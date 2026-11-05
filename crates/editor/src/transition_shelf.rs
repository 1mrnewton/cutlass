//! The transitions shelf: the blends the app offers, and dropping one on a
//! cut between two touching main-track clips.

use cutlass_model::{Draft, MaterialId, MaterialTransition, Micros, SegmentId, TrackAttribute};
use cutlass_session::Request;

use crate::hit::{BadScale, Pointer, TimelineScale, hit_test};

/// A blend the shelf can drop onto a main-track cut.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransitionOffer {
    pub id: &'static str,
    pub name: &'static str,
    /// Preset length in microseconds. A short clip stores a shorter blend.
    pub duration: i64,
    /// `true` splits the window across the cut. `false` keeps it on the outgoing clip.
    pub overlap: bool,
}

impl TransitionOffer {
    #[must_use]
    pub fn card_id(self) -> String {
        format!("transition:{}", self.id)
    }
}

/// Dissolve crosses the cut. Dip stays on the outgoing clip.
pub const TRANSITION_OFFERS: &[TransitionOffer] = &[
    TransitionOffer {
        id: "dissolve",
        name: "Dissolve",
        duration: 500_000,
        overlap: true,
    },
    TransitionOffer {
        id: "dip",
        name: "Dip",
        duration: 400_000,
        overlap: false,
    },
];

/// The offer a shelf card names.
#[must_use]
pub fn transition_of_card(card_id: &str) -> Option<TransitionOffer> {
    TRANSITION_OFFERS
        .iter()
        .copied()
        .find(|offer| offer.card_id() == card_id)
}

/// Why a transition drop stores nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransitionMiss {
    /// The pixel size or the lane height cannot map the pointer.
    BadScale,
    /// The pointer is outside every lane.
    OffLanes,
    /// The pointer is on a lane that is not the main track.
    NotMain,
    /// The main track is locked.
    Locked,
    /// No touching cut is within reach of the pointer.
    NoCut,
    /// The clips are too short for any blend.
    TooShort,
}

/// The outgoing clip of a cut between two touching main-track clips.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MainCut {
    pub outgoing_id: SegmentId,
    pub outgoing: Micros,
    pub incoming: Micros,
}

/// The touching main-track cut nearest `time`, within `reach`. A gap is not a cut.
pub fn main_cut(draft: &Draft, time: Micros, reach: Micros) -> Result<MainCut, TransitionMiss> {
    let track = draft
        .tracks
        .iter()
        .find(|track| track.is_main())
        .ok_or(TransitionMiss::NoCut)?;
    if track.attribute.contains(TrackAttribute::LOCK) {
        return Err(TransitionMiss::Locked);
    }
    let mut ordered: Vec<_> = track.segments.iter().collect();
    ordered.sort_by_key(|segment| segment.target.start.0);
    let mut best: Option<(i64, MainCut)> = None;
    for pair in ordered.windows(2) {
        let (left, right) = (pair[0], pair[1]);
        if left.target.duration.0 <= 0 || right.target.duration.0 <= 0 {
            continue;
        }
        let cut = left.target.end().0;
        if right.target.start.0 != cut {
            continue;
        }
        let distance = (time.0 - cut).saturating_abs();
        if distance > reach.0 || best.as_ref().is_some_and(|(near, _)| distance >= *near) {
            continue;
        }
        best = Some((
            distance,
            MainCut {
                outgoing_id: left.id.clone(),
                outgoing: left.target.duration,
                incoming: right.target.duration,
            },
        ));
    }
    best.map(|(_, cut)| cut).ok_or(TransitionMiss::NoCut)
}

/// Shorten `preset` so the blend window stays inside both clips.
#[must_use]
pub fn fit_transition(preset: i64, outgoing: i64, incoming: i64, overlap: bool) -> Option<i64> {
    if preset <= 0 || outgoing <= 0 || incoming <= 0 {
        return None;
    }
    let fitted = if overlap {
        let even = outgoing.min(incoming).saturating_mul(2);
        let odd = outgoing
            .min(incoming.saturating_sub(1))
            .saturating_mul(2)
            .saturating_add(1);
        even.max(odd).min(preset)
    } else {
        preset.min(outgoing)
    };
    (fitted > 0).then_some(fitted)
}

/// The request that stores `offer` on the main-track cut under the pointer.
///
/// The blend opens no lane and moves neither clip. A preset longer than the
/// clips is shortened to fit. `reach` is how far from the pointer a cut may be.
pub fn transition_request(
    draft: &Draft,
    offer: TransitionOffer,
    pointer: Pointer,
    scale: TimelineScale,
    reach: Micros,
    material_id: MaterialId,
) -> Result<Request, TransitionMiss> {
    let hit = match hit_test(draft, scale, pointer) {
        Ok(Some(hit)) => hit,
        Ok(None) => return Err(TransitionMiss::OffLanes),
        Err(BadScale) => return Err(TransitionMiss::BadScale),
    };
    if !hit.lane.main {
        return Err(TransitionMiss::NotMain);
    }
    let cut = main_cut(draft, hit.time, reach)?;
    let fitted = fit_transition(
        offer.duration,
        cut.outgoing.0,
        cut.incoming.0,
        offer.overlap,
    )
    .ok_or(TransitionMiss::TooShort)?;
    Ok(Request::SetTransition {
        segment_id: cut.outgoing_id,
        transition: Box::new(MaterialTransition {
            id: material_id,
            kind: "transition".to_owned(),
            name: offer.name.to_owned(),
            duration: Micros(fitted),
            is_overlap: offer.overlap,
            ..Default::default()
        }),
    })
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Segment, TimeRange, TrackId};

    use super::*;

    fn draft(clips: &[(i64, i64)]) -> Draft {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        for (index, (start, duration)) in clips.iter().enumerate() {
            draft.tracks[0].segments.push(Segment::new(
                format!("s{index}"),
                "m",
                TimeRange::new(*start, *duration),
            ));
        }
        draft
    }

    #[test]
    fn the_nearest_touching_cut_wins_and_a_gap_is_not_one() {
        let draft = draft(&[(0, 1_000), (1_000, 1_000), (2_500, 1_000)]);
        let cut = main_cut(&draft, Micros(1_100), Micros(200)).unwrap();
        assert_eq!(cut.outgoing_id, SegmentId::new("s0"));
        assert_eq!(
            main_cut(&draft, Micros(2_250), Micros(300)),
            Err(TransitionMiss::NoCut)
        );
    }

    #[test]
    fn a_locked_main_track_refuses() {
        let mut draft = draft(&[(0, 1_000), (1_000, 1_000)]);
        draft.tracks[0].attribute = TrackAttribute(TrackAttribute::LOCK);
        assert_eq!(
            main_cut(&draft, Micros(1_000), Micros(10)),
            Err(TransitionMiss::Locked)
        );
    }

    #[test]
    fn a_preset_shrinks_to_fit_the_clips() {
        assert_eq!(fit_transition(500_000, 100, 100, true), Some(200));
        assert_eq!(fit_transition(500_000, 100, 101, true), Some(201));
        assert_eq!(fit_transition(400_000, 100, 5, false), Some(100));
        assert_eq!(
            fit_transition(400_000, 1_000_000, 1_000_000, false),
            Some(400_000)
        );
        assert_eq!(fit_transition(0, 100, 100, true), None);
    }

    #[test]
    fn a_drop_on_the_cut_stores_a_blend_on_the_outgoing_clip() {
        let draft = draft(&[(0, 1_000_000), (1_000_000, 1_000_000)]);
        let scale = TimelineScale {
            lane_height: 40.0,
            pixel_size: 1.0e-3,
            empty_lanes: false,
        };
        let request = transition_request(
            &draft,
            TRANSITION_OFFERS[0],
            Pointer { x: 1_000.0, y: 8.0 },
            scale,
            Micros(12_000),
            MaterialId::new("t"),
        )
        .unwrap();
        let Request::SetTransition {
            segment_id,
            transition,
        } = request
        else {
            panic!("the drop stores a blend");
        };
        assert_eq!(segment_id, SegmentId::new("s0"));
        assert_eq!(transition.duration, Micros(500_000));
        assert!(transition.is_overlap);
        assert_eq!(
            transition_of_card("transition:dip"),
            Some(TRANSITION_OFFERS[1])
        );
        assert_eq!(transition_of_card("transition:none"), None);
    }
}
