//! The transitions shelf: the blends the app offers, and dropping one on a
//! cut between two touching main-track clips.
//!
//! A blend on a cut shows on the timeline as a bar over its window. The
//! bar is picked, deleted, and made longer or shorter by its edges, between
//! the shortest blend and the longest the two clips have room for.

use cutlass_effects::TransitionKind;
use cutlass_model::{
    Draft, MaterialId, MaterialTransition, Micros, Segment, SegmentId, Track, TrackAttribute,
    TrackId, blend_extents, blend_window,
};
use cutlass_session::Request;

use crate::hit::{BadScale, Pointer, TimelineScale, hit_test};

/// What a transition card's id starts with; the transition's id follows.
pub const TRANSITION_CARD_PREFIX: &str = "transition:";

/// A blend the shelf can drop onto a main-track cut.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransitionOffer {
    pub id: &'static str,
    pub name: &'static str,
    /// Preset length in microseconds. A short clip stores a shorter blend.
    pub duration: i64,
    /// `true` splits the window across the cut. `false` keeps it on the outgoing clip.
    pub overlap: bool,
    /// What it draws.
    pub kind: TransitionKind,
    /// The shelf category, one of [`cutlass_effects::TRANSITION_CATEGORIES`].
    pub category: &'static str,
}

impl TransitionOffer {
    #[must_use]
    pub fn card_id(self) -> String {
        format!("{TRANSITION_CARD_PREFIX}{}", self.id)
    }
}

const fn offer(kind: TransitionKind, id: &'static str, name: &'static str) -> TransitionOffer {
    let dip = matches!(
        kind,
        TransitionKind::DipToBlack | TransitionKind::DipToWhite
    );
    TransitionOffer {
        id,
        name,
        duration: if dip { 400_000 } else { 500_000 },
        overlap: !dip,
        kind,
        category: "",
    }
}

/// The twenty built-in transitions. Dips stay on the outgoing clip; the
/// rest cross the cut.
const BUILT_IN_OFFERS: &[TransitionOffer] = &[
    offer(TransitionKind::Dissolve, "dissolve", "Dissolve"),
    offer(TransitionKind::DipToBlack, "dip", "Dip to Black"),
    offer(TransitionKind::DipToWhite, "dip-white", "Dip to White"),
    offer(TransitionKind::WipeLeft, "wipe-left", "Wipe Left"),
    offer(TransitionKind::WipeRight, "wipe-right", "Wipe Right"),
    offer(TransitionKind::WipeUp, "wipe-up", "Wipe Up"),
    offer(TransitionKind::WipeDown, "wipe-down", "Wipe Down"),
    offer(TransitionKind::SlideLeft, "slide-left", "Slide Left"),
    offer(TransitionKind::SlideRight, "slide-right", "Slide Right"),
    offer(TransitionKind::SlideUp, "slide-up", "Slide Up"),
    offer(TransitionKind::SlideDown, "slide-down", "Slide Down"),
    offer(TransitionKind::PushLeft, "push-left", "Push Left"),
    offer(TransitionKind::PushRight, "push-right", "Push Right"),
    offer(TransitionKind::PushUp, "push-up", "Push Up"),
    offer(TransitionKind::PushDown, "push-down", "Push Down"),
    offer(TransitionKind::ZoomIn, "zoom-in", "Zoom In"),
    offer(TransitionKind::ZoomOut, "zoom-out", "Zoom Out"),
    offer(TransitionKind::Spin, "spin", "Spin"),
    offer(TransitionKind::Blur, "blur", "Blur"),
    offer(TransitionKind::Glitch, "glitch", "Glitch"),
];

/// Every transition the shelf offers: the built-in ones, then each
/// bundled recipe, then each from the studio folder, each with its
/// category. The studio's change as its files do.
#[must_use]
pub fn transition_offers() -> Vec<TransitionOffer> {
    let built_in = BUILT_IN_OFFERS.iter().map(|offer| TransitionOffer {
        category: offer.kind.category(),
        ..*offer
    });
    let recipes = TransitionKind::every().into_iter().filter_map(|kind| {
        let recipe = kind.recipe()?;
        Some(TransitionOffer {
            id: recipe.id.as_str(),
            name: recipe.name.as_str(),
            duration: recipe.duration().0,
            overlap: true,
            kind,
            category: recipe.category.as_str(),
        })
    });
    built_in.chain(recipes).collect()
}

/// The shelf's categories that hold at least one transition, in shelf
/// order.
#[must_use]
pub fn transition_categories() -> Vec<&'static str> {
    cutlass_effects::TRANSITION_CATEGORIES
        .iter()
        .copied()
        .filter(|category| {
            transition_offers()
                .iter()
                .any(|offer| offer.category == *category)
        })
        .collect()
}

/// The offer a shelf card names.
#[must_use]
pub fn transition_of_card(card_id: &str) -> Option<TransitionOffer> {
    transition_offers()
        .into_iter()
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

/// The clip under `time` on `track`, and the clips that touch it before
/// and after.
fn clip_at(track: &Track, time: Micros) -> Option<(Option<&Segment>, &Segment, Option<&Segment>)> {
    let under = track.segments.iter().find(|segment| {
        segment.target.duration.0 > 0
            && segment.target.start.0 <= time.0
            && time.0 < segment.target.end().0
    })?;
    let touching = |at: Micros, end: bool| {
        track.segments.iter().find(|other| {
            other.id != under.id
                && other.target.duration.0 > 0
                && if end {
                    other.target.end() == at
                } else {
                    other.target.start == at
                }
        })
    };
    Some((
        touching(under.target.start, true),
        under,
        touching(under.target.end(), false),
    ))
}

/// The cut a transition dropped at `time` on the main track lands on.
///
/// A touching cut within `reach` of the pointer takes it. Otherwise the
/// drop is on a clip: the blend goes on the cut where that clip begins, and
/// on the cut where it ends when no clip touches its start, as the first
/// clip of the track has none. A clip that touches nothing has no cut.
pub fn drop_cut(draft: &Draft, time: Micros, reach: Micros) -> Result<MainCut, TransitionMiss> {
    match main_cut(draft, time, reach) {
        Err(TransitionMiss::NoCut) => {}
        found => return found,
    }
    let track = draft
        .tracks
        .iter()
        .find(|track| track.is_main())
        .ok_or(TransitionMiss::NoCut)?;
    let (before, under, after) = clip_at(track, time).ok_or(TransitionMiss::NoCut)?;
    let (outgoing, incoming) = match (before, after) {
        (Some(before), _) => (before, under),
        (None, Some(after)) => (under, after),
        (None, None) => return Err(TransitionMiss::NoCut),
    };
    Ok(MainCut {
        outgoing_id: outgoing.id.clone(),
        outgoing: outgoing.target.duration,
        incoming: incoming.target.duration,
    })
}

/// The shortest blend an edge can be dragged to, in microseconds.
pub const TRANSITION_SHORTEST: i64 = 100_000;

/// The longest blend, in microseconds, when the clips have the room.
pub const TRANSITION_LONGEST: i64 = 5_000_000;

/// The blend a segment carries out of itself.
#[must_use]
pub fn blend_of<'a>(draft: &'a Draft, segment: &Segment) -> Option<&'a MaterialTransition> {
    segment.extra_material_refs.iter().find_map(|id| {
        draft
            .materials
            .transitions
            .iter()
            .find(|item| &item.id == id)
    })
}

/// A blend on a cut, as the timeline draws and picks it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransitionBar {
    /// The outgoing clip, which names the blend.
    pub segment_id: SegmentId,
    pub track_id: TrackId,
    pub name: String,
    pub cut: Micros,
    /// The window the blend plays over.
    pub start: Micros,
    pub end: Micros,
    pub duration: Micros,
    /// The window crosses the cut; off, it stays on the outgoing clip.
    pub overlap: bool,
}

/// Every blend that sits on a cut, in track order. A blend with no clip
/// after it, or with no length, is not drawn.
#[must_use]
pub fn transition_bars(draft: &Draft) -> Vec<TransitionBar> {
    if draft.materials.transitions.is_empty() {
        return Vec::new();
    }
    let mut bars = Vec::new();
    for track in &draft.tracks {
        for segment in &track.segments {
            let Some(blend) = blend_of(draft, segment) else {
                continue;
            };
            let cut = segment.target.end();
            let meets = track
                .segments
                .iter()
                .any(|other| other.id != segment.id && other.target.start == cut);
            let Some((start, end)) = blend_window(cut, blend.duration, blend.is_overlap) else {
                continue;
            };
            if meets {
                bars.push(TransitionBar {
                    segment_id: segment.id.clone(),
                    track_id: track.id.clone(),
                    name: blend.name.clone(),
                    cut,
                    start,
                    end,
                    duration: blend.duration,
                    overlap: blend.is_overlap,
                });
            }
        }
    }
    bars
}

/// The room a blend on the cut after `outgoing_id` has on each side: the
/// outgoing clip less the blend that comes into it from the cut before,
/// and the incoming clip less its own blend into the cut after. Also the
/// blend itself.
fn cut_room<'a>(
    draft: &'a Draft,
    outgoing_id: &SegmentId,
) -> Option<(i64, i64, Option<&'a MaterialTransition>)> {
    let track = draft.tracks.iter().find(|track| {
        track
            .segments
            .iter()
            .any(|segment| &segment.id == outgoing_id)
    })?;
    let outgoing = track
        .segments
        .iter()
        .find(|segment| &segment.id == outgoing_id)?;
    let incoming = track.segments.iter().find(|other| {
        other.id != outgoing.id
            && other.target.duration.0 > 0
            && other.target.start == outgoing.target.end()
    })?;
    let extents = |segment: &Segment| {
        blend_of(draft, segment)
            .and_then(|blend| blend_extents(blend.duration, blend.is_overlap))
            .map_or((0, 0), |(left, right)| (left.0, right.0))
    };
    let before = track
        .segments
        .iter()
        .find(|other| other.id != outgoing.id && other.target.end() == outgoing.target.start)
        .map_or(0, |before| extents(before).1);
    let after = extents(incoming).0;
    Some((
        outgoing.target.duration.0.saturating_sub(before),
        incoming.target.duration.0.saturating_sub(after),
        blend_of(draft, outgoing),
    ))
}

/// The shortest and the longest the blend on the cut after `outgoing_id`
/// can be: [`TRANSITION_SHORTEST`] to [`TRANSITION_LONGEST`], held to the
/// room the two clips have. `None` when the clip has no blend on a cut.
#[must_use]
pub fn transition_limits(draft: &Draft, outgoing_id: &SegmentId) -> Option<(Micros, Micros)> {
    let (outgoing, incoming, blend) = cut_room(draft, outgoing_id)?;
    let longest = fit_transition(TRANSITION_LONGEST, outgoing, incoming, blend?.is_overlap)?;
    Some((Micros(TRANSITION_SHORTEST.min(longest)), Micros(longest)))
}

/// The request that makes the blend on the cut after `outgoing_id` last
/// `duration`, held to [`transition_limits`]. `None` when the clip has no
/// blend on a cut, or the blend already lasts that long.
#[must_use]
pub fn transition_length_request(
    draft: &Draft,
    outgoing_id: &SegmentId,
    duration: Micros,
) -> Option<Request> {
    let (shortest, longest) = transition_limits(draft, outgoing_id)?;
    let (_, _, blend) = cut_room(draft, outgoing_id)?;
    let blend = blend?;
    let duration = Micros(duration.0.clamp(shortest.0, longest.0));
    (duration != blend.duration).then(|| Request::SetTransition {
        segment_id: outgoing_id.clone(),
        transition: Box::new(MaterialTransition {
            duration,
            ..blend.clone()
        }),
    })
}

/// An edge of a transition bar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarEdge {
    Start,
    End,
}

/// How long `bar` lasts with its `edge` dragged to `time`. A blend that
/// crosses the cut grows both ways from it; one that stays on the outgoing
/// clip follows its start, and its end is the cut.
#[must_use]
pub fn bar_length(bar: &TransitionBar, edge: BarEdge, time: Micros) -> Micros {
    let length = match (bar.overlap, edge) {
        (true, BarEdge::Start) => bar.cut.0.saturating_sub(time.0).saturating_mul(2),
        (true, BarEdge::End) => time.0.saturating_sub(bar.cut.0).saturating_mul(2),
        (false, BarEdge::Start) => bar.cut.0.saturating_sub(time.0),
        (false, BarEdge::End) => bar.duration.0,
    };
    Micros(length.max(0))
}

/// The requests that put the blend of `outgoing_id` on every other cut of
/// its track, each held to half of the two clips it sits between. A cut
/// that has a blend keeps its material id; `fresh` names one for a cut that
/// has none.
pub fn transition_everywhere(
    draft: &Draft,
    outgoing_id: &SegmentId,
    fresh: &mut dyn FnMut(&str) -> String,
) -> Vec<Request> {
    let Some(track) = draft.tracks.iter().find(|track| {
        track
            .segments
            .iter()
            .any(|segment| &segment.id == outgoing_id)
    }) else {
        return Vec::new();
    };
    let Some(source) = track
        .segments
        .iter()
        .find(|segment| &segment.id == outgoing_id)
        .and_then(|segment| blend_of(draft, segment))
    else {
        return Vec::new();
    };
    let mut ordered: Vec<&Segment> = track.segments.iter().collect();
    ordered.sort_by_key(|segment| segment.target.start.0);
    let mut requests = Vec::new();
    for pair in ordered.windows(2) {
        let (outgoing, incoming) = (pair[0], pair[1]);
        if &outgoing.id == outgoing_id || incoming.target.start != outgoing.target.end() {
            continue;
        }
        // Half of each clip at most, so the blends on a clip's two cuts
        // never run into each other.
        let Some(duration) = fit_transition(
            source.duration.0,
            outgoing.target.duration.0 / 2,
            incoming.target.duration.0 / 2,
            source.is_overlap,
        ) else {
            continue;
        };
        let id = blend_of(draft, outgoing).map_or_else(
            || MaterialId::new(fresh("transition")),
            |existing| existing.id.clone(),
        );
        requests.push(Request::SetTransition {
            segment_id: outgoing.id.clone(),
            transition: Box::new(MaterialTransition {
                id,
                duration: Micros(duration),
                ..source.clone()
            }),
        });
    }
    requests
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

/// The request that stores `offer` on the main-track cut the pointer names
/// ([`drop_cut`]): the cut within `reach`, or the one the clip under the
/// pointer begins at.
///
/// The blend opens no lane and moves neither clip. A preset longer than the
/// clips is shortened to fit. A cut that has a blend takes this one in its
/// place.
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
    let cut = drop_cut(draft, hit.time, reach)?;
    // Held to the room beside the blends on the cuts before and after.
    let (outgoing, incoming) = cut_room(draft, &cut.outgoing_id).map_or(
        (cut.outgoing.0, cut.incoming.0),
        |(outgoing, incoming, _)| (outgoing, incoming),
    );
    let fitted = fit_transition(offer.duration, outgoing, incoming, offer.overlap)
        .ok_or(TransitionMiss::TooShort)?;
    Ok(Request::SetTransition {
        segment_id: cut.outgoing_id,
        transition: Box::new(MaterialTransition {
            id: material_id,
            kind: "transition".to_owned(),
            name: offer.name.to_owned(),
            effect_id: offer.kind.effect_id(),
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

    fn blend(id: &str, duration: i64, overlap: bool) -> MaterialTransition {
        MaterialTransition {
            id: MaterialId::new(id),
            name: id.to_owned(),
            duration: Micros(duration),
            is_overlap: overlap,
            ..Default::default()
        }
    }

    /// `draft` with `blend` on the cut after clip `index`.
    fn with_blend(mut draft: Draft, index: usize, blend: MaterialTransition) -> Draft {
        draft.tracks[0].segments[index]
            .extra_material_refs
            .push(blend.id.clone());
        draft.materials.transitions.push(blend);
        draft
    }

    #[test]
    fn a_drop_on_a_clip_takes_the_cut_it_begins_at_or_the_one_it_ends_at() {
        // s0 and s1 touch, s2 follows s1, s3 stands alone after a gap.
        let draft = draft(&[
            (0, 1_000_000),
            (1_000_000, 1_000_000),
            (2_000_000, 1_000_000),
            (5_000_000, 1_000_000),
        ]);
        let reach = Micros(10_000);
        let on = |time: i64| drop_cut(&draft, Micros(time), reach).map(|cut| cut.outgoing_id);
        // The middle of the first clip: no clip before it, so the cut after.
        assert_eq!(on(500_000), Ok(SegmentId::new("s0")));
        // The middle of the second and of the third: the cut each begins at.
        assert_eq!(on(1_500_000), Ok(SegmentId::new("s0")));
        assert_eq!(on(2_500_000), Ok(SegmentId::new("s1")));
        // Right beside a cut, that cut, whichever clip the pointer is on.
        assert_eq!(on(1_995_000), Ok(SegmentId::new("s1")));
        assert_eq!(on(2_005_000), Ok(SegmentId::new("s1")));
        // A clip that touches nothing, and the gap, have no cut.
        assert_eq!(on(5_500_000), Err(TransitionMiss::NoCut));
        assert_eq!(on(4_000_000), Err(TransitionMiss::NoCut));
    }

    #[test]
    fn bars_are_the_blends_on_cuts_with_their_windows() {
        let plain = draft(&[
            (0, 1_000_000),
            (1_000_000, 1_000_000),
            (5_000_000, 1_000_000),
        ]);
        assert!(transition_bars(&plain).is_empty());
        let draft = with_blend(plain, 0, blend("cross", 400_000, true));
        // A blend where no clip follows is not on a cut.
        let draft = with_blend(draft, 2, blend("alone", 400_000, true));
        let bars = transition_bars(&draft);
        assert_eq!(bars.len(), 1);
        let bar = &bars[0];
        assert_eq!(bar.segment_id, SegmentId::new("s0"));
        assert_eq!(bar.name, "cross");
        assert_eq!(
            (bar.start, bar.cut, bar.end),
            (Micros(800_000), Micros(1_000_000), Micros(1_200_000))
        );

        // An edge dragged out makes a crossing blend grow both ways.
        assert_eq!(
            bar_length(bar, BarEdge::Start, Micros(700_000)),
            Micros(600_000)
        );
        assert_eq!(
            bar_length(bar, BarEdge::End, Micros(1_100_000)),
            Micros(200_000)
        );
        assert_eq!(bar_length(bar, BarEdge::End, Micros(900_000)), Micros(0));
        let dip = TransitionBar {
            overlap: false,
            ..bar.clone()
        };
        assert_eq!(
            bar_length(&dip, BarEdge::Start, Micros(700_000)),
            Micros(300_000)
        );
        assert_eq!(
            bar_length(&dip, BarEdge::End, Micros(1_500_000)),
            dip.duration
        );
    }

    #[test]
    fn a_blend_is_as_long_as_asked_between_the_shortest_and_the_room() {
        let draft = with_blend(
            draft(&[(0, 1_000_000), (1_000_000, 300_000), (1_300_000, 1_000_000)]),
            0,
            blend("cross", 400_000, true),
        );
        let id = SegmentId::new("s0");
        // The incoming clip is 0.3s, so the blend's half on it is at most that.
        assert_eq!(
            transition_limits(&draft, &id),
            Some((Micros(TRANSITION_SHORTEST), Micros(600_000)))
        );
        let length = |draft: &Draft, asked: i64| match transition_length_request(
            draft,
            &id,
            Micros(asked),
        ) {
            Some(Request::SetTransition { transition, .. }) => Some(transition.duration.0),
            _ => None,
        };
        assert_eq!(length(&draft, 500_000), Some(500_000));
        assert_eq!(length(&draft, 9_000_000), Some(600_000));
        assert_eq!(length(&draft, 1), Some(TRANSITION_SHORTEST));
        assert_eq!(length(&draft, 400_000), None, "already that long");

        // A blend on the next cut takes its half of the short clip first.
        let crowded = with_blend(draft.clone(), 1, blend("next", 200_000, true));
        assert_eq!(
            transition_limits(&crowded, &id).map(|(_, longest)| longest),
            Some(Micros(400_000))
        );
        // No blend, or no clip after, and there is nothing to make longer.
        assert_eq!(transition_limits(&draft, &SegmentId::new("s1")), None);
        assert_eq!(transition_limits(&draft, &SegmentId::new("s2")), None);
        // Long clips stop at the longest blend.
        let roomy = with_blend(
            self::draft(&[(0, 60_000_000), (60_000_000, 60_000_000)]),
            0,
            blend("cross", 400_000, true),
        );
        assert_eq!(
            transition_limits(&roomy, &id).map(|(_, longest)| longest),
            Some(Micros(TRANSITION_LONGEST))
        );
    }

    #[test]
    fn a_blend_goes_on_every_other_cut_fitted_to_its_clips() {
        // Cuts after s0, s1 and s2; s3 touches nothing after it.
        let draft = with_blend(
            draft(&[
                (0, 2_000_000),
                (2_000_000, 2_000_000),
                (4_000_000, 100_000),
                (4_100_000, 2_000_000),
                (9_000_000, 1_000_000),
            ]),
            0,
            blend("cross", 800_000, true),
        );
        let draft = with_blend(draft, 1, blend("old", 50_000, false));
        let mut next = 0;
        let mut fresh = |prefix: &str| {
            next += 1;
            format!("{prefix}-{next}")
        };
        let requests = transition_everywhere(&draft, &SegmentId::new("s0"), &mut fresh);
        let placed: Vec<(String, String, String, i64)> = requests
            .iter()
            .map(|request| match request {
                Request::SetTransition {
                    segment_id,
                    transition,
                } => (
                    segment_id.as_str().to_owned(),
                    transition.id.as_str().to_owned(),
                    transition.name.clone(),
                    transition.duration.0,
                ),
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(
            placed,
            [
                // The cut that had a blend keeps its material id; half of
                // the 0.1s clip between them holds each blend to 0.1s.
                (
                    "s1".to_owned(),
                    "old".to_owned(),
                    "cross".to_owned(),
                    100_000
                ),
                (
                    "s2".to_owned(),
                    "transition-1".to_owned(),
                    "cross".to_owned(),
                    100_001
                ),
            ]
        );
        assert!(transition_everywhere(&draft, &SegmentId::new("s3"), &mut fresh).is_empty());
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
            transition_offers()[0],
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
            Some(transition_offers()[1])
        );
        assert_eq!(transition_of_card("transition:none"), None);
    }

    #[test]
    fn the_shelf_offers_every_built_in_transition_by_its_own_key() {
        let every = TransitionKind::every();
        assert_eq!(transition_offers().len(), every.len());
        for (offer, kind) in transition_offers().iter().zip(every) {
            assert_eq!(offer.kind, kind);
            assert_eq!(offer.id, kind.key());
            assert_eq!(offer.name, kind.label());
            assert!(
                cutlass_effects::TRANSITION_CATEGORIES.contains(&offer.category),
                "{}",
                offer.id
            );
            assert_eq!(transition_of_card(&offer.card_id()), Some(*offer));
        }
    }
}
