//! Load-time report.
//!
//! Repair names a missing main track, a bad track order, a range conflict, or
//! a transition window that runs past a segment. It does not move segments,
//! and it does not change the transition material.
//!
//! File order is the row order, from the top of the timeline to the bottom.
//! Audio is the bottom block. The main track sits just above it. Overlay,
//! sticker, text, effect, filter, adjust, and plugin effect sit above the
//! main track. A track on the wrong side of that main track is a bad order.
//! A second main track is a bad order. Kinds outside those two lists are not
//! given a side.

use std::collections::BTreeSet;

use cutlass_model::{
    Draft, MaterialId, MaterialRef, MaterialTransition, Segment, SegmentId, Track, TrackId,
    TrackType,
};

/// One problem found on open. The draft that was read is not changed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RepairNote {
    /// No video or mixed track with flag 0.
    MissingMainTrack,
    /// This track is not on its side of the main track, or it is another main track.
    BadTrackOrder { track_id: TrackId },
    /// A segment duration is not positive, or two positive spans on one track overlap.
    /// Touching at an endpoint is not a conflict.
    RangeConflict {
        track_id: TrackId,
        segment_id: SegmentId,
    },
    /// The blend window runs past a segment, or `duration` is negative.
    /// The material and the ranges stay as stored.
    TransitionWindow {
        track_id: TrackId,
        segment_id: SegmentId,
        material_id: MaterialId,
    },
    /// The top draft and the main timeline differ. The timeline folder was loaded.
    OutwardCopyDiffers,
    /// This timeline folder's draft id is not the directory name. It was not loaded.
    TimelineIdMismatch { directory: String, draft_id: String },
}

/// The problems in `draft`. The same draft is the one the caller still holds.
#[must_use]
pub fn repair(draft: &Draft) -> Vec<RepairNote> {
    let mut notes = Vec::new();
    match draft.tracks.iter().position(Track::is_main) {
        None => notes.push(RepairNote::MissingMainTrack),
        Some(main_index) => {
            for (index, track) in draft.tracks.iter().enumerate() {
                let wrong = (index != main_index && track.is_main())
                    || (index < main_index && is_under_main(track))
                    || (index > main_index && is_above_main(track));
                if wrong {
                    notes.push(RepairNote::BadTrackOrder {
                        track_id: track.id.clone(),
                    });
                }
            }
        }
    }
    for track in &draft.tracks {
        range_conflicts(track, &mut notes);
        transition_windows(draft, track, &mut notes);
    }
    notes
}

fn is_under_main(track: &Track) -> bool {
    track.kind == TrackType::Audio
}

fn is_above_main(track: &Track) -> bool {
    if track.is_main() {
        return false;
    }
    matches!(
        track.kind,
        TrackType::Video
            | TrackType::Mixed
            | TrackType::Sticker
            | TrackType::Text
            | TrackType::Effect
            | TrackType::Filter
            | TrackType::Adjust
            | TrackType::PluginEffect
    )
}

fn range_conflicts(track: &Track, notes: &mut Vec<RepairNote>) {
    let mut positive = Vec::new();
    for (index, segment) in track.segments.iter().enumerate() {
        let usable = segment.target.duration.0 > 0
            && segment
                .target
                .start
                .0
                .checked_add(segment.target.duration.0)
                .is_some();
        if !usable {
            notes.push(RepairNote::RangeConflict {
                track_id: track.id.clone(),
                segment_id: segment.id.clone(),
            });
            continue;
        }
        positive.push(index);
    }
    positive.sort_by(|&left, &right| {
        track.segments[left]
            .target
            .start
            .cmp(&track.segments[right].target.start)
            .then(left.cmp(&right))
    });
    let mut reported = BTreeSet::new();
    for (offset, &index) in positive.iter().enumerate() {
        let end = track.segments[index].target.end().0;
        for &other in positive.iter().skip(offset + 1) {
            if track.segments[other].target.start.0 >= end {
                break;
            }
            if reported.insert(other) {
                notes.push(RepairNote::RangeConflict {
                    track_id: track.id.clone(),
                    segment_id: track.segments[other].id.clone(),
                });
            }
        }
    }
}

/// A transition is a material on the outgoing segment, not a segment of its own.
/// The cut is that segment's target end. A duration of 0 is a hard cut.
fn transition_windows(draft: &Draft, track: &Track, notes: &mut Vec<RepairNote>) {
    for segment in &track.segments {
        for material_id in &segment.extra_material_refs {
            let Some(MaterialRef::Transition(transition)) =
                draft.materials.get(material_id.as_str())
            else {
                continue;
            };
            if window_overruns(track, segment, transition) {
                notes.push(RepairNote::TransitionWindow {
                    track_id: track.id.clone(),
                    segment_id: segment.id.clone(),
                    material_id: material_id.clone(),
                });
            }
        }
    }
}

fn window_overruns(track: &Track, segment: &Segment, transition: &MaterialTransition) -> bool {
    let duration = transition.duration.0;
    if duration < 0 {
        return true;
    }
    if duration == 0 {
        return false;
    }
    let Some(cut) = segment
        .target
        .start
        .0
        .checked_add(segment.target.duration.0)
    else {
        return true;
    };
    let (left, right) = if transition.is_overlap {
        let left = duration / 2;
        (left, duration - left)
    } else {
        (duration, 0)
    };
    let Some(window_start) = cut.checked_sub(left) else {
        return true;
    };
    if window_start < segment.target.start.0 {
        return true;
    }
    if right == 0 {
        return false;
    }
    let Some(window_end) = cut.checked_add(right) else {
        return true;
    };
    let Some(incoming) = track
        .segments
        .iter()
        .find(|other| other.id != segment.id && other.target.start.0 == cut)
    else {
        return true;
    };
    let Some(incoming_end) = incoming
        .target
        .start
        .0
        .checked_add(incoming.target.duration.0)
    else {
        return true;
    };
    window_end > incoming_end
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        Draft, MaterialId, MaterialTransition, Micros, Segment, SegmentId, TimeRange, Track,
        TrackFlag, TrackId, TrackType,
    };

    use super::{RepairNote, repair};

    fn note_order(track: &str) -> RepairNote {
        RepairNote::BadTrackOrder {
            track_id: TrackId::new(track),
        }
    }

    fn note_range(track: &str, segment: &str) -> RepairNote {
        RepairNote::RangeConflict {
            track_id: TrackId::new(track),
            segment_id: SegmentId::new(segment),
        }
    }

    fn note_window(track: &str, segment: &str, material: &str) -> RepairNote {
        RepairNote::TransitionWindow {
            track_id: TrackId::new(track),
            segment_id: SegmentId::new(segment),
            material_id: MaterialId::new(material),
        }
    }

    fn with_transition(
        draft: &mut Draft,
        segment: &str,
        material: &str,
        duration: i64,
        overlap: bool,
    ) {
        draft.materials.transitions.push(MaterialTransition {
            id: MaterialId::new(material),
            duration: Micros(duration),
            is_overlap: overlap,
            ..MaterialTransition::default()
        });
        let segment = draft
            .tracks
            .iter_mut()
            .flat_map(|track| track.segments.iter_mut())
            .find(|item| item.id.as_str() == segment)
            .unwrap();
        segment.extra_material_refs.push(MaterialId::new(material));
    }

    fn lane(id: &str, kind: TrackType) -> Track {
        let mut track = Track::main(id);
        track.kind = kind;
        track
    }

    fn overlay(id: &str) -> Track {
        let mut track = Track::main(id);
        track.flag = TrackFlag::SubVideo;
        track
    }

    fn clip(id: &str, start: i64, duration: i64) -> Segment {
        Segment::new(id, "m", TimeRange::new(start, duration))
    }

    #[test]
    fn a_missing_main_track_is_named_and_nothing_moves() {
        let mut draft = Draft::empty("D");
        draft.tracks = vec![lane("audio", TrackType::Audio)];
        let before = draft.clone();
        assert_eq!(repair(&draft), vec![RepairNote::MissingMainTrack]);
        assert_eq!(draft, before);
    }

    #[test]
    fn audio_above_main_and_text_below_main_are_bad_order() {
        let mut draft = Draft::empty("D");
        draft.tracks = vec![
            lane("audio", TrackType::Audio),
            Track::main("main"),
            lane("title", TrackType::Text),
        ];
        assert_eq!(
            repair(&draft),
            vec![note_order("audio"), note_order("title")]
        );
    }

    #[test]
    fn overlay_text_main_and_audio_are_in_order() {
        let mut draft = Draft::empty("D");
        draft.tracks = vec![
            overlay("overlay"),
            lane("title", TrackType::Text),
            Track::main("main"),
            lane("audio", TrackType::Audio),
        ];
        assert!(repair(&draft).is_empty());
    }

    #[test]
    fn a_second_main_track_is_a_bad_order() {
        let mut draft = Draft::empty("D");
        draft.tracks = vec![Track::main("first"), Track::main("second")];
        assert_eq!(repair(&draft), vec![note_order("second")]);
    }

    #[test]
    fn a_composition_below_main_is_not_assigned_a_side() {
        let mut draft = Draft::empty("D");
        draft.tracks = vec![
            Track::main("main"),
            lane("comp", TrackType::Composition),
            lane("audio", TrackType::Audio),
        ];
        assert!(repair(&draft).is_empty());
    }

    #[test]
    fn an_overlap_is_named_and_the_segment_list_stays() {
        let mut draft = Draft::empty("D");
        let mut main = Track::main("main");
        main.segments = vec![clip("a", 0, 100), clip("b", 50, 50)];
        draft.tracks = vec![main];
        let before = draft.clone();
        assert_eq!(repair(&draft), vec![note_range("main", "b")]);
        assert_eq!(draft.tracks[0].segments, before.tracks[0].segments);
        assert_eq!(draft, before);
    }

    #[test]
    fn touching_endpoints_are_not_a_conflict() {
        let mut draft = Draft::empty("D");
        let mut main = Track::main("main");
        main.segments = vec![clip("a", 0, 100), clip("b", 100, 50)];
        draft.tracks = vec![main];
        assert!(repair(&draft).is_empty());
    }

    #[test]
    fn a_zero_duration_is_a_range_conflict() {
        let mut draft = Draft::empty("D");
        let mut main = Track::main("main");
        main.segments = vec![clip("a", 0, 100), clip("b", 50, 0)];
        draft.tracks = vec![main];
        let before = draft.clone();
        assert_eq!(repair(&draft), vec![note_range("main", "b")]);
        assert_eq!(draft, before);
    }

    #[test]
    fn a_missing_main_track_still_reports_a_range_conflict() {
        let mut draft = Draft::empty("D");
        let mut audio = lane("audio", TrackType::Audio);
        audio.segments = vec![clip("a", 0, 100), clip("b", 10, 10)];
        draft.tracks = vec![audio];
        assert_eq!(
            repair(&draft),
            vec![RepairNote::MissingMainTrack, note_range("audio", "b")]
        );
    }

    #[test]
    fn a_window_past_the_outgoing_segment_is_reported_and_the_material_stays() {
        let mut draft = Draft::empty("D");
        let mut main = Track::main("main");
        main.segments = vec![clip("a", 0, 100), clip("b", 100, 50)];
        draft.tracks = vec![main];
        with_transition(&mut draft, "a", "blend", 150, false);
        let before = draft.clone();
        assert_eq!(repair(&draft), vec![note_window("main", "a", "blend")]);
        assert_eq!(draft.materials.transitions[0].duration, Micros(150));
        assert_eq!(draft, before);
    }

    #[test]
    fn a_window_that_fills_the_outgoing_segment_is_not_past_it() {
        let mut draft = Draft::empty("D");
        let mut main = Track::main("main");
        main.segments = vec![clip("a", 0, 100), clip("b", 100, 50)];
        draft.tracks = vec![main];
        with_transition(&mut draft, "a", "blend", 100, false);
        assert!(repair(&draft).is_empty());
    }

    #[test]
    fn a_hard_cut_is_not_a_window() {
        let mut draft = Draft::empty("D");
        let mut main = Track::main("main");
        main.segments = vec![clip("a", 0, 100), clip("b", 100, 50)];
        draft.tracks = vec![main];
        with_transition(&mut draft, "a", "blend", 0, false);
        assert!(repair(&draft).is_empty());
    }

    #[test]
    fn a_negative_duration_is_reported_and_stays_negative() {
        let mut draft = Draft::empty("D");
        let mut main = Track::main("main");
        main.segments = vec![clip("a", 0, 100), clip("b", 100, 50)];
        draft.tracks = vec![main];
        with_transition(&mut draft, "a", "blend", -1, false);
        let before = draft.clone();
        assert_eq!(repair(&draft), vec![note_window("main", "a", "blend")]);
        assert_eq!(draft.materials.transitions[0].duration, Micros(-1));
        assert_eq!(draft, before);
    }

    #[test]
    fn an_overlap_that_fits_both_sides_is_not_reported() {
        let mut draft = Draft::empty("D");
        let mut main = Track::main("main");
        main.segments = vec![clip("a", 0, 50), clip("b", 50, 51)];
        draft.tracks = vec![main];
        with_transition(&mut draft, "a", "blend", 101, true);
        assert!(repair(&draft).is_empty());
    }

    #[test]
    fn an_overlap_past_the_incoming_segment_is_reported() {
        let mut draft = Draft::empty("D");
        let mut main = Track::main("main");
        main.segments = vec![clip("a", 0, 50), clip("b", 50, 50)];
        draft.tracks = vec![main];
        with_transition(&mut draft, "a", "blend", 101, true);
        let before = draft.clone();
        assert_eq!(repair(&draft), vec![note_window("main", "a", "blend")]);
        assert!(draft.materials.transitions[0].is_overlap);
        assert_eq!(draft.tracks[0].segments, before.tracks[0].segments);
    }

    #[test]
    fn an_overlap_with_no_incoming_segment_is_reported() {
        let mut draft = Draft::empty("D");
        let mut main = Track::main("main");
        main.segments = vec![clip("a", 0, 100)];
        draft.tracks = vec![main];
        with_transition(&mut draft, "a", "blend", 10, true);
        assert_eq!(repair(&draft), vec![note_window("main", "a", "blend")]);
    }

    #[test]
    fn a_ref_that_is_not_a_transition_is_not_a_window() {
        let mut draft = Draft::empty("D");
        let mut main = Track::main("main");
        let mut segment = clip("a", 0, 100);
        segment.extra_material_refs.push(MaterialId::new("missing"));
        main.segments = vec![segment];
        draft.tracks = vec![main];
        assert!(repair(&draft).is_empty());
    }
}
