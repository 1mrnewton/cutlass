//! Load-time report.
//!
//! Repair names a missing main track, a bad track order, a range conflict, a
//! transition window that runs past a segment, or a speed whose target
//! duration is not the mapped source duration. It does not move segments,
//! and it does not change the transition material or the speed points.
//!
//! File order is the row order, from the top of the timeline to the bottom.
//! Audio is the bottom block. The main track sits just above it. Overlay,
//! sticker, text, effect, filter, adjust, and plugin effect sit above the
//! main track. A track on the wrong side of that main track is a bad order.
//! A second main track is a bad order. Kinds outside those two lists are not
//! given a side.

use std::collections::BTreeSet;

use cutlass_model::{
    ConstantSpeedError, Draft, MaterialId, MaterialRef, MaterialSpeed, MaterialTransition, Micros,
    Segment, SegmentId, SpeedMode, SpeedPoint, Track, TrackId, TrackType, constant_target_duration,
    curve_speed_factor,
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
    /// The stored target duration is not the source duration mapped by speed.
    /// The segment and the speed points are left as stored.
    SpeedDuration {
        track_id: TrackId,
        segment_id: SegmentId,
    },
    /// Curve mode has fewer than two usable points, so the constant speed number stays in force.
    CurveUnused { material_id: MaterialId },
    /// A curve point lies outside x in 0 to 1. The point stays stored.
    SpeedPointOutside { material_id: MaterialId },
    /// A curve point has a speed at or below 0, or a non-finite coordinate.
    SpeedPoint { material_id: MaterialId },
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
        speed_durations(draft, track, &mut notes);
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
    let Some((left, right)) = cutlass_model::blend_extents(Micros(duration), transition.is_overlap)
    else {
        return false;
    };
    let left = left.0;
    let right = right.0;
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

/// Names a duration the speed mapping does not produce. Points stay stored.
fn speed_durations(draft: &Draft, track: &Track, notes: &mut Vec<RepairNote>) {
    for segment in &track.segments {
        if let Some(speed) = curve_material(draft, segment) {
            report_curve(track, segment, speed, notes);
        } else if let Some(source) = &segment.source_range {
            report_constant(track, segment, source.duration, segment.speed, notes);
        }
    }
}

fn curve_material<'a>(draft: &'a Draft, segment: &Segment) -> Option<&'a MaterialSpeed> {
    segment.extra_material_refs.iter().find_map(|material_id| {
        match draft.materials.get(material_id.as_str()) {
            Some(MaterialRef::Speed(speed)) if speed.mode == SpeedMode::Curve => Some(speed),
            _ => None,
        }
    })
}

fn report_curve(
    track: &Track,
    segment: &Segment,
    speed: &MaterialSpeed,
    notes: &mut Vec<RepairNote>,
) {
    let points = speed
        .curve_speed
        .as_ref()
        .map(|curve| curve.speed_points.as_slice())
        .unwrap_or(&[]);
    for point in points {
        if point.x.is_finite() && !(0.0..=1.0).contains(&point.x) {
            notes.push(RepairNote::SpeedPointOutside {
                material_id: speed.id.clone(),
            });
        }
    }
    match curve_speed_factor(points) {
        Err(ConstantSpeedError::TooFewPoints) => {
            notes.push(RepairNote::CurveUnused {
                material_id: speed.id.clone(),
            });
        }
        Err(ConstantSpeedError::NonPositiveSpeed | ConstantSpeedError::Unrepresentable) => {
            notes.push(RepairNote::SpeedPoint {
                material_id: speed.id.clone(),
            });
            return;
        }
        Ok(_) | Err(ConstantSpeedError::NonPositiveDuration | ConstantSpeedError::Curve) => {}
    }
    let Some(source) = &segment.source_range else {
        return;
    };
    let mapped = if matches!(
        curve_speed_factor(points),
        Err(ConstantSpeedError::TooFewPoints)
    ) {
        constant_target_duration(source.duration, segment.speed).ok()
    } else {
        curve_target_duration_ok(source.duration, points)
    };
    if let Some(mapped) = mapped
        && mapped != segment.target.duration
    {
        notes.push(RepairNote::SpeedDuration {
            track_id: track.id.clone(),
            segment_id: segment.id.clone(),
        });
    }
}

fn curve_target_duration_ok(source: Micros, points: &[SpeedPoint]) -> Option<Micros> {
    cutlass_model::curve_target_duration(source, points).ok()
}

fn report_constant(
    track: &Track,
    segment: &Segment,
    source: Micros,
    speed: f64,
    notes: &mut Vec<RepairNote>,
) {
    let disagrees = match constant_target_duration(source, speed) {
        Ok(mapped) => mapped != segment.target.duration,
        Err(ConstantSpeedError::NonPositiveSpeed | ConstantSpeedError::Unrepresentable) => true,
        Err(
            ConstantSpeedError::NonPositiveDuration
            | ConstantSpeedError::Curve
            | ConstantSpeedError::TooFewPoints,
        ) => false,
    };
    if disagrees {
        notes.push(RepairNote::SpeedDuration {
            track_id: track.id.clone(),
            segment_id: segment.id.clone(),
        });
    }
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

    #[test]
    fn a_constant_speed_duration_that_disagrees_is_reported_and_left_stored() {
        let mut draft = Draft::empty("D");
        let mut main = Track::main("main");
        let mut wrong = clip("a", 0, 2_000_000);
        wrong.source_range = Some(TimeRange::new(0, 2_000_000));
        wrong.speed = 2.0;
        let mut matched = clip("b", 2_000_000, 1_000_000);
        matched.source_range = Some(TimeRange::new(0, 2_000_000));
        matched.speed = 2.0;
        let mut curved = clip("c", 3_000_000, 2_000_000);
        curved.source_range = Some(TimeRange::new(0, 2_000_000));
        curved.speed = 2.0;
        curved.extra_material_refs.push(MaterialId::new("curve"));
        main.segments = vec![wrong, matched, curved];
        draft.tracks = vec![main];
        draft.materials.speeds.push(cutlass_model::MaterialSpeed {
            id: MaterialId::new("curve"),
            mode: cutlass_model::SpeedMode::Curve,
            speed: 2.0,
            ..cutlass_model::MaterialSpeed::default()
        });
        let before = draft.clone();
        assert_eq!(
            repair(&draft),
            vec![
                RepairNote::SpeedDuration {
                    track_id: TrackId::new("main"),
                    segment_id: SegmentId::new("a"),
                },
                RepairNote::CurveUnused {
                    material_id: MaterialId::new("curve"),
                },
                RepairNote::SpeedDuration {
                    track_id: TrackId::new("main"),
                    segment_id: SegmentId::new("c"),
                },
            ]
        );
        assert_eq!(draft, before);
    }

    #[test]
    fn a_curve_disagreement_is_reported_and_the_points_stay() {
        let mut draft = Draft::empty("D");
        let mut main = Track::main("main");
        let mut segment = clip("a", 0, 500_000);
        segment.source_range = Some(TimeRange::new(0, 1_000_000));
        segment.speed = 2.0;
        segment.extra_material_refs.push(MaterialId::new("curve"));
        main.segments = vec![segment];
        draft.tracks = vec![main];
        draft.materials.speeds.push(cutlass_model::MaterialSpeed {
            id: MaterialId::new("curve"),
            mode: cutlass_model::SpeedMode::Curve,
            speed: 2.0,
            curve_speed: Some(cutlass_model::CurveSpeed {
                speed_points: vec![
                    cutlass_model::SpeedPoint {
                        x: -0.25,
                        y: 1.0,
                        rest: serde_json::Map::new(),
                    },
                    cutlass_model::SpeedPoint {
                        x: 1.25,
                        y: 1.0,
                        rest: serde_json::Map::new(),
                    },
                ],
                ..cutlass_model::CurveSpeed::default()
            }),
            ..cutlass_model::MaterialSpeed::default()
        });
        let before = draft.clone();
        assert_eq!(
            repair(&draft),
            vec![
                RepairNote::SpeedPointOutside {
                    material_id: MaterialId::new("curve"),
                },
                RepairNote::SpeedPointOutside {
                    material_id: MaterialId::new("curve"),
                },
                RepairNote::SpeedDuration {
                    track_id: TrackId::new("main"),
                    segment_id: SegmentId::new("a"),
                },
            ]
        );
        assert_eq!(draft, before);
        assert_eq!(
            draft.materials.speeds[0]
                .curve_speed
                .as_ref()
                .map(|curve| curve.speed_points.len()),
            Some(2)
        );
    }
}
