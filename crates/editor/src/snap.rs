use cutlass_model::{
    Draft, MaterialRef, MaterialTransition, Micros, Segment, SegmentId, TimeRange,
};

/// What a drag lined up with. The time is on the timeline, in microseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapKind {
    Playhead,
    SegmentEdge,
    Keyframe,
    Marker,
    Beat,
    TransitionMiddle,
    /// The start of the timeline, or its stored duration.
    TimelineEnd,
}

/// A snap reports a time. It does not move a segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnapTarget {
    pub time: Micros,
    pub kind: SnapKind,
}

/// The nearest snap time within `threshold` microseconds, inclusive.
///
/// `ignore` is the segment being dragged. Its edges, keyframes, beats, and
/// transition stay out of the search, because they move with that drag.
/// A negative threshold snaps to nothing.
#[must_use]
pub fn snap(
    draft: &Draft,
    time: Micros,
    playhead: Micros,
    threshold: Micros,
    ignore: Option<&SegmentId>,
) -> Option<SnapTarget> {
    let limit = u64::try_from(threshold.0).ok()?;
    let mut best: Option<Best> = None;
    consider(&mut best, playhead, SnapKind::Playhead, time.0, limit);
    for track in &draft.tracks {
        for segment in &track.segments {
            if ignore.is_some_and(|id| id == &segment.id) {
                continue;
            }
            collect_segment(draft, segment, &mut best, time.0, limit);
        }
    }
    collect_markers(draft, &mut best, time.0, limit);
    consider(&mut best, Micros(0), SnapKind::TimelineEnd, time.0, limit);
    consider(
        &mut best,
        draft.duration,
        SnapKind::TimelineEnd,
        time.0,
        limit,
    );
    best.map(|found| SnapTarget {
        time: found.time,
        kind: found.kind,
    })
}

/// The middle of a transition window: `window.start + duration / 2`.
///
/// A duration of zero or less is a hard cut and has no middle. `is_overlap`
/// false places the window on the outgoing side. `is_overlap` true gives
/// that side `duration / 2`, rounded toward zero.
#[must_use]
pub fn transition_middle(cut: Micros, transition: &MaterialTransition) -> Option<Micros> {
    let duration = transition.duration.0;
    if duration <= 0 {
        return None;
    }
    let window_start = if transition.is_overlap {
        cut.0.checked_sub(duration / 2)?
    } else {
        cut.0.checked_sub(duration)?
    };
    window_start.checked_add(duration / 2).map(Micros)
}

struct Best {
    time: Micros,
    kind: SnapKind,
    distance: u64,
}

fn consider(best: &mut Option<Best>, time: Micros, kind: SnapKind, query: i64, limit: u64) {
    let distance = time.0.abs_diff(query);
    if distance > limit {
        return;
    }
    let replace = match best {
        None => true,
        Some(current) => {
            distance < current.distance
                || (distance == current.distance && kind.rank() < current.kind.rank())
                || (distance == current.distance
                    && kind.rank() == current.kind.rank()
                    && time.0 < current.time.0)
        }
    };
    if replace {
        *best = Some(Best {
            time,
            kind,
            distance,
        });
    }
}

impl SnapKind {
    fn rank(self) -> u8 {
        match self {
            Self::Playhead => 0,
            Self::SegmentEdge => 1,
            Self::Keyframe => 2,
            Self::Marker => 3,
            Self::Beat => 4,
            Self::TransitionMiddle => 5,
            Self::TimelineEnd => 6,
        }
    }
}

fn collect_segment(
    draft: &Draft,
    segment: &Segment,
    best: &mut Option<Best>,
    query: i64,
    limit: u64,
) {
    let mut note = |time: Micros, kind: SnapKind| consider(best, time, kind, query, limit);
    if let Some(end) = positive_end(&segment.target) {
        note(segment.target.start, SnapKind::SegmentEdge);
        note(Micros(end), SnapKind::SegmentEdge);
        collect_keyframes(segment, &mut note);
        collect_beats(draft, segment, &mut note);
        collect_transition(draft, segment, Micros(end), &mut note);
    }
}

fn collect_keyframes(segment: &Segment, note: &mut impl FnMut(Micros, SnapKind)) {
    let duration = segment.target.duration.0;
    for group in &segment.common_keyframes {
        for key in &group.keyframe_list {
            if key.time_offset.0 < 0 || key.time_offset.0 >= duration {
                continue;
            }
            if let Some(time) = segment.target.start.0.checked_add(key.time_offset.0) {
                note(Micros(time), SnapKind::Keyframe);
            }
        }
    }
}

fn collect_beats(draft: &Draft, segment: &Segment, note: &mut impl FnMut(Micros, SnapKind)) {
    for id in &segment.extra_material_refs {
        let Some(MaterialRef::Other(raw)) = draft.materials.get(id.as_str()) else {
            continue;
        };
        if raw.kind != "beats" {
            continue;
        }
        let Some(beats) = raw.fields.get("user_beats") else {
            continue;
        };
        each_integer(beats, &mut |beat| {
            if let Some(time) = beat_on_timeline(segment, beat) {
                note(time, SnapKind::Beat);
            }
        });
    }
}

fn collect_transition(
    draft: &Draft,
    segment: &Segment,
    cut: Micros,
    note: &mut impl FnMut(Micros, SnapKind),
) {
    for id in &segment.extra_material_refs {
        let Some(MaterialRef::Transition(transition)) = draft.materials.get(id.as_str()) else {
            continue;
        };
        if let Some(middle) = transition_middle(cut, transition) {
            note(middle, SnapKind::TransitionMiddle);
        }
    }
}

fn collect_markers(draft: &Draft, best: &mut Option<Best>, query: i64, limit: u64) {
    let mut note = |time: Micros| consider(best, time, SnapKind::Marker, query, limit);
    for material in &draft.materials.time_marks {
        mark_items(material.fields.get("mark_items"), &mut note);
    }
    let Some(root) = draft.rest.get("time_marks") else {
        return;
    };
    if let Some(items) = root.as_array() {
        for item in items {
            if item.get("mark_items").is_some() {
                mark_items(item.get("mark_items"), &mut note);
            } else {
                one_mark(item, &mut note);
            }
        }
        return;
    }
    if root.as_object().is_some() {
        mark_items(root.get("mark_items"), &mut note);
    }
}

fn mark_items(value: Option<&serde_json::Value>, note: &mut impl FnMut(Micros)) {
    let Some(items) = value.and_then(serde_json::Value::as_array) else {
        return;
    };
    for item in items {
        one_mark(item, note);
    }
}

fn one_mark(item: &serde_json::Value, note: &mut impl FnMut(Micros)) {
    let Some(start) = item
        .get("time_range")
        .and_then(|range| range.get("start"))
        .and_then(serde_json::Value::as_i64)
    else {
        return;
    };
    note(Micros(start));
}

fn each_integer(value: &serde_json::Value, note: &mut impl FnMut(i64)) {
    if let Some(beat) = value.as_i64() {
        note(beat);
        return;
    }
    let Some(items) = value.as_array() else {
        return;
    };
    for item in items {
        if let Some(beat) = item.as_i64() {
            note(beat);
        }
    }
}

/// A beat is a source time. It lands on the timeline at the segment start
/// plus that offset through the source range. Speed other than 1 divides
/// the source offset, toward zero.
fn beat_on_timeline(segment: &Segment, beat: i64) -> Option<Micros> {
    let (source_start, source_end) = if let Some(source) = &segment.source_range {
        let end = positive_end(source)?;
        (source.start.0, end)
    } else {
        (0, segment.target.duration.0)
    };
    if beat < source_start || beat >= source_end {
        return None;
    }
    let delta = beat.checked_sub(source_start)?;
    let delta_target = if segment.speed == 1.0 {
        delta
    } else {
        speed_offset(delta, segment.speed)?
    };
    segment.target.start.0.checked_add(delta_target).map(Micros)
}

fn speed_offset(delta: i64, speed: f64) -> Option<i64> {
    if delta < 0 || !speed.is_finite() || speed <= 0.0 {
        return None;
    }
    let mapped = (delta as f64) / speed;
    if !mapped.is_finite() || mapped < 0.0 || mapped >= (1_u64 << 63) as f64 {
        return None;
    }
    Some(mapped.trunc() as i64)
}

fn positive_end(range: &TimeRange) -> Option<i64> {
    if range.duration.0 <= 0 {
        return None;
    }
    range.start.0.checked_add(range.duration.0)
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        CommonKeyframe, CommonKeyframeGroup, Draft, MaterialId, MaterialTransition, RawMaterial,
        Segment, TimeRange, TrackId,
    };

    use super::*;

    fn segment(id: &str, start: i64, duration: i64) -> Segment {
        Segment::new(id, "media", TimeRange::new(start, duration))
    }

    fn main_with(segment: Segment) -> Draft {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        draft.duration = segment.target.end();
        draft.tracks[0].segments.push(segment);
        draft
    }

    #[test]
    fn snap_reports_an_edge_and_leaves_the_segment() {
        let draft = main_with(segment("clip", 0, 1_000_000));
        let before = draft.clone();
        let target = snap(
            &draft,
            Micros(1_100_000),
            Micros(9_000_000),
            Micros(200_000),
            None,
        )
        .unwrap();
        assert_eq!(draft, before);
        assert_eq!(target.time, Micros(1_000_000));
        assert_eq!(target.kind, SnapKind::SegmentEdge);
        assert_eq!(draft.tracks[0].segments[0].target.start, Micros(0));
    }

    #[test]
    fn a_time_past_the_threshold_does_not_snap() {
        let draft = main_with(segment("clip", 0, 1_000_000));
        assert!(
            snap(
                &draft,
                Micros(1_500_000),
                Micros(9_000_000),
                Micros(100),
                None,
            )
            .is_none()
        );
        assert!(snap(&draft, Micros(1_000_000), Micros(0), Micros(-1), None,).is_none());
    }

    #[test]
    fn the_playhead_wins_a_tie_with_an_edge() {
        let draft = main_with(segment("clip", 0, 1_000));
        let target = snap(&draft, Micros(1_000), Micros(1_000), Micros(0), None).unwrap();
        assert_eq!(target.kind, SnapKind::Playhead);
        assert_eq!(target.time, Micros(1_000));
    }

    #[test]
    fn an_equal_distance_picks_the_earlier_edge() {
        let draft = main_with(segment("clip", 0, 1_000));
        let target = snap(&draft, Micros(500), Micros(9_000), Micros(500), None).unwrap();
        assert_eq!(target.time, Micros(0));
        assert_eq!(target.kind, SnapKind::SegmentEdge);
    }

    #[test]
    fn the_dragged_segment_is_not_a_target() {
        let mut draft = main_with(segment("kept", 0, 1_000_000));
        draft.tracks[0]
            .segments
            .push(segment("dragged", 3_000_000, 1_000_000));
        draft.duration = Micros(4_000_000);
        let target = snap(
            &draft,
            Micros(1_100_000),
            Micros(9_000_000),
            Micros(200_000),
            Some(&SegmentId::new("dragged")),
        )
        .unwrap();
        assert_eq!(target.time, Micros(1_000_000));
        assert_eq!(target.kind, SnapKind::SegmentEdge);
    }

    #[test]
    fn the_middle_of_a_transition_follows_the_window() {
        let mut on_outgoing = MaterialTransition {
            duration: Micros(1_001),
            ..MaterialTransition::default()
        };
        assert_eq!(
            transition_middle(Micros(5_000), &on_outgoing),
            Some(Micros(4_499))
        );
        on_outgoing.is_overlap = true;
        assert_eq!(
            transition_middle(Micros(5_000), &on_outgoing),
            Some(Micros(5_000))
        );
        on_outgoing.duration = Micros(0);
        assert!(transition_middle(Micros(5_000), &on_outgoing).is_none());
        on_outgoing.duration = Micros(-4);
        assert!(transition_middle(Micros(5_000), &on_outgoing).is_none());

        let transition = MaterialTransition {
            id: MaterialId::new("blend"),
            duration: Micros(1_000),
            ..MaterialTransition::default()
        };
        let mut clip = segment("out", 0, 10_000);
        clip.extra_material_refs.push(transition.id.clone());
        let mut draft = main_with(clip);
        draft.materials.transitions.push(transition);
        let target = snap(&draft, Micros(9_500), Micros(0), Micros(0), None).unwrap();
        assert_eq!(target.time, Micros(9_500));
        assert_eq!(target.kind, SnapKind::TransitionMiddle);
        assert_eq!(draft.tracks[0].segments[0].target.end(), Micros(10_000));
    }

    #[test]
    fn a_keyframe_a_marker_and_a_beat_are_targets() {
        let mut clip = segment("clip", 1_000_000, 1_000_000);
        clip.source_range = Some(TimeRange::new(0, 1_000_000));
        clip.common_keyframes.push(CommonKeyframeGroup {
            keyframe_list: vec![CommonKeyframe {
                time_offset: Micros(400_000),
                ..CommonKeyframe::default()
            }],
            ..CommonKeyframeGroup::default()
        });
        clip.extra_material_refs.push(MaterialId::new("beat"));
        let mut draft = main_with(clip);
        draft.materials.beats.push(RawMaterial {
            id: MaterialId::new("beat"),
            kind: "beats".to_owned(),
            fields: serde_json::from_str(r#"{"user_beats":[200000]}"#).unwrap(),
        });
        draft.materials.time_marks.push(RawMaterial {
            id: MaterialId::new("marks"),
            kind: "time_mark".to_owned(),
            fields: serde_json::from_str(
                r##"{"mark_items":[{"time_range":{"start":250000,"duration":10},"color":"#ffffff","title":"note"}]}"##,
            )
            .unwrap(),
        });

        let key = snap(&draft, Micros(1_400_100), Micros(0), Micros(200), None).unwrap();
        assert_eq!(key.time, Micros(1_400_000));
        assert_eq!(key.kind, SnapKind::Keyframe);

        let mark = snap(&draft, Micros(250_050), Micros(0), Micros(100), None).unwrap();
        assert_eq!(mark.time, Micros(250_000));
        assert_eq!(mark.kind, SnapKind::Marker);

        let beat = snap(&draft, Micros(1_200_000), Micros(0), Micros(0), None).unwrap();
        assert_eq!(beat.time, Micros(1_200_000));
        assert_eq!(beat.kind, SnapKind::Beat);
    }

    #[test]
    fn a_beat_at_double_speed_uses_half_the_source_offset() {
        let mut clip = segment("clip", 1_000_000, 500_000);
        clip.speed = 2.0;
        clip.source_range = Some(TimeRange::new(0, 1_000_000));
        clip.extra_material_refs.push(MaterialId::new("beat"));
        let mut draft = main_with(clip);
        draft.materials.beats.push(RawMaterial {
            id: MaterialId::new("beat"),
            kind: "beats".to_owned(),
            fields: serde_json::from_str(r#"{"user_beats":200000}"#).unwrap(),
        });
        let beat = snap(&draft, Micros(1_100_000), Micros(0), Micros(0), None).unwrap();
        assert_eq!(beat.time, Micros(1_100_000));
        assert_eq!(beat.kind, SnapKind::Beat);
    }

    #[test]
    fn the_timeline_end_is_a_target() {
        let mut draft = Draft::empty("D");
        draft.duration = Micros(5_000_000);
        let target = snap(&draft, Micros(4_900_000), Micros(0), Micros(200_000), None).unwrap();
        assert_eq!(target.time, Micros(5_000_000));
        assert_eq!(target.kind, SnapKind::TimelineEnd);
    }

    #[test]
    fn the_last_segment_edge_is_still_found() {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        for index in 0..64 {
            draft.tracks[0]
                .segments
                .push(segment(&format!("s-{index}"), index * 20, 10));
        }
        let target = snap(&draft, Micros(1_270), Micros(0), Micros(0), None).unwrap();
        assert_eq!(target.time, Micros(1_270));
        assert_eq!(target.kind, SnapKind::SegmentEdge);
    }
}
