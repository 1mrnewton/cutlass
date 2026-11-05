use cutlass_model::{
    CommonKeyframeGroup, Draft, GRAPH_PRESETS, KeyframeGraph, Micros, POSE_KEYS, Segment, SegmentId,
};

use super::locate;
use super::{EditError, ensure_unlocked, placement, rotate};

/// Replace the segment's keyframe groups and leave every other field.
pub fn set_keyframes(
    draft: &mut Draft,
    segment_id: &SegmentId,
    keyframes: &[CommonKeyframeGroup],
) -> Result<(), EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    draft.tracks[track_index].segments[segment_index].common_keyframes = keyframes.to_vec();
    Ok(())
}

/// Where a picture sits, as [`super::placement`] and [`super::rotate`] store it.
#[derive(Clone, Copy, Debug)]
pub struct Pose {
    pub x: f64,
    pub y: f64,
    pub scale: f64,
    pub rotation: f64,
}

/// Key the clip's place, size, turn, and opacity at `offset` from its start.
pub fn add_keyframe(
    draft: &mut Draft,
    segment_id: &SegmentId,
    offset: Micros,
) -> Result<(), EditError> {
    let graphs = draft.keyframe_graph_list.clone();
    let segment = keyed(draft, segment_id, offset)?;
    cutlass_model::add_keys(segment, &graphs, offset).ok_or(EditError::KeyTime)
}

/// Remove every key at `offset` from the clip's start.
pub fn remove_keyframe(
    draft: &mut Draft,
    segment_id: &SegmentId,
    offset: Micros,
) -> Result<(), EditError> {
    let segment = keyed(draft, segment_id, offset)?;
    let graphs: Vec<String> = segment
        .common_keyframes
        .iter()
        .flat_map(|group| &group.keyframe_list)
        .filter(|key| key.time_offset == offset && !key.graph_id.is_empty())
        .map(|key| key.graph_id.clone())
        .collect();
    if !cutlass_model::remove_keys(segment, offset) {
        return Err(EditError::NoKey);
    }
    prune(draft, &graphs);
    Ok(())
}

/// Ease the span that ends at the key `offset` from the clip's start along
/// the Graphs panel curve `graph`, or make it straight with `None`, on each
/// group with a key there and one before it. Each group gets its own copy
/// of the curve, as the reference editor stores it.
pub fn set_keyframe_graph(
    draft: &mut Draft,
    segment_id: &SegmentId,
    offset: Micros,
    graph: Option<&str>,
) -> Result<(), EditError> {
    let preset = match graph {
        Some(name) => Some(
            GRAPH_PRESETS
                .iter()
                .find(|preset| preset.name == name)
                .ok_or(EditError::Graph)?,
        ),
        None => None,
    };
    let segment = keyed(draft, segment_id, offset)?;
    let owner = segment.id.as_str().to_owned();
    let (mut old, mut new, mut spans) = (Vec::new(), Vec::new(), 0);
    for group in &mut segment.common_keyframes {
        let earlier = group
            .keyframe_list
            .iter()
            .any(|key| key.time_offset.0 < offset.0);
        let Some(key) = group
            .keyframe_list
            .iter_mut()
            .find(|key| key.time_offset == offset)
        else {
            continue;
        };
        if !earlier {
            continue;
        }
        spans += 1;
        if !key.graph_id.is_empty() {
            old.push(std::mem::take(&mut key.graph_id));
        }
        if let Some(preset) = preset {
            let id = format!("{owner}-{}-{}-graph", group.property_type, offset.0);
            key.graph_id.clone_from(&id);
            new.push(KeyframeGraph::of(id, preset));
        }
    }
    if spans == 0 {
        return Err(EditError::NoKey);
    }
    let list = &mut draft.keyframe_graph_list;
    for graph in new {
        match list.iter_mut().find(|kept| kept.id == graph.id) {
            Some(kept) => *kept = graph,
            None => list.push(graph),
        }
    }
    prune(draft, &old);
    Ok(())
}

/// Drop the graphs `ids` that no key points at any more.
fn prune(draft: &mut Draft, ids: &[String]) {
    let used: Vec<String> = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .flat_map(|segment| &segment.common_keyframes)
        .flat_map(|group| &group.keyframe_list)
        .map(|key| key.graph_id.clone())
        .filter(|id| ids.contains(id))
        .collect();
    draft
        .keyframe_graph_list
        .retain(|graph| !ids.contains(&graph.id) || used.contains(&graph.id));
}

/// Store the pose on the clip, and key it at `offset` from its start.
pub fn key_pose(
    draft: &mut Draft,
    segment_id: &SegmentId,
    offset: Micros,
    pose: Pose,
) -> Result<(), EditError> {
    keyed(draft, segment_id, offset)?;
    placement::set_position(draft, segment_id, pose.x, pose.y)?;
    placement::set_scale(draft, segment_id, pose.scale)?;
    rotate::set_rotation(draft, segment_id, pose.rotation)?;
    let segment = keyed(draft, segment_id, offset)?;
    let values = [pose.x, pose.y, pose.rotation, pose.scale, pose.scale];
    for (property, value) in POSE_KEYS.iter().zip(values) {
        cutlass_model::put_key(segment, property, offset, value);
    }
    Ok(())
}

/// The unlocked clip `segment_id`, when `offset` is on it.
fn keyed<'a>(
    draft: &'a mut Draft,
    segment_id: &SegmentId,
    offset: Micros,
) -> Result<&'a mut Segment, EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let segment = &mut draft.tracks[track_index].segments[segment_index];
    if offset.0 < 0 || offset.0 > segment.target.duration.0 {
        return Err(EditError::KeyTime);
    }
    Ok(segment)
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, Micros, Segment, TimeRange, key_offsets};

    use crate::edit::EditError;
    use crate::request::Request;
    use crate::session::Session;

    fn open() -> Session {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = "main".into();
        let mut session = Session::open(draft);
        let mut segment = Segment::new("a", "media", TimeRange::new(0, 1_000_000));
        segment.uniform_scale.on = true;
        session
            .apply(
                0,
                &[Request::Place {
                    track_id: "main".into(),
                    segment: Box::new(segment),
                }],
            )
            .unwrap();
        session
    }

    fn pose(offset: i64, x: f64) -> Request {
        Request::KeyPose {
            segment_id: "a".into(),
            offset: Micros(offset),
            x,
            y: 0.0,
            scale: 2.0,
            rotation: 0.0,
        }
    }

    #[test]
    fn a_keyed_pose_moves_the_clip_between_its_keys_and_undoes() {
        let mut session = open();
        let revision = session
            .apply(
                1,
                &[Request::AddKeyframe {
                    segment_id: "a".into(),
                    offset: Micros(0),
                }],
            )
            .unwrap();
        let revision = session.apply(revision, &[pose(500_000, 1.0)]).unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        assert_eq!(key_offsets(segment), [Micros(0), Micros(500_000)]);
        assert_eq!(segment.clip.transform.x, 1.0);
        let (clip, uniform) = cutlass_model::clip_at(segment, &[], Micros(250_000)).unwrap();
        assert_eq!(clip.transform.x, 0.5);
        assert_eq!(uniform.value, 1.5);

        session
            .apply(
                revision,
                &[Request::RemoveKeyframe {
                    segment_id: "a".into(),
                    offset: Micros(0),
                }],
            )
            .unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        assert_eq!(key_offsets(segment), [Micros(500_000)]);
        session.undo().unwrap();
        session.undo().unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        assert_eq!(key_offsets(segment), [Micros(0)]);
        assert_eq!(segment.clip.transform.x, 0.0);
    }

    #[test]
    fn a_split_or_a_start_trim_keeps_each_key_on_its_moment() {
        let mut session = open();
        let revision = session
            .apply(1, &[pose(0, 0.0), pose(800_000, 1.0)])
            .unwrap();
        let x_at = |session: &Session, id: &str, time: i64| {
            let segment = session.draft().tracks[0]
                .segments
                .iter()
                .find(|segment| segment.id.as_str() == id)
                .unwrap();
            cutlass_model::clip_at(segment, &[], Micros(time))
                .unwrap()
                .0
                .transform
                .x
        };
        let before = x_at(&session, "a", 600_000);
        let revision = session
            .apply(
                revision,
                &[Request::Split {
                    segment_id: "a".into(),
                    at: Micros(400_000),
                    right_id: "b".into(),
                }],
            )
            .unwrap();
        assert_eq!(x_at(&session, "b", 600_000), before);
        assert_eq!(x_at(&session, "a", 200_000), 0.25);
        session
            .apply(
                revision,
                &[Request::Trim {
                    segment_id: "b".into(),
                    edge: crate::request::TrimEdge::Start,
                    time: Micros(500_000),
                }],
            )
            .unwrap();
        // Main packs, so the trimmed clip slides back to the cut; the
        // footage a tenth of a second in is still the old 0.6 s.
        let right = &session.draft().tracks[0].segments[1];
        assert_eq!(key_offsets(right), [Micros(-500_000), Micros(300_000)]);
        let start = right.target.start.0;
        assert_eq!(x_at(&session, "b", start + 100_000), before);
    }

    #[test]
    fn a_key_off_the_clip_or_missing_is_refused() {
        let mut session = open();
        assert_eq!(
            session.apply(1, &[pose(1_000_001, 0.0)]),
            Err(EditError::KeyTime)
        );
        assert_eq!(
            session.apply(
                1,
                &[Request::RemoveKeyframe {
                    segment_id: "a".into(),
                    offset: Micros(0),
                }]
            ),
            Err(EditError::NoKey)
        );
    }

    fn graph(offset: i64, graph: Option<&str>) -> Request {
        Request::SetKeyframeGraph {
            segment_id: "a".into(),
            offset: Micros(offset),
            graph: graph.map(str::to_owned),
        }
    }

    #[test]
    fn a_curve_eases_each_pose_group_and_goes_with_its_key() {
        let mut session = open();
        let revision = session
            .apply(
                1,
                &[Request::AddKeyframe {
                    segment_id: "a".into(),
                    offset: Micros(0),
                }],
            )
            .unwrap();
        let revision = session.apply(revision, &[pose(500_000, 1.0)]).unwrap();
        assert_eq!(
            session.apply(revision, &[graph(0, Some("Ease In"))]),
            Err(EditError::NoKey)
        );
        assert_eq!(
            session.apply(revision, &[graph(500_000, Some("Wobble"))]),
            Err(EditError::Graph)
        );
        let revision = session
            .apply(revision, &[graph(500_000, Some("Ease In"))])
            .unwrap();
        let draft = session.draft();
        assert_eq!(draft.keyframe_graph_list.len(), 5);
        assert!(
            draft
                .keyframe_graph_list
                .iter()
                .all(|graph| graph.resource_name == "Ease In")
        );
        let segment = &draft.tracks[0].segments[0];
        let (clip, _) =
            cutlass_model::clip_at(segment, &draft.keyframe_graph_list, Micros(250_000)).unwrap();
        assert!(clip.transform.x < 0.4, "{}", clip.transform.x);

        let revision = session
            .apply(revision, &[graph(500_000, Some("Ease Out"))])
            .unwrap();
        let names: Vec<&str> = session
            .draft()
            .keyframe_graph_list
            .iter()
            .map(|graph| graph.resource_name.as_str())
            .collect();
        assert_eq!(names, ["Ease Out"; 5]);

        session.apply(revision, &[graph(500_000, None)]).unwrap();
        assert!(session.draft().keyframe_graph_list.is_empty());
        session.undo().unwrap();
        assert_eq!(session.draft().keyframe_graph_list.len(), 5);
        session
            .apply(
                session.revision(),
                &[Request::RemoveKeyframe {
                    segment_id: "a".into(),
                    offset: Micros(500_000),
                }],
            )
            .unwrap();
        assert!(session.draft().keyframe_graph_list.is_empty());
    }
}
