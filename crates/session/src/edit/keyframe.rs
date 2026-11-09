use cutlass_model::{
    CommonKeyframeGroup, Draft, GRAPH_PRESETS, KeyframeGraph, Micros, OPACITY_KEYS, POSITION_KEYS,
    ROTATION_KEYS, SCALE_KEYS, Segment, SegmentId, has_keys, is_keyable, is_pose_key,
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

/// Remove the pose keys at `offset` from the clip's start.
pub fn remove_keyframe(
    draft: &mut Draft,
    segment_id: &SegmentId,
    offset: Micros,
) -> Result<(), EditError> {
    let segment = keyed(draft, segment_id, offset)?;
    let graphs: Vec<String> = segment
        .common_keyframes
        .iter()
        .filter(|group| is_pose_key(&group.property_type))
        .flat_map(|group| &group.keyframe_list)
        .filter(|key| key.time_offset == offset && !key.graph_id.is_empty())
        .map(|key| key.graph_id.clone())
        .collect();
    if !cutlass_model::remove_keys_where(segment, offset, is_pose_key) {
        return Err(EditError::NoKey);
    }
    prune(draft, &graphs);
    Ok(())
}

/// Ease the span that ends at the key `offset` from the clip's start along
/// the Graphs panel curve `graph`, or make it straight with `None`, on each
/// pose group with a key there and one before it. Each group gets its own copy
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
        if !is_pose_key(&group.property_type) {
            continue;
        }
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

/// Change the clip's pose at `offset` from its start to `pose`. Each part
/// that changed (position, scale, rotation) is keyed there when it already
/// has keys, and stored on the clip when it has none. A part that did not
/// change is left alone, so moving a clip never keys or stores its scale.
pub fn key_pose(
    draft: &mut Draft,
    segment_id: &SegmentId,
    offset: Micros,
    pose: Pose,
) -> Result<(), EditError> {
    if !pose.x.is_finite() || !pose.y.is_finite() {
        return Err(EditError::Position);
    }
    if !pose.scale.is_finite() || pose.scale <= 0.0 {
        return Err(EditError::Scale);
    }
    if !pose.rotation.is_finite() {
        return Err(EditError::Angle);
    }
    let graphs = draft.keyframe_graph_list.clone();
    let segment = keyed(draft, segment_id, offset)?;
    let at = Micros(segment.target.start.0.saturating_add(offset.0));
    let (clip, uniform) = cutlass_model::clip_at(segment, &graphs, at).ok_or(EditError::KeyTime)?;
    let was_scale = if uniform.on {
        uniform.value
    } else {
        clip.scale.x
    };
    let same = |a: f64, b: f64| (a - b).abs() <= 1e-9;
    let moved = !same(clip.transform.x, pose.x) || !same(clip.transform.y, pose.y);
    let sized = !same(was_scale.abs(), pose.scale);
    let turned = !same(
        clip.rotation.rem_euclid(360.0),
        pose.rotation.rem_euclid(360.0),
    );
    let keyed_place = has_keys(segment, &POSITION_KEYS);
    let keyed_size = has_keys(segment, &SCALE_KEYS);
    let keyed_turn = has_keys(segment, &ROTATION_KEYS);
    if moved && keyed_place {
        put_part(segment, Part::Position, offset, pose);
    }
    if sized && keyed_size {
        put_part(segment, Part::Scale, offset, pose);
    }
    if turned && keyed_turn {
        put_part(segment, Part::Rotation, offset, pose);
    }
    if moved && !keyed_place {
        placement::set_position(draft, segment_id, pose.x, pose.y)?;
    }
    if sized && !keyed_size {
        placement::set_scale(draft, segment_id, pose.scale)?;
    }
    if turned && !keyed_turn {
        rotate::set_rotation(draft, segment_id, pose.rotation)?;
    }
    Ok(())
}

/// A part of the pose [`key_pose`] keys.
#[derive(Clone, Copy)]
enum Part {
    Position,
    Scale,
    Rotation,
}

/// Key `part` of `pose` at `offset`. Scale keys both axes, and the name
/// that drives both when the clip already keys it.
fn put_part(segment: &mut Segment, part: Part, offset: Micros, pose: Pose) {
    match part {
        Part::Position => {
            cutlass_model::put_key(segment, POSITION_KEYS[0], offset, pose.x);
            cutlass_model::put_key(segment, POSITION_KEYS[1], offset, pose.y);
        }
        Part::Scale => {
            cutlass_model::put_key(segment, SCALE_KEYS[0], offset, pose.scale);
            cutlass_model::put_key(segment, SCALE_KEYS[1], offset, pose.scale);
            if has_keys(segment, &SCALE_KEYS[2..]) {
                cutlass_model::put_key(segment, SCALE_KEYS[2], offset, pose.scale);
            }
        }
        Part::Rotation => {
            cutlass_model::put_key(segment, ROTATION_KEYS[0], offset, pose.rotation);
        }
    }
}

/// Key each of `values` (a property and its value) at `offset` from the
/// clip's start. Only properties a reader plays are taken.
pub fn key_values(
    draft: &mut Draft,
    segment_id: &SegmentId,
    offset: Micros,
    values: &[(String, f64)],
) -> Result<(), EditError> {
    if values.is_empty()
        || values
            .iter()
            .any(|(property, value)| !is_keyable(property) || !in_range(property, *value))
    {
        return Err(EditError::KeyValue);
    }
    let segment = keyed(draft, segment_id, offset)?;
    for (property, value) in values {
        cutlass_model::put_key(segment, property, offset, *value);
    }
    Ok(())
}

/// Whether `value` is one `property` can hold, as the edit that stores it
/// takes: a scale above zero, an opacity from 0 to 1, anything else finite.
fn in_range(property: &str, value: f64) -> bool {
    if !value.is_finite() {
        return false;
    }
    if SCALE_KEYS.contains(&property) {
        return value > 0.0;
    }
    if OPACITY_KEYS.contains(&property) {
        return (0.0..=1.0).contains(&value);
    }
    true
}

/// Take the keys on `properties` at `offset` from the clip's start off.
pub fn unkey_values(
    draft: &mut Draft,
    segment_id: &SegmentId,
    offset: Micros,
    properties: &[String],
) -> Result<(), EditError> {
    if properties.is_empty() || properties.iter().any(|property| !is_keyable(property)) {
        return Err(EditError::KeyValue);
    }
    let segment = keyed(draft, segment_id, offset)?;
    let graphs: Vec<String> = segment
        .common_keyframes
        .iter()
        .filter(|group| properties.contains(&group.property_type))
        .flat_map(|group| &group.keyframe_list)
        .filter(|key| key.time_offset == offset && !key.graph_id.is_empty())
        .map(|key| key.graph_id.clone())
        .collect();
    if !cutlass_model::remove_keys_where(segment, offset, |property| {
        properties.iter().any(|wanted| wanted == property)
    }) {
        return Err(EditError::NoKey);
    }
    prune(draft, &graphs);
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
    use cutlass_model::{Micros, Segment, TimeRange, key_offsets};

    use crate::edit::EditError;
    use crate::request::Request;
    use crate::session::Session;

    fn open() -> Session {
        let mut draft = crate::edit::test_draft("D");
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
        // A keyed part is keyed, not stored.
        assert_eq!(segment.clip.transform.x, 0.0);
        let (clip, uniform) = cutlass_model::clip_at(segment, &[], Micros(250_000)).unwrap();
        assert_eq!(clip.transform.x, 0.5);
        assert_eq!(uniform.value, 1.5);
        // The turn and opacity did not change, so they get no key there.
        for name in ["KFTypeRotation", "KFTypeAlpha"] {
            let group = segment
                .common_keyframes
                .iter()
                .find(|group| group.property_type == name)
                .unwrap();
            assert_eq!(group.keyframe_list.len(), 1, "{name}");
        }

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
            .apply(
                1,
                &[
                    values(0, &[("KFTypePositionX", 0.0)]),
                    values(800_000, &[("KFTypePositionX", 1.0)]),
                ],
            )
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
    fn keying_position_leaves_scale_and_the_turn_unkeyed() {
        let mut session = open();
        let revision = session
            .apply(
                1,
                &[
                    values(0, &[("KFTypePositionX", 0.0), ("KFTypePositionY", 0.0)]),
                    values(
                        800_000,
                        &[("KFTypePositionX", 1.0), ("KFTypePositionY", 0.0)],
                    ),
                ],
            )
            .unwrap();
        // A drag that moves the clip and resizes it: the move is keyed, the
        // new size is stored, and the turn is left alone.
        let revision = session
            .apply(
                revision,
                &[Request::KeyPose {
                    segment_id: "a".into(),
                    offset: Micros(400_000),
                    x: -0.5,
                    y: 0.25,
                    scale: 3.0,
                    rotation: 0.0,
                }],
            )
            .unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        let names: Vec<&str> = segment
            .common_keyframes
            .iter()
            .map(|group| group.property_type.as_str())
            .collect();
        assert_eq!(names, ["KFTypePositionX", "KFTypePositionY"]);
        assert_eq!(
            key_offsets(segment),
            [Micros(0), Micros(400_000), Micros(800_000)]
        );
        assert_eq!(segment.uniform_scale.value, 3.0);
        assert_eq!(segment.clip.transform.x, 0.0);
        // The scale is the same all along the clip.
        for time in [0, 200_000, 600_000, 1_000_000] {
            let (_, uniform) = cutlass_model::clip_at(segment, &[], Micros(time)).unwrap();
            assert_eq!(uniform.value, 3.0);
        }
        // Resizing alone adds no position key.
        session
            .apply(
                revision,
                &[Request::KeyPose {
                    segment_id: "a".into(),
                    offset: Micros(600_000),
                    x: 0.25,
                    y: 0.125,
                    scale: 2.0,
                    rotation: 0.0,
                }],
            )
            .unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        assert_eq!(key_offsets(segment).len(), 3);
        assert_eq!(segment.uniform_scale.value, 2.0);
    }

    #[test]
    fn a_pose_key_out_of_range_is_refused() {
        let mut session = open();
        for (name, value) in [
            ("KFTypeScaleX", 0.0),
            ("KFTypeScale", -1.0),
            ("KFTypeAlpha", 1.5),
            ("alpha", -0.1),
            ("KFTypeRotation", f64::NAN),
        ] {
            assert_eq!(
                session.apply(1, &[values(0, &[(name, value)])]),
                Err(EditError::KeyValue),
                "{name}"
            );
        }
        assert_eq!(
            session.apply(1, &[pose(0, f64::INFINITY)]),
            Err(EditError::Position)
        );
        assert!(
            session.draft().tracks[0].segments[0]
                .common_keyframes
                .is_empty()
        );
        session
            .apply(
                1,
                &[values(
                    0,
                    &[("KFTypeAlpha", 0.0), ("KFTypeRotation", 720.0)],
                )],
            )
            .unwrap();
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
        // Position and scale changed at the second key; the turn did not.
        assert_eq!(draft.keyframe_graph_list.len(), 4);
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
        assert_eq!(names, ["Ease Out"; 4]);

        session.apply(revision, &[graph(500_000, None)]).unwrap();
        assert!(session.draft().keyframe_graph_list.is_empty());
        session.undo().unwrap();
        assert_eq!(session.draft().keyframe_graph_list.len(), 4);
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

    fn values(offset: i64, values: &[(&str, f64)]) -> Request {
        Request::KeyValues {
            segment_id: "a".into(),
            offset: Micros(offset),
            values: values.iter().map(|(p, v)| ((*p).to_owned(), *v)).collect(),
        }
    }

    #[test]
    fn named_values_are_keyed_and_taken_off_apart_from_the_pose() {
        let mut session = open();
        let x = cutlass_model::MASK_KEYS[0];
        let revision = session
            .apply(
                1,
                &[
                    values(0, &[(x, -0.5), ("KFTypeVolume", 1.0)]),
                    values(800_000, &[(x, 0.5), ("KFTypeVolume", 0.25)]),
                    Request::AddKeyframe {
                        segment_id: "a".into(),
                        offset: Micros(800_000),
                    },
                ],
            )
            .unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        let at = |segment: &Segment, name: &str, offset: i64| {
            cutlass_model::sample(&segment.common_keyframes, &[], &[name], offset)
                .flatten()
                .unwrap()
        };
        assert!((at(segment, x, 400_000) - 0.0).abs() < 1e-9);
        assert!((at(segment, "KFTypeVolume", 400_000) - 0.625).abs() < 1e-9);
        // A key already there takes the new value.
        let revision = session
            .apply(revision, &[values(800_000, &[(x, 0.75)])])
            .unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        assert_eq!(at(segment, x, 800_000), 0.75);

        // The pose's ◆ takes only the pose off.
        let revision = session
            .apply(
                revision,
                &[Request::RemoveKeyframe {
                    segment_id: "a".into(),
                    offset: Micros(800_000),
                }],
            )
            .unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        assert!(cutlass_model::key_offsets_where(segment, cutlass_model::is_pose_key).is_empty());
        assert_eq!(key_offsets(segment), [Micros(0), Micros(800_000)]);

        session
            .apply(
                revision,
                &[Request::UnkeyValues {
                    segment_id: "a".into(),
                    offset: Micros(800_000),
                    properties: vec![x.to_owned()],
                }],
            )
            .unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        assert_eq!(at(segment, x, 800_000), -0.5, "only the first key is left");
        assert_eq!(at(segment, "KFTypeVolume", 800_000), 0.25);
    }

    #[test]
    fn a_value_nothing_plays_or_off_the_clip_is_refused() {
        let mut session = open();
        for (request, error) in [
            (values(0, &[("KFTypeBrightness", 0.5)]), EditError::KeyValue),
            (
                values(0, &[("KFTypeVolume", f64::NAN)]),
                EditError::KeyValue,
            ),
            (values(0, &[]), EditError::KeyValue),
            (
                values(2_000_000, &[("KFTypeVolume", 1.0)]),
                EditError::KeyTime,
            ),
            (
                Request::UnkeyValues {
                    segment_id: "a".into(),
                    offset: Micros(0),
                    properties: vec!["KFTypeVolume".to_owned()],
                },
                EditError::NoKey,
            ),
        ] {
            assert_eq!(session.apply(1, &[request]), Err(error));
        }
    }
}
