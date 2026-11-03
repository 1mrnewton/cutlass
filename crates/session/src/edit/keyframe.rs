use cutlass_model::{CommonKeyframeGroup, Draft, Micros, POSE_KEYS, Segment, SegmentId};

use super::speed::locate;
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
    let segment = keyed(draft, segment_id, offset)?;
    cutlass_model::add_keys(segment, offset).ok_or(EditError::KeyTime)
}

/// Remove every key at `offset` from the clip's start.
pub fn remove_keyframe(
    draft: &mut Draft,
    segment_id: &SegmentId,
    offset: Micros,
) -> Result<(), EditError> {
    let segment = keyed(draft, segment_id, offset)?;
    if cutlass_model::remove_keys(segment, offset) {
        Ok(())
    } else {
        Err(EditError::NoKey)
    }
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
        let (clip, uniform) = cutlass_model::clip_at(segment, Micros(250_000)).unwrap();
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
}
