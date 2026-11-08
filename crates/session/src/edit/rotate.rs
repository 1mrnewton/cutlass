use cutlass_model::{Draft, SegmentId};

use super::locate;
use super::{EditError, ensure_unlocked};

/// Store the clip rotation in degrees. The clip stays where it is.
pub fn set_rotation(
    draft: &mut Draft,
    segment_id: &SegmentId,
    degrees: f64,
) -> Result<(), EditError> {
    if !degrees.is_finite() {
        return Err(EditError::Angle);
    }
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    draft.tracks[track_index].segments[segment_index]
        .clip
        .rotation = degrees;
    Ok(())
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Micros, Segment, TimeRange, TrackAttribute, TrackId};

    use crate::edit::EditError;
    use crate::request::Request;
    use crate::session::Session;

    fn open() -> (Session, TrackId) {
        let mut draft = crate::edit::test_draft("D");
        draft.tracks[0].id = TrackId::new("main");
        (Session::open(draft), TrackId::new("main"))
    }

    #[test]
    fn a_rotation_turns_the_picture_and_leaves_the_clip_in_place() {
        let (mut session, track) = open();
        let mut segment = Segment::new("a", "media", TimeRange::new(0, 1_000_000));
        segment.source_range = Some(TimeRange::new(10, 1_000_000));
        segment.volume = 0.4;
        segment.clip.flip.horizontal = true;
        segment.clip.scale.x = 1.5;
        session
            .apply(
                0,
                &[Request::Place {
                    track_id: track.clone(),
                    segment: Box::new(segment),
                }],
            )
            .unwrap();
        session
            .apply(
                1,
                &[Request::SetRotation {
                    segment_id: "a".into(),
                    degrees: 270.0,
                }],
            )
            .unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        assert_eq!(segment.clip.rotation, 270.0);
        assert!(segment.clip.flip.horizontal);
        assert_eq!(segment.clip.scale.x, 1.5);
        assert_eq!(segment.volume, 0.4);
        assert_eq!(segment.target, TimeRange::new(0, 1_000_000));
        assert_eq!(segment.source_range, Some(TimeRange::new(10, 1_000_000)));
        session
            .apply(
                2,
                &[Request::SetRotation {
                    segment_id: "a".into(),
                    degrees: 12.5,
                }],
            )
            .unwrap();
        assert_eq!(session.draft().tracks[0].segments[0].clip.rotation, 12.5);
        assert_eq!(
            session.draft().tracks[0].segments[0].target.start,
            Micros(0)
        );

        let error = session
            .apply(
                3,
                &[Request::SetRotation {
                    segment_id: "a".into(),
                    degrees: f64::NAN,
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::Angle);
        assert_eq!(session.draft().tracks[0].segments[0].clip.rotation, 12.5);

        session
            .apply(
                3,
                &[Request::SetTrackAttribute {
                    track_id: track,
                    bit: TrackAttribute::LOCK,
                    on: true,
                }],
            )
            .unwrap();
        let error = session
            .apply(
                4,
                &[Request::SetRotation {
                    segment_id: "a".into(),
                    degrees: 90.0,
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::LockedTrack);
        assert_eq!(session.draft().tracks[0].segments[0].clip.rotation, 12.5);
    }

    #[test]
    fn a_missing_segment_leaves_the_rotation_unset() {
        let (mut session, track) = open();
        session
            .apply(
                0,
                &[Request::Place {
                    track_id: track,
                    segment: Box::new(Segment::new("a", "media", TimeRange::new(0, 1_000_000))),
                }],
            )
            .unwrap();
        let before = session.draft().clone();
        let error = session
            .apply(
                1,
                &[
                    Request::SetRotation {
                        segment_id: "a".into(),
                        degrees: 90.0,
                    },
                    Request::SetRotation {
                        segment_id: "missing".into(),
                        degrees: 180.0,
                    },
                ],
            )
            .unwrap_err();
        assert_eq!(error, EditError::MissingSegment);
        assert_eq!(session.draft(), &before);
        assert_eq!(session.draft().tracks[0].segments[0].clip.rotation, 0.0);
    }
}
