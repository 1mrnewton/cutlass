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

/// How far from the picture's middle a pivot may be, in half sizes. Past
/// the picture's own edge, so a clip can swing about a point beside it.
pub const PIVOT_REACH: f64 = 8.0;

/// Store the point the picture turns and is sized about: `x` and `y` are
/// shares of its own half size from its middle, y up. The clip's rotation,
/// scale and place stay as they are, so a picture that is neither turned
/// nor sized does not move.
pub fn set_pivot(
    draft: &mut Draft,
    segment_id: &SegmentId,
    x: f64,
    y: f64,
) -> Result<(), EditError> {
    let within = |value: f64| value.is_finite() && value.abs() <= PIVOT_REACH;
    if !within(x) || !within(y) {
        return Err(EditError::Pivot);
    }
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let pivot = &mut draft.tracks[track_index].segments[segment_index].clip.pivot;
    pivot.x = x;
    pivot.y = y;
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
    fn a_pivot_is_stored_and_moves_nothing_else() {
        let (mut session, track) = open();
        let mut segment = Segment::new("a", "media", TimeRange::new(0, 1_000_000));
        segment.clip.rotation = 30.0;
        segment.clip.transform.x = 0.25;
        session
            .apply(
                0,
                &[Request::Place {
                    track_id: track.clone(),
                    segment: Box::new(segment),
                }],
            )
            .unwrap();
        let pivot = |x: f64, y: f64| Request::SetPivot {
            segment_id: "a".into(),
            x,
            y,
        };
        session
            .apply(session.revision(), &[pivot(-1.0, 0.5)])
            .unwrap();
        let clip = &session.draft().tracks[0].segments[0].clip;
        assert_eq!((clip.pivot.x, clip.pivot.y), (-1.0, 0.5));
        assert_eq!((clip.rotation, clip.transform.x), (30.0, 0.25));
        // It is one undo step, and the middle is a pivot like any other.
        session.undo().unwrap();
        assert!(session.draft().tracks[0].segments[0].clip.pivot.is_origin());
        session.redo().unwrap();
        session
            .apply(session.revision(), &[pivot(0.0, 0.0)])
            .unwrap();
        assert!(session.draft().tracks[0].segments[0].clip.pivot.is_origin());
        // A point beside the picture is allowed; one too far, or not a
        // number, changes nothing.
        session
            .apply(session.revision(), &[pivot(8.0, -8.0)])
            .unwrap();
        let before = session.draft().clone();
        for (x, y) in [
            (8.5, 0.0),
            (0.0, -9.0),
            (f64::NAN, 0.0),
            (0.0, f64::INFINITY),
        ] {
            assert_eq!(
                session.apply(session.revision(), &[pivot(x, y)]),
                Err(EditError::Pivot)
            );
            assert_eq!(session.draft(), &before);
        }
        assert_eq!(
            session.apply(
                session.revision(),
                &[Request::SetPivot {
                    segment_id: "missing".into(),
                    x: 0.0,
                    y: 0.0,
                }]
            ),
            Err(EditError::MissingSegment)
        );
        session
            .apply(
                session.revision(),
                &[Request::SetTrackAttribute {
                    track_id: track,
                    bit: TrackAttribute::LOCK,
                    on: true,
                }],
            )
            .unwrap();
        assert_eq!(
            session.apply(session.revision(), &[pivot(0.0, 0.0)]),
            Err(EditError::LockedTrack)
        );
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
