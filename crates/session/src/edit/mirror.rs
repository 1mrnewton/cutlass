use cutlass_model::{Draft, SegmentId};

use super::locate;
use super::{EditError, ensure_unlocked};

/// Store whether the clip is active: drawn and heard. The clip stays where
/// it is and keeps its edits.
pub fn set_active(
    draft: &mut Draft,
    segment_id: &SegmentId,
    active: bool,
) -> Result<(), EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    draft.tracks[track_index].segments[segment_index].visible = active;
    Ok(())
}

/// Store a horizontal mirror on the segment. The clip stays where it is.
pub fn set_mirror(
    draft: &mut Draft,
    segment_id: &SegmentId,
    horizontal: bool,
) -> Result<(), EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    draft.tracks[track_index].segments[segment_index]
        .clip
        .flip
        .horizontal = horizontal;
    Ok(())
}

/// Store a top-to-bottom flip on the segment. The clip stays where it is.
pub fn set_vertical(
    draft: &mut Draft,
    segment_id: &SegmentId,
    vertical: bool,
) -> Result<(), EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    draft.tracks[track_index].segments[segment_index]
        .clip
        .flip
        .vertical = vertical;
    Ok(())
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, Micros, Segment, TimeRange, TrackAttribute, TrackId};

    use crate::edit::EditError;
    use crate::request::Request;
    use crate::session::Session;

    fn open() -> (Session, TrackId) {
        let mut draft = crate::edit::test_draft("D");
        draft.tracks[0].id = TrackId::new("main");
        (Session::open(draft), TrackId::new("main"))
    }

    #[test]
    fn a_mirror_flips_the_picture_and_leaves_the_clip_in_place() {
        let (mut session, track) = open();
        let mut segment = Segment::new("a", "media", TimeRange::new(0, 1_000_000));
        segment.source_range = Some(TimeRange::new(10, 1_000_000));
        segment.volume = 0.4;
        segment.clip.flip.vertical = true;
        segment.clip.rotation = 12.0;
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
                &[Request::SetMirror {
                    segment_id: "a".into(),
                    horizontal: true,
                }],
            )
            .unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        assert!(segment.clip.flip.horizontal);
        assert!(segment.clip.flip.vertical);
        assert_eq!(segment.clip.rotation, 12.0);
        assert_eq!(segment.volume, 0.4);
        assert_eq!(segment.target, TimeRange::new(0, 1_000_000));
        assert_eq!(segment.source_range, Some(TimeRange::new(10, 1_000_000)));
        session
            .apply(
                2,
                &[Request::SetMirror {
                    segment_id: "a".into(),
                    horizontal: false,
                }],
            )
            .unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        assert!(!segment.clip.flip.horizontal);
        assert!(segment.clip.flip.vertical);
        assert_eq!(segment.target.start, Micros(0));

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
                &[Request::SetMirror {
                    segment_id: "a".into(),
                    horizontal: true,
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::LockedTrack);
        assert!(!session.draft().tracks[0].segments[0].clip.flip.horizontal);
    }

    #[test]
    fn a_missing_segment_leaves_the_mirror_unset() {
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
                    Request::SetMirror {
                        segment_id: "a".into(),
                        horizontal: true,
                    },
                    Request::SetMirror {
                        segment_id: "missing".into(),
                        horizontal: true,
                    },
                ],
            )
            .unwrap_err();
        assert_eq!(error, EditError::MissingSegment);
        assert_eq!(session.draft(), &before);
        assert!(!session.draft().tracks[0].segments[0].clip.flip.horizontal);
    }

    #[test]
    fn a_vertical_flip_turns_the_picture_over_and_leaves_the_clip_in_place() {
        let (mut session, track) = open();
        let mut segment = Segment::new("a", "media", TimeRange::new(0, 1_000_000));
        segment.source_range = Some(TimeRange::new(10, 1_000_000));
        segment.clip.flip.horizontal = true;
        segment.clip.rotation = 12.0;
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
                &[Request::SetVerticalFlip {
                    segment_id: "a".into(),
                    vertical: true,
                }],
            )
            .unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        assert!(segment.clip.flip.vertical);
        assert!(segment.clip.flip.horizontal);
        assert_eq!(segment.clip.rotation, 12.0);
        assert_eq!(segment.target, TimeRange::new(0, 1_000_000));
        assert_eq!(segment.source_range, Some(TimeRange::new(10, 1_000_000)));

        session
            .apply(
                2,
                &[Request::SetTrackAttribute {
                    track_id: track,
                    bit: TrackAttribute::LOCK,
                    on: true,
                }],
            )
            .unwrap();
        let before = session.draft().clone();
        let error = session
            .apply(
                3,
                &[Request::SetVerticalFlip {
                    segment_id: "a".into(),
                    vertical: false,
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::LockedTrack);
        assert_eq!(session.draft(), &before);
    }

    #[test]
    fn a_deactivated_clip_keeps_its_place_and_undo_brings_it_back() {
        let mut draft = Draft::empty("draft");
        draft.tracks[0]
            .segments
            .push(Segment::new("a", "m", TimeRange::new(0, 1_000_000)));
        let mut session = Session::open(draft);
        let off = Request::SetActive {
            segment_id: "a".into(),
            active: false,
        };
        session.apply(0, &[off]).unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        assert!(!segment.visible);
        assert_eq!(segment.target, TimeRange::new(0, 1_000_000));
        session.undo().unwrap();
        assert!(session.draft().tracks[0].segments[0].visible);
    }
}
