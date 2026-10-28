use cutlass_model::{Draft, MaterialVideo, SegmentId};

use super::speed::locate;
use super::{EditError, ensure_unlocked};

/// Store the reverse flag and, when the video material is there, its path.
///
/// Playback is the part that seeks. This edit does not.
pub fn set_reverse(
    draft: &mut Draft,
    segment_id: &SegmentId,
    reverse: bool,
    path: &str,
) -> Result<(), EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let material_id = draft.tracks[track_index].segments[segment_index]
        .material_id
        .clone();
    let exists = draft
        .materials
        .videos
        .iter()
        .any(|video| video.id == material_id);
    if !exists && !path.is_empty() && material_id.is_empty() {
        return Err(EditError::MissingMaterial);
    }
    if exists {
        let Some(video) = draft
            .materials
            .videos
            .iter_mut()
            .find(|video| video.id == material_id)
        else {
            return Err(EditError::MissingMaterial);
        };
        video.reverse_path = path.to_owned();
    } else if !path.is_empty() {
        draft.materials.videos.push(MaterialVideo {
            id: material_id,
            reverse_path: path.to_owned(),
            ..MaterialVideo::default()
        });
    }
    draft.tracks[track_index].segments[segment_index].reverse = reverse;
    Ok(())
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, Micros, Segment, TimeRange, TrackId};

    use crate::edit::EditError;
    use crate::request::Request;
    use crate::session::Session;

    fn open() -> (Session, TrackId) {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        (Session::open(draft), TrackId::new("main"))
    }

    #[test]
    fn reverse_stores_the_flag_and_the_path_without_moving_the_clip() {
        let (mut session, track) = open();
        let mut segment = Segment::new("a", "media", TimeRange::new(0, 1_000_000));
        segment.volume = 0.4;
        segment.source_range = Some(TimeRange::new(10, 1_000_000));
        session
            .apply(
                0,
                &[Request::Place {
                    track_id: track,
                    segment: Box::new(segment),
                }],
            )
            .unwrap();
        session
            .apply(
                1,
                &[Request::SetReverse {
                    segment_id: "a".into(),
                    reverse: true,
                    path: "reversed".to_owned(),
                }],
            )
            .unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        assert!(segment.reverse);
        assert_eq!(segment.volume, 0.4);
        assert_eq!(segment.target, TimeRange::new(0, 1_000_000));
        assert_eq!(segment.source_range, Some(TimeRange::new(10, 1_000_000)));
        assert_eq!(session.draft().materials.videos[0].reverse_path, "reversed");
        assert_eq!(session.draft().materials.videos[0].id.as_str(), "media");
        session
            .apply(
                2,
                &[Request::SetReverse {
                    segment_id: "a".into(),
                    reverse: true,
                    path: String::new(),
                }],
            )
            .unwrap();
        assert!(session.draft().tracks[0].segments[0].reverse);
        assert_eq!(session.draft().materials.videos[0].reverse_path, "");
        assert_eq!(
            session.draft().tracks[0].segments[0].target.start,
            Micros(0)
        );
    }

    #[test]
    fn a_missing_segment_leaves_the_reverse_unset() {
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
                    Request::SetReverse {
                        segment_id: "a".into(),
                        reverse: true,
                        path: "reversed".to_owned(),
                    },
                    Request::SetReverse {
                        segment_id: "missing".into(),
                        reverse: true,
                        path: "other".to_owned(),
                    },
                ],
            )
            .unwrap_err();
        assert_eq!(error, EditError::MissingSegment);
        assert_eq!(session.draft(), &before);
        assert!(!session.draft().tracks[0].segments[0].reverse);
    }
}
