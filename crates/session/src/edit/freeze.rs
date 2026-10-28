use cutlass_model::{Draft, FreezeInfo, MaterialVideo, SegmentId};

use super::speed::locate;
use super::{EditError, ensure_unlocked};

/// Store a freeze mapping on the video material. This does not seek.
pub fn set_freeze(
    draft: &mut Draft,
    segment_id: &SegmentId,
    freeze: Option<FreezeInfo>,
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
    if freeze.is_some() && !exists && material_id.is_empty() {
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
        video.freeze = freeze;
        return Ok(());
    }
    if let Some(freeze) = freeze {
        draft.materials.videos.push(MaterialVideo {
            id: material_id,
            freeze: Some(freeze),
            ..MaterialVideo::default()
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, FreezeInfo, Micros, Segment, TimeRange, TrackId};

    use crate::edit::EditError;
    use crate::request::Request;
    use crate::session::Session;

    #[test]
    fn a_freeze_is_stored_on_the_video_and_the_ranges_stay() {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        let mut session = Session::open(draft);
        let mut segment = Segment::new("a", "media", TimeRange::new(0, 1_000_000));
        segment.source_range = Some(TimeRange::new(20, 1_000_000));
        segment.volume = 0.4;
        segment.reverse = false;
        session
            .apply(
                0,
                &[Request::Place {
                    track_id: TrackId::new("main"),
                    segment: Box::new(segment),
                }],
            )
            .unwrap();
        session
            .apply(
                1,
                &[Request::SetFreeze {
                    segment_id: "a".into(),
                    freeze: Some(FreezeInfo {
                        source_material_id: "src".to_owned(),
                        source_material_path: "clip".to_owned(),
                        reverse_material_path: String::new(),
                        timestamp: Micros(1500),
                        ..FreezeInfo::default()
                    }),
                }],
            )
            .unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        assert_eq!(segment.target, TimeRange::new(0, 1_000_000));
        assert_eq!(segment.source_range, Some(TimeRange::new(20, 1_000_000)));
        assert_eq!(segment.volume, 0.4);
        assert!(!segment.reverse);
        let freeze = session.draft().materials.videos[0].freeze.clone().unwrap();
        assert_eq!(freeze.timestamp, Micros(1500));
        assert_eq!(freeze.source_material_id, "src");
        let before = session.draft().clone();
        let error = session
            .apply(
                2,
                &[
                    Request::SetFreeze {
                        segment_id: "a".into(),
                        freeze: None,
                    },
                    Request::SetFreeze {
                        segment_id: "missing".into(),
                        freeze: None,
                    },
                ],
            )
            .unwrap_err();
        assert_eq!(error, EditError::MissingSegment);
        assert_eq!(session.draft(), &before);
    }
}
