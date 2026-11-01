use cutlass_model::{Crop, Draft, MaterialVideo, SegmentId};

use super::speed::locate;
use super::{EditError, ensure_unlocked};

/// Store the crop quad. Ratio and scale stay as they were.
pub fn set_crop(draft: &mut Draft, segment_id: &SegmentId, crop: Crop) -> Result<(), EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let material_id = draft.tracks[track_index].segments[segment_index]
        .material_id
        .clone();
    if material_id.is_empty() {
        return Err(EditError::MissingMaterial);
    }
    if let Some(video) = draft
        .materials
        .videos
        .iter_mut()
        .find(|video| video.id == material_id)
    {
        let ratio = video.crop_ratio.clone();
        let scale = video.crop_scale;
        video.crop = crop;
        video.crop_ratio = ratio;
        video.crop_scale = scale;
        return Ok(());
    }
    if let Some(compound) = draft
        .materials
        .drafts
        .iter_mut()
        .find(|compound| compound.id == material_id)
    {
        compound.crop = crop;
        return Ok(());
    }
    if draft.materials.get(material_id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    draft.materials.videos.push(MaterialVideo {
        id: material_id,
        crop,
        ..MaterialVideo::default()
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        Crop, Draft, MaterialAudio, MaterialDraft, MaterialId, MaterialVideo, Segment, TimeRange,
        Track, TrackAttribute, TrackId, TrackType,
    };

    use crate::edit::EditError;
    use crate::request::Request;
    use crate::session::Session;

    fn middle() -> Crop {
        Crop {
            upper_left_x: 0.25,
            upper_left_y: 0.25,
            upper_right_x: 0.75,
            upper_right_y: 0.25,
            lower_left_x: 0.25,
            lower_left_y: 0.75,
            lower_right_x: 0.75,
            lower_right_y: 0.75,
        }
    }

    #[test]
    fn a_crop_stays_on_the_video_and_on_a_compound() {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        draft.config.maintrack_adsorb = false;
        let mut video = MaterialVideo::new("picture");
        video.crop_ratio = "16:9".to_owned();
        video.crop_scale = 2.0;
        draft.materials.videos.push(video);
        draft.tracks[0].segments.push(Segment::new(
            "picture",
            "picture",
            TimeRange::new(0, 1_000_000),
        ));
        draft.materials.drafts.push(MaterialDraft {
            id: MaterialId::new("nest"),
            ..MaterialDraft::default()
        });
        let mut compound = Track::main("compound");
        compound.kind = TrackType::Composition;
        compound.segments.push(Segment::new(
            "compound",
            "nest",
            TimeRange::new(0, 1_000_000),
        ));
        draft.tracks.push(compound);
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("tone"),
            ..MaterialAudio::default()
        });
        let mut sound = Track::main("sound");
        sound.kind = TrackType::Audio;
        sound
            .segments
            .push(Segment::new("tone", "tone", TimeRange::new(0, 1_000_000)));
        draft.tracks.push(sound);
        let mut session = Session::open(draft);

        session
            .apply(
                0,
                &[Request::SetCrop {
                    segment_id: "picture".into(),
                    crop: middle(),
                }],
            )
            .unwrap();
        let video = &session.draft().materials.videos[0];
        assert_eq!(video.crop, middle());
        assert_eq!(video.crop_ratio, "16:9");
        assert_eq!(video.crop_scale, 2.0);
        assert_eq!(
            session.draft().tracks[0].segments[0].target,
            TimeRange::new(0, 1_000_000)
        );

        session
            .apply(
                1,
                &[Request::SetCrop {
                    segment_id: "compound".into(),
                    crop: middle(),
                }],
            )
            .unwrap();
        assert_eq!(session.draft().materials.drafts[0].crop, middle());
        assert_eq!(session.draft().materials.videos.len(), 1);

        let error = session
            .apply(
                2,
                &[Request::SetCrop {
                    segment_id: "tone".into(),
                    crop: middle(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::DuplicateMaterial);
        assert!(
            session
                .draft()
                .materials
                .videos
                .iter()
                .all(|video| video.id.as_str() != "tone")
        );

        session
            .apply(
                2,
                &[Request::SetTrackAttribute {
                    track_id: TrackId::new("main"),
                    bit: TrackAttribute::LOCK,
                    on: true,
                }],
            )
            .unwrap();
        let error = session
            .apply(
                3,
                &[Request::SetCrop {
                    segment_id: "picture".into(),
                    crop: Crop::default(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::LockedTrack);
        assert_eq!(session.draft().materials.videos[0].crop, middle());

        let before = session.draft().clone();
        let error = session
            .apply(
                3,
                &[
                    Request::SetCrop {
                        segment_id: "compound".into(),
                        crop: Crop::default(),
                    },
                    Request::SetCrop {
                        segment_id: "missing".into(),
                        crop: middle(),
                    },
                ],
            )
            .unwrap_err();
        assert_eq!(error, EditError::MissingSegment);
        assert_eq!(session.draft(), &before);
    }
}
