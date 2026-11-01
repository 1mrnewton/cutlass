use cutlass_model::{Draft, MaterialId};

use crate::request::ImportedMedium;

use super::EditError;

/// Store the material in the library. The timeline is left as it is.
pub fn import(draft: &mut Draft, medium: ImportedMedium) -> Result<(), EditError> {
    let material_id = medium_id(&medium);
    if material_id.is_empty() {
        return Err(EditError::MissingMaterial);
    }
    if draft.materials.get(material_id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    match medium {
        ImportedMedium::Video(video) => draft.materials.videos.push(*video),
        ImportedMedium::Audio(audio) => draft.materials.audios.push(*audio),
    }
    Ok(())
}

fn medium_id(medium: &ImportedMedium) -> MaterialId {
    match medium {
        ImportedMedium::Video(video) => video.id.clone(),
        ImportedMedium::Audio(audio) => audio.id.clone(),
    }
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        Draft, MaterialAudio, MaterialVideo, Micros, TrackAttribute, TrackId, TrackType,
    };

    use super::*;
    use crate::request::{ImportedMedium, Request};
    use crate::session::Session;

    fn video_request(id: &str) -> Request {
        let mut material = MaterialVideo::new(id);
        material.kind = "video".to_owned();
        material.path = format!("{id}.mp4");
        material.duration = Micros(1_000_000);
        Request::Import {
            medium: ImportedMedium::Video(Box::new(material)),
        }
    }

    #[test]
    fn an_import_lands_in_the_library_and_undoes_in_one_step() {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        let mut session = Session::open(draft);
        session.apply(0, &[video_request("clip")]).unwrap();
        assert_eq!(session.draft().materials.videos[0].path, "clip.mp4");
        assert!(session.draft().tracks[0].segments.is_empty());
        assert_eq!(session.draft().tracks.len(), 1);
        assert_eq!(session.draft().duration, Micros(0));
        session.undo().unwrap();
        assert!(session.draft().materials.videos.is_empty());
        assert!(session.draft().tracks[0].segments.is_empty());
    }

    #[test]
    fn audio_stays_off_the_timeline_and_a_locked_track_still_accepts_it() {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        draft.tracks[0].attribute = TrackAttribute(TrackAttribute::LOCK);
        let mut session = Session::open(draft);
        let material = MaterialAudio {
            id: "tone".into(),
            kind: "music".to_owned(),
            path: "tone.wav".to_owned(),
            duration: Micros(500_000),
            ..MaterialAudio::default()
        };
        session
            .apply(
                0,
                &[Request::Import {
                    medium: ImportedMedium::Audio(Box::new(material)),
                }],
            )
            .unwrap();
        assert_eq!(session.draft().tracks.len(), 1);
        assert_eq!(session.draft().tracks[0].kind, TrackType::Video);
        assert!(session.draft().tracks[0].segments.is_empty());
        assert!(
            session.draft().tracks[0]
                .attribute
                .contains(TrackAttribute::LOCK)
        );
        assert_eq!(session.draft().materials.audios[0].path, "tone.wav");
    }

    #[test]
    fn a_duplicate_or_empty_material_leaves_the_draft() {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        let mut session = Session::open(draft);
        session.apply(0, &[video_request("clip")]).unwrap();
        let before = session.draft().clone();
        let error = session.apply(1, &[video_request("clip")]).unwrap_err();
        assert_eq!(error, EditError::DuplicateMaterial);
        assert_eq!(session.draft(), &before);
        let error = session
            .apply(
                1,
                &[Request::Import {
                    medium: ImportedMedium::Video(Box::new(MaterialVideo::new(""))),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::MissingMaterial);
        assert_eq!(session.draft(), &before);
    }
}
