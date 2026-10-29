use cutlass_model::{Draft, MaterialId, Segment, Track, TrackFlag, TrackId, TrackType};

use crate::request::ImportedMedium;

use super::{EditError, ensure_unlocked, place::place};

/// Insert the material and the segment. A missing track is created first.
pub fn import(
    draft: &mut Draft,
    track_id: &TrackId,
    kind: &TrackType,
    medium: ImportedMedium,
    segment: Segment,
) -> Result<(), EditError> {
    let material_id = medium_id(&medium);
    if material_id.is_empty() || material_id != segment.material_id {
        return Err(EditError::MissingMaterial);
    }
    if draft.materials.get(material_id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    ensure_track(draft, track_id, kind)?;
    if let Some(track) = draft.tracks.iter().find(|track| &track.id == track_id) {
        ensure_unlocked(track.attribute)?;
    }
    match medium {
        ImportedMedium::Video(video) => draft.materials.videos.push(*video),
        ImportedMedium::Audio(audio) => draft.materials.audios.push(*audio),
    }
    place(draft, track_id, segment)
}

fn medium_id(medium: &ImportedMedium) -> MaterialId {
    match medium {
        ImportedMedium::Video(video) => video.id.clone(),
        ImportedMedium::Audio(audio) => audio.id.clone(),
    }
}

fn ensure_track(draft: &mut Draft, track_id: &TrackId, kind: &TrackType) -> Result<(), EditError> {
    if let Some(track) = draft.tracks.iter().find(|track| &track.id == track_id) {
        if &track.kind == kind {
            Ok(())
        } else {
            Err(EditError::TrackKind)
        }
    } else {
        let mut track = Track::main(track_id.clone());
        track.kind = kind.clone();
        let main_exists = draft.tracks.iter().any(Track::is_main);
        if main_exists && matches!(kind, TrackType::Video | TrackType::Mixed) {
            track.flag = TrackFlag::SubVideo;
        }
        draft.tracks.push(track);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        Draft, MaterialAudio, MaterialVideo, Micros, Segment, TimeRange, TrackAttribute, TrackId,
        TrackType,
    };

    use super::*;
    use crate::request::{ImportedMedium, Request};
    use crate::session::Session;

    fn video_request(track: &str, id: &str) -> Request {
        let mut material = MaterialVideo::new(id);
        material.kind = "video".to_owned();
        material.path = format!("{id}.mp4");
        material.duration = Micros(1_000_000);
        Request::Import {
            track_id: TrackId::new(track),
            kind: TrackType::Video,
            medium: ImportedMedium::Video(Box::new(material)),
            segment: Box::new(Segment::new(id, id, TimeRange::new(0, 1_000_000))),
        }
    }

    #[test]
    fn an_import_lands_the_material_and_undoes_in_one_step() {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        let mut session = Session::open(draft);
        session
            .apply(0, &[video_request("main", "clip")])
            .unwrap();
        assert_eq!(session.draft().materials.videos[0].path, "clip.mp4");
        assert_eq!(session.draft().tracks[0].segments.len(), 1);
        assert_eq!(session.draft().duration, Micros(1_000_000));
        session.undo().unwrap();
        assert!(session.draft().materials.videos.is_empty());
        assert!(session.draft().tracks[0].segments.is_empty());
    }

    #[test]
    fn audio_creates_its_own_track_and_a_locked_track_leaves_the_draft() {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        let mut session = Session::open(draft);
        let mut material = MaterialAudio::default();
        material.id = "tone".into();
        material.kind = "music".to_owned();
        material.path = "tone.wav".to_owned();
        material.duration = Micros(500_000);
        session
            .apply(
                0,
                &[Request::Import {
                    track_id: TrackId::new("audio"),
                    kind: TrackType::Audio,
                    medium: ImportedMedium::Audio(Box::new(material)),
                    segment: Box::new(Segment::new("tone", "tone", TimeRange::new(0, 500_000))),
                }],
            )
            .unwrap();
        assert_eq!(session.draft().tracks.len(), 2);
        assert_eq!(session.draft().tracks[1].kind, TrackType::Audio);
        assert!(!session.draft().tracks[1].is_main());
        assert_eq!(session.draft().materials.audios[0].path, "tone.wav");

        session
            .apply(
                1,
                &[Request::SetTrackAttribute {
                    track_id: TrackId::new("main"),
                    bit: TrackAttribute::LOCK,
                    on: true,
                }],
            )
            .unwrap();
        let before = session.draft().clone();
        let error = session
            .apply(2, &[video_request("main", "other")])
            .unwrap_err();
        assert_eq!(error, EditError::LockedTrack);
        assert_eq!(session.draft(), &before);
    }

    #[test]
    fn a_duplicate_material_or_the_wrong_track_type_leaves_the_draft() {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        let mut session = Session::open(draft);
        session
            .apply(0, &[video_request("main", "clip")])
            .unwrap();
        let before = session.draft().clone();
        let error = session
            .apply(1, &[video_request("main", "clip")])
            .unwrap_err();
        assert_eq!(error, EditError::DuplicateMaterial);
        assert_eq!(session.draft(), &before);
        let error = session
            .apply(
                1,
                &[Request::Import {
                    track_id: TrackId::new("main"),
                    kind: TrackType::Audio,
                    medium: ImportedMedium::Audio(Box::new(MaterialAudio {
                        id: "tone".into(),
                        duration: Micros(100),
                        ..MaterialAudio::default()
                    })),
                    segment: Box::new(Segment::new("tone", "tone", TimeRange::new(0, 100))),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::TrackKind);
        assert_eq!(session.draft(), &before);
    }
}
