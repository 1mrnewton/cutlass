use cutlass_model::{
    Draft, MaterialAudio, MaterialId, MaterialRef, MaterialVideo, Segment, SegmentId, TrackId,
    TrackType,
};

use super::EditError;

/// Place the picture's sound on its own clip and link that clip to the picture.
///
/// The sound keeps the picture's range, speed, gain, and reverse. A later mix
/// reads the sound clip and skips the picture, so the file is not heard twice.
/// One undo removes the sound clip.
pub fn extract_audio(
    draft: &mut Draft,
    source_id: &SegmentId,
    track_id: &TrackId,
    segment_id: &SegmentId,
    material_id: &MaterialId,
) -> Result<(), EditError> {
    let (source_track, source_index) = super::speed::locate(draft, source_id)?;
    super::ensure_unlocked(draft.tracks[source_track].attribute)?;
    let source = draft.tracks[source_track].segments[source_index].clone();
    let Some(MaterialRef::Video(video)) = draft.materials.get(source.material_id.as_str()) else {
        return Err(EditError::NotVideo);
    };
    let (path, reverse) = sound_path(video, &source);
    if path.is_empty() {
        return Err(EditError::NoSound);
    }
    if sound_is_separate(draft, &source) {
        return Err(EditError::AlreadyExtracted);
    }
    if segment_id.as_str().is_empty() {
        return Err(EditError::DuplicateSegment);
    }
    if material_id.as_str().is_empty() || draft.materials.get(material_id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    if draft.tracks.iter().any(|track| {
        track
            .segments
            .iter()
            .any(|segment| &segment.id == segment_id)
    }) {
        return Err(EditError::DuplicateSegment);
    }
    let carried = carried_refs(draft, &source);
    let duration = video.duration;
    if let Some(track) = draft.tracks.iter().find(|track| &track.id == track_id) {
        if track.kind != TrackType::Audio {
            return Err(EditError::TrackKind);
        }
        super::ensure_unlocked(track.attribute)?;
    } else {
        super::track::insert_track(draft, track_id, &TrackType::Audio)?;
    }
    draft.materials.audios.push(MaterialAudio {
        id: material_id.clone(),
        kind: "video_original_sound".to_owned(),
        path,
        duration,
        video_id: source.material_id.clone(),
        rest: Default::default(),
    });
    let mut segment = Segment::new(
        segment_id.clone(),
        material_id.clone(),
        source.target.clone(),
    );
    segment.source_range.clone_from(&source.source_range);
    segment.speed = source.speed;
    segment.volume = source.volume;
    segment.last_nonzero_volume = source.last_nonzero_volume;
    segment.is_tone_modify = source.is_tone_modify;
    segment.reverse = reverse;
    segment.extra_material_refs = carried;
    super::place::place(draft, track_id, segment)?;
    super::linkage::link(draft, source_id, segment_id)
}

fn sound_path(video: &MaterialVideo, segment: &Segment) -> (String, bool) {
    if segment.reverse && !video.reverse_path.is_empty() {
        (video.reverse_path.clone(), false)
    } else {
        (video.path.clone(), segment.reverse)
    }
}

fn sound_is_separate(draft: &Draft, segment: &Segment) -> bool {
    draft
        .relationships
        .iter()
        .filter(|relationship| relationship.kind == "linkage")
        .flat_map(|relationship| &relationship.id_to_id)
        .any(|pair| {
            pair.owner == segment.id && sound_for(draft, &pair.dependent, &segment.material_id)
        })
}

fn sound_for(draft: &Draft, id: &SegmentId, video_id: &MaterialId) -> bool {
    let Some(segment) = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .find(|segment| &segment.id == id)
    else {
        return false;
    };
    matches!(
        draft.materials.get(segment.material_id.as_str()),
        Some(MaterialRef::Audio(audio))
            if audio.kind == "video_original_sound"
                && draft.materials.same_video(audio.video_id.as_str(), video_id.as_str())
    )
}

fn carried_refs(draft: &Draft, segment: &Segment) -> Vec<MaterialId> {
    segment
        .extra_material_refs
        .iter()
        .filter(|id| {
            matches!(
                draft.materials.get(id.as_str()),
                Some(MaterialRef::Speed(_) | MaterialRef::Fade(_))
            )
        })
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        MaterialAudioFade, MaterialId, MaterialSpeed, MaterialVideo, Micros, Segment, TimeRange,
        Track, TrackAttribute, TrackId, TrackType,
    };

    use super::*;
    use crate::request::Request;
    use crate::session::Session;

    fn picture() -> (Draft, Segment) {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        let mut video = MaterialVideo::new("picture");
        video.path = "clip.mp4".to_owned();
        video.duration = Micros(2_000_000);
        video.reverse_path = "clip-back.mp4".to_owned();
        draft.materials.videos.push(video);
        draft.materials.speeds.push(MaterialSpeed {
            id: MaterialId::new("fast"),
            speed: 2.0,
            ..MaterialSpeed::default()
        });
        draft.materials.audio_fades.push(MaterialAudioFade {
            id: MaterialId::new("fade"),
            fade_in_duration: Micros(40_000),
            ..MaterialAudioFade::default()
        });
        let mut segment = Segment::new("picture", "picture", TimeRange::new(0, 1_000_000));
        segment.source_range = Some(TimeRange::new(100_000, 2_000_000));
        segment.speed = 2.0;
        segment.volume = 0.5;
        segment.reverse = true;
        segment.extra_material_refs = vec![MaterialId::new("fast"), MaterialId::new("fade")];
        (draft, segment)
    }

    fn request() -> Request {
        Request::ExtractAudio {
            source_id: "picture".into(),
            track_id: "sound".into(),
            segment_id: "sound-clip".into(),
            material_id: "sound-material".into(),
        }
    }

    #[test]
    fn a_picture_with_its_own_copy_of_the_video_keeps_its_sound_extracted() {
        let (mut draft, segment) = picture();
        draft.tracks[0].segments.push(segment);
        let mut session = Session::open(draft);
        session.apply(0, &[request()]).unwrap();
        // Split, so both halves show one video, then crop the left half so
        // it takes its own copy.
        session
            .apply(
                1,
                &[
                    Request::Split {
                        segment_id: "picture".into(),
                        at: Micros(500_000),
                        right_id: "right".into(),
                    },
                    Request::SetCrop {
                        segment_id: "picture".into(),
                        crop: cutlass_model::Crop::default(),
                    },
                ],
            )
            .unwrap();
        let draft = session.draft();
        let left = &draft.tracks[0].segments[0];
        assert_ne!(left.material_id.as_str(), "picture");
        assert!(sound_is_separate(draft, left));
        assert!(sound_is_separate(draft, &draft.tracks[0].segments[1]));
    }

    #[test]
    fn extract_audio_places_the_sound_on_its_own_clip() {
        let (mut draft, segment) = picture();
        draft.tracks[0].segments.push(segment);
        let mut session = Session::open(draft);
        session.apply(0, &[request()]).unwrap();
        let draft = session.draft();
        let sound = draft
            .tracks
            .iter()
            .find(|track| track.id.as_str() == "sound")
            .expect("track");
        assert_eq!(sound.kind, TrackType::Audio);
        let clip = &sound.segments[0];
        assert_eq!(clip.target, TimeRange::new(0, 1_000_000));
        assert_eq!(clip.source_range, Some(TimeRange::new(100_000, 2_000_000)));
        assert_eq!(clip.speed, 2.0);
        assert_eq!(clip.volume, 0.5);
        assert!(!clip.reverse);
        assert_eq!(
            clip.extra_material_refs,
            [MaterialId::new("fast"), MaterialId::new("fade")]
        );
        let audio = &draft.materials.audios[0];
        assert_eq!(audio.kind, "video_original_sound");
        assert_eq!(audio.path, "clip-back.mp4");
        assert_eq!(audio.video_id.as_str(), "picture");
        assert_eq!(audio.duration, Micros(2_000_000));
        let pair = &draft
            .relationships
            .iter()
            .find(|item| item.kind == "linkage")
            .expect("link")
            .id_to_id[0];
        assert_eq!(pair.owner.as_str(), "picture");
        assert_eq!(pair.dependent.as_str(), "sound-clip");
        assert_eq!(draft.materials.videos[0].path, "clip.mp4");

        session.undo().unwrap();
        assert!(session.draft().materials.audios.is_empty());
        assert!(
            session
                .draft()
                .tracks
                .iter()
                .all(|track| track.id.as_str() != "sound")
        );
        session.redo().unwrap();

        let before = session.draft().clone();
        let error = session.apply(3, &[request()]).unwrap_err();
        assert_eq!(error, EditError::AlreadyExtracted);
        assert_eq!(session.draft(), &before);

        session
            .apply(
                3,
                &[Request::Remove {
                    segment_id: "sound-clip".into(),
                }],
            )
            .unwrap();
        assert!(
            session
                .draft()
                .relationships
                .iter()
                .all(|item| { item.kind != "linkage" || item.id_to_id.is_empty() })
        );
        session
            .apply(
                4,
                &[Request::ExtractAudio {
                    source_id: "picture".into(),
                    track_id: "sound".into(),
                    segment_id: "sound-again".into(),
                    material_id: "sound-again".into(),
                }],
            )
            .unwrap();
        assert_eq!(
            session
                .draft()
                .tracks
                .iter()
                .find(|track| track.id.as_str() == "sound")
                .expect("sound")
                .segments
                .len(),
            1
        );
    }

    #[test]
    fn a_forward_file_keeps_the_reverse_flag_and_a_locked_lane_refuses() {
        let (mut draft, mut segment) = picture();
        segment.reverse = true;
        draft.materials.videos[0].reverse_path.clear();
        draft.tracks[0].segments.push(segment);
        let mut session = Session::open(draft);
        session.apply(0, &[request()]).unwrap();
        let clip = &session
            .draft()
            .tracks
            .iter()
            .find(|track| track.kind == TrackType::Audio)
            .expect("audio")
            .segments[0];
        assert!(clip.reverse);
        assert_eq!(session.draft().materials.audios[0].path, "clip.mp4");

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
            .apply(
                2,
                &[Request::ExtractAudio {
                    source_id: "picture".into(),
                    track_id: "other".into(),
                    segment_id: "other-clip".into(),
                    material_id: "other-material".into(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::LockedTrack);
        assert_eq!(session.draft(), &before);
    }

    #[test]
    fn a_clip_with_no_picture_sound_or_the_wrong_kind_leaves_the_draft() {
        let (mut draft, mut segment) = picture();
        draft.materials.videos[0].path.clear();
        draft.materials.videos[0].reverse_path.clear();
        segment.reverse = false;
        draft.tracks[0].segments.push(segment);
        let mut tone = Track::main("tone");
        tone.kind = TrackType::Audio;
        tone.segments
            .push(Segment::new("tone", "tone", TimeRange::new(0, 500_000)));
        draft.tracks.push(tone);
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("tone"),
            path: "tone.wav".to_owned(),
            ..MaterialAudio::default()
        });
        let mut session = Session::open(draft);
        let before = session.draft().clone();
        let error = session.apply(0, &[request()]).unwrap_err();
        assert_eq!(error, EditError::NoSound);
        assert_eq!(session.draft(), &before);
        let error = session
            .apply(
                0,
                &[Request::ExtractAudio {
                    source_id: "tone".into(),
                    track_id: "sound".into(),
                    segment_id: "sound-clip".into(),
                    material_id: "sound-material".into(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::NotVideo);
        assert_eq!(session.draft(), &before);
    }

    #[test]
    fn a_locked_sound_lane_is_refused_and_another_clip_can_still_extract() {
        let (mut draft, segment) = picture();
        draft.tracks[0].segments.push(segment);
        let mut other = Segment::new("other", "picture", TimeRange::new(2_000_000, 1_000_000));
        other.source_range = Some(TimeRange::new(0, 2_000_000));
        draft.tracks[0].segments.push(other);
        let mut lane = Track::main("taken");
        lane.kind = TrackType::Audio;
        lane.attribute = TrackAttribute(TrackAttribute::LOCK);
        draft.tracks.push(lane);
        let mut session = Session::open(draft);
        let before = session.draft().clone();
        let error = session
            .apply(
                0,
                &[Request::ExtractAudio {
                    source_id: "picture".into(),
                    track_id: "taken".into(),
                    segment_id: "sound-clip".into(),
                    material_id: "sound-material".into(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::LockedTrack);
        assert_eq!(session.draft(), &before);

        session
            .apply(
                0,
                &[Request::ExtractAudio {
                    source_id: "picture".into(),
                    track_id: "sound".into(),
                    segment_id: "sound-clip".into(),
                    material_id: "sound-material".into(),
                }],
            )
            .unwrap();
        session
            .apply(
                1,
                &[Request::ExtractAudio {
                    source_id: "other".into(),
                    track_id: "sound".into(),
                    segment_id: "other-sound".into(),
                    material_id: "other-material".into(),
                }],
            )
            .unwrap();
        let sounds = session
            .draft()
            .tracks
            .iter()
            .find(|track| track.id.as_str() == "sound")
            .expect("sound");
        assert_eq!(sounds.segments.len(), 2);
        assert_eq!(session.draft().materials.audios.len(), 2);
    }
}
