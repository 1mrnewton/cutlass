//! Replace a clip's media and keep its edits: its place on the timeline,
//! speed, look, keys and effects stay; only the file it plays changes.

use cutlass_model::{Draft, MaterialId, MaterialRef, MetaType, Micros, SegmentId, TimeRange};

use super::locate;
use super::{EditError, ensure_unlocked};

/// What a material plays, and for how long; `None` for a still.
fn medium(draft: &Draft, material_id: &MaterialId) -> Option<(bool, Option<Micros>)> {
    match draft.materials.get(material_id.as_str())? {
        MaterialRef::Video(video) if video.kind == MetaType::Photo.as_str() => Some((true, None)),
        MaterialRef::Video(video) => Some((true, Some(video.duration))),
        MaterialRef::Audio(audio) => Some((false, Some(audio.duration))),
        _ => None,
    }
}

/// Point `segment_id` at `material_id`. A picture takes a picture and a
/// sound a sound. The new media must be as long as the part the clip
/// plays; the clip keeps its source start when the new media reaches that
/// far, and starts at the beginning otherwise.
pub fn replace_media(
    draft: &mut Draft,
    segment_id: &SegmentId,
    material_id: &MaterialId,
) -> Result<(), EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let segment = &draft.tracks[track_index].segments[segment_index];
    let (was_picture, _) = medium(draft, &segment.material_id).ok_or(EditError::MissingMaterial)?;
    let (picture, length) = medium(draft, material_id).ok_or(EditError::MissingMaterial)?;
    if picture != was_picture {
        return Err(EditError::LaneKind);
    }
    let source = match (segment.source_range.clone(), length) {
        (Some(range), Some(length)) => {
            if range.duration.0 > length.0 {
                return Err(EditError::SourceBounds);
            }
            let start = if range.start.0 + range.duration.0 <= length.0 {
                range.start.0
            } else {
                0
            };
            Some(TimeRange::new(start, range.duration.0))
        }
        (Some(range), None) => Some(TimeRange::new(0, range.duration.0)),
        (None, _) => None,
    };
    let segment = &mut draft.tracks[track_index].segments[segment_index];
    segment.material_id = material_id.clone();
    segment.source_range = source;
    Ok(())
}

/// Point material `material_id` at the file stored at `path`, `width` by
/// `height` and `duration` long, as Link to media does for a file that
/// has moved. Every clip of the material then plays the new file where it
/// played the old one, so the file must reach as far as the furthest of
/// them does. A still has no length to reach. Other materials stay.
pub fn link_media(
    draft: &mut Draft,
    material_id: &MaterialId,
    path: &str,
    (width, height): (i32, i32),
    duration: Micros,
) -> Result<(), EditError> {
    if path.is_empty() {
        return Err(EditError::MissingMaterial);
    }
    let (_, length) = medium(draft, material_id).ok_or(EditError::MissingMaterial)?;
    if length.is_some() {
        if duration.0 <= 0 {
            return Err(EditError::SourceBounds);
        }
        let reach = draft
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .filter(|segment| &segment.material_id == material_id)
            .filter_map(|segment| segment.source_range.as_ref())
            .map(|range| range.end().0)
            .max()
            .unwrap_or(0);
        if reach > duration.0 {
            return Err(EditError::SourceBounds);
        }
    }
    if let Some(video) = draft
        .materials
        .videos
        .iter_mut()
        .find(|video| &video.id == material_id)
    {
        path.clone_into(&mut video.path);
        video.width = width;
        video.height = height;
        if length.is_some() {
            video.duration = duration;
        }
        // A reversed copy was made from the file that is gone.
        video.reverse_path.clear();
        video.reverse_intensifies_path.clear();
    } else if let Some(audio) = draft
        .materials
        .audios
        .iter_mut()
        .find(|audio| &audio.id == material_id)
    {
        path.clone_into(&mut audio.path);
        audio.duration = duration;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        Draft, MaterialAudio, MaterialVideo, Micros, Segment, TimeRange, TrackAttribute,
    };

    use crate::edit::EditError;
    use crate::request::{ImportedMedium, Request};
    use crate::session::Session;

    fn video(id: &str, kind: &str, length: i64) -> Request {
        let mut material = MaterialVideo::new(id);
        material.kind = kind.to_owned();
        material.duration = Micros(length);
        Request::Import {
            medium: ImportedMedium::Video(Box::new(material)),
        }
    }

    fn replace(material: &str) -> Request {
        Request::ReplaceMedia {
            segment_id: "a".into(),
            material_id: material.into(),
        }
    }

    fn session() -> Session {
        let mut draft = Draft::empty("D");
        let mut material = MaterialVideo::new("old");
        material.kind = "video".to_owned();
        material.duration = Micros(10_000_000);
        draft.materials.videos.push(material);
        let mut segment = Segment::new("a", "old", TimeRange::new(0, 1_000_000));
        segment.source_range = Some(TimeRange::new(3_000_000, 2_000_000));
        segment.speed = 2.0;
        segment.volume = 0.5;
        draft.tracks[0].segments.push(segment);
        Session::open(draft)
    }

    #[test]
    fn a_replaced_clip_keeps_its_place_speed_and_source_start_in_one_step() {
        let mut session = session();
        session
            .apply(0, &[video("new", "video", 8_000_000), replace("new")])
            .unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        assert_eq!(segment.material_id.as_str(), "new");
        assert_eq!(segment.target, TimeRange::new(0, 1_000_000));
        assert_eq!(
            segment.source_range,
            Some(TimeRange::new(3_000_000, 2_000_000))
        );
        assert_eq!((segment.speed, segment.volume), (2.0, 0.5));
        session.undo().unwrap();
        assert_eq!(
            session.draft().tracks[0].segments[0].material_id.as_str(),
            "old"
        );
        assert!(session.draft().materials.videos.len() == 1);
    }

    #[test]
    fn shorter_media_starts_at_its_beginning_and_too_short_media_is_refused() {
        let mut session = session();
        session
            .apply(0, &[video("short", "video", 4_000_000), replace("short")])
            .unwrap();
        assert_eq!(
            session.draft().tracks[0].segments[0].source_range,
            Some(TimeRange::new(0, 2_000_000))
        );
        let refused = session.apply(1, &[video("tiny", "video", 1_000_000), replace("tiny")]);
        assert_eq!(refused.unwrap_err(), EditError::SourceBounds);
        // A photo is as long as it needs to be.
        session
            .apply(1, &[video("still", "photo", 0), replace("still")])
            .unwrap();
        assert_eq!(
            session.draft().tracks[0].segments[0].source_range,
            Some(TimeRange::new(0, 2_000_000))
        );
    }

    #[test]
    fn a_picture_takes_no_sound_and_a_locked_track_is_refused() {
        let mut session = session();
        let sound = Request::Import {
            medium: ImportedMedium::Audio(Box::new(MaterialAudio {
                id: "tone".into(),
                kind: "music".to_owned(),
                duration: Micros(9_000_000),
                ..MaterialAudio::default()
            })),
        };
        assert_eq!(
            session.apply(0, &[sound, replace("tone")]).unwrap_err(),
            EditError::LaneKind
        );
        assert_eq!(
            session.apply(0, &[replace("missing")]).unwrap_err(),
            EditError::MissingMaterial
        );
        let mut draft = session.draft().clone();
        draft.tracks[0].attribute = TrackAttribute(TrackAttribute::LOCK);
        let mut locked = Session::open(draft);
        assert_eq!(
            locked
                .apply(0, &[video("new", "video", 8_000_000), replace("new")])
                .unwrap_err(),
            EditError::LockedTrack
        );
    }

    #[test]
    fn a_material_is_linked_to_a_file_that_reaches_as_far_as_its_clips() {
        let mut session = session();
        let link = |path: &str, length: i64| Request::LinkMedia {
            material_id: "old".into(),
            path: path.to_owned(),
            width: 1280,
            height: 720,
            duration: Micros(length),
        };
        let before = session.draft().clone();
        // The clip plays up to five seconds in: a file that long will do.
        session
            .apply(0, &[link("Resources/moved.mp4", 5_000_000)])
            .unwrap();
        let draft = session.draft();
        let video = &draft.materials.videos[0];
        assert_eq!(video.path, "Resources/moved.mp4");
        assert_eq!((video.width, video.height), (1280, 720));
        assert_eq!(video.duration, Micros(5_000_000));
        // The clip is as it was: only the file under it changed.
        assert_eq!(draft.tracks[0].segments, before.tracks[0].segments);
        session.undo().unwrap();
        assert_eq!(session.draft(), &before);

        // A shorter file, no path, and a material that is not there.
        for (request, error) in [
            (link("short.mp4", 4_999_999), EditError::SourceBounds),
            (link("empty.mp4", 0), EditError::SourceBounds),
            (link("", 9_000_000), EditError::MissingMaterial),
            (
                Request::LinkMedia {
                    material_id: "nowhere".into(),
                    path: "a.mp4".to_owned(),
                    width: 1,
                    height: 1,
                    duration: Micros(9_000_000),
                },
                EditError::MissingMaterial,
            ),
        ] {
            assert_eq!(session.apply(2, &[request]).unwrap_err(), error);
            assert_eq!(session.draft(), &before);
        }

        // A still has no length, and a sound takes the new length.
        let mut draft = before.clone();
        draft.materials.videos[0].kind = "photo".to_owned();
        draft.materials.videos[0].reverse_path = "back.mp4".to_owned();
        draft.materials.audios.push(MaterialAudio {
            id: "tone".into(),
            duration: Micros(1_000_000),
            ..MaterialAudio::default()
        });
        let mut stills = Session::open(draft);
        stills
            .apply(
                0,
                &[
                    link("still.png", 0),
                    Request::LinkMedia {
                        material_id: "tone".into(),
                        path: "tone.m4a".to_owned(),
                        width: 0,
                        height: 0,
                        duration: Micros(2_000_000),
                    },
                ],
            )
            .unwrap();
        let materials = &stills.draft().materials;
        assert_eq!(materials.videos[0].path, "still.png");
        assert_eq!(materials.videos[0].duration, Micros(10_000_000));
        assert_eq!(materials.videos[0].reverse_path, "");
        assert_eq!(materials.audios[0].path, "tone.m4a");
        assert_eq!(materials.audios[0].duration, Micros(2_000_000));
    }
}
