use cutlass_model::{Draft, MaterialRef, SegmentId, VideoStable};

use super::locate;
use super::{EditError, ensure_unlocked};

/// Steady the clip at `level`, or stop steadying it at 0 or below. The
/// clip's own copy of its material changes; the kept file and any keys
/// the block holds stay.
pub fn set_stable(draft: &mut Draft, segment_id: &SegmentId, level: i64) -> Result<(), EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let material_id = draft.tracks[track_index].segments[segment_index]
        .material_id
        .clone();
    match draft.materials.get(material_id.as_str()) {
        Some(MaterialRef::Video(_)) => {}
        Some(_) => return Err(EditError::NotVideo),
        None => return Err(EditError::MissingMaterial),
    }
    let material_id = super::own::own(draft, track_index, segment_index, &material_id)?;
    let Some(video) = draft
        .materials
        .videos
        .iter_mut()
        .find(|video| video.id == material_id)
    else {
        return Err(EditError::MissingMaterial);
    };
    if level <= 0 {
        video.stable = None;
    } else {
        video.stable.get_or_insert_with(VideoStable::default).level = level;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        Draft, MaterialAudio, MaterialId, MaterialVideo, Segment, TimeRange, Track, TrackAttribute,
        TrackId, TrackType, VideoStable,
    };

    use crate::edit::EditError;
    use crate::request::Request;
    use crate::session::Session;

    fn session() -> Session {
        session_with(None)
    }

    fn session_with(stable: Option<VideoStable>) -> Session {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        draft.config.maintrack_adsorb = false;
        draft.materials.videos.push(MaterialVideo {
            stable,
            ..MaterialVideo::new("picture")
        });
        draft.tracks[0].segments.push(Segment::new(
            "picture",
            "picture",
            TimeRange::new(0, 1_000_000),
        ));
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
        Session::open(draft)
    }

    fn set(session: &mut Session, segment: &str, level: i64) -> Result<u64, EditError> {
        let revision = session.revision();
        session.apply(
            revision,
            &[Request::SetStable {
                segment_id: segment.into(),
                level,
            }],
        )
    }

    fn stable(session: &Session) -> Option<VideoStable> {
        session.draft().materials.videos[0].stable.clone()
    }

    #[test]
    fn a_level_steadies_the_clip_and_zero_stops_it_and_undo_returns() {
        let mut session = session();
        set(&mut session, "picture", VideoStable::RECOMMENDED).unwrap();
        assert_eq!(stable(&session).unwrap().level, 2);
        // Another level keeps the rest of the block.
        let mut session = session_with(Some(VideoStable {
            level: 1,
            matrix_path: "Resources/video_stable/a.json".into(),
            ..VideoStable::default()
        }));
        set(&mut session, "picture", VideoStable::MOST_STABLE).unwrap();
        let kept = stable(&session).unwrap();
        assert_eq!(kept.level, 3);
        assert_eq!(kept.matrix_path, "Resources/video_stable/a.json");
        set(&mut session, "picture", 0).unwrap();
        assert_eq!(stable(&session), None);
        session.undo().unwrap();
        assert_eq!(stable(&session).unwrap().level, 3);
    }

    #[test]
    fn a_sound_a_missing_clip_or_a_locked_lane_is_refused_and_the_draft_stays() {
        let mut session = session();
        let before = session.draft().clone();
        assert_eq!(set(&mut session, "tone", 2), Err(EditError::NotVideo));
        assert_eq!(set(&mut session, "nope", 2), Err(EditError::MissingSegment));
        assert_eq!(session.draft(), &before);
        let revision = session.revision();
        session
            .apply(
                revision,
                &[Request::SetTrackAttribute {
                    track_id: TrackId::new("main"),
                    bit: TrackAttribute::LOCK,
                    on: true,
                }],
            )
            .unwrap();
        assert_eq!(set(&mut session, "picture", 2), Err(EditError::LockedTrack));
        assert_eq!(stable(&session), None);
    }
}
