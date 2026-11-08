use cutlass_model::{Algorithm, Draft, MaterialRef, SegmentId, VideoAlgorithm};

use super::locate;
use super::{EditError, ensure_unlocked};

/// Enhance the clip at `level` (HD or UHD), or stop at 0 or below. The
/// clip's own copy of its material changes; its other algorithms and any
/// keys the block holds stay.
pub fn set_enhance(draft: &mut Draft, segment_id: &SegmentId, level: i64) -> Result<(), EditError> {
    set_algorithm(draft, segment_id, Algorithm::QUALITY_ENHANCE, level)
}

/// Reduce the clip's noise at `level` (Weak or Strong), or stop at 0 or
/// below, as [`set_enhance`] does.
pub fn set_denoise(draft: &mut Draft, segment_id: &SegmentId, level: i64) -> Result<(), EditError> {
    set_algorithm(draft, segment_id, Algorithm::NOISE_REDUCTION, level)
}

/// Fill the clip to `fps` frames a second with Optical flow, one of
/// [`Algorithm::FRAME_RATES`], or stop at 0 or below, as [`set_enhance`]
/// does.
pub fn set_interpolate(
    draft: &mut Draft,
    segment_id: &SegmentId,
    fps: i64,
) -> Result<(), EditError> {
    if fps > 0 && !Algorithm::FRAME_RATES.contains(&fps) {
        return Err(EditError::FlowRate);
    }
    set_algorithm(draft, segment_id, Algorithm::COMPLEMENT_FRAME, fps)
}

fn set_algorithm(
    draft: &mut Draft,
    segment_id: &SegmentId,
    kind: &str,
    level: i64,
) -> Result<(), EditError> {
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
    let block = video
        .video_algorithm
        .get_or_insert_with(VideoAlgorithm::default);
    let found = block
        .algorithms
        .iter()
        .position(|algorithm| algorithm.kind == kind);
    match (found, level > 0) {
        (Some(index), true) => block.algorithms[index].level = level,
        (Some(index), false) => {
            block.algorithms.remove(index);
        }
        (None, true) => block.algorithms.push(Algorithm {
            kind: kind.to_owned(),
            level,
            ..Algorithm::default()
        }),
        (None, false) => {}
    }
    if block.algorithms.is_empty() && block.rest.is_empty() {
        video.video_algorithm = None;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        Algorithm, Draft, MaterialAudio, MaterialId, MaterialVideo, Segment, TimeRange, Track,
        TrackAttribute, TrackId, TrackType, VideoAlgorithm,
    };

    use crate::edit::EditError;
    use crate::request::Request;
    use crate::session::Session;

    fn session_with(algorithm: Option<VideoAlgorithm>) -> Session {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        draft.config.maintrack_adsorb = false;
        draft.materials.videos.push(MaterialVideo {
            video_algorithm: algorithm,
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
            &[Request::SetEnhance {
                segment_id: segment.into(),
                level,
            }],
        )
    }

    fn block(session: &Session) -> Option<VideoAlgorithm> {
        session.draft().materials.videos[0].video_algorithm.clone()
    }

    fn level(session: &Session) -> Option<i64> {
        session.draft().materials.videos[0]
            .enhance()
            .map(|enhance| enhance.level)
    }

    #[test]
    fn a_level_enhances_the_clip_and_zero_stops_it_and_undo_returns() {
        let mut session = session_with(None);
        set(&mut session, "picture", Algorithm::HD).unwrap();
        assert_eq!(level(&session), Some(Algorithm::HD));
        set(&mut session, "picture", Algorithm::UHD).unwrap();
        assert_eq!(level(&session), Some(Algorithm::UHD));
        assert_eq!(block(&session).unwrap().algorithms.len(), 1);
        set(&mut session, "picture", 0).unwrap();
        assert_eq!(block(&session), None);
        session.undo().unwrap();
        assert_eq!(level(&session), Some(Algorithm::UHD));
    }

    #[test]
    fn other_algorithms_and_unknown_keys_stay() {
        let deflicker = Algorithm {
            kind: "Deflicker".into(),
            level: 1,
            ..Algorithm::default()
        };
        let mut rest = serde_json::Map::new();
        rest.insert("gameplay".into(), serde_json::Value::Null);
        let mut session = session_with(Some(VideoAlgorithm {
            algorithms: vec![deflicker.clone()],
            rest: rest.clone(),
        }));
        set(&mut session, "picture", Algorithm::HD).unwrap();
        set(&mut session, "picture", 0).unwrap();
        assert_eq!(
            block(&session),
            Some(VideoAlgorithm {
                algorithms: vec![deflicker],
                rest,
            })
        );
    }

    #[test]
    fn a_sound_a_missing_clip_or_a_locked_lane_is_refused_and_the_draft_stays() {
        let mut session = session_with(None);
        let before = session.draft().clone();
        assert_eq!(set(&mut session, "tone", 1), Err(EditError::NotVideo));
        assert_eq!(set(&mut session, "nope", 1), Err(EditError::MissingSegment));
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
        assert_eq!(set(&mut session, "picture", 1), Err(EditError::LockedTrack));
        assert_eq!(block(&session), None);
    }

    #[test]
    fn noise_reduction_sits_beside_enhance_and_each_stops_alone() {
        let mut session = session_with(None);
        set(&mut session, "picture", Algorithm::HD).unwrap();
        let denoise = |session: &mut Session, level: i64| {
            let revision = session.revision();
            session.apply(
                revision,
                &[Request::SetDenoise {
                    segment_id: "picture".into(),
                    level,
                }],
            )
        };
        denoise(&mut session, Algorithm::STRONG).unwrap();
        let video = &session.draft().materials.videos[0];
        assert_eq!(video.denoise().map(|a| a.level), Some(Algorithm::STRONG));
        assert_eq!(level(&session), Some(Algorithm::HD));
        set(&mut session, "picture", 0).unwrap();
        assert_eq!(level(&session), None);
        assert_eq!(
            session.draft().materials.videos[0]
                .denoise()
                .map(|a| a.level),
            Some(Algorithm::STRONG)
        );
        denoise(&mut session, 0).unwrap();
        assert_eq!(block(&session), None);
        session.undo().unwrap();
        assert!(session.draft().materials.videos[0].denoise().is_some());
        assert!(denoise(&mut session, Algorithm::WEAK).is_ok());
        assert_eq!(
            {
                let revision = session.revision();
                session.apply(
                    revision,
                    &[Request::SetDenoise {
                        segment_id: "tone".into(),
                        level: 1,
                    }],
                )
            },
            Err(EditError::NotVideo)
        );
    }

    fn interpolate(session: &mut Session, segment: &str, fps: i64) -> Result<u64, EditError> {
        let revision = session.revision();
        session.apply(
            revision,
            &[Request::SetInterpolate {
                segment_id: segment.into(),
                fps,
            }],
        )
    }

    fn fps(session: &Session) -> Option<i64> {
        session.draft().materials.videos[0]
            .interpolate()
            .map(|flow| flow.level)
    }

    #[test]
    fn optical_flow_takes_an_offered_rate_beside_the_others_and_zero_stops_it() {
        let mut session = session_with(None);
        set(&mut session, "picture", Algorithm::HD).unwrap();
        for rate in Algorithm::FRAME_RATES {
            interpolate(&mut session, "picture", rate).unwrap();
            assert_eq!(fps(&session), Some(rate));
        }
        assert_eq!(level(&session), Some(Algorithm::HD));
        let before = session.draft().clone();
        for odd in [24, 1, 120, i64::MAX] {
            assert_eq!(
                interpolate(&mut session, "picture", odd),
                Err(EditError::FlowRate),
                "{odd}"
            );
        }
        assert_eq!(session.draft(), &before);
        assert_eq!(
            interpolate(&mut session, "tone", 30),
            Err(EditError::NotVideo)
        );
        interpolate(&mut session, "picture", 0).unwrap();
        assert_eq!(fps(&session), None);
        assert_eq!(level(&session), Some(Algorithm::HD));
        session.undo().unwrap();
        assert_eq!(fps(&session), Some(60));
    }

    #[test]
    fn optical_flow_on_every_clip_at_once_is_one_undo_step() {
        let mut draft = session_with(None).draft().clone();
        draft.tracks[0].segments.push(Segment::new(
            "second",
            "picture",
            TimeRange::new(1_000_000, 1_000_000),
        ));
        let mut session = Session::open(draft);
        let revision = session.revision();
        session
            .apply(
                revision,
                &[
                    Request::SetInterpolate {
                        segment_id: "picture".into(),
                        fps: 50,
                    },
                    Request::SetInterpolate {
                        segment_id: "second".into(),
                        fps: 50,
                    },
                ],
            )
            .unwrap();
        let flowed = |session: &Session| {
            session
                .draft()
                .materials
                .videos
                .iter()
                .filter(|video| video.interpolate().is_some_and(|flow| flow.level == 50))
                .count()
        };
        // Each clip owns its copy of the material, both filled.
        assert_eq!(flowed(&session), 2);
        session.undo().unwrap();
        assert_eq!(flowed(&session), 0);
    }
}
