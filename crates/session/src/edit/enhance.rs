use cutlass_model::{
    Algorithm, Draft, MaterialRef, MotionBlur, Reframe, SegmentId, VideoAlgorithm,
};

/// The longest preset id Relight stores.
const MAX_PRESET: usize = 64;

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

/// Light the clip again with the preset `preset` at `strength` (0 to 1),
/// the Custom light taking `color` (`#rrggbb`) and `angle` (degrees), or
/// stop with an empty preset, as [`set_enhance`] does.
pub fn set_relight(
    draft: &mut Draft,
    segment_id: &SegmentId,
    preset: &str,
    strength: f64,
    color: &str,
    angle: f64,
) -> Result<(), EditError> {
    let named = preset.len() <= MAX_PRESET
        && preset.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_' || byte == b'-'
        });
    let hex = color.is_empty()
        || (color.len() == 7
            && color.starts_with('#')
            && color.bytes().skip(1).all(|byte| byte.is_ascii_hexdigit()));
    if !named || !hex || !(0.0..=1.0).contains(&strength) || !angle.is_finite() {
        return Err(EditError::Relight);
    }
    let level = i64::from(!preset.is_empty());
    set_filled(draft, segment_id, Algorithm::RELIGHT, level, |algorithm| {
        algorithm.preset = preset.to_owned();
        algorithm.strength = Some(strength);
        algorithm.color = color.to_owned();
        algorithm.angle = Some(angle.rem_euclid(360.0));
    })
}

/// Cut the clip to `ratio` around its subject with these settings, or
/// stop with an empty ratio, as [`set_enhance`] does.
pub fn set_reframe(
    draft: &mut Draft,
    segment_id: &SegmentId,
    ratio: &str,
    stability: i64,
    speed: i64,
) -> Result<(), EditError> {
    let settings = Reframe::FLEXIBLE..=Reframe::STABLE;
    if !(ratio.is_empty() || Reframe::RATIOS.contains(&ratio))
        || !settings.contains(&stability)
        || !settings.contains(&speed)
    {
        return Err(EditError::Reframe);
    }
    let level = i64::from(!ratio.is_empty());
    set_filled(
        draft,
        segment_id,
        Algorithm::AUTO_REFRAME,
        level,
        |algorithm| {
            algorithm.ratio = ratio.to_owned();
            algorithm.stability = Some(stability);
            algorithm.speed = Some(speed);
        },
    )
}

/// Blur the clip along its motion as `blur` says, or stop when `on` is
/// false, as [`set_enhance`] does.
pub fn set_motion_blur(
    draft: &mut Draft,
    segment_id: &SegmentId,
    on: bool,
    blur: &MotionBlur,
) -> Result<(), EditError> {
    let share = 0.0..=1.0;
    if !share.contains(&blur.blur)
        || !share.contains(&blur.blend)
        || !(MotionBlur::BOTH..=MotionBlur::BACKWARD).contains(&blur.direction)
        || !MotionBlur::TIMES.contains(&blur.times)
    {
        return Err(EditError::MotionBlur);
    }
    set_filled(
        draft,
        segment_id,
        Algorithm::MOTION_BLUR,
        i64::from(on),
        |algorithm| {
            algorithm.strength = Some(blur.blur);
            algorithm.blend = Some(blur.blend);
            algorithm.direction = Some(blur.direction);
            algorithm.speed = Some(blur.times);
        },
    )
}

fn set_algorithm(
    draft: &mut Draft,
    segment_id: &SegmentId,
    kind: &str,
    level: i64,
) -> Result<(), EditError> {
    set_filled(draft, segment_id, kind, level, |_| {})
}

/// [`set_algorithm`], with `fill` setting the algorithm's other fields
/// when it is on.
fn set_filled(
    draft: &mut Draft,
    segment_id: &SegmentId,
    kind: &str,
    level: i64,
    fill: impl FnOnce(&mut Algorithm),
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
        (Some(index), true) => {
            let algorithm = &mut block.algorithms[index];
            algorithm.level = level;
            fill(algorithm);
        }
        (Some(index), false) => {
            block.algorithms.remove(index);
        }
        (None, true) => {
            let mut algorithm = Algorithm {
                kind: kind.to_owned(),
                level,
                ..Algorithm::default()
            };
            fill(&mut algorithm);
            block.algorithms.push(algorithm);
        }
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
        Algorithm, Draft, MaterialAudio, MaterialId, MaterialVideo, MotionBlur, Reframe, Segment,
        TimeRange, Track, TrackAttribute, TrackId, TrackType, VideoAlgorithm,
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

    fn relight(
        session: &mut Session,
        preset: &str,
        strength: f64,
        color: &str,
    ) -> Result<u64, EditError> {
        let revision = session.revision();
        session.apply(
            revision,
            &[Request::SetRelight {
                segment_id: "picture".into(),
                preset: preset.into(),
                strength,
                color: color.into(),
                angle: -45.0,
            }],
        )
    }

    fn reframe(
        session: &mut Session,
        ratio: &str,
        stability: i64,
        speed: i64,
    ) -> Result<u64, EditError> {
        let revision = session.revision();
        session.apply(
            revision,
            &[Request::SetReframe {
                segment_id: "picture".into(),
                ratio: ratio.into(),
                stability,
                speed,
            }],
        )
    }

    fn motion_blur(
        session: &mut Session,
        on: bool,
        blur: f64,
        blend: f64,
        direction: i64,
        times: i64,
    ) -> Result<u64, EditError> {
        let revision = session.revision();
        session.apply(
            revision,
            &[Request::SetMotionBlur {
                segment_id: "picture".into(),
                on,
                blur,
                blend,
                direction,
                times,
            }],
        )
    }

    #[test]
    fn motion_blur_stores_its_settings_and_off_stops_it() {
        let mut session = session_with(None);
        set(&mut session, "picture", Algorithm::HD).unwrap();
        motion_blur(&mut session, true, 0.8, 0.2, MotionBlur::FORWARD, 4).unwrap();
        assert_eq!(
            session.draft().materials.videos[0].motion_blur(),
            Some(MotionBlur {
                blur: 0.8,
                blend: 0.2,
                direction: MotionBlur::FORWARD,
                times: 4
            })
        );
        let before = session.draft().clone();
        for (blur, blend, direction, times) in [
            (1.5, 0.0, 0, 1),
            (0.5, -0.1, 0, 1),
            (f64::NAN, 0.0, 0, 1),
            (0.5, 0.0, 3, 1),
            (0.5, 0.0, 0, 3),
        ] {
            assert_eq!(
                motion_blur(&mut session, true, blur, blend, direction, times),
                Err(EditError::MotionBlur),
                "{blur} {blend} {direction} {times}"
            );
        }
        assert_eq!(session.draft(), &before);
        motion_blur(&mut session, false, 1.0, 0.0, 0, 1).unwrap();
        assert!(session.draft().materials.videos[0].motion_blur().is_none());
        assert_eq!(level(&session), Some(Algorithm::HD));
        session.undo().unwrap();
        assert_eq!(
            session.draft().materials.videos[0]
                .motion_blur()
                .map(|blur| blur.times),
            Some(4)
        );
    }

    #[test]
    fn reframe_stores_its_shape_and_settings_and_an_empty_ratio_stops_it() {
        let mut session = session_with(None);
        set(&mut session, "picture", Algorithm::HD).unwrap();
        reframe(&mut session, "9:16", Reframe::STABLE, Reframe::FAST).unwrap();
        let cut = session.draft().materials.videos[0].reframe().unwrap();
        assert_eq!(
            (cut.ratio.as_str(), cut.stability, cut.speed),
            ("9:16", Reframe::STABLE, Reframe::FAST)
        );
        reframe(&mut session, Reframe::ORIGINAL, 2, 2).unwrap();
        assert_eq!(
            session.draft().materials.videos[0].reframe().unwrap().ratio,
            Reframe::ORIGINAL
        );
        let before = session.draft().clone();
        for (ratio, stability, speed) in [
            ("7:5", 2, 2),
            ("9:16", 0, 2),
            ("9:16", 2, 4),
            ("../x", 2, 2),
        ] {
            assert_eq!(
                reframe(&mut session, ratio, stability, speed),
                Err(EditError::Reframe),
                "{ratio} {stability} {speed}"
            );
        }
        assert_eq!(session.draft(), &before);
        reframe(&mut session, "", 2, 2).unwrap();
        assert!(session.draft().materials.videos[0].reframe().is_none());
        assert_eq!(level(&session), Some(Algorithm::HD));
        session.undo().unwrap();
        assert!(session.draft().materials.videos[0].reframe().is_some());
    }

    #[test]
    fn relight_stores_its_light_beside_the_others_and_an_empty_preset_stops_it() {
        let mut session = session_with(None);
        set(&mut session, "picture", Algorithm::HD).unwrap();
        relight(&mut session, "warm_key", 0.6, "").unwrap();
        let lit = session.draft().materials.videos[0].relight().unwrap();
        assert_eq!(
            (lit.preset.as_str(), lit.strength, lit.angle),
            ("warm_key", 0.6, 315.0)
        );
        relight(&mut session, "custom", 1.0, "#33aaFF").unwrap();
        let lit = session.draft().materials.videos[0].relight().unwrap();
        assert_eq!(
            (lit.preset.as_str(), lit.color.as_str()),
            ("custom", "#33aaFF")
        );
        assert_eq!(level(&session), Some(Algorithm::HD));
        let before = session.draft().clone();
        for (preset, strength, color) in [
            ("Warm Key", 1.0, ""),
            ("../x", 1.0, ""),
            ("warm", 1.5, ""),
            ("warm", f64::NAN, ""),
            ("warm", 1.0, "red"),
            ("warm", 1.0, "#12345g"),
        ] {
            assert_eq!(
                relight(&mut session, preset, strength, color),
                Err(EditError::Relight),
                "{preset} {strength} {color}"
            );
        }
        assert_eq!(
            relight(&mut session, &"a".repeat(65), 1.0, ""),
            Err(EditError::Relight)
        );
        assert_eq!(session.draft(), &before);
        relight(&mut session, "", 1.0, "").unwrap();
        assert!(session.draft().materials.videos[0].relight().is_none());
        assert_eq!(level(&session), Some(Algorithm::HD));
        session.undo().unwrap();
        assert_eq!(
            session.draft().materials.videos[0]
                .relight()
                .map(|lit| lit.preset),
            Some("custom".into())
        );
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
