//! Validation + lowering for the audio / retiming command family:
//! extract_audio, speed and ramps, pitch lock, clip audio mix, denoise.

use super::*;

use crate::wire::{
    ExtractAudio, SetClipAudio, SetClipPitch, SetClipSpeed, SetDenoise, SetSpeedCurve,
};

pub(super) fn extract_audio(
    project: &Project,
    args: &ExtractAudio,
) -> Result<EditCommand, Rejection> {
    let clip = clip_ref(project, args.clip)?;
    let timeline = project.timeline();
    let source_track_id = timeline
        .track_of(clip.id)
        .ok_or_else(|| Rejection::new(format!("clip {} is not on the timeline", args.clip)))?;
    let source_track = timeline.track(source_track_id).ok_or_else(|| {
        Rejection::new(format!(
            "clip {} refers to missing track {}",
            args.clip,
            source_track_id.raw()
        ))
    })?;
    let cutlass_models::ClipSource::Media { media, .. } = &clip.content else {
        return Err(Rejection::new(format!(
            "clip {} is generated; extract_audio requires a media-backed video clip",
            args.clip
        )));
    };
    if source_track.kind != TrackKind::Video {
        return Err(Rejection::new(format!(
            "clip {} is on a {} track; extract_audio requires a video-track clip",
            args.clip,
            kind_name(source_track.kind)
        )));
    }
    if source_track.locked {
        return Err(Rejection::new(format!(
            "video track {} is locked; unlock it before extracting clip {}'s audio",
            source_track_id.raw(),
            args.clip
        )));
    }
    let media = media_ref(project, media.raw())?;
    if media.kind() != cutlass_models::MediaKind::Video {
        return Err(Rejection::new(format!(
            "clip {} does not reference video media; extract_audio requires a video file",
            args.clip
        )));
    }
    if !media.has_audio {
        return Err(Rejection::new(format!(
            "clip {} uses media {} with no audio stream to extract",
            args.clip,
            media.id.raw()
        )));
    }
    if timeline.detached_to_audio_lane(clip.id) {
        return Err(Rejection::new(format!(
            "clip {} already has extracted audio; use its linked audio-lane companion",
            args.clip
        )));
    }
    if let Some(link) = clip.link {
        let other_members: Vec<u64> = timeline
            .tracks_ordered()
            .flat_map(|track| track.clips())
            .filter(|candidate| candidate.id != clip.id && candidate.link == Some(link))
            .map(|candidate| candidate.id.raw())
            .collect();
        if !other_members.is_empty() {
            return Err(Rejection::new(format!(
                "clip {} is already linked to clips {}; call unlink_clips first, then \
                         retry extract_audio",
                args.clip,
                list_ids(other_members)
            )));
        }
    }

    let target = track_ref(project, args.track)?;
    if target.kind != TrackKind::Audio {
        return Err(Rejection::new(format!(
            "track {} is a {} track; extract_audio needs an audio track",
            args.track,
            kind_name(target.kind)
        )));
    }
    if target.locked {
        return Err(Rejection::new(format!(
            "audio track {} is locked; unlock it or choose another audio track",
            args.track
        )));
    }
    if target
        .has_overlap(clip.timeline, None)
        .map_err(|error| Rejection::new(error.to_string()))?
    {
        return Err(Rejection::new(format!(
            "audio track {} already has a clip overlapping clip {}'s exact timeline \
                     range; choose or add a free audio track",
            args.track, args.clip
        )));
    }
    Ok(EditCommand::ExtractAudio {
        clip: clip.id,
        to_track: Some(target.id),
    })
}

pub(super) fn set_clip_speed(
    project: &Project,
    args: &SetClipSpeed,
) -> Result<EditCommand, Rejection> {
    let clip = clip_ref(project, args.clip)?;
    if clip.is_generated() {
        return Err(Rejection::new(format!(
            "clip {} is a generated clip; set_clip_speed only works on media \
                     clips (footage with a source file)",
            args.clip
        )));
    }
    // Omitted fields keep the clip's current retiming.
    let speed = match args.speed {
        Some(speed) => rational_speed(speed)?,
        None => clip.speed,
    };
    Ok(EditCommand::SetClipSpeed {
        clip: clip.id,
        speed,
        reversed: args.reversed.unwrap_or(clip.reversed),
    })
}

pub(super) fn set_speed_curve(
    project: &Project,
    args: &SetSpeedCurve,
) -> Result<EditCommand, Rejection> {
    let clip = clip_ref(project, args.clip)?;
    if clip.is_generated() {
        return Err(Rejection::new(format!(
            "clip {} is a generated clip; set_speed_curve only works on media \
                     clips (footage with a source file)",
            args.clip
        )));
    }
    let curve = match &args.preset {
        Some(name) => Some(cutlass_models::speed_preset(name).ok_or_else(|| {
            Rejection::new(format!(
                "unknown speed ramp preset '{name}'; choose one of: \
                         ramp_up, ramp_down, montage, hero, bullet"
            ))
        })?),
        None => None,
    };
    Ok(EditCommand::SetSpeedCurve {
        clip: clip.id,
        curve,
    })
}

pub(super) fn set_clip_pitch(
    project: &Project,
    args: &SetClipPitch,
) -> Result<EditCommand, Rejection> {
    let clip = clip_ref(project, args.clip)?;
    if clip.is_generated() {
        return Err(Rejection::new(format!(
            "clip {} is a generated clip; set_clip_pitch only works on media \
                     clips (footage with a source file)",
            args.clip
        )));
    }
    Ok(EditCommand::SetClipPitch {
        clip: clip.id,
        preserve_pitch: args.preserve_pitch,
    })
}

pub(super) fn set_denoise(project: &Project, args: &SetDenoise) -> Result<EditCommand, Rejection> {
    let clip = clip_ref(project, args.clip)?;
    if clip.is_generated() {
        return Err(Rejection::new(format!(
            "clip {} is a generated clip; set_denoise only works on media \
                     clips (footage with a source file)",
            args.clip
        )));
    }
    // CapCut keeps a video's sound on the clip itself, so denoise lands
    // on the clip directly — unless its audio was detached to a linked
    // audio lane, where the audible half now lives (same rule as
    // set_clip_audio).
    let timeline = project.timeline();
    if !timeline.carries_own_audio(clip.id) {
        let companion = clip.link.and_then(|link| {
            timeline
                .tracks_ordered()
                .filter(|t| t.kind == TrackKind::Audio)
                .flat_map(|t| t.clips())
                .find(|c| c.link == Some(link))
                .map(|c| c.id.raw())
        });
        return Err(Rejection::new(match companion {
            Some(id) => format!(
                "clip {} is not on an audio lane; its audio plays through \
                         linked clip {id} — call set_denoise on clip {id} instead",
                args.clip
            ),
            None => format!(
                "clip {} is not on an audio lane and has no linked audio \
                         companion; there is nothing audible to clean",
                args.clip
            ),
        }));
    }
    Ok(EditCommand::SetClipDenoise {
        clip: clip.id,
        denoise: args.denoise,
    })
}

pub(super) fn set_clip_audio(
    project: &Project,
    args: &SetClipAudio,
) -> Result<EditCommand, Rejection> {
    let clip = clip_ref(project, args.clip)?;
    if clip.is_generated() {
        return Err(Rejection::new(format!(
            "clip {} is a generated clip; set_clip_audio only works on media \
                     clips (footage with a source file)",
            args.clip
        )));
    }
    // CapCut keeps a video's sound on the clip itself, so volume/fades
    // land on the clip directly — unless its audio was detached to a
    // linked audio lane, where the audible half now lives.
    let timeline = project.timeline();
    if !timeline.carries_own_audio(clip.id) {
        let companion = clip.link.and_then(|link| {
            timeline
                .tracks_ordered()
                .filter(|t| t.kind == TrackKind::Audio)
                .flat_map(|t| t.clips())
                .find(|c| c.link == Some(link))
                .map(|c| c.id.raw())
        });
        return Err(Rejection::new(match companion {
            Some(id) => format!(
                "clip {} is not on an audio lane; its audio plays through \
                         linked clip {id} — call set_clip_audio on clip {id} instead",
                args.clip
            ),
            None => format!(
                "clip {} is not on an audio lane and has no linked audio \
                         companion; there is nothing audible to adjust",
                args.clip
            ),
        }));
    }
    // Omitted volume keeps the clip's gain untouched — a flat level
    // stays flat and, crucially, an envelope is preserved (so "fade
    // the music out" past a keyframed clip doesn't wipe the
    // automation). A present volume sets a flat level (basic slider).
    let volume = match args.volume {
        Some(volume) => {
            if !volume.is_finite() || !(0.0..=10.0).contains(&volume) {
                return Err(Rejection::new(format!(
                    "volume must be between 0 (mute) and 10 (got {volume})"
                )));
            }
            Some(volume as f32)
        }
        None => None,
    };
    let rate = timeline_rate(project);
    let clip_ticks = clip.timeline.duration.value;
    let fade =
        |current: i64, requested: Option<f64>, what: &str| -> Result<RationalTime, Rejection> {
            let Some(seconds) = requested else {
                return Ok(RationalTime::new(current, rate));
            };
            require_non_negative(seconds, what)?;
            let time = timeline_time(project, seconds, what)?;
            if time.value > clip_ticks {
                return Err(Rejection::new(format!(
                    "{what} of {seconds}s is longer than clip {} ({:.3}s)",
                    args.clip,
                    ticks_to_seconds(clip_ticks, rate),
                )));
            }
            Ok(time)
        };
    Ok(EditCommand::SetClipAudio {
        clip: clip.id,
        volume,
        fade_in: fade(clip.fade_in, args.fade_in, "fade_in")?,
        fade_out: fade(clip.fade_out, args.fade_out, "fade_out")?,
    })
}
