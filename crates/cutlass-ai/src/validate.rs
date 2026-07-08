//! Validation + lowering: wire commands → engine commands, against a live
//! project snapshot.
//!
//! This is the guardrail chokepoint. Every rejection carries a
//! model-readable message (including the ids that *do* exist) so the agent
//! loop can feed it straight back as a tool result and let the model correct
//! course. Checks here are for good error messages and the whitelist; the
//! engine remains the authority (overlaps, source bounds, rate math) and
//! re-validates everything on apply.

use cutlass_commands::{Command, EditCommand};
use cutlass_models::{
    AnimationRef, AnimationSlot, AudioRole, CanvasAspect, ChromaKey, Clip, ClipId, ClipParam,
    ClipTransform, CropRect, Easing, Filter, Generator, Marker, MarkerColor, MarkerId, Mask,
    MaskKind, MediaId, Param, ParamValue, Project, Rational, RationalTime, StabilizeLevel,
    TimeRange, TrackId, TrackKind, animation_spec, filter_catalog, filter_spec,
};

use crate::wire::{
    WireAnimationSlot, WireAudioRole, WireCanvasAspect, WireChromaKey, WireClipParam, WireCommand,
    WireEasing, WireGenerator, WireMarkerColor, WireMask, WireMaskKind, WireShape,
    WireStabilizeLevel, WireTrackKind,
};

/// A wire command the project as it stands cannot accept. The message is
/// written for the model: it names the problem and the valid alternatives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rejection {
    pub message: String,
}

impl Rejection {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for Rejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Rejection {}

/// Validate `command` against `project` and lower it to an engine command.
pub fn validate(command: &WireCommand, project: &Project) -> Result<Command, Rejection> {
    let edit = match command {
        WireCommand::AddTrack(args) => EditCommand::AddTrack {
            kind: track_kind(args.kind),
            name: args.name.clone(),
            index: args.index.map(|i| i as usize),
            pinned: false,
        },
        WireCommand::AddClip(args) => {
            let track = track_ref(project, args.track)?;
            let media = media_ref(project, args.media)?;
            require_media_lane(track.kind, args.track)?;
            let (source, start) = media_placement(
                project,
                media,
                args.source_start,
                args.source_duration,
                args.start,
                "start",
            )?;
            EditCommand::AddClip {
                track: track.id,
                media: media.id,
                source,
                start,
            }
        }
        WireCommand::AddGenerated(args) => {
            let track = track_ref(project, args.track)?;
            let generator = lower_generator(&args.generator, None);
            require_generator_lane(track, &generator)?;
            let timeline = timeline_range(project, args.start, args.duration)?;
            EditCommand::AddGenerated {
                track: track.id,
                generator,
                timeline,
            }
        }
        WireCommand::SetGenerator(args) => {
            let clip = clip_ref(project, args.clip)?;
            let Some(current) = generated_content(clip) else {
                return Err(Rejection::new(format!(
                    "clip {} is a media clip; set_generator only works on generated \
                     clips (text, solid, shape)",
                    args.clip
                )));
            };
            EditCommand::SetGenerator {
                clip: clip.id,
                generator: lower_generator(&args.generator, Some(current)),
            }
        }
        WireCommand::SetClipTransform(args) => {
            let clip = clip_ref(project, args.clip)?;
            // Omitted properties keep their current value — sampled at the
            // clip start for animated params (the agent edits whole-clip
            // placement; keyframe-level edits get their own commands).
            let current = clip.transform.sample(0);
            let transform = ClipTransform {
                position: [
                    args.position_x.map_or(current.position[0], |v| v as f32),
                    args.position_y.map_or(current.position[1], |v| v as f32),
                ],
                anchor_point: [
                    args.anchor_x.map_or(current.anchor_point[0], |v| v as f32),
                    args.anchor_y.map_or(current.anchor_point[1], |v| v as f32),
                ],
                scale: args.scale.map_or(current.scale, |v| v as f32),
                rotation: args.rotation.map_or(current.rotation, |v| v as f32),
                opacity: args.opacity.map_or(current.opacity, |v| v as f32),
            };
            transform
                .validate()
                .map_err(|e| Rejection::new(format!("invalid transform: {e}")))?;
            EditCommand::SetClipTransform {
                clip: clip.id,
                transform,
                at: None,
            }
        }
        WireCommand::SetClipCrop(args) => {
            let clip = clip_ref(project, args.clip)?;
            let timeline = project.timeline();
            let on_audio_lane = timeline
                .track_of(clip.id)
                .and_then(|id| timeline.track(id))
                .is_some_and(|t| t.kind == TrackKind::Audio);
            if on_audio_lane {
                return Err(Rejection::new(format!(
                    "clip {} is on an audio lane; there is no frame to crop",
                    args.clip
                )));
            }
            // Omitted edges keep the clip's current framing. Current insets
            // derive from the stored kept-region rect.
            let current = clip.crop;
            let inset = |requested: Option<f64>, current: f32, what: &str| {
                let Some(v) = requested else {
                    return Ok(current);
                };
                if !v.is_finite() || !(0.0..1.0).contains(&v) {
                    return Err(Rejection::new(format!(
                        "{what} must be a fraction between 0 and 1 (got {v})"
                    )));
                }
                Ok(v as f32)
            };
            let left = inset(args.left, current.x, "left")?;
            let top = inset(args.top, current.y, "top")?;
            let right = inset(args.right, 1.0 - current.x - current.w, "right")?;
            let bottom = inset(args.bottom, 1.0 - current.y - current.h, "bottom")?;
            let crop = CropRect {
                x: left,
                y: top,
                w: 1.0 - left - right,
                h: 1.0 - top - bottom,
            };
            crop.validate().map_err(|_| {
                Rejection::new(format!(
                    "crop leaves no visible frame: left {left} + right {right} and \
                     top {top} + bottom {bottom} must each keep at least 1% of the frame"
                ))
            })?;
            EditCommand::SetClipCrop {
                clip: clip.id,
                crop,
                flip_h: args.flip_h.unwrap_or(clip.flip_h),
                flip_v: args.flip_v.unwrap_or(clip.flip_v),
            }
        }
        WireCommand::AddEffect(args) => {
            let clip = clip_ref(project, args.clip)?;
            reject_audio_lane(project, clip, "effects need a visual frame", args.clip)?;
            if cutlass_models::effect_spec(&args.effect).is_none() {
                return Err(Rejection::new(format!(
                    "unknown effect '{}'; available effects: {}",
                    args.effect,
                    effect_ids()
                )));
            }
            EditCommand::AddEffect {
                clip: clip.id,
                effect_id: args.effect.clone(),
            }
        }
        WireCommand::RemoveEffect(args) => {
            let clip = clip_ref(project, args.clip)?;
            let index = effect_index(clip, args.index, args.clip)?;
            EditCommand::RemoveEffect {
                clip: clip.id,
                index,
            }
        }
        WireCommand::SetEffectParam(args) => {
            let clip = clip_ref(project, args.clip)?;
            let index = effect_index(clip, args.index, args.clip)?;
            let instance = &clip.effects[index];
            let spec = cutlass_models::effect_spec(&instance.effect_id).ok_or_else(|| {
                Rejection::new(format!(
                    "effect '{}' on clip {} is not in the catalog",
                    instance.effect_id, args.clip
                ))
            })?;
            let slot = spec
                .params
                .iter()
                .position(|p| p.name == args.param)
                .ok_or_else(|| {
                    let names: Vec<&str> = spec.params.iter().map(|p| p.name).collect();
                    Rejection::new(format!(
                        "effect '{}' has no parameter '{}'; parameters: {}",
                        instance.effect_id,
                        args.param,
                        names.join(", ")
                    ))
                })?;
            let p = &spec.params[slot];
            let v = args.value;
            if !v.is_finite() || v < f64::from(p.min) || v > f64::from(p.max) {
                return Err(Rejection::new(format!(
                    "{} must be between {} and {} (got {})",
                    args.param, p.min, p.max, v
                )));
            }
            EditCommand::SetEffectParam {
                clip: clip.id,
                index,
                param: slot,
                value: v as f32,
            }
        }
        WireCommand::AddTransition(args) => {
            let clip = clip_ref(project, args.clip)?;
            reject_audio_lane(project, clip, "transitions need a visual frame", args.clip)?;
            if cutlass_models::transition_spec(&args.transition).is_none() {
                return Err(Rejection::new(format!(
                    "unknown transition '{}'; available transitions: {}",
                    args.transition,
                    transition_ids()
                )));
            }
            if !has_right_neighbor(project, clip) {
                return Err(Rejection::new(format!(
                    "clip {} has no clip butting against its right edge to transition into",
                    args.clip
                )));
            }
            EditCommand::AddTransition {
                clip: clip.id,
                transition_id: args.transition.clone(),
            }
        }
        WireCommand::RemoveTransition(args) => {
            let clip = clip_ref(project, args.clip)?;
            require_transition(project, clip, args.clip)?;
            EditCommand::RemoveTransition { clip: clip.id }
        }
        WireCommand::SetTransition(args) => {
            let clip = clip_ref(project, args.clip)?;
            require_transition(project, clip, args.clip)?;
            if !(args.seconds.is_finite()) || args.seconds <= 0.0 {
                return Err(Rejection::new(format!(
                    "transition duration must be positive (got {}s)",
                    args.seconds
                )));
            }
            let duration =
                seconds_to_ticks(args.seconds, timeline_rate(project), "duration")?.max(1);
            EditCommand::SetTransition {
                clip: clip.id,
                duration,
            }
        }
        WireCommand::SetParamKeyframe(args) => {
            let clip = clip_ref(project, args.clip)?;
            let at = keyframe_position(project, clip, args.at)?;
            let value = param_value(args.param, args.value, args.position)?;
            EditCommand::SetParamKeyframe {
                clip: clip.id,
                param: clip_param(args.param),
                at,
                value,
                easing: easing(args.easing),
            }
        }
        WireCommand::RemoveParamKeyframe(args) => {
            let clip = clip_ref(project, args.clip)?;
            let at = keyframe_position(project, clip, args.at)?;
            EditCommand::RemoveParamKeyframe {
                clip: clip.id,
                param: clip_param(args.param),
                at,
            }
        }
        WireCommand::SetClipSpeed(args) => {
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
            EditCommand::SetClipSpeed {
                clip: clip.id,
                speed,
                reversed: args.reversed.unwrap_or(clip.reversed),
            }
        }
        WireCommand::SetSpeedCurve(args) => {
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
            EditCommand::SetSpeedCurve {
                clip: clip.id,
                curve,
            }
        }
        WireCommand::SetClipPitch(args) => {
            let clip = clip_ref(project, args.clip)?;
            if clip.is_generated() {
                return Err(Rejection::new(format!(
                    "clip {} is a generated clip; set_clip_pitch only works on media \
                     clips (footage with a source file)",
                    args.clip
                )));
            }
            EditCommand::SetClipPitch {
                clip: clip.id,
                preserve_pitch: args.preserve_pitch,
            }
        }
        WireCommand::SetDenoise(args) => {
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
            EditCommand::SetClipDenoise {
                clip: clip.id,
                denoise: args.denoise,
            }
        }
        WireCommand::SetClipMask(args) => {
            let clip = clip_ref(project, args.clip)?;
            reject_audio_lane(project, clip, "masks need a visual frame", args.clip)?;
            if clip.is_generated() {
                return Err(Rejection::new(format!(
                    "clip {} is a generated clip; set_clip_mask only works on media \
                     clips (footage with a source file)",
                    args.clip
                )));
            }
            let mask = match &args.mask {
                None => None,
                Some(wire) => Some(lower_mask(wire)?),
            };
            EditCommand::SetClipMask {
                clip: clip.id,
                mask,
            }
        }
        WireCommand::SetClipChroma(args) => {
            let clip = clip_ref(project, args.clip)?;
            reject_audio_lane(project, clip, "chroma key needs a visual frame", args.clip)?;
            if clip.is_generated() {
                return Err(Rejection::new(format!(
                    "clip {} is a generated clip; set_clip_chroma only works on media \
                     clips (footage with a source file)",
                    args.clip
                )));
            }
            let chroma = match &args.chroma {
                None => None,
                Some(wire) => Some(lower_chroma(wire)?),
            };
            EditCommand::SetClipChroma {
                clip: clip.id,
                chroma,
            }
        }
        WireCommand::SetClipStabilize(args) => {
            let clip = clip_ref(project, args.clip)?;
            reject_audio_lane(
                project,
                clip,
                "stabilization needs a visual frame",
                args.clip,
            )?;
            if clip.is_generated() {
                return Err(Rejection::new(format!(
                    "clip {} is a generated clip; set_clip_stabilize only works on media \
                     clips (footage with a source file)",
                    args.clip
                )));
            }
            if let cutlass_models::ClipSource::Media { media, .. } = &clip.content
                && project.media(*media).is_some_and(|m| m.is_image)
            {
                return Err(Rejection::new(format!(
                    "clip {} is a still image; stabilization requires video",
                    args.clip
                )));
            }
            let stabilize = args.level.map(lower_stabilize);
            EditCommand::SetClipStabilize {
                clip: clip.id,
                stabilize,
            }
        }
        WireCommand::SetClipFilter(args) => {
            let clip = clip_ref(project, args.clip)?;
            reject_audio_lane(project, clip, "filters need a visual frame", args.clip)?;
            let filter = match &args.filter {
                None => None,
                Some(wire) => Some(lower_filter(wire)?),
            };
            EditCommand::SetClipFilter {
                clip: clip.id,
                filter,
            }
        }
        WireCommand::SetClipAdjustments(args) => {
            let clip = clip_ref(project, args.clip)?;
            reject_audio_lane(project, clip, "adjustments need a visual frame", args.clip)?;
            let mut adjust = clip.adjust;
            if let Some(v) = args.brightness {
                adjust.brightness = unit_slider(v, "brightness")?;
            }
            if let Some(v) = args.contrast {
                adjust.contrast = unit_slider(v, "contrast")?;
            }
            if let Some(v) = args.saturation {
                adjust.saturation = unit_slider(v, "saturation")?;
            }
            if let Some(v) = args.exposure {
                adjust.exposure = unit_slider(v, "exposure")?;
            }
            if let Some(v) = args.temperature {
                adjust.temperature = unit_slider(v, "temperature")?;
            }
            adjust
                .validate()
                .map_err(|e| Rejection::new(e.to_string()))?;
            EditCommand::SetClipAdjustments {
                clip: clip.id,
                adjust,
            }
        }
        WireCommand::SetClipAnimation(args) => {
            let clip = clip_ref(project, args.clip)?;
            reject_audio_lane(project, clip, "animations need a visual frame", args.clip)?;
            let slot = lower_animation_slot(args.slot);
            let animation = match &args.animation {
                None => None,
                Some(id) => {
                    let spec = animation_spec(id).ok_or_else(|| {
                        Rejection::new(format!(
                            "unknown animation '{id}'; available animations include fade_in, \
                             fade_out, pulse, slide_up, zoom_in, and others from the catalog"
                        ))
                    })?;
                    if spec.slot != slot {
                        return Err(Rejection::new(format!(
                            "animation '{id}' does not fit the {} slot",
                            match args.slot {
                                WireAnimationSlot::In => "in",
                                WireAnimationSlot::Out => "out",
                                WireAnimationSlot::Combo => "combo",
                            }
                        )));
                    }
                    if spec.text_only
                        && !matches!(
                            clip.content,
                            cutlass_models::ClipSource::Generated(Generator::Text { .. })
                        )
                    {
                        return Err(Rejection::new(format!(
                            "animation '{id}' is a text-only preset"
                        )));
                    }
                    Some(AnimationRef::new(id.clone()))
                }
            };
            EditCommand::SetClipAnimation {
                clip: clip.id,
                slot,
                animation,
            }
        }
        WireCommand::SetAudioRole(args) => {
            let clip = clip_ref(project, args.clip)?;
            let timeline = project.timeline();
            let on_audio = timeline
                .track_of(clip.id)
                .and_then(|id| timeline.track(id))
                .is_some_and(|t| t.kind == TrackKind::Audio);
            if !on_audio {
                return Err(Rejection::new(format!(
                    "clip {} is not on an audio lane; set_audio_role only works on audio clips",
                    args.clip
                )));
            }
            EditCommand::SetAudioRole {
                clip: clip.id,
                role: args.role.map(lower_audio_role),
            }
        }
        WireCommand::SetClipAudio(args) => {
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
            let fade = |current: i64,
                        requested: Option<f64>,
                        what: &str|
             -> Result<RationalTime, Rejection> {
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
            EditCommand::SetClipAudio {
                clip: clip.id,
                volume,
                fade_in: fade(clip.fade_in, args.fade_in, "fade_in")?,
                fade_out: fade(clip.fade_out, args.fade_out, "fade_out")?,
            }
        }
        WireCommand::SetParamConstant(args) => {
            let clip = clip_ref(project, args.clip)?;
            let value = param_value(args.param, args.value, args.position)?;
            EditCommand::SetParamConstant {
                clip: clip.id,
                param: clip_param(args.param),
                value,
            }
        }
        WireCommand::SplitClip(args) => {
            let clip = clip_ref(project, args.clip)?;
            let at = timeline_time(project, args.at, "at")?;
            let tl = clip.timeline;
            if at.value <= tl.start.value || at.value >= tl.end_tick() {
                let rate = timeline_rate(project);
                return Err(Rejection::new(format!(
                    "split position {:.3}s is not strictly inside clip {} \
                     ({:.3}s to {:.3}s)",
                    args.at,
                    args.clip,
                    ticks_to_seconds(tl.start.value, rate),
                    ticks_to_seconds(tl.end_tick(), rate),
                )));
            }
            EditCommand::SplitClip { clip: clip.id, at }
        }
        WireCommand::TrimClip(args) => {
            let clip = clip_ref(project, args.clip)?;
            let timeline = timeline_range(project, args.start, args.duration)?;
            EditCommand::TrimClip {
                clip: clip.id,
                timeline,
            }
        }
        WireCommand::MoveClip(args) => {
            let clip = clip_ref(project, args.clip)?;
            let track = track_ref(project, args.to_track)?;
            if !track.kind.accepts_content(&clip.content) {
                return Err(Rejection::new(format!(
                    "clip {} cannot live on track {} ({} lane)",
                    args.clip,
                    args.to_track,
                    kind_name(track.kind),
                )));
            }
            let start = timeline_time(project, args.start, "start")?;
            require_non_negative(args.start, "start")?;
            EditCommand::MoveClip {
                clip: clip.id,
                to_track: track.id,
                start,
            }
        }
        WireCommand::RemoveClip(args) => EditCommand::RemoveClip {
            clip: clip_ref(project, args.clip)?.id,
        },
        WireCommand::RemoveTrack(args) => EditCommand::RemoveTrack {
            track: track_ref(project, args.track)?.id,
        },
        WireCommand::SetTrackEnabled(args) => EditCommand::SetTrackEnabled {
            track: track_ref(project, args.track)?.id,
            enabled: args.enabled,
        },
        WireCommand::SetTrackMuted(args) => EditCommand::SetTrackMuted {
            track: track_ref(project, args.track)?.id,
            muted: args.muted,
        },
        WireCommand::SetTrackLocked(args) => EditCommand::SetTrackLocked {
            track: track_ref(project, args.track)?.id,
            locked: args.locked,
        },
        WireCommand::RippleDelete(args) => EditCommand::RippleDelete {
            clip: clip_ref(project, args.clip)?.id,
        },
        WireCommand::ShiftClips(args) => {
            let track = track_ref(project, args.track)?;
            require_non_negative(args.from, "from")?;
            let from = timeline_time(project, args.from, "from")?;
            let delta = timeline_time_signed(project, args.delta, "delta")?;
            if delta.value == 0 {
                return Err(Rejection::new(format!(
                    "delta of {:+.4}s rounds to zero frames at the timeline rate; \
                     nothing would move",
                    args.delta
                )));
            }
            EditCommand::ShiftClips {
                track: track.id,
                from,
                delta,
            }
        }
        WireCommand::RippleInsert(args) => {
            let track = track_ref(project, args.track)?;
            let media = media_ref(project, args.media)?;
            require_media_lane(track.kind, args.track)?;
            let (source, at) = media_placement(
                project,
                media,
                args.source_start,
                args.source_duration,
                args.at,
                "at",
            )?;
            EditCommand::RippleInsert {
                track: track.id,
                media: media.id,
                source,
                at,
            }
        }
        WireCommand::LinkClips(args) => {
            if args.clips.len() < 2 {
                return Err(Rejection::new(
                    "link_clips needs at least two clip ids".to_string(),
                ));
            }
            let mut clips = Vec::with_capacity(args.clips.len());
            for &raw in &args.clips {
                clips.push(clip_ref(project, raw)?.id);
            }
            EditCommand::LinkClips { clips }
        }
        WireCommand::AddMarker(args) => {
            require_non_negative(args.at, "at")?;
            let at = timeline_time(project, args.at, "at")?;
            EditCommand::AddMarker {
                at,
                name: args.name.clone().unwrap_or_default(),
                color: args.color.map(marker_color),
            }
        }
        WireCommand::RemoveMarker(args) => EditCommand::RemoveMarker {
            marker: marker_ref(project, args.marker)?.id,
        },
        WireCommand::SetMarker(args) => {
            let marker = marker_ref(project, args.marker)?;
            let at = match args.at {
                Some(seconds) => {
                    require_non_negative(seconds, "at")?;
                    timeline_time(project, seconds, "at")?
                }
                None => marker.tick,
            };
            EditCommand::SetMarker {
                marker: marker.id,
                at,
                name: args.name.clone().unwrap_or_else(|| marker.name.clone()),
                color: args.color.map(marker_color).unwrap_or(marker.color),
            }
        }
        WireCommand::SetCanvas(args) => {
            let current = project.timeline().canvas();
            EditCommand::SetCanvas {
                aspect: args.aspect.map(canvas_aspect).unwrap_or(current.aspect),
                background: args.background.unwrap_or(current.background),
            }
        }
    };
    Ok(Command::Edit(edit))
}

// --- entity lookups (errors list what exists) ------------------------------

const MAX_LISTED_IDS: usize = 32;

fn list_ids(mut ids: Vec<u64>) -> String {
    if ids.is_empty() {
        return "none".to_string();
    }
    ids.sort_unstable();
    let extra = ids.len().saturating_sub(MAX_LISTED_IDS);
    let mut out = ids
        .into_iter()
        .take(MAX_LISTED_IDS)
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    if extra > 0 {
        out.push_str(&format!(" (and {extra} more)"));
    }
    out
}

fn unit_slider(value: f64, name: &str) -> Result<f32, Rejection> {
    if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
        return Err(Rejection::new(format!(
            "{name} must be between -1 and 1 (got {value})"
        )));
    }
    Ok(value as f32)
}

fn lower_mask_kind(kind: WireMaskKind) -> MaskKind {
    match kind {
        WireMaskKind::Linear => MaskKind::Linear,
        WireMaskKind::Mirror => MaskKind::Mirror,
        WireMaskKind::Circle => MaskKind::Circle,
        WireMaskKind::Rectangle => MaskKind::Rectangle,
        WireMaskKind::Heart => MaskKind::Heart,
        WireMaskKind::Star => MaskKind::Star,
    }
}

fn lower_mask(wire: &WireMask) -> Result<Mask, Rejection> {
    let feather = wire.feather.unwrap_or(0.0);
    if !feather.is_finite() || !(0.0..=1.0).contains(&feather) {
        return Err(Rejection::new(format!(
            "mask feather must be between 0 and 1 (got {feather})"
        )));
    }
    let mask = Mask {
        kind: lower_mask_kind(wire.kind),
        feather: feather as f32,
        invert: wire.invert.unwrap_or(false),
    };
    mask.validate().map_err(|e| Rejection::new(e.to_string()))?;
    Ok(mask)
}

fn lower_chroma(wire: &WireChromaKey) -> Result<ChromaKey, Rejection> {
    let strength = wire.strength.unwrap_or(0.0);
    let shadow = wire.shadow.unwrap_or(0.0);
    for (name, value) in [("strength", strength), ("shadow", shadow)] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(Rejection::new(format!(
                "chroma {name} must be between 0 and 1 (got {value})"
            )));
        }
    }
    let chroma = ChromaKey {
        rgb: wire.rgb,
        strength: strength as f32,
        shadow: shadow as f32,
    };
    chroma
        .validate()
        .map_err(|e| Rejection::new(e.to_string()))?;
    Ok(chroma)
}

fn lower_stabilize(level: WireStabilizeLevel) -> StabilizeLevel {
    match level {
        WireStabilizeLevel::Recommended => StabilizeLevel::Recommended,
        WireStabilizeLevel::Smooth => StabilizeLevel::Smooth,
        WireStabilizeLevel::MaxSmooth => StabilizeLevel::MaxSmooth,
    }
}

fn lower_filter(wire: &crate::wire::WireFilter) -> Result<Filter, Rejection> {
    let intensity = wire.intensity.unwrap_or(0.8);
    if !intensity.is_finite() || !(0.0..=1.0).contains(&intensity) {
        return Err(Rejection::new(format!(
            "filter intensity must be between 0 and 1 (got {intensity})"
        )));
    }
    let filter = Filter {
        id: wire.id.clone(),
        intensity: intensity as f32,
    };
    filter.validate().map_err(|e| {
        let ids = filter_catalog()
            .iter()
            .map(|s| s.id)
            .collect::<Vec<_>>()
            .join(", ");
        if filter_spec(&wire.id).is_none() {
            Rejection::new(format!(
                "unknown filter '{}'; available filters: {ids}",
                wire.id
            ))
        } else {
            Rejection::new(e.to_string())
        }
    })?;
    Ok(filter)
}

fn lower_animation_slot(slot: WireAnimationSlot) -> AnimationSlot {
    match slot {
        WireAnimationSlot::In => AnimationSlot::In,
        WireAnimationSlot::Out => AnimationSlot::Out,
        WireAnimationSlot::Combo => AnimationSlot::Combo,
    }
}

fn lower_audio_role(role: WireAudioRole) -> AudioRole {
    match role {
        WireAudioRole::Music => AudioRole::Music,
        WireAudioRole::Sfx => AudioRole::Sfx,
        WireAudioRole::Voiceover => AudioRole::Voiceover,
        WireAudioRole::Extracted => AudioRole::Extracted,
    }
}

fn clip_ref(project: &Project, raw: u64) -> Result<&Clip, Rejection> {
    project.clip(ClipId::from_raw(raw)).ok_or_else(|| {
        let existing = project
            .timeline()
            .tracks_ordered()
            .flat_map(|t| t.clips())
            .map(|c| c.id.raw())
            .collect();
        Rejection::new(format!(
            "clip {raw} does not exist; clips on the timeline: {}",
            list_ids(existing)
        ))
    })
}

/// Comma-separated catalog effect ids, for rejection messages.
fn effect_ids() -> String {
    cutlass_models::effect_catalog()
        .iter()
        .map(|s| s.id)
        .collect::<Vec<_>>()
        .join(", ")
}

fn transition_ids() -> String {
    cutlass_models::transition_catalog()
        .iter()
        .map(|s| s.id)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Whether a clip butts directly against a following clip on its track (the
/// next clip starts exactly where this one ends) — the precondition for a
/// transition at its right junction.
fn has_right_neighbor(project: &Project, clip: &Clip) -> bool {
    let timeline = project.timeline();
    let Some(track) = timeline.track_of(clip.id).and_then(|id| timeline.track(id)) else {
        return false;
    };
    let end = clip.timeline.end_tick();
    track
        .clips()
        .any(|c| c.id != clip.id && c.timeline.start.value == end)
}

/// Reject when the clip carries no transition at its right junction.
fn require_transition(project: &Project, clip: &Clip, raw_clip: u64) -> Result<(), Rejection> {
    let timeline = project.timeline();
    let has = timeline
        .track_of(clip.id)
        .and_then(|id| timeline.track(id))
        .is_some_and(|t| t.transition_at(clip.id).is_some());
    if !has {
        return Err(Rejection::new(format!(
            "clip {raw_clip} has no transition at its right cut"
        )));
    }
    Ok(())
}

/// Resolve a wire effect-chain index against a clip, rejecting out-of-range
/// indices with the clip's current chain length.
fn effect_index(clip: &Clip, index: u32, raw_clip: u64) -> Result<usize, Rejection> {
    let index = index as usize;
    if index >= clip.effects.len() {
        return Err(Rejection::new(format!(
            "clip {raw_clip} has no effect at index {index} ({} on the chain)",
            clip.effects.len()
        )));
    }
    Ok(index)
}

/// Reject clips on an audio lane, which have no frame to operate on.
fn reject_audio_lane(
    project: &Project,
    clip: &Clip,
    why: &str,
    raw_clip: u64,
) -> Result<(), Rejection> {
    let timeline = project.timeline();
    let on_audio = timeline
        .track_of(clip.id)
        .and_then(|id| timeline.track(id))
        .is_some_and(|t| t.kind == TrackKind::Audio);
    if on_audio {
        return Err(Rejection::new(format!(
            "clip {raw_clip} is on an audio lane; {why}"
        )));
    }
    Ok(())
}

fn track_ref(project: &Project, raw: u64) -> Result<&cutlass_models::Track, Rejection> {
    project
        .timeline()
        .track(TrackId::from_raw(raw))
        .ok_or_else(|| {
            let existing = project
                .timeline()
                .tracks_ordered()
                .map(|t| t.id.raw())
                .collect();
            Rejection::new(format!(
                "track {raw} does not exist; tracks on the timeline: {}",
                list_ids(existing)
            ))
        })
}

fn media_ref(project: &Project, raw: u64) -> Result<&cutlass_models::MediaSource, Rejection> {
    project.media(MediaId::from_raw(raw)).ok_or_else(|| {
        let existing = project.media_iter().map(|m| m.id.raw()).collect();
        Rejection::new(format!(
            "media {raw} does not exist; media in the pool: {}",
            list_ids(existing)
        ))
    })
}

fn marker_ref(project: &Project, raw: u64) -> Result<&Marker, Rejection> {
    project
        .timeline()
        .marker(MarkerId::from_raw(raw))
        .ok_or_else(|| {
            let existing = project
                .timeline()
                .markers()
                .iter()
                .map(|m| m.id.raw())
                .collect();
            Rejection::new(format!(
                "marker {raw} does not exist; markers on the timeline: {}",
                list_ids(existing)
            ))
        })
}

fn marker_color(color: WireMarkerColor) -> MarkerColor {
    match color {
        WireMarkerColor::Teal => MarkerColor::Teal,
        WireMarkerColor::Blue => MarkerColor::Blue,
        WireMarkerColor::Purple => MarkerColor::Purple,
        WireMarkerColor::Pink => MarkerColor::Pink,
        WireMarkerColor::Red => MarkerColor::Red,
        WireMarkerColor::Orange => MarkerColor::Orange,
        WireMarkerColor::Yellow => MarkerColor::Yellow,
        WireMarkerColor::Green => MarkerColor::Green,
    }
}

fn canvas_aspect(aspect: WireCanvasAspect) -> CanvasAspect {
    match aspect {
        WireCanvasAspect::Auto => CanvasAspect::Auto,
        WireCanvasAspect::Wide16x9 => CanvasAspect::Wide16x9,
        WireCanvasAspect::Tall9x16 => CanvasAspect::Tall9x16,
        WireCanvasAspect::Square1x1 => CanvasAspect::Square1x1,
        WireCanvasAspect::Portrait4x5 => CanvasAspect::Portrait4x5,
        WireCanvasAspect::Cinema21x9 => CanvasAspect::Cinema21x9,
    }
}

// --- kind rules -------------------------------------------------------------

fn track_kind(kind: WireTrackKind) -> TrackKind {
    match kind {
        WireTrackKind::Video => TrackKind::Video,
        WireTrackKind::Audio => TrackKind::Audio,
        WireTrackKind::Text => TrackKind::Text,
        WireTrackKind::Sticker => TrackKind::Sticker,
    }
}

fn kind_name(kind: TrackKind) -> &'static str {
    match kind {
        TrackKind::Video => "video",
        TrackKind::Audio => "audio",
        TrackKind::Text => "text",
        TrackKind::Sticker => "sticker",
        TrackKind::Effect => "effect",
        TrackKind::Filter => "filter",
        TrackKind::Adjustment => "adjustment",
    }
}

fn require_media_lane(kind: TrackKind, raw: u64) -> Result<(), Rejection> {
    if matches!(kind, TrackKind::Video | TrackKind::Audio) {
        Ok(())
    } else {
        Err(Rejection::new(format!(
            "track {raw} is a {} lane; media clips need a video or audio track",
            kind_name(kind)
        )))
    }
}

fn require_generator_lane(
    track: &cutlass_models::Track,
    generator: &Generator,
) -> Result<(), Rejection> {
    let content = cutlass_models::ClipSource::Generated(generator.clone());
    if track.kind.accepts_content(&content) {
        return Ok(());
    }
    let needed = match generator {
        Generator::Text { .. } => "a text track",
        _ => "a sticker (overlay) track",
    };
    Err(Rejection::new(format!(
        "track {} is a {} lane; this generator needs {needed}",
        track.id.raw(),
        kind_name(track.kind),
    )))
}

/// Lower a wire generator. When replacing the content of an existing text
/// clip, the current style is preserved (the agent edits words, not looks).
fn lower_generator(wire: &WireGenerator, current: Option<&Generator>) -> Generator {
    match wire {
        WireGenerator::Text { content } => {
            let style = match current {
                Some(Generator::Text { style, .. }) => style.clone(),
                _ => Default::default(),
            };
            Generator::Text {
                content: content.clone(),
                style,
            }
        }
        WireGenerator::Solid { rgba } => Generator::SolidColor { rgba: *rgba },
        WireGenerator::Shape {
            shape,
            rgba,
            width,
            height,
        } => {
            let (shape_w, shape_h, corner_radius, stroke) = match current {
                Some(Generator::Shape {
                    width: w,
                    height: h,
                    corner_radius,
                    stroke,
                    ..
                }) => (
                    w.sample(0),
                    h.sample(0),
                    corner_radius.clone(),
                    stroke.clone(),
                ),
                _ => (
                    cutlass_models::SHAPE_DROP_WIDTH,
                    cutlass_models::SHAPE_DROP_HEIGHT,
                    Param::Constant(0.0),
                    None,
                ),
            };
            Generator::Shape {
                shape: match shape {
                    WireShape::Rectangle => cutlass_models::Shape::Rectangle,
                    WireShape::Ellipse => cutlass_models::Shape::Ellipse,
                },
                rgba: Param::Constant(*rgba),
                width: Param::Constant(width.unwrap_or(shape_w)),
                height: Param::Constant(height.unwrap_or(shape_h)),
                corner_radius,
                stroke,
            }
        }
    }
}

fn generated_content(clip: &Clip) -> Option<&Generator> {
    match &clip.content {
        cutlass_models::ClipSource::Generated(g) => Some(g),
        cutlass_models::ClipSource::Media { .. } => None,
    }
}

// --- seconds → ticks ---------------------------------------------------------

fn timeline_rate(project: &Project) -> Rational {
    project.timeline().frame_rate
}

/// Lower a wire speed multiplier to the engine's exact rational, snapped to
/// hundredths (2.0 → 2/1, 0.5 → 1/2, 0.333 → 33/100). CapCut's UI range.
fn rational_speed(speed: f64) -> Result<Rational, Rejection> {
    if !speed.is_finite() || !(0.05..=100.0).contains(&speed) {
        return Err(Rejection::new(format!(
            "speed must be between 0.05 and 100 (got {speed})"
        )));
    }
    let num = (speed * 100.0).round() as i32;
    let g = gcd(num, 100);
    Ok(Rational::new(num / g, 100 / g))
}

fn gcd(mut a: i32, mut b: i32) -> i32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

fn clip_param(param: WireClipParam) -> ClipParam {
    match param {
        WireClipParam::Position => ClipParam::Position,
        WireClipParam::Scale => ClipParam::Scale,
        WireClipParam::Rotation => ClipParam::Rotation,
        WireClipParam::Opacity => ClipParam::Opacity,
        WireClipParam::Volume => ClipParam::Volume,
    }
}

fn easing(easing: Option<WireEasing>) -> Easing {
    match easing {
        None | Some(WireEasing::Linear) => Easing::Linear,
        Some(WireEasing::EaseIn) => Easing::EaseIn,
        Some(WireEasing::EaseOut) => Easing::EaseOut,
        Some(WireEasing::EaseInOut) => Easing::EaseInOut,
    }
}

/// Build the typed parameter value from the wire's `value` / `position`
/// fields, rejecting the wrong shape with a message naming the right one.
fn param_value(
    param: WireClipParam,
    value: Option<f64>,
    position: Option<[f64; 2]>,
) -> Result<ParamValue, Rejection> {
    match param {
        WireClipParam::Position => position
            .map(|p| ParamValue::Vec2([p[0] as f32, p[1] as f32]))
            .ok_or_else(|| {
                Rejection::new("param 'position' needs the 'position' argument as [x, y]")
            }),
        WireClipParam::Scale
        | WireClipParam::Rotation
        | WireClipParam::Opacity
        | WireClipParam::Volume => value.map(|v| ParamValue::Scalar(v as f32)).ok_or_else(|| {
            Rejection::new(
                format!("param '{param:?}' needs the 'value' argument (a number)",).to_lowercase(),
            )
        }),
    }
}

/// A keyframe's timeline position in seconds, pre-checked against the
/// clip's extent so the model gets a message naming where the clip sits.
fn keyframe_position(
    project: &Project,
    clip: &Clip,
    seconds: f64,
) -> Result<RationalTime, Rejection> {
    let at = timeline_time(project, seconds, "at")?;
    let tl = clip.timeline;
    if at.value < tl.start.value || at.value >= tl.end_tick() {
        let rate = timeline_rate(project);
        return Err(Rejection::new(format!(
            "keyframe position {seconds:.3}s is outside clip {} ({:.3}s to {:.3}s)",
            clip.id.raw(),
            ticks_to_seconds(tl.start.value, rate),
            ticks_to_seconds(tl.end_tick(), rate),
        )));
    }
    Ok(at)
}

fn ticks_to_seconds(ticks: i64, rate: Rational) -> f64 {
    ticks as f64 * rate.seconds_per_unit()
}

fn seconds_to_ticks(seconds: f64, rate: Rational, what: &str) -> Result<i64, Rejection> {
    if !seconds.is_finite() {
        return Err(Rejection::new(format!("{what} must be a finite number")));
    }
    let ticks = seconds * f64::from(rate.num) / f64::from(rate.den);
    if !(-(2f64.powi(53))..=2f64.powi(53)).contains(&ticks) {
        return Err(Rejection::new(format!(
            "{what} of {seconds}s is out of range"
        )));
    }
    Ok(ticks.round() as i64)
}

fn require_non_negative(seconds: f64, what: &str) -> Result<(), Rejection> {
    if seconds < 0.0 {
        return Err(Rejection::new(format!(
            "{what} must not be negative (got {seconds}s)"
        )));
    }
    Ok(())
}

/// A non-negative timeline position, frame-snapped to the project rate.
fn timeline_time(project: &Project, seconds: f64, what: &str) -> Result<RationalTime, Rejection> {
    let ticks = seconds_to_ticks(seconds, timeline_rate(project), what)?;
    Ok(RationalTime::new(ticks, timeline_rate(project)))
}

/// A signed timeline delta, frame-snapped to the project rate.
fn timeline_time_signed(
    project: &Project,
    seconds: f64,
    what: &str,
) -> Result<RationalTime, Rejection> {
    let ticks = seconds_to_ticks(seconds, timeline_rate(project), what)?;
    Ok(RationalTime::new(ticks, timeline_rate(project)))
}

/// A timeline range from `start`/`duration` seconds; duration must survive
/// frame snapping with at least one frame.
fn timeline_range(project: &Project, start: f64, duration: f64) -> Result<TimeRange, Rejection> {
    require_non_negative(start, "start")?;
    if duration <= 0.0 {
        return Err(Rejection::new(format!(
            "duration must be positive (got {duration}s)"
        )));
    }
    let rate = timeline_rate(project);
    let start_ticks = seconds_to_ticks(start, rate, "start")?;
    let duration_ticks = seconds_to_ticks(duration, rate, "duration")?.max(1);
    Ok(TimeRange::at_rate(start_ticks, duration_ticks, rate))
}

/// Source range (at the media's native rate) + timeline position for
/// placing media. Pre-checks bounds so the model gets a message naming the
/// media's actual extent.
fn media_placement(
    project: &Project,
    media: &cutlass_models::MediaSource,
    source_start: f64,
    source_duration: f64,
    timeline_seconds: f64,
    timeline_what: &str,
) -> Result<(TimeRange, RationalTime), Rejection> {
    require_non_negative(source_start, "source_start")?;
    if source_duration <= 0.0 {
        return Err(Rejection::new(format!(
            "source_duration must be positive (got {source_duration}s)"
        )));
    }
    require_non_negative(timeline_seconds, timeline_what)?;

    let rate = media.frame_rate;
    let start_ticks = seconds_to_ticks(source_start, rate, "source_start")?;
    let duration_ticks = seconds_to_ticks(source_duration, rate, "source_duration")?.max(1);
    if start_ticks + duration_ticks > media.duration.value {
        return Err(Rejection::new(format!(
            "source range {:.3}s + {:.3}s exceeds media {} which is {:.3}s long",
            source_start,
            source_duration,
            media.id.raw(),
            ticks_to_seconds(media.duration.value, rate),
        )));
    }
    let source = TimeRange::at_rate(start_ticks, duration_ticks, rate);
    let at = timeline_time(project, timeline_seconds, timeline_what)?;
    Ok((source, at))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire;
    use cutlass_models::MediaSource;

    const R24: Rational = Rational::FPS_24;

    /// 24 fps project: one video track with a 10 s media clip at 0 s, one
    /// text track with a title from 2 s to 5 s, and a 60 s media source.
    fn fixture() -> (Project, u64, u64, u64, u64, u64) {
        let mut project = Project::new("fixture", R24);
        let media = project.add_media(MediaSource::new(
            "/tmp/agent-fixture.mp4",
            1920,
            1080,
            R24,
            60 * 24,
            true,
        ));
        let video = project.add_track(TrackKind::Video, "V1");
        let text = project.add_track(TrackKind::Text, "Titles");
        let clip = project
            .add_clip(
                video,
                media,
                TimeRange::at_rate(0, 240, R24),
                RationalTime::new(0, R24),
            )
            .unwrap();
        let title = project
            .add_generated(
                text,
                Generator::text("INTRO"),
                TimeRange::at_rate(48, 72, R24),
            )
            .unwrap();
        (
            project,
            media.raw(),
            video.raw(),
            text.raw(),
            clip.raw(),
            title.raw(),
        )
    }

    fn lower(project: &Project, cmd: WireCommand) -> EditCommand {
        match validate(&cmd, project).expect("command should validate") {
            Command::Edit(edit) => edit,
            other => panic!("expected edit, got {other:?}"),
        }
    }

    fn reject(project: &Project, cmd: WireCommand) -> String {
        validate(&cmd, project)
            .expect_err("command should be rejected")
            .message
    }

    #[test]
    fn trim_clip_converts_seconds_to_frame_ticks() {
        let (project, _, _, _, clip, _) = fixture();
        let edit = lower(
            &project,
            WireCommand::TrimClip(wire::TrimClip {
                clip,
                start: 4.0,
                duration: 6.0,
            }),
        );
        assert_eq!(
            edit,
            EditCommand::TrimClip {
                clip: ClipId::from_raw(clip),
                timeline: TimeRange::at_rate(96, 144, R24),
            }
        );
    }

    #[test]
    fn fractional_seconds_snap_to_nearest_frame() {
        let (project, _, _, _, clip, _) = fixture();
        // 1.02 s at 24 fps = 24.48 frames -> 24; duration 0.01 s -> 0.24
        // frames -> clamps to 1 frame.
        let edit = lower(
            &project,
            WireCommand::TrimClip(wire::TrimClip {
                clip,
                start: 1.02,
                duration: 0.01,
            }),
        );
        assert_eq!(
            edit,
            EditCommand::TrimClip {
                clip: ClipId::from_raw(clip),
                timeline: TimeRange::at_rate(24, 1, R24),
            }
        );
    }

    #[test]
    fn add_clip_uses_media_rate_for_source_and_timeline_rate_for_start() {
        let mut project = Project::new("mixed-rates", R24);
        let media = project.add_media(MediaSource::new(
            "/tmp/30fps.mp4",
            1920,
            1080,
            Rational::FPS_30,
            300,
            true,
        ));
        let track = project.add_track(TrackKind::Video, "V1");

        let edit = lower(
            &project,
            WireCommand::AddClip(wire::AddClip {
                track: track.raw(),
                media: media.raw(),
                source_start: 1.0,
                source_duration: 4.0,
                start: 2.0,
            }),
        );
        assert_eq!(
            edit,
            EditCommand::AddClip {
                track,
                media,
                source: TimeRange::at_rate(30, 120, Rational::FPS_30),
                start: RationalTime::new(48, R24),
            }
        );
    }

    #[test]
    fn add_clip_rejects_out_of_bounds_source_with_media_extent() {
        let (project, media, video, _, _, _) = fixture();
        let msg = reject(
            &project,
            WireCommand::AddClip(wire::AddClip {
                track: video,
                media,
                source_start: 55.0,
                source_duration: 10.0,
                start: 0.0,
            }),
        );
        assert!(msg.contains("exceeds media"), "{msg}");
        assert!(msg.contains("60.000s"), "{msg}");
    }

    #[test]
    fn unknown_ids_list_existing_ones() {
        let (project, _, video, text, clip, title) = fixture();

        let msg = reject(
            &project,
            WireCommand::RemoveClip(wire::RemoveClip { clip: 999 }),
        );
        assert!(msg.contains("clip 999 does not exist"), "{msg}");
        assert!(msg.contains(&clip.to_string()), "{msg}");
        assert!(msg.contains(&title.to_string()), "{msg}");

        let msg = reject(
            &project,
            WireCommand::RemoveTrack(wire::RemoveTrack { track: 999 }),
        );
        assert!(msg.contains("track 999 does not exist"), "{msg}");
        assert!(msg.contains(&video.to_string()), "{msg}");
        assert!(msg.contains(&text.to_string()), "{msg}");

        let msg = reject(
            &project,
            WireCommand::AddClip(wire::AddClip {
                track: video,
                media: 999,
                source_start: 0.0,
                source_duration: 1.0,
                start: 0.0,
            }),
        );
        assert!(msg.contains("media 999 does not exist"), "{msg}");
    }

    #[test]
    fn generators_must_match_lane_kind() {
        let (project, _, video, text, _, _) = fixture();

        let msg = reject(
            &project,
            WireCommand::AddGenerated(wire::AddGenerated {
                track: video,
                generator: WireGenerator::Text {
                    content: "hi".into(),
                },
                start: 0.0,
                duration: 2.0,
            }),
        );
        assert!(msg.contains("needs a text track"), "{msg}");

        let msg = reject(
            &project,
            WireCommand::AddGenerated(wire::AddGenerated {
                track: text,
                generator: WireGenerator::Solid {
                    rgba: [0, 0, 0, 255],
                },
                start: 0.0,
                duration: 2.0,
            }),
        );
        assert!(msg.contains("sticker (overlay) track"), "{msg}");
    }

    #[test]
    fn media_clips_cannot_land_on_generator_lanes() {
        let (project, media, _, text, _, _) = fixture();
        let msg = reject(
            &project,
            WireCommand::AddClip(wire::AddClip {
                track: text,
                media,
                source_start: 0.0,
                source_duration: 1.0,
                start: 0.0,
            }),
        );
        assert!(
            msg.contains("media clips need a video or audio track"),
            "{msg}"
        );
    }

    #[test]
    fn set_generator_preserves_text_style_and_rejects_media_clips() {
        let (mut project, _, _, _, clip, title) = fixture();

        // Give the title a non-default style, then replace its content.
        let styled = Generator::Text {
            content: "INTRO".into(),
            style: cutlass_models::TextStyle {
                size: 120.0,
                ..Default::default()
            },
        };
        project
            .set_generator(ClipId::from_raw(title), styled.clone())
            .unwrap();

        let edit = lower(
            &project,
            WireCommand::SetGenerator(wire::SetGenerator {
                clip: title,
                generator: WireGenerator::Text {
                    content: "OUTRO".into(),
                },
            }),
        );
        match edit {
            EditCommand::SetGenerator {
                generator: Generator::Text { content, style },
                ..
            } => {
                assert_eq!(content, "OUTRO");
                assert_eq!(style.size, 120.0, "existing style must be preserved");
            }
            other => panic!("unexpected lowering: {other:?}"),
        }

        let msg = reject(
            &project,
            WireCommand::SetGenerator(wire::SetGenerator {
                clip,
                generator: WireGenerator::Text {
                    content: "nope".into(),
                },
            }),
        );
        assert!(msg.contains("is a media clip"), "{msg}");
    }

    #[test]
    fn transform_merges_with_current_values() {
        let (mut project, _, _, _, clip, _) = fixture();
        project
            .set_transform(
                ClipId::from_raw(clip),
                ClipTransform {
                    position: [0.25, 0.0],
                    scale: 0.5,
                    rotation: 10.0,
                    opacity: 0.8,
                    ..ClipTransform::IDENTITY
                },
                None,
            )
            .unwrap();

        let edit = lower(
            &project,
            WireCommand::SetClipTransform(wire::SetClipTransform {
                clip,
                position_x: None,
                position_y: Some(-0.1),
                anchor_x: None,
                anchor_y: None,
                scale: None,
                rotation: None,
                opacity: Some(1.0),
            }),
        );
        assert_eq!(
            edit,
            EditCommand::SetClipTransform {
                clip: ClipId::from_raw(clip),
                transform: ClipTransform {
                    position: [0.25, -0.1],
                    scale: 0.5,
                    rotation: 10.0,
                    opacity: 1.0,
                    ..ClipTransform::IDENTITY
                },
                at: None,
            }
        );

        let msg = reject(
            &project,
            WireCommand::SetClipTransform(wire::SetClipTransform {
                clip,
                position_x: None,
                position_y: None,
                anchor_x: None,
                anchor_y: None,
                scale: Some(0.0),
                rotation: None,
                opacity: None,
            }),
        );
        assert!(msg.contains("invalid transform"), "{msg}");
    }

    #[test]
    fn clip_crop_merges_edges_and_rejects_empty_frames() {
        let (mut project, _, _, _, clip, _) = fixture();

        // Fresh clip: edges lower straight into the kept-region rect.
        let edit = lower(
            &project,
            WireCommand::SetClipCrop(wire::SetClipCrop {
                clip,
                left: Some(0.25),
                top: None,
                right: Some(0.25),
                bottom: None,
                flip_h: Some(true),
                flip_v: None,
            }),
        );
        assert_eq!(
            edit,
            EditCommand::SetClipCrop {
                clip: ClipId::from_raw(clip),
                crop: CropRect {
                    x: 0.25,
                    y: 0.0,
                    w: 0.5,
                    h: 1.0
                },
                flip_h: true,
                flip_v: false,
            }
        );

        // Omitted fields keep the stored framing: crop the top, keep the
        // earlier horizontal window and flip.
        project
            .set_clip_crop(
                ClipId::from_raw(clip),
                CropRect {
                    x: 0.25,
                    y: 0.0,
                    w: 0.5,
                    h: 1.0,
                },
                true,
                false,
            )
            .unwrap();
        let edit = lower(
            &project,
            WireCommand::SetClipCrop(wire::SetClipCrop {
                clip,
                left: None,
                top: Some(0.1),
                right: None,
                bottom: None,
                flip_h: None,
                flip_v: None,
            }),
        );
        match edit {
            EditCommand::SetClipCrop { crop, flip_h, .. } => {
                assert_eq!(crop.x, 0.25);
                assert_eq!(crop.w, 0.5);
                assert_eq!(crop.y, 0.1);
                assert!((crop.h - 0.9).abs() < 1e-6);
                assert!(flip_h, "stored flip must be kept");
            }
            other => panic!("unexpected lowering: {other:?}"),
        }

        // Edges that eat the whole frame are rejected with a hint.
        let msg = reject(
            &project,
            WireCommand::SetClipCrop(wire::SetClipCrop {
                clip,
                left: Some(0.6),
                top: None,
                right: Some(0.6),
                bottom: None,
                flip_h: None,
                flip_v: None,
            }),
        );
        assert!(msg.contains("leaves no visible frame"), "{msg}");

        // Out-of-range fractions are rejected by name.
        let msg = reject(
            &project,
            WireCommand::SetClipCrop(wire::SetClipCrop {
                clip,
                left: None,
                top: None,
                right: None,
                bottom: Some(1.5),
                flip_h: None,
                flip_v: None,
            }),
        );
        assert!(msg.contains("bottom must be a fraction"), "{msg}");
    }

    #[test]
    fn clip_crop_rejects_audio_lane_clips() {
        let (mut project, media, _, _, _, _) = fixture();
        let lane = project.add_track(TrackKind::Audio, "A1");
        let audio_clip = project
            .add_clip(
                lane,
                cutlass_models::MediaId::from_raw(media),
                TimeRange::at_rate(0, 240, R24),
                RationalTime::new(0, R24),
            )
            .unwrap();
        let msg = reject(
            &project,
            WireCommand::SetClipCrop(wire::SetClipCrop {
                clip: audio_clip.raw(),
                left: Some(0.2),
                top: None,
                right: None,
                bottom: None,
                flip_h: None,
                flip_v: None,
            }),
        );
        assert!(msg.contains("no frame to crop"), "{msg}");
    }

    #[test]
    fn clip_speed_lowers_to_exact_rationals() {
        let (project, _, _, _, clip, title) = fixture();

        let edit = lower(
            &project,
            WireCommand::SetClipSpeed(wire::SetClipSpeed {
                clip,
                speed: Some(2.0),
                reversed: None,
            }),
        );
        assert_eq!(
            edit,
            EditCommand::SetClipSpeed {
                clip: ClipId::from_raw(clip),
                speed: Rational::new(2, 1),
                reversed: false,
            }
        );

        // Hundredth snapping: 0.5 → 1/2, 0.75 → 3/4, 0.333 → 33/100.
        assert_eq!(rational_speed(0.5).unwrap(), Rational::new(1, 2));
        assert_eq!(rational_speed(0.75).unwrap(), Rational::new(3, 4));
        assert_eq!(rational_speed(0.333).unwrap(), Rational::new(33, 100));

        // Omitted fields keep the clip's current retiming (reverse-only).
        let edit = lower(
            &project,
            WireCommand::SetClipSpeed(wire::SetClipSpeed {
                clip,
                speed: None,
                reversed: Some(true),
            }),
        );
        assert_eq!(
            edit,
            EditCommand::SetClipSpeed {
                clip: ClipId::from_raw(clip),
                speed: Rational::new(1, 1),
                reversed: true,
            }
        );

        // Out-of-range speeds and generated clips are rejected with names.
        let msg = reject(
            &project,
            WireCommand::SetClipSpeed(wire::SetClipSpeed {
                clip,
                speed: Some(0.0),
                reversed: None,
            }),
        );
        assert!(msg.contains("between 0.05 and 100"), "{msg}");
        let msg = reject(
            &project,
            WireCommand::SetClipSpeed(wire::SetClipSpeed {
                clip: title,
                speed: Some(2.0),
                reversed: None,
            }),
        );
        assert!(msg.contains("generated clip"), "{msg}");
    }

    #[test]
    fn clip_pitch_lowers_and_rejects_generated() {
        let (project, _, _, _, clip, title) = fixture();

        let edit = lower(
            &project,
            WireCommand::SetClipPitch(wire::SetClipPitch {
                clip,
                preserve_pitch: false,
            }),
        );
        assert_eq!(
            edit,
            EditCommand::SetClipPitch {
                clip: ClipId::from_raw(clip),
                preserve_pitch: false,
            }
        );

        // Generated clips have no footage to stretch, so pitch is meaningless.
        let msg = reject(
            &project,
            WireCommand::SetClipPitch(wire::SetClipPitch {
                clip: title,
                preserve_pitch: true,
            }),
        );
        assert!(msg.contains("generated clip"), "{msg}");
    }

    #[test]
    fn look_commands_lower_and_guard() {
        let (project, _, _, _, clip, title) = fixture();

        let edit = lower(
            &project,
            WireCommand::SetClipFilter(wire::SetClipFilter {
                clip,
                filter: Some(wire::WireFilter {
                    id: "vivid".to_string(),
                    intensity: Some(0.9),
                }),
            }),
        );
        match edit {
            EditCommand::SetClipFilter { filter, .. } => {
                let filter = filter.expect("filter set");
                assert_eq!(filter.id, "vivid");
                assert!((filter.intensity - 0.9).abs() < 1e-6);
            }
            other => panic!("expected SetClipFilter, got {other:?}"),
        }

        let edit = lower(
            &project,
            WireCommand::SetClipAnimation(wire::SetClipAnimation {
                clip,
                slot: wire::WireAnimationSlot::In,
                animation: Some("fade_in".to_string()),
            }),
        );
        assert_eq!(
            edit,
            EditCommand::SetClipAnimation {
                clip: ClipId::from_raw(clip),
                slot: AnimationSlot::In,
                animation: Some(AnimationRef::new("fade_in")),
            }
        );

        let msg = reject(
            &project,
            WireCommand::SetClipAnimation(wire::SetClipAnimation {
                clip,
                slot: wire::WireAnimationSlot::Out,
                animation: Some("fade_in".to_string()),
            }),
        );
        assert!(msg.contains("does not fit"), "{msg}");

        let msg = reject(
            &project,
            WireCommand::SetClipMask(wire::SetClipMask {
                clip: title,
                mask: Some(wire::WireMask {
                    kind: wire::WireMaskKind::Circle,
                    feather: None,
                    invert: None,
                }),
            }),
        );
        assert!(msg.contains("generated clip"), "{msg}");
    }

    #[test]
    fn set_denoise_lowers_steers_and_rejects_generated() {
        let (mut project, media, _, _, video_clip, title) = fixture();
        // An audio lane carrying the linked companion of the video clip.
        let lane = project.add_track(TrackKind::Audio, "A1");
        let audio_clip = project
            .add_clip(
                lane,
                cutlass_models::MediaId::from_raw(media),
                TimeRange::at_rate(0, 240, R24),
                RationalTime::new(0, R24),
            )
            .unwrap();
        let link = cutlass_models::LinkId::next();
        for id in [ClipId::from_raw(video_clip), audio_clip] {
            project.timeline_mut().clip_mut(id).unwrap().link = Some(link);
        }

        // An audio-lane target lowers straight through.
        let edit = lower(
            &project,
            WireCommand::SetDenoise(wire::SetDenoise {
                clip: audio_clip.raw(),
                denoise: true,
            }),
        );
        assert_eq!(
            edit,
            EditCommand::SetClipDenoise {
                clip: audio_clip,
                denoise: true,
            }
        );

        // A video-lane target is steered to its linked audio companion.
        let msg = reject(
            &project,
            WireCommand::SetDenoise(wire::SetDenoise {
                clip: video_clip,
                denoise: true,
            }),
        );
        assert!(
            msg.contains(&format!("linked clip {}", audio_clip.raw())),
            "{msg}"
        );

        // A generated clip has no footage to clean.
        let msg = reject(
            &project,
            WireCommand::SetDenoise(wire::SetDenoise {
                clip: title,
                denoise: true,
            }),
        );
        assert!(msg.contains("generated clip"), "{msg}");
    }

    #[test]
    fn clip_audio_lowers_volume_and_fades() {
        let (mut project, media, _, _, video_clip, title) = fixture();
        // An audio lane carrying the linked companion of the video clip.
        let lane = project.add_track(TrackKind::Audio, "A1");
        let audio_clip = project
            .add_clip(
                lane,
                cutlass_models::MediaId::from_raw(media),
                TimeRange::at_rate(0, 240, R24),
                RationalTime::new(0, R24),
            )
            .unwrap();
        let link = cutlass_models::LinkId::next();
        for id in [ClipId::from_raw(video_clip), audio_clip] {
            project.timeline_mut().clip_mut(id).unwrap().link = Some(link);
        }

        // Volume + fades lower to ticks at the timeline rate (1s = 24).
        let edit = lower(
            &project,
            WireCommand::SetClipAudio(wire::SetClipAudio {
                clip: audio_clip.raw(),
                volume: Some(0.5),
                fade_in: Some(1.0),
                fade_out: Some(0.5),
            }),
        );
        assert_eq!(
            edit,
            EditCommand::SetClipAudio {
                clip: audio_clip,
                volume: Some(0.5),
                fade_in: RationalTime::new(24, R24),
                fade_out: RationalTime::new(12, R24),
            }
        );

        // Omitted fields keep the clip's current mix.
        let edit = lower(
            &project,
            WireCommand::SetClipAudio(wire::SetClipAudio {
                clip: audio_clip.raw(),
                volume: Some(0.0),
                fade_in: None,
                fade_out: None,
            }),
        );
        assert_eq!(
            edit,
            EditCommand::SetClipAudio {
                clip: audio_clip,
                volume: Some(0.0),
                fade_in: RationalTime::new(0, R24),
                fade_out: RationalTime::new(0, R24),
            }
        );

        // Omitting volume lowers to `None`, so the clip's gain (a flat level
        // or an M8 envelope) is preserved — only the fades change.
        let edit = lower(
            &project,
            WireCommand::SetClipAudio(wire::SetClipAudio {
                clip: audio_clip.raw(),
                volume: None,
                fade_in: Some(0.5),
                fade_out: None,
            }),
        );
        assert_eq!(
            edit,
            EditCommand::SetClipAudio {
                clip: audio_clip,
                volume: None,
                fade_in: RationalTime::new(12, R24),
                fade_out: RationalTime::new(0, R24),
            }
        );

        // A video-lane target is steered to its linked audio companion.
        let msg = reject(
            &project,
            WireCommand::SetClipAudio(wire::SetClipAudio {
                clip: video_clip,
                volume: Some(0.5),
                fade_in: None,
                fade_out: None,
            }),
        );
        assert!(
            msg.contains(&format!("linked clip {}", audio_clip.raw())),
            "{msg}"
        );

        // Out-of-range volume, over-long fades, generated clips: rejected.
        let msg = reject(
            &project,
            WireCommand::SetClipAudio(wire::SetClipAudio {
                clip: audio_clip.raw(),
                volume: Some(11.0),
                fade_in: None,
                fade_out: None,
            }),
        );
        assert!(msg.contains("between 0 (mute) and 10"), "{msg}");
        let msg = reject(
            &project,
            WireCommand::SetClipAudio(wire::SetClipAudio {
                clip: audio_clip.raw(),
                volume: None,
                fade_in: Some(60.0),
                fade_out: None,
            }),
        );
        assert!(msg.contains("longer than clip"), "{msg}");
        let msg = reject(
            &project,
            WireCommand::SetClipAudio(wire::SetClipAudio {
                clip: title,
                volume: Some(0.5),
                fade_in: None,
                fade_out: None,
            }),
        );
        assert!(msg.contains("generated clip"), "{msg}");
    }

    #[test]
    fn audio_edits_on_a_video_clip_with_sound_target_the_clip_itself() {
        // CapCut keeps a video's audio on the clip — a drop lands a single
        // clip, not a linked audio companion — so volume/fades and denoise on
        // the video clip adjust it directly, with no steering.
        let (project, _, _, _, video_clip, _) = fixture();

        let edit = lower(
            &project,
            WireCommand::SetClipAudio(wire::SetClipAudio {
                clip: video_clip,
                volume: Some(0.5),
                fade_in: None,
                fade_out: None,
            }),
        );
        assert_eq!(
            edit,
            EditCommand::SetClipAudio {
                clip: ClipId::from_raw(video_clip),
                volume: Some(0.5),
                fade_in: RationalTime::new(0, R24),
                fade_out: RationalTime::new(0, R24),
            }
        );

        let edit = lower(
            &project,
            WireCommand::SetDenoise(wire::SetDenoise {
                clip: video_clip,
                denoise: true,
            }),
        );
        assert_eq!(
            edit,
            EditCommand::SetClipDenoise {
                clip: ClipId::from_raw(video_clip),
                denoise: true,
            }
        );
    }

    #[test]
    fn split_outside_clip_names_its_extent() {
        let (project, _, _, _, clip, _) = fixture();
        let msg = reject(
            &project,
            WireCommand::SplitClip(wire::SplitClip { clip, at: 10.0 }),
        );
        assert!(msg.contains("not strictly inside clip"), "{msg}");
        assert!(msg.contains("0.000s"), "{msg}");
        assert!(msg.contains("10.000s"), "{msg}");
    }

    #[test]
    fn shift_rejects_sub_frame_delta_and_link_needs_two() {
        let (project, _, video, _, clip, _) = fixture();
        let msg = reject(
            &project,
            WireCommand::ShiftClips(wire::ShiftClips {
                track: video,
                from: 0.0,
                delta: 0.001,
            }),
        );
        assert!(msg.contains("rounds to zero frames"), "{msg}");

        let msg = reject(
            &project,
            WireCommand::LinkClips(wire::LinkClips { clips: vec![clip] }),
        );
        assert!(msg.contains("at least two"), "{msg}");
    }

    #[test]
    fn marker_commands_lower_to_engine() {
        let (project, _, _, _, _, _) = fixture();
        let edit = lower(
            &project,
            WireCommand::AddMarker(wire::AddMarker {
                at: 2.0,
                name: Some("intro".into()),
                color: None,
            }),
        );
        assert_eq!(
            edit,
            EditCommand::AddMarker {
                at: RationalTime::new(48, R24),
                name: "intro".into(),
                color: None,
            }
        );

        // Negative positions are rejected before reaching the engine.
        let msg = reject(
            &project,
            WireCommand::AddMarker(wire::AddMarker {
                at: -1.0,
                name: None,
                color: None,
            }),
        );
        assert!(msg.contains("must not be negative"), "{msg}");

        let mut project = project;
        let id = project
            .timeline_mut()
            .add_marker(cutlass_models::Marker::new(
                RationalTime::new(48, R24),
                "mid",
                MarkerColor::Blue,
            ))
            .unwrap();
        let edit = lower(
            &project,
            WireCommand::SetMarker(wire::SetMarker {
                marker: id.raw(),
                at: Some(3.0),
                name: Some("outro".into()),
                color: Some(WireMarkerColor::Green),
            }),
        );
        assert_eq!(
            edit,
            EditCommand::SetMarker {
                marker: id,
                at: RationalTime::new(72, R24),
                name: "outro".into(),
                color: MarkerColor::Green,
            }
        );

        // Omitted fields keep the marker's current state.
        let edit = lower(
            &project,
            WireCommand::SetMarker(wire::SetMarker {
                marker: id.raw(),
                at: None,
                name: None,
                color: None,
            }),
        );
        assert_eq!(
            edit,
            EditCommand::SetMarker {
                marker: id,
                at: RationalTime::new(48, R24),
                name: "mid".into(),
                color: MarkerColor::Blue,
            }
        );

        let msg = reject(
            &project,
            WireCommand::RemoveMarker(wire::RemoveMarker { marker: 404 }),
        );
        assert!(msg.contains("marker 404 does not exist"), "{msg}");
        assert!(msg.contains(&id.raw().to_string()), "{msg}");
    }

    #[test]
    fn set_canvas_lowers_and_keeps_omitted_fields() {
        let (mut project, _, _, _, _, _) = fixture();
        let edit = lower(
            &project,
            WireCommand::SetCanvas(wire::SetCanvas {
                aspect: Some(wire::WireCanvasAspect::Tall9x16),
                background: Some([20, 20, 28]),
            }),
        );
        assert_eq!(
            edit,
            EditCommand::SetCanvas {
                aspect: CanvasAspect::Tall9x16,
                background: [20, 20, 28],
            }
        );

        // Omitted fields keep the project's current canvas settings.
        project
            .timeline_mut()
            .set_canvas(cutlass_models::CanvasSettings {
                aspect: CanvasAspect::Square1x1,
                background: [255, 0, 0],
            });
        let edit = lower(
            &project,
            WireCommand::SetCanvas(wire::SetCanvas {
                aspect: None,
                background: Some([0, 0, 0]),
            }),
        );
        assert_eq!(
            edit,
            EditCommand::SetCanvas {
                aspect: CanvasAspect::Square1x1,
                background: [0, 0, 0],
            }
        );
        let edit = lower(
            &project,
            WireCommand::SetCanvas(wire::SetCanvas {
                aspect: Some(wire::WireCanvasAspect::Auto),
                background: None,
            }),
        );
        assert_eq!(
            edit,
            EditCommand::SetCanvas {
                aspect: CanvasAspect::Auto,
                background: [255, 0, 0],
            }
        );
    }

    #[test]
    fn non_finite_and_negative_times_are_rejected() {
        let (project, _, _, _, clip, _) = fixture();
        let msg = reject(
            &project,
            WireCommand::TrimClip(wire::TrimClip {
                clip,
                start: f64::NAN,
                duration: 1.0,
            }),
        );
        assert!(msg.contains("finite"), "{msg}");

        let msg = reject(
            &project,
            WireCommand::TrimClip(wire::TrimClip {
                clip,
                start: -1.0,
                duration: 1.0,
            }),
        );
        assert!(msg.contains("must not be negative"), "{msg}");
    }
}
