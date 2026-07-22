use super::*;

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
        WireCommand::ExtractAudio(args) => {
            let clip = clip_ref(project, args.clip)?;
            let timeline = project.timeline();
            let source_track_id = timeline.track_of(clip.id).ok_or_else(|| {
                Rejection::new(format!("clip {} is not on the timeline", args.clip))
            })?;
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
            EditCommand::ExtractAudio {
                clip: clip.id,
                to_track: Some(target.id),
            }
        }
        WireCommand::DuplicateClip(args) => {
            let clip = clip_ref(project, args.clip)?;
            let target = track_ref(project, args.to_track)?;
            if !target.kind.accepts_content(&clip.content) {
                return Err(Rejection::new(format!(
                    "clip {} cannot be duplicated onto track {} ({} lane); choose a track \
                     compatible with the source clip's content",
                    args.clip,
                    args.to_track,
                    kind_name(target.kind),
                )));
            }
            if target.locked {
                return Err(Rejection::new(format!(
                    "destination track {} is locked; unlock it or choose another compatible track",
                    args.to_track
                )));
            }
            let start = timeline_time(project, args.start, "start")?;
            require_non_negative(args.start, "start")?;
            let mut destination = clip.timeline;
            destination.start = start;
            if target
                .has_overlap(destination, None)
                .map_err(|error| Rejection::new(error.to_string()))?
            {
                let rate = timeline_rate(project);
                return Err(Rejection::new(format!(
                    "track {} has a clip overlapping duplicate of clip {} in the exact \
                     destination range {:.3}s to {:.3}s; choose another target track or a start \
                     with {:.3}s of free space — duplicate_clip does not ripple or search for space",
                    args.to_track,
                    args.clip,
                    ticks_to_seconds(destination.start.value, rate),
                    ticks_to_seconds(destination.end_tick(), rate),
                    ticks_to_seconds(destination.duration.value, rate),
                )));
            }
            EditCommand::DuplicateClip {
                clip: clip.id,
                to_track: target.id,
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
        WireCommand::MoveEffect(args) => {
            let clip = clip_ref(project, args.clip)?;
            let from_index = effect_index(clip, args.from_index, args.clip)?;
            let to_index = effect_index(clip, args.to_index, args.clip)?;
            if from_index == to_index {
                return Err(Rejection::new(format!(
                    "effect {from_index} on clip {} is already at index {to_index}; \
                     move_effect would make no change",
                    args.clip
                )));
            }
            EditCommand::MoveEffect {
                clip: clip.id,
                from_index,
                to_index,
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
            reject_non_transition_lane(project, clip, args.clip)?;
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
            let animation =
                match &args.animation {
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
                        let anim = AnimationRef {
                            id: id.clone(),
                            speed: args
                                .speed
                                .unwrap_or(cutlass_models::ANIMATION_PARAM_DEFAULT),
                            intensity: args
                                .intensity
                                .unwrap_or(cutlass_models::ANIMATION_PARAM_DEFAULT),
                            stagger: args
                                .stagger
                                .unwrap_or(cutlass_models::ANIMATION_PARAM_DEFAULT),
                        };
                        Some(anim.normalized_for(spec).map_err(|e| {
                            Rejection::new(format!("invalid animation params: {e}"))
                        })?)
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
        WireCommand::RemoveTrack(args) => {
            let track = track_ref(project, args.track)?;
            if track.main {
                return Err(Rejection::new(format!(
                    "track {} is the main track and cannot be removed; \
                     it is the permanent video lane every edit builds on",
                    args.track,
                )));
            }
            EditCommand::RemoveTrack { track: track.id }
        }
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
            if args.clips.len() > MAX_MULTI_CLIP_REFS {
                return Err(Rejection::new(format!(
                    "link_clips accepts at most {MAX_MULTI_CLIP_REFS} clip ids (got {})",
                    args.clips.len()
                )));
            }
            let mut clips = Vec::with_capacity(args.clips.len());
            for &raw in &args.clips {
                clips.push(clip_ref(project, raw)?.id);
            }
            EditCommand::LinkClips { clips }
        }
        WireCommand::UnlinkClips(args) => {
            if args.clips.is_empty() {
                return Err(Rejection::new(
                    "unlink_clips needs at least one clip id".to_string(),
                ));
            }
            if args.clips.len() > MAX_MULTI_CLIP_REFS {
                return Err(Rejection::new(format!(
                    "unlink_clips accepts at most {MAX_MULTI_CLIP_REFS} clip ids (got {})",
                    args.clips.len()
                )));
            }
            let mut seen = HashSet::with_capacity(args.clips.len());
            for &raw in &args.clips {
                if !seen.insert(raw) {
                    return Err(Rejection::new(format!(
                        "unlink_clips contains duplicate clip id {raw}; list each clip once"
                    )));
                }
            }

            let mut clips = Vec::with_capacity(args.clips.len());
            let mut any_linked = false;
            for &raw in &args.clips {
                let clip = clip_ref(project, raw)?;
                any_linked |= clip.link.is_some();
                clips.push(clip.id);
            }
            if !any_linked {
                return Err(Rejection::new(
                    "unlink_clips found no linked clips; all referenced clips are already unlinked",
                ));
            }
            EditCommand::UnlinkClips { clips }
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
