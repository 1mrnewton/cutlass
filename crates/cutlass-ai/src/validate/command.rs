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
        WireCommand::ExtractAudio(args) => extract_audio(project, args)?,
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
            // Lowering uses `at: None`, which flattens every transform param
            // to a constant — refuse when any param is already keyframed.
            check_set_clip_transform_preserves_keyframes(clip, args.clip)?;
            // Omitted properties keep their current constant value.
            if let Some(x) = args.position_x {
                check_position_component(x)?;
            }
            if let Some(y) = args.position_y {
                check_position_component(y)?;
            }
            if let Some(x) = args.anchor_x {
                check_anchor_component(x)?;
            }
            if let Some(y) = args.anchor_y {
                check_anchor_component(y)?;
            }
            if let Some(scale) = args.scale {
                check_wire_scale(scale)?;
            }
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
                scale: args.scale.map_or(current.scale, |v| v.to_scale2()),
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
            // Lowering uses `at: None`, which flattens crop to a constant —
            // refuse when crop is already keyframed (flips are separate bools).
            check_set_clip_crop_preserves_keyframes(clip, args.clip)?;
            // Omitted edges keep the clip's current framing (constant crop).
            let current = clip.crop.sample(0);
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
                at: None,
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
            let (index, slot, p) = effect_param_slot(clip, args.index, &args.param, args.clip)?;
            match p.kind {
                cutlass_models::EffectParamKind::Scalar => {
                    let v = args.value.ok_or_else(|| {
                        Rejection::new(format!(
                            "effect param '{}' is a scalar; pass 'value' (a number)",
                            args.param
                        ))
                    })?;
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
                cutlass_models::EffectParamKind::Color => {
                    let rgba = args.rgba.ok_or_else(|| {
                        Rejection::new(format!(
                            "effect param '{}' is a color; pass 'rgba' as \
                             [red, green, blue, alpha]",
                            args.param
                        ))
                    })?;
                    EditCommand::SetParamConstant {
                        clip: clip.id,
                        param: ClipParam::Effect {
                            effect: index as u32,
                            param: slot as u32,
                        },
                        value: ParamValue::Color(rgba),
                    }
                }
                cutlass_models::EffectParamKind::Vec2 => {
                    let position = args.position.ok_or_else(|| {
                        Rejection::new(format!(
                            "effect param '{}' is a vec2; pass 'position' as [x, y]",
                            args.param
                        ))
                    })?;
                    let v = [position[0] as f32, position[1] as f32];
                    if !v[0].is_finite()
                        || !v[1].is_finite()
                        || v[0] < p.min
                        || v[0] > p.max
                        || v[1] < p.min
                        || v[1] > p.max
                    {
                        return Err(Rejection::new(format!(
                            "{} components must be between {} and {} (got [{}, {}])",
                            args.param, p.min, p.max, v[0], v[1]
                        )));
                    }
                    EditCommand::SetParamConstant {
                        clip: clip.id,
                        param: ClipParam::Effect {
                            effect: index as u32,
                            param: slot as u32,
                        },
                        value: ParamValue::Vec2(v),
                    }
                }
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
            reject_speed_keyframe_param(&args.param)?;
            let at = keyframe_position(project, clip, args.at)?;
            check_motion_param_args(&args.param, args.value, args.position)?;
            let value = param_value(
                clip,
                args.clip,
                &args.param,
                args.value,
                args.position,
                args.rgba,
                args.rect,
            )?;
            let tangents = spatial_tangents(args.tangent_out, args.tangent_in, &args.param)?;
            EditCommand::SetParamKeyframe {
                clip: clip.id,
                param: clip_param(&args.param, clip, args.clip)?,
                at,
                value,
                easing: easing(args.easing)?,
                tangents,
            }
        }
        WireCommand::RemoveParamKeyframe(args) => {
            let clip = clip_ref(project, args.clip)?;
            reject_speed_keyframe_param(&args.param)?;
            let at = keyframe_position(project, clip, args.at)?;
            EditCommand::RemoveParamKeyframe {
                clip: clip.id,
                param: clip_param(&args.param, clip, args.clip)?,
                at,
            }
        }
        WireCommand::SetClipSpeed(args) => set_clip_speed(project, args)?,
        WireCommand::SetSpeedCurve(args) => set_speed_curve(project, args)?,
        WireCommand::SetClipPitch(args) => set_clip_pitch(project, args)?,
        WireCommand::SetDenoise(args) => set_denoise(project, args)?,
        WireCommand::SetClipMask(args) => set_clip_mask(project, args)?,
        WireCommand::SetClipChroma(args) => set_clip_chroma(project, args)?,
        WireCommand::SetClipStabilize(args) => set_clip_stabilize(project, args)?,
        WireCommand::SetClipFilter(args) => set_clip_filter(project, args)?,
        WireCommand::SetClipBlendMode(args) => set_clip_blend_mode(project, args)?,
        WireCommand::SetMotionBlur(args) => set_motion_blur(project, args)?,
        WireCommand::SetClipLayerStyles(args) => set_clip_layer_styles(project, args)?,
        WireCommand::SetClipAdjustments(args) => set_clip_adjustments(project, args)?,
        WireCommand::SetClipAnimation(args) => set_clip_animation(project, args)?,
        WireCommand::SetAudioRole(args) => set_audio_role(project, args)?,
        WireCommand::SetClipAudio(args) => set_clip_audio(project, args)?,
        WireCommand::SetParamConstant(args) => {
            let clip = clip_ref(project, args.clip)?;
            reject_speed_keyframe_param(&args.param)?;
            check_motion_param_args(&args.param, args.value, args.position)?;
            let value = param_value(
                clip,
                args.clip,
                &args.param,
                args.value,
                args.position,
                args.rgba,
                args.rect,
            )?;
            EditCommand::SetParamConstant {
                clip: clip.id,
                param: clip_param(&args.param, clip, args.clip)?,
                value,
            }
        }
        WireCommand::ApplyEasingPreset(args) => {
            let clip = clip_ref(project, args.clip)?;
            let at = keyframe_position(project, clip, args.from_tick)?;
            let param = clip_param(&args.param, clip, args.clip)?;
            // Fail closed early with a clear message for unsupported kinds.
            if matches!(param, ClipParam::Crop | ClipParam::Speed)
                || matches!(
                    param,
                    ClipParam::Effect { .. } | ClipParam::Shape { .. } | ClipParam::Text { .. }
                )
            {
                return Err(Rejection::new(
                    "apply_easing_preset supports scalar/vec2 transform, volume, pan, look, \
                     and non-color style params only",
                ));
            }
            if matches!(
                param,
                ClipParam::Style {
                    param: StyleParam::ShadowColor
                        | StyleParam::GlowColor
                        | StyleParam::OutlineColor
                        | StyleParam::BackgroundColor,
                }
            ) {
                return Err(Rejection::new(
                    "apply_easing_preset does not support color parameters",
                ));
            }
            EditCommand::ApplyEasingPreset {
                clip: clip.id,
                param,
                at,
                preset: match args.preset {
                    WireEasingPreset::BounceOut => PiecewiseEasingPreset::BounceOut,
                    WireEasingPreset::ElasticOut => PiecewiseEasingPreset::ElasticOut,
                    WireEasingPreset::BackOut => PiecewiseEasingPreset::BackOut,
                },
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
