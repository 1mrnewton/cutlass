//! Tool-spec generation: the model-facing tool catalog over the wire types.

use schemars::JsonSchema;

use super::*;

/// One LLM tool: name, model-facing description, and a JSON Schema for its
/// arguments.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    /// Owned so host/MCP tool catalogs discovered at runtime can share the
    /// same provider wire as the static edit vocabulary.
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

fn spec<T: JsonSchema>(name: &'static str, description: &'static str) -> ToolSpec {
    // Subschemas are inlined (no `$defs` / `$ref`): small local models
    // routinely fail to follow reference indirection and then guess the
    // argument shape (e.g. passing a bare string where the tagged
    // `WireGenerator` object is required).
    let mut settings = schemars::generate::SchemaSettings::draft2020_12();
    settings.inline_subschemas = true;
    let mut parameters =
        serde_json::to_value(settings.into_generator().into_root_schema_for::<T>())
            .expect("tool argument schemas are plain data and always serialize");
    compact_schema_json(&mut parameters);
    ToolSpec {
        name: name.to_string(),
        description: description.to_string(),
        parameters,
    }
}

/// Strip schemars noise that costs tokens without helping the model:
/// `format`, `title`, `$schema`, and single-element `allOf` wrappers.
/// Numeric `minimum`/`maximum` are kept — they are load-bearing for `u8`
/// RGBA components and unsigned ids that serde rejects outside range.
fn compact_schema_json(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            map.remove("format");
            map.remove("title");
            map.remove("$schema");
            if map.get("description") == Some(&serde_json::json!("")) {
                map.remove("description");
            }
            // Collapse `allOf: [inner]` into the wrapper, keeping any
            // description / keywords already on the wrapper.
            if let Some(serde_json::Value::Array(all_of)) = map.get("allOf")
                && all_of.len() == 1
                && let serde_json::Value::Object(inner) = &all_of[0]
            {
                let inner = inner.clone();
                map.remove("allOf");
                for (k, v) in inner {
                    map.entry(k).or_insert(v);
                }
            }
            for child in map.values_mut() {
                compact_schema_json(child);
            }
        }
        serde_json::Value::Array(items) => {
            for child in items {
                compact_schema_json(child);
            }
        }
        _ => {}
    }
}

/// A corrective example appended to argument-decode rejections, for the
/// tools whose nested shapes models most often get wrong. The model reads
/// this and retries; without it, weak models tend to give up and ask the
/// user instead.
fn argument_hint(tool: &str) -> Option<&'static str> {
    match tool {
        "add_generated" | "set_generator" => Some(
            "'generator' must be a tagged object: {\"type\": \"text\", \"content\": \"Hello\"} \
             or {\"type\": \"solid\", \"rgba\": [0, 0, 0, 255]} \
             or {\"type\": \"shape\", \"shape\": \"ellipse\", \"rgba\": [255, 0, 0, 255]}",
        ),
        _ => None,
    }
}

/// The read-only tool: returns the current project summary + editor
/// context. Not a [`WireCommand`] — the agent loop answers it without
/// touching dispatch.
pub fn describe_project_spec() -> ToolSpec {
    ToolSpec {
        name: "describe_project".into(),
        description: "Get the current state of the project: tracks, clips with ids and \
                      times in seconds, the media pool, and the user's selection and \
                      playhead. Call this whenever you are unsure about ids or timing."
            .into(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {},
        }),
    }
}

macro_rules! tools {
    ($( $name:literal => $variant:ident ( $args:ty ), $desc:literal; )+) => {
        /// The full tool surface, in stable order.
        pub fn tool_specs() -> Vec<ToolSpec> {
            vec![ $( spec::<$args>($name, $desc) ),+ ]
        }

        impl WireCommand {
            /// The tool name this command arrives under.
            pub fn tool_name(&self) -> &'static str {
                match self {
                    $( WireCommand::$variant(_) => $name, )+
                }
            }

            /// Decode a provider tool call. Unknown names and malformed
            /// arguments come back as model-readable messages.
            pub fn from_tool_call(
                name: &str,
                arguments: serde_json::Value,
            ) -> Result<WireCommand, String> {
                match name {
                    $(
                        $name => serde_json::from_value::<$args>(arguments)
                            .map(WireCommand::$variant)
                            .map_err(|e| {
                                let hint = argument_hint(name)
                                    .map(|h| format!(" ({h})"))
                                    .unwrap_or_default();
                                format!("invalid arguments for {name}: {e}{hint}")
                            }),
                    )+
                    other => Err(format!(
                        "unknown tool '{other}'; available tools: {}",
                        [$($name),+].join(", ")
                    )),
                }
            }
        }
    };
}

tools! {
    "add_track" => AddTrack(AddTrack),
        "Add a timeline track (video, audio, text, sticker). CapCut zones: audio bottom, then main video, overlays, text top — index only orders within a zone.";
    "add_clip" => AddClip(AddClip),
        "Place a trimmed range of imported media on a video or audio track. Times in seconds.";
    "extract_audio" => ExtractAudio(ExtractAudio),
        "Detach a video clip's embedded sound onto an unlocked audio track (placement and audio/retime preserved). Track id required: add_track(kind=audio) first if needed.";
    "duplicate_clip" => DuplicateClip(DuplicateClip),
        "Deep-copy a clip to an explicit track and start (timeline seconds). Fresh unlinked id; does not ripple or find space.";
    "add_generated" => AddGenerated(AddGenerated),
        "Place a generated clip (text, solid, or shape) on a matching track. Times in seconds.";
    "set_generator" => SetGenerator(SetGenerator),
        "Replace generated clip content (text keeps styling). Not valid for media clips.";
    "set_clip_transform" => SetClipTransform(SetClipTransform),
        "Set canvas placement: position (canvas fractions from center), scale (number or [x,y]; 1.0=fit), rotation (deg CW), opacity. Omitted fields keep. Not for audio.";
    "set_clip_crop" => SetClipCrop(SetClipCrop),
        "Crop by edge-trim fractions (0 restores) and/or flip_h/flip_v. Kept region still aspect-fits (crop does not move the layer). Not for audio.";
    "add_effect" => AddEffect(AddEffect),
        "Append a visual effect (e.g. gaussian_blur/radius, vignette/amount). Chain order; not for audio.";
    "remove_effect" => RemoveEffect(RemoveEffect),
        "Remove an effect by chain index (0 = first). See describe_project.";
    "move_effect" => MoveEffect(MoveEffect),
        "Reorder effects; from_index/to_index address the pre-move chain. See describe_project.";
    "set_effect_param" => SetEffectParam(SetEffectParam),
        "Set an effect param: `value` scalars, `position` vec2, `rgba` colors. See describe_project for indices.";
    "add_transition" => AddTransition(AddTransition),
        "Add a transition at the cut to the next abutting clip: crossfade, dip_to_black/white, wipe_*, slide. Not for audio.";
    "remove_transition" => RemoveTransition(RemoveTransition),
        "Remove the transition at a clip's right cut.";
    "set_transition" => SetTransition(SetTransition),
        "Set transition duration in seconds (window centered on the cut).";
    "set_param_keyframe" => SetParamKeyframe(SetParamKeyframe),
        "Add/replace a keyframe at timeline seconds. See `param` schema for selectors and value args (`value`/`position`/`rgba`/`rect`). Optional tangent_out/tangent_in are canvas-fraction handles for position motion paths only.";
    "remove_param_keyframe" => RemoveParamKeyframe(RemoveParamKeyframe),
        "Remove a keyframe at timeline seconds (same `param` as set_param_keyframe). Last keyframe freezes the value.";
    "set_param_constant" => SetParamConstant(SetParamConstant),
        "Set an animatable property to a fixed value and clear its keyframes (same `param`/value args as set_param_keyframe).";
    "apply_easing_preset" => ApplyEasingPreset(ApplyEasingPreset),
        "Replace the outgoing KF segment at from_tick with bounce_out/elastic_out/back_out. Needs a following keyframe; scalar/vec2 only.";
    "set_clip_speed" => SetClipSpeed(SetClipSpeed),
        "Set media playback speed (2.0=double, 0.5=slow-mo) and/or reverse. Length re-derives; audio time-stretches. Not for generated clips.";
    "set_speed_curve" => SetSpeedCurve(SetSpeedCurve),
        "Apply/clear a speed ramp preset (ramp_up, ramp_down, montage, hero, bullet). Length re-derives from average speed. Not for generated clips.";
    "set_clip_pitch" => SetClipPitch(SetClipPitch),
        "preserve_pitch true keeps pitch while time-stretching; false lets pitch follow speed. Retimed media only.";
    "set_clip_audio" => SetClipAudio(SetClipAudio),
        "Set volume (0=mute, 1=unchanged, max 10) and/or fade-in/out seconds. Target video-with-sound or audio clips directly.";
    "set_denoise" => SetDenoise(SetDenoise),
        "Toggle speech-preserving denoise (steady hum/hiss/room tone). Media clips only; target video-with-sound directly.";
    "set_clip_mask" => SetClipMask(SetClipMask),
        "Set/clear a mask (linear/mirror/circle/rectangle/heart/star). feather 0..1; center/size = layer fractions; rotation deg CW; roundness 0..1 (rectangle). Animate via look params. null clears.";
    "set_clip_chroma" => SetClipChroma(SetClipChroma),
        "Set/clear chroma key on a media visual clip. null clears.";
    "set_clip_stabilize" => SetClipStabilize(SetClipStabilize),
        "Set/clear stabilize on a media video clip: recommended, smooth, max_smooth. null clears.";
    "set_clip_filter" => SetClipFilter(SetClipFilter),
        "Set/clear a color-grade filter (vivid, warm, cool, noir, …). null clears.";
    "set_clip_blend_mode" => SetClipBlendMode(SetClipBlendMode),
        "Set blend mode over layers below: normal, darken, multiply, color_burn, lighten, screen, color_dodge, add, overlay, soft_light, hard_light, difference, exclusion.";
    "set_motion_blur" => SetMotionBlur(SetMotionBlur),
        "Transform motion blur on a visual clip (not animatable): enabled, shutter_deg 0..720 (default 180), samples 2..32 (clamped to 16). Needs an animated transform.";
    "set_layer_styles" => SetClipLayerStyles(SetClipLayerStyles),
        "Set layer shadow/glow/outline/background (reference px @1080p; not text glyph styles). Omitted blocks removed; empty styles clears all. Animate via style params.";
    "set_clip_adjustments" => SetClipAdjustments(SetClipAdjustments),
        "Manual color adjust: brightness/contrast/saturation/exposure/temperature/tint/hue/highlights/shadows (−1..1); sharpness/vignette (0..1). Omitted keep.";
    "set_clip_animation" => SetClipAnimation(SetClipAnimation),
        "Set/clear look animation. Slots: in/out/combo. Optional speed 0.25–4, intensity 0–2, stagger 0–2 (text). null animation clears.";
    "set_audio_role" => SetAudioRole(SetAudioRole),
        "Tag an audio-lane clip: music, sfx, voiceover, extracted. null clears.";
    "split_clip" => SplitClip(SplitClip),
        "Split a clip at timeline seconds into two abutting clips.";
    "trim_clip" => TrimClip(TrimClip),
        "Re-place/trim to new start and duration (seconds). Media head trim advances source in-point.";
    "move_clip" => MoveClip(MoveClip),
        "Move a clip to a track at a new start (seconds), keeping duration.";
    "remove_clip" => RemoveClip(RemoveClip),
        "Remove a clip, leaving a gap.";
    "remove_track" => RemoveTrack(RemoveTrack),
        "Remove a track and its clips. Main track (see describe_project) cannot be removed.";
    "set_track_enabled" => SetTrackEnabled(SetTrackEnabled),
        "Show or hide a visual track in the composite.";
    "set_track_muted" => SetTrackMuted(SetTrackMuted),
        "Mute or unmute an audio track.";
    "set_track_locked" => SetTrackLocked(SetTrackLocked),
        "Lock or unlock a track against editing.";
    "ripple_delete" => RippleDelete(RippleDelete),
        "Remove a clip and slide later clips left to close the gap.";
    "shift_clips" => ShiftClips(ShiftClips),
        "Shift clips on a track at/after a position by signed seconds.";
    "ripple_insert" => RippleInsert(RippleInsert),
        "Insert trimmed media at timeline seconds, shifting later clips right.";
    "link_clips" => LinkClips(LinkClips),
        "Link two or more clips to select/move/trim together (replaces prior links).";
    "unlink_clips" => UnlinkClips(UnlinkClips),
        "Dissolve every link group touched by the given clip ids.";
    "add_marker" => AddMarker(AddMarker),
        "Drop a named, colored ruler marker at timeline seconds. Omit color to cycle palette.";
    "remove_marker" => RemoveMarker(RemoveMarker),
        "Remove a ruler marker by id.";
    "set_marker" => SetMarker(SetMarker),
        "Move, rename, or recolor a ruler marker. Omitted fields keep.";
    "set_canvas" => SetCanvas(SetCanvas),
        "Set canvas aspect ('auto'|'16:9'|'9:16'|'1:1'|'4:5'|'21:9') and/or background color. Omitted keep.";
}
