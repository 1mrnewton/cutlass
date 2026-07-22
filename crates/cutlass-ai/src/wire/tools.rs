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
    let parameters = serde_json::to_value(settings.into_generator().into_root_schema_for::<T>())
        .expect("tool argument schemas are plain data and always serialize");
    ToolSpec {
        name: name.to_string(),
        description: description.to_string(),
        parameters,
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
        "Add a track to the timeline stack (video, audio, text, or sticker overlay lane). Lanes keep CapCut zones: audio at the bottom, then the main video track, overlays above it, text on top — the index only orders a lane within its zone.";
    "add_clip" => AddClip(AddClip),
        "Place a trimmed range of an imported media file on a video or audio track. Times are in seconds.";
    "extract_audio" => ExtractAudio(ExtractAudio),
        "Detach a video clip's embedded sound onto an existing unlocked audio track, preserving its exact placement and audio/retime settings. The track id is required: call add_track with kind audio first when needed, then pass the returned id. Keeping the target explicit lets planned track ids remap correctly during replay.";
    "duplicate_clip" => DuplicateClip(DuplicateClip),
        "Make a deep property-preserving copy of one clip at an explicit target track and start (timeline seconds). The copy gets a fresh unlinked clip id. This tool does not ripple clips or search for space; choose a non-overlapping destination explicitly.";
    "add_generated" => AddGenerated(AddGenerated),
        "Place a generated clip (text title, solid color, or shape) on a matching track. Times are in seconds.";
    "set_generator" => SetGenerator(SetGenerator),
        "Replace a generated clip's content: change a title's text (styling preserved) or recolor a solid/shape. Not valid for media clips.";
    "set_clip_transform" => SetClipTransform(SetClipTransform),
        "Change a clip's placement on the canvas: position, scale, rotation, opacity. Omitted fields keep their current value. Not valid on audio tracks.";
    "set_clip_crop" => SetClipCrop(SetClipCrop),
        "Crop a clip to a sub-region of its frame (fractions trimmed off each edge; 0 restores an edge) and/or mirror it with flip_h / flip_v. Omitted fields keep their current value. Not valid on audio tracks.";
    "add_effect" => AddEffect(AddEffect),
        "Add a visual effect to a clip's effect chain. Available effects: gaussian_blur (param 'radius'), vignette (param 'amount'). Effects render on the placed layer, in chain order. Not valid on audio tracks.";
    "remove_effect" => RemoveEffect(RemoveEffect),
        "Remove an effect from a clip's chain by its index (0 = first). See describe_project for a clip's current effects.";
    "move_effect" => MoveEffect(MoveEffect),
        "Reorder a clip's effect chain. Both from_index and to_index address the current pre-move chain; to_index is the effect's final index. See describe_project for the current order.";
    "set_effect_param" => SetEffectParam(SetEffectParam),
        "Set a parameter of an effect on a clip to a value (e.g. gaussian_blur 'radius', vignette 'amount'). Use describe_project to see effect indices and current params.";
    "add_transition" => AddTransition(AddTransition),
        "Add a transition at the cut where a clip meets the next clip on its track. Available transitions: crossfade, dip_to_black, dip_to_white, wipe_left, wipe_right, wipe_up, wipe_down, slide. The clip must butt directly against a following clip. Not valid on audio tracks.";
    "remove_transition" => RemoveTransition(RemoveTransition),
        "Remove the transition at a clip's right cut. See describe_project for which clips carry one.";
    "set_transition" => SetTransition(SetTransition),
        "Set the duration in seconds of the transition at a clip's right cut (centered on the cut).";
    "set_param_keyframe" => SetParamKeyframe(SetParamKeyframe),
        "Add or replace a keyframe on any animatable clip property at a timeline position in seconds. The `param` selector supports transform properties (position, anchor_point, scale, rotation, opacity), volume, speed, an effect `{effect:{index,param}}`, generated-shape `{shape:{param}}`, generated-text `{text:{param}}`, and color-look `{look:{param}}` properties. Use `value` for scalars, `position` for position/anchor_point, and `rgba` for text or shape colors. Effect names and valid properties are visible in describe_project.";
    "remove_param_keyframe" => RemoveParamKeyframe(RemoveParamKeyframe),
        "Remove the keyframe at a timeline position (seconds) on any animatable clip property. Use the same `param` selector as set_param_keyframe. Removing the last keyframe freezes the property at that value.";
    "set_param_constant" => SetParamConstant(SetParamConstant),
        "Set any animatable clip property to a fixed value and remove all its keyframes (stops its animation). Use the same `param` selector as set_param_keyframe, with `value` for scalars, `position` for position/anchor_point, or `rgba` for text and shape colors.";
    "set_clip_speed" => SetClipSpeed(SetClipSpeed),
        "Change a media clip's playback speed (2.0 = double speed, 0.5 = slow motion) and/or play it in reverse. The clip's timeline length re-derives from the speed; its audio time-stretches to match (pitch preserved by default). Not valid for generated clips.";
    "set_speed_curve" => SetSpeedCurve(SetSpeedCurve),
        "Apply a CapCut-style speed ramp to a media clip so its speed varies across its length: preset 'ramp_up' (slow to fast), 'ramp_down' (fast to slow), 'montage' (fast/slow/fast), 'hero' (slow-mo on the action), or 'bullet' (fast/hard-slow/fast). Omit preset to clear the ramp. The clip's length re-derives from the ramp's average speed; its audio time-stretches along the ramp. Not valid for generated clips.";
    "set_clip_pitch" => SetClipPitch(SetClipPitch),
        "Lock or unlock a retimed media clip's pitch. preserve_pitch true (default) keeps the original pitch while time-stretching; false lets pitch follow speed (the chipmunk effect when sped up). Only affects a clip that is retimed (speed change, reverse, or ramp). Not valid for generated clips.";
    "set_clip_audio" => SetClipAudio(SetClipAudio),
        "Set a clip's volume (0.0 mutes, 1.0 unchanged, 2.0 doubles) and/or fade-in/fade-out durations in seconds. Omitted fields keep their current value. A video clip keeps its own sound, so target it directly.";
    "set_denoise" => SetDenoise(SetDenoise),
        "Turn noise reduction on or off for a media clip: runs its audio through a speech-preserving denoiser that suppresses steady background noise (hum, hiss, air-conditioning, room tone) while keeping voice. Use on clips with a constant background drone. A video clip keeps its own sound, so target it directly. Not valid for generated clips.";
    "set_clip_mask" => SetClipMask(SetClipMask),
        "Set or clear a shaped alpha mask on a media-backed visual clip. Mask kinds: linear, mirror, circle, rectangle, heart, star. Optional constants: feather (0..1), invert, center/size as fractions of the layer, rotation in degrees clockwise, roundness (0..1, rectangle only). Omitted geometry uses defaults. Constants only — animate with set_param_keyframe using look params (mask_feather, mask_center, mask_size, mask_rotation, mask_roundness). Pass null for mask to clear.";
    "set_clip_chroma" => SetClipChroma(SetClipChroma),
        "Set or clear chroma keying (green screen) on a media-backed visual clip. Pass null for chroma to clear.";
    "set_clip_stabilize" => SetClipStabilize(SetClipStabilize),
        "Set or clear video stabilization on a media-backed video clip (not still images). Levels: recommended, smooth, max_smooth. Pass null for level to clear.";
    "set_clip_filter" => SetClipFilter(SetClipFilter),
        "Set or clear a color-grade filter preset on any visual clip. Filter ids include vivid, warm, cool, noir, sunset, and others from the catalog. Pass null for filter to clear.";
    "set_clip_blend_mode" => SetClipBlendMode(SetClipBlendMode),
        "Set how a visual clip composites over the layers below it (CapCut Blend). Modes: normal, darken, multiply, color_burn, lighten, screen, color_dodge, add, overlay, soft_light, hard_light, difference, exclusion. Visual clips only; normal resets to plain source-over.";
    "set_layer_styles" => SetClipLayerStyles(SetClipLayerStyles),
        "Set layer-quad shadow/glow/outline/background styles on any visual clip (distinct from text glyph treatments). Each block is optional constants in reference pixels (1080p baseline); omitted blocks are removed. Empty styles clears every block. Constants only — animate with set_param_keyframe using style params (shadow_blur, glow_color, …).";
    "set_clip_adjustments" => SetClipAdjustments(SetClipAdjustments),
        "Set manual color adjustments (brightness, contrast, saturation, exposure, temperature) on any visual clip. Each slider is -1..1; omitted sliders keep their current value.";
    "set_clip_animation" => SetClipAnimation(SetClipAnimation),
        "Set or clear a look animation preset on any visual clip. Slots: in (entrance), out (exit), combo (looping). Animation ids include fade_in, slide_up, pulse, typewriter, wave, etc. Optional speed (0.25–4), intensity (0–2), and stagger (0–2, text presets). Pass null for animation to clear the slot.";
    "set_audio_role" => SetAudioRole(SetAudioRole),
        "Tag or untag what an audio-lane clip is: music, sfx, voiceover, or extracted. Pass null for role to clear the tag. Audio-track clips only.";
    "split_clip" => SplitClip(SplitClip),
        "Split a clip at a timeline position (seconds) into two abutting clips.";
    "trim_clip" => TrimClip(TrimClip),
        "Re-place / trim a clip to a new timeline start and duration in seconds. Trimming a media clip's head advances its source in-point.";
    "move_clip" => MoveClip(MoveClip),
        "Move a clip to a track at a new start time (seconds), keeping its duration.";
    "remove_clip" => RemoveClip(RemoveClip),
        "Remove a clip, leaving a gap where it sat.";
    "remove_track" => RemoveTrack(RemoveTrack),
        "Remove a track and any clips still on it. The main track (marked in describe_project) is permanent and cannot be removed.";
    "set_track_enabled" => SetTrackEnabled(SetTrackEnabled),
        "Show or hide a visual track in the composite.";
    "set_track_muted" => SetTrackMuted(SetTrackMuted),
        "Mute or unmute an audio track.";
    "set_track_locked" => SetTrackLocked(SetTrackLocked),
        "Lock or unlock a track's clips against editing.";
    "ripple_delete" => RippleDelete(RippleDelete),
        "Remove a clip and slide later clips on its track left to close the gap.";
    "shift_clips" => ShiftClips(ShiftClips),
        "Shift every clip on a track starting at/after a position by a signed number of seconds.";
    "ripple_insert" => RippleInsert(RippleInsert),
        "Insert a trimmed range of media at a timeline position, shifting later clips right to make room. Times are in seconds.";
    "link_clips" => LinkClips(LinkClips),
        "Link two or more clips so they select, move, and trim together (replaces their previous links).";
    "unlink_clips" => UnlinkClips(UnlinkClips),
        "Dissolve every link group touched by one or more clip ids. Naming any member clears the complete group; distinct members of the same group are coalesced.";
    "add_marker" => AddMarker(AddMarker),
        "Drop a named, colored marker on the timeline ruler at a position in seconds. Omit color to cycle the palette.";
    "remove_marker" => RemoveMarker(RemoveMarker),
        "Remove a ruler marker by id.";
    "set_marker" => SetMarker(SetMarker),
        "Move, rename, or recolor a ruler marker. Omitted fields keep their current value.";
    "set_canvas" => SetCanvas(SetCanvas),
        "Set the project canvas: aspect ratio preset ('auto' follows the footage; '16:9', '9:16', '1:1', '4:5', '21:9' reshape it) and/or the background color shown where no clip covers the canvas. Omitted fields keep their current value.";
}
