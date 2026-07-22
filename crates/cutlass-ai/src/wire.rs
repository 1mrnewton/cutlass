//! The agent-facing wire format: the JSON surface the LLM sees and emits.
//!
//! Deliberately *not* serde derives on `cutlass-commands` — the wire layer is
//! shaped for LLM ergonomics (times in fractional seconds, ids as plain
//! integers, flat tagged objects) and keeps internal refactors from silently
//! changing the prompt-visible schema. Lowering to real engine commands (and
//! every guardrail) lives in [`crate::validate`].
//!
//! The vocabulary is closed by construction: project commands (open / save /
//! export / import) are not representable here, and [`WireGenerator`] carries
//! only the generator kinds the compositor actually renders — the phantom
//! sticker/effect/filter/adjustment variants cannot be expressed.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Bumped whenever the prompt-visible tool surface changes shape.
/// The snapshot test in `tests/tool_schema.rs` makes drift a reviewed diff.
///
/// 2: M2 keyframe commands (`set_param_keyframe`, `remove_param_keyframe`,
///    `set_param_constant`).
/// 3: M1 clip speed (`set_clip_speed`).
/// 4: M1 clip audio mix (`set_clip_audio`).
/// 5: M1 timeline markers (`add_marker`, `remove_marker`, `set_marker`).
/// 6: M1 crop + flip (`set_clip_crop`).
/// 7: subschemas inlined (no `$defs`/`$ref`) + `generator` field examples,
///    so small local models stop guessing nested argument shapes.
/// 8: M1 canvas settings (`set_canvas`).
/// 9: M4 effects (`add_effect`, `remove_effect`, `set_effect_param`).
/// 10: M4 transitions (`add_transition`, `remove_transition`, `set_transition`).
/// 11: M2 speed ramps (`set_speed_curve`).
/// 12: M8 volume envelopes (`volume` joins the keyframe param enum).
/// 13: M8 varispeed pitch lock (`set_clip_pitch`); retimed-audio descriptions
///     drop the "muted" language now that speed/reverse/ramp clips sound.
/// 14: M8 sidechain ducking (`duck`).
/// 15: shape generators gain optional width/height.
/// 16: optional `anchor_x`/`anchor_y` on `set_clip_transform`.
/// 17: M8 beat detection (`detect_beats`).
/// 18: M8 noise reduction (`set_denoise`).
/// 19: video clips carry their own audio (CapCut embedded audio) — the audio
///     tool descriptions drop the "linked audio companion" steering.
/// 20: look tools (mask/chroma/stabilize/filter/adjust/animation/audio_role);
///     removed unsupported `duck` and `detect_beats`.
/// 21: prompt extensions add the read-only `read_skill` tool.
/// 22: complete-group unlinking (`unlink_clips`) and bounded link-group lists.
/// 23: effect-chain reordering (`move_effect`).
/// 24: explicit-target audio extraction (`extract_audio`).
/// 25: explicit-target, property-preserving clip duplication (`duplicate_clip`).
/// 26: named easing presets (snappy/overshoot/anticipate) and raw bezier on
///     `set_param_keyframe`.
/// 27: `hold` (step) easing on `set_param_keyframe`.
/// 28: clip blend mode (`set_clip_blend_mode`).
/// 29: layer styles (`set_layer_styles`) and style params on keyframes.
/// 30: mask geometry (`center` / `size` / `rotation` / `roundness`) on
///     `set_clip_mask`.
/// 31: six new adjust sliders (tint/hue/highlights/shadows/sharpness/vignette)
///     on `set_clip_adjustments` and look keyframe params.
/// 32: typed effect params — Color via `rgba`, Vec2 via `position` on
///     `set_effect_param` / `set_param_keyframe` / `set_param_constant`.
/// 33: animatable crop — `crop` on `WireClipParam` with `rect: [x,y,w,h]`
///     on `set_param_keyframe` / `set_param_constant`.
/// 34: animatable pan — `pan` on `WireClipParam` (−1…+1 stereo balance).
/// 35: animatable text background radius/color on text keyframe params.
/// 36: per-axis scale — `set_clip_transform.scale` accepts a number or
///     `[x, y]`; scale keyframes accept `value` (uniform) or `position`.
/// 37: spatial motion-path tangents — optional `tangent_out` / `tangent_in`
///     on `set_param_keyframe` (position only, canvas-fraction handles).
/// 38: multi-keyframe easing presets — `apply_easing_preset` (bounce_out /
///     elastic_out / back_out) on scalar/vec2 animated params.
/// 39: per-clip transform motion blur (`set_motion_blur`).
pub const TOOL_SCHEMA_VERSION: u32 = 39;

mod dtos;
mod tools;

pub(crate) use dtos::MAX_MULTI_CLIP_REFS;
pub use dtos::{
    AddClip, AddEffect, AddGenerated, AddMarker, AddTrack, AddTransition, ApplyEasingPreset,
    DuplicateClip, ExtractAudio, LinkClips, MoveClip, MoveEffect, RemoveClip, RemoveEffect,
    RemoveMarker, RemoveParamKeyframe, RemoveTrack, RemoveTransition, RippleDelete, RippleInsert,
    SetAudioRole, SetCanvas, SetClipAdjustments, SetClipAnimation, SetClipAudio, SetClipBlendMode,
    SetClipChroma, SetClipCrop, SetClipFilter, SetClipLayerStyles, SetClipMask, SetClipPitch,
    SetClipSpeed, SetClipStabilize, SetClipTransform, SetDenoise, SetEffectParam, SetGenerator,
    SetMarker, SetMotionBlur, SetParamConstant, SetParamKeyframe, SetSpeedCurve, SetTrackEnabled,
    SetTrackLocked, SetTrackMuted, SetTransition, ShiftClips, SplitClip, TrimClip, UnlinkClips,
    WireAnimationSlot, WireAudioRole, WireBlendMode, WireCanvasAspect, WireChromaKey,
    WireClipParam, WireEasing, WireEasingPreset, WireFilter, WireGenerator, WireLayerBackground,
    WireLayerGlow, WireLayerOutline, WireLayerShadow, WireLayerStyles, WireLookParam,
    WireMarkerColor, WireMask, WireMaskKind, WireScale, WireShape, WireShapeParam,
    WireStabilizeLevel, WireStyleParam, WireTextParam, WireTrackKind,
};
pub use tools::{ToolSpec, describe_project_spec, tool_specs};

/// Every timeline edit the agent may request, as one tagged value.
///
/// Tool calls arrive as `(name, arguments)` pairs and convert through
/// [`WireCommand::from_tool_call`]; serialized plans (dry-run previews,
/// eval fixtures) use the `command`-tagged JSON representation directly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "command", rename_all = "snake_case")]
pub enum WireCommand {
    AddTrack(AddTrack),
    AddClip(AddClip),
    ExtractAudio(ExtractAudio),
    DuplicateClip(DuplicateClip),
    AddGenerated(AddGenerated),
    SetGenerator(SetGenerator),
    SetClipTransform(SetClipTransform),
    SetClipCrop(SetClipCrop),
    AddEffect(AddEffect),
    RemoveEffect(RemoveEffect),
    MoveEffect(MoveEffect),
    SetEffectParam(SetEffectParam),
    AddTransition(AddTransition),
    RemoveTransition(RemoveTransition),
    SetTransition(SetTransition),
    SetParamKeyframe(SetParamKeyframe),
    RemoveParamKeyframe(RemoveParamKeyframe),
    SetParamConstant(SetParamConstant),
    ApplyEasingPreset(ApplyEasingPreset),
    SetClipSpeed(SetClipSpeed),
    SetSpeedCurve(SetSpeedCurve),
    SetClipPitch(SetClipPitch),
    SetClipAudio(SetClipAudio),
    SetDenoise(SetDenoise),
    SetClipMask(SetClipMask),
    SetClipChroma(SetClipChroma),
    SetClipStabilize(SetClipStabilize),
    SetClipFilter(SetClipFilter),
    SetClipBlendMode(SetClipBlendMode),
    SetMotionBlur(SetMotionBlur),
    SetClipLayerStyles(SetClipLayerStyles),
    SetClipAdjustments(SetClipAdjustments),
    SetClipAnimation(SetClipAnimation),
    SetAudioRole(SetAudioRole),
    SplitClip(SplitClip),
    TrimClip(TrimClip),
    MoveClip(MoveClip),
    RemoveClip(RemoveClip),
    RemoveTrack(RemoveTrack),
    SetTrackEnabled(SetTrackEnabled),
    SetTrackMuted(SetTrackMuted),
    SetTrackLocked(SetTrackLocked),
    RippleDelete(RippleDelete),
    ShiftClips(ShiftClips),
    RippleInsert(RippleInsert),
    LinkClips(LinkClips),
    UnlinkClips(UnlinkClips),
    AddMarker(AddMarker),
    RemoveMarker(RemoveMarker),
    SetMarker(SetMarker),
    SetCanvas(SetCanvas),
}

impl WireCommand {
    /// Rewrite clip/track/marker references through the given maps (ids
    /// absent from a map pass through unchanged).
    ///
    /// This is what makes plan replay work: a plan is recorded against a
    /// sandbox where `add_track`/`split_clip` allocated sandbox-local ids;
    /// when the live engine replays the plan, each created entity gets a
    /// fresh id, and later steps that referenced the sandbox id must be
    /// remapped onto the real one.
    pub fn remap_ids(
        &mut self,
        clip_map: &std::collections::HashMap<u64, u64>,
        track_map: &std::collections::HashMap<u64, u64>,
        marker_map: &std::collections::HashMap<u64, u64>,
    ) {
        let clip = |id: &mut u64| {
            if let Some(mapped) = clip_map.get(id) {
                *id = *mapped;
            }
        };
        let track = |id: &mut u64| {
            if let Some(mapped) = track_map.get(id) {
                *id = *mapped;
            }
        };
        let marker = |id: &mut u64| {
            if let Some(mapped) = marker_map.get(id) {
                *id = *mapped;
            }
        };
        match self {
            WireCommand::AddTrack(_) => {}
            WireCommand::AddClip(a) => track(&mut a.track),
            WireCommand::ExtractAudio(a) => {
                clip(&mut a.clip);
                track(&mut a.track);
            }
            WireCommand::DuplicateClip(a) => {
                clip(&mut a.clip);
                track(&mut a.to_track);
            }
            WireCommand::AddGenerated(a) => track(&mut a.track),
            WireCommand::SetGenerator(a) => clip(&mut a.clip),
            WireCommand::SetClipTransform(a) => clip(&mut a.clip),
            WireCommand::SetClipCrop(a) => clip(&mut a.clip),
            WireCommand::AddEffect(a) => clip(&mut a.clip),
            WireCommand::RemoveEffect(a) => clip(&mut a.clip),
            WireCommand::MoveEffect(a) => clip(&mut a.clip),
            WireCommand::SetEffectParam(a) => clip(&mut a.clip),
            WireCommand::AddTransition(a) => clip(&mut a.clip),
            WireCommand::RemoveTransition(a) => clip(&mut a.clip),
            WireCommand::SetTransition(a) => clip(&mut a.clip),
            WireCommand::SetParamKeyframe(a) => clip(&mut a.clip),
            WireCommand::RemoveParamKeyframe(a) => clip(&mut a.clip),
            WireCommand::SetParamConstant(a) => clip(&mut a.clip),
            WireCommand::ApplyEasingPreset(a) => clip(&mut a.clip),
            WireCommand::SetClipSpeed(a) => clip(&mut a.clip),
            WireCommand::SetSpeedCurve(a) => clip(&mut a.clip),
            WireCommand::SetClipPitch(a) => clip(&mut a.clip),
            WireCommand::SetClipAudio(a) => clip(&mut a.clip),
            WireCommand::SetDenoise(a) => clip(&mut a.clip),
            WireCommand::SetClipMask(a) => clip(&mut a.clip),
            WireCommand::SetClipChroma(a) => clip(&mut a.clip),
            WireCommand::SetClipStabilize(a) => clip(&mut a.clip),
            WireCommand::SetClipFilter(a) => clip(&mut a.clip),
            WireCommand::SetClipBlendMode(a) => clip(&mut a.clip),
            WireCommand::SetMotionBlur(a) => clip(&mut a.clip),
            WireCommand::SetClipLayerStyles(a) => clip(&mut a.clip),
            WireCommand::SetClipAdjustments(a) => clip(&mut a.clip),
            WireCommand::SetClipAnimation(a) => clip(&mut a.clip),
            WireCommand::SetAudioRole(a) => clip(&mut a.clip),
            WireCommand::SplitClip(a) => clip(&mut a.clip),
            WireCommand::TrimClip(a) => clip(&mut a.clip),
            WireCommand::MoveClip(a) => {
                clip(&mut a.clip);
                track(&mut a.to_track);
            }
            WireCommand::RemoveClip(a) => clip(&mut a.clip),
            WireCommand::RemoveTrack(a) => track(&mut a.track),
            WireCommand::SetTrackEnabled(a) => track(&mut a.track),
            WireCommand::SetTrackMuted(a) => track(&mut a.track),
            WireCommand::SetTrackLocked(a) => track(&mut a.track),
            WireCommand::RippleDelete(a) => clip(&mut a.clip),
            WireCommand::ShiftClips(a) => track(&mut a.track),
            WireCommand::RippleInsert(a) => track(&mut a.track),
            WireCommand::LinkClips(a) => a.clips.iter_mut().for_each(clip),
            WireCommand::UnlinkClips(a) => a.clips.iter_mut().for_each(clip),
            WireCommand::AddMarker(_) => {}
            WireCommand::RemoveMarker(a) => marker(&mut a.marker),
            WireCommand::SetMarker(a) => marker(&mut a.marker),
            WireCommand::SetCanvas(_) => {}
        }
    }
}

#[cfg(test)]
mod tests;
