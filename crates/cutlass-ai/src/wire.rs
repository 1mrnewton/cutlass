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
pub const TOOL_SCHEMA_VERSION: u32 = 25;

/// Model-facing clip lists stay small enough for deterministic validation and
/// useful rejection messages while covering realistic linked groups.
pub(crate) const MAX_MULTI_CLIP_REFS: usize = 64;

/// Track lane categories the agent may create or target.
///
/// The engine has more kinds (effect / filter / adjustment lanes); the agent
/// cannot create those yet — it applies effects, filters, and adjustments to
/// clips directly instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireTrackKind {
    /// Footage and other imported picture media.
    Video,
    /// Imported sound media.
    Audio,
    /// Titles and captions.
    Text,
    /// Graphic overlays: solid colors and shapes.
    Sticker,
}

/// Geometry of a generated shape clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireShape {
    Rectangle,
    Ellipse,
}

/// Synthetic clip content the agent may create.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WireGenerator {
    /// A title / text layer (rendered with the default style; styling is
    /// preserved when replacing the text of an existing text clip).
    Text {
        /// The text to display.
        content: String,
    },
    /// A solid color fill covering the canvas.
    Solid {
        /// Fill color as `[red, green, blue, alpha]`, each 0-255.
        rgba: [u8; 4],
    },
    /// A filled vector shape centered on the canvas.
    Shape {
        shape: WireShape,
        /// Fill color as `[red, green, blue, alpha]`, each 0-255.
        rgba: [u8; 4],
        /// Width in reference pixels (1080px-tall canvas). Omit to keep the
        /// clip's current size when editing an existing shape.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        width: Option<f32>,
        /// Height in reference pixels. Omit to keep the clip's current size.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        height: Option<f32>,
    },
}

/// Add a track to the timeline stack.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AddTrack {
    pub kind: WireTrackKind,
    /// Display name, e.g. "V2" or "Music".
    pub name: String,
    /// Stack position (0 = bottom layer, composited first). Omit to add on
    /// top of the stack.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
}

/// Place a trimmed range of imported media on a video or audio track.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AddClip {
    /// Target track id.
    pub track: u64,
    /// Media pool id of the source file.
    pub media: u64,
    /// In-point within the source media, in seconds.
    pub source_start: f64,
    /// Length of the source range to use, in seconds.
    pub source_duration: f64,
    /// Where the clip begins on the timeline, in seconds.
    pub start: f64,
}

/// Detach a video clip's embedded sound onto an explicit audio track.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ExtractAudio {
    /// Source media clip on a video track.
    pub clip: u64,
    /// Target unlocked audio track. This is required: call `add_track` first
    /// when no suitable audio lane exists, then use its returned id.
    pub track: u64,
}

/// Make a deep property-preserving copy of one clip at an explicit target
/// track and timeline start. The copy receives a fresh unlinked clip id.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DuplicateClip {
    /// Source clip to copy.
    pub clip: u64,
    /// Explicit destination track id.
    pub to_track: u64,
    /// Explicit destination start in timeline seconds.
    pub start: f64,
}

/// Place a generated clip (text, solid color, shape) on a matching track.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AddGenerated {
    /// Target track id. Text goes on text tracks; solids and shapes go on
    /// sticker (overlay) tracks.
    pub track: u64,
    /// The content to generate, as a tagged object — e.g.
    /// `{"type": "text", "content": "Hello"}`,
    /// `{"type": "solid", "rgba": [0, 0, 0, 255]}`, or
    /// `{"type": "shape", "shape": "ellipse", "rgba": [255, 0, 0, 255]}`.
    pub generator: WireGenerator,
    /// Where the clip begins on the timeline, in seconds.
    pub start: f64,
    /// Clip length on the timeline, in seconds.
    pub duration: f64,
}

/// Replace a generated clip's content (edit a title's text, recolor a
/// shape). Rejected for media-backed clips. Replacing the text of a text
/// clip keeps its current styling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetGenerator {
    /// The generated clip to modify.
    pub clip: u64,
    /// The replacement content, as a tagged object — e.g.
    /// `{"type": "text", "content": "Hello"}`,
    /// `{"type": "solid", "rgba": [0, 0, 0, 255]}`, or
    /// `{"type": "shape", "shape": "ellipse", "rgba": [255, 0, 0, 255]}`.
    pub generator: WireGenerator,
}

/// Change a clip's placement on the canvas. Omitted fields keep their
/// current value. Rejected for clips on audio tracks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetClipTransform {
    pub clip: u64,
    /// Horizontal offset of the content center from the canvas center, as a
    /// fraction of canvas width (+ is right; 0.5 puts the center on the
    /// right edge).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position_x: Option<f64>,
    /// Vertical offset of the content center from the canvas center, as a
    /// fraction of canvas height (+ is down).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position_y: Option<f64>,
    /// Horizontal anchor within the content bounds (0 = left, 0.5 = center).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor_x: Option<f64>,
    /// Vertical anchor within the content bounds (0 = top, 0.5 = center).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor_y: Option<f64>,
    /// Uniform scale; 1.0 fits the content inside the canvas (100%).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<f64>,
    /// Clockwise rotation in degrees about the content center.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotation: Option<f64>,
    /// Layer opacity, 0.0 (transparent) to 1.0 (opaque).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opacity: Option<f64>,
}

/// Crop a clip to a sub-region of its frame and/or mirror it. Crop values
/// are the fractions trimmed off each edge (left 0.25 removes the left
/// quarter); the kept region aspect-fits the canvas exactly like the full
/// frame did, so cropping never moves the layer. Omitted fields keep
/// their current value. Rejected for clips on audio tracks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetClipCrop {
    pub clip: u64,
    /// Fraction of the frame width trimmed off the left edge (0–1). Omit
    /// to keep the current value; 0 restores the edge.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub left: Option<f64>,
    /// Fraction of the frame height trimmed off the top edge (0–1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top: Option<f64>,
    /// Fraction of the frame width trimmed off the right edge (0–1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub right: Option<f64>,
    /// Fraction of the frame height trimmed off the bottom edge (0–1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bottom: Option<f64>,
    /// Mirror the content left-right. Omit to keep the current state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flip_h: Option<bool>,
    /// Mirror the content top-bottom. Omit to keep the current state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flip_v: Option<bool>,
}

/// Add a visual effect to the end of a clip's effect chain. Effects run on
/// the placed layer before it composites, in chain order. Use
/// `describe_project` to see a clip's current effects and their indices.
/// Rejected for clips on audio tracks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AddEffect {
    pub clip: u64,
    /// Effect id from the catalog, e.g. `gaussian_blur` or `vignette`.
    pub effect: String,
}

/// Remove an effect from a clip's chain by its position (0 = the first
/// effect). See `describe_project` for the current chain order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoveEffect {
    pub clip: u64,
    /// Index of the effect in the clip's chain (0 = first).
    pub index: u32,
}

/// Reorder one effect within a clip's chain. Both indices address the current
/// pre-move chain; `to_index` is the moved effect's final index after removal
/// and insertion. Use `describe_project` to inspect the current order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MoveEffect {
    pub clip: u64,
    /// Current index of the effect to move (0 = first).
    pub from_index: u32,
    /// Final index for the moved effect (0 = first).
    pub to_index: u32,
}

/// Set one parameter of an effect already on a clip to a fixed value. The
/// value is range-checked against the catalog. Use `set_param_keyframe`
/// with an effect param to animate it instead.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetEffectParam {
    pub clip: u64,
    /// Index of the effect in the clip's chain (0 = first).
    pub index: u32,
    /// Parameter name, e.g. `radius` (gaussian_blur) or `amount` (vignette).
    pub param: String,
    /// New value (clamped to the parameter's catalog range).
    pub value: f64,
}

/// Add a transition at the junction where `clip` abuts the next clip on its
/// track. `clip` must butt directly against a following clip (same track, the
/// next clip starts exactly where this one ends). Rejected for clips on audio
/// tracks and clips with no abutting neighbor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AddTransition {
    pub clip: u64,
    /// Transition id from the catalog, e.g. `crossfade` or `dip_to_black`.
    pub transition: String,
}

/// Remove the transition at `clip`'s right junction. See `describe_project`
/// for which clips currently carry one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoveTransition {
    pub clip: u64,
}

/// Set the duration (in seconds) of the transition at `clip`'s right junction.
/// The window is centered on the cut.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetTransition {
    pub clip: u64,
    /// New window length in seconds (must be positive).
    pub seconds: f64,
}

/// An animatable clip property the keyframe commands can address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireClipParam {
    /// Canvas placement of the content center (vec2: use the `position`
    /// argument, not `value`).
    Position,
    /// Uniform scale (1.0 = fit inside the canvas).
    Scale,
    /// Clockwise rotation in degrees.
    Rotation,
    /// Layer opacity 0.0–1.0.
    Opacity,
    /// Audio gain envelope (0.0 = mute, 1.0 = unchanged, up to 10.0 boost).
    /// Keyframing it draws volume automation; this is how you fade audio in
    /// or out over time or duck music under a voice. Media-backed clips only.
    Volume,
}

/// Interpolation toward the next keyframe. (Custom bezier curves exist in
/// the engine but are inspector-only; the agent surface stays simple.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireEasing {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
}

/// Add or replace a keyframe on one animatable clip property, making the
/// property animate over time. The first keyframe on a property turns its
/// fixed value into a curve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetParamKeyframe {
    pub clip: u64,
    pub param: WireClipParam,
    /// Timeline position of the keyframe, in seconds. Must fall inside the
    /// clip.
    pub at: f64,
    /// New value for `scale` / `rotation` / `opacity`. Ignored for
    /// `position`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    /// New `[x, y]` for `position` (fractions of canvas size from center,
    /// +x right, +y down). Ignored for scalar params.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<[f64; 2]>,
    /// Interpolation toward the next keyframe. Defaults to linear.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub easing: Option<WireEasing>,
}

/// Remove the keyframe at exactly a timeline position on one property.
/// Removing the last keyframe freezes the property at that value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoveParamKeyframe {
    pub clip: u64,
    pub param: WireClipParam,
    /// Timeline position of the keyframe to remove, in seconds.
    pub at: f64,
}

/// Set one animatable property to a fixed value, removing all its
/// keyframes (stops the animation).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetParamConstant {
    pub clip: u64,
    pub param: WireClipParam,
    /// New value for `scale` / `rotation` / `opacity`. Ignored for
    /// `position`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    /// New `[x, y]` for `position`. Ignored for scalar params.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<[f64; 2]>,
}

/// Change a media clip's constant playback speed and/or direction. The clip
/// keeps its timeline start and source footage; its timeline length
/// re-derives from the speed (a 2x clip takes half the time). Audio
/// time-stretches to match (pitch preserved by default; see set_clip_pitch).
/// Not valid for generated clips (text/solid/shape).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetClipSpeed {
    /// The media clip to retime.
    pub clip: u64,
    /// Playback rate multiplier: 2.0 plays at double speed (half as long on
    /// the timeline), 0.5 is half-speed slow motion. Allowed range 0.05 to
    /// 100. Omit to keep the current speed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
    /// Play the clip's footage backwards. Omit to keep the current
    /// direction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reversed: Option<bool>,
}

/// Apply (or clear) a CapCut-style speed ramp on a media clip: its playback
/// speed varies across its length following a named preset, instead of a
/// single constant speed. The clip keeps its source footage; its timeline
/// length re-derives from the ramp's average speed. The audio time-stretches
/// along the ramp. Not valid for generated clips.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetSpeedCurve {
    /// The media clip to ramp.
    pub clip: u64,
    /// Named ramp preset: "ramp_up" (slow→fast), "ramp_down" (fast→slow),
    /// "montage" (fast/slow/fast), "hero" (dip to slow-mo on the action),
    /// "bullet" (fast / hard slow / fast). Omit or set null to clear the ramp
    /// back to a constant speed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset: Option<String>,
}

/// Lock or unlock a retimed media clip's pitch (CapCut "pitch" switch). When
/// preserved (the default), the audio time-stretches and keeps its original
/// pitch; when not, pitch rides the playback speed — the "chipmunk" effect on
/// a sped-up clip, a deep growl on a slowed one. Only affects sound on a
/// retimed clip (a speed change, reverse, or ramp). Not valid for generated
/// clips.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetClipPitch {
    /// The media clip whose pitch handling to set.
    pub clip: u64,
    /// True keeps the original pitch (time-stretch); false lets the pitch
    /// follow the playback speed.
    pub preserve_pitch: bool,
}

/// Set a clip's audio mix: a constant volume gain plus linear fade-in/out
/// ramps. Volume 1.0 is unchanged, 0.0 mutes, 2.0 doubles (max 10). Fades
/// are seconds of ramp at the clip's head/tail. A video clip keeps its own
/// sound, so target it directly; only when a clip's audio was separated onto a
/// linked audio lane do you target that clip instead.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetClipAudio {
    /// The media clip to adjust — a video clip with sound, or an audio clip.
    pub clip: u64,
    /// Gain multiplier: 0.0 mutes, 1.0 keeps the recorded level, up to a
    /// maximum of 10. Omit to keep the current volume.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub volume: Option<f64>,
    /// Fade-in duration in seconds from the clip's start (0 removes the
    /// fade). Omit to keep the current fade-in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fade_in: Option<f64>,
    /// Fade-out duration in seconds ending at the clip's end (0 removes the
    /// fade). Omit to keep the current fade-out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fade_out: Option<f64>,
}

/// Turn noise reduction on or off for a media clip (CapCut "Reduce noise").
/// Runs the clip's audio through a speech-preserving denoiser that suppresses
/// steady background noise — fan hum, hiss, air-conditioning, room tone — while
/// keeping voice. Best on clips with a constant background drone. A video clip
/// keeps its own sound, so target it directly; only when its audio was
/// separated onto a linked audio lane do you target that clip instead. Not
/// valid for generated clips.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetDenoise {
    /// The media clip to clean.
    pub clip: u64,
    /// True turns noise reduction on, false off.
    pub denoise: bool,
}

/// Mask shape kinds (CapCut mask presets).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireMaskKind {
    Linear,
    Mirror,
    Circle,
    Rectangle,
    Heart,
    Star,
}

/// A shaped alpha mask on a media-backed visual clip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct WireMask {
    pub kind: WireMaskKind,
    /// Edge softness, 0 (hard) … 1 (fully feathered). Defaults to 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feather: Option<f64>,
    /// Keep the outside instead of the inside.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invert: Option<bool>,
}

/// Set (or clear) a per-clip mask on a media-backed visual clip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetClipMask {
    pub clip: u64,
    /// Mask to apply. `null` clears the mask.
    pub mask: Option<WireMask>,
}

/// Chroma key settings for green-screen style removal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct WireChromaKey {
    /// Key color as `[red, green, blue]`, each 0-255.
    pub rgb: [u8; 3],
    /// Keying strength (tolerance), 0 … 1. Defaults to 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strength: Option<f64>,
    /// Shadow retention, 0 … 1. Defaults to 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow: Option<f64>,
}

/// Set (or clear) chroma keying on a media-backed visual clip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetClipChroma {
    pub clip: u64,
    /// Chroma settings. `null` clears chroma key.
    pub chroma: Option<WireChromaKey>,
}

/// Stabilization strength presets (CapCut stabilize).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireStabilizeLevel {
    Recommended,
    Smooth,
    MaxSmooth,
}

/// Set (or clear) video stabilization on a media-backed video clip (not stills).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetClipStabilize {
    pub clip: u64,
    /// Stabilization preset. `null` clears stabilization.
    pub level: Option<WireStabilizeLevel>,
}

/// A color-grade filter preset and blend intensity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct WireFilter {
    /// Catalog id (e.g. "vivid", "warm", "noir").
    pub id: String,
    /// Blend of the graded result over the original, 0 … 1. Defaults to 0.8.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intensity: Option<f64>,
}

/// Set (or clear) a color-grade filter preset on any visual clip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetClipFilter {
    pub clip: u64,
    /// Filter preset. `null` clears the filter.
    pub filter: Option<WireFilter>,
}

/// Set manual color adjustments (CapCut adjust) on any visual clip. Omitted
/// sliders keep their current value; all-neutral clears the grade.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetClipAdjustments {
    pub clip: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub brightness: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contrast: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub saturation: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exposure: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
}

/// Animation slot (CapCut In / Out / Combo tabs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireAnimationSlot {
    In,
    Out,
    Combo,
}

/// Set (or clear) a look animation preset on any visual clip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetClipAnimation {
    pub clip: u64,
    /// Which animation slot to set.
    pub slot: WireAnimationSlot,
    /// Catalog animation id (e.g. "fade_in", "pulse"). `null` clears the slot.
    pub animation: Option<String>,
}

/// Audio role tags for clips on audio tracks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireAudioRole {
    Music,
    Sfx,
    Voiceover,
    Extracted,
}

/// Tag (or untag) what an audio-lane clip is (music / sfx / voiceover / extracted).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetAudioRole {
    pub clip: u64,
    /// Role to apply. `null` clears the tag.
    pub role: Option<WireAudioRole>,
}

/// Split a clip at a timeline position into two abutting clips.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SplitClip {
    pub clip: u64,
    /// Timeline position of the cut, in seconds. Must fall strictly inside
    /// the clip.
    pub at: f64,
}

/// Re-place / trim a clip to a new timeline range. Trimming the head of a
/// media clip advances its source in-point to match (like dragging a trim
/// handle).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TrimClip {
    pub clip: u64,
    /// New timeline start of the clip, in seconds.
    pub start: f64,
    /// New clip length, in seconds.
    pub duration: f64,
}

/// Move a clip to a track at a new start time, keeping its duration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MoveClip {
    pub clip: u64,
    /// Destination track id (may be the clip's current track).
    pub to_track: u64,
    /// New timeline start, in seconds.
    pub start: f64,
}

/// Remove a clip, leaving a gap where it sat.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoveClip {
    pub clip: u64,
}

/// Remove a track and any clips still on it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoveTrack {
    pub track: u64,
}

/// Toggle whether a visual track contributes to the composite.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetTrackEnabled {
    pub track: u64,
    pub enabled: bool,
}

/// Toggle whether an audio track is silenced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetTrackMuted {
    pub track: u64,
    pub muted: bool,
}

/// Toggle whether a track's clips are editable (selection / move / trim).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetTrackLocked {
    pub track: u64,
    pub locked: bool,
}

/// Remove a clip and slide later clips on its track left to close the gap.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RippleDelete {
    pub clip: u64,
}

/// Shift every clip on a track that starts at or after `from` by `delta`
/// seconds (negative shifts left). Rejected if a left shift would collide
/// with an earlier clip or push a clip before time 0.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ShiftClips {
    pub track: u64,
    /// Clips starting at or after this timeline position (seconds) shift.
    pub from: f64,
    /// Signed shift amount in seconds.
    pub delta: f64,
}

/// Insert a trimmed range of media at a timeline position, first shifting
/// every clip at or after that position right to make room.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RippleInsert {
    /// Target track id (video or audio).
    pub track: u64,
    /// Media pool id of the source file.
    pub media: u64,
    /// In-point within the source media, in seconds.
    pub source_start: f64,
    /// Length of the source range to use, in seconds.
    pub source_duration: f64,
    /// Timeline position of the insert, in seconds.
    pub at: f64,
}

/// Put two or more clips into one link group: linked clips select, move,
/// and trim together. Replaces any previous links on those clips.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LinkClips {
    /// Ids of the clips to link (at least two).
    #[schemars(length(min = 2, max = MAX_MULTI_CLIP_REFS))]
    pub clips: Vec<u64>,
}

/// Dissolve every link group touched by one or more clips. Naming any member
/// clears the complete group; distinct members of the same group are harmless
/// and coalesced by the engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct UnlinkClips {
    /// Ids of linked clips whose complete groups should be dissolved. List
    /// each clip id once; callers do not need to enumerate every group member.
    #[schemars(length(min = 1, max = MAX_MULTI_CLIP_REFS))]
    pub clips: Vec<u64>,
}

/// Marker flag colors (the editor's fixed palette).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireMarkerColor {
    Teal,
    Blue,
    Purple,
    Pink,
    Red,
    Orange,
    Yellow,
    Green,
}

/// Drop a named, colored marker on the timeline ruler — an anchor for
/// navigation and for aligning edits ("cut at the marker", beat sync).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AddMarker {
    /// Timeline position of the marker, in seconds.
    pub at: f64,
    /// Short label shown beside the flag (e.g. "Drop", "Beat 1"). Omit for
    /// an unnamed marker.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Flag color. Omit to cycle the palette automatically.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<WireMarkerColor>,
}

/// Remove a timeline marker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoveMarker {
    /// The marker to remove.
    pub marker: u64,
}

/// Move, rename, or recolor an existing timeline marker. Omitted fields
/// keep their current value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetMarker {
    /// The marker to change.
    pub marker: u64,
    /// New timeline position in seconds. Omit to keep the position.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<f64>,
    /// New label ("" clears it). Omit to keep the name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// New flag color. Omit to keep the color.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<WireMarkerColor>,
}

/// Canvas aspect-ratio presets the agent may pick.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum WireCanvasAspect {
    /// Follow the footage: canvas shape and size derive from the largest
    /// video media on the timeline (the default).
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "16:9")]
    Wide16x9,
    #[serde(rename = "9:16")]
    Tall9x16,
    #[serde(rename = "1:1")]
    Square1x1,
    #[serde(rename = "4:5")]
    Portrait4x5,
    #[serde(rename = "21:9")]
    Cinema21x9,
}

impl WireCanvasAspect {
    /// The serialized name (`"auto"`, `"16:9"`, …), for transcripts.
    pub fn name(self) -> &'static str {
        match self {
            WireCanvasAspect::Auto => "auto",
            WireCanvasAspect::Wide16x9 => "16:9",
            WireCanvasAspect::Tall9x16 => "9:16",
            WireCanvasAspect::Square1x1 => "1:1",
            WireCanvasAspect::Portrait4x5 => "4:5",
            WireCanvasAspect::Cinema21x9 => "21:9",
        }
    }
}

/// Set the project canvas: the aspect-ratio preset the composite renders
/// at, and/or the background color shown where no clip covers the canvas.
/// Omitted fields keep their current value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetCanvas {
    /// Canvas aspect preset: "auto" follows the footage; "16:9", "9:16",
    /// "1:1", "4:5", and "21:9" fix the shape (clips re-fit automatically).
    /// Omit to keep the current preset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aspect: Option<WireCanvasAspect>,
    /// Canvas background color as `[red, green, blue]`, each 0-255. Omit
    /// to keep the current background.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background: Option<[u8; 3]>,
}

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
    SetClipSpeed(SetClipSpeed),
    SetSpeedCurve(SetSpeedCurve),
    SetClipPitch(SetClipPitch),
    SetClipAudio(SetClipAudio),
    SetDenoise(SetDenoise),
    SetClipMask(SetClipMask),
    SetClipChroma(SetClipChroma),
    SetClipStabilize(SetClipStabilize),
    SetClipFilter(SetClipFilter),
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
            WireCommand::SetClipSpeed(a) => clip(&mut a.clip),
            WireCommand::SetSpeedCurve(a) => clip(&mut a.clip),
            WireCommand::SetClipPitch(a) => clip(&mut a.clip),
            WireCommand::SetClipAudio(a) => clip(&mut a.clip),
            WireCommand::SetDenoise(a) => clip(&mut a.clip),
            WireCommand::SetClipMask(a) => clip(&mut a.clip),
            WireCommand::SetClipChroma(a) => clip(&mut a.clip),
            WireCommand::SetClipStabilize(a) => clip(&mut a.clip),
            WireCommand::SetClipFilter(a) => clip(&mut a.clip),
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
        "Add or replace a keyframe on a clip property (position, scale, rotation, opacity) at a timeline position in seconds, animating it over time. Use 'value' for scalar params, 'position' for position.";
    "remove_param_keyframe" => RemoveParamKeyframe(RemoveParamKeyframe),
        "Remove the keyframe at a timeline position (seconds) on a clip property. Removing the last keyframe freezes the property at that value.";
    "set_param_constant" => SetParamConstant(SetParamConstant),
        "Set a clip property to a fixed value and remove all its keyframes (stops its animation).";
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
        "Set or clear a shaped alpha mask on a media-backed visual clip. Mask kinds: linear, mirror, circle, rectangle, heart, star. Pass null for mask to clear.";
    "set_clip_chroma" => SetClipChroma(SetClipChroma),
        "Set or clear chroma keying (green screen) on a media-backed visual clip. Pass null for chroma to clear.";
    "set_clip_stabilize" => SetClipStabilize(SetClipStabilize),
        "Set or clear video stabilization on a media-backed video clip (not still images). Levels: recommended, smooth, max_smooth. Pass null for level to clear.";
    "set_clip_filter" => SetClipFilter(SetClipFilter),
        "Set or clear a color-grade filter preset on any visual clip. Filter ids include vivid, warm, cool, noir, sunset, and others from the catalog. Pass null for filter to clear.";
    "set_clip_adjustments" => SetClipAdjustments(SetClipAdjustments),
        "Set manual color adjustments (brightness, contrast, saturation, exposure, temperature) on any visual clip. Each slider is -1..1; omitted sliders keep their current value.";
    "set_clip_animation" => SetClipAnimation(SetClipAnimation),
        "Set or clear a look animation preset on any visual clip. Slots: in (entrance), out (exit), combo (looping). Animation ids include fade_in, slide_up, pulse, etc. Pass null for animation to clear the slot.";
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tagged_json_round_trips() {
        let cmd = WireCommand::TrimClip(TrimClip {
            clip: 12,
            start: 14.0,
            duration: 4.0,
        });
        let json = serde_json::to_value(&cmd).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "command": "trim_clip",
                "clip": 12,
                "start": 14.0,
                "duration": 4.0,
            })
        );
        let back: WireCommand = serde_json::from_value(json).unwrap();
        assert_eq!(back, cmd);
    }

    #[test]
    fn from_tool_call_decodes_arguments() {
        let cmd =
            WireCommand::from_tool_call("split_clip", serde_json::json!({ "clip": 7, "at": 12.4 }))
                .unwrap();
        assert_eq!(cmd, WireCommand::SplitClip(SplitClip { clip: 7, at: 12.4 }));
        assert_eq!(cmd.tool_name(), "split_clip");
    }

    #[test]
    fn duplicate_clip_tagged_json_and_tool_arguments_are_strict() {
        let json = serde_json::json!({
            "command": "duplicate_clip",
            "clip": 7,
            "to_track": 3,
            "start": 12.5,
        });
        let command: WireCommand = serde_json::from_value(json).unwrap();
        assert_eq!(
            command,
            WireCommand::DuplicateClip(DuplicateClip {
                clip: 7,
                to_track: 3,
                start: 12.5,
            })
        );
        assert_eq!(command.tool_name(), "duplicate_clip");

        let missing = WireCommand::from_tool_call(
            "duplicate_clip",
            serde_json::json!({ "clip": 7, "to_track": 3 }),
        )
        .unwrap_err();
        assert!(missing.contains("missing field `start`"), "{missing}");

        let extra = WireCommand::from_tool_call(
            "duplicate_clip",
            serde_json::json!({
                "clip": 7,
                "to_track": 3,
                "start": 12.5,
                "ripple": true,
            }),
        )
        .unwrap_err();
        assert!(extra.contains("unknown field `ripple`"), "{extra}");
    }

    #[test]
    fn from_tool_call_rejects_unknown_tool_and_bad_args() {
        let err = WireCommand::from_tool_call("save_project", serde_json::json!({})).unwrap_err();
        assert!(err.contains("unknown tool 'save_project'"));
        assert!(err.contains("add_clip"));

        let err =
            WireCommand::from_tool_call("trim_clip", serde_json::json!({ "clip": "not-a-number" }))
                .unwrap_err();
        assert!(err.contains("invalid arguments for trim_clip"));
    }

    #[test]
    fn generator_decode_rejection_carries_a_corrective_example() {
        // The historical failure mode: a model passes the title text as a
        // bare string instead of the tagged object.
        let err = WireCommand::from_tool_call(
            "add_generated",
            serde_json::json!({
                "track": 2,
                "generator": "Hello world",
                "start": 0.0,
                "duration": 3.0,
            }),
        )
        .unwrap_err();
        assert!(err.contains("invalid arguments for add_generated"), "{err}");
        assert!(
            err.contains("{\"type\": \"text\", \"content\": \"Hello\"}"),
            "{err}"
        );
    }

    #[test]
    fn tool_schemas_are_fully_inlined() {
        // No `$ref` indirection anywhere: weak local models read schemas
        // literally and guess when the shape is behind a reference.
        for spec in tool_specs() {
            let rendered = spec.parameters.to_string();
            assert!(
                !rendered.contains("$ref") && !rendered.contains("$defs"),
                "{} schema is not self-contained: {rendered}",
                spec.name
            );
        }
    }

    #[test]
    fn generator_wire_format_is_tagged_lowercase() {
        let shape = WireGenerator::Shape {
            shape: WireShape::Ellipse,
            rgba: [255, 0, 0, 255],
            width: None,
            height: None,
        };
        assert_eq!(
            serde_json::to_value(&shape).unwrap(),
            serde_json::json!({ "type": "shape", "shape": "ellipse", "rgba": [255, 0, 0, 255] })
        );
    }

    #[test]
    fn remap_ids_rewrites_only_mapped_references() {
        let clip_map = std::collections::HashMap::from([(10u64, 99u64)]);
        let track_map = std::collections::HashMap::from([(2u64, 7u64)]);
        let marker_map = std::collections::HashMap::from([(4u64, 40u64)]);

        let mut mv = WireCommand::MoveClip(MoveClip {
            clip: 10,
            to_track: 2,
            start: 1.0,
        });
        mv.remap_ids(&clip_map, &track_map, &marker_map);
        assert_eq!(
            mv,
            WireCommand::MoveClip(MoveClip {
                clip: 99,
                to_track: 7,
                start: 1.0,
            })
        );

        let mut move_effect = WireCommand::MoveEffect(MoveEffect {
            clip: 10,
            from_index: 0,
            to_index: 2,
        });
        move_effect.remap_ids(&clip_map, &track_map, &marker_map);
        assert_eq!(
            move_effect,
            WireCommand::MoveEffect(MoveEffect {
                clip: 99,
                from_index: 0,
                to_index: 2,
            })
        );

        let mut extract = WireCommand::ExtractAudio(ExtractAudio { clip: 10, track: 2 });
        extract.remap_ids(&clip_map, &track_map, &marker_map);
        assert_eq!(
            extract,
            WireCommand::ExtractAudio(ExtractAudio { clip: 99, track: 7 })
        );

        let mut duplicate = WireCommand::DuplicateClip(DuplicateClip {
            clip: 10,
            to_track: 2,
            start: 8.0,
        });
        duplicate.remap_ids(&clip_map, &track_map, &marker_map);
        assert_eq!(
            duplicate,
            WireCommand::DuplicateClip(DuplicateClip {
                clip: 99,
                to_track: 7,
                start: 8.0,
            })
        );

        // Unmapped ids pass through; link lists remap element-wise.
        let mut link = WireCommand::LinkClips(LinkClips {
            clips: vec![10, 11],
        });
        link.remap_ids(&clip_map, &track_map, &marker_map);
        assert_eq!(
            link,
            WireCommand::LinkClips(LinkClips {
                clips: vec![99, 11],
            })
        );

        let mut unlink = WireCommand::UnlinkClips(UnlinkClips {
            clips: vec![11, 10],
        });
        unlink.remap_ids(&clip_map, &track_map, &marker_map);
        assert_eq!(
            unlink,
            WireCommand::UnlinkClips(UnlinkClips {
                clips: vec![11, 99],
            })
        );

        // Marker references follow the marker map (sandbox add_marker ids
        // land on the live engine's ids during plan replay).
        let mut set = WireCommand::SetMarker(SetMarker {
            marker: 4,
            at: Some(2.0),
            name: None,
            color: None,
        });
        set.remap_ids(&clip_map, &track_map, &marker_map);
        assert_eq!(
            set,
            WireCommand::SetMarker(SetMarker {
                marker: 40,
                at: Some(2.0),
                name: None,
                color: None,
            })
        );
    }

    #[test]
    fn tool_specs_cover_every_command_with_object_schemas() {
        let specs = tool_specs();
        assert_eq!(specs.len(), 47);
        for spec in &specs {
            assert!(
                !spec.description.is_empty(),
                "{} missing description",
                spec.name
            );
            assert_eq!(
                spec.parameters.get("type").and_then(|t| t.as_str()),
                Some("object"),
                "{} schema is not an object",
                spec.name
            );
        }
    }

    #[test]
    fn extract_audio_schema_requires_explicit_clip_and_track() {
        let specs = tool_specs();
        let extract = specs
            .iter()
            .find(|spec| spec.name == "extract_audio")
            .expect("extract_audio tool");
        assert_eq!(
            extract.parameters["required"],
            serde_json::json!(["clip", "track"])
        );
        assert!(
            extract
                .description
                .contains("planned track ids remap correctly")
        );
        assert!(
            WireCommand::from_tool_call("extract_audio", serde_json::json!({"clip": 7}))
                .unwrap_err()
                .contains("missing field `track`")
        );
    }

    #[test]
    fn duplicate_clip_schema_requires_only_explicit_placement_fields() {
        let specs = tool_specs();
        let duplicate = specs
            .iter()
            .find(|spec| spec.name == "duplicate_clip")
            .expect("duplicate_clip tool");
        assert_eq!(
            duplicate.parameters["required"],
            serde_json::json!(["clip", "to_track", "start"])
        );
        assert_eq!(duplicate.parameters["additionalProperties"], false);
        assert!(
            duplicate
                .description
                .contains("deep property-preserving copy")
        );
        assert!(duplicate.description.contains("fresh unlinked clip id"));
        assert!(
            duplicate
                .description
                .contains("explicit target track and start")
        );
        assert!(
            duplicate
                .description
                .contains("does not ripple clips or search for space")
        );
    }

    #[test]
    fn move_effect_schema_uses_u32_indices() {
        let specs = tool_specs();
        let move_effect = specs
            .iter()
            .find(|spec| spec.name == "move_effect")
            .expect("move_effect tool");
        assert_eq!(
            move_effect.parameters["required"],
            serde_json::json!(["clip", "from_index", "to_index"])
        );
        for field in ["from_index", "to_index"] {
            let index = &move_effect.parameters["properties"][field];
            assert_eq!(index["type"], "integer");
            assert_eq!(index["format"], "uint32");
            assert_eq!(index["minimum"], 0);
        }
    }

    #[test]
    fn unlink_schema_is_a_bounded_nonempty_clip_list() {
        let specs = tool_specs();
        let unlink = specs
            .iter()
            .find(|spec| spec.name == "unlink_clips")
            .expect("unlink_clips tool");
        let clips = &unlink.parameters["properties"]["clips"];
        assert_eq!(clips["type"], "array");
        assert_eq!(clips["minItems"], 1);
        assert_eq!(clips["maxItems"], MAX_MULTI_CLIP_REFS);
        assert_eq!(clips["items"]["type"], "integer");
        assert_eq!(unlink.parameters["required"], serde_json::json!(["clips"]));
    }
}
