//! The command DTOs: every argument struct and enum in the wire vocabulary.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireClipParam {
    /// Canvas placement of the content center (vec2: use the `position`
    /// argument, not `value`).
    Position,
    /// Canvas point around which the clip scales and rotates (vec2: use the
    /// `position` argument, not `value`).
    AnchorPoint,
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
    /// Playback-rate ramp (scalar: 1.0 = normal speed). Media-backed clips
    /// only; keyframes use normalized positions within the clip.
    Speed,
    /// A scalar parameter of an effect already on the clip. `index` is the
    /// effect-chain position and `param` is its catalog name, as in
    /// `set_effect_param`.
    Effect { index: u32, param: String },
    /// A visual property of a generated shape clip.
    Shape { param: WireShapeParam },
    /// A visual style property of a generated text clip.
    Text { param: WireTextParam },
    /// A scalar property of the clip's color look.
    Look { param: WireLookParam },
    /// A property of the clip's layer-quad styles (shadow/glow/outline/
    /// background). Color params use `rgba`; `shadow_offset` uses
    /// `position`; the rest use scalar `value`.
    Style { param: WireStyleParam },
}

/// Animatable properties of a generated shape clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireShapeParam {
    Width,
    Height,
    CornerRadius,
    InnerRatio,
    Fill,
    StrokeColor,
    StrokeWidth,
}

/// Animatable visual-style properties of a generated text clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireTextParam {
    Size,
    Fill,
    LetterSpacing,
    LineSpacing,
    StrokeWidth,
    StrokeColor,
    ShadowBlur,
    ShadowDistance,
    ShadowColor,
}

/// Animatable scalar properties of a clip's color look.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireLookParam {
    FilterIntensity,
    LutIntensity,
    AdjustBrightness,
    AdjustContrast,
    AdjustSaturation,
    AdjustExposure,
    AdjustTemperature,
    MaskFeather,
    ChromaStrength,
    ChromaShadow,
}

/// Animatable properties of a clip's layer-quad styles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireStyleParam {
    ShadowColor,
    ShadowOffset,
    ShadowBlur,
    GlowColor,
    GlowRadius,
    GlowIntensity,
    OutlineColor,
    OutlineWidth,
    BackgroundColor,
    BackgroundPadding,
    BackgroundRadius,
}

/// Interpolation toward the next keyframe.
///
/// Named presets (`snappy` / `overshoot` / `anticipate`) encode as cubic
/// beziers in the engine. `hold` is step interpolation — the value stays at
/// this keyframe until the next one. `bezier` accepts raw CSS-style control
/// points `(x1, y1, x2, y2)` with `x` in `0..=1` (y may overshoot).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireEasing {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    Snappy,
    Overshoot,
    Anticipate,
    Hold,
    Bezier {
        /// Control points `[x1, y1, x2, y2]`.
        points: [f32; 4],
    },
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
    /// New value for scalar parameters. Ignored for position and color
    /// parameters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    /// New `[x, y]` for `position` or `anchor_point` (fractions of canvas size
    /// from center, +x right, +y down). Ignored for scalar and color params.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<[f64; 2]>,
    /// New `[red, green, blue, alpha]` color for shape or text color
    /// parameters. Ignored for scalar and vec2 params.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rgba: Option<[u8; 4]>,
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
    /// New value for scalar parameters. Ignored for position and color
    /// parameters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    /// New `[x, y]` for `position` or `anchor_point`. Ignored for scalar and
    /// color params.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<[f64; 2]>,
    /// New `[red, green, blue, alpha]` color for shape or text color
    /// parameters. Ignored for scalar and vec2 params.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rgba: Option<[u8; 4]>,
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

/// How a visual clip composites over the stack below (CapCut "Blend").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireBlendMode {
    Normal,
    Darken,
    Multiply,
    ColorBurn,
    Lighten,
    Screen,
    ColorDodge,
    Add,
    Overlay,
    SoftLight,
    HardLight,
    Difference,
    Exclusion,
}

/// Set how a clip composites over the stack below (CapCut "Blend").
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SetClipBlendMode {
    /// Target clip id.
    pub clip: u64,
    /// Blend mode id: normal, darken, multiply, color_burn, lighten,
    /// screen, color_dodge, add, overlay, soft_light, hard_light,
    /// difference, exclusion.
    pub mode: WireBlendMode,
}

/// Drop shadow drawn from the layer's alpha. Lengths are reference pixels
/// (1080p baseline); color is RGBA 0–255.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WireLayerShadow {
    pub rgba: [u8; 4],
    /// Offset in reference pixels (`+x` right, `+y` down).
    pub offset: [f32; 2],
    /// Blur radius in reference pixels (`>= 0`).
    pub blur: f32,
}

/// Soft glow bloom drawn from the layer's alpha.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WireLayerGlow {
    pub rgba: [u8; 4],
    /// Glow radius in reference pixels (`>= 0`).
    pub radius: f32,
    /// Strength multiplier, `0` … `4`.
    pub intensity: f32,
}

/// Hard outline / stroke around the layer's alpha silhouette.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WireLayerOutline {
    pub rgba: [u8; 4],
    /// Outline width in reference pixels (`>= 0`).
    pub width: f32,
}

/// Solid plate behind the layer (padded AABB of the alpha bounds).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WireLayerBackground {
    pub rgba: [u8; 4],
    /// Padding around the alpha bounds in reference pixels (`>= 0`).
    pub padding: f32,
    /// Corner radius in reference pixels (`>= 0`).
    pub radius: f32,
}

/// Layer-quad styles (CapCut shadow/glow/outline/background) for any visual
/// clip — distinct from text glyph treatments. Lengths are reference pixels
/// (1080p baseline). Omitted blocks are removed; included blocks replace the
/// previous block with the given constant values. Animate individual fields
/// afterward with `set_param_keyframe` and style params.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WireLayerStyles {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow: Option<WireLayerShadow>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub glow: Option<WireLayerGlow>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outline: Option<WireLayerOutline>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background: Option<WireLayerBackground>,
}

/// Replace a visual clip's layer styles (shadow/glow/outline/background).
/// Pass an empty `styles` object to clear every block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SetClipLayerStyles {
    /// Target clip id.
    pub clip: u64,
    /// Style blocks to install. Omitted blocks are removed.
    pub styles: WireLayerStyles,
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
    /// Playback rate of the entrance/exit window or combo period (`1` = catalog).
    /// Omit or `null` for the default. Range `0.25..=4`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
    /// Magnitude of motion / opacity swing (`1` = catalog, `0` = none).
    /// Omit or `null` for the default. Range `0..=2`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intensity: Option<f32>,
    /// Per-character stagger stretch (`1` = catalog). Text presets only.
    /// Omit or `null` for the default. Range `0..=2`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stagger: Option<f32>,
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
