use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Per-axis clip scale on the wire: a bare number (uniform, legacy agents)
/// or `[x, y]` when split. Schema is `anyOf` number/array via untagged serde.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum WireScale {
    /// Uniform scale on both axes (1.0 = aspect-fit / CapCut 100%).
    Uniform(f64),
    /// Per-axis scale `[x, y]`.
    Axes([f64; 2]),
}

impl WireScale {
    pub fn to_scale2(self) -> cutlass_models::Scale2 {
        match self {
            WireScale::Uniform(s) => cutlass_models::Scale2::uniform(s as f32),
            WireScale::Axes([x, y]) => cutlass_models::Scale2 {
                x: x as f32,
                y: y as f32,
            },
        }
    }
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
    /// Scale: a number for uniform (1.0 = fit inside the canvas) or `[x, y]`
    /// for per-axis stretch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<WireScale>,
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
/// value is range-checked against the catalog. Use `value` for scalars,
/// `position` for vec2 params (e.g. color_overlay `offset`), and `rgba` for
/// color params (e.g. duotone `shadow_color`). Use `set_param_keyframe`
/// with an effect param to animate it instead.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetEffectParam {
    pub clip: u64,
    /// Index of the effect in the clip's chain (0 = first).
    pub index: u32,
    /// Parameter name, e.g. `radius` (gaussian_blur) or `shadow_color` (duotone).
    pub param: String,
    /// New scalar value. Required for scalar params; ignored for color/vec2.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    /// New `[x, y]` for vec2 effect params. Ignored for scalar and color.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<[f64; 2]>,
    /// New `[red, green, blue, alpha]` for color effect params. Ignored for
    /// scalar and vec2.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rgba: Option<[u8; 4]>,
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
    /// Scale (1.0 = fit inside the canvas). Use `value` for a uniform number
    /// or `position` as `[x, y]` for per-axis.
    Scale,
    /// Clockwise rotation in degrees.
    Rotation,
    /// Layer opacity 0.0–1.0.
    Opacity,
    /// Kept-region crop window as content fractions (rect: use the `rect`
    /// argument `[x, y, w, h]`, not `value`). Visual clips only.
    Crop,
    /// Audio gain envelope (0.0 = mute, 1.0 = unchanged, up to 10.0 boost).
    /// Keyframing it draws volume automation; this is how you fade audio in
    /// or out over time or duck music under a voice. Media-backed clips only.
    Volume,
    /// Stereo pan envelope (−1.0 = full left, 0.0 = center, +1.0 = full
    /// right). Keyframing it draws balance automation. Media-backed clips
    /// only (including video clips with sound — same target rule as volume).
    Pan,
    /// Playback-rate ramp (scalar: 1.0 = normal speed). Media-backed clips
    /// only; keyframes use normalized positions within the clip.
    Speed,
    /// A parameter of an effect already on the clip. `index` is the
    /// effect-chain position and `param` is its catalog name, as in
    /// `set_effect_param`. Use `value` for scalars, `position` for vec2
    /// params, and `rgba` for color params.
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
    BackgroundColor,
    BackgroundRadius,
}

/// Animatable properties of a clip's color look / mask.
///
/// `mask_center` / `mask_size` carry vec2 values; the rest are scalars.
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
    AdjustTint,
    AdjustHue,
    AdjustHighlights,
    AdjustShadows,
    AdjustSharpness,
    AdjustVignette,
    MaskFeather,
    MaskCenter,
    MaskSize,
    MaskRotation,
    MaskRoundness,
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
    /// New value for scalar parameters. Ignored for position/vec2, color,
    /// and rect parameters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    /// New `[x, y]` for `position`, `anchor_point`, style `shadow_offset`,
    /// look `mask_center` / `mask_size`, or vec2 effect params. Ignored for
    /// scalar, color, and rect params.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<[f64; 2]>,
    /// New `[red, green, blue, alpha]` for shape, text, style, or effect
    /// color parameters. Ignored for scalar, vec2, and rect params.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rgba: Option<[u8; 4]>,
    /// New `[x, y, w, h]` kept-region for `crop` (content fractions).
    /// Ignored for scalar, vec2, and color params.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rect: Option<[f64; 4]>,
    /// Interpolation toward the next keyframe. Defaults to linear.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub easing: Option<WireEasing>,
    /// Outgoing spatial bezier handle for a **position** motion path
    /// (After Effects–style). Offset from this keyframe's value, in canvas
    /// fractions. Ignored / rejected on non-position params. Pair with
    /// `tangent_in` on the next keyframe to shape the segment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tangent_out: Option<[f64; 2]>,
    /// Incoming spatial bezier handle for a **position** motion path.
    /// Offset from this keyframe's value, in canvas fractions. Position
    /// only; see `tangent_out`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tangent_in: Option<[f64; 2]>,
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

/// Multi-keyframe easing preset applied to the outgoing segment at `from_tick`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireEasingPreset {
    BounceOut,
    ElasticOut,
    BackOut,
}

/// Expand the keyframe segment leaving `from_tick` into a bounce / elastic /
/// back approximation (multiple keyframes). Scalar and vec2 params only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ApplyEasingPreset {
    pub clip: u64,
    pub param: WireClipParam,
    /// Timeline position of the departing keyframe, in seconds. Must have a
    /// following keyframe on the same param.
    pub from_tick: f64,
    pub preset: WireEasingPreset,
}

/// Set one animatable property to a fixed value, removing all its
/// keyframes (stops the animation).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetParamConstant {
    pub clip: u64,
    pub param: WireClipParam,
    /// New value for scalar parameters. Ignored for position/vec2, color,
    /// and rect parameters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    /// New `[x, y]` for `position`, `anchor_point`, style `shadow_offset`,
    /// look `mask_center` / `mask_size`, or vec2 effect params. Ignored for
    /// scalar, color, and rect params.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<[f64; 2]>,
    /// New `[red, green, blue, alpha]` for shape, text, style, or effect
    /// color parameters. Ignored for scalar, vec2, and rect params.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rgba: Option<[u8; 4]>,
    /// New `[x, y, w, h]` kept-region for `crop` (content fractions).
    /// Ignored for scalar, vec2, and color params.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rect: Option<[f64; 4]>,
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
