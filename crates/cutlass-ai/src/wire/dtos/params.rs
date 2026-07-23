use std::borrow::Cow;

use schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};
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

/// Change a clip's canvas placement. Omitted fields keep their value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetClipTransform {
    pub clip: u64,
    /// Anchor offset from canvas center, canvas-width fraction (+x right);
    /// 0 = centered, 0.5 = right edge.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position_x: Option<f64>,
    /// Anchor offset from canvas center, canvas-height fraction (+y down);
    /// 0 = centered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position_y: Option<f64>,
    /// Pivot in content bounds (0 = left, 0.5 = content center).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor_x: Option<f64>,
    /// Pivot in content bounds (0 = top, 0.5 = content center).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor_y: Option<f64>,
    /// Uniform number (1.0 = fit) or `[x, y]` per-axis.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<WireScale>,
    /// Clockwise degrees about the anchor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotation: Option<f64>,
    /// Layer opacity 0.0–1.0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opacity: Option<f64>,
}

/// Crop (edge trim fractions 0–1) and/or flip. Kept region still aspect-fits;
/// crop does not move the layer. Omitted fields keep their value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetClipCrop {
    pub clip: u64,
    /// Fraction trimmed from the left edge (0 restores).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub left: Option<f64>,
    /// Fraction trimmed from the top edge (0–1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top: Option<f64>,
    /// Fraction trimmed from the right edge (0–1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub right: Option<f64>,
    /// Fraction trimmed from the bottom edge (0–1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bottom: Option<f64>,
    /// Mirror left-right.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flip_h: Option<bool>,
    /// Mirror top-bottom.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flip_v: Option<bool>,
}

/// Append a visual effect to a clip's chain (catalog id, e.g. gaussian_blur).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AddEffect {
    pub clip: u64,
    /// Effect catalog id (e.g. `gaussian_blur`, `vignette`).
    pub effect: String,
}

/// Remove an effect by chain index (0 = first).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoveEffect {
    pub clip: u64,
    /// Effect chain index (0 = first).
    pub index: u32,
}

/// Reorder an effect; both indices address the pre-move chain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MoveEffect {
    pub clip: u64,
    /// Current index (0 = first).
    pub from_index: u32,
    /// Final index after the move (0 = first).
    pub to_index: u32,
}

/// Set one effect parameter (catalog-range-checked).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetEffectParam {
    pub clip: u64,
    /// Effect chain index (0 = first).
    pub index: u32,
    /// Param name (e.g. `radius`, `shadow_color`).
    pub param: String,
    /// Scalar value (ignored for color/vec2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    /// `[x, y]` for vec2 effect params.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<[f64; 2]>,
    /// `[r, g, b, a]` 0–255 for color effect params.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rgba: Option<[u8; 4]>,
}

/// Add a transition where `clip` abuts the next clip on its track.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AddTransition {
    pub clip: u64,
    /// Transition catalog id (e.g. `crossfade`, `dip_to_black`).
    pub transition: String,
}

/// Remove the transition at `clip`'s right cut.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RemoveTransition {
    pub clip: u64,
}

/// Set transition duration (seconds, positive); window centered on the cut.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetTransition {
    pub clip: u64,
    /// Window length in seconds (must be positive).
    pub seconds: f64,
}

/// Unit (string) variants of [`WireClipParam`] — serde `rename_all = "snake_case"`.
pub(crate) const WIRE_CLIP_PARAM_UNIT_TOKENS: &[&str] = &[
    "position",
    "anchor_point",
    "scale",
    "rotation",
    "opacity",
    "crop",
    "volume",
    "pan",
    "speed",
];

/// Snake_case wire tokens for [`WireShapeParam`].
pub(crate) const WIRE_SHAPE_PARAM_TOKENS: &[&str] = &[
    "width",
    "height",
    "corner_radius",
    "inner_ratio",
    "fill",
    "stroke_color",
    "stroke_width",
];

/// Snake_case wire tokens for [`WireTextParam`].
pub(crate) const WIRE_TEXT_PARAM_TOKENS: &[&str] = &[
    "size",
    "fill",
    "letter_spacing",
    "line_spacing",
    "stroke_width",
    "stroke_color",
    "shadow_blur",
    "shadow_distance",
    "shadow_color",
    "background_color",
    "background_radius",
];

/// Snake_case wire tokens for [`WireLookParam`].
pub(crate) const WIRE_LOOK_PARAM_TOKENS: &[&str] = &[
    "filter_intensity",
    "lut_intensity",
    "adjust_brightness",
    "adjust_contrast",
    "adjust_saturation",
    "adjust_exposure",
    "adjust_temperature",
    "adjust_tint",
    "adjust_hue",
    "adjust_highlights",
    "adjust_shadows",
    "adjust_sharpness",
    "adjust_vignette",
    "mask_feather",
    "mask_center",
    "mask_size",
    "mask_rotation",
    "mask_roundness",
    "chroma_strength",
    "chroma_shadow",
];

/// Snake_case wire tokens for [`WireStyleParam`].
pub(crate) const WIRE_STYLE_PARAM_TOKENS: &[&str] = &[
    "shadow_color",
    "shadow_offset",
    "shadow_blur",
    "glow_color",
    "glow_radius",
    "glow_intensity",
    "outline_color",
    "outline_width",
    "background_color",
    "background_padding",
    "background_radius",
];

/// An animatable clip property the keyframe commands can address.
///
/// Wire serde stays externally tagged (`"position"` / `{"effect":{…}}`).
/// JSON Schema is hand-written so the four keyframe tools do not each
/// re-inline a long `oneOf` of per-variant descriptions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WireClipParam {
    Position,
    AnchorPoint,
    Scale,
    Rotation,
    Opacity,
    Crop,
    Volume,
    Pan,
    Speed,
    Effect { index: u32, param: String },
    Shape { param: WireShapeParam },
    Text { param: WireTextParam },
    Look { param: WireLookParam },
    Style { param: WireStyleParam },
}

fn schema_string_enum(tokens: &[&str]) -> serde_json::Value {
    serde_json::json!({
        "type": "string",
        "enum": tokens,
    })
}

/// Externally-tagged `{"tag":{"param":…}}` object branch with closed shapes.
fn schema_tagged_param_branch(tag: &str, tokens: &[&str]) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": [tag],
        "properties": {
            tag: {
                "type": "object",
                "additionalProperties": false,
                "required": ["param"],
                "properties": {
                    "param": schema_string_enum(tokens),
                }
            }
        }
    })
}

impl JsonSchema for WireClipParam {
    fn inline_schema() -> bool {
        true
    }

    fn schema_name() -> Cow<'static, str> {
        "WireClipParam".into()
    }

    fn json_schema(_gen: &mut SchemaGenerator) -> Schema {
        // Compact shared description; nested param names are exact enums in
        // each tagged branch (same tokens serde accepts).
        let schema = serde_json::json!({
            "description": "Clip property. Strings: position|anchor_point|scale|rotation|opacity|crop|volume|pan|speed. Tagged: {\"effect\":{\"index\":N,\"param\":effectParamName}}, {\"shape|text|look|style\":{\"param\":enum}}. Args: position=[x,y] = anchor offset from canvas center in canvas fractions (vec2 also: anchor_point, per-axis scale, mask_center/size, shadow_offset, vec2 effects); value=scalars (rot° CW about anchor, opacity 0..1, scale 1.0=fit, volume 0..10, pan −1..+1); rgba=[r,g,b,a] 0–255; rect=[x,y,w,h] content fractions (crop). speed is not keyframable here — use set_clip_speed / set_speed_curve. volume/pan=media; crop=visual. Keyframe `at` = absolute timeline seconds inside the clip.",
            "oneOf": [
                schema_string_enum(WIRE_CLIP_PARAM_UNIT_TOKENS),
                {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["effect"],
                    "properties": {
                        "effect": {
                            "type": "object",
                            "additionalProperties": false,
                            "required": ["index", "param"],
                            "properties": {
                                "index": { "type": "integer", "minimum": 0 },
                                "param": { "type": "string" }
                            }
                        }
                    }
                },
                schema_tagged_param_branch("shape", WIRE_SHAPE_PARAM_TOKENS),
                schema_tagged_param_branch("text", WIRE_TEXT_PARAM_TOKENS),
                schema_tagged_param_branch("look", WIRE_LOOK_PARAM_TOKENS),
                schema_tagged_param_branch("style", WIRE_STYLE_PARAM_TOKENS),
            ]
        });
        Schema::try_from(schema).expect("WireClipParam schema is plain JSON data")
    }
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

/// Animatable look/mask params (`mask_center`/`mask_size` = vec2; else scalar).
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
/// Named presets encode as cubic beziers; `hold` is step; `bezier` takes
/// CSS-style `[x1,y1,x2,y2]` with `x` in `0..=1` (y may overshoot).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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

impl JsonSchema for WireEasing {
    fn inline_schema() -> bool {
        true
    }

    fn schema_name() -> Cow<'static, str> {
        "WireEasing".into()
    }

    fn json_schema(_gen: &mut SchemaGenerator) -> Schema {
        json_schema!({
            "description": "linear|ease_*|snappy|overshoot|anticipate|hold|{bezier:{points:[x1,y1,x2,y2]}}",
            "oneOf": [
                {
                    "type": "string",
                    "enum": [
                        "linear",
                        "ease_in",
                        "ease_out",
                        "ease_in_out",
                        "snappy",
                        "overshoot",
                        "anticipate",
                        "hold"
                    ]
                },
                {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["bezier"],
                    "properties": {
                        "bezier": {
                            "type": "object",
                            "additionalProperties": false,
                            "required": ["points"],
                            "properties": {
                                "points": {
                                    "type": "array",
                                    "items": { "type": "number" },
                                    "minItems": 4,
                                    "maxItems": 4
                                }
                            }
                        }
                    }
                }
            ]
        })
    }
}

/// Add or replace a keyframe on one animatable clip property.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(description = "")]
pub struct SetParamKeyframe {
    pub clip: u64,
    pub param: WireClipParam,
    /// Absolute timeline seconds (must lie inside the clip).
    pub at: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    /// Vec2 `[x,y]`. For position: anchor offset from canvas center
    /// (canvas fractions; [0,0]=centered). Also per-axis scale / other vec2s.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<[f64; 2]>,
    /// `[r,g,b,a]` 0–255.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rgba: Option<[u8; 4]>,
    /// Crop `[x,y,w,h]` content fractions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rect: Option<[f64; 4]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub easing: Option<WireEasing>,
    /// Position motion-path out-handle (canvas-fraction offset from value).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tangent_out: Option<[f64; 2]>,
    /// Position motion-path in-handle (canvas-fraction offset from value).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tangent_in: Option<[f64; 2]>,
}

/// Remove the keyframe at a timeline position on one property.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(description = "")]
pub struct RemoveParamKeyframe {
    pub clip: u64,
    pub param: WireClipParam,
    /// Timeline seconds of the keyframe to remove.
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

/// Expand the outgoing keyframe segment into a bounce/elastic/back curve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(description = "")]
pub struct ApplyEasingPreset {
    pub clip: u64,
    pub param: WireClipParam,
    /// Timeline seconds of the departing keyframe (needs a following KF).
    pub from_tick: f64,
    pub preset: WireEasingPreset,
}

/// Set one animatable property to a fixed value and clear its keyframes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(description = "")]
pub struct SetParamConstant {
    pub clip: u64,
    pub param: WireClipParam,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    /// Vec2 `[x,y]`. For position: anchor offset from canvas center
    /// (canvas fractions; [0,0]=centered).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<[f64; 2]>,
    /// `[r,g,b,a]` 0–255.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rgba: Option<[u8; 4]>,
    /// Crop `[x,y,w,h]` content fractions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rect: Option<[f64; 4]>,
}

/// Change a media clip's constant playback speed and/or direction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetClipSpeed {
    pub clip: u64,
    /// Playback rate: 2.0 = double speed, 0.5 = slow-mo; range 0.05..100.
    /// Timeline length re-derives; audio time-stretches. Omit to keep.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
    /// Play footage backwards. Omit to keep.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reversed: Option<bool>,
}

/// Apply or clear a CapCut-style speed ramp on a media clip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetSpeedCurve {
    pub clip: u64,
    /// Preset: ramp_up, ramp_down, montage, hero, bullet. Null clears to
    /// constant speed. Length re-derives from average speed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset: Option<String>,
}

/// Lock or unlock pitch on a retimed media clip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetClipPitch {
    pub clip: u64,
    /// true = keep pitch while time-stretching; false = pitch follows speed.
    pub preserve_pitch: bool,
}

/// Set a clip's volume gain and/or fade-in/out (seconds).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetClipAudio {
    /// Video clip with sound, or an audio clip.
    pub clip: u64,
    /// Gain: 0.0 mute, 1.0 unchanged, max 10. Omit to keep.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub volume: Option<f64>,
    /// Fade-in seconds from clip start (0 clears). Omit to keep.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fade_in: Option<f64>,
    /// Fade-out seconds ending at clip end (0 clears). Omit to keep.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fade_out: Option<f64>,
}

/// Enable/disable speech-preserving noise reduction on a media clip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetDenoise {
    pub clip: u64,
    pub denoise: bool,
}
