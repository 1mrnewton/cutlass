use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Mask shape kinds (CapCut mask presets).
///
/// `mirror` is a parallel band (thickness = `size[0]` layer-width fraction)
/// centered on the mask line at `rotation`, not a half-plane. Feather softens
/// both band edges symmetrically.
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

/// Shaped alpha mask on a media-backed visual clip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct WireMask {
    pub kind: WireMaskKind,
    /// Edge softness 0 (hard) … 1 (fully feathered). Default 0.
    /// For `mirror`, feather applies symmetrically to both band edges.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feather: Option<f64>,
    /// Keep outside instead of inside.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invert: Option<bool>,
    /// Center offset from layer center as layer-size fractions
    /// (`[0,0]` centered, `[0.5,0]` right edge). Default `[0,0]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub center: Option<[f32; 2]>,
    /// Size as layer-size fractions (`[1,1]` covers layer). For `mirror`,
    /// `size[0]` is band thickness (width fraction; model default `0.5` when
    /// omitted). Default `[1,1]` for other kinds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<[f32; 2]>,
    /// Rotation degrees CW about mask center. Default 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotation: Option<f32>,
    /// Rectangle corner roundness 0…1 (other kinds ignore). Default 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roundness: Option<f32>,
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

/// Transform motion blur (not animatable). Omitted fields keep current values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SetMotionBlur {
    pub clip: u64,
    /// false skips supersampling entirely.
    pub enabled: bool,
    /// Shutter angle degrees (`0..=720`; 360 = full frame; 0 disables).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shutter_deg: Option<f32>,
    /// Sub-frame samples (`2..=32`; render clamps to 16).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub samples: Option<u32>,
}

/// Drop shadow from layer alpha. Lengths = reference px (1080p); rgba 0–255.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WireLayerShadow {
    pub rgba: [u8; 4],
    /// Offset in reference px (`+x` right, `+y` down).
    pub offset: [f32; 2],
    /// Blur radius in reference px (`>= 0`).
    pub blur: f32,
}

/// Soft glow from layer alpha. Lengths = reference px (1080p); rgba 0–255.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WireLayerGlow {
    pub rgba: [u8; 4],
    /// Radius in reference px (`>= 0`).
    pub radius: f32,
    /// Strength `0`…`4`.
    pub intensity: f32,
}

/// Hard outline around layer alpha. Width = reference px (1080p); rgba 0–255.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WireLayerOutline {
    pub rgba: [u8; 4],
    /// Width in reference px (`>= 0`).
    pub width: f32,
}

/// Solid plate behind layer alpha AABB. Lengths = reference px (1080p).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WireLayerBackground {
    pub rgba: [u8; 4],
    /// Padding in reference px (`>= 0`).
    pub padding: f32,
    /// Corner radius in reference px (`>= 0`).
    pub radius: f32,
}

/// Layer-quad styles (shadow/glow/outline/background). Lengths = reference px
/// (1080p). Omitted blocks are removed.
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

/// Replace layer styles; empty `styles` clears every block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SetClipLayerStyles {
    pub clip: u64,
    /// Style blocks to install; omitted blocks are removed.
    pub styles: WireLayerStyles,
}

/// Manual color adjust. Signed sliders `-1..=1`; sharpness/vignette `0..=1`.
/// Omitted sliders keep their value.
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
    /// Green (−) ↔ magenta (+), `-1..=1`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tint: Option<f64>,
    /// Hue rotation; ±1 → ±30°.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hue: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub highlights: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadows: Option<f64>,
    /// Unsharp-mask strength `0..=1`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sharpness: Option<f64>,
    /// Radial darkening `0..=1`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vignette: Option<f64>,
}

/// Animation slot (CapCut In / Out / Combo tabs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WireAnimationSlot {
    In,
    Out,
    Combo,
}

/// Set or clear a look animation preset on a visual clip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SetClipAnimation {
    pub clip: u64,
    pub slot: WireAnimationSlot,
    /// Catalog id (e.g. fade_in, pulse). `null` clears the slot.
    pub animation: Option<String>,
    /// Playback rate `0.25..=4` (`1` = catalog). Omit for default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
    /// Motion/opacity magnitude `0..=2` (`1` = catalog). Omit for default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intensity: Option<f32>,
    /// Per-character stagger `0..=2` (text presets only). Omit for default.
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
