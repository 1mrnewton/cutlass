//! Clip "look" extensions (mobile-support Phase I): mask, chroma key,
//! stabilization, filter presets, color adjustments, entrance/exit
//! animations, and the audio role tag.
//!
//! These persist and validate like every other clip property. Color
//! adjustments and filter presets are composited per-clip (see
//! `cutlass-render` / `cutlass-compositor`); mask and chroma key are
//! composited per-clip; look animations drive transform/opacity at
//! resolve time; stabilization remains render-neutral this milestone.
//!
//! The catalogs here follow the effect-catalog pattern: they are the
//! validation *and* UI source of truth (stable ids, display labels), so the
//! shells never hard-code preset lists.

use serde::{Deserialize, Serialize};

use crate::error::ModelError;
use crate::param::Param;

// --- Mask -------------------------------------------------------------------

/// Mask shapes (CapCut mask panel). Serialized by snake_case id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskKind {
    Linear,
    Mirror,
    Circle,
    Rectangle,
    Heart,
    Star,
}

impl MaskKind {
    /// Stable wire/catalog id (the serde name).
    pub const fn id(self) -> &'static str {
        match self {
            MaskKind::Linear => "linear",
            MaskKind::Mirror => "mirror",
            MaskKind::Circle => "circle",
            MaskKind::Rectangle => "rectangle",
            MaskKind::Heart => "heart",
            MaskKind::Star => "star",
        }
    }
}

/// A shaped alpha mask over a clip's content. `None` on the clip ⇔ no mask.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mask {
    pub kind: MaskKind,
    /// Edge softness, `0` (hard) … `1` (fully feathered).
    #[serde(default = "default_zero_param", skip_serializing_if = "is_zero_param")]
    pub feather: Param<f32>,
    /// Keep the outside instead of the inside.
    #[serde(default, skip_serializing_if = "is_false")]
    pub invert: bool,
}

impl Mask {
    /// A hard, non-inverted mask of `kind`.
    pub fn new(kind: MaskKind) -> Self {
        Self {
            kind,
            feather: Param::Constant(0.0),
            invert: false,
        }
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        validate_unit_param("mask feather", &self.feather)
    }
}

/// One mask catalog entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskSpec {
    pub kind: MaskKind,
    pub label: &'static str,
}

const MASKS: &[MaskSpec] = &[
    MaskSpec {
        kind: MaskKind::Linear,
        label: "Linear",
    },
    MaskSpec {
        kind: MaskKind::Mirror,
        label: "Mirror",
    },
    MaskSpec {
        kind: MaskKind::Circle,
        label: "Circle",
    },
    MaskSpec {
        kind: MaskKind::Rectangle,
        label: "Rectangle",
    },
    MaskSpec {
        kind: MaskKind::Heart,
        label: "Heart",
    },
    MaskSpec {
        kind: MaskKind::Star,
        label: "Star",
    },
];

/// Every mask shape (UI browsing order).
pub fn mask_catalog() -> &'static [MaskSpec] {
    MASKS
}

// --- Chroma key ---------------------------------------------------------------

/// Green-screen keying (CapCut chroma key): pixels near `rgb` turn
/// transparent. `None` on the clip ⇔ keying off.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChromaKey {
    /// Key color, opaque `[r, g, b]`.
    pub rgb: [u8; 3],
    /// Keying strength (tolerance), `0` … `1`.
    #[serde(default = "default_zero_param", skip_serializing_if = "is_zero_param")]
    pub strength: Param<f32>,
    /// Shadow retention, `0` … `1`.
    #[serde(default = "default_zero_param", skip_serializing_if = "is_zero_param")]
    pub shadow: Param<f32>,
}

impl ChromaKey {
    pub fn validate(&self) -> Result<(), ModelError> {
        validate_unit_param("chroma strength", &self.strength)?;
        validate_unit_param("chroma shadow", &self.shadow)
    }
}

// --- Stabilization ------------------------------------------------------------

/// Stabilization strength (CapCut stabilize panel). `None` on the clip ⇔ off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StabilizeLevel {
    Recommended,
    Smooth,
    MaxSmooth,
}

impl StabilizeLevel {
    /// Stable wire/catalog id (the serde name).
    pub const fn id(self) -> &'static str {
        match self {
            StabilizeLevel::Recommended => "recommended",
            StabilizeLevel::Smooth => "smooth",
            StabilizeLevel::MaxSmooth => "max_smooth",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            StabilizeLevel::Recommended => "Recommended",
            StabilizeLevel::Smooth => "Smooth",
            StabilizeLevel::MaxSmooth => "Max smooth",
        }
    }

    /// Every level (UI browsing order).
    pub const ALL: [StabilizeLevel; 3] = [
        StabilizeLevel::Recommended,
        StabilizeLevel::Smooth,
        StabilizeLevel::MaxSmooth,
    ];
}

// --- Filter presets -------------------------------------------------------------

/// A color-grade filter applied to a clip (CapCut filters). `None` on the
/// clip ⇔ no filter. Also the payload persisted on `Generator::Filter` lane
/// bars, which grade everything beneath them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Filter {
    /// Catalog id (see [`filter_catalog`]).
    pub id: String,
    /// Blend of the graded result over the original, `0` … `1`.
    #[serde(
        default = "default_filter_intensity_param",
        skip_serializing_if = "is_default_filter_intensity"
    )]
    pub intensity: Param<f32>,
}

impl Filter {
    /// A filter at the default intensity.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            intensity: Param::Constant(default_filter_intensity()),
        }
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        if filter_spec(&self.id).is_none() {
            return Err(ModelError::InvalidParam(format!(
                "unknown filter '{}'",
                self.id
            )));
        }
        validate_unit_param("filter intensity", &self.intensity)
    }
}

fn default_filter_intensity() -> f32 {
    0.8
}

fn default_filter_intensity_param() -> Param<f32> {
    Param::Constant(default_filter_intensity())
}

fn is_default_filter_intensity(v: &Param<f32>) -> bool {
    v.constant() == Some(default_filter_intensity())
}

/// One filter catalog entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilterSpec {
    pub id: &'static str,
    pub label: &'static str,
}

const FILTERS: &[FilterSpec] = &[
    FilterSpec {
        id: "vivid",
        label: "Vivid",
    },
    FilterSpec {
        id: "warm",
        label: "Warm",
    },
    FilterSpec {
        id: "cool",
        label: "Cool",
    },
    FilterSpec {
        id: "mono",
        label: "Mono",
    },
    FilterSpec {
        id: "fade",
        label: "Fade",
    },
    FilterSpec {
        id: "chrome",
        label: "Chrome",
    },
    FilterSpec {
        id: "noir",
        label: "Noir",
    },
    FilterSpec {
        id: "sunset",
        label: "Sunset",
    },
    FilterSpec {
        id: "forest",
        label: "Forest",
    },
    FilterSpec {
        id: "berry",
        label: "Berry",
    },
];

/// Every filter preset (UI browsing order).
pub fn filter_catalog() -> &'static [FilterSpec] {
    FILTERS
}

/// The catalog entry for `id`, or `None`.
pub fn filter_spec(id: &str) -> Option<&'static FilterSpec> {
    FILTERS.iter().find(|s| s.id == id)
}

// --- 3D LUTs ----------------------------------------------------------------

/// A `.cube` 3D LUT applied to a clip after its filter/adjust grade. `None`
/// on the clip ⇔ no LUT. File-backed like Lottie animations: `path` points at
/// a `.cube` file on disk (downloaded from the asset catalog or supplied by
/// the user); the renderer parses and uploads it lazily and skips missing
/// files gracefully. Also valid on `Generator::Filter` lane bars, which apply
/// the LUT to everything composited beneath them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lut {
    /// Absolute path to the `.cube` file.
    pub path: String,
    /// Blend of the looked-up result over the original, `0` … `1`.
    #[serde(
        default = "default_filter_intensity_param",
        skip_serializing_if = "is_default_filter_intensity"
    )]
    pub intensity: Param<f32>,
}

impl Lut {
    /// A LUT at the default intensity.
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            intensity: Param::Constant(default_filter_intensity()),
        }
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        if self.path.trim().is_empty() {
            return Err(ModelError::InvalidParam("empty LUT path".into()));
        }
        validate_unit_param("LUT intensity", &self.intensity)
    }
}

// --- Color adjustments -----------------------------------------------------------

/// Manual color grade (CapCut adjust panel): signed strengths, `0` neutral.
/// Lives on visual clips and on `Generator::Adjustment` lane bars.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColorAdjustments {
    #[serde(default = "default_zero_param", skip_serializing_if = "is_zero_param")]
    pub brightness: Param<f32>,
    #[serde(default = "default_zero_param", skip_serializing_if = "is_zero_param")]
    pub contrast: Param<f32>,
    #[serde(default = "default_zero_param", skip_serializing_if = "is_zero_param")]
    pub saturation: Param<f32>,
    #[serde(default = "default_zero_param", skip_serializing_if = "is_zero_param")]
    pub exposure: Param<f32>,
    #[serde(default = "default_zero_param", skip_serializing_if = "is_zero_param")]
    pub temperature: Param<f32>,
}

impl ColorAdjustments {
    /// True iff every slider sits at neutral — the serde skip predicate.
    pub fn is_neutral(&self) -> bool {
        [
            &self.brightness,
            &self.contrast,
            &self.saturation,
            &self.exposure,
            &self.temperature,
        ]
        .iter()
        .all(|param| is_zero_param(param))
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        for (name, value) in [
            ("brightness", &self.brightness),
            ("contrast", &self.contrast),
            ("saturation", &self.saturation),
            ("exposure", &self.exposure),
            ("temperature", &self.temperature),
        ] {
            validate_adjust_param(name, value)?;
        }
        Ok(())
    }
}

impl Default for ColorAdjustments {
    fn default() -> Self {
        Self {
            brightness: default_zero_param(),
            contrast: default_zero_param(),
            saturation: default_zero_param(),
            exposure: default_zero_param(),
            temperature: default_zero_param(),
        }
    }
}

// --- Animations -----------------------------------------------------------------

/// Which animation slot a preset occupies (CapCut In / Out / Combo tabs).
/// A combo replaces both entrance and exit; setting one side clears a combo
/// and vice versa (enforced by `Project::set_clip_animation`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnimationSlot {
    In,
    Out,
    Combo,
}

impl AnimationSlot {
    /// Stable wire/catalog id (the serde name).
    pub const fn id(self) -> &'static str {
        match self {
            AnimationSlot::In => "in",
            AnimationSlot::Out => "out",
            AnimationSlot::Combo => "combo",
        }
    }
}

/// Default value for [`AnimationRef`] tunable knobs (identity / catalog feel).
pub const ANIMATION_PARAM_DEFAULT: f32 = 1.0;
/// Inclusive range for [`AnimationRef::speed`].
pub const ANIMATION_SPEED_RANGE: (f32, f32) = (0.25, 4.0);
/// Inclusive range for [`AnimationRef::intensity`].
pub const ANIMATION_INTENSITY_RANGE: (f32, f32) = (0.0, 2.0);
/// Inclusive range for [`AnimationRef::stagger`].
pub const ANIMATION_STAGGER_RANGE: (f32, f32) = (0.0, 2.0);

fn default_anim_param() -> f32 {
    ANIMATION_PARAM_DEFAULT
}

fn is_default_anim_param(v: &f32) -> bool {
    (*v - ANIMATION_PARAM_DEFAULT).abs() < f32::EPSILON
}

/// Which user-tunable knobs a preset exposes in the inspector / AI wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnimationKnobs {
    pub speed: bool,
    pub intensity: bool,
    pub stagger: bool,
}

impl AnimationKnobs {
    /// Speed + intensity (whole-layer and most text presets).
    pub const SPEED_INTENSITY: Self = Self {
        speed: true,
        intensity: true,
        stagger: false,
    };
    /// Speed + intensity + stagger (per-character presets).
    pub const SPEED_INTENSITY_STAGGER: Self = Self {
        speed: true,
        intensity: true,
        stagger: true,
    };
}

/// A reference to a catalog animation, stored per slot on the clip.
///
/// `speed` / `intensity` / `stagger` default to [`ANIMATION_PARAM_DEFAULT`]
/// and serialize only when non-default (additive — old projects still load).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnimationRef {
    /// Catalog id (see [`animation_catalog`]).
    pub id: String,
    /// Playback rate of the entrance/exit window or combo period (`1` = catalog).
    #[serde(
        default = "default_anim_param",
        skip_serializing_if = "is_default_anim_param"
    )]
    pub speed: f32,
    /// Magnitude of motion / opacity swing (`1` = catalog, `0` = no motion).
    #[serde(
        default = "default_anim_param",
        skip_serializing_if = "is_default_anim_param"
    )]
    pub intensity: f32,
    /// Per-character stagger stretch (`1` = catalog; ignored when the preset
    /// has no stagger knob).
    #[serde(
        default = "default_anim_param",
        skip_serializing_if = "is_default_anim_param"
    )]
    pub stagger: f32,
}

impl AnimationRef {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            speed: ANIMATION_PARAM_DEFAULT,
            intensity: ANIMATION_PARAM_DEFAULT,
            stagger: ANIMATION_PARAM_DEFAULT,
        }
    }

    /// Clamp / zero unsupported knobs against `spec`, rejecting out-of-range
    /// values on knobs the preset exposes.
    pub fn normalized_for(self, spec: &AnimationSpec) -> Result<Self, ModelError> {
        let speed = normalize_knob("speed", self.speed, spec.knobs.speed, ANIMATION_SPEED_RANGE)?;
        let intensity = normalize_knob(
            "intensity",
            self.intensity,
            spec.knobs.intensity,
            ANIMATION_INTENSITY_RANGE,
        )?;
        let stagger = normalize_knob(
            "stagger",
            self.stagger,
            spec.knobs.stagger,
            ANIMATION_STAGGER_RANGE,
        )?;
        Ok(Self {
            id: self.id,
            speed,
            intensity,
            stagger,
        })
    }
}

fn normalize_knob(
    name: &str,
    value: f32,
    supported: bool,
    (lo, hi): (f32, f32),
) -> Result<f32, ModelError> {
    if !supported {
        return Ok(ANIMATION_PARAM_DEFAULT);
    }
    if !value.is_finite() || value < lo || value > hi {
        return Err(ModelError::InvalidParam(format!(
            "animation {name} {value} outside [{lo}, {hi}]"
        )));
    }
    Ok(value)
}

/// One animation catalog entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnimationSpec {
    pub id: &'static str,
    pub label: &'static str,
    pub slot: AnimationSlot,
    /// Presets designed for text clips only (the text panel's animation
    /// chips); rejected on other content.
    pub text_only: bool,
    /// Which of speed / intensity / stagger the UI and AI may expose.
    pub knobs: AnimationKnobs,
}

const fn anim(
    id: &'static str,
    label: &'static str,
    slot: AnimationSlot,
    text_only: bool,
    knobs: AnimationKnobs,
) -> AnimationSpec {
    AnimationSpec {
        id,
        label,
        slot,
        text_only,
        knobs,
    }
}

const SI: AnimationKnobs = AnimationKnobs::SPEED_INTENSITY;
const SIS: AnimationKnobs = AnimationKnobs::SPEED_INTENSITY_STAGGER;

const ANIMATIONS: &[AnimationSpec] = &[
    // Entrances.
    anim("fade_in", "Fade in", AnimationSlot::In, false, SI),
    anim("slide_up", "Slide up", AnimationSlot::In, false, SI),
    anim("zoom_in", "Zoom in", AnimationSlot::In, false, SI),
    anim("spin_in", "Spin in", AnimationSlot::In, false, SI),
    anim("bounce", "Bounce", AnimationSlot::In, false, SI),
    // Exits.
    anim("fade_out", "Fade out", AnimationSlot::Out, false, SI),
    anim("slide_down", "Slide down", AnimationSlot::Out, false, SI),
    anim("zoom_out", "Zoom out", AnimationSlot::Out, false, SI),
    anim("spin_out", "Spin out", AnimationSlot::Out, false, SI),
    anim("drop", "Drop", AnimationSlot::Out, false, SI),
    // Combos (looping presence animations).
    anim("pulse", "Pulse", AnimationSlot::Combo, false, SI),
    anim("rock", "Rock", AnimationSlot::Combo, false, SI),
    anim("swing", "Swing", AnimationSlot::Combo, false, SI),
    anim("flicker", "Flicker", AnimationSlot::Combo, false, SI),
    anim("breathe", "Breathe", AnimationSlot::Combo, false, SI),
    // Text-only per-character presets (glyph atlas path).
    anim(
        "char_typewriter",
        "Typewriter",
        AnimationSlot::In,
        true,
        SIS,
    ),
    anim("char_fade_in", "Fade in", AnimationSlot::In, true, SIS),
    anim("char_bounce_in", "Bounce in", AnimationSlot::In, true, SIS),
    anim("char_slide_in", "Slide in", AnimationSlot::In, true, SIS),
    anim("char_pop_in", "Pop in", AnimationSlot::In, true, SIS),
    anim("char_fade_out", "Fade out", AnimationSlot::Out, true, SIS),
    anim("char_fall_away", "Fall away", AnimationSlot::Out, true, SIS),
    anim(
        "char_typewriter_out",
        "Type out",
        AnimationSlot::Out,
        true,
        SIS,
    ),
    // Text-only combos (the text panel's looping chips).
    anim("typewriter", "Typewriter", AnimationSlot::Combo, true, SIS),
    anim("text_fade", "Fade", AnimationSlot::Combo, true, SIS),
    anim("text_bounce", "Bounce", AnimationSlot::Combo, true, SIS),
    anim("text_slide", "Slide", AnimationSlot::Combo, true, SIS),
    anim("pop", "Pop", AnimationSlot::Combo, true, SIS),
    anim("wave", "Wave", AnimationSlot::Combo, true, SIS),
    anim("char_jitter", "Jitter", AnimationSlot::Combo, true, SIS),
    anim("char_pulse", "Pulse", AnimationSlot::Combo, true, SIS),
];

/// Every animation preset (UI browsing order; filter by slot / text_only).
pub fn animation_catalog() -> &'static [AnimationSpec] {
    ANIMATIONS
}

/// The catalog entry for `id`, or `None`.
pub fn animation_spec(id: &str) -> Option<&'static AnimationSpec> {
    ANIMATIONS.iter().find(|s| s.id == id)
}

// --- Audio roles ------------------------------------------------------------------

/// What an audio-lane clip *is* (CapCut's music / sound-FX / voiceover /
/// extracted grouping) — drives badges and future mixing defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioRole {
    Music,
    Sfx,
    Voiceover,
    Extracted,
}

impl AudioRole {
    /// Stable wire/catalog id (the serde name).
    pub const fn id(self) -> &'static str {
        match self {
            AudioRole::Music => "music",
            AudioRole::Sfx => "sfx",
            AudioRole::Voiceover => "voiceover",
            AudioRole::Extracted => "extracted",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            AudioRole::Music => "Music",
            AudioRole::Sfx => "Sound FX",
            AudioRole::Voiceover => "Voiceover",
            AudioRole::Extracted => "Extracted",
        }
    }

    pub const ALL: [AudioRole; 4] = [
        AudioRole::Music,
        AudioRole::Sfx,
        AudioRole::Voiceover,
        AudioRole::Extracted,
    ];
}

// --- Text effect presets --------------------------------------------------------------

use crate::clip::{TextBackground, TextShadow, TextStroke};

/// A text effect preset (CapCut text effects): a named combination of the
/// stroke / shadow / background treatments [`crate::TextStyle`] already
/// persists. Applying a preset bakes these fields onto the style (see
/// [`crate::Generator::resolve_presets`]), so the file stays self-describing
/// and renderers never need the catalog.
#[derive(Debug, Clone, PartialEq)]
pub struct TextEffectSpec {
    pub id: &'static str,
    pub label: &'static str,
    pub stroke: Option<TextStroke>,
    pub shadow: Option<TextShadow>,
    pub background: Option<TextBackground>,
}

const TEXT_EFFECTS: &[TextEffectSpec] = &[
    TextEffectSpec {
        id: "neon",
        label: "Neon",
        stroke: Some(TextStroke {
            rgba: crate::Param::Constant([57, 255, 20, 255]),
            width: crate::Param::Constant(4.0),
        }),
        shadow: Some(TextShadow {
            rgba: crate::Param::Constant([57, 255, 20, 200]),
            blur: crate::Param::Constant(0.35),
            distance: crate::Param::Constant(0.0),
        }),
        background: None,
    },
    TextEffectSpec {
        id: "shadow",
        label: "Shadow",
        stroke: None,
        shadow: Some(TextShadow {
            rgba: crate::Param::Constant([0, 0, 0, 230]),
            blur: crate::Param::Constant(0.15),
            distance: crate::Param::Constant(8.0),
        }),
        background: None,
    },
    TextEffectSpec {
        id: "outline",
        label: "Outline",
        stroke: Some(TextStroke {
            rgba: crate::Param::Constant([0, 0, 0, 255]),
            width: crate::Param::Constant(8.0),
        }),
        shadow: None,
        background: None,
    },
    TextEffectSpec {
        id: "glow",
        label: "Glow",
        stroke: None,
        shadow: Some(TextShadow {
            rgba: crate::Param::Constant([255, 255, 255, 220]),
            blur: crate::Param::Constant(0.4),
            distance: crate::Param::Constant(0.0),
        }),
        background: None,
    },
    TextEffectSpec {
        id: "retro",
        label: "Retro",
        stroke: Some(TextStroke {
            rgba: crate::Param::Constant([255, 140, 60, 255]),
            width: crate::Param::Constant(5.0),
        }),
        shadow: Some(TextShadow {
            rgba: crate::Param::Constant([120, 40, 160, 255]),
            blur: crate::Param::Constant(0.05),
            distance: crate::Param::Constant(10.0),
        }),
        background: None,
    },
    TextEffectSpec {
        id: "chrome",
        label: "Chrome",
        stroke: Some(TextStroke {
            rgba: crate::Param::Constant([230, 230, 240, 255]),
            width: crate::Param::Constant(3.0),
        }),
        shadow: Some(TextShadow {
            rgba: crate::Param::Constant([40, 60, 90, 200]),
            blur: crate::Param::Constant(0.2),
            distance: crate::Param::Constant(6.0),
        }),
        background: None,
    },
];

/// Every text effect preset (UI browsing order).
pub fn text_effect_catalog() -> &'static [TextEffectSpec] {
    TEXT_EFFECTS
}

/// The catalog entry for `id`, or `None`.
pub fn text_effect_spec(id: &str) -> Option<&'static TextEffectSpec> {
    TEXT_EFFECTS.iter().find(|s| s.id == id)
}

// --- Shared validation helpers -------------------------------------------------------

fn validate_unit(what: &str, v: f32) -> Result<(), ModelError> {
    if !v.is_finite() || !(0.0..=1.0).contains(&v) {
        return Err(ModelError::InvalidParam(format!(
            "{what} = {v} out of range [0, 1]"
        )));
    }
    Ok(())
}

fn default_zero_param() -> Param<f32> {
    Param::Constant(0.0)
}

fn is_zero_param(v: &Param<f32>) -> bool {
    v.constant() == Some(0.0)
}

fn validate_unit_param(what: &str, param: &Param<f32>) -> Result<(), ModelError> {
    param.validate_shape()?;
    param.for_each_value(|value| validate_unit(what, *value))
}

fn validate_adjust_param(what: &str, param: &Param<f32>) -> Result<(), ModelError> {
    param.validate_shape()?;
    param.for_each_value(|value| {
        if value.is_finite() && (-1.0..=1.0).contains(value) {
            Ok(())
        } else {
            Err(ModelError::InvalidParam(format!(
                "{what} = {value} out of range [-1, 1]"
            )))
        }
    })
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_false(b: &bool) -> bool {
    !*b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_ids_are_unique() {
        fn assert_unique(ids: Vec<&str>, what: &str) {
            let mut sorted = ids.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), ids.len(), "duplicate {what} id");
        }
        assert_unique(mask_catalog().iter().map(|s| s.kind.id()).collect(), "mask");
        assert_unique(filter_catalog().iter().map(|s| s.id).collect(), "filter");
        assert_unique(
            animation_catalog().iter().map(|s| s.id).collect(),
            "animation",
        );
        assert_unique(
            text_effect_catalog().iter().map(|s| s.id).collect(),
            "text effect",
        );
    }

    #[test]
    fn enum_ids_match_their_serde_names() {
        for spec in mask_catalog() {
            let json = serde_json::to_value(spec.kind).unwrap();
            assert_eq!(json, serde_json::json!(spec.kind.id()));
        }
        for level in StabilizeLevel::ALL {
            let json = serde_json::to_value(level).unwrap();
            assert_eq!(json, serde_json::json!(level.id()));
        }
        for role in AudioRole::ALL {
            let json = serde_json::to_value(role).unwrap();
            assert_eq!(json, serde_json::json!(role.id()));
        }
    }

    #[test]
    fn defaults_are_elided_from_the_wire() {
        let mask = Mask::new(MaskKind::Circle);
        assert_eq!(
            serde_json::to_value(mask).unwrap(),
            serde_json::json!({"kind": "circle"})
        );

        let chroma = ChromaKey {
            rgb: [0, 255, 0],
            strength: 0.0.into(),
            shadow: 0.0.into(),
        };
        assert_eq!(
            serde_json::to_value(chroma).unwrap(),
            serde_json::json!({"rgb": [0, 255, 0]})
        );

        let filter = Filter::new("vivid");
        assert_eq!(
            serde_json::to_value(filter).unwrap(),
            serde_json::json!({"id": "vivid"})
        );
    }

    #[test]
    fn validation_rejects_out_of_range_values() {
        let mut mask = Mask::new(MaskKind::Linear);
        mask.feather = 1.5.into();
        assert!(mask.validate().is_err());

        let chroma = ChromaKey {
            rgb: [0, 255, 0],
            strength: (-0.1).into(),
            shadow: 0.0.into(),
        };
        assert!(chroma.validate().is_err());

        assert!(Filter::new("nope").validate().is_err());
        let mut filter = Filter::new("vivid");
        filter.intensity = 2.0.into();
        assert!(filter.validate().is_err());

        let adjust = ColorAdjustments {
            brightness: (-1.5).into(),
            ..Default::default()
        };
        assert!(adjust.validate().is_err());
        assert!(ColorAdjustments::default().is_neutral());
    }

    #[test]
    fn animation_catalog_slots_and_text_flags() {
        assert_eq!(animation_spec("fade_in").unwrap().slot, AnimationSlot::In);
        assert_eq!(animation_spec("drop").unwrap().slot, AnimationSlot::Out);
        assert_eq!(animation_spec("pulse").unwrap().slot, AnimationSlot::Combo);
        assert!(animation_spec("typewriter").unwrap().text_only);
        assert!(animation_spec("missing").is_none());
        // Whole-layer presets expose speed/intensity; per-char also stagger.
        let fade = animation_spec("fade_in").unwrap();
        assert!(fade.knobs.speed && fade.knobs.intensity && !fade.knobs.stagger);
        let wave = animation_spec("wave").unwrap();
        assert!(wave.knobs.speed && wave.knobs.intensity && wave.knobs.stagger);
    }

    #[test]
    fn animation_ref_defaults_and_normalization() {
        let spec = animation_spec("wave").unwrap();
        let a = AnimationRef::new("wave");
        assert_eq!(a.speed, ANIMATION_PARAM_DEFAULT);
        let ok = a.clone().normalized_for(spec).unwrap();
        assert_eq!(ok.intensity, 1.0);

        let mut hot = AnimationRef::new("wave");
        hot.intensity = 1.5;
        hot.stagger = 0.5;
        let hot = hot.normalized_for(spec).unwrap();
        assert_eq!(hot.intensity, 1.5);
        assert_eq!(hot.stagger, 0.5);

        let mut bad = AnimationRef::new("wave");
        bad.speed = 99.0;
        assert!(bad.normalized_for(spec).is_err());

        // Unsupported knobs snap back to default.
        let fade = animation_spec("fade_in").unwrap();
        let mut staggered = AnimationRef::new("fade_in");
        staggered.stagger = 1.5;
        let cleaned = staggered.normalized_for(fade).unwrap();
        assert_eq!(cleaned.stagger, ANIMATION_PARAM_DEFAULT);
    }

    #[test]
    fn animation_ref_serde_omits_default_knobs() {
        let a = AnimationRef::new("pulse");
        assert_eq!(
            serde_json::to_value(&a).unwrap(),
            serde_json::json!({"id": "pulse"})
        );
        let mut tuned = AnimationRef::new("pulse");
        tuned.speed = 2.0;
        assert_eq!(
            serde_json::to_value(&tuned).unwrap(),
            serde_json::json!({"id": "pulse", "speed": 2.0})
        );
        // Old id-only JSON still loads with defaults.
        let loaded: AnimationRef =
            serde_json::from_value(serde_json::json!({"id": "pulse"})).unwrap();
        assert_eq!(loaded.speed, ANIMATION_PARAM_DEFAULT);
        assert_eq!(loaded.intensity, ANIMATION_PARAM_DEFAULT);
    }

    #[test]
    fn text_effect_presets_resolve() {
        let neon = text_effect_spec("neon").unwrap();
        assert!(neon.stroke.is_some() && neon.shadow.is_some());
        assert!(text_effect_spec("nope").is_none());
    }
}
