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
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Mask {
    pub kind: MaskKind,
    /// Edge softness, `0` (hard) … `1` (fully feathered).
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub feather: f32,
    /// Keep the outside instead of the inside.
    #[serde(default, skip_serializing_if = "is_false")]
    pub invert: bool,
}

impl Mask {
    /// A hard, non-inverted mask of `kind`.
    pub fn new(kind: MaskKind) -> Self {
        Self {
            kind,
            feather: 0.0,
            invert: false,
        }
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        validate_unit("mask feather", self.feather)
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
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ChromaKey {
    /// Key color, opaque `[r, g, b]`.
    pub rgb: [u8; 3],
    /// Keying strength (tolerance), `0` … `1`.
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub strength: f32,
    /// Shadow retention, `0` … `1`.
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub shadow: f32,
}

impl ChromaKey {
    pub fn validate(&self) -> Result<(), ModelError> {
        validate_unit("chroma strength", self.strength)?;
        validate_unit("chroma shadow", self.shadow)
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
        default = "default_filter_intensity",
        skip_serializing_if = "is_default_filter_intensity"
    )]
    pub intensity: f32,
}

impl Filter {
    /// A filter at the default intensity.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            intensity: default_filter_intensity(),
        }
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        if filter_spec(&self.id).is_none() {
            return Err(ModelError::InvalidParam(format!(
                "unknown filter '{}'",
                self.id
            )));
        }
        validate_unit("filter intensity", self.intensity)
    }
}

fn default_filter_intensity() -> f32 {
    0.8
}

fn is_default_filter_intensity(v: &f32) -> bool {
    *v == default_filter_intensity()
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
        default = "default_filter_intensity",
        skip_serializing_if = "is_default_filter_intensity"
    )]
    pub intensity: f32,
}

impl Lut {
    /// A LUT at the default intensity.
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            intensity: default_filter_intensity(),
        }
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        if self.path.trim().is_empty() {
            return Err(ModelError::InvalidParam("empty LUT path".into()));
        }
        validate_unit("LUT intensity", self.intensity)
    }
}

// --- Color adjustments -----------------------------------------------------------

/// Manual color grade (CapCut adjust panel): signed strengths, `0` neutral.
/// Lives on visual clips and on `Generator::Adjustment` lane bars.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct ColorAdjustments {
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub brightness: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub contrast: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub saturation: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub exposure: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub temperature: f32,
}

impl ColorAdjustments {
    /// True iff every slider sits at neutral — the serde skip predicate.
    pub fn is_neutral(&self) -> bool {
        *self == Self::default()
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        for (name, value) in [
            ("brightness", self.brightness),
            ("contrast", self.contrast),
            ("saturation", self.saturation),
            ("exposure", self.exposure),
            ("temperature", self.temperature),
        ] {
            if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
                return Err(ModelError::InvalidParam(format!(
                    "{name} = {value} out of range [-1, 1]"
                )));
            }
        }
        Ok(())
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

/// A reference to a catalog animation, stored per slot on the clip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnimationRef {
    /// Catalog id (see [`animation_catalog`]).
    pub id: String,
}

impl AnimationRef {
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }
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
}

const ANIMATIONS: &[AnimationSpec] = &[
    // Entrances.
    AnimationSpec {
        id: "fade_in",
        label: "Fade in",
        slot: AnimationSlot::In,
        text_only: false,
    },
    AnimationSpec {
        id: "slide_up",
        label: "Slide up",
        slot: AnimationSlot::In,
        text_only: false,
    },
    AnimationSpec {
        id: "zoom_in",
        label: "Zoom in",
        slot: AnimationSlot::In,
        text_only: false,
    },
    AnimationSpec {
        id: "spin_in",
        label: "Spin in",
        slot: AnimationSlot::In,
        text_only: false,
    },
    AnimationSpec {
        id: "bounce",
        label: "Bounce",
        slot: AnimationSlot::In,
        text_only: false,
    },
    // Exits.
    AnimationSpec {
        id: "fade_out",
        label: "Fade out",
        slot: AnimationSlot::Out,
        text_only: false,
    },
    AnimationSpec {
        id: "slide_down",
        label: "Slide down",
        slot: AnimationSlot::Out,
        text_only: false,
    },
    AnimationSpec {
        id: "zoom_out",
        label: "Zoom out",
        slot: AnimationSlot::Out,
        text_only: false,
    },
    AnimationSpec {
        id: "spin_out",
        label: "Spin out",
        slot: AnimationSlot::Out,
        text_only: false,
    },
    AnimationSpec {
        id: "drop",
        label: "Drop",
        slot: AnimationSlot::Out,
        text_only: false,
    },
    // Combos (looping presence animations).
    AnimationSpec {
        id: "pulse",
        label: "Pulse",
        slot: AnimationSlot::Combo,
        text_only: false,
    },
    AnimationSpec {
        id: "rock",
        label: "Rock",
        slot: AnimationSlot::Combo,
        text_only: false,
    },
    AnimationSpec {
        id: "swing",
        label: "Swing",
        slot: AnimationSlot::Combo,
        text_only: false,
    },
    AnimationSpec {
        id: "flicker",
        label: "Flicker",
        slot: AnimationSlot::Combo,
        text_only: false,
    },
    AnimationSpec {
        id: "breathe",
        label: "Breathe",
        slot: AnimationSlot::Combo,
        text_only: false,
    },
    // Text-only per-character presets (glyph atlas path).
    AnimationSpec {
        id: "char_typewriter",
        label: "Typewriter",
        slot: AnimationSlot::In,
        text_only: true,
    },
    AnimationSpec {
        id: "char_fade_in",
        label: "Fade in",
        slot: AnimationSlot::In,
        text_only: true,
    },
    AnimationSpec {
        id: "char_bounce_in",
        label: "Bounce in",
        slot: AnimationSlot::In,
        text_only: true,
    },
    AnimationSpec {
        id: "char_slide_in",
        label: "Slide in",
        slot: AnimationSlot::In,
        text_only: true,
    },
    AnimationSpec {
        id: "char_pop_in",
        label: "Pop in",
        slot: AnimationSlot::In,
        text_only: true,
    },
    AnimationSpec {
        id: "char_fade_out",
        label: "Fade out",
        slot: AnimationSlot::Out,
        text_only: true,
    },
    AnimationSpec {
        id: "char_fall_away",
        label: "Fall away",
        slot: AnimationSlot::Out,
        text_only: true,
    },
    AnimationSpec {
        id: "char_typewriter_out",
        label: "Type out",
        slot: AnimationSlot::Out,
        text_only: true,
    },
    // Text-only combos (the text panel's looping chips).
    AnimationSpec {
        id: "typewriter",
        label: "Typewriter",
        slot: AnimationSlot::Combo,
        text_only: true,
    },
    AnimationSpec {
        id: "text_fade",
        label: "Fade",
        slot: AnimationSlot::Combo,
        text_only: true,
    },
    AnimationSpec {
        id: "text_bounce",
        label: "Bounce",
        slot: AnimationSlot::Combo,
        text_only: true,
    },
    AnimationSpec {
        id: "text_slide",
        label: "Slide",
        slot: AnimationSlot::Combo,
        text_only: true,
    },
    AnimationSpec {
        id: "pop",
        label: "Pop",
        slot: AnimationSlot::Combo,
        text_only: true,
    },
    AnimationSpec {
        id: "wave",
        label: "Wave",
        slot: AnimationSlot::Combo,
        text_only: true,
    },
    AnimationSpec {
        id: "char_jitter",
        label: "Jitter",
        slot: AnimationSlot::Combo,
        text_only: true,
    },
    AnimationSpec {
        id: "char_pulse",
        label: "Pulse",
        slot: AnimationSlot::Combo,
        text_only: true,
    },
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
#[derive(Debug, Clone, Copy, PartialEq)]
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
            rgba: [57, 255, 20, 255],
            width: 4.0,
        }),
        shadow: Some(TextShadow {
            rgba: [57, 255, 20, 200],
            blur: 0.35,
            distance: 0.0,
        }),
        background: None,
    },
    TextEffectSpec {
        id: "shadow",
        label: "Shadow",
        stroke: None,
        shadow: Some(TextShadow {
            rgba: [0, 0, 0, 230],
            blur: 0.15,
            distance: 8.0,
        }),
        background: None,
    },
    TextEffectSpec {
        id: "outline",
        label: "Outline",
        stroke: Some(TextStroke {
            rgba: [0, 0, 0, 255],
            width: 8.0,
        }),
        shadow: None,
        background: None,
    },
    TextEffectSpec {
        id: "glow",
        label: "Glow",
        stroke: None,
        shadow: Some(TextShadow {
            rgba: [255, 255, 255, 220],
            blur: 0.4,
            distance: 0.0,
        }),
        background: None,
    },
    TextEffectSpec {
        id: "retro",
        label: "Retro",
        stroke: Some(TextStroke {
            rgba: [255, 140, 60, 255],
            width: 5.0,
        }),
        shadow: Some(TextShadow {
            rgba: [120, 40, 160, 255],
            blur: 0.05,
            distance: 10.0,
        }),
        background: None,
    },
    TextEffectSpec {
        id: "chrome",
        label: "Chrome",
        stroke: Some(TextStroke {
            rgba: [230, 230, 240, 255],
            width: 3.0,
        }),
        shadow: Some(TextShadow {
            rgba: [40, 60, 90, 200],
            blur: 0.2,
            distance: 6.0,
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

fn is_zero_f32(v: &f32) -> bool {
    *v == 0.0
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
            strength: 0.0,
            shadow: 0.0,
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
        mask.feather = 1.5;
        assert!(mask.validate().is_err());

        let chroma = ChromaKey {
            rgb: [0, 255, 0],
            strength: -0.1,
            shadow: 0.0,
        };
        assert!(chroma.validate().is_err());

        assert!(Filter::new("nope").validate().is_err());
        let mut filter = Filter::new("vivid");
        filter.intensity = 2.0;
        assert!(filter.validate().is_err());

        let adjust = ColorAdjustments {
            brightness: -1.5,
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
    }

    #[test]
    fn text_effect_presets_resolve() {
        let neon = text_effect_spec("neon").unwrap();
        assert!(neon.stroke.is_some() && neon.shadow.is_some());
        assert!(text_effect_spec("nope").is_none());
    }
}
