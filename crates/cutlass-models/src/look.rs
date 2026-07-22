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

mod adjust;
mod animation;
mod chroma;
mod filter;
mod mask;
mod text_effect;

pub use adjust::ColorAdjustments;
pub use animation::{
    ANIMATION_INTENSITY_RANGE, ANIMATION_PARAM_DEFAULT, ANIMATION_SPEED_RANGE,
    ANIMATION_STAGGER_RANGE, AnimationKnobs, AnimationRef, AnimationSlot, AnimationSpec,
    animation_catalog, animation_spec,
};
pub use chroma::{ChromaKey, StabilizeLevel};
pub use filter::{Filter, FilterSpec, Lut, filter_catalog, filter_spec};
pub use mask::{Mask, MaskKind, MaskSpec, mask_catalog};
pub use text_effect::{TextEffectSpec, text_effect_catalog, text_effect_spec};

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

#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_false(b: &bool) -> bool {
    !*b
}

#[cfg(test)]
mod tests;
