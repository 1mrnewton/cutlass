// --- Animations -----------------------------------------------------------------

use serde::{Deserialize, Serialize};

use crate::error::ModelError;

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
