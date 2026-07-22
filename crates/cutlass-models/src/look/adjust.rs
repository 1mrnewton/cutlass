// --- Color adjustments -----------------------------------------------------------

use serde::{Deserialize, Serialize};

use crate::error::ModelError;
use crate::param::Param;

use super::{default_zero_param, is_zero_param};

/// Manual color grade (CapCut adjust panel): signed strengths, `0` neutral.
/// Lives on visual clips and on `Generator::Adjustment` lane bars.
///
/// # Slider semantics
///
/// - **brightness / contrast / saturation / exposure / temperature**: signed
///   `[-1, 1]`, `0` neutral (existing).
/// - **tint**: green (−) ↔ magenta (+), `[-1, 1]`.
/// - **hue**: hue rotation; ±1 maps to ±30°, `[-1, 1]`.
/// - **highlights / shadows**: lift/compress tones above / below mid-luma,
///   `[-1, 1]`.
/// - **sharpness / vignette**: one-directional; stored as `Param<f32>` like the
///   others for slider uniformity, but validated in `[0, 1]` (negatives
///   rejected). Softening / inverse vignette are not supported.
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
    /// Green (−) ↔ magenta (+).
    #[serde(default = "default_zero_param", skip_serializing_if = "is_zero_param")]
    pub tint: Param<f32>,
    /// Hue rotation; ±1 → ±30°.
    #[serde(default = "default_zero_param", skip_serializing_if = "is_zero_param")]
    pub hue: Param<f32>,
    /// Lift/compress values above mid-luma.
    #[serde(default = "default_zero_param", skip_serializing_if = "is_zero_param")]
    pub highlights: Param<f32>,
    /// Lift/compress values below mid-luma.
    #[serde(default = "default_zero_param", skip_serializing_if = "is_zero_param")]
    pub shadows: Param<f32>,
    /// Unsharp-mask strength; validated in `[0, 1]` (no softening).
    #[serde(default = "default_zero_param", skip_serializing_if = "is_zero_param")]
    pub sharpness: Param<f32>,
    /// Radial darkening from layer center; validated in `[0, 1]`.
    #[serde(default = "default_zero_param", skip_serializing_if = "is_zero_param")]
    pub vignette: Param<f32>,
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
            &self.tint,
            &self.hue,
            &self.highlights,
            &self.shadows,
            &self.sharpness,
            &self.vignette,
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
            ("tint", &self.tint),
            ("hue", &self.hue),
            ("highlights", &self.highlights),
            ("shadows", &self.shadows),
        ] {
            validate_adjust_param(name, value, -1.0, 1.0)?;
        }
        for (name, value) in [("sharpness", &self.sharpness), ("vignette", &self.vignette)] {
            validate_adjust_param(name, value, 0.0, 1.0)?;
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
            tint: default_zero_param(),
            hue: default_zero_param(),
            highlights: default_zero_param(),
            shadows: default_zero_param(),
            sharpness: default_zero_param(),
            vignette: default_zero_param(),
        }
    }
}

fn validate_adjust_param(
    what: &str,
    param: &Param<f32>,
    lo: f32,
    hi: f32,
) -> Result<(), ModelError> {
    param.validate_shape()?;
    param.for_each_value(|value| {
        if value.is_finite() && (*value >= lo && *value <= hi) {
            Ok(())
        } else {
            Err(ModelError::InvalidParam(format!(
                "{what} = {value} out of range [{lo}, {hi}]"
            )))
        }
    })
}
