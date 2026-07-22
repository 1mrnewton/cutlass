// --- Color adjustments -----------------------------------------------------------

use serde::{Deserialize, Serialize};

use crate::error::ModelError;
use crate::param::Param;

use super::{default_zero_param, is_zero_param};

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
