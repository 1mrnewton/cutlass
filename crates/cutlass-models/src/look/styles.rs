// --- Layer styles -------------------------------------------------------------

use serde::{Deserialize, Serialize};

use crate::error::ModelError;
use crate::param::Param;

/// Layer-quad styles applied to a clip's rendered pixels (CapCut-style
/// shadow/glow/outline/background for ANY visual clip). Rendered from the
/// layer's alpha by the compositor after transform; distinct from text's
/// glyph-level treatments. Lengths are reference pixels (1080p baseline).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct LayerStyles {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow: Option<LayerShadow>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub glow: Option<LayerGlow>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outline: Option<LayerOutline>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background: Option<LayerBackground>,
}

impl LayerStyles {
    /// True iff no style block is present — the serde skip predicate on
    /// [`crate::Clip::styles`].
    pub fn is_empty(&self) -> bool {
        self.shadow.is_none()
            && self.glow.is_none()
            && self.outline.is_none()
            && self.background.is_none()
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        if let Some(shadow) = &self.shadow {
            shadow.validate()?;
        }
        if let Some(glow) = &self.glow {
            glow.validate()?;
        }
        if let Some(outline) = &self.outline {
            outline.validate()?;
        }
        if let Some(background) = &self.background {
            background.validate()?;
        }
        Ok(())
    }
}

/// Drop shadow drawn from the layer's alpha (offset + blur).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerShadow {
    /// Shadow color (RGBA, 0-255).
    pub rgba: Param<[u8; 4]>,
    /// Offset in reference pixels (`+x` right, `+y` down).
    pub offset: Param<[f32; 2]>,
    /// Blur radius in reference pixels (`>= 0`).
    pub blur: Param<f32>,
}

impl Default for LayerShadow {
    fn default() -> Self {
        Self {
            rgba: Param::Constant([0, 0, 0, 128]),
            offset: Param::Constant([4.0, 4.0]),
            blur: Param::Constant(8.0),
        }
    }
}

impl LayerShadow {
    pub fn validate(&self) -> Result<(), ModelError> {
        self.rgba.validate_shape()?;
        self.offset.validate_shape()?;
        validate_finite_vec2_param("layer shadow offset", &self.offset)?;
        validate_non_negative_param("layer shadow blur", &self.blur)
    }
}

/// Soft glow bloom drawn from the layer's alpha.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerGlow {
    /// Glow color (RGBA, 0-255).
    pub rgba: Param<[u8; 4]>,
    /// Glow radius in reference pixels (`>= 0`).
    pub radius: Param<f32>,
    /// Strength multiplier, `0` … `4`.
    pub intensity: Param<f32>,
}

impl Default for LayerGlow {
    fn default() -> Self {
        Self {
            rgba: Param::Constant([255, 255, 255, 255]),
            radius: Param::Constant(12.0),
            intensity: Param::Constant(1.0),
        }
    }
}

impl LayerGlow {
    pub fn validate(&self) -> Result<(), ModelError> {
        self.rgba.validate_shape()?;
        validate_non_negative_param("layer glow radius", &self.radius)?;
        validate_intensity_param("layer glow intensity", &self.intensity)
    }
}

/// Hard outline / stroke around the layer's alpha silhouette.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerOutline {
    /// Outline color (RGBA, 0-255).
    pub rgba: Param<[u8; 4]>,
    /// Outline width in reference pixels (`>= 0`).
    pub width: Param<f32>,
}

impl Default for LayerOutline {
    fn default() -> Self {
        Self {
            rgba: Param::Constant([255, 255, 255, 255]),
            width: Param::Constant(4.0),
        }
    }
}

impl LayerOutline {
    pub fn validate(&self) -> Result<(), ModelError> {
        self.rgba.validate_shape()?;
        validate_non_negative_param("layer outline width", &self.width)
    }
}

/// Solid plate behind the layer (padded AABB of the alpha bounds).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerBackground {
    /// Plate color (RGBA, 0-255).
    pub rgba: Param<[u8; 4]>,
    /// Padding around the alpha bounds in reference pixels (`>= 0`).
    pub padding: Param<f32>,
    /// Corner radius in reference pixels (`>= 0`).
    pub radius: Param<f32>,
}

impl Default for LayerBackground {
    fn default() -> Self {
        Self {
            rgba: Param::Constant([0, 0, 0, 255]),
            padding: Param::Constant(12.0),
            radius: Param::Constant(0.0),
        }
    }
}

impl LayerBackground {
    pub fn validate(&self) -> Result<(), ModelError> {
        self.rgba.validate_shape()?;
        validate_non_negative_param("layer background padding", &self.padding)?;
        validate_non_negative_param("layer background radius", &self.radius)
    }
}

fn validate_non_negative_param(what: &str, param: &Param<f32>) -> Result<(), ModelError> {
    param.validate_shape()?;
    param.for_each_value(|value| {
        if value.is_finite() && *value >= 0.0 {
            Ok(())
        } else {
            Err(ModelError::InvalidParam(format!(
                "{what} = {value} must be finite and >= 0"
            )))
        }
    })
}

fn validate_intensity_param(what: &str, param: &Param<f32>) -> Result<(), ModelError> {
    param.validate_shape()?;
    param.for_each_value(|value| {
        if value.is_finite() && (0.0..=4.0).contains(value) {
            Ok(())
        } else {
            Err(ModelError::InvalidParam(format!(
                "{what} = {value} out of range [0, 4]"
            )))
        }
    })
}

fn validate_finite_vec2_param(what: &str, param: &Param<[f32; 2]>) -> Result<(), ModelError> {
    param.for_each_value(|value| {
        if value[0].is_finite() && value[1].is_finite() {
            Ok(())
        } else {
            Err(ModelError::InvalidParam(format!(
                "{what} = [{}, {}] must be finite",
                value[0], value[1]
            )))
        }
    })
}
