// --- Layer styles -------------------------------------------------------------

use serde::{Deserialize, Serialize};

use crate::clip::{ParamValue, StyleParam};
use crate::error::ModelError;
use crate::param::{Easing, Param};

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

fn missing_style(block: &str) -> ModelError {
    ModelError::InvalidParam(format!("clip has no {block} layer style"))
}

/// Resolve a scalar [`StyleParam`] to the [`Param`] it names on `styles`.
pub(crate) fn style_scalar_mut(
    styles: &mut LayerStyles,
    param: StyleParam,
) -> Result<&mut Param<f32>, ModelError> {
    match param {
        StyleParam::ShadowBlur => styles
            .shadow
            .as_mut()
            .map(|shadow| &mut shadow.blur)
            .ok_or_else(|| missing_style("shadow")),
        StyleParam::GlowRadius => styles
            .glow
            .as_mut()
            .map(|glow| &mut glow.radius)
            .ok_or_else(|| missing_style("glow")),
        StyleParam::GlowIntensity => styles
            .glow
            .as_mut()
            .map(|glow| &mut glow.intensity)
            .ok_or_else(|| missing_style("glow")),
        StyleParam::OutlineWidth => styles
            .outline
            .as_mut()
            .map(|outline| &mut outline.width)
            .ok_or_else(|| missing_style("outline")),
        StyleParam::BackgroundPadding => styles
            .background
            .as_mut()
            .map(|background| &mut background.padding)
            .ok_or_else(|| missing_style("background")),
        StyleParam::BackgroundRadius => styles
            .background
            .as_mut()
            .map(|background| &mut background.radius)
            .ok_or_else(|| missing_style("background")),
        StyleParam::ShadowColor
        | StyleParam::ShadowOffset
        | StyleParam::GlowColor
        | StyleParam::OutlineColor
        | StyleParam::BackgroundColor => Err(ModelError::InvalidParam(format!(
            "style parameter {param:?} is not a scalar"
        ))),
    }
}

/// Resolve a color [`StyleParam`] to the [`Param`] it names on `styles`.
pub(crate) fn style_color_mut(
    styles: &mut LayerStyles,
    param: StyleParam,
) -> Result<&mut Param<[u8; 4]>, ModelError> {
    match param {
        StyleParam::ShadowColor => styles
            .shadow
            .as_mut()
            .map(|shadow| &mut shadow.rgba)
            .ok_or_else(|| missing_style("shadow")),
        StyleParam::GlowColor => styles
            .glow
            .as_mut()
            .map(|glow| &mut glow.rgba)
            .ok_or_else(|| missing_style("glow")),
        StyleParam::OutlineColor => styles
            .outline
            .as_mut()
            .map(|outline| &mut outline.rgba)
            .ok_or_else(|| missing_style("outline")),
        StyleParam::BackgroundColor => styles
            .background
            .as_mut()
            .map(|background| &mut background.rgba)
            .ok_or_else(|| missing_style("background")),
        StyleParam::ShadowOffset
        | StyleParam::ShadowBlur
        | StyleParam::GlowRadius
        | StyleParam::GlowIntensity
        | StyleParam::OutlineWidth
        | StyleParam::BackgroundPadding
        | StyleParam::BackgroundRadius => Err(ModelError::InvalidParam(format!(
            "style parameter {param:?} is not a color"
        ))),
    }
}

/// Resolve the vec2 [`StyleParam::ShadowOffset`] to the [`Param`] it names.
pub(crate) fn style_vec2_mut(
    styles: &mut LayerStyles,
    param: StyleParam,
) -> Result<&mut Param<[f32; 2]>, ModelError> {
    match param {
        StyleParam::ShadowOffset => styles
            .shadow
            .as_mut()
            .map(|shadow| &mut shadow.offset)
            .ok_or_else(|| missing_style("shadow")),
        StyleParam::ShadowColor
        | StyleParam::ShadowBlur
        | StyleParam::GlowColor
        | StyleParam::GlowRadius
        | StyleParam::GlowIntensity
        | StyleParam::OutlineColor
        | StyleParam::OutlineWidth
        | StyleParam::BackgroundColor
        | StyleParam::BackgroundPadding
        | StyleParam::BackgroundRadius => Err(ModelError::InvalidParam(format!(
            "style parameter {param:?} is not a vec2"
        ))),
    }
}

/// Range-check a scalar style value (matches [`LayerStyles::validate`]).
pub(crate) fn validate_style_value(param: StyleParam, value: f32) -> Result<(), ModelError> {
    let ok = match param {
        StyleParam::ShadowBlur
        | StyleParam::GlowRadius
        | StyleParam::OutlineWidth
        | StyleParam::BackgroundPadding
        | StyleParam::BackgroundRadius => value.is_finite() && value >= 0.0,
        StyleParam::GlowIntensity => value.is_finite() && (0.0..=4.0).contains(&value),
        StyleParam::ShadowColor
        | StyleParam::ShadowOffset
        | StyleParam::GlowColor
        | StyleParam::OutlineColor
        | StyleParam::BackgroundColor => {
            return Err(ModelError::InvalidParam(format!(
                "style parameter {param:?} is not a scalar"
            )));
        }
    };
    if ok {
        Ok(())
    } else {
        Err(ModelError::InvalidParam(format!(
            "style parameter {param:?} = {value} is out of range"
        )))
    }
}

fn expect_scalar(value: ParamValue) -> Result<f32, ModelError> {
    match value {
        ParamValue::Scalar(v) => Ok(v),
        ParamValue::Vec2(_) | ParamValue::Color(_) | ParamValue::Rect(_) => {
            Err(ModelError::InvalidParam("expected a scalar value".into()))
        }
    }
}

fn expect_vec2(value: ParamValue) -> Result<[f32; 2], ModelError> {
    match value {
        ParamValue::Vec2(v) => Ok(v),
        ParamValue::Scalar(_) | ParamValue::Color(_) | ParamValue::Rect(_) => {
            Err(ModelError::InvalidParam("expected a vec2 value".into()))
        }
    }
}

fn expect_color(value: ParamValue) -> Result<[u8; 4], ModelError> {
    match value {
        ParamValue::Color(v) => Ok(v),
        ParamValue::Scalar(_) | ParamValue::Vec2(_) | ParamValue::Rect(_) => {
            Err(ModelError::InvalidParam("expected a color value".into()))
        }
    }
}

fn validate_style_offset(value: [f32; 2]) -> Result<(), ModelError> {
    if value[0].is_finite() && value[1].is_finite() {
        Ok(())
    } else {
        Err(ModelError::InvalidParam(format!(
            "style parameter ShadowOffset = [{}, {}] must be finite",
            value[0], value[1]
        )))
    }
}

/// Insert or replace a keyframe on one layer-style property.
pub(crate) fn set_style_param_keyframe(
    styles: &mut LayerStyles,
    param: StyleParam,
    tick: i64,
    value: ParamValue,
    easing: Easing,
) -> Result<(), ModelError> {
    easing.validate()?;
    match param {
        StyleParam::ShadowBlur
        | StyleParam::GlowRadius
        | StyleParam::GlowIntensity
        | StyleParam::OutlineWidth
        | StyleParam::BackgroundPadding
        | StyleParam::BackgroundRadius => {
            let v = expect_scalar(value)?;
            validate_style_value(param, v)?;
            style_scalar_mut(styles, param)?.set_keyframe(tick, v, easing);
            Ok(())
        }
        StyleParam::ShadowOffset => {
            let v = expect_vec2(value)?;
            validate_style_offset(v)?;
            style_vec2_mut(styles, param)?.set_keyframe(tick, v, easing);
            Ok(())
        }
        StyleParam::ShadowColor
        | StyleParam::GlowColor
        | StyleParam::OutlineColor
        | StyleParam::BackgroundColor => {
            let v = expect_color(value)?;
            style_color_mut(styles, param)?.set_keyframe(tick, v, easing);
            Ok(())
        }
    }
}

/// Remove the keyframe at exactly `tick` on one layer-style property.
pub(crate) fn remove_style_param_keyframe(
    styles: &mut LayerStyles,
    param: StyleParam,
    tick: i64,
) -> Result<(), ModelError> {
    let removed = match param {
        StyleParam::ShadowBlur
        | StyleParam::GlowRadius
        | StyleParam::GlowIntensity
        | StyleParam::OutlineWidth
        | StyleParam::BackgroundPadding
        | StyleParam::BackgroundRadius => style_scalar_mut(styles, param)?.remove_keyframe(tick),
        StyleParam::ShadowOffset => style_vec2_mut(styles, param)?.remove_keyframe(tick),
        StyleParam::ShadowColor
        | StyleParam::GlowColor
        | StyleParam::OutlineColor
        | StyleParam::BackgroundColor => style_color_mut(styles, param)?.remove_keyframe(tick),
    };
    if removed {
        Ok(())
    } else {
        Err(ModelError::InvalidParam(format!(
            "no {param:?} keyframe at tick {tick}"
        )))
    }
}

/// Replace one layer-style property with a constant, dropping its keyframes.
pub(crate) fn set_style_param_constant(
    styles: &mut LayerStyles,
    param: StyleParam,
    value: ParamValue,
) -> Result<(), ModelError> {
    match param {
        StyleParam::ShadowBlur
        | StyleParam::GlowRadius
        | StyleParam::GlowIntensity
        | StyleParam::OutlineWidth
        | StyleParam::BackgroundPadding
        | StyleParam::BackgroundRadius => {
            let v = expect_scalar(value)?;
            validate_style_value(param, v)?;
            style_scalar_mut(styles, param)?.set_constant(v);
            Ok(())
        }
        StyleParam::ShadowOffset => {
            let v = expect_vec2(value)?;
            validate_style_offset(v)?;
            style_vec2_mut(styles, param)?.set_constant(v);
            Ok(())
        }
        StyleParam::ShadowColor
        | StyleParam::GlowColor
        | StyleParam::OutlineColor
        | StyleParam::BackgroundColor => {
            let v = expect_color(value)?;
            style_color_mut(styles, param)?.set_constant(v);
            Ok(())
        }
    }
}
