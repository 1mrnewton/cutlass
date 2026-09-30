//! Clip-local constant param writes shared by project edits and live preview
//! overrides.

use crate::error::ModelError;
use crate::look::mask::{is_mask_param, set_mask_param_constant};
use crate::look::styles::set_style_param_constant;
use crate::param::Param;

use super::{Clip, ClipParam, ClipSource, CropRect, LookParam, ParamValue};

impl Clip {
    /// Replace one animatable property with a constant, dropping its keyframes.
    ///
    /// Same addressing as [`crate::Project::set_param_constant`], but without
    /// project-level target checks (audio-capable clip, visual track, …). Used
    /// by live preview overrides after the UI/worker has already validated the
    /// edit, and by the project mutator after those checks.
    pub fn set_param_constant(
        &mut self,
        param: ClipParam,
        value: ParamValue,
    ) -> Result<(), ModelError> {
        match param {
            ClipParam::Volume => {
                let v = expect_scalar(value)?;
                crate::clip::validate_volume(v)?;
                self.volume.set_constant(v);
                Ok(())
            }
            ClipParam::Pan => {
                let v = expect_scalar(value)?;
                crate::clip::validate_pan(v)?;
                self.pan.set_constant(v);
                Ok(())
            }
            ClipParam::Crop => {
                let [x, y, w, h] = value.rect()?;
                let crop = CropRect { x, y, w, h };
                crop.validate()?;
                self.crop.set_constant(crop);
                Ok(())
            }
            ClipParam::Effect { effect, param } => self
                .effects
                .get_mut(effect as usize)
                .ok_or_else(|| {
                    ModelError::InvalidParam(format!("effect index {effect} out of range"))
                })?
                .set_param_value_constant(param as usize, value),
            ClipParam::Shape { param } => match &mut self.content {
                ClipSource::Generated(generator) => {
                    generator.set_shape_param_constant(param, value)
                }
                ClipSource::Media { .. } => Err(ModelError::InvalidParam(
                    "shape parameters apply only to generated clips".into(),
                )),
            },
            ClipParam::Text { param } => match &mut self.content {
                ClipSource::Generated(generator) => generator.set_text_param_constant(param, value),
                ClipSource::Media { .. } => Err(ModelError::InvalidParam(
                    "text parameters apply only to generated clips".into(),
                )),
            },
            ClipParam::Look { param } if is_mask_param(param) => {
                set_mask_param_constant(&mut self.mask, param, value)
            }
            ClipParam::Look { param } => {
                let value = expect_scalar(value)?;
                validate_look_value(param, value)?;
                look_param_mut(self, param)?.set_constant(value);
                Ok(())
            }
            ClipParam::Style { param } => set_style_param_constant(&mut self.styles, param, value),
            ClipParam::Speed => Err(ModelError::InvalidParam(
                "speed curve is not written through set_param_constant".into(),
            )),
            ClipParam::Position
            | ClipParam::AnchorPoint
            | ClipParam::Scale
            | ClipParam::Rotation
            | ClipParam::Opacity => self.transform.set_param_constant(param, value),
        }
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

fn look_param_mut(clip: &mut Clip, param: LookParam) -> Result<&mut Param<f32>, ModelError> {
    let missing =
        |name: &str| ModelError::InvalidParam(format!("{name} is not enabled on this clip"));
    match param {
        LookParam::FilterIntensity => clip
            .filter
            .as_mut()
            .map(|filter| &mut filter.intensity)
            .ok_or_else(|| missing("filter")),
        LookParam::LutIntensity => clip
            .lut
            .as_mut()
            .map(|lut| &mut lut.intensity)
            .ok_or_else(|| missing("LUT")),
        LookParam::AdjustBrightness => Ok(&mut clip.adjust.brightness),
        LookParam::AdjustContrast => Ok(&mut clip.adjust.contrast),
        LookParam::AdjustSaturation => Ok(&mut clip.adjust.saturation),
        LookParam::AdjustExposure => Ok(&mut clip.adjust.exposure),
        LookParam::AdjustTemperature => Ok(&mut clip.adjust.temperature),
        LookParam::AdjustTint => Ok(&mut clip.adjust.tint),
        LookParam::AdjustHue => Ok(&mut clip.adjust.hue),
        LookParam::AdjustHighlights => Ok(&mut clip.adjust.highlights),
        LookParam::AdjustShadows => Ok(&mut clip.adjust.shadows),
        LookParam::AdjustSharpness => Ok(&mut clip.adjust.sharpness),
        LookParam::AdjustVignette => Ok(&mut clip.adjust.vignette),
        LookParam::ChromaStrength => clip
            .chroma_key
            .as_mut()
            .map(|chroma| &mut chroma.strength)
            .ok_or_else(|| missing("chroma key")),
        LookParam::ChromaShadow => clip
            .chroma_key
            .as_mut()
            .map(|chroma| &mut chroma.shadow)
            .ok_or_else(|| missing("chroma key")),
        LookParam::MaskFeather
        | LookParam::MaskCenter
        | LookParam::MaskSize
        | LookParam::MaskRotation
        | LookParam::MaskRoundness => Err(ModelError::InvalidParam(format!(
            "look parameter {param:?} is routed through the mask param helpers"
        ))),
    }
}

fn validate_look_value(param: LookParam, value: f32) -> Result<(), ModelError> {
    let valid = match param {
        LookParam::FilterIntensity
        | LookParam::LutIntensity
        | LookParam::ChromaStrength
        | LookParam::ChromaShadow => (0.0..=1.0).contains(&value),
        LookParam::AdjustBrightness
        | LookParam::AdjustContrast
        | LookParam::AdjustSaturation
        | LookParam::AdjustExposure
        | LookParam::AdjustTemperature
        | LookParam::AdjustTint
        | LookParam::AdjustHue
        | LookParam::AdjustHighlights
        | LookParam::AdjustShadows => (-1.0..=1.0).contains(&value),
        LookParam::AdjustSharpness | LookParam::AdjustVignette => (0.0..=1.0).contains(&value),
        LookParam::MaskFeather
        | LookParam::MaskCenter
        | LookParam::MaskSize
        | LookParam::MaskRotation
        | LookParam::MaskRoundness => {
            return Err(ModelError::InvalidParam(format!(
                "look parameter {param:?} is routed through the mask param helpers"
            )));
        }
    };
    if value.is_finite() && valid {
        Ok(())
    } else {
        Err(ModelError::InvalidParam(format!(
            "look parameter {param:?} = {value} is out of range"
        )))
    }
}
