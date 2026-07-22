// --- Mask -------------------------------------------------------------------

use serde::{Deserialize, Serialize};

use crate::clip::{LookParam, ParamValue};
use crate::error::ModelError;
use crate::param::{Easing, Param};

use super::{default_zero_param, is_false, is_zero_param, validate_unit_param};

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
    /// Mask center offset from the layer's center, as a fraction of the layer's
    /// size per axis (`[0,0]` = centered, `[0.5,0]` = right edge). May exceed ±1.
    #[serde(
        default = "default_center_param",
        skip_serializing_if = "is_default_center"
    )]
    pub center: Param<[f32; 2]>,
    /// Mask size as a fraction of the layer's size per axis. `[1,1]` covers the
    /// layer exactly (legacy behavior).
    #[serde(
        default = "default_size_param",
        skip_serializing_if = "is_default_size"
    )]
    pub size: Param<[f32; 2]>,
    /// Mask rotation in degrees, clockwise, about the mask center.
    #[serde(default = "default_zero_param", skip_serializing_if = "is_zero_param")]
    pub rotation: Param<f32>,
    /// Rectangle corner rounding, `0` (sharp) … `1` (fully round). Ignored by
    /// other kinds.
    #[serde(default = "default_zero_param", skip_serializing_if = "is_zero_param")]
    pub roundness: Param<f32>,
    /// Keep the outside instead of the inside.
    #[serde(default, skip_serializing_if = "is_false")]
    pub invert: bool,
}

fn default_center_param() -> Param<[f32; 2]> {
    Param::Constant([0.0, 0.0])
}

fn is_default_center(p: &Param<[f32; 2]>) -> bool {
    p.constant() == Some([0.0, 0.0])
}

fn default_size_param() -> Param<[f32; 2]> {
    Param::Constant([1.0, 1.0])
}

fn is_default_size(p: &Param<[f32; 2]>) -> bool {
    p.constant() == Some([1.0, 1.0])
}

fn validate_center_value(value: [f32; 2]) -> Result<(), ModelError> {
    if value[0].is_finite()
        && value[1].is_finite()
        && value[0].abs() <= 10.0
        && value[1].abs() <= 10.0
    {
        Ok(())
    } else {
        Err(ModelError::InvalidParam(format!(
            "mask center = [{}, {}] must be finite and within ±10",
            value[0], value[1]
        )))
    }
}

fn validate_size_value(value: [f32; 2]) -> Result<(), ModelError> {
    if value[0].is_finite() && value[1].is_finite() && value[0] > 0.0 && value[1] > 0.0 {
        Ok(())
    } else {
        Err(ModelError::InvalidParam(format!(
            "mask size = [{}, {}] must be finite and > 0",
            value[0], value[1]
        )))
    }
}

fn validate_rotation_value(value: f32) -> Result<(), ModelError> {
    if value.is_finite() && value.abs() <= 3600.0 {
        Ok(())
    } else {
        Err(ModelError::InvalidParam(format!(
            "mask rotation = {value} must be finite and within ±3600"
        )))
    }
}

impl Mask {
    /// A hard, non-inverted mask of `kind`.
    pub fn new(kind: MaskKind) -> Self {
        Self {
            kind,
            feather: Param::Constant(0.0),
            center: Param::Constant([0.0, 0.0]),
            size: Param::Constant([1.0, 1.0]),
            rotation: Param::Constant(0.0),
            roundness: Param::Constant(0.0),
            invert: false,
        }
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        validate_unit_param("mask feather", &self.feather)?;
        self.center.validate_shape()?;
        self.center
            .for_each_value(|value| validate_center_value(*value))?;
        self.size.validate_shape()?;
        self.size
            .for_each_value(|value| validate_size_value(*value))?;
        self.rotation.validate_shape()?;
        self.rotation
            .for_each_value(|value| validate_rotation_value(*value))?;
        validate_unit_param("mask roundness", &self.roundness)
    }
}

fn missing_mask() -> ModelError {
    ModelError::InvalidParam("clip has no mask".into())
}

/// Resolve a scalar mask [`LookParam`] to the [`Param`] it names.
pub(crate) fn mask_scalar_mut(
    mask: &mut Option<Mask>,
    param: LookParam,
) -> Result<&mut Param<f32>, ModelError> {
    let mask = mask.as_mut().ok_or_else(missing_mask)?;
    match param {
        LookParam::MaskFeather => Ok(&mut mask.feather),
        LookParam::MaskRotation => Ok(&mut mask.rotation),
        LookParam::MaskRoundness => Ok(&mut mask.roundness),
        LookParam::MaskCenter
        | LookParam::MaskSize
        | LookParam::FilterIntensity
        | LookParam::LutIntensity
        | LookParam::AdjustBrightness
        | LookParam::AdjustContrast
        | LookParam::AdjustSaturation
        | LookParam::AdjustExposure
        | LookParam::AdjustTemperature
        | LookParam::ChromaStrength
        | LookParam::ChromaShadow => Err(ModelError::InvalidParam(format!(
            "look parameter {param:?} is not a scalar mask property"
        ))),
    }
}

/// Resolve a vec2 mask [`LookParam`] to the [`Param`] it names.
pub(crate) fn mask_vec2_mut(
    mask: &mut Option<Mask>,
    param: LookParam,
) -> Result<&mut Param<[f32; 2]>, ModelError> {
    let mask = mask.as_mut().ok_or_else(missing_mask)?;
    match param {
        LookParam::MaskCenter => Ok(&mut mask.center),
        LookParam::MaskSize => Ok(&mut mask.size),
        LookParam::MaskFeather
        | LookParam::MaskRotation
        | LookParam::MaskRoundness
        | LookParam::FilterIntensity
        | LookParam::LutIntensity
        | LookParam::AdjustBrightness
        | LookParam::AdjustContrast
        | LookParam::AdjustSaturation
        | LookParam::AdjustExposure
        | LookParam::AdjustTemperature
        | LookParam::ChromaStrength
        | LookParam::ChromaShadow => Err(ModelError::InvalidParam(format!(
            "look parameter {param:?} is not a vec2 mask property"
        ))),
    }
}

/// Range-check a scalar mask value (matches [`Mask::validate`]).
pub(crate) fn validate_mask_value(param: LookParam, value: f32) -> Result<(), ModelError> {
    match param {
        LookParam::MaskFeather | LookParam::MaskRoundness => {
            if value.is_finite() && (0.0..=1.0).contains(&value) {
                Ok(())
            } else {
                Err(ModelError::InvalidParam(format!(
                    "look parameter {param:?} = {value} is out of range"
                )))
            }
        }
        LookParam::MaskRotation => validate_rotation_value(value).map_err(|_| {
            ModelError::InvalidParam(format!(
                "look parameter {param:?} = {value} is out of range"
            ))
        }),
        LookParam::MaskCenter
        | LookParam::MaskSize
        | LookParam::FilterIntensity
        | LookParam::LutIntensity
        | LookParam::AdjustBrightness
        | LookParam::AdjustContrast
        | LookParam::AdjustSaturation
        | LookParam::AdjustExposure
        | LookParam::AdjustTemperature
        | LookParam::ChromaStrength
        | LookParam::ChromaShadow => Err(ModelError::InvalidParam(format!(
            "look parameter {param:?} is not a scalar"
        ))),
    }
}

fn expect_scalar(value: ParamValue) -> Result<f32, ModelError> {
    match value {
        ParamValue::Scalar(v) => Ok(v),
        ParamValue::Vec2(_) | ParamValue::Color(_) => {
            Err(ModelError::InvalidParam("expected a scalar value".into()))
        }
    }
}

fn expect_vec2(value: ParamValue) -> Result<[f32; 2], ModelError> {
    match value {
        ParamValue::Vec2(v) => Ok(v),
        ParamValue::Scalar(_) | ParamValue::Color(_) => {
            Err(ModelError::InvalidParam("expected a vec2 value".into()))
        }
    }
}

fn validate_mask_vec2(param: LookParam, value: [f32; 2]) -> Result<(), ModelError> {
    match param {
        LookParam::MaskCenter => validate_center_value(value),
        LookParam::MaskSize => validate_size_value(value),
        _ => Err(ModelError::InvalidParam(format!(
            "look parameter {param:?} is not a vec2"
        ))),
    }
}

/// True iff `param` names a property on [`Mask`].
pub(crate) fn is_mask_param(param: LookParam) -> bool {
    matches!(
        param,
        LookParam::MaskFeather
            | LookParam::MaskCenter
            | LookParam::MaskSize
            | LookParam::MaskRotation
            | LookParam::MaskRoundness
    )
}

/// Insert or replace a keyframe on one mask property.
pub(crate) fn set_mask_param_keyframe(
    mask: &mut Option<Mask>,
    param: LookParam,
    tick: i64,
    value: ParamValue,
    easing: Easing,
) -> Result<(), ModelError> {
    easing.validate()?;
    match param {
        LookParam::MaskFeather | LookParam::MaskRotation | LookParam::MaskRoundness => {
            let v = expect_scalar(value)?;
            validate_mask_value(param, v)?;
            mask_scalar_mut(mask, param)?.set_keyframe(tick, v, easing);
            Ok(())
        }
        LookParam::MaskCenter | LookParam::MaskSize => {
            let v = expect_vec2(value)?;
            validate_mask_vec2(param, v)?;
            mask_vec2_mut(mask, param)?.set_keyframe(tick, v, easing);
            Ok(())
        }
        _ => Err(ModelError::InvalidParam(format!(
            "look parameter {param:?} is not a mask property"
        ))),
    }
}

/// Remove the keyframe at exactly `tick` on one mask property.
pub(crate) fn remove_mask_param_keyframe(
    mask: &mut Option<Mask>,
    param: LookParam,
    tick: i64,
) -> Result<(), ModelError> {
    let removed = match param {
        LookParam::MaskFeather | LookParam::MaskRotation | LookParam::MaskRoundness => {
            mask_scalar_mut(mask, param)?.remove_keyframe(tick)
        }
        LookParam::MaskCenter | LookParam::MaskSize => {
            mask_vec2_mut(mask, param)?.remove_keyframe(tick)
        }
        _ => {
            return Err(ModelError::InvalidParam(format!(
                "look parameter {param:?} is not a mask property"
            )));
        }
    };
    if removed {
        Ok(())
    } else {
        Err(ModelError::InvalidParam(format!(
            "no {param:?} keyframe at tick {tick}"
        )))
    }
}

/// Replace one mask property with a constant, dropping its keyframes.
pub(crate) fn set_mask_param_constant(
    mask: &mut Option<Mask>,
    param: LookParam,
    value: ParamValue,
) -> Result<(), ModelError> {
    match param {
        LookParam::MaskFeather | LookParam::MaskRotation | LookParam::MaskRoundness => {
            let v = expect_scalar(value)?;
            validate_mask_value(param, v)?;
            mask_scalar_mut(mask, param)?.set_constant(v);
            Ok(())
        }
        LookParam::MaskCenter | LookParam::MaskSize => {
            let v = expect_vec2(value)?;
            validate_mask_vec2(param, v)?;
            mask_vec2_mut(mask, param)?.set_constant(v);
            Ok(())
        }
        _ => Err(ModelError::InvalidParam(format!(
            "look parameter {param:?} is not a mask property"
        ))),
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
