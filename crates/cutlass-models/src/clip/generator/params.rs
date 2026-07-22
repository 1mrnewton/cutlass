//! Param routing for [`Generator`]: resolving a [`ShapeParam`] / [`TextParam`]
//! name to the [`Param`] it addresses and applying keyframe / constant edits.

use crate::error::ModelError;
use crate::param::{Easing, Param};

use crate::clip::transform::{ParamValue, ShapeParam, TextParam};

use super::{
    Generator, Shape, validate_corner_radius, validate_shape_dim, validate_stroke_width,
    validate_unit_fraction,
};

impl Generator {
    /// Insert or replace a keyframe on one animatable shape property.
    /// Errors when the generator is not a shape, the property does not
    /// apply to its kind (e.g. `InnerRatio` on a rectangle, stroke params
    /// with no stroke set), or the value is out of range.
    pub fn set_shape_param_keyframe(
        &mut self,
        param: ShapeParam,
        tick: i64,
        value: ParamValue,
        easing: Easing,
    ) -> Result<(), ModelError> {
        easing.validate()?;
        self.with_shape_param(param, |target, kind| match kind {
            ShapeParamKind::Scalar { validate } => {
                let v = value.scalar()?;
                validate(v)?;
                target.scalar()?.set_keyframe(tick, v, easing);
                Ok(())
            }
            ShapeParamKind::Color => {
                let v = value.color()?;
                target.color()?.set_keyframe(tick, v, easing);
                Ok(())
            }
        })
    }

    /// Remove the keyframe at exactly `tick` on one shape property. Errors
    /// when no keyframe sits there.
    pub fn remove_shape_param_keyframe(
        &mut self,
        param: ShapeParam,
        tick: i64,
    ) -> Result<(), ModelError> {
        self.with_shape_param(param, |target, kind| {
            let removed = match kind {
                ShapeParamKind::Scalar { .. } => target.scalar()?.remove_keyframe(tick),
                ShapeParamKind::Color => target.color()?.remove_keyframe(tick),
            };
            if removed {
                Ok(())
            } else {
                Err(ModelError::InvalidParam(format!(
                    "no {param:?} keyframe at tick {tick}"
                )))
            }
        })
    }

    /// Replace one shape property with a constant, dropping its keyframes.
    pub fn set_shape_param_constant(
        &mut self,
        param: ShapeParam,
        value: ParamValue,
    ) -> Result<(), ModelError> {
        self.with_shape_param(param, |target, kind| match kind {
            ShapeParamKind::Scalar { validate } => {
                let v = value.scalar()?;
                validate(v)?;
                target.scalar()?.set_constant(v);
                Ok(())
            }
            ShapeParamKind::Color => {
                target.color()?.set_constant(value.color()?);
                Ok(())
            }
        })
    }

    /// Insert or replace a keyframe on one animatable text style property.
    pub fn set_text_param_keyframe(
        &mut self,
        param: TextParam,
        tick: i64,
        value: ParamValue,
        easing: Easing,
    ) -> Result<(), ModelError> {
        easing.validate()?;
        self.with_text_param(param, |target, kind| match kind {
            TextParamKind::Scalar { validate } => {
                let v = value.scalar()?;
                validate(v)?;
                target.scalar()?.set_keyframe(tick, v, easing);
                Ok(())
            }
            TextParamKind::Color => {
                target.color()?.set_keyframe(tick, value.color()?, easing);
                Ok(())
            }
        })
    }

    /// Remove the keyframe at exactly `tick` from a text style property.
    pub fn remove_text_param_keyframe(
        &mut self,
        param: TextParam,
        tick: i64,
    ) -> Result<(), ModelError> {
        self.with_text_param(param, |target, kind| {
            let removed = match kind {
                TextParamKind::Scalar { .. } => target.scalar()?.remove_keyframe(tick),
                TextParamKind::Color => target.color()?.remove_keyframe(tick),
            };
            removed.then_some(()).ok_or_else(|| {
                ModelError::InvalidParam(format!("no {param:?} keyframe at tick {tick}"))
            })
        })
    }

    /// Replace one text style property with a constant, dropping keyframes.
    pub fn set_text_param_constant(
        &mut self,
        param: TextParam,
        value: ParamValue,
    ) -> Result<(), ModelError> {
        self.with_text_param(param, |target, kind| match kind {
            TextParamKind::Scalar { validate } => {
                let v = value.scalar()?;
                validate(v)?;
                target.scalar()?.set_constant(v);
                Ok(())
            }
            TextParamKind::Color => {
                target.color()?.set_constant(value.color()?);
                Ok(())
            }
        })
    }

    /// Resolve `param` to the [`Param`] it names on this generator and run
    /// `f` on it — the single routing point for the three mutators above.
    fn with_shape_param<R>(
        &mut self,
        param: ShapeParam,
        f: impl FnOnce(ShapeParamTarget<'_>, ShapeParamKind) -> Result<R, ModelError>,
    ) -> Result<R, ModelError> {
        let Generator::Shape {
            shape,
            rgba,
            width,
            height,
            corner_radius,
            stroke,
        } = self
        else {
            return Err(ModelError::InvalidParam(
                "shape parameters apply only to shape generator clips".into(),
            ));
        };
        let scalar =
            |validate: fn(f32) -> Result<(), ModelError>| ShapeParamKind::Scalar { validate };
        match param {
            ShapeParam::Width => f(ShapeParamTarget::Scalar(width), scalar(validate_shape_dim)),
            ShapeParam::Height => f(ShapeParamTarget::Scalar(height), scalar(validate_shape_dim)),
            ShapeParam::CornerRadius => f(
                ShapeParamTarget::Scalar(corner_radius),
                scalar(validate_corner_radius),
            ),
            ShapeParam::Fill => f(ShapeParamTarget::Color(rgba), ShapeParamKind::Color),
            ShapeParam::InnerRatio => match shape {
                Shape::Star { inner_ratio, .. } => f(
                    ShapeParamTarget::Scalar(inner_ratio),
                    scalar(|v| validate_unit_fraction(v, "star inner_ratio")),
                ),
                _ => Err(ModelError::InvalidParam(
                    "inner_ratio applies only to star shapes".into(),
                )),
            },
            ShapeParam::StrokeWidth | ShapeParam::StrokeColor => match stroke {
                Some(s) => match param {
                    ShapeParam::StrokeWidth => f(
                        ShapeParamTarget::Scalar(&mut s.width),
                        scalar(validate_stroke_width),
                    ),
                    _ => f(ShapeParamTarget::Color(&mut s.rgba), ShapeParamKind::Color),
                },
                None => Err(ModelError::InvalidParam(
                    "shape has no stroke — set one via SetGenerator first".into(),
                )),
            },
        }
    }

    fn with_text_param<R>(
        &mut self,
        param: TextParam,
        f: impl FnOnce(TextParamTarget<'_>, TextParamKind) -> Result<R, ModelError>,
    ) -> Result<R, ModelError> {
        let Generator::Text { style, .. } = self else {
            return Err(ModelError::InvalidParam(
                "text parameters apply only to text generator clips".into(),
            ));
        };
        let scalar =
            |validate: fn(f32) -> Result<(), ModelError>| TextParamKind::Scalar { validate };
        match param {
            TextParam::Size => f(
                TextParamTarget::Scalar(&mut style.size),
                scalar(|v| {
                    validate_text_range(
                        v,
                        f32::EPSILON,
                        crate::clip::text::MAX_TEXT_SIZE,
                        "text size",
                    )
                }),
            ),
            TextParam::Fill => f(
                TextParamTarget::Color(&mut style.fill),
                TextParamKind::Color,
            ),
            TextParam::LetterSpacing => f(
                TextParamTarget::Scalar(&mut style.letter_spacing),
                scalar(validate_text_letter_spacing),
            ),
            TextParam::LineSpacing => f(
                TextParamTarget::Scalar(&mut style.line_spacing),
                scalar(|v| {
                    validate_text_range(
                        v,
                        f32::EPSILON,
                        crate::clip::text::MAX_TEXT_LINE_SPACING,
                        "text line spacing",
                    )
                }),
            ),
            TextParam::StrokeWidth | TextParam::StrokeColor => match &mut style.stroke {
                Some(stroke) => match param {
                    TextParam::StrokeWidth => f(
                        TextParamTarget::Scalar(&mut stroke.width),
                        scalar(|v| {
                            validate_text_range(
                                v,
                                0.0,
                                crate::clip::text::MAX_TEXT_STROKE_WIDTH,
                                "text stroke width",
                            )
                        }),
                    ),
                    _ => f(
                        TextParamTarget::Color(&mut stroke.rgba),
                        TextParamKind::Color,
                    ),
                },
                None => Err(ModelError::InvalidParam(
                    "text has no stroke — set one via SetGenerator first".into(),
                )),
            },
            TextParam::ShadowBlur | TextParam::ShadowDistance | TextParam::ShadowColor => {
                match &mut style.shadow {
                    Some(shadow) => match param {
                        TextParam::ShadowBlur => f(
                            TextParamTarget::Scalar(&mut shadow.blur),
                            scalar(|v| validate_text_range(v, 0.0, 1.0, "text shadow blur")),
                        ),
                        TextParam::ShadowDistance => f(
                            TextParamTarget::Scalar(&mut shadow.distance),
                            scalar(|v| {
                                validate_text_range(
                                    v,
                                    0.0,
                                    crate::clip::text::MAX_TEXT_SHADOW_DISTANCE,
                                    "text shadow distance",
                                )
                            }),
                        ),
                        _ => f(
                            TextParamTarget::Color(&mut shadow.rgba),
                            TextParamKind::Color,
                        ),
                    },
                    None => Err(ModelError::InvalidParam(
                        "text has no shadow — set one via SetGenerator first".into(),
                    )),
                }
            }
        }
    }
}

/// A mutable reference to one animatable shape property, typed by value kind.
enum ShapeParamTarget<'a> {
    Scalar(&'a mut Param<f32>),
    Color(&'a mut Param<[u8; 4]>),
}

enum TextParamTarget<'a> {
    Scalar(&'a mut Param<f32>),
    Color(&'a mut Param<[u8; 4]>),
}

impl<'a> TextParamTarget<'a> {
    fn scalar(self) -> Result<&'a mut Param<f32>, ModelError> {
        match self {
            Self::Scalar(p) => Ok(p),
            Self::Color(_) => Err(ModelError::InvalidParam(
                "expected a scalar value for this text parameter".into(),
            )),
        }
    }

    fn color(self) -> Result<&'a mut Param<[u8; 4]>, ModelError> {
        match self {
            Self::Color(p) => Ok(p),
            Self::Scalar(_) => Err(ModelError::InvalidParam(
                "expected a color value for this text parameter".into(),
            )),
        }
    }
}

impl<'a> ShapeParamTarget<'a> {
    fn scalar(self) -> Result<&'a mut Param<f32>, ModelError> {
        match self {
            ShapeParamTarget::Scalar(p) => Ok(p),
            ShapeParamTarget::Color(_) => Err(ModelError::InvalidParam(
                "expected a color value for this shape parameter".into(),
            )),
        }
    }

    fn color(self) -> Result<&'a mut Param<[u8; 4]>, ModelError> {
        match self {
            ShapeParamTarget::Color(p) => Ok(p),
            ShapeParamTarget::Scalar(_) => Err(ModelError::InvalidParam(
                "expected a scalar value for this shape parameter".into(),
            )),
        }
    }
}

/// Value kind (and range rule) of one [`ShapeParam`].
enum ShapeParamKind {
    Scalar {
        validate: fn(f32) -> Result<(), ModelError>,
    },
    Color,
}

enum TextParamKind {
    Scalar {
        validate: fn(f32) -> Result<(), ModelError>,
    },
    Color,
}

fn validate_text_range(v: f32, min: f32, max: f32, what: &str) -> Result<(), ModelError> {
    if !v.is_finite() || !(min..=max).contains(&v) {
        return Err(ModelError::InvalidParam(format!(
            "{what} must be in {min}..={max}"
        )));
    }
    Ok(())
}

fn validate_text_letter_spacing(v: f32) -> Result<(), ModelError> {
    if !v.is_finite() || v.abs() > crate::clip::text::MAX_TEXT_LETTER_SPACING {
        return Err(ModelError::InvalidParam(format!(
            "text letter spacing must be finite and within -{}..={} reference px",
            crate::clip::text::MAX_TEXT_LETTER_SPACING,
            crate::clip::text::MAX_TEXT_LETTER_SPACING,
        )));
    }
    Ok(())
}
