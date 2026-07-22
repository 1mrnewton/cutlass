use super::*;

/// Lower the agent-facing property selector to the model selector. Effect
/// parameters retain the wire's stable catalog names and resolve to their
/// instance slot here.
pub(super) fn clip_param(
    param: &WireClipParam,
    clip: &Clip,
    wire_clip: u64,
) -> Result<ClipParam, Rejection> {
    use crate::wire::{WireLookParam, WireShapeParam, WireStyleParam, WireTextParam};

    Ok(match param {
        WireClipParam::Position => ClipParam::Position,
        WireClipParam::AnchorPoint => ClipParam::AnchorPoint,
        WireClipParam::Scale => ClipParam::Scale,
        WireClipParam::Rotation => ClipParam::Rotation,
        WireClipParam::Opacity => ClipParam::Opacity,
        WireClipParam::Volume => ClipParam::Volume,
        WireClipParam::Speed => ClipParam::Speed,
        WireClipParam::Effect { index, param } => {
            let (effect, slot, _) = effect_param_slot(clip, *index, param, wire_clip)?;
            ClipParam::Effect {
                effect: effect as u32,
                param: slot as u32,
            }
        }
        WireClipParam::Shape { param } => ClipParam::Shape {
            param: match param {
                WireShapeParam::Width => cutlass_models::ShapeParam::Width,
                WireShapeParam::Height => cutlass_models::ShapeParam::Height,
                WireShapeParam::CornerRadius => cutlass_models::ShapeParam::CornerRadius,
                WireShapeParam::InnerRatio => cutlass_models::ShapeParam::InnerRatio,
                WireShapeParam::Fill => cutlass_models::ShapeParam::Fill,
                WireShapeParam::StrokeColor => cutlass_models::ShapeParam::StrokeColor,
                WireShapeParam::StrokeWidth => cutlass_models::ShapeParam::StrokeWidth,
            },
        },
        WireClipParam::Text { param } => ClipParam::Text {
            param: match param {
                WireTextParam::Size => cutlass_models::TextParam::Size,
                WireTextParam::Fill => cutlass_models::TextParam::Fill,
                WireTextParam::LetterSpacing => cutlass_models::TextParam::LetterSpacing,
                WireTextParam::LineSpacing => cutlass_models::TextParam::LineSpacing,
                WireTextParam::StrokeWidth => cutlass_models::TextParam::StrokeWidth,
                WireTextParam::StrokeColor => cutlass_models::TextParam::StrokeColor,
                WireTextParam::ShadowBlur => cutlass_models::TextParam::ShadowBlur,
                WireTextParam::ShadowDistance => cutlass_models::TextParam::ShadowDistance,
                WireTextParam::ShadowColor => cutlass_models::TextParam::ShadowColor,
            },
        },
        WireClipParam::Look { param } => ClipParam::Look {
            param: match param {
                WireLookParam::FilterIntensity => cutlass_models::LookParam::FilterIntensity,
                WireLookParam::LutIntensity => cutlass_models::LookParam::LutIntensity,
                WireLookParam::AdjustBrightness => cutlass_models::LookParam::AdjustBrightness,
                WireLookParam::AdjustContrast => cutlass_models::LookParam::AdjustContrast,
                WireLookParam::AdjustSaturation => cutlass_models::LookParam::AdjustSaturation,
                WireLookParam::AdjustExposure => cutlass_models::LookParam::AdjustExposure,
                WireLookParam::AdjustTemperature => cutlass_models::LookParam::AdjustTemperature,
                WireLookParam::AdjustTint => cutlass_models::LookParam::AdjustTint,
                WireLookParam::AdjustHue => cutlass_models::LookParam::AdjustHue,
                WireLookParam::AdjustHighlights => cutlass_models::LookParam::AdjustHighlights,
                WireLookParam::AdjustShadows => cutlass_models::LookParam::AdjustShadows,
                WireLookParam::AdjustSharpness => cutlass_models::LookParam::AdjustSharpness,
                WireLookParam::AdjustVignette => cutlass_models::LookParam::AdjustVignette,
                WireLookParam::MaskFeather => cutlass_models::LookParam::MaskFeather,
                WireLookParam::MaskCenter => cutlass_models::LookParam::MaskCenter,
                WireLookParam::MaskSize => cutlass_models::LookParam::MaskSize,
                WireLookParam::MaskRotation => cutlass_models::LookParam::MaskRotation,
                WireLookParam::MaskRoundness => cutlass_models::LookParam::MaskRoundness,
                WireLookParam::ChromaStrength => cutlass_models::LookParam::ChromaStrength,
                WireLookParam::ChromaShadow => cutlass_models::LookParam::ChromaShadow,
            },
        },
        WireClipParam::Style { param } => ClipParam::Style {
            param: match param {
                WireStyleParam::ShadowColor => cutlass_models::StyleParam::ShadowColor,
                WireStyleParam::ShadowOffset => cutlass_models::StyleParam::ShadowOffset,
                WireStyleParam::ShadowBlur => cutlass_models::StyleParam::ShadowBlur,
                WireStyleParam::GlowColor => cutlass_models::StyleParam::GlowColor,
                WireStyleParam::GlowRadius => cutlass_models::StyleParam::GlowRadius,
                WireStyleParam::GlowIntensity => cutlass_models::StyleParam::GlowIntensity,
                WireStyleParam::OutlineColor => cutlass_models::StyleParam::OutlineColor,
                WireStyleParam::OutlineWidth => cutlass_models::StyleParam::OutlineWidth,
                WireStyleParam::BackgroundColor => cutlass_models::StyleParam::BackgroundColor,
                WireStyleParam::BackgroundPadding => cutlass_models::StyleParam::BackgroundPadding,
                WireStyleParam::BackgroundRadius => cutlass_models::StyleParam::BackgroundRadius,
            },
        },
    })
}

pub(super) fn unit_slider(value: f64, name: &str) -> Result<f32, Rejection> {
    if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
        return Err(Rejection::new(format!(
            "{name} must be between -1 and 1 (got {value})"
        )));
    }
    Ok(value as f32)
}

/// One-directional adjust sliders (`sharpness` / `vignette`): `0..=1`.
pub(super) fn unit_positive_slider(value: f64, name: &str) -> Result<f32, Rejection> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(Rejection::new(format!(
            "{name} must be between 0 and 1 (got {value})"
        )));
    }
    Ok(value as f32)
}

pub(super) fn lower_mask_kind(kind: WireMaskKind) -> MaskKind {
    match kind {
        WireMaskKind::Linear => MaskKind::Linear,
        WireMaskKind::Mirror => MaskKind::Mirror,
        WireMaskKind::Circle => MaskKind::Circle,
        WireMaskKind::Rectangle => MaskKind::Rectangle,
        WireMaskKind::Heart => MaskKind::Heart,
        WireMaskKind::Star => MaskKind::Star,
    }
}

pub(super) fn lower_mask(wire: &WireMask) -> Result<Mask, Rejection> {
    let feather = wire.feather.unwrap_or(0.0);
    if !feather.is_finite() || !(0.0..=1.0).contains(&feather) {
        return Err(Rejection::new(format!(
            "mask feather must be between 0 and 1 (got {feather})"
        )));
    }
    let mut mask = Mask::new(lower_mask_kind(wire.kind));
    mask.feather = Param::Constant(feather as f32);
    mask.invert = wire.invert.unwrap_or(false);
    if let Some(center) = wire.center {
        mask.center = Param::Constant(center);
    }
    if let Some(size) = wire.size {
        mask.size = Param::Constant(size);
    }
    if let Some(rotation) = wire.rotation {
        mask.rotation = Param::Constant(rotation);
    }
    if let Some(roundness) = wire.roundness {
        mask.roundness = Param::Constant(roundness);
    }
    // Model `Mask::validate` range-checks every geometry field.
    mask.validate().map_err(|e| Rejection::new(e.to_string()))?;
    Ok(mask)
}

pub(super) fn lower_chroma(wire: &WireChromaKey) -> Result<ChromaKey, Rejection> {
    let strength = wire.strength.unwrap_or(0.0);
    let shadow = wire.shadow.unwrap_or(0.0);
    for (name, value) in [("strength", strength), ("shadow", shadow)] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(Rejection::new(format!(
                "chroma {name} must be between 0 and 1 (got {value})"
            )));
        }
    }
    let chroma = ChromaKey {
        rgb: wire.rgb,
        strength: (strength as f32).into(),
        shadow: (shadow as f32).into(),
    };
    chroma
        .validate()
        .map_err(|e| Rejection::new(e.to_string()))?;
    Ok(chroma)
}

pub(super) fn lower_stabilize(level: WireStabilizeLevel) -> StabilizeLevel {
    match level {
        WireStabilizeLevel::Recommended => StabilizeLevel::Recommended,
        WireStabilizeLevel::Smooth => StabilizeLevel::Smooth,
        WireStabilizeLevel::MaxSmooth => StabilizeLevel::MaxSmooth,
    }
}

pub(super) fn lower_blend_mode(mode: WireBlendMode) -> BlendMode {
    match mode {
        WireBlendMode::Normal => BlendMode::Normal,
        WireBlendMode::Darken => BlendMode::Darken,
        WireBlendMode::Multiply => BlendMode::Multiply,
        WireBlendMode::ColorBurn => BlendMode::ColorBurn,
        WireBlendMode::Lighten => BlendMode::Lighten,
        WireBlendMode::Screen => BlendMode::Screen,
        WireBlendMode::ColorDodge => BlendMode::ColorDodge,
        WireBlendMode::Add => BlendMode::Add,
        WireBlendMode::Overlay => BlendMode::Overlay,
        WireBlendMode::SoftLight => BlendMode::SoftLight,
        WireBlendMode::HardLight => BlendMode::HardLight,
        WireBlendMode::Difference => BlendMode::Difference,
        WireBlendMode::Exclusion => BlendMode::Exclusion,
    }
}

pub(super) fn lower_filter(wire: &crate::wire::WireFilter) -> Result<Filter, Rejection> {
    let intensity = wire.intensity.unwrap_or(0.8);
    if !intensity.is_finite() || !(0.0..=1.0).contains(&intensity) {
        return Err(Rejection::new(format!(
            "filter intensity must be between 0 and 1 (got {intensity})"
        )));
    }
    let filter = Filter {
        id: wire.id.clone(),
        intensity: (intensity as f32).into(),
    };
    filter.validate().map_err(|e| {
        let ids = filter_catalog()
            .iter()
            .map(|s| s.id)
            .collect::<Vec<_>>()
            .join(", ");
        if filter_spec(&wire.id).is_none() {
            Rejection::new(format!(
                "unknown filter '{}'; available filters: {ids}",
                wire.id
            ))
        } else {
            Rejection::new(e.to_string())
        }
    })?;
    Ok(filter)
}

pub(super) fn lower_animation_slot(slot: WireAnimationSlot) -> AnimationSlot {
    match slot {
        WireAnimationSlot::In => AnimationSlot::In,
        WireAnimationSlot::Out => AnimationSlot::Out,
        WireAnimationSlot::Combo => AnimationSlot::Combo,
    }
}

pub(super) fn lower_audio_role(role: WireAudioRole) -> AudioRole {
    match role {
        WireAudioRole::Music => AudioRole::Music,
        WireAudioRole::Sfx => AudioRole::Sfx,
        WireAudioRole::Voiceover => AudioRole::Voiceover,
        WireAudioRole::Extracted => AudioRole::Extracted,
    }
}

/// Lower a wire generator. When replacing the content of an existing text
/// clip, the current style is preserved (the agent edits words, not looks).
pub(super) fn lower_generator(wire: &WireGenerator, current: Option<&Generator>) -> Generator {
    match wire {
        WireGenerator::Text { content } => {
            let style = match current {
                Some(Generator::Text { style, .. }) => style.clone(),
                _ => Default::default(),
            };
            Generator::Text {
                content: content.clone(),
                style,
            }
        }
        WireGenerator::Solid { rgba } => Generator::SolidColor { rgba: *rgba },
        WireGenerator::Shape {
            shape,
            rgba,
            width,
            height,
        } => {
            let (shape_w, shape_h, corner_radius, stroke) = match current {
                Some(Generator::Shape {
                    width: w,
                    height: h,
                    corner_radius,
                    stroke,
                    ..
                }) => (
                    w.sample(0),
                    h.sample(0),
                    corner_radius.clone(),
                    stroke.clone(),
                ),
                _ => (
                    cutlass_models::SHAPE_DROP_WIDTH,
                    cutlass_models::SHAPE_DROP_HEIGHT,
                    Param::Constant(0.0),
                    None,
                ),
            };
            Generator::Shape {
                shape: match shape {
                    WireShape::Rectangle => cutlass_models::Shape::Rectangle,
                    WireShape::Ellipse => cutlass_models::Shape::Ellipse,
                },
                rgba: Param::Constant(*rgba),
                width: Param::Constant(width.unwrap_or(shape_w)),
                height: Param::Constant(height.unwrap_or(shape_h)),
                corner_radius,
                stroke,
            }
        }
    }
}

pub(super) fn generated_content(clip: &Clip) -> Option<&Generator> {
    match &clip.content {
        cutlass_models::ClipSource::Generated(g) => Some(g),
        cutlass_models::ClipSource::Media { .. } => None,
    }
}
