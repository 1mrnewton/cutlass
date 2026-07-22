use super::*;

pub(super) fn unit_slider(value: f64, name: &str) -> Result<f32, Rejection> {
    if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
        return Err(Rejection::new(format!(
            "{name} must be between -1 and 1 (got {value})"
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
    let mask = Mask {
        kind: lower_mask_kind(wire.kind),
        feather: (feather as f32).into(),
        invert: wire.invert.unwrap_or(false),
    };
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
