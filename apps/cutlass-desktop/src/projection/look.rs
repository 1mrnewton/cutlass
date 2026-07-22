use super::*;

/// Projected mask fields + geometry keyframe lists for one clip.
pub(super) struct ProjectedMask {
    pub(super) kind: String,
    pub(super) label: String,
    pub(super) invert: bool,
    pub(super) feather: f32,
    pub(super) center: [f32; 2],
    pub(super) size: [f32; 2],
    pub(super) rotation: f32,
    pub(super) roundness: f32,
    pub(super) kf_feather: ModelRc<ParamKeyframe>,
    pub(super) kf_center: ModelRc<ParamKeyframe>,
    pub(super) kf_size: ModelRc<ParamKeyframe>,
    pub(super) kf_rotation: ModelRc<ParamKeyframe>,
    pub(super) kf_roundness: ModelRc<ParamKeyframe>,
}

pub(super) fn clip_mask(clip: &EngineClip, clip_start: i64) -> ProjectedMask {
    let Some(mask) = &clip.mask else {
        let defaults = cutlass_models::Mask::new(cutlass_models::MaskKind::Circle);
        return ProjectedMask {
            kind: String::new(),
            label: String::new(),
            invert: false,
            feather: defaults.feather.sample(0),
            center: defaults.center.sample(0),
            size: defaults.size.sample(0),
            rotation: defaults.rotation.sample(0),
            roundness: defaults.roundness.sample(0),
            kf_feather: empty_keyframes(),
            kf_center: empty_keyframes(),
            kf_size: empty_keyframes(),
            kf_rotation: empty_keyframes(),
            kf_roundness: empty_keyframes(),
        };
    };
    let label = cutlass_models::mask_catalog()
        .iter()
        .find(|s| s.kind == mask.kind)
        .map(|s| s.label.to_string())
        .unwrap_or_else(|| mask.kind.id().to_string());
    ProjectedMask {
        kind: mask.kind.id().to_string(),
        label,
        invert: mask.invert,
        feather: mask.feather.sample(0),
        center: mask.center.sample(0),
        size: mask.size.sample(0),
        rotation: mask.rotation.sample(0),
        roundness: mask.roundness.sample(0),
        kf_feather: keyframes_to_slint(&mask.feather, clip_start, |v| (*v, 0.0)),
        kf_center: keyframes_to_slint(&mask.center, clip_start, |v| (v[0], v[1])),
        kf_size: keyframes_to_slint(&mask.size, clip_start, |v| (v[0], v[1])),
        kf_rotation: keyframes_to_slint(&mask.rotation, clip_start, |v| (*v, 0.0)),
        kf_roundness: keyframes_to_slint(&mask.roundness, clip_start, |v| (*v, 0.0)),
    }
}

/// Projected chroma-key fields + strength/shadow keyframe lists.
pub(super) struct ProjectedChroma {
    pub(super) enabled: bool,
    pub(super) color: Color,
    pub(super) strength: f32,
    pub(super) shadow: f32,
    pub(super) kf_strength: ModelRc<ParamKeyframe>,
    pub(super) kf_shadow: ModelRc<ParamKeyframe>,
}

pub(super) fn clip_chroma(clip: &EngineClip, clip_start: i64) -> ProjectedChroma {
    let Some(chroma) = &clip.chroma_key else {
        return ProjectedChroma {
            enabled: false,
            color: Color::from_rgb_u8(0, 255, 0),
            strength: 0.5,
            shadow: 0.0,
            kf_strength: empty_keyframes(),
            kf_shadow: empty_keyframes(),
        };
    };
    ProjectedChroma {
        enabled: true,
        color: Color::from_rgb_u8(chroma.rgb[0], chroma.rgb[1], chroma.rgb[2]),
        strength: chroma.strength.sample(0),
        shadow: chroma.shadow.sample(0),
        kf_strength: keyframes_to_slint(&chroma.strength, clip_start, |v| (*v, 0.0)),
        kf_shadow: keyframes_to_slint(&chroma.shadow, clip_start, |v| (*v, 0.0)),
    }
}

/// Projected layer-style fields + scalar/vec2 keyframe lists for one clip.
/// Colors are clip-start samples only (no kf lists — AI-only for now).
pub(super) struct ProjectedLayerStyles {
    pub(super) shadow_enabled: bool,
    pub(super) shadow_color: Color,
    pub(super) shadow_offset: [f32; 2],
    pub(super) shadow_blur: f32,
    pub(super) glow_enabled: bool,
    pub(super) glow_color: Color,
    pub(super) glow_radius: f32,
    pub(super) glow_intensity: f32,
    pub(super) outline_enabled: bool,
    pub(super) outline_color: Color,
    pub(super) outline_width: f32,
    pub(super) background_enabled: bool,
    pub(super) background_color: Color,
    pub(super) background_padding: f32,
    pub(super) background_radius: f32,
    pub(super) kf_shadow_offset: ModelRc<ParamKeyframe>,
    pub(super) kf_shadow_blur: ModelRc<ParamKeyframe>,
    pub(super) kf_glow_radius: ModelRc<ParamKeyframe>,
    pub(super) kf_glow_intensity: ModelRc<ParamKeyframe>,
    pub(super) kf_outline_width: ModelRc<ParamKeyframe>,
    pub(super) kf_background_padding: ModelRc<ParamKeyframe>,
    pub(super) kf_background_radius: ModelRc<ParamKeyframe>,
}

pub(super) fn clip_layer_styles(clip: &EngineClip, clip_start: i64) -> ProjectedLayerStyles {
    let shadow_enabled = clip.styles.shadow.is_some();
    let shadow = clip.styles.shadow.clone().unwrap_or_default();
    let glow_enabled = clip.styles.glow.is_some();
    let glow = clip.styles.glow.clone().unwrap_or_default();
    let outline_enabled = clip.styles.outline.is_some();
    let outline = clip.styles.outline.clone().unwrap_or_default();
    let background_enabled = clip.styles.background.is_some();
    let background = clip.styles.background.clone().unwrap_or_default();

    ProjectedLayerStyles {
        shadow_enabled,
        shadow_color: rgba_color(shadow.rgba.sample(0)),
        shadow_offset: shadow.offset.sample(0),
        shadow_blur: shadow.blur.sample(0),
        glow_enabled,
        glow_color: rgba_color(glow.rgba.sample(0)),
        glow_radius: glow.radius.sample(0),
        glow_intensity: glow.intensity.sample(0),
        outline_enabled,
        outline_color: rgba_color(outline.rgba.sample(0)),
        outline_width: outline.width.sample(0),
        background_enabled,
        background_color: rgba_color(background.rgba.sample(0)),
        background_padding: background.padding.sample(0),
        background_radius: background.radius.sample(0),
        kf_shadow_offset: clip
            .styles
            .shadow
            .as_ref()
            .map_or_else(empty_keyframes, |s| {
                keyframes_to_slint(&s.offset, clip_start, |v| (v[0], v[1]))
            }),
        kf_shadow_blur: clip
            .styles
            .shadow
            .as_ref()
            .map_or_else(empty_keyframes, |s| {
                keyframes_to_slint(&s.blur, clip_start, |v| (*v, 0.0))
            }),
        kf_glow_radius: clip.styles.glow.as_ref().map_or_else(empty_keyframes, |s| {
            keyframes_to_slint(&s.radius, clip_start, |v| (*v, 0.0))
        }),
        kf_glow_intensity: clip.styles.glow.as_ref().map_or_else(empty_keyframes, |s| {
            keyframes_to_slint(&s.intensity, clip_start, |v| (*v, 0.0))
        }),
        kf_outline_width: clip
            .styles
            .outline
            .as_ref()
            .map_or_else(empty_keyframes, |s| {
                keyframes_to_slint(&s.width, clip_start, |v| (*v, 0.0))
            }),
        kf_background_padding: clip
            .styles
            .background
            .as_ref()
            .map_or_else(empty_keyframes, |s| {
                keyframes_to_slint(&s.padding, clip_start, |v| (*v, 0.0))
            }),
        kf_background_radius: clip
            .styles
            .background
            .as_ref()
            .map_or_else(empty_keyframes, |s| {
                keyframes_to_slint(&s.radius, clip_start, |v| (*v, 0.0))
            }),
    }
}

pub(super) fn clip_filter(clip: &EngineClip) -> (String, String, f32) {
    let Some(filter) = &clip.filter else {
        return (String::new(), String::new(), 0.0);
    };
    let label = cutlass_models::filter_spec(&filter.id)
        .map(|s| s.label)
        .unwrap_or(filter.id.as_str())
        .to_string();
    (filter.id.clone(), label, filter.intensity.sample(0))
}

/// Project a clip's `.cube` LUT as (id, label, intensity). The id is the
/// file's stem — for catalog downloads that IS the catalog id (the LUT
/// worker names files `<id>.cube`) — and the label prettifies it
/// (`cutlass-vivid` → "Vivid").
pub(super) fn clip_lut(clip: &EngineClip) -> (String, String, String, f32) {
    let Some(lut) = &clip.lut else {
        return (String::new(), String::new(), String::new(), 0.0);
    };
    let id = std::path::Path::new(&lut.path)
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| lut.path.clone());
    (
        id.clone(),
        lut_label(&id),
        lut.path.clone(),
        lut.intensity.sample(0),
    )
}

/// Human label for a LUT id/stem: strip the first-party prefix, split on
/// separators, capitalize words (`cutlass-teal_orange` → "Teal Orange").
pub(crate) fn lut_label(id: &str) -> String {
    let base = id.strip_prefix("cutlass-").unwrap_or(id);
    base.split(['-', '_', ' '])
        .filter(|word| !word.is_empty())
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Projected look-animation slot: id, label, knobs, and which knobs the
/// catalog exposes for the inspector sliders.
pub(super) struct ClipAnimationProj {
    pub(super) id: String,
    pub(super) label: String,
    pub(super) speed: f32,
    pub(super) intensity: f32,
    pub(super) stagger: f32,
    pub(super) has_speed: bool,
    pub(super) has_intensity: bool,
    pub(super) has_stagger: bool,
}

pub(super) fn clip_animation(
    animation: Option<&cutlass_models::AnimationRef>,
) -> ClipAnimationProj {
    let Some(animation) = animation else {
        return ClipAnimationProj {
            id: String::new(),
            label: String::new(),
            speed: 1.0,
            intensity: 1.0,
            stagger: 1.0,
            has_speed: false,
            has_intensity: false,
            has_stagger: false,
        };
    };
    let spec = cutlass_models::animation_spec(&animation.id);
    let label = spec
        .map(|s| s.label)
        .unwrap_or(animation.id.as_str())
        .to_string();
    let knobs = spec
        .map(|s| s.knobs)
        .unwrap_or(cutlass_models::AnimationKnobs {
            speed: false,
            intensity: false,
            stagger: false,
        });
    ClipAnimationProj {
        id: animation.id.clone(),
        label,
        speed: animation.speed,
        intensity: animation.intensity,
        stagger: animation.stagger,
        has_speed: knobs.speed,
        has_intensity: knobs.intensity,
        has_stagger: knobs.stagger,
    }
}
