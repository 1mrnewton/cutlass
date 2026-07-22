use cutlass_compositor::{
    BlendMode, LayerChromaKey, LayerEffects, LayerMask, PassInstance, mask_kind,
};
use cutlass_models::MaskKind;

use crate::scene::ResolvedPass;

/// Packed effect chain: catalog-static ids plus owned parameter values.
///
/// [`PassInstance`] wants a `&'static str` id and borrowed params. Ids are
/// interned against the compositor's static effect catalog (unknown ids are
/// dropped here — they'd dispatch as no-op passthroughs anyway), and params
/// stay owned so the instances built by [`EffectChain::instances`] borrow from
/// this store for the duration of one render instead of leaking.
pub(super) struct EffectChain {
    passes: Vec<(&'static str, Vec<f32>)>,
}

impl EffectChain {
    pub(super) fn instances(&self) -> Vec<PassInstance<'_>> {
        self.passes
            .iter()
            .map(|(id, params)| PassInstance { id, params })
            .collect()
    }
}

pub(super) fn pack_effects(resolved: &[ResolvedPass]) -> EffectChain {
    let passes = resolved
        .iter()
        .filter_map(|pass| {
            let id = cutlass_compositor::effect_descriptors()
                .iter()
                .find(|d| d.id == pass.id)?
                .id;
            Some((id, pass.params.clone()))
        })
        .collect();
    EffectChain { passes }
}

pub(super) fn layer_effects(layer: &crate::scene::SceneLayer) -> LayerEffects {
    let mask = layer.mask.map(|m| LayerMask {
        kind: mask_kind_id(m.kind),
        feather: m.feather,
        invert: u32::from(m.invert),
    });
    let chroma_key = layer.chroma_key.map(|c| LayerChromaKey {
        rgb: [
            f32::from(c.rgb[0]) / 255.0,
            f32::from(c.rgb[1]) / 255.0,
            f32::from(c.rgb[2]) / 255.0,
        ],
        strength: c.strength,
        shadow: c.shadow,
    });
    LayerEffects { mask, chroma_key }
}

fn mask_kind_id(kind: MaskKind) -> u32 {
    match kind {
        MaskKind::Linear => mask_kind::LINEAR,
        MaskKind::Mirror => mask_kind::MIRROR,
        MaskKind::Circle => mask_kind::CIRCLE,
        MaskKind::Rectangle => mask_kind::RECTANGLE,
        MaskKind::Heart => mask_kind::HEART,
        MaskKind::Star => mask_kind::STAR,
    }
}

/// Map a model blend mode onto the compositor's GPU blend id.
pub(super) fn blend_mode(mode: cutlass_models::BlendMode) -> BlendMode {
    match mode {
        cutlass_models::BlendMode::Normal => BlendMode::Normal,
        cutlass_models::BlendMode::Darken => BlendMode::Darken,
        cutlass_models::BlendMode::Multiply => BlendMode::Multiply,
        cutlass_models::BlendMode::ColorBurn => BlendMode::ColorBurn,
        cutlass_models::BlendMode::Lighten => BlendMode::Lighten,
        cutlass_models::BlendMode::Screen => BlendMode::Screen,
        cutlass_models::BlendMode::ColorDodge => BlendMode::ColorDodge,
        cutlass_models::BlendMode::Add => BlendMode::Add,
        cutlass_models::BlendMode::Overlay => BlendMode::Overlay,
        cutlass_models::BlendMode::SoftLight => BlendMode::SoftLight,
        cutlass_models::BlendMode::HardLight => BlendMode::HardLight,
        cutlass_models::BlendMode::Difference => BlendMode::Difference,
        cutlass_models::BlendMode::Exclusion => BlendMode::Exclusion,
    }
}
