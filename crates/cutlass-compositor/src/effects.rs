//! Effect registry: maps effect ids to ordered GPU passes.
//!
//! Effects are **data** (see the v1 roadmap M4 invariant): the model stores
//! `{effect_id, params}` and this crate owns the WGSL that renders each id.
//! An effect is a list of full-screen fragment passes (sharing
//! [`effect_header.wgsl`]); a multi-pass effect (e.g. separable blur) repeats
//! the same fragment with a different `pass_index`.
//!
//! [`effect_descriptors`] is the canonical list of what the compositor can
//! render and the slot order of each effect's parameters. The
//! `cutlass-models` effect catalog (defaults / ranges / display names, used
//! for validation and the UI) is drift-checked against this list from the
//! engine, so the two never disagree on ids or parameter names.

use std::collections::HashMap;

/// Number of scalar parameter slots packed into the effect uniform (`p0..p3`,
/// `p1.x..p1.w`). Effects use as many as they need; the rest stay zero.
pub const EFFECT_PARAM_SLOTS: usize = 8;

/// A renderable effect: its stable id and the ordered names of its scalar
/// parameters (slot 0 first). The engine packs model parameters into the
/// uniform in this order.
#[derive(Debug, Clone, Copy)]
pub struct EffectDescriptor {
    pub id: &'static str,
    pub params: &'static [&'static str],
}

/// Build-time blueprint: an id, its parameter slot names, and the fragment
/// source of each pass (concatenated after the shared header). Two passes
/// referencing the same source compile to one pipeline (deduped by source).
struct EffectBlueprint {
    id: &'static str,
    params: &'static [&'static str],
    passes: &'static [&'static str],
}

const BLUR_FS: &str = include_str!("../shaders/effect_blur.wgsl");
const VIGNETTE_FS: &str = include_str!("../shaders/effect_vignette.wgsl");
const SHARPEN_FS: &str = include_str!("../shaders/effect_sharpen.wgsl");
const PIXELATE_FS: &str = include_str!("../shaders/effect_pixelate.wgsl");
const GLITCH_FS: &str = include_str!("../shaders/effect_glitch.wgsl");
const CHROMATIC_FS: &str = include_str!("../shaders/effect_chromatic.wgsl");
const GRAIN_FS: &str = include_str!("../shaders/effect_grain.wgsl");
const GLOW_FS: &str = include_str!("../shaders/effect_glow.wgsl");
const ZOOM_BLUR_FS: &str = include_str!("../shaders/effect_zoom_blur.wgsl");
const MIRROR_FS: &str = include_str!("../shaders/effect_mirror.wgsl");

/// The starter pack (M4). Ids and parameter names/order are drift-checked
/// against the `cutlass-models` catalog from the engine.
const BLUEPRINTS: &[EffectBlueprint] = &[
    EffectBlueprint {
        id: "gaussian_blur",
        params: &["radius"],
        passes: &[BLUR_FS, BLUR_FS],
    },
    EffectBlueprint {
        id: "vignette",
        params: &["amount"],
        passes: &[VIGNETTE_FS],
    },
    EffectBlueprint {
        id: "sharpen",
        params: &["amount"],
        passes: &[SHARPEN_FS],
    },
    EffectBlueprint {
        id: "pixelate",
        params: &["size"],
        passes: &[PIXELATE_FS],
    },
    EffectBlueprint {
        id: "glitch",
        params: &["amount", "seed"],
        passes: &[GLITCH_FS],
    },
    EffectBlueprint {
        id: "chromatic_aberration",
        params: &["amount"],
        passes: &[CHROMATIC_FS],
    },
    EffectBlueprint {
        id: "grain",
        params: &["amount", "seed"],
        passes: &[GRAIN_FS],
    },
    EffectBlueprint {
        id: "glow",
        params: &["threshold", "intensity"],
        passes: &[GLOW_FS, GLOW_FS],
    },
    EffectBlueprint {
        id: "zoom_blur",
        params: &["amount"],
        passes: &[ZOOM_BLUR_FS],
    },
    EffectBlueprint {
        id: "mirror",
        params: &["mode"],
        passes: &[MIRROR_FS],
    },
];

/// Canonical descriptors for every effect the compositor can render.
pub fn effect_descriptors() -> Vec<EffectDescriptor> {
    BLUEPRINTS
        .iter()
        .map(|bp| EffectDescriptor {
            id: bp.id,
            params: bp.params,
        })
        .collect()
}

/// Slot index of `param` within `effect_id`, or `None` if either is unknown.
/// The engine uses this to pack model parameters into the uniform.
pub fn effect_param_index(effect_id: &str, param: &str) -> Option<usize> {
    BLUEPRINTS
        .iter()
        .find(|bp| bp.id == effect_id)?
        .params
        .iter()
        .position(|p| *p == param)
}

/// GPU pipelines for the effect catalog, built once at compositor
/// construction. `effects` maps an id to the ordered pipeline indices of its
/// passes (into `pipelines`).
pub(crate) struct EffectRegistry {
    pub(crate) pipelines: Vec<wgpu::RenderPipeline>,
    effects: HashMap<&'static str, Vec<usize>>,
}

impl EffectRegistry {
    pub(crate) fn build(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        format: wgpu::TextureFormat,
        header: &str,
    ) -> Self {
        let mut pipelines = Vec::new();
        let mut by_source: HashMap<&'static str, usize> = HashMap::new();
        let mut effects: HashMap<&'static str, Vec<usize>> = HashMap::new();

        for bp in BLUEPRINTS {
            let mut pass_indices = Vec::with_capacity(bp.passes.len());
            for &frag in bp.passes {
                let index = *by_source.entry(frag).or_insert_with(|| {
                    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("effect_pass"),
                        source: wgpu::ShaderSource::Wgsl(format!("{header}\n{frag}").into()),
                    });
                    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some("effect_pipeline"),
                        layout: Some(layout),
                        vertex: wgpu::VertexState {
                            module: &module,
                            entry_point: Some("vs"),
                            buffers: &[],
                            compilation_options: Default::default(),
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &module,
                            entry_point: Some("fs"),
                            // Passes fully cover the target (fullscreen
                            // triangle) and replace it; no blending.
                            targets: &[Some(wgpu::ColorTargetState {
                                format,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                            compilation_options: Default::default(),
                        }),
                        primitive: wgpu::PrimitiveState {
                            topology: wgpu::PrimitiveTopology::TriangleList,
                            ..Default::default()
                        },
                        depth_stencil: None,
                        multisample: wgpu::MultisampleState::default(),
                        multiview_mask: None,
                        cache: None,
                    });
                    pipelines.push(pipeline);
                    pipelines.len() - 1
                });
                pass_indices.push(index);
            }
            effects.insert(bp.id, pass_indices);
        }

        Self { pipelines, effects }
    }

    /// The ordered pass pipeline indices for `effect_id`, or `None` if the id
    /// is not in the registry (the compositor skips unknown effects).
    pub(crate) fn passes(&self, effect_id: &str) -> Option<&[usize]> {
        self.effects.get(effect_id).map(Vec::as_slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptors_have_unique_ids_and_fit_the_slots() {
        let descriptors = effect_descriptors();
        assert!(!descriptors.is_empty());
        let mut ids: Vec<&str> = descriptors.iter().map(|d| d.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), descriptors.len(), "effect ids are unique");
        for d in &descriptors {
            assert!(
                d.params.len() <= EFFECT_PARAM_SLOTS,
                "{} declares more params than slots",
                d.id
            );
        }
    }

    #[test]
    fn param_index_resolves_known_and_rejects_unknown() {
        assert_eq!(effect_param_index("gaussian_blur", "radius"), Some(0));
        assert_eq!(effect_param_index("vignette", "amount"), Some(0));
        assert_eq!(effect_param_index("gaussian_blur", "nope"), None);
        assert_eq!(effect_param_index("no_such_effect", "radius"), None);
    }
}
