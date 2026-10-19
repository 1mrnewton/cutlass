//! Transition registry: maps transition ids to a single GPU blend pass.
//!
//! Transitions are **data** (v1 roadmap M4), like effects: the model stores
//! `{transition_id, duration}` and this crate owns the WGSL that blends the
//! two frames by progress. Each transition is a single fragment pass sharing
//! the effect header ([`effect_header.wgsl`]): `orig_tex` is the outgoing
//! frame, `src_tex` is the incoming frame, and progress (0..1) rides in
//! `fx.p0.x`.
//!
//! [`transition_ids`] is the canonical renderable set; the `cutlass-models`
//! transition catalog (display names, used for validation/UI) is drift-checked
//! against it from the engine.

use std::collections::HashMap;

struct TransitionBlueprint {
    id: &'static str,
    fragment: &'static str,
}

const CROSSFADE_FS: &str = include_str!("../shaders/transition_crossfade.wgsl");
const DIP_TO_BLACK_FS: &str = include_str!("../shaders/transition_dip_to_black.wgsl");
const DIP_TO_WHITE_FS: &str = include_str!("../shaders/transition_dip_to_white.wgsl");
const WIPE_LEFT_FS: &str = include_str!("../shaders/transition_wipe_left.wgsl");
const WIPE_RIGHT_FS: &str = include_str!("../shaders/transition_wipe_right.wgsl");
const WIPE_UP_FS: &str = include_str!("../shaders/transition_wipe_up.wgsl");
const WIPE_DOWN_FS: &str = include_str!("../shaders/transition_wipe_down.wgsl");
const SLIDE_FS: &str = include_str!("../shaders/transition_slide.wgsl");

/// The starter set (M4). Ids are drift-checked against the `cutlass-models`
/// transition catalog from the engine.
const BLUEPRINTS: &[TransitionBlueprint] = &[
    TransitionBlueprint {
        id: "crossfade",
        fragment: CROSSFADE_FS,
    },
    TransitionBlueprint {
        id: "dip_to_black",
        fragment: DIP_TO_BLACK_FS,
    },
    TransitionBlueprint {
        id: "dip_to_white",
        fragment: DIP_TO_WHITE_FS,
    },
    TransitionBlueprint {
        id: "wipe_left",
        fragment: WIPE_LEFT_FS,
    },
    TransitionBlueprint {
        id: "wipe_right",
        fragment: WIPE_RIGHT_FS,
    },
    TransitionBlueprint {
        id: "wipe_up",
        fragment: WIPE_UP_FS,
    },
    TransitionBlueprint {
        id: "wipe_down",
        fragment: WIPE_DOWN_FS,
    },
    TransitionBlueprint {
        id: "slide",
        fragment: SLIDE_FS,
    },
];

/// Canonical ids for every transition the compositor can render.
pub fn transition_ids() -> Vec<&'static str> {
    BLUEPRINTS.iter().map(|bp| bp.id).collect()
}

/// GPU pipelines for the transition catalog, built once at compositor
/// construction. `transitions` maps an id to its single pass pipeline index.
pub(crate) struct TransitionRegistry {
    pub(crate) pipelines: Vec<wgpu::RenderPipeline>,
    transitions: HashMap<&'static str, usize>,
}

impl TransitionRegistry {
    pub(crate) fn build(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        format: wgpu::TextureFormat,
        header: &str,
    ) -> Self {
        let mut pipelines = Vec::new();
        let mut transitions: HashMap<&'static str, usize> = HashMap::new();

        for bp in BLUEPRINTS {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("transition_pass"),
                source: wgpu::ShaderSource::Wgsl(format!("{header}\n{}", bp.fragment).into()),
            });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("transition_pipeline"),
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
            transitions.insert(bp.id, pipelines.len() - 1);
        }

        Self {
            pipelines,
            transitions,
        }
    }

    /// The pass pipeline index for `transition_id`, or `None` if the id is not
    /// in the registry (the compositor falls back to a cut).
    pub(crate) fn pipeline(&self, transition_id: &str) -> Option<usize> {
        self.transitions.get(transition_id).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique_and_eight() {
        let mut ids = transition_ids();
        let count = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), count, "transition ids are unique");
        assert_eq!(count, 8, "the M4 starter set is eight transitions");
    }
}
