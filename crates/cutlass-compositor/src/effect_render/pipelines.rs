use crate::passes::resolve_transition_pass;

use super::RT_FORMAT;
use super::blend::build_blend_pipeline;
use super::blit::{
    build_additive_blit_pipeline, build_blit_pipeline, build_offset_blit_pipeline,
    build_replace_pipeline,
};

/// GPU pipelines for catalog effect and transition passes.
pub(crate) struct PassRegistry {
    pub passthrough: wgpu::RenderPipeline,
    pub blur_h: wgpu::RenderPipeline,
    pub blur_v: wgpu::RenderPipeline,
    pub vignette: wgpu::RenderPipeline,
    pub pixelate: wgpu::RenderPipeline,
    pub crossfade: wgpu::RenderPipeline,
    pub wipe: wgpu::RenderPipeline,
    pub grade: wgpu::RenderPipeline,
    pub lut: wgpu::RenderPipeline,
    pub effect_layout: wgpu::BindGroupLayout,
    pub lut_layout: wgpu::BindGroupLayout,
    pub transition_layout: wgpu::BindGroupLayout,
    pub sampler: wgpu::Sampler,
    /// Full-canvas blit of an offscreen texture (premultiplied src-over).
    pub blit_pipeline: wgpu::RenderPipeline,
    /// Full-canvas replacement blit for canvas-wide passes.
    pub replace_pipeline: wgpu::RenderPipeline,
    pub blit_layout: wgpu::BindGroupLayout,
    /// Dst-sampling blend modes (non-`Normal`).
    pub blend_pipeline: wgpu::RenderPipeline,
    pub blend_layout: wgpu::BindGroupLayout,
    /// Layer-alpha silhouette tint (shadow/glow seed).
    pub silhouette_pipeline: wgpu::RenderPipeline,
    /// Premultiplied blit with UV offset (shadow placement).
    pub offset_blit_pipeline: wgpu::RenderPipeline,
    pub offset_blit_layout: wgpu::BindGroupLayout,
    /// Additive premultiplied blit (glow).
    pub additive_blit_pipeline: wgpu::RenderPipeline,
    /// Harden blurred silhouette into an outside stroke (outline).
    pub harden_pipeline: wgpu::RenderPipeline,
    pub harden_layout: wgpu::BindGroupLayout,
}

impl PassRegistry {
    pub fn new(device: &wgpu::Device) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cutlass.effect.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let effect_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cutlass.effect.bgl"),
            entries: &[tex_entry(0), sampler_entry(1), uniform_entry(2)],
        });

        let transition_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cutlass.transition.bgl"),
            entries: &[
                tex_entry(0),
                tex_entry(1),
                sampler_entry(2),
                uniform_entry(3),
            ],
        });

        let blit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cutlass.blit.bgl"),
            entries: &[tex_entry(0), sampler_entry(1)],
        });

        let lut_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cutlass.lut.bgl"),
            entries: &[
                tex_entry(0),
                sampler_entry(1),
                tex3d_entry(2),
                sampler_entry(3),
                uniform_entry(4),
            ],
        });

        let blend_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cutlass.blend.bgl"),
            entries: &[
                tex_entry(0),
                tex_entry(1),
                sampler_entry(2),
                uniform_entry(3),
            ],
        });

        let offset_blit_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cutlass.offset_blit.bgl"),
                entries: &[tex_entry(0), sampler_entry(1), uniform_entry(2)],
            });

        let harden_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cutlass.harden.bgl"),
            entries: &[
                tex_entry(0),
                tex_entry(1),
                sampler_entry(2),
                uniform_entry(3),
            ],
        });

        let passthrough = build_effect_pipeline(
            device,
            "cutlass.passthrough",
            &effect_layout,
            include_str!("../../shaders/effect_passthrough.wgsl"),
        );
        let blur_h = build_effect_pipeline(
            device,
            "cutlass.blur_h",
            &effect_layout,
            include_str!("../../shaders/effect_blur_h.wgsl"),
        );
        let blur_v = build_effect_pipeline(
            device,
            "cutlass.blur_v",
            &effect_layout,
            include_str!("../../shaders/effect_blur_v.wgsl"),
        );
        let vignette = build_effect_pipeline(
            device,
            "cutlass.vignette",
            &effect_layout,
            include_str!("../../shaders/effect_vignette.wgsl"),
        );
        let pixelate = build_effect_pipeline(
            device,
            "cutlass.pixelate",
            &effect_layout,
            include_str!("../../shaders/effect_pixelate.wgsl"),
        );
        let crossfade = build_transition_pipeline(
            device,
            "cutlass.crossfade",
            &transition_layout,
            include_str!("../../shaders/transition_crossfade.wgsl"),
        );
        let wipe = build_transition_pipeline(
            device,
            "cutlass.wipe",
            &transition_layout,
            include_str!("../../shaders/transition_wipe.wgsl"),
        );
        let grade = build_effect_pipeline(
            device,
            "cutlass.canvas_grade",
            &effect_layout,
            &format!(
                "{}\n{}",
                include_str!("../../shaders/grade.wgsl"),
                include_str!("../../shaders/canvas_grade.wgsl")
            ),
        );
        let lut = build_effect_pipeline(
            device,
            "cutlass.lut",
            &lut_layout,
            include_str!("../../shaders/lut.wgsl"),
        );
        let blit_pipeline = build_blit_pipeline(device, &blit_layout);
        let replace_pipeline = build_replace_pipeline(device, &blit_layout);
        let blend_pipeline = build_blend_pipeline(device, &blend_layout);
        let silhouette_pipeline = build_effect_pipeline(
            device,
            "cutlass.style_silhouette",
            &effect_layout,
            include_str!("../../shaders/style_silhouette.wgsl"),
        );
        let offset_blit_pipeline = build_offset_blit_pipeline(device, &offset_blit_layout);
        let additive_blit_pipeline = build_additive_blit_pipeline(device, &blit_layout);
        let harden_pipeline = build_effect_pipeline(
            device,
            "cutlass.style_harden",
            &harden_layout,
            include_str!("../../shaders/style_harden.wgsl"),
        );

        Self {
            passthrough,
            blur_h,
            blur_v,
            vignette,
            pixelate,
            crossfade,
            wipe,
            grade,
            lut,
            effect_layout,
            lut_layout,
            transition_layout,
            sampler,
            blit_pipeline,
            replace_pipeline,
            blit_layout,
            blend_pipeline,
            blend_layout,
            silhouette_pipeline,
            offset_blit_pipeline,
            offset_blit_layout,
            additive_blit_pipeline,
            harden_pipeline,
            harden_layout,
        }
    }

    pub(super) fn effect_pipeline(&self, id: &str) -> &wgpu::RenderPipeline {
        match id {
            "gaussian_blur" => &self.blur_h,
            "vignette" => &self.vignette,
            "pixelate" => &self.pixelate,
            _ => &self.passthrough,
        }
    }

    pub(super) fn transition_pipeline(&self, id: &str) -> &wgpu::RenderPipeline {
        match resolve_transition_pass(id) {
            "wipe_left" => &self.wipe,
            _ => &self.crossfade,
        }
    }
}

fn tex_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn tex3d_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D3,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(super) const PREMULT_OVER: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
};

pub(super) fn build_effect_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    source: &str,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            targets: &[Some(wgpu::ColorTargetState {
                format: RT_FORMAT,
                blend: Some(PREMULT_OVER),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn build_transition_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    source: &str,
) -> wgpu::RenderPipeline {
    build_effect_pipeline(device, label, layout, source)
}
