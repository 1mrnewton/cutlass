//! Offscreen targets, pass pipelines, and effect/transition execution.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::error::CompositorError;
use crate::grade::ColorGrade;
use crate::passes::{PassInstance, effect_is_noop, resolve_transition_pass};

const RT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TexelUniforms {
    texel_size: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EffectUniforms {
    texel_size: [f32; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TransitionUniforms {
    texel_size: [f32; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GradeUniforms {
    grade0: [f32; 4],
    grade1: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LutUniforms {
    /// intensity, lut size, pad, pad.
    params0: [f32; 4],
    domain_lo: [f32; 4],
    domain_scale: [f32; 4],
}

/// Reused canvas-sized RGBA textures for effect ping-pong and transitions.
pub(crate) struct OffscreenPool {
    pub(crate) width: u32,
    pub(crate) height: u32,
    // The views keep the underlying textures alive; no need to store them.
    views: [wgpu::TextureView; 3],
}

impl OffscreenPool {
    pub fn ensure(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let make = |label: &str| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: RT_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };
        let (_t0, v0) = make("cutlass.offscreen.a");
        let (_t1, v1) = make("cutlass.offscreen.b");
        let (_t2, v2) = make("cutlass.offscreen.c");
        Self {
            width,
            height,
            views: [v0, v1, v2],
        }
    }

    pub fn view(&self, index: usize) -> &wgpu::TextureView {
        &self.views[index % 3]
    }
}

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

        let passthrough = build_effect_pipeline(
            device,
            "cutlass.passthrough",
            &effect_layout,
            include_str!("../shaders/effect_passthrough.wgsl"),
        );
        let blur_h = build_effect_pipeline(
            device,
            "cutlass.blur_h",
            &effect_layout,
            include_str!("../shaders/effect_blur_h.wgsl"),
        );
        let blur_v = build_effect_pipeline(
            device,
            "cutlass.blur_v",
            &effect_layout,
            include_str!("../shaders/effect_blur_v.wgsl"),
        );
        let vignette = build_effect_pipeline(
            device,
            "cutlass.vignette",
            &effect_layout,
            include_str!("../shaders/effect_vignette.wgsl"),
        );
        let pixelate = build_effect_pipeline(
            device,
            "cutlass.pixelate",
            &effect_layout,
            include_str!("../shaders/effect_pixelate.wgsl"),
        );
        let crossfade = build_transition_pipeline(
            device,
            "cutlass.crossfade",
            &transition_layout,
            include_str!("../shaders/transition_crossfade.wgsl"),
        );
        let wipe = build_transition_pipeline(
            device,
            "cutlass.wipe",
            &transition_layout,
            include_str!("../shaders/transition_wipe.wgsl"),
        );
        let grade = build_effect_pipeline(
            device,
            "cutlass.canvas_grade",
            &effect_layout,
            include_str!("../shaders/canvas_grade.wgsl"),
        );
        let lut = build_effect_pipeline(
            device,
            "cutlass.lut",
            &lut_layout,
            include_str!("../shaders/lut.wgsl"),
        );
        let blit_pipeline = build_blit_pipeline(device, &blit_layout);
        let replace_pipeline = build_replace_pipeline(device, &blit_layout);

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
        }
    }

    fn effect_pipeline(&self, id: &str) -> &wgpu::RenderPipeline {
        match id {
            "gaussian_blur" => &self.blur_h,
            "vignette" => &self.vignette,
            "pixelate" => &self.pixelate,
            _ => &self.passthrough,
        }
    }

    fn transition_pipeline(&self, id: &str) -> &wgpu::RenderPipeline {
        match resolve_transition_pass(id) {
            "wipe_left" => &self.wipe,
            _ => &self.crossfade,
        }
    }
}

/// Run an effect chain on `input` view, ping-ponging through `pool`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_effect_chain<'a>(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    registry: &PassRegistry,
    pool: &'a OffscreenPool,
    input_view: &'a wgpu::TextureView,
    effects: &[PassInstance<'_>],
    width: u32,
    height: u32,
) -> &'a wgpu::TextureView {
    let mut read_idx = 0usize;
    let mut current_input = input_view;

    for effect in effects {
        if effect_is_noop(effect.id, effect.params) {
            continue;
        }
        let write_idx = read_idx ^ 1;
        let write_view = pool.view(write_idx);

        match effect.id {
            "gaussian_blur" => {
                let radius = effect.params.first().copied().unwrap_or(0.0);
                let mid_idx = 1usize;
                let out_idx = 2usize;
                let mid_view = pool.view(mid_idx);
                let out_view = pool.view(out_idx);
                draw_effect_pass(
                    device,
                    encoder,
                    registry,
                    &registry.blur_h,
                    current_input,
                    mid_view,
                    width,
                    height,
                    radius,
                );
                draw_effect_pass(
                    device,
                    encoder,
                    registry,
                    &registry.blur_v,
                    mid_view,
                    out_view,
                    width,
                    height,
                    radius,
                );
                read_idx = out_idx;
                current_input = out_view;
                continue;
            }
            _ => {
                let pipeline = registry.effect_pipeline(effect.id);
                let param0 = effect.params.first().copied().unwrap_or(0.0);
                draw_effect_pass(
                    device,
                    encoder,
                    registry,
                    pipeline,
                    current_input,
                    write_view,
                    width,
                    height,
                    param0,
                );
            }
        }
        read_idx = write_idx;
        current_input = pool.view(read_idx);
    }

    current_input
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_transition_pass(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    registry: &PassRegistry,
    outgoing: &wgpu::TextureView,
    incoming: &wgpu::TextureView,
    output: &wgpu::TextureView,
    transition_id: &str,
    progress: f32,
    width: u32,
    height: u32,
) {
    let pipeline = registry.transition_pipeline(transition_id);
    let uniforms = TransitionUniforms {
        texel_size: texel_size(width, height),
        params: [progress, 0.0, 0.0, 0.0],
    };
    let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cutlass.transition.uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cutlass.transition.bg"),
        layout: &registry.transition_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(outgoing),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(incoming),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&registry.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: ubuf.as_entire_binding(),
            },
        ],
    });

    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("cutlass.transition.pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: output,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..6, 0..1);
}

pub(crate) fn run_grade_pass<'a>(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    registry: &PassRegistry,
    input: &'a wgpu::TextureView,
    output: &'a wgpu::TextureView,
    grade: ColorGrade,
) -> &'a wgpu::TextureView {
    let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cutlass.canvas_grade.uniforms"),
        contents: bytemuck::bytes_of(&GradeUniforms {
            grade0: [
                grade.exposure,
                grade.brightness,
                grade.contrast,
                grade.saturation,
            ],
            grade1: [grade.temperature, grade.tint, 0.0, 0.0],
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cutlass.canvas_grade.bg"),
        layout: &registry.effect_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(input),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&registry.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: ubuf.as_entire_binding(),
            },
        ],
    });

    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("cutlass.canvas_grade.pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: output,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(&registry.grade);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..6, 0..1);
    output
}

/// Map `input` through a 3D LUT texture into `output` (premultiplied in/out;
/// the shader un-premultiplies around the lookup).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_lut_pass<'a>(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    registry: &PassRegistry,
    input: &wgpu::TextureView,
    output: &'a wgpu::TextureView,
    lut_view: &wgpu::TextureView,
    lut_size: u32,
    domain_min: [f32; 3],
    domain_max: [f32; 3],
    intensity: f32,
) -> &'a wgpu::TextureView {
    let scale = [
        1.0 / (domain_max[0] - domain_min[0]),
        1.0 / (domain_max[1] - domain_min[1]),
        1.0 / (domain_max[2] - domain_min[2]),
    ];
    let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cutlass.lut.uniforms"),
        contents: bytemuck::bytes_of(&LutUniforms {
            params0: [intensity.clamp(0.0, 1.0), lut_size as f32, 0.0, 0.0],
            domain_lo: [domain_min[0], domain_min[1], domain_min[2], 0.0],
            domain_scale: [scale[0], scale[1], scale[2], 0.0],
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cutlass.lut.bg"),
        layout: &registry.lut_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(input),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&registry.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(lut_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&registry.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: ubuf.as_entire_binding(),
            },
        ],
    });

    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("cutlass.lut.pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: output,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(&registry.lut);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..6, 0..1);
    output
}

/// Alpha-over a full-canvas offscreen texture onto `canvas_view`.
pub(crate) fn blit_premultiplied_to_canvas(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    registry: &PassRegistry,
    source: &wgpu::TextureView,
    canvas_view: &wgpu::TextureView,
) {
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cutlass.blit.bg"),
        layout: &registry.blit_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(source),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&registry.sampler),
            },
        ],
    });
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("cutlass.blit.pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: canvas_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(&registry.blit_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..6, 0..1);
}

/// Replace a full-canvas render target with `source`.
pub(crate) fn blit_replace(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    registry: &PassRegistry,
    source: &wgpu::TextureView,
    output: &wgpu::TextureView,
) {
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cutlass.blit_replace.bg"),
        layout: &registry.blit_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(source),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&registry.sampler),
            },
        ],
    });
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("cutlass.blit_replace.pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: output,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(&registry.replace_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..6, 0..1);
}

#[allow(clippy::too_many_arguments)]
fn draw_effect_pass(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    registry: &PassRegistry,
    pipeline: &wgpu::RenderPipeline,
    input: &wgpu::TextureView,
    output: &wgpu::TextureView,
    width: u32,
    height: u32,
    param0: f32,
) {
    let uniforms = EffectUniforms {
        texel_size: texel_size(width, height),
        params: [param0, 0.0, 0.0, 0.0],
    };
    let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cutlass.effect.uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cutlass.effect.bg"),
        layout: &registry.effect_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(input),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&registry.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: ubuf.as_entire_binding(),
            },
        ],
    });
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("cutlass.effect.pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: output,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..6, 0..1);
}

fn texel_size(width: u32, height: u32) -> [f32; 4] {
    [
        1.0 / width.max(1) as f32,
        1.0 / height.max(1) as f32,
        0.0,
        0.0,
    ]
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

const PREMULT_OVER: wgpu::BlendState = wgpu::BlendState {
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

fn build_effect_pipeline(
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

fn build_blit_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let source = r#"
struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var positions = array<vec2<f32>, 6>(
        vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0),
        vec2(-1.0,1.0), vec2(1.0,-1.0), vec2(1.0,1.0));
    var uvs = array<vec2<f32>, 6>(
        vec2(0.0,1.0), vec2(1.0,1.0), vec2(0.0,0.0),
        vec2(0.0,0.0), vec2(1.0,1.0), vec2(1.0,0.0));
    var o: VsOut;
    o.pos = vec4(positions[vi], 0.0, 1.0);
    o.uv = uvs[vi];
    return o;
}
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@fragment fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, in.uv);
}
"#;
    build_effect_pipeline(device, "cutlass.blit", layout, source)
}

fn build_replace_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let source = r#"
struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var positions = array<vec2<f32>, 6>(
        vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0),
        vec2(-1.0,1.0), vec2(1.0,-1.0), vec2(1.0,1.0));
    var uvs = array<vec2<f32>, 6>(
        vec2(0.0,1.0), vec2(1.0,1.0), vec2(0.0,0.0),
        vec2(0.0,0.0), vec2(1.0,1.0), vec2(1.0,0.0));
    var o: VsOut;
    o.pos = vec4(positions[vi], 0.0, 1.0);
    o.uv = uvs[vi];
    return o;
}
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@fragment fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, in.uv);
}
"#;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cutlass.blit_replace"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cutlass.blit_replace"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("cutlass.blit_replace"),
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
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// Draw one layer to an offscreen (or canvas) target.
///
/// `instances` is `Some((buffer, count))` for instanced glyph draws; `None`
/// draws a single unit-quad like the rest of the layer pipelines.
pub(crate) fn draw_layer_to_offscreen(
    pass: &mut wgpu::RenderPass<'_>,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    instances: Option<(&wgpu::Buffer, u32)>,
) {
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    if let Some((buffer, count)) = instances {
        pass.set_vertex_buffer(0, buffer.slice(..));
        pass.draw(0..6, 0..count);
    } else {
        pass.draw(0..6, 0..1);
    }
}

/// Returns `true` when the effect chain is empty or all passes are no-ops.
pub(crate) fn effects_need_offscreen(effects: &[PassInstance<'_>]) -> bool {
    !effects.is_empty() && effects.iter().any(|e| !effect_is_noop(e.id, e.params))
}

#[allow(dead_code)]
pub(crate) fn check_dimensions(width: u32, height: u32) -> Result<(), CompositorError> {
    if width == 0 || height == 0 {
        return Err(CompositorError::InvalidDimensions { width, height });
    }
    Ok(())
}
