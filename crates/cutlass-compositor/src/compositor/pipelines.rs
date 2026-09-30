use super::glyphs::GlyphInstanceGpu;
use crate::effect_render::PassRegistry;
use crate::gpu::GpuContext;

use super::{CANVAS_FORMAT, Compositor};

const GRADE_WGSL: &str = include_str!("../../shaders/grade.wgsl");

/// Straight-alpha src-over: the source fragment carries un-premultiplied color,
/// so the source color is weighted by its own alpha. Used by the solid and YUV
/// (fully-opaque) pipelines.
const STRAIGHT_OVER: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::SrcAlpha,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
};

/// Premultiplied src-over: the source fragment is already alpha-weighted (the
/// RGBA pipeline), so the color factor is `One`. Correct for anti-aliased edges.
const PREMULTIPLIED_OVER: wgpu::BlendState = wgpu::BlendState {
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

impl Compositor {
    /// Build the compositor's pipelines on an existing GPU context.
    pub fn new(gpu: &GpuContext) -> Self {
        let device = &gpu.device;

        let yuv_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cutlass.yuv.bgl"),
            entries: &[
                plane_tex_entry(0),
                plane_tex_entry(1),
                plane_tex_entry(2),
                sampler_entry(3),
                uniform_entry(4),
            ],
        });

        let yuv_fx_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cutlass.yuv_fx.bgl"),
            entries: &[
                plane_tex_entry(0),
                plane_tex_entry(1),
                plane_tex_entry(2),
                sampler_entry(3),
                uniform_entry(4),
                uniform_entry(5),
            ],
        });

        let rgba_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cutlass.rgba.bgl"),
            entries: &[plane_tex_entry(0), sampler_entry(1), uniform_entry(2)],
        });

        let rgba_fx_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cutlass.rgba_fx.bgl"),
            entries: &[
                plane_tex_entry(0),
                sampler_entry(1),
                uniform_entry(2),
                uniform_entry(3),
            ],
        });

        let solid_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cutlass.solid.bgl"),
            entries: &[uniform_entry(0)],
        });

        let sdf_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cutlass.sdf.bgl"),
            entries: &[uniform_entry(0)],
        });

        let glyphs_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cutlass.glyphs.bgl"),
            entries: &[plane_tex_entry(0), sampler_entry(1), uniform_entry(2)],
        });

        let yuv_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cutlass.yuv.wgsl"),
            source: wgsl_with_grade(include_str!("../../shaders/yuv.wgsl")),
        });
        let yuv_fx_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cutlass.yuv_fx.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{GRADE_WGSL}
{}{}",
                    include_str!("../../shaders/mask.wgsl"),
                    include_str!("../../shaders/yuv_fx.wgsl"),
                )
                .into(),
            ),
        });
        let rgba_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cutlass.rgba.wgsl"),
            source: wgsl_with_grade(include_str!("../../shaders/rgba.wgsl")),
        });
        let rgba_fx_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cutlass.rgba_fx.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{GRADE_WGSL}
{}{}",
                    include_str!("../../shaders/mask.wgsl"),
                    include_str!("../../shaders/rgba_fx.wgsl"),
                )
                .into(),
            ),
        });
        let solid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cutlass.solid.wgsl"),
            source: wgsl_with_grade(include_str!("../../shaders/solid.wgsl")),
        });
        let sdf_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cutlass.shape.wgsl"),
            source: wgsl_with_grade(include_str!("../../shaders/shape.wgsl")),
        });
        let glyphs_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cutlass.glyph.wgsl"),
            source: wgsl_with_grade(include_str!("../../shaders/glyph.wgsl")),
        });

        // Opaque video and solids use straight-alpha src-over; RGBA bitmaps
        // (text/shape coverage) are premultiplied for clean edges.
        let yuv_pipeline = build_pipeline(
            device,
            "cutlass.yuv",
            &yuv_layout,
            &yuv_shader,
            STRAIGHT_OVER,
        );
        let yuv_fx_pipeline = build_pipeline(
            device,
            "cutlass.yuv_fx",
            &yuv_fx_layout,
            &yuv_fx_shader,
            STRAIGHT_OVER,
        );
        let rgba_pipeline = build_pipeline(
            device,
            "cutlass.rgba",
            &rgba_layout,
            &rgba_shader,
            PREMULTIPLIED_OVER,
        );
        let rgba_fx_pipeline = build_pipeline(
            device,
            "cutlass.rgba_fx",
            &rgba_fx_layout,
            &rgba_fx_shader,
            PREMULTIPLIED_OVER,
        );
        let solid_pipeline = build_pipeline(
            device,
            "cutlass.solid",
            &solid_layout,
            &solid_shader,
            STRAIGHT_OVER,
        );
        // SDF shapes emit straight-alpha fragments (coverage folded into
        // alpha), like solids.
        let sdf_pipeline = build_pipeline(
            device,
            "cutlass.sdf",
            &sdf_layout,
            &sdf_shader,
            STRAIGHT_OVER,
        );
        // Glyphs are premultiplied atlas samples, like RGBA bitmaps.
        let glyphs_pipeline = build_instanced_pipeline(
            device,
            "cutlass.glyphs",
            &glyphs_layout,
            &glyphs_shader,
            PREMULTIPLIED_OVER,
            &[GlyphInstanceGpu::LAYOUT],
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cutlass.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Self {
            yuv_pipeline,
            yuv_fx_pipeline,
            rgba_pipeline,
            rgba_fx_pipeline,
            solid_pipeline,
            sdf_pipeline,
            glyphs_pipeline,
            yuv_layout,
            yuv_fx_layout,
            rgba_layout,
            rgba_fx_layout,
            solid_layout,
            sdf_layout,
            glyphs_layout,
            sampler,
            target: None,
            pass_registry: PassRegistry::new(device),
            offscreen: None,
            lut_cache: std::collections::HashMap::new(),
            glyph_atlases: std::cell::RefCell::new(cutlass_core::ByteBudgetLru::new(
                cutlass_core::RASTER_MEMO_BUDGET_BYTES,
            )),
            #[cfg(target_vendor = "apple")]
            metal_import: crate::metal_import::MetalSurfaceImporter::new(gpu),
            #[cfg(target_os = "windows")]
            dx12_import: crate::dx12_import::Dx12SurfaceImporter::new(gpu),
        }
    }
}

fn plane_tex_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn build_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
    blend: wgpu::BlendState,
) -> wgpu::RenderPipeline {
    build_instanced_pipeline(device, label, layout, shader, blend, &[])
}

fn build_instanced_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
    blend: wgpu::BlendState,
    vertex_buffers: &[wgpu::VertexBufferLayout<'_>],
) -> wgpu::RenderPipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs"),
            buffers: vertex_buffers,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs"),
            targets: &[Some(wgpu::ColorTargetState {
                format: CANVAS_FORMAT,
                blend: Some(blend),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn wgsl_with_grade(shader: &str) -> wgpu::ShaderSource<'static> {
    wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(format!("{GRADE_WGSL}\n{shader}")))
}
