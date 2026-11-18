//! The keep pass: an effect's result shows inside its masks and the
//! picture from before the effect shows outside them, softly across a
//! feathered edge. The masks are `cutlass_effects`'s WGSL `mask_stack`, the
//! arithmetic the CPU path uses for an adjustment layer's masks.

use crate::pool::Bind;
use crate::stack::{STACK_BYTES, atlas_view, stack_bytes};
use cutlass_effects::{CanvasMasks, MASK_WGSL, STACK_WGSL};

use crate::CANVAS_FORMAT;

const SHADER: &str = r"
@group(0) @binding(0) var before: texture_2d<f32>;
@group(0) @binding(1) var after: texture_2d<f32>;
@group(0) @binding(2) var<uniform> stack: MaskStack;
@group(0) @binding(3) var atlas: texture_2d<f32>;

struct Out {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs(@builtin(vertex_index) index: u32) -> Out {
    let x = f32((index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(index & 2u) * 2.0 - 1.0;
    var out: Out;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    let at = vec2<i32>(in.position.xy);
    let was = textureLoad(before, at, 0);
    let now = textureLoad(after, at, 0);
    return mix(was, now, mask_stack(stack, atlas, in.position.xy));
}
";

pub(crate) struct Keep {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    pool: std::sync::Arc<crate::pool::Pool>,
}

impl Keep {
    pub(crate) fn new(device: &wgpu::Device, pool: std::sync::Arc<crate::pool::Pool>) -> Self {
        let source = format!("{SHADER}\n{MASK_WGSL}\n{STACK_WGSL}");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cutlass-render keep"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let texture = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cutlass-render keep"),
            entries: &[
                texture(0),
                texture(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(STACK_BYTES),
                    },
                    count: None,
                },
                texture(3),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cutlass-render keep"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cutlass-render keep"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: CANVAS_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            pipeline,
            layout,
            pool,
        }
    }

    /// Write `after` where `masks` keep it and `before` elsewhere into `to`.
    /// Drawn masks whose pictures are larger than the device's textures
    /// are refused.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        before: &wgpu::TextureView,
        after: &wgpu::TextureView,
        to: &wgpu::TextureView,
        masks: &CanvasMasks,
    ) -> Result<(), crate::RenderError> {
        let atlas = atlas_view(&self.pool, device, queue, masks)?;
        let bytes = stack_bytes(masks);
        let buffer = self.pool.uniform(
            device,
            queue,
            "cutlass-render keep params",
            STACK_BYTES,
            &bytes,
        );
        let group = self.pool.group(
            device,
            "cutlass-render keep",
            &self.layout,
            &[
                (0, Bind::View(before)),
                (1, Bind::View(after)),
                (2, Bind::Buffer(&buffer)),
                (3, Bind::View(&atlas)),
            ],
        );
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("cutlass-render keep"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: to,
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
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
        drop(pass);
        Ok(())
    }
}
