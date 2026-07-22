//! Dst-sampling blend composite pass (non-`Normal` layer blend modes).

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::layer::BlendMode;

use super::RT_FORMAT;
use super::pipelines::PassRegistry;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlendUniforms {
    mode: u32,
    _pad: [u32; 3],
}

/// Composite premultiplied `src` over a canvas `dst` snapshot into `output`
/// using `mode`. Fixed-function blending is disabled — the shader writes the
/// final premultiplied result.
pub(crate) fn run_blend_composite(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    registry: &PassRegistry,
    dst_snapshot: &wgpu::TextureView,
    src_layer: &wgpu::TextureView,
    output: &wgpu::TextureView,
    mode: BlendMode,
) {
    let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cutlass.blend.uniforms"),
        contents: bytemuck::bytes_of(&BlendUniforms {
            mode: mode.shader_id(),
            _pad: [0; 3],
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cutlass.blend.bg"),
        layout: &registry.blend_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(dst_snapshot),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(src_layer),
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
        label: Some("cutlass.blend.pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: output,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                // Every pixel is written; Load preserves unread edges if any.
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(&registry.blend_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..6, 0..1);
}

pub(super) fn build_blend_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let source = include_str!("../../shaders/blend_composite.wgsl");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cutlass.blend_composite"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cutlass.blend_composite"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("cutlass.blend_composite"),
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
                // Shader computes final composition; no fixed-function blend.
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        multiview_mask: None,
        cache: None,
    })
}
