use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::pipelines::PassRegistry;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LutUniforms {
    /// intensity, lut size, pad, pad.
    params0: [f32; 4],
    domain_lo: [f32; 4],
    domain_scale: [f32; 4],
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
