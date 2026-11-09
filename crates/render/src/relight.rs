//! The relight pass: a picture lit again from a grey depth picture, as
//! `cutlass_effects::relight_rgba` lights it on the CPU after eval
//! stretches the depth to the picture. The arithmetic is
//! `cutlass_effects`'s WGSL `relight_pixel`.

use cutlass_effects::{
    Lighting, MAX_LIGHTS, RELIGHT_UNIFORM_BYTES, RELIGHT_WGSL, relight_uniform, stretch_axis,
};

use crate::{CANVAS_FORMAT, RenderError, Rgba};

const SHADER: &str = r"
@group(0) @binding(0) var<uniform> relight: Relight;
@group(0) @binding(1) var relight_picture: texture_2d<f32>;
@group(0) @binding(2) var relight_depth_map: texture_2d<f32>;
@group(0) @binding(3) var relight_axes: texture_2d<f32>;

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
    return relight_pixel(u32(in.position.x), u32(in.position.y));
}
";

/// The depth a picture is lit from, its red nearer when lighter, and the
/// light over it.
#[derive(Clone, Copy, Debug)]
pub struct Relit<'a> {
    pub depth: Rgba<'a>,
    pub lighting: &'a Lighting,
}

pub(crate) struct Relights {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
}

impl Relights {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let source = format!("{RELIGHT_WGSL}\n{SHADER}");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cutlass-render relight"),
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
            label: Some("cutlass-render relight"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(RELIGHT_UNIFORM_BYTES as u64),
                    },
                    count: None,
                },
                texture(1),
                texture(2),
                texture(3),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cutlass-render relight"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cutlass-render relight"),
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
        Self { pipeline, layout }
    }

    /// `from`, `size`, lit by `relit` into `to`. A depth of no size, or
    /// whose bytes do not match it, or more lights than a preset uses, is
    /// refused.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        from: &wgpu::TextureView,
        to: &wgpu::TextureView,
        size: (u32, u32),
        relit: &Relit<'_>,
    ) -> Result<(), RenderError> {
        if relit.lighting.lights.len() > MAX_LIGHTS {
            return Err(RenderError::Bytes {
                expected: MAX_LIGHTS,
                actual: relit.lighting.lights.len(),
            });
        }
        let depth = crate::layer::Pipelines::upload_rgba(device, queue, &relit.depth)?;
        let axes = axes(device, queue, size, (relit.depth.width, relit.depth.height))?;
        let uniform = relight_uniform(
            relit.lighting,
            size,
            (relit.depth.width, relit.depth.height),
        );
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cutlass-render relight params"),
            size: RELIGHT_UNIFORM_BYTES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, &uniform);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cutlass-render relight"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(from),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&depth.views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&axes),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("cutlass-render relight"),
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
        Ok(())
    }
}

/// The depth stretch from `depth` to `size` as a texture the shader
/// reads: across in row 0 and down in row 1, each texel the low index,
/// high index and weight.
fn axes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    size: (u32, u32),
    depth: (u32, u32),
) -> Result<wgpu::TextureView, RenderError> {
    let width = size.0.max(size.1);
    let limit = device.limits().max_texture_dimension_2d;
    if width == 0 || width > limit {
        return Err(RenderError::Size {
            width: size.0,
            height: size.1,
        });
    }
    let mut bytes = vec![0u8; width as usize * 2 * 16];
    let rows = [
        stretch_axis(size.0 as usize, depth.0 as usize),
        stretch_axis(size.1 as usize, depth.1 as usize),
    ];
    for (row, axis) in rows.iter().enumerate() {
        for (index, &(lo, hi, weight)) in axis.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let texel = [lo as f32, hi as f32, weight, 0.0];
            let at = (row * width as usize + index) * 16;
            for (offset, value) in texel.iter().enumerate() {
                bytes[at + offset * 4..at + offset * 4 + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
    }
    let extent = wgpu::Extent3d {
        width,
        height: 2,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("cutlass-render relight axes"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 16),
            rows_per_image: Some(2),
        },
        extent,
    );
    Ok(texture.create_view(&wgpu::TextureViewDescriptor::default()))
}

#[cfg(test)]
mod tests;
