//! The cut passes: a clip's mask or selection keeps part of the picture
//! drawn so far and clears the rest, as `cutlass_effects::Mask::apply` and
//! `Selection::apply` multiply it on the CPU.
//!
//! A mask's shape is `cutlass_effects`'s WGSL `mask_coverage`. A selection
//! is its one-byte-a-pixel picture, read bilinearly at the clip point each
//! canvas pixel shows, as the CPU reads it.

use cutlass_effects::{CanvasMask, MASK_WGSL};

use crate::CANVAS_FORMAT;

const SHADER: &str = r"
@group(0) @binding(0) var picture: texture_2d<f32>;
@group(0) @binding(1) var selection: texture_2d<f32>;
// A mask, or for a selection: the canvas-to-clip frame in frame0 and
// frame1.xy, invert in frame1.w, and the selection's size in shape0.xy.
@group(0) @binding(2) var<uniform> mask: CanvasMask;

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
fn fs_mask(in: Out) -> @location(0) vec4<f32> {
    let c = textureLoad(picture, vec2<i32>(in.position.xy), 0);
    return c * clamp(mask_coverage(mask, in.position.xy), 0.0, 1.0);
}

fn selected(x: i32, y: i32) -> f32 {
    return textureLoad(selection, vec2<i32>(x, y), 0).r;
}

// How much of the object is at canvas pixel centre `pixel`: 0 off the clip.
fn selection_coverage(pixel: vec2<f32>) -> f32 {
    let u = mask.frame0.x * pixel.x + mask.frame0.y * pixel.y + mask.frame0.z;
    let v = mask.frame0.w * pixel.x + mask.frame1.x * pixel.y + mask.frame1.y;
    if (u < -1.0 || u > 1.0 || v < -1.0 || v > 1.0) {
        return 0.0;
    }
    let w = mask.shape0.x;
    let h = mask.shape0.y;
    let x = clamp((u + 1.0) * 0.5 * w - 0.5, 0.0, w - 1.0);
    let y = clamp((1.0 - v) * 0.5 * h - 0.5, 0.0, h - 1.0);
    let x0 = i32(x);
    let y0 = i32(y);
    let x1 = min(x0 + 1, i32(w) - 1);
    let y1 = min(y0 + 1, i32(h) - 1);
    let fx = x - f32(x0);
    let fy = y - f32(y0);
    let top = selected(x0, y0) + (selected(x1, y0) - selected(x0, y0)) * fx;
    let bottom = selected(x0, y1) + (selected(x1, y1) - selected(x0, y1)) * fx;
    return top + (bottom - top) * fy;
}

@fragment
fn fs_select(in: Out) -> @location(0) vec4<f32> {
    let c = textureLoad(picture, vec2<i32>(in.position.xy), 0);
    var keep = selection_coverage(in.position.xy);
    if (mask.frame1.w > 0.5) {
        keep = 1.0 - keep;
    }
    return c * clamp(keep, 0.0, 1.0);
}
";

/// Bytes in the uniform block.
const PARAMS: u64 = 64;

/// A selection placed on the canvas: its pixels, one byte each and nonzero
/// selected, and `frame`, which takes a canvas pixel centre to the clip
/// point it shows (-1 to 1 across the clip, y up) as
/// `(a·x + b·y + c, d·x + e·y + f)`.
#[derive(Clone, Copy, Debug)]
pub struct Selected<'a> {
    pub width: u32,
    pub height: u32,
    pub alpha: &'a [u8],
    pub invert: bool,
    pub frame: [f32; 6],
}

pub(crate) struct Cuts {
    mask: wgpu::RenderPipeline,
    select: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    /// Bound in place of a selection by the mask pass.
    none: wgpu::TextureView,
}

impl Cuts {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let source = format!("{SHADER}\n{MASK_WGSL}");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cutlass-render cut"),
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
            label: Some("cutlass-render cut"),
            entries: &[
                texture(0),
                texture(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(PARAMS),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cutlass-render cut"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("cutlass-render cut"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry),
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
            })
        };
        let none =
            selection_texture(device, 1, 1).create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            mask: pipeline("fs_mask"),
            select: pipeline("fs_select"),
            layout,
            none,
        }
    }

    /// `from` kept as much as `mask` keeps, into `to`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mask(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        from: &wgpu::TextureView,
        to: &wgpu::TextureView,
        mask: &CanvasMask,
    ) {
        let uniform: Vec<u8> = mask
            .uniform()
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        self.draw(
            device, queue, encoder, &self.mask, from, &self.none, to, &uniform,
        );
    }

    /// `from` kept only where `selected` is, into `to`. A selection larger
    /// than the device's textures is refused.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn select(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        from: &wgpu::TextureView,
        to: &wgpu::TextureView,
        selected: &Selected<'_>,
    ) -> Result<(), crate::RenderError> {
        let (width, height) = (selected.width, selected.height);
        let limit = device.limits().max_texture_dimension_2d;
        if width == 0 || height == 0 || width > limit || height > limit {
            return Err(crate::RenderError::Size { width, height });
        }
        let expected = width as usize * height as usize;
        if selected.alpha.len() != expected {
            return Err(crate::RenderError::Bytes {
                expected,
                actual: selected.alpha.len(),
            });
        }
        let texture = selection_texture(device, width, height);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            selected.alpha,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let [a, b, c, d, e, f] = selected.frame;
        #[allow(clippy::cast_precision_loss)]
        let values = [
            a,
            b,
            c,
            d,
            e,
            f,
            0.0,
            if selected.invert { 1.0 } else { 0.0 },
            width as f32,
            height as f32,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        let uniform: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        self.draw(
            device,
            queue,
            encoder,
            &self.select,
            from,
            &view,
            to,
            &uniform,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn draw(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        from: &wgpu::TextureView,
        selection: &wgpu::TextureView,
        to: &wgpu::TextureView,
        uniform: &[u8],
    ) {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cutlass-render cut params"),
            size: PARAMS,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, uniform);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cutlass-render cut"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(from),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(selection),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("cutlass-render cut"),
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
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn selection_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("cutlass-render selection"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

#[cfg(test)]
mod tests;
