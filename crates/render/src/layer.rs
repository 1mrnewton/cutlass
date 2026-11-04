//! One layer: a source texture, where it lands on the canvas, and how much.
//!
//! A canvas pixel center goes through three maps, each floored to a whole
//! pixel and moved to that pixel's center before the next, as the CPU path
//! does:
//!
//! 1. canvas to the placed picture (fit, transform, flip). Outside the
//!    picture is transparent.
//! 2. picture to the oriented source (crop). Outside it is transparent.
//! 3. oriented source to the stored texture (a quarter turn).
//!
//! Texels are fetched, not filtered. RGBA sources are premultiplied. A 4:2:0
//! source is converted with limited-range BT.709, the conversion the CPU
//! path applies, and has an alpha of one. Opacity scales the premultiplied
//! color.

use cutlass_effects::Step;
use cutlass_gpu::{Affine, Placement};

use crate::{CANVAS_FORMAT, RenderError};

/// Premultiplied RGBA bytes, row by row with no padding.
#[derive(Clone, Copy, Debug)]
pub struct Rgba<'a> {
    pub width: u32,
    pub height: u32,
    pub bytes: &'a [u8],
}

/// One 8-bit plane, row by row with no padding.
#[derive(Clone, Copy, Debug)]
pub struct Plane<'a> {
    pub width: u32,
    pub height: u32,
    pub bytes: &'a [u8],
}

/// Y, U, and V planes of one 4:2:0 picture. Chroma is half size, rounded down.
#[derive(Clone, Copy, Debug)]
pub struct Planes<'a> {
    pub y: Plane<'a>,
    pub u: Plane<'a>,
    pub v: Plane<'a>,
}

impl<'a> Planes<'a> {
    /// Split packed Y, then U, then V, as the decoders write 4:2:0.
    pub fn packed(width: u32, height: u32, bytes: &'a [u8]) -> Result<Self, RenderError> {
        let (cw, ch) = (width / 2, height / 2);
        let luma = width as usize * height as usize;
        let chroma = cw as usize * ch as usize;
        if bytes.len() != luma + 2 * chroma || cw == 0 || ch == 0 {
            return Err(RenderError::Bytes {
                expected: luma + 2 * chroma,
                actual: bytes.len(),
            });
        }
        let (y, rest) = bytes.split_at(luma);
        let (u, v) = rest.split_at(chroma);
        let plane = |width, height, bytes| Plane {
            width,
            height,
            bytes,
        };
        Ok(Self {
            y: plane(width, height, y),
            u: plane(cw, ch, u),
            v: plane(cw, ch, v),
        })
    }
}

/// What one layer draws.
#[derive(Clone, Debug)]
pub enum Layer<'a> {
    Rgba(Rgba<'a>, Placement),
    Yuv(Planes<'a>, Placement),
    /// A 4:2:0 picture over a blurred copy of itself that covers the canvas.
    /// `cover` maps canvas pixels to the picture for the copy; `radius` is the
    /// box blur's half width in canvas pixels. The copy is not faded.
    YuvOverBlur {
        planes: Planes<'a>,
        placement: Placement,
        cover: Placement,
        radius: u32,
    },
    /// Layers drawn onto their own `size` canvas first, then that canvas
    /// placed like a picture, as a compound clip is.
    Nested {
        layers: Vec<Layer<'a>>,
        size: (u32, u32),
        placement: Placement,
    },
    /// An effect over everything drawn before it in the same list.
    Effect(&'a [Step]),
}

/// A layer's textures once uploaded: three views (the same one three times
/// for RGBA), the source size, and whether it is 4:2:0.
pub(crate) struct Uploaded {
    pub(crate) views: [wgpu::TextureView; 3],
    pub(crate) size: (u32, u32),
    pub(crate) yuv: bool,
}

const SHADER: &str = r"
struct Params {
    picture_x: vec4<f32>,
    picture_y: vec4<f32>,
    oriented_x: vec4<f32>,
    oriented_y: vec4<f32>,
    texture_x: vec4<f32>,
    texture_y: vec4<f32>,
    // Picture width and height, oriented width and height.
    sizes: vec4<f32>,
    // Texture width and height, opacity, kind: 0 rgba, 1 yuv.
    source: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var tex0: texture_2d<f32>;
@group(0) @binding(2) var tex1: texture_2d<f32>;
@group(0) @binding(3) var tex2: texture_2d<f32>;

struct Out {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs(@builtin(vertex_index) index: u32) -> Out {
    // One triangle that covers the whole canvas.
    let x = f32((index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(index & 2u) * 2.0 - 1.0;
    var out: Out;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

fn map(row_x: vec4<f32>, row_y: vec4<f32>, p: vec2<f32>) -> vec2<f32> {
    let q = vec3<f32>(p, 1.0);
    return vec2<f32>(dot(row_x.xyz, q), dot(row_y.xyz, q));
}

fn outside(p: vec2<f32>, size: vec2<f32>) -> bool {
    return p.x < 0.0 || p.y < 0.0 || p.x >= size.x || p.y >= size.y;
}

// Picture pixel `index`, through the crop and the file turn to the
// texture. A pixel the crop leaves outside the source is clear.
fn picture_pixel(index: vec2<f32>) -> vec4<f32> {
    let oriented = map(params.oriented_x, params.oriented_y, index + 0.5);
    if (outside(oriented, params.sizes.zw)) {
        return vec4<f32>(0.0);
    }
    let stored = map(params.texture_x, params.texture_y, floor(oriented) + 0.5);
    let size = params.source.xy;
    let texel = vec2<i32>(clamp(floor(stored), vec2<f32>(0.0), size - 1.0));
    if (params.source.w < 0.5) {
        return textureLoad(tex0, texel, 0);
    }
    let chroma = texel / 2;
    // Limited-range BT.709, as the CPU conversion.
    let y = (textureLoad(tex0, texel, 0).r * 255.0 - 16.0) / 219.0;
    let cb = (textureLoad(tex1, chroma, 0).r * 255.0 - 128.0) / 224.0;
    let cr = (textureLoad(tex2, chroma, 0).r * 255.0 - 128.0) / 224.0;
    let kr = 0.2126;
    let kb = 0.0722;
    let kg = 1.0 - kr - kb;
    let rgb = vec3<f32>(
        y + 2.0 * (1.0 - kr) * cr,
        y - 2.0 * kb * (1.0 - kb) / kg * cb - 2.0 * kr * (1.0 - kr) / kg * cr,
        y + 2.0 * (1.0 - kb) * cb,
    );
    // Each channel is stored as a byte, as the CPU picture holds it.
    return vec4<f32>(round(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)) * 255.0) / 255.0, 1.0);
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    // `position.xy` is the canvas pixel center, and `picture` the matching
    // point of the placed picture, where pixel i spans i to i + 1.
    let picture = map(params.picture_x, params.picture_y, in.position.xy);
    if (outside(picture, params.sizes.xy)) {
        discard;
    }
    // Bilinear between the four nearest pixel centers, the edge pixel
    // repeated past the edge.
    let at = picture - 0.5;
    let base = floor(at);
    let weight = at - base;
    let last = params.sizes.xy - 1.0;
    let near = clamp(base, vec2<f32>(0.0), last);
    let far = clamp(base + 1.0, vec2<f32>(0.0), last);
    let top = mix(
        picture_pixel(near),
        picture_pixel(vec2<f32>(far.x, near.y)),
        weight.x,
    );
    let bottom = mix(
        picture_pixel(vec2<f32>(near.x, far.y)),
        picture_pixel(far),
        weight.x,
    );
    return mix(top, bottom, weight.y) * params.source.z;
}
";

/// Bytes in [`SHADER`]'s `Params`.
const PARAMS: u64 = 8 * 16;

pub(crate) struct Pipelines {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
}

pub(crate) struct Bound {
    group: wgpu::BindGroup,
}

impl Pipelines {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cutlass-render layer"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
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
            label: Some("cutlass-render layer"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(PARAMS),
                    },
                    count: None,
                },
                texture(1),
                texture(2),
                texture(3),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cutlass-render layer"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let over = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cutlass-render layer"),
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
                    blend: Some(wgpu::BlendState {
                        color: over,
                        alpha: over,
                    }),
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

    /// Upload one layer's pixels and parameters.
    /// Upload an RGBA picture.
    pub(crate) fn upload_rgba(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        rgba: &Rgba<'_>,
    ) -> Result<Uploaded, RenderError> {
        let view = upload(
            device,
            queue,
            rgba.width,
            rgba.height,
            rgba.bytes,
            wgpu::TextureFormat::Rgba8Unorm,
            4,
        )?;
        Ok(Uploaded {
            views: [view.clone(), view.clone(), view],
            size: (rgba.width, rgba.height),
            yuv: false,
        })
    }

    /// Upload the three planes of a 4:2:0 picture.
    pub(crate) fn upload_planes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        planes: &Planes<'_>,
    ) -> Result<Uploaded, RenderError> {
        let plane = |plane: &Plane<'_>| {
            upload(
                device,
                queue,
                plane.width,
                plane.height,
                plane.bytes,
                wgpu::TextureFormat::R8Unorm,
                1,
            )
        };
        Ok(Uploaded {
            views: [plane(&planes.y)?, plane(&planes.u)?, plane(&planes.v)?],
            size: (planes.y.width, planes.y.height),
            yuv: true,
        })
    }

    /// Bind uploaded textures with a placement.
    pub(crate) fn bind(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        uploaded: &Uploaded,
        placement: Placement,
    ) -> Bound {
        let size = uploaded.size;
        let kind = if uploaded.yuv { 1.0 } else { 0.0 };
        let row = |affine: Affine, index: usize| {
            let a = affine.0;
            [
                a[index * 3] as f32,
                a[index * 3 + 1] as f32,
                a[index * 3 + 2] as f32,
                0.0,
            ]
        };
        let values: [[f32; 4]; 8] = [
            row(placement.to_picture, 0),
            row(placement.to_picture, 1),
            row(placement.to_oriented, 0),
            row(placement.to_oriented, 1),
            row(placement.to_texture, 0),
            row(placement.to_texture, 1),
            [
                placement.picture.0 as f32,
                placement.picture.1 as f32,
                placement.oriented.0 as f32,
                placement.oriented.1 as f32,
            ],
            [
                size.0 as f32,
                size.1 as f32,
                placement.opacity.clamp(0.0, 1.0) as f32,
                kind,
            ],
        ];
        let mut bytes = Vec::with_capacity(PARAMS as usize);
        for value in values.iter().flatten() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cutlass-render params"),
            size: PARAMS,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, &bytes);
        let [t0, t1, t2] = &uploaded.views;
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cutlass-render layer"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(t0),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(t1),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(t2),
                },
            ],
        });
        Bound { group }
    }

    pub(crate) fn draw(&self, pass: &mut wgpu::RenderPass<'_>, layer: &Bound) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &layer.group, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    bytes: &[u8],
    format: wgpu::TextureFormat,
    texel: u32,
) -> Result<wgpu::TextureView, RenderError> {
    let limit = device.limits().max_texture_dimension_2d;
    if width == 0 || height == 0 || width > limit || height > limit {
        return Err(RenderError::Size { width, height });
    }
    let expected = width as usize * height as usize * texel as usize;
    if bytes.len() != expected {
        return Err(RenderError::Bytes {
            expected,
            actual: bytes.len(),
        });
    }
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("cutlass-render layer"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
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
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * texel),
            rows_per_image: Some(height),
        },
        size,
    );
    Ok(texture.create_view(&wgpu::TextureViewDescriptor::default()))
}
