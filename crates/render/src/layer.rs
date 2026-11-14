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
//! source is converted with limited-range BT.709 and then its [`YuvTone`],
//! the conversion the CPU path applies, and has an alpha of one. Opacity
//! scales the premultiplied color.

use cutlass_decode::Surface;
use cutlass_effects::Step;
use cutlass_gpu::{Affine, Placement, ToneCurve, YuvMatrix, YuvTone};

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

/// Where one 4:2:0 picture's samples are.
#[derive(Clone, Copy, Debug)]
pub enum PlaneSource<'a> {
    /// Y, U, and V planes. Chroma is half size, rounded down.
    Packed {
        y: Plane<'a>,
        u: Plane<'a>,
        v: Plane<'a>,
    },
    /// A platform GPU buffer of `width` by `height`. The renderer's
    /// [`SurfaceImport`](crate::SurfaceImport) samples it in place; without
    /// one, the bytes it copies out are drawn.
    Surface {
        surface: &'a Surface,
        width: u32,
        height: u32,
    },
}

/// One 4:2:0 picture. `tone` names its range and matrix and what follows
/// the matrix.
#[derive(Clone, Copy, Debug)]
pub struct Planes<'a> {
    pub source: PlaneSource<'a>,
    pub tone: YuvTone,
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
            source: PlaneSource::Packed {
                y: plane(width, height, y),
                u: plane(cw, ch, u),
                v: plane(cw, ch, v),
            },
            tone: YuvTone::NONE,
        })
    }

    /// A platform surface of `width` by `height`.
    #[must_use]
    pub fn surface(surface: &'a Surface, width: u32, height: u32) -> Self {
        Self {
            source: PlaneSource::Surface {
                surface,
                width,
                height,
            },
            tone: YuvTone::NONE,
        }
    }

    /// The same planes read with the range, matrix and transfer step of
    /// `tone`.
    #[must_use]
    pub fn with_tone(self, tone: YuvTone) -> Self {
        Self { tone, ..self }
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
    Effect(&'a [Step], Option<&'a cutlass_effects::CanvasMask>),
    /// Colour work (a chroma key, grades, adjust sliders, a LUT) over
    /// everything drawn before it in the same list, shown only inside
    /// `mask` when it has one. `seed` moves the grain.
    Colour {
        colour: &'a cutlass_effects::Colour,
        seed: u32,
        mask: Option<&'a cutlass_effects::CanvasMask>,
    },
    /// Everything drawn before it in the same list, kept as much as the
    /// mask keeps: a clip's own mask.
    Cut(&'a cutlass_effects::CanvasMask),
    /// Everything drawn before it in the same list, kept only where a
    /// clip's selection is.
    Select(crate::Selected<'a>),
    /// Everything drawn before it in the same list, lit again from a depth
    /// picture stretched to the canvas. It reads the canvas at the size it
    /// is drawn, so it belongs on a [`Layer::Nested`] canvas, which is
    /// drawn whole.
    Relight(crate::Relit<'a>),
    /// Everything drawn before it in the same list, painted by an effect
    /// package's stand-in. A blend package's own pass is only its fade;
    /// [`Layer::Blend`] mixes.
    Package(&'a cutlass_effects::PackagePass),
    /// `layers` drawn on a canvas the size of this one, then mixed by
    /// `mode` with everything drawn before it in the same list.
    Blend {
        mode: cutlass_effects::BlendMode,
        layers: Vec<Layer<'a>>,
    },
    /// Moment `t` of a built-in transition from `outgoing` to `incoming`,
    /// both the size of the canvas, drawn as a picture over the whole
    /// canvas.
    Transition {
        kind: cutlass_effects::TransitionKind,
        t: f32,
        outgoing: Rgba<'a>,
        incoming: Rgba<'a>,
    },
    /// One moment of a transition recipe from `outgoing` to `incoming`,
    /// both the size of the canvas: each picture through its own steps,
    /// then the swap. A [`Self::Transition`] whose kind is a recipe draws
    /// the same; this one takes a recipe that has no kind yet, such as one
    /// being made.
    Crossing {
        crossing: &'a cutlass_effects::Crossing,
        outgoing: Rgba<'a>,
        incoming: Rgba<'a>,
    },
    /// A Lottie frame drawn from its edges onto its own canvas, then that
    /// canvas placed like a picture.
    Vector {
        scene: &'a cutlass_lottie::Scene,
        placement: Placement,
    },
}

/// A layer's textures once uploaded: three views (the same one three times
/// for RGBA), the source size, and whether it is 4:2:0.
pub(crate) struct Uploaded {
    pub(crate) views: [wgpu::TextureView; 3],
    pub(crate) size: (u32, u32),
    pub(crate) yuv: bool,
    pub(crate) tone: YuvTone,
    /// Chroma is one two-channel texture, Cb in red and Cr in green.
    pub(crate) interleaved: bool,
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
    // Texture width and height, opacity, kind: 0 rgba, 1 yuv, 2 yuv with
    // Cb and Cr interleaved in one texture.
    source: vec4<f32>,
    // Tone after the matrix: source curve, destination curve (0 Rec.709,
    // 1 HLG, 2 PQ), PQ reference nits, and 1 when it changes the sample.
    tone: vec4<f32>,
    // Linear gamut adaptation rows; w is 1 on the first row when there is
    // one. On the second row w is the matrix of a yuv picture (0 BT.709,
    // 1 BT.601, 2 BT.2020), and on the third 1 when its samples are full
    // range.
    adapt_r: vec4<f32>,
    adapt_g: vec4<f32>,
    adapt_b: vec4<f32>,
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

// The curves in the color crate's transfer module.
const HLG_A: f32 = 0.17883277;
const HLG_B: f32 = 0.28466892;
const HLG_C: f32 = 0.55991073;
const PQ_C1: f32 = 0.8359375;
const PQ_C2: f32 = 18.8515625;
const PQ_C3: f32 = 18.6875;
const PQ_M1: f32 = 0.1593017578125;
const PQ_M2: f32 = 78.84375;

fn decode_curve(encoded: vec3<f32>, curve: f32, nits: f32) -> vec3<f32> {
    let e = clamp(encoded, vec3<f32>(0.0), vec3<f32>(1.0));
    if (curve < 0.5) {
        return select(
            pow((e + 0.099) / 1.099, vec3<f32>(1.0 / 0.45)),
            e / 4.5,
            e < vec3<f32>(0.081),
        );
    }
    if (curve < 1.5) {
        return select(
            (exp((e - HLG_C) / HLG_A) + HLG_B) / 12.0,
            e * e / 3.0,
            e <= vec3<f32>(0.5),
        );
    }
    let raised = pow(e, vec3<f32>(1.0 / PQ_M2));
    let ratio = max(raised - PQ_C1, vec3<f32>(0.0)) / (PQ_C2 - PQ_C3 * raised);
    let light = select(10000.0 * pow(ratio, vec3<f32>(1.0 / PQ_M1)), vec3<f32>(0.0), raised <= vec3<f32>(PQ_C1));
    return clamp(light / nits, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn encode_curve(linear: vec3<f32>, curve: f32, nits: f32) -> vec3<f32> {
    let l = clamp(linear, vec3<f32>(0.0), vec3<f32>(1.0));
    if (curve < 0.5) {
        return select(1.099 * pow(l, vec3<f32>(0.45)) - 0.099, 4.5 * l, l < vec3<f32>(0.018));
    }
    if (curve < 1.5) {
        return select(
            HLG_A * log(max(12.0 * l - HLG_B, vec3<f32>(1.0e-6))) + HLG_C,
            sqrt(3.0 * l),
            l <= vec3<f32>(1.0 / 12.0),
        );
    }
    let y = clamp(l * nits / 10000.0, vec3<f32>(0.0), vec3<f32>(1.0));
    let ym = pow(y, vec3<f32>(PQ_M1));
    let signal = pow((PQ_C1 + PQ_C2 * ym) / (1.0 + PQ_C3 * ym), vec3<f32>(PQ_M2));
    return select(signal, vec3<f32>(0.0), l * nits <= vec3<f32>(0.0));
}

fn tone(rgb: vec3<f32>) -> vec3<f32> {
    if (params.tone.w < 0.5) {
        return rgb;
    }
    var linear = decode_curve(rgb, params.tone.x, params.tone.z);
    if (params.adapt_r.w > 0.5) {
        linear = clamp(
            vec3<f32>(
                dot(params.adapt_r.xyz, linear),
                dot(params.adapt_g.xyz, linear),
                dot(params.adapt_b.xyz, linear),
            ),
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );
    }
    return encode_curve(linear, params.tone.y, params.tone.z);
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
    // The range and the matrix the stream names, as the CPU conversion:
    // studio swing is (Y - 16) / 219 and (C - 128) / 224, full range
    // Y / 255 and (C - 128) / 255.
    var y = textureLoad(tex0, texel, 0).r * 255.0;
    var cbcr = vec2<f32>(textureLoad(tex1, chroma, 0).r, textureLoad(tex2, chroma, 0).r);
    if (params.source.w > 1.5) {
        cbcr = textureLoad(tex1, chroma, 0).rg;
    }
    var cb = cbcr.x * 255.0 - 128.0;
    var cr = cbcr.y * 255.0 - 128.0;
    if (params.adapt_b.w > 0.5) {
        y = y / 255.0;
        cb = cb / 255.0;
        cr = cr / 255.0;
    } else {
        y = (y - 16.0) / 219.0;
        cb = cb / 224.0;
        cr = cr / 224.0;
    }
    var kr = 0.2126;
    var kb = 0.0722;
    if (params.adapt_g.w > 1.5) {
        kr = 0.2627;
        kb = 0.0593;
    } else if (params.adapt_g.w > 0.5) {
        kr = 0.299;
        kb = 0.114;
    }
    let kg = 1.0 - kr - kb;
    let rgb = tone(clamp(
        vec3<f32>(
            y + 2.0 * (1.0 - kr) * cr,
            y - 2.0 * kb * (1.0 - kb) / kg * cb - 2.0 * kr * (1.0 - kr) / kg * cr,
            y + 2.0 * (1.0 - kb) * cb,
        ),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    ));
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
const PARAMS: u64 = 12 * 16;

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
            tone: YuvTone::NONE,
            interleaved: false,
        })
    }

    /// Upload the three planes of a 4:2:0 picture, or import its surface
    /// through `import`. A surface that does not import is copied out and
    /// uploaded.
    pub(crate) fn upload_planes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        planes: &Planes<'_>,
        import: Option<&crate::SurfaceImport>,
    ) -> Result<Uploaded, RenderError> {
        let (y, u, v) = match planes.source {
            PlaneSource::Packed { y, u, v } => (y, u, v),
            PlaneSource::Surface {
                surface,
                width,
                height,
            } => {
                if let Some([luma, chroma]) = import.and_then(|import| import(device, surface)) {
                    let view = |texture: &wgpu::Texture| {
                        texture.create_view(&wgpu::TextureViewDescriptor::default())
                    };
                    let chroma = view(&chroma);
                    return Ok(Uploaded {
                        views: [view(&luma), chroma.clone(), chroma],
                        size: (width, height),
                        yuv: true,
                        tone: planes.tone,
                        interleaved: true,
                    });
                }
                let bytes = surface.yuv420p().map_err(|_| RenderError::Bytes {
                    expected: width as usize * height as usize * 3 / 2,
                    actual: 0,
                })?;
                let packed = Planes::packed(width, height, &bytes)?.with_tone(planes.tone);
                return Self::upload_planes(device, queue, &packed, None);
            }
        };
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
            views: [plane(&y)?, plane(&u)?, plane(&v)?],
            size: (y.width, y.height),
            yuv: true,
            tone: planes.tone,
            interleaved: false,
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
        let kind = match (uploaded.yuv, uploaded.interleaved) {
            (false, _) => 0.0,
            (true, false) => 1.0,
            (true, true) => 2.0,
        };
        let row = |affine: Affine, index: usize| {
            let a = affine.0;
            [
                a[index * 3] as f32,
                a[index * 3 + 1] as f32,
                a[index * 3 + 2] as f32,
                0.0,
            ]
        };
        let tone = uploaded.tone;
        let curve = |curve| match curve {
            ToneCurve::Rec709 => 0.0,
            ToneCurve::Hlg => 1.0,
            ToneCurve::Pq => 2.0,
        };
        let adapt = tone.adapt.unwrap_or([[0.0; 3]; 3]);
        let has_adapt = if tone.adapt.is_some() { 1.0 } else { 0.0 };
        let adapt_row = |index: usize, flag: f32| {
            let r = adapt[index];
            [r[0] as f32, r[1] as f32, r[2] as f32, flag]
        };
        let values: [[f32; 4]; 12] = [
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
            [
                curve(tone.source),
                curve(tone.dest),
                tone.nits as f32,
                if uploaded.yuv && !tone.is_none() {
                    1.0
                } else {
                    0.0
                },
            ],
            adapt_row(0, has_adapt),
            adapt_row(
                1,
                match tone.matrix {
                    YuvMatrix::Bt709 => 0.0,
                    YuvMatrix::Bt601 => 1.0,
                    YuvMatrix::Bt2020 => 2.0,
                },
            ),
            adapt_row(2, if tone.full_range { 1.0 } else { 0.0 }),
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
