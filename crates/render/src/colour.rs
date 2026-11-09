//! Colour passes: a clip's chroma key, grade, adjust sliders and LUT run
//! over the picture drawn so far, as `cutlass_effects::Colour::apply` runs
//! them on the CPU.
//!
//! Each pass reads premultiplied RGBA and writes it again in the canvas
//! format, so values round to bytes between passes as they do on the CPU.
//! Adjust sliders that look at neighbours (clarity, sharpen) keep the
//! CPU's unrounded straight colour between their steps in a half-float
//! texture: the tone, a box blur across and down, the mix with the blur,
//! then vignette and grain on the way back to bytes.

use cutlass_effects::{COLOUR_UNIFORM_BYTES, COLOUR_WGSL, Colour, ColourPass, ColourStage};

use crate::CANVAS_FORMAT;

/// Unrounded straight colour between the steps of clarity and sharpen.
const FLOAT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The LUT's format: its points exactly as the CPU reads them.
const LUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

const SHADER: &str = r"
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var other: texture_2d<f32>;
@group(0) @binding(2) var<uniform> colour_params: ColourParams;
@group(0) @binding(3) var colour_lut: texture_3d<f32>;

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

fn here(in: Out) -> vec2<i32> {
    return vec2<i32>(in.position.xy);
}

@fragment
fn fs_chroma(in: Out) -> @location(0) vec4<f32> {
    return chroma_key(textureLoad(source, here(in), 0));
}

@fragment
fn fs_grade(in: Out) -> @location(0) vec4<f32> {
    return grade_pixel(textureLoad(source, here(in), 0));
}

@fragment
fn fs_lut(in: Out) -> @location(0) vec4<f32> {
    return lut_pixel(textureLoad(source, here(in), 0));
}

@fragment
fn fs_adjust(in: Out) -> @location(0) vec4<f32> {
    let toned = adjust_toned(textureLoad(source, here(in), 0));
    return adjust_finish(toned.rgb, toned.a, vec2<u32>(in.position.xy));
}

@fragment
fn fs_tone(in: Out) -> @location(0) vec4<f32> {
    return adjust_toned(textureLoad(source, here(in), 0));
}

// The mean of `2r + 1` colours along `axis`, the edge colour repeated past
// the edge; alpha is the pixel's own.
fn box_mean(at: vec2<i32>, axis: vec2<i32>) -> vec4<f32> {
    let radius = i32(colour_params.frame.w);
    let last = vec2<i32>(textureDimensions(source)) - vec2<i32>(1);
    var sum = vec3<f32>(0.0);
    for (var k = -radius; k <= radius; k = k + 1) {
        let near = clamp(at + axis * k, vec2<i32>(0), last);
        sum = sum + textureLoad(source, near, 0).rgb;
    }
    return vec4<f32>(sum / f32(2 * radius + 1), textureLoad(source, at, 0).a);
}

@fragment
fn fs_box_across(in: Out) -> @location(0) vec4<f32> {
    return box_mean(here(in), vec2<i32>(1, 0));
}

@fragment
fn fs_box_down(in: Out) -> @location(0) vec4<f32> {
    return box_mean(here(in), vec2<i32>(0, 1));
}

@fragment
fn fs_clarity(in: Out) -> @location(0) vec4<f32> {
    let base = textureLoad(source, here(in), 0);
    let soft = textureLoad(other, here(in), 0).rgb;
    let weight = colour_midtones(colour_luma(base.rgb));
    return vec4<f32>(base.rgb + colour_params.v[3].y * weight * (base.rgb - soft), base.a);
}

@fragment
fn fs_sharpen(in: Out) -> @location(0) vec4<f32> {
    let base = textureLoad(source, here(in), 0);
    let soft = textureLoad(other, here(in), 0).rgb;
    return vec4<f32>(base.rgb + 1.5 * colour_params.v[3].z * (base.rgb - soft), base.a);
}

@fragment
fn fs_finish(in: Out) -> @location(0) vec4<f32> {
    let toned = textureLoad(source, here(in), 0);
    return adjust_finish(toned.rgb, toned.a, vec2<u32>(in.position.xy));
}
";

pub(crate) struct Colours {
    layout: wgpu::BindGroupLayout,
    chroma: wgpu::RenderPipeline,
    grade: wgpu::RenderPipeline,
    adjust: wgpu::RenderPipeline,
    lut: wgpu::RenderPipeline,
    tone: wgpu::RenderPipeline,
    across: wgpu::RenderPipeline,
    down: wgpu::RenderPipeline,
    clarity: wgpu::RenderPipeline,
    sharpen: wgpu::RenderPipeline,
    finish: wgpu::RenderPipeline,
    /// Bound in place of a LUT by every pass but the LUT's own.
    no_lut: wgpu::TextureView,
}

/// Where one pass reads and writes, and with what.
struct Draw<'a> {
    pipeline: &'a wgpu::RenderPipeline,
    from: &'a wgpu::TextureView,
    other: &'a wgpu::TextureView,
    to: &'a wgpu::TextureView,
    uniform: &'a [u8],
    lut: &'a wgpu::TextureView,
}

impl Colours {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let source = format!("{SHADER}\n{COLOUR_WGSL}");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cutlass-render colour"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let texture = |binding, view_dimension| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cutlass-render colour"),
            entries: &[
                texture(0, wgpu::TextureViewDimension::D2),
                texture(1, wgpu::TextureViewDimension::D2),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(COLOUR_UNIFORM_BYTES as u64),
                    },
                    count: None,
                },
                texture(3, wgpu::TextureViewDimension::D3),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cutlass-render colour"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str, format| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("cutlass-render colour"),
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
                        format,
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
        let no_lut = lut_texture(device, 1).create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            chroma: pipeline("fs_chroma", CANVAS_FORMAT),
            grade: pipeline("fs_grade", CANVAS_FORMAT),
            adjust: pipeline("fs_adjust", CANVAS_FORMAT),
            lut: pipeline("fs_lut", CANVAS_FORMAT),
            tone: pipeline("fs_tone", FLOAT_FORMAT),
            across: pipeline("fs_box_across", FLOAT_FORMAT),
            down: pipeline("fs_box_down", FLOAT_FORMAT),
            clarity: pipeline("fs_clarity", FLOAT_FORMAT),
            sharpen: pipeline("fs_sharpen", FLOAT_FORMAT),
            finish: pipeline("fs_finish", CANVAS_FORMAT),
            layout,
            no_lut,
        }
    }

    /// Run `colour` over `input`, a `size` texture in the canvas format.
    /// `seed` moves the grain. Returns the texture that holds the result;
    /// `input` may be written.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        size: (u32, u32),
        colour: &Colour,
        seed: u32,
    ) -> wgpu::TextureView {
        let mut current = input.clone();
        let mut spare: Option<wgpu::TextureView> = None;
        for pass in colour.passes() {
            let to = spare
                .take()
                .unwrap_or_else(|| scratch(device, size, CANVAS_FORMAT));
            let frame = [size.0, size.1, seed, 0];
            let uniform = pass.uniform(frame);
            match pass.stage {
                ColourStage::Chroma | ColourStage::Grade | ColourStage::Lut => {
                    let pipeline = match pass.stage {
                        ColourStage::Chroma => &self.chroma,
                        ColourStage::Grade => &self.grade,
                        ColourStage::Adjust | ColourStage::Lut => &self.lut,
                    };
                    let table = pass.lut.map(|lut| upload_lut(device, queue, lut));
                    self.draw(
                        device,
                        queue,
                        encoder,
                        &Draw {
                            pipeline,
                            from: &current,
                            other: &current,
                            to: &to,
                            uniform: &uniform,
                            lut: table.as_ref().unwrap_or(&self.no_lut),
                        },
                    );
                }
                ColourStage::Adjust if pass.reads_neighbours() => {
                    self.neighbours(device, queue, encoder, &current, &to, size, &pass, seed);
                }
                ColourStage::Adjust => {
                    self.draw(
                        device,
                        queue,
                        encoder,
                        &Draw {
                            pipeline: &self.adjust,
                            from: &current,
                            other: &current,
                            to: &to,
                            uniform: &uniform,
                            lut: &self.no_lut,
                        },
                    );
                }
            }
            spare = Some(std::mem::replace(&mut current, to));
        }
        current
    }

    /// An adjust pass with clarity or sharpen, from `from` into `to`.
    #[allow(clippy::too_many_arguments)]
    fn neighbours(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        from: &wgpu::TextureView,
        to: &wgpu::TextureView,
        size: (u32, u32),
        pass: &ColourPass<'_>,
        seed: u32,
    ) {
        let mut step = |pipeline: &wgpu::RenderPipeline,
                        from: &wgpu::TextureView,
                        other: &wgpu::TextureView,
                        to: &wgpu::TextureView,
                        radius: u32| {
            let uniform = pass.uniform([size.0, size.1, seed, radius]);
            self.draw(
                device,
                queue,
                encoder,
                &Draw {
                    pipeline,
                    from,
                    other,
                    to,
                    uniform: &uniform,
                    lut: &self.no_lut,
                },
            );
        };
        let mut base = scratch(device, size, FLOAT_FORMAT);
        let mut line = scratch(device, size, FLOAT_FORMAT);
        let soft = scratch(device, size, FLOAT_FORMAT);
        step(&self.tone, from, from, &base, 0);
        for (amount, pipeline, radius) in [
            (
                pass.clarity(),
                &self.clarity,
                cutlass_effects::clarity_radius(size.0, size.1),
            ),
            (pass.sharpen(), &self.sharpen, 1),
        ] {
            if amount == 0.0 {
                continue;
            }
            step(&self.across, &base, &base, &line, radius);
            step(&self.down, &line, &line, &soft, radius);
            step(pipeline, &base, &soft, &line, radius);
            std::mem::swap(&mut base, &mut line);
        }
        step(&self.finish, &base, &base, to, 0);
    }

    fn draw(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        draw: &Draw<'_>,
    ) {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cutlass-render colour params"),
            size: COLOUR_UNIFORM_BYTES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, draw.uniform);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cutlass-render colour"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(draw.from),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(draw.other),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(draw.lut),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("cutlass-render colour"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: draw.to,
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
        pass.set_pipeline(draw.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn scratch(
    device: &wgpu::Device,
    size: (u32, u32),
    format: wgpu::TextureFormat,
) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("cutlass-render colour scratch"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

fn lut_texture(device: &wgpu::Device, side: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("cutlass-render lut"),
        size: wgpu::Extent3d {
            width: side,
            height: side,
            depth_or_array_layers: side,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: LUT_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

#[cfg(test)]
mod tests;

/// `lut` as a 3D texture: red across, green down, blue deep, as the table
/// is laid out.
fn upload_lut(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    lut: &cutlass_effects::Lut,
) -> wgpu::TextureView {
    let side = u32::try_from(lut.size()).unwrap_or(1);
    let texture = lut_texture(device, side);
    let bytes: Vec<u8> = lut
        .table()
        .iter()
        .flat_map(|[r, g, b]| [*r, *g, *b, 1.0])
        .flat_map(f32::to_le_bytes)
        .collect();
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
            bytes_per_row: Some(side * 16),
            rows_per_image: Some(side),
        },
        wgpu::Extent3d {
            width: side,
            height: side,
            depth_or_array_layers: side,
        },
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}
