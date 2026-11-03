//! Effect passes: one shader that runs any block of a recipe.
//!
//! An effect layer applies to everything drawn below it. Each step reads
//! the picture so far and writes a new one of the same size; two textures
//! take turns. Colours stay premultiplied between steps; a step that works
//! on colour divides by alpha first and multiplies again after. Sampling is
//! linear and mirrors at the edges, so a shaken or turned frame shows a
//! reflection at its border instead of a hole. A blur runs as two steps,
//! across and then down.

use cutlass_effects::{Block, Step};

use crate::CANVAS_FORMAT;

const SHADER: &str = r"
struct Effect {
    // The block, then for a blur 0 across and 1 down.
    code: vec2<u32>,
    size: vec2<f32>,
    a: vec4<f32>,
    b: vec4<f32>,
};

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var<uniform> fx: Effect;

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

fn fetch(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(source, linear_sampler, uv, 0.0);
}

fn pcg(value: u32) -> u32 {
    let state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// A number in [0, 1] from three whole numbers.
fn rand(a: u32, b: u32, seed: u32) -> f32 {
    return f32(pcg(a ^ pcg(b ^ pcg(seed)))) / 4294967295.0;
}

fn straight(c: vec4<f32>) -> vec3<f32> {
    if (c.a <= 0.0) {
        return vec3<f32>(0.0);
    }
    return c.rgb / c.a;
}

fn premultiplied(rgb: vec3<f32>, alpha: f32) -> vec4<f32> {
    return vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)) * alpha, alpha);
}

fn transform(uv: vec2<f32>) -> vec4<f32> {
    let scale = fx.a.x;
    let turn = radians(fx.a.w);
    let d = (uv - vec2<f32>(0.5) - fx.a.yz) * fx.size;
    let c = cos(turn);
    let s = sin(turn);
    let back = vec2<f32>(c * d.x + s * d.y, -s * d.x + c * d.y) / scale;
    return fetch(back / fx.size + vec2<f32>(0.5));
}

fn rgb_split(uv: vec2<f32>) -> vec4<f32> {
    let shift = fx.a.xy;
    let red = fetch(uv + shift);
    let middle = fetch(uv);
    let blue = fetch(uv - shift);
    return vec4<f32>(red.r, middle.g, blue.b, max(middle.a, max(red.a, blue.a)));
}

fn slices(uv: vec2<f32>) -> vec4<f32> {
    let count = max(floor(fx.a.x), 1.0);
    let band = u32(floor(uv.y * count));
    let seed = u32(fx.a.z);
    if (rand(band, 1u, seed) < 0.5) {
        return fetch(uv);
    }
    let shift = (rand(band, 2u, seed) * 2.0 - 1.0) * fx.a.y;
    return fetch(uv + vec2<f32>(shift, 0.0));
}

fn blur(uv: vec2<f32>) -> vec4<f32> {
    let short = min(fx.size.x, fx.size.y);
    let reach = fx.a.x * short;
    if (reach < 0.5) {
        return fetch(uv);
    }
    var direction = vec2<f32>(1.0 / fx.size.x, 0.0);
    if (fx.code.y == 1u) {
        direction = vec2<f32>(0.0, 1.0 / fx.size.y);
    }
    let stride = direction * reach / 7.0;
    var sum = vec4<f32>(0.0);
    for (var k = -7; k <= 7; k = k + 1) {
        sum = sum + fetch(uv + stride * f32(k));
    }
    return sum / 15.0;
}

fn grade(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    var rgb = straight(c) + vec3<f32>(fx.a.x);
    rgb = (rgb - vec3<f32>(0.5)) * fx.a.y + vec3<f32>(0.5);
    let luma = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    rgb = mix(vec3<f32>(luma), rgb, fx.a.z);
    return premultiplied(rgb, c.a);
}

fn flash(uv: vec2<f32>) -> vec4<f32> {
    return mix(fetch(uv), vec4<f32>(fx.a.yzw, 1.0), fx.a.x);
}

fn vignette(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    let d = length(uv - vec2<f32>(0.5)) * 1.4142;
    let dark = smoothstep(fx.a.y, fx.a.y + 0.5, d) * fx.a.x;
    return vec4<f32>(c.rgb * (1.0 - dark), c.a);
}

fn noise(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    let pixel = vec2<u32>(uv * fx.size);
    let grain = (rand(pixel.x, pixel.y, u32(fx.a.y)) - 0.5) * fx.a.x;
    return vec4<f32>(clamp(c.rgb + vec3<f32>(grain * c.a), vec3<f32>(0.0), vec3<f32>(c.a)), c.a);
}

fn scanlines(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    let line = u32(floor(uv.y * fx.a.y));
    if ((line & 1u) == 1u) {
        return vec4<f32>(c.rgb * (1.0 - fx.a.x), c.a);
    }
    return c;
}

fn mirror(uv: vec2<f32>) -> vec4<f32> {
    var p = uv;
    let side = u32(fx.a.x);
    if (side == 0u && p.x > 0.5) {
        p.x = 1.0 - p.x;
    } else if (side == 1u && p.x < 0.5) {
        p.x = 1.0 - p.x;
    } else if (side == 2u && p.y > 0.5) {
        p.y = 1.0 - p.y;
    } else if (side == 3u && p.y < 0.5) {
        p.y = 1.0 - p.y;
    }
    return fetch(p);
}

fn pixelate(uv: vec2<f32>) -> vec4<f32> {
    let cell = fx.a.x * min(fx.size.x, fx.size.y);
    if (cell < 1.0) {
        return fetch(uv);
    }
    let pixel = (floor(uv * fx.size / cell) + vec2<f32>(0.5)) * cell;
    return fetch(pixel / fx.size);
}

fn invert(uv: vec2<f32>) -> vec4<f32> {
    let c = fetch(uv);
    let rgb = straight(c);
    return premultiplied(mix(rgb, vec3<f32>(1.0) - rgb, fx.a.x), c.a);
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    let uv = in.position.xy / fx.size;
    switch (fx.code.x) {
        case 0u: { return transform(uv); }
        case 1u: { return rgb_split(uv); }
        case 2u: { return slices(uv); }
        case 3u: { return blur(uv); }
        case 4u: { return grade(uv); }
        case 5u: { return flash(uv); }
        case 6u: { return vignette(uv); }
        case 7u: { return noise(uv); }
        case 8u: { return scanlines(uv); }
        case 9u: { return mirror(uv); }
        case 10u: { return pixelate(uv); }
        case 11u: { return invert(uv); }
        default: { return fetch(uv); }
    }
}
";

/// Bytes in one step's uniform block.
const PARAMS: u64 = 48;

pub(crate) struct Effects {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl Effects {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cutlass-render effect"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cutlass-render effect"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
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
            label: Some("cutlass-render effect"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cutlass-render effect"),
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
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cutlass-render effect"),
            address_mode_u: wgpu::AddressMode::MirrorRepeat,
            address_mode_v: wgpu::AddressMode::MirrorRepeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..wgpu::SamplerDescriptor::default()
        });
        Self {
            pipeline,
            layout,
            sampler,
        }
    }

    /// Run `steps` over `first`, using `second` in between. Returns the
    /// texture that holds the result.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run<'v>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        first: &'v wgpu::TextureView,
        second: &'v wgpu::TextureView,
        size: (u32, u32),
        steps: &[Step],
    ) -> &'v wgpu::TextureView {
        let (mut from, mut to) = (first, second);
        for step in steps {
            let directions: &[u32] = if step.block == Block::Blur {
                &[0, 1]
            } else {
                &[0]
            };
            for direction in directions {
                self.pass(device, queue, encoder, from, to, size, step, *direction);
                std::mem::swap(&mut from, &mut to);
            }
        }
        from
    }

    #[allow(clippy::too_many_arguments)]
    fn pass(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        from: &wgpu::TextureView,
        to: &wgpu::TextureView,
        size: (u32, u32),
        step: &Step,
        direction: u32,
    ) {
        let mut bytes = Vec::with_capacity(PARAMS as usize);
        bytes.extend_from_slice(&step.block.code().to_le_bytes());
        bytes.extend_from_slice(&direction.to_le_bytes());
        for value in [size.0 as f32, size.1 as f32] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in step.values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cutlass-render effect params"),
            size: PARAMS,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, &bytes);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cutlass-render effect"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(from),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("cutlass-render effect"),
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
    }
}

#[cfg(test)]
mod tests {
    use cutlass_effects::{Block, MAX_VALUES, Step, bundled};
    use cutlass_gpu::Placement;
    use cutlass_model::Micros;

    use crate::{Layer, Renderer, Rgba};

    fn renderer() -> Option<Renderer> {
        let renderer = Renderer::for_test().expect("a GPU is required");
        if renderer.is_none() {
            eprintln!("skipped: no GPU");
        }
        renderer
    }

    fn step(block: Block, set: &[(usize, f32)]) -> Step {
        let mut values = [0.0; MAX_VALUES];
        for (slot, param) in values.iter_mut().zip(block.params()) {
            *slot = param.default;
        }
        for (index, value) in set {
            values[*index] = *value;
        }
        Step { block, values }
    }

    /// Red, green, blue, white, all opaque.
    const ROW: [u8; 16] = [
        255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
    ];

    fn draw(renderer: &Renderer, bytes: &[u8], size: (u32, u32), steps: &[Step]) -> Vec<u8> {
        let canvas = renderer.canvas(size.0, size.1).expect("canvas");
        let picture = Rgba {
            width: size.0,
            height: size.1,
            bytes,
        };
        renderer
            .draw(
                &canvas,
                &[
                    Layer::Rgba(picture, Placement::canvas(size.0, size.1)),
                    Layer::Effect(steps),
                ],
            )
            .expect("draw");
        renderer.read(&canvas).expect("read")
    }

    fn close(got: &[u8], expected: &[u8]) {
        assert_eq!(got.len(), expected.len());
        for (index, (a, b)) in got.iter().zip(expected).enumerate() {
            assert!(a.abs_diff(*b) <= 2, "byte {index}: {got:?} vs {expected:?}");
        }
    }

    #[test]
    fn no_steps_and_resting_steps_leave_the_picture_alone() {
        let Some(renderer) = renderer() else { return };
        close(&draw(&renderer, &ROW, (4, 1), &[]), &ROW);
        let resting = [
            step(Block::Transform, &[]),
            step(Block::RgbSplit, &[]),
            step(Block::Blur, &[]),
            step(Block::Grade, &[]),
            step(Block::Flash, &[]),
            step(Block::Pixelate, &[]),
            step(Block::Invert, &[(0, 0.0)]),
        ];
        close(&draw(&renderer, &ROW, (4, 1), &resting), &ROW);
    }

    #[test]
    fn invert_grade_and_flash_change_colour_as_named() {
        let Some(renderer) = renderer() else { return };
        let inverted = draw(&renderer, &ROW, (4, 1), &[step(Block::Invert, &[])]);
        close(
            &inverted,
            &[
                0, 255, 255, 255, 255, 0, 255, 255, 255, 255, 0, 255, 0, 0, 0, 255,
            ],
        );
        let grey = draw(&renderer, &ROW, (4, 1), &[step(Block::Grade, &[(2, 0.0)])]);
        for pixel in grey.chunks(4) {
            assert!(pixel[0].abs_diff(pixel[1]) <= 1 && pixel[1].abs_diff(pixel[2]) <= 1);
        }
        let white = draw(&renderer, &ROW, (4, 1), &[step(Block::Flash, &[(0, 1.0)])]);
        close(&white, &[255; 16]);
    }

    #[test]
    fn mirror_copies_the_left_half_onto_the_right() {
        let Some(renderer) = renderer() else { return };
        let mirrored = draw(&renderer, &ROW, (4, 1), &[step(Block::Mirror, &[])]);
        close(
            &mirrored,
            &[
                255, 0, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 255, 0, 0, 255,
            ],
        );
    }

    #[test]
    fn an_effect_changes_only_what_is_below_it() {
        let Some(renderer) = renderer() else { return };
        let canvas = renderer.canvas(4, 1).expect("canvas");
        let blue = [0, 0, 255, 255];
        let mut place = Placement::canvas(1, 1);
        place.to_picture = cutlass_gpu::Affine([1.0, 0.0, -3.0, 0.0, 1.0, 0.0]);
        let steps = [step(Block::Invert, &[])];
        renderer
            .draw(
                &canvas,
                &[
                    Layer::Rgba(
                        Rgba {
                            width: 4,
                            height: 1,
                            bytes: &ROW,
                        },
                        Placement::canvas(4, 1),
                    ),
                    Layer::Effect(&steps),
                    Layer::Rgba(
                        Rgba {
                            width: 1,
                            height: 1,
                            bytes: &blue,
                        },
                        place,
                    ),
                ],
            )
            .expect("draw");
        let got = renderer.read(&canvas).expect("read");
        close(
            &got[..12],
            &[0, 255, 255, 255, 255, 0, 255, 255, 255, 255, 0, 255],
        );
        close(&got[12..], &blue);
    }

    #[test]
    fn colour_steps_keep_transparent_pixels_transparent() {
        let Some(renderer) = renderer() else { return };
        let clear = [0u8; 16];
        let steps = [
            step(Block::Grade, &[(0, 0.5), (1, 2.0)]),
            step(Block::Invert, &[]),
            step(Block::Noise, &[(0, 1.0), (1, 9.0)]),
            step(Block::Vignette, &[(0, 1.0)]),
        ];
        close(&draw(&renderer, &clear, (4, 1), &steps), &clear);
    }

    #[test]
    fn every_bundled_recipe_runs_and_moves_the_picture() {
        let Some(renderer) = renderer() else { return };
        let (width, height) = (32u32, 24u32);
        let scene: Vec<u8> = (0..width * height)
            .flat_map(|index| {
                let (x, y) = (index % width, index / width);
                [(x * 8) as u8, (y * 10) as u8, ((x + y) * 4) as u8, 255]
            })
            .collect();
        for recipe in bundled() {
            let mut changed = false;
            for tenth in 0..20 {
                let steps = recipe.steps_at(Micros(tenth * 100_000));
                let got = draw(&renderer, &scene, (width, height), &steps);
                assert_eq!(got.len(), scene.len(), "{}", recipe.id);
                changed |= got != scene;
            }
            assert!(changed, "{} never changed the picture", recipe.id);
        }
    }
}
