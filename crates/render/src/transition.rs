//! The transition pass: one moment of a built-in transition from the
//! outgoing picture to the incoming one.
//!
//! Both pictures are canvas-sized. Each pixel is mixed by
//! `cutlass_effects`'s WGSL `transition`, the same arithmetic the CPU path
//! runs; a blur transition gets both pictures already blurred.

use cutlass_effects::{TRANSITION_WGSL, TransitionKind};

use crate::CANVAS_FORMAT;

const SHADER: &str = r"
@group(0) @binding(0) var outgoing_picture: texture_2d<f32>;
@group(0) @binding(1) var incoming_picture: texture_2d<f32>;
// Kind, t as bits, width, height.
@group(0) @binding(2) var<uniform> params: vec4<u32>;

fn tr_load(which: u32, texel: vec2<i32>) -> vec4<f32> {
    if (which == 0u) {
        return textureLoad(outgoing_picture, texel, 0);
    }
    return textureLoad(incoming_picture, texel, 0);
}

@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(index & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<f32>(f32(params.z), f32(params.w));
    return transition(params.x, bitcast<f32>(params.y), vec2<i32>(position.xy), size);
}
";

/// Bytes in the uniform block.
const PARAMS: u64 = 16;

pub(crate) struct Transition {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
}

impl Transition {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let source = format!("{SHADER}\n{TRANSITION_WGSL}");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cutlass-render transition"),
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
            label: Some("cutlass-render transition"),
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
            label: Some("cutlass-render transition"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cutlass-render transition"),
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

    /// Write moment `t` of `kind` from `outgoing` to `incoming`, both
    /// `size`, into `out`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        outgoing: &wgpu::TextureView,
        incoming: &wgpu::TextureView,
        out: &wgpu::TextureView,
        (kind, t, size): (TransitionKind, f32, (u32, u32)),
    ) {
        let mut bytes = [0u8; 16];
        bytes[..4].copy_from_slice(&kind.code().to_le_bytes());
        bytes[4..8].copy_from_slice(&t.to_bits().to_le_bytes());
        bytes[8..12].copy_from_slice(&size.0.to_le_bytes());
        bytes[12..].copy_from_slice(&size.1.to_le_bytes());
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cutlass-render transition params"),
            size: PARAMS,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, &bytes);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cutlass-render transition"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(outgoing),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(incoming),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("cutlass-render transition"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: out,
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
    use cutlass_effects::{TransitionKind, mix_transition};

    use crate::{Layer, Renderer, Rgba};

    const W: u32 = 24;
    const H: u32 = 16;

    /// A picture with something different in every pixel, part clear in
    /// one corner, so moves and turns show.
    fn picture(seed: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        for y in 0..H {
            for x in 0..W {
                let value = |k: u32| u8::try_from((x * 37 + y * 91 + seed * k) % 256).unwrap();
                let alpha = if x < 3 && y < 3 { 0 } else { 255 };
                let premultiplied = |v: u8| u8::try_from(u32::from(v) * alpha / 255).unwrap();
                bytes.extend([
                    premultiplied(value(13)),
                    premultiplied(value(29)),
                    premultiplied(value(53)),
                    u8::try_from(alpha).unwrap(),
                ]);
            }
        }
        bytes
    }

    #[test]
    fn every_recipe_transition_starts_on_the_outgoing_and_ends_on_the_incoming() {
        let Some(renderer) = Renderer::for_test().expect("a GPU is required") else {
            eprintln!("skipped: no GPU");
            return;
        };
        let (w, h) = (64u32, 48u32);
        let scene = |seed: u32| -> Vec<u8> {
            (0..w * h)
                .flat_map(|at| {
                    let (x, y) = (at % w, at / w);
                    let value =
                        |k: u32| u8::try_from((x * 4 * k + y * 5 + seed * 40) % 256).unwrap();
                    [value(1), value(2), value(3), 255]
                })
                .collect()
        };
        let (outgoing, incoming) = (scene(1), scene(4));
        let canvas = renderer.canvas(w, h).expect("canvas");
        let draw = |kind, t| {
            let picture = |bytes| Rgba {
                width: w,
                height: h,
                bytes,
            };
            renderer
                .draw(
                    &canvas,
                    &[Layer::Transition {
                        kind,
                        t,
                        outgoing: picture(&outgoing),
                        incoming: picture(&incoming),
                    }],
                )
                .expect("draw");
            renderer.read(&canvas).expect("read")
        };
        let apart = |a: &[u8], b: &[u8]| {
            let total: u64 = a
                .iter()
                .zip(b)
                .map(|(a, b)| u64::from(a.abs_diff(*b)))
                .sum();
            total as f64 / a.len() as f64
        };
        let recipes: Vec<TransitionKind> = TransitionKind::every()
            .into_iter()
            .filter(|kind| kind.recipe().is_some())
            .collect();
        assert!(!recipes.is_empty());
        for kind in recipes {
            let key = kind.key();
            let start = draw(kind, 0.0);
            assert!(
                apart(&start, &outgoing) < 3.0,
                "{key} starts at {}",
                apart(&start, &outgoing)
            );
            let end = draw(kind, 1.0);
            assert!(
                apart(&end, &incoming) < 3.0,
                "{key} ends at {}",
                apart(&end, &incoming)
            );
            let middle = draw(kind, 0.5);
            assert!(
                apart(&middle, &outgoing) > 2.0 && apart(&middle, &incoming) > 2.0,
                "{key} does nothing in the middle"
            );
        }
    }

    #[test]
    fn photo_slide_swaps_the_pictures_as_white_prints_over_a_dim_copy() {
        let Some(renderer) = Renderer::for_test().expect("a GPU is required") else {
            eprintln!("skipped: no GPU");
            return;
        };
        let kind = TransitionKind::every()
            .into_iter()
            .find(|kind| kind.key() == "photo-slide")
            .expect("photo slide ships");
        let (w, h) = (96u32, 64u32);
        let red: Vec<u8> = [240, 20, 20, 255].repeat((w * h) as usize);
        let blue: Vec<u8> = [20, 20, 240, 255].repeat((w * h) as usize);
        let canvas = renderer.canvas(w, h).expect("canvas");
        let draw = |t| {
            let picture = |bytes| Rgba {
                width: w,
                height: h,
                bytes,
            };
            renderer
                .draw(
                    &canvas,
                    &[Layer::Transition {
                        kind,
                        t,
                        outgoing: picture(&red),
                        incoming: picture(&blue),
                    }],
                )
                .expect("draw");
            renderer.read(&canvas).expect("read")
        };
        let at = |frame: &[u8], x: u32, y: u32| {
            let i = 4 * (y * w + x) as usize;
            [frame[i], frame[i + 1], frame[i + 2]]
        };
        let white = |c: [u8; 3]| c.iter().all(|v| *v > 220);
        // Early on, the outgoing picture has shrunk onto its print and is
        // sliding right: the left edge shows the dim copy, the middle the
        // picture, the paper between.
        let early = draw(0.15);
        let left = at(&early, 1, 32);
        assert!((150..220).contains(&left[0]) && left[2] < 40, "{left:?}");
        assert!(at(&early, 60, 32)[0] > 220);
        assert!(
            (0..w).any(|x| white(at(&early, x, 32))),
            "no paper across the middle"
        );
        // Then the incoming print settles in the middle, over the dim copy
        // of the outgoing picture, its paper round it.
        let settled = draw(0.6);
        assert!(at(&settled, 48, 30)[2] > 220);
        let corner = at(&settled, 1, 1);
        assert!(
            (150..220).contains(&corner[0]) && corner[2] < 40,
            "{corner:?}"
        );
        assert!((0..w / 2).any(|x| white(at(&settled, x, 32))));
        // Its bottom paper is deeper than its top.
        let paper_rows = |from: u32, step: i32| {
            let mut y = from;
            let mut rows = 0;
            while white(at(&settled, 48, y)) {
                rows += 1;
                y = y.wrapping_add_signed(step);
            }
            rows
        };
        let top = (0..32)
            .rev()
            .find(|y| white(at(&settled, 48, *y)))
            .expect("top paper");
        let bottom = (32..h)
            .find(|y| white(at(&settled, 48, *y)))
            .expect("bottom paper");
        assert!(
            paper_rows(bottom, 1) > paper_rows(top, -1),
            "the foot is deeper"
        );
    }

    #[test]
    fn every_transition_on_the_gpu_matches_the_cpu() {
        let Some(renderer) = Renderer::for_test().expect("a GPU is required") else {
            eprintln!("skipped: no GPU");
            return;
        };
        let (outgoing, incoming) = (picture(1), picture(7));
        let canvas = renderer.canvas(W, H).expect("canvas");
        for kind in TransitionKind::ALL {
            for t in [0.0, 0.17, 0.43, 0.61, 0.87, 1.0] {
                renderer
                    .draw(
                        &canvas,
                        &[Layer::Transition {
                            kind,
                            t,
                            outgoing: Rgba {
                                width: W,
                                height: H,
                                bytes: &outgoing,
                            },
                            incoming: Rgba {
                                width: W,
                                height: H,
                                bytes: &incoming,
                            },
                        }],
                    )
                    .expect("draw");
                let got = renderer.read(&canvas).expect("read");
                let expected = mix_transition(kind, &outgoing, &incoming, W, H, t).expect("mix");
                // A move or turn can land a sample right on a pixel edge,
                // where the GPU's division rounds to the neighbour; a few
                // such pixels are allowed for those. Anything else is the
                // same within one step.
                let moving = !matches!(
                    kind,
                    TransitionKind::Dissolve
                        | TransitionKind::DipToBlack
                        | TransitionKind::DipToWhite
                        | TransitionKind::Blur
                );
                let allowed = if moving { (W * H / 20) as usize } else { 0 };
                let off: Vec<usize> = got
                    .chunks(4)
                    .zip(expected.chunks(4))
                    .enumerate()
                    .filter(|(_, (a, b))| a.iter().zip(b.iter()).any(|(a, b)| a.abs_diff(*b) > 1))
                    .map(|(index, _)| index)
                    .collect();
                assert!(
                    off.len() <= allowed,
                    "{kind:?} at {t}: {} pixels differ, first {:?}",
                    off.len(),
                    off.first().map(|index| (
                        index % W as usize,
                        index / W as usize,
                        &got[index * 4..index * 4 + 4],
                        &expected[index * 4..index * 4 + 4]
                    ))
                );
            }
        }
        // The pictures must be the canvas's size.
        let small = [0_u8; 4];
        let error = renderer.draw(
            &canvas,
            &[Layer::Transition {
                kind: TransitionKind::Dissolve,
                t: 0.5,
                outgoing: Rgba {
                    width: 1,
                    height: 1,
                    bytes: &small,
                },
                incoming: Rgba {
                    width: 1,
                    height: 1,
                    bytes: &small,
                },
            }],
        );
        assert!(error.is_err());
    }
}
