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
