//! The blend pass: a clip drawn alone, mixed with what is drawn below it.
//!
//! Both pictures are canvas-sized and premultiplied. Each pixel is read
//! with `textureLoad`, not sampled, and mixed by `cutlass_effects`'s WGSL
//! `blend`, the same arithmetic the CPU path runs.

use cutlass_effects::{BLEND_WGSL, BlendMode};

use crate::CANVAS_FORMAT;

const SHADER: &str = r"
@group(0) @binding(0) var top_picture: texture_2d<f32>;
@group(0) @binding(1) var below_picture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> mode: vec4<u32>;

@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(index & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let texel = vec2<i32>(position.xy);
    let source = textureLoad(top_picture, texel, 0);
    let backdrop = textureLoad(below_picture, texel, 0);
    return blend(mode.x, source, backdrop);
}
";

/// Bytes in the uniform block: the mode, padded to sixteen.
const PARAMS: u64 = 16;

pub(crate) struct Mix {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
}

impl Mix {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let source = format!("{SHADER}\n{BLEND_WGSL}");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cutlass-render mix"),
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
            label: Some("cutlass-render mix"),
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
            label: Some("cutlass-render mix"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cutlass-render mix"),
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

    /// Write `top` mixed over `below` by `mode` into `out`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        top: &wgpu::TextureView,
        below: &wgpu::TextureView,
        out: &wgpu::TextureView,
        mode: BlendMode,
    ) {
        let mut bytes = [0u8; 16];
        bytes[..4].copy_from_slice(&mode.code().to_le_bytes());
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cutlass-render mix mode"),
            size: PARAMS,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, &bytes);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cutlass-render mix"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(top),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(below),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("cutlass-render mix"),
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
    use cutlass_effects::{BlendMode, blend_pixel};
    use cutlass_gpu::Placement;

    use crate::{Layer, Renderer, Rgba};

    /// Pairs of premultiplied pixels: some opaque, some part clear.
    const TOP: [[u8; 4]; 6] = [
        [255, 255, 255, 255],
        [0, 0, 0, 255],
        [200, 40, 90, 255],
        [64, 10, 0, 128],
        [0, 0, 0, 0],
        [30, 120, 220, 255],
    ];
    const BELOW: [[u8; 4]; 6] = [
        [128, 128, 128, 255],
        [128, 128, 128, 255],
        [20, 160, 240, 255],
        [100, 60, 0, 200],
        [90, 90, 90, 255],
        [250, 10, 60, 255],
    ];

    #[test]
    fn every_mode_on_the_gpu_matches_the_cpu() {
        let Some(renderer) = Renderer::for_test().expect("a GPU is required") else {
            eprintln!("skipped: no GPU");
            return;
        };
        let width = u32::try_from(TOP.len()).expect("width");
        let top: Vec<u8> = TOP.iter().flatten().copied().collect();
        let below: Vec<u8> = BELOW.iter().flatten().copied().collect();
        let canvas = renderer.canvas(width, 1).expect("canvas");
        for mode in BlendMode::ALL {
            renderer
                .draw(
                    &canvas,
                    &[
                        Layer::Rgba(
                            Rgba {
                                width,
                                height: 1,
                                bytes: &below,
                            },
                            Placement::canvas(width, 1),
                        ),
                        Layer::Blend {
                            mode,
                            layers: vec![Layer::Rgba(
                                Rgba {
                                    width,
                                    height: 1,
                                    bytes: &top,
                                },
                                Placement::canvas(width, 1),
                            )],
                        },
                    ],
                )
                .expect("draw");
            let got = renderer.read(&canvas).expect("read");
            for (index, (pixel, (source, backdrop))) in
                got.chunks(4).zip(TOP.iter().zip(BELOW.iter())).enumerate()
            {
                let expected = blend_pixel(mode, *source, *backdrop);
                for (a, b) in pixel.iter().zip(expected) {
                    assert!(
                        a.abs_diff(b) <= 1,
                        "{mode:?} pixel {index}: gpu {pixel:?} cpu {expected:?}"
                    );
                }
            }
        }
    }
}
