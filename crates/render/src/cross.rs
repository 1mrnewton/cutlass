//! The cross pass of a recipe transition: with each picture already run
//! through its own steps, pick pixel by pixel which shows, by a wipe, a
//! shape opening from a point, a fade, brightness, or one laid over the
//! other. An `edge` lays a coloured band along the swapping line.

use cutlass_effects::Crossing;

use crate::CANVAS_FORMAT;

const SHADER: &str = r"
struct Cross {
    // progress, kind, angle, soft
    a: vec4<f32>,
    // count, x, y, edge
    b: vec4<f32>,
    // red, green, blue
    c: vec4<f32>,
    size: vec4<f32>,
};

@group(0) @binding(0) var outgoing_picture: texture_2d<f32>;
@group(0) @binding(1) var incoming_picture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> cx: Cross;

@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(index & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn pcg(value: u32) -> u32 {
    let state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand(a: u32, b: u32, seed: u32) -> f32 {
    return f32(pcg(a ^ pcg(b ^ pcg(seed)))) / 4294967295.0;
}

// How far out a point `q` (y up, shorter side 1) is for a shape of size 1.
fn heart_inside(q: vec2<f32>, k: f32) -> bool {
    let x = q.x / k * 1.2;
    let y = q.y / k * 1.2 + 0.15;
    let ring = x * x + y * y - 1.0;
    return ring * ring * ring - x * x * y * y * y <= 0.0;
}

fn reach(kind: u32, q: vec2<f32>) -> f32 {
    if (kind == 5u) {
        return abs(q.x) + abs(q.y);
    }
    if (kind == 12u) {
        let angle = atan2(q.y, q.x) - 1.5707964;
        let sector = 6.2831855 / 5.0;
        let wrapped = angle - sector * floor(angle / sector);
        let into = abs(wrapped / sector - 0.5) * 2.0;
        return length(q) / (0.45 + 0.55 * into);
    }
    if (kind == 13u) {
        return max(abs(q.y), abs(q.x) * 0.8660254 + abs(q.y) * 0.5);
    }
    if (kind == 14u) {
        var lo = 0.0;
        var hi = 8.0;
        for (var i = 0; i < 18; i = i + 1) {
            let mid = (lo + hi) * 0.5;
            if (heart_inside(q, mid)) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        return hi;
    }
    return length(q);
}

fn luma(rgb: vec3<f32>) -> f32 {
    return dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@fragment
fn fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let texel = vec2<i32>(position.xy);
    let a = textureLoad(outgoing_picture, texel, 0);
    let b = textureLoad(incoming_picture, texel, 0);
    let kind = u32(cx.a.y + 0.5);
    if (kind == 10u) {
        return b + a * (1.0 - b.a);
    }
    if (kind == 11u) {
        return a + b * (1.0 - a.a);
    }
    let size = cx.size.xy;
    let short = min(size.x, size.y);
    let uv = position.xy / size;
    let turn = radians(cx.a.z);
    let along = vec2<f32>(cos(turn), -sin(turn));
    let spread = (abs(along.x) + abs(along.y)) * 0.5;
    let count = max(floor(cx.b.x + 0.5), 1.0);
    let progress = cx.a.x;
    let soft = cx.a.w;
    // The point shapes open from, and the frame's corners from it.
    let point = vec2<f32>(cx.b.y, cx.b.z);
    let p = (uv - vec2<f32>(0.5) - point) * size / short;
    let half = size / short * 0.5;
    var run = 0.0;
    switch kind {
        case 0u: { run = dot(uv - vec2<f32>(0.5), along) / spread * 0.5 + 0.5; }
        case 2u: { run = fract((dot(uv - vec2<f32>(0.5), along) / spread * 0.5 + 0.5) * count); }
        case 3u: { run = abs(dot(uv - vec2<f32>(0.5), along)) / spread; }
        case 4u: {
            let round_ = atan2(p.x, -p.y) / 6.2831853 - cx.a.z / 360.0;
            run = fract(round_ + 1.0);
        }
        case 6u: {
            let at = position.xy / (short / count);
            let cell = floor(at);
            let odd = f32((u32(cell.x) + u32(cell.y)) & 1u);
            run = odd * 0.5 + fract(at.x) * 0.5;
        }
        case 7u: {
            let grain = vec2<u32>(position.xy / 3.0);
            run = rand(grain.x, grain.y, 7u);
        }
        case 8u: { run = 0.5; }
        case 9u: {
            var rgb = vec3<f32>(0.0);
            if (a.a > 0.0) {
                rgb = a.rgb / a.a;
            }
            run = luma(rgb);
        }
        case 15u: {
            let cell = vec2<u32>(position.xy / (short / count));
            run = rand(cell.x, cell.y, 15u);
        }
        case 16u: {
            let across = vec2<f32>(-along.y, along.x);
            let stripe = u32(floor((dot(uv - vec2<f32>(0.5), across) / spread * 0.5 + 0.5) * count + 64.0));
            run = dot(uv - vec2<f32>(0.5), along) / spread * 0.5 + 0.5;
            if ((stripe & 1u) == 1u) {
                run = 1.0 - run;
            }
        }
        default: {
            // A shape out from the point, measured so it just covers the
            // frame's farthest corner at the end.
            let flip = vec2<f32>(1.0, -1.0);
            let far = max(
                max(reach(kind, (half - point * size / short) * flip), reach(kind, (vec2<f32>(-half.x, half.y) - point * size / short) * flip)),
                max(reach(kind, (-half - point * size / short) * flip), reach(kind, (vec2<f32>(half.x, -half.y) - point * size / short) * flip))
            );
            run = reach(kind, p * flip) / max(far, 0.0001);
        }
    }
    var shown = progress;
    if (kind != 8u) {
        shown = 1.0 - smoothstep(progress * (1.0 + soft) - soft, progress * (1.0 + soft), run);
    }
    var out = mix(a, b, shown);
    let edge = cx.b.w;
    if (edge > 0.0 && kind != 8u && progress > 0.0 && progress < 1.0) {
        let line = progress * (1.0 + soft) - soft * 0.5;
        let band = 1.0 - clamp(abs(run - line) / edge, 0.0, 1.0);
        let alpha = max(out.a, band);
        out = vec4<f32>(mix(out.rgb, cx.c.rgb * alpha, band), alpha);
    }
    return out;
}
";

/// Bytes in the uniform block.
const PARAMS: u64 = 64;

pub(crate) struct Cross {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
}

impl Cross {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cutlass-render cross"),
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
            label: Some("cutlass-render cross"),
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
            label: Some("cutlass-render cross"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cutlass-render cross"),
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

    /// Write `crossing`'s swap of `outgoing` and `incoming`, both `size`,
    /// into `out`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        outgoing: &wgpu::TextureView,
        incoming: &wgpu::TextureView,
        out: &wgpu::TextureView,
        crossing: &Crossing,
        size: (u32, u32),
    ) {
        let mut values = [0.0f32; 16];
        // In the order of `CROSS`: four to a vec4, the colour third.
        values[..11].copy_from_slice(&crossing.cross[..11]);
        #[allow(clippy::cast_precision_loss)]
        {
            values[12] = size.0 as f32;
            values[13] = size.1 as f32;
        }
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cutlass-render cross params"),
            size: PARAMS,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, &bytes);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cutlass-render cross"),
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
            label: Some("cutlass-render cross"),
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
