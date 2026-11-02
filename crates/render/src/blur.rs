//! A canvas-sized box blur, as the CPU backdrop draws it.
//!
//! Rows are blurred, then columns. Each output is the window of `2r + 1`
//! samples around it, an edge sample repeated past the edge, summed and
//! divided with the remainder dropped, on all four channels. One invocation
//! walks one row or column with a running sum, so the cost does not grow
//! with the radius. Sums of whole byte values stay exact in `f32`.

use crate::CANVAS_FORMAT;

const SHADER: &str = r"
struct Blur {
    // Radius, 1 for rows and 0 for columns, width, height.
    params: vec4<u32>,
};

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> blur: Blur;

fn at(line: i32, index: i32, rows: bool) -> vec2<i32> {
    if (rows) {
        return vec2<i32>(index, line);
    }
    return vec2<i32>(line, index);
}

fn sample(line: i32, index: i32, length: i32, rows: bool) -> vec4<f32> {
    let clamped = clamp(index, 0, length - 1);
    return round(textureLoad(source, at(line, clamped, rows), 0) * 255.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let radius = i32(blur.params.x);
    let rows = blur.params.y == 1u;
    let width = i32(blur.params.z);
    let height = i32(blur.params.w);
    let line = i32(id.x);
    var lines = width;
    var length = height;
    if (rows) {
        lines = height;
        length = width;
    }
    if (line >= lines) {
        return;
    }
    let window = f32(2 * radius + 1);
    var sum = vec4<f32>(0.0);
    for (var k = -radius; k <= radius; k = k + 1) {
        sum = sum + sample(line, k, length, rows);
    }
    for (var index = 0; index < length; index = index + 1) {
        textureStore(output, at(line, index, rows), floor(sum / window) / 255.0);
        sum = sum + sample(line, index + radius + 1, length, rows)
            - sample(line, index - radius, length, rows);
    }
}
";

pub(crate) struct Blurs {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

impl Blurs {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cutlass-render blur"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cutlass-render blur"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: CANVAS_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cutlass-render blur"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cutlass-render blur"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self { pipeline, layout }
    }

    /// Blur `first` in place by `radius`, rows then columns, through
    /// `second`. Both are canvas-sized storage textures.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        first: &wgpu::TextureView,
        second: &wgpu::TextureView,
        size: (u32, u32),
        radius: u32,
    ) {
        if radius == 0 {
            return;
        }
        let pass = |encoder: &mut wgpu::CommandEncoder,
                    source: &wgpu::TextureView,
                    target: &wgpu::TextureView,
                    rows: bool| {
            let mut bytes = Vec::with_capacity(16);
            for value in [radius, u32::from(rows), size.0, size.1] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cutlass-render blur params"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buffer, 0, &bytes);
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cutlass-render blur"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(source),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(target),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffer.as_entire_binding(),
                    },
                ],
            });
            let lines = if rows { size.1 } else { size.0 };
            let mut compute = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cutlass-render blur"),
                timestamp_writes: None,
            });
            compute.set_pipeline(&self.pipeline);
            compute.set_bind_group(0, &group, &[]);
            compute.dispatch_workgroups(lines.div_ceil(64), 1, 1);
        };
        pass(encoder, first, second, true);
        pass(encoder, second, first, false);
    }
}

/// A canvas-sized texture the layer pipeline draws into and the blur reads
/// and writes.
pub(crate) fn scratch(device: &wgpu::Device, size: (u32, u32)) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("cutlass-render blur scratch"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: CANVAS_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}
