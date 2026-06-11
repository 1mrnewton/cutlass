use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::error::CompositorError;
use crate::gpu::GpuContext;
use crate::image::RgbaImage;
use crate::layer::{CompositeLayer, CompositorConfig};
use crate::yuv::{Yuv420pImage, Yuv420pLayer};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const R8: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SolidUniforms {
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RgbaToYuvParams {
    width: u32,
    height: u32,
    y_stride: u32,
    uv_stride: u32,
}

/// WGPU alpha-over compositor. Layers are composited bottom-to-top with
/// standard src-over blending onto a single offscreen target.
pub struct Compositor {
    solid_pipeline: wgpu::RenderPipeline,
    blit_pipeline: wgpu::RenderPipeline,
    yuv_pipeline: wgpu::RenderPipeline,
    rgba_to_yuv_pipeline: wgpu::ComputePipeline,
    solid_bind_layout: wgpu::BindGroupLayout,
    blit_bind_layout: wgpu::BindGroupLayout,
    yuv_bind_layout: wgpu::BindGroupLayout,
    rgba_to_yuv_bind_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    solid_uniform: wgpu::Buffer,
    rgba_to_yuv_params: wgpu::Buffer,
    /// Reused each composite when canvas size matches.
    target: Option<CachedTarget>,
}

struct CachedTarget {
    width: u32,
    height: u32,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    readback: wgpu::Buffer,
    readback_stride: u32,
}

impl Compositor {
    pub fn new(gpu: &GpuContext) -> Result<Self, CompositorError> {
        let blend = src_over_blend();

        let solid_shader = shader(gpu, "solid", include_str!("../shaders/solid.wgsl"));
        let blit_shader = shader(gpu, "blit", include_str!("../shaders/blit.wgsl"));
        let yuv_shader = shader(gpu, "yuv_blit", include_str!("../shaders/yuv_blit.wgsl"));
        let rgba_to_yuv_shader = shader(
            gpu,
            "rgba_to_yuv",
            include_str!("../shaders/rgba_to_yuv.wgsl"),
        );

        let solid_bind_layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("solid_bind"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let blit_bind_layout = texture_sampler_bind_layout(gpu, "blit_bind");

        let yuv_bind_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("yuv_bind"),
                    entries: &[
                        texture_binding(0, wgpu::ShaderStages::FRAGMENT),
                        texture_binding(1, wgpu::ShaderStages::FRAGMENT),
                        texture_binding(2, wgpu::ShaderStages::FRAGMENT),
                        sampler_binding(3, wgpu::ShaderStages::FRAGMENT),
                    ],
                });

        let rgba_to_yuv_bind_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("rgba_to_yuv_bind"),
                    entries: &[
                        texture_binding(0, wgpu::ShaderStages::COMPUTE),
                        storage_binding(1, true),
                        storage_binding(2, true),
                        storage_binding(3, true),
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let solid_pipeline = render_pipeline(
            gpu,
            "solid_pipeline",
            &solid_shader,
            &pipeline_layout(gpu, "solid_layout", &solid_bind_layout),
            blend,
        );
        let blit_pipeline = render_pipeline(
            gpu,
            "blit_pipeline",
            &blit_shader,
            &pipeline_layout(gpu, "blit_layout", &blit_bind_layout),
            blend,
        );
        let yuv_pipeline = render_pipeline(
            gpu,
            "yuv_pipeline",
            &yuv_shader,
            &pipeline_layout(gpu, "yuv_layout", &yuv_bind_layout),
            blend,
        );

        let rgba_to_yuv_pipeline =
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("rgba_to_yuv_pipeline"),
                    layout: Some(
                        &gpu.device
                            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                                label: Some("rgba_to_yuv_layout"),
                                bind_group_layouts: &[&rgba_to_yuv_bind_layout],
                                push_constant_ranges: &[],
                            }),
                    ),
                    module: &rgba_to_yuv_shader,
                    entry_point: Some("cs"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("layer_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let solid_uniform = uniform_buffer(gpu, "solid_uniform", &SolidUniforms {
            color: [0.0; 4],
        });
        let rgba_to_yuv_params = uniform_buffer(
            gpu,
            "rgba_to_yuv_params",
            &RgbaToYuvParams {
                width: 0,
                height: 0,
                y_stride: 0,
                uv_stride: 0,
            },
        );

        Ok(Self {
            solid_pipeline,
            blit_pipeline,
            yuv_pipeline,
            rgba_to_yuv_pipeline,
            solid_bind_layout,
            blit_bind_layout,
            yuv_bind_layout,
            rgba_to_yuv_bind_layout,
            sampler,
            solid_uniform,
            rgba_to_yuv_params,
            target: None,
        })
    }

    /// Composite layers bottom-to-top and read back RGBA8 bytes.
    pub fn composite(
        &mut self,
        gpu: &GpuContext,
        config: &CompositorConfig,
        layers: &[CompositeLayer],
    ) -> Result<RgbaImage, CompositorError> {
        validate_layers(config, layers)?;
        self.ensure_target(gpu, config)?;
        self.render_layers(gpu, config, layers)?;
        self.readback_rgba(gpu, config)
    }

    /// Composite layers, convert the canvas to YUV420P on GPU, and read back.
    pub fn composite_yuv420p(
        &mut self,
        gpu: &GpuContext,
        config: &CompositorConfig,
        layers: &[CompositeLayer],
    ) -> Result<Yuv420pImage, CompositorError> {
        validate_layers(config, layers)?;
        self.ensure_target(gpu, config)?;

        let width = config.width;
        let height = config.height;
        let y_stride = width;
        let uv_stride = width / 2;
        let y_count = (y_stride * height) as u64;
        let uv_count = (uv_stride * (height / 2)) as u64;
        let y_buf = storage_buffer(gpu, "y_out", y_count * 4);
        let u_buf = storage_buffer(gpu, "u_out", uv_count * 4);
        let v_buf = storage_buffer(gpu, "v_out", uv_count * 4);

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("composite_yuv_encoder"),
            });
        self.render_layers_into(&mut encoder, gpu, config, layers)?;
        self.encode_yuv_into(
            &mut encoder,
            gpu,
            config,
            &y_buf,
            &u_buf,
            &v_buf,
        )?;
        gpu.queue.submit(Some(encoder.finish()));

        let y = read_storage_u8(gpu, &y_buf, y_count as usize)?;
        let u = read_storage_u8(gpu, &u_buf, uv_count as usize)?;
        let v = read_storage_u8(gpu, &v_buf, uv_count as usize)?;
        Ok(Yuv420pImage {
            width,
            height,
            y,
            u,
            v,
        })
    }

    fn render_layers(
        &mut self,
        gpu: &GpuContext,
        config: &CompositorConfig,
        layers: &[CompositeLayer],
    ) -> Result<(), CompositorError> {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("composite_encoder"),
            });
        self.render_layers_into(&mut encoder, gpu, config, layers)?;
        gpu.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn render_layers_into(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        gpu: &GpuContext,
        config: &CompositorConfig,
        layers: &[CompositeLayer],
    ) -> Result<(), CompositorError> {
        let target = self.target.as_ref().expect("target initialized");

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("composite_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            for layer in layers {
                match layer {
                    CompositeLayer::Solid { rgba } => {
                        let uniforms = SolidUniforms {
                            color: [
                                rgba[0] as f32 / 255.0,
                                rgba[1] as f32 / 255.0,
                                rgba[2] as f32 / 255.0,
                                rgba[3] as f32 / 255.0,
                            ],
                        };
                        gpu.queue.write_buffer(
                            &self.solid_uniform,
                            0,
                            bytemuck::bytes_of(&uniforms),
                        );
                        let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("solid_bind"),
                            layout: &self.solid_bind_layout,
                            entries: &[wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.solid_uniform.as_entire_binding(),
                            }],
                        });
                        pass.set_pipeline(&self.solid_pipeline);
                        pass.set_bind_group(0, &bind, &[]);
                        pass.draw(0..3, 0..1);
                    }
                    CompositeLayer::Rgba { bytes } => {
                        let texture = upload_rgba_texture(gpu, bytes, config.width, config.height);
                        let view = texture.create_view(&Default::default());
                        let bind = blit_bind(gpu, &self.blit_bind_layout, &view, &self.sampler);
                        pass.set_pipeline(&self.blit_pipeline);
                        pass.set_bind_group(0, &bind, &[]);
                        pass.draw(0..3, 0..1);
                    }
                    CompositeLayer::Yuv420p(layer) => {
                        let y_tex = upload_r8_texture(
                            gpu,
                            "y_plane",
                            &layer.tight_y(),
                            layer.width,
                            layer.height,
                        );
                        let uv_w = layer.width / 2;
                        let uv_h = layer.height / 2;
                        let u_tex =
                            upload_r8_texture(gpu, "u_plane", &layer.tight_u(), uv_w, uv_h);
                        let v_tex =
                            upload_r8_texture(gpu, "v_plane", &layer.tight_v(), uv_w, uv_h);
                        let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("yuv_bind"),
                            layout: &self.yuv_bind_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(
                                        &y_tex.create_view(&Default::default()),
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(
                                        &u_tex.create_view(&Default::default()),
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::TextureView(
                                        &v_tex.create_view(&Default::default()),
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                                },
                            ],
                        });
                        pass.set_pipeline(&self.yuv_pipeline);
                        pass.set_bind_group(0, &bind, &[]);
                        pass.draw(0..3, 0..1);
                    }
                }
            }
        }

        Ok(())
    }

    fn encode_yuv_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gpu: &GpuContext,
        config: &CompositorConfig,
        y_buf: &wgpu::Buffer,
        u_buf: &wgpu::Buffer,
        v_buf: &wgpu::Buffer,
    ) -> Result<(), CompositorError> {
        let target = self.target.as_ref().expect("target initialized");
        let width = config.width;
        let height = config.height;
        let params = RgbaToYuvParams {
            width,
            height,
            y_stride: width,
            uv_stride: width / 2,
        };
        gpu.queue
            .write_buffer(&self.rgba_to_yuv_params, 0, bytemuck::bytes_of(&params));

        let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rgba_to_yuv_bind"),
            layout: &self.rgba_to_yuv_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&target.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: y_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: v_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.rgba_to_yuv_params.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rgba_to_yuv_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.rgba_to_yuv_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        Ok(())
    }

    fn readback_rgba(
        &mut self,
        gpu: &GpuContext,
        config: &CompositorConfig,
    ) -> Result<RgbaImage, CompositorError> {
        let target = self.target.as_ref().expect("target initialized");
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rgba_readback_encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &target.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(target.readback_stride),
                    rows_per_image: Some(config.height),
                },
            },
            wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit(Some(encoder.finish()));

        let row_bytes = usize::try_from(config.width * 4).expect("width");
        let padded_row = usize::try_from(target.readback_stride).expect("stride");
        let height = usize::try_from(config.height).expect("height");
        let mut tight = vec![0u8; row_bytes * height];
        map_read_buffer(gpu, &target.readback, |mapped| {
            for row in 0..height {
                let src = row * padded_row;
                let dst = row * row_bytes;
                tight[dst..dst + row_bytes].copy_from_slice(&mapped[src..src + row_bytes]);
            }
        })?;
        RgbaImage::new(config.width, config.height, tight)
    }

    fn ensure_target(
        &mut self,
        gpu: &GpuContext,
        config: &CompositorConfig,
    ) -> Result<(), CompositorError> {
        let needs_new = self
            .target
            .as_ref()
            .is_none_or(|t| t.width != config.width || t.height != config.height);
        if needs_new {
            let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("composite_target"),
                size: wgpu::Extent3d {
                    width: config.width,
                    height: config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            let readback_stride = align_row_bytes(config.width * 4);
            let readback_size = u64::from(readback_stride) * u64::from(config.height);
            let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: readback_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            self.target = Some(CachedTarget {
                width: config.width,
                height: config.height,
                texture,
                view,
                readback,
                readback_stride,
            });
        }
        Ok(())
    }
}

fn validate_layers(
    config: &CompositorConfig,
    layers: &[CompositeLayer],
) -> Result<(), CompositorError> {
    validate_config(config)?;
    let expected = layer_byte_len(config);
    for layer in layers {
        match layer {
            CompositeLayer::Rgba { bytes } if bytes.len() != expected => {
                return Err(CompositorError::LayerSizeMismatch {
                    got: bytes.len(),
                    expected,
                });
            }
            CompositeLayer::Yuv420p(yuv) => validate_yuv_layer(yuv)?,
            _ => {}
        }
    }
    Ok(())
}

fn validate_yuv_layer(layer: &Yuv420pLayer) -> Result<(), CompositorError> {
    if layer.width == 0
        || layer.height == 0
        || !layer.width.is_multiple_of(2)
        || !layer.height.is_multiple_of(2)
    {
        return Err(CompositorError::InvalidYuvDimensions {
            width: layer.width,
            height: layer.height,
        });
    }
    Ok(())
}

fn validate_config(config: &CompositorConfig) -> Result<(), CompositorError> {
    if config.width == 0 || config.height == 0 {
        return Err(CompositorError::InvalidDimensions {
            width: config.width,
            height: config.height,
        });
    }
    Ok(())
}

fn layer_byte_len(config: &CompositorConfig) -> usize {
    usize::try_from(config.width)
        .unwrap_or(0)
        .saturating_mul(usize::try_from(config.height).unwrap_or(0))
        .saturating_mul(4)
}

fn align_row_bytes(bytes_per_row: u32) -> u32 {
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    bytes_per_row.div_ceil(align) * align
}

fn src_over_blend() -> wgpu::BlendState {
    wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
    }
}

fn shader(gpu: &GpuContext, label: &str, source: &str) -> wgpu::ShaderModule {
    gpu.device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        })
}

fn pipeline_layout(
    gpu: &GpuContext,
    label: &str,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::PipelineLayout {
    gpu.device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[layout],
            push_constant_ranges: &[],
        })
}

fn render_pipeline(
    gpu: &GpuContext,
    label: &str,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    blend: wgpu::BlendState,
) -> wgpu::RenderPipeline {
    gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs"),
            targets: &[Some(wgpu::ColorTargetState {
                format: FORMAT,
                blend: Some(blend),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

fn texture_binding(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_binding(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn storage_binding(binding: u32, read_write: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: if read_write {
                wgpu::BufferBindingType::Storage { read_only: false }
            } else {
                wgpu::BufferBindingType::Storage { read_only: true }
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_sampler_bind_layout(gpu: &GpuContext, label: &str) -> wgpu::BindGroupLayout {
    gpu.device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &[
                texture_binding(0, wgpu::ShaderStages::FRAGMENT),
                sampler_binding(1, wgpu::ShaderStages::FRAGMENT),
            ],
        })
}

fn uniform_buffer<T: Pod>(gpu: &GpuContext, label: &str, value: &T) -> wgpu::Buffer {
    gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(value),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

fn storage_buffer(gpu: &GpuContext, label: &str, size: u64) -> wgpu::Buffer {
    gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn upload_rgba_texture(
    gpu: &GpuContext,
    bytes: &[u8],
    width: u32,
    height: u32,
) -> wgpu::Texture {
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("layer_upload"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    texture
}

fn upload_r8_texture(
    gpu: &GpuContext,
    label: &str,
    bytes: &[u8],
    width: u32,
    height: u32,
) -> wgpu::Texture {
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: R8,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
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
    texture
}

fn blit_bind(
    gpu: &GpuContext,
    layout: &wgpu::BindGroupLayout,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("blit_bind"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

fn map_read_buffer<F>(gpu: &GpuContext, buffer: &wgpu::Buffer, f: F) -> Result<(), CompositorError>
where
    F: FnOnce(&[u8]),
{
    let slice = buffer.slice(..);
    let (tx, rx) = mpsc::sync_channel(1);
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = gpu.device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|_| CompositorError::MapFailed)?
        .map_err(|_| CompositorError::MapFailed)?;
    let mapped = slice.get_mapped_range();
    f(&mapped);
    drop(mapped);
    buffer.unmap();
    Ok(())
}

fn read_storage_u8(
    gpu: &GpuContext,
    buffer: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<u8>, CompositorError> {
    let byte_len = count * 4;
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("storage_readback"),
        size: byte_len as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("storage_copy"),
        });
    encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, byte_len as u64);
    gpu.queue.submit(Some(encoder.finish()));

    let mut out = vec![0u8; count];
    map_read_buffer(gpu, &readback, |mapped| {
        for (i, chunk) in mapped.chunks_exact(4).take(count).enumerate() {
            out[i] = chunk[0];
        }
    })?;
    Ok(out)
}
