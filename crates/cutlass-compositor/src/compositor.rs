use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::error::CompositorError;
use crate::gpu::GpuContext;
use crate::image::RgbaImage;
use crate::layer::{CompositeLayer, CompositorConfig};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SolidUniforms {
    color: [f32; 4],
}

/// WGPU alpha-over compositor. Layers are composited bottom-to-top with
/// standard src-over blending onto a single offscreen target.
pub struct Compositor {
    solid_pipeline: wgpu::RenderPipeline,
    blit_pipeline: wgpu::RenderPipeline,
    solid_bind_layout: wgpu::BindGroupLayout,
    blit_bind_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    solid_uniform: wgpu::Buffer,
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
        let solid_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("solid"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/solid.wgsl").into()),
            });
        let blit_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("blit"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/blit.wgsl").into()),
            });

        let solid_bind_layout =
            gpu.device
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

        let blit_bind_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("blit_bind"),
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
                    ],
                });

        let blend = wgpu::BlendState {
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
        };

        let pipeline_layout = |label: &str, layout: &wgpu::BindGroupLayout| {
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &[layout],
                    push_constant_ranges: &[],
                })
        };

        let vertex_buffers: &[wgpu::VertexBufferLayout<'_>] = &[];

        let solid_pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("solid_pipeline"),
            layout: Some(&pipeline_layout("solid_layout", &solid_bind_layout)),
            vertex: wgpu::VertexState {
                module: &solid_shader,
                entry_point: Some("vs"),
                buffers: vertex_buffers,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &solid_shader,
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
        });

        let blit_pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit_pipeline"),
            layout: Some(&pipeline_layout("blit_layout", &blit_bind_layout)),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs"),
                buffers: vertex_buffers,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
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
        });

        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("layer_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let solid_uniform = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("solid_uniform"),
            contents: bytemuck::bytes_of(&SolidUniforms {
                color: [0.0, 0.0, 0.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Ok(Self {
            solid_pipeline,
            blit_pipeline,
            solid_bind_layout,
            blit_bind_layout,
            sampler,
            solid_uniform,
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
        validate_config(config)?;
        let expected = layer_byte_len(config);

        for layer in layers {
            if let CompositeLayer::Rgba { bytes } = layer
                && bytes.len() != expected
            {
                return Err(CompositorError::LayerSizeMismatch {
                    got: bytes.len(),
                    expected,
                });
            }
        }

        self.ensure_target(gpu, config)?;
        let target = self.target.as_ref().expect("target initialized");

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("composite_encoder"),
            });

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
                        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
                            label: Some("layer_upload"),
                            size: wgpu::Extent3d {
                                width: config.width,
                                height: config.height,
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
                                bytes_per_row: Some(config.width * 4),
                                rows_per_image: Some(config.height),
                            },
                            wgpu::Extent3d {
                                width: config.width,
                                height: config.height,
                                depth_or_array_layers: 1,
                            },
                        );
                        let view = texture.create_view(&Default::default());
                        let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("blit_bind"),
                            layout: &self.blit_bind_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(&view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                                },
                            ],
                        });
                        pass.set_pipeline(&self.blit_pipeline);
                        pass.set_bind_group(0, &bind, &[]);
                        pass.draw(0..3, 0..1);
                    }
                }
            }
        }

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

        let slice = target.readback.slice(..);
        let (tx, rx) = mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| CompositorError::MapFailed)?
            .map_err(|_| CompositorError::MapFailed)?;
        let mapped = slice.get_mapped_range();
        for row in 0..height {
            let src = row * padded_row;
            let dst = row * row_bytes;
            tight[dst..dst + row_bytes].copy_from_slice(&mapped[src..src + row_bytes]);
        }
        drop(mapped);
        target.readback.unmap();

        RgbaImage::new(config.width, config.height, tight)
    }

    fn ensure_target(
        &mut self,
        gpu: &GpuContext,
        config: &CompositorConfig,
    ) -> Result<(), CompositorError> {
        let needs_new = self.target.as_ref().is_none_or(|t| {
            t.width != config.width || t.height != config.height
        });
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
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
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
