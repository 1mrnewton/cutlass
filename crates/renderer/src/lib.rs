//! GPU quad preview for Slint: renders into an `Rgba8Unorm` texture compatible with [`slint::Image::try_from`].

use slint::wgpu_28::wgpu;
use std::sync::Mutex;

use slint::Image;

const QUAD_WGSL: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shader/quad.wgsl"));
const TEXTURE_PNG: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/texture.png"));

/// Holds pipeline + imported Slint image for the preview texture.
pub struct QuadPreviewRenderer {
    inner: Mutex<Inner>,
}

struct Inner {
    pipeline: Option<wgpu::RenderPipeline>,
    bind_group: Option<wgpu::BindGroup>,
    /// Keeps `texture.png` resident on the GPU for the bind group’s [`wgpu::TextureView`].
    source_texture: Option<wgpu::Texture>,
    image: Option<Image>,
    extent_px: (u32, u32),
}

impl QuadPreviewRenderer {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                pipeline: None,
                bind_group: None,
                source_texture: None,
                image: None,
                extent_px: (0, 0),
            }),
        }
    }

    /// Call from [`slint::RenderingState::RenderingSetup`] with Slint’s WGPU device and queue.
    pub fn setup(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cutlass-quad-texture-bind-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cutlass-quad-layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = create_pipeline(device, &pipeline_layout, wgpu::TextureFormat::Rgba8Unorm);

        let rgba = image::load_from_memory(TEXTURE_PNG)
            .expect("texture.png: invalid image")
            .to_rgba8();
        let (tw, th) = rgba.dimensions();

        let photo_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cutlass-quad-source-texture"),
            size: wgpu::Extent3d {
                width: tw,
                height: th,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            photo_tex.as_image_copy(),
            rgba.as_raw(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * tw),
                rows_per_image: Some(th),
            },
            wgpu::Extent3d {
                width: tw,
                height: th,
                depth_or_array_layers: 1,
            },
        );

        let photo_view = photo_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cutlass-quad-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cutlass-quad-bind-group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&photo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let mut guard = self.inner.lock().expect("quad preview poisoned");
        guard.source_texture = Some(photo_tex);
        guard.pipeline = Some(pipeline);
        guard.bind_group = Some(bind_group);
    }

    /// Call from [`slint::RenderingState::BeforeRendering`].
    ///
    /// Redraws the quad each frame. Returns [`Some`] with a fresh [`Image`] when the backing
    /// texture was recreated (window resize); assign that to your Slint `Image` property.
    pub fn render(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
    ) -> Option<Image> {
        if width == 0 || height == 0 {
            return None;
        }

        let mut guard = self.inner.lock().expect("quad preview poisoned");

        let extent_changed = guard.extent_px != (width, height) || guard.image.is_none();
        let mut image_for_ui = None;

        if extent_changed {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("cutlass-preview-rgba"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

            let image = match Image::try_from(texture) {
                Ok(img) => img,
                Err(e) => {
                    tracing::error!(?e, "texture import for Slint Image failed");
                    return None;
                }
            };

            image_for_ui = Some(image.clone());
            guard.image = Some(image);
            guard.extent_px = (width, height);
        }

        let pipeline = guard.pipeline.as_ref().expect("setup before render");
        let bind_group = guard.bind_group.as_ref().expect("setup before render");
        let tex = guard
            .image
            .as_ref()
            .expect("image")
            .to_wgpu_28_texture()
            .expect("wgpu texture");
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        render_quad_to_view(device, queue, pipeline, bind_group, &view);

        image_for_ui
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cutlass-quad-shader"),
        source: wgpu::ShaderSource::Wgsl(QUAD_WGSL.into()),
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("cutlass-quad-pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn render_quad_to_view(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    view: &wgpu::TextureView,
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("cutlass-quad-encoder"),
    });

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("cutlass-quad-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    queue.submit([encoder.finish()]);
}
