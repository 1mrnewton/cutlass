mod decode;
mod types;
use bytemuck::{Pod, Zeroable};
use std::num::NonZero;
use std::path::Path;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use decode::VideoDecoder;

#[cfg(target_os = "macos")]
use decode::VideoDecoderGpu;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
    /// Normalized texture coords; (0,0) = top-left of the image, (1,1) = bottom-right.
    uv: [f32; 2],
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Nv12FragParams {
    unpack_for_srgb_target: u32,
    /// WebGPU aligns `uniform` structs to vec4 granularity.
    _pad: [u32; 3],
}

const VERTICES: &[Vertex] = &[
    Vertex {
        position: [-1.0, 1.0],
        uv: [0.0, 0.0],
    },
    Vertex {
        position: [-1.0, -1.0],
        uv: [0.0, 1.0],
    },
    Vertex {
        position: [1.0, -1.0],
        uv: [1.0, 1.0],
    },
    Vertex {
        position: [-1.0, 1.0],
        uv: [0.0, 0.0],
    },
    Vertex {
        position: [1.0, -1.0],
        uv: [1.0, 1.0],
    },
    Vertex {
        position: [1.0, 1.0],
        uv: [1.0, 0.0],
    },
];

enum AppDecoder {
    Cpu(VideoDecoder),
    #[cfg(target_os = "macos")]
    ZeroCopy(VideoDecoderGpu),
}

impl AppDecoder {
    fn size(&self) -> (u32, u32) {
        match self {
            AppDecoder::Cpu(d) => d.size(),
            #[cfg(target_os = "macos")]
            AppDecoder::ZeroCopy(d) => d.size(),
        }
    }
}

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    window: Arc<Window>,
    vertex_buffer: wgpu::Buffer,

    pipeline_rgba: wgpu::RenderPipeline,
    bind_group_rgba: wgpu::BindGroup,
    diffuse_texture: wgpu::Texture,
    sampler: wgpu::Sampler,

    #[cfg(target_os = "macos")]
    pipeline_nv12: wgpu::RenderPipeline,
    #[cfg(target_os = "macos")]
    nv12_bind_group_layout: wgpu::BindGroupLayout,
    #[cfg(target_os = "macos")]
    nv12_bind_group: Option<wgpu::BindGroup>,
    #[cfg(target_os = "macos")]
    nv12_frame: Option<decode::GpuFrame>,
    #[cfg(target_os = "macos")]
    nv12_params_buffer: wgpu::Buffer,
    #[cfg(target_os = "macos")]
    present_nv12: bool,
}

impl GpuState {
    async fn new(window: Arc<Window>, video_w: u32, video_h: u32) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find an adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Main Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                ..Default::default()
            })
            .await
            .expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        #[cfg(target_os = "macos")]
        let swapchain_targets_srgb = surface_format.is_srgb();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader_rgba = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RGBA quad"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/triangle.wgsl").into()),
        });

        let layout_rgba = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RGBA texture layout"),
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

        let pl_rgba = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RGBA pipeline layout"),
            bind_group_layouts: &[Some(&layout_rgba)],
            immediate_size: 0,
        });

        let pipeline_rgba = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("RGBA textured quad"),
            layout: Some(&pl_rgba),
            vertex: wgpu::VertexState {
                module: &shader_rgba,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_rgba,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let diffuse_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("RGBA upload texture"),
            size: wgpu::Extent3d {
                width: video_w,
                height: video_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let diffuse_view = diffuse_texture.create_view(&Default::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("video sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bind_group_rgba = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RGBA bind group"),
            layout: &layout_rgba,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Quad vertex buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        #[cfg(target_os = "macos")]
        let (pipeline_nv12, nv12_bind_group_layout, nv12_params_buffer) = {
            let shader_nv12 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("NV12 quad"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/video_nv12.wgsl").into()),
            });

            let nv12_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("NV12 fragment params"),
                contents: bytemuck::bytes_of(&Nv12FragParams {
                    unpack_for_srgb_target: u32::from(swapchain_targets_srgb),
                    _pad: [0; 3],
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            let layout_nv12 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("NV12 layout"),
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
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZero::new(
                                std::mem::size_of::<Nv12FragParams>() as u64,
                            ),
                        },
                        count: None,
                    },
                ],
            });

            let pl_nv12 = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("NV12 pipeline layout"),
                bind_group_layouts: &[Some(&layout_nv12)],
                immediate_size: 0,
            });

            let pipeline_nv12 = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("NV12 textured quad"),
                layout: Some(&pl_nv12),
                vertex: wgpu::VertexState {
                    module: &shader_nv12,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 0,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                            wgpu::VertexAttribute {
                                offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                        ],
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_nv12,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

            (pipeline_nv12, layout_nv12, nv12_params_buffer)
        };

        Self {
            surface,
            device,
            queue,
            config,
            window,
            vertex_buffer,
            pipeline_rgba,
            bind_group_rgba,
            diffuse_texture,

            sampler,

            #[cfg(target_os = "macos")]
            pipeline_nv12,
            #[cfg(target_os = "macos")]
            nv12_bind_group_layout,
            #[cfg(target_os = "macos")]
            nv12_bind_group: None,
            #[cfg(target_os = "macos")]
            nv12_frame: None,
            #[cfg(target_os = "macos")]
            nv12_params_buffer,
            #[cfg(target_os = "macos")]
            present_nv12: false,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn render(&mut self) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                let size = self.window.inner_size();
                self.resize(size.width, size.height);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return,
            wgpu::CurrentSurfaceTexture::Validation => {
                eprintln!("Surface validation error in get_current_texture");
                return;
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            #[cfg(target_os = "macos")]
            if self.present_nv12 && self.nv12_bind_group.is_some() {
                pass.set_pipeline(&self.pipeline_nv12);
                pass.set_bind_group(
                    0,
                    self.nv12_bind_group.as_ref().expect("checked is_some"),
                    &[],
                );
            } else {
                pass.set_pipeline(&self.pipeline_rgba);
                pass.set_bind_group(0, &self.bind_group_rgba, &[]);
            }

            #[cfg(not(target_os = "macos"))]
            {
                pass.set_pipeline(&self.pipeline_rgba);
                pass.set_bind_group(0, &self.bind_group_rgba, &[]);
            }
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.draw(0..VERTICES.len() as u32, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
    }

    pub fn upload_frame_rgba(&mut self, frame: &decode::DecodedFrame) {
        #[cfg(target_os = "macos")]
        {
            self.present_nv12 = false;
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.diffuse_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &frame.data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(frame.stride),
                rows_per_image: Some(frame.height),
            },
            wgpu::Extent3d {
                width: frame.width,
                height: frame.height,
                depth_or_array_layers: 1,
            },
        );
    }

    #[cfg(target_os = "macos")]
    pub fn upload_frame_nv12(&mut self, gpu: decode::GpuFrame) {
        let y_view = gpu.y.create_view(&Default::default());
        let cbcr_view = gpu.cbcr.create_view(&Default::default());

        self.nv12_bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("NV12 bind group"),
            layout: &self.nv12_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&y_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&cbcr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.nv12_params_buffer.as_entire_binding(),
                },
            ],
        }));

        self.nv12_frame = Some(gpu);
        self.present_nv12 = true;
    }
}

struct App {
    state: Option<GpuState>,
    decoder: AppDecoder,
    #[cfg(target_os = "macos")]
    vt_frame: Option<ffmpeg_next::util::frame::video::Video>,
}

impl App {
    fn new(video_path: impl AsRef<Path>) -> Self {
        let path = video_path.as_ref();

        #[cfg(target_os = "macos")]
        let (decoder, vt_frame) = match VideoDecoderGpu::open(path) {
            Ok(d) => {
                log::info!("using zero-copy VideoToolbox → IOSurface → wgpu path");
                (AppDecoder::ZeroCopy(d), Some(ffmpeg_next::util::frame::video::Video::empty()))
            }
            Err(e) => {
                log::warn!("zero-copy decoder unavailable ({e}); falling back to CPU RGBA");
                (
                    AppDecoder::Cpu(VideoDecoder::open(path).expect("open video")),
                    None,
                )
            }
        };

        #[cfg(not(target_os = "macos"))]
        let decoder = AppDecoder::Cpu(VideoDecoder::open(path).expect("open video"));

        Self {
            state: None,
            decoder,
            #[cfg(target_os = "macos")]
            vt_frame,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("cutlass scrub"))
                .unwrap(),
        );
        let (w, h) = self.decoder.size();
        let state = pollster::block_on(GpuState::new(window, w, h));

        self.state = Some(state);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::RedrawRequested => {
                state.render();
                state.window.request_redraw();
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Space),
                        state: ElementState::Pressed,
                        repeat: false,
                        ..
                    },
                ..
            } => match &mut self.decoder {
                AppDecoder::Cpu(decoder) => match decoder.decode_one() {
                    Ok(Some(frame)) => state.upload_frame_rgba(&frame),
                    Ok(None) => println!("eof"),
                    Err(e) => eprintln!("decode err: {e}"),
                },
                #[cfg(target_os = "macos")]
                AppDecoder::ZeroCopy(decoder) => {
                    let Some(vt_frame) = self.vt_frame.as_mut() else {
                        unreachable!("ZeroCopy implies vt_frame is Some");
                    };
                    match decoder.decode_into(vt_frame) {
                        Ok(Some(())) => match decode::GpuFrame::from_vt_video_frame(&state.device, vt_frame)
                        {
                            Ok(gpu) => state.upload_frame_nv12(gpu),
                            Err(e) => eprintln!("gpu import err: {e:?}"),
                        },
                        Ok(None) => println!("eof"),
                        Err(e) => eprintln!("decode err: {e}"),
                    }
                }
            },

            _ => {}
        }
    }
}

fn main() {
    env_logger::init();
    let path = "assets/13232364_3840_2160_24fps.mp4";
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(path);
    event_loop.run_app(&mut app).unwrap();
}
