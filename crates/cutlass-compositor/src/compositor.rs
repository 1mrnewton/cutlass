//! The composite pass: scene description in, RGBA canvas out.
//!
//! [`Compositor::render`] clears the canvas to the config background, then draws
//! each layer's placed quad bottom-to-top and reads the result back to an
//! [`RgbaImage`]. Three layer kinds share one placement model and target:
//!
//! - **video frames** — uploaded as planes and converted YUV→RGB on the GPU
//!   using the frame's own [`cutlass_core::ColorSpace`]; straight-alpha src-over.
//! - **RGBA bitmaps** — pre-rasterized generators (text/shapes/stickers) or
//!   stills; premultiplied on upload and composited premultiplied src-over so
//!   anti-aliased edges stay clean.
//! - **solid fills** — a fast no-texture path; straight-alpha src-over.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use cutlass_core::{
    CpuImage, FrameData, MatrixCoefficients, PixelFormat, Plane, RgbaImage, VideoFrame,
};

use crate::error::CompositorError;
use crate::gpu::GpuContext;
use crate::layer::{CompositeLayer, CompositorConfig, LayerContent, LayerPlacement};

/// Canvas pixel format. Plain (non-sRGB) `Unorm`: the YUV shader already emits
/// gamma-encoded R'G'B', so we store those bytes verbatim (display-ready SDR)
/// and read them straight back into PNG/export without an extra transfer.
const CANVAS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Caller-provided destination for a composited frame's pixels, so a render
/// can write its readback rows straight into storage the caller owns (a UI
/// framebuffer) instead of allocating an [`RgbaImage`] and copying again.
pub trait FrameSink {
    /// Called **once per successful render**, after the GPU work and buffer
    /// map succeeded, with the output dimensions. Must return exactly
    /// `width * height * 4` bytes; the compositor fills them with tightly
    /// packed row-major RGBA. A failed render never calls this.
    fn pixels(&mut self, width: u32, height: u32) -> &mut [u8];
}

/// The [`FrameSink`] behind the [`RgbaImage`]-returning render paths: it
/// allocates the image at the reported size and hands out its bytes.
#[derive(Default)]
pub struct ImageSink(Option<RgbaImage>);

impl ImageSink {
    /// The rendered image; `None` until a render succeeded into this sink.
    pub fn into_image(self) -> Option<RgbaImage> {
        self.0
    }
}

impl FrameSink for ImageSink {
    fn pixels(&mut self, width: u32, height: u32) -> &mut [u8] {
        let image = self.0.insert(RgbaImage::new(
            width,
            height,
            vec![0; (width as usize) * (height as usize) * 4],
        ));
        &mut image.pixels
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct YuvUniforms {
    linear: [f32; 4],
    trans_opacity: [f32; 4],
    uv_rect: [f32; 4],
    /// Kr, Kb, full-range flag, plane mode (0 = planar I420, 1 = biplanar NV12).
    coeffs: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SolidUniforms {
    color: [f32; 4],
    linear: [f32; 4],
    trans_opacity: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RgbaUniforms {
    linear: [f32; 4],
    trans_opacity: [f32; 4],
    uv_rect: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SdfUniforms {
    fill: [f32; 4],
    stroke_color: [f32; 4],
    linear: [f32; 4],
    /// Clip translation (x, y), layer opacity (z), stroke width px (w).
    trans_opacity: [f32; 4],
    /// Shape half-extents px (x, y), shape kind id (z), corner radius px (w).
    geo: [f32; 4],
    /// Star points (x), inner fraction (y); quad half-extents px (z, w).
    star: [f32; 4],
}

/// Which pipeline draws a built layer.
#[derive(Clone, Copy)]
enum LayerPipeline {
    Yuv,
    Rgba,
    Solid,
    Sdf,
}

/// GPU resources backing one layer for the duration of a render. The bind group
/// retains its textures/buffer, but we hold them too so the lifetimes are
/// unmistakable across the pass.
struct LayerGpu {
    pipeline: LayerPipeline,
    bind_group: wgpu::BindGroup,
    _textures: Vec<wgpu::Texture>,
    _uniform: wgpu::Buffer,
    /// Platform keep-alive for zero-copy imported surfaces (e.g. the Apple
    /// `CVMetalTexture` wrappers a GPU frame's planes are mapped through). Held
    /// until the layer's GPU resources drop, i.e. after the pass is submitted
    /// and waited on. `None` for CPU-uploaded and generated layers.
    _keep_alive: Option<Box<dyn core::any::Any + Send>>,
}

/// Straight-alpha src-over: the source fragment carries un-premultiplied color,
/// so the source color is weighted by its own alpha. Used by the solid and YUV
/// (fully-opaque) pipelines.
const STRAIGHT_OVER: wgpu::BlendState = wgpu::BlendState {
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

/// Premultiplied src-over: the source fragment is already alpha-weighted (the
/// RGBA pipeline), so the color factor is `One`. Correct for anti-aliased edges.
const PREMULTIPLIED_OVER: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
};

/// Canvas texture + readback buffer reused across renders while the size is
/// stable. Interactive preview (and export) renders the same size every
/// frame, so recreating these per render is pure allocation overhead on the
/// hot path; the pass clears the texture, so no state leaks between frames.
struct TargetCache {
    width: u32,
    height: u32,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    readback: wgpu::Buffer,
    /// Row pitch of `readback` (256-byte copy alignment).
    padded_bytes_per_row: u32,
}

/// A WGPU alpha-over compositor. Build once (pipelines are reused), render many.
pub struct Compositor {
    yuv_pipeline: wgpu::RenderPipeline,
    rgba_pipeline: wgpu::RenderPipeline,
    solid_pipeline: wgpu::RenderPipeline,
    sdf_pipeline: wgpu::RenderPipeline,
    yuv_layout: wgpu::BindGroupLayout,
    rgba_layout: wgpu::BindGroupLayout,
    solid_layout: wgpu::BindGroupLayout,
    sdf_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    target: Option<TargetCache>,
    /// Maps Apple `CVPixelBuffer` GPU surfaces into `wgpu` textures with no CPU
    /// copy. `None` when the device isn't a Metal device (e.g. a software
    /// adapter) — GPU frames then fall back to [`CompositorError::UnsupportedFormat`]
    /// and the caller can decode in CPU mode instead.
    #[cfg(target_vendor = "apple")]
    metal_import: Option<crate::metal_import::MetalSurfaceImporter>,
    /// Maps Windows shared D3D11 NV12 textures into `wgpu` textures with no
    /// CPU copy. `None` when the device isn't the D3D12 backend or lacks
    /// `TEXTURE_FORMAT_NV12` — GPU frames then fall back to
    /// [`CompositorError::UnsupportedFormat`] and the caller can decode in CPU
    /// mode instead.
    #[cfg(target_os = "windows")]
    dx12_import: Option<crate::dx12_import::Dx12SurfaceImporter>,
}

impl Compositor {
    /// Build the compositor's pipelines on an existing GPU context.
    pub fn new(gpu: &GpuContext) -> Self {
        let device = &gpu.device;

        let yuv_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cutlass.yuv.bgl"),
            entries: &[
                plane_tex_entry(0),
                plane_tex_entry(1),
                plane_tex_entry(2),
                sampler_entry(3),
                uniform_entry(4),
            ],
        });

        let rgba_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cutlass.rgba.bgl"),
            entries: &[plane_tex_entry(0), sampler_entry(1), uniform_entry(2)],
        });

        let solid_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cutlass.solid.bgl"),
            entries: &[uniform_entry(0)],
        });

        let sdf_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cutlass.sdf.bgl"),
            entries: &[uniform_entry(0)],
        });

        let yuv_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cutlass.yuv.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/yuv.wgsl").into()),
        });
        let rgba_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cutlass.rgba.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/rgba.wgsl").into()),
        });
        let solid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cutlass.solid.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/solid.wgsl").into()),
        });
        let sdf_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cutlass.shape.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shape.wgsl").into()),
        });

        // Opaque video and solids use straight-alpha src-over; RGBA bitmaps
        // (text/shape coverage) are premultiplied for clean edges.
        let yuv_pipeline = build_pipeline(
            device,
            "cutlass.yuv",
            &yuv_layout,
            &yuv_shader,
            STRAIGHT_OVER,
        );
        let rgba_pipeline = build_pipeline(
            device,
            "cutlass.rgba",
            &rgba_layout,
            &rgba_shader,
            PREMULTIPLIED_OVER,
        );
        let solid_pipeline = build_pipeline(
            device,
            "cutlass.solid",
            &solid_layout,
            &solid_shader,
            STRAIGHT_OVER,
        );
        // SDF shapes emit straight-alpha fragments (coverage folded into
        // alpha), like solids.
        let sdf_pipeline = build_pipeline(
            device,
            "cutlass.sdf",
            &sdf_layout,
            &sdf_shader,
            STRAIGHT_OVER,
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cutlass.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Self {
            yuv_pipeline,
            rgba_pipeline,
            solid_pipeline,
            sdf_pipeline,
            yuv_layout,
            rgba_layout,
            solid_layout,
            sdf_layout,
            sampler,
            target: None,
            #[cfg(target_vendor = "apple")]
            metal_import: crate::metal_import::MetalSurfaceImporter::new(gpu),
            #[cfg(target_os = "windows")]
            dx12_import: crate::dx12_import::Dx12SurfaceImporter::new(gpu),
        }
    }

    /// Make `self.target` hold a canvas + readback buffer for `width`×`height`,
    /// reusing the cached pair when the size already matches.
    fn ensure_target(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self
            .target
            .as_ref()
            .is_some_and(|t| t.width == width && t.height == height)
        {
            return;
        }

        let unpadded = width * 4;
        let padded = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cutlass.canvas"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: CANVAS_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cutlass.readback"),
            size: u64::from(padded) * u64::from(height),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.target = Some(TargetCache {
            width,
            height,
            texture,
            view,
            readback,
            padded_bytes_per_row: padded,
        });
    }

    /// Composite `layers` (bottom-to-top) onto a cleared canvas and read it
    /// back. The canvas texture and readback buffer are cached on `self` and
    /// reused while the render size stays the same.
    pub fn render(
        &mut self,
        gpu: &GpuContext,
        config: &CompositorConfig,
        layers: &[CompositeLayer<'_>],
    ) -> Result<RgbaImage, CompositorError> {
        let mut sink = ImageSink::default();
        self.render_into(gpu, config, layers, &mut sink)?;
        Ok(sink
            .into_image()
            .expect("render_into fills the sink on success"))
    }

    /// [`render`](Self::render) writing the readback rows directly into
    /// `sink`-provided storage — the interactive-preview path, where the
    /// tight `Vec` this would otherwise allocate is immediately copied into
    /// a UI pixel buffer and dropped.
    pub fn render_into(
        &mut self,
        gpu: &GpuContext,
        config: &CompositorConfig,
        layers: &[CompositeLayer<'_>],
        sink: &mut dyn FrameSink,
    ) -> Result<(), CompositorError> {
        if config.width == 0 || config.height == 0 {
            return Err(CompositorError::InvalidDimensions {
                width: config.width,
                height: config.height,
            });
        }

        let device = &gpu.device;

        // Build every layer's GPU resources before the pass so they outlive it.
        let mut built = Vec::with_capacity(layers.len());
        for layer in layers {
            built.push(self.build_layer(gpu, config, layer)?);
        }

        self.ensure_target(device, config.width, config.height);
        let target = self
            .target
            .as_ref()
            .expect("ensure_target populates the cache");

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cutlass.encoder"),
        });
        {
            let bg = config.background;
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("cutlass.pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: f64::from(bg[0]) / 255.0,
                            g: f64::from(bg[1]) / 255.0,
                            b: f64::from(bg[2]) / 255.0,
                            a: f64::from(bg[3]) / 255.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            for layer in &built {
                let pipeline = match layer.pipeline {
                    LayerPipeline::Yuv => &self.yuv_pipeline,
                    LayerPipeline::Rgba => &self.rgba_pipeline,
                    LayerPipeline::Solid => &self.solid_pipeline,
                    LayerPipeline::Sdf => &self.sdf_pipeline,
                };
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &layer.bind_group, &[]);
                pass.draw(0..6, 0..1);
            }
        }

        // Record the readback copy, then submit, *then* map — the copy must
        // execute on the queue before the buffer holds anything.
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
                    bytes_per_row: Some(target.padded_bytes_per_row),
                    rows_per_image: Some(config.height),
                },
            },
            wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit(std::iter::once(encoder.finish()));

        let result = map_readback(
            gpu,
            &target.readback,
            target.padded_bytes_per_row,
            config.width,
            config.height,
            sink,
        );
        if result.is_err() {
            // A failed map can leave the buffer in an unmappable state; drop
            // the cache so the next render starts from fresh resources.
            self.target = None;
        }
        result
    }

    fn build_layer(
        &self,
        gpu: &GpuContext,
        config: &CompositorConfig,
        layer: &CompositeLayer<'_>,
    ) -> Result<LayerGpu, CompositorError> {
        match &layer.content {
            LayerContent::Solid(rgba) => self.build_solid(gpu, config, &layer.placement, *rgba),
            LayerContent::Frame(frame) => self.build_frame(gpu, config, layer, frame),
            LayerContent::Rgba(image) => self.build_rgba(gpu, config, layer, image),
            LayerContent::Sdf(shape) => self.build_sdf(gpu, config, &layer.placement, shape),
        }
    }

    fn build_sdf(
        &self,
        gpu: &GpuContext,
        config: &CompositorConfig,
        placement: &LayerPlacement,
        shape: &crate::layer::SdfLayer,
    ) -> Result<LayerGpu, CompositorError> {
        use cutlass_shapes::SdfParams;

        let (linear, mut trans) = placement_affine(config, placement, 0.0);
        let stroke = shape.stroke.filter(|s| s.width > 0.0);
        trans[3] = stroke.map_or(0.0, |s| s.width);

        let h = shape.shape.half;
        // Shape kind ids and parameter packing per shape.wgsl's `Sdf` block.
        let (kind, radius, star) = match shape.shape.params {
            SdfParams::RoundedRect { radius } => (0.0, radius, [0.0, 0.0]),
            SdfParams::Ellipse => (1.0, 0.0, [0.0, 0.0]),
            SdfParams::Star {
                points,
                inner,
                round,
            } => (2.0, round, [points as f32, inner]),
            SdfParams::Line => (3.0, 0.0, [0.0, 0.0]),
            SdfParams::Arrow => (4.0, 0.0, [0.0, 0.0]),
            SdfParams::Heart => (5.0, 0.0, [0.0, 0.0]),
        };

        let color = |rgba: [u8; 4]| {
            [
                f32::from(rgba[0]) / 255.0,
                f32::from(rgba[1]) / 255.0,
                f32::from(rgba[2]) / 255.0,
                f32::from(rgba[3]) / 255.0,
            ]
        };
        let uniforms = SdfUniforms {
            fill: color(shape.fill),
            stroke_color: stroke.map_or([0.0; 4], |s| color(s.rgba)),
            linear,
            trans_opacity: trans,
            geo: [h[0], h[1], kind, radius],
            star: [
                star[0],
                star[1],
                placement.size[0] * 0.5,
                placement.size[1] * 0.5,
            ],
        };
        let buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cutlass.sdf.uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cutlass.sdf.bg"),
            layout: &self.sdf_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Ok(LayerGpu {
            pipeline: LayerPipeline::Sdf,
            bind_group,
            _textures: Vec::new(),
            _uniform: buffer,
            _keep_alive: None,
        })
    }

    fn build_solid(
        &self,
        gpu: &GpuContext,
        config: &CompositorConfig,
        placement: &LayerPlacement,
        rgba: [u8; 4],
    ) -> Result<LayerGpu, CompositorError> {
        let (linear, trans) = placement_affine(config, placement, 0.0);
        let uniforms = SolidUniforms {
            color: [
                f32::from(rgba[0]) / 255.0,
                f32::from(rgba[1]) / 255.0,
                f32::from(rgba[2]) / 255.0,
                f32::from(rgba[3]) / 255.0,
            ],
            linear,
            trans_opacity: trans,
        };
        let buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cutlass.solid.uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cutlass.solid.bg"),
            layout: &self.solid_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Ok(LayerGpu {
            pipeline: LayerPipeline::Solid,
            bind_group,
            _textures: Vec::new(),
            _uniform: buffer,
            _keep_alive: None,
        })
    }

    fn build_frame(
        &self,
        gpu: &GpuContext,
        config: &CompositorConfig,
        layer: &CompositeLayer<'_>,
        frame: &VideoFrame,
    ) -> Result<LayerGpu, CompositorError> {
        let (cw, ch) = frame.coded_size;
        if cw == 0 || ch == 0 {
            return Err(CompositorError::MalformedFrame(format!(
                "coded size is {cw}x{ch}"
            )));
        }

        // Acquire the plane textures: CPU frames upload their planes; GPU frames
        // map their native surface straight into textures with no pixel copy.
        // `nv12` selects the shader's chroma path; `keep_alive` parks any native
        // wrappers (e.g. CVMetalTexture) that must outlive the GPU read.
        let (textures, nv12, keep_alive) = match &frame.data {
            FrameData::Cpu(image) => {
                let (textures, nv12) = upload_cpu_planes(gpu, frame, image, cw, ch)?;
                (textures, nv12, None)
            }
            FrameData::Gpu(surface) => self.import_gpu_planes(gpu, frame, surface, cw, ch)?,
        };

        // For NV12 the V slot is bound to the same CbCr texture (unsampled).
        let y_view = &textures[0].1;
        let u_view = &textures[1].1;
        let v_view = if nv12 { &textures[1].1 } else { &textures[2].1 };

        let resolved = frame.color.resolved();
        let (kr, kb) = luma_coeffs(resolved.matrix);
        let full = if resolved.range.is_full() { 1.0 } else { 0.0 };
        let plane_mode = if nv12 { 1.0 } else { 0.0 };

        // Container rotation rides on top of the placement rotation.
        let rotation = placement_rotation(layer, frame);
        let (linear, trans) = placement_affine_rot(config, &layer.placement, rotation);
        let uv_rect = visible_uv_rect(frame, layer.uv);

        let uniforms = YuvUniforms {
            linear,
            trans_opacity: trans,
            uv_rect,
            coeffs: [kr, kb, full, plane_mode],
        };
        let buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cutlass.yuv.uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cutlass.yuv.bg"),
            layout: &self.yuv_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(y_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(u_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(v_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffer.as_entire_binding(),
                },
            ],
        });

        Ok(LayerGpu {
            pipeline: LayerPipeline::Yuv,
            bind_group,
            _textures: textures.into_iter().map(|(tex, _)| tex).collect(),
            _uniform: buffer,
            _keep_alive: keep_alive,
        })
    }

    /// Map a zero-copy [`FrameData::Gpu`] surface into plane textures the YUV
    /// pipeline can sample. Returns the textures, the NV12 flag, and a native
    /// keep-alive bundle. On platforms/formats without an import path this is a
    /// [`CompositorError::UnsupportedFormat`] so the caller can fall back to a
    /// CPU-mode decode.
    #[allow(clippy::type_complexity)]
    fn import_gpu_planes(
        &self,
        gpu: &GpuContext,
        frame: &VideoFrame,
        surface: &cutlass_core::GpuSurface,
        cw: u32,
        ch: u32,
    ) -> Result<
        (
            Vec<(wgpu::Texture, wgpu::TextureView)>,
            bool,
            Option<Box<dyn core::any::Any + Send>>,
        ),
        CompositorError,
    > {
        #[cfg(target_vendor = "apple")]
        {
            use cutlass_core::GpuSurfaceKind;
            match surface.kind() {
                GpuSurfaceKind::CoreVideoPixelBuffer => {
                    if frame.format != PixelFormat::Nv12 {
                        return Err(CompositorError::UnsupportedFormat(format!(
                            "zero-copy import supports NV12 CoreVideo surfaces; got {:?}",
                            frame.format
                        )));
                    }
                    let importer = self.metal_import.as_ref().ok_or_else(|| {
                        CompositorError::UnsupportedFormat(
                            "Metal surface import is unavailable on this device".into(),
                        )
                    })?;
                    let (textures, keep) = importer.import_nv12(gpu, surface, (cw, ch))?;
                    Ok((textures, true, Some(Box::new(keep))))
                }
                other => Err(CompositorError::UnsupportedFormat(format!(
                    "GPU surface kind {other:?} import is not implemented yet"
                ))),
            }
        }
        #[cfg(target_os = "windows")]
        {
            use cutlass_core::GpuSurfaceKind;
            match surface.kind() {
                GpuSurfaceKind::D3D11Texture2D => {
                    if frame.format != PixelFormat::Nv12 {
                        return Err(CompositorError::UnsupportedFormat(format!(
                            "zero-copy import supports NV12 D3D11 surfaces; got {:?}",
                            frame.format
                        )));
                    }
                    let importer = self.dx12_import.as_ref().ok_or_else(|| {
                        CompositorError::UnsupportedFormat(
                            "D3D12 surface import is unavailable on this device".into(),
                        )
                    })?;
                    let textures = importer.import_nv12(gpu, surface, (cw, ch))?;
                    // No compositor-side keep-alive needed: the wgpu texture
                    // holds the opened ID3D12Resource, and the decoder-side
                    // pool slot rides in the *frame's* keep-alive, which the
                    // renderer holds until the pass completes.
                    Ok((textures, true, None))
                }
                other => Err(CompositorError::UnsupportedFormat(format!(
                    "GPU surface kind {other:?} import is not implemented yet"
                ))),
            }
        }
        // Android (`AHardwareBuffer`) import is not wired yet; the decoder
        // delivers CPU planes there, so this branch is currently unreachable
        // in practice.
        //
        // Android plan: decode into a `Surface`/`ImageReader` to obtain an
        // `AHardwareBuffer`, then import on the Vulkan backend via
        // `VK_ANDROID_external_memory_android_hardware_buffer`: create a
        // `VkImage` with external memory, bind the imported allocation, and
        // adopt it with `wgpu::hal::vulkan::Device::texture_from_raw`.
        //
        // Blocker (researched against wgpu 28, gfx-rs/wgpu#2320 / #3145 / #7324):
        // robust YCbCr sampling needs a `VkSamplerYcbcrConversion` bound as an
        // *immutable* sampler, and real devices hand back vendor formats
        // (`VK_FORMAT_UNDEFINED` + `VkExternalFormatANDROID`, e.g. UBWC) that
        // *only* resolve through that conversion. wgpu-hal hardcodes
        // `p_immutable_samplers = null` and its format map rejects
        // `VK_FORMAT_UNDEFINED`, so neither the immutable-sampler binding nor the
        // external format is expressible through wgpu's safe API today. The
        // Metal trick (split planes into R8/RG8 and convert in our shader) does
        // not carry over: per-plane VkImage aliasing only works for the rare
        // concrete-format AHBs, not the vendor-tiled ones that dominate.
        //
        // So the viable route is a raw-Vulkan YCbCr->RGBA conversion pass on a
        // device shared with wgpu (own `VkInstance`/`VkDevice` with the AHB +
        // ycbcr extensions, handed to wgpu per #7324/#7829), writing into an
        // RGBA `VkImage` that wgpu then samples — GPU-resident throughout, no CPU
        // copy, but not a single-sample import. Until that lands, Android stays
        // on the hardware-decode + CPU-plane path (one plane copy/frame), which
        // is already wired and correct.
        #[cfg(not(any(target_vendor = "apple", target_os = "windows")))]
        {
            let _ = (gpu, frame, surface, cw, ch);
            Err(CompositorError::UnsupportedFormat(
                "zero-copy GPU surface import is not implemented on this platform yet".into(),
            ))
        }
    }

    fn build_rgba(
        &self,
        gpu: &GpuContext,
        config: &CompositorConfig,
        layer: &CompositeLayer<'_>,
        image: &RgbaImage,
    ) -> Result<LayerGpu, CompositorError> {
        if image.width == 0 || image.height == 0 {
            return Err(CompositorError::MalformedFrame(format!(
                "rgba bitmap is {}x{}",
                image.width, image.height
            )));
        }
        let expected = (image.width as usize)
            .checked_mul(image.height as usize)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| CompositorError::MalformedFrame("rgba bitmap size overflow".into()))?;
        if image.pixels.len() < expected {
            return Err(CompositorError::MalformedFrame(format!(
                "rgba bitmap: {} bytes < {}×{}×4",
                image.pixels.len(),
                image.width,
                image.height
            )));
        }

        // Premultiply straight-alpha input so bilinear sampling and src-over
        // blending don't leak color from transparent texels into glyph edges.
        let mut premul = vec![0u8; expected];
        for (dst, src) in premul.chunks_exact_mut(4).zip(image.pixels.chunks_exact(4)) {
            let a = u16::from(src[3]);
            dst[0] = ((u16::from(src[0]) * a + 127) / 255) as u8;
            dst[1] = ((u16::from(src[1]) * a + 127) / 255) as u8;
            dst[2] = ((u16::from(src[2]) * a + 127) / 255) as u8;
            dst[3] = src[3];
        }

        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cutlass.rgba.tex"),
            size: wgpu::Extent3d {
                width: image.width,
                height: image.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
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
            &premul,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(image.width * 4),
                rows_per_image: Some(image.height),
            },
            wgpu::Extent3d {
                width: image.width,
                height: image.height,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let (linear, trans) = placement_affine(config, &layer.placement, 0.0);
        let uniforms = RgbaUniforms {
            linear,
            trans_opacity: trans,
            uv_rect: layer.uv,
        };
        let buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cutlass.rgba.uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cutlass.rgba.bg"),
            layout: &self.rgba_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
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

        Ok(LayerGpu {
            pipeline: LayerPipeline::Rgba,
            bind_group,
            _textures: vec![texture],
            _uniform: buffer,
            _keep_alive: None,
        })
    }
}

/// Map the (already-submitted) readback buffer and copy its rows — stripped
/// of the per-row copy padding — into `sink`-provided storage. The sink is
/// only consulted once the map succeeded.
fn map_readback(
    gpu: &GpuContext,
    buffer: &wgpu::Buffer,
    padded: u32,
    width: u32,
    height: u32,
    sink: &mut dyn FrameSink,
) -> Result<(), CompositorError> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|_| CompositorError::MapFailed)?;
    rx.recv()
        .map_err(|_| CompositorError::MapFailed)?
        .map_err(|_| CompositorError::MapFailed)?;

    let unpadded = (width * 4) as usize;
    let mapped = slice.get_mapped_range();
    let pixels = sink.pixels(width, height);
    debug_assert_eq!(pixels.len(), unpadded * height as usize);
    for row in 0..height as usize {
        let src = row * padded as usize;
        let dst = row * unpadded;
        pixels[dst..dst + unpadded].copy_from_slice(&mapped[src..src + unpadded]);
    }
    drop(mapped);
    buffer.unmap();

    Ok(())
}

fn plane_tex_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn build_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
    blend: wgpu::BlendState,
) -> wgpu::RenderPipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[layout],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs"),
            targets: &[Some(wgpu::ColorTargetState {
                format: CANVAS_FORMAT,
                blend: Some(blend),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// Upload a CPU frame's planes to fresh textures per its pixel layout, honoring
/// each plane's byte stride. Returns the plane textures and whether the layout
/// is NV12 (biplanar) — which selects the YUV shader's chroma path.
fn upload_cpu_planes(
    gpu: &GpuContext,
    frame: &VideoFrame,
    image: &CpuImage,
    cw: u32,
    ch: u32,
) -> Result<(Vec<(wgpu::Texture, wgpu::TextureView)>, bool), CompositorError> {
    match frame.format {
        PixelFormat::Nv12 => {
            expect_planes(image.planes.len(), 2, frame.format)?;
            let y = upload_plane(
                gpu,
                "y",
                wgpu::TextureFormat::R8Unorm,
                cw,
                ch,
                &image.planes[0],
            )?;
            let uv = upload_plane(
                gpu,
                "uv",
                wgpu::TextureFormat::Rg8Unorm,
                cw.div_ceil(2),
                ch.div_ceil(2),
                &image.planes[1],
            )?;
            Ok((vec![y, uv], true))
        }
        PixelFormat::Yuv420p => {
            expect_planes(image.planes.len(), 3, frame.format)?;
            let y = upload_plane(
                gpu,
                "y",
                wgpu::TextureFormat::R8Unorm,
                cw,
                ch,
                &image.planes[0],
            )?;
            let u = upload_plane(
                gpu,
                "u",
                wgpu::TextureFormat::R8Unorm,
                cw.div_ceil(2),
                ch.div_ceil(2),
                &image.planes[1],
            )?;
            let v = upload_plane(
                gpu,
                "v",
                wgpu::TextureFormat::R8Unorm,
                cw.div_ceil(2),
                ch.div_ceil(2),
                &image.planes[2],
            )?;
            Ok((vec![y, u, v], false))
        }
        other => Err(CompositorError::UnsupportedFormat(format!(
            "{other:?} is not handled by the compositor yet (8-bit NV12 / I420 only)"
        ))),
    }
}

/// Upload one plane to a fresh texture, honoring its byte stride.
fn upload_plane(
    gpu: &GpuContext,
    label: &str,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    plane: &Plane,
) -> Result<(wgpu::Texture, wgpu::TextureView), CompositorError> {
    let need = plane
        .stride
        .checked_mul(height as usize)
        .ok_or_else(|| CompositorError::MalformedFrame("plane size overflow".into()))?;
    if plane.data.len() < need {
        return Err(CompositorError::MalformedFrame(format!(
            "{label} plane: {} bytes < stride {} × {} rows",
            plane.data.len(),
            plane.stride,
            height
        )));
    }

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
        format,
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
        &plane.data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(plane.stride as u32),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    Ok((texture, view))
}

fn expect_planes(got: usize, want: usize, format: PixelFormat) -> Result<(), CompositorError> {
    if got == want {
        Ok(())
    } else {
        Err(CompositorError::MalformedFrame(format!(
            "{format:?} needs {want} planes, got {got}"
        )))
    }
}

/// Luma coefficients (Kr, Kb) for the YUV→RGB matrix. `resolved()` upstream maps
/// `Unspecified` to BT.709, so only concrete matrices reach here; `Identity` is
/// degenerate on YUV and falls back to BT.709.
fn luma_coeffs(matrix: MatrixCoefficients) -> (f32, f32) {
    match matrix {
        MatrixCoefficients::Bt601 => (0.299, 0.114),
        MatrixCoefficients::Bt2020Ncl => (0.2627, 0.0593),
        _ => (0.2126, 0.0722),
    }
}

/// Effective clockwise rotation (radians): the placement plus the frame's
/// container rotation, so sideways-recorded phone video plays upright.
fn placement_rotation(layer: &CompositeLayer<'_>, frame: &VideoFrame) -> f32 {
    layer.placement.rotation + (frame.rotation.degrees() as f32).to_radians()
}

/// Map the layer's UV rect (in visible-picture space) into the coded texture,
/// since the visible picture may be a sub-rect of a padded decode buffer.
fn visible_uv_rect(frame: &VideoFrame, uv: [f32; 4]) -> [f32; 4] {
    let (cw, ch) = (frame.coded_size.0 as f32, frame.coded_size.1 as f32);
    let vx = frame.visible.x as f32 / cw;
    let vy = frame.visible.y as f32 / ch;
    let vw = frame.visible.width as f32 / cw;
    let vh = frame.visible.height as f32 / ch;
    [
        vx + uv[0] * vw,
        vy + uv[1] * vh,
        vx + uv[2] * vw,
        vy + uv[3] * vh,
    ]
}

fn placement_affine(
    config: &CompositorConfig,
    p: &LayerPlacement,
    extra_rotation: f32,
) -> ([f32; 4], [f32; 4]) {
    placement_affine_rot(config, p, p.rotation + extra_rotation)
}

/// Build the unit-quad ([-0.5, 0.5]², +y down) → clip-space affine for a
/// placement, using `rotation` (radians, clockwise) instead of `p.rotation`.
fn placement_affine_rot(
    config: &CompositorConfig,
    p: &LayerPlacement,
    rotation: f32,
) -> ([f32; 4], [f32; 4]) {
    let (cw, ch) = (config.width as f32, config.height as f32);
    let (cos, sin) = (rotation.cos(), rotation.sin());
    // Canvas space (+y down): pos = center + R·(corner ⊙ size), R the clockwise
    // rotation [cos, -sin; sin, cos] in screen coords.
    let a = cos * p.size[0];
    let b = sin * p.size[0];
    let c = -sin * p.size[1];
    let d = cos * p.size[1];
    // Canvas px → clip space: x' = 2x/cw − 1, y' = 1 − 2y/ch (flip y).
    let linear = [2.0 * a / cw, -2.0 * b / ch, 2.0 * c / cw, -2.0 * d / ch];
    let trans = [
        2.0 * p.center[0] / cw - 1.0,
        1.0 - 2.0 * p.center[1] / ch,
        p.opacity,
        0.0,
    ];
    (linear, trans)
}
