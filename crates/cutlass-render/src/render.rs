//! The GPU renderer: realize a [`Scene`] into a composited [`RgbaImage`].
//!
//! [`Renderer`] owns the expensive, reusable pieces — a `wgpu` device, the
//! compositor pipelines, a text rasterizer, and a per-media decoder cache — so
//! a single instance renders many frames (preview scrub, export) without
//! re-initializing the GPU or re-opening decoders.

use std::collections::HashMap;
use std::path::Path;

use cutlass_compositor::{
    CompositeLayer, Compositor, CompositorConfig, CompositorError, GpuContext, LayerPlacement,
    RgbaImage, SdfLayer,
};
use cutlass_core::{RationalTime, VideoDecoder, VideoFrame};
use cutlass_decoder::OutputMode;
use cutlass_models::{MediaId, Project};
use cutlass_shapes::{PathRaster, ShapeStyle};
use cutlass_text::TextRenderer;

use crate::error::RenderError;
use crate::resolve::resolve;
use crate::scene::{LayerSource, Scene, SizeSpec};

/// Renders project frames on a headless (or shared) GPU.
pub struct Renderer {
    gpu: GpuContext,
    compositor: Compositor,
    text: TextRenderer,
    /// Pen-path rasterizer (memoized, like `text`). Parametric shapes never
    /// touch it — they realize as GPU SDF layers.
    paths: PathRaster,
    /// One open decoder per media source, reused across frames. Decoders are
    /// stateful (seek + walk), so keeping them warm makes sequential export and
    /// nearby scrubbing cheap.
    decoders: HashMap<MediaId, Box<dyn VideoDecoder>>,
    /// Preferred decoder output mode. Apple starts in [`OutputMode::Gpu`] so
    /// hardware-decoded `CVPixelBuffer`s import into the compositor with no CPU
    /// copy; if a produced surface can't be imported (e.g. 10-bit/HDR), the
    /// renderer permanently falls back to [`OutputMode::Cpu`] and retries.
    decode_mode: OutputMode,
}

impl Renderer {
    /// Bring up a headless GPU and build the renderer. Use this for export and
    /// tests; the desktop UI will instead share its device via the compositor's
    /// `GpuContext::from_parts`.
    pub fn new_headless() -> Result<Self, RenderError> {
        let gpu = GpuContext::new_headless_blocking()?;
        let compositor = Compositor::new(&gpu);
        Ok(Self {
            gpu,
            compositor,
            text: TextRenderer::new(),
            paths: PathRaster::new(),
            decoders: HashMap::new(),
            decode_mode: default_decode_mode(),
        })
    }

    /// Add a font face (TTF/OTF bytes) for deterministic text rendering. Without
    /// this the renderer uses the host's installed fonts.
    pub fn load_font(&mut self, data: Vec<u8>) {
        self.text.load_font(data);
    }

    /// Resolve `project` at `t` and composite the result into an [`RgbaImage`].
    pub fn render_frame(
        &mut self,
        project: &Project,
        t: RationalTime,
    ) -> Result<RgbaImage, RenderError> {
        let scene = resolve(project, t)?;
        self.render_scene(project, &scene)
    }

    /// [`render_frame`](Self::render_frame) scaled to fit within
    /// `max_width`×`max_height` (aspect preserved, never upscaled) — the
    /// interactive-preview path, where compositing and reading back a full
    /// 4K canvas per scrub tick would waste most of its pixels.
    pub fn render_frame_fit(
        &mut self,
        project: &Project,
        t: RationalTime,
        max_width: u32,
        max_height: u32,
    ) -> Result<RgbaImage, RenderError> {
        let mut scene = resolve(project, t)?;
        scene.fit_within(max_width, max_height);
        self.render_scene(project, &scene)
    }

    /// [`render_frame`](Self::render_frame) at an exact output size: content
    /// uniformly scaled (up or down) and centered, aspect mismatches
    /// letterboxed over the canvas background — the export path for a
    /// requested resolution.
    pub fn render_frame_sized(
        &mut self,
        project: &Project,
        t: RationalTime,
        width: u32,
        height: u32,
    ) -> Result<RgbaImage, RenderError> {
        let mut scene = resolve(project, t)?;
        scene.fit_into(width, height);
        self.render_scene(project, &scene)
    }

    /// Composite an already-resolved [`Scene`]. `project` supplies media file
    /// paths for the decoder cache.
    ///
    /// When decoding zero-copy (Apple's [`OutputMode::Gpu`]) produces a surface
    /// the compositor can't import, this falls back to CPU decode once and
    /// retries, so unusual formats (10-bit/HDR) still render.
    pub fn render_scene(
        &mut self,
        project: &Project,
        scene: &Scene,
    ) -> Result<RgbaImage, RenderError> {
        match self.render_scene_once(project, scene) {
            Err(RenderError::Compositor(CompositorError::UnsupportedFormat(_)))
                if self.decode_mode == OutputMode::Gpu =>
            {
                // A zero-copy surface couldn't be imported (e.g. 10-bit/HDR).
                // Permanently fall back to CPU decode and retry; the dropped
                // decoders reopen in CPU mode on the next decode.
                self.decode_mode = OutputMode::Cpu;
                self.decoders.clear();
                self.render_scene_once(project, scene)
            }
            other => other,
        }
    }

    fn render_scene_once(
        &mut self,
        project: &Project,
        scene: &Scene,
    ) -> Result<RgbaImage, RenderError> {
        // First pass: decode/rasterize each layer into owned pixels and a final
        // placement. Held in `realized` so the borrowed `CompositeLayer`s built
        // below stay valid through the composite call.
        let mut realized: Vec<Realized> = Vec::with_capacity(scene.layers.len());
        for layer in &scene.layers {
            let place = |size: [f32; 2]| LayerPlacement {
                center: layer.center,
                size,
                rotation: layer.rotation,
                opacity: layer.opacity,
            };
            match &layer.source {
                LayerSource::Solid(rgba) => {
                    let size = fixed_size(layer.size, [scene.width as f32, scene.height as f32]);
                    realized.push(Realized::Solid {
                        rgba: *rgba,
                        placement: place(size),
                    });
                }
                LayerSource::Text { content, style } => {
                    let image = self.text.rasterize(content, style);
                    if image.width == 0 || image.height == 0 {
                        continue; // nothing rasterized (no fonts / empty run)
                    }
                    let scale = match layer.size {
                        SizeSpec::BitmapScaled(s) => s,
                        SizeSpec::Fixed(_) => 1.0,
                    };
                    let size = [image.width as f32 * scale, image.height as f32 * scale];
                    realized.push(Realized::Bitmap {
                        image,
                        placement: place(size),
                        uv: layer.uv,
                    });
                }
                LayerSource::Media { media, source_time } => {
                    let frame = self.decode(project, *media, *source_time)?;
                    let size = fixed_size(layer.size, [scene.width as f32, scene.height as f32]);
                    realized.push(Realized::Frame {
                        frame,
                        placement: place(size),
                        uv: layer.uv,
                    });
                }
                LayerSource::Shape {
                    params,
                    fill,
                    stroke,
                    pad,
                } => {
                    // The resolver sized the quad as shape + pad per side;
                    // recover the shape's own half-extents for the shader.
                    let size = fixed_size(layer.size, [scene.width as f32, scene.height as f32]);
                    let half = [
                        (size[0] * 0.5 - pad).max(0.0),
                        (size[1] * 0.5 - pad).max(0.0),
                    ];
                    realized.push(Realized::Sdf {
                        shape: SdfLayer {
                            shape: params.with_half(half),
                            fill: *fill,
                            stroke: *stroke,
                        },
                        placement: place(size),
                    });
                }
                LayerSource::PathShape {
                    path,
                    fill,
                    stroke,
                    raster_scale,
                } => {
                    let style = ShapeStyle {
                        fill: Some(*fill).filter(|c| c[3] > 0),
                        stroke: *stroke,
                    };
                    let image = self.paths.rasterize(path, &style, *raster_scale);
                    if image.width == 0 || image.height == 0 {
                        continue; // nothing inked (degenerate path or style)
                    }
                    let scale = match layer.size {
                        SizeSpec::BitmapScaled(s) => s,
                        SizeSpec::Fixed(_) => 1.0,
                    };
                    let size = [image.width as f32 * scale, image.height as f32 * scale];
                    realized.push(Realized::Bitmap {
                        image,
                        placement: place(size),
                        uv: layer.uv,
                    });
                }
            }
        }

        // Second pass: borrow the owned pixels into the compositor's layer type.
        let layers: Vec<CompositeLayer> = realized
            .iter()
            .map(|r| match r {
                Realized::Solid { rgba, placement } => CompositeLayer::solid(*rgba, *placement),
                Realized::Bitmap {
                    image,
                    placement,
                    uv,
                } => CompositeLayer::rgba(image, *placement).with_uv(*uv),
                Realized::Frame {
                    frame,
                    placement,
                    uv,
                } => CompositeLayer::frame(frame, *placement).with_uv(*uv),
                Realized::Sdf { shape, placement } => CompositeLayer::sdf(*shape, *placement),
            })
            .collect();

        let config = CompositorConfig::new(scene.width, scene.height).with_background(scene.background);
        Ok(self.compositor.render(&self.gpu, &config, &layers)?)
    }

    /// Decode the frame of `media` at `source_time`, opening (and caching) a
    /// decoder for the source on first use.
    fn decode(
        &mut self,
        project: &Project,
        media: MediaId,
        source_time: RationalTime,
    ) -> Result<VideoFrame, RenderError> {
        let mode = self.decode_mode;
        let decoder = match self.decoders.entry(media) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let src = project.media(media).ok_or(RenderError::MissingMedia(media))?;
                e.insert(open_decoder(src.path(), mode)?)
            }
        };
        decoder
            .frame_at(source_time)?
            .ok_or(RenderError::NoFrame {
                media,
                time: source_time,
            })
    }
}

/// An owned, decoded/rasterized layer kept alive while the compositor borrows it.
enum Realized {
    Frame {
        frame: VideoFrame,
        placement: LayerPlacement,
        uv: [f32; 4],
    },
    Bitmap {
        image: RgbaImage,
        placement: LayerPlacement,
        uv: [f32; 4],
    },
    Solid {
        rgba: [u8; 4],
        placement: LayerPlacement,
    },
    Sdf {
        shape: SdfLayer,
        placement: LayerPlacement,
    },
}

/// The on-canvas size for a non-text layer, falling back to the canvas if a
/// bitmap-scaled spec ever reaches here (it shouldn't for media/solid).
fn fixed_size(size: SizeSpec, canvas: [f32; 2]) -> [f32; 2] {
    match size {
        SizeSpec::Fixed(s) => s,
        SizeSpec::BitmapScaled(_) => canvas,
    }
}

/// The renderer's starting decode mode: zero-copy GPU surfaces on Apple (with a
/// CPU fallback in [`Renderer::render_scene`]), CPU planes elsewhere until those
/// backends grow a zero-copy import path.
#[cfg(target_vendor = "apple")]
fn default_decode_mode() -> OutputMode {
    OutputMode::Gpu
}

#[cfg(not(target_vendor = "apple"))]
fn default_decode_mode() -> OutputMode {
    OutputMode::Cpu
}

/// Open the platform's native decoder for `path` in `mode`. Only Apple has a
/// zero-copy import path today, so the other backends always decode to CPU
/// planes regardless of the requested mode.
#[cfg(target_vendor = "apple")]
fn open_decoder(path: &Path, mode: OutputMode) -> Result<Box<dyn VideoDecoder>, RenderError> {
    Ok(Box::new(cutlass_decoder::AvfDecoder::open(path, mode)?))
}

#[cfg(target_os = "windows")]
fn open_decoder(path: &Path, _mode: OutputMode) -> Result<Box<dyn VideoDecoder>, RenderError> {
    Ok(Box::new(cutlass_decoder::WmfDecoder::open(
        path,
        OutputMode::Cpu,
    )?))
}

#[cfg(target_os = "android")]
fn open_decoder(path: &Path, _mode: OutputMode) -> Result<Box<dyn VideoDecoder>, RenderError> {
    Ok(Box::new(cutlass_decoder::MediaCodecDecoder::open(
        path,
        OutputMode::Cpu,
    )?))
}

#[cfg(not(any(target_vendor = "apple", target_os = "windows", target_os = "android")))]
fn open_decoder(_path: &Path, _mode: OutputMode) -> Result<Box<dyn VideoDecoder>, RenderError> {
    Err(RenderError::unsupported(
        "no native video decoder for this platform",
    ))
}
