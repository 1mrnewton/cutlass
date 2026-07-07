//! The GPU renderer: realize a [`Scene`] into a composited [`RgbaImage`].
//!
//! [`Renderer`] owns the expensive, reusable pieces — a `wgpu` device, the
//! compositor pipelines, a text rasterizer, and a per-media decoder cache — so
//! a single instance renders many frames (preview scrub, export) without
//! re-initializing the GPU or re-opening decoders.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use cutlass_compositor::{
    CompositeLayer, Compositor, CompositorConfig, CompositorError, FrameSink, GpuContext,
    ImageSink, LayerPlacement, RgbaImage, SdfLayer,
};
use cutlass_core::{RationalTime, VideoDecoder, VideoFrame};
use cutlass_decoder::OutputMode;
use cutlass_models::{MediaId, Project};
use cutlass_shapes::{PathRaster, ShapeStyle};
use cutlass_text::TextRenderer;

use crate::error::RenderError;
use crate::resolve::{ResolveOverrides, resolve, resolve_with};
use crate::scene::{LayerSource, Scene, SizeSpec};

/// A composited frame slower than this logs its stage breakdown at `info`
/// (default-visible): interactive preview budgets a few frames of latency,
/// and anything past this is worth attributing to decode vs GPU work.
const SLOW_FRAME_LOG_MS: f64 = 150.0;

/// Per-stage timing of the most recent successful frame render.
///
/// Callers that adapt render *resolution* to cost (the preview quality
/// ladder) need the split, not the total: decode runs at the source's native
/// size no matter how small the output canvas is, so only
/// [`scaled_cost_ms`](Self::scaled_cost_ms) responds to rendering smaller.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameStats {
    /// Media decode time summed across layers (resolution-independent).
    pub decode_ms: f64,
    /// Text/shape/still realize time (raster caches, still decodes).
    pub raster_ms: f64,
    /// GPU composite + readback — scales with output pixels.
    pub composite_ms: f64,
}

impl FrameStats {
    /// The portion of the frame cost that shrinks with output resolution
    /// (composite + raster) — what a quality ladder can actually buy back.
    pub fn scaled_cost_ms(&self) -> f64 {
        self.raster_ms + self.composite_ms
    }

    /// Whole-frame cost (decode + raster + composite).
    pub fn total_ms(&self) -> f64 {
        self.decode_ms + self.raster_ms + self.composite_ms
    }
}

/// How media decoders are positioned when realizing a frame.
///
/// `Exact` is correctness (export, settled preview); `NearestSync` is the
/// scrub-latency escape hatch: on long-GOP sources an exact mid-GOP target
/// costs a keyframe-prefix walk of hundreds of decodes, where the nearest
/// sync frame costs one. Frames rendered under `NearestSync` may show
/// content up to a GOP *before* the requested time — callers own the
/// follow-up exact render and must never cache snapped output under an
/// exact key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SeekPolicy {
    /// Decode the exact frame covering each layer's source time.
    #[default]
    Exact,
    /// Snap to the cheapest frame near the target: one decode from the
    /// sync point at/before it (or a short exact roll when the target is
    /// just ahead of the decoder's position).
    NearestSync,
}

/// Renders project frames on a headless (or shared) GPU.
pub struct Renderer {
    gpu: GpuContext,
    compositor: Compositor,
    text: TextRenderer,
    /// Pen-path rasterizer (memoized, like `text`). Parametric shapes never
    /// touch it — they realize as GPU SDF layers.
    paths: PathRaster,
    /// One open decoder per **on-screen use** of a media source, reused
    /// across frames. Decoders are stateful (seek + walk), so keeping them
    /// warm makes sequential export and nearby scrubbing cheap. Keyed by
    /// `(media, occurrence slot)` — the per-scene index of that media in the
    /// layer walk — because two simultaneously visible clips of the same
    /// file sit at *different* source times: sharing one decoder cursor
    /// would ping-pong it between them, paying a seek plus GOP-prefix
    /// re-decode per layer per frame. Slots are stable while the stack is
    /// (same walk order), so each clip keeps its warm decoder.
    decoders: HashMap<(MediaId, u32), Box<dyn VideoDecoder>>,
    /// Decode-once cache for still images: one straight-alpha RGBA bitmap per
    /// media source, reused for every frame the still is on screen. Bounded by
    /// the project's still count, with each entry capped at
    /// [`cutlass_decoder::image::MAX_DECODE_DIMENSION`] on the long side.
    stills: HashMap<MediaId, RgbaImage>,
    /// Preferred decoder output mode. Apple and Windows start in
    /// [`OutputMode::Gpu`] so hardware-decoded surfaces (`CVPixelBuffer` /
    /// shared D3D11 NV12 textures) import into the compositor with no CPU
    /// copy; if a produced surface can't be imported (e.g. 10-bit/HDR, or a
    /// GPU without NV12 texture support), the renderer permanently falls back
    /// to [`OutputMode::Cpu`] and retries.
    decode_mode: OutputMode,
    /// Runtime-only substitute decode paths (preview proxies): when present
    /// (and [`use_proxies`](Self::use_proxies) holds), decoders for a media
    /// id open this file instead of the project's. Session state — never
    /// serialized, cleared when the session's media-id space changes
    /// (open/load/relink). Content must match the original frame-for-frame;
    /// only resolution/GOP may differ, so placement geometry (driven by the
    /// model's dimensions) stays valid and normalized UVs sample correctly.
    proxies: HashMap<MediaId, PathBuf>,
    /// Whether [`decode`](Self::decode) honors `proxies`. Preview leaves
    /// this on; full-quality paths sharing this renderer (the engine's
    /// export command) flip it off for the pass. Toggling drops the
    /// proxied media's open decoders so no cursor outlives its file.
    use_proxies: bool,
    /// Stage timings of the last successful render (see [`FrameStats`]).
    last_stats: FrameStats,
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
            stills: HashMap::new(),
            decode_mode: default_decode_mode(),
            proxies: HashMap::new(),
            use_proxies: true,
            last_stats: FrameStats::default(),
        })
    }

    /// Stage timings of the most recent successful render — how the last
    /// frame's cost split between decode, raster, and composite. Zeroed
    /// until the first render completes.
    pub fn last_frame_stats(&self) -> FrameStats {
        self.last_stats
    }

    /// Decode `media` from the file at `path` (a preview proxy) instead of
    /// the project's own path, from the next frame on. Drops the media's
    /// open decoders so no cursor keeps reading the original.
    pub fn set_proxy(&mut self, media: MediaId, path: PathBuf) {
        self.proxies.insert(media, path);
        self.drop_decoders_for(media);
    }

    /// Remove `media`'s proxy substitution (e.g. the media was relinked to a
    /// new file), returning decode to the project's path.
    pub fn clear_proxy(&mut self, media: MediaId) {
        if self.proxies.remove(&media).is_some() {
            self.drop_decoders_for(media);
        }
    }

    /// Remove every proxy substitution — required when the session swaps
    /// projects: media ids persist in project files, so an id from the old
    /// registry can name a different file in the new one.
    pub fn clear_proxies(&mut self) {
        let stale: Vec<MediaId> = self.proxies.keys().copied().collect();
        self.proxies.clear();
        for media in stale {
            self.drop_decoders_for(media);
        }
    }

    /// The proxy path registered for `media`, if any (regardless of
    /// [`set_use_proxies`](Self::set_use_proxies)).
    pub fn proxy_for(&self, media: MediaId) -> Option<&Path> {
        self.proxies.get(&media).map(PathBuf::as_path)
    }

    /// Turn proxy substitution on/off for subsequent decodes. Off renders
    /// full quality from the originals (the engine's in-place export);
    /// proxied media's open decoders drop on every change of state so a
    /// stale cursor can never serve the wrong file.
    pub fn set_use_proxies(&mut self, on: bool) {
        if self.use_proxies == on {
            return;
        }
        self.use_proxies = on;
        let proxied: Vec<MediaId> = self.proxies.keys().copied().collect();
        for media in proxied {
            self.drop_decoders_for(media);
        }
    }

    /// Drop every open decoder slot for `media` (see `decoders` — one entry
    /// per on-screen occurrence).
    fn drop_decoders_for(&mut self, media: MediaId) {
        self.decoders.retain(|(id, _), _| *id != media);
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

    /// [`render_frame`](Self::render_frame) with live-preview
    /// [`ResolveOverrides`] applied — the gesture/inspector preview path.
    pub fn render_frame_with(
        &mut self,
        project: &Project,
        t: RationalTime,
        overrides: ResolveOverrides<'_>,
    ) -> Result<RgbaImage, RenderError> {
        let scene = resolve_with(project, t, overrides)?;
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
        self.render_frame_fit_with(
            project,
            t,
            max_width,
            max_height,
            ResolveOverrides::default(),
        )
    }

    /// [`render_frame_fit`](Self::render_frame_fit) with live-preview
    /// [`ResolveOverrides`] applied.
    pub fn render_frame_fit_with(
        &mut self,
        project: &Project,
        t: RationalTime,
        max_width: u32,
        max_height: u32,
        overrides: ResolveOverrides<'_>,
    ) -> Result<RgbaImage, RenderError> {
        let mut scene = resolve_with(project, t, overrides)?;
        scene.fit_within(max_width, max_height);
        self.render_scene(project, &scene)
    }

    /// [`render_frame_with`](Self::render_frame_with) writing the composited
    /// rows directly into `sink`-provided storage (see
    /// [`render_scene_into`](Self::render_scene_into)), decoding under
    /// `policy` — the interactive-preview entry point, where a scrub drag
    /// passes [`SeekPolicy::NearestSync`].
    pub fn render_frame_into_with(
        &mut self,
        project: &Project,
        t: RationalTime,
        overrides: ResolveOverrides<'_>,
        policy: SeekPolicy,
        sink: &mut dyn FrameSink,
    ) -> Result<(), RenderError> {
        let scene = resolve_with(project, t, overrides)?;
        self.render_scene_into_policy(project, &scene, sink, policy)
    }

    /// [`render_frame_fit_with`](Self::render_frame_fit_with) writing the
    /// composited rows directly into `sink`-provided storage — the
    /// interactive-preview path, which hands the pixels straight to the UI's
    /// frame buffer instead of round-tripping through an [`RgbaImage`].
    /// Decodes under `policy` (see [`render_frame_into_with`](Self::render_frame_into_with)).
    #[allow(clippy::too_many_arguments)] // the preview call: bound + overrides + policy are all load-bearing
    pub fn render_frame_fit_into_with(
        &mut self,
        project: &Project,
        t: RationalTime,
        max_width: u32,
        max_height: u32,
        overrides: ResolveOverrides<'_>,
        policy: SeekPolicy,
        sink: &mut dyn FrameSink,
    ) -> Result<(), RenderError> {
        let mut scene = resolve_with(project, t, overrides)?;
        scene.fit_within(max_width, max_height);
        self.render_scene_into_policy(project, &scene, sink, policy)
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
    /// When decoding zero-copy ([`OutputMode::Gpu`] on Apple/Windows) produces
    /// a surface the compositor can't import, this falls back to CPU decode
    /// once and retries, so unusual formats (10-bit/HDR) still render.
    pub fn render_scene(
        &mut self,
        project: &Project,
        scene: &Scene,
    ) -> Result<RgbaImage, RenderError> {
        let mut sink = ImageSink::default();
        self.render_scene_into(project, scene, &mut sink)?;
        Ok(sink
            .into_image()
            .expect("render_scene_into fills the sink on success"))
    }

    /// [`render_scene`](Self::render_scene) writing the composited rows
    /// directly into `sink`-provided storage. The sink is consulted only
    /// after the GPU work succeeded, so the CPU-decode fallback retry can
    /// reuse it — at most one attempt ever writes.
    pub fn render_scene_into(
        &mut self,
        project: &Project,
        scene: &Scene,
        sink: &mut dyn FrameSink,
    ) -> Result<(), RenderError> {
        self.render_scene_into_policy(project, scene, sink, SeekPolicy::Exact)
    }

    fn render_scene_into_policy(
        &mut self,
        project: &Project,
        scene: &Scene,
        sink: &mut dyn FrameSink,
        policy: SeekPolicy,
    ) -> Result<(), RenderError> {
        match self.render_scene_once(project, scene, sink, policy) {
            Err(RenderError::Compositor(CompositorError::UnsupportedFormat(_)))
                if self.decode_mode == OutputMode::Gpu =>
            {
                // A zero-copy surface couldn't be imported (e.g. 10-bit/HDR).
                // Permanently fall back to CPU decode and retry; the dropped
                // decoders reopen in CPU mode on the next decode.
                self.decode_mode = OutputMode::Cpu;
                self.decoders.clear();
                self.render_scene_once(project, scene, sink, policy)
            }
            other => other,
        }
    }

    fn render_scene_once(
        &mut self,
        project: &Project,
        scene: &Scene,
        sink: &mut dyn FrameSink,
        policy: SeekPolicy,
    ) -> Result<(), RenderError> {
        let realize_started = Instant::now();
        // Decode time accumulated across media layers — on weak machines this
        // is where whole-frame seconds hide, so the stage log splits it out.
        let mut decode_ms = 0.0f64;
        // First pass: decode/rasterize each layer into owned pixels and a final
        // placement. Held in `realized` so the borrowed `CompositeLayer`s built
        // below stay valid through the composite call.
        let mut realized: Vec<Realized> = Vec::with_capacity(scene.layers.len());
        // Per-scene occurrence index of each media source: the decoder-cache
        // slot for repeated uses of one file (see `decoders`).
        let mut occurrence: HashMap<MediaId, u32> = HashMap::new();
        for layer in &scene.layers {
            let color_grade = layer.color_grade;
            // The layer carries the anchor position; the quad center falls out
            // of the final pixel size (bitmap sizes only exist after raster).
            let place = |size: [f32; 2]| LayerPlacement {
                center: layer.quad_center(size),
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
                        color_grade,
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
                        color_grade,
                    });
                }
                LayerSource::Media { media, source_time } => {
                    let slot = occurrence.entry(*media).or_insert(0);
                    let decode_started = Instant::now();
                    let frame = self.decode(project, *media, *slot, *source_time, policy)?;
                    decode_ms += decode_started.elapsed().as_secs_f64() * 1000.0;
                    *slot += 1;
                    let size = fixed_size(layer.size, [scene.width as f32, scene.height as f32]);
                    realized.push(Realized::Frame {
                        frame,
                        placement: place(size),
                        uv: layer.uv,
                        color_grade,
                    });
                }
                LayerSource::Still { media } => {
                    self.ensure_still(project, *media)?;
                    let size = fixed_size(layer.size, [scene.width as f32, scene.height as f32]);
                    realized.push(Realized::Still {
                        media: *media,
                        placement: place(size),
                        uv: layer.uv,
                        color_grade,
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
                        color_grade,
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
                        color_grade,
                    });
                }
            }
        }

        // Second pass: borrow the owned pixels into the compositor's layer type.
        let layers: Vec<CompositeLayer> = realized
            .iter()
            .map(|r| match r {
                Realized::Solid {
                    rgba,
                    placement,
                    color_grade,
                } => CompositeLayer::solid(*rgba, *placement).with_color_grade(*color_grade),
                Realized::Bitmap {
                    image,
                    placement,
                    uv,
                    color_grade,
                } => CompositeLayer::rgba(image, *placement)
                    .with_uv(*uv)
                    .with_color_grade(*color_grade),
                Realized::Frame {
                    frame,
                    placement,
                    uv,
                    color_grade,
                } => CompositeLayer::frame(frame, *placement)
                    .with_uv(*uv)
                    .with_color_grade(*color_grade),
                // `ensure_still` populated the cache in the first pass.
                Realized::Still {
                    media,
                    placement,
                    uv,
                    color_grade,
                } => CompositeLayer::rgba(&self.stills[media], *placement)
                    .with_uv(*uv)
                    .with_color_grade(*color_grade),
                Realized::Sdf {
                    shape,
                    placement,
                    color_grade,
                } => CompositeLayer::sdf(*shape, *placement).with_color_grade(*color_grade),
            })
            .collect();

        let config =
            CompositorConfig::new(scene.width, scene.height).with_background(scene.background);
        let realize_ms = realize_started.elapsed().as_secs_f64() * 1000.0;
        let composite_started = Instant::now();
        self.compositor
            .render_into(&self.gpu, &config, &layers, sink)?;

        // Stage breakdown per frame: decode (media layers), raster (text/
        // shape/still realize minus decode), composite (GPU submit + mapped
        // readback). Slow frames surface at `info` so a default-filtered log
        // shows where the seconds go on decode- or GPU-bound machines.
        let composite_ms = composite_started.elapsed().as_secs_f64() * 1000.0;
        let raster_ms = (realize_ms - decode_ms).max(0.0);
        let total_ms = realize_ms + composite_ms;
        self.last_stats = FrameStats {
            decode_ms,
            raster_ms,
            composite_ms,
        };
        if total_ms > SLOW_FRAME_LOG_MS {
            tracing::info!(
                decode_ms = %format_args!("{decode_ms:.1}"),
                raster_ms = %format_args!("{raster_ms:.1}"),
                composite_ms = %format_args!("{composite_ms:.1}"),
                layers = layers.len(),
                width = scene.width,
                height = scene.height,
                "slow frame render: {total_ms:.0} ms"
            );
        } else {
            tracing::trace!(
                decode_ms = %format_args!("{decode_ms:.1}"),
                raster_ms = %format_args!("{raster_ms:.1}"),
                composite_ms = %format_args!("{composite_ms:.1}"),
                layers = layers.len(),
                "frame render: {total_ms:.1} ms"
            );
        }
        Ok(())
    }

    /// Decode the frame of `media` at `source_time`, opening (and caching) a
    /// decoder for this `(media, slot)` use on first sight — over the
    /// media's proxy file when one is registered (see
    /// [`set_proxy`](Self::set_proxy)). Under [`SeekPolicy::NearestSync`]
    /// the frame may be the sync point before `source_time` rather than the
    /// exact frame (see [`SeekPolicy`]).
    fn decode(
        &mut self,
        project: &Project,
        media: MediaId,
        slot: u32,
        source_time: RationalTime,
        policy: SeekPolicy,
    ) -> Result<VideoFrame, RenderError> {
        let mode = self.decode_mode;
        let proxy = self.use_proxies.then(|| self.proxies.get(&media)).flatten();
        let decoder = match self.decoders.entry((media, slot)) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let src = project
                    .media(media)
                    .ok_or(RenderError::MissingMedia(media))?;
                let path = proxy.map(PathBuf::as_path).unwrap_or_else(|| src.path());
                e.insert(open_decoder(path, mode)?)
            }
        };
        let mut frame = match policy {
            SeekPolicy::Exact => decoder.frame_at(source_time)?,
            SeekPolicy::NearestSync => decoder.frame_at_nearest(source_time)?,
        };
        // Proxies run a frame short by construction (generation drops the
        // container-reported tail, which routinely over-counts by one): an
        // exact target at the media's very end can overshoot the proxy's
        // EOF. Show the nearest decodable frame instead of failing the
        // whole composite over the final tick.
        if frame.is_none() && proxy.is_some() {
            frame = decoder.frame_at_nearest(source_time)?;
        }
        frame.ok_or(RenderError::NoFrame {
            media,
            time: source_time,
        })
    }

    /// Tight size (canvas px, at transform scale 1.0) of the content
    /// `generator` draws on a `canvas_w`×`canvas_h` canvas — what a preview
    /// selection box should hug. Animated params sample at clip-local `tick`.
    /// `None` for generators that draw nothing (empty text, degenerate
    /// shapes, kinds the compositor doesn't draw yet).
    ///
    /// `&mut self`: text and pen-path sizes come from the memoized
    /// rasterizers, so a miss here warms the composite path too.
    pub fn generator_content_size(
        &mut self,
        generator: &cutlass_models::Generator,
        canvas_w: u32,
        canvas_h: u32,
        tick: i64,
    ) -> Option<(u32, u32)> {
        let layer = crate::resolve::resolve_generator(
            generator,
            [0.0, 0.0],
            [0.5, 0.5],
            0.0,
            1.0,
            [0.0, 0.0, 1.0, 1.0],
            canvas_w as f32,
            canvas_h as f32,
            1.0,
            tick,
        )?;
        match layer.size {
            SizeSpec::Fixed(size) => Some((size[0].round() as u32, size[1].round() as u32)),
            // Bitmap layers (text, pen paths) size to their raster.
            SizeSpec::BitmapScaled(_) => {
                let image = match &layer.source {
                    LayerSource::Text { content, style } => self.text.rasterize(content, style),
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
                        self.paths.rasterize(path, &style, *raster_scale)
                    }
                    _ => return None,
                };
                (image.width > 0 && image.height > 0).then_some((image.width, image.height))
            }
        }
    }

    /// Decode `media`'s single still frame into the cache on first use.
    fn ensure_still(&mut self, project: &Project, media: MediaId) -> Result<(), RenderError> {
        if self.stills.contains_key(&media) {
            return Ok(());
        }
        let src = project
            .media(media)
            .ok_or(RenderError::MissingMedia(media))?;
        let image = cutlass_decoder::decode_image(src.path())?;
        self.stills.insert(media, image);
        Ok(())
    }
}

/// An owned, decoded/rasterized layer kept alive while the compositor borrows it.
enum Realized {
    Frame {
        frame: VideoFrame,
        placement: LayerPlacement,
        uv: [f32; 4],
        color_grade: Option<cutlass_compositor::ColorGrade>,
    },
    /// A still image already decoded into the renderer's cache; the composite
    /// pass borrows the pixels from there instead of cloning them per frame.
    Still {
        media: MediaId,
        placement: LayerPlacement,
        uv: [f32; 4],
        color_grade: Option<cutlass_compositor::ColorGrade>,
    },
    Bitmap {
        image: RgbaImage,
        placement: LayerPlacement,
        uv: [f32; 4],
        color_grade: Option<cutlass_compositor::ColorGrade>,
    },
    Solid {
        rgba: [u8; 4],
        placement: LayerPlacement,
        color_grade: Option<cutlass_compositor::ColorGrade>,
    },
    Sdf {
        shape: SdfLayer,
        placement: LayerPlacement,
        color_grade: Option<cutlass_compositor::ColorGrade>,
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

/// The renderer's starting decode mode: zero-copy GPU surfaces on Apple and
/// Windows (with a CPU fallback in [`Renderer::render_scene`]), CPU planes
/// elsewhere until those backends grow a zero-copy import path.
#[cfg(any(target_vendor = "apple", target_os = "windows"))]
fn default_decode_mode() -> OutputMode {
    OutputMode::Gpu
}

#[cfg(not(any(target_vendor = "apple", target_os = "windows")))]
fn default_decode_mode() -> OutputMode {
    OutputMode::Cpu
}

/// Open the platform's native decoder for `path` in `mode`. Only Apple and
/// Windows have a zero-copy import path today, so the other backends always
/// decode to CPU planes regardless of the requested mode.
#[cfg(target_vendor = "apple")]
fn open_decoder(path: &Path, mode: OutputMode) -> Result<Box<dyn VideoDecoder>, RenderError> {
    Ok(Box::new(cutlass_decoder::AvfDecoder::open(path, mode)?))
}

#[cfg(target_os = "windows")]
fn open_decoder(path: &Path, mode: OutputMode) -> Result<Box<dyn VideoDecoder>, RenderError> {
    Ok(Box::new(cutlass_decoder::WmfDecoder::open(path, mode)?))
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
