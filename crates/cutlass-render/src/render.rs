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
    ColorGrade, CompositeLayer, Compositor, CompositorConfig, CompositorError, CompositorLayer,
    CubeLut, FrameSink, GpuContext, ImageSink, LayerChromaKey, LayerEffects, LayerLut, LayerMask,
    LayerPlacement, PassInstance, RgbaImage, SdfLayer, mask_kind,
};
use cutlass_core::{RationalTime, VideoDecoder, VideoFrame};
use cutlass_decoder::OutputMode;
use cutlass_models::{ClipId, MaskKind, MediaId, Project};
use cutlass_shapes::{PathRaster, ShapeStyle};
use cutlass_text::TextRenderer;

use crate::error::RenderError;
use crate::resolve::{ResolveOverrides, resolve, resolve_gesture_partitions, resolve_with};
use crate::scene::{LayerSource, ResolvedPass, Scene, SceneLut, SizeSpec};

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
    /// Decode-once cache for bundled stickers, keyed by catalog id: the whole
    /// frame sequence (a static sticker is one frame) plus per-frame delays.
    /// Bounded by the catalog (small, embedded assets); frame lookup on the
    /// hot path is O(frames) over the delay table, no decode.
    stickers: HashMap<String, StickerSequence>,
    /// File-backed Lottie animations, keyed by path: parsed composition +
    /// LRU of rasterized frames (capped-fps sampling, per-asset byte budget
    /// — never the pre-render-everything sticker strategy; see
    /// `docs/lottie-design.md`). Failed loads are remembered so a missing
    /// file logs once and draws nothing instead of re-probing every frame.
    lottie: HashMap<String, LottieState>,
    /// Monotonic per-scene stamp for the Lottie frame LRU: frames touched
    /// by the scene currently being composed are never evicted, so two
    /// clips of one asset can't alias mid-frame.
    lottie_stamp: u64,
    /// Parsed `.cube` LUTs, keyed by path. The GPU texture lives in the
    /// compositor's own cache (same key); this holds the CPU parse so a
    /// re-render never re-reads the file. Failed loads are remembered so a
    /// missing file logs once and grades nothing instead of re-probing
    /// every frame.
    luts: HashMap<String, CubeLutState>,
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
            stickers: HashMap::new(),
            lottie: HashMap::new(),
            lottie_stamp: 0,
            luts: HashMap::new(),
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

    /// Partitioned preview frames for a zero-drift transform gesture: layers
    /// below the clip (opaque background), the clip alone at identity
    /// transform (transparent background), and layers above (transparent).
    /// Each pass is fit to `max_width`×`max_height`. Returns `None` when the
    /// clip can't be partitioned (transitions, etc.).
    pub fn render_gesture_frames(
        &mut self,
        project: &Project,
        t: RationalTime,
        clip_id: ClipId,
        max_width: u32,
        max_height: u32,
    ) -> Result<Option<GestureFrames>, RenderError> {
        let Some(partitions) = resolve_gesture_partitions(project, t, clip_id)? else {
            return Ok(None);
        };

        let mut below = partitions.below;
        below.fit_within(max_width, max_height);
        let below = self.render_scene(project, &below)?;

        let mut sprite_scene = partitions.sprite;
        sprite_scene.fit_within(max_width, max_height);
        let mut sprite = self.render_scene(project, &sprite_scene)?;
        straighten_alpha(&mut sprite);

        let above = if partitions.above.layers.is_empty() {
            None
        } else {
            let mut above_scene = partitions.above;
            above_scene.fit_within(max_width, max_height);
            let mut image = self.render_scene(project, &above_scene)?;
            straighten_alpha(&mut image);
            Some(image)
        };

        Ok(Some(GestureFrames {
            below,
            sprite,
            above,
        }))
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
        // New scene, new LRU stamp: frames touched below are eviction-exempt
        // until the next scene.
        self.lottie_stamp += 1;
        // Decode time accumulated across media layers — on weak machines this
        // is where whole-frame seconds hide, so the stage log splits it out.
        let mut decode_ms = 0.0f64;
        // First pass: decode/rasterize each layer into owned pixels and a final
        // placement. Held in `realized` so the borrowed `CompositeLayer`s built
        // below stay valid through the composite call.
        let mut realized: Vec<Realized> = Vec::with_capacity(scene.layers.len());
        let mut effect_store: Vec<EffectChain> = Vec::new();
        let mut occurrence: HashMap<MediaId, u32> = HashMap::new();
        for layer in &scene.layers {
            let fx = layer_effects(layer);
            let color_grade = layer.color_grade;
            // Load (or recall) the layer's .cube table; unreadable files
            // resolve to None and grade nothing.
            let scene_lut = self.resolve_scene_lut(&layer.lut);
            // The layer carries the anchor position; the quad center falls out
            // of the final pixel size (bitmap sizes only exist after raster).
            let place = |size: [f32; 2]| LayerPlacement {
                center: layer.quad_center(size),
                size,
                rotation: layer.rotation,
                opacity: layer.opacity,
            };
            match &layer.source {
                LayerSource::CanvasPass => {
                    realized.push(Realized::CanvasPass {
                        effects: layer.effects.clone(),
                        grade: color_grade,
                        lut: scene_lut,
                    });
                }
                LayerSource::Transition {
                    outgoing,
                    incoming,
                    transition_id,
                    progress,
                } => {
                    let out = self.realize_subscene_layer(project, scene, outgoing, policy)?;
                    let inc = self.realize_subscene_layer(project, scene, incoming, policy)?;
                    realized.push(Realized::Transition {
                        outgoing: out,
                        incoming: inc,
                        transition_id: transition_id.clone(),
                        progress: *progress,
                    });
                }
                LayerSource::Solid(rgba) => {
                    let size = fixed_size(layer.size, [scene.width as f32, scene.height as f32]);
                    realized.push(Realized::Solid {
                        rgba: *rgba,
                        placement: place(size),
                        effects: layer.effects.clone(),
                        fx,
                        color_grade,
                        lut: scene_lut,
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
                        effects: layer.effects.clone(),
                        fx,
                        color_grade,
                        lut: scene_lut,
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
                        effects: layer.effects.clone(),
                        fx,
                        color_grade,
                        lut: scene_lut,
                    });
                }
                LayerSource::Still { media } => {
                    self.ensure_still(project, *media)?;
                    let size = fixed_size(layer.size, [scene.width as f32, scene.height as f32]);
                    realized.push(Realized::Still {
                        media: *media,
                        placement: place(size),
                        uv: layer.uv,
                        effects: layer.effects.clone(),
                        fx,
                        color_grade,
                        lut: scene_lut,
                    });
                }
                LayerSource::Lottie { path, local_time } => {
                    // A missing or unsupported file draws nothing (the media
                    // offline story — projects move machines), never an error.
                    let Some(frame_index) = self.ensure_lottie_frame(path, *local_time) else {
                        continue;
                    };
                    let size = fixed_size(layer.size, [scene.width as f32, scene.height as f32]);
                    realized.push(Realized::Lottie {
                        path: path.clone(),
                        frame_index,
                        placement: place(size),
                        uv: layer.uv,
                        effects: layer.effects.clone(),
                        fx,
                        color_grade,
                        lut: scene_lut,
                    });
                }
                LayerSource::Sticker { asset, local_time } => {
                    // The resolver only emits catalog ids, but stay graceful:
                    // an unknown id draws nothing rather than failing a frame.
                    let Some(spec) = cutlass_models::sticker_spec(asset) else {
                        continue;
                    };
                    self.ensure_sticker(spec)?;
                    let frame_index = self.stickers[spec.id].frame_at(*local_time);
                    let size = fixed_size(layer.size, [scene.width as f32, scene.height as f32]);
                    realized.push(Realized::Sticker {
                        asset: asset.clone(),
                        frame_index,
                        placement: place(size),
                        uv: layer.uv,
                        effects: layer.effects.clone(),
                        fx,
                        color_grade,
                        lut: scene_lut,
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
                        effects: layer.effects.clone(),
                        fx,
                        color_grade,
                        lut: scene_lut,
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
                        effects: layer.effects.clone(),
                        fx,
                        color_grade,
                        lut: scene_lut,
                    });
                }
            }
        }

        // Pack effect chains and build compositor layers with stable borrows.
        for r in &realized {
            if let Some(effects) = r.effects().filter(|e| !e.is_empty()) {
                effect_store.push(pack_effects(effects));
            }
        }
        let instance_store: Vec<Vec<PassInstance<'_>>> =
            effect_store.iter().map(EffectChain::instances).collect();

        let mut effect_idx = 0usize;
        let mut layer_storage: Vec<CompositeLayer<'_>> = Vec::new();
        // Phase 1: build all composite layers (indices only for transitions).
        enum LayerJob<'a> {
            Plain {
                storage_idx: usize,
            },
            CanvasPass {
                effects: &'a [PassInstance<'a>],
                grade: Option<ColorGrade>,
                lut: &'a Option<SceneLut>,
            },
            Transition {
                out_idx: usize,
                in_idx: usize,
                transition_id: &'a str,
                progress: f32,
            },
        }
        let mut jobs: Vec<LayerJob<'_>> = Vec::new();

        for r in &realized {
            match r {
                Realized::CanvasPass {
                    effects,
                    grade,
                    lut,
                } => {
                    let effects = if effects.is_empty() {
                        &[]
                    } else {
                        let chain = &instance_store[effect_idx];
                        effect_idx += 1;
                        chain.as_slice()
                    };
                    jobs.push(LayerJob::CanvasPass {
                        effects,
                        grade: *grade,
                        lut,
                    });
                }
                Realized::Transition {
                    outgoing,
                    incoming,
                    transition_id,
                    progress,
                } => {
                    let out_effects = outgoing
                        .effects()
                        .filter(|e| !e.is_empty())
                        .map(|_| {
                            let chain = &instance_store[effect_idx];
                            effect_idx += 1;
                            chain.as_slice()
                        })
                        .unwrap_or(&[]);
                    layer_storage.push(composite_from_realized(
                        outgoing.as_ref(),
                        &self.stills,
                        &self.stickers,
                        &self.lottie,
                        &self.luts,
                        out_effects,
                    ));
                    let out_idx = layer_storage.len() - 1;
                    let in_effects = incoming
                        .effects()
                        .filter(|e| !e.is_empty())
                        .map(|_| {
                            let chain = &instance_store[effect_idx];
                            effect_idx += 1;
                            chain.as_slice()
                        })
                        .unwrap_or(&[]);
                    layer_storage.push(composite_from_realized(
                        incoming.as_ref(),
                        &self.stills,
                        &self.stickers,
                        &self.lottie,
                        &self.luts,
                        in_effects,
                    ));
                    let in_idx = layer_storage.len() - 1;
                    jobs.push(LayerJob::Transition {
                        out_idx,
                        in_idx,
                        transition_id: transition_id.as_str(),
                        progress: *progress,
                    });
                }
                other => {
                    let effects = other
                        .effects()
                        .filter(|e| !e.is_empty())
                        .map(|_| {
                            let chain = &instance_store[effect_idx];
                            effect_idx += 1;
                            chain.as_slice()
                        })
                        .unwrap_or(&[]);
                    layer_storage.push(composite_from_realized(
                        other,
                        &self.stills,
                        &self.stickers,
                        &self.lottie,
                        &self.luts,
                        effects,
                    ));
                    jobs.push(LayerJob::Plain {
                        storage_idx: layer_storage.len() - 1,
                    });
                }
            }
        }

        // Phase 2: borrow storage immutably for compositor dispatch.
        let compositor_layers: Vec<CompositorLayer<'_>> = jobs
            .iter()
            .map(|job| match job {
                LayerJob::Plain { storage_idx } => {
                    CompositorLayer::layer(&layer_storage[*storage_idx])
                }
                LayerJob::CanvasPass {
                    effects,
                    grade,
                    lut,
                } => CompositorLayer::CanvasPass {
                    effects,
                    grade: *grade,
                    lut: layer_lut(lut, &self.luts),
                },
                LayerJob::Transition {
                    out_idx,
                    in_idx,
                    transition_id,
                    progress,
                } => CompositorLayer::Transition {
                    outgoing: &layer_storage[*out_idx],
                    incoming: &layer_storage[*in_idx],
                    transition_id,
                    progress: *progress,
                },
            })
            .collect();

        let config =
            CompositorConfig::new(scene.width, scene.height).with_background(scene.background);
        let realize_ms = realize_started.elapsed().as_secs_f64() * 1000.0;
        let composite_started = Instant::now();
        self.compositor.render_compositor_layers_into(
            &self.gpu,
            &config,
            &compositor_layers,
            sink,
        )?;

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
                layers = compositor_layers.len(),
                width = scene.width,
                height = scene.height,
                "slow frame render: {total_ms:.0} ms"
            );
        } else {
            tracing::trace!(
                decode_ms = %format_args!("{decode_ms:.1}"),
                raster_ms = %format_args!("{raster_ms:.1}"),
                composite_ms = %format_args!("{composite_ms:.1}"),
                layers = compositor_layers.len(),
                "frame render: {total_ms:.1} ms"
            );
        }
        Ok(())
    }

    fn realize_subscene_layer(
        &mut self,
        project: &Project,
        scene: &Scene,
        layer: &crate::scene::SceneLayer,
        policy: SeekPolicy,
    ) -> Result<Box<Realized>, RenderError> {
        let place = |size: [f32; 2]| LayerPlacement {
            center: layer.quad_center(size),
            size,
            rotation: layer.rotation,
            opacity: layer.opacity,
        };
        let fx = layer_effects(layer);
        let color_grade = layer.color_grade;
        let realized = match &layer.source {
            LayerSource::Solid(rgba) => {
                let size = fixed_size(layer.size, [scene.width as f32, scene.height as f32]);
                Realized::Solid {
                    rgba: *rgba,
                    placement: place(size),
                    effects: layer.effects.clone(),
                    fx,
                    color_grade,
                    lut: None,
                }
            }
            LayerSource::Text { content, style } => {
                let image = self.text.rasterize(content, style);
                if image.width == 0 || image.height == 0 {
                    return Err(RenderError::unsupported("empty text layer"));
                }
                let scale = match layer.size {
                    SizeSpec::BitmapScaled(s) => s,
                    SizeSpec::Fixed(_) => 1.0,
                };
                let size = [image.width as f32 * scale, image.height as f32 * scale];
                Realized::Bitmap {
                    image,
                    placement: place(size),
                    uv: layer.uv,
                    effects: layer.effects.clone(),
                    fx,
                    color_grade,
                    lut: None,
                }
            }
            LayerSource::Media { media, source_time } => {
                let frame = self.decode(project, *media, 0, *source_time, policy)?;
                let size = fixed_size(layer.size, [scene.width as f32, scene.height as f32]);
                Realized::Frame {
                    frame,
                    placement: place(size),
                    uv: layer.uv,
                    effects: layer.effects.clone(),
                    fx,
                    color_grade,
                    lut: None,
                }
            }
            LayerSource::Still { media } => {
                self.ensure_still(project, *media)?;
                let size = fixed_size(layer.size, [scene.width as f32, scene.height as f32]);
                Realized::Still {
                    media: *media,
                    placement: place(size),
                    uv: layer.uv,
                    effects: layer.effects.clone(),
                    fx,
                    color_grade,
                    lut: None,
                }
            }
            LayerSource::Lottie { path, local_time } => {
                let size = fixed_size(layer.size, [scene.width as f32, scene.height as f32]);
                match self.ensure_lottie_frame(path, *local_time) {
                    Some(frame_index) => Realized::Lottie {
                        path: path.clone(),
                        frame_index,
                        placement: place(size),
                        uv: layer.uv,
                        effects: layer.effects.clone(),
                        fx,
                        color_grade,
                        lut: None,
                    },
                    // Missing file inside a transition: a transparent side,
                    // matching the draw-nothing policy of the main path.
                    None => Realized::Solid {
                        rgba: [0, 0, 0, 0],
                        placement: place(size),
                        effects: layer.effects.clone(),
                        fx,
                        color_grade,
                        lut: None,
                    },
                }
            }
            LayerSource::Sticker { asset, local_time } => {
                let spec = cutlass_models::sticker_spec(asset)
                    .ok_or_else(|| RenderError::unsupported("unknown sticker asset"))?;
                self.ensure_sticker(spec)?;
                let frame_index = self.stickers[spec.id].frame_at(*local_time);
                let size = fixed_size(layer.size, [scene.width as f32, scene.height as f32]);
                Realized::Sticker {
                    asset: asset.clone(),
                    frame_index,
                    placement: place(size),
                    uv: layer.uv,
                    effects: layer.effects.clone(),
                    fx,
                    color_grade,
                    lut: None,
                }
            }
            LayerSource::Shape {
                params,
                fill,
                stroke,
                pad,
            } => {
                let size = fixed_size(layer.size, [scene.width as f32, scene.height as f32]);
                let half = [
                    (size[0] * 0.5 - pad).max(0.0),
                    (size[1] * 0.5 - pad).max(0.0),
                ];
                Realized::Sdf {
                    shape: SdfLayer {
                        shape: params.with_half(half),
                        fill: *fill,
                        stroke: *stroke,
                    },
                    placement: place(size),
                    effects: layer.effects.clone(),
                    fx,
                    color_grade,
                    lut: None,
                }
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
                    return Err(RenderError::unsupported("degenerate path layer"));
                }
                let scale = match layer.size {
                    SizeSpec::BitmapScaled(s) => s,
                    SizeSpec::Fixed(_) => 1.0,
                };
                let size = [image.width as f32 * scale, image.height as f32 * scale];
                Realized::Bitmap {
                    image,
                    placement: place(size),
                    uv: layer.uv,
                    effects: layer.effects.clone(),
                    fx,
                    color_grade,
                    lut: None,
                }
            }
            LayerSource::Transition { .. } => {
                return Err(RenderError::unsupported("nested transitions"));
            }
            LayerSource::CanvasPass => {
                return Err(RenderError::unsupported("nested canvas pass"));
            }
        };
        Ok(Box::new(realized))
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
        // An exact target at the media's very end can overshoot EOF: the
        // pool trusts the container's frame count, which routinely
        // over-reports by one, and proxies additionally run a frame short
        // by construction (generation drops that untrustworthy tail). Show
        // the nearest decodable frame instead of failing the whole
        // composite — transitions pin the outgoing side at the final
        // source frame, so a miss here would error every frame of the
        // window.
        if frame.is_none() {
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
            None,
            None,
            canvas_w as f32,
            canvas_h as f32,
            1.0,
            tick,
            // Sizing doesn't depend on the animation clock.
            0.0,
            Vec::new(),
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

    /// Decode a bundled sticker's whole frame sequence into the cache on
    /// first use (mirrors [`ensure_still`](Self::ensure_still)).
    fn ensure_sticker(&mut self, spec: &cutlass_models::StickerSpec) -> Result<(), RenderError> {
        if self.stickers.contains_key(spec.id) {
            return Ok(());
        }
        let decoded = cutlass_decoder::decode_animation(spec.bytes)?;
        let delays_ms: Vec<u32> = decoded.iter().map(|f| f.delay_ms).collect();
        let total_ms = delays_ms.iter().map(|d| u64::from(*d)).sum();
        self.stickers.insert(
            spec.id.to_owned(),
            StickerSequence {
                frames: decoded.into_iter().map(|f| f.image).collect(),
                delays_ms,
                total_ms,
            },
        );
        Ok(())
    }

    /// Make the Lottie frame for `local_time` resident (parse the file on
    /// first sight, rasterize the frame unless cached) and return its
    /// sampled-frame index. `None` — draw nothing — for files that are
    /// missing, unparseable, or fail to rasterize; the failure is
    /// remembered and logged once, not re-probed per frame.
    fn ensure_lottie_frame(&mut self, path: &str, local_time: f64) -> Option<usize> {
        let stamp = self.lottie_stamp;
        let state = self.lottie.entry(path.to_owned()).or_insert_with(|| {
            match cutlass_decoder::LottieAnimation::load(std::path::Path::new(path)) {
                Ok(animation) => LottieState::Loaded(Box::new(LottiePlayer {
                    animation,
                    frames: HashMap::new(),
                })),
                Err(e) => {
                    tracing::warn!("lottie '{path}' failed to load: {e}");
                    LottieState::Failed
                }
            }
        });
        let LottieState::Loaded(player) = state else {
            return None;
        };

        let index = player.animation.frame_index_at(local_time);
        if let Some((_, used)) = player.frames.get_mut(&index) {
            *used = stamp;
            return Some(index);
        }
        let image = match player.animation.render_frame(index) {
            Ok(image) => image,
            Err(e) => {
                tracing::warn!("lottie '{path}' frame {index} failed to render: {e}");
                return None;
            }
        };
        player.frames.insert(index, (image, stamp));

        // Enforce the per-asset byte budget, oldest-stamp first. Frames
        // stamped by the current scene are exempt (still borrowed below).
        let mut total: usize = player.frames.values().map(|(f, _)| f.pixels.len()).sum();
        while total > LOTTIE_CACHE_BYTES {
            let Some((&victim, _)) = player
                .frames
                .iter()
                .filter(|(_, (_, used))| *used != stamp)
                .min_by_key(|(_, (_, used))| *used)
            else {
                break;
            };
            let (evicted, _) = player.frames.remove(&victim).expect("victim exists");
            total -= evicted.pixels.len();
        }
        Some(index)
    }

    /// Resolve a layer's `.cube` LUT reference: parse and cache the file on
    /// first sight, and return the reference only when the table is loadable.
    /// Missing/unparseable files log once and grade nothing — the media
    /// offline story, never an error.
    fn resolve_scene_lut(&mut self, lut: &Option<SceneLut>) -> Option<SceneLut> {
        let lut = lut.as_ref()?;
        let state = self
            .luts
            .entry(lut.path.clone())
            .or_insert_with(|| match std::fs::read_to_string(&lut.path) {
                Ok(text) => match CubeLut::parse(&text) {
                    Ok(cube) => CubeLutState::Loaded(Box::new(cube)),
                    Err(e) => {
                        tracing::warn!("LUT '{}' failed to parse: {e}", lut.path);
                        CubeLutState::Failed
                    }
                },
                Err(e) => {
                    tracing::warn!("LUT '{}' failed to read: {e}", lut.path);
                    CubeLutState::Failed
                }
            });
        matches!(state, CubeLutState::Loaded(_)).then(|| lut.clone())
    }
}

/// A `.cube` path the renderer has seen: parsed, or failed (missing or
/// malformed file — grades nothing, logged once at load).
enum CubeLutState {
    Loaded(Box<CubeLut>),
    Failed,
}

/// Borrow the parsed table behind a realized LUT reference for the
/// compositor's [`LayerLut`] (keyed uploads; see `cutlass-compositor`).
/// Realize only emits references [`Renderer::resolve_scene_lut`] loaded.
fn layer_lut<'a>(
    lut: &'a Option<SceneLut>,
    luts: &'a HashMap<String, CubeLutState>,
) -> Option<LayerLut<'a>> {
    let lut = lut.as_ref()?;
    let CubeLutState::Loaded(cube) = &luts[&lut.path] else {
        unreachable!("realized LUT reference without a loaded table")
    };
    Some(LayerLut {
        key: &lut.path,
        lut: cube,
        intensity: lut.intensity,
    })
}

/// Per-asset Lottie frame-cache budget. At the 512 px render cap
/// (~1 MB/frame) this holds ~32 frames — a full loop of a typical short
/// sticker, so steady-state playback rasterizes nothing.
const LOTTIE_CACHE_BYTES: usize = 32 << 20;

/// A Lottie path the renderer has seen: parsed and playable, or failed
/// (missing/unsupported file — draws nothing, logged once at load).
enum LottieState {
    Loaded(Box<LottiePlayer>),
    Failed,
}

struct LottiePlayer {
    animation: cutlass_decoder::LottieAnimation,
    /// Rasterized frames keyed by sampled-frame index, stamped with the
    /// scene counter of their last use (the LRU key).
    frames: HashMap<usize, (RgbaImage, u64)>,
}

/// A decoded sticker asset: every frame up front plus per-frame delays, so
/// per-composite frame selection is a table walk instead of a decode.
struct StickerSequence {
    frames: Vec<RgbaImage>,
    /// Display duration per frame in milliseconds (parallel to `frames`).
    delays_ms: Vec<u32>,
    /// Sum of `delays_ms`.
    total_ms: u64,
}

impl StickerSequence {
    /// Index of the frame on screen at `local_time` seconds, looping over
    /// the sequence. Static stickers (one frame) always show frame 0.
    fn frame_at(&self, local_time: f64) -> usize {
        if self.frames.len() <= 1 || self.total_ms == 0 {
            return 0;
        }
        let ms = (local_time.max(0.0) * 1000.0) as u64 % self.total_ms;
        let mut acc = 0u64;
        for (index, delay) in self.delays_ms.iter().enumerate() {
            acc += u64::from(*delay);
            if ms < acc {
                return index;
            }
        }
        self.frames.len() - 1
    }
}

/// Packed effect chain: catalog-static ids plus owned parameter values.
///
/// [`PassInstance`] wants a `&'static str` id and borrowed params. Ids are
/// interned against the compositor's static effect catalog (unknown ids are
/// dropped here — they'd dispatch as no-op passthroughs anyway), and params
/// stay owned so the instances built by [`EffectChain::instances`] borrow from
/// this store for the duration of one render instead of leaking.
struct EffectChain {
    passes: Vec<(&'static str, Vec<f32>)>,
}

impl EffectChain {
    fn instances(&self) -> Vec<PassInstance<'_>> {
        self.passes
            .iter()
            .map(|(id, params)| PassInstance { id, params })
            .collect()
    }
}

fn pack_effects(resolved: &[ResolvedPass]) -> EffectChain {
    let passes = resolved
        .iter()
        .filter_map(|pass| {
            let id = cutlass_compositor::effect_descriptors()
                .iter()
                .find(|d| d.id == pass.id)?
                .id;
            Some((id, pass.params.clone()))
        })
        .collect();
    EffectChain { passes }
}

fn composite_from_realized<'a>(
    r: &'a Realized,
    stills: &'a HashMap<MediaId, RgbaImage>,
    stickers: &'a HashMap<String, StickerSequence>,
    lottie: &'a HashMap<String, LottieState>,
    luts: &'a HashMap<String, CubeLutState>,
    effects: &'a [PassInstance<'a>],
) -> CompositeLayer<'a> {
    match r {
        Realized::Solid {
            rgba,
            placement,
            fx,
            color_grade,
            lut,
            ..
        } => CompositeLayer::solid(*rgba, *placement)
            .with_fx(*fx)
            .with_effects(effects)
            .with_color_grade(*color_grade)
            .with_lut(layer_lut(lut, luts)),
        Realized::Bitmap {
            image,
            placement,
            uv,
            fx,
            color_grade,
            lut,
            ..
        } => CompositeLayer::rgba(image, *placement)
            .with_uv(*uv)
            .with_fx(*fx)
            .with_effects(effects)
            .with_color_grade(*color_grade)
            .with_lut(layer_lut(lut, luts)),
        Realized::Frame {
            frame,
            placement,
            uv,
            fx,
            color_grade,
            lut,
            ..
        } => CompositeLayer::frame(frame, *placement)
            .with_uv(*uv)
            .with_fx(*fx)
            .with_effects(effects)
            .with_color_grade(*color_grade)
            .with_lut(layer_lut(lut, luts)),
        Realized::Still {
            media,
            placement,
            uv,
            fx,
            color_grade,
            lut,
            ..
        } => CompositeLayer::rgba(&stills[media], *placement)
            .with_uv(*uv)
            .with_fx(*fx)
            .with_effects(effects)
            .with_color_grade(*color_grade)
            .with_lut(layer_lut(lut, luts)),
        Realized::Sticker {
            asset,
            frame_index,
            placement,
            uv,
            fx,
            color_grade,
            lut,
            ..
        } => CompositeLayer::rgba(&stickers[asset].frames[*frame_index], *placement)
            .with_uv(*uv)
            .with_fx(*fx)
            .with_effects(effects)
            .with_color_grade(*color_grade)
            .with_lut(layer_lut(lut, luts)),
        Realized::Lottie {
            path,
            frame_index,
            placement,
            uv,
            fx,
            color_grade,
            lut,
            ..
        } => {
            // Realize only emits `Realized::Lottie` after `ensure_lottie_frame`
            // cached this exact frame, and the LRU never evicts frames stamped
            // by the scene being composed.
            let LottieState::Loaded(player) = &lottie[path] else {
                unreachable!("realized lottie layer without a loaded player")
            };
            CompositeLayer::rgba(&player.frames[frame_index].0, *placement)
                .with_uv(*uv)
                .with_fx(*fx)
                .with_effects(effects)
                .with_color_grade(*color_grade)
                .with_lut(layer_lut(lut, luts))
        }
        Realized::Sdf {
            shape,
            placement,
            fx,
            color_grade,
            lut,
            ..
        } => CompositeLayer::sdf(*shape, *placement)
            .with_fx(*fx)
            .with_effects(effects)
            .with_color_grade(*color_grade)
            .with_lut(layer_lut(lut, luts)),
        Realized::Transition { .. } | Realized::CanvasPass { .. } => {
            unreachable!("non-layer realized items handled separately")
        }
    }
}

/// An owned, decoded/rasterized layer kept alive while the compositor borrows it.
enum Realized {
    Transition {
        outgoing: Box<Realized>,
        incoming: Box<Realized>,
        transition_id: String,
        progress: f32,
    },
    CanvasPass {
        effects: Vec<ResolvedPass>,
        grade: Option<ColorGrade>,
        lut: Option<SceneLut>,
    },
    Frame {
        frame: VideoFrame,
        placement: LayerPlacement,
        uv: [f32; 4],
        effects: Vec<ResolvedPass>,
        fx: LayerEffects,
        color_grade: Option<ColorGrade>,
        lut: Option<SceneLut>,
    },
    Still {
        media: MediaId,
        placement: LayerPlacement,
        uv: [f32; 4],
        effects: Vec<ResolvedPass>,
        fx: LayerEffects,
        color_grade: Option<ColorGrade>,
        lut: Option<SceneLut>,
    },
    Sticker {
        asset: String,
        frame_index: usize,
        placement: LayerPlacement,
        uv: [f32; 4],
        effects: Vec<ResolvedPass>,
        fx: LayerEffects,
        color_grade: Option<ColorGrade>,
        lut: Option<SceneLut>,
    },
    Lottie {
        path: String,
        frame_index: usize,
        placement: LayerPlacement,
        uv: [f32; 4],
        effects: Vec<ResolvedPass>,
        fx: LayerEffects,
        color_grade: Option<ColorGrade>,
        lut: Option<SceneLut>,
    },
    Bitmap {
        image: RgbaImage,
        placement: LayerPlacement,
        uv: [f32; 4],
        effects: Vec<ResolvedPass>,
        fx: LayerEffects,
        color_grade: Option<ColorGrade>,
        lut: Option<SceneLut>,
    },
    Solid {
        rgba: [u8; 4],
        placement: LayerPlacement,
        effects: Vec<ResolvedPass>,
        fx: LayerEffects,
        color_grade: Option<ColorGrade>,
        lut: Option<SceneLut>,
    },
    Sdf {
        shape: SdfLayer,
        placement: LayerPlacement,
        effects: Vec<ResolvedPass>,
        fx: LayerEffects,
        color_grade: Option<ColorGrade>,
        lut: Option<SceneLut>,
    },
}

impl Realized {
    fn effects(&self) -> Option<&[ResolvedPass]> {
        match self {
            Realized::Transition { .. } => None,
            Realized::CanvasPass { effects, .. } => Some(effects),
            Realized::Frame { effects, .. }
            | Realized::Still { effects, .. }
            | Realized::Sticker { effects, .. }
            | Realized::Lottie { effects, .. }
            | Realized::Bitmap { effects, .. }
            | Realized::Solid { effects, .. }
            | Realized::Sdf { effects, .. } => Some(effects),
        }
    }
}

fn layer_effects(layer: &crate::scene::SceneLayer) -> LayerEffects {
    let mask = layer.mask.map(|m| LayerMask {
        kind: mask_kind_id(m.kind),
        feather: m.feather,
        invert: u32::from(m.invert),
    });
    let chroma_key = layer.chroma_key.map(|c| LayerChromaKey {
        rgb: [
            f32::from(c.rgb[0]) / 255.0,
            f32::from(c.rgb[1]) / 255.0,
            f32::from(c.rgb[2]) / 255.0,
        ],
        strength: c.strength,
        shadow: c.shadow,
    });
    LayerEffects { mask, chroma_key }
}

fn mask_kind_id(kind: MaskKind) -> u32 {
    match kind {
        MaskKind::Linear => mask_kind::LINEAR,
        MaskKind::Mirror => mask_kind::MIRROR,
        MaskKind::Circle => mask_kind::CIRCLE,
        MaskKind::Rectangle => mask_kind::RECTANGLE,
        MaskKind::Heart => mask_kind::HEART,
        MaskKind::Star => mask_kind::STAR,
    }
}

/// The on-canvas size for a non-text layer, falling back to the canvas if a
/// bitmap-scaled spec ever reaches here (it shouldn't for media/solid).
fn fixed_size(size: SizeSpec, canvas: [f32; 2]) -> [f32; 2] {
    match size {
        SizeSpec::Fixed(s) => s,
        SizeSpec::BitmapScaled(_) => canvas,
    }
}

/// Three fit-sized RGBA images for a zero-drift preview transform gesture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GestureFrames {
    pub below: RgbaImage,
    pub sprite: RgbaImage,
    pub above: Option<RgbaImage>,
}

/// Convert premultiplied RGBA readbacks into straight alpha for Slint.
fn straighten_alpha(image: &mut RgbaImage) {
    for chunk in image.pixels.chunks_exact_mut(4) {
        let a = chunk[3];
        if a == 0 || a == 255 {
            continue;
        }
        let af = f32::from(a) / 255.0;
        for c in &mut chunk[..3] {
            *c = ((*c as f32 / af).round() as u32).min(255) as u8;
        }
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

#[cfg(test)]
mod tests {
    use super::StickerSequence;
    use cutlass_core::RgbaImage;

    fn seq(delays_ms: &[u32]) -> StickerSequence {
        StickerSequence {
            frames: delays_ms
                .iter()
                .map(|_| RgbaImage::new(1, 1, vec![0; 4]))
                .collect(),
            delays_ms: delays_ms.to_vec(),
            total_ms: delays_ms.iter().map(|d| u64::from(*d)).sum(),
        }
    }

    #[test]
    fn sticker_frame_selection_walks_delays_and_loops() {
        let s = seq(&[100, 50, 100]);
        assert_eq!(s.frame_at(0.0), 0);
        assert_eq!(s.frame_at(0.099), 0);
        assert_eq!(s.frame_at(0.100), 1);
        assert_eq!(s.frame_at(0.149), 1);
        assert_eq!(s.frame_at(0.150), 2);
        // Loops at total (250 ms) and clamps negatives to the first frame.
        assert_eq!(s.frame_at(0.250), 0);
        assert_eq!(s.frame_at(0.601), 1);
        assert_eq!(s.frame_at(-1.0), 0);
    }

    #[test]
    fn static_stickers_always_show_frame_zero() {
        let s = seq(&[100]);
        assert_eq!(s.frame_at(12.34), 0);
    }
}
