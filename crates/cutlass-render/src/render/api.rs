use std::collections::HashMap;
use std::path::{Path, PathBuf};

use cutlass_compositor::{
    Compositor, CompositorError, FrameSink, GpuContext, ImageSink, RgbaImage,
};
use cutlass_core::RationalTime;
use cutlass_decoder::OutputMode;
use cutlass_models::{ClipId, MediaId, Project};
use cutlass_shapes::{PathRaster, ShapeStyle};
use cutlass_text::TextRenderer;

use crate::error::RenderError;
use crate::resolve::{ResolveOverrides, resolve, resolve_gesture_partitions, resolve_with};
use crate::scene::{LayerSource, Scene, SizeSpec};

use super::media_cache::default_decode_mode;
use super::{FrameStats, GestureFrames, Renderer, SeekPolicy};

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

    /// Remove every proxy substitution while preserving original-media
    /// decoders. Use [`reset_media_sources`](Self::reset_media_sources) when
    /// the project/media-id namespace itself changes.
    pub fn clear_proxies(&mut self) {
        let stale: Vec<MediaId> = self.proxies.keys().copied().collect();
        self.proxies.clear();
        for media in stale {
            self.drop_decoders_for(media);
        }
    }

    /// Drop every cache keyed by the current project's media-id namespace.
    ///
    /// Call this before reusing one renderer with a different project or a
    /// relinked media catalog. Media ids are persisted per project, so the
    /// same numeric id can name a different file after a session switch;
    /// retaining its decoder or still bitmap would render the old asset.
    /// Path-keyed and bundled-asset caches remain warm.
    pub fn reset_media_sources(&mut self) {
        self.decoders.clear();
        self.stills.clear();
        self.proxies.clear();
    }

    /// Invalidate one media id after its source path or probed metadata
    /// changes. The next frame reopens/redecodes it from the project.
    pub fn invalidate_media_source(&mut self, media: MediaId) {
        self.proxies.remove(&media);
        self.drop_decoders_for(media);
        self.stills.remove(&media);
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
                    LayerSource::Text { content, style, .. } => self.text.rasterize(content, style),
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
