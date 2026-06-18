//! Timeline layer resolution and decode helpers for GPU compositing.

use std::time::Instant;

use cutlass_cache::{FrameCache, SourceFingerprint};
use cutlass_compositor::{CompositeLayer, CompositorConfig, LayerEffect, LayerPlacement};
use cutlass_decoder::DecodedFrame;
use cutlass_models::{
    Clip, ClipId, ClipTransform, CropRect, Generator, ModelError, Project, RationalTime, TrackKind,
};
use tracing::debug;

use crate::ColorConvertPath;
use crate::decoder_pool::DecoderPool;
use crate::error::EngineError;
use crate::frame::{RgbaFrame, decoded_to_yuv_layer, legacy_decoded_to_rgba};
use crate::generator_raster::GeneratorRaster;

const DEFAULT_WIDTH: u32 = 1920;
const DEFAULT_HEIGHT: u32 = 1080;

/// Output canvas size, honoring the project's canvas aspect preset (M1
/// canvas settings).
///
/// The *base* size is the max *video* media dimensions on the timeline, or
/// 1920×1080 fallback. Stills don't vote: a 12MP photo must not balloon the
/// canvas (and the encode) past what the footage calls for — it aspect-fits
/// like any other layer.
///
/// - `Auto` (default): the base size as-is — the pre-canvas-settings
///   behavior.
/// - A fixed ratio: the canvas short edge keeps the base's short edge (the
///   footage's quality tier survives a ratio change — 4K stays 4K-class
///   when flipped to 9:16), the long edge follows the ratio.
///
/// Width and height are forced even for downstream H.264 export.
pub fn composite_canvas_size(project: &Project) -> (u32, u32) {
    let mut max_w = 0u32;
    let mut max_h = 0u32;

    for track in project.timeline().tracks_ordered() {
        if track.kind != TrackKind::Video {
            continue;
        }
        for clip in track.clips() {
            if let Some(media_id) = clip.media()
                && let Some(media) = project.media(media_id)
                && !media.is_image
            {
                max_w = max_w.max(media.width);
                max_h = max_h.max(media.height);
            }
        }
    }

    let (base_w, base_h) = if max_w == 0 || max_h == 0 {
        (DEFAULT_WIDTH, DEFAULT_HEIGHT)
    } else {
        (max_w, max_h)
    };

    match project.timeline().canvas().aspect.ratio() {
        None => (to_even(base_w), to_even(base_h)),
        Some((rw, rh)) => {
            let tier = u64::from(base_w.min(base_h));
            let (rw, rh) = (u64::from(rw), u64::from(rh));
            let (w, h) = if rw >= rh {
                (tier * rw / rh, tier)
            } else {
                (tier, tier * rh / rw)
            };
            (to_even(w as u32), to_even(h as u32))
        }
    }
}

/// The full compositor canvas for a project: derived size plus the
/// project's background color. Export builds its pass config here; preview
/// goes through [`preview_canvas_config`], which scales this down.
pub fn composite_canvas_config(project: &Project) -> CompositorConfig {
    let (width, height) = composite_canvas_size(project);
    CompositorConfig::new(width, height).with_background(project.timeline().canvas().background)
}

/// Preview render-resolution knob — the single source of truth until the UI
/// drives it. Flip it; everything downstream of decode follows.
///
/// `Some(h)` caps the preview pipeline to height `h`: a freshly decoded frame
/// is downscaled to this height *before* it enters the frame cache, so the
/// cache stores the smaller frame, the compositor reads and composites it, and
/// the canvas / RGBA readback / UI upload all shrink with it (~9× fewer bytes
/// at 720p from 4K). Decode itself still runs at native resolution — the codec
/// has no cheaper option — and export ignores this entirely (always full
/// source, see [`composite_canvas_config`] + a `None` cache).
///
/// Flipping it is cache-safe: [`import_media`](crate::import_media) registers
/// each source's [`CacheSpec`](cutlass_cache::CacheSpec) at the scaled dims, so
/// a changed height is a changed spec and the stale on-disk index is dropped.
///
/// `None` (or `0`) means full-resolution preview (the legacy behavior).
pub const PREVIEW_MAX_HEIGHT: Option<u32> = Some(720);

/// Scale `width × height` down so its height is at most [`PREVIEW_MAX_HEIGHT`],
/// preserving aspect ratio and forcing even dimensions (YUV420P / H.264 need
/// even). Sizes already within the cap (or no cap) pass through unchanged; the
/// result never goes below 2 and never upscales.
pub fn preview_scaled_dims(width: u32, height: u32) -> (u32, u32) {
    scale_to_max_height(width, height, PREVIEW_MAX_HEIGHT)
}

/// Core of [`preview_scaled_dims`], parameterized on the cap for testing.
fn scale_to_max_height(width: u32, height: u32, max_height: Option<u32>) -> (u32, u32) {
    let Some(cap) = max_height.filter(|c| *c > 0) else {
        return (to_even(width), to_even(height));
    };
    if height <= cap {
        return (to_even(width), to_even(height));
    }
    let scale = f64::from(cap) / f64::from(height);
    let w = ((f64::from(width) * scale).round() as u32).max(2);
    (to_even(w), to_even(cap.max(2)))
}

/// Preview canvas size: [`composite_canvas_size`] capped by
/// [`PREVIEW_MAX_HEIGHT`]. The source layers are downscaled to match (see
/// [`decode_media_frame`]), so the compositor works end-to-end at this size.
pub fn preview_canvas_size(project: &Project) -> (u32, u32) {
    let (width, height) = composite_canvas_size(project);
    preview_scaled_dims(width, height)
}

/// Preview compositor canvas: [`composite_canvas_config`] scaled to the
/// preview resolution. Export uses [`composite_canvas_config`] (full source).
pub fn preview_canvas_config(project: &Project) -> CompositorConfig {
    let (width, height) = preview_canvas_size(project);
    CompositorConfig::new(width, height).with_background(project.timeline().canvas().background)
}

/// Canvas placement for content of `content_w × content_h` under a clip
/// transform (CapCut semantics: scale 1.0 aspect-fits the content inside the
/// canvas, centered; position offsets are normalized to canvas dimensions;
/// rotation is degrees clockwise about the anchor).
///
/// This is *the* geometry: the compositor draws it, and preview hit-testing
/// (preview roadmap Phase 2) inverts it — the two can never disagree.
pub fn layer_placement(
    transform: &ClipTransform,
    content_w: u32,
    content_h: u32,
    canvas: &CompositorConfig,
) -> LayerPlacement {
    cropped_layer_placement(transform, content_w, content_h, &CropRect::FULL, canvas)
}

/// [`layer_placement`] for cropped content (CapCut crop, M1): the kept
/// region is the content — it aspect-fits the canvas at scale 1.0 and
/// transforms exactly like a full frame of that shape would. Preview
/// hit-testing applies the same crop, so the selection box hugs the kept
/// pixels.
pub fn cropped_layer_placement(
    transform: &ClipTransform,
    content_w: u32,
    content_h: u32,
    crop: &CropRect,
    canvas: &CompositorConfig,
) -> LayerPlacement {
    let (cw, ch) = (canvas.width as f32, canvas.height as f32);
    let (w, h) = (content_w as f32 * crop.w, content_h as f32 * crop.h);
    let fit = if w > 0.0 && h > 0.0 {
        (cw / w).min(ch / h)
    } else {
        1.0
    };
    placement_from_size(transform, w, h, fit, canvas)
}

/// Placement for a generator raster authored at canvas pixel scale: its pixels
/// map 1:1 onto the canvas (no aspect-fit), centered, then the clip transform
/// applies. A text raster can exceed the canvas to hold content that overflows
/// the frame; the excess simply falls outside the frame until the clip is
/// moved or scaled down, instead of being clipped away during rasterization.
pub fn generator_layer_placement(
    transform: &ClipTransform,
    content_w: u32,
    content_h: u32,
    crop: &CropRect,
    canvas: &CompositorConfig,
) -> LayerPlacement {
    let (w, h) = (content_w as f32 * crop.w, content_h as f32 * crop.h);
    placement_from_size(transform, w, h, 1.0, canvas)
}

/// Shared geometry behind [`cropped_layer_placement`] and
/// [`generator_layer_placement`]: place content of effective size `w × h`
/// (already reduced by crop) at `fit · transform.scale`, centered on the
/// canvas and offset/rotated by the transform. The two callers differ only in
/// their `fit` choice (aspect-fit for sampled media, 1:1 for generator
/// rasters).
fn placement_from_size(
    transform: &ClipTransform,
    w: f32,
    h: f32,
    fit: f32,
    canvas: &CompositorConfig,
) -> LayerPlacement {
    let (cw, ch) = (canvas.width as f32, canvas.height as f32);
    let scale = fit * transform.scale;
    let size = [w * scale, h * scale];
    let anchor = [
        cw * 0.5 + transform.position[0] * cw,
        ch * 0.5 + transform.position[1] * ch,
    ];
    let to_center = [
        (0.5 - transform.anchor_point[0]) * size[0],
        (0.5 - transform.anchor_point[1]) * size[1],
    ];
    let (sin, cos) = transform.rotation.to_radians().sin_cos();
    let mut center = [
        anchor[0] + to_center[0] * cos - to_center[1] * sin,
        anchor[1] + to_center[0] * sin + to_center[1] * cos,
    ];
    // Unrotated layers snap their top-left corner to whole canvas pixels.
    // The bilinear sampler then sees the same sub-texel phase every frame,
    // so an animated position translates the layer as an exact pixel-shifted
    // copy of itself instead of pulsing between sharp and blurred as the
    // fractional offset drifts — the "shaking text" artifact. At 1:1 (text
    // and other full-canvas rasters) sampling lands exactly on texel
    // centers, keeping glyphs bit-crisp while they move. Rotated layers are
    // resampled off-grid by nature, so they keep continuous placement.
    if transform.rotation == 0.0 {
        for axis in 0..2 {
            let half = size[axis] * 0.5;
            center[axis] = (center[axis] - half).round() + half;
        }
    }
    LayerPlacement {
        center,
        size,
        rotation: transform.rotation.to_radians(),
        opacity: transform.opacity.clamp(0.0, 1.0),
    }
}

/// The anchor pivot in canvas pixels for a placed layer — the point scale and
/// rotation gestures pivot about, and what `ClipTransform::position` places.
pub fn anchor_canvas_position(transform: &ClipTransform, placement: &LayerPlacement) -> [f32; 2] {
    let offset = [
        (transform.anchor_point[0] - 0.5) * placement.size[0],
        (transform.anchor_point[1] - 0.5) * placement.size[1],
    ];
    let (sin, cos) = placement.rotation.sin_cos();
    [
        placement.center[0] + offset[0] * cos - offset[1] * sin,
        placement.center[1] + offset[0] * sin + offset[1] * cos,
    ]
}

/// Given a fixed content center and placed size, derive the normalized
/// anchor + position that keep the frame unchanged while moving the pivot to
/// `anchor_canvas`.
pub fn reposition_anchor(
    anchor_canvas: [f32; 2],
    center: [f32; 2],
    size: [f32; 2],
    rotation_deg: f32,
    canvas: &CompositorConfig,
) -> ([f32; 2], [f32; 2]) {
    let (cw, ch) = (canvas.width as f32, canvas.height as f32);
    let (sin, cos) = rotation_deg.to_radians().sin_cos();
    let delta = [center[0] - anchor_canvas[0], center[1] - anchor_canvas[1]];
    // Invert the clockwise rotation used by placement (same matrix as hit-test).
    let to_center = [
        delta[0] * cos + delta[1] * sin,
        -delta[0] * sin + delta[1] * cos,
    ];
    let anchor_point = [0.5 - to_center[0] / size[0], 0.5 - to_center[1] / size[1]];
    let position = [
        (anchor_canvas[0] - cw * 0.5) / cw,
        (anchor_canvas[1] - ch * 0.5) / ch,
    ];
    (anchor_point, position)
}

/// Canvas `position` that keeps `center` fixed when `anchor_point` changes
/// (inspector anchor sliders and keyframe edits that should not shift pixels).
pub fn position_preserving_center(
    center: [f32; 2],
    size: [f32; 2],
    anchor_point: [f32; 2],
    rotation_deg: f32,
    canvas: &CompositorConfig,
) -> [f32; 2] {
    let to_center = [
        (0.5 - anchor_point[0]) * size[0],
        (0.5 - anchor_point[1]) * size[1],
    ];
    let (sin, cos) = rotation_deg.to_radians().sin_cos();
    let anchor = [
        center[0] - (to_center[0] * cos - to_center[1] * sin),
        center[1] - (to_center[0] * sin + to_center[1] * cos),
    ];
    let (cw, ch) = (canvas.width as f32, canvas.height as f32);
    [(anchor[0] - cw * 0.5) / cw, (anchor[1] - ch * 0.5) / ch]
}

/// The compositor UV rect sampling a clip's kept region: the crop window,
/// with a flipped axis encoded as a reversed UV span.
pub fn content_uv(crop: &CropRect, flip_h: bool, flip_v: bool) -> [f32; 4] {
    let (mut u0, mut u1) = (crop.x, crop.x + crop.w);
    let (mut v0, mut v1) = (crop.y, crop.y + crop.h);
    if flip_h {
        std::mem::swap(&mut u0, &mut u1);
    }
    if flip_v {
        std::mem::swap(&mut v0, &mut v1);
    }
    [u0, v0, u1, v1]
}

/// Resolve enabled video layers at `time`, bottom to top.
///
/// Pass `cache: Some(...)` for interactive preview (disk cache on hit). Pass
/// `None` for export so every media frame is decoded from the original source
/// file — never from cached YUV blobs (and never from future proxy paths).
///
/// `anim_phase` is the fraction of a timeline tick past `time` at which
/// animated clip transforms sample (media frames stay on the whole tick).
/// Preview passes `0.0`; export passes the exact output-frame phase so a
/// 60 fps export of a 24 fps timeline animates at 60 Hz instead of
/// repeating 24 Hz positions in an uneven 3-2 cadence.
///
/// `override_transform` substitutes one clip's transform for this resolve
/// only — the live preview of an uncommitted drag gesture (preview roadmap
/// Phase 3). Session state, never project state: export passes `None`.
///
/// `override_generator` likewise substitutes one generated clip's generator
/// (e.g. a live inspector font-size drag) for this resolve only; export and
/// prefetch pass `None`.
#[allow(clippy::too_many_arguments)]
pub fn resolve_layers(
    project: &Project,
    cache: Option<&FrameCache>,
    pool: &mut DecoderPool,
    raster: &mut GeneratorRaster,
    time: RationalTime,
    anim_phase: f32,
    canvas: &CompositorConfig,
    color_convert: ColorConvertPath,
    override_transform: Option<(ClipId, ClipTransform)>,
    override_generator: Option<(ClipId, &Generator)>,
) -> Result<Vec<CompositeLayer>, EngineError> {
    let mut layers = Vec::new();

    for track in project.timeline().tracks_ordered() {
        if !track.kind.is_visual() || !track.enabled {
            continue;
        }
        let Some(clip) = track.clip_at(time)? else {
            continue;
        };

        // Transitions (M4): if `time` falls inside a junction's window on this
        // track, blend the outgoing and incoming clips instead of drawing the
        // single active one. Each side samples its own frame, holding the edge
        // frame where the window spills past that clip's range.
        if let Some(window) = transition_window(track, clip, time) {
            let from = resolve_clip_layer(
                project,
                cache,
                pool,
                raster,
                window.from,
                window.from_sample,
                anim_phase,
                canvas,
                color_convert,
                override_transform,
                override_generator,
            )?;
            let to = resolve_clip_layer(
                project,
                cache,
                pool,
                raster,
                window.to,
                window.to_sample,
                anim_phase,
                canvas,
                color_convert,
                override_transform,
                override_generator,
            )?;
            if let (Some(from), Some(to)) = (from, to) {
                layers.push(CompositeLayer::transition(
                    from,
                    to,
                    window.transition_id,
                    window.progress,
                ));
                continue;
            }
            // A side failed to resolve (e.g. missing media): fall back to the
            // plain active clip below.
        }

        if let Some(layer) = resolve_clip_layer(
            project,
            cache,
            pool,
            raster,
            clip,
            time,
            anim_phase,
            canvas,
            color_convert,
            override_transform,
            override_generator,
        )? {
            layers.push(layer);
        }
    }

    Ok(layers)
}

/// A resolved transition window at `time`: which clips blend, the frame time
/// each side samples (clamped into its own range), and the blend progress.
struct ResolvedTransition<'a> {
    from: &'a Clip,
    to: &'a Clip,
    from_sample: RationalTime,
    to_sample: RationalTime,
    transition_id: &'a str,
    progress: f32,
}

/// If `time` falls inside the window of a transition on `track` whose junction
/// touches `active` (the clip `clip_at(time)` returned), resolve it. The
/// window is centered on the cut (`left.end == right.start`) and spans the
/// transition's `duration` in ticks.
fn transition_window<'a>(
    track: &'a cutlass_models::Track,
    active: &Clip,
    time: RationalTime,
) -> Option<ResolvedTransition<'a>> {
    for transition in track.transitions() {
        if active.id != transition.left && active.id != transition.right {
            continue;
        }
        let left = track.clip(transition.left)?;
        let right = track.clip(transition.right)?;
        // The cut sits where the two clips abut; bail if they no longer do
        // (a stale junction the prune pass hasn't swept yet).
        let center = left.timeline.end_tick();
        if center != right.timeline.start.value {
            continue;
        }
        let half = transition.duration / 2;
        let start = center - half;
        let end = start + transition.duration;
        if time.value < start || time.value >= end {
            continue;
        }
        let rate = time.rate;
        let progress = ((time.value - start) as f32 / transition.duration as f32).clamp(0.0, 1.0);
        // Hold each side's edge frame where the window spills past its range.
        let from_sample = RationalTime::new(
            time.value
                .clamp(left.timeline.start.value, left.timeline.end_tick() - 1),
            rate,
        );
        let to_sample = RationalTime::new(
            time.value
                .clamp(right.timeline.start.value, right.timeline.end_tick() - 1),
            rate,
        );
        return Some(ResolvedTransition {
            from: left,
            to: right,
            from_sample,
            to_sample,
            transition_id: &transition.transition_id,
            progress,
        });
    }
    None
}

/// Resolve a single clip into a [`CompositeLayer`] at sample time `time`.
/// `None` when the content contributes nothing (e.g. a generator with no
/// raster). The active-clip path and both transition sides share this so they
/// can never disagree on placement, framing, or effects.
#[allow(clippy::too_many_arguments)]
fn resolve_clip_layer(
    project: &Project,
    cache: Option<&FrameCache>,
    pool: &mut DecoderPool,
    raster: &mut GeneratorRaster,
    clip: &Clip,
    time: RationalTime,
    anim_phase: f32,
    canvas: &CompositorConfig,
    color_convert: ColorConvertPath,
    override_transform: Option<(ClipId, ClipTransform)>,
    override_generator: Option<(ClipId, &Generator)>,
) -> Result<Option<CompositeLayer>, EngineError> {
    // Animated params sample at the clip-relative tick (M2): pure binary
    // search + lerp per property, allocation-free. A live gesture
    // override replaces the whole sampled value for this resolve only.
    let anim_tick = clip.animation_tick_f(time.value as f64 + f64::from(anim_phase));
    let transform = match &override_transform {
        Some((id, t)) if *id == clip.id => *t,
        _ => clip.transform.sample_at(anim_tick),
    };
    let transform = &transform;
    // Framing (M1): the kept region shapes the placement, the UV rect
    // samples it (flips reverse the sampled axis).
    let crop = &clip.crop;
    let uv = content_uv(crop, clip.flip_h, clip.flip_v);
    // Effects (M4): sample each animated param at the same clip tick and
    // pack into the compositor's slot order. Empty for clips without
    // effects (no allocation), keeping the no-effect path untouched.
    let effects = resolve_effects(clip, anim_tick);
    match &clip.content {
        cutlass_models::ClipSource::Media { .. } => {
            // Still images bypass the decode/cache pipeline entirely:
            // one cached RGBA upload, identical for every tick the clip
            // covers (and for both color-convert paths).
            if let Some(media_id) = clip.media()
                && let Some(media) = project.media(media_id)
                && media.is_image
            {
                let (bytes, width, height) = pool.still(media_id, media.path())?;
                let placement = cropped_layer_placement(transform, width, height, crop, canvas);
                return Ok(Some(
                    CompositeLayer::rgba(bytes, width, height, placement)
                        .with_uv(uv)
                        .with_effects(effects),
                ));
            }
            let layer = match color_convert {
                ColorConvertPath::Gpu => {
                    let decoded = decode_media_frame(project, cache, pool, clip, time)?;
                    let yuv = decoded_to_yuv_layer(&decoded)?;
                    let placement =
                        cropped_layer_placement(transform, yuv.width, yuv.height, crop, canvas);
                    CompositeLayer::yuv420p(yuv, placement).with_uv(uv)
                }
                ColorConvertPath::LegacyCpu => {
                    // Native-size upload; the GPU scales it into place
                    // (the old CPU bilinear resize-to-canvas is gone).
                    let frame = decode_media_rgba_legacy(project, cache, pool, clip, time)?;
                    let placement =
                        cropped_layer_placement(transform, frame.width, frame.height, crop, canvas);
                    CompositeLayer::rgba(
                        std::sync::Arc::new(frame.bytes),
                        frame.width,
                        frame.height,
                        placement,
                    )
                    .with_uv(uv)
                }
            };
            Ok(Some(layer.with_effects(effects)))
        }
        cutlass_models::ClipSource::Generated(generator) => {
            // A live inspector edit (e.g. font-size drag) renders this clip
            // from the override generator instead of its committed one.
            let generator = match override_generator {
                Some((id, g)) if id == clip.id => g,
                _ => generator,
            };
            // Solids fill the canvas. Text/shape rasters carry their own size
            // (text can exceed the canvas to hold content overflowing the
            // frame) and place 1:1 with canvas pixels, so the clip transform
            // applies on top of that raster.
            let placement =
                cropped_layer_placement(transform, canvas.width, canvas.height, crop, canvas);
            match generator {
                Generator::SolidColor { rgba } => {
                    // Solids have no texture to sample: crop shrinks the
                    // quad, flips are invisible.
                    Ok(Some(
                        CompositeLayer::solid(*rgba, placement).with_effects(effects),
                    ))
                }
                Generator::Text { .. } | Generator::Shape { .. } => {
                    match raster.raster_layer(generator, canvas.width, canvas.height) {
                        Some((bytes, raster_w, raster_h)) => {
                            let placement = generator_layer_placement(
                                transform, raster_w, raster_h, crop, canvas,
                            );
                            Ok(Some(
                                CompositeLayer::rgba(bytes, raster_w, raster_h, placement)
                                    .with_uv(uv)
                                    .with_effects(effects),
                            ))
                        }
                        None => {
                            debug!(?generator, "generator produced no raster");
                            Ok(None)
                        }
                    }
                }
                Generator::Adjustment => {
                    // Adjustment layers (M4) carry no content of their
                    // own; the compositor applies their effect chain to
                    // the accumulated canvas below. An empty chain is a
                    // harmless no-op there, so emit unconditionally.
                    Ok(Some(CompositeLayer::adjustment(
                        effects,
                        transform.opacity.clamp(0.0, 1.0),
                    )))
                }
                Generator::Sticker | Generator::Effect | Generator::Filter => {
                    debug!(?generator, "skipping unsupported generator for composite");
                    Ok(None)
                }
            }
        }
    }
}

/// Sample a clip's effect chain at clip-relative `tick` into compositor
/// [`LayerEffect`]s. Each model parameter is looked up by name in the
/// compositor's slot table, so the catalog (defaults / ranges) and the WGSL
/// (slot order) bridge by name and can't silently drift. Unknown effects or
/// params are skipped here; the model layer validates them on edit.
fn resolve_effects(clip: &Clip, tick: f64) -> Vec<LayerEffect> {
    if clip.effects.is_empty() {
        return Vec::new();
    }
    clip.effects
        .iter()
        .filter_map(|fx| {
            let spec = cutlass_models::effect_spec(&fx.effect_id)?;
            let mut layer_effect = LayerEffect::new(&fx.effect_id);
            for pspec in spec.params {
                let value = fx.sample_param(pspec.name, tick).unwrap_or(pspec.default);
                if let Some(slot) =
                    cutlass_compositor::effect_param_index(&fx.effect_id, pspec.name)
                {
                    layer_effect.params[slot] = value;
                }
            }
            Some(layer_effect)
        })
        .collect()
}

fn decode_media_frame(
    project: &Project,
    cache: Option<&FrameCache>,
    pool: &mut DecoderPool,
    clip: &Clip,
    time: RationalTime,
) -> Result<DecodedFrame, EngineError> {
    let source_time = clip
        .source_time_at(time)?
        .ok_or_else(|| EngineError::Preview("timeline position outside clip".into()))?;

    let media_id = clip
        .media()
        .ok_or_else(|| EngineError::Preview("clip has no backing media".into()))?;
    let media = project
        .media(media_id)
        .ok_or(ModelError::UnknownMedia(media_id))?;

    let fingerprint = SourceFingerprint::from_path(media.path())?;
    let source_id = fingerprint.id();

    let (decoder, index) = pool.decoder_and_index(media_id, media.path())?;
    // Exact rational → stream-tick conversion. The old `RationalTime →
    // Duration → ticks` path truncated twice, landing rate-matched targets
    // one tick below the frame's stored PTS — a guaranteed cache miss on
    // every revisit (measured: 1080p24 media on the 24fps timeline only hit
    // on ticks where the nanosecond hop happened to be exact).
    let target_ticks = index.rate_ticks_to_stream_ticks(
        source_time.value,
        source_time.rate.num,
        source_time.rate.den,
    );

    // Preview (cache = Some) renders at PREVIEW_MAX_HEIGHT; export (cache =
    // None) keeps full source. The cache stores frames at the preview dims, so
    // hits unpack at the same size that misses scale down to.
    let (preview_w, preview_h) = if cache.is_some() {
        preview_scaled_dims(media.width, media.height)
    } else {
        (media.width, media.height)
    };

    let start = Instant::now();
    if let Some(cache) = cache
        && let Some(packed) = cache.get(source_id, target_ticks)
    {
        debug!(
            us = start.elapsed().as_micros() as u64,
            pts = target_ticks,
            "preview frame cache hit"
        );
        return crate::frame::unpack_yuv420p(&packed, preview_w, preview_h);
    }

    let decoded = decoder
        .frame_at_ticks(target_ticks)?
        .ok_or_else(|| EngineError::Preview("decoder returned no frame".into()))?;
    debug!(
        ms = start.elapsed().as_secs_f64() * 1000.0,
        pts = decoded.pts_ticks,
        "decoded media frame"
    );

    // Downscale once on the decode miss (native → preview height) so every
    // later cache hit, GPU upload, composite, and readback is the small size.
    let decoded = if cache.is_some() && (decoded.width, decoded.height) != (preview_w, preview_h) {
        cutlass_decoder::scale_yuv420p(&decoded, preview_w, preview_h)?
    } else {
        decoded
    };

    // Cache under the *requested* `target_ticks`, not the decoded frame's
    // PTS. The reader looks up by `target_ticks`, and a target rarely lands
    // exactly on a frame's stored PTS (any rate that isn't a clean multiple
    // of the stream time_base — e.g. 60.03fps media). Keying by PTS made
    // every revisit miss, and during playback the read-ahead prefetch had
    // already rolled the decoder *past* the playhead, so each missed frame
    // re-decoded as a backward seek (≈3fps, plus the `mmco` warning flood).
    // Same key on both sides ⇒ prefetch warms exactly what the render reads.
    if let Some(cache) = cache
        && let Ok(packed) = crate::frame::pack_yuv420p(&decoded)
    {
        cache.cache_frame(source_id, target_ticks, packed);
    }

    Ok(decoded)
}

fn decode_media_rgba_legacy(
    project: &Project,
    cache: Option<&FrameCache>,
    pool: &mut DecoderPool,
    clip: &Clip,
    time: RationalTime,
) -> Result<RgbaFrame, EngineError> {
    let decoded = decode_media_frame(project, cache, pool, clip, time)?;
    legacy_decoded_to_rgba(&decoded)
}

fn to_even(v: u32) -> u32 {
    if v.is_multiple_of(2) { v } else { v + 1 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_compositor::LayerContent;
    use cutlass_models::Rational;

    #[test]
    fn composite_canvas_size_defaults_without_media() {
        let project = Project::new("test", Rational::FPS_24);
        assert_eq!(composite_canvas_size(&project), (1920, 1080));
    }

    #[test]
    fn scale_to_max_height_no_cap_is_identity() {
        assert_eq!(scale_to_max_height(3840, 2160, None), (3840, 2160));
        assert_eq!(scale_to_max_height(3840, 2160, Some(0)), (3840, 2160));
    }

    #[test]
    fn scale_to_max_height_passes_through_when_within_cap() {
        assert_eq!(scale_to_max_height(1920, 1080, Some(1080)), (1920, 1080));
        assert_eq!(scale_to_max_height(1280, 720, Some(1080)), (1280, 720));
    }

    #[test]
    fn scale_to_max_height_scales_landscape_4k_to_720p() {
        assert_eq!(scale_to_max_height(3840, 2160, Some(720)), (1280, 720));
    }

    #[test]
    fn scale_to_max_height_scales_portrait_by_height() {
        // Height drops to the cap; width follows, aspect kept and even.
        assert_eq!(scale_to_max_height(2160, 3840, Some(720)), (406, 720));
    }

    #[test]
    fn scale_to_max_height_forces_even_dimensions() {
        let (w, h) = scale_to_max_height(1999, 1001, Some(501));
        assert_eq!(h, 502);
        assert!(w.is_multiple_of(2) && h.is_multiple_of(2), "got {w}x{h}");
    }

    #[test]
    fn preview_scaled_dims_uses_the_constant() {
        // Whatever the shipped cap is, a 4K frame must come back capped and even.
        let (w, h) = preview_scaled_dims(3840, 2160);
        if let Some(cap) = PREVIEW_MAX_HEIGHT {
            assert_eq!(h, cap);
            assert!(w.is_multiple_of(2) && h.is_multiple_of(2), "got {w}x{h}");
        } else {
            assert_eq!((w, h), (3840, 2160));
        }
    }

    fn rgba_layer(layer: &CompositeLayer) -> &std::sync::Arc<Vec<u8>> {
        match &layer.content {
            LayerContent::Rgba { bytes, .. } => bytes,
            other => panic!("expected Rgba layer, got {other:?}"),
        }
    }

    // --- layer_placement ---------------------------------------------------

    const CANVAS: CompositorConfig = CompositorConfig::new(1920, 1080);

    #[test]
    fn identity_transform_on_canvas_sized_content_is_full_canvas() {
        let p = layer_placement(&ClipTransform::IDENTITY, 1920, 1080, &CANVAS);
        assert_eq!(p, LayerPlacement::full_canvas(&CANVAS));
    }

    #[test]
    fn mismatched_aspect_fits_inside_canvas() {
        // Portrait 1080×1920 into a 1920×1080 canvas: height-limited.
        let p = layer_placement(&ClipTransform::IDENTITY, 1080, 1920, &CANVAS);
        let fit = 1080.0 / 1920.0; // canvas_h / content_h
        assert_eq!(p.size, [1080.0 * fit, 1920.0 * fit]);
        assert!(p.size[1] <= 1080.0 + 1e-3);
        // Unrotated: the corner pixel-snaps, so the center sits within half
        // a pixel of true center with an integral left edge.
        assert_eq!(p.center[1], 540.0);
        assert!((p.center[0] - 960.0).abs() <= 0.5);
        let left = p.center[0] - p.size[0] / 2.0;
        assert_eq!(left, left.round());
    }

    #[test]
    fn unrotated_placement_snaps_corner_to_whole_pixels() {
        // A fractional position offset must not leave the layer sampling
        // between texels (per-frame sub-pixel phase = moving-text shimmer).
        let t = ClipTransform {
            position: [0.1234, -0.0567],
            ..ClipTransform::IDENTITY
        };
        let p = layer_placement(&t, 1920, 1080, &CANVAS);
        for axis in 0..2 {
            let corner = p.center[axis] - p.size[axis] / 2.0;
            assert_eq!(corner, corner.round(), "axis {axis} corner {corner}");
        }
        // Snapping moves the layer by less than half a pixel.
        assert!((p.center[0] - (960.0 + 0.1234 * 1920.0)).abs() <= 0.5);
        assert!((p.center[1] - (540.0 - 0.0567 * 1080.0)).abs() <= 0.5);

        // Rotated layers resample off-grid regardless; they keep the
        // continuous (unsnapped) placement.
        let rotated = ClipTransform {
            rotation: 30.0,
            ..t
        };
        let p = layer_placement(&rotated, 1920, 1080, &CANVAS);
        assert_eq!(p.center, [960.0 + 0.1234 * 1920.0, 540.0 - 0.0567 * 1080.0]);
    }

    #[test]
    fn position_scale_rotation_opacity_apply() {
        let t = ClipTransform {
            position: [0.25, -0.5],
            scale: 0.5,
            rotation: 90.0,
            opacity: 0.4,
            ..ClipTransform::IDENTITY
        };
        let p = layer_placement(&t, 1920, 1080, &CANVAS);
        assert_eq!(p.center, [960.0 + 0.25 * 1920.0, 540.0 - 0.5 * 1080.0]);
        assert_eq!(p.size, [960.0, 540.0]);
        assert!((p.rotation - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        assert_eq!(p.opacity, 0.4);
    }

    #[test]
    fn off_center_anchor_shifts_placement() {
        // Anchor at the content's top-left corner; position places that
        // corner on the canvas center — the layer hangs down-right.
        let t = ClipTransform {
            anchor_point: [0.0, 0.0],
            ..ClipTransform::IDENTITY
        };
        let p = layer_placement(&t, 1920, 1080, &CANVAS);
        assert_eq!(p.size, [1920.0, 1080.0]);
        assert_eq!(p.center, [960.0 + 960.0, 540.0 + 540.0]);
        assert_eq!(anchor_canvas_position(&t, &p), [960.0, 540.0]);
    }

    #[test]
    fn position_preserving_center_roundtrips_with_placement() {
        let base = ClipTransform {
            position: [0.125, -0.130],
            anchor_point: [0.25, 0.75],
            rotation: 30.0,
            ..ClipTransform::IDENTITY
        };
        let p0 = layer_placement(&base, 1920, 1080, &CANVAS);
        let pos = position_preserving_center(p0.center, p0.size, [0.8, 0.2], 30.0, &CANVAS);
        let t = ClipTransform {
            position: pos,
            anchor_point: [0.8, 0.2],
            rotation: 30.0,
            ..ClipTransform::IDENTITY
        };
        let p1 = layer_placement(&t, 1920, 1080, &CANVAS);
        assert!((p1.center[0] - p0.center[0]).abs() < 1e-3);
        assert!((p1.center[1] - p0.center[1]).abs() < 1e-3);
    }

    #[test]
    fn scale_about_off_center_anchor_keeps_canvas_pivot() {
        let t0 = ClipTransform {
            anchor_point: [0.25, 0.5],
            scale: 1.0,
            ..ClipTransform::IDENTITY
        };
        let p0 = layer_placement(&t0, 1920, 1080, &CANVAS);
        let a0 = anchor_canvas_position(&t0, &p0);
        let t1 = ClipTransform { scale: 0.5, ..t0 };
        let p1 = layer_placement(&t1, 1920, 1080, &CANVAS);
        let a1 = anchor_canvas_position(&t1, &p1);
        assert!((a0[0] - a1[0]).abs() < 1e-3);
        assert!((a0[1] - a1[1]).abs() < 1e-3);
        assert!((p1.size[0] - p0.size[0] * 0.5).abs() < 1e-3);
    }

    #[test]
    fn reposition_anchor_preserves_center() {
        let base = ClipTransform {
            position: [0.125, -0.130],
            rotation: 30.0,
            ..ClipTransform::IDENTITY
        };
        let p0 = layer_placement(&base, 1920, 1080, &CANVAS);
        let center = p0.center;
        let anchor = anchor_canvas_position(&base, &p0);
        let (ap, pos) = reposition_anchor(anchor, center, p0.size, 30.0, &CANVAS);
        let t = ClipTransform {
            position: pos,
            anchor_point: ap,
            rotation: 30.0,
            ..ClipTransform::IDENTITY
        };
        let p = layer_placement(&t, 1920, 1080, &CANVAS);
        assert!(
            (p.center[0] - center[0]).abs() < 1e-3,
            "{:?} vs {:?}",
            p.center,
            center
        );
        assert!((p.center[1] - center[1]).abs() < 1e-3);
        let a = anchor_canvas_position(&t, &p);
        assert!((a[0] - anchor[0]).abs() < 1e-3);
        assert!((a[1] - anchor[1]).abs() < 1e-3);
    }

    #[test]
    fn cropped_placement_fits_the_kept_region() {
        // Keep the center half horizontally of 1920×1080 content: the kept
        // region is 960×1080 (portrait-ish), so the fit is height-limited
        // on the 1920×1080 canvas — no stretch, aspect preserved.
        let crop = CropRect {
            x: 0.25,
            y: 0.0,
            w: 0.5,
            h: 1.0,
        };
        let p = cropped_layer_placement(&ClipTransform::IDENTITY, 1920, 1080, &crop, &CANVAS);
        assert_eq!(p.center, [960.0, 540.0]);
        assert_eq!(p.size, [960.0, 1080.0]);

        // Full crop matches the uncropped geometry exactly.
        let full = cropped_layer_placement(
            &ClipTransform::IDENTITY,
            1920,
            1080,
            &CropRect::FULL,
            &CANVAS,
        );
        assert_eq!(
            full,
            layer_placement(&ClipTransform::IDENTITY, 1920, 1080, &CANVAS)
        );
    }

    #[test]
    fn content_uv_crops_and_mirrors() {
        let crop = CropRect {
            x: 0.1,
            y: 0.2,
            w: 0.5,
            h: 0.25,
        };
        assert_eq!(content_uv(&crop, false, false), [0.1, 0.2, 0.6, 0.45]);
        // Flips reverse the sampled axis, keeping the same window.
        assert_eq!(content_uv(&crop, true, false), [0.6, 0.2, 0.1, 0.45]);
        assert_eq!(content_uv(&crop, false, true), [0.1, 0.45, 0.6, 0.2]);
        assert_eq!(
            content_uv(&CropRect::FULL, true, true),
            [1.0, 1.0, 0.0, 0.0]
        );
    }

    #[test]
    fn resolve_applies_crop_to_placement_and_uv() {
        use cutlass_models::TimeRange;
        let rate = cutlass_models::Rational::FPS_24;
        let mut project = Project::new("t", rate);
        let track = project.add_track(TrackKind::Text, "T1");
        let clip = project
            .add_generated(
                track,
                Generator::text("Hi"),
                TimeRange::at_rate(0, 24, rate),
            )
            .unwrap();
        let crop = CropRect {
            x: 0.0,
            y: 0.25,
            w: 1.0,
            h: 0.5,
        };
        project.set_clip_crop(clip, crop, true, false).unwrap();

        let mut pool = DecoderPool::new();
        let mut raster = GeneratorRaster::new();
        let canvas = CompositorConfig::new(320, 240);
        let layers = resolve_layers(
            &project,
            None,
            &mut pool,
            &mut raster,
            RationalTime::new(0, rate),
            0.0,
            &canvas,
            ColorConvertPath::Gpu,
            None,
            None,
        )
        .unwrap();
        assert_eq!(layers.len(), 1);
        // Kept region of the 320×240 raster is 320×120 → width-limited fit
        // fills the canvas width at half height.
        assert_eq!(layers[0].placement.size, [320.0, 120.0]);
        // UV samples the kept band, mirrored horizontally.
        assert_eq!(layers[0].uv, [1.0, 0.25, 0.0, 0.75]);
    }

    #[test]
    fn resolve_attaches_sampled_effects_to_layer() {
        use cutlass_models::TimeRange;
        let rate = cutlass_models::Rational::FPS_24;
        let mut project = Project::new("t", rate);
        let track = project.add_track(TrackKind::Text, "T1");
        let clip = project
            .add_generated(
                track,
                Generator::text("Hi"),
                TimeRange::at_rate(0, 24, rate),
            )
            .unwrap();
        project.add_effect(clip, "vignette").unwrap();
        // amount is slot 0 of vignette.
        project.set_effect_param(clip, 0, 0, 0.5).unwrap();

        let mut pool = DecoderPool::new();
        let mut raster = GeneratorRaster::new();
        let canvas = CompositorConfig::new(64, 64);
        let layers = resolve_layers(
            &project,
            None,
            &mut pool,
            &mut raster,
            RationalTime::new(0, rate),
            0.0,
            &canvas,
            ColorConvertPath::Gpu,
            None,
            None,
        )
        .unwrap();
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].effects.len(), 1);
        assert_eq!(layers[0].effects[0].effect_id, "vignette");
        let slot = cutlass_compositor::effect_param_index("vignette", "amount").unwrap();
        assert_eq!(layers[0].effects[0].params[slot], 0.5);
    }

    #[test]
    fn resolve_emits_adjustment_layer_with_its_chain() {
        use cutlass_compositor::LayerContent;
        use cutlass_models::TimeRange;
        let rate = cutlass_models::Rational::FPS_24;
        let mut project = Project::new("t", rate);
        let track = project.add_track(TrackKind::Adjustment, "FX");
        let clip = project
            .add_generated(
                track,
                Generator::Adjustment,
                TimeRange::at_rate(0, 24, rate),
            )
            .unwrap();
        project.add_effect(clip, "vignette").unwrap();

        let mut pool = DecoderPool::new();
        let mut raster = GeneratorRaster::new();
        let canvas = CompositorConfig::new(64, 64);
        let layers = resolve_layers(
            &project,
            None,
            &mut pool,
            &mut raster,
            RationalTime::new(0, rate),
            0.0,
            &canvas,
            ColorConvertPath::Gpu,
            None,
            None,
        )
        .unwrap();
        // The adjustment clip resolves to an Adjustment layer carrying its
        // effect chain (not a textured layer).
        assert_eq!(layers.len(), 1);
        assert!(matches!(layers[0].content, LayerContent::Adjustment));
        assert_eq!(layers[0].effects.len(), 1);
        assert_eq!(layers[0].effects[0].effect_id, "vignette");
    }

    #[test]
    fn resolve_emits_transition_layer_inside_the_window() {
        use cutlass_compositor::LayerContent;
        use cutlass_models::TimeRange;
        let rate = cutlass_models::Rational::FPS_24;
        let mut project = Project::new("t", rate);
        let track = project.add_track(TrackKind::Sticker, "S");
        let left = project
            .add_generated(
                track,
                Generator::SolidColor {
                    rgba: [255, 0, 0, 255],
                },
                TimeRange::at_rate(0, 24, rate),
            )
            .unwrap();
        let _right = project
            .add_generated(
                track,
                Generator::SolidColor {
                    rgba: [0, 0, 255, 255],
                },
                TimeRange::at_rate(24, 24, rate),
            )
            .unwrap();
        project.add_transition(left, "crossfade").unwrap();

        let mut pool = DecoderPool::new();
        let mut raster = GeneratorRaster::new();
        let canvas = CompositorConfig::new(64, 64);
        let resolve = |t: i64, pool: &mut DecoderPool, raster: &mut GeneratorRaster| {
            resolve_layers(
                &project,
                None,
                pool,
                raster,
                RationalTime::new(t, rate),
                0.0,
                &canvas,
                ColorConvertPath::Gpu,
                None,
                None,
            )
            .unwrap()
        };

        // Centered on the cut at tick 24 with a 24-tick window → [12, 36). At
        // the cut the blend is half-way.
        let mid = resolve(24, &mut pool, &mut raster);
        assert_eq!(mid.len(), 1);
        match &mid[0].content {
            LayerContent::Transition {
                transition_id,
                progress,
                ..
            } => {
                assert_eq!(transition_id, "crossfade");
                assert!((*progress - 0.5).abs() < 1e-4, "progress {progress}");
            }
            other => panic!("expected a transition layer, got {other:?}"),
        }

        // Outside the window it is a plain solid layer again.
        let before = resolve(4, &mut pool, &mut raster);
        assert!(matches!(before[0].content, LayerContent::Solid { .. }));
    }

    #[test]
    fn resolve_without_effects_leaves_empty_chain() {
        use cutlass_models::TimeRange;
        let rate = cutlass_models::Rational::FPS_24;
        let mut project = Project::new("t", rate);
        let track = project.add_track(TrackKind::Text, "T1");
        project
            .add_generated(
                track,
                Generator::text("Hi"),
                TimeRange::at_rate(0, 24, rate),
            )
            .unwrap();

        let mut pool = DecoderPool::new();
        let mut raster = GeneratorRaster::new();
        let canvas = CompositorConfig::new(64, 64);
        let layers = resolve_layers(
            &project,
            None,
            &mut pool,
            &mut raster,
            RationalTime::new(0, rate),
            0.0,
            &canvas,
            ColorConvertPath::Gpu,
            None,
            None,
        )
        .unwrap();
        assert!(layers[0].effects.is_empty());
    }

    #[test]
    fn resolve_text_generator_yields_rgba_layer() {
        use cutlass_models::TimeRange;
        let mut project = Project::new("t", cutlass_models::Rational::FPS_24);
        let track = project.add_track(TrackKind::Text, "T1");
        project
            .add_generated(
                track,
                Generator::text("Hi"),
                TimeRange::at_rate(0, 24, cutlass_models::Rational::FPS_24),
            )
            .unwrap();

        let mut pool = DecoderPool::new();
        let mut raster = GeneratorRaster::new();
        let canvas = CompositorConfig::new(320, 240);
        let layers = resolve_layers(
            &project,
            None,
            &mut pool,
            &mut raster,
            RationalTime::new(0, cutlass_models::Rational::FPS_24),
            0.0,
            &canvas,
            ColorConvertPath::Gpu,
            None,
            None,
        )
        .unwrap();
        assert_eq!(layers.len(), 1);
        let bytes = rgba_layer(&layers[0]);
        assert_eq!(bytes.len(), (320 * 240 * 4) as usize);
        // Text rasterizes some visible (non-transparent) pixels.
        assert!(bytes.chunks_exact(4).any(|p| p[3] > 0));
    }

    #[test]
    fn resolve_samples_keyframed_transform_at_frame_tick() {
        use cutlass_models::{ClipParam, Easing, ParamValue, RationalTime as RT, TimeRange};
        let rate = cutlass_models::Rational::FPS_24;
        let mut project = Project::new("t", rate);
        let track = project.add_track(TrackKind::Sticker, "ST1");
        let clip = project
            .add_generated(
                track,
                Generator::SolidColor {
                    rgba: [255, 0, 0, 255],
                },
                // Clip starts at tick 12, not 0 — sampling must be clip-relative.
                TimeRange::at_rate(12, 48, rate),
            )
            .unwrap();
        // Opacity fades 0 → 1 over the first 24 clip ticks.
        project
            .set_param_keyframe(
                clip,
                ClipParam::Opacity,
                RT::new(12, rate),
                ParamValue::Scalar(0.0),
                Easing::Linear,
            )
            .unwrap();
        project
            .set_param_keyframe(
                clip,
                ClipParam::Opacity,
                RT::new(36, rate),
                ParamValue::Scalar(1.0),
                Easing::Linear,
            )
            .unwrap();

        let mut pool = DecoderPool::new();
        let mut raster = GeneratorRaster::new();
        let canvas = CompositorConfig::new(64, 64);
        let opacity_at =
            |pool: &mut DecoderPool, raster: &mut GeneratorRaster, tick: i64, phase: f32| {
                let layers = resolve_layers(
                    &project,
                    None,
                    pool,
                    raster,
                    RationalTime::new(tick, rate),
                    phase,
                    &canvas,
                    ColorConvertPath::Gpu,
                    None,
                    None,
                )
                .unwrap();
                layers[0].placement.opacity
            };

        assert_eq!(opacity_at(&mut pool, &mut raster, 12, 0.0), 0.0);
        assert_eq!(opacity_at(&mut pool, &mut raster, 24, 0.0), 0.5);
        assert_eq!(opacity_at(&mut pool, &mut raster, 36, 0.0), 1.0);
        // Past the last keyframe the value holds.
        assert_eq!(opacity_at(&mut pool, &mut raster, 50, 0.0), 1.0);
        // Sub-frame phases sample between ticks (export above the timeline
        // rate): half a tick past 24 is 12.5/24 of the fade.
        assert_eq!(opacity_at(&mut pool, &mut raster, 24, 0.5), 12.5 / 24.0);
    }

    fn png_asset() -> Option<std::path::PathBuf> {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../local-assets/assets/texture.png");
        path.exists().then_some(path)
    }

    #[test]
    fn resolve_image_still_yields_cached_rgba_layer() {
        use cutlass_models::MediaSource;
        let Some(path) = png_asset() else {
            return;
        };
        let rate = cutlass_models::Rational::FPS_24;
        let mut project = Project::new("t", rate);
        let media_id = project.add_media(MediaSource::image(&path, 0, 0));
        let source = project.media(media_id).unwrap().full_range();
        let track = project.add_track(TrackKind::Video, "V1");
        project
            .add_clip(track, media_id, source, RationalTime::new(0, rate))
            .unwrap();

        let mut pool = DecoderPool::new();
        let mut raster = GeneratorRaster::new();
        let canvas = CompositorConfig::new(320, 240);
        let resolve = |pool: &mut DecoderPool, raster: &mut GeneratorRaster, tick: i64| {
            resolve_layers(
                &project,
                None,
                pool,
                raster,
                RationalTime::new(tick, rate),
                0.0,
                &canvas,
                ColorConvertPath::Gpu,
                None,
                None,
            )
            .unwrap()
        };

        // 5s at 24fps = 120 ticks; the still covers all of them.
        let first = resolve(&mut pool, &mut raster, 0);
        assert_eq!(first.len(), 1);
        let bytes = rgba_layer(&first[0]);
        assert!(!bytes.is_empty());

        // Same Arc on a later tick: decoded once, reused per frame.
        let later = resolve(&mut pool, &mut raster, 119);
        assert!(std::sync::Arc::ptr_eq(
            rgba_layer(&first[0]),
            rgba_layer(&later[0])
        ));

        // Past the clip: gap, no layers.
        assert!(resolve(&mut pool, &mut raster, 120).is_empty());
    }

    #[test]
    fn canvas_size_ignores_image_media() {
        use cutlass_models::{MediaSource, TimeRange};
        let rate = cutlass_models::Rational::FPS_24;
        let mut project = Project::new("t", rate);
        // A 12MP still on the timeline must not balloon the canvas.
        let media_id = project.add_media(MediaSource::image("/photos/p.png", 4000, 3000));
        let track = project.add_track(TrackKind::Video, "V1");
        project
            .add_clip(
                track,
                media_id,
                TimeRange::at_rate(0, 5_000, cutlass_models::STILL_TICK_RATE),
                RationalTime::new(0, rate),
            )
            .unwrap();
        assert_eq!(composite_canvas_size(&project), (1920, 1080));
    }

    #[test]
    fn canvas_size_honors_aspect_presets() {
        use cutlass_models::{CanvasAspect, CanvasSettings, MediaSource, TimeRange};
        let rate = cutlass_models::Rational::FPS_24;
        let mut project = Project::new("t", rate);
        let media_id = project.add_media(MediaSource::new(
            "/tmp/clip.mp4",
            1920,
            1080,
            rate,
            240,
            false,
        ));
        let track = project.add_track(TrackKind::Video, "V1");
        project
            .add_clip(
                track,
                media_id,
                TimeRange::at_rate(0, 100, rate),
                RationalTime::new(0, rate),
            )
            .unwrap();

        // The short edge keeps the footage's 1080 tier; the long edge
        // follows the chosen ratio.
        let expect = [
            (CanvasAspect::Auto, (1920, 1080)),
            (CanvasAspect::Wide16x9, (1920, 1080)),
            (CanvasAspect::Tall9x16, (1080, 1920)),
            (CanvasAspect::Square1x1, (1080, 1080)),
            (CanvasAspect::Portrait4x5, (1080, 1350)),
            (CanvasAspect::Cinema21x9, (2520, 1080)),
        ];
        for (aspect, size) in expect {
            project.timeline_mut().set_canvas(CanvasSettings {
                aspect,
                background: [0, 0, 0],
            });
            assert_eq!(composite_canvas_size(&project), size, "{}", aspect.name());
        }
    }

    #[test]
    fn canvas_aspect_applies_to_the_empty_project_fallback() {
        use cutlass_models::{CanvasAspect, CanvasSettings};
        let mut project = Project::new("t", cutlass_models::Rational::FPS_24);
        project.timeline_mut().set_canvas(CanvasSettings {
            aspect: CanvasAspect::Tall9x16,
            background: [0, 0, 0],
        });
        // No media: the 1080 tier of the 1920×1080 fallback, reshaped.
        assert_eq!(composite_canvas_size(&project), (1080, 1920));
    }

    #[test]
    fn canvas_config_carries_the_project_background() {
        use cutlass_models::{CanvasAspect, CanvasSettings};
        let mut project = Project::new("t", cutlass_models::Rational::FPS_24);
        assert_eq!(composite_canvas_config(&project).background, [0, 0, 0]);
        project.timeline_mut().set_canvas(CanvasSettings {
            aspect: CanvasAspect::Auto,
            background: [30, 60, 90],
        });
        let config = composite_canvas_config(&project);
        assert_eq!(config.background, [30, 60, 90]);
        assert_eq!((config.width, config.height), (1920, 1080));
    }

    #[test]
    fn resolve_shape_generator_is_cached_across_frames() {
        use cutlass_models::{Shape, TimeRange};
        let mut project = Project::new("t", cutlass_models::Rational::FPS_24);
        let track = project.add_track(TrackKind::Sticker, "ST1");
        project
            .add_generated(
                track,
                Generator::shape(Shape::Ellipse, [10, 200, 50, 255]),
                TimeRange::at_rate(0, 24, cutlass_models::Rational::FPS_24),
            )
            .unwrap();

        let mut pool = DecoderPool::new();
        let mut raster = GeneratorRaster::new();
        let canvas = CompositorConfig::new(160, 160);
        let resolve = |raster: &mut GeneratorRaster, pool: &mut DecoderPool, tick: i64| {
            resolve_layers(
                &project,
                None,
                pool,
                raster,
                RationalTime::new(tick, cutlass_models::Rational::FPS_24),
                0.0,
                &canvas,
                ColorConvertPath::Gpu,
                None,
                None,
            )
            .unwrap()
        };

        let first = resolve(&mut raster, &mut pool, 0);
        let second = resolve(&mut raster, &mut pool, 5);
        // Same generator + canvas on a later frame reuses the cached raster.
        assert!(std::sync::Arc::ptr_eq(
            rgba_layer(&first[0]),
            rgba_layer(&second[0])
        ));
    }
}
