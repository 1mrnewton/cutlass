//! Timeline layer resolution and decode helpers for GPU compositing.

use std::time::Instant;

use cutlass_cache::{FrameCache, SourceFingerprint};
use cutlass_compositor::{CompositeLayer, CompositorConfig, LayerPlacement};
use cutlass_decoder::DecodedFrame;
use cutlass_models::{
    Clip, ClipId, ClipTransform, Generator, ModelError, Project, RationalTime, TrackKind,
};
use tracing::debug;

use crate::ColorConvertPath;
use crate::decoder_pool::DecoderPool;
use crate::error::EngineError;
use crate::frame::{decoded_to_yuv_layer, legacy_decoded_to_rgba, RgbaFrame};
use crate::generator_raster::GeneratorRaster;

const DEFAULT_WIDTH: u32 = 1920;
const DEFAULT_HEIGHT: u32 = 1080;

/// Output canvas size: max media dimensions on the timeline, or 1920×1080 fallback.
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
            {
                max_w = max_w.max(media.width);
                max_h = max_h.max(media.height);
            }
        }
    }

    if max_w == 0 || max_h == 0 {
        (DEFAULT_WIDTH, DEFAULT_HEIGHT)
    } else {
        (to_even(max_w), to_even(max_h))
    }
}

/// Canvas placement for content of `content_w × content_h` under a clip
/// transform (CapCut semantics: scale 1.0 aspect-fits the content inside the
/// canvas, centered; position offsets are normalized to canvas dimensions;
/// rotation is degrees clockwise about the content center).
///
/// This is *the* geometry: the compositor draws it, and preview hit-testing
/// (preview roadmap Phase 2) inverts it — the two can never disagree.
pub fn layer_placement(
    transform: &ClipTransform,
    content_w: u32,
    content_h: u32,
    canvas: &CompositorConfig,
) -> LayerPlacement {
    let (cw, ch) = (canvas.width as f32, canvas.height as f32);
    let (w, h) = (content_w as f32, content_h as f32);
    let fit = if w > 0.0 && h > 0.0 {
        (cw / w).min(ch / h)
    } else {
        1.0
    };
    let scale = fit * transform.scale;
    LayerPlacement {
        center: [
            cw * 0.5 + transform.position[0] * cw,
            ch * 0.5 + transform.position[1] * ch,
        ],
        size: [w * scale, h * scale],
        rotation: transform.rotation.to_radians(),
        opacity: transform.opacity.clamp(0.0, 1.0),
    }
}

/// Resolve enabled video layers at `time`, bottom to top.
///
/// Pass `cache: Some(...)` for interactive preview (disk cache on hit). Pass
/// `None` for export so every media frame is decoded from the original source
/// file — never from cached YUV blobs (and never from future proxy paths).
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

        // Animated params sample at the clip-relative tick (M2): pure binary
        // search + lerp per property, allocation-free. A live gesture
        // override replaces the whole sampled value for this resolve only.
        let transform = match &override_transform {
            Some((id, t)) if *id == clip.id => *t,
            _ => clip.transform.sample(clip.animation_tick(time.value)),
        };
        let transform = &transform;
        match &clip.content {
            cutlass_models::ClipSource::Media { .. } => {
                let layer = match color_convert {
                    ColorConvertPath::Gpu => {
                        let decoded = decode_media_frame(project, cache, pool, clip, time)?;
                        let yuv = decoded_to_yuv_layer(&decoded)?;
                        let placement =
                            layer_placement(transform, yuv.width, yuv.height, canvas);
                        CompositeLayer::yuv420p(yuv, placement)
                    }
                    ColorConvertPath::LegacyCpu => {
                        // Native-size upload; the GPU scales it into place
                        // (the old CPU bilinear resize-to-canvas is gone).
                        let frame = decode_media_rgba_legacy(project, cache, pool, clip, time)?;
                        let placement =
                            layer_placement(transform, frame.width, frame.height, canvas);
                        CompositeLayer::rgba(
                            std::sync::Arc::new(frame.bytes),
                            frame.width,
                            frame.height,
                            placement,
                        )
                    }
                };
                layers.push(layer);
            }
            cutlass_models::ClipSource::Generated(generator) => {
                // A live inspector edit (e.g. font-size drag) renders this clip
                // from the override generator instead of its committed one.
                let generator = match override_generator {
                    Some((id, g)) if id == clip.id => g,
                    _ => generator,
                };
                // Generators raster at canvas size, so their fit is 1:1 and
                // the clip transform applies on top of the full canvas.
                let placement =
                    layer_placement(transform, canvas.width, canvas.height, canvas);
                match generator {
                    Generator::SolidColor { rgba } => {
                        layers.push(CompositeLayer::solid(*rgba, placement));
                    }
                    Generator::Text { .. } | Generator::Shape { .. } => {
                        match raster.raster(generator, canvas.width, canvas.height) {
                            Some(bytes) => layers.push(CompositeLayer::rgba(
                                bytes,
                                canvas.width,
                                canvas.height,
                                placement,
                            )),
                            None => {
                                debug!(?generator, "generator produced no raster");
                            }
                        }
                    }
                    Generator::Sticker
                    | Generator::Effect
                    | Generator::Filter
                    | Generator::Adjustment => {
                        debug!(?generator, "skipping unsupported generator for composite");
                    }
                }
            }
        }
    }

    Ok(layers)
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

    let start = Instant::now();
    if let Some(cache) = cache
        && let Some(packed) = cache.get(source_id, target_ticks)
    {
        debug!(
            us = start.elapsed().as_micros() as u64,
            pts = target_ticks,
            "preview frame cache hit"
        );
        return crate::frame::unpack_yuv420p(&packed, media.width, media.height);
    }

    let decoded = decoder
        .frame_at_ticks(target_ticks)?
        .ok_or_else(|| EngineError::Preview("decoder returned no frame".into()))?;
    debug!(
        ms = start.elapsed().as_secs_f64() * 1000.0,
        pts = decoded.pts_ticks,
        "decoded media frame"
    );

    if let Some(cache) = cache
        && let Ok(packed) = crate::frame::pack_yuv420p(&decoded)
    {
        cache.cache_frame(source_id, decoded.pts_ticks, packed);
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

    fn rgba_layer(layer: &CompositeLayer) -> &std::sync::Arc<Vec<u8>> {
        match &layer.content {
            LayerContent::Rgba { bytes, .. } => bytes,
            other => panic!("expected Rgba layer, got {other:?}"),
        }
    }

    // --- layer_placement ---------------------------------------------------

    const CANVAS: CompositorConfig = CompositorConfig {
        width: 1920,
        height: 1080,
    };

    #[test]
    fn identity_transform_on_canvas_sized_content_is_full_canvas() {
        let p = layer_placement(&ClipTransform::IDENTITY, 1920, 1080, &CANVAS);
        assert_eq!(p, LayerPlacement::full_canvas(&CANVAS));
    }

    #[test]
    fn mismatched_aspect_fits_inside_canvas() {
        // Portrait 1080×1920 into a 1920×1080 canvas: height-limited.
        let p = layer_placement(&ClipTransform::IDENTITY, 1080, 1920, &CANVAS);
        assert_eq!(p.center, [960.0, 540.0]);
        let fit = 1080.0 / 1920.0; // canvas_h / content_h
        assert_eq!(p.size, [1080.0 * fit, 1920.0 * fit]);
        assert!(p.size[1] <= 1080.0 + 1e-3);
    }

    #[test]
    fn position_scale_rotation_opacity_apply() {
        let t = ClipTransform {
            position: [0.25, -0.5],
            scale: 0.5,
            rotation: 90.0,
            opacity: 0.4,
        };
        let p = layer_placement(&t, 1920, 1080, &CANVAS);
        assert_eq!(p.center, [960.0 + 0.25 * 1920.0, 540.0 - 0.5 * 1080.0]);
        assert_eq!(p.size, [960.0, 540.0]);
        assert!((p.rotation - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        assert_eq!(p.opacity, 0.4);
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
                Generator::SolidColor { rgba: [255, 0, 0, 255] },
                // Clip starts at tick 12, not 0 — sampling must be clip-relative.
                TimeRange::at_rate(12, 48, rate),
            )
            .unwrap();
        // Opacity fades 0 → 1 over the first 24 clip ticks.
        project
            .set_param_keyframe(clip, ClipParam::Opacity, RT::new(12, rate), ParamValue::Scalar(0.0), Easing::Linear)
            .unwrap();
        project
            .set_param_keyframe(clip, ClipParam::Opacity, RT::new(36, rate), ParamValue::Scalar(1.0), Easing::Linear)
            .unwrap();

        let mut pool = DecoderPool::new();
        let mut raster = GeneratorRaster::new();
        let canvas = CompositorConfig::new(64, 64);
        let opacity_at = |pool: &mut DecoderPool, raster: &mut GeneratorRaster, tick: i64| {
            let layers = resolve_layers(
                &project,
                None,
                pool,
                raster,
                RationalTime::new(tick, rate),
                &canvas,
                ColorConvertPath::Gpu,
                None,
                None,
            )
            .unwrap();
            layers[0].placement.opacity
        };

        assert_eq!(opacity_at(&mut pool, &mut raster, 12), 0.0);
        assert_eq!(opacity_at(&mut pool, &mut raster, 24), 0.5);
        assert_eq!(opacity_at(&mut pool, &mut raster, 36), 1.0);
        // Past the last keyframe the value holds.
        assert_eq!(opacity_at(&mut pool, &mut raster, 50), 1.0);
    }

    #[test]
    fn resolve_shape_generator_is_cached_across_frames() {
        use cutlass_models::{Shape, TimeRange};
        let mut project = Project::new("t", cutlass_models::Rational::FPS_24);
        let track = project.add_track(TrackKind::Sticker, "ST1");
        project
            .add_generated(
                track,
                Generator::Shape {
                    shape: Shape::Ellipse,
                    rgba: [10, 200, 50, 255],
                },
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
