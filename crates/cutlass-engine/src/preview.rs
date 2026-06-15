//! Timeline preview: resolve layers and composite via WGPU.

use std::time::Instant;

use cutlass_cache::FrameCache;
use cutlass_compositor::{Compositor, CompositorConfig, GpuContext, Yuv420pImage};
use cutlass_models::{ClipId, ClipTransform, Generator, ModelError, Project, RationalTime};
use tracing::debug;

use crate::ColorConvertPath;
use crate::composite::{composite_canvas_config, preview_canvas_config, resolve_layers};
use crate::decoder_pool::DecoderPool;
use crate::error::EngineError;
use crate::frame::RgbaFrame;
use crate::generator_raster::GeneratorRaster;

#[allow(clippy::too_many_arguments)]
pub fn get_frame(
    project: &Project,
    cache: &FrameCache,
    pool: &mut DecoderPool,
    raster: &mut GeneratorRaster,
    gpu: &GpuContext,
    compositor: &mut Compositor,
    time: RationalTime,
    color_convert: ColorConvertPath,
    override_transform: Option<(ClipId, ClipTransform)>,
    override_generator: Option<(ClipId, &Generator)>,
) -> Result<RgbaFrame, EngineError> {
    let tl_rate = project.timeline().frame_rate;
    if time.rate != tl_rate {
        return Err(ModelError::RateMismatch {
            expected: tl_rate,
            got: time.rate,
        }
        .into());
    }

    let config = preview_canvas_config(project);

    // Stage timings (playback roadmap Phase 2): resolve covers decode or
    // cache read; composite covers GPU submit + RGBA readback.
    let start = Instant::now();
    let layers = resolve_layers(
        project,
        Some(cache),
        pool,
        raster,
        time,
        0.0,
        &config,
        color_convert,
        override_transform,
        override_generator,
    )?;
    let resolve_ms = start.elapsed().as_secs_f64() * 1000.0;

    // A timeline gap isn't an error: the canvas composites bottom-up from
    // the background color, so zero layers is just the bare canvas. Skip
    // the GPU round-trip and hand back the solid background directly.
    if layers.is_empty() {
        return background_rgba_frame(&config);
    }

    let start = Instant::now();
    let image = compositor
        .composite(gpu, &config, &layers)
        .map_err(|e| EngineError::Preview(e.to_string()))?;
    debug!(
        resolve_ms,
        composite_ms = start.elapsed().as_secs_f64() * 1000.0,
        tick = time.value,
        "preview frame stages"
    );

    RgbaFrame::new(image.width, image.height, image.bytes)
}

/// Warm the decode path for `time` without compositing: resolve every layer
/// (sequential decode + cache fill) and drop the pixels. Playback read-ahead
/// (roadmap Phase 2) calls this for the ticks just past the playhead while
/// the worker is idle, so a GOP boundary's decode spike is paid *before* the
/// cadence reaches it instead of hitching a frame.
pub fn prefetch_frame(
    project: &Project,
    cache: &FrameCache,
    pool: &mut DecoderPool,
    raster: &mut GeneratorRaster,
    time: RationalTime,
    color_convert: ColorConvertPath,
) -> Result<(), EngineError> {
    let config = preview_canvas_config(project);
    resolve_layers(
        project,
        Some(cache),
        pool,
        raster,
        time,
        0.0,
        &config,
        color_convert,
        None,
        None,
    )?;
    Ok(())
}

/// Export frame path: no disk cache; returns GPU-composited YUV420P for encode.
///
/// `anim_phase` is the sub-frame animation sampling offset in timeline
/// ticks (see [`resolve_layers`]).
#[allow(clippy::too_many_arguments)]
pub fn get_export_yuv_frame(
    project: &Project,
    pool: &mut DecoderPool,
    raster: &mut GeneratorRaster,
    gpu: &GpuContext,
    compositor: &mut Compositor,
    time: RationalTime,
    anim_phase: f32,
    color_convert: ColorConvertPath,
) -> Result<Yuv420pImage, EngineError> {
    let tl_rate = project.timeline().frame_rate;
    if time.rate != tl_rate {
        return Err(ModelError::RateMismatch {
            expected: tl_rate,
            got: time.rate,
        }
        .into());
    }

    let config = composite_canvas_config(project);
    // Export never sees a gesture override: committed project state only.
    let layers = resolve_layers(
        project,
        None,
        pool,
        raster,
        time,
        anim_phase,
        &config,
        color_convert,
        None,
        None,
    )?;

    // Same gap policy as preview: a tick no clip covers exports as the
    // bare canvas background.
    if layers.is_empty() {
        return Ok(background_yuv420p(&config));
    }

    match color_convert {
        ColorConvertPath::Gpu => compositor
            .composite_yuv420p(gpu, &config, &layers)
            .map_err(|e| EngineError::Preview(e.to_string())),
        ColorConvertPath::LegacyCpu => {
            let image = compositor
                .composite(gpu, &config, &layers)
                .map_err(|e| EngineError::Preview(e.to_string()))?;
            Ok(cutlass_compositor::legacy_rgba_to_yuv420p(
                &image.bytes,
                image.width,
                image.height,
            ))
        }
    }
}

/// Opaque canvas background — what compositing zero layers produces, without
/// the GPU submit + readback.
fn background_rgba_frame(config: &CompositorConfig) -> Result<RgbaFrame, EngineError> {
    let [r, g, b] = config.background;
    let px = [r, g, b, 255];
    let count = config.width as usize * config.height as usize;
    let mut bytes = Vec::with_capacity(count * 4);
    for _ in 0..count {
        bytes.extend_from_slice(&px);
    }
    RgbaFrame::new(config.width, config.height, bytes)
}

/// The canvas background as limited-range YUV, matching what the GPU and
/// legacy CPU RGBA→YUV converters (BT.601) emit for that solid RGB frame.
fn background_yuv420p(config: &CompositorConfig) -> Yuv420pImage {
    let [r, g, b] = config.background.map(i32::from);
    let y = (((66 * r + 129 * g + 25 * b + 128) >> 8) + 16).clamp(16, 235) as u8;
    let u = (((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128).clamp(16, 240) as u8;
    let v = (((112 * r - 94 * g - 18 * b + 128) >> 8) + 128).clamp(16, 240) as u8;
    let (w, h) = (config.width as usize, config.height as usize);
    Yuv420pImage {
        width: config.width,
        height: config.height,
        y: vec![y; w * h],
        u: vec![u; (w / 2) * (h / 2)],
        v: vec![v; (w / 2) * (h / 2)],
    }
}
