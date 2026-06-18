//! Timeline → composited RGBA frames + mixed audio → H.264/AAC MP4.
//!
//! Export decodes media from original source files, composites on GPU, converts
//! to YUV420P on GPU, then encodes. It does not read or write the session frame
//! cache, and must not use proxy transcodes (preview-only when proxies land).
//! Audio follows the same rule: every audible span is decoded from its source
//! and mixed fresh (see [`crate::export_audio`]).

use std::path::Path;

use cutlass_compositor::{Compositor, CompositorError, GpuContext};
use cutlass_decoder::{AUDIO_CHANNELS, HwAccel};
use cutlass_encoder::{ExportConfig, ExportStats, VideoExport};
use cutlass_models::{Project, Rational, RationalTime};
use tracing::info;

use crate::ColorConvertPath;
use crate::composite::composite_canvas_size;
use crate::decoder_pool::DecoderPool;
use crate::error::EngineError;
use crate::export_audio::{EXPORT_AUDIO_RATE, ExportAudioMixer, sample_boundary};
use crate::generator_raster::GeneratorRaster;
use crate::preview;

fn gpu_err(err: CompositorError) -> EngineError {
    EngineError::Export(err.to_string())
}

/// User-tunable export settings (the export dialog). Every field falls back
/// to the project's own values, so `default()` reproduces the plain
/// canvas-size, timeline-rate export.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExportSettings {
    /// Target output height; width follows the canvas aspect ratio, both
    /// rounded to even (H.264 4:2:0). Honored exactly — picking 2160p over
    /// a 1080p canvas upscales, like CapCut. `None` ⇒ the composite canvas
    /// size.
    pub target_height: Option<u32>,
    /// Output frame rate. `None` ⇒ the timeline rate. Other rates resample
    /// by nearest-tick: output frame `n` composites the timeline frame under
    /// `n / fps` seconds. Animated clip transforms sample at the exact
    /// output frame time (sub-frame), so rates above the timeline rate
    /// genuinely smooth keyframed motion.
    pub fps: Option<Rational>,
    /// Constant-quality level (libx264 CRF, 0–51, lower = better).
    /// `None` ⇒ the encoder default (18, visually near-transparent).
    pub quality: Option<u8>,
}

/// Per-frame export progress: `(frames_done, frames_total)`. Return `false`
/// to abort the export ([`EngineError::ExportCancelled`]).
pub type ExportProgress<'a> = &'a mut dyn FnMut(u64, u64) -> bool;

/// Build encoder settings from the project timeline rate and composite canvas.
pub fn export_config_for(project: &Project) -> Result<ExportConfig, EngineError> {
    export_config_with(project, ExportSettings::default())
}

/// [`export_config_for`] with dialog overrides applied.
pub fn export_config_with(
    project: &Project,
    settings: ExportSettings,
) -> Result<ExportConfig, EngineError> {
    let (canvas_w, canvas_h) = composite_canvas_size(project);
    let rate = project.timeline().frame_rate;
    if !rate.is_valid() {
        return Err(EngineError::Export("invalid timeline frame rate".into()));
    }
    let out_rate = settings.fps.unwrap_or(rate);
    if !out_rate.is_valid() {
        return Err(EngineError::Export("invalid export frame rate".into()));
    }
    let (width, height) = match settings.target_height {
        Some(target) => export_dims(canvas_w, canvas_h, target),
        // The canvas follows media dimensions, which may be odd; H.264
        // 4:2:0 needs even, so the no-preset path still rounds.
        None => (canvas_w.max(2) & !1, canvas_h.max(2) & !1),
    };
    let defaults = ExportConfig::default();
    Ok(ExportConfig {
        width,
        height,
        frame_rate_num: out_rate.num,
        frame_rate_den: out_rate.den,
        quality: settings.quality.unwrap_or(defaults.quality),
        // Prefer the platform hardware H.264 encoder (VideoToolbox / NVENC /
        // …): far faster on big 4K timelines. The encoder selection falls back
        // to software libx264 (CRF `quality`) when no hardware encoder opens,
        // so this is safe everywhere. Hardware uses bitrate, not CRF, hence the
        // resolution-aware target below.
        hardware: true,
        bitrate: target_bitrate(width, height, out_rate),
        ..defaults
    })
}

/// Target H.264 bitrate (bits/sec) for the hardware encoder, which — unlike
/// libx264's constant-quality CRF — needs an explicit rate. Scales with pixels
/// and frame rate at ~0.15 bits/pixel/frame, the high-quality-upload ballpark
/// (≈9 Mb/s for 1080p30, ≈37 Mb/s for 4K30, ≈75 Mb/s for 4K60), with a floor
/// so tiny frames still get a usable rate. Ignored when the encoder falls back
/// to software CRF.
fn target_bitrate(width: u32, height: u32, rate: Rational) -> usize {
    const BITS_PER_PIXEL: f64 = 0.15;
    const FLOOR: usize = 2_000_000;
    let fps = f64::from(rate.num) / f64::from(rate.den.max(1));
    let bits = f64::from(width) * f64::from(height) * fps * BITS_PER_PIXEL;
    (bits as usize).max(FLOOR)
}

/// Output dimensions for a resolution preset: exactly `target_h` tall (up
/// *or* down from the canvas — the user's pick wins), width following the
/// canvas aspect ratio, both rounded to even for H.264 4:2:0. Unlike the
/// proxy path's `scaled_dims`, this never clamps to the source size.
fn export_dims(canvas_w: u32, canvas_h: u32, target_h: u32) -> (u32, u32) {
    if canvas_w == 0 || canvas_h == 0 {
        return (canvas_w.max(2) & !1, target_h.max(2) & !1);
    }
    let h = target_h.max(2);
    let w = ((canvas_w as u64 * h as u64) / canvas_h as u64) as u32;
    (w.max(2) & !1, h & !1)
}

/// Output frames for `tl_frames` timeline ticks resampled from `tl` to `out`
/// frames per second: the count covering the same wall-clock span, rounded up
/// so trailing content is never dropped.
fn output_frame_count(tl_frames: i64, tl: Rational, out: Rational) -> i64 {
    // All factors are positive (rates validated, tl_frames > 0), so the
    // manual ceiling is exact. Signed `div_ceil` is still unstable.
    let num = tl_frames as i128 * tl.den as i128 * out.num as i128;
    let den = tl.num as i128 * out.den as i128;
    ((num + den - 1) / den) as i64
}

/// Timeline tick composited for output frame `n`: the frame the timeline
/// shows at `n / out` seconds (floor — mid-frame time shows the started
/// frame, same convention as the playback transport).
fn source_tick_for(n: i64, tl: Rational, out: Rational) -> i64 {
    let num = n as i128 * out.den as i128 * tl.num as i128;
    let den = out.num as i128 * tl.den as i128;
    (num / den) as i64
}

/// Fraction of a timeline tick by which output frame `n`'s exact time sits
/// past `tick` — the sub-frame phase animated clip transforms sample at.
/// Zero whenever the rates line up; can exceed 1.0 only for trailing frames
/// whose tick was clamped to the timeline end (per-clip clamping holds the
/// last value there).
fn source_phase_for(n: i64, tl: Rational, out: Rational, tick: i64) -> f32 {
    let num = n as i128 * out.den as i128 * tl.num as i128;
    let den = out.num as i128 * tl.den as i128;
    ((num - tick as i128 * den) as f64 / den as f64) as f32
}

/// Composite every timeline frame `0..duration` and mux to `output`.
pub fn export_timeline(
    project: &Project,
    pool: &mut DecoderPool,
    gpu: &GpuContext,
    compositor: &mut Compositor,
    output: &Path,
    color_convert: ColorConvertPath,
) -> Result<ExportStats, EngineError> {
    export_timeline_with(
        project,
        pool,
        gpu,
        compositor,
        output,
        color_convert,
        ExportSettings::default(),
        &mut |_, _| true,
    )
}

/// [`export_timeline`] with dialog settings and per-frame progress/cancel.
///
/// `progress` is called once with `(0, total)` before the first frame and
/// after every encoded frame; returning `false` aborts with
/// [`EngineError::ExportCancelled`] (the partial file is left on disk for the
/// caller to clean up).
#[allow(clippy::too_many_arguments)]
pub fn export_timeline_with(
    project: &Project,
    pool: &mut DecoderPool,
    gpu: &GpuContext,
    compositor: &mut Compositor,
    output: &Path,
    color_convert: ColorConvertPath,
    settings: ExportSettings,
    progress: ExportProgress<'_>,
) -> Result<ExportStats, EngineError> {
    let tl_frames = project.timeline().duration().value;
    if tl_frames <= 0 {
        return Err(EngineError::Export(
            "timeline has no content to export".into(),
        ));
    }

    let mut mixer = ExportAudioMixer::for_project(project);
    let mut config = export_config_with(project, settings)?;
    config.audio_rate = mixer.as_ref().map(|_| EXPORT_AUDIO_RATE);
    let tl_rate = project.timeline().frame_rate;
    let out_rate = settings.fps.unwrap_or(tl_rate);
    let out_frames = output_frame_count(tl_frames, tl_rate, out_rate);
    let mut sink = VideoExport::create(output, config)?;

    info!(
        frames = out_frames,
        width = config.width,
        height = config.height,
        fps_num = config.frame_rate_num,
        fps_den = config.frame_rate_den,
        crf = config.quality,
        audio = mixer.is_some(),
        path = %output.display(),
        "exporting timeline"
    );

    if !progress(0, out_frames as u64) {
        return Err(EngineError::ExportCancelled);
    }

    // Sub-frame animation phases only matter when something is keyframed;
    // a fully static timeline keeps phase at zero so repeated ticks reuse
    // the composite below.
    let animated = project
        .timeline()
        .tracks_ordered()
        .any(|t| t.clips().any(|c| c.transform.is_animated()));

    // When the output rate exceeds the timeline rate, consecutive output
    // frames repeat a (tick, phase) pair; keep the last composite so a
    // repeat costs a plane copy instead of a decode + GPU round-trip.
    let mut last: Option<(i64, f32, cutlass_compositor::Yuv420pImage)> = None;
    // Generator rasters (text, shapes) are cached per export; a static title
    // composites once, not once per frame.
    let mut raster = GeneratorRaster::new();
    // Audio streams in lockstep: after video frame `n`, the samples covering
    // `[boundary(n), boundary(n+1))` — so the muxer interleaves cleanly and
    // both tracks end at the same wall-clock instant.
    let mut audio_pos = 0i64;
    let mut audio_buf: Vec<f32> = Vec::new();
    for n in 0..out_frames {
        let tick = source_tick_for(n, tl_rate, out_rate).min(tl_frames - 1);
        let phase = if animated {
            source_phase_for(n, tl_rate, out_rate, tick)
        } else {
            0.0
        };
        if last.as_ref().map(|(t, p, _)| (*t, *p)) != Some((tick, phase)) {
            let yuv = preview::get_export_yuv_frame(
                project,
                pool,
                &mut raster,
                gpu,
                compositor,
                RationalTime::new(tick, tl_rate),
                phase,
                color_convert,
            )?;
            last = Some((tick, phase, yuv));
        }
        let (_, _, yuv) = last.as_ref().expect("composite for tick was just stored");
        sink.push_yuv420p(yuv.width, yuv.height, &yuv.y, &yuv.u, &yuv.v)?;

        if let Some(mixer) = &mut mixer {
            let next = sample_boundary(n + 1, out_rate.num, out_rate.den);
            let frames = (next - audio_pos).max(0) as usize;
            audio_buf.resize(frames * AUDIO_CHANNELS, 0.0);
            mixer.mix_into(audio_pos, &mut audio_buf)?;
            sink.push_audio(&audio_buf)?;
            audio_pos = next;
        }

        if !progress(n as u64 + 1, out_frames as u64) {
            return Err(EngineError::ExportCancelled);
        }
    }

    sink.finish().map_err(Into::into)
}

/// Standalone export for background threads: owns its own GPU context and decoder pool.
pub fn export_project(
    project: &Project,
    output: &Path,
    color_convert: ColorConvertPath,
) -> Result<ExportStats, EngineError> {
    export_project_with(
        project,
        output,
        color_convert,
        ExportSettings::default(),
        &mut |_, _| true,
    )
}

/// [`export_project`] with dialog settings and per-frame progress/cancel —
/// what the UI's export job runs on its own thread.
pub fn export_project_with(
    project: &Project,
    output: &Path,
    color_convert: ColorConvertPath,
    settings: ExportSettings,
    progress: ExportProgress<'_>,
) -> Result<ExportStats, EngineError> {
    let gpu = GpuContext::new_headless_blocking().map_err(gpu_err)?;
    let mut compositor = Compositor::new(&gpu).map_err(gpu_err)?;
    // Hardware decode (probe the platform media engine, fall back to software):
    // its own pool with no frame cache, so NV12 hardware frames never reach the
    // preview cache's YUV420P pack path.
    let mut pool = DecoderPool::with_hw_accel(HwAccel::Auto);
    export_timeline_with(
        project,
        &mut pool,
        &gpu,
        &mut compositor,
        output,
        color_convert,
        settings,
        progress,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_models::{Rational, TrackKind};

    #[test]
    fn export_config_uses_timeline_rate_and_canvas_fallback() {
        let project = Project::new("test", Rational::FPS_24);
        let cfg = export_config_for(&project).unwrap();
        assert_eq!(cfg.width, 1920);
        assert_eq!(cfg.height, 1080);
        assert_eq!(cfg.frame_rate_num, 24);
        assert_eq!(cfg.frame_rate_den, 1);
    }

    #[test]
    fn settings_override_resolution_fps_and_quality() {
        let project = Project::new("test", Rational::FPS_24);
        let cfg = export_config_with(
            &project,
            ExportSettings {
                target_height: Some(720),
                fps: Some(Rational::FPS_30),
                quality: Some(23),
            },
        )
        .unwrap();
        assert_eq!((cfg.width, cfg.height), (1280, 720));
        assert_eq!((cfg.frame_rate_num, cfg.frame_rate_den), (30, 1));
        assert_eq!(cfg.quality, 23);
    }

    #[test]
    fn settings_upscale_to_requested_resolution() {
        // 4K preset over the default 1080p canvas: the pick wins (CapCut
        // behavior), so the file really is 3840×2160.
        let project = Project::new("test", Rational::FPS_24);
        let cfg = export_config_with(
            &project,
            ExportSettings {
                target_height: Some(2160),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!((cfg.width, cfg.height), (3840, 2160));
    }

    #[test]
    fn export_prefers_hardware_with_resolution_scaled_bitrate() {
        let project = Project::new("test", Rational::FPS_24);
        let cfg = export_config_with(
            &project,
            ExportSettings {
                target_height: Some(2160),
                fps: Some(Rational::FPS_30),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(cfg.hardware);
        // 4K30 lands near the YouTube high-quality upload ballpark, not the
        // 12 Mb/s default that would crush a 4K render.
        assert_eq!(cfg.bitrate, target_bitrate(3840, 2160, Rational::FPS_30));
        assert!(
            cfg.bitrate > 30_000_000,
            "4K30 bitrate too low: {}",
            cfg.bitrate
        );
    }

    #[test]
    fn target_bitrate_scales_and_floors() {
        // Scales with pixels and fps.
        assert!(
            target_bitrate(3840, 2160, Rational::FPS_60)
                > target_bitrate(3840, 2160, Rational::FPS_30)
        );
        assert!(
            target_bitrate(3840, 2160, Rational::FPS_30)
                > target_bitrate(1920, 1080, Rational::FPS_30)
        );
        // Tiny frame still gets the floor.
        assert_eq!(target_bitrate(2, 2, Rational::FPS_24), 2_000_000);
    }

    #[test]
    fn export_dims_follow_aspect_and_round_even() {
        // Down, up, and odd-aspect rounding.
        assert_eq!(export_dims(1920, 1080, 540), (960, 540));
        assert_eq!(export_dims(1920, 1080, 2160), (3840, 2160));
        // 1280×720 → 480 tall: 853.33 wide floors to 853, rounds even to 852.
        assert_eq!(export_dims(1280, 720, 480), (852, 480));
        // Degenerate canvas still yields legal even dims.
        assert_eq!(export_dims(0, 0, 720), (2, 720));
    }

    #[test]
    fn frame_count_resamples_across_rates() {
        let tl = Rational::FPS_24;
        // Same rate: identity.
        assert_eq!(output_frame_count(48, tl, tl), 48);
        // Halve the rate: half the frames.
        assert_eq!(output_frame_count(48, tl, Rational::new(12, 1)), 24);
        // 24 → 30: same wall-clock span at more frames.
        assert_eq!(output_frame_count(48, tl, Rational::FPS_30), 60);
        // Non-integer span rounds up so trailing content is kept.
        assert_eq!(output_frame_count(25, tl, Rational::new(12, 1)), 13);
    }

    #[test]
    fn source_tick_maps_output_frames_to_timeline_frames() {
        let tl = Rational::FPS_24;
        // Same rate: identity.
        assert_eq!(source_tick_for(7, tl, tl), 7);
        // Half rate: every other timeline frame.
        assert_eq!(source_tick_for(5, tl, Rational::new(12, 1)), 10);
        // Faster output repeats ticks (60 out of 24: 0,0,0,1,1,2,…).
        let out = Rational::FPS_60;
        let ticks: Vec<i64> = (0..6).map(|n| source_tick_for(n, tl, out)).collect();
        assert_eq!(ticks, vec![0, 0, 0, 1, 1, 2]);
    }

    #[test]
    fn source_phase_tracks_exact_output_time() {
        let tl = Rational::FPS_24;
        // Same rate: every output frame sits exactly on its tick.
        for n in 0..5 {
            assert_eq!(source_phase_for(n, tl, tl, source_tick_for(n, tl, tl)), 0.0);
        }
        // 24 → 60: output frames land 0.4 ticks apart, so the repeated
        // ticks carry distinct sub-frame phases (smooth motion at 60 Hz).
        let out = Rational::FPS_60;
        let phases: Vec<f32> = (0..6)
            .map(|n| {
                let tick = source_tick_for(n, tl, out);
                source_phase_for(n, tl, out, tick)
            })
            .collect();
        let expect = [0.0, 0.4, 0.8, 0.2, 0.6, 0.0];
        for (got, want) in phases.iter().zip(expect) {
            assert!((got - want).abs() < 1e-6, "{phases:?}");
        }
        // A tail frame whose tick was clamped samples past the tick; the
        // per-clip clamp holds the last value.
        let clamped = source_phase_for(5, tl, out, 1);
        assert!((clamped - 1.0).abs() < 1e-6);
    }

    #[test]
    fn export_config_matches_media_dimensions() {
        let mut project = Project::new("test", Rational::FPS_24);
        let media_id = project.add_media(cutlass_models::MediaSource::new(
            "/tmp/x.mp4",
            1280,
            720,
            Rational::FPS_24,
            100,
            false,
        ));
        let track = project.add_track(TrackKind::Video, "V1");
        project
            .add_clip(
                track,
                media_id,
                cutlass_models::TimeRange::at_rate(0, 48, Rational::FPS_24),
                RationalTime::new(0, Rational::FPS_24),
            )
            .unwrap();
        let cfg = export_config_for(&project).unwrap();
        assert_eq!(cfg.width, 1280);
        assert_eq!(cfg.height, 720);
    }
}
