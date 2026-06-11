//! Timeline → composited RGBA frames + mixed audio → H.264/AAC MP4.
//!
//! Export decodes media from original source files, composites on GPU, converts
//! to YUV420P on GPU, then encodes. It does not read or write the session frame
//! cache, and must not use proxy transcodes (preview-only when proxies land).
//! Audio follows the same rule: every audible span is decoded from its source
//! and mixed fresh (see [`crate::export_audio`]).

use std::path::Path;

use cutlass_compositor::{Compositor, CompositorError, GpuContext};
use cutlass_decoder::AUDIO_CHANNELS;
use cutlass_encoder::{ExportConfig, ExportStats, VideoExport, scaled_dims};
use cutlass_models::{Project, Rational, RationalTime};
use tracing::info;

use crate::ColorConvertPath;
use crate::composite::composite_canvas_size;
use crate::decoder_pool::DecoderPool;
use crate::error::EngineError;
use crate::export_audio::{EXPORT_AUDIO_RATE, ExportAudioMixer, sample_boundary};
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
    /// rounded to even (H.264 4:2:0) and never upscaled past the canvas.
    /// `None` ⇒ the composite canvas size.
    pub target_height: Option<u32>,
    /// Output frame rate. `None` ⇒ the timeline rate. Other rates resample
    /// by nearest-tick: output frame `n` composites the timeline frame under
    /// `n / fps` seconds.
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
        Some(target) => scaled_dims(canvas_w, canvas_h, target),
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
        ..defaults
    })
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
        return Err(EngineError::Export("timeline has no content to export".into()));
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

    // When the output rate exceeds the timeline rate, consecutive output
    // frames repeat a tick; keep the last composite so a repeat costs a
    // plane copy instead of a decode + GPU round-trip.
    let mut last: Option<(i64, cutlass_compositor::Yuv420pImage)> = None;
    // Audio streams in lockstep: after video frame `n`, the samples covering
    // `[boundary(n), boundary(n+1))` — so the muxer interleaves cleanly and
    // both tracks end at the same wall-clock instant.
    let mut audio_pos = 0i64;
    let mut audio_buf: Vec<f32> = Vec::new();
    for n in 0..out_frames {
        let tick = source_tick_for(n, tl_rate, out_rate).min(tl_frames - 1);
        if last.as_ref().map(|(t, _)| *t) != Some(tick) {
            let yuv = preview::get_export_yuv_frame(
                project,
                pool,
                gpu,
                compositor,
                RationalTime::new(tick, tl_rate),
                color_convert,
            )?;
            last = Some((tick, yuv));
        }
        let (_, yuv) = last.as_ref().expect("composite for tick was just stored");
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
    let mut pool = DecoderPool::new();
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
        let cfg = export_config_with(&project, ExportSettings {
            target_height: Some(720),
            fps: Some(Rational::FPS_30),
            quality: Some(23),
        })
        .unwrap();
        assert_eq!((cfg.width, cfg.height), (1280, 720));
        assert_eq!((cfg.frame_rate_num, cfg.frame_rate_den), (30, 1));
        assert_eq!(cfg.quality, 23);
    }

    #[test]
    fn settings_never_upscale_past_canvas() {
        let project = Project::new("test", Rational::FPS_24);
        let cfg = export_config_with(&project, ExportSettings {
            target_height: Some(2160),
            ..Default::default()
        })
        .unwrap();
        assert_eq!((cfg.width, cfg.height), (1920, 1080));
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
