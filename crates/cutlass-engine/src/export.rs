//! Timeline → composited RGBA frames → H.264 MP4.
//!
//! Export decodes media from original source files, composites on GPU, converts
//! to YUV420P on GPU, then encodes. It does not read or write the session frame
//! cache, and must not use proxy transcodes (preview-only when proxies land).

use std::path::Path;

use cutlass_compositor::{Compositor, CompositorError, GpuContext};
use cutlass_encoder::{ExportConfig, ExportStats, VideoExport};
use cutlass_models::{Project, RationalTime};
use tracing::info;

use crate::ColorConvertPath;
use crate::composite::composite_canvas_size;
use crate::decoder_pool::DecoderPool;
use crate::error::EngineError;
use crate::preview;

fn gpu_err(err: CompositorError) -> EngineError {
    EngineError::Export(err.to_string())
}

/// Build encoder settings from the project timeline rate and composite canvas.
pub fn export_config_for(project: &Project) -> Result<ExportConfig, EngineError> {
    let (width, height) = composite_canvas_size(project);
    let rate = project.timeline().frame_rate;
    if !rate.is_valid() {
        return Err(EngineError::Export("invalid timeline frame rate".into()));
    }
    Ok(ExportConfig {
        width,
        height,
        frame_rate_num: rate.num,
        frame_rate_den: rate.den,
        ..Default::default()
    })
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
    let frame_count = project.timeline().duration().value;
    if frame_count <= 0 {
        return Err(EngineError::Export("timeline has no content to export".into()));
    }

    let config = export_config_for(project)?;
    let rate = project.timeline().frame_rate;
    let mut sink = VideoExport::create(output, config)?;

    info!(
        frames = frame_count,
        width = config.width,
        height = config.height,
        path = %output.display(),
        "exporting timeline"
    );

    for tick in 0..frame_count {
        let time = RationalTime::new(tick, rate);
        let yuv = preview::get_export_yuv_frame(
            project,
            pool,
            gpu,
            compositor,
            time,
            color_convert,
        )?;
        if yuv.width != config.width || yuv.height != config.height {
            return Err(EngineError::Export(format!(
                "composite size {}x{} does not match export {}x{}",
                yuv.width, yuv.height, config.width, config.height
            )));
        }
        sink.push_yuv420p(yuv.width, yuv.height, &yuv.y, &yuv.u, &yuv.v)?;
    }

    sink.finish().map_err(Into::into)
}

/// Standalone export for background threads: owns its own GPU context and decoder pool.
pub fn export_project(
    project: &Project,
    output: &Path,
    color_convert: ColorConvertPath,
) -> Result<ExportStats, EngineError> {
    let gpu = GpuContext::new_headless_blocking().map_err(gpu_err)?;
    let mut compositor = Compositor::new(&gpu).map_err(gpu_err)?;
    let mut pool = DecoderPool::new();
    export_timeline(
        project,
        &mut pool,
        &gpu,
        &mut compositor,
        output,
        color_convert,
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
