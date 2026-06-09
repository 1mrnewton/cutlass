//! Timeline preview: resolve layers and composite via WGPU.

use cutlass_compositor::{Compositor, GpuContext};
use cutlass_cache::FrameCache;
use cutlass_models::{ModelError, Project, RationalTime};

use crate::composite::{composite_canvas_size, resolve_layers};
use crate::decoder_pool::DecoderPool;
use crate::error::EngineError;
use crate::frame::RgbaFrame;

pub fn get_frame(
    project: &Project,
    cache: &FrameCache,
    pool: &mut DecoderPool,
    gpu: &GpuContext,
    compositor: &mut Compositor,
    time: RationalTime,
) -> Result<RgbaFrame, EngineError> {
    let tl_rate = project.timeline().frame_rate;
    if time.rate != tl_rate {
        return Err(ModelError::RateMismatch {
            expected: tl_rate,
            got: time.rate,
        }
        .into());
    }

    let (width, height) = composite_canvas_size(project);
    let config = cutlass_compositor::CompositorConfig::new(width, height);
    let layers = resolve_layers(project, cache, pool, time, &config)?;

    if layers.is_empty() {
        return Err(EngineError::Preview("no video at timeline position".into()));
    }

    let image = compositor
        .composite(gpu, &config, &layers)
        .map_err(|e| EngineError::Preview(e.to_string()))?;

    RgbaFrame::new(image.width, image.height, image.bytes)
}
