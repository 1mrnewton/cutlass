//! Media probing: inspect files for container, codec, and stream metadata
//! without opening a full decode pipeline.
//!
//! Used at import time (duration, frame rate, resolution, audio layout) before
//! [`cutlass-decoder`] takes over for frame reads.

mod error;
mod media;
mod probe;

pub use error::ProbeError;
pub use media::MediaProbe;
pub use probe::{duration_ticks_from_micros, is_image_path, probe};

use tracing::info;

#[cfg(feature = "ffmpeg")]
pub fn init() {
    let _ = probe::ensure_ffmpeg_init();
    info!("cutlass-probe ready");
}

/// FFmpeg-free build: probing opens the native decoder per-file; nothing global.
#[cfg(not(feature = "ffmpeg"))]
pub fn init() {
    info!("cutlass-probe ready (native probe, no FFmpeg)");
}
