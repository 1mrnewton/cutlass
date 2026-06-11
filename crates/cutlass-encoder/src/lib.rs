//! Video encode + mux.
//!
//! - [`proxy`]: transcode a source file into an all-intra H.264 proxy for fast seeks.
//! - [`export`]: push composited RGBA8 frames into an H.264 MP4 deliverable.

mod error;
mod export;
mod h264;
mod proxy;

pub use error::EncodeError;
pub use export::{AUDIO_CHANNELS, ExportConfig, ExportStats, VideoExport};
pub use h264::scaled_dims;
pub use proxy::{ProxyBuildOptions, ProxyConfig, ProxyStats, build_proxy, build_proxy_with};

use tracing::info;

pub fn init() {
    let _ = h264::ensure_ffmpeg_init();
    info!(
        ffmpeg = %cutlass_decoder::ffmpeg_version(),
        "cutlass-encoder ready"
    );
}
