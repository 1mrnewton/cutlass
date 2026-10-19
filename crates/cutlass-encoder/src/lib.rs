//! Video encode + mux.
//!
//! - [`proxy`]: transcode a source file into an all-intra H.264 proxy for fast seeks.
//! - [`export`]: push composited RGBA8 frames into an H.264 MP4 deliverable.

mod error;
#[cfg(feature = "ffmpeg")]
mod export;
#[cfg(feature = "ffmpeg")]
mod ffmpeg_backend;
#[cfg(feature = "ffmpeg")]
mod h264;
#[cfg(feature = "ffmpeg")]
mod proxy;

pub use error::EncodeError;
#[cfg(feature = "ffmpeg")]
pub use export::{AUDIO_CHANNELS, ExportConfig, ExportStats, VideoExport};
#[cfg(feature = "ffmpeg")]
pub use ffmpeg_backend::FfmpegEncoder;
#[cfg(feature = "ffmpeg")]
pub use proxy::{ProxyBuildOptions, ProxyConfig, ProxyStats, build_proxy, build_proxy_with};

use tracing::info;

#[cfg(feature = "ffmpeg")]
pub fn init() {
    let _ = h264::ensure_ffmpeg_init();
    info!(
        ffmpeg = %cutlass_decoder::ffmpeg_version(),
        "cutlass-encoder ready"
    );
}

/// FFmpeg-free build: the native encoder opens per-file, nothing to warm up.
#[cfg(not(feature = "ffmpeg"))]
pub fn init() {
    info!("cutlass-encoder ready (native backend, no FFmpeg)");
}

// ---- Platform-native (FFmpeg-free) backends -------------------------------
//
// Additive to the FFmpeg path above: implements the `cutlass_core` encode trait
// with each platform's hardware codec + native muxer. Desktop still drives
// FFmpeg today; the mobile shells reach for this via `open_encoder`.

#[cfg(target_vendor = "apple")]
mod apple;
#[cfg(target_vendor = "apple")]
pub use apple::AvfEncoder;

/// Open the platform's native encoder for `path` with `config`, behind the
/// [`cutlass_core::VideoEncoder`] trait so the export loop stays
/// platform-agnostic. The container is inferred from the path extension by the
/// backend (`.mp4` on Apple). Any existing file at `path` is overwritten.
pub fn open_encoder(
    path: &std::path::Path,
    config: cutlass_core::EncoderConfig,
) -> Result<Box<dyn cutlass_core::VideoEncoder>, cutlass_core::EncodeError> {
    #[cfg(target_vendor = "apple")]
    {
        Ok(Box::new(AvfEncoder::open(path, config)?))
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (path, config);
        Err(cutlass_core::EncodeError::unsupported(
            "no native video encoder for this platform yet",
        ))
    }
}
