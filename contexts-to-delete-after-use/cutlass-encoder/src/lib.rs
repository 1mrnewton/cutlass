//! cutlass-encoder: platform-native video encoding behind
//! [`cutlass_core::VideoEncoder`].
//!
//! The mirror of `cutlass-decoder`: the renderer composites the timeline to
//! frames, and a native encoder muxes them to a file. Like decode, the codec
//! call is platform-native (VideoToolbox on Apple, Media Foundation on Windows,
//! MediaCodec on Android) while the *control* — frames in, file out — stays in
//! Rust behind the trait, so desktop, mobile, and the Python bindings share one
//! export path.
//!
//! The **Apple** backend ([`AvfEncoder`]) uses `AVAssetWriter` (VideoToolbox
//! H.264 + AAC, mp4 mux); the **Android** backend ([`MediaCodecEncoder`]) uses
//! MediaCodec (H.264 + AAC) with `AMediaMuxer`. Other platforms return
//! [`EncodeError::Unsupported`] from [`open_encoder`] until their backends land.

use std::path::Path;

use cutlass_core::{EncodeError, EncoderConfig, VideoEncoder};

#[cfg(target_vendor = "apple")]
mod apple;
#[cfg(target_vendor = "apple")]
pub use apple::AvfEncoder;

#[cfg(target_os = "android")]
mod android;
#[cfg(target_os = "android")]
pub use android::MediaCodecEncoder;

// RGBA→I420 conversion lives outside the Android cfg so it's unit-testable on
// the host; the Android backend is its only consumer today.
mod yuv;

/// Open the platform's native encoder for `path` with `config`, returning it
/// behind the [`VideoEncoder`] trait so the export loop is platform-agnostic.
///
/// The container is inferred from the path extension by the backend (`.mp4` on
/// Apple). Any existing file at `path` is overwritten.
pub fn open_encoder(
    path: &Path,
    config: EncoderConfig,
) -> Result<Box<dyn VideoEncoder>, EncodeError> {
    #[cfg(target_vendor = "apple")]
    {
        Ok(Box::new(AvfEncoder::open(path, config)?))
    }
    #[cfg(target_os = "android")]
    {
        Ok(Box::new(MediaCodecEncoder::open(path, config)?))
    }
    #[cfg(not(any(target_vendor = "apple", target_os = "android")))]
    {
        let _ = (path, config);
        Err(EncodeError::unsupported(
            "no native video encoder for this platform yet",
        ))
    }
}
