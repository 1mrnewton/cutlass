//! The encode contract (export seam).
//!
//! The mirror image of [`crate::decode`]: the compositor renders the timeline
//! to frames, and a native encoder muxes them to a file. Like decode, the codec
//! call is platform-native (VideoToolbox / MediaCodec / Media Foundation, or a
//! software fallback) while control stays in Rust behind [`VideoEncoder`]. This
//! is the "composite every frame `0..duration` → mux to mp4" path that the
//! engine, the future Rust exporter, and the Python bindings all build on.
//!
//! This is deliberately a thin seam for now — enough to pin the architecture
//! (frames in, file out, codec behind the trait); bitrate/quality/codec
//! selection will grow on [`EncoderConfig`] as the exporter lands.

use crate::audio::AudioEncoderConfig;
use crate::color::ColorSpace;
use crate::frame::VideoFrame;
use crate::time::{Rational, RationalTime};

/// An encode failure, backend-agnostic across native and software encoders.
#[derive(Debug, thiserror::Error)]
pub enum EncodeError {
    /// Creating the encoder or output file failed.
    #[error("failed to start encoder: {0}")]
    Start(String),
    /// Encoding or muxing a frame failed.
    #[error("encode failed: {0}")]
    Encode(String),
    /// The requested output format / codec / pixel format isn't supported.
    #[error("unsupported: {0}")]
    Unsupported(String),
}

impl EncodeError {
    pub fn unsupported(msg: impl Into<String>) -> Self {
        EncodeError::Unsupported(msg.into())
    }
}

/// Output configuration for an encode session.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderConfig {
    /// Output frame size in pixels.
    pub size: (u32, u32),
    /// Output frame rate.
    pub frame_rate: Rational,
    /// Colorimetry to tag the output with.
    pub color: ColorSpace,
    /// Audio track configuration. `None` ⇒ a video-only file (the encoder adds
    /// no audio track and [`push_audio`](VideoEncoder::push_audio) is unused).
    pub audio: Option<AudioEncoderConfig>,
    /// Maximum keyframe (sync-point) spacing in frames. `None` leaves the
    /// codec's default GOP policy (typically seconds-long for delivery).
    /// Preview proxies set a short interval (~15) so scrub seeks decode at
    /// most a few frames to reach any target.
    #[cfg_attr(feature = "serde", serde(default))]
    pub keyframe_interval: Option<u32>,
}

impl EncoderConfig {
    /// A video-only encode config (no audio track).
    pub fn new(size: (u32, u32), frame_rate: Rational, color: ColorSpace) -> Self {
        Self {
            size,
            frame_rate,
            color,
            audio: None,
            keyframe_interval: None,
        }
    }

    /// Add an audio track to this config.
    pub fn with_audio(mut self, audio: AudioEncoderConfig) -> Self {
        self.audio = Some(audio);
        self
    }

    /// Cap the keyframe (sync-point) spacing at `frames`.
    pub fn with_keyframe_interval(mut self, frames: u32) -> Self {
        self.keyframe_interval = Some(frames);
        self
    }
}

/// A push-based video encoder.
///
/// Push composited frames in presentation order, then [`finish`](VideoEncoder::finish)
/// to flush and finalize the container. `Send` so the encoder can run on an
/// export worker thread.
pub trait VideoEncoder: Send {
    /// Encode and mux one frame. Frames are expected in presentation order.
    fn push(&mut self, frame: &VideoFrame) -> Result<(), EncodeError>;

    /// Encode and mux one block of interleaved `f32` audio at `pts`, in
    /// presentation order alongside the video [`push`](VideoEncoder::push)es.
    ///
    /// `samples` is `frames * channels` long, matching the
    /// [`AudioEncoderConfig`] the encoder was opened with. Only meaningful when
    /// the config carried `Some(audio)`; the default implementation errors, so
    /// audio-free encoders (and the PNG sink) need not implement it.
    fn push_audio(&mut self, samples: &[f32], pts: RationalTime) -> Result<(), EncodeError> {
        let _ = (samples, pts);
        Err(EncodeError::unsupported(
            "this encoder has no audio track (config.audio was None)",
        ))
    }

    /// Flush buffered frames and finalize the output. The encoder must not be
    /// used after this returns.
    fn finish(&mut self) -> Result<(), EncodeError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_config_roundtrips_fields() {
        let cfg = EncoderConfig::new((1920, 1080), Rational::FPS_30, ColorSpace::BT709);
        assert_eq!(cfg.size, (1920, 1080));
        assert_eq!(cfg.frame_rate, Rational::FPS_30);
        assert_eq!(cfg.color, ColorSpace::BT709);
    }

    #[test]
    fn encode_error_unsupported_helper() {
        let e = EncodeError::unsupported("prores not wired up");
        assert_eq!(e.to_string(), "unsupported: prores not wired up");
    }
}
