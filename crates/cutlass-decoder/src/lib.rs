//! Media demux + decode.
//!
//! Video decode lives under [`video`]; [`audio`] currently covers waveform
//! peak extraction (playback decode comes later).

pub mod audio;
mod error;
pub mod video;

pub use audio::{AudioPeaks, audio_peaks, audio_peaks_per_second};
pub use error::DecodeError;
pub use video::{
    DecodeOptions, DecodedFrame, Decoder, HwAccel, KeyframeIndex, PixelFormat, Plane, SourceInfo,
    ThumbnailImage, attach_hwaccel, duration_to_ticks, ffmpeg_version, hw_accel_from_env,
    is_hardware_pixel_format, ticks_to_duration, transfer_hw_frame_to_cpu, video_strip,
    video_thumbnail,
};

use tracing::info;

pub fn init() {
    info!(
        ffmpeg = %ffmpeg_version(),
        "cutlass-decoder ready"
    );
}
