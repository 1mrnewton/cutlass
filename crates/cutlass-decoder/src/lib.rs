//! Media demux + decode.
//!
//! Video decode lives under [`video`]; [`audio`] covers waveform peak
//! extraction and clocked playback streaming ([`AudioReader`]).

// The FFmpeg demux/decode path and everything layered on it (audio DSP, stills,
// scaling, keyframe index). Compiled out on platforms that can't link libav*
// (iOS/Android), which use the platform-native backends below instead.
#[cfg(feature = "ffmpeg")]
pub mod audio;
mod error;
#[cfg(feature = "ffmpeg")]
pub mod ffmpeg_backend;
#[cfg(feature = "ffmpeg")]
pub mod image;
#[cfg(feature = "ffmpeg")]
pub mod video;

#[cfg(feature = "ffmpeg")]
pub use audio::{
    AUDIO_CHANNELS, AudioPeaks, AudioReader, CONTROL_HZ, DuckSettings, audio_peaks,
    audio_peaks_per_second, denoise_interleaved, detect_beats, duck_gain, onset_envelope,
    reduce_curve, render_denoised, render_stretched, render_stretched_curve, speech_band_energy,
};
pub use error::DecodeError;
#[cfg(feature = "ffmpeg")]
pub use ffmpeg_backend::{FfmpegDecoder, core_time_base, video_frame_from_decoded};
#[cfg(feature = "ffmpeg")]
pub use image::{STILL_MAX_DIM, decode_image};
#[cfg(feature = "ffmpeg")]
pub use video::{
    DecodeOptions, DecodedFrame, Decoder, HwAccel, KeyframeIndex, PixelFormat, Plane, SourceInfo,
    ThumbnailImage, attach_hwaccel, duration_to_ticks, ffmpeg_version, hw_accel_from_env,
    is_hardware_pixel_format, scale_yuv420p, ticks_to_duration, transfer_hw_frame_to_cpu,
    video_strip, video_thumbnail,
};

use tracing::info;

#[cfg(feature = "ffmpeg")]
pub fn init() {
    info!(
        ffmpeg = %ffmpeg_version(),
        "cutlass-decoder ready"
    );
}

/// FFmpeg-free build: nothing global to warm up; the native backends open
/// per-file. Kept so callers can `init()` unconditionally.
#[cfg(not(feature = "ffmpeg"))]
pub fn init() {
    info!("cutlass-decoder ready (native backends, no FFmpeg)");
}

// ---- Platform-native (FFmpeg-free) backends -------------------------------
//
// Additive to the FFmpeg path above: these implement the `cutlass_core` decode
// traits with each platform's hardware codec (no AVC/HEVC royalties, no FFmpeg
// link). Desktop still drives FFmpeg today; the mobile shells (and a future
// macOS hardware path) reach for these via `open_video_decoder`.

#[cfg(target_vendor = "apple")]
mod apple;
#[cfg(target_vendor = "apple")]
pub use apple::AvfDecoder;
#[cfg(target_vendor = "apple")]
mod apple_audio;
#[cfg(target_vendor = "apple")]
pub use apple_audio::AvfAudioReader;

#[cfg(target_os = "android")]
mod android;
#[cfg(target_os = "android")]
pub use android::AndroidDecoder;

#[cfg(target_os = "windows")]
mod wmf;
#[cfg(target_os = "windows")]
pub use wmf::WmfDecoder;

/// How a native decoder should deliver frame pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputMode {
    /// Zero-copy: hand back the decoder's native GPU surface
    /// ([`cutlass_core::FrameData::Gpu`]) for the renderer to import. The fast
    /// path for preview/playback.
    #[default]
    Gpu,
    /// Copy decoded pixels into CPU planes ([`cutlass_core::FrameData::Cpu`]).
    /// The portable path — export readback, tests, and platforms without
    /// zero-copy import.
    Cpu,
}

/// Open the platform-native video decoder for `path`, delivering frames in
/// `mode`. Returns [`cutlass_core::DecodeError::Unsupported`] on platforms
/// without a native backend yet (those keep using the FFmpeg [`Decoder`]).
pub fn open_video_decoder(
    path: &std::path::Path,
    mode: OutputMode,
) -> Result<Box<dyn cutlass_core::VideoDecoder>, cutlass_core::DecodeError> {
    #[cfg(target_vendor = "apple")]
    {
        Ok(Box::new(AvfDecoder::open(path, mode)?))
    }
    #[cfg(target_os = "android")]
    {
        Ok(Box::new(AndroidDecoder::open(path, mode)?))
    }
    #[cfg(target_os = "windows")]
    {
        Ok(Box::new(WmfDecoder::open(path, mode)?))
    }
    #[cfg(not(any(target_vendor = "apple", target_os = "android", target_os = "windows")))]
    {
        let _ = (path, mode);
        Err(cutlass_core::DecodeError::unsupported(
            "no native video decoder for this platform",
        ))
    }
}

/// Open the platform-native audio reader for `path`, decoding to interleaved
/// `f32` at `out_rate` Hz and `channels` channels. Returns
/// [`cutlass_core::DecodeError::Unsupported`] on platforms without a native
/// audio backend.
pub fn open_audio_reader(
    path: &std::path::Path,
    out_rate: u32,
    channels: u16,
) -> Result<Box<dyn cutlass_core::AudioReader>, cutlass_core::DecodeError> {
    #[cfg(target_vendor = "apple")]
    {
        Ok(Box::new(AvfAudioReader::open(path, out_rate, channels)?))
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (path, out_rate, channels);
        Err(cutlass_core::DecodeError::unsupported(
            "no native audio decoder for this platform",
        ))
    }
}
