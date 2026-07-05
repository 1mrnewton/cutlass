//! Platform-native video decoding for Cutlass.
//!
//! Cutlass never ships its own AVC/HEVC decoder (royalties + performance);
//! instead each platform's hardware codec is wrapped behind
//! [`cutlass_core::VideoDecoder`], so the decode *control* (open, seek, frame
//! selection) is shared Rust while only the codec call is native.
//!
//! - **Apple** ([`AvfDecoder`]): `AVAssetReader` does demux + VideoToolbox
//!   decode in one, handing back IOSurface-backed `CVPixelBuffer`s. Those map
//!   into a `wgpu` texture with no copy on the GPU path, or read back to planes
//!   on the CPU path.
//! - **Windows** ([`WmfDecoder`]): Media Foundation's Source Reader demuxes +
//!   decodes, handing back NV12 planes on the CPU path. (Zero-copy DXGI output
//!   is not wired up yet.)
//! - **Android** ([`MediaCodecDecoder`]): NDK `AMediaExtractor` demuxes and
//!   `AMediaCodec` hardware-decodes, handing back NV12/I420 planes on the CPU
//!   path. (Zero-copy `AHardwareBuffer` output is not wired up yet.)
//! - **Linux**: to follow (VAAPI or software).
//!
//! All backends share one `frame_at` seek policy (the `seek` module):
//! sequential and near-forward targets roll forward from the last emitted
//! frame instead of seeking, so playback and forward scrubbing never pay a
//! per-frame seek + GOP re-decode.

#[cfg(target_vendor = "apple")]
mod apple;
#[cfg(target_vendor = "apple")]
pub use apple::AvfDecoder;
#[cfg(target_vendor = "apple")]
mod apple_audio;
#[cfg(target_vendor = "apple")]
pub use apple_audio::AvfAudioReader;

#[cfg(target_os = "windows")]
mod wmf;
#[cfg(target_os = "windows")]
pub use wmf::WmfDecoder;
/// Fragmented-MP4 duration recovery, for demuxers that report none
/// (Media Foundation today; usable by any backend with the same gap).
#[cfg(target_os = "windows")]
mod mp4_duration;

#[cfg(target_os = "android")]
mod android;
#[cfg(target_os = "android")]
pub use android::{MediaCodecAudioReader, MediaCodecDecoder};

/// Shared `frame_at` roll-forward policy used by every backend.
#[cfg(any(target_vendor = "apple", target_os = "windows", target_os = "android"))]
mod seek;

mod probe;
pub use probe::{MediaProbe, probe};

mod peaks;
pub use peaks::{AudioPeaks, audio_peaks, audio_peaks_per_second};

pub mod image;
#[cfg(target_vendor = "apple")]
mod image_apple;
pub use image::{ImageFormat, ImageInfo, decode_image, probe_image, sniff_image};

use std::path::Path;

use cutlass_core::{AudioReader, DecodeError, VideoDecoder};

/// Open the platform-native video decoder for `path`, delivering frames in
/// `mode`.
///
/// Mirrors [`open_audio_reader`]: shared control flow, native codec only.
/// Returns [`DecodeError::Unsupported`] on platforms without a backend. Note
/// that Windows and Android currently ignore a [`OutputMode::Gpu`] request at
/// the decoder level (no zero-copy surface output yet); pass [`OutputMode::Cpu`]
/// there, or let the caller fall back.
pub fn open_video_decoder(
    path: &Path,
    mode: OutputMode,
) -> Result<Box<dyn VideoDecoder>, DecodeError> {
    #[cfg(target_vendor = "apple")]
    {
        Ok(Box::new(AvfDecoder::open(path, mode)?))
    }
    #[cfg(target_os = "windows")]
    {
        Ok(Box::new(WmfDecoder::open(path, mode)?))
    }
    #[cfg(target_os = "android")]
    {
        Ok(Box::new(MediaCodecDecoder::open(path, mode)?))
    }
    #[cfg(not(any(target_vendor = "apple", target_os = "windows", target_os = "android")))]
    {
        let _ = (path, mode);
        Err(DecodeError::unsupported(
            "no native video decoder for this platform",
        ))
    }
}

/// Open the platform-native audio reader for `path`, decoding to interleaved
/// `f32` at `out_rate` Hz and `channels` channels (the mixer's format).
///
/// Mirrors the video decoder's platform dispatch; returns
/// [`DecodeError::Unsupported`] on platforms without an audio backend.
pub fn open_audio_reader(
    path: &Path,
    out_rate: u32,
    channels: u16,
) -> Result<Box<dyn AudioReader>, DecodeError> {
    #[cfg(target_vendor = "apple")]
    {
        Ok(Box::new(AvfAudioReader::open(path, out_rate, channels)?))
    }
    #[cfg(target_os = "android")]
    {
        Ok(Box::new(MediaCodecAudioReader::open(
            path, out_rate, channels,
        )?))
    }
    #[cfg(not(any(target_vendor = "apple", target_os = "android")))]
    {
        let _ = (path, out_rate, channels);
        Err(DecodeError::unsupported(
            "no native audio decoder for this platform",
        ))
    }
}

/// How a decoder should deliver frame pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputMode {
    /// Zero-copy: hand back the decoder's native GPU surface
    /// ([`cutlass_core::FrameData::Gpu`]) for the renderer to import. The fast
    /// path for preview/playback.
    #[default]
    Gpu,
    /// Copy decoded pixels into CPU planes ([`cutlass_core::FrameData::Cpu`]).
    /// The portable path — used for export readback, tests, and platforms
    /// without zero-copy import.
    Cpu,
}
