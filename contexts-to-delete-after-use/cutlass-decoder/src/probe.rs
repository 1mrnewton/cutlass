//! Lightweight media probing: open the native decoder, read its [`SourceInfo`],
//! and drop it — no frames are decoded.
//!
//! This is the FFmpeg-free replacement for container probing. Every platform
//! decoder reports its source's dimensions, frame rate, and duration at open
//! time, which is all the media pool needs to register a clip.

use std::path::Path;

use cutlass_core::{DecodeError, Rational, SourceInfo, VideoDecoder, resample};

use crate::OutputMode;

/// Static metadata for a media file, read without decoding any frames.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaProbe {
    /// Visible frame size in pixels, after applying container rotation.
    pub width: u32,
    pub height: u32,
    /// Native (nominal) frame rate.
    pub frame_rate: Rational,
    /// Source length in whole frames at [`frame_rate`](Self::frame_rate); `0`
    /// when the container reports no duration.
    pub frame_count: i64,
    /// Whether the source carries an audio track, detected per-platform
    /// ([`has_audio_track`]). The [`VideoDecoder`] surfaces only the video
    /// track, so audio presence is queried from the demuxer separately.
    pub has_audio: bool,
}

impl MediaProbe {
    fn from_info(info: &SourceInfo, has_audio: bool) -> Self {
        let (width, height) = info.display_size_rotated();
        // Reduce the container's timescale-derived rate (e.g. `600/25`) to its
        // canonical form (`24/1`) so it matches source ranges authored at the
        // `FPS_*` constants under structural rate equality.
        let frame_rate = info.frame_rate.reduced();
        let frame_count = info
            .duration
            .map(|d| resample(d, frame_rate).value.max(0))
            .unwrap_or(0);
        Self {
            width,
            height,
            frame_rate,
            frame_count,
            has_audio,
        }
    }
}

/// Probe `path` with the platform-native decoder.
///
/// Opens the source (demux + codec setup) far enough to read its
/// [`SourceInfo`], then drops the decoder, and separately checks for an audio
/// track. Returns [`DecodeError::Unsupported`] on platforms without a native
/// decoder.
pub fn probe(path: &Path) -> Result<MediaProbe, DecodeError> {
    let decoder = open(path)?;
    Ok(MediaProbe::from_info(decoder.info(), has_audio_track(path)))
}

#[cfg(target_vendor = "apple")]
fn open(path: &Path) -> Result<Box<dyn VideoDecoder>, DecodeError> {
    Ok(Box::new(crate::AvfDecoder::open(path, OutputMode::Cpu)?))
}

#[cfg(target_os = "windows")]
fn open(path: &Path) -> Result<Box<dyn VideoDecoder>, DecodeError> {
    Ok(Box::new(crate::WmfDecoder::open(path, OutputMode::Cpu)?))
}

#[cfg(target_os = "android")]
fn open(path: &Path) -> Result<Box<dyn VideoDecoder>, DecodeError> {
    Ok(Box::new(crate::MediaCodecDecoder::open(
        path,
        OutputMode::Cpu,
    )?))
}

#[cfg(not(any(target_vendor = "apple", target_os = "windows", target_os = "android")))]
fn open(_path: &Path) -> Result<Box<dyn VideoDecoder>, DecodeError> {
    Err(DecodeError::unsupported(
        "no native video decoder for this platform",
    ))
}

/// Whether `path` has at least one audio track, queried from the platform
/// demuxer. `false` on any error or on platforms without a backend — a missing
/// audio track is never fatal to probing.
#[cfg(target_vendor = "apple")]
fn has_audio_track(path: &Path) -> bool {
    crate::apple::has_audio_track(path)
}

#[cfg(target_os = "android")]
fn has_audio_track(path: &Path) -> bool {
    crate::android::has_audio_track(path)
}

#[cfg(not(any(target_vendor = "apple", target_os = "android")))]
fn has_audio_track(_path: &Path) -> bool {
    false
}
