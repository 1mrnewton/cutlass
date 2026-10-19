//! FFmpeg backend for the [`cutlass_core::VideoDecoder`] contract.
//!
//! The platform-native backends (VideoToolbox / MediaCodec / Media Foundation)
//! cover Apple, Android, and Windows; this wraps the in-tree FFmpeg [`Decoder`]
//! so Linux and desktop *dev* builds get a `cutlass_core::VideoDecoder` too —
//! the cross-platform fallback the reference design leaves as "to follow".
//!
//! Demux and decode both come from FFmpeg here, and only CPU frames are produced
//! (no zero-copy GPU surface). The wrapper's whole job is translation: map the
//! crate's [`DecodedFrame`]/[`SourceInfo`]/[`DecodeError`](crate::DecodeError)
//! onto the core types, and express timeline `RationalTime` targets in the
//! stream's own tick base so seeks stay exact.

use std::path::Path;

use cutlass_core::{
    ColorRange, ColorSpace, CpuImage, DecodeError, FrameData, PixelFormat, Plane, Rational,
    RationalTime, Rect, Rotation, SourceInfo, VideoDecoder, VideoFrame, resample,
};

use crate::DecodeError as FfDecodeError;
use crate::{DecodedFrame, Decoder};

/// A [`cutlass_core::VideoDecoder`] backed by the FFmpeg [`Decoder`].
pub struct FfmpegDecoder {
    inner: Decoder,
    info: SourceInfo,
}

// SAFETY: the wrapped FFmpeg decoder — including its hardware-device
// `AVBufferRef` — is only ever touched by one thread at a time. The trait is
// `&mut self` and a decode-ahead worker owns the decoder exclusively (it is
// never shared as `&self` across threads). FFmpeg's buffer refcounts are atomic,
// so moving the handle to another thread is sound; this mirrors the platform
// backends' `unsafe impl Send`.
unsafe impl Send for FfmpegDecoder {}

impl FfmpegDecoder {
    /// Open `path` with FFmpeg and read its source properties.
    pub fn open(path: &Path) -> Result<Self, DecodeError> {
        let inner = Decoder::open(path).map_err(to_core_err)?;
        let info = to_core_info(inner.info(), inner.duration());
        Ok(Self { inner, info })
    }

    /// `target` expressed in the source's own time-base ticks (what the inner
    /// decoder's `*_ticks` entry points expect) — exact, no `Duration` float.
    fn target_ticks(&self, target: RationalTime) -> i64 {
        resample(target, self.info.time_base).value
    }
}

impl VideoDecoder for FfmpegDecoder {
    fn info(&self) -> &SourceInfo {
        &self.info
    }

    fn seek(&mut self, target: RationalTime) -> Result<(), DecodeError> {
        let ticks = self.target_ticks(target);
        self.inner.seek_ticks(ticks).map_err(to_core_err)
    }

    fn next_frame(&mut self) -> Result<Option<VideoFrame>, DecodeError> {
        let tb = self.info.time_base;
        let rotation = self.info.rotation;
        Ok(self
            .inner
            .next_frame()
            .map_err(to_core_err)?
            .map(|frame| video_frame_from_decoded(frame, tb, rotation)))
    }

    /// Override the default seek+walk with the inner decoder's roll-forward
    /// `frame_at_ticks`, which skips the re-seek when `target` is just ahead of
    /// the current position within the same GOP (the playback/scrub fast path).
    fn frame_at(&mut self, target: RationalTime) -> Result<Option<VideoFrame>, DecodeError> {
        let ticks = self.target_ticks(target);
        let tb = self.info.time_base;
        let rotation = self.info.rotation;
        Ok(self
            .inner
            .frame_at_ticks(ticks)
            .map_err(to_core_err)?
            .map(|frame| video_frame_from_decoded(frame, tb, rotation)))
    }
}

/// Core PTS time base (ticks per second) for an FFmpeg source: the inverse of
/// FFmpeg's seconds-per-tick `time_base`. A frame's `pts.rate` equals this, so
/// callers driving the concrete [`Decoder`] (which keeps the FFmpeg keyframe
/// index) can build core PTS and convert `RationalTime` targets to stream ticks
/// without re-opening through [`FfmpegDecoder`].
pub fn core_time_base(info: &crate::SourceInfo) -> Rational {
    let tb_num = info.time_base.numerator().max(1);
    let tb_den = info.time_base.denominator().max(1);
    Rational::new(tb_den, tb_num)
}

/// Build the core [`SourceInfo`] from the FFmpeg one plus the container duration.
fn to_core_info(info: &crate::SourceInfo, duration: Option<std::time::Duration>) -> SourceInfo {
    // FFmpeg's `time_base` is seconds-per-tick (`num/den`); core's is ticks per
    // second, so a frame's `pts.rate` equals it — invert.
    let time_base = core_time_base(info);

    let frame_rate = Rational::new(
        info.frame_rate.numerator().max(1),
        info.frame_rate.denominator().max(1),
    );

    let duration = duration.map(|d| {
        let ticks = (d.as_secs_f64() * time_base.as_f64()).round() as i64;
        RationalTime::new(ticks, time_base)
    });

    SourceInfo {
        coded_size: (info.width, info.height),
        display_size: (info.width, info.height),
        // The FFmpeg decoder doesn't surface container rotation yet; keep
        // upright until the display-matrix side data is wired through.
        rotation: Rotation::None,
        pixel_format: to_core_pixel_format(info.pixel_format),
        color: to_color(info.full_range),
        frame_rate,
        time_base,
        duration,
    }
}

/// Translate a decoded FFmpeg frame into a core CPU [`VideoFrame`].
///
/// Exposed so the engine's desktop decode path (which drives the concrete
/// [`Decoder`] directly to keep its FFmpeg keyframe index and frame cache) can
/// produce the same `cutlass_core::VideoFrame` currency as [`FfmpegDecoder`],
/// rather than re-implementing the plane/colorimetry mapping. `time_base` comes
/// from [`core_time_base`]; `rotation` is the container display rotation
/// ([`Rotation::None`] until that side data is wired through).
pub fn video_frame_from_decoded(
    frame: DecodedFrame,
    time_base: Rational,
    rotation: Rotation,
) -> VideoFrame {
    let (width, height) = (frame.width, frame.height);
    let format = to_core_pixel_format(frame.format);
    let color = to_color(frame.full_range);
    let pts = RationalTime::new(frame.pts_ticks, time_base);

    let rows = plane_rows(frame.format, height);
    let planes = frame
        .planes
        .into_iter()
        .zip(rows)
        .map(|(plane, plane_rows)| Plane::new(plane.data, plane.stride, plane_rows as usize))
        .collect();

    VideoFrame::new(
        pts,
        format,
        color,
        (width, height),
        Rect::from_size(width, height),
        rotation,
        FrameData::Cpu(CpuImage::new(planes)),
    )
}

/// Per-plane sample-row counts for a format at `height` (chroma is half-height
/// for the 4:2:0 layouts).
fn plane_rows(format: crate::PixelFormat, height: u32) -> Vec<u32> {
    match format {
        crate::PixelFormat::Yuv420p => vec![height, height / 2, height / 2],
        crate::PixelFormat::Nv12 => vec![height, height / 2],
        crate::PixelFormat::Rgba8 => vec![height],
    }
}

fn to_core_pixel_format(format: crate::PixelFormat) -> PixelFormat {
    match format {
        crate::PixelFormat::Yuv420p => PixelFormat::Yuv420p,
        crate::PixelFormat::Nv12 => PixelFormat::Nv12,
        crate::PixelFormat::Rgba8 => PixelFormat::Rgba8,
    }
}

/// The FFmpeg path only tracks sample range (full for `yuvj*`), assuming BT.709
/// otherwise — carry that across rather than inventing colorimetry it can't see.
fn to_color(full_range: bool) -> ColorSpace {
    ColorSpace {
        range: if full_range {
            ColorRange::Full
        } else {
            ColorRange::Limited
        },
        ..ColorSpace::BT709
    }
}

fn to_core_err(err: FfDecodeError) -> DecodeError {
    match err {
        FfDecodeError::Open(e) => DecodeError::Open(e.to_string()),
        FfDecodeError::Io(e) => DecodeError::Io(e.to_string()),
        FfDecodeError::Decode(e) => DecodeError::Decode(e.to_string()),
        FfDecodeError::Unsupported { what } => DecodeError::Unsupported(what),
        FfDecodeError::HwAccelUnavailable { accel } => {
            DecodeError::Unsupported(format!("hardware acceleration unavailable: {accel}"))
        }
    }
}
