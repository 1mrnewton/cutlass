//! The decode contract.
//!
//! Cutlass decodes with the platform's native codec — VideoToolbox, MediaCodec,
//! Media Foundation, or a software fallback — but *control* stays in Rust behind
//! [`VideoDecoder`]. That split is what lets the desktop app, the mobile shells,
//! and the Python bindings all drive one decode path: only the codec call is
//! native, everything above the trait is shared.
//!
//! The trait is a **synchronous pull**: `seek`, then `next_frame` until the
//! frame you want. That is the natural shape of every hardware codec (feed
//! packets, drain frames). The asynchronous *decode-ahead* model — running the
//! decoder on a worker thread that fills a small pool of frames keyed by PTS so
//! the render thread never blocks — is layered *on top* of this trait by the
//! engine, not baked into it.

use crate::color::ColorSpace;
use crate::frame::VideoFrame;
use crate::geometry::Rotation;
use crate::pixel::PixelFormat;
use crate::time::{Rational, RationalTime};

/// A decode failure, backend-agnostic so VideoToolbox / MediaCodec / Media
/// Foundation / software backends all map their native errors into one type.
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    /// Opening or probing the source failed (bad container, missing stream).
    #[error("failed to open source: {0}")]
    Open(String),
    /// I/O error reading the source.
    #[error("source i/o error: {0}")]
    Io(String),
    /// The codec rejected a packet or failed to produce a frame.
    #[error("decode failed: {0}")]
    Decode(String),
    /// Seeking failed (e.g. target before the first keyframe).
    #[error("seek failed: {0}")]
    Seek(String),
    /// The source uses a codec, pixel format, or feature we don't support.
    #[error("unsupported: {0}")]
    Unsupported(String),
}

impl DecodeError {
    pub fn unsupported(msg: impl Into<String>) -> Self {
        DecodeError::Unsupported(msg.into())
    }
}

/// Static properties of a decoded video source, read once at open time.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceInfo {
    /// Full coded frame size in pixels (may be padded past the visible area).
    pub coded_size: (u32, u32),
    /// Visible frame size in pixels (before rotation).
    pub display_size: (u32, u32),
    /// Display rotation from container metadata.
    pub rotation: Rotation,
    /// Pixel format the decoder is expected to emit (hardware paths may only
    /// confirm this after the first frame).
    pub pixel_format: PixelFormat,
    /// Colorimetry of the decoded samples.
    pub color: ColorSpace,
    /// Average/nominal frame rate (may be approximate for VFR sources).
    pub frame_rate: Rational,
    /// PTS time base as ticks-per-second; a frame's `pts.rate` equals this.
    pub time_base: Rational,
    /// Source duration, if the container reports one.
    pub duration: Option<RationalTime>,
}

impl SourceInfo {
    /// Displayed frame size after applying [`SourceInfo::rotation`].
    pub fn display_size_rotated(&self) -> (u32, u32) {
        if self.rotation.swaps_axes() {
            (self.display_size.1, self.display_size.0)
        } else {
            self.display_size
        }
    }
}

/// A pull-based video decoder over a single source.
///
/// Implementors wrap one native codec instance. The contract:
///
/// - [`seek`](VideoDecoder::seek) positions the decoder at the keyframe **at or
///   before** `target` and flushes codec buffers, so a subsequent
///   `next_frame` walk reaches `target` by decoding forward.
/// - [`next_frame`](VideoDecoder::next_frame) returns the next frame in
///   presentation order (PTS-increasing between seeks), or `None` at end of
///   stream.
///
/// `Send` is required so a decoder can be owned by a decode-ahead worker thread.
pub trait VideoDecoder: Send {
    /// Static properties of the source.
    fn info(&self) -> &SourceInfo;

    /// Seek to the keyframe at or before `target` and flush buffers.
    fn seek(&mut self, target: RationalTime) -> Result<(), DecodeError>;

    /// Decode and return the next frame, or `None` at end of stream.
    fn next_frame(&mut self) -> Result<Option<VideoFrame>, DecodeError>;

    /// Decode the first frame whose PTS is at or after `target` — the scrub /
    /// seek-to-time primitive. Relies on the [`seek`](VideoDecoder::seek)
    /// contract (lands at/before `target`) and walks forward from there.
    ///
    /// Backends can override this with a roll-forward optimization that skips
    /// the re-seek when `target` is just ahead of the current position within
    /// the same GOP (the common sequential-playback case).
    fn frame_at(&mut self, target: RationalTime) -> Result<Option<VideoFrame>, DecodeError> {
        use core::cmp::Ordering;
        self.seek(target)?;
        while let Some(frame) = self.next_frame()? {
            if frame.pts.compare(target) != Ordering::Less {
                return Ok(Some(frame));
            }
        }
        Ok(None)
    }

    /// Decode *a* frame near `target`, trading exactness for speed — the
    /// keyframe-snap primitive for interactive scrubbing. Where
    /// [`frame_at`](VideoDecoder::frame_at) pays a keyframe-prefix walk
    /// (potentially a whole GOP — hundreds of decodes on long-GOP sources)
    /// to land exactly, this seeks (landing on the sync frame at or before
    /// `target`) and decodes exactly one frame.
    ///
    /// The returned frame's PTS is therefore usually *before* `target`, by
    /// up to a GOP. Callers own the follow-up exact decode once the
    /// interaction settles, and must not cache the result under the exact
    /// target. `Ok(None)` means the seek landed at/past end of stream.
    ///
    /// Backends with roll-forward state should override this to delegate to
    /// `frame_at` when the target is just ahead of the last emitted frame
    /// (inside the roll window) — there the exact walk is cheaper than a
    /// container seek *and* lands exactly.
    fn frame_at_nearest(
        &mut self,
        target: RationalTime,
    ) -> Result<Option<VideoFrame>, DecodeError> {
        self.seek(target)?;
        self.next_frame()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::ColorSpace;
    use crate::frame::{CpuImage, FrameData, Plane, VideoFrame};
    use crate::geometry::Rect;

    /// A trivial decoder over N synthetic 30 fps frames (PTS = frame index),
    /// just enough to exercise the default `frame_at` walk.
    struct MockDecoder {
        info: SourceInfo,
        cursor: usize,
        count: i64,
    }

    impl MockDecoder {
        fn new(count: i64) -> Self {
            let info = SourceInfo {
                coded_size: (2, 2),
                display_size: (2, 2),
                rotation: Rotation::None,
                pixel_format: PixelFormat::Nv12,
                color: ColorSpace::BT709,
                frame_rate: Rational::FPS_30,
                time_base: Rational::FPS_30,
                duration: Some(RationalTime::new(count, Rational::FPS_30)),
            };
            Self {
                info,
                cursor: 0,
                count,
            }
        }
    }

    impl VideoDecoder for MockDecoder {
        fn info(&self) -> &SourceInfo {
            &self.info
        }

        fn seek(&mut self, _target: RationalTime) -> Result<(), DecodeError> {
            // Mock has a keyframe on every frame, so "at or before target" is
            // just the start — walking forward reaches any target.
            self.cursor = 0;
            Ok(())
        }

        fn next_frame(&mut self) -> Result<Option<VideoFrame>, DecodeError> {
            if (self.cursor as i64) >= self.count {
                return Ok(None);
            }
            let pts = RationalTime::new(self.cursor as i64, Rational::FPS_30);
            self.cursor += 1;
            let y = Plane::new(vec![16u8; 4], 2, 2);
            let uv = Plane::new(vec![128u8; 2], 2, 1);
            Ok(Some(VideoFrame::new(
                pts,
                PixelFormat::Nv12,
                ColorSpace::BT709,
                (2, 2),
                Rect::from_size(2, 2),
                Rotation::None,
                FrameData::Cpu(CpuImage::new(vec![y, uv])),
            )))
        }
    }

    #[test]
    fn frame_at_default_walks_to_first_frame_at_or_after_target() {
        let mut dec = MockDecoder::new(30);
        let f = dec
            .frame_at(RationalTime::new(5, Rational::FPS_30))
            .unwrap()
            .unwrap();
        assert_eq!(f.pts.value, 5);
    }

    #[test]
    fn frame_at_handles_cross_rate_target() {
        // Target expressed at 24 fps: 4/24 s = 1/6 s lands on 30fps frame 5.
        let mut dec = MockDecoder::new(30);
        let f = dec
            .frame_at(RationalTime::new(4, Rational::FPS_24))
            .unwrap()
            .unwrap();
        assert_eq!(f.pts.value, 5);
    }

    #[test]
    fn frame_at_past_end_returns_none() {
        let mut dec = MockDecoder::new(10);
        assert!(
            dec.frame_at(RationalTime::new(100, Rational::FPS_30))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn source_info_rotated_display_size_swaps_for_portrait() {
        let info = SourceInfo {
            coded_size: (1920, 1088),
            display_size: (1920, 1080),
            rotation: Rotation::Cw90,
            pixel_format: PixelFormat::Nv12,
            color: ColorSpace::BT709,
            frame_rate: Rational::FPS_30,
            time_base: Rational::new(90000, 1),
            duration: None,
        };
        assert_eq!(info.display_size_rotated(), (1080, 1920));
    }

    #[test]
    fn decode_error_unsupported_helper() {
        let e = DecodeError::unsupported("av1 not wired up");
        assert!(matches!(e, DecodeError::Unsupported(_)));
        assert_eq!(e.to_string(), "unsupported: av1 not wired up");
    }
}
