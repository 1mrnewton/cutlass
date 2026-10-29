//! One picture inside a still file.
//!
//! Ranges point into the file buffer. They are not copied until a packet
//! is read.

use crate::fail::{Fail, as_u64, hostile};
use crate::limits::{GIF_TIMESCALE, WEBP_TIMESCALE};
use cutlass_codec::CodecId;

/// Which signature opened the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StillKind {
    Jpeg,
    Png,
    Webp,
    Gif,
}

impl StillKind {
    #[must_use]
    pub const fn codec(self) -> CodecId {
        match self {
            Self::Jpeg => CodecId::Jpeg,
            Self::Png => CodecId::Png,
            Self::Webp => CodecId::Webp,
            Self::Gif => CodecId::Gif,
        }
    }

    /// Ticks per second for a sequence. A single picture has no clock.
    #[must_use]
    pub const fn clock(self, animated: bool) -> u32 {
        match self {
            Self::Gif => GIF_TIMESCALE,
            Self::Webp if animated => WEBP_TIMESCALE,
            Self::Jpeg | Self::Png | Self::Webp => 0,
        }
    }
}

/// Disposal and, for WebP, the blending bit stored with one frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameControl {
    /// GIF disposal method, 0 through 7, or the WebP disposal bit, 0 or 1.
    pub disposal: u8,
    /// WebP blending bit from the `ANMF` flags. `None` for GIF and for a
    /// single picture. `1` means do not blend. `0` means alpha-blend.
    pub webp_blend: Option<u8>,
}

pub(crate) struct Frame {
    pub start: usize,
    pub end: usize,
    pub pts: i64,
    pub duration: i64,
    pub timed: bool,
    pub keyframe: bool,
    pub disposal: u8,
    pub blend: Option<u8>,
}

impl Frame {
    pub(crate) fn still(len: usize) -> Self {
        Self {
            start: 0,
            end: len,
            pts: 0,
            duration: 0,
            timed: false,
            keyframe: true,
            disposal: 0,
            blend: None,
        }
    }
}

pub(crate) struct Sequence {
    pub kind: StillKind,
    pub animated: bool,
    pub width: u32,
    pub height: u32,
    pub loop_count: Option<u16>,
    pub frames: Vec<Frame>,
}

impl Sequence {
    pub(crate) fn exclusive_end(&self) -> Result<Option<i64>, Fail> {
        if self.kind.clock(self.animated) == 0 {
            return Ok(None);
        }
        let Some(last) = self.frames.last() else {
            return Err(Fail::Bad("still has no picture"));
        };
        let end = last
            .pts
            .checked_add(last.duration)
            .ok_or_else(|| hostile(as_u64(self.frames.len()), u64::from(u32::MAX)))?;
        Ok(Some(end))
    }
}
