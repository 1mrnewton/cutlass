//! Coded-domain pictures exchanged by a video codec.
//!
//! [`Yuv420p`] is tightly packed 8-bit 4:2:0 with no stride. [`VideoFrame`]
//! carries planes that may have a stride, plus the color tags. Display
//! conversion lives outside this crate.

use std::fmt;

use super::error::CodecError;
use super::pixel::{ColorPrimaries, ColorRange, MatrixCoefficients, PixelFormat, Transfer};

/// One sample plane, row-major, with `stride` bytes from one row to the next.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plane {
    pub width: u32,
    pub height: u32,
    pub stride: usize,
    pub samples: Vec<u8>,
}

impl Plane {
    /// A plane whose rows are packed with no padding.
    pub fn tight(width: u32, height: u32, samples: Vec<u8>) -> Result<Self, CodecError> {
        let stride = usize::try_from(width).map_err(|_| CodecError::Limit {
            what: "plane width",
        })?;
        let rows = usize::try_from(height).map_err(|_| CodecError::Limit {
            what: "plane height",
        })?;
        let need = stride.checked_mul(rows).ok_or(CodecError::Limit {
            what: "plane samples",
        })?;
        if samples.len() != need {
            return Err(CodecError::InvalidApi {
                what: "plane sample count",
            });
        }
        Ok(Self {
            width,
            height,
            stride,
            samples,
        })
    }

    /// A zero-filled tight plane.
    pub fn blank(width: u32, height: u32) -> Result<Self, CodecError> {
        let stride = usize::try_from(width).map_err(|_| CodecError::Limit {
            what: "plane width",
        })?;
        let rows = usize::try_from(height).map_err(|_| CodecError::Limit {
            what: "plane height",
        })?;
        let need = stride.checked_mul(rows).ok_or(CodecError::Limit {
            what: "plane samples",
        })?;
        Ok(Self {
            width,
            height,
            stride,
            samples: vec![0; need],
        })
    }

    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> Option<u8> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let row = usize::try_from(y).ok()?;
        let col = usize::try_from(x).ok()?;
        self.samples.get(row * self.stride + col).copied()
    }
}

/// One decoded or encoder-input picture.
///
/// `y`, `u`, and `v` are present for [`PixelFormat::Yuv420p`]. The chroma
/// planes are `(width + 1) / 2` by `(height + 1) / 2`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoFrame {
    pub format: PixelFormat,
    pub width: u32,
    pub height: u32,
    pub y: Plane,
    pub u: Plane,
    pub v: Plane,
    pub range: ColorRange,
    pub primaries: ColorPrimaries,
    pub transfer: Transfer,
    pub matrix: MatrixCoefficients,
}

impl VideoFrame {
    /// Builds a 4:2:0 picture. Width and height must be non-zero.
    pub fn yuv420p(
        width: u32,
        height: u32,
        y: Plane,
        u: Plane,
        v: Plane,
    ) -> Result<Self, CodecError> {
        if width == 0 || height == 0 {
            return Err(CodecError::InvalidApi {
                what: "empty picture",
            });
        }
        let chroma_w = width.div_ceil(2);
        let chroma_h = height.div_ceil(2);
        if y.width != width || y.height != height {
            return Err(CodecError::InvalidApi {
                what: "luma plane size",
            });
        }
        if u.width != chroma_w
            || u.height != chroma_h
            || v.width != chroma_w
            || v.height != chroma_h
        {
            return Err(CodecError::InvalidApi {
                what: "chroma plane size",
            });
        }
        Ok(Self {
            format: PixelFormat::Yuv420p,
            width,
            height,
            y,
            u,
            v,
            range: ColorRange::Unspecified,
            primaries: ColorPrimaries::Unspecified,
            transfer: Transfer::Unspecified,
            matrix: MatrixCoefficients::Unspecified,
        })
    }

    /// A black 4:2:0 picture with tight planes.
    pub fn blank_yuv420p(width: u32, height: u32) -> Result<Self, CodecError> {
        let chroma_w = width.div_ceil(2);
        let chroma_h = height.div_ceil(2);
        Self::yuv420p(
            width,
            height,
            Plane::blank(width, height)?,
            Plane::blank(chroma_w, chroma_h)?,
            Plane::blank(chroma_w, chroma_h)?,
        )
    }
}

/// Why a 4:2:0 picture could not be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameError {
    /// A dimension is zero.
    Empty,
    /// 4:2:0 needs even edges.
    Odd,
    /// A plane length does not match the size.
    Length,
    /// The picture would pass 8192 on an edge, or the product overflowed.
    Limit,
}

impl fmt::Display for FrameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Empty => "frame size is empty",
            Self::Odd => "4:2:0 frame size is odd",
            Self::Length => "plane length does not match the frame size",
            Self::Limit => "frame size is past the limit",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for FrameError {}

/// Tightly packed 8-bit 4:2:0, visible region only.
///
/// `y` is `width * height` bytes. `u` and `v` are each `(width / 2) * (height / 2)`
/// bytes. Rows are contiguous. There is no padding and no separate stride.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Yuv420p {
    width: u32,
    height: u32,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

impl Yuv420p {
    /// Both edges must be even, non-zero, and at most 8192.
    pub fn try_new(
        width: u32,
        height: u32,
        y: Vec<u8>,
        u: Vec<u8>,
        v: Vec<u8>,
    ) -> Result<Self, FrameError> {
        if width == 0 || height == 0 {
            return Err(FrameError::Empty);
        }
        if width > 8192 || height > 8192 {
            return Err(FrameError::Limit);
        }
        if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(FrameError::Odd);
        }
        let y_len = (width as usize)
            .checked_mul(height as usize)
            .ok_or(FrameError::Limit)?;
        let c_len = ((width / 2) as usize)
            .checked_mul((height / 2) as usize)
            .ok_or(FrameError::Limit)?;
        if y.len() != y_len || u.len() != c_len || v.len() != c_len {
            return Err(FrameError::Length);
        }
        Ok(Self {
            width,
            height,
            y,
            u,
            v,
        })
    }

    /// A solid picture. Luma and both chroma planes are filled with one value each.
    pub fn solid(width: u32, height: u32, y: u8, u: u8, v: u8) -> Result<Self, FrameError> {
        if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Self::try_new(width, height, Vec::new(), Vec::new(), Vec::new());
        }
        let y_len = (width as usize)
            .checked_mul(height as usize)
            .ok_or(FrameError::Limit)?;
        let c_len = ((width / 2) as usize)
            .checked_mul((height / 2) as usize)
            .ok_or(FrameError::Limit)?;
        Self::try_new(
            width,
            height,
            vec![y; y_len],
            vec![u; c_len],
            vec![v; c_len],
        )
    }

    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    #[must_use]
    pub fn y(&self) -> &[u8] {
        &self.y
    }

    #[must_use]
    pub fn u(&self) -> &[u8] {
        &self.u
    }

    #[must_use]
    pub fn v(&self) -> &[u8] {
        &self.v
    }

    #[must_use]
    pub fn y_mut(&mut self) -> &mut [u8] {
        &mut self.y
    }

    #[must_use]
    pub fn u_mut(&mut self) -> &mut [u8] {
        &mut self.u
    }

    #[must_use]
    pub fn v_mut(&mut self) -> &mut [u8] {
        &mut self.v
    }
}

#[cfg(test)]
mod tests {
    use super::{FrameError, Plane, VideoFrame, Yuv420p};

    #[test]
    fn a_solid_picture_keeps_even_planes() {
        let frame = Yuv420p::solid(4, 2, 16, 128, 128).expect("frame");
        assert_eq!(frame.width(), 4);
        assert_eq!(frame.height(), 2);
        assert_eq!(frame.y().len(), 8);
        assert_eq!(frame.u().len(), 2);
        assert_eq!(frame.v(), &[128, 128]);
    }

    #[test]
    fn an_odd_or_short_picture_is_refused() {
        assert_eq!(
            Yuv420p::try_new(1, 2, vec![0, 0], vec![], vec![]),
            Err(FrameError::Odd)
        );
        assert_eq!(
            Yuv420p::try_new(2, 2, vec![0, 0, 0], vec![0], vec![0]),
            Err(FrameError::Length)
        );
        assert_eq!(
            Yuv420p::try_new(0, 2, Vec::new(), Vec::new(), Vec::new()),
            Err(FrameError::Empty)
        );
    }

    #[test]
    fn yuv420_rejects_a_short_chroma_plane() {
        let y = Plane::blank(4, 4).expect("y");
        let u = Plane::blank(1, 2).expect("u");
        let v = Plane::blank(2, 2).expect("v");
        assert!(VideoFrame::yuv420p(4, 4, y, u, v).is_err());
    }

    #[test]
    fn odd_width_uses_ceil_chroma() {
        let frame = VideoFrame::blank_yuv420p(3, 2).expect("frame");
        assert_eq!(frame.u.width, 2);
        assert_eq!(frame.u.height, 1);
        assert_eq!(frame.y.samples.len(), 6);
    }
}
