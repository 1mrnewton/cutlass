//! Coded-domain pictures exchanged by a video codec.
//!
//! Samples stay in the format the bitstream names. This module does not
//! convert Y′CbCr to RGB.

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

#[cfg(test)]
mod tests {
    use super::{Plane, VideoFrame};

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
