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

    /// A tight plane whose samples are `sample_bytes` each, little-endian.
    pub fn bytes(
        width: u32,
        height: u32,
        sample_bytes: usize,
        samples: Vec<u8>,
    ) -> Result<Self, CodecError> {
        let columns = usize::try_from(width).map_err(|_| CodecError::Limit {
            what: "plane width",
        })?;
        let rows = usize::try_from(height).map_err(|_| CodecError::Limit {
            what: "plane height",
        })?;
        let stride = columns.checked_mul(sample_bytes).ok_or(CodecError::Limit {
            what: "plane samples",
        })?;
        let need = stride.checked_mul(rows).ok_or(CodecError::Limit {
            what: "plane samples",
        })?;
        if samples.len() != need || sample_bytes == 0 {
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
    /// Display width. Equal to [`Self::width`] when the frame names no other size.
    pub render_width: u32,
    /// Display height. Equal to [`Self::height`] when the frame names no other size.
    pub render_height: u32,
    /// Straight alpha at full size, when the stream carries it.
    pub alpha: Option<Alpha>,
}

/// An alpha plane carried beside the color planes, at its coded depth.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Alpha {
    /// Bits per sample, 1 to 16. Up to 8 bits store one byte per sample;
    /// more store two, little-endian.
    pub bit_depth: u8,
    /// One sample per pixel, the frame's width by its height.
    pub plane: Plane,
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
            render_width: width,
            render_height: height,
            alpha: None,
        })
    }

    /// A planar picture whose chroma size follows `format`.
    pub fn planar(
        format: PixelFormat,
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
        let Some((chroma_w, chroma_h)) = format.chroma_wh(width, height) else {
            return Err(CodecError::InvalidApi {
                what: "picture format",
            });
        };
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
            format,
            width,
            height,
            y,
            u,
            v,
            range: ColorRange::Unspecified,
            primaries: ColorPrimaries::Unspecified,
            transfer: Transfer::Unspecified,
            matrix: MatrixCoefficients::Unspecified,
            render_width: width,
            render_height: height,
            alpha: None,
        })
    }

    /// Attaches a full-size alpha plane of `bit_depth` bits per sample.
    pub fn with_alpha(mut self, bit_depth: u8, plane: Plane) -> Result<Self, CodecError> {
        if bit_depth == 0 || bit_depth > 16 {
            return Err(CodecError::InvalidApi {
                what: "alpha bit depth",
            });
        }
        let sample_bytes = if bit_depth > 8 { 2 } else { 1 };
        let row = usize::try_from(plane.width)
            .ok()
            .and_then(|w| w.checked_mul(sample_bytes));
        let need = usize::try_from(plane.height)
            .ok()
            .and_then(|h| plane.stride.checked_mul(h));
        let fits = matches!((row, need), (Some(row), Some(need))
            if row <= plane.stride && need <= plane.samples.len());
        if plane.width != self.width || plane.height != self.height || !fits {
            return Err(CodecError::InvalidApi {
                what: "alpha plane size",
            });
        }
        self.alpha = Some(Alpha { bit_depth, plane });
        Ok(self)
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

/// Longest edge a stored picture allows. H.264 level 6.2's coded edge is 16,880.
const MAX_EDGE: u32 = 16_880;

/// Why a 4:2:0 picture could not be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameError {
    /// A dimension is zero.
    Empty,
    /// 4:2:0 needs even edges.
    Odd,
    /// A plane length does not match the size.
    Length,
    /// The picture would pass 16,880 on an edge, or the product overflowed.
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
    /// Both edges must be even, non-zero, and at most 16,880.
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
        if width > MAX_EDGE || height > MAX_EDGE {
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

/// Tightly packed 10-bit 4:2:0, visible region only.
///
/// Each sample is a `u16` in `0..=1023`, stored in the low 10 bits. `y` is
/// `width * height` samples. `u` and `v` are each `(width / 2) * (height / 2)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Yuv420p10 {
    width: u32,
    height: u32,
    y: Vec<u16>,
    u: Vec<u16>,
    v: Vec<u16>,
}

impl Yuv420p10 {
    /// Both edges must be even, non-zero, and at most 16,880.
    pub fn try_new(
        width: u32,
        height: u32,
        y: Vec<u16>,
        u: Vec<u16>,
        v: Vec<u16>,
    ) -> Result<Self, FrameError> {
        if width == 0 || height == 0 {
            return Err(FrameError::Empty);
        }
        if width > MAX_EDGE || height > MAX_EDGE {
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

    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    #[must_use]
    pub fn y(&self) -> &[u16] {
        &self.y
    }

    #[must_use]
    pub fn u(&self) -> &[u16] {
        &self.u
    }

    #[must_use]
    pub fn v(&self) -> &[u16] {
        &self.v
    }
}

/// Tightly packed 4:2:0 at 9 through 14 bits, visible region only.
///
/// Each sample is a `u16` in `0..(1 << bit_depth)`, stored in the low
/// `bit_depth` bits. `y` is `width * height` samples. `u` and `v` are each
/// `(width / 2) * (height / 2)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Yuv420High {
    bit_depth: u8,
    width: u32,
    height: u32,
    y: Vec<u16>,
    u: Vec<u16>,
    v: Vec<u16>,
}

impl Yuv420High {
    /// `bit_depth` is 9 through 14. Both edges must be even, non-zero, and at most 16,880.
    pub fn try_new(
        bit_depth: u8,
        width: u32,
        height: u32,
        y: Vec<u16>,
        u: Vec<u16>,
        v: Vec<u16>,
    ) -> Result<Self, FrameError> {
        if !(9..=14).contains(&bit_depth) {
            return Err(FrameError::Limit);
        }
        if width == 0 || height == 0 {
            return Err(FrameError::Empty);
        }
        if width > MAX_EDGE || height > MAX_EDGE {
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
            bit_depth,
            width,
            height,
            y,
            u,
            v,
        })
    }

    #[must_use]
    pub const fn bit_depth(&self) -> u8 {
        self.bit_depth
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
    pub fn y(&self) -> &[u16] {
        &self.y
    }

    #[must_use]
    pub fn u(&self) -> &[u16] {
        &self.u
    }

    #[must_use]
    pub fn v(&self) -> &[u16] {
        &self.v
    }
}

fn checked_chroma(
    width: u32,
    height: u32,
    y: usize,
    u: usize,
    v: usize,
    sub_w: u32,
    sub_h: u32,
) -> Result<(), FrameError> {
    if width == 0 || height == 0 {
        return Err(FrameError::Empty);
    }
    if width > MAX_EDGE || height > MAX_EDGE {
        return Err(FrameError::Limit);
    }
    if sub_w == 2 && !width.is_multiple_of(2) {
        return Err(FrameError::Odd);
    }
    if sub_h == 2 && !height.is_multiple_of(2) {
        return Err(FrameError::Odd);
    }
    let y_len = (width as usize)
        .checked_mul(height as usize)
        .ok_or(FrameError::Limit)?;
    let c_len = ((width / sub_w) as usize)
        .checked_mul((height / sub_h) as usize)
        .ok_or(FrameError::Limit)?;
    if y != y_len || u != c_len || v != c_len {
        return Err(FrameError::Length);
    }
    Ok(())
}

/// Tightly packed 4:2:2. `u` and `v` are each `(width / 2) * height`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Yuv422p {
    width: u32,
    height: u32,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

impl Yuv422p {
    /// Width is even, non-zero, and at most 16,880. Height is non-zero and at most 16,880.
    pub fn try_new(
        width: u32,
        height: u32,
        y: Vec<u8>,
        u: Vec<u8>,
        v: Vec<u8>,
    ) -> Result<Self, FrameError> {
        checked_chroma(width, height, y.len(), u.len(), v.len(), 2, 1)?;
        Ok(Self {
            width,
            height,
            y,
            u,
            v,
        })
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
}

/// Tightly packed 4:4:4. `u` and `v` are each `width * height`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Yuv444p {
    width: u32,
    height: u32,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

impl Yuv444p {
    /// Both edges are non-zero and at most 16,880.
    pub fn try_new(
        width: u32,
        height: u32,
        y: Vec<u8>,
        u: Vec<u8>,
        v: Vec<u8>,
    ) -> Result<Self, FrameError> {
        checked_chroma(width, height, y.len(), u.len(), v.len(), 1, 1)?;
        Ok(Self {
            width,
            height,
            y,
            u,
            v,
        })
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
}

/// Tightly packed 4:2:2 at 9 through 14 bits.
///
/// Each sample is a `u16` in `0..(1 << bit_depth)`. `u` and `v` are each
/// `(width / 2) * height`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Yuv422High {
    bit_depth: u8,
    width: u32,
    height: u32,
    y: Vec<u16>,
    u: Vec<u16>,
    v: Vec<u16>,
}

impl Yuv422High {
    /// `bit_depth` is 9 through 14. Width is even. Both edges are non-zero and at most 16,880.
    pub fn try_new(
        bit_depth: u8,
        width: u32,
        height: u32,
        y: Vec<u16>,
        u: Vec<u16>,
        v: Vec<u16>,
    ) -> Result<Self, FrameError> {
        if !(9..=14).contains(&bit_depth) {
            return Err(FrameError::Limit);
        }
        checked_chroma(width, height, y.len(), u.len(), v.len(), 2, 1)?;
        Ok(Self {
            bit_depth,
            width,
            height,
            y,
            u,
            v,
        })
    }

    #[must_use]
    pub const fn bit_depth(&self) -> u8 {
        self.bit_depth
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
    pub fn y(&self) -> &[u16] {
        &self.y
    }

    #[must_use]
    pub fn u(&self) -> &[u16] {
        &self.u
    }

    #[must_use]
    pub fn v(&self) -> &[u16] {
        &self.v
    }
}

/// Tightly packed 4:4:4 at 9 through 14 bits.
///
/// Each sample is a `u16` in `0..(1 << bit_depth)`. `u` and `v` are each
/// `width * height`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Yuv444High {
    bit_depth: u8,
    width: u32,
    height: u32,
    y: Vec<u16>,
    u: Vec<u16>,
    v: Vec<u16>,
}

impl Yuv444High {
    /// `bit_depth` is 9 through 14. Both edges are non-zero and at most 16,880.
    pub fn try_new(
        bit_depth: u8,
        width: u32,
        height: u32,
        y: Vec<u16>,
        u: Vec<u16>,
        v: Vec<u16>,
    ) -> Result<Self, FrameError> {
        if !(9..=14).contains(&bit_depth) {
            return Err(FrameError::Limit);
        }
        checked_chroma(width, height, y.len(), u.len(), v.len(), 1, 1)?;
        Ok(Self {
            bit_depth,
            width,
            height,
            y,
            u,
            v,
        })
    }

    #[must_use]
    pub const fn bit_depth(&self) -> u8 {
        self.bit_depth
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
    pub fn y(&self) -> &[u16] {
        &self.y
    }

    #[must_use]
    pub fn u(&self) -> &[u16] {
        &self.u
    }

    #[must_use]
    pub fn v(&self) -> &[u16] {
        &self.v
    }
}

#[cfg(test)]
mod tests {
    use super::super::pixel::PixelFormat;
    use super::{
        FrameError, Plane, VideoFrame, Yuv420High, Yuv420p, Yuv420p10, Yuv422High, Yuv422p,
        Yuv444High, Yuv444p,
    };

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

    #[test]
    fn ten_bit_420_keeps_the_sample_count() {
        let frame =
            Yuv420p10::try_new(4, 2, vec![1000; 8], vec![512, 512], vec![0, 4]).expect("frame");
        assert_eq!(frame.y(), &[1000; 8]);
        assert_eq!(frame.u(), &[512, 512]);
        assert_eq!(
            Yuv420p10::try_new(4, 2, vec![0; 8], vec![0], vec![0, 0]),
            Err(FrameError::Length)
        );
    }

    #[test]
    fn high_bit_420_records_the_depth() {
        let frame = Yuv420High::try_new(12, 4, 2, vec![4000; 8], vec![100, 200], vec![0, 4])
            .expect("frame");
        assert_eq!(frame.bit_depth(), 12);
        assert_eq!(frame.y(), &[4000; 8]);
        assert_eq!(
            Yuv420High::try_new(8, 4, 2, vec![0; 8], vec![0, 0], vec![0, 0]),
            Err(FrameError::Limit)
        );
        assert_eq!(
            Yuv420High::try_new(14, 4, 2, vec![0; 8], vec![0], vec![0, 0]),
            Err(FrameError::Length)
        );
    }

    #[test]
    fn chroma_422_and_444_keep_their_plane_sizes() {
        let yuv422 = Yuv422p::try_new(4, 2, vec![1; 8], vec![2; 4], vec![3; 4]).expect("422");
        assert_eq!(yuv422.u().len(), 4);
        assert_eq!(
            Yuv422p::try_new(1, 2, vec![0, 0], vec![0, 0], vec![0, 0]),
            Err(FrameError::Odd)
        );
        let yuv444 =
            Yuv444p::try_new(3, 1, vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]).expect("444");
        assert_eq!(yuv444.v(), &[7, 8, 9]);
        let high = Yuv422High::try_new(10, 2, 2, vec![0; 4], vec![1; 2], vec![2; 2]).expect("10");
        assert_eq!(high.bit_depth(), 10);
        assert_eq!(
            Yuv444High::try_new(8, 2, 2, vec![0; 4], vec![0; 4], vec![0; 4]),
            Err(FrameError::Limit)
        );
    }

    #[test]
    fn the_stored_edge_reaches_the_level_6_picture() {
        let y = vec![0u8; 16_880 * 16];
        let chroma = vec![0u8; 8_440 * 8];
        assert!(Yuv420p::try_new(16_880, 16, y, chroma.clone(), chroma).is_ok());
        let y = vec![0u8; 16_896 * 16];
        let chroma = vec![0u8; 8_448 * 8];
        assert_eq!(
            Yuv420p::try_new(16_896, 16, y, chroma.clone(), chroma),
            Err(FrameError::Limit)
        );
    }

    #[test]
    fn alpha_rides_a_frame_only_at_the_frame_size() {
        let frame = VideoFrame::planar(
            PixelFormat::Yuv444p12,
            3,
            2,
            Plane::bytes(3, 2, 2, vec![0; 12]).expect("y"),
            Plane::bytes(3, 2, 2, vec![0; 12]).expect("u"),
            Plane::bytes(3, 2, 2, vec![0; 12]).expect("v"),
        )
        .expect("frame");
        assert_eq!(frame.alpha, None);
        let alpha = Plane::bytes(3, 2, 2, vec![0xff; 12]).expect("alpha");
        let with = frame.clone().with_alpha(16, alpha.clone()).expect("alpha");
        assert_eq!(with.alpha.as_ref().map(|a| a.bit_depth), Some(16));
        assert!(frame.clone().with_alpha(0, alpha.clone()).is_err());
        assert!(frame.clone().with_alpha(17, alpha).is_err());
        let narrow = Plane::bytes(2, 2, 2, vec![0; 8]).expect("narrow");
        assert!(frame.clone().with_alpha(16, narrow).is_err());
        let bytes = Plane::bytes(3, 2, 1, vec![0; 6]).expect("bytes");
        assert!(frame.clone().with_alpha(16, bytes.clone()).is_err());
        assert!(frame.with_alpha(8, bytes).is_ok());
    }
}
