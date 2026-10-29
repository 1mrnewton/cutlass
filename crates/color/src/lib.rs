//! CPU color conversion for the software path.
//!
//! A picture is planar Y′CbCr or a platform buffer. Planar 4:2:0, 4:2:2, and
//! 4:4:4 become tightly packed 8-bit RGBA, the bytes the software path samples.
//! Semi-planar [`Layout::Nv12`] is refused. 10-bit samples are scaled into that
//! same 8-bit RGBA; a right shift is not the conversion.
//!
//! Rec.709, HLG, and PQ are transfer functions. HLG here is the BT.2100 OETF
//! and its inverse. The HLG system-gamma OOTF is not applied. A segment HDR
//! mode and the draft color-space integer are both inputs, and neither is
//! rewritten into the other. A platform buffer stays a platform buffer when
//! the preview and the encoder both accept it. This crate does not download it.
//!
//! The crate depends on `cutlass-codec` only. It does not sample a GPU texture.

#![forbid(unsafe_code)]

mod convert;
mod lut;
mod transfer;

use std::fmt;

use cutlass_codec::access::{
    ColorPrimaries, ColorRange, MatrixCoefficients, PixelFormat, Transfer, Yuv420p,
};

pub use lut::{Cube, Lut, ThreeDl};
pub use transfer::Curve;

/// Longest edge this crate will convert. Matches the software framebuffer.
pub const MAX_EDGE: u32 = 8_192;

/// Planar layouts this crate understands. [`Layout::Nv12`] is named and refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Planar 4:2:0. Chroma is half the width and half the height.
    Yuv420,
    /// Planar 4:2:2. Chroma is half the width and the full height.
    Yuv422,
    /// Planar 4:4:4. Chroma matches the luma grid.
    Yuv444,
    /// Interleaved chroma, semi-planar 4:2:0. Not converted.
    Nv12,
}

/// Stored sample width. Ten-bit samples are little-endian `u16` in `0..=1023`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Depth {
    Eight,
    Ten,
}

/// Y′CbCr matrix. Unspecified codec tags become [`Matrix::Bt709`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Matrix {
    #[default]
    Bt709,
    Bt601,
    Bt2020,
}

impl Matrix {
    /// Codec coefficients. [`MatrixCoefficients::Rgb`] is not a Y′CbCr matrix.
    pub fn from_coefficients(value: MatrixCoefficients) -> Result<Self, Error> {
        match value {
            MatrixCoefficients::Unspecified | MatrixCoefficients::Bt709 => Ok(Self::Bt709),
            MatrixCoefficients::Bt601 => Ok(Self::Bt601),
            MatrixCoefficients::Rgb => Err(Error::Matrix),
        }
    }
}

/// One CPU picture. Planes are borrowed. `stride` is bytes per row; `0` is tight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Picture<'a> {
    pub layout: Layout,
    pub depth: Depth,
    pub width: u32,
    pub height: u32,
    pub y: &'a [u8],
    pub u: &'a [u8],
    pub v: &'a [u8],
    pub y_stride: usize,
    pub chroma_stride: usize,
    pub range: ColorRange,
    pub matrix: Matrix,
    pub primaries: ColorPrimaries,
    pub curve: Curve,
}

impl<'a> Picture<'a> {
    /// Tight 8-bit 4:2:0 from a codec picture. Tags are passed beside the planes.
    #[must_use]
    pub fn from_yuv420p(
        frame: &'a Yuv420p,
        range: ColorRange,
        matrix: Matrix,
        primaries: ColorPrimaries,
        curve: Curve,
    ) -> Self {
        Self {
            layout: Layout::Yuv420,
            depth: Depth::Eight,
            width: frame.width(),
            height: frame.height(),
            y: frame.y(),
            u: frame.u(),
            v: frame.v(),
            y_stride: 0,
            chroma_stride: 0,
            range,
            matrix,
            primaries,
            curve,
        }
    }
}

/// HDR display mode on one segment. `mode` is not a draft color-space code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SegmentHdr {
    pub mode: i32,
    pub intensity: f64,
    pub nits: f64,
}

impl SegmentHdr {
    /// Mode 0, full intensity, no peak override.
    #[must_use]
    pub const fn off() -> Self {
        Self {
            mode: 0,
            intensity: 1.0,
            nits: 0.0,
        }
    }
}

impl Default for SegmentHdr {
    fn default() -> Self {
        Self::off()
    }
}

/// What the converter needs besides the picture.
///
/// `draft` is the integer stored on the draft. `segment` is that clip's HDR
/// setting. Conversion does not write one into the other.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Request {
    pub draft: i32,
    pub segment: SegmentHdr,
}

impl Request {
    /// Draft code with segment HDR off.
    #[must_use]
    pub const fn draft(code: i32) -> Self {
        Self {
            draft: code,
            segment: SegmentHdr::off(),
        }
    }
}

impl Default for Request {
    fn default() -> Self {
        Self::draft(0)
    }
}

/// A platform buffer the CPU path may leave in place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlatformView {
    pub token: u64,
    pub preview_accepts: bool,
    pub encoder_accepts: bool,
}

/// CPU planes or a platform buffer. The platform arm carries no pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Input<'a> {
    Cpu(Picture<'a>),
    Platform(PlatformView),
}

/// 8-bit RGBA the software path samples, plus the draft and segment values.
#[derive(Clone, Debug, PartialEq)]
pub struct Working {
    pub width: u32,
    pub height: u32,
    /// [`PixelFormat::Rgba8`], four bytes per pixel, row by row.
    pub format: PixelFormat,
    pub rgba: Vec<u8>,
    /// The draft color-space integer that was requested.
    pub draft: i32,
    /// The segment HDR setting that was requested.
    pub segment: SegmentHdr,
}

impl Working {
    /// One pixel, or nothing when `x` or `y` is outside the picture.
    #[must_use]
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let row = usize::try_from(y).ok()?;
        let col = usize::try_from(x).ok()?;
        let width = usize::try_from(self.width).ok()?;
        let index = row.checked_mul(width)?.checked_add(col)?.checked_mul(4)?;
        let bytes = self.rgba.get(index..index + 4)?;
        Some([bytes[0], bytes[1], bytes[2], bytes[3]])
    }
}

/// Either working RGBA, or a platform buffer that was not downloaded.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    Working(Working),
    /// Both consumers accept the buffer, so the CPU converter did not run.
    Platform {
        token: u64,
    },
}

/// Why a conversion or a LUT read failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// Width or height is zero.
    Empty,
    /// 4:2:0 needs even edges. 4:2:2 needs an even width.
    Odd,
    /// A plane is shorter than its width, height, and stride.
    Length,
    /// An edge, a LUT, or a byte length is past the limit.
    Limit,
    /// A 10-bit sample is above 1023, or a LUT value is not usable.
    Range,
    /// This layout is not converted. [`Layout::Nv12`] is the named case.
    Layout(Layout),
    /// The codec matrix is RGB, which is not Y′CbCr.
    Matrix,
    /// The draft integer is not one of -1, 0, 2, 67, or 2084.
    Draft(i32),
    /// The segment mode is outside 0 through 11.
    Segment(i32),
    /// Intensity is non-finite or outside 0 through 1.
    Intensity,
    /// Peak nits are negative or non-finite.
    Nits,
    /// The platform buffer is missing a consumer, and this crate will not download it.
    Platform,
    /// The LUT text is not a complete cube or 3dl.
    Lut(LutIssue),
}

/// Why a LUT file was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LutIssue {
    /// The file ended before every lattice point was present.
    Truncated,
    /// A cube is missing `LUT_3D_SIZE`, or a 3dl is missing its mesh line.
    Header,
    /// The lattice size is outside 2 through 64.
    Size,
    /// A 3dl mesh is not strictly increasing.
    Mesh,
    /// A sample could not be parsed, or a cube domain is empty.
    Value,
    /// `LUT_1D_SIZE` is not a 3D LUT.
    OneDimensional,
    /// The text is longer than 8 MiB.
    Limit,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::Empty => "picture size is empty",
            Self::Odd => "subsampling needs an even edge",
            Self::Length => "plane is shorter than its size",
            Self::Limit => "picture or lut is past the limit",
            Self::Range => "sample is outside the stored range",
            Self::Layout(Layout::Nv12) => "nv12 is not converted",
            Self::Layout(_) => "layout is not converted",
            Self::Matrix => "rgb is not a ycbcr matrix",
            Self::Draft(_) => "draft color space is not a known code",
            Self::Segment(_) => "segment hdr mode is outside 0 through 11",
            Self::Intensity => "segment hdr intensity is not in 0 through 1",
            Self::Nits => "segment hdr nits are not usable",
            Self::Platform => "platform buffer was not converted on the cpu",
            Self::Lut(LutIssue::Truncated) => "lut is truncated",
            Self::Lut(LutIssue::Header) => "lut header is missing",
            Self::Lut(LutIssue::Size) => "lut size is outside 2 through 64",
            Self::Lut(LutIssue::Mesh) => "3dl mesh is not strictly increasing",
            Self::Lut(LutIssue::Value) => "lut sample is not usable",
            Self::Lut(LutIssue::OneDimensional) => "1d lut is not converted",
            Self::Lut(LutIssue::Limit) => "lut text is past the limit",
        };
        formatter.write_str(text)
    }
}

impl std::error::Error for Error {}

/// Convert `input` for `request`.
///
/// A platform buffer is returned as [`Outcome::Platform`] only when both the
/// preview and the encoder accept it. Otherwise the error is [`Error::Platform`]
/// and no CPU pixels are produced.
pub fn convert(input: Input<'_>, request: &Request) -> Result<Outcome, Error> {
    match input {
        Input::Platform(view) => {
            if view.preview_accepts && view.encoder_accepts {
                return Ok(Outcome::Platform { token: view.token });
            }
            Err(Error::Platform)
        }
        Input::Cpu(picture) => convert::working(&picture, request).map(Outcome::Working),
    }
}

/// Remap RGB through a cube or a 3dl. Alpha is left as it was. Draft and
/// segment values on `working` are left as they were.
pub fn apply_lut(working: &mut Working, lut: &Lut) -> Result<(), Error> {
    lut::apply(working, lut)
}

/// Codec transfer tags. BT.2020's 10-bit curve is the Rec.709 OETF.
/// HLG and PQ are not codec tags; they are [`Curve::Hlg`] and [`Curve::Pq`].
#[must_use]
pub fn curve_from_codec(value: Transfer) -> Curve {
    Curve::from_codec(value)
}

#[cfg(test)]
mod tests;
