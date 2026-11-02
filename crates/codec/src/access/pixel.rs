/// Pixel layout for a video packet or a decoded picture description.
///
/// Audio codecs leave this as [`PixelFormat::None`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum PixelFormat {
    #[default]
    None,
    Rgba8,
    /// Tightly packed 8-bit 4:2:0. A full-size Y plane, then U, then V.
    Yuv420p,
    /// 8-bit 4:2:2. Chroma is `(width + 1) / 2` by `height`.
    Yuv422p,
    /// 8-bit 4:4:0. Chroma is `width` by `(height + 1) / 2`.
    Yuv440p,
    /// 8-bit 4:4:4.
    Yuv444p,
    /// 8-bit 4:4:4 stored as G, B, R.
    Gbrp,
    /// 10-bit 4:2:0, little-endian samples.
    Yuv420p10,
    /// 12-bit 4:2:0, little-endian samples.
    Yuv420p12,
    /// 10-bit 4:2:2, little-endian samples.
    Yuv422p10,
    /// 12-bit 4:2:2, little-endian samples.
    Yuv422p12,
    /// 10-bit 4:4:0, little-endian samples.
    Yuv440p10,
    /// 12-bit 4:4:0, little-endian samples.
    Yuv440p12,
    /// 10-bit 4:4:4, little-endian samples.
    Yuv444p10,
    /// 12-bit 4:4:4, little-endian samples.
    Yuv444p12,
    /// 10-bit 4:4:4 stored as G, B, R, little-endian samples.
    Gbrp10,
    /// 12-bit 4:4:4 stored as G, B, R, little-endian samples.
    Gbrp12,
    /// One tightly packed 8-bit plane, `width` by `height`.
    Gray8,
    /// One 10-bit plane. Each sample is a little-endian `u16` in the low 10 bits.
    Gray10,
}

/// Color primaries named on a video stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ColorPrimaries {
    #[default]
    Unspecified,
    Bt709,
    /// SMPTE 170M / BT.601.
    Bt601,
    Bt2020,
}

/// Transfer function named on a video stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Transfer {
    #[default]
    Unspecified,
    Bt709,
    /// SMPTE 170M transfer, the usual partner of [`ColorPrimaries::Bt601`].
    Bt601,
    Bt2020,
}

/// Luma/chroma matrix named on a video stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ColorMatrix {
    #[default]
    Unspecified,
    Bt709,
    Bt601,
    Bt2020,
}

/// Where a 4:2:0 chroma sample sits relative to luma.
///
/// The values are the AV1 `chroma_sample_position` codes. A sequence that does
/// not carry the field uses [`ChromaSamplePosition::Unknown`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ChromaSamplePosition {
    #[default]
    Unknown,
    Vertical,
    Colocated,
    Reserved,
}

/// Sample range of an 8-bit picture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ColorRange {
    #[default]
    Unspecified,
    /// Studio swing, luma 16–235.
    Limited,
    /// The same studio swing. A VP9 header records this name.
    Studio,
    /// Full swing, 0–255.
    Full,
}

/// Matrix used to relate Y′CbCr to RGB. The codec does not convert.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum MatrixCoefficients {
    #[default]
    Unspecified,
    /// ITU-R BT.709.
    Bt709,
    /// ITU-R BT.601.
    Bt601,
    /// ITU-R BT.2020.
    Bt2020,
    /// Identity / RGB, stored only as a tag.
    Rgb,
}

impl PixelFormat {
    /// Chroma width and height for a planar format. `None` when the format has no chroma planes.
    #[must_use]
    pub const fn chroma_wh(self, width: u32, height: u32) -> Option<(u32, u32)> {
        match self {
            Self::Yuv420p | Self::Yuv420p10 | Self::Yuv420p12 => {
                Some((width.div_ceil(2), height.div_ceil(2)))
            }
            Self::Yuv422p | Self::Yuv422p10 | Self::Yuv422p12 => Some((width.div_ceil(2), height)),
            Self::Yuv440p | Self::Yuv440p10 | Self::Yuv440p12 => Some((width, height.div_ceil(2))),
            Self::Yuv444p
            | Self::Yuv444p10
            | Self::Yuv444p12
            | Self::Gbrp
            | Self::Gbrp10
            | Self::Gbrp12 => Some((width, height)),
            Self::None | Self::Rgba8 | Self::Gray8 | Self::Gray10 => None,
        }
    }
}
