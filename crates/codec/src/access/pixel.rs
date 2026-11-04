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
///
/// The names are the `colour_primaries` values in ISO/IEC 23091-2. Values 5
/// and 6 share [`Self::Bt601`]: both are BT.601, 625-line and 525-line.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ColorPrimaries {
    #[default]
    Unspecified,
    Bt709,
    /// SMPTE 170M / BT.601.
    Bt601,
    Bt2020,
    /// BT.470 System M. Value 4.
    Bt470m,
    /// SMPTE 240M. Value 7. The chromaticities match SMPTE 170M.
    Smpte240m,
    /// Generic film, illuminant C. Value 8.
    Film,
    /// SMPTE ST 428-1. Value 10.
    Smpte428,
    /// SMPTE ST 431-2, DCI P3. Value 11.
    Smpte431,
    /// SMPTE ST 432-1, Display P3. Value 12.
    Smpte432,
    /// EBU Tech. 3213-E. Value 22.
    Ebu3213,
}

/// Transfer function named on a video stream.
///
/// The names are the `transfer_characteristics` values in ISO/IEC 23091-2.
/// Value 6 is the same function as value 1, so both are [`Self::Bt709`].
/// Values 14 and 15 are the BT.2020 10-bit and 12-bit curves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Transfer {
    #[default]
    Unspecified,
    Bt709,
    /// Gamma 2.2 and gamma 2.8, values 4 and 5.
    Bt601,
    /// BT.2020 10-bit and 12-bit curves, values 14 and 15.
    Bt2020,
    /// SMPTE 240M. Value 7.
    Smpte240m,
    /// Linear. Value 8.
    Linear,
    /// Logarithmic, 100:1. Value 9.
    Log100,
    /// Logarithmic, 100·√10:1. Value 10.
    LogSqrt,
    /// IEC 61966-2-4. Value 11.
    Iec61966_2_4,
    /// BT.1361 extended gamut. Value 12.
    Bt1361,
    /// IEC 61966-2-1, sRGB. Value 13.
    Srgb,
    /// SMPTE ST 2084, perceptual quantizer. Value 16.
    Pq,
    /// SMPTE ST 428-1. Value 17.
    Smpte428,
    /// ARIB STD-B67, hybrid log-gamma. Value 18.
    Hlg,
}

/// Luma/chroma matrix named on a video stream.
///
/// The names are the `matrix_coefficients` values in ISO/IEC 23091-2.
/// Values 5 and 6 are the same coefficients, so both are [`Self::Bt601`].
/// Values 9 and 10 are BT.2020 non-constant and constant luminance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ColorMatrix {
    #[default]
    Unspecified,
    Bt709,
    Bt601,
    Bt2020,
    /// Identity, GBR. Value 0.
    Identity,
    /// FCC. Value 4.
    Fcc,
    /// SMPTE 240M. Value 7.
    Smpte240m,
    /// YCgCo. Value 8.
    Ycgco,
    /// SMPTE 2085. Value 11.
    Smpte2085,
    /// Chromaticity-derived, non-constant luminance. Value 12.
    ChromaDerived,
    /// Chromaticity-derived, constant luminance. Value 13.
    ChromaDerivedConstant,
    /// ICtCp. Value 14.
    Ictcp,
    /// IPT-C2. Value 15.
    IptC2,
    /// YCgCo-R with even bit addition. Value 16.
    YcgcoRe,
    /// YCgCo-R with odd bit addition. Value 17.
    YcgcoRo,
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
