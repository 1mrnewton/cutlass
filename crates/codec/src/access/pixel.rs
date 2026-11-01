/// Pixel layout for a video packet or a decoded picture description.
///
/// Audio codecs leave this as [`PixelFormat::None`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum PixelFormat {
    #[default]
    None,
    Rgba8,
    Yuv420p,
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
    /// Identity / RGB, stored only as a tag.
    Rgb,
}
