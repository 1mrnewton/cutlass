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
    /// Tightly packed 10-bit 4:2:0. Each sample is a little-endian `u16` in the low 10 bits.
    Yuv420p10,
    /// Tightly packed 12-bit 4:2:0. Each sample is a little-endian `u16` in the low 12 bits.
    Yuv420p12,
    /// Tightly packed 8-bit 4:2:2. A full-size Y plane, then U and V at half width and full height.
    Yuv422p,
    /// Tightly packed 10-bit 4:2:2. Each sample is a little-endian `u16` in the low 10 bits.
    Yuv422p10,
    /// Tightly packed 12-bit 4:2:2. Each sample is a little-endian `u16` in the low 12 bits.
    Yuv422p12,
    /// Tightly packed 8-bit 4:4:4. A full-size Y plane, then U, then V.
    Yuv444p,
    /// Tightly packed 10-bit 4:4:4. Each sample is a little-endian `u16` in the low 10 bits.
    Yuv444p10,
    /// Tightly packed 12-bit 4:4:4. Each sample is a little-endian `u16` in the low 12 bits.
    Yuv444p12,
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
    /// Identity / RGB, stored only as a tag.
    Rgb,
}
