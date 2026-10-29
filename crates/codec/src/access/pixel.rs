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
