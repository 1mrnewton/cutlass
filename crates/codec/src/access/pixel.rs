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
}

/// Transfer function named on a video stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Transfer {
    #[default]
    Unspecified,
    Bt709,
}

/// Sample range named on a video stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ColorRange {
    #[default]
    Unspecified,
    /// Studio swing. 8-bit luma occupies 16..=235.
    Limited,
    /// Full swing. 8-bit samples occupy 0..=255.
    Full,
}

/// Matrix coefficients named on a video stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum MatrixCoefficients {
    #[default]
    Unspecified,
    Bt709,
}
