//! Pixel layouts that cross crate boundaries.

/// Memory layout of a decoded image, or of an image going into an encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// 8-bit 4:2:0. A Y plane, then one plane of interleaved Cb and Cr.
    Nv12,
    /// 10-bit 4:2:0 in the NV12 layout. Each sample is a little-endian `u16`
    /// with the value in the top 10 bits.
    P010,
}

/// Which code values carry picture data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorRange {
    /// Video range, 16 to 235 for 8-bit luma. Most camera and web video.
    Limited,
    /// Every code value, 0 to 255 for 8-bit. JPEG, screen recordings, and
    /// some phones.
    Full,
}
