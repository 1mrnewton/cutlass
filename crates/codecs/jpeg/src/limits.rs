//! Bounds for one JPEG bitstream.
//!
//! The frame header is what allocates the picture. These caps are below the
//! 16-bit marker maximum so a hostile size returns an error before the
//! coefficient planes exist.

/// Longest edge this codec will decode or encode.
pub const MAX_EDGE: u32 = 16_384;

/// Most pixels in one still: 8192 by 8192, 64 megapixels.
///
/// Cameras and phones save 24 to 61 megapixels (6000 by 4000, 8064 by 6048,
/// 9504 by 6336), so this leaves room for them. The WebP decoder has the same
/// cap.
pub const MAX_PIXELS: u64 = 67_108_864;

/// Largest accepted bitstream. The caller owns the bytes; this stops a scan
/// of a buffer that cannot be a still under [`MAX_PIXELS`].
pub const MAX_INPUT: usize = 64 * 1024 * 1024;

/// Most marker segments in one image. Each segment is at most 65 533 bytes.
pub(crate) const MAX_MARKERS: usize = 4_096;

/// Successive-approximation low bit. ITU-T T.81 limits the point transform.
pub(crate) const MAX_AL: u8 = 13;

pub(crate) fn dimensions_ok(width: u32, height: u32) -> bool {
    width > 0
        && height > 0
        && width <= MAX_EDGE
        && height <= MAX_EDGE
        && u64::from(width) * u64::from(height) <= MAX_PIXELS
}
