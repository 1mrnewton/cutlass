//! Bounds for one PNG datastream.
//!
//! The inflated image has a size implied by the header. Anything larger is
//! rejected before it is stored. These caps are below the format maximum so
//! a hostile header cannot allocate an unbounded picture.

/// Longest edge this codec will decode or encode.
pub(crate) const MAX_EDGE: u32 = 16_384;

/// Most pixels in one still.
pub(crate) const MAX_PIXELS: u64 = 16_777_216;

/// Largest single chunk, including an embedded profile.
pub(crate) const MAX_CHUNK: usize = 16 * 1024 * 1024;

/// Largest concatenated IDAT payload.
pub(crate) const MAX_IDAT: usize = 128 * 1024 * 1024;

/// Largest inflated text or profile.
pub(crate) const MAX_INFLATED_META: usize = 1024 * 1024;

pub(crate) fn edge_ok(width: u32, height: u32) -> bool {
    width > 0
        && height > 0
        && width <= MAX_EDGE
        && height <= MAX_EDGE
        && width <= 0x7fff_ffff
        && height <= 0x7fff_ffff
        && u64::from(width) * u64::from(height) <= MAX_PIXELS
}
