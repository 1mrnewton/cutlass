//! Bounds for one untrusted WebP packet.
//!
//! The format allows 16 383 × 16 383 pixels for a bitstream and 2^24 for an
//! animation canvas. A few bytes of VP8L can describe the largest of those,
//! so every size is checked against these caps before a buffer is
//! allocated. The caps are below the format maxima on purpose.

use cutlass_decode::CodecError;

/// Longest edge of one VP8 or VP8L bitstream. 14 bits in both headers.
pub(crate) const MAX_EDGE: u32 = 16_383;

/// Most pixels in one decoded picture or canvas: 8192 × 8192.
///
/// A 16 383 × 16 383 bitstream is legal WebP and is refused here with
/// [`CodecError::Limit`]. At this cap the RGBA picture is 256 MiB and the
/// 16-bit [`cutlass_decode::StillImage`] copy is 512 MiB.
pub(crate) const MAX_PIXELS: u64 = 1 << 26;

/// Most decoding-table entries one VP8L image may build, all groups
/// together. Each entry is four bytes.
pub(crate) const MAX_TABLE_ENTRIES: usize = 1 << 25;

/// Most chunks one packet may carry.
pub(crate) const MAX_CHUNKS_PER_PACKET: usize = 16_384;

/// Largest `ICCP` payload kept.
pub(crate) const MAX_ICC_BYTES: usize = 4 * 1024 * 1024;

/// Width and height of one bitstream, or the reason it is refused.
pub(crate) fn picture_ok(width: u32, height: u32) -> Result<(), CodecError> {
    if width == 0 || height == 0 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "webp picture size",
        });
    }
    if width > MAX_EDGE || height > MAX_EDGE {
        return Err(CodecError::Limit {
            what: "webp picture edge",
        });
    }
    pixels_ok(width, height)
}

fn pixels_ok(width: u32, height: u32) -> Result<(), CodecError> {
    if u64::from(width) * u64::from(height) > MAX_PIXELS {
        return Err(CodecError::Limit {
            what: "webp picture pixels",
        });
    }
    Ok(())
}

/// `width * height` as a buffer length. Call after [`picture_ok`].
pub(crate) fn pixel_count(width: u32, height: u32) -> Result<usize, CodecError> {
    usize::try_from(u64::from(width) * u64::from(height)).map_err(|_| CodecError::Limit {
        what: "webp picture pixels",
    })
}
