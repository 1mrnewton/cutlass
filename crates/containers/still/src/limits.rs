//! Bounds for an untrusted still file.
//!
//! Sizes are checked before a buffer of that size is allocated. A declared
//! chunk, marker, or RIFF size larger than the bytes still in the file is
//! refused. The file itself is not read past [`MAX_FILE_BYTES`].

use cutlass_container::Limits;

/// Largest still file this crate will accept.
///
/// One JPEG, PNG, or still WebP is one packet, so this is also the largest
/// packet. GIF and animated WebP frames share the same budget. 32 MiB covers
/// a large photograph and stops a declared-size allocation.
pub const MAX_FILE_BYTES: u64 = 32 * 1024 * 1024;

/// Most frames one GIF or animated WebP may contain.
///
/// The loop count is not applied on top of this. A file that repeats forever
/// is still one pass of its frames.
pub const MAX_FRAMES: u32 = 4096;

/// WebP chunk nesting this crate will walk.
///
/// A RIFF form contains chunks. An `ANMF` payload contains chunks. A chunk
/// inside that payload is the bottom. Deeper nesting is refused.
pub const MAX_CHUNK_DEPTH: u32 = 2;

/// GIF graphic-control delays are hundredths of a second.
///
/// GIF89a names this unit. It is not a microsecond clock.
pub const GIF_TIMESCALE: u32 = 100;

/// Delay used when a GIF frame has no graphic-control delay, or the stored
/// delay is 0.
///
/// GIF89a says a delay of 0 does not name a wait. FFmpeg 8.1.1's GIF parser
/// substitutes 10 hundredths of a second, which is 100 ms. This crate does
/// the same. A stored delay above 0 is kept.
pub const GIF_DEFAULT_DELAY: i64 = 10;

/// WebP `ANMF` durations are milliseconds.
pub const WEBP_TIMESCALE: u32 = 1000;

/// Duration used when an `ANMF` frame stores 0.
///
/// The WebP container spec leaves 0 to the implementation and notes that
/// tools use a GIF-like minimum. 100 ms is the same wait as
/// [`GIF_DEFAULT_DELAY`]. A stored duration above 0 is kept, including
/// 1 through 10.
pub const WEBP_DEFAULT_DELAY_MS: i64 = 100;

/// Cases this crate names and does not implement.
///
/// Removing a line is a matrix change. The support record in `SUPPORT.md`
/// describes each one.
pub const UNSUPPORTED: &[&str] = &[
    "apng frame sequence",
    "heic",
    "avif",
    "pixel decode",
    "huffman",
    "inflate",
    "lzw",
    "fragmented output",
    "invented frame rate",
    "infinite loop as packets",
    "gif plain text",
];

/// [`Limits`] whose packet cap is [`MAX_FILE_BYTES`].
///
/// Open also refuses a file larger than the [`Input`](cutlass_container::Input)
/// packet limit when that limit is lower.
#[must_use]
pub fn still_limits() -> Limits {
    Limits {
        max_header_bytes: MAX_FILE_BYTES,
        max_packet_bytes: MAX_FILE_BYTES,
        ..Limits::standard()
    }
}

/// The cap open will actually accept for this input.
#[must_use]
pub fn file_cap(limits: Limits) -> u64 {
    limits.max_packet_bytes.min(MAX_FILE_BYTES)
}
