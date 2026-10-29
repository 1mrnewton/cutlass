//! Bounds checked before a WAV chunk is stored.
//!
//! A multi-gigabyte `data` chunk is not buffered. These caps apply to the
//! header, one packet, and the metadata chunks this crate keeps.

/// LIST and RIFF nesting. A deeper file is a nested bomb.
pub const MAX_DEPTH: u32 = 8;

/// Chunk headers examined in one file, including chunks inside LIST.
pub const MAX_CHUNKS: u32 = 4_096;

/// Unknown chunk payload kept for a mux round trip.
///
/// A larger payload that still fits in its parent is skipped and named in
/// [`crate::DroppedChunk`]. It is not allocated.
pub const MAX_KEPT_CHUNK: u64 = 64 * 1024;

/// `ds64` table rows. Each row is 12 bytes. A longer table is refused.
pub const MAX_DS64_ROWS: u32 = 64;

/// Bytes scanned for an MPEG sync inside a compressed WAV payload.
pub const MAX_MPEG_RESYNC: u64 = 64 * 1024;

/// Header bytes that accept or refuse this family: `RIFF`/`RF64`, size, `WAVE`.
pub const PROBE_BYTES: usize = 12;

/// Codec-configuration header stored in stream extradata.
///
/// The channel mask is a little-endian `u32` at [`CHANNEL_MASK_OFFSET`].
pub const EXTRA_HEADER: usize = 16;

/// Byte offset of `dwChannelMask` in the extradata header.
pub const CHANNEL_MASK_OFFSET: usize = 8;
