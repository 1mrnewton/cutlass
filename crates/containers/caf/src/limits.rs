//! Bounds checked before anything a header declares is allocated or walked.
//!
//! The sound data is never buffered whole. These caps apply to the header
//! chunks this crate keeps, to one packet, and to the packet table.

/// Bytes `probe` needs: `caff`, version, flags, then the `desc` chunk type.
pub const PROBE_BYTES: usize = 12;

/// Chunk headers examined in one file. A longer chain is refused.
pub const MAX_CHUNKS: u32 = 4_096;

/// Channels in one frame. `StreamParams` carries a `u16`; 64 matches AIFF.
pub const MAX_CHANNELS: u32 = 64;

/// Frames one packet may describe, in `desc` or in a `pakt` entry.
///
/// The largest real values are ALAC's 2^18 limit and FLAC's 65,535. A
/// larger count is refused before it becomes a timestamp.
pub const MAX_FRAMES_PER_PACKET: u32 = 1 << 20;

/// Bits per channel stored in `desc`.
pub const MAX_BITS_PER_CHANNEL: u32 = 64;

/// Largest `info` chunk that is parsed. A larger one is skipped and named.
pub const MAX_INFO_BYTES: u64 = 64 * 1024;

/// Bytes of kept chunks (`mark`, `regn`, `strg`, `uuid`, `midi`) held for
/// one file, all of them together. On open, a kept chunk that would pass
/// this is skipped, still listed in `chunks()`; the muxer refuses it.
pub const MAX_KEPT_BYTES: u64 = 1024 * 1024;

/// Largest `chan` chunk: the 12-byte head and one 20-byte description per
/// channel, for [`MAX_CHANNELS`] channels.
pub const MAX_CHAN_BYTES: u64 = 12 + 20 * MAX_CHANNELS as u64;

/// Linear PCM is returned in packets of whole frames up to this many bytes,
/// the grouping libavformat's CAF demuxer uses. A frame larger than this is
/// returned alone.
pub const PCM_PACKET_BYTES: u64 = 4_096;

/// One seek checkpoint is kept for this many packets of a packet table.
pub const CHECKPOINT_EVERY: u64 = 64;

/// Bytes read at a time while decoding a packet table.
pub(crate) const PAKT_BLOCK: usize = 64 * 1024;

/// Longest variable-length integer in a packet table: five bytes carry 35
/// bits, enough for any `u32`.
pub const MAX_VARINT_BYTES: u32 = 5;
