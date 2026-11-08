//! Bounds checked before an Ogg page or packet is stored.
//!
//! A long file is read page by page. These caps bound one page, one packet,
//! one resync scan, and the work a seek or an open may do on hostile input.

/// Bytes a probe looks at: the 4-byte capture pattern and the version.
pub const PROBE_BYTES: usize = 5;

/// Fixed page header, before the segment table (RFC 3533 section 6).
pub const PAGE_HEADER_BYTES: usize = 27;

/// Largest page: 27 header bytes, 255 lacing values, 255 segments of 255.
pub const MAX_PAGE_BYTES: usize = PAGE_HEADER_BYTES + 255 + 255 * 255;

/// Bytes searched for the next capture pattern after garbage or a page
/// whose checksum fails. A larger run of garbage ends the read.
pub const MAX_RESYNC_BYTES: u64 = 256 * 1024;

/// Largest Opus audio packet per Opus stream inside one Ogg packet
/// (RFC 7845 section 6). A larger packet is refused, not allocated.
pub const MAX_OPUS_PACKET_PER_STREAM: u64 = 61_440;

/// Largest user comment count kept from `OpusTags`.
pub const MAX_COMMENTS: u32 = 65_536;

/// Bytes scanned after the headers for each Opus stream's first page with
/// a granule position, at open.
pub const MAX_START_SCAN_BYTES: u64 = 8 * 1024 * 1024;

/// Bytes scanned backwards from the end of the first link for a stream's
/// last pages, at open.
pub const MAX_TAIL_SCAN_BYTES: u64 = 8 * 1024 * 1024;

/// Bisection probes in one seek.
pub const MAX_SEEK_PROBES: u32 = 64;

/// Bisection probes in the open-time search for the end of the first link.
pub const MAX_LINK_PROBES: u32 = 64;

/// Opus seek pre-roll: 80 ms at 48 kHz (RFC 7845 section 4.6).
pub const OPUS_PREROLL: i64 = 3_840;

/// Opus granule positions count 48 kHz samples (RFC 7845 section 4).
pub const OPUS_RATE: u32 = 48_000;

/// The mux closes a page once its body reaches this many bytes.
pub const MUX_PAGE_BYTES: usize = 4_096;
