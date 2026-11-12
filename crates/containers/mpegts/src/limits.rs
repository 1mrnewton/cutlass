//! Work and memory bounds for a hostile transport stream.
//!
//! `cutlass_container::Limits` caps one access unit, the track count, and
//! the header window. These constants bound what a transport stream adds on
//! top: the resync scan, one PSI section, the open-time scan, the tail scan
//! that finds the duration, and the seek bisection.

/// One transport packet, ITU-T H.222.0 clause 2.4.3.2.
pub const TS_PACKET_BYTES: usize = 188;

/// The sync byte that starts every transport packet.
pub const SYNC_BYTE: u8 = 0x47;

/// Bytes the probe looks at: five packets at the widest stride (204) with
/// the first sync anywhere in the first packet.
pub const PROBE_BYTES: usize = 1024;

/// Consecutive sync bytes at one stride that settle the packet size.
pub const SYNC_RUN: usize = 5;

/// Bytes searched for sync bytes after a lost sync before a read fails.
/// The next read continues the search, so a long run of garbage costs a
/// bounded amount of work per call.
pub const MAX_RESYNC_BYTES: usize = 1 << 20;

/// Largest PSI section, `section_length` 4093 plus its three header bytes
/// (clause 2.4.4.1). PAT and PMT are capped lower, at 1024, by the
/// specification; private sections use the full span.
pub const MAX_SECTION_BYTES: usize = 4096;

/// Programs read from the PAT. A PAT section holds at most 253.
pub const MAX_PROGRAMS: usize = 256;

/// Descriptors walked in one descriptor loop.
pub const MAX_DESCRIPTORS: usize = 256;

/// Bytes read from the start of the file while open looks for the PAT,
/// the PMT, and the first access unit of each stream.
pub const MAX_OPEN_SCAN_BYTES: u64 = 16 * 1024 * 1024;

/// Bytes read back from the end of the file to find the last timestamp.
/// The window starts at 1 MiB and doubles up to this bound.
pub const MAX_TAIL_SCAN_BYTES: u64 = 16 * 1024 * 1024;

/// Bisection probes in one seek.
pub const MAX_SEEK_PROBES: u32 = 64;

/// Bytes one bisection probe reads before it gives up on finding a
/// timestamp of the track it seeks.
pub const MAX_PROBE_WINDOW_BYTES: u64 = 4 * 1024 * 1024;

/// Seek windows widened backwards, each twice the last, while looking for
/// the keyframe before the target.
pub const MAX_SEEK_WIDENINGS: u32 = 24;

/// Packets of other tracks held after a seek until the landing packet is
/// returned.
pub const MAX_HELD_PACKETS: usize = 4096;

/// Bytes the PES buffers and pending access units of all tracks may hold
/// together. One access unit is still capped by `Limits::max_packet_bytes`.
pub const MAX_BUFFERED_BYTES: usize = 96 * 1024 * 1024;

/// Bytes of a PES payload kept at open to learn a stream's parameters.
pub const MAX_PARAMS_BYTES: usize = 1 << 20;

/// PES packets of one stream searched at open for its parameters.
pub const MAX_PARAMS_PES: u32 = 64;
