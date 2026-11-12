//! Bounds a hostile file cannot push past.
//!
//! The shared [`cutlass_container::Limits`] caps one header element, one
//! access unit, the track count, one extradata blob, and the seek index.
//! These are the Matroska-specific caps on top of those.

/// Bytes [`crate::probe`] needs to see the EBML magic. A longer window lets
/// it read the DocType as well.
pub const PROBE_BYTES: usize = 4;

/// Largest EBML header element. Real headers are about 40 bytes.
pub const MAX_EBML_HEADER: u64 = 4 * 1024;

/// Top-level elements walked between the Segment start and the first
/// Cluster, and elements walked looking for the Segment.
pub const MAX_TOP_LEVEL: u32 = 4_096;

/// Largest Cues element held in memory. About 20 bytes per cue point, so
/// this is over three million points.
pub const MAX_CUES_BYTES: u64 = 64 * 1024 * 1024;

/// Frames in one laced block: the frame count is one octet (RFC 9559, 10.3).
pub const MAX_LACED_FRAMES: usize = 256;

/// Block headers a seek walks from the cue's Cluster to the landing and the
/// preroll after it. No payload byte is read on that walk.
pub const MAX_SEEK_WALK: u32 = 1_000_000;

/// Block headers seeks keep from their walks, so that a later seek over the
/// same stretch reads nothing (about 90 bytes each). Past it the kept walks
/// are dropped and kept again from the next seek on.
pub const MAX_WALKED_BLOCKS: usize = 16 * 1024;

/// Cluster headers a seek walks to build an index when the file has no Cues.
pub const MAX_CLUSTER_SCAN: u32 = 4_000_000;

/// Elements inside one Cluster that are not blocks and are skipped in a row
/// (Void, CRC-32, an unknown ID). A file of nothing but those still ends.
pub const MAX_SKIPPED_IN_A_ROW: u32 = 1_000_000;

/// Header-stripping bytes kept per track (ContentCompSettings).
pub const MAX_STRIPPED_HEADER: usize = 4 * 1024;

/// Largest CodecID string kept. Longer ids are cut for the unknown-codec tag.
pub const MAX_CODEC_ID: usize = 64;

/// Cluster time span the muxer allows before it starts a new Cluster, in
/// nanoseconds (FFmpeg's and the spec's 5-second recommendation, RFC 9559 25.1).
pub const CLUSTER_SPAN_NS: u64 = 5_000_000_000;

/// Cluster byte size the muxer allows before it starts a new Cluster
/// (RFC 9559, 25.1 recommends at most five megabytes).
pub const CLUSTER_BYTES: usize = 5 * 1024 * 1024;

/// Payload bytes the muxer holds before it writes the Tracks element when a
/// track has not delivered its first packet (and its configuration) yet.
pub const MAX_HELD_BEFORE_HEADER: u64 = 64 * 1024 * 1024;
