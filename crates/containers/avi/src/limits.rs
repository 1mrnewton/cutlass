//! Bounds this crate checks before it walks, stores, or allocates.
//!
//! [`cutlass_container::Limits`] still caps one header, one packet, one
//! extradata blob, the track count, and the index. These constants bound the
//! AVI-specific walks a hostile file could stretch.

/// Header bytes that accept or refuse this family: `RIFF`, size, `AVI `.
pub const PROBE_BYTES: usize = 12;

/// Stream numbers are two decimal digits in a chunk id (`00dc` to `99wb`).
pub const MAX_STREAMS: u32 = 100;

/// Chunks examined at one level of `hdrl`, or at the top of one RIFF.
pub const MAX_CHUNKS: u32 = 4_096;

/// `RIFF` segments: the first `AVI ` one plus OpenDML `AVIX` extensions.
pub const MAX_RIFFS: u32 = 65_536;

/// Entries one OpenDML super index may name.
pub const MAX_SUPER_ENTRIES: u32 = 65_536;

/// Bytes a rebuild scan looks through for the next chunk header after a
/// broken one. Past this the rest of that `movi` list is not indexed.
pub const MAX_RESYNC: u64 = 1024 * 1024;

/// Bytes of a video chunk a rebuild scan reads to tell a keyframe.
pub const SNIFF_BYTES: u64 = 4_096;

/// A file is read in time order rather than byte order when one stream
/// runs more than this many seconds behind another in the byte order.
pub const MAX_DRIFT_SECONDS: i128 = 2;

/// Empty frames the muxer writes to fill one gap in a video stream.
pub const MAX_SKIPPED_FRAMES: i64 = 60_000;
