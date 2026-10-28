//! Resource limits for an untrusted ISO base media file.
//!
//! Sizes are checked before allocation. A declared size larger than the
//! remaining parent is an error, not an allocation of that size.

/// Nested box depth. A movie is `moov/trak/mdia/minf/stbl`, well under this.
pub const MAX_DEPTH: u32 = 32;

/// Largest `moov`, `moof`, `meta`, or `sidx` loaded into memory.
pub const MAX_METADATA_BYTES: u64 = 64 * 1024 * 1024;

/// Tracks in one file.
pub const MAX_TRACKS: usize = 64;

/// Samples indexed for one track, including samples appended from fragments.
pub const MAX_SAMPLES: u64 = 4_000_000;

/// One access unit.
pub const MAX_PACKET_BYTES: u64 = 64 * 1024 * 1024;

/// Edit list entries on one track.
pub const MAX_EDITS: usize = 4_096;

/// Compatible brands stored from `ftyp`.
pub const MAX_BRANDS: usize = 32;

/// Sample descriptions in one `stsd`.
pub const MAX_STSD: usize = 16;

/// Run-length rows in `stts`, `ctts`, or `stsc`, and chunk offsets.
pub const MAX_TABLE_ROWS: usize = 4_000_000;

/// Boxes visited while opening one file.
pub const MAX_ATOMS: u32 = 2_000_000;

/// Bytes of a header `probe` needs to see a brand. `open` reads the real size.
pub const PROBE_BYTES: usize = 64;

/// Payloads buffered when `moov` is written before `mdat`.
pub const MAX_BUFFERED_MUX_BYTES: u64 = 256 * 1024 * 1024;

/// Samples gathered into one movie fragment before it is flushed.
pub const DEFAULT_SAMPLES_PER_FRAGMENT: u32 = 16;
