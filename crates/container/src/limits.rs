//! Bounds on what a demuxer or muxer will allocate.
//!
//! A multi-gigabyte file is read packet by packet. These limits bound one
//! header, one access unit, and the counts a hostile size field must not
//! turn into an allocation. They are not a cap on the length of a movie.

/// Allocation bounds shared by open, read, and mux.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Largest header window, in bytes.
    pub max_header_bytes: u64,
    /// Largest track count.
    pub max_tracks: u32,
    /// Largest single access unit, in bytes.
    pub max_packet_bytes: u64,
    /// Largest codec configuration blob, in bytes.
    pub max_extradata_bytes: u64,
    /// Largest edit list on one track.
    pub max_edits: u32,
    /// Largest seek index, in entries.
    pub max_index_entries: u32,
}

impl Limits {
    /// Bounds for a long recording on a desktop machine.
    ///
    /// The header cap is 8 MiB. One access unit is 16 MiB. Extradata is
    /// 1 MiB. Track count is 64. An edit list is 4,096 entries. A seek
    /// index is 16,777,216 entries, above a day of frames at 60 fps.
    /// There is no total file-size cap: the input is not buffered whole.
    #[must_use]
    pub const fn standard() -> Self {
        Self {
            max_header_bytes: 8 * 1024 * 1024,
            max_tracks: 64,
            max_packet_bytes: 16 * 1024 * 1024,
            max_extradata_bytes: 1024 * 1024,
            max_edits: 4096,
            max_index_entries: 16 * 1024 * 1024,
        }
    }
}

impl Default for Limits {
    fn default() -> Self {
        Self::standard()
    }
}
