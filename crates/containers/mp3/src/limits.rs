//! Bounds for untrusted MP3 bytes.
//!
//! Sizes are checked before a buffer is allocated. A declared ID3 size is
//! never used as an allocation length.

/// Largest ID3v2 tag this demuxer will skip.
///
/// The tag is not loaded. A larger declared size is [`crate::ContainerError::Limit`].
pub const MAX_ID3_BYTES: u64 = 16 * 1024 * 1024;

/// Largest APEv2 footer span this demuxer will skip.
pub const MAX_APE_BYTES: u64 = 16 * 1024 * 1024;

/// How far a resync may walk without a confirmed Layer III frame.
///
/// FFmpeg's MP3 demuxer uses the same 64 KiB ceiling when it hunts for two
/// consecutive frames.
pub const MAX_RESYNC_BYTES: u64 = 64 * 1024;

/// Largest Layer III frame this demuxer will accept.
///
/// A legal MPEG-1 frame at 320 kbit/s and 32 kHz is 1440 bytes. Free format
/// is capped here as well. LAME treats 2880 as the free-format ceiling.
pub const MAX_FRAME_BYTES: u32 = 4096;

/// Bytes searched on either side of an index position.
///
/// The search does not restart at the beginning of the audio.
pub const SEEK_WINDOW_BYTES: u64 = 4096;

/// Largest VBRI seek table this demuxer will store.
pub const MAX_VBRI_ENTRIES: usize = 4096;

/// Leading ID3v2 tags skipped before the first frame.
pub const MAX_LEADING_ID3: u32 = 4;
