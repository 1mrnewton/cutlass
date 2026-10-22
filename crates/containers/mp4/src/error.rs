//! Failures from reading an MP4 or MOV file.

use crate::Fourcc;

/// Why a file could not be read.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// The file is cut off.
    #[error("file ended at byte {at} while reading {need} bytes")]
    Truncated { at: u64, need: u64 },
    /// A box claims a size smaller than its header.
    #[error("box {kind} has size {size}, which is shorter than its header")]
    BadBoxSize { kind: Fourcc, size: u64 },
    /// A required box is missing.
    #[error("missing box {0}")]
    MissingBox(Fourcc),
    /// A box that must be unique appears twice.
    #[error("duplicate box {0}")]
    DuplicateBox(Fourcc),
    /// The box version is not one this reader handles.
    #[error("box {kind} has version {version}")]
    BadVersion { kind: Fourcc, version: u8 },
    /// A number does not fit the field it is stored in.
    #[error("a number in the file does not fit")]
    TooLarge,
    /// The track id is zero.
    #[error("track id is zero")]
    BadTrackId,
    /// A track timescale is zero.
    #[error("track timescale is zero")]
    ZeroTimescale,
    /// A video sample entry has a zero width or height.
    #[error("video width and height must be non-zero")]
    ZeroDimension,
    /// More than one sample description is not read yet.
    #[error("a track has more than one sample description")]
    MultipleDescriptions,
    /// The video codec is not one this reader knows.
    #[error("unsupported video sample entry {0}")]
    UnsupportedVideo(Fourcc),
    /// A required codec configuration box is missing.
    #[error("missing codec configuration {0}")]
    MissingConfig(Fourcc),
    /// An elementary-stream descriptor is not usable.
    #[error("elementary stream descriptor is invalid")]
    BadDescriptor,
    /// An audio sample rate is missing or not a positive number.
    #[error("audio sample rate is invalid")]
    BadSampleRate,
    /// An audio track reports no channels.
    #[error("audio channel count is zero")]
    ZeroChannels,
    /// Sample tables, chunk offsets, and the file bytes disagree.
    #[error("{0}")]
    Inconsistent(&'static str),
}
