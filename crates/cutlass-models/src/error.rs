use thiserror::Error;

use crate::ids::{ClipId, MediaId, TrackId};

/// Errors from model mutations that would violate a referential or layout
/// invariant.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ModelError {
    #[error("unknown track: {0}")]
    UnknownTrack(TrackId),

    #[error("unknown media: {0}")]
    UnknownMedia(MediaId),

    #[error("unknown clip: {0}")]
    UnknownClip(ClipId),

    #[error("clip overlaps an existing clip on {0}")]
    Overlap(TrackId),

    #[error("media {0} is still referenced by one or more clips")]
    MediaReferenced(MediaId),

    #[error("source range is outside the media bounds")]
    SourceOutOfBounds,

    #[error("invalid time range (negative or zero duration where positive required)")]
    InvalidRange,
}
