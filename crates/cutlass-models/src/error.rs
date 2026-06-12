use thiserror::Error;

use crate::ids::{ClipId, MarkerId, MediaId, TrackId};
use crate::schema::ProjectSchema;
use crate::time::Rational;
use crate::track::TrackKind;

/// Errors from model mutations that would violate a referential or layout
/// invariant.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ModelError {
    #[error("unsupported project schema (found {found:?}, expected {expected:?})")]
    UnsupportedProjectSchema {
        found: ProjectSchema,
        expected: ProjectSchema,
    },

    #[error("invalid project file: {0}")]
    InvalidProjectFile(String),

    #[error("unknown track: {0}")]
    UnknownTrack(TrackId),

    #[error("unknown media: {0}")]
    UnknownMedia(MediaId),

    #[error("unknown clip: {0}")]
    UnknownClip(ClipId),

    #[error("unknown marker: {0}")]
    UnknownMarker(MarkerId),

    #[error("clip overlaps an existing clip on {0}")]
    Overlap(TrackId),

    #[error("track {track} ({kind:?}) cannot hold this clip")]
    IncompatibleTrackKind { track: TrackId, kind: TrackKind },

    #[error("media {0} is still referenced by one or more clips")]
    MediaReferenced(MediaId),

    #[error("source range is outside the media bounds")]
    SourceOutOfBounds,

    #[error("invalid time range (negative or zero duration where positive required)")]
    InvalidRange,

    #[error("invalid transform: {0}")]
    InvalidTransform(String),

    #[error("invalid parameter: {0}")]
    InvalidParam(String),

    #[error("rate mismatch: expected {expected:?}, got {got:?}")]
    RateMismatch { expected: Rational, got: Rational },

    #[error("time arithmetic overflow")]
    TimeOverflow,
}
