use thiserror::Error;

use crate::clip::SlotMedia;
use crate::ids::{CaptionGroupId, ClipId, MarkerId, MediaId, TrackId};
use crate::media::MediaKind;
use crate::schema::ProjectSchema;
use crate::time::{Rational, TimeError};
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

    #[error("clip id {0} already exists on the timeline")]
    DuplicateClip(ClipId),

    #[error("unknown marker: {0}")]
    UnknownMarker(MarkerId),

    #[error("unknown caption group: {0}")]
    UnknownCaptionGroup(CaptionGroupId),

    #[error("caption group {0} already exists on the timeline")]
    DuplicateCaptionGroup(CaptionGroupId),

    #[error("clip {0} is not a caption cue")]
    NotACaptionCue(ClipId),

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

    // --- templates --------------------------------------------------------
    #[error("template slot {slot} accepts {accepts:?} but {found:?} media was supplied")]
    SlotMediaMismatch {
        slot: ClipId,
        accepts: SlotMedia,
        found: MediaKind,
    },

    #[error("media supplied for template slot {slot} cannot cover its locked duration")]
    SlotDurationUnmet { slot: ClipId },

    #[error("template slot {slot} is speed-ramped or reversed; filling it is not supported")]
    SlotRetimeUnsupported { slot: ClipId },

    #[error("{given} picks supplied for a template with {slots} fill slots")]
    TooManyPicks { given: usize, slots: usize },
}

/// Bridge the shared `cutlass-core` time errors into [`ModelError`] so model
/// methods that propagate rational-time arithmetic with `?` keep returning a
/// single error type.
impl From<TimeError> for ModelError {
    fn from(err: TimeError) -> Self {
        match err {
            TimeError::RateMismatch { expected, got } => ModelError::RateMismatch { expected, got },
            TimeError::Overflow => ModelError::TimeOverflow,
        }
    }
}
