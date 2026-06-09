//! Structured editor commands.
//!
//! UI gestures and the AI agent both emit these values; the engine applies them
//! against project/timeline state with undo/redo.

use std::path::PathBuf;

use cutlass_models::{ClipId, Generator, MediaId, RationalTime, TimeRange, TrackId, TrackKind};

/// A project-level action (media pool, not timeline placement).
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectCommand {
    /// Register a file in the media pool.
    Import { path: PathBuf },
    /// Write the current project to a `.cutlass` file.
    Save { path: PathBuf },
    /// Replace the session from a project file; every media path must exist.
    Open { path: PathBuf },
    /// Replace the session from a project file; missing media paths are kept but not relinked.
    Load { path: PathBuf },
}

/// A single structured edit against the timeline.
#[derive(Debug, Clone, PartialEq)]
pub enum EditCommand {
    /// Append a track to the timeline stack.
    AddTrack {
        kind: TrackKind,
        name: String,
    },
    /// Place a trimmed range of imported media on a track.
    AddClip {
        track: TrackId,
        media: MediaId,
        source: TimeRange,
        start: RationalTime,
    },
    /// Place a generated clip (text, solid, shape, …) on a track.
    AddGenerated {
        track: TrackId,
        generator: Generator,
        timeline: TimeRange,
    },
    /// Split a clip at a timeline position into two abutting clips.
    SplitClip { clip: ClipId, at: RationalTime },
    /// Re-place / trim a clip to occupy `timeline`.
    TrimClip { clip: ClipId, timeline: TimeRange },
    /// Move a clip to `to_track` starting at `start`, keeping its duration.
    MoveClip {
        clip: ClipId,
        to_track: TrackId,
        start: RationalTime,
    },
    /// Remove a clip, leaving a gap where it sat.
    RemoveClip { clip: ClipId },
    /// Remove a clip and slide later clips on its track left to close the gap.
    RippleDelete { clip: ClipId },
}

/// Top-level command surface: media registration or a timeline edit.
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    Project(ProjectCommand),
    Edit(EditCommand),
}

/// What an applied edit produced, for callers to act on (e.g. select the new clip).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditOutcome {
    Created(ClipId),
    CreatedTrack(TrackId),
    Updated(ClipId),
    Removed(ClipId),
}
