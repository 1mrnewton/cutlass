//! Structured editor commands.
//!
//! UI gestures and the AI agent both emit these values; the engine applies them
//! against project/timeline state with undo/redo.

use std::path::PathBuf;

use cutlass_models::{
    ClipId, ClipParam, ClipTransform, Easing, Generator, MediaId, ParamValue, RationalTime,
    TimeRange, TrackId, TrackKind,
};

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
    /// Render the timeline to an H.264 MP4 at the project frame rate.
    Export { path: PathBuf },
}

/// A single structured edit against the timeline.
#[derive(Debug, Clone, PartialEq)]
pub enum EditCommand {
    /// Add a track to the timeline stack.
    AddTrack {
        kind: TrackKind,
        name: String,
        /// Stack position (0 = bottom layer, composited first). `None`
        /// appends to the top of the stack. Clamped to the stack height.
        index: Option<usize>,
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
    /// Replace a generated clip's content (e.g. edit a title's text, recolor
    /// a shape). Rejected for media-backed clips. The inverse restores the
    /// previous generator.
    SetGenerator { clip: ClipId, generator: Generator },
    /// Set a clip's spatial transform (position/scale/rotation/opacity on
    /// the canvas). Rejected for audio-track clips. The inverse restores the
    /// previous transform.
    ///
    /// `at` composes the edit with animation: `Some(playhead)` writes a
    /// keyframe at that timeline position on properties that already have
    /// keyframes (the CapCut gesture semantics); `None` flattens every
    /// property to a constant. Identical behavior on never-animated clips.
    SetClipTransform {
        clip: ClipId,
        transform: ClipTransform,
        at: Option<RationalTime>,
    },
    /// Insert or replace a keyframe on one animatable clip property at an
    /// absolute timeline position (must fall inside the clip). A constant
    /// property becomes a single-keyframe curve. The inverse restores the
    /// previous parameter state.
    SetParamKeyframe {
        clip: ClipId,
        param: ClipParam,
        at: RationalTime,
        value: ParamValue,
        easing: Easing,
    },
    /// Remove the keyframe at exactly `at` on one property. Removing the
    /// last keyframe collapses the property to a constant of that
    /// keyframe's value. Rejected when no keyframe sits at `at`.
    RemoveParamKeyframe {
        clip: ClipId,
        param: ClipParam,
        at: RationalTime,
    },
    /// Replace one animatable property with a constant, dropping all its
    /// keyframes.
    SetParamConstant {
        clip: ClipId,
        param: ClipParam,
        value: ParamValue,
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
    /// Remove a track (and any clips still on it) from the stack.
    RemoveTrack { track: TrackId },
    /// Toggle whether a (visual) track contributes to the composite.
    SetTrackEnabled { track: TrackId, enabled: bool },
    /// Toggle whether an audio track is silenced.
    SetTrackMuted { track: TrackId, muted: bool },
    /// Toggle whether a track's clips are editable (selection/move/trim).
    SetTrackLocked { track: TrackId, locked: bool },
    /// Remove a clip and slide later clips on its track left to close the gap.
    RippleDelete { clip: ClipId },
    /// Shift every clip on `track` whose start is ≥ `from` by `delta` ticks
    /// (signed). The ripple primitive: opens a hole for an insert when
    /// positive, closes a gap when negative; rejected if a left shift would
    /// collide or push below tick 0.
    ShiftClips {
        track: TrackId,
        from: RationalTime,
        delta: RationalTime,
    },
    /// Insert a trimmed range of media at `at`, first shifting every clip
    /// starting at/after `at` right by the new clip's duration (CapCut
    /// main-track insert). Atomic: a rejected placement restores the shift.
    RippleInsert {
        track: TrackId,
        media: MediaId,
        source: TimeRange,
        at: RationalTime,
    },
    /// Put `clips` into one fresh link group (CapCut linkage): linked clips
    /// select, move, and trim together. Any previous links on the clips are
    /// replaced; the inverse restores them.
    LinkClips { clips: Vec<ClipId> },
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
    RemovedTrack(TrackId),
    /// Clips on the track were ripple-shifted (no single clip to point at).
    ShiftedTrack(TrackId),
    /// A track flag (enabled / muted / locked) was changed.
    UpdatedTrack(TrackId),
}
