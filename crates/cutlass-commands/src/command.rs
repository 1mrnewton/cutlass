//! Structured editor commands.
//!
//! UI gestures and the AI agent both emit these values; the engine applies them
//! against project/timeline state with undo/redo.

use std::path::PathBuf;

use cutlass_models::{
    CanvasAspect, ClipId, ClipParam, ClipTransform, CropRect, Easing, Generator, MarkerColor,
    MarkerId, MediaId, Param, ParamValue, Rational, RationalTime, TimeRange, TrackId, TrackKind,
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
    /// Re-point a media-pool entry at a new file (missing-media relink):
    /// re-probe `path` and refresh the entry's metadata in place, keeping
    /// its id so clips stay attached. Not undoable by design — it repairs
    /// project state to match the disk, and undoing back to a dead path is
    /// never what the user wants.
    RelinkMedia { media: MediaId, path: PathBuf },
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
    /// Retime a media clip (CapCut speed, M1): `speed` is the positive
    /// playback-rate multiplier (2/1 = double speed), `reversed` plays the
    /// source window backward. Keeps the clip's timeline start and source
    /// window; the timeline duration re-derives (source ÷ speed). Rejected
    /// on generated clips and when the new extent would overlap a
    /// neighbor. The inverse restores the previous clip state.
    SetClipSpeed {
        clip: ClipId,
        speed: Rational,
        reversed: bool,
    },
    /// Set (or clear) a media clip's playback-rate ramp (CapCut speed curves,
    /// M2): `curve` is the normalized speed-multiplier curve over the clip's
    /// span (ticks `0..=SPEED_CURVE_SCALE`); `None` clears it to a flat unit
    /// ramp. Keeps the clip's base `speed`/`reversed` and source window; the
    /// timeline duration re-derives from `source ÷ (base_speed × average
    /// curve)`. Rejected on generated clips, malformed curves, and when the
    /// new extent would overlap a neighbor. The inverse restores the previous
    /// clip state.
    SetSpeedCurve {
        clip: ClipId,
        curve: Option<Param<f32>>,
    },
    /// Toggle pitch preservation on a retimed media clip (CapCut's "pitch"
    /// switch, M8 Phase 3): `true` time-stretches so the audio keeps its
    /// pitch, `false` lets pitch follow the speed ("chipmunk"). A pure audio
    /// property — changes no duration. Rejected on generated clips. The
    /// inverse restores the previous clip state.
    SetClipPitch { clip: ClipId, preserve_pitch: bool },
    /// Set a clip's framing (CapCut crop, M1): `crop` is the normalized
    /// kept region of the content (applied before placement, so the kept
    /// pixels aspect-fit and transform exactly like the full frame did);
    /// `flip_h`/`flip_v` mirror the content. Rejected on audio-track clips
    /// and on degenerate/out-of-frame rects. The inverse restores the
    /// previous framing.
    SetClipCrop {
        clip: ClipId,
        crop: CropRect,
        flip_h: bool,
        flip_v: bool,
    },
    /// Set a media clip's audio mix (CapCut volume + fades): `volume` is the
    /// flat gain (`0` mutes, `1` = unchanged, up to 10× boost) — `Some`
    /// overwrites the clip's gain with that constant (the basic slider,
    /// flattening any M8 envelope), `None` leaves the gain (constant or
    /// envelope) untouched and only updates the fades. Fades are linear
    /// in/out durations at the timeline rate. Audible for clips on audio
    /// lanes; rejected on generated clips and on fades longer than the clip.
    /// The inverse restores the previous clip state.
    SetClipAudio {
        clip: ClipId,
        volume: Option<f32>,
        fade_in: RationalTime,
        fade_out: RationalTime,
    },
    /// Append a GPU effect (M4) to a visual clip's chain. `effect_id` must
    /// exist in the effect catalog. The inverse removes it (clip snapshot).
    AddEffect { clip: ClipId, effect_id: String },
    /// Remove the effect at `index` from a clip's chain. The inverse restores
    /// it (clip snapshot).
    RemoveEffect { clip: ClipId, index: usize },
    /// Set one effect parameter to a constant (the non-animated quick edit;
    /// animated edits go through `SetParamKeyframe` with `ClipParam::Effect`).
    /// `param` is the catalog slot index. The inverse restores the previous
    /// clip state.
    SetEffectParam {
        clip: ClipId,
        index: usize,
        param: usize,
        value: f32,
    },
    /// Add a transition (M4) at the junction where `clip` abuts the next clip
    /// on its track. `transition_id` must exist in the transition catalog. The
    /// inverse removes it (track-transitions snapshot).
    AddTransition { clip: ClipId, transition_id: String },
    /// Remove the transition at `clip`'s right junction. The inverse restores
    /// it (track-transitions snapshot).
    RemoveTransition { clip: ClipId },
    /// Set the window length (timeline ticks) of the transition at `clip`'s
    /// right junction. The inverse restores the previous length.
    SetTransition { clip: ClipId, duration: i64 },
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
    /// Tag (or untag) an audio track as a sidechain "voice" source for audio
    /// ducking (M8 Phase 4): a "duck under voice" gesture dips music under the
    /// clips on every lane flagged here. The inverse restores the flag.
    SetTrackDuckSource { track: TrackId, duck_source: bool },
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
    /// Sidechain-duck the `music` clips under the `voice` clips (CapCut audio
    /// ducking, M8 Phase 4): analyze speech-band energy on the voice clips and
    /// write volume keyframes that dip each music clip while the voice is
    /// present. `amount` is the fractional duck depth (`0` none … `1` to
    /// silence), `threshold` the linear speech-band RMS gate, `attack`/`release`
    /// the dip/recover times in seconds. Reuses the M8 volume envelope, so the
    /// result is ordinary, editable keyframes; the inverse restores every
    /// touched clip's prior volume in one undo.
    DuckLanes {
        voice: Vec<ClipId>,
        music: Vec<ClipId>,
        threshold: f32,
        amount: f32,
        attack: f32,
        release: f32,
    },
    /// Drop a named, colored marker on the timeline ruler at `at` (M1
    /// markers). `color: None` cycles the fixed palette. The inverse
    /// removes it.
    AddMarker {
        at: RationalTime,
        name: String,
        color: Option<MarkerColor>,
    },
    /// Remove a ruler marker. The inverse restores it (same id).
    RemoveMarker { marker: MarkerId },
    /// Move / rename / recolor a ruler marker in one shot (callers resolve
    /// "keep current" before dispatch). The inverse restores the previous
    /// marker state.
    SetMarker {
        marker: MarkerId,
        at: RationalTime,
        name: String,
        color: MarkerColor,
    },
    /// Set the project canvas (M1 canvas settings): aspect-ratio preset +
    /// opaque background color, in one shot (callers resolve "keep current"
    /// before dispatch). The inverse restores the previous settings.
    SetCanvas {
        aspect: CanvasAspect,
        background: [u8; 3],
    },
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
    /// A ruler marker was added / changed / removed (M1 markers).
    CreatedMarker(MarkerId),
    UpdatedMarker(MarkerId),
    RemovedMarker(MarkerId),
    /// The project canvas settings changed (M1 canvas settings).
    UpdatedCanvas,
}
