//! Background preview rendering: engine and decode/composite stay off the UI thread.
//!
//! PORT IN PROGRESS (from main's crates/cutlass-ui): this is the slim Phase 1
//! worker — engine ownership, the full edit/project message set, debounced
//! autosave, and the fit-sized preview pump. Deferred to later phases and
//! noted inline: audio snapshots (Phase 3), thumbnail/strip registration
//! (Phase 4), export (Phase 5), live gesture overrides + generator content
//! sizes (Phase 6), and the AI agent bridge (separate port).

use std::cell::Cell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, bounded, unbounded};
use cutlass_commands::{Command, EditCommand, EditOutcome, ProjectCommand};
use cutlass_engine::{ApplyOutcome, Engine, EngineConfig};
use cutlass_models::{
    AnimatedTransform, ClipId, ClipParam, ClipSource, ClipTransform, CropRect, Easing, Generator,
    LinkId, MAX_SPEED, MIN_SPEED, MarkerColor, MarkerId, MediaId, Param, ParamValue, Rational,
    RationalTime, TimeRange, Track, TrackId, TrackKind, resample,
};
use tracing::{debug, error, info, warn};

use crate::{EditorStore, PreviewStore};

/// Everything a mutation publishes to: the Slint view model. The audio
/// mixer's snapshot rejoins in Phase 3 so sound can never diverge from the
/// projected timeline.
struct UiSink {
    editor: slint::Weak<EditorStore<'static>>,
}

pub struct PreviewSession {
    pub duration_ticks: i64,
    pub tl_rate: Rational,
}

/// Work submitted to the engine thread. Scrub frames coalesce to the latest
/// pending tick; imports must not be dropped by that coalescing (see
/// [`worker_loop`]).
// PORT (Phase 2): the whole edit message set is live worker-side, but main.rs
// only wires sessions/imports/drops/frames so far — the rest of the EditorStore
// callbacks bind in Phase 2 and this allow goes away.
#[allow(dead_code)]
enum WorkerMsg {
    Frame(i64),
    /// The preview panel's on-screen size in physical pixels — the fit bound
    /// for every subsequent frame render (scrubbing pays for view-sized
    /// pixels, not the full canvas). Re-renders the current frame on change.
    Viewport {
        width: u32,
        height: u32,
    },
    Import(PathBuf),
    /// Place the full range of `media` (raw id from the Slint projection) at
    /// `start_tick` sequence ticks. `track` is the targeted video lane's raw
    /// id, or empty to create a new video lane at `drop_row` (the lane-list
    /// row under the cursor, top-first; may be out of range). `insert`
    /// (main-track magnet) ripple-inserts at `start_tick`, shifting later
    /// clips right instead of first-fit sliding.
    AddClip {
        media: String,
        track: String,
        start_tick: i64,
        drop_row: i64,
        insert: bool,
    },
    /// Place a generated clip (text title, solid, shape) at `start_tick` on
    /// `track` (raw id of a matching-kind lane), or create a lane of the
    /// generator's kind at `drop_row` when `track` is empty. Generated lanes
    /// are never the main track, so there's no ripple-insert path.
    AddGenerated {
        generator: Generator,
        track: String,
        start_tick: i64,
        duration_ticks: i64,
        drop_row: i64,
    },
    /// Move `clip` (raw id) to `track` at `start_tick`, or — when `track` is
    /// empty — to a new lane of the clip's kind inserted at `insert_row`.
    /// `insert` (main-track magnet) ripple-inserts on the main lane; for
    /// reorders `start_tick` is in post-close space (the resolver already
    /// subtracted the clip's own span).
    MoveClip {
        clip: String,
        track: String,
        insert_row: i64,
        start_tick: i64,
        insert: bool,
    },
    /// Move a multi-selection in one history entry. Each entry is fully
    /// resolved (existing target lane + start) by the group drag resolver;
    /// the batch lands via park-then-place so members can never transiently
    /// collide with each other regardless of order.
    MoveGroup {
        moves: Vec<GroupMove>,
    },
    /// Re-place `clip` (raw id) at `[start_tick, start_tick + duration_ticks)`
    /// on its own lane (edge trim; the engine re-derives the source in/out).
    TrimClip {
        clip: String,
        start_tick: i64,
        duration_ticks: i64,
    },
    /// Remove every clip in `clips` (raw ids) as one history entry; lanes
    /// the removals empty are removed too (same policy as drag-moves).
    RemoveClips {
        clips: Vec<String>,
    },
    /// Replace a generated clip's content (raw id) — e.g. an inspector title
    /// edit. One undoable history entry per committed edit.
    SetGenerator {
        clip: String,
        generator: Generator,
    },
    /// Resize a shape clip's reference-pixel dimensions. Preserves shape kind
    /// and fill from the committed generator.
    ///
    /// PORT (Phase 6): main also had a `PreviewShapeSize` live-drag variant
    /// riding the engine's generator override; it returns with the override
    /// APIs. Until then the slider commits on release only.
    SetShapeSize {
        clip: String,
        width: f32,
        height: f32,
    },
    /// Retime a media clip (CapCut speed, M1): positive rational `num/den`
    /// playback rate plus the reverse flag. The engine re-derives the
    /// timeline duration; one undoable history entry.
    SetClipSpeed {
        clip: String,
        num: i32,
        den: i32,
        reversed: bool,
    },
    /// Toggle pitch preservation on a retimed media clip (CapCut "pitch"
    /// switch, M8 Phase 3): `true` keeps the original pitch (time-stretch),
    /// `false` lets pitch ride the speed. With linkage on the clip's
    /// audio-lane link partners follow; one undoable history entry.
    SetClipPitch {
        clip: String,
        preserve: bool,
    },
    /// Toggle noise reduction on a media clip (CapCut "Reduce noise", M8
    /// Phase 5): `true` runs the clip's audio through RNNoise in both mixers.
    /// Routed to the clip's audio-lane link partners when a video half is
    /// targeted; one undoable history entry.
    SetDenoise {
        clip: String,
        denoise: bool,
    },
    /// Set (or clear) a media clip's speed ramp (CapCut speed curves, M2):
    /// `curve` is the normalized rate curve, `None` clears it. The engine
    /// re-derives the timeline duration from the ramp's average; one undoable
    /// history entry (the whole link group when linkage is on).
    SetSpeedCurve {
        clip: String,
        curve: Option<Param<f32>>,
    },
    /// Adjust one existing ramp point's multiplier (velocity-graph drag): the
    /// worker reads the clip's current curve, replaces point `index`, and
    /// re-commits as a `SetSpeedCurve`. One undoable history entry.
    SetSpeedCurvePoint {
        clip: String,
        index: usize,
        value: f32,
    },
    /// Set a clip's audio mix (CapCut volume + fades): `volume` is `Some` for
    /// the basic flat-level slider (flattening any M8 envelope) or `None` to
    /// keep the gain and change only the fades. Fade durations are seconds
    /// (converted to ticks at the timeline rate worker-side). Routed to the
    /// clip's audio-lane link partners when a video half is targeted; one
    /// undoable history entry.
    SetClipAudio {
        clip: String,
        volume: Option<f32>,
        fade_in_s: f32,
        fade_out_s: f32,
    },
    /// Duck a music clip under the voice lanes (M8 Phase 4): gather every clip
    /// on a voice-tagged (`duck_source`) audio lane overlapping `clip` and dip
    /// its volume under them, written as ordinary M8 volume keyframes. One
    /// undoable history entry.
    DuckUnderVoice {
        clip: String,
    },
    /// Detect beat markers on a media clip (CapCut "Beat", M8 Phase 6): the
    /// worker decodes the clip's audio, runs onset/tempo analysis, and stores
    /// the beat grid on the clip so the timeline magnet can snap to it. One
    /// undoable history entry.
    DetectBeats {
        clip: String,
    },
    /// Clear a clip's detected beat markers (M8 Phase 6). One undoable entry.
    ClearBeats {
        clip: String,
    },
    /// Set a visual clip's crop window + mirroring (CapCut crop, M1): the
    /// normalized kept-region rect plus flip flags. One undoable history
    /// entry; the engine rejects audio-lane clips and degenerate rects.
    SetClipCrop {
        clip: String,
        crop: CropRect,
        flip_h: bool,
        flip_v: bool,
    },
    /// Append a catalog effect to a clip's chain (M4). One undoable entry.
    AddEffect {
        clip: String,
        effect_id: String,
    },
    /// Remove the effect at `index` from a clip's chain (M4).
    RemoveEffect {
        clip: String,
        index: u32,
    },
    /// Set one effect parameter (by catalog name) to a constant (M4).
    SetEffectParam {
        clip: String,
        index: u32,
        param: String,
        value: f32,
    },
    /// Add a catalog transition at the junction after `clip` (M4).
    AddTransition {
        clip: String,
        transition_id: String,
    },
    /// Remove the transition at the junction after `clip` (M4).
    RemoveTransition {
        clip: String,
    },
    /// Set the window length (timeline ticks) of the transition after `clip`.
    SetTransition {
        clip: String,
        duration: i64,
    },
    /// Set the project canvas (M1 canvas settings): preset index in
    /// `CanvasAspect::ALL` order plus the opaque background color. One
    /// undoable history entry.
    SetCanvas {
        aspect_index: i32,
        background: [u8; 3],
    },
    /// Fit/fill clip helper (M1 canvas settings): re-place the clip centered
    /// at aspect-fit scale (`fill: false`) or the cover scale that fills the
    /// canvas (`fill: true`). Rides `SetClipTransform`, so it composes with
    /// keyframes at `tick` like any transform gesture and undoes in one step.
    FitClip {
        clip: String,
        fill: bool,
        tick: i64,
    },
    /// Commit a transform gesture as one undoable `SetClipTransform`, then
    /// re-render `tick` so the committed placement is visible immediately.
    ///
    /// PORT (Phase 6): main previewed the drag live through engine-side
    /// `TransformOverride` / `GeneratorOverride` session state; those
    /// messages return with the override APIs. Until then gestures are
    /// commit-on-release (the mobile pattern).
    SetTransform {
        clip: String,
        transform: ClipTransform,
        tick: i64,
    },
    /// Insert or replace a keyframe on one animatable property of `clip`
    /// (raw id) at the absolute sequence tick (the playhead — must fall
    /// inside the clip; the engine validates). One undoable edit; the
    /// projection republish carries the updated curve back to the UI.
    SetParamKeyframe {
        clip: String,
        param: ClipParam,
        tick: i64,
        value: ParamValue,
        easing: Easing,
    },
    /// Remove the keyframe sitting exactly at `tick` on one property of
    /// `clip`. Removing the last keyframe collapses the property to a
    /// constant of that keyframe's value (engine semantics). Undoable.
    RemoveParamKeyframe {
        clip: String,
        param: ClipParam,
        tick: i64,
    },
    /// Move every keyframe sitting at `from_tick` (across all animated
    /// properties of `clip`) to `to_tick` — the timeline diamond drag
    /// (keyframes roadmap Phase 2). One history group: a single undo puts
    /// the merged diamond back.
    RetimeKeyframes {
        clip: String,
        from_tick: i64,
        to_tick: i64,
    },
    /// Remove every keyframe sitting at `tick` across all animated
    /// properties of `clip` (timeline diamond right-click). One history
    /// group.
    RemoveKeyframesAt {
        clip: String,
        tick: i64,
    },
    /// Split `clip` (raw id) at `at_tick` (sequence ticks). The UI gates on
    /// the playhead being strictly inside the clip; the engine re-validates.
    SplitClip {
        clip: String,
        at_tick: i64,
    },
    /// Drop a ruler marker at `at_tick`. `color` is a palette name
    /// ("teal", "blue", …) or empty to cycle. One undoable history entry.
    AddMarker {
        at_tick: i64,
        name: String,
        color: String,
    },
    /// Remove a ruler marker by raw id. One undoable history entry.
    RemoveMarker {
        marker: String,
    },
    /// Step the engine history one entry back / forward.
    Undo,
    Redo,
    /// Snapshot `clips` (raw ids — the whole selection) into the worker
    /// clipboard as one block. A snapshot, not a reference — pasting works
    /// after the originals are deleted.
    CopyClips {
        clips: Vec<String>,
    },
    /// Place the clipboard block at `tick`: members keep their lanes and
    /// relative placement, the whole block slides right as one unit until
    /// every member fits.
    PasteAt {
        tick: i64,
    },
    /// Place copies of `clips` (the whole selection) right after the block
    /// they form, keeping lanes and relative placement.
    DuplicateClips {
        clips: Vec<String>,
    },
    /// Dissolve the link group of every clip in `clips` (raw ids): all
    /// members of the touched groups — selected or not — end up unlinked.
    UnlinkClips {
        clips: Vec<String>,
    },
    /// Mirror of the UI's main-track magnet toggle. The worker needs it for
    /// ops without a drag resolution (delete/paste/duplicate); enabling also
    /// packs the main lane gapless (one history entry).
    SetMainMagnet(bool),
    /// Mirror of the UI's linkage toggle: drops of media with audio create
    /// linked pairs, trims/splits follow link groups.
    SetLinkage(bool),
    /// Set a track header flag (hide/mute/lock) on `track` (raw id). Undoable.
    SetTrackFlag {
        track: String,
        flag: TrackFlag,
        value: bool,
    },
    // PORT (Phase 5): main's `Export(ExportRequest)` / `CancelExport` job
    // messages return with the export thread (export_to_file_observed).
    /// Flush the live draft to its project file. `None` reuses the engine's
    /// current draft path (the normal case: explicit Cmd+S, or the debounce /
    /// session-swap / close flush); `Some` rebinds it (binding a freshly
    /// created draft after `NewProject`). Not undoable; on success the
    /// projection republish refreshes the title and the meta sidecar is
    /// rewritten so the gallery name tracks the project name.
    SaveProject {
        path: Option<PathBuf>,
    },
    /// Replace the session from a `.cutlass` file (tolerant: entries whose
    /// media file is gone are kept and surface through the relink flow —
    /// the projection republish carries the missing set, and app.slint
    /// raises the relink dialog on the epoch bump). Success re-registers
    /// still-present pool media with the thumbnail and strip workers,
    /// republishes everything, and bumps the session epoch so the UI
    /// resets its session state (playhead, selection, range). Failure
    /// publishes `session-error`. The unsaved-changes guard ran UI-side
    /// before this message was sent.
    OpenProject {
        path: PathBuf,
    },
    /// Re-point a media-pool entry (raw id) at a new file (missing-media
    /// relink, M0): the engine re-probes the file and swaps the entry's
    /// path/metadata in place (id and clips untouched), the tile workers
    /// re-register, and the projection republish drops the entry from the
    /// missing set. Not undoable — state repair, not an edit.
    RelinkMedia {
        media: String,
        path: PathBuf,
    },
    /// Try `folder/<filename>` for every missing pool entry (locate-folder
    /// gesture in the relink dialog).
    RelinkFolder {
        folder: PathBuf,
    },
    /// Delete a source (raw id) from the media pool / Library bin. `force`
    /// false removes only when no clip references it (the engine rejects a
    /// referenced source); `force` true first deletes every referencing clip
    /// and then the source, all in one undo. One history entry either way.
    RemoveMedia {
        media: String,
        force: bool,
    },
    /// Replace the session with a fresh, empty project (the draft's
    /// `project.cutlass` is written by the `SaveProject` that follows).
    /// Same epoch bump as `OpenProject`.
    NewProject,
    /// Rename the current draft (its display name). Applied as one undoable
    /// edit; the projection republish updates the title bar and the next
    /// auto-save writes the name into the draft's project file and meta.
    RenameProject {
        name: String,
    },
    // PORT (later): main's `SnapshotProject` / `AgentApplyPlan` round-trips
    // return with the cutlass-ai port (the agent rehearses on a sandbox
    // clone, then replays the plan here as one history group).
}

/// One clip's resolved landing inside a [`WorkerMsg::MoveGroup`] batch.
/// All raw ids from the Slint projection.
pub struct GroupMove {
    pub clip: String,
    pub track: String,
    pub start_tick: i64,
}

/// Which track header toggle a [`WorkerMsg::SetTrackFlag`] addresses.
// PORT (Phase 2): constructed once the track-header callbacks wire up.
#[allow(dead_code)]
#[derive(Clone, Copy)]
pub enum TrackFlag {
    /// Video: contributes to the composite (the eye toggle).
    Enabled,
    /// Audio: silenced (the speaker toggle).
    Muted,
    /// Clips can't be selected / moved / trimmed (the lock toggle).
    Locked,
    /// Audio: tagged as a sidechain "voice" source for ducking (M8 Phase 4).
    DuckSource,
}

/// Worker-side clipboard: one member of the copied block, everything needed
/// to re-issue it as a fresh `AddClip` / `AddGenerated` later, independent
/// of the original. A copy snapshots the whole selection as a `Vec` of these
/// (single-clip copy ⇒ a block of one).
struct ClipboardClip {
    /// Lane the clip was copied from (preferred paste target).
    track: TrackId,
    /// Lane kind, for recreating a lane when `track` is gone by paste time.
    kind: TrackKind,
    content: ClipSource,
    /// Timeline-rate duration, for first-fit placement.
    duration_ticks: i64,
    /// Start offset from the block's earliest member — paste keeps the
    /// members' relative placement.
    offset_ticks: i64,
    /// The original's link group, as a grouping key only: members copied
    /// from the same group are re-linked as a fresh group on paste.
    link: Option<LinkId>,
}

/// Cheap, cloneable sender to the engine thread. Hand one clone to each UI
/// callback that needs to talk to the engine (scrub, import, …). Cloning keeps
/// the channel — and therefore the worker loop — alive.
#[derive(Clone)]
pub struct WorkerHandle {
    tx: Sender<WorkerMsg>,
}

// PORT (Phase 2): most senders bind to EditorStore callbacks in Phase 2;
// until then only the session/import/drop/frame subset is called from main.
#[allow(dead_code)]
impl WorkerHandle {
    pub fn request_frame(&self, tick: i64) {
        let _ = self.tx.send(WorkerMsg::Frame(tick));
    }

    /// Report the preview panel's on-screen size (physical px). The worker
    /// fits every subsequent render inside it and repaints the current frame.
    pub fn set_viewport(&self, width: u32, height: u32) {
        let _ = self.tx.send(WorkerMsg::Viewport { width, height });
    }

    pub fn import(&self, path: PathBuf) {
        let _ = self.tx.send(WorkerMsg::Import(path));
    }

    pub fn save_project(&self, path: Option<PathBuf>) {
        let _ = self.tx.send(WorkerMsg::SaveProject { path });
    }

    pub fn open_project(&self, path: PathBuf) {
        let _ = self.tx.send(WorkerMsg::OpenProject { path });
    }

    pub fn new_project(&self) {
        let _ = self.tx.send(WorkerMsg::NewProject);
    }

    pub fn rename_project(&self, name: String) {
        let _ = self.tx.send(WorkerMsg::RenameProject { name });
    }

    pub fn relink_media(&self, media: String, path: PathBuf) {
        let _ = self.tx.send(WorkerMsg::RelinkMedia { media, path });
    }

    pub fn relink_folder(&self, folder: PathBuf) {
        let _ = self.tx.send(WorkerMsg::RelinkFolder { folder });
    }

    pub fn remove_media(&self, media: String, force: bool) {
        let _ = self.tx.send(WorkerMsg::RemoveMedia { media, force });
    }

    pub fn add_clip(
        &self,
        media: String,
        track: String,
        start_tick: i64,
        drop_row: i64,
        insert: bool,
    ) {
        let _ = self.tx.send(WorkerMsg::AddClip {
            media,
            track,
            start_tick,
            drop_row,
            insert,
        });
    }

    pub fn add_generated(
        &self,
        generator: Generator,
        track: String,
        start_tick: i64,
        duration_ticks: i64,
        drop_row: i64,
    ) {
        let _ = self.tx.send(WorkerMsg::AddGenerated {
            generator,
            track,
            start_tick,
            duration_ticks,
            drop_row,
        });
    }

    pub fn move_clip(
        &self,
        clip: String,
        track: String,
        insert_row: i64,
        start_tick: i64,
        insert: bool,
    ) {
        let _ = self.tx.send(WorkerMsg::MoveClip {
            clip,
            track,
            insert_row,
            start_tick,
            insert,
        });
    }

    pub fn move_group(&self, moves: Vec<GroupMove>) {
        let _ = self.tx.send(WorkerMsg::MoveGroup { moves });
    }

    pub fn trim_clip(&self, clip: String, start_tick: i64, duration_ticks: i64) {
        let _ = self.tx.send(WorkerMsg::TrimClip {
            clip,
            start_tick,
            duration_ticks,
        });
    }

    pub fn remove_clips(&self, clips: Vec<String>) {
        let _ = self.tx.send(WorkerMsg::RemoveClips { clips });
    }

    pub fn split_clip(&self, clip: String, at_tick: i64) {
        let _ = self.tx.send(WorkerMsg::SplitClip { clip, at_tick });
    }

    pub fn add_marker(&self, at_tick: i64, name: String, color: String) {
        let _ = self.tx.send(WorkerMsg::AddMarker {
            at_tick,
            name,
            color,
        });
    }

    pub fn remove_marker(&self, marker: String) {
        let _ = self.tx.send(WorkerMsg::RemoveMarker { marker });
    }

    pub fn set_generator(&self, clip: String, generator: Generator) {
        let _ = self.tx.send(WorkerMsg::SetGenerator { clip, generator });
    }

    pub fn set_shape_size(&self, clip: String, width: f32, height: f32) {
        let _ = self.tx.send(WorkerMsg::SetShapeSize {
            clip,
            width,
            height,
        });
    }

    pub fn set_clip_speed(&self, clip: String, num: i32, den: i32, reversed: bool) {
        let _ = self.tx.send(WorkerMsg::SetClipSpeed {
            clip,
            num,
            den,
            reversed,
        });
    }

    pub fn set_clip_pitch(&self, clip: String, preserve: bool) {
        let _ = self.tx.send(WorkerMsg::SetClipPitch { clip, preserve });
    }

    pub fn set_denoise(&self, clip: String, denoise: bool) {
        let _ = self.tx.send(WorkerMsg::SetDenoise { clip, denoise });
    }

    /// Resolve a speed-ramp preset name (CapCut speed curves, M2) and dispatch
    /// the edit. `""` / `"none"` / `"normal"` clears the ramp; an unknown name
    /// is dropped with a warning so a stray UI string can't apply garbage.
    pub fn set_speed_curve(&self, clip: String, preset: String) {
        let curve = match preset.trim() {
            "" | "none" | "normal" => None,
            name => match cutlass_models::speed_preset(name) {
                Some(curve) => Some(curve),
                None => {
                    warn!(preset = name, "set-speed-curve ignored: unknown preset");
                    return;
                }
            },
        };
        let _ = self.tx.send(WorkerMsg::SetSpeedCurve { clip, curve });
    }

    pub fn set_speed_curve_point(&self, clip: String, index: i32, value: f32) {
        let Ok(index) = usize::try_from(index) else {
            warn!(index, "set-speed-curve-point ignored: negative index");
            return;
        };
        let _ = self
            .tx
            .send(WorkerMsg::SetSpeedCurvePoint { clip, index, value });
    }

    /// Set the flat volume level + fades (CapCut's basic slider): `volume` is
    /// `Some`, flattening any envelope.
    pub fn set_clip_audio(&self, clip: String, volume: f32, fade_in_s: f32, fade_out_s: f32) {
        let _ = self.tx.send(WorkerMsg::SetClipAudio {
            clip,
            volume: Some(volume),
            fade_in_s,
            fade_out_s,
        });
    }

    /// Duck `clip` (a music clip) under the voice-tagged lanes (M8 Phase 4).
    pub fn duck_under_voice(&self, clip: String) {
        let _ = self.tx.send(WorkerMsg::DuckUnderVoice { clip });
    }

    /// Detect beat markers on `clip` (CapCut "Beat", M8 Phase 6).
    pub fn detect_beats(&self, clip: String) {
        let _ = self.tx.send(WorkerMsg::DetectBeats { clip });
    }

    /// Clear `clip`'s detected beat markers (M8 Phase 6).
    pub fn clear_beats(&self, clip: String) {
        let _ = self.tx.send(WorkerMsg::ClearBeats { clip });
    }

    /// Set only the fades, preserving the clip's gain (constant or a
    /// keyframed M8 envelope) — `volume` lowers to `None`.
    pub fn set_clip_fades(&self, clip: String, fade_in_s: f32, fade_out_s: f32) {
        let _ = self.tx.send(WorkerMsg::SetClipAudio {
            clip,
            volume: None,
            fade_in_s,
            fade_out_s,
        });
    }

    pub fn set_clip_crop(&self, clip: String, crop: CropRect, flip_h: bool, flip_v: bool) {
        let _ = self.tx.send(WorkerMsg::SetClipCrop {
            clip,
            crop,
            flip_h,
            flip_v,
        });
    }

    pub fn add_effect(&self, clip: String, effect_id: String) {
        let _ = self.tx.send(WorkerMsg::AddEffect { clip, effect_id });
    }

    pub fn remove_effect(&self, clip: String, index: u32) {
        let _ = self.tx.send(WorkerMsg::RemoveEffect { clip, index });
    }

    pub fn set_effect_param(&self, clip: String, index: u32, param: String, value: f32) {
        let _ = self.tx.send(WorkerMsg::SetEffectParam {
            clip,
            index,
            param,
            value,
        });
    }

    pub fn add_transition(&self, clip: String, transition_id: String) {
        let _ = self.tx.send(WorkerMsg::AddTransition {
            clip,
            transition_id,
        });
    }

    pub fn remove_transition(&self, clip: String) {
        let _ = self.tx.send(WorkerMsg::RemoveTransition { clip });
    }

    pub fn set_transition(&self, clip: String, duration: i64) {
        let _ = self.tx.send(WorkerMsg::SetTransition { clip, duration });
    }

    pub fn set_canvas(&self, aspect_index: i32, background: [u8; 3]) {
        let _ = self.tx.send(WorkerMsg::SetCanvas {
            aspect_index,
            background,
        });
    }

    pub fn fit_clip(&self, clip: String, fill: bool, tick: i64) {
        let _ = self.tx.send(WorkerMsg::FitClip { clip, fill, tick });
    }

    pub fn set_param_keyframe(
        &self,
        clip: String,
        param: ClipParam,
        tick: i64,
        value: ParamValue,
        easing: Easing,
    ) {
        let _ = self.tx.send(WorkerMsg::SetParamKeyframe {
            clip,
            param,
            tick,
            value,
            easing,
        });
    }

    pub fn remove_param_keyframe(&self, clip: String, param: ClipParam, tick: i64) {
        let _ = self
            .tx
            .send(WorkerMsg::RemoveParamKeyframe { clip, param, tick });
    }

    pub fn retime_keyframes(&self, clip: String, from_tick: i64, to_tick: i64) {
        let _ = self.tx.send(WorkerMsg::RetimeKeyframes {
            clip,
            from_tick,
            to_tick,
        });
    }

    pub fn remove_keyframes_at(&self, clip: String, tick: i64) {
        let _ = self.tx.send(WorkerMsg::RemoveKeyframesAt { clip, tick });
    }

    pub fn set_transform(&self, clip: String, transform: ClipTransform, tick: i64) {
        let _ = self.tx.send(WorkerMsg::SetTransform {
            clip,
            transform,
            tick,
        });
    }

    pub fn undo(&self) {
        let _ = self.tx.send(WorkerMsg::Undo);
    }

    pub fn redo(&self) {
        let _ = self.tx.send(WorkerMsg::Redo);
    }

    pub fn copy_clips(&self, clips: Vec<String>) {
        let _ = self.tx.send(WorkerMsg::CopyClips { clips });
    }

    pub fn paste_at(&self, tick: i64) {
        let _ = self.tx.send(WorkerMsg::PasteAt { tick });
    }

    pub fn duplicate_clips(&self, clips: Vec<String>) {
        let _ = self.tx.send(WorkerMsg::DuplicateClips { clips });
    }

    pub fn unlink_clips(&self, clips: Vec<String>) {
        let _ = self.tx.send(WorkerMsg::UnlinkClips { clips });
    }

    pub fn set_main_magnet(&self, enabled: bool) {
        let _ = self.tx.send(WorkerMsg::SetMainMagnet(enabled));
    }

    pub fn set_linkage(&self, enabled: bool) {
        let _ = self.tx.send(WorkerMsg::SetLinkage(enabled));
    }

    pub fn set_track_flag(&self, track: String, flag: TrackFlag, value: bool) {
        let _ = self.tx.send(WorkerMsg::SetTrackFlag { track, flag, value });
    }
}

pub struct PreviewWorker {
    handle: WorkerHandle,
    _join: JoinHandle<()>,
}

impl PreviewWorker {
    /// Spawns a dedicated thread that owns the [`Engine`] (required: decoders are not `Send`).
    pub fn spawn(
        config: EngineConfig,
        preview_weak: slint::Weak<PreviewStore<'static>>,
        editor_weak: slint::Weak<EditorStore<'static>>,
    ) -> Result<(Self, PreviewSession), String> {
        let (ready_tx, ready_rx) = bounded(1);
        let (req_tx, req_rx) = unbounded();

        let join = std::thread::Builder::new()
            .name("cutlass-preview".into())
            .spawn(move || {
                if let Err(e) = worker_main(config, preview_weak, editor_weak, req_rx, ready_tx) {
                    error!("preview worker exited: {e}");
                }
            })
            .map_err(|e| e.to_string())?;

        let session = ready_rx
            .recv()
            .map_err(|e| e.to_string())?
            .map_err(|e: String| e)?;

        Ok((
            Self {
                handle: WorkerHandle { tx: req_tx },
                _join: join,
            },
            session,
        ))
    }

    /// Clone a sender for a UI callback.
    pub fn handle(&self) -> WorkerHandle {
        self.handle.clone()
    }
}

fn worker_main(
    config: EngineConfig,
    preview_weak: slint::Weak<PreviewStore<'static>>,
    editor_weak: slint::Weak<EditorStore<'static>>,
    req_rx: Receiver<WorkerMsg>,
    ready_tx: Sender<Result<PreviewSession, String>>,
) -> Result<(), String> {
    // Start from an empty project: media arrives via user-driven imports
    // (Library → engine), not a hardcoded bootstrap asset.
    let mut engine = Engine::new(config).map_err(|e| e.to_string())?;
    let timeline = engine.project().timeline();
    let session = PreviewSession {
        duration_ticks: timeline.duration().value,
        tl_rate: timeline.frame_rate,
    };
    // Debug, not info: the worker boots an empty engine so it's ready behind
    // the launch screen, but no project exists yet — the user-facing project
    // lifecycle ("new session" / "opened project") logs at info once they
    // actually create or open one.
    debug!(
        duration_ticks = session.duration_ticks,
        tl_rate = ?session.tl_rate,
        "preview worker ready (empty project)"
    );
    let tl_rate = session.tl_rate;
    ready_tx.send(Ok(session)).map_err(|e| e.to_string())?;

    // Seed the UI with the engine's project so the editor reads from the engine
    // from the first frame (rather than any Slint-side placeholder).
    let ui = UiSink {
        editor: editor_weak,
    };
    publish_projection(&mut engine, &ui);

    worker_loop(&mut engine, tl_rate, preview_weak, ui, req_rx);
    Ok(())
}

/// Single consumer for the engine thread. Scrub frames coalesce to the latest
/// pending tick, but every mutation (import, add-clip, move-clip, …) is
/// executed in order — it must never be discarded by the coalescing drain.
fn worker_loop(
    engine: &mut Engine,
    tl_rate: Rational,
    preview_weak: slint::Weak<PreviewStore<'static>>,
    ui: UiSink,
    req_rx: Receiver<WorkerMsg>,
) {
    // Clipboard lives with the loop: it's edit-session state, not project
    // state — copies survive any number of edits/undos and die with the app.
    // One block per copy (the whole selection); never empty when `Some`.
    let mut clipboard: Option<Vec<ClipboardClip>> = None;
    // Mirror of TimelineStore.main-magnet-enabled (must match its default).
    // Drag gestures carry their resolved insert flag; this drives the ops
    // without a drag resolution (delete/paste/duplicate) and pack-on-enable.
    let mut main_magnet = true;
    // Mirror of TimelineStore.link-enabled (must match its default): drops
    // of media with audio create linked pairs, trims/splits follow links.
    let mut linkage = true;
    // Fit bound + quality ladder for every preview render (the PreviewFeed
    // model). `Cell`s because the `mutate` closure repaints too.
    let fit = FrameFit::default();
    // Debounced auto-save: an edit arms a deadline; once the worker has been
    // idle of further work for `PERSIST_DEBOUNCE` the dirty draft is written
    // to its `project.cutlass`. Scrub/seek `Frame`s deliberately don't push
    // the deadline out (they're reads), so playback over a pending edit still
    // flushes between frames; a session swap / close forces an immediate
    // write through `SaveProject`.
    let mut persist_deadline: Option<Instant> = None;
    // Last tick the preview rendered (the playhead). Scrub/seek `Frame`s keep
    // it current; edits re-render here so the composite reflects a delete,
    // generator change, etc. without waiting for the user to move the playhead.
    let mut last_tick: i64 = 0;

    let mutate = |engine: &mut Engine,
                  clipboard: &mut Option<Vec<ClipboardClip>>,
                  main_magnet: &mut bool,
                  linkage: &mut bool,
                  msg: WorkerMsg| {
        match msg {
            // Reached only when the viewport report drains behind a frame or
            // mutation; the follow-up render picks the new bound up.
            WorkerMsg::Viewport { width, height } => {
                fit.set_viewport(width, height);
            }
            WorkerMsg::Import(path) => import_and_publish(engine, &path, &ui),
            WorkerMsg::AddClip {
                media,
                track,
                start_tick,
                drop_row,
                insert,
            } => add_clip_and_publish(engine, &media, &track, start_tick, drop_row, insert, &ui),
            WorkerMsg::AddGenerated {
                generator,
                track,
                start_tick,
                duration_ticks,
                drop_row,
            } => add_generated_and_publish(
                engine,
                generator,
                &track,
                start_tick,
                duration_ticks,
                drop_row,
                &ui,
            ),
            WorkerMsg::MoveClip {
                clip,
                track,
                insert_row,
                start_tick,
                insert,
            } => move_clip_and_publish(
                engine,
                &clip,
                &track,
                insert_row,
                start_tick,
                insert,
                *main_magnet,
                &ui,
            ),
            WorkerMsg::MoveGroup { moves } => move_group_and_publish(engine, &moves, &ui),
            WorkerMsg::TrimClip {
                clip,
                start_tick,
                duration_ticks,
            } => trim_clip_and_publish(
                engine,
                &clip,
                start_tick,
                duration_ticks,
                *linkage,
                *main_magnet,
                &ui,
            ),
            WorkerMsg::RemoveClips { clips } => {
                remove_clips_and_publish(engine, &clips, *main_magnet, &ui)
            }
            WorkerMsg::SetGenerator { clip, generator } => {
                set_generator_and_publish(engine, &clip, generator, &ui)
            }
            WorkerMsg::SetShapeSize {
                clip,
                width,
                height,
            } => {
                if let Some(generator) = shape_size_from_engine(engine, &clip, width, height) {
                    set_generator_and_publish(engine, &clip, generator, &ui);
                }
            }
            WorkerMsg::SetClipSpeed {
                clip,
                num,
                den,
                reversed,
            } => set_clip_speed_and_publish(engine, &clip, num, den, reversed, *linkage, &ui),
            WorkerMsg::SetClipPitch { clip, preserve } => {
                set_clip_pitch_and_publish(engine, &clip, preserve, *linkage, &ui)
            }
            WorkerMsg::SetDenoise { clip, denoise } => {
                set_denoise_and_publish(engine, &clip, denoise, *linkage, &ui)
            }
            WorkerMsg::SetSpeedCurve { clip, curve } => {
                set_speed_curve_and_publish(engine, &clip, &curve, *linkage, &ui)
            }
            WorkerMsg::SetSpeedCurvePoint { clip, index, value } => {
                set_speed_curve_point_and_publish(engine, &clip, index, value, *linkage, &ui)
            }
            WorkerMsg::SetClipAudio {
                clip,
                volume,
                fade_in_s,
                fade_out_s,
            } => set_clip_audio_and_publish(engine, &clip, volume, fade_in_s, fade_out_s, &ui),
            WorkerMsg::DuckUnderVoice { clip } => duck_under_voice_and_publish(engine, &clip, &ui),
            WorkerMsg::DetectBeats { clip } => detect_beats_and_publish(engine, &clip, &ui),
            WorkerMsg::ClearBeats { clip } => clear_beats_and_publish(engine, &clip, &ui),
            WorkerMsg::SetClipCrop {
                clip,
                crop,
                flip_h,
                flip_v,
            } => set_clip_crop_and_publish(engine, &clip, crop, flip_h, flip_v, &ui),
            WorkerMsg::AddEffect { clip, effect_id } => {
                add_effect_and_publish(engine, &clip, &effect_id, &ui)
            }
            WorkerMsg::RemoveEffect { clip, index } => {
                remove_effect_and_publish(engine, &clip, index, &ui)
            }
            WorkerMsg::SetEffectParam {
                clip,
                index,
                param,
                value,
            } => set_effect_param_and_publish(engine, &clip, index, &param, value, &ui),
            WorkerMsg::AddTransition {
                clip,
                transition_id,
            } => add_transition_and_publish(engine, &clip, &transition_id, &ui),
            WorkerMsg::RemoveTransition { clip } => {
                remove_transition_and_publish(engine, &clip, &ui)
            }
            WorkerMsg::SetTransition { clip, duration } => {
                set_transition_and_publish(engine, &clip, duration, &ui)
            }
            WorkerMsg::SetCanvas {
                aspect_index,
                background,
            } => set_canvas_and_publish(engine, aspect_index, background, &ui),
            WorkerMsg::SetTransform {
                clip,
                transform,
                tick,
            } => {
                // The gesture happened at the visible frame: pass the playhead
                // so animated properties get a keyframe there instead of being
                // flattened (M2 compose semantics).
                let at = RationalTime::new(tick, tl_rate);
                set_transform_and_publish(engine, &clip, transform, at, &ui);
                render_frame(engine, tl_rate, &preview_weak, tick, &fit);
            }
            WorkerMsg::FitClip { clip, fill, tick } => {
                fit_clip_and_publish(engine, &clip, fill, tick, tl_rate, &ui);
                render_frame(engine, tl_rate, &preview_weak, tick, &fit);
            }
            WorkerMsg::SetParamKeyframe {
                clip,
                param,
                tick,
                value,
                easing,
            } => set_param_keyframe_and_publish(
                engine,
                &clip,
                param,
                RationalTime::new(tick, tl_rate),
                value,
                easing,
                &ui,
            ),
            WorkerMsg::RemoveParamKeyframe { clip, param, tick } => {
                remove_param_keyframe_and_publish(
                    engine,
                    &clip,
                    param,
                    RationalTime::new(tick, tl_rate),
                    &ui,
                )
            }
            WorkerMsg::RetimeKeyframes {
                clip,
                from_tick,
                to_tick,
            } => retime_keyframes_and_publish(engine, &clip, from_tick, to_tick, tl_rate, &ui),
            WorkerMsg::RemoveKeyframesAt { clip, tick } => {
                remove_keyframes_at_and_publish(engine, &clip, tick, tl_rate, &ui)
            }
            WorkerMsg::SplitClip { clip, at_tick } => {
                split_clip_and_publish(engine, &clip, at_tick, *linkage, &ui)
            }
            WorkerMsg::AddMarker {
                at_tick,
                name,
                color,
            } => add_marker_and_publish(engine, at_tick, &name, &color, tl_rate, &ui),
            WorkerMsg::RemoveMarker { marker } => remove_marker_and_publish(engine, &marker, &ui),
            WorkerMsg::Undo => history_step_and_publish(engine, false, &ui),
            WorkerMsg::Redo => history_step_and_publish(engine, true, &ui),
            WorkerMsg::CopyClips { clips } => {
                // The block origin only matters to duplicate; paste re-bases
                // on the playhead tick.
                if let Some((_, block)) = snapshot_block(engine, &clips) {
                    info!(count = block.len(), "copied clips to clipboard");
                    *clipboard = Some(block);
                }
            }
            WorkerMsg::PasteAt { tick } => match clipboard {
                Some(block) => paste_and_publish(engine, block, tick, *main_magnet, &ui),
                None => info!("paste ignored: clipboard empty"),
            },
            WorkerMsg::DuplicateClips { clips } => {
                duplicate_clips_and_publish(engine, &clips, *main_magnet, &ui)
            }
            WorkerMsg::UnlinkClips { clips } => unlink_clips_and_publish(engine, &clips, &ui),
            WorkerMsg::SetMainMagnet(enabled) => {
                *main_magnet = enabled;
                info!(enabled, "main-track magnet toggled");
                if enabled {
                    pack_main_track_and_publish(engine, &ui);
                }
            }
            WorkerMsg::SetLinkage(enabled) => {
                *linkage = enabled;
                info!(enabled, "linkage toggled");
            }
            WorkerMsg::SetTrackFlag { track, flag, value } => {
                set_track_flag_and_publish(engine, &track, flag, value, &ui)
            }
            WorkerMsg::SaveProject { path } => save_project_and_publish(engine, path, &ui),
            WorkerMsg::OpenProject { path } => open_project_and_publish(engine, path, &ui),
            WorkerMsg::RelinkMedia { media, path } => {
                relink_media_and_publish(engine, &media, &path, &ui)
            }
            WorkerMsg::RelinkFolder { folder } => relink_folder_and_publish(engine, folder, &ui),
            WorkerMsg::RemoveMedia { media, force } => {
                remove_media_and_publish(engine, &media, force, &ui)
            }
            WorkerMsg::NewProject => new_project_and_publish(engine, &ui),
            WorkerMsg::RenameProject { name } => rename_project_and_publish(engine, name, &ui),
            WorkerMsg::Frame(_) => unreachable!("frames are handled by the drain below"),
        }
    };

    loop {
        let msg = match next_message(&req_rx, persist_deadline) {
            Wake::Stop => break,
            // The debounce elapsed: write the dirty draft to its project file.
            Wake::Persist => {
                save_project_and_publish(engine, None, &ui);
                persist_deadline = None;
                continue;
            }
            Wake::Message(msg) => msg,
        };
        // Scrub/seek frames are reads — they must not push the auto-save
        // deadline out, or playback over a pending edit would never flush.
        let resets_deadline = !matches!(msg, WorkerMsg::Frame(_));
        match msg {
            WorkerMsg::Frame(mut tick) => {
                while let Ok(next) = req_rx.try_recv() {
                    match next {
                        WorkerMsg::Frame(latest) => tick = latest,
                        other => mutate(
                            engine,
                            &mut clipboard,
                            &mut main_magnet,
                            &mut linkage,
                            other,
                        ),
                    }
                }
                last_tick = tick;
                render_frame(engine, tl_rate, &preview_weak, tick, &fit);
            }
            // The preview panel resized (or first laid out): renders now fit
            // the new bound. Repaint the current frame only when the bucketed
            // size actually changed — live window resizes report every frame.
            WorkerMsg::Viewport { width, height } => {
                if fit.set_viewport(width, height) {
                    render_frame(engine, tl_rate, &preview_weak, last_tick, &fit);
                }
            }
            other => {
                let redraw = mutation_redraws_preview(&other);
                mutate(
                    engine,
                    &mut clipboard,
                    &mut main_magnet,
                    &mut linkage,
                    other,
                );
                // Edits otherwise only repaint when the playhead moves; refresh
                // the current frame so the change is visible immediately.
                if redraw {
                    render_frame(engine, tl_rate, &preview_weak, last_tick, &fit);
                }
            }
        }
        // After any non-scrub work, (re)arm the debounce while the draft has
        // unsaved edits, or clear it once a save/swap left the draft clean.
        if resets_deadline {
            persist_deadline = engine.is_dirty().then(|| Instant::now() + PERSIST_DEBOUNCE);
        }
    }
}

/// Idle gap after the last edit before the dirty draft is written. Long
/// enough to coalesce a burst of rapid edits (typing a title, dragging a
/// slider) into one write, short enough that work is never far from disk.
const PERSIST_DEBOUNCE: Duration = Duration::from_millis(300);

/// What the worker should do next (see [`next_message`]).
enum Wake {
    /// A request arrived; process it.
    Message(WorkerMsg),
    /// The debounce elapsed with the draft dirty; write it.
    Persist,
    /// The request channel closed; the loop should exit.
    Stop,
}

/// Block for the next request, waking to auto-save when `deadline` passes.
/// With no edit pending (`deadline` is `None`) it's a plain blocking receive.
fn next_message(req_rx: &Receiver<WorkerMsg>, deadline: Option<Instant>) -> Wake {
    match deadline {
        None => match req_rx.recv() {
            Ok(msg) => Wake::Message(msg),
            Err(_) => Wake::Stop,
        },
        Some(deadline) => {
            let timeout = deadline.saturating_duration_since(Instant::now());
            match req_rx.recv_timeout(timeout) {
                Ok(msg) => Wake::Message(msg),
                Err(RecvTimeoutError::Timeout) => Wake::Persist,
                Err(RecvTimeoutError::Disconnected) => Wake::Stop,
            }
        }
    }
}

/// Whether an executed mutation changes the visible composite at the current
/// playhead and should therefore trigger a preview re-render. The only frame
/// trigger used to be playhead movement, so edits (delete, generator/font
/// change, …) looked stale until the user scrubbed. `SetTransform` and
/// `ClearTransformOverride` render themselves with their own tick, so they're
/// excluded here to avoid a redundant second composite; pure session ops
/// (import, copy, auto-save, export, linkage, rename) don't alter the canvas.
fn mutation_redraws_preview(msg: &WorkerMsg) -> bool {
    matches!(
        msg,
        WorkerMsg::AddClip { .. }
            | WorkerMsg::AddGenerated { .. }
            | WorkerMsg::MoveClip { .. }
            | WorkerMsg::MoveGroup { .. }
            | WorkerMsg::TrimClip { .. }
            | WorkerMsg::RemoveClips { .. }
            | WorkerMsg::SetGenerator { .. }
            | WorkerMsg::SetClipSpeed { .. }
            | WorkerMsg::SetClipPitch { .. }
            | WorkerMsg::SetSpeedCurve { .. }
            | WorkerMsg::SetSpeedCurvePoint { .. }
            | WorkerMsg::SetClipCrop { .. }
            // Effects and transitions repaint the canvas at the playhead.
            | WorkerMsg::AddEffect { .. }
            | WorkerMsg::RemoveEffect { .. }
            | WorkerMsg::SetEffectParam { .. }
            | WorkerMsg::AddTransition { .. }
            | WorkerMsg::RemoveTransition { .. }
            | WorkerMsg::SetTransition { .. }
            // Aspect reshapes the composite, background recolors it.
            | WorkerMsg::SetCanvas { .. }
            | WorkerMsg::SetParamKeyframe { .. }
            | WorkerMsg::RemoveParamKeyframe { .. }
            | WorkerMsg::RetimeKeyframes { .. }
            | WorkerMsg::RemoveKeyframesAt { .. }
            | WorkerMsg::SplitClip { .. }
            | WorkerMsg::PasteAt { .. }
            | WorkerMsg::DuplicateClips { .. }
            | WorkerMsg::Undo
            | WorkerMsg::Redo
            | WorkerMsg::SetMainMagnet(_)
            | WorkerMsg::SetTrackFlag { .. }
            | WorkerMsg::OpenProject { .. }
            | WorkerMsg::NewProject
            // Relinked media decodes again — refresh the stale composite.
            | WorkerMsg::RelinkMedia { .. }
            | WorkerMsg::RelinkFolder { .. }
            // A forced library delete removes the source's clips too; an
            // unreferenced delete touches nothing on the canvas.
            | WorkerMsg::RemoveMedia { force: true, .. }
    )
}

/// Fit/fill helper (M1 canvas settings): compute the centered fit (scale
/// 1.0) or cover transform for a clip and commit it through the regular
/// `SetClipTransform` path, so it keyframes at the playhead on animated
/// clips and undoes in one step like any gesture.
fn fit_clip_and_publish(
    engine: &mut Engine,
    clip: &str,
    fill: bool,
    tick: i64,
    tl_rate: Rational,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "fit/fill ignored: unparsable clip id");
        return;
    };
    let Some(transform) = fit_clip_transform(engine, clip_id, fill, tick) else {
        error!(%clip_id, "fit/fill ignored: unknown clip or degenerate content");
        return;
    };
    let at = RationalTime::new(tick, tl_rate);
    set_transform_and_publish(engine, clip, transform, at, ui);
}

/// The transform that centers a clip at aspect-fit (scale 1.0 by the
/// placement convention) or at the cover scale that fills the canvas — the
/// crop's kept region is what aspect-fits, so it is also what must cover.
/// Rotation and opacity keep their playhead-sampled values; position resets
/// to center (CapCut fit/fill semantics).
fn fit_clip_transform(
    engine: &Engine,
    clip_id: ClipId,
    fill: bool,
    tick: i64,
) -> Option<ClipTransform> {
    let project = engine.project();
    let clip = project.clip(clip_id)?;
    let (canvas_w, canvas_h) = cutlass_render::canvas_size(project);
    let (content_w, content_h) = match clip.media() {
        Some(media_id) => {
            let media = project.media(media_id)?;
            (media.width, media.height)
        }
        // Generators raster at canvas size: fit and fill are both 1.0.
        None => (canvas_w, canvas_h),
    };
    let (w, h) = (
        content_w as f32 * clip.crop.w,
        content_h as f32 * clip.crop.h,
    );
    if w <= 0.0 || h <= 0.0 || canvas_w == 0 || canvas_h == 0 {
        return None;
    }
    let (cw, ch) = (canvas_w as f32, canvas_h as f32);
    let fit = (cw / w).min(ch / h);
    let cover = (cw / w).max(ch / h);
    let scale = if fill { cover / fit } else { 1.0 };
    let sampled = clip.transform.sample_at(clip.animation_tick_f(tick as f64));
    Some(ClipTransform {
        position: [0.0, 0.0],
        anchor_point: sampled.anchor_point,
        scale,
        rotation: sampled.rotation,
        opacity: sampled.opacity,
    })
}

/// Commit a transform gesture as one undoable `SetClipTransform`, keyframing
/// at `at` (the playhead) when the property is animated.
fn set_transform_and_publish(
    engine: &mut Engine,
    clip: &str,
    transform: ClipTransform,
    at: RationalTime,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-transform ignored: unparsable clip id");
        return;
    };
    // CapCut compose semantics: on a clip with animated properties this
    // commit writes keyframes at the playhead instead of flattening. Note it
    // before applying so the UI can surface "a gesture added a keyframe".
    let wrote_keyframe = engine
        .project()
        .clip(clip_id)
        .is_some_and(|c| c.transform.is_animated());
    match engine.apply(Command::Edit(EditCommand::SetClipTransform {
        clip: clip_id,
        transform,
        at: Some(at),
    })) {
        Ok(_) => {
            info!(%clip_id, ?transform, "set clip transform");
            if wrote_keyframe {
                bump_keyframe_commit_epoch(ui);
            }
            publish_projection(engine, ui);
        }
        Err(e) => {
            error!(%clip_id, "set transform failed: {e}");
            publish_projection(engine, ui);
        }
    }
}

/// Signal the inspector that a transform gesture just wrote keyframes (the
/// transient "keyframe added" chip): bump `EditorStore.keyframe-commit-epoch`.
fn bump_keyframe_commit_epoch(ui: &UiSink) {
    let editor_weak = ui.editor.clone();
    if let Err(e) = slint::invoke_from_event_loop(move || {
        if let Some(store) = editor_weak.upgrade() {
            store.set_keyframe_commit_epoch(store.get_keyframe_commit_epoch().wrapping_add(1));
        }
    }) {
        error!("failed to bump keyframe commit epoch: {e}");
    }
}

/// Insert or replace one property keyframe at `at` (absolute playhead
/// position) as one undoable edit (keyframes roadmap Phase 1: the inspector
/// diamond / easing picker). Engine-rejected positions (playhead outside the
/// clip — the UI gates, but a stale projection can race) only log.
fn set_param_keyframe_and_publish(
    engine: &mut Engine,
    clip: &str,
    param: ClipParam,
    at: RationalTime,
    value: ParamValue,
    easing: Easing,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-param-keyframe ignored: unparsable clip id");
        return;
    };
    match engine.apply(Command::Edit(EditCommand::SetParamKeyframe {
        clip: clip_id,
        param,
        at,
        value,
        easing,
    })) {
        Ok(_) => {
            info!(%clip_id, ?param, tick = at.value, "set param keyframe");
            publish_projection(engine, ui);
        }
        Err(e) => error!(%clip_id, ?param, "set param keyframe failed: {e}"),
    }
}

/// Remove the keyframe at exactly `at` on one property (inspector diamond
/// toggled off). The engine rejects when nothing sits there.
fn remove_param_keyframe_and_publish(
    engine: &mut Engine,
    clip: &str,
    param: ClipParam,
    at: RationalTime,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "remove-param-keyframe ignored: unparsable clip id");
        return;
    };
    match engine.apply(Command::Edit(EditCommand::RemoveParamKeyframe {
        clip: clip_id,
        param,
        at,
    })) {
        Ok(_) => {
            info!(%clip_id, ?param, tick = at.value, "removed param keyframe");
            publish_projection(engine, ui);
        }
        Err(e) => error!(%clip_id, ?param, "remove param keyframe failed: {e}"),
    }
}

/// Every animated property with a keyframe exactly at the clip-relative
/// `rel_tick`, with that keyframe's value and easing — the slice of one
/// merged timeline diamond (the timeline draws one diamond per tick across
/// all properties, CapCut-style).
fn keyframes_at(
    transform: &AnimatedTransform,
    rel_tick: i64,
) -> Vec<(ClipParam, ParamValue, Easing)> {
    let mut hits = Vec::new();
    if let Some(kf) = transform
        .position
        .keyframes()
        .iter()
        .find(|k| k.tick == rel_tick)
    {
        hits.push((ClipParam::Position, ParamValue::Vec2(kf.value), kf.easing));
    }
    let scalars = [
        (ClipParam::Scale, &transform.scale),
        (ClipParam::Rotation, &transform.rotation),
        (ClipParam::Opacity, &transform.opacity),
    ];
    for (param, p) in scalars {
        if let Some(kf) = p.keyframes().iter().find(|k| k.tick == rel_tick) {
            hits.push((param, ParamValue::Scalar(kf.value), kf.easing));
        }
    }
    hits
}

/// Move every keyframe at `from_tick` to `to_tick` (timeline diamond drag,
/// keyframes roadmap Phase 2): per property a remove + re-set with the same
/// value and easing, all in one history group so a single undo puts the
/// diamond back. A keyframe already sitting at the destination on the same
/// property is replaced (the diamonds merge, like CapCut). The engine
/// re-validates that `to_tick` falls inside the clip.
fn retime_keyframes_and_publish(
    engine: &mut Engine,
    clip: &str,
    from_tick: i64,
    to_tick: i64,
    tl_rate: Rational,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "retime-keyframes ignored: unparsable clip id");
        return;
    };
    if from_tick == to_tick {
        return;
    }
    let Some(model) = engine.project().clip(clip_id) else {
        error!(%clip_id, "retime-keyframes ignored: clip not on the timeline");
        return;
    };
    let moved = keyframes_at(&model.transform, from_tick - model.timeline.start.value);
    if moved.is_empty() {
        error!(%clip_id, from_tick, "retime-keyframes ignored: no keyframes at tick");
        return;
    }

    engine.begin_group();
    for (param, value, easing) in moved {
        if let Err(e) = engine.apply(Command::Edit(EditCommand::RemoveParamKeyframe {
            clip: clip_id,
            param,
            at: RationalTime::new(from_tick, tl_rate),
        })) {
            error!(%clip_id, ?param, "retime keyframes failed removing: {e}");
            engine.rollback_group();
            return;
        }
        if let Err(e) = engine.apply(Command::Edit(EditCommand::SetParamKeyframe {
            clip: clip_id,
            param,
            at: RationalTime::new(to_tick, tl_rate),
            value,
            easing,
        })) {
            error!(%clip_id, ?param, "retime keyframes failed setting: {e}");
            engine.rollback_group();
            return;
        }
    }
    engine.commit_group();
    info!(%clip_id, from_tick, to_tick, "retimed keyframes");
    publish_projection(engine, ui);
}

/// Remove every property's keyframe at `tick` (timeline diamond
/// right-click) as one history group — one undo restores the whole merged
/// diamond.
fn remove_keyframes_at_and_publish(
    engine: &mut Engine,
    clip: &str,
    tick: i64,
    tl_rate: Rational,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "remove-keyframes ignored: unparsable clip id");
        return;
    };
    let Some(model) = engine.project().clip(clip_id) else {
        error!(%clip_id, "remove-keyframes ignored: clip not on the timeline");
        return;
    };
    let hits = keyframes_at(&model.transform, tick - model.timeline.start.value);
    if hits.is_empty() {
        error!(%clip_id, tick, "remove-keyframes ignored: no keyframes at tick");
        return;
    }

    engine.begin_group();
    for (param, _, _) in hits {
        if let Err(e) = engine.apply(Command::Edit(EditCommand::RemoveParamKeyframe {
            clip: clip_id,
            param,
            at: RationalTime::new(tick, tl_rate),
        })) {
            error!(%clip_id, ?param, "remove keyframes failed: {e}");
            engine.rollback_group();
            return;
        }
    }
    engine.commit_group();
    info!(%clip_id, tick, "removed keyframes at tick");
    publish_projection(engine, ui);
}

// PORT: main warmed the disk cache here with a `prefetch_ahead` read-ahead
// after idle frames; this branch's engine keeps decode read-ahead internal
// to its native decoders, so the worker sends nothing.

fn import_and_publish(engine: &mut Engine, path: &Path, ui: &UiSink) {
    match engine.apply(Command::Project(ProjectCommand::Import {
        path: path.to_path_buf(),
    })) {
        Ok(ApplyOutcome::Imported { media }) => {
            info!(
                ?media,
                path = %path.display(),
                pool = engine.project().media_count(),
                "imported media into pool"
            );
            // PORT (Phase 4): register the source with the thumbnail/strip
            // workers here so tiles and filmstrips generate off-thread.
            publish_projection(engine, ui);
        }
        Ok(other) => error!(path = %path.display(), "unexpected import outcome: {other:?}"),
        Err(e) => error!(path = %path.display(), "import failed: {e}"),
    }
}

/// Persist the session to its draft file. `path` is the draft's
/// `project.cutlass` (binding a freshly created draft); `None` reuses the
/// engine's current path — the debounced auto-save and the flush before a
/// session swap / close. Success refreshes the draft's name sidecar and
/// republishes the projection; failure surfaces through `session-error`. A
/// `None` flush with no bound draft (e.g. New from the launch screen over the
/// empty boot session) has nothing to persist and is a quiet no-op.
fn save_project_and_publish(engine: &mut Engine, path: Option<PathBuf>, ui: &UiSink) {
    let Some(path) = path.or_else(|| engine.project_path().cloned()) else {
        return;
    };
    match engine.apply(Command::Project(ProjectCommand::Save {
        path: path.clone(),
    })) {
        Ok(ApplyOutcome::Saved) => {
            crate::drafts::write_meta(&path, &engine.project().name);
            publish_projection(engine, ui);
        }
        Ok(other) => error!(path = %path.display(), "unexpected save outcome: {other:?}"),
        Err(e) => {
            error!(path = %path.display(), "auto-save failed: {e}");
            publish_session_error(
                ui,
                format!("Couldn't save the project to {}: {e}", path.display()),
            );
        }
    }
}

/// Replace the session from a `.cutlass` file. Tolerant (`Load`, not the
/// strict `Open`): entries whose media file is gone are kept so the user
/// can relink them instead of being locked out of the project — the
/// projection republish carries the missing set (count + per-tile badges)
/// and app.slint raises the relink dialog on the epoch bump. On success
/// every still-present pool media re-registers with the thumbnail and
/// strip workers — the same bookkeeping an import does — the projection
/// republish swaps the UI over, and the session epoch bump resets UI
/// session state (playhead, selection, in/out range). On failure the
/// current session is untouched (the engine rejects before replacing) and
/// `session-error` names the offending path.
fn open_project_and_publish(engine: &mut Engine, path: PathBuf, ui: &UiSink) {
    match engine.apply(Command::Project(ProjectCommand::Load {
        path: path.clone(),
    })) {
        Ok(ApplyOutcome::Loaded) => {
            info!(
                path = %path.display(),
                pool = engine.project().media_count(),
                "opened project"
            );
            // PORT (Phase 4): re-register still-present pool media with the
            // thumbnail/strip workers here (the same bookkeeping an import
            // does).
            publish_projection(engine, ui);
            bump_session_epoch(ui);
        }
        Ok(other) => error!(path = %path.display(), "unexpected open outcome: {other:?}"),
        Err(e) => {
            error!(path = %path.display(), "open failed: {e}");
            publish_session_error(ui, format!("Couldn't open {}: {e}", path.display()));
        }
    }
}

/// Re-point a pool entry at a user-picked file (missing-media relink, M0).
/// The engine re-probes and swaps the entry's path/metadata in place (same
/// id — clips recover without being touched); the tile workers re-register
/// so the thumbnail and filmstrips regenerate from the new file; the
/// projection republish clears the entry's missing badge and decrements
/// the dialog's count. Failures (unreadable file, probe error) surface
/// through `session-error` and leave the entry untouched.
fn relink_media_and_publish(engine: &mut Engine, media: &str, path: &Path, ui: &UiSink) {
    let Some(media_id) = parse_raw_id(media).map(MediaId::from_raw) else {
        error!(media, "relink ignored: unparsable media id");
        return;
    };
    match engine.apply(Command::Project(ProjectCommand::RelinkMedia {
        media: media_id,
        path: path.to_path_buf(),
    })) {
        Ok(ApplyOutcome::Relinked { media }) => {
            info!(?media, path = %path.display(), "relinked media");
            // PORT (Phase 4): re-register with the tile workers so the
            // thumbnail and filmstrips regenerate from the new file.
            publish_projection(engine, ui);
        }
        Ok(other) => error!(path = %path.display(), "unexpected relink outcome: {other:?}"),
        Err(e) => {
            error!(path = %path.display(), "relink failed: {e}");
            publish_session_error(ui, format!("Couldn't relink to {}: {e}", path.display()));
        }
    }
}

/// Try `folder/<filename>` for every missing pool entry; relink each match.
fn relink_folder_and_publish(engine: &mut Engine, folder: PathBuf, ui: &UiSink) {
    let candidates: Vec<(MediaId, PathBuf)> = engine
        .project()
        .media_iter()
        .filter(|media| !media.path().exists())
        .filter_map(|media| {
            media
                .path()
                .file_name()
                .map(|name| (media.id, folder.join(name)))
        })
        .filter(|(_, candidate)| candidate.exists())
        .collect();

    if candidates.is_empty() {
        publish_session_error(
            ui,
            format!(
                "No missing media files were found in {}. \
                 Pick individual files or choose a folder that contains them.",
                folder.display()
            ),
        );
        return;
    }

    let mut relinked = 0usize;
    for (media_id, path) in candidates {
        match engine.apply(Command::Project(ProjectCommand::RelinkMedia {
            media: media_id,
            path: path.clone(),
        })) {
            // PORT (Phase 4): re-register each relinked source with the tile
            // workers so thumbnails/filmstrips regenerate.
            Ok(ApplyOutcome::Relinked { .. }) => relinked += 1,
            Ok(other) => error!(path = %path.display(), "unexpected relink outcome: {other:?}"),
            Err(e) => error!(path = %path.display(), "folder relink failed: {e}"),
        }
    }

    if relinked > 0 {
        info!(count = relinked, folder = %folder.display(), "relinked media from folder");
        publish_projection(engine, ui);
    }
}

/// Delete a source from the media pool (Library bin). `force` false removes
/// only unreferenced media — the engine rejects a referenced source, which the
/// UI prevents by gating on the tile's usage count. `force` true first deletes
/// every clip referencing the source and then the source, all in one history
/// group, so a single undo restores both. The thumbnail cache entry is evicted
/// on success. Lanes the cascade empties are pruned, matching the clip-delete
/// flow (`remove_clips_and_publish`).
fn remove_media_and_publish(engine: &mut Engine, media: &str, force: bool, ui: &UiSink) {
    let Some(media_id) = parse_raw_id(media).map(MediaId::from_raw) else {
        error!(media, "delete-media ignored: unparsable media id");
        return;
    };
    if engine.project().media(media_id).is_none() {
        error!(%media_id, "delete-media ignored: not in the pool");
        return;
    }

    if !force {
        match engine.apply(Command::Project(ProjectCommand::RemoveMedia {
            media: media_id,
        })) {
            Ok(ApplyOutcome::RemovedMedia { media }) => {
                info!(?media, "removed media from pool");
                // PORT (Phase 4): evict the thumbnail cache entry here.
                publish_projection(engine, ui);
            }
            Ok(other) => error!(%media_id, "unexpected remove-media outcome: {other:?}"),
            // The UI only sends the unforced delete for an unreferenced tile,
            // so a rejection here is a race (a clip landed on it between the
            // projection and the click) — surface it instead of dropping it.
            Err(e) => {
                error!(%media_id, "remove media failed: {e}");
                publish_session_error(ui, format!("Couldn't remove the media: {e}"));
            }
        }
        return;
    }

    // Cascade: gather every clip that references the source up front (a Library
    // delete leaves gaps where the clips sat — it isn't a timeline-timing
    // edit), then remove them and the source as one undoable group.
    let mut doomed: Vec<(ClipId, TrackId)> = Vec::new();
    for track in engine.project().timeline().tracks_ordered() {
        for clip in track.clips() {
            if clip.media() == Some(media_id) {
                doomed.push((clip.id, track.id));
            }
        }
    }

    engine.begin_group();
    for &(clip_id, _) in &doomed {
        if let Err(e) = apply_edit(engine, EditCommand::RemoveClip { clip: clip_id }) {
            error!(%clip_id, "remove referencing clip failed: {e}");
            engine.rollback_group();
            publish_projection(engine, ui);
            return;
        }
    }
    if let Err(e) = engine.apply(Command::Project(ProjectCommand::RemoveMedia {
        media: media_id,
    })) {
        error!(%media_id, "remove media failed after clearing clips: {e}");
        engine.rollback_group();
        publish_projection(engine, ui);
        return;
    }
    // Prune lanes the removals emptied (CapCut drops emptied overlay tracks).
    let mut lanes: Vec<TrackId> = doomed.iter().map(|&(_, track)| track).collect();
    lanes.sort();
    lanes.dedup();
    for lane in lanes {
        remove_track_if_empty(engine, lane);
    }
    engine.commit_group();
    info!(%media_id, clips = doomed.len(), "removed media and its referencing clips");
    // PORT (Phase 4): evict the thumbnail cache entry here.
    publish_projection(engine, ui);
}

/// Replace the session with a fresh, empty project (the launch screen's New,
/// or New from the editor). The draft's `project.cutlass` is written by the
/// `SaveProject { Some(path) }` main.rs sends right after.
fn new_project_and_publish(engine: &mut Engine, ui: &UiSink) {
    engine.new_session();
    info!("new session");
    publish_projection(engine, ui);
    bump_session_epoch(ui);
}

/// Rename the current draft (title bar). Applied as one undoable edit so it
/// joins the undo history and dirties the session; the projection republish
/// updates the title, and the debounced auto-save writes the new name into
/// the draft's project file and meta sidecar.
fn rename_project_and_publish(engine: &mut Engine, name: String, ui: &UiSink) {
    // The title field commits on blur as well as Enter, so a focus-and-leave
    // with no change arrives here unchanged — skip it so it never spends an
    // undo entry or dirties the draft.
    if engine.project().name == name {
        return;
    }
    match engine.apply(Command::Edit(EditCommand::SetProjectName { name })) {
        Ok(_) => publish_projection(engine, ui),
        Err(e) => error!("project rename failed: {e}"),
    }
}

/// Place the full source range of `media` on a video track (audio-only media
/// lands on an audio track), then republish the projection so the clip appears.
///
/// Placement policy (CapCut-ish):
/// - dropped on a lane of the media's kind → that lane, sliding right into the
///   first gap that fits when the drop tick overlaps existing clips;
/// - dropped on empty timeline space (`track` empty) or a lane of another
///   kind → a fresh track of the media's kind inserted at `drop_row`, so the
///   new lane appears where the user dropped (above the lanes ⇒ top of the
///   stack, below ⇒ bottom);
/// - dropped on the main lane with the magnet on (`insert`) → ripple-insert
///   at `start_tick`, shifting later clips right (atomic engine command).
///
/// A video drop whose media carries audio lands a *single* clip — CapCut keeps
/// the sound on the video clip and the audio mixers read it from that lane, so
/// no separate audio lane is spawned.
#[allow(clippy::too_many_arguments)]
fn add_clip_and_publish(
    engine: &mut Engine,
    media: &str,
    track: &str,
    start_tick: i64,
    drop_row: i64,
    insert: bool,
    ui: &UiSink,
) {
    let Some(media_id) = parse_raw_id(media).map(MediaId::from_raw) else {
        error!(media, "drop ignored: unparsable media id");
        return;
    };
    let Some((source, audio_only)) = engine
        .project()
        .media(media_id)
        .map(|m| (m.full_range(), m.is_audio_only()))
    else {
        error!(%media_id, "drop ignored: media not in pool");
        return;
    };
    let lane_kind = if audio_only {
        TrackKind::Audio
    } else {
        TrackKind::Video
    };
    let tl_rate = engine.project().timeline().frame_rate;
    // Mirror Project::add_clip's source→timeline resampling so first-fit sees
    // the same extent the engine will validate.
    let duration_ticks = resample(source.duration, tl_rate).value.max(1);

    // CapCut keeps a video's sound on the video clip itself — a drop lands one
    // clip and the audio mixers read its audio from that lane (see
    // `audio_snapshot`). No companion lane is spawned; "Separate audio" is a
    // later, explicit gesture.

    // The main-track magnet only applies to the main *video* lane.
    if insert
        && !audio_only
        && let Some(lane) = lane_of_kind(engine, track, TrackKind::Video)
    {
        let at = start_tick.max(0);
        engine.begin_group();
        match engine.apply(Command::Edit(EditCommand::RippleInsert {
            track: lane,
            media: media_id,
            source,
            at: RationalTime::new(at, tl_rate),
        })) {
            Ok(ApplyOutcome::Edited(EditOutcome::Created(clip))) => {
                engine.commit_group();
                info!(%clip, %lane, %media_id, at, "ripple-inserted clip from library drop");
            }
            Ok(other) => {
                error!(%media_id, "unexpected ripple-insert outcome: {other:?}");
                engine.rollback_group();
            }
            Err(e) => {
                error!(%media_id, %lane, start_tick, "ripple insert failed: {e}");
                engine.rollback_group();
            }
        }
        publish_projection(engine, ui);
        return;
    }
    let desired = start_tick.max(0);

    // One history entry per drop, even when it creates the landing lane.
    engine.begin_group();
    let (track_id, start_value) = match lane_of_kind(engine, track, lane_kind) {
        Some(lane) => {
            let lane_track = engine
                .project()
                .timeline()
                .track(lane)
                .expect("lane_of_kind returned an existing track");
            (lane, first_fit_start(lane_track, desired, duration_ticks))
        }
        None => match create_track(engine, lane_kind, drop_row) {
            Ok(id) => (id, desired),
            Err(e) => {
                error!(%media_id, "drop failed creating {lane_kind:?} track: {e}");
                engine.rollback_group();
                return;
            }
        },
    };

    match engine.apply(Command::Edit(EditCommand::AddClip {
        track: track_id,
        media: media_id,
        source,
        start: RationalTime::new(start_value, tl_rate),
    })) {
        Ok(ApplyOutcome::Edited(EditOutcome::Created(clip))) => {
            engine.commit_group();
            info!(
                %clip, %track_id, %media_id,
                start_tick = start_value,
                desired,
                "added clip from library drop"
            );
            publish_projection(engine, ui);
        }
        // First-fit should have made the placement valid; the engine still
        // rejects atomically if not. Surface the reason and roll the group
        // back so a lane created for this drop doesn't linger.
        Ok(other) => {
            error!(%media_id, "unexpected add-clip outcome: {other:?}");
            engine.rollback_group();
            publish_projection(engine, ui);
        }
        Err(e) => {
            error!(%media_id, %track_id, start_tick = start_value, "add clip failed: {e}");
            engine.rollback_group();
            publish_projection(engine, ui);
        }
    }
}

/// Place a generated clip (text/solid/shape) from a library-tile drop. One
/// history entry, even when it creates the landing lane; rolled back on a
/// rejected placement so a lane made for the drop doesn't linger.
fn add_generated_and_publish(
    engine: &mut Engine,
    generator: Generator,
    track: &str,
    start_tick: i64,
    duration_ticks: i64,
    drop_row: i64,
    ui: &UiSink,
) {
    let Some(lane_kind) = TrackKind::for_generator(&generator) else {
        error!(
            ?generator,
            "generated drop ignored: no lane kind for generator"
        );
        return;
    };
    let desired = start_tick.max(0);
    let duration = duration_ticks.max(1);

    engine.begin_group();
    let track_id = match lane_of_kind(engine, track, lane_kind) {
        Some(lane) => {
            let lane_track = engine
                .project()
                .timeline()
                .track(lane)
                .expect("lane_of_kind returned an existing track");
            let start = first_fit_start(lane_track, desired, duration);
            (lane, start)
        }
        None => match create_track(engine, lane_kind, drop_row) {
            Ok(id) => (id, desired),
            Err(e) => {
                error!(
                    ?generator,
                    "generated drop failed creating {lane_kind:?} track: {e}"
                );
                engine.rollback_group();
                return;
            }
        },
    };
    let (track_id, start_value) = track_id;

    let content = ClipSource::Generated(generator);
    match add_clip_content(engine, track_id, &content, duration, start_value) {
        Ok(clip) => {
            engine.commit_group();
            info!(%clip, %track_id, start_tick = start_value, "added generated clip from drop");
            publish_projection(engine, ui);
        }
        Err(e) => {
            error!(%track_id, start_tick = start_value, "add generated clip failed: {e}");
            engine.rollback_group();
            publish_projection(engine, ui);
        }
    }
}

/// Retime a media clip (CapCut speed, M1). The engine validates (positive
/// speed, media-backed clip, no neighbor overlap) and re-derives the
/// timeline duration; one undoable history entry. With linkage on, the
/// clip's link partners (the video+audio pair from one media drop) retime
/// together in one history group, so the pair stays in sync and one undo
/// restores both. Audio of retimed clips is muted by the snapshot builder,
/// so the republish silences it immediately.
fn set_clip_speed_and_publish(
    engine: &mut Engine,
    clip: &str,
    num: i32,
    den: i32,
    reversed: bool,
    linkage: bool,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-clip-speed ignored: unparsable clip id");
        return;
    };
    let targets = if linkage {
        link_group_ids(engine, clip_id)
    } else {
        vec![clip_id]
    };

    engine.begin_group();
    for target in &targets {
        if let Err(e) = engine.apply(Command::Edit(EditCommand::SetClipSpeed {
            clip: *target,
            speed: Rational::new(num, den),
            reversed,
        })) {
            error!(clip_id = %target, "set clip speed failed: {e}");
            engine.rollback_group();
            publish_projection(engine, ui);
            return;
        }
    }
    engine.commit_group();
    info!(%clip_id, num, den, reversed, clips = targets.len(), "retimed clip");
    publish_projection(engine, ui);
}

/// Toggle pitch preservation on a retimed media clip (CapCut "pitch" switch,
/// M8 Phase 3). With linkage on the whole link group flips together so an A/V
/// pair stays consistent — one undoable history entry. The republish
/// re-snapshots the mixer so the new stretch mode is audible immediately.
fn set_clip_pitch_and_publish(
    engine: &mut Engine,
    clip: &str,
    preserve: bool,
    linkage: bool,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-clip-pitch ignored: unparsable clip id");
        return;
    };
    let targets = if linkage {
        link_group_ids(engine, clip_id)
    } else {
        vec![clip_id]
    };

    engine.begin_group();
    for target in &targets {
        if let Err(e) = engine.apply(Command::Edit(EditCommand::SetClipPitch {
            clip: *target,
            preserve_pitch: preserve,
        })) {
            error!(clip_id = %target, "set clip pitch failed: {e}");
            engine.rollback_group();
            publish_projection(engine, ui);
            return;
        }
    }
    engine.commit_group();
    info!(%clip_id, preserve, clips = targets.len(), "set clip pitch");
    publish_projection(engine, ui);
}

/// Toggle noise reduction on a media clip (CapCut "Reduce noise", M8 Phase 5).
/// With linkage on, the whole link group follows so selecting a video half
/// still cleans its audio companion — one undoable history group. The
/// republish re-snapshots the mixer, which renders the cleaned signal.
fn set_denoise_and_publish(
    engine: &mut Engine,
    clip: &str,
    denoise: bool,
    linkage: bool,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-denoise ignored: unparsable clip id");
        return;
    };
    let targets = if linkage {
        link_group_ids(engine, clip_id)
    } else {
        vec![clip_id]
    };

    engine.begin_group();
    for target in &targets {
        if let Err(e) = engine.apply(Command::Edit(EditCommand::SetClipDenoise {
            clip: *target,
            denoise,
        })) {
            error!(clip_id = %target, "set denoise failed: {e}");
            engine.rollback_group();
            publish_projection(engine, ui);
            return;
        }
    }
    engine.commit_group();
    info!(%clip_id, denoise, clips = targets.len(), "set denoise");
    publish_projection(engine, ui);
}

/// Set (or clear) a media clip's speed ramp (CapCut speed curves, M2). Like
/// constant-speed retiming the engine re-derives each clip's timeline
/// duration from the ramp average, so with linkage on every link partner
/// ramps in lockstep to keep A/V in sync — one undoable history group. The
/// republish re-snapshots the mixer, which now plays the ramp time-stretched
/// along its curve (M8 Phase 3).
fn set_speed_curve_and_publish(
    engine: &mut Engine,
    clip: &str,
    curve: &Option<Param<f32>>,
    linkage: bool,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-speed-curve ignored: unparsable clip id");
        return;
    };
    let targets = if linkage {
        link_group_ids(engine, clip_id)
    } else {
        vec![clip_id]
    };

    engine.begin_group();
    for target in &targets {
        if let Err(e) = engine.apply(Command::Edit(EditCommand::SetSpeedCurve {
            clip: *target,
            curve: curve.clone(),
        })) {
            error!(clip_id = %target, "set speed curve failed: {e}");
            engine.rollback_group();
            publish_projection(engine, ui);
            return;
        }
    }
    engine.commit_group();
    info!(%clip_id, points = curve.as_ref().map_or(0, |c| c.keyframes().len()), clips = targets.len(), "set speed ramp");
    publish_projection(engine, ui);
}

/// Adjust one existing ramp point's multiplier (velocity-graph drag). Reads
/// the addressed clip's current curve, replaces point `index`'s value, and
/// re-commits through [`set_speed_curve_and_publish`] so duration re-derive,
/// linkage, and undo all flow through the one path.
fn set_speed_curve_point_and_publish(
    engine: &mut Engine,
    clip: &str,
    index: usize,
    value: f32,
    linkage: bool,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-speed-curve-point ignored: unparsable clip id");
        return;
    };
    let Some(mut curve) = engine
        .project()
        .clip(clip_id)
        .map(|c| c.speed_curve.clone())
    else {
        error!(%clip_id, "set-speed-curve-point ignored: unknown clip");
        return;
    };
    // Address the point by index, but edit it through the keyframe API at its
    // own tick so the curve keeps its shape (tick + easing) and stays sorted.
    let Some(&point) = curve.keyframes().get(index) else {
        warn!(%clip_id, index, "set-speed-curve-point ignored: index out of range");
        return;
    };
    curve.set_keyframe(point.tick, value.clamp(MIN_SPEED, MAX_SPEED), point.easing);
    set_speed_curve_and_publish(engine, clip, &Some(curve), linkage, ui);
}

/// Set a clip's audio mix (CapCut volume + fades, M1). A video clip carries
/// its own sound, so the edit lands on the clicked clip; only when its audio
/// was detached to a linked audio lane does it follow to the audible half
/// there. One history group; the republish re-snapshots the playback mixer, so
/// the change is audible within a block.
fn set_clip_audio_and_publish(
    engine: &mut Engine,
    clip: &str,
    volume: Option<f32>,
    fade_in_s: f32,
    fade_out_s: f32,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-clip-audio ignored: unparsable clip id");
        return;
    };
    // CapCut keeps a video's sound on its own clip, so volume/fades land on the
    // clicked clip when it carries its own audio (a video drop, or an audio
    // lane). Only when its sound was detached to a linked audio lane does the
    // edit follow to the audible half there.
    let targets: Vec<ClipId> = if engine.project().timeline().carries_own_audio(clip_id) {
        vec![clip_id]
    } else {
        link_group_ids(engine, clip_id)
            .into_iter()
            .filter(|id| engine.project().timeline().carries_own_audio(*id))
            .collect()
    };
    if targets.is_empty() {
        warn!(%clip_id, "set-clip-audio ignored: no audible clip to adjust");
        return;
    }

    let tl_rate = engine.project().timeline().frame_rate;
    let to_ticks = |seconds: f32| {
        let ticks = (f64::from(seconds) * f64::from(tl_rate.num) / f64::from(tl_rate.den)).round();
        RationalTime::new(ticks.max(0.0) as i64, tl_rate)
    };
    let (fade_in, fade_out) = (to_ticks(fade_in_s), to_ticks(fade_out_s));

    engine.begin_group();
    for target in &targets {
        if let Err(e) = engine.apply(Command::Edit(EditCommand::SetClipAudio {
            clip: *target,
            volume,
            fade_in,
            fade_out,
        })) {
            error!(clip_id = %target, "set clip audio failed: {e}");
            engine.rollback_group();
            publish_projection(engine, ui);
            return;
        }
    }
    engine.commit_group();
    info!(%clip_id, ?volume, fade_in_s, fade_out_s, clips = targets.len(), "set clip audio");
    publish_projection(engine, ui);
}

/// Duck a music clip under the voice lanes (M8 Phase 4). Gathers every clip on
/// a voice-tagged (`duck_source`) audio lane that overlaps the selected music
/// clip and lowers `DuckLanes` onto it — the engine writes the dip as ordinary
/// M8 volume keyframes, so the result is one undoable edit, audible on the next
/// mixer snapshot and editable through the volume envelope afterwards. The
/// defaults mirror the decoder's broadcast-typical ducker (and the agent
/// `duck` tool); the linear speech-band threshold stays an internal detail.
fn duck_under_voice_and_publish(engine: &mut Engine, clip: &str, ui: &UiSink) {
    let Some(music_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "duck-under-voice ignored: unparsable clip id");
        return;
    };

    // Resolve the overlapping voice clips against an immutable view, never
    // ducking a clip under its own lane.
    let voice: Vec<ClipId> = {
        let project = engine.project();
        let timeline = project.timeline();
        let Some(music) = project.clip(music_id) else {
            warn!(%music_id, "duck-under-voice ignored: unknown clip");
            return;
        };
        let music_track = timeline.track_of(music_id);
        let music_range = music.timeline;
        timeline
            .tracks_ordered()
            .filter(|track| {
                track.kind == TrackKind::Audio && track.duck_source && Some(track.id) != music_track
            })
            .flat_map(|track| track.clips_ordered())
            .filter(|c| c.timeline.overlaps(music_range).unwrap_or(false))
            .map(|c| c.id)
            .collect()
    };
    if voice.is_empty() {
        warn!(%music_id, "duck-under-voice: no voice-lane clips overlap the selected music");
        return;
    }

    match engine.apply(Command::Edit(EditCommand::DuckLanes {
        voice,
        music: vec![music_id],
        // Mirror `DuckSettings::default()` / the agent `duck` tool defaults.
        threshold: 0.025,
        amount: 0.66,
        attack: 0.08,
        release: 0.32,
    })) {
        Ok(ApplyOutcome::Edited(EditOutcome::Updated(_))) => {
            info!(%music_id, "ducked music under voice");
            publish_projection(engine, ui);
        }
        Ok(other) => error!(%music_id, "unexpected duck-under-voice outcome: {other:?}"),
        Err(e) => error!(%music_id, "duck under voice failed: {e}"),
    }
}

/// Detect beat markers on a media clip (CapCut "Beat", M8 Phase 6): the engine
/// decodes the clip's audio, runs onset/tempo analysis, and stores the beat
/// grid (source ticks) so the timeline magnet can snap clip edges to it. One
/// undoable history entry. A rejection (generated clip / no audio) just logs —
/// the inspector only offers the button on media clips with sound, so it would
/// be a stale-projection race.
fn detect_beats_and_publish(engine: &mut Engine, clip: &str, ui: &UiSink) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "detect-beats ignored: unparsable clip id");
        return;
    };
    match engine.apply(Command::Edit(EditCommand::DetectBeats { clip: clip_id })) {
        Ok(ApplyOutcome::Edited(EditOutcome::Updated(_))) => {
            info!(%clip_id, "detected beats");
            publish_projection(engine, ui);
        }
        Ok(other) => error!(%clip_id, "unexpected detect-beats outcome: {other:?}"),
        Err(e) => error!(%clip_id, "detect beats failed: {e}"),
    }
}

/// Clear a clip's detected beat markers (M8 Phase 6). One undoable entry.
fn clear_beats_and_publish(engine: &mut Engine, clip: &str, ui: &UiSink) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "clear-beats ignored: unparsable clip id");
        return;
    };
    match engine.apply(Command::Edit(EditCommand::ClearBeats { clip: clip_id })) {
        Ok(ApplyOutcome::Edited(EditOutcome::Updated(_))) => {
            info!(%clip_id, "cleared beats");
            publish_projection(engine, ui);
        }
        Ok(other) => error!(%clip_id, "unexpected clear-beats outcome: {other:?}"),
        Err(e) => error!(%clip_id, "clear beats failed: {e}"),
    }
}

/// Set the project canvas settings (M1): aspect preset + background color
/// in one undoable history entry. An out-of-range preset index falls back
/// to auto (defensive — the dialog's list is index-aligned with the model).
fn set_canvas_and_publish(
    engine: &mut Engine,
    aspect_index: i32,
    background: [u8; 3],
    ui: &UiSink,
) {
    let aspect = usize::try_from(aspect_index)
        .ok()
        .and_then(|i| cutlass_models::CanvasAspect::ALL.get(i).copied())
        .unwrap_or_default();
    match engine.apply(Command::Edit(EditCommand::SetCanvas { aspect, background })) {
        Ok(_) => {
            info!(aspect = aspect.name(), ?background, "set canvas settings");
            publish_projection(engine, ui);
        }
        Err(e) => error!("set canvas failed: {e}"),
    }
}

/// Set a visual clip's crop window + mirroring (CapCut crop, M1). One
/// undoable history entry; the engine validates the rect and rejects
/// audio-lane clips, so a failure here just logs (the inspector only shows
/// crop controls for visual clips — a rejection is a stale-projection race).
fn set_clip_crop_and_publish(
    engine: &mut Engine,
    clip: &str,
    crop: CropRect,
    flip_h: bool,
    flip_v: bool,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-clip-crop ignored: unparsable clip id");
        return;
    };
    if let Err(e) = engine.apply(Command::Edit(EditCommand::SetClipCrop {
        clip: clip_id,
        crop,
        flip_h,
        flip_v,
    })) {
        error!(%clip_id, "set clip crop failed: {e}");
        return;
    }
    info!(
        %clip_id,
        x = crop.x, y = crop.y, w = crop.w, h = crop.h, flip_h, flip_v,
        "set clip crop"
    );
    publish_projection(engine, ui);
}

/// Append a catalog effect to a clip's chain (M4). One undoable entry; the
/// composite repaints because effects are visual.
fn add_effect_and_publish(engine: &mut Engine, clip: &str, effect_id: &str, ui: &UiSink) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "add-effect ignored: unparsable clip id");
        return;
    };
    if let Err(e) = engine.apply(Command::Edit(EditCommand::AddEffect {
        clip: clip_id,
        effect_id: effect_id.to_string(),
    })) {
        error!(%clip_id, effect_id, "add effect failed: {e}");
        return;
    }
    info!(%clip_id, effect_id, "added effect");
    publish_projection(engine, ui);
}

/// Remove the effect at `index` from a clip's chain (M4).
fn remove_effect_and_publish(engine: &mut Engine, clip: &str, index: u32, ui: &UiSink) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "remove-effect ignored: unparsable clip id");
        return;
    };
    if let Err(e) = engine.apply(Command::Edit(EditCommand::RemoveEffect {
        clip: clip_id,
        index: index as usize,
    })) {
        error!(%clip_id, index, "remove effect failed: {e}");
        return;
    }
    info!(%clip_id, index, "removed effect");
    publish_projection(engine, ui);
}

/// Set one effect parameter to a constant (M4). The inspector addresses the
/// parameter by its catalog name; resolve it to the uniform slot index the
/// command expects from the clip's current effect.
fn set_effect_param_and_publish(
    engine: &mut Engine,
    clip: &str,
    index: u32,
    param: &str,
    value: f32,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-effect-param ignored: unparsable clip id");
        return;
    };
    let slot = engine
        .project()
        .clip(clip_id)
        .and_then(|c| c.effects.get(index as usize))
        .and_then(|fx| cutlass_models::effect_spec(&fx.effect_id))
        .and_then(|spec| spec.params.iter().position(|p| p.name == param));
    let Some(slot) = slot else {
        error!(%clip_id, index, param, "set-effect-param ignored: unknown param");
        return;
    };
    if let Err(e) = engine.apply(Command::Edit(EditCommand::SetEffectParam {
        clip: clip_id,
        index: index as usize,
        param: slot,
        value,
    })) {
        error!(%clip_id, index, param, value, "set effect param failed: {e}");
        return;
    }
    info!(%clip_id, index, param, value, "set effect param");
    publish_projection(engine, ui);
}

/// Add a catalog transition at the junction after `clip` (M4). Requires a
/// right-neighbor clip that abuts; the engine rejects otherwise.
fn add_transition_and_publish(engine: &mut Engine, clip: &str, transition_id: &str, ui: &UiSink) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "add-transition ignored: unparsable clip id");
        return;
    };
    if let Err(e) = engine.apply(Command::Edit(EditCommand::AddTransition {
        clip: clip_id,
        transition_id: transition_id.to_string(),
    })) {
        error!(%clip_id, transition_id, "add transition failed: {e}");
        return;
    }
    info!(%clip_id, transition_id, "added transition");
    publish_projection(engine, ui);
}

/// Remove the transition at `clip`'s right junction (M4).
fn remove_transition_and_publish(engine: &mut Engine, clip: &str, ui: &UiSink) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "remove-transition ignored: unparsable clip id");
        return;
    };
    if let Err(e) = engine.apply(Command::Edit(EditCommand::RemoveTransition {
        clip: clip_id,
    })) {
        error!(%clip_id, "remove transition failed: {e}");
        return;
    }
    info!(%clip_id, "removed transition");
    publish_projection(engine, ui);
}

/// Set the window length (timeline ticks) of the transition after `clip` (M4).
fn set_transition_and_publish(engine: &mut Engine, clip: &str, duration: i64, ui: &UiSink) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-transition ignored: unparsable clip id");
        return;
    };
    if let Err(e) = engine.apply(Command::Edit(EditCommand::SetTransition {
        clip: clip_id,
        duration,
    })) {
        error!(%clip_id, duration, "set transition failed: {e}");
        return;
    }
    info!(%clip_id, duration, "set transition duration");
    publish_projection(engine, ui);
}

/// Drop a ruler marker (M1). Empty `color` cycles the palette; one undoable
/// history entry.
fn add_marker_and_publish(
    engine: &mut Engine,
    at_tick: i64,
    name: &str,
    color: &str,
    tl_rate: Rational,
    ui: &UiSink,
) {
    let at = RationalTime::new(at_tick.max(0), tl_rate);
    let color = parse_marker_color(color);
    match engine.apply(Command::Edit(EditCommand::AddMarker {
        at,
        name: name.to_string(),
        color,
    })) {
        Ok(ApplyOutcome::Edited(EditOutcome::CreatedMarker(id))) => {
            info!(%id, at_tick, "added timeline marker");
            publish_projection(engine, ui);
        }
        Ok(other) => error!(at_tick, "unexpected add-marker outcome: {other:?}"),
        Err(e) => error!(at_tick, "add marker failed: {e}"),
    }
}

/// Remove a ruler marker by raw id (M1). One undoable history entry.
fn remove_marker_and_publish(engine: &mut Engine, marker: &str, ui: &UiSink) {
    let Some(marker_id) = parse_raw_id(marker).map(MarkerId::from_raw) else {
        error!(marker, "remove-marker ignored: unparsable marker id");
        return;
    };
    match engine.apply(Command::Edit(EditCommand::RemoveMarker {
        marker: marker_id,
    })) {
        Ok(_) => {
            info!(%marker_id, "removed timeline marker");
            publish_projection(engine, ui);
        }
        Err(e) => error!(%marker_id, "remove marker failed: {e}"),
    }
}

fn parse_marker_color(name: &str) -> Option<MarkerColor> {
    match name {
        "teal" => Some(MarkerColor::Teal),
        "blue" => Some(MarkerColor::Blue),
        "purple" => Some(MarkerColor::Purple),
        "pink" => Some(MarkerColor::Pink),
        "red" => Some(MarkerColor::Red),
        "orange" => Some(MarkerColor::Orange),
        "yellow" => Some(MarkerColor::Yellow),
        "green" => Some(MarkerColor::Green),
        _ => None,
    }
}

/// Build a shape generator with new reference-pixel dimensions, preserving the
/// clip's shape kind and fill. `None` when the clip is missing or not a shape.
///
/// Dimensions are floored at 1px and non-finite input is rejected: the slider
/// stays in `8..=1920`, but a typed entry or double-click reset can deliver
/// anything, and a zero/negative extent would collapse the raster's `Rect` to
/// an invisible shape.
fn shape_size_from_engine(
    engine: &Engine,
    clip: &str,
    width: f32,
    height: f32,
) -> Option<Generator> {
    if !width.is_finite() || !height.is_finite() {
        return None;
    }
    let clip_id = parse_raw_id(clip).map(ClipId::from_raw)?;
    let generator = match &engine.project().timeline().clip(clip_id)?.content {
        ClipSource::Generated(g) => g,
        ClipSource::Media { .. } => return None,
    };
    match generator {
        // A slider commit sets an absolute size, so the animated params
        // collapse to constants (matching the pre-keyframe behavior); corner
        // rounding and stroke ride along untouched.
        Generator::Shape {
            shape,
            rgba,
            corner_radius,
            stroke,
            ..
        } => Some(Generator::Shape {
            shape: shape.clone(),
            rgba: rgba.clone(),
            width: Param::Constant(width.max(1.0)),
            height: Param::Constant(height.max(1.0)),
            corner_radius: corner_radius.clone(),
            stroke: stroke.clone(),
        }),
        _ => None,
    }
}

/// Replace a generated clip's content (inspector title edit). One history
/// entry per committed edit; the engine rejects non-generated clips.
fn set_generator_and_publish(engine: &mut Engine, clip: &str, generator: Generator, ui: &UiSink) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-generator ignored: unparsable clip id");
        return;
    };
    match engine.apply(Command::Edit(EditCommand::SetGenerator {
        clip: clip_id,
        generator,
    })) {
        Ok(_) => {
            info!(%clip_id, "updated generated clip content");
            publish_projection(engine, ui);
        }
        Err(e) => error!(%clip_id, "set generator failed: {e}"),
    }
}

/// Whether `[start, start + duration)` overlaps no clip on `track`.
fn span_free(track: &Track, start: i64, duration: i64) -> bool {
    let end = start + duration;
    track
        .clips_ordered()
        .iter()
        .all(|c| c.timeline.end_tick() <= start || c.timeline.start.value >= end)
}

/// `track` (raw id from the Slint projection) when it names an existing lane
/// of `kind`.
fn lane_of_kind(engine: &Engine, track: &str, kind: TrackKind) -> Option<TrackId> {
    let id = TrackId::from_raw(parse_raw_id(track)?);
    engine
        .project()
        .timeline()
        .track(id)
        .is_some_and(|t| t.kind == kind)
        .then_some(id)
}

/// Move a dragged clip to its resolved landing spot: an existing lane
/// (`track` set) or a new lane of the clip's kind inserted at `insert_row`.
/// A cross-lane move that empties its source lane removes that lane
/// (CapCut deletes overlay tracks that empty out). With `insert` (main-track
/// magnet) the landing is an insertion on the main lane; with the magnet on,
/// a move *off* the main lane also closes the gap it leaves. Every variant
/// is one history group, so one undo reverts the whole gesture.
#[allow(clippy::too_many_arguments)]
fn move_clip_and_publish(
    engine: &mut Engine,
    clip: &str,
    track: &str,
    insert_row: i64,
    start_tick: i64,
    insert: bool,
    main_magnet: bool,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "move ignored: unparsable clip id");
        return;
    };
    let Some(source_track) = engine.project().timeline().track_of(clip_id) else {
        error!(%clip_id, "move ignored: clip not on the timeline");
        return;
    };
    let kind = engine
        .project()
        .timeline()
        .track(source_track)
        .expect("track_of returned an existing track")
        .kind;
    let placed = engine
        .project()
        .clip(clip_id)
        .expect("track_of returned a placed clip")
        .timeline;
    let tl_rate = engine.project().timeline().frame_rate;
    // Decided before the gesture mutates anything: a new lane created below
    // the stack would become the bottom video lane and steal main status.
    let source_is_main = main_magnet && Some(source_track) == main_video_track(engine);

    if insert {
        // Main-track magnet: the resolver targets the existing main lane.
        let Some(to_track) = parse_raw_id(track).map(TrackId::from_raw) else {
            error!(%clip_id, track, "insert-move ignored: unparsable track id");
            return;
        };
        engine.begin_group();
        let result = if to_track == source_track {
            ripple_reorder(engine, clip_id, to_track, start_tick.max(0))
        } else {
            ripple_move_in(engine, clip_id, source_track, to_track, start_tick.max(0))
        };
        match result {
            Ok(()) => {
                engine.commit_group();
                info!(%clip_id, %to_track, start_tick, "ripple-inserted moved clip");
            }
            Err(e) => {
                error!(%clip_id, %to_track, start_tick, "insert move failed: {e}");
                engine.rollback_group();
            }
        }
        publish_projection(engine, ui);
        return;
    }

    // One history entry per move, including a created destination lane and a
    // removed emptied source lane.
    engine.begin_group();
    let to_track = match parse_raw_id(track).map(TrackId::from_raw) {
        Some(id) => id,
        None => match create_track(engine, kind, insert_row) {
            Ok(id) => id,
            Err(e) => {
                error!(%clip_id, "move failed creating {kind:?} track: {e}");
                engine.rollback_group();
                return;
            }
        },
    };

    match engine.apply(Command::Edit(EditCommand::MoveClip {
        clip: clip_id,
        to_track,
        start: RationalTime::new(start_tick.max(0), tl_rate),
    })) {
        Ok(ApplyOutcome::Edited(EditOutcome::Updated(_))) => {
            let mut completed = true;
            if source_track != to_track {
                // Leaving the main lane with the magnet on closes the gap
                // the clip vacated (CapCut ripple). Can't collide: the first
                // shifted clip lands exactly where the moved clip started.
                if source_is_main {
                    completed = apply_edit(
                        engine,
                        EditCommand::ShiftClips {
                            track: source_track,
                            from: placed.start,
                            delta: RationalTime::new(-placed.duration.value, tl_rate),
                        },
                    )
                    .map_err(|e| error!(%clip_id, "move failed closing main-lane gap: {e}"))
                    .is_ok();
                }
                if completed {
                    remove_track_if_empty(engine, source_track);
                }
            }
            if completed {
                engine.commit_group();
                info!(%clip_id, %to_track, start_tick, "moved clip");
            } else {
                engine.rollback_group();
            }
            publish_projection(engine, ui);
        }
        Ok(other) => {
            error!(%clip_id, "unexpected move-clip outcome: {other:?}");
            engine.rollback_group();
            publish_projection(engine, ui);
        }
        // The drag resolver previewed a valid spot; the engine still rejects
        // atomically if the projection raced a concurrent edit. Rolling back
        // removes a lane this move just created.
        Err(e) => {
            error!(%clip_id, %to_track, start_tick, "move clip failed: {e}");
            engine.rollback_group();
            publish_projection(engine, ui);
        }
    }
}

/// Reorder within the main lane as one group of four commands: park the clip
/// past the lane's content end (never rendered — the projection publishes
/// only after the group resolves), close its old gap, open the new hole at
/// `at` (post-close space, straight from the drag resolver), and land in it.
fn ripple_reorder(
    engine: &mut Engine,
    clip_id: ClipId,
    track: TrackId,
    at: i64,
) -> Result<(), String> {
    let tl_rate = engine.project().timeline().frame_rate;
    let placed = engine
        .project()
        .clip(clip_id)
        .ok_or("clip not on the timeline")?
        .timeline;
    let duration = placed.duration.value;
    let park = engine
        .project()
        .timeline()
        .track(track)
        .ok_or("main lane missing")?
        .content_end();

    apply_edit(
        engine,
        EditCommand::MoveClip {
            clip: clip_id,
            to_track: track,
            start: RationalTime::new(park, tl_rate),
        },
    )?;
    // Both shifts also carry the parked clip along (its start stays past the
    // rest of the lane), so it never collides with the clips in between.
    apply_edit(
        engine,
        EditCommand::ShiftClips {
            track,
            from: placed.start,
            delta: RationalTime::new(-duration, tl_rate),
        },
    )?;
    apply_edit(
        engine,
        EditCommand::ShiftClips {
            track,
            from: RationalTime::new(at, tl_rate),
            delta: RationalTime::new(duration, tl_rate),
        },
    )?;
    apply_edit(
        engine,
        EditCommand::MoveClip {
            clip: clip_id,
            to_track: track,
            start: RationalTime::new(at, tl_rate),
        },
    )
}

/// Cross-lane move onto the main lane: open the hole at `at`, move the clip
/// in, and drop the source lane when this emptied it (same overlay policy as
/// freeform moves).
fn ripple_move_in(
    engine: &mut Engine,
    clip_id: ClipId,
    source_track: TrackId,
    to_track: TrackId,
    at: i64,
) -> Result<(), String> {
    let tl_rate = engine.project().timeline().frame_rate;
    let duration = engine
        .project()
        .clip(clip_id)
        .ok_or("clip not on the timeline")?
        .timeline
        .duration
        .value;

    apply_edit(
        engine,
        EditCommand::ShiftClips {
            track: to_track,
            from: RationalTime::new(at, tl_rate),
            delta: RationalTime::new(duration, tl_rate),
        },
    )?;
    apply_edit(
        engine,
        EditCommand::MoveClip {
            clip: clip_id,
            to_track,
            start: RationalTime::new(at, tl_rate),
        },
    )?;
    remove_track_if_empty(engine, source_track);
    Ok(())
}

/// Re-place a trimmed clip at its resolved extent. The trim resolver already
/// clamped to neighbors and source headroom, so this should always apply; the
/// engine still validates atomically (overlap, source bounds) and we surface
/// any rejection rather than mutating the projection optimistically.
///
/// With `linkage` on, the same edge delta applies to every clip in the
/// trimmed clip's link group (the resolver intersected the clamps, so the
/// partners' extents are valid too) — one history entry for the group.
///
/// With the main-track magnet on and the trim touching the main lane, the
/// trim *ripples* instead of leaving/eating a gap: downstream clips follow
/// the dragged edge (timeline roadmap Phase 7's deliberate gap). See
/// [`commit_trims`]; still one history entry — a single undo restores the
/// trim and every shifted clip.
fn trim_clip_and_publish(
    engine: &mut Engine,
    clip: &str,
    start_tick: i64,
    duration_ticks: i64,
    linkage: bool,
    main_magnet: bool,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "trim ignored: unparsable clip id");
        return;
    };
    let Some(placed) = engine.project().clip(clip_id).map(|c| c.timeline) else {
        error!(%clip_id, "trim ignored: clip not on the timeline");
        return;
    };
    let tl_rate = engine.project().timeline().frame_rate;
    let start = start_tick.max(0);
    let duration = duration_ticks.max(1);
    // The same edge motion, expressed as deltas the partners can replay.
    let delta_start = start - placed.start.value;
    let delta_duration = duration - placed.duration.value;

    let mut trims = vec![(clip_id, TimeRange::at_rate(start, duration, tl_rate))];
    if linkage {
        for partner in link_group_ids(engine, clip_id) {
            if partner == clip_id {
                continue;
            }
            let Some(extent) = engine.project().clip(partner).map(|c| c.timeline) else {
                continue;
            };
            trims.push((
                partner,
                TimeRange::at_rate(
                    extent.start.value + delta_start,
                    (extent.duration.value + delta_duration).max(1),
                    tl_rate,
                ),
            ));
        }
    }

    match commit_trims(engine, &trims, main_magnet) {
        Ok(ripple) => info!(%clip_id, start_tick, duration_ticks, linkage, ripple, "trimmed clip"),
        Err(e) => error!(%clip_id, "trim clip failed: {e}"),
    }
    publish_projection(engine, ui);
}

/// Apply a resolved set of member trims as one history group.
///
/// With the main-track magnet on and any member sitting on the main lane
/// (dragging the audio half of a linked pair must still keep the main lane
/// gapless), every member's trim ripples on its own lane — linked pairs and
/// their downstream neighbors all shift by the same duration delta, so
/// cross-lane alignment survives. Otherwise members get plain `TrimClip`s.
///
/// A rejected step rolls the whole group back — no half-applied ripple.
/// Returns whether the group rippled.
fn commit_trims(
    engine: &mut Engine,
    trims: &[(ClipId, TimeRange)],
    main_magnet: bool,
) -> Result<bool, String> {
    let main = main_video_track(engine);
    let ripple = main_magnet
        && main.is_some_and(|m| {
            trims
                .iter()
                .any(|&(id, _)| engine.project().timeline().track_of(id) == Some(m))
        });

    engine.begin_group();
    for &(id, timeline) in trims {
        let result = if ripple {
            apply_ripple_trim(engine, id, timeline)
        } else {
            apply_edit(engine, EditCommand::TrimClip { clip: id, timeline })
        };
        if let Err(e) = result {
            engine.rollback_group();
            return Err(format!("clip {id}: {e}"));
        }
    }
    engine.commit_group();
    Ok(ripple)
}

/// One member's ripple trim: `TrimClip` + `ShiftClips` composed on the
/// member's own lane, ordered so the engine's atomic validation accepts the
/// intermediate state (open room before growing into it, trim before
/// closing the gap behind a shrink).
///
/// Semantics (CapCut): the trimmed clip stays anchored at its old start and
/// every downstream clip shifts by the duration delta — the lane neither
/// leaves nor eats a gap.
/// - Trailing edge: only the duration changes; downstream (clips starting at
///   or after the old end) shifts by the delta.
/// - Leading edge: the resolved extent moves the start — that start delta is
///   what the engine derives the new source in-point from — and the shift
///   then re-anchors the clip at its old start, carrying downstream along.
///   A leading grow shifts everything from the old start right first, then
///   trims anchored there, which yields the same negative start delta.
///
/// The caller wraps members in one history group and rolls back on error,
/// so a rejected step never leaves a half-applied ripple.
fn apply_ripple_trim(engine: &mut Engine, clip: ClipId, timeline: TimeRange) -> Result<(), String> {
    let Some(old) = engine.project().clip(clip).map(|c| c.timeline) else {
        return Err("clip is not on the timeline".into());
    };
    let Some(track) = engine.project().timeline().track_of(clip) else {
        return Err("clip has no track".into());
    };
    let tl_rate = engine.project().timeline().frame_rate;
    let delta_dur = timeline.duration.value - old.duration.value;
    let trim = EditCommand::TrimClip { clip, timeline };

    if timeline.start.value != old.start.value {
        // Leading edge (the resolver anchors the end, so the start moved).
        if delta_dur > 0 {
            // Grow: open room first (the clip and everything after it move
            // right), then trim anchored at the old start.
            apply_edit(
                engine,
                EditCommand::ShiftClips {
                    track,
                    from: old.start,
                    delta: RationalTime::new(delta_dur, tl_rate),
                },
            )?;
            apply_edit(
                engine,
                EditCommand::TrimClip {
                    clip,
                    timeline: TimeRange::at_rate(old.start.value, timeline.duration.value, tl_rate),
                },
            )
        } else {
            // Shrink: trim to the resolved extent (a gap opens at the old
            // start), then slide the clip and downstream left into it.
            apply_edit(engine, trim)?;
            apply_edit(
                engine,
                EditCommand::ShiftClips {
                    track,
                    from: timeline.start,
                    delta: RationalTime::new(old.start.value - timeline.start.value, tl_rate),
                },
            )
        }
    } else if delta_dur > 0 {
        // Trailing grow: push downstream right, then extend into the hole.
        apply_edit(
            engine,
            EditCommand::ShiftClips {
                track,
                from: RationalTime::new(old.end_tick(), tl_rate),
                delta: RationalTime::new(delta_dur, tl_rate),
            },
        )?;
        apply_edit(engine, trim)
    } else if delta_dur < 0 {
        // Trailing shrink: pull the edge in, then close the gap behind it.
        apply_edit(engine, trim)?;
        apply_edit(
            engine,
            EditCommand::ShiftClips {
                track,
                from: RationalTime::new(old.end_tick(), tl_rate),
                delta: RationalTime::new(delta_dur, tl_rate),
            },
        )
    } else {
        // No edge moved (defensive — the UI skips noop trims).
        apply_edit(engine, trim)
    }
}

/// Every clip sharing `clip`'s link group (including itself); just the clip
/// when it's unlinked. O(total clips) — cold per-gesture path.
fn link_group_ids(engine: &Engine, clip: ClipId) -> Vec<ClipId> {
    let Some(link) = engine.project().clip(clip).and_then(|c| c.link) else {
        return vec![clip];
    };
    engine
        .project()
        .timeline()
        .tracks_ordered()
        .flat_map(|t| t.clips_ordered())
        .filter(|c| c.link == Some(link))
        .map(|c| c.id)
        .collect()
}

/// Toggle a track header flag (hide/mute/lock). Undoable like any edit; the
/// republished projection carries the new flag to the lane header. Disabling
/// a visual track drops it from the composite (the engine skips `!enabled`
/// visual tracks), so the preview catches up on the next scrub.
fn set_track_flag_and_publish(
    engine: &mut Engine,
    track: &str,
    flag: TrackFlag,
    value: bool,
    ui: &UiSink,
) {
    let Some(track_id) = parse_raw_id(track).map(TrackId::from_raw) else {
        error!(track, "set-track-flag ignored: unparsable track id");
        return;
    };
    let command = match flag {
        TrackFlag::Enabled => EditCommand::SetTrackEnabled {
            track: track_id,
            enabled: value,
        },
        TrackFlag::Muted => EditCommand::SetTrackMuted {
            track: track_id,
            muted: value,
        },
        TrackFlag::Locked => EditCommand::SetTrackLocked {
            track: track_id,
            locked: value,
        },
        TrackFlag::DuckSource => EditCommand::SetTrackDuckSource {
            track: track_id,
            duck_source: value,
        },
    };
    match engine.apply(Command::Edit(command)) {
        Ok(ApplyOutcome::Edited(EditOutcome::UpdatedTrack(_))) => {
            info!(%track_id, value, "set track flag");
            publish_projection(engine, ui);
        }
        Ok(other) => error!(%track_id, "unexpected set-track-flag outcome: {other:?}"),
        Err(e) => error!(%track_id, "set track flag failed: {e}"),
    }
}

// PORT (Phase 5): main ran the export job here — a dedicated thread cloning
// the project, rendering through the engine's export path, and publishing
// progress/cancel state to the Slint `ExportBackend`. It returns on this
// branch's `cutlass_render::export_to_file_observed` (the mobile
// `export_job.rs` pattern).

/// Remove every clip in `clips`; lanes the removals empty are removed with
/// them (CapCut deletes emptied overlay tracks — same policy the drag-moves
/// use). With the main-track magnet on, main-lane deletions ripple their
/// gaps closed. Everything forms one history group: one undo restores the
/// whole selection.
fn remove_clips_and_publish(engine: &mut Engine, clips: &[String], main_magnet: bool, ui: &UiSink) {
    let main = main_video_track(engine);
    // Resolve every member up front: a single bad id voids the whole batch
    // rather than half-deleting the selection.
    let mut targets = Vec::with_capacity(clips.len());
    for clip in clips {
        let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
            error!(clip, "delete ignored: unparsable clip id");
            return;
        };
        let Some(track) = engine.project().timeline().track_of(clip_id) else {
            error!(%clip_id, "delete ignored: clip not on the timeline");
            return;
        };
        targets.push((clip_id, track));
    }
    if targets.is_empty() {
        return;
    }
    // Ripple deletes shift later main-lane clips left; deleting right-to-left
    // keeps each pending member's recorded position valid.
    targets.sort_by_key(|(clip_id, _)| {
        std::cmp::Reverse(
            engine
                .project()
                .clip(*clip_id)
                .map(|c| c.timeline.start.value)
                .unwrap_or(0),
        )
    });

    engine.begin_group();
    for &(clip_id, track) in &targets {
        let command = if main_magnet && Some(track) == main {
            EditCommand::RippleDelete { clip: clip_id }
        } else {
            EditCommand::RemoveClip { clip: clip_id }
        };
        if let Err(e) = apply_edit(engine, command) {
            error!(%clip_id, "remove clip failed: {e}");
            engine.rollback_group();
            publish_projection(engine, ui);
            return;
        }
    }
    // Lane cleanup after all removals: dedupe so each lane is checked once.
    let mut lanes: Vec<TrackId> = targets.iter().map(|&(_, track)| track).collect();
    lanes.sort();
    lanes.dedup();
    for lane in lanes {
        remove_track_if_empty(engine, lane);
    }
    engine.commit_group();
    info!(count = targets.len(), "removed clips");
    publish_projection(engine, ui);
}

/// Split a clip into two abutting clips at `at_tick`. The UI only offers the
/// split while the playhead is strictly inside the clip; the engine still
/// validates the position atomically.
///
/// With `linkage` on, every linked partner that also spans `at_tick` splits
/// at the same tick, and the resulting tails are linked into a fresh group
/// (heads keep the original link) — one history entry for the lot.
fn split_clip_and_publish(
    engine: &mut Engine,
    clip: &str,
    at_tick: i64,
    linkage: bool,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "split ignored: unparsable clip id");
        return;
    };
    let tl_rate = engine.project().timeline().frame_rate;
    let at = RationalTime::new(at_tick, tl_rate);

    // Partners split only where the tick is strictly inside their extent
    // (linked clips can have different lengths after asymmetric edits).
    let members: Vec<ClipId> = if linkage {
        link_group_ids(engine, clip_id)
            .into_iter()
            .filter(|&id| {
                engine.project().clip(id).is_some_and(|c| {
                    at_tick > c.timeline.start.value && at_tick < c.timeline.end_tick()
                })
            })
            .collect()
    } else {
        vec![clip_id]
    };
    if members.is_empty() {
        error!(%clip_id, at_tick, "split ignored: tick outside the clip");
        return;
    }

    engine.begin_group();
    let mut tails = Vec::with_capacity(members.len());
    for member in &members {
        match engine.apply(Command::Edit(EditCommand::SplitClip { clip: *member, at })) {
            Ok(ApplyOutcome::Edited(EditOutcome::Created(tail))) => tails.push(tail),
            Ok(other) => {
                error!(%member, "unexpected split-clip outcome: {other:?}");
                engine.rollback_group();
                return;
            }
            Err(e) => {
                error!(%member, at_tick, "split clip failed: {e}");
                engine.rollback_group();
                return;
            }
        }
    }
    // Tails are born unlinked (split copies content, not links); pair them
    // back up so each half keeps moving as a unit.
    if tails.len() > 1
        && let Err(e) = apply_edit(
            engine,
            EditCommand::LinkClips {
                clips: tails.clone(),
            },
        )
    {
        error!(%clip_id, "split failed linking tails: {e}");
        engine.rollback_group();
        return;
    }
    engine.commit_group();
    info!(%clip_id, ?tails, at_tick, "split clip");
    publish_projection(engine, ui);
}

/// Land a group-drag batch. The resolver already validated every member
/// against everything outside the selection, but members can still collide
/// with *each other's* old positions mid-batch, so the batch goes
/// park-then-place: every member first parks past the global content end on
/// its target lane, then lands on its resolved start. One history group —
/// one undo reverts the whole gesture. Source lanes the moves empty are
/// removed (same overlay policy as single moves). Group moves are freeform —
/// the main-track magnet's ripple-insert applies to single-clip drags only.
fn move_group_and_publish(engine: &mut Engine, moves: &[GroupMove], ui: &UiSink) {
    // Resolve raw ids up front; any stale entry voids the batch.
    let mut resolved = Vec::with_capacity(moves.len());
    for entry in moves {
        let Some(clip_id) = parse_raw_id(&entry.clip).map(ClipId::from_raw) else {
            error!(clip = entry.clip, "group move ignored: unparsable clip id");
            return;
        };
        let Some(to_track) = parse_raw_id(&entry.track).map(TrackId::from_raw) else {
            error!(
                track = entry.track,
                "group move ignored: unparsable track id"
            );
            return;
        };
        let Some(source_track) = engine.project().timeline().track_of(clip_id) else {
            error!(%clip_id, "group move ignored: clip not on the timeline");
            return;
        };
        resolved.push((clip_id, to_track, source_track, entry.start_tick.max(0)));
    }
    if resolved.is_empty() {
        return;
    }
    let tl_rate = engine.project().timeline().frame_rate;
    // Parking starts past everything on any lane; spaced by each member's
    // duration so parked members can't collide either.
    let mut park = engine
        .project()
        .timeline()
        .tracks_ordered()
        .map(|t| t.content_end())
        .max()
        .unwrap_or(0);

    engine.begin_group();
    for &(clip_id, to_track, _, _) in &resolved {
        let duration = engine
            .project()
            .clip(clip_id)
            .map(|c| c.timeline.duration.value)
            .unwrap_or(1);
        if let Err(e) = apply_edit(
            engine,
            EditCommand::MoveClip {
                clip: clip_id,
                to_track,
                start: RationalTime::new(park, tl_rate),
            },
        ) {
            error!(%clip_id, %to_track, "group move failed parking: {e}");
            engine.rollback_group();
            publish_projection(engine, ui);
            return;
        }
        park += duration;
    }
    for &(clip_id, to_track, _, start_tick) in &resolved {
        if let Err(e) = apply_edit(
            engine,
            EditCommand::MoveClip {
                clip: clip_id,
                to_track,
                start: RationalTime::new(start_tick, tl_rate),
            },
        ) {
            error!(%clip_id, %to_track, start_tick, "group move failed landing: {e}");
            engine.rollback_group();
            publish_projection(engine, ui);
            return;
        }
    }
    // Lane cleanup after all landings (dedupe: one check per source lane).
    let mut sources: Vec<TrackId> = resolved.iter().map(|&(_, _, source, _)| source).collect();
    sources.sort();
    sources.dedup();
    for source in sources {
        remove_track_if_empty(engine, source);
    }
    engine.commit_group();
    info!(count = resolved.len(), "moved clip group");
    publish_projection(engine, ui);
}

/// Step the engine history (`redo == false` ⇒ undo). Publishes even on a
/// no-op so the UI's can-undo / can-redo flags stay honest.
fn history_step_and_publish(engine: &mut Engine, redo: bool, ui: &UiSink) {
    let stepped = if redo { engine.redo() } else { engine.undo() };
    info!(redo, stepped, "history step");
    publish_projection(engine, ui);
}

/// Snapshot `clips` (raw ids — the selection) as one clipboard block:
/// members in start order, offsets rebased to the earliest start. Returns
/// the block origin (that earliest start) alongside, for callers that place
/// relative to the originals (duplicate). Ids that no longer resolve are
/// skipped; an empty result is `None`.
fn snapshot_block(engine: &Engine, clips: &[String]) -> Option<(i64, Vec<ClipboardClip>)> {
    let timeline = engine.project().timeline();
    let mut members = Vec::with_capacity(clips.len());
    for raw in clips {
        let Some(clip_id) = parse_raw_id(raw).map(ClipId::from_raw) else {
            continue;
        };
        let Some(track) = timeline.track_of(clip_id) else {
            continue;
        };
        let Some(kind) = timeline.track(track).map(|t| t.kind) else {
            continue;
        };
        let Some(clip) = engine.project().clip(clip_id) else {
            continue;
        };
        members.push(ClipboardClip {
            track,
            kind,
            content: clip.content.clone(),
            duration_ticks: clip.timeline.duration.value,
            // Absolute start for now; rebased to the block origin below.
            offset_ticks: clip.timeline.start.value,
            link: clip.link,
        });
    }
    if members.is_empty() {
        return None;
    }
    members.sort_by_key(|m| m.offset_ticks);
    let origin = members[0].offset_ticks;
    for member in &mut members {
        member.offset_ticks -= origin;
    }
    Some((origin, members))
}

/// Smallest uniform right-shift (≥ 0) that lets every `(lane, start,
/// duration)` span land without overlapping existing clips. Members can't
/// collide with each other (a uniform shift preserves their relative,
/// originally disjoint placement), so only 0 and the "blocked member
/// becomes left-flush against an existing clip's end" shifts can be the
/// minimum — the group analogue of `first_fit_start`'s gap scan. O(n·m)
/// per candidate on this cold, user-triggered path.
fn block_fit_dx(engine: &Engine, spans: &[(TrackId, i64, i64)]) -> i64 {
    let timeline = engine.project().timeline();
    let fits = |dx: i64| {
        spans.iter().all(|&(track, start, duration)| {
            timeline
                .track(track)
                .is_some_and(|t| span_free(t, start + dx, duration))
        })
    };
    let mut candidates: Vec<i64> = vec![0];
    for &(track, start, _) in spans {
        let Some(track) = timeline.track(track) else {
            continue;
        };
        for clip in track.clips_ordered() {
            let dx = clip.timeline.end_tick() - start;
            if dx > 0 {
                candidates.push(dx);
            }
        }
    }
    candidates.sort_unstable();
    candidates.dedup();
    // The largest candidate parks every member at/after the last clip on
    // its lane, so a fit always exists; 0 covers the all-lanes-empty case.
    candidates.into_iter().find(|&dx| fits(dx)).unwrap_or(0)
}

/// Place every member of a resolved block — `(landing lane, desired start,
/// member)` — inside the caller's open history group: one uniform
/// right-shift until everything fits, then re-issue each member's content
/// and re-link copies whose originals shared a link group (singleton
/// leftovers of partially copied groups stay unlinked).
fn place_block(
    engine: &mut Engine,
    members: &[(TrackId, i64, &ClipboardClip)],
) -> Result<(), String> {
    let spans: Vec<(TrackId, i64, i64)> = members
        .iter()
        .map(|&(track, start, member)| (track, start, member.duration_ticks.max(1)))
        .collect();
    let dx = block_fit_dx(engine, &spans);

    let mut created: Vec<(Option<LinkId>, ClipId)> = Vec::with_capacity(members.len());
    for &(track, start, member) in members {
        let id = add_clip_content(
            engine,
            track,
            &member.content,
            member.duration_ticks,
            start + dx,
        )?;
        created.push((member.link, id));
    }

    let mut seen: Vec<LinkId> = Vec::new();
    for &(link, _) in &created {
        let Some(link) = link else { continue };
        if seen.contains(&link) {
            continue;
        }
        seen.push(link);
        let group: Vec<ClipId> = created
            .iter()
            .filter(|(l, _)| *l == Some(link))
            .map(|&(_, id)| id)
            .collect();
        if group.len() >= 2 {
            apply_edit(engine, EditCommand::LinkClips { clips: group })?;
        }
    }
    Ok(())
}

/// Paste the clipboard block at `tick`: members land on the lanes they were
/// copied from (recreated by kind when gone), keeping relative placement;
/// the whole block slides right as one unit until every member fits — the
/// group analogue of the library-drop policy. A single-member block keeps
/// the magnet behavior: pasted on the main lane with the magnet on, it
/// ripple-inserts at the clip boundary nearest `tick` instead (groups stay
/// freeform, same policy as group drags).
fn paste_and_publish(
    engine: &mut Engine,
    block: &[ClipboardClip],
    tick: i64,
    main_magnet: bool,
    ui: &UiSink,
) {
    let tl_rate = engine.project().timeline().frame_rate;

    // One history entry per paste, even when it recreates copied lanes.
    engine.begin_group();

    // Landing lane per source lane: the original when it still exists, one
    // fresh lane of its kind (top of the stack, as single-paste always did)
    // per vanished track id.
    let mut lanes: HashMap<TrackId, TrackId> = HashMap::new();
    for member in block {
        if lanes.contains_key(&member.track) {
            continue;
        }
        let landing = if engine.project().timeline().track(member.track).is_some() {
            member.track
        } else {
            match create_track(engine, member.kind, 0) {
                Ok(id) => id,
                Err(e) => {
                    error!("paste failed creating {:?} track: {e}", member.kind);
                    engine.rollback_group();
                    return;
                }
            }
        };
        lanes.insert(member.track, landing);
    }

    // Single-clip ripple-insert (magnet) keeps its dedicated path.
    if let [only] = block {
        let track = lanes[&only.track];
        if main_magnet && Some(track) == main_video_track(engine) {
            let duration = only.duration_ticks.max(1);
            let lane = engine
                .project()
                .timeline()
                .track(track)
                .expect("paste target track exists");
            let start = nearest_boundary(lane, tick.max(0));
            let result = apply_edit(
                engine,
                EditCommand::ShiftClips {
                    track,
                    from: RationalTime::new(start, tl_rate),
                    delta: RationalTime::new(duration, tl_rate),
                },
            )
            .and_then(|_| {
                add_clip_content(engine, track, &only.content, only.duration_ticks, start)
            });
            match result {
                Ok(clip_id) => {
                    engine.commit_group();
                    info!(%clip_id, %track, start_tick = start, "ripple-pasted clip");
                }
                Err(e) => {
                    error!(%track, start_tick = start, "paste failed: {e}");
                    engine.rollback_group();
                }
            }
            publish_projection(engine, ui);
            return;
        }
    }

    let members: Vec<(TrackId, i64, &ClipboardClip)> = block
        .iter()
        .map(|member| {
            (
                lanes[&member.track],
                tick.max(0) + member.offset_ticks,
                member,
            )
        })
        .collect();
    match place_block(engine, &members) {
        Ok(()) => {
            engine.commit_group();
            info!(count = block.len(), tick, "pasted clipboard block");
        }
        // Rolling back also removes lanes this paste just recreated.
        Err(e) => {
            error!(tick, "paste failed: {e}");
            engine.rollback_group();
        }
    }
    publish_projection(engine, ui);
}

/// Duplicate the selection as one block: copies keep their lanes and
/// relative placement, landing right after the block's end — slid further
/// right as one unit when something is in the way. Copies of linked members
/// re-link as fresh groups; one history entry for everything. Freeform like
/// group drags (no group ripple-insert) — a single clip keeps the
/// magnet-aware single-duplicate path below.
fn duplicate_clips_and_publish(
    engine: &mut Engine,
    clips: &[String],
    main_magnet: bool,
    ui: &UiSink,
) {
    if let [only] = clips {
        duplicate_clip_and_publish(engine, only, main_magnet, ui);
        return;
    }
    let Some((origin, block)) = snapshot_block(engine, clips) else {
        info!("duplicate ignored: no valid clips in selection");
        return;
    };
    let span = block
        .iter()
        .map(|m| m.offset_ticks + m.duration_ticks.max(1))
        .max()
        .unwrap_or(1);
    // Copies land right after the originals' span; lanes all exist (the
    // originals are live), so no lane resolution is needed.
    let base = origin + span;
    let members: Vec<(TrackId, i64, &ClipboardClip)> = block
        .iter()
        .map(|member| (member.track, base + member.offset_ticks, member))
        .collect();

    engine.begin_group();
    match place_block(engine, &members) {
        Ok(()) => {
            engine.commit_group();
            info!(count = block.len(), "duplicated clip block");
        }
        Err(e) => {
            error!("duplicate failed: {e}");
            engine.rollback_group();
        }
    }
    publish_projection(engine, ui);
}

/// Dissolve the link groups of `clips` (raw ids): every member of every
/// touched group — selected or not — ends up unlinked. Implemented with the
/// existing `LinkClips` command by giving each member a fresh *singleton*
/// group, which behaves exactly like no link everywhere links are read
/// (selection expansion, linked trims/splits, drops). One history entry;
/// undo restores the old groups (the link action snapshots prior values).
/// A dedicated `UnlinkClips` (link = None) can replace the singleton trick
/// once the command surface is open again post-M1.
fn unlink_clips_and_publish(engine: &mut Engine, clips: &[String], ui: &UiSink) {
    // Link ids represented in the selection…
    let mut links: Vec<LinkId> = Vec::new();
    for raw in clips {
        let Some(clip_id) = parse_raw_id(raw).map(ClipId::from_raw) else {
            continue;
        };
        if let Some(link) = engine.project().clip(clip_id).and_then(|c| c.link)
            && !links.contains(&link)
        {
            links.push(link);
        }
    }
    if links.is_empty() {
        info!("unlink ignored: selection has no linked clips");
        return;
    }
    // …expanded to full membership, so groups dissolve as a whole.
    let members: Vec<ClipId> = engine
        .project()
        .timeline()
        .tracks_ordered()
        .flat_map(|t| t.clips_ordered())
        .filter(|c| c.link.is_some_and(|l| links.contains(&l)))
        .map(|c| c.id)
        .collect();

    engine.begin_group();
    for member in &members {
        if let Err(e) = apply_edit(
            engine,
            EditCommand::LinkClips {
                clips: vec![*member],
            },
        ) {
            error!(%member, "unlink failed: {e}");
            engine.rollback_group();
            publish_projection(engine, ui);
            return;
        }
    }
    engine.commit_group();
    info!(
        groups = links.len(),
        members = members.len(),
        "unlinked clip groups"
    );
    publish_projection(engine, ui);
}

/// Place a copy of `clip` immediately after it on its own lane (first gap
/// that fits from the clip's end). With the main-track magnet on, a main-lane
/// duplicate ripple-inserts right after the original, shifting later clips.
fn duplicate_clip_and_publish(engine: &mut Engine, clip: &str, main_magnet: bool, ui: &UiSink) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "duplicate ignored: unparsable clip id");
        return;
    };
    let Some(track) = engine.project().timeline().track_of(clip_id) else {
        error!(%clip_id, "duplicate ignored: clip not on the timeline");
        return;
    };
    let original = engine
        .project()
        .clip(clip_id)
        .expect("track_of returned a placed clip");
    let content = original.content.clone();
    let duration_ticks = original.timeline.duration.value.max(1);
    let end_tick = original.timeline.end_tick();
    let tl_rate = engine.project().timeline().frame_rate;

    if main_magnet && Some(track) == main_video_track(engine) {
        // Open a hole right after the original, land the copy in it — one
        // history entry for the pair.
        engine.begin_group();
        let result = apply_edit(
            engine,
            EditCommand::ShiftClips {
                track,
                from: RationalTime::new(end_tick, tl_rate),
                delta: RationalTime::new(duration_ticks, tl_rate),
            },
        )
        .and_then(|_| add_clip_content(engine, track, &content, duration_ticks, end_tick));
        match result {
            Ok(copy_id) => {
                engine.commit_group();
                info!(%clip_id, %copy_id, %track, start_tick = end_tick, "ripple-duplicated clip");
            }
            Err(e) => {
                error!(%clip_id, start_tick = end_tick, "duplicate failed: {e}");
                engine.rollback_group();
            }
        }
        publish_projection(engine, ui);
        return;
    }

    let lane = engine
        .project()
        .timeline()
        .track(track)
        .expect("track_of returned an existing track");
    let start = first_fit_start(lane, end_tick, duration_ticks);

    match add_clip_content(engine, track, &content, duration_ticks, start) {
        Ok(copy_id) => {
            info!(%clip_id, %copy_id, %track, start_tick = start, "duplicated clip");
            publish_projection(engine, ui);
        }
        Err(e) => error!(%clip_id, start_tick = start, "duplicate failed: {e}"),
    }
}

/// Close every gap on the main lane, including leading space before the
/// first clip — CapCut's lane is gapless the moment the magnet turns on.
/// One history group: a single undo restores the gaps.
fn pack_main_track_and_publish(engine: &mut Engine, ui: &UiSink) {
    let Some(track) = main_video_track(engine) else {
        return;
    };
    let tl_rate = engine.project().timeline().frame_rate;
    // (start, duration) snapshot in start order. Each shift slides the whole
    // suffix left, so positions after it are tracked via the running offset
    // instead of re-reading the engine.
    let clips: Vec<(i64, i64)> = engine
        .project()
        .timeline()
        .track(track)
        .map(|t| {
            t.clips_ordered()
                .iter()
                .map(|c| (c.timeline.start.value, c.timeline.duration.value))
                .collect()
        })
        .unwrap_or_default();

    let mut shifted_so_far = 0;
    let mut expected = 0;
    engine.begin_group();
    for (start, duration) in clips {
        let current = start - shifted_so_far;
        if current > expected {
            if let Err(e) = apply_edit(
                engine,
                EditCommand::ShiftClips {
                    track,
                    from: RationalTime::new(current, tl_rate),
                    delta: RationalTime::new(expected - current, tl_rate),
                },
            ) {
                error!(%track, "magnet pack failed: {e}");
                engine.rollback_group();
                publish_projection(engine, ui);
                return;
            }
            shifted_so_far += current - expected;
        }
        expected += duration;
    }
    // An already-packed lane records nothing (empty groups are dropped).
    engine.commit_group();
    publish_projection(engine, ui);
}

/// The main track under CapCut's magnet: the *bottom* video lane (the engine
/// stacks bottom→top, so the first video track in stack order).
fn main_video_track(engine: &Engine) -> Option<TrackId> {
    let timeline = engine.project().timeline();
    timeline.order().iter().copied().find(|id| {
        timeline
            .track(*id)
            .is_some_and(|t| t.kind == TrackKind::Video)
    })
}

/// Clip boundary on `track` nearest to `tick`: every clip start plus the
/// content end (0 on an empty lane). Ties resolve to the earlier boundary.
fn nearest_boundary(track: &Track, tick: i64) -> i64 {
    let mut best = 0;
    let mut best_distance = i64::MAX;
    let mut consider = |boundary: i64| {
        let distance = (tick - boundary).abs();
        if distance < best_distance {
            best = boundary;
            best_distance = distance;
        }
    };
    for clip in track.clips_ordered() {
        consider(clip.timeline.start.value);
    }
    consider(track.content_end());
    best
}

// PORT (later): main's `agent_replay` lived here — the AI agent's rehearsed
// plan replayed on the live engine as one history group, with sandbox ids
// remapped. It returns with the cutlass-ai port.

/// Apply a single edit command, flattening the outcome — for compositions
/// where only success/failure matters (the group publishes once at the end).
fn apply_edit(engine: &mut Engine, command: EditCommand) -> Result<(), String> {
    engine
        .apply(Command::Edit(command))
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// Re-issue snapshotted clip content as a fresh engine command: `AddClip`
/// for media-backed content, `AddGenerated` for generated content.
fn add_clip_content(
    engine: &mut Engine,
    track: TrackId,
    content: &ClipSource,
    duration_ticks: i64,
    start_tick: i64,
) -> Result<ClipId, String> {
    let tl_rate = engine.project().timeline().frame_rate;
    let command = match content {
        ClipSource::Media { media, source } => EditCommand::AddClip {
            track,
            media: *media,
            source: *source,
            start: RationalTime::new(start_tick, tl_rate),
        },
        ClipSource::Generated(generator) => EditCommand::AddGenerated {
            track,
            generator: generator.clone(),
            timeline: TimeRange::at_rate(start_tick, duration_ticks.max(1), tl_rate),
        },
    };
    match engine.apply(Command::Edit(command)) {
        Ok(ApplyOutcome::Edited(EditOutcome::Created(id))) => Ok(id),
        Ok(other) => Err(format!("unexpected add outcome: {other:?}")),
        Err(e) => Err(e.to_string()),
    }
}

/// Remove `track` when an edit left it empty (CapCut removes emptied lanes).
fn remove_track_if_empty(engine: &mut Engine, track: TrackId) {
    let emptied = engine
        .project()
        .timeline()
        .track(track)
        .is_some_and(|t| t.is_empty());
    if !emptied {
        return;
    }
    if let Err(e) = engine.apply(Command::Edit(EditCommand::RemoveTrack { track })) {
        error!(%track, "failed to remove emptied track: {e}");
    }
}

/// Create a new track of `kind` for drops/moves that don't target an existing
/// lane, inserted so it appears at `drop_row` in the lane list. Named by
/// kind + per-kind count (V1, V2, A1, …).
fn create_track(engine: &mut Engine, kind: TrackKind, drop_row: i64) -> Result<TrackId, String> {
    let timeline = engine.project().timeline();
    // The lane list shows the stack top-first (see projection.rs), so the new
    // lane appears at UI row r when inserted at stack index (len - r). The
    // clamp covers drops above the first lane (⇒ top of stack) and below the
    // last (⇒ bottom).
    let stack_len = timeline.order().len() as i64;
    let order_index = (stack_len - drop_row).clamp(0, stack_len) as usize;
    let count = timeline.tracks_ordered().filter(|t| t.kind == kind).count();
    match engine.apply(Command::Edit(EditCommand::AddTrack {
        kind,
        name: format!("{}{}", kind_prefix(kind), count + 1),
        index: Some(order_index),
    })) {
        Ok(ApplyOutcome::Edited(EditOutcome::CreatedTrack(id))) => Ok(id),
        Ok(other) => Err(format!("unexpected add-track outcome: {other:?}")),
        Err(e) => Err(e.to_string()),
    }
}

fn kind_prefix(kind: TrackKind) -> &'static str {
    match kind {
        TrackKind::Video => "V",
        TrackKind::Audio => "A",
        TrackKind::Text => "T",
        TrackKind::Sticker => "ST",
        TrackKind::Effect => "FX",
        TrackKind::Filter => "F",
        TrackKind::Adjustment => "ADJ",
    }
}

/// First start ≥ `desired` where `[start, start + duration)` fits in a gap on
/// `track`. Clips are scanned in start order (they never overlap), so a blocked
/// candidate just slides to the blocker's end — O(n) on this cold per-drop path.
fn first_fit_start(track: &Track, desired: i64, duration_ticks: i64) -> i64 {
    let mut start = desired;
    for clip in track.clips_ordered() {
        if start + duration_ticks <= clip.timeline.start.value {
            break; // fits entirely before this clip
        }
        start = start.max(clip.timeline.end_tick());
    }
    start
}

fn parse_raw_id(raw: &str) -> Option<u64> {
    raw.parse().ok()
}

/// Snapshot the engine's project and hand it to the UI thread, which rebuilds
/// the Slint view model. The snapshot crosses the thread boundary (`Send`);
/// the `!Send` Slint model types are constructed inside the event-loop closure.
/// History availability rides along so the toolbar's undo/redo states always
/// match the projection they were published with.
///
/// PORT (Phase 3): the audio mixer's timeline snapshot publishes from this
/// same chokepoint on main, so what playback sounds like can never diverge
/// from what the UI shows.
fn publish_projection(engine: &mut Engine, ui: &UiSink) {
    let generator_sizes = generator_content_sizes(engine);
    // Pool entries whose backing file is gone (raw ids) — drives the relink
    // dialog count and the library tiles' missing badges. Computed here on
    // the worker thread so the UI thread never stats the filesystem (a dead
    // network mount must not hitch painting).
    let missing_media: std::collections::HashSet<u64> = engine
        .project()
        .media_iter()
        .filter(|m| !m.path().exists())
        .map(|m| m.id.raw())
        .collect();
    let project = engine.project().clone();
    let can_undo = engine.can_undo();
    let can_redo = engine.can_redo();
    // Session save state rides the same chokepoint as the project view, so
    // the title bar's dirty dot can never disagree with the engine.
    let dirty = engine.is_dirty();
    let file_name = engine
        .project_path()
        .and_then(|p| p.file_stem())
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let has_path = engine.project_path().is_some();
    // Full path (not just the stem): main.rs needs it to address the
    // session's autosave slot when a close discards unsaved work.
    let file_path = engine
        .project_path()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();
    let editor_weak = ui.editor.clone();
    if let Err(e) = slint::invoke_from_event_loop(move || {
        if let Some(store) = editor_weak.upgrade() {
            store.set_project(crate::projection::project_to_slint(
                &project,
                &generator_sizes,
                &missing_media,
            ));
            store.set_missing_media_count(missing_media.len() as i32);
            store.set_can_undo(can_undo);
            store.set_can_redo(can_redo);
            store.set_projection_revision(store.get_projection_revision().saturating_add(1));
            store.set_project_dirty(dirty);
            store.set_project_has_path(has_path);
            store.set_project_file_name(file_name.into());
            store.set_project_file_path(file_path.into());
        }
    }) {
        error!("failed to publish project projection to UI: {e}");
    }
}

/// Bump `EditorStore.session-epoch`: the session was replaced wholesale
/// (open / new), and UI-side session state — playhead, selection, in/out
/// range, playback — must reset. The watcher lives in `app.slint`.
fn bump_session_epoch(ui: &UiSink) {
    let editor_weak = ui.editor.clone();
    if let Err(e) = slint::invoke_from_event_loop(move || {
        if let Some(store) = editor_weak.upgrade() {
            store.set_session_epoch(store.get_session_epoch() + 1);
        }
    }) {
        error!("failed to bump session epoch: {e}");
    }
}

/// Surface a session-level failure (save/open) to the user: sets
/// `EditorStore.session-error`, which mounts the message dialog until the
/// user dismisses it (clearing the property).
fn publish_session_error(ui: &UiSink, message: String) {
    let editor_weak = ui.editor.clone();
    if let Err(e) = slint::invoke_from_event_loop(move || {
        if let Some(store) = editor_weak.upgrade() {
            store.set_session_error(message.into());
        }
    }) {
        error!("failed to publish session error: {e}");
    }
}

/// Drawn-content size (canvas px) for every generated clip, keyed by raw clip
/// id — the preview's selection box and hit-test hug what the generator
/// actually draws instead of its full-canvas raster.
///
/// PORT (Phase 6): served from the engine's raster cache on main
/// (`Engine::generator_content_size`), which this branch doesn't expose yet.
/// Empty means the selection geometry falls back to canvas-sized boxes —
/// functional, just looser around text/shapes.
fn generator_content_sizes(_engine: &mut Engine) -> HashMap<u64, (i32, i32)> {
    HashMap::new()
}

// PORT (Phase 3): main built the playback mixer's `AudioSnapshot` here (every
// audible clip with its source window, retime, volume envelope, and fades).
// It returns with the cpal audio system on `ExportAudioMixer`.

/// The preview fit bound + quality ladder (the Swift `PreviewFeed` model,
/// worker-side): renders fit inside the reported viewport, and a slow run of
/// renders steps the resolution down a tier so scrubbing stays interactive.
/// `Cell`s because the coalescing loop and the `mutate` closure both repaint.
struct FrameFit {
    /// Bucketed viewport (physical px); `(0, 0)` until the panel reports.
    viewport: Cell<(u32, u32)>,
    /// EMA of recent render costs, milliseconds.
    avg_ms: Cell<f64>,
    /// Index into [`QUALITY_LADDER`].
    tier: Cell<usize>,
}

/// Resolution multipliers, best first. Tier moves down when renders run slow
/// and back up when they're comfortably fast.
const QUALITY_LADDER: [f64; 3] = [1.0, 0.7, 0.5];
/// Never request more than this many pixels on the long side — retina
/// monitors ask for huge viewports whose cost buys invisible detail.
const MAX_LONG_SIDE: f64 = 1440.0;
/// EMA above this drops a tier; below `RAISE_BELOW_MS` climbs back.
const DROP_ABOVE_MS: f64 = 45.0;
const RAISE_BELOW_MS: f64 = 18.0;
/// Viewport reports quantize to this grid so a live window resize (a report
/// per frame) doesn't churn re-renders for sub-bucket changes.
const VIEWPORT_BUCKET: u32 = 64;

impl Default for FrameFit {
    fn default() -> Self {
        Self {
            viewport: Cell::new((0, 0)),
            avg_ms: Cell::new(0.0),
            tier: Cell::new(0),
        }
    }
}

impl FrameFit {
    /// Record a new on-screen size. Returns whether the bucketed bound
    /// changed (⇒ the caller should repaint the current frame).
    fn set_viewport(&self, width: u32, height: u32) -> bool {
        let bucket = |v: u32| v.max(1).div_ceil(VIEWPORT_BUCKET) * VIEWPORT_BUCKET;
        let next = (bucket(width), bucket(height));
        let changed = self.viewport.get() != next;
        self.viewport.set(next);
        changed
    }

    /// The `(max_width, max_height)` to request for the next render: the
    /// viewport scaled by the current quality tier, capped at
    /// [`MAX_LONG_SIDE`]. `None` until the viewport is known (render at full
    /// canvas size — the launch state before the panel lays out).
    fn fit_bound(&self) -> Option<(u32, u32)> {
        let (w, h) = self.viewport.get();
        if w == 0 || h == 0 {
            return None;
        }
        let mut scale = QUALITY_LADDER[self.tier.get()];
        let long_side = f64::from(w.max(h)) * scale;
        if long_side > MAX_LONG_SIDE {
            scale *= MAX_LONG_SIDE / long_side;
        }
        let dim = |v: u32| ((f64::from(v) * scale).round() as u32).max(1);
        Some((dim(w), dim(h)))
    }

    /// Fold one render's cost into the EMA and step the quality tier: slow
    /// renders drop a tier immediately (their cost dominates the average),
    /// sustained fast ones climb back up.
    fn note_render_cost(&self, elapsed: Duration) {
        let ms = elapsed.as_secs_f64() * 1000.0;
        let avg = self.avg_ms.get();
        let next = if avg == 0.0 { ms } else { avg * 0.75 + ms * 0.25 };
        self.avg_ms.set(next);

        let tier = self.tier.get();
        if next > DROP_ABOVE_MS && tier + 1 < QUALITY_LADDER.len() {
            self.tier.set(tier + 1);
            self.avg_ms.set(0.0);
            debug!(tier = tier + 1, avg_ms = next, "preview quality tier down");
        } else if next < RAISE_BELOW_MS && tier > 0 {
            self.tier.set(tier - 1);
            self.avg_ms.set(0.0);
            debug!(tier = tier - 1, avg_ms = next, "preview quality tier up");
        }
    }
}

fn render_frame(
    engine: &mut Engine,
    tl_rate: Rational,
    preview_weak: &slint::Weak<PreviewStore<'static>>,
    tick: i64,
    fit: &FrameFit,
) {
    let at = RationalTime::new(tick, tl_rate);
    let started = Instant::now();
    let result = match fit.fit_bound() {
        Some((max_w, max_h)) => engine.get_frame_fit(at, max_w, max_h),
        None => engine.get_frame(at),
    };
    match result {
        Ok(frame) => {
            fit.note_render_cost(started.elapsed());
            let weak = preview_weak.clone();
            if let Err(e) = slint::invoke_from_event_loop(move || {
                if let Some(store) = weak.upgrade() {
                    store.set_frame(crate::preview::to_slint_image(frame));
                }
            }) {
                error!("failed to deliver preview frame to UI: {e}");
            }
        }
        Err(e) => error!(tick, "preview frame failed: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_models::Project;

    /// `keyframes_at` slices one merged timeline diamond: only the
    /// properties keyframed exactly at the tick, each with its own value
    /// and easing, position as vec2.
    #[test]
    fn keyframes_at_collects_per_property_hits() {
        let mut t = AnimatedTransform::identity();
        t.set_param_keyframe(
            ClipParam::Scale,
            10,
            ParamValue::Scalar(2.0),
            Easing::EaseIn,
        )
        .unwrap();
        t.set_param_keyframe(
            ClipParam::Scale,
            20,
            ParamValue::Scalar(3.0),
            Easing::Linear,
        )
        .unwrap();
        t.set_param_keyframe(
            ClipParam::Position,
            10,
            ParamValue::Vec2([0.1, -0.2]),
            Easing::Linear,
        )
        .unwrap();
        t.set_param_keyframe(
            ClipParam::Opacity,
            30,
            ParamValue::Scalar(0.5),
            Easing::Linear,
        )
        .unwrap();

        let hits = keyframes_at(&t, 10);
        assert_eq!(
            hits,
            vec![
                (
                    ClipParam::Position,
                    ParamValue::Vec2([0.1, -0.2]),
                    Easing::Linear
                ),
                (ClipParam::Scale, ParamValue::Scalar(2.0), Easing::EaseIn),
            ]
        );

        assert!(keyframes_at(&t, 15).is_empty());
        assert_eq!(keyframes_at(&t, 30).len(), 1);
    }

    // --- magnet ripple trim (commit path) -----------------------------------

    /// Engine over an empty project that carries one 1000-tick media entry
    /// (no real file needed — nothing here decodes).
    fn trim_test_engine() -> (Engine, MediaId) {
        let r = Rational::FPS_24;
        let mut project = Project::new("trim-fixture", r);
        let media = project.add_media(cutlass_models::MediaSource::new(
            "/tmp/trim-fixture.mp4",
            1920,
            1080,
            r,
            1000,
            false,
        ));
        let engine = Engine::with_project(EngineConfig::default(), project).expect("engine");
        (engine, media)
    }

    fn add_video_track(engine: &mut Engine, name: &str) -> TrackId {
        match engine
            .apply(Command::Edit(EditCommand::AddTrack {
                kind: TrackKind::Video,
                name: name.into(),
                index: None,
            }))
            .expect("add track")
        {
            ApplyOutcome::Edited(EditOutcome::CreatedTrack(id)) => id,
            other => panic!("expected CreatedTrack, got {other:?}"),
        }
    }

    /// Media-backed clip at `[start, start+duration)`; the source window is
    /// offset 100 ticks into the fixture media so both edges keep headroom.
    fn add_media_clip(
        engine: &mut Engine,
        track: TrackId,
        media: MediaId,
        start: i64,
        duration: i64,
    ) -> ClipId {
        match engine
            .apply(Command::Edit(EditCommand::AddClip {
                track,
                media,
                source: TimeRange::at_rate(100 + start, duration, Rational::FPS_24),
                start: RationalTime::new(start, Rational::FPS_24),
            }))
            .expect("add clip")
        {
            ApplyOutcome::Edited(EditOutcome::Created(id)) => id,
            other => panic!("expected Created, got {other:?}"),
        }
    }

    fn clip_starts(engine: &Engine, clips: &[ClipId]) -> Vec<i64> {
        clips
            .iter()
            .map(|id| engine.project().clip(*id).expect("clip").start().value)
            .collect()
    }

    #[test]
    fn ripple_tail_shrink_shifts_downstream_on_main_lane() {
        let (mut engine, media) = trim_test_engine();
        let track = add_video_track(&mut engine, "V1");
        let a = add_media_clip(&mut engine, track, media, 0, 50);
        let b = add_media_clip(&mut engine, track, media, 50, 30);
        let c = add_media_clip(&mut engine, track, media, 80, 40);

        commit_trims(
            &mut engine,
            &[(b, TimeRange::at_rate(50, 20, Rational::FPS_24))],
            true,
        )
        .expect("ripple tail shrink");

        assert_eq!(clip_starts(&engine, &[a, b, c]), [0, 50, 70]);
        assert!(engine.undo());
        assert_eq!(clip_starts(&engine, &[a, b, c]), [0, 50, 80]);
    }

    #[test]
    fn ripple_tail_grow_opens_room_before_extending() {
        let (mut engine, media) = trim_test_engine();
        let track = add_video_track(&mut engine, "V1");
        let a = add_media_clip(&mut engine, track, media, 0, 50);
        let b = add_media_clip(&mut engine, track, media, 50, 30);

        commit_trims(
            &mut engine,
            &[(a, TimeRange::at_rate(0, 60, Rational::FPS_24))],
            true,
        )
        .expect("ripple tail grow");

        assert_eq!(clip_starts(&engine, &[a, b]), [0, 60]);
    }

    #[test]
    fn ripple_head_shrink_reanchors_at_old_start() {
        let (mut engine, media) = trim_test_engine();
        let track = add_video_track(&mut engine, "V1");
        let a = add_media_clip(&mut engine, track, media, 0, 50);
        let b = add_media_clip(&mut engine, track, media, 50, 30);

        commit_trims(
            &mut engine,
            &[(a, TimeRange::at_rate(10, 40, Rational::FPS_24))],
            true,
        )
        .expect("ripple head shrink");

        assert_eq!(clip_starts(&engine, &[a, b]), [0, 40]);
    }

    #[test]
    fn plain_trim_off_magnet_leaves_gap() {
        let (mut engine, media) = trim_test_engine();
        let track = add_video_track(&mut engine, "V1");
        let a = add_media_clip(&mut engine, track, media, 0, 50);
        let b = add_media_clip(&mut engine, track, media, 50, 30);
        let c = add_media_clip(&mut engine, track, media, 80, 40);

        commit_trims(
            &mut engine,
            &[(b, TimeRange::at_rate(50, 20, Rational::FPS_24))],
            false,
        )
        .expect("plain trim");

        assert_eq!(clip_starts(&engine, &[a, b, c]), [0, 50, 80]);
    }

    // --- magnet ripple trim: media-backed clips (source derivation, links,
    // --- rollback) -----------------------------------------------------------

    /// Main lane `V1` packed gapless — A [0,100) B [100,200) C [200,300) —
    /// each clip cut from the middle of a 1000-tick media, so both edges
    /// have plenty of source headroom: A source [100,200), B [300,400),
    /// C [500,600).
    fn ripple_fixture() -> (Engine, [ClipId; 3], TrackId) {
        let r = Rational::FPS_24;
        let mut project = Project::new("ripple-fixture", r);
        let media = project.add_media(cutlass_models::MediaSource::new(
            "/tmp/ripple-fixture.mp4",
            1920,
            1080,
            r,
            1000,
            false,
        ));
        let track = project.add_track(TrackKind::Video, "V1");
        let a = project
            .add_clip(
                track,
                media,
                TimeRange::at_rate(100, 100, r),
                RationalTime::new(0, r),
            )
            .expect("clip A");
        let b = project
            .add_clip(
                track,
                media,
                TimeRange::at_rate(300, 100, r),
                RationalTime::new(100, r),
            )
            .expect("clip B");
        let c = project
            .add_clip(
                track,
                media,
                TimeRange::at_rate(500, 100, r),
                RationalTime::new(200, r),
            )
            .expect("clip C");

        let engine = Engine::with_project(EngineConfig::default(), project).expect("engine");
        (engine, [a, b, c], track)
    }

    fn extent(engine: &Engine, clip: ClipId) -> (i64, i64) {
        let placed = engine.project().clip(clip).expect("clip exists").timeline;
        (placed.start.value, placed.duration.value)
    }

    fn source_start(engine: &Engine, clip: ClipId) -> i64 {
        engine
            .project()
            .clip(clip)
            .expect("clip exists")
            .source_range()
            .expect("media clip has a source range")
            .start
            .value
    }

    fn tr24(start: i64, duration: i64) -> TimeRange {
        TimeRange::at_rate(start, duration, Rational::FPS_24)
    }

    /// Leading-edge shrink: the resolved extent moves the start right (that
    /// delta advances the source in-point), the commit re-anchors at the old
    /// start, and downstream follows — the lane stays gapless.
    #[test]
    fn ripple_head_shrink_advances_source_and_stays_anchored() {
        let (mut engine, [a, b, c], _track) = ripple_fixture();

        let rippled =
            commit_trims(&mut engine, &[(b, tr24(120, 80))], true).expect("ripple head shrink");
        assert!(rippled);

        assert_eq!(extent(&engine, b), (100, 80));
        assert_eq!(source_start(&engine, b), 320);
        assert_eq!(extent(&engine, c), (180, 100));
        assert_eq!(extent(&engine, a), (0, 100));

        // One undo restores the trim and the shift together.
        assert!(engine.undo());
        assert_eq!(extent(&engine, b), (100, 100));
        assert_eq!(source_start(&engine, b), 300);
        assert_eq!(extent(&engine, c), (200, 100));
    }

    /// Leading-edge grow: earlier source is revealed (in-point retreats),
    /// the clip stays anchored, downstream moves right by the delta.
    #[test]
    fn ripple_head_grow_reveals_earlier_source() {
        let (mut engine, [a, b, c], _track) = ripple_fixture();

        let rippled =
            commit_trims(&mut engine, &[(b, tr24(50, 150))], true).expect("ripple head grow");
        assert!(rippled);

        assert_eq!(extent(&engine, b), (100, 150));
        assert_eq!(source_start(&engine, b), 250);
        assert_eq!(extent(&engine, c), (250, 100));
        assert_eq!(extent(&engine, a), (0, 100));
    }

    /// Trailing-edge trims keep the source in-point and move the out-point;
    /// the shift only touches clips after the old end.
    #[test]
    fn ripple_tail_trims_keep_source_in_point() {
        let (mut engine, [a, b, c], _track) = ripple_fixture();

        commit_trims(&mut engine, &[(b, tr24(100, 140))], true).expect("ripple tail grow");
        assert_eq!(extent(&engine, b), (100, 140));
        assert_eq!(source_start(&engine, b), 300);
        assert_eq!(extent(&engine, c), (240, 100));
        assert_eq!(extent(&engine, a), (0, 100));
    }

    /// The last clip on the lane has nothing downstream — the ripple is a
    /// plain trim (the shift selects an empty set and stays a no-op).
    #[test]
    fn ripple_trim_of_last_clip_has_no_downstream() {
        let (mut engine, [a, b, c], _track) = ripple_fixture();

        commit_trims(&mut engine, &[(c, tr24(200, 40))], true).expect("ripple last clip");
        assert_eq!(extent(&engine, c), (200, 40));
        assert_eq!(extent(&engine, a), (0, 100));
        assert_eq!(extent(&engine, b), (100, 100));
    }

    /// The magnet only governs the main lane (bottom video track): an
    /// overlay-lane trim stays plain even with the magnet on.
    #[test]
    fn overlay_lane_trim_does_not_ripple() {
        let (mut engine, [a, _, _], _track) = ripple_fixture();
        let media = engine
            .project()
            .clip(a)
            .expect("clip")
            .media()
            .expect("media");
        let overlay = add_video_track(&mut engine, "V2");
        let d = add_media_clip(&mut engine, overlay, media, 0, 100);
        let e = add_media_clip(&mut engine, overlay, media, 100, 100);

        let rippled = commit_trims(&mut engine, &[(d, tr24(0, 60))], true).expect("overlay trim");
        assert!(!rippled);
        assert_eq!(extent(&engine, d), (0, 60));
        assert_eq!(extent(&engine, e), (100, 100));
    }

    /// A trim the engine rejects (source bounds) rolls the whole group back:
    /// the shift that already opened room is undone, history stays untouched.
    #[test]
    fn rejected_ripple_rolls_back_whole_group() {
        let (mut engine, [a, b, c], _track) = ripple_fixture();

        // B's source is [300,400) of a 1000-tick media: growing the tail by
        // 700 ticks would need source up to 1100 — rejected after the shift.
        let result = commit_trims(&mut engine, &[(b, tr24(100, 800))], true);
        assert!(result.is_err());

        assert_eq!(extent(&engine, a), (0, 100));
        assert_eq!(extent(&engine, b), (100, 100));
        assert_eq!(extent(&engine, c), (200, 100));
        assert!(
            !engine.can_undo(),
            "rolled-back group must leave no history"
        );
    }

    /// Linked-pair trim (video on the main lane, audio partner elsewhere):
    /// both members ripple on their own lanes, so the pair stays in sync and
    /// downstream clips on both lanes shift by the same delta.
    #[test]
    fn linked_pair_ripples_on_both_lanes() {
        let (mut engine, [_, b, c], _track) = ripple_fixture();
        let r = Rational::FPS_24;
        let audio = match engine
            .apply(Command::Edit(EditCommand::AddTrack {
                kind: TrackKind::Audio,
                name: "A1".into(),
                index: None,
            }))
            .expect("add audio track")
        {
            ApplyOutcome::Edited(EditOutcome::CreatedTrack(id)) => id,
            other => panic!("expected CreatedTrack, got {other:?}"),
        };
        let media = engine
            .project()
            .clip(b)
            .expect("clip B")
            .media()
            .expect("media");
        let add_audio_clip = |engine: &mut Engine, source: TimeRange, start: i64| match engine
            .apply(Command::Edit(EditCommand::AddClip {
                track: audio,
                media,
                source,
                start: RationalTime::new(start, r),
            }))
            .expect("add audio clip")
        {
            ApplyOutcome::Edited(EditOutcome::Created(id)) => id,
            other => panic!("expected Created, got {other:?}"),
        };
        // P mirrors B; Q sits downstream on the audio lane, aligned with C.
        let p = add_audio_clip(&mut engine, tr24(300, 100), 100);
        let q = add_audio_clip(&mut engine, tr24(500, 100), 200);

        // Head-shrink both members by 20 (the worker's trim path hands the
        // same edge delta to every link-group member).
        commit_trims(&mut engine, &[(b, tr24(120, 80)), (p, tr24(120, 80))], true)
            .expect("linked ripple trim");

        assert_eq!(extent(&engine, b), (100, 80));
        assert_eq!(extent(&engine, p), (100, 80));
        assert_eq!(source_start(&engine, b), 320);
        assert_eq!(source_start(&engine, p), 320);
        // Downstream on both lanes shifted left by 20, staying aligned.
        assert_eq!(extent(&engine, c), (180, 100));
        assert_eq!(extent(&engine, q), (180, 100));
    }
}
