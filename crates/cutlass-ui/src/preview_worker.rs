//! Background preview rendering: engine and decode/composite stay off the UI thread.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use cutlass_commands::{Command, EditCommand, EditOutcome, ProjectCommand};
use cutlass_engine::{ApplyOutcome, Engine, EngineConfig, EngineError, ExportSettings};
use cutlass_models::{
    AnimatedTransform, ClipId, ClipParam, ClipSource, ClipTransform, CropRect, Easing, Generator,
    LinkId, MAX_SPEED, MIN_SPEED, MarkerColor, MarkerId, MediaId, Param, ParamValue, Project,
    Rational, RationalTime, TimeRange, Track, TrackId, TrackKind, resample,
};
use tracing::{error, info, warn};

use crate::agent::{AgentCreated, AgentPlanStep};
use crate::audio::{AudioHandle, AudioSnapshot, AudioSpan};
use crate::strips::StripHandle;
use crate::thumbnails::{ThumbKind, ThumbnailHandle};
use crate::{EditorStore, ExportBackend, PreviewStore};

/// Everything a mutation publishes to: the Slint view model and the audio
/// mixer's timeline snapshot. One value threaded through the worker so the
/// two can never diverge.
struct UiSink {
    editor: slint::Weak<EditorStore<'static>>,
    export: slint::Weak<ExportBackend<'static>>,
    audio: AudioHandle,
}

pub struct PreviewSession {
    pub duration_ticks: i64,
    pub tl_rate: Rational,
}

/// Work submitted to the engine thread. Scrub frames coalesce to the latest
/// pending tick; imports must not be dropped by that coalescing (see
/// [`worker_loop`]).
enum WorkerMsg {
    Frame(i64),
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
    MoveGroup { moves: Vec<GroupMove> },
    /// Re-place `clip` (raw id) at `[start_tick, start_tick + duration_ticks)`
    /// on its own lane (edge trim; the engine re-derives the source in/out).
    TrimClip {
        clip: String,
        start_tick: i64,
        duration_ticks: i64,
    },
    /// Remove every clip in `clips` (raw ids) as one history entry; lanes
    /// the removals empty are removed too (same policy as drag-moves).
    RemoveClips { clips: Vec<String> },
    /// Replace a generated clip's content (raw id) — e.g. an inspector title
    /// edit. One undoable history entry per committed edit.
    SetGenerator { clip: String, generator: Generator },
    /// Resize a shape clip's reference-pixel dimensions. Preserves shape kind
    /// and fill from the committed generator.
    SetShapeSize {
        clip: String,
        width: f32,
        height: f32,
    },
    /// Live preview of a shape resize at `tick` — no history entry.
    PreviewShapeSize {
        clip: String,
        width: f32,
        height: f32,
        tick: i64,
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
    /// Live drag override (preview roadmap Phase 3): render `tick` with
    /// `clip`'s transform replaced — session state on the engine, no history
    /// entry, no projection republish. Bursts coalesce to the newest value
    /// like `Frame` requests do.
    TransformOverride {
        clip: String,
        transform: ClipTransform,
        tick: i64,
    },
    /// Drop the gesture override (no-op release / cancelled drag) and
    /// re-render `tick` from committed state.
    ClearTransformOverride { tick: i64 },
    /// Live inspector edit preview (e.g. font-size slider drag): render `tick`
    /// with `clip`'s generator replaced — session state on the engine, no
    /// history entry, no projection republish. Coalesces with `Frame`/itself
    /// like `TransformOverride` so a fast drag can't back the queue up.
    GeneratorOverride {
        clip: String,
        generator: Generator,
        tick: i64,
    },
    /// Drop the generator override (control released with no net change) and
    /// re-render `tick` from committed state.
    ClearGeneratorOverride { tick: i64 },
    /// Commit a transform gesture: clear any override and apply one undoable
    /// `SetClipTransform`, then re-render `tick` (a nudge has no preceding
    /// override, so the frame must refresh here).
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
    RemoveKeyframesAt { clip: String, tick: i64 },
    /// Split `clip` (raw id) at `at_tick` (sequence ticks). The UI gates on
    /// the playhead being strictly inside the clip; the engine re-validates.
    SplitClip { clip: String, at_tick: i64 },
    /// Drop a ruler marker at `at_tick`. `color` is a palette name
    /// ("teal", "blue", …) or empty to cycle. One undoable history entry.
    AddMarker {
        at_tick: i64,
        name: String,
        color: String,
    },
    /// Remove a ruler marker by raw id. One undoable history entry.
    RemoveMarker { marker: String },
    /// Step the engine history one entry back / forward.
    Undo,
    Redo,
    /// Snapshot `clips` (raw ids — the whole selection) into the worker
    /// clipboard as one block. A snapshot, not a reference — pasting works
    /// after the originals are deleted.
    CopyClips { clips: Vec<String> },
    /// Place the clipboard block at `tick`: members keep their lanes and
    /// relative placement, the whole block slides right as one unit until
    /// every member fits.
    PasteAt { tick: i64 },
    /// Place copies of `clips` (the whole selection) right after the block
    /// they form, keeping lanes and relative placement.
    DuplicateClips { clips: Vec<String> },
    /// Dissolve the link group of every clip in `clips` (raw ids): all
    /// members of the touched groups — selected or not — end up unlinked.
    UnlinkClips { clips: Vec<String> },
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
    /// Start an export job: the worker clones the project and hands it to a
    /// dedicated thread (decode + GPU composite + encode must not stall
    /// preview). One job at a time; a second request while one runs is
    /// refused with a status message.
    Export(ExportRequest),
    /// Flag the running export job to stop after the frame in flight.
    CancelExport,
    /// Write the session to a `.cutlass` file. `None` reuses the engine's
    /// current project path (plain Cmd+S on a saved project — the UI gates
    /// on a path existing); `Some` rebinds it (first save / Save As). Not
    /// undoable; on success the projection republish clears the dirty dot.
    /// Either way `save-finished(ok)` fires so a pending guarded transition
    /// (open/new/close waiting on "Save") can continue or abort.
    SaveProject { path: Option<PathBuf> },
    /// Replace the session from a `.cutlass` file (tolerant: entries whose
    /// media file is gone are kept and surface through the relink flow —
    /// the projection republish carries the missing set, and app.slint
    /// raises the relink dialog on the epoch bump). Success re-registers
    /// still-present pool media with the thumbnail and strip workers,
    /// republishes everything, and bumps the session epoch so the UI
    /// resets its session state (playhead, selection, range). Failure
    /// publishes `session-error`. The unsaved-changes guard ran UI-side
    /// before this message was sent.
    OpenProject { path: PathBuf },
    /// Re-point a media-pool entry (raw id) at a new file (missing-media
    /// relink, M0): the engine re-probes the file and swaps the entry's
    /// path/metadata in place (id and clips untouched), the tile workers
    /// re-register, and the projection republish drops the entry from the
    /// missing set. Not undoable — state repair, not an edit.
    RelinkMedia { media: String, path: PathBuf },
    /// Try `folder/<filename>` for every missing pool entry (locate-folder
    /// gesture in the relink dialog).
    RelinkFolder { folder: PathBuf },
    /// Replace the session with a fresh, empty, unsaved project (File →
    /// New). Same epoch bump as `OpenProject`; guard ran UI-side.
    NewProject,
    /// Periodic autosave sweep (UI timer, every
    /// [`autosave::SWEEP_INTERVAL`](crate::autosave::SWEEP_INTERVAL)).
    /// Dirty session ⇒ snapshot to the sidecar slot; clean ⇒ the slot is
    /// stale and gets removed. Failures only log — autosave must never
    /// interrupt editing.
    Autosave,
    /// Restore a crash-recovery snapshot (launch offer, accepted). Loads
    /// `autosave` tolerantly, binds the session to `source` (the user's
    /// file — `None` for a never-saved session), and leaves it dirty so
    /// Cmd+S writes the recovered work back to the real file.
    RestoreAutosave {
        autosave: PathBuf,
        source: Option<PathBuf>,
    },
    /// Clone the live project for the AI agent's sandbox rehearsal
    /// (`src/agent.rs`). Ordered with mutations, so the snapshot always
    /// reflects every edit sent before it.
    SnapshotProject { reply: Sender<Project> },
    /// Replay a rehearsed agent plan as one history group, re-validating
    /// every step against the live project and remapping ids the sandbox
    /// allocated. All-or-nothing: any failure rolls the group back.
    AgentApplyPlan {
        steps: Vec<AgentPlanStep>,
        reply: Sender<Result<(), String>>,
    },
}

/// Dialog settings for one export job (see `ui/lib/export-backend.slint`).
pub struct ExportRequest {
    pub path: PathBuf,
    /// Target output height; `None` ⇒ the composite canvas size.
    pub target_height: Option<u32>,
    /// Output frame rate; `None` ⇒ the timeline rate. Dialog presets are
    /// integer rates.
    pub fps_num: Option<i32>,
    /// libx264 constant-quality level.
    pub crf: u8,
}

/// One clip's resolved landing inside a [`WorkerMsg::MoveGroup`] batch.
/// All raw ids from the Slint projection.
pub struct GroupMove {
    pub clip: String,
    pub track: String,
    pub start_tick: i64,
}

/// Which track header toggle a [`WorkerMsg::SetTrackFlag`] addresses.
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

impl WorkerHandle {
    pub fn request_frame(&self, tick: i64) {
        let _ = self.tx.send(WorkerMsg::Frame(tick));
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

    pub fn autosave(&self) {
        let _ = self.tx.send(WorkerMsg::Autosave);
    }

    pub fn restore_autosave(&self, autosave: PathBuf, source: Option<PathBuf>) {
        let _ = self.tx.send(WorkerMsg::RestoreAutosave { autosave, source });
    }

    pub fn relink_media(&self, media: String, path: PathBuf) {
        let _ = self.tx.send(WorkerMsg::RelinkMedia { media, path });
    }

    pub fn relink_folder(&self, folder: PathBuf) {
        let _ = self.tx.send(WorkerMsg::RelinkFolder { folder });
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

    pub fn preview_shape_size(&self, clip: String, width: f32, height: f32, tick: i64) {
        let _ = self.tx.send(WorkerMsg::PreviewShapeSize {
            clip,
            width,
            height,
            tick,
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

    pub fn transform_override(&self, clip: String, transform: ClipTransform, tick: i64) {
        let _ = self.tx.send(WorkerMsg::TransformOverride {
            clip,
            transform,
            tick,
        });
    }

    pub fn clear_transform_override(&self, tick: i64) {
        let _ = self.tx.send(WorkerMsg::ClearTransformOverride { tick });
    }

    pub fn generator_override(&self, clip: String, generator: Generator, tick: i64) {
        let _ = self.tx.send(WorkerMsg::GeneratorOverride {
            clip,
            generator,
            tick,
        });
    }

    pub fn clear_generator_override(&self, tick: i64) {
        let _ = self.tx.send(WorkerMsg::ClearGeneratorOverride { tick });
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

    /// Synchronous round-trip: clone of the live project as of every edit
    /// sent before this call. `None` only if the worker thread is gone.
    pub fn snapshot_project(&self) -> Option<Project> {
        let (reply, rx) = bounded(1);
        self.tx.send(WorkerMsg::SnapshotProject { reply }).ok()?;
        rx.recv().ok()
    }

    /// Synchronous round-trip: replay a rehearsed agent plan as one undo
    /// entry. `None` only if the worker thread is gone.
    pub fn agent_apply_plan(&self, steps: Vec<AgentPlanStep>) -> Option<Result<(), String>> {
        let (reply, rx) = bounded(1);
        self.tx
            .send(WorkerMsg::AgentApplyPlan { steps, reply })
            .ok()?;
        rx.recv().ok()
    }

    pub fn export(&self, request: ExportRequest) {
        let _ = self.tx.send(WorkerMsg::Export(request));
    }

    pub fn cancel_export(&self) {
        let _ = self.tx.send(WorkerMsg::CancelExport);
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
        export_weak: slint::Weak<ExportBackend<'static>>,
        thumbs: ThumbnailHandle,
        strips: StripHandle,
        audio: AudioHandle,
    ) -> Result<(Self, PreviewSession), String> {
        let (ready_tx, ready_rx) = bounded(1);
        let (req_tx, req_rx) = unbounded();

        let join = std::thread::Builder::new()
            .name("cutlass-preview".into())
            .spawn(move || {
                if let Err(e) = worker_main(
                    config,
                    preview_weak,
                    editor_weak,
                    export_weak,
                    thumbs,
                    strips,
                    audio,
                    req_rx,
                    ready_tx,
                ) {
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

#[allow(clippy::too_many_arguments)]
fn worker_main(
    config: EngineConfig,
    preview_weak: slint::Weak<PreviewStore<'static>>,
    editor_weak: slint::Weak<EditorStore<'static>>,
    export_weak: slint::Weak<ExportBackend<'static>>,
    thumbs: ThumbnailHandle,
    strips: StripHandle,
    audio: AudioHandle,
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
    info!(
        duration_ticks = session.duration_ticks,
        tl_rate = ?session.tl_rate,
        "preview worker ready (empty project)"
    );
    let tl_rate = session.tl_rate;
    ready_tx
        .send(Ok(session))
        .map_err(|e| e.to_string())?;

    // Seed the UI with the engine's project so the editor reads from the engine
    // from the first frame (rather than any Slint-side placeholder).
    let ui = UiSink {
        editor: editor_weak,
        export: export_weak,
        audio,
    };
    publish_projection(&mut engine, &ui);

    worker_loop(&mut engine, tl_rate, preview_weak, ui, thumbs, strips, req_rx);
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
    thumbs: ThumbnailHandle,
    strips: StripHandle,
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
    // One export job at a time. `active` outlives jobs (the export thread
    // clears it on exit); `cancel` is reset at every job start so a stale
    // cancel can't kill the next run.
    let export_state = ExportJobState::default();
    // The autosave slot last written, with the engine revision it captured:
    // a dirty-but-idle session skips the redundant rewrite, and a session
    // identity change (Save As / Open / New) cleans the orphaned slot up.
    let mut autosave_slot: Option<(PathBuf, u64)> = None;
    // Last tick the preview rendered (the playhead). Scrub/seek `Frame`s keep
    // it current; edits re-render here so the composite reflects a delete,
    // generator change, etc. without waiting for the user to move the playhead.
    let mut last_tick: i64 = 0;

    let mutate = |engine: &mut Engine,
                  clipboard: &mut Option<Vec<ClipboardClip>>,
                  main_magnet: &mut bool,
                  linkage: &mut bool,
                  autosave_slot: &mut Option<(PathBuf, u64)>,
                  msg: WorkerMsg| {
        match msg {
            WorkerMsg::Import(path) => {
                import_and_publish(engine, &path, &ui, &thumbs, &strips)
            }
            WorkerMsg::AddClip {
                media,
                track,
                start_tick,
                drop_row,
                insert,
            } => add_clip_and_publish(
                engine,
                &media,
                &track,
                start_tick,
                drop_row,
                insert,
                *linkage,
                &ui,
            ),
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
            WorkerMsg::MoveGroup { moves } => {
                move_group_and_publish(engine, &moves, &ui)
            }
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
            WorkerMsg::SetShapeSize { clip, width, height } => {
                if let Some(generator) = shape_size_from_engine(engine, &clip, width, height) {
                    set_generator_and_publish(engine, &clip, generator, &ui);
                }
            }
            // Only reached when a shape-resize burst interleaves with another
            // coalesced gesture's drain (practically impossible — one slider at
            // a time). The dedicated loop arm coalesces the common case.
            WorkerMsg::PreviewShapeSize {
                clip,
                width,
                height,
                tick,
            } => {
                if let Some(generator) = shape_size_from_engine(engine, &clip, width, height) {
                    apply_generator_override(engine, &clip, generator);
                    render_frame(engine, tl_rate, &preview_weak, tick);
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
            WorkerMsg::ClearTransformOverride { tick } => {
                engine.set_transform_override(None);
                render_frame(engine, tl_rate, &preview_weak, tick);
            }
            WorkerMsg::ClearGeneratorOverride { tick } => {
                engine.set_generator_override(None);
                render_frame(engine, tl_rate, &preview_weak, tick);
            }
            // Only reached if a generator-override burst interleaves with
            // another coalesced gesture's drain (practically impossible — you
            // can't drag two controls at once). The dedicated loop arm handles
            // the common case with coalescing.
            WorkerMsg::GeneratorOverride {
                clip,
                generator,
                tick,
            } => {
                apply_generator_override(engine, &clip, generator);
                render_frame(engine, tl_rate, &preview_weak, tick);
            }
            WorkerMsg::SetTransform {
                clip,
                transform,
                tick,
            } => {
                // The override previewed this exact transform; clearing it as
                // the command lands means the next render is identical — no
                // flicker between gesture end and commit.
                engine.set_transform_override(None);
                // The gesture happened at the visible frame: pass the playhead
                // so animated properties get a keyframe there instead of being
                // flattened (M2 compose semantics).
                let at = RationalTime::new(tick, tl_rate);
                set_transform_and_publish(engine, &clip, transform, at, &ui);
                render_frame(engine, tl_rate, &preview_weak, tick);
            }
            WorkerMsg::FitClip { clip, fill, tick } => {
                fit_clip_and_publish(engine, &clip, fill, tick, tl_rate, &ui);
                render_frame(engine, tl_rate, &preview_weak, tick);
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
            WorkerMsg::RemoveMarker { marker } => {
                remove_marker_and_publish(engine, &marker, &ui)
            }
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
            WorkerMsg::OpenProject { path } => {
                open_project_and_publish(engine, path, &ui, &thumbs, &strips)
            }
            WorkerMsg::RelinkMedia { media, path } => {
                relink_media_and_publish(engine, &media, &path, &ui, &thumbs, &strips)
            }
            WorkerMsg::RelinkFolder { folder } => {
                relink_folder_and_publish(engine, folder, &ui, &thumbs, &strips)
            }
            WorkerMsg::NewProject => new_project_and_publish(engine, &ui),
            WorkerMsg::Autosave => autosave_sweep(engine, autosave_slot),
            WorkerMsg::RestoreAutosave { autosave, source } => restore_autosave_and_publish(
                engine,
                autosave,
                source,
                autosave_slot,
                &ui,
                &thumbs,
                &strips,
            ),
            WorkerMsg::SnapshotProject { reply } => {
                let _ = reply.send(engine.project().clone());
            }
            WorkerMsg::AgentApplyPlan { steps, reply } => {
                let _ = reply.send(agent_apply_and_publish(engine, steps, &ui));
            }
            WorkerMsg::Export(request) => start_export(engine, &ui, &export_state, request),
            WorkerMsg::CancelExport => {
                info!("export cancel requested");
                export_state.cancel.store(true, Ordering::Relaxed);
            }
            WorkerMsg::Frame(_) => unreachable!("frames are handled by the drain below"),
            WorkerMsg::TransformOverride { .. } => {
                unreachable!("overrides are handled by the drain below")
            }
        }
    };

    while let Ok(msg) = req_rx.recv() {
        match msg {
            WorkerMsg::Frame(mut tick) => {
                while let Ok(next) = req_rx.try_recv() {
                    match next {
                        WorkerMsg::Frame(latest) => tick = latest,
                        WorkerMsg::TransformOverride {
                            clip,
                            transform,
                            tick: at,
                        } => {
                            apply_transform_override(engine, &clip, transform);
                            tick = at;
                        }
                        other => {
                            mutate(engine, &mut clipboard, &mut main_magnet, &mut linkage, &mut autosave_slot, other)
                        }
                    }
                }
                last_tick = tick;
                render_frame(engine, tl_rate, &preview_weak, tick);
                prefetch_ahead(engine, tl_rate, tick, &req_rx);
            }
            // Drag-gesture overrides arrive at pointer-move rate; render only
            // the newest one (same coalescing as scrub frames) so a fast drag
            // can't back the queue up behind stale composites.
            WorkerMsg::TransformOverride {
                mut clip,
                mut transform,
                mut tick,
            } => {
                // Queue order must hold against drained mutations: the
                // release's SetTransform often lands right behind the last
                // pointer-move, and it commits + clears the override. Apply
                // the coalesced override *before* such a mutation and never
                // after it, or the stale gesture override outlives the commit
                // and pins the clip's transform on every later frame
                // (keyframed animation freezes in preview until re-cleared).
                let mut pending = true;
                while let Ok(next) = req_rx.try_recv() {
                    match next {
                        WorkerMsg::Frame(latest) => tick = latest,
                        WorkerMsg::TransformOverride {
                            clip: c,
                            transform: t,
                            tick: at,
                        } => {
                            clip = c;
                            transform = t;
                            tick = at;
                            pending = true;
                        }
                        other => {
                            if std::mem::take(&mut pending) {
                                apply_transform_override(engine, &clip, transform);
                            }
                            mutate(engine, &mut clipboard, &mut main_magnet, &mut linkage, &mut autosave_slot, other)
                        }
                    }
                }
                last_tick = tick;
                if pending {
                    apply_transform_override(engine, &clip, transform);
                }
                render_frame(engine, tl_rate, &preview_weak, tick);
            }
            // Live inspector edits (font-size drag) arrive at pointer-move
            // rate; coalesce to the newest like transform overrides do.
            WorkerMsg::GeneratorOverride {
                mut clip,
                mut generator,
                mut tick,
            } => {
                // Same ordering rule as TransformOverride above: a drained
                // mutation (the release's SetGenerator / ClearGeneratorOverride)
                // must not be followed by a re-apply of the override it ended.
                let mut pending = true;
                while let Ok(next) = req_rx.try_recv() {
                    match next {
                        WorkerMsg::Frame(latest) => tick = latest,
                        WorkerMsg::GeneratorOverride {
                            clip: c,
                            generator: g,
                            tick: at,
                        } => {
                            clip = c;
                            generator = g;
                            tick = at;
                            pending = true;
                        }
                        other => {
                            if std::mem::take(&mut pending) {
                                apply_generator_override(engine, &clip, generator.clone());
                            }
                            mutate(engine, &mut clipboard, &mut main_magnet, &mut linkage, &mut autosave_slot, other)
                        }
                    }
                }
                last_tick = tick;
                if pending {
                    apply_generator_override(engine, &clip, generator);
                }
                render_frame(engine, tl_rate, &preview_weak, tick);
            }
            // Shape resize drags (width/height sliders) arrive at pointer-move
            // rate; coalesce to the newest like the generator/transform
            // overrides so a fast drag can't back the render queue up. The
            // override generator is rebuilt from committed engine state, so the
            // drained-mutation ordering rule (above) applies unchanged.
            WorkerMsg::PreviewShapeSize {
                mut clip,
                mut width,
                mut height,
                mut tick,
            } => {
                let mut pending = true;
                while let Ok(next) = req_rx.try_recv() {
                    match next {
                        WorkerMsg::Frame(latest) => tick = latest,
                        WorkerMsg::PreviewShapeSize {
                            clip: c,
                            width: w,
                            height: h,
                            tick: at,
                        } => {
                            clip = c;
                            width = w;
                            height = h;
                            tick = at;
                            pending = true;
                        }
                        other => {
                            if std::mem::take(&mut pending)
                                && let Some(generator) =
                                    shape_size_from_engine(engine, &clip, width, height)
                            {
                                apply_generator_override(engine, &clip, generator);
                            }
                            mutate(engine, &mut clipboard, &mut main_magnet, &mut linkage, &mut autosave_slot, other)
                        }
                    }
                }
                last_tick = tick;
                if pending && let Some(generator) =
                    shape_size_from_engine(engine, &clip, width, height)
                {
                    apply_generator_override(engine, &clip, generator);
                }
                render_frame(engine, tl_rate, &preview_weak, tick);
            }
            other => {
                let redraw = mutation_redraws_preview(&other);
                mutate(engine, &mut clipboard, &mut main_magnet, &mut linkage, &mut autosave_slot, other);
                // Edits otherwise only repaint when the playhead moves; refresh
                // the current frame so the change is visible immediately.
                if redraw {
                    render_frame(engine, tl_rate, &preview_weak, last_tick);
                }
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
/// (import, copy, save, autosave, export, linkage) don't alter the canvas.
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
            | WorkerMsg::RestoreAutosave { .. }
            // Relinked media decodes again — refresh the stale composite.
            | WorkerMsg::RelinkMedia { .. }
            | WorkerMsg::RelinkFolder { .. }
    )
}

/// Point the engine's session override at `clip` (raw id) for the next
/// renders. Unparsable ids are dropped — the gesture came from a projected
/// clip, so this only fires on a stale projection race.
fn apply_transform_override(engine: &mut Engine, clip: &str, transform: ClipTransform) {
    match parse_raw_id(clip).map(ClipId::from_raw) {
        Some(id) => engine.set_transform_override(Some((id, transform))),
        None => error!(clip, "transform override ignored: unparsable clip id"),
    }
}

/// Point the engine's generator override at `clip` (raw id) for the next
/// renders — the live preview of an uncommitted inspector edit. Unparsable
/// ids are dropped (stale projection race), same as the transform override.
fn apply_generator_override(engine: &mut Engine, clip: &str, generator: Generator) {
    match parse_raw_id(clip).map(ClipId::from_raw) {
        Some(id) => engine.set_generator_override(Some((id, generator))),
        None => error!(clip, "generator override ignored: unparsable clip id"),
    }
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
    let (canvas_w, canvas_h) = cutlass_engine::composite_canvas_size(project);
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
    let sampled = clip
        .transform
        .sample_at(clip.animation_tick_f(tick as f64));
    Some(ClipTransform {
        position: [0.0, 0.0],
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

/// Playback read-ahead (playback roadmap Phase 2): with the queue idle after
/// a rendered frame, warm the decode/cache path for the next few ticks so a
/// GOP boundary's decode spike is paid *before* the playback cadence reaches
/// it. Stops the instant a new message arrives (the real request supersedes
/// the guess) and at the sequence end; a wrong guess (about-to-seek, reverse
/// shuttle) only warms the cache.
const READ_AHEAD_TICKS: i64 = 4;

fn prefetch_ahead(
    engine: &mut Engine,
    tl_rate: Rational,
    tick: i64,
    req_rx: &Receiver<WorkerMsg>,
) {
    let end = engine.project().timeline().duration().value;
    for ahead in 1..=READ_AHEAD_TICKS {
        let target = tick + ahead;
        if target >= end || !req_rx.is_empty() {
            return;
        }
        // Failures are expected (gap in the timeline, mid-edit churn) and
        // the real request will surface them if they matter.
        let _ = engine.prefetch(RationalTime::new(target, tl_rate));
    }
}

fn import_and_publish(
    engine: &mut Engine,
    path: &Path,
    ui: &UiSink,
    thumbs: &ThumbnailHandle,
    strips: &StripHandle,
) {
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
            // Kick off tile thumbnail generation off-thread; the tile shows
            // its placeholder until the image lands (see src/thumbnails.rs).
            if let Some(source) = engine.project().media(media) {
                register_media_with_workers(source, thumbs, strips);
            }
            publish_projection(engine, ui);
        }
        Ok(other) => error!(path = %path.display(), "unexpected import outcome: {other:?}"),
        Err(e) => error!(path = %path.display(), "import failed: {e}"),
    }
}

/// Write the session to `path` — or the engine's current project path when
/// `None` (plain save on an already-saved project). Success republishes the
/// projection, which is what clears the title bar's dirty dot; failure
/// publishes `session-error` and leaves the dot on (honest: the file on
/// disk is still stale). Either way `save-finished(ok)` fires so a pending
/// guarded transition in main.rs can continue or abort. A `None` path with
/// no current path is a UI gating bug, not a user state.
fn save_project_and_publish(engine: &mut Engine, path: Option<PathBuf>, ui: &UiSink) {
    let Some(path) = path.or_else(|| engine.project_path().cloned()) else {
        error!("save requested with no target path and no current project path");
        notify_save_finished(ui, false);
        return;
    };
    match engine.apply(Command::Project(ProjectCommand::Save { path: path.clone() })) {
        Ok(ApplyOutcome::Saved) => {
            info!(path = %path.display(), "project saved");
            note_recent_project(&path, ui);
            publish_projection(engine, ui);
            notify_save_finished(ui, true);
        }
        Ok(other) => {
            error!(path = %path.display(), "unexpected save outcome: {other:?}");
            notify_save_finished(ui, false);
        }
        Err(e) => {
            error!(path = %path.display(), "save failed: {e}");
            publish_session_error(ui, format!("Couldn't save the project to {}: {e}", path.display()));
            notify_save_finished(ui, false);
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
fn open_project_and_publish(
    engine: &mut Engine,
    path: PathBuf,
    ui: &UiSink,
    thumbs: &ThumbnailHandle,
    strips: &StripHandle,
) {
    match engine.apply(Command::Project(ProjectCommand::Load { path: path.clone() })) {
        Ok(ApplyOutcome::Loaded) => {
            info!(
                path = %path.display(),
                pool = engine.project().media_count(),
                "opened project"
            );
            for media in engine.project().media_iter() {
                if media.path().exists() {
                    register_media_with_workers(media, thumbs, strips);
                }
            }
            note_recent_project(&path, ui);
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
fn relink_media_and_publish(
    engine: &mut Engine,
    media: &str,
    path: &Path,
    ui: &UiSink,
    thumbs: &ThumbnailHandle,
    strips: &StripHandle,
) {
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
            if let Some(source) = engine.project().media(media) {
                register_media_with_workers(source, thumbs, strips);
            }
            publish_projection(engine, ui);
        }
        Ok(other) => error!(path = %path.display(), "unexpected relink outcome: {other:?}"),
        Err(e) => {
            error!(path = %path.display(), "relink failed: {e}");
            publish_session_error(
                ui,
                format!("Couldn't relink to {}: {e}", path.display()),
            );
        }
    }
}

/// Try `folder/<filename>` for every missing pool entry; relink each match.
fn relink_folder_and_publish(
    engine: &mut Engine,
    folder: PathBuf,
    ui: &UiSink,
    thumbs: &ThumbnailHandle,
    strips: &StripHandle,
) {
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
            Ok(ApplyOutcome::Relinked { media }) => {
                relinked += 1;
                if let Some(source) = engine.project().media(media) {
                    register_media_with_workers(source, thumbs, strips);
                }
            }
            Ok(other) => error!(path = %path.display(), "unexpected relink outcome: {other:?}"),
            Err(e) => error!(path = %path.display(), "folder relink failed: {e}"),
        }
    }

    if relinked > 0 {
        info!(count = relinked, folder = %folder.display(), "relinked media from folder");
        publish_projection(engine, ui);
    }
}

/// Replace the session with a fresh, empty, unsaved project (File → New).
/// The unsaved-changes guard ran UI-side; this is unconditional.
fn new_project_and_publish(engine: &mut Engine, ui: &UiSink) {
    engine.new_session();
    info!("new session");
    publish_projection(engine, ui);
    bump_session_epoch(ui);
}

/// One autosave sweep: snapshot a dirty session to its sidecar slot
/// (`~/.cutlass/autosave/`), never to the user's file. Clean session ⇒ any
/// existing slot is stale (just saved, or untouched) and gets removed. A
/// session whose identity changed since the last write (Save As / Open /
/// New) cleans its orphaned slot up first. `slot_state` carries the slot
/// path and the engine revision it captured, so a dirty-but-idle session
/// doesn't rewrite an identical snapshot every sweep. Failures only log:
/// autosave is invisible by contract.
fn autosave_sweep(engine: &Engine, slot_state: &mut Option<(PathBuf, u64)>) {
    let dir = crate::autosave::default_dir();
    let source = engine.project_path().map(PathBuf::as_path);
    let slot = crate::autosave::slot_for(&dir, source);

    if let Some((old, _)) = slot_state.take_if(|(old, _)| *old != slot) {
        crate::autosave::discard(&old);
    }
    if !engine.is_dirty() {
        crate::autosave::discard(&slot);
        *slot_state = None;
        return;
    }
    let revision = engine.revision();
    if slot_state.as_ref().is_some_and(|(_, rev)| *rev == revision) {
        return; // dirty but idle: the snapshot on disk is already current
    }
    if let Err(e) = std::fs::create_dir_all(&dir) {
        warn!(dir = %dir.display(), "autosave skipped: couldn't create dir: {e}");
        return;
    }
    match engine.project().save_to_file(&slot) {
        Ok(()) => {
            // Meta lands after the snapshot: a crash between the two writes
            // degrades to "no candidate", never to a mislabeled restore.
            if let Err(e) = crate::autosave::write_meta(&slot, source) {
                warn!(slot = %slot.display(), "autosave meta write failed: {e}");
                return;
            }
            *slot_state = Some((slot, revision));
        }
        Err(e) => warn!(slot = %slot.display(), "autosave failed: {e}"),
    }
}

/// Restore an accepted crash-recovery snapshot: tolerant load (missing
/// media entries survive, like `Load`), session bound to `source` — the
/// user's file, not the sidecar — and left dirty so the first Cmd+S writes
/// the recovered work where it belongs. Media that still exist on disk
/// re-register with the tile workers; the epoch bump resets UI session
/// state, same as an open.
fn restore_autosave_and_publish(
    engine: &mut Engine,
    autosave: PathBuf,
    source: Option<PathBuf>,
    slot_state: &mut Option<(PathBuf, u64)>,
    ui: &UiSink,
    thumbs: &ThumbnailHandle,
    strips: &StripHandle,
) {
    match engine.restore_session(&autosave, source) {
        Ok(()) => {
            info!(
                autosave = %autosave.display(),
                pool = engine.project().media_count(),
                "restored autosave"
            );
            for media in engine.project().media_iter() {
                if media.path().exists() {
                    register_media_with_workers(media, thumbs, strips);
                }
            }
            // The snapshot becomes the session's live slot (it already holds
            // this exact content); a pid-named orphan gets swept into the
            // current slot — and deleted — on the next dirty sweep.
            *slot_state = Some((autosave, engine.revision()));
            publish_projection(engine, ui);
            bump_session_epoch(ui);
        }
        Err(e) => {
            error!(autosave = %autosave.display(), "restore failed: {e}");
            publish_session_error(
                ui,
                format!("Couldn't restore the recovered project {}: {e}", autosave.display()),
            );
        }
    }
}

/// Register one pool media with the off-thread tile workers: a library
/// thumbnail render and the strip worker's id → path record (filmstrips /
/// waveforms resolve by media id alone). Shared by import and open.
fn register_media_with_workers(
    media: &cutlass_models::MediaSource,
    thumbs: &ThumbnailHandle,
    strips: &StripHandle,
) {
    let kind = match media.kind() {
        cutlass_models::MediaKind::Audio => ThumbKind::Audio,
        cutlass_models::MediaKind::Image => ThumbKind::Image,
        cutlass_models::MediaKind::Video => ThumbKind::Video,
    };
    thumbs.request(media.id.raw(), media.path().to_path_buf(), kind);
    // Stills register too: the strip sampler repeats the one picture across
    // the clip's filmstrip tiles.
    strips.register_media(media.id.raw(), media.path().to_path_buf());
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
///   at `start_tick`, shifting later clips right (atomic engine command);
/// - with `linkage` on, a video drop whose media carries audio also lands a
///   linked audio clip at the same tick (an existing unlocked audio lane
///   with the span free, else a fresh bottom lane) — one history entry for
///   the pair.
fn add_clip_and_publish(
    engine: &mut Engine,
    media: &str,
    track: &str,
    start_tick: i64,
    drop_row: i64,
    insert: bool,
    linkage: bool,
    ui: &UiSink,
) {
    let Some(media_id) = parse_raw_id(media).map(MediaId::from_raw) else {
        error!(media, "drop ignored: unparsable media id");
        return;
    };
    let Some((source, audio_only, has_audio)) = engine
        .project()
        .media(media_id)
        .map(|m| (m.full_range(), m.is_audio_only(), m.has_audio))
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
    // Mirror Project::add_clip's source→timeline resampling so first-fit and
    // the audio companion see the same extent the engine will validate.
    let duration_ticks = resample(source.duration, tl_rate).value.max(1);
    let wants_companion = linkage && !audio_only && has_audio;

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
                let linked = !wants_companion
                    || add_linked_audio(engine, clip, media_id, source, at, duration_ticks)
                        .map_err(|e| error!(%clip, "linked audio drop failed: {e}"))
                        .is_ok();
                if linked {
                    engine.commit_group();
                    info!(%clip, %lane, %media_id, at, "ripple-inserted clip from library drop");
                } else {
                    engine.rollback_group();
                }
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

    // One history entry per drop, even when it creates the landing lane(s)
    // and the linked audio companion.
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
            let linked = !wants_companion
                || add_linked_audio(engine, clip, media_id, source, start_value, duration_ticks)
                    .map_err(|e| error!(%clip, "linked audio drop failed: {e}"))
                    .is_ok();
            if linked {
                engine.commit_group();
                info!(
                    %clip, %track_id, %media_id,
                    start_tick = start_value,
                    desired,
                    "added clip from library drop"
                );
            } else {
                engine.rollback_group();
            }
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
        error!(?generator, "generated drop ignored: no lane kind for generator");
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
                error!(?generator, "generated drop failed creating {lane_kind:?} track: {e}");
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

/// Set a clip's audio mix (CapCut volume + fades, M1). Audio rides
/// audio-lane clips, so a video half of a linked pair routes to its
/// audio-lane link partners — the inspector edit lands where the sound is.
/// One history group; the republish re-snapshots the playback mixer, so the
/// change is audible within a block.
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
    let on_audio_lane = |engine: &Engine, id: ClipId| {
        let timeline = engine.project().timeline();
        timeline
            .track_of(id)
            .and_then(|t| timeline.track(t))
            .is_some_and(|t| t.kind == TrackKind::Audio)
    };
    let targets: Vec<ClipId> = if on_audio_lane(engine, clip_id) {
        vec![clip_id]
    } else {
        // Always follow linkage here: volume on the video half alone is
        // inaudible, so the edit must land on the audio companions.
        link_group_ids(engine, clip_id)
            .into_iter()
            .filter(|id| on_audio_lane(engine, *id))
            .collect()
    };
    if targets.is_empty() {
        warn!(%clip_id, "set-clip-audio ignored: no audio-lane clip to adjust");
        return;
    }

    let tl_rate = engine.project().timeline().frame_rate;
    let to_ticks = |seconds: f32| {
        let ticks =
            (f64::from(seconds) * f64::from(tl_rate.num) / f64::from(tl_rate.den)).round();
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
                track.kind == TrackKind::Audio
                    && track.duck_source
                    && Some(track.id) != music_track
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

/// Set the project canvas settings (M1): aspect preset + background color
/// in one undoable history entry. An out-of-range preset index falls back
/// to auto (defensive — the dialog's list is index-aligned with the model).
fn set_canvas_and_publish(engine: &mut Engine, aspect_index: i32, background: [u8; 3], ui: &UiSink) {
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
    if let Err(e) = engine.apply(Command::Edit(EditCommand::RemoveTransition { clip: clip_id })) {
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
        Generator::Shape { shape, rgba, .. } => Some(Generator::Shape {
            shape: *shape,
            rgba: *rgba,
            width: width.max(1.0),
            height: height.max(1.0),
        }),
        _ => None,
    }
}

/// Replace a generated clip's content (inspector title edit). One history
/// entry per committed edit; the engine rejects non-generated clips.
fn set_generator_and_publish(engine: &mut Engine, clip: &str, generator: Generator, ui: &UiSink) {
    // A live font-size drag may have left an override in place; the commit is
    // the authoritative value, so clear it (the next render is identical — no
    // flicker between drag end and commit, mirroring `SetTransform`).
    engine.set_generator_override(None);
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

/// Land the audio companion of a linked drop: the same source range at the
/// same tick, on the topmost unlocked audio lane with the span free (a new
/// bottom lane when none has room), then link the pair. Runs inside the
/// drop's history group.
fn add_linked_audio(
    engine: &mut Engine,
    video_clip: ClipId,
    media_id: MediaId,
    source: TimeRange,
    start_tick: i64,
    duration_ticks: i64,
) -> Result<(), String> {
    let tl_rate = engine.project().timeline().frame_rate;
    // UI rows show the stack top-first, so scanning the order back-to-front
    // prefers the audio lane closest to the video lanes.
    let lane = {
        let timeline = engine.project().timeline();
        timeline.order().iter().rev().copied().find(|id| {
            timeline.track(*id).is_some_and(|t| {
                t.kind == TrackKind::Audio
                    && !t.locked
                    && span_free(t, start_tick, duration_ticks)
            })
        })
    };
    let lane = match lane {
        Some(lane) => lane,
        // drop_row == lane count ⇒ stack index 0 ⇒ bottom lane in the UI.
        None => {
            let bottom = engine.project().timeline().order().len() as i64;
            create_track(engine, TrackKind::Audio, bottom)?
        }
    };

    let audio_clip = match engine.apply(Command::Edit(EditCommand::AddClip {
        track: lane,
        media: media_id,
        source,
        start: RationalTime::new(start_tick, tl_rate),
    })) {
        Ok(ApplyOutcome::Edited(EditOutcome::Created(id))) => id,
        Ok(other) => return Err(format!("unexpected audio add outcome: {other:?}")),
        Err(e) => return Err(e.to_string()),
    };
    apply_edit(engine, EditCommand::LinkClips {
        clips: vec![video_clip, audio_clip],
    })?;
    info!(%video_clip, %audio_clip, %lane, start_tick, "linked audio companion");
    Ok(())
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

    apply_edit(engine, EditCommand::MoveClip {
        clip: clip_id,
        to_track: track,
        start: RationalTime::new(park, tl_rate),
    })?;
    // Both shifts also carry the parked clip along (its start stays past the
    // rest of the lane), so it never collides with the clips in between.
    apply_edit(engine, EditCommand::ShiftClips {
        track,
        from: placed.start,
        delta: RationalTime::new(-duration, tl_rate),
    })?;
    apply_edit(engine, EditCommand::ShiftClips {
        track,
        from: RationalTime::new(at, tl_rate),
        delta: RationalTime::new(duration, tl_rate),
    })?;
    apply_edit(engine, EditCommand::MoveClip {
        clip: clip_id,
        to_track: track,
        start: RationalTime::new(at, tl_rate),
    })
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

    apply_edit(engine, EditCommand::ShiftClips {
        track: to_track,
        from: RationalTime::new(at, tl_rate),
        delta: RationalTime::new(duration, tl_rate),
    })?;
    apply_edit(engine, EditCommand::MoveClip {
        clip: clip_id,
        to_track,
        start: RationalTime::new(at, tl_rate),
    })?;
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
fn apply_ripple_trim(
    engine: &mut Engine,
    clip: ClipId,
    timeline: TimeRange,
) -> Result<(), String> {
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
            apply_edit(engine, EditCommand::ShiftClips {
                track,
                from: old.start,
                delta: RationalTime::new(delta_dur, tl_rate),
            })?;
            apply_edit(engine, EditCommand::TrimClip {
                clip,
                timeline: TimeRange::at_rate(old.start.value, timeline.duration.value, tl_rate),
            })
        } else {
            // Shrink: trim to the resolved extent (a gap opens at the old
            // start), then slide the clip and downstream left into it.
            apply_edit(engine, trim)?;
            apply_edit(engine, EditCommand::ShiftClips {
                track,
                from: timeline.start,
                delta: RationalTime::new(old.start.value - timeline.start.value, tl_rate),
            })
        }
    } else if delta_dur > 0 {
        // Trailing grow: push downstream right, then extend into the hole.
        apply_edit(engine, EditCommand::ShiftClips {
            track,
            from: RationalTime::new(old.end_tick(), tl_rate),
            delta: RationalTime::new(delta_dur, tl_rate),
        })?;
        apply_edit(engine, trim)
    } else if delta_dur < 0 {
        // Trailing shrink: pull the edge in, then close the gap behind it.
        apply_edit(engine, trim)?;
        apply_edit(engine, EditCommand::ShiftClips {
            track,
            from: RationalTime::new(old.end_tick(), tl_rate),
            delta: RationalTime::new(delta_dur, tl_rate),
        })
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

/// Shared flags between the worker loop and the export thread. `active`
/// gates one-job-at-a-time (the export thread clears it when it exits);
/// `cancel` is reset at job start and set by [`WorkerMsg::CancelExport`].
#[derive(Default)]
struct ExportJobState {
    active: Arc<AtomicBool>,
    cancel: Arc<AtomicBool>,
}

/// One snapshot of the export job for the Slint `ExportBackend` global.
#[derive(Default)]
struct ExportUiState {
    running: bool,
    done: u64,
    total: u64,
    completed: bool,
    failed: bool,
    status: String,
}

fn publish_export_state(weak: &slint::Weak<ExportBackend<'static>>, state: ExportUiState) {
    let weak = weak.clone();
    if let Err(e) = slint::invoke_from_event_loop(move || {
        if let Some(backend) = weak.upgrade() {
            backend.set_running(state.running);
            backend.set_frames_done(state.done.min(i32::MAX as u64) as i32);
            backend.set_frames_total(state.total.min(i32::MAX as u64) as i32);
            backend.set_progress(if state.total > 0 {
                (state.done as f32 / state.total as f32).clamp(0.0, 1.0)
            } else {
                0.0
            });
            backend.set_completed(state.completed);
            backend.set_failed(state.failed);
            backend.set_status(state.status.into());
        }
    }) {
        error!("failed to publish export state to UI: {e}");
    }
}

/// Snapshot the project and run the export on a dedicated thread: decode +
/// GPU composite + encode would otherwise freeze preview and edits for the
/// whole render. The thread owns its own GPU context and decoder pool
/// (`export_project_with`), publishes progress to the UI at most ~10×/sec,
/// and tears the `active` gate down when it exits — whatever the outcome.
fn start_export(engine: &Engine, ui: &UiSink, state: &ExportJobState, request: ExportRequest) {
    if state.active.swap(true, Ordering::SeqCst) {
        warn!("export refused: a job is already running");
        return;
    }
    state.cancel.store(false, Ordering::SeqCst);

    let project = engine.project().clone();
    let color_convert = engine.config().color_convert;
    let settings = ExportSettings {
        target_height: request.target_height,
        fps: request.fps_num.map(|num| Rational::new(num, 1)),
        quality: Some(request.crf),
    };
    let export_weak = ui.export.clone();
    let active = state.active.clone();
    let cancel = state.cancel.clone();
    let path = request.path;

    publish_export_state(&export_weak, ExportUiState {
        running: true,
        ..Default::default()
    });

    let spawned = std::thread::Builder::new()
        .name("cutlass-export".into())
        .spawn(move || {
            info!(path = %path.display(), "export job started");
            let weak = export_weak.clone();
            let mut last_publish = Instant::now();
            let mut published_once = false;
            let result = cutlass_engine::export_project_with(
                &project,
                &path,
                color_convert,
                settings,
                &mut |done, total| {
                    if cancel.load(Ordering::Relaxed) {
                        return false;
                    }
                    // Throttle event-loop traffic, but always deliver the
                    // first call (the dialog learns the total) and the last.
                    if !published_once
                        || done == total
                        || last_publish.elapsed() >= Duration::from_millis(100)
                    {
                        published_once = true;
                        last_publish = Instant::now();
                        publish_export_state(&weak, ExportUiState {
                            running: true,
                            done,
                            total,
                            ..Default::default()
                        });
                    }
                    true
                },
            );

            let outcome = match result {
                Ok(stats) => {
                    info!(
                        frames = stats.frames,
                        width = stats.width,
                        height = stats.height,
                        path = %path.display(),
                        "export job finished"
                    );
                    ExportUiState {
                        done: stats.frames,
                        total: stats.frames,
                        completed: true,
                        status: format!(
                            "Saved {}×{}, {} frames to {}",
                            stats.width,
                            stats.height,
                            stats.frames,
                            path.display()
                        ),
                        ..Default::default()
                    }
                }
                Err(EngineError::ExportCancelled) => {
                    // The half-written file is junk; don't leave it behind.
                    let _ = std::fs::remove_file(&path);
                    info!(path = %path.display(), "export job cancelled");
                    ExportUiState {
                        failed: true,
                        status: "Export cancelled".into(),
                        ..Default::default()
                    }
                }
                Err(e) => {
                    error!(path = %path.display(), "export job failed: {e}");
                    ExportUiState {
                        failed: true,
                        status: format!("Export failed: {e}"),
                        ..Default::default()
                    }
                }
            };
            publish_export_state(&weak, outcome);
            active.store(false, Ordering::SeqCst);
        });

    if let Err(e) = spawned {
        error!("failed to spawn export thread: {e}");
        state.active.store(false, Ordering::SeqCst);
        publish_export_state(&ui.export, ExportUiState {
            failed: true,
            status: format!("Export failed to start: {e}"),
            ..Default::default()
        });
    }
}

/// Remove every clip in `clips`; lanes the removals empty are removed with
/// them (CapCut deletes emptied overlay tracks — same policy the drag-moves
/// use). With the main-track magnet on, main-lane deletions ripple their
/// gaps closed. Everything forms one history group: one undo restores the
/// whole selection.
fn remove_clips_and_publish(
    engine: &mut Engine,
    clips: &[String],
    main_magnet: bool,
    ui: &UiSink,
) {
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
        && let Err(e) = apply_edit(engine, EditCommand::LinkClips { clips: tails.clone() })
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
fn move_group_and_publish(
    engine: &mut Engine,
    moves: &[GroupMove],
    ui: &UiSink,
) {
    // Resolve raw ids up front; any stale entry voids the batch.
    let mut resolved = Vec::with_capacity(moves.len());
    for entry in moves {
        let Some(clip_id) = parse_raw_id(&entry.clip).map(ClipId::from_raw) else {
            error!(clip = entry.clip, "group move ignored: unparsable clip id");
            return;
        };
        let Some(to_track) = parse_raw_id(&entry.track).map(TrackId::from_raw) else {
            error!(track = entry.track, "group move ignored: unparsable track id");
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
        if let Err(e) = apply_edit(engine, EditCommand::MoveClip {
            clip: clip_id,
            to_track,
            start: RationalTime::new(park, tl_rate),
        }) {
            error!(%clip_id, %to_track, "group move failed parking: {e}");
            engine.rollback_group();
            publish_projection(engine, ui);
            return;
        }
        park += duration;
    }
    for &(clip_id, to_track, _, start_tick) in &resolved {
        if let Err(e) = apply_edit(engine, EditCommand::MoveClip {
            clip: clip_id,
            to_track,
            start: RationalTime::new(start_tick, tl_rate),
        }) {
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
fn history_step_and_publish(
    engine: &mut Engine,
    redo: bool,
    ui: &UiSink,
) {
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
        let id = add_clip_content(engine, track, &member.content, member.duration_ticks, start + dx)?;
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
            let result = apply_edit(engine, EditCommand::ShiftClips {
                track,
                from: RationalTime::new(start, tl_rate),
                delta: RationalTime::new(duration, tl_rate),
            })
            .and_then(|_| add_clip_content(engine, track, &only.content, only.duration_ticks, start));
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
        .map(|member| (lanes[&member.track], tick.max(0) + member.offset_ticks, member))
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
        if let Err(e) = apply_edit(engine, EditCommand::LinkClips { clips: vec![*member] }) {
            error!(%member, "unlink failed: {e}");
            engine.rollback_group();
            publish_projection(engine, ui);
            return;
        }
    }
    engine.commit_group();
    info!(groups = links.len(), members = members.len(), "unlinked clip groups");
    publish_projection(engine, ui);
}

/// Place a copy of `clip` immediately after it on its own lane (first gap
/// that fits from the clip's end). With the main-track magnet on, a main-lane
/// duplicate ripple-inserts right after the original, shifting later clips.
fn duplicate_clip_and_publish(
    engine: &mut Engine,
    clip: &str,
    main_magnet: bool,
    ui: &UiSink,
) {
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
        let result = apply_edit(engine, EditCommand::ShiftClips {
            track,
            from: RationalTime::new(end_tick, tl_rate),
            delta: RationalTime::new(duration_ticks, tl_rate),
        })
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
fn pack_main_track_and_publish(
    engine: &mut Engine,
    ui: &UiSink,
) {
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
            if let Err(e) = apply_edit(engine, EditCommand::ShiftClips {
                track,
                from: RationalTime::new(current, tl_rate),
                delta: RationalTime::new(expected - current, tl_rate),
            }) {
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
    timeline
        .order()
        .iter()
        .copied()
        .find(|id| timeline.track(*id).is_some_and(|t| t.kind == TrackKind::Video))
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

/// Apply a single edit command, flattening the outcome — for compositions
/// where only success/failure matters (the group publishes once at the end).
/// Replay a rehearsed agent plan (see `src/agent.rs`) on the live engine:
/// one history group, re-validated step by step, with sandbox-allocated
/// ids remapped onto the ids the live engine hands out. `after_step` runs
/// after every applied step (the worker publishes there, so the user
/// watches the plan land) and after the rollback/commit. Any failure rolls
/// the whole group back — the project changed mid-prompt is the only way
/// a rehearsed step can fail here.
pub(crate) fn agent_replay(
    engine: &mut Engine,
    steps: Vec<AgentPlanStep>,
    mut after_step: impl FnMut(&mut Engine),
) -> Result<(), String> {
    use std::collections::HashMap as Map;
    let mut clip_map: Map<u64, u64> = Map::new();
    let mut track_map: Map<u64, u64> = Map::new();
    let mut marker_map: Map<u64, u64> = Map::new();

    let total = steps.len();
    engine.begin_group();
    for (index, mut step) in steps.into_iter().enumerate() {
        step.command.remap_ids(&clip_map, &track_map, &marker_map);
        let outcome = cutlass_ai::validate(&step.command, engine.project())
            .map_err(|r| r.message)
            .and_then(|lowered| engine.apply(lowered).map_err(|e| e.to_string()));
        match outcome {
            Ok(ApplyOutcome::Edited(edited)) => {
                match (step.created, &edited) {
                    (Some(AgentCreated::Clip(sandbox)), EditOutcome::Created(live)) => {
                        clip_map.insert(sandbox, live.raw());
                    }
                    (Some(AgentCreated::Track(sandbox)), EditOutcome::CreatedTrack(live)) => {
                        track_map.insert(sandbox, live.raw());
                    }
                    (Some(AgentCreated::Marker(sandbox)), EditOutcome::CreatedMarker(live)) => {
                        marker_map.insert(sandbox, live.raw());
                    }
                    _ => {}
                }
                after_step(engine);
            }
            Ok(other) => {
                engine.rollback_group();
                after_step(engine);
                return Err(format!(
                    "step {}/{total}: unexpected engine outcome {other:?}",
                    index + 1
                ));
            }
            Err(reason) => {
                engine.rollback_group();
                after_step(engine);
                return Err(format!("step {}/{total}: {reason}", index + 1));
            }
        }
    }
    engine.commit_group();
    info!(steps = total, "agent plan applied");
    after_step(engine);
    Ok(())
}

fn agent_apply_and_publish(
    engine: &mut Engine,
    steps: Vec<AgentPlanStep>,
    ui: &UiSink,
) -> Result<(), String> {
    agent_replay(engine, steps, |engine| publish_projection(engine, ui))
}

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
/// The audio mixer gets its own snapshot from the same chokepoint, so what
/// playback *sounds* like always matches the project the UI shows (mute
/// toggles, trims, moves — every mutation lands here).
fn publish_projection(engine: &mut Engine, ui: &UiSink) {
    ui.audio.publish_snapshot(audio_snapshot(engine));

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

/// Record `path` at the front of the recent-projects MRU (lifecycle
/// roadmap Phase 3) and push the refreshed list to
/// `EditorStore.recent-projects`. Called on every successful save and
/// open — the moments a `.cutlass` path is proven real and current.
fn note_recent_project(path: &Path, ui: &UiSink) {
    let entries = crate::recent::note(&crate::recent::default_path(), path);
    let editor_weak = ui.editor.clone();
    if let Err(e) = slint::invoke_from_event_loop(move || {
        if let Some(store) = editor_weak.upgrade() {
            store.set_recent_projects(slint::ModelRc::new(slint::VecModel::from(
                crate::recent::to_rows(&entries),
            )));
        }
    }) {
        error!("failed to publish recent projects: {e}");
    }
}

/// Fire `EditorStore.save-finished(ok)` — a Rust→Rust completion signal:
/// main.rs handles it to continue (or abort) a guarded transition waiting
/// on "Save". Plain saves fire it too; with nothing pending it's a no-op.
fn notify_save_finished(ui: &UiSink, ok: bool) {
    let editor_weak = ui.editor.clone();
    if let Err(e) = slint::invoke_from_event_loop(move || {
        if let Some(store) = editor_weak.upgrade() {
            store.invoke_save_finished(ok);
        }
    }) {
        error!("failed to notify save completion: {e}");
    }
}

/// Drawn-content size (canvas px) for every generated clip, keyed by raw clip
/// id. Rides the projection so the preview's selection box and hit-test hug
/// what the generator actually draws instead of its full-canvas raster (see
/// `src/preview_select.rs`). Served from the engine's raster cache; clips the
/// compositor doesn't draw are absent (the UI falls back to canvas size).
fn generator_content_sizes(engine: &mut Engine) -> HashMap<u64, (i32, i32)> {
    let generators: Vec<(u64, Generator)> = engine
        .project()
        .timeline()
        .tracks_ordered()
        .flat_map(|track| track.clips())
        .filter_map(|clip| match &clip.content {
            ClipSource::Generated(generator) => Some((clip.id.raw(), generator.clone())),
            ClipSource::Media { .. } => None,
        })
        .collect();
    generators
        .into_iter()
        .filter_map(|(id, generator)| {
            let (w, h) = engine.generator_content_size(&generator)?;
            Some((id, (w as i32, h as i32)))
        })
        .collect()
}

/// Every audible clip on the timeline, in rational time: clips on unmuted
/// audio lanes whose media carries an audio stream. Video lanes contribute
/// no sound — imports land a linked audio companion for that (linkage), so
/// audio is always *on* audio lanes, CapCut-style.
fn audio_snapshot(engine: &Engine) -> AudioSnapshot {
    let project = engine.project();
    let timeline = project.timeline();
    let fps = timeline.frame_rate;
    let mut spans = Vec::new();
    for track in timeline.tracks_ordered() {
        if track.kind != TrackKind::Audio || track.muted {
            continue;
        }
        for clip in track.clips_ordered() {
            // Constant-zero clips are silent either way. Retimed clips —
            // constant speed, reverse, and now speed ramps (M2) — all
            // time-stretch (M8 Phase 3); the export mixer matches, so what you
            // hear is what you ship.
            if clip.is_silent() {
                continue;
            }
            let Some(media_id) = clip.media() else {
                continue;
            };
            let Some(media) = project.media(media_id) else {
                continue;
            };
            if !media.has_audio {
                continue;
            }
            let Some(source) = clip.source_range() else {
                continue;
            };
            spans.push(AudioSpan {
                path: media.path().to_path_buf(),
                start_tick: clip.timeline.start.value,
                end_tick: clip.timeline.end_tick(),
                source_start: source.start.value,
                source_rate: (source.start.rate.num, source.start.rate.den),
                source_duration: source.duration.value,
                retimed: clip.is_retimed(),
                reversed: clip.reversed,
                pitch_factor: clip.audio_pitch_factor(),
                speed_curve: clip.has_speed_curve().then(|| clip.speed_curve.clone()),
                volume: clip.volume.clone(),
                fade_in_ticks: clip.fade_in,
                fade_out_ticks: clip.fade_out,
            });
        }
    }
    AudioSnapshot {
        fps: (fps.num, fps.den),
        spans,
    }
}

fn render_frame(
    engine: &mut Engine,
    tl_rate: Rational,
    preview_weak: &slint::Weak<PreviewStore<'static>>,
    tick: i64,
) {
    match engine.get_frame(RationalTime::new(tick, tl_rate)) {
        Ok(frame) => {
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

    /// `keyframes_at` slices one merged timeline diamond: only the
    /// properties keyframed exactly at the tick, each with its own value
    /// and easing, position as vec2.
    #[test]
    fn keyframes_at_collects_per_property_hits() {
        let mut t = AnimatedTransform::identity();
        t.set_param_keyframe(ClipParam::Scale, 10, ParamValue::Scalar(2.0), Easing::EaseIn)
            .unwrap();
        t.set_param_keyframe(ClipParam::Scale, 20, ParamValue::Scalar(3.0), Easing::Linear)
            .unwrap();
        t.set_param_keyframe(
            ClipParam::Position,
            10,
            ParamValue::Vec2([0.1, -0.2]),
            Easing::Linear,
        )
        .unwrap();
        t.set_param_keyframe(ClipParam::Opacity, 30, ParamValue::Scalar(0.5), Easing::Linear)
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
    fn trim_test_engine() -> (tempfile::TempDir, Engine, MediaId) {
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
        let dir = tempfile::tempdir().expect("tempdir");
        let config = EngineConfig {
            cache_dir: dir.path().join("cache"),
            cache_budget_bytes: 16 * 1024 * 1024,
            ..EngineConfig::default()
        };
        let engine = Engine::with_project(config, project).expect("engine");
        (dir, engine, media)
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
        let (_dir, mut engine, media) = trim_test_engine();
        let track = add_video_track(&mut engine, "V1");
        let a = add_media_clip(&mut engine, track, media, 0, 50);
        let b = add_media_clip(&mut engine, track, media, 50, 30);
        let c = add_media_clip(&mut engine, track, media, 80, 40);

        commit_trims(
            &mut engine,
            &[(
                b,
                TimeRange::at_rate(50, 20, Rational::FPS_24),
            )],
            true,
        )
        .expect("ripple tail shrink");

        assert_eq!(clip_starts(&engine, &[a, b, c]), [0, 50, 70]);
        assert!(engine.undo());
        assert_eq!(clip_starts(&engine, &[a, b, c]), [0, 50, 80]);
    }

    #[test]
    fn ripple_tail_grow_opens_room_before_extending() {
        let (_dir, mut engine, media) = trim_test_engine();
        let track = add_video_track(&mut engine, "V1");
        let a = add_media_clip(&mut engine, track, media, 0, 50);
        let b = add_media_clip(&mut engine, track, media, 50, 30);

        commit_trims(
            &mut engine,
            &[(
                a,
                TimeRange::at_rate(0, 60, Rational::FPS_24),
            )],
            true,
        )
        .expect("ripple tail grow");

        assert_eq!(clip_starts(&engine, &[a, b]), [0, 60]);
    }

    #[test]
    fn ripple_head_shrink_reanchors_at_old_start() {
        let (_dir, mut engine, media) = trim_test_engine();
        let track = add_video_track(&mut engine, "V1");
        let a = add_media_clip(&mut engine, track, media, 0, 50);
        let b = add_media_clip(&mut engine, track, media, 50, 30);

        commit_trims(
            &mut engine,
            &[(
                a,
                TimeRange::at_rate(10, 40, Rational::FPS_24),
            )],
            true,
        )
        .expect("ripple head shrink");

        assert_eq!(clip_starts(&engine, &[a, b]), [0, 40]);
    }

    #[test]
    fn plain_trim_off_magnet_leaves_gap() {
        let (_dir, mut engine, media) = trim_test_engine();
        let track = add_video_track(&mut engine, "V1");
        let a = add_media_clip(&mut engine, track, media, 0, 50);
        let b = add_media_clip(&mut engine, track, media, 50, 30);
        let c = add_media_clip(&mut engine, track, media, 80, 40);

        commit_trims(
            &mut engine,
            &[(
                b,
                TimeRange::at_rate(50, 20, Rational::FPS_24),
            )],
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
    fn ripple_fixture() -> (tempfile::TempDir, Engine, [ClipId; 3], TrackId) {
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
            .add_clip(track, media, TimeRange::at_rate(100, 100, r), RationalTime::new(0, r))
            .expect("clip A");
        let b = project
            .add_clip(track, media, TimeRange::at_rate(300, 100, r), RationalTime::new(100, r))
            .expect("clip B");
        let c = project
            .add_clip(track, media, TimeRange::at_rate(500, 100, r), RationalTime::new(200, r))
            .expect("clip C");

        let dir = tempfile::tempdir().expect("tempdir");
        let config = EngineConfig {
            cache_dir: dir.path().join("cache"),
            cache_budget_bytes: 16 * 1024 * 1024,
            ..EngineConfig::default()
        };
        let engine = Engine::with_project(config, project).expect("engine");
        (dir, engine, [a, b, c], track)
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
        let (_dir, mut engine, [a, b, c], _track) = ripple_fixture();

        let rippled = commit_trims(&mut engine, &[(b, tr24(120, 80))], true)
            .expect("ripple head shrink");
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
        let (_dir, mut engine, [a, b, c], _track) = ripple_fixture();

        let rippled = commit_trims(&mut engine, &[(b, tr24(50, 150))], true)
            .expect("ripple head grow");
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
        let (_dir, mut engine, [a, b, c], _track) = ripple_fixture();

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
        let (_dir, mut engine, [a, b, c], _track) = ripple_fixture();

        commit_trims(&mut engine, &[(c, tr24(200, 40))], true).expect("ripple last clip");
        assert_eq!(extent(&engine, c), (200, 40));
        assert_eq!(extent(&engine, a), (0, 100));
        assert_eq!(extent(&engine, b), (100, 100));
    }

    /// The magnet only governs the main lane (bottom video track): an
    /// overlay-lane trim stays plain even with the magnet on.
    #[test]
    fn overlay_lane_trim_does_not_ripple() {
        let (_dir, mut engine, [a, _, _], _track) = ripple_fixture();
        let media = engine.project().clip(a).expect("clip").media().expect("media");
        let overlay = add_video_track(&mut engine, "V2");
        let d = add_media_clip(&mut engine, overlay, media, 0, 100);
        let e = add_media_clip(&mut engine, overlay, media, 100, 100);

        let rippled = commit_trims(&mut engine, &[(d, tr24(0, 60))], true)
            .expect("overlay trim");
        assert!(!rippled);
        assert_eq!(extent(&engine, d), (0, 60));
        assert_eq!(extent(&engine, e), (100, 100));
    }

    /// A trim the engine rejects (source bounds) rolls the whole group back:
    /// the shift that already opened room is undone, history stays untouched.
    #[test]
    fn rejected_ripple_rolls_back_whole_group() {
        let (_dir, mut engine, [a, b, c], _track) = ripple_fixture();

        // B's source is [300,400) of a 1000-tick media: growing the tail by
        // 700 ticks would need source up to 1100 — rejected after the shift.
        let result = commit_trims(&mut engine, &[(b, tr24(100, 800))], true);
        assert!(result.is_err());

        assert_eq!(extent(&engine, a), (0, 100));
        assert_eq!(extent(&engine, b), (100, 100));
        assert_eq!(extent(&engine, c), (200, 100));
        assert!(!engine.can_undo(), "rolled-back group must leave no history");
    }

    /// Linked-pair trim (video on the main lane, audio partner elsewhere):
    /// both members ripple on their own lanes, so the pair stays in sync and
    /// downstream clips on both lanes shift by the same delta.
    #[test]
    fn linked_pair_ripples_on_both_lanes() {
        let (_dir, mut engine, [_, b, c], _track) = ripple_fixture();
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
        let media = engine.project().clip(b).expect("clip B").media().expect("media");
        let add_audio_clip = |engine: &mut Engine, source: TimeRange, start: i64| {
            match engine
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
            }
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
