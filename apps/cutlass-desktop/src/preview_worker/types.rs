use super::*;

/// Everything a mutation publishes to: the Slint view model. The audio
/// mixer's snapshot rejoins in Phase 3 so sound can never diverge from the
/// projected timeline.
pub(super) struct UiSink {
    pub(super) editor: slint::Weak<EditorStore<'static>>,
    /// Export job progress/outcome lands here (from the export thread).
    pub(super) export: slint::Weak<ExportBackend<'static>>,
    /// Audio mixer inbox: every projection republish also publishes the
    /// project snapshot here, so mid-playback edits become audible (the
    /// mixer reopens over the new snapshot at its current position).
    pub(super) audio: crate::audio::AudioHandle,
    /// Off-thread tile workers: pool changes (import/open/relink) register
    /// media for library thumbnails and filmstrip/waveform decode.
    pub(super) thumbs: ThumbnailHandle,
    pub(super) strips: StripHandle,
    /// Preview-proxy generator: large video sources queue a background
    /// re-encode; results come back as [`WorkerMsg::ProxyReady`].
    pub(super) proxy: ProxyHandle,
}

pub struct PreviewSession {
    pub duration_ticks: i64,
    pub tl_rate: Rational,
}

/// Exact in-memory usage of the composited preview-frame cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct PreviewCacheStats {
    pub(crate) entries: usize,
    pub(crate) bytes: u64,
}

/// Result of an acknowledged media-pool import. The path is read back from
/// the engine after import, so callers receive its canonical/current value
/// rather than merely the path they requested.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ImportMediaRpcResult {
    pub(crate) media_id: u64,
    pub(crate) path: PathBuf,
}

/// Result of an acknowledged project save.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SaveProjectRpcResult {
    pub(crate) path: PathBuf,
    /// Always false. An `Ok` result is withheld if the engine remains dirty.
    pub(crate) dirty: bool,
}

/// One acknowledged media relink, also used by folder-relink results.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RelinkMediaRpcResult {
    pub(crate) media_id: u64,
    pub(crate) path: PathBuf,
}

/// Result of an acknowledged folder relink. Entries are sorted by raw media
/// id, independent of media-pool iteration order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RelinkFolderRpcResult {
    pub(crate) relinked: Vec<RelinkMediaRpcResult>,
}

/// Result of an acknowledged project open. `path` is read back from the
/// engine after the load, and therefore names the session's actual binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OpenProjectRpcResult {
    pub(crate) path: PathBuf,
    pub(crate) project_name: String,
    pub(crate) missing_media_count: usize,
}

/// Result of an acknowledged fresh-session replacement. A new session is
/// intentionally unbound until the host creates an app-owned draft and sends
/// a separate [`WorkerHandle::save_project_rpc`] in queue order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NewProjectRpcResult {
    pub(crate) path: Option<PathBuf>,
    pub(crate) project_name: String,
    pub(crate) missing_media_count: usize,
    pub(crate) requires_save_binding: bool,
}

/// Result of an acknowledged template application. An `Ok` result is only
/// returned after the filled in-memory project has been saved and the engine
/// reports this actual app-owned draft binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ApplyTemplateRpcResult {
    pub(crate) path: PathBuf,
    pub(crate) project_name: String,
    pub(crate) missing_media_count: usize,
}

/// Work submitted to the engine thread. Scrub frames coalesce to the latest
/// pending tick; imports must not be dropped by that coalescing (see
/// [`worker_loop`]).
pub(super) enum WorkerMsg {
    Frame(i64),
    /// The preview panel's on-screen size in physical pixels — the fit bound
    /// for every subsequent frame render (scrubbing pays for view-sized
    /// pixels, not the full canvas). Re-renders the current frame on change.
    Viewport {
        width: u32,
        height: u32,
    },
    Import(PathBuf),
    /// Queue-ordered, acknowledged counterpart to [`WorkerMsg::Import`].
    ImportMediaRpc {
        path: PathBuf,
        reply: Sender<Result<ImportMediaRpcResult, String>>,
        operation: Arc<WorkerRpcOperation>,
    },
    /// OS files dropped on the window (Finder / Explorer). With `target`
    /// — the drop landed on the timeline: (lane-list row, sequence tick)
    /// under the cursor — every file is imported and placed end-to-end from
    /// the drop point in one undo group: videos/images on a video lane
    /// (falling back to the empty main track, CapCut-style), audio-only
    /// files on an audio lane (the model's lane zones keep audio at the
    /// bottom). Without `target` the drop is a plain pool import, the same
    /// as [`WorkerMsg::Import`] per file.
    DropFiles {
        paths: Vec<PathBuf>,
        target: Option<(i64, i64)>,
    },
    /// A preview proxy for pool media `media_id` is ready at `proxy`
    /// (from the proxy worker thread). `source` is the file the job was
    /// keyed to; the handler binds the proxy only while the pool entry
    /// still names that exact path (media ids persist in project files and
    /// across relinks, so the id alone can go stale in flight).
    ProxyReady {
        media_id: u64,
        source: PathBuf,
        proxy: PathBuf,
    },
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
    /// Place a generated clip (text title, solid, shape, effect) at
    /// `start_tick` on `track` (raw id of a matching-kind lane), or create a
    /// lane of the generator's kind at `drop_row` when `track` is empty.
    /// Generated lanes are never the main track, so there's no ripple-insert
    /// path. `effect` seeds the new clip's effect chain — a standalone
    /// effect-lane segment dropped from the Effects catalog (CapCut's
    /// effect-as-track-clip). `animations` are `(slot, catalog id)` pairs a
    /// text-preset drop attaches to the fresh clip (unknown ids are skipped,
    /// never errors — a served preset must not brick the drop).
    AddGenerated {
        generator: Generator,
        track: String,
        start_tick: i64,
        duration_ticks: i64,
        drop_row: i64,
        effect: Option<String>,
        animations: Vec<(String, String)>,
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
    /// Delete every clip in `clips` and close each lane's gap (`RippleDelete`),
    /// regardless of the main-track magnet — the explicit "ripple delete"
    /// gesture. One history group.
    RippleDeleteClips {
        clips: Vec<String>,
    },
    /// Toggle reverse playback on a media clip: reads the clip's current
    /// speed and flips `reversed`. One undoable history entry.
    ReverseClip {
        clip: String,
    },
    /// CapCut "extract audio": place the video clip's sound on an audio lane
    /// (same media, no new library asset), link the pair, and tag the audio
    /// half as `Extracted`. The video goes silent via `carries_own_audio`.
    ExtractAudio {
        clip: String,
    },
    /// Replace a generated clip's content (raw id) — e.g. an inspector title
    /// edit. One undoable history entry per committed edit.
    SetGenerator {
        clip: String,
        generator: Generator,
    },
    /// Resize a shape clip's reference-pixel dimensions. Preserves shape kind
    /// and fill from the committed generator.
    SetShapeSize {
        clip: String,
        width: f32,
        height: f32,
    },
    /// Live shape-resize drag (width/height sliders): rebuild the generator
    /// from committed state at the new dimensions and ride the engine's
    /// generator override — no history entry until `SetShapeSize` commits.
    /// Coalesces to the newest like `Frame` so a fast drag can't back the
    /// queue up.
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
    /// Set a visual clip's blend mode (CapCut "Blend"). `mode` is a
    /// `BlendMode` id (e.g. `"multiply"`); unknown ids are ignored.
    /// One undoable history entry.
    SetBlendMode {
        clip: String,
        mode: String,
    },
    /// Replace a visual clip's layer styles (shadow/glow/outline/background).
    /// One undoable history entry.
    SetLayerStyles {
        clip: String,
        styles: LayerStyles,
    },
    /// Set (or clear) a visual clip's shaped alpha mask. `None` clears.
    /// One undoable history entry.
    SetMask {
        clip: String,
        mask: Option<Mask>,
    },
    /// Set (or clear) chroma keying on a visual clip. `None` clears.
    /// One undoable history entry.
    SetChroma {
        clip: String,
        chroma: Option<ChromaKey>,
    },
    /// Set (or clear) a visual clip's filter preset. `filter_id == ""`
    /// clears; intensity is normalized 0..=1. One undoable history entry.
    SetClipFilter {
        clip: String,
        filter_id: String,
        intensity: f32,
    },
    /// Set (or clear) a visual clip's `.cube` LUT. `path == ""` clears;
    /// intensity is normalized 0..=1. One undoable history entry.
    SetClipLut {
        clip: String,
        path: String,
        intensity: f32,
    },
    /// Set all five manual color adjustments in one undoable history entry.
    SetClipAdjust {
        clip: String,
        adjust: ColorAdjustments,
    },
    /// Save edited per-project agent rules into `ProjectMetadata`.
    /// Metadata, not a timeline command: dirties the session (rules save
    /// with the project) but is not undoable, like relink.
    SetAgentRules {
        rules: String,
    },
    /// Set (or clear) one look-animation slot on a visual clip.
    SetClipAnimation {
        clip: String,
        slot: String,
        animation_id: String,
        speed: f32,
        intensity: f32,
        stagger: f32,
    },
    /// Live color-grading preview: replace one clip's filter + adjustments
    /// through the engine's session-only look override. Bursts coalesce to
    /// the newest like transform/generator overrides.
    PreviewClipLook {
        clip: String,
        filter_id: String,
        intensity: f32,
        adjust: ColorAdjustments,
        tick: i64,
    },
    /// Live layer-styles preview: replace one clip's styles through the
    /// engine's session-only styles override. Bursts coalesce to the newest
    /// like look/transform/generator overrides.
    PreviewClipStyles {
        clip: String,
        styles: LayerStyles,
        tick: i64,
    },
    /// Drop the styles override (control released with no net change) and
    /// re-render `tick` from committed state. Look clears on commit only;
    /// styles also clear on SetLayerStyles / style param commits.
    ClearStylesOverride {
        tick: i64,
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
    /// Live drag override: render `tick` with `clip`'s transform replaced —
    /// session state on the engine, no history entry, no projection
    /// republish. Bursts coalesce to the newest value like `Frame` requests.
    TransformOverride {
        clip: String,
        transform: ClipTransform,
        tick: i64,
    },
    /// Render partitioned below/sprite/above frames once for a zero-drift
    /// transform gesture. On failure the per-move override path is used.
    BeginTransformGesture {
        clip: String,
        tick: i64,
    },
    /// Press ended without a drag: drop prepared sprite frames.
    EndTransformGesture,
    /// Drop the gesture override (no-op release / cancelled drag) and
    /// re-render `tick` from committed state.
    ClearTransformOverride {
        tick: i64,
    },
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
    ClearGeneratorOverride {
        tick: i64,
    },
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
    /// Replace one animatable property with a constant, dropping its
    /// keyframes. One undoable edit.
    SetParamConstant {
        clip: String,
        param: ClipParam,
        value: ParamValue,
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
    /// Move / rename / recolor a ruler marker. One undoable history entry.
    SetMarker {
        marker: String,
        at_tick: i64,
        name: String,
        color: String,
    },
    /// Remove a track explicitly (context menu), even when non-empty.
    RemoveTrackManual {
        track: String,
    },
    /// Reorder a track in the stack (0 = bottom layer).
    MoveTrackManual {
        track: String,
        index: usize,
    },
    /// Rename a track lane.
    SetTrackName {
        track: String,
        name: String,
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
    /// Start an export job: the worker clones the project and hands it to a
    /// dedicated export thread (fresh renderer, own GPU queue + decoders),
    /// which publishes progress into `ExportBackend`. One job at a time.
    Export(ExportRequest),
    /// Flag the running export job to stop after the frame in flight.
    CancelExport,
    /// Flush the live draft to its project file. `None` reuses the engine's
    /// current draft path (the normal case: explicit Cmd+S, or the debounce /
    /// session-swap / close flush); `Some` rebinds it (binding a freshly
    /// created draft after `NewProject`). Not undoable; on success the
    /// projection republish refreshes the title and the meta sidecar is
    /// rewritten so the gallery name tracks the project name.
    SaveProject {
        path: Option<PathBuf>,
    },
    /// Queue-ordered, acknowledged counterpart to
    /// [`WorkerMsg::SaveProject`].
    SaveProjectRpc {
        path: Option<PathBuf>,
        reply: Sender<Result<SaveProjectRpcResult, String>>,
        operation: Arc<WorkerRpcOperation>,
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
    /// Queue-ordered, acknowledged counterpart to
    /// [`WorkerMsg::OpenProject`].
    OpenProjectRpc {
        path: PathBuf,
        reply: Sender<Result<OpenProjectRpcResult, String>>,
        operation: Arc<WorkerRpcOperation>,
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
    /// Queue-ordered, acknowledged counterpart to
    /// [`WorkerMsg::RelinkMedia`].
    RelinkMediaRpc {
        media: String,
        path: PathBuf,
        reply: Sender<Result<RelinkMediaRpcResult, String>>,
        operation: Arc<WorkerRpcOperation>,
    },
    /// Try `folder/<filename>` for every missing pool entry (locate-folder
    /// gesture in the relink dialog).
    RelinkFolder {
        folder: PathBuf,
    },
    /// Queue-ordered, acknowledged counterpart to
    /// [`WorkerMsg::RelinkFolder`].
    RelinkFolderRpc {
        folder: PathBuf,
        reply: Sender<Result<RelinkFolderRpcResult, String>>,
        operation: Arc<WorkerRpcOperation>,
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
    /// Queue-ordered, acknowledged counterpart to [`WorkerMsg::NewProject`].
    /// It deliberately does not create or bind a draft; the orchestration
    /// layer follows it with an acknowledged save to its chosen app-owned path.
    NewProjectRpc {
        reply: Sender<Result<NewProjectRpcResult, String>>,
        operation: Arc<WorkerRpcOperation>,
    },
    /// Replace the session with an installed `.cutlasst` template filled by
    /// `picks` (CapCut "use template"). On success a fresh draft directory
    /// is created and the filled project saved into it (the engine resets
    /// the project path, so the bind happens here, not via a queued
    /// `SaveProject` that would also run after a failure). Same epoch bump
    /// as `OpenProject`. A pre-apply failure leaves the current session
    /// untouched; a later draft/binding failure publishes the applied
    /// in-memory session and reports that partial outcome explicitly.
    ApplyTemplate {
        path: PathBuf,
        picks: Vec<TemplatePick>,
    },
    /// Queue-ordered, acknowledged counterpart to
    /// [`WorkerMsg::ApplyTemplate`].
    ApplyTemplateRpc {
        path: PathBuf,
        picks: Vec<TemplatePick>,
        reply: Sender<Result<ApplyTemplateRpcResult, String>>,
        operation: Arc<WorkerRpcOperation>,
    },
    /// Rename the current draft (its display name). Applied as one undoable
    /// edit; the projection republish updates the title bar and the next
    /// auto-save writes the name into the draft's project file and meta.
    RenameProject {
        name: String,
    },
    /// Report the preview-frame cache's exact current in-memory usage.
    #[allow(dead_code)] // Wired into the cache registry in its follow-up phase.
    GetPreviewCacheStats {
        reply: Sender<PreviewCacheStats>,
        operation: Arc<WorkerRpcOperation>,
    },
    /// Drop every composited preview frame and report the exact pre-clear
    /// usage (the entries and bytes removed).
    #[allow(dead_code)] // Wired into the cache registry in its follow-up phase.
    ClearPreviewCache {
        reply: Sender<PreviewCacheStats>,
        operation: Arc<WorkerRpcOperation>,
    },
    /// Hand an off-UI maintenance worker a coherent clone of the live project,
    /// then stop consuming this queue until its guard sends or disconnects
    /// `resume`. The operation state prevents an abandoned queued request from
    /// pausing the worker when it is eventually dequeued.
    #[allow(dead_code)] // Wired into cache relocation by its coordination slice.
    BeginProjectMaintenance {
        reply: Sender<Result<Project, ()>>,
        resume: Receiver<ProjectMaintenanceResumeAction>,
        operation: Arc<WorkerRpcOperation>,
    },
    /// Clone the live project for the AI agent's sandbox rehearsal
    /// (`src/agent.rs`). Ordered with mutations, so the snapshot always
    /// reflects every edit sent before it.
    SnapshotProject {
        reply: Sender<Project>,
    },
    /// Replay a rehearsed agent plan, one history group per phase,
    /// re-validating every step against the live project and remapping ids
    /// the sandbox allocated. A failure rolls back the failing phase only
    /// and stops; phases already committed stay, each its own undo step.
    AgentApplyPlan {
        phases: Vec<Vec<AgentPlanStep>>,
        reply: Sender<Result<(), String>>,
    },
}

/// Dialog settings for one export job (see `ui/lib/export-backend.slint`).
pub struct ExportRequest {
    /// Destination file (the platform encoder writes H.264/AAC mp4).
    pub path: PathBuf,
    /// Scale the output to this height (aspect preserved, evened for H.264);
    /// `None` exports at the composite canvas size.
    pub target_height: Option<u32>,
    /// Resample to this integer frame rate; `None` keeps the timeline rate.
    pub fps_num: Option<i32>,
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
pub(super) struct ClipboardClip {
    /// Lane the clip was copied from (preferred paste target).
    pub(super) track: TrackId,
    /// Lane kind, for recreating a lane when `track` is gone by paste time.
    pub(super) kind: TrackKind,
    pub(super) content: ClipSource,
    /// Timeline-rate duration, for first-fit placement.
    pub(super) duration_ticks: i64,
    /// Start offset from the block's earliest member — paste keeps the
    /// members' relative placement.
    pub(super) offset_ticks: i64,
    /// The original's link group, as a grouping key only: members copied
    /// from the same group are re-linked as a fresh group on paste.
    pub(super) link: Option<LinkId>,
}

/// Cheap, cloneable sender to the engine thread. Hand one clone to each UI
/// callback that needs to talk to the engine (scrub, import, …). Cloning keeps
/// the channel — and therefore the worker loop — alive.
#[derive(Clone)]
pub struct WorkerHandle {
    pub(super) tx: Sender<WorkerMsg>,
}
