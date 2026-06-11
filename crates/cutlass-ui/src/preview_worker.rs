//! Background preview rendering: engine and decode/composite stay off the UI thread.

use std::path::{Path, PathBuf};
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use cutlass_commands::{Command, EditCommand, EditOutcome, ProjectCommand};
use cutlass_engine::{ApplyOutcome, Engine, EngineConfig};
use cutlass_models::{
    ClipId, ClipSource, MediaId, Rational, RationalTime, TimeRange, Track, TrackId, TrackKind,
    resample,
};
use tracing::{error, info};

use crate::strips::StripHandle;
use crate::thumbnails::{ThumbKind, ThumbnailHandle};
use crate::{EditorStore, PreviewStore};

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
    /// Re-place `clip` (raw id) at `[start_tick, start_tick + duration_ticks)`
    /// on its own lane (edge trim; the engine re-derives the source in/out).
    TrimClip {
        clip: String,
        start_tick: i64,
        duration_ticks: i64,
    },
    /// Remove `clip` (raw id); a lane this empties is removed too (same
    /// policy as drag-moves).
    RemoveClip { clip: String },
    /// Split `clip` (raw id) at `at_tick` (sequence ticks). The UI gates on
    /// the playhead being strictly inside the clip; the engine re-validates.
    SplitClip { clip: String, at_tick: i64 },
    /// Step the engine history one entry back / forward.
    Undo,
    Redo,
    /// Snapshot `clip` (raw id) into the worker clipboard. A snapshot, not a
    /// reference — pasting works after the original is deleted.
    CopyClip { clip: String },
    /// Place the clipboard content at `tick` on the copied clip's lane,
    /// sliding right into the first gap that fits.
    PasteAt { tick: i64 },
    /// Place a copy of `clip` right after it on its own lane.
    DuplicateClip { clip: String },
    /// Mirror of the UI's main-track magnet toggle. The worker needs it for
    /// ops without a drag resolution (delete/paste/duplicate); enabling also
    /// packs the main lane gapless (one history entry).
    SetMainMagnet(bool),
}

/// Worker-side clipboard: everything needed to re-issue the copied clip as a
/// fresh `AddClip` / `AddGenerated` later, independent of the original.
struct ClipboardClip {
    /// Lane the clip was copied from (preferred paste target).
    track: TrackId,
    /// Lane kind, for recreating a lane when `track` is gone by paste time.
    kind: TrackKind,
    content: ClipSource,
    /// Timeline-rate duration, for first-fit placement.
    duration_ticks: i64,
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

    pub fn trim_clip(&self, clip: String, start_tick: i64, duration_ticks: i64) {
        let _ = self.tx.send(WorkerMsg::TrimClip {
            clip,
            start_tick,
            duration_ticks,
        });
    }

    pub fn remove_clip(&self, clip: String) {
        let _ = self.tx.send(WorkerMsg::RemoveClip { clip });
    }

    pub fn split_clip(&self, clip: String, at_tick: i64) {
        let _ = self.tx.send(WorkerMsg::SplitClip { clip, at_tick });
    }

    pub fn undo(&self) {
        let _ = self.tx.send(WorkerMsg::Undo);
    }

    pub fn redo(&self) {
        let _ = self.tx.send(WorkerMsg::Redo);
    }

    pub fn copy_clip(&self, clip: String) {
        let _ = self.tx.send(WorkerMsg::CopyClip { clip });
    }

    pub fn paste_at(&self, tick: i64) {
        let _ = self.tx.send(WorkerMsg::PasteAt { tick });
    }

    pub fn duplicate_clip(&self, clip: String) {
        let _ = self.tx.send(WorkerMsg::DuplicateClip { clip });
    }

    pub fn set_main_magnet(&self, enabled: bool) {
        let _ = self.tx.send(WorkerMsg::SetMainMagnet(enabled));
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
        thumbs: ThumbnailHandle,
        strips: StripHandle,
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
                    thumbs,
                    strips,
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

fn worker_main(
    config: EngineConfig,
    preview_weak: slint::Weak<PreviewStore<'static>>,
    editor_weak: slint::Weak<EditorStore<'static>>,
    thumbs: ThumbnailHandle,
    strips: StripHandle,
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
    publish_projection(&engine, &editor_weak);

    worker_loop(&mut engine, tl_rate, preview_weak, editor_weak, thumbs, strips, req_rx);
    Ok(())
}

/// Single consumer for the engine thread. Scrub frames coalesce to the latest
/// pending tick, but every mutation (import, add-clip, move-clip, …) is
/// executed in order — it must never be discarded by the coalescing drain.
fn worker_loop(
    engine: &mut Engine,
    tl_rate: Rational,
    preview_weak: slint::Weak<PreviewStore<'static>>,
    editor_weak: slint::Weak<EditorStore<'static>>,
    thumbs: ThumbnailHandle,
    strips: StripHandle,
    req_rx: Receiver<WorkerMsg>,
) {
    // Clipboard lives with the loop: it's edit-session state, not project
    // state — copies survive any number of edits/undos and die with the app.
    let mut clipboard: Option<ClipboardClip> = None;
    // Mirror of TimelineStore.main-magnet-enabled (must match its default).
    // Drag gestures carry their resolved insert flag; this drives the ops
    // without a drag resolution (delete/paste/duplicate) and pack-on-enable.
    let mut main_magnet = true;

    let mutate = |engine: &mut Engine,
                  clipboard: &mut Option<ClipboardClip>,
                  main_magnet: &mut bool,
                  msg: WorkerMsg| {
        match msg {
            WorkerMsg::Import(path) => {
                import_and_publish(engine, &path, &editor_weak, &thumbs, &strips)
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
                &editor_weak,
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
                &editor_weak,
            ),
            WorkerMsg::TrimClip {
                clip,
                start_tick,
                duration_ticks,
            } => trim_clip_and_publish(engine, &clip, start_tick, duration_ticks, &editor_weak),
            WorkerMsg::RemoveClip { clip } => {
                remove_clip_and_publish(engine, &clip, *main_magnet, &editor_weak)
            }
            WorkerMsg::SplitClip { clip, at_tick } => {
                split_clip_and_publish(engine, &clip, at_tick, &editor_weak)
            }
            WorkerMsg::Undo => history_step_and_publish(engine, false, &editor_weak),
            WorkerMsg::Redo => history_step_and_publish(engine, true, &editor_weak),
            WorkerMsg::CopyClip { clip } => {
                if let Some(snapshot) = snapshot_clip(engine, &clip) {
                    info!(clip, "copied clip to clipboard");
                    *clipboard = Some(snapshot);
                }
            }
            WorkerMsg::PasteAt { tick } => match clipboard {
                Some(content) => paste_and_publish(engine, content, tick, *main_magnet, &editor_weak),
                None => info!("paste ignored: clipboard empty"),
            },
            WorkerMsg::DuplicateClip { clip } => {
                duplicate_clip_and_publish(engine, &clip, *main_magnet, &editor_weak)
            }
            WorkerMsg::SetMainMagnet(enabled) => {
                *main_magnet = enabled;
                info!(enabled, "main-track magnet toggled");
                if enabled {
                    pack_main_track_and_publish(engine, &editor_weak);
                }
            }
            WorkerMsg::Frame(_) => unreachable!("frames are handled by the drain below"),
        }
    };

    while let Ok(msg) = req_rx.recv() {
        match msg {
            WorkerMsg::Frame(mut tick) => {
                while let Ok(next) = req_rx.try_recv() {
                    match next {
                        WorkerMsg::Frame(latest) => tick = latest,
                        other => mutate(engine, &mut clipboard, &mut main_magnet, other),
                    }
                }
                render_frame(engine, tl_rate, &preview_weak, tick);
            }
            other => mutate(engine, &mut clipboard, &mut main_magnet, other),
        }
    }
}

fn import_and_publish(
    engine: &mut Engine,
    path: &Path,
    editor_weak: &slint::Weak<EditorStore<'static>>,
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
                let kind = if source.is_audio_only() {
                    ThumbKind::Audio
                } else {
                    ThumbKind::Video
                };
                thumbs.request(media.raw(), source.path().to_path_buf(), kind);
                // The strip worker resolves filmstrip/waveform requests by
                // media id alone; it needs the path on record (src/strips.rs).
                strips.register_media(media.raw(), source.path().to_path_buf());
            }
            publish_projection(engine, editor_weak);
        }
        Ok(other) => error!(path = %path.display(), "unexpected import outcome: {other:?}"),
        Err(e) => error!(path = %path.display(), "import failed: {e}"),
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
fn add_clip_and_publish(
    engine: &mut Engine,
    media: &str,
    track: &str,
    start_tick: i64,
    drop_row: i64,
    insert: bool,
    editor_weak: &slint::Weak<EditorStore<'static>>,
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

    // The main-track magnet only applies to the main *video* lane.
    if insert
        && !audio_only
        && let Some(lane) = lane_of_kind(engine, track, TrackKind::Video)
    {
        match engine.apply(Command::Edit(EditCommand::RippleInsert {
            track: lane,
            media: media_id,
            source,
            at: RationalTime::new(start_tick.max(0), tl_rate),
        })) {
            Ok(ApplyOutcome::Edited(EditOutcome::Created(clip))) => {
                info!(%clip, %lane, %media_id, start_tick, "ripple-inserted clip from library drop");
            }
            Ok(other) => error!(%media_id, "unexpected ripple-insert outcome: {other:?}"),
            Err(e) => error!(%media_id, %lane, start_tick, "ripple insert failed: {e}"),
        }
        publish_projection(engine, editor_weak);
        return;
    }
    // Mirror Project::add_clip's source→timeline resampling so first-fit sees
    // the same extent the engine will validate.
    let duration_ticks = resample(source.duration, tl_rate).value.max(1);
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
            publish_projection(engine, editor_weak);
        }
        // First-fit should have made the placement valid; the engine still
        // rejects atomically if not. Surface the reason and roll the group
        // back so a lane created for this drop doesn't linger.
        Ok(other) => {
            error!(%media_id, "unexpected add-clip outcome: {other:?}");
            engine.rollback_group();
            publish_projection(engine, editor_weak);
        }
        Err(e) => {
            error!(%media_id, %track_id, start_tick = start_value, "add clip failed: {e}");
            engine.rollback_group();
            publish_projection(engine, editor_weak);
        }
    }
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
    editor_weak: &slint::Weak<EditorStore<'static>>,
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
        publish_projection(engine, editor_weak);
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
            publish_projection(engine, editor_weak);
        }
        Ok(other) => {
            error!(%clip_id, "unexpected move-clip outcome: {other:?}");
            engine.rollback_group();
            publish_projection(engine, editor_weak);
        }
        // The drag resolver previewed a valid spot; the engine still rejects
        // atomically if the projection raced a concurrent edit. Rolling back
        // removes a lane this move just created.
        Err(e) => {
            error!(%clip_id, %to_track, start_tick, "move clip failed: {e}");
            engine.rollback_group();
            publish_projection(engine, editor_weak);
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
fn trim_clip_and_publish(
    engine: &mut Engine,
    clip: &str,
    start_tick: i64,
    duration_ticks: i64,
    editor_weak: &slint::Weak<EditorStore<'static>>,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "trim ignored: unparsable clip id");
        return;
    };
    let tl_rate = engine.project().timeline().frame_rate;
    let timeline = TimeRange::at_rate(start_tick.max(0), duration_ticks.max(1), tl_rate);

    match engine.apply(Command::Edit(EditCommand::TrimClip {
        clip: clip_id,
        timeline,
    })) {
        Ok(ApplyOutcome::Edited(EditOutcome::Updated(_))) => {
            info!(%clip_id, start_tick, duration_ticks, "trimmed clip");
            publish_projection(engine, editor_weak);
        }
        Ok(other) => error!(%clip_id, "unexpected trim-clip outcome: {other:?}"),
        Err(e) => error!(%clip_id, start_tick, duration_ticks, "trim clip failed: {e}"),
    }
}

/// Remove a clip; a lane the removal empties is removed with it (CapCut
/// deletes emptied overlay tracks — same policy the drag-moves use). With
/// the main-track magnet on, deleting from the main lane ripples the gap
/// closed. Everything forms one history group: one undo restores it all.
fn remove_clip_and_publish(
    engine: &mut Engine,
    clip: &str,
    main_magnet: bool,
    editor_weak: &slint::Weak<EditorStore<'static>>,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "delete ignored: unparsable clip id");
        return;
    };
    let Some(track) = engine.project().timeline().track_of(clip_id) else {
        error!(%clip_id, "delete ignored: clip not on the timeline");
        return;
    };
    let ripple = main_magnet && Some(track) == main_video_track(engine);

    // One history entry per delete, including the removal of an emptied lane.
    engine.begin_group();
    let command = if ripple {
        EditCommand::RippleDelete { clip: clip_id }
    } else {
        EditCommand::RemoveClip { clip: clip_id }
    };
    match engine.apply(Command::Edit(command)) {
        Ok(ApplyOutcome::Edited(EditOutcome::Removed(_))) => {
            remove_track_if_empty(engine, track);
            engine.commit_group();
            info!(%clip_id, %track, ripple, "removed clip");
            publish_projection(engine, editor_weak);
        }
        Ok(other) => {
            error!(%clip_id, "unexpected remove-clip outcome: {other:?}");
            engine.rollback_group();
        }
        Err(e) => {
            error!(%clip_id, "remove clip failed: {e}");
            engine.rollback_group();
        }
    }
}

/// Split a clip into two abutting clips at `at_tick`. The UI only offers the
/// split while the playhead is strictly inside the clip; the engine still
/// validates the position atomically.
fn split_clip_and_publish(
    engine: &mut Engine,
    clip: &str,
    at_tick: i64,
    editor_weak: &slint::Weak<EditorStore<'static>>,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "split ignored: unparsable clip id");
        return;
    };
    let tl_rate = engine.project().timeline().frame_rate;

    match engine.apply(Command::Edit(EditCommand::SplitClip {
        clip: clip_id,
        at: RationalTime::new(at_tick, tl_rate),
    })) {
        Ok(ApplyOutcome::Edited(EditOutcome::Created(tail))) => {
            info!(%clip_id, %tail, at_tick, "split clip");
            publish_projection(engine, editor_weak);
        }
        Ok(other) => error!(%clip_id, "unexpected split-clip outcome: {other:?}"),
        Err(e) => error!(%clip_id, at_tick, "split clip failed: {e}"),
    }
}

/// Step the engine history (`redo == false` ⇒ undo). Publishes even on a
/// no-op so the UI's can-undo / can-redo flags stay honest.
fn history_step_and_publish(
    engine: &mut Engine,
    redo: bool,
    editor_weak: &slint::Weak<EditorStore<'static>>,
) {
    let stepped = if redo { engine.redo() } else { engine.undo() };
    info!(redo, stepped, "history step");
    publish_projection(engine, editor_weak);
}

/// Snapshot `clip` (raw id) for the clipboard.
fn snapshot_clip(engine: &Engine, clip: &str) -> Option<ClipboardClip> {
    let clip_id = parse_raw_id(clip).map(ClipId::from_raw)?;
    let timeline = engine.project().timeline();
    let track = timeline.track_of(clip_id)?;
    let kind = timeline.track(track)?.kind;
    let clip = engine.project().clip(clip_id)?;
    Some(ClipboardClip {
        track,
        kind,
        content: clip.content.clone(),
        duration_ticks: clip.timeline.duration.value,
    })
}

/// Paste the clipboard at `tick`: lands on the copied clip's lane (or a new
/// lane of its kind when that lane is gone), sliding right into the first
/// gap that fits — the same placement policy as library drops. With the
/// main-track magnet on, pasting on the main lane ripple-inserts at the
/// clip boundary nearest `tick` instead.
fn paste_and_publish(
    engine: &mut Engine,
    content: &ClipboardClip,
    tick: i64,
    main_magnet: bool,
    editor_weak: &slint::Weak<EditorStore<'static>>,
) {
    let tl_rate = engine.project().timeline().frame_rate;
    let duration = content.duration_ticks.max(1);

    // One history entry per paste, even when it recreates the copied lane.
    engine.begin_group();
    let track = if engine.project().timeline().track(content.track).is_some() {
        content.track
    } else {
        match create_track(engine, content.kind, 0) {
            Ok(id) => id,
            Err(e) => {
                error!("paste failed creating {:?} track: {e}", content.kind);
                engine.rollback_group();
                return;
            }
        }
    };

    let lane = engine
        .project()
        .timeline()
        .track(track)
        .expect("paste target track exists");
    let ripple = main_magnet && Some(track) == main_video_track(engine);
    let start = if ripple {
        nearest_boundary(lane, tick.max(0))
    } else {
        first_fit_start(lane, tick.max(0), duration)
    };

    if ripple
        && let Err(e) = apply_edit(engine, EditCommand::ShiftClips {
            track,
            from: RationalTime::new(start, tl_rate),
            delta: RationalTime::new(duration, tl_rate),
        })
    {
        error!(%track, start_tick = start, "paste failed opening hole: {e}");
        engine.rollback_group();
        publish_projection(engine, editor_weak);
        return;
    }

    match add_clip_content(engine, track, &content.content, content.duration_ticks, start) {
        Ok(clip_id) => {
            engine.commit_group();
            info!(%clip_id, %track, start_tick = start, ripple, "pasted clip");
            publish_projection(engine, editor_weak);
        }
        // Rolling back removes a lane this paste just recreated (and closes
        // a hole the ripple shift just opened).
        Err(e) => {
            error!(%track, start_tick = start, "paste failed: {e}");
            engine.rollback_group();
            publish_projection(engine, editor_weak);
        }
    }
}

/// Place a copy of `clip` immediately after it on its own lane (first gap
/// that fits from the clip's end). With the main-track magnet on, a main-lane
/// duplicate ripple-inserts right after the original, shifting later clips.
fn duplicate_clip_and_publish(
    engine: &mut Engine,
    clip: &str,
    main_magnet: bool,
    editor_weak: &slint::Weak<EditorStore<'static>>,
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
        publish_projection(engine, editor_weak);
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
            publish_projection(engine, editor_weak);
        }
        Err(e) => error!(%clip_id, start_tick = start, "duplicate failed: {e}"),
    }
}

/// Close every gap on the main lane, including leading space before the
/// first clip — CapCut's lane is gapless the moment the magnet turns on.
/// One history group: a single undo restores the gaps.
fn pack_main_track_and_publish(
    engine: &mut Engine,
    editor_weak: &slint::Weak<EditorStore<'static>>,
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
                publish_projection(engine, editor_weak);
                return;
            }
            shifted_so_far += current - expected;
        }
        expected += duration;
    }
    // An already-packed lane records nothing (empty groups are dropped).
    engine.commit_group();
    publish_projection(engine, editor_weak);
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
fn publish_projection(engine: &Engine, editor_weak: &slint::Weak<EditorStore<'static>>) {
    let project = engine.project().clone();
    let can_undo = engine.can_undo();
    let can_redo = engine.can_redo();
    let editor_weak = editor_weak.clone();
    if let Err(e) = slint::invoke_from_event_loop(move || {
        if let Some(store) = editor_weak.upgrade() {
            store.set_project(crate::projection::project_to_slint(&project));
            store.set_can_undo(can_undo);
            store.set_can_redo(can_redo);
        }
    }) {
        error!("failed to publish project projection to UI: {e}");
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
