//! Session-scoped editing runtime.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use cutlass_cache::FrameCache;
use cutlass_commands::{Command, EditCommand, EditOutcome, ProjectCommand};
use cutlass_compositor::{Compositor, GpuContext};
use cutlass_ml::{CaptionLayout, Transcribe, TranscribeOptions};
use cutlass_models::Project;

use cutlass_models::{ClipId, ClipTransform, Generator, RationalTime, TimeRange, TrackId, TrackKind};

use crate::action::{ApplyContext, ApplyOutcome, History, dispatch};
use crate::decoder_pool::DecoderPool;
use crate::error::EngineError;
use crate::frame::RgbaFrame;
use crate::generator_raster::GeneratorRaster;
use crate::preview;

fn gpu_init_err(err: cutlass_compositor::CompositorError) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Unsupported, err.to_string())
}

/// Default on-disk frame cache budget (50 GiB).
pub const DEFAULT_CACHE_BUDGET_BYTES: u64 = 50 * 1024 * 1024 * 1024;

/// Where YUV ↔ RGBA conversion runs for preview and export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorConvertPath {
    /// GPU shaders in `cutlass-compositor` (default).
    #[default]
    Gpu,
    /// Legacy CPU routines in `cutlass-engine::frame` / `composite`.
    LegacyCpu,
}

/// Session configuration for [`Engine`].
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Directory for per-source YUV frame blobs and index sidecars.
    pub cache_dir: PathBuf,
    /// Global frame cache byte budget; LRU eviction runs across all sources.
    pub cache_budget_bytes: u64,
    /// Maximum inverse actions retained on the undo stack.
    pub undo_limit: usize,
    /// YUV/RGBA conversion path for preview and export compositing.
    pub color_convert: ColorConvertPath,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            cache_dir: PathBuf::from(".cutlass/cache"),
            cache_budget_bytes: DEFAULT_CACHE_BUDGET_BYTES,
            undo_limit: 100,
            color_convert: ColorConvertPath::Gpu,
        }
    }
}

/// Cutlass editing engine: project state, inverse undo/redo, session infrastructure.
pub struct Engine {
    project: Project,
    cache: FrameCache,
    config: EngineConfig,
    history: History,
    project_path: Option<PathBuf>,
    decoder_pool: DecoderPool,
    raster: GeneratorRaster,
    gpu: GpuContext,
    compositor: Compositor,
    /// Live gesture override (preview roadmap Phase 3): one clip rendered
    /// with this transform instead of its committed one. Session state —
    /// never serialized, never in history, never seen by export.
    transform_override: Option<(ClipId, ClipTransform)>,
    /// Live edit override (inspector live preview): one generated clip
    /// rasterized from this generator instead of its committed one — e.g.
    /// dragging the font-size slider repaints without an undo entry per tick.
    /// Session state — never serialized, never in history, never seen by export.
    generator_override: Option<(ClipId, cutlass_models::Generator)>,
    /// Session revision: bumped on every successful project mutation
    /// (edits, imports, open/load, undo, redo). Never serialized.
    revision: u64,
    /// The revision last written to (or read from) disk; together with
    /// `revision` this is the dirty flag (see [`is_dirty`](Self::is_dirty)).
    saved_revision: u64,
    /// Runtime-injected speech-to-text backend for auto-captions (M9 Phase 4).
    /// `None` until a caller wires one (the UI from `[ml]` config, tests a
    /// stub) — the crate stays free of any model runtime, so the lean build
    /// never pulls whisper.cpp. Shared (`Arc`) so a worker can hold it while
    /// the engine moves between jobs.
    transcriber: Option<Arc<dyn Transcribe + Send + Sync>>,
}

impl Engine {
    pub fn new(config: EngineConfig) -> std::io::Result<Self> {
        let undo_limit = config.undo_limit;
        let cache = FrameCache::new(config.cache_dir.clone(), config.cache_budget_bytes)?;
        let gpu = GpuContext::new_headless_blocking().map_err(gpu_init_err)?;
        let compositor = Compositor::new(&gpu).map_err(gpu_init_err)?;
        Ok(Self {
            project: Project::new("untitled", cutlass_models::Rational::FPS_24),
            cache,
            history: History::new(undo_limit),
            project_path: None,
            decoder_pool: DecoderPool::new(),
            raster: GeneratorRaster::new(),
            gpu,
            compositor,
            config,
            transform_override: None,
            generator_override: None,
            revision: 0,
            saved_revision: 0,
            transcriber: None,
        })
    }

    pub fn with_project(config: EngineConfig, project: Project) -> std::io::Result<Self> {
        let undo_limit = config.undo_limit;
        let cache = FrameCache::new(config.cache_dir.clone(), config.cache_budget_bytes)?;
        let gpu = GpuContext::new_headless_blocking().map_err(gpu_init_err)?;
        let compositor = Compositor::new(&gpu).map_err(gpu_init_err)?;
        Ok(Self {
            project,
            cache,
            history: History::new(undo_limit),
            project_path: None,
            decoder_pool: DecoderPool::new(),
            raster: GeneratorRaster::new(),
            gpu,
            compositor,
            config,
            transform_override: None,
            generator_override: None,
            revision: 0,
            saved_revision: 0,
            transcriber: None,
        })
    }

    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Read-only view of the session project. Timeline and media mutations must
    /// go through [`apply`](Self::apply) so undo/redo stays consistent.
    pub fn project(&self) -> &Project {
        &self.project
    }

    pub fn cache(&self) -> &FrameCache {
        &self.cache
    }

    /// Path last written with [`Save`](cutlass_commands::ProjectCommand::Save) or
    /// loaded with [`Open`](cutlass_commands::ProjectCommand::Open) /
    /// [`Load`](cutlass_commands::ProjectCommand::Load).
    pub fn project_path(&self) -> Option<&PathBuf> {
        self.project_path.as_ref()
    }

    /// Monotonic session revision: bumped by every successful project
    /// mutation (edit commands, imports, open/load, undo, redo). Session
    /// state — never serialized; restarts from 0 with each engine.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// True when the session has mutations not yet written with
    /// [`Save`](cutlass_commands::ProjectCommand::Save). Conservative by
    /// design: a rolled-back gesture or an undo back to the saved content
    /// still reads dirty (revisions only grow) — false-positives only,
    /// never a false "saved".
    pub fn is_dirty(&self) -> bool {
        self.revision != self.saved_revision
    }

    pub fn can_undo(&self) -> bool {
        self.history.can_undo()
    }

    pub fn can_redo(&self) -> bool {
        self.history.can_redo()
    }

    /// Group every command applied until [`commit_group`](Self::commit_group)
    /// into a single history entry, so a gesture that dispatches several
    /// commands (new-lane move, drop that creates a lane, delete that empties
    /// its lane) reverts with one undo.
    pub fn begin_group(&mut self) {
        self.history.begin_group();
    }

    /// Close the open group and record it as one undo entry (no-op if the
    /// group made no edits).
    pub fn commit_group(&mut self) {
        self.history.commit_group();
    }

    /// Abort the open group: revert its commands in reverse order, restoring
    /// the pre-group state. History is left untouched — a rolled-back gesture
    /// records nothing and preserves the redo stack.
    pub fn rollback_group(&mut self) {
        for inverse in self.history.take_group().into_iter().rev() {
            if self.run_action(inverse).is_err() {
                // Inverses are written to be infallible once recorded (same
                // policy as undo); nothing sensible to do beyond stopping.
                tracing::error!("history group rollback failed; state may be partial");
                return;
            }
        }
    }

    /// Replace the session with a fresh, empty, unsaved project (File →
    /// New). Clears history, decoders, any gesture override, and the
    /// project path; the session rebaselines as clean. Mirrors what
    /// [`new`](Self::new) builds, without tearing down the GPU context or
    /// caches.
    pub fn new_session(&mut self) {
        self.project = Project::new("untitled", cutlass_models::Rational::FPS_24);
        self.history.clear();
        self.decoder_pool.clear();
        self.transform_override = None;
        self.generator_override = None;
        self.project_path = None;
        self.revision += 1;
        self.saved_revision = self.revision;
    }

    /// Replace the session with the given project (AI-agent sandbox: the
    /// agent rehearses a prompt's edits against a snapshot clone before the
    /// live engine replays the validated plan). Same teardown as
    /// [`new_session`](Self::new_session): history, decoders, and gesture
    /// overrides clear, no project path binds, the session rebaselines.
    pub fn reset_project(&mut self, project: Project) {
        self.project = project;
        self.history.clear();
        self.decoder_pool.clear();
        self.transform_override = None;
        self.generator_override = None;
        self.project_path = None;
        self.revision += 1;
        self.saved_revision = self.revision;
    }

    /// Replace the session from an autosave snapshot, binding it to the
    /// file it stands in for (crash recovery). Loads tolerantly (missing
    /// media entries are kept, like `Load`), points `project_path` at
    /// `bind_to` — the user's file, **not** the autosave — and marks the
    /// session dirty: the restored content is by definition not what's on
    /// disk at `bind_to`, so the first Cmd+S writes it back there.
    pub fn restore_session(
        &mut self,
        autosave: &std::path::Path,
        bind_to: Option<PathBuf>,
    ) -> Result<(), EngineError> {
        let loaded = Project::load_from_file(autosave)?;
        crate::action::project::relink_media_cache(&self.cache, &loaded, false)?;
        self.project = loaded;
        self.history.clear();
        self.decoder_pool.clear();
        self.transform_override = None;
        self.generator_override = None;
        self.project_path = bind_to;
        self.saved_revision = self.revision;
        self.revision += 1;
        Ok(())
    }

    /// Apply a wire command. On success, pushes the inverse action onto the undo stack.
    pub fn apply(&mut self, command: Command) -> Result<ApplyOutcome, EngineError> {
        if let Command::Project(ProjectCommand::Export { path }) = command {
            let stats = crate::export::export_timeline(
                &self.project,
                &mut self.decoder_pool,
                &self.gpu,
                &mut self.compositor,
                &path,
                self.config.color_convert,
            )?;
            return Ok(ApplyOutcome::Exported { stats });
        }

        let mut ctx = ApplyContext {
            project: &mut self.project,
            cache: &self.cache,
            project_path: &mut self.project_path,
            history: &mut self.history,
        };
        let (outcome, inverse) = dispatch(command, &mut ctx)?;
        match outcome {
            ApplyOutcome::Opened | ApplyOutcome::Loaded => {
                self.decoder_pool.clear();
                // The session now mirrors the file it came from: rebaseline
                // as clean (revision still bumps so observers see a change).
                self.revision += 1;
                self.saved_revision = self.revision;
            }
            ApplyOutcome::Saved => self.saved_revision = self.revision,
            // Relink dirties the session: the repaired path needs saving.
            ApplyOutcome::Imported { .. }
            | ApplyOutcome::RemovedMedia { .. }
            | ApplyOutcome::Edited(_)
            | ApplyOutcome::Relinked { .. } => self.revision += 1,
            // Export is handled by the early return above.
            ApplyOutcome::Exported { .. } => {}
        }
        if let Some(inverse) = inverse {
            self.history.record_do(inverse);
        }
        Ok(outcome)
    }

    /// Inject the speech-to-text backend used by [`generate_captions`]. Called
    /// once at startup by the UI (built from `[ml]` config) or by a test with a
    /// stub; replaces any previously set backend.
    pub fn set_transcriber(&mut self, transcriber: Arc<dyn Transcribe + Send + Sync>) {
        self.transcriber = Some(transcriber);
    }

    /// Whether a transcribe backend is wired (so the UI can gate the captions
    /// affordance instead of offering a button that always errors).
    pub fn has_transcriber(&self) -> bool {
        self.transcriber.is_some()
    }

    /// Auto-caption a media clip (M9 Phase 4): decode its speech, transcribe it
    /// with the injected backend, and add the words as subtitle-styled text
    /// clips on a fresh "Captions" lane — one undoable history entry. Captions
    /// are ordinary text clips, so they edit, restyle, move, and delete like any
    /// title afterward.
    ///
    /// Slow (the transcribe call can run for seconds): callers run it off the UI
    /// thread, passing `cancel`/`on_progress` to abort and report. Errors when
    /// no backend is set, the clip is generated / retimed / silent, or the
    /// backend fails; on any error nothing is added.
    pub fn generate_captions(
        &mut self,
        clip: ClipId,
        options: TranscribeOptions,
        layout: CaptionLayout,
        cancel: &AtomicBool,
        on_progress: &mut dyn FnMut(f32),
    ) -> Result<ApplyOutcome, EngineError> {
        let transcriber = self
            .transcriber
            .clone()
            .ok_or_else(|| EngineError::Transcribe("no transcription backend configured".into()))?;

        // Decode + transcribe + map against an immutable view, then mutate.
        let placements = crate::caption::plan(
            &self.project,
            clip,
            transcriber.as_ref(),
            &options,
            &layout,
            cancel,
            on_progress,
        )?;
        if placements.is_empty() {
            return Err(EngineError::Transcribe("no speech detected to caption".into()));
        }

        // All the inserts (the new lane + every cue) collapse into one undo.
        self.begin_group();
        match self.insert_captions(placements) {
            Ok(track) => {
                self.commit_group();
                Ok(ApplyOutcome::Edited(EditOutcome::CreatedTrack(track)))
            }
            Err(e) => {
                self.rollback_group();
                Err(e)
            }
        }
    }

    /// Add a fresh text lane and place every caption clip on it. Caller owns the
    /// history group; a partial failure rolls back through it.
    fn insert_captions(
        &mut self,
        placements: Vec<(Generator, TimeRange)>,
    ) -> Result<TrackId, EngineError> {
        let track = match self.apply(Command::Edit(EditCommand::AddTrack {
            kind: TrackKind::Text,
            name: "Captions".into(),
            index: None,
        }))? {
            ApplyOutcome::Edited(EditOutcome::CreatedTrack(id)) => id,
            other => {
                return Err(EngineError::Transcribe(format!(
                    "captions: unexpected add-track outcome: {other:?}"
                )));
            }
        };
        for (generator, timeline) in placements {
            self.apply(Command::Edit(EditCommand::AddGenerated {
                track,
                generator,
                timeline,
            }))?;
        }
        Ok(track)
    }

    /// Warm decoders and the frame cache for `time` without compositing —
    /// playback read-ahead (see [`preview::prefetch_frame`]). Errors are the
    /// caller's to ignore: a tick past the content or mid-edit is expected.
    pub fn prefetch(&mut self, time: RationalTime) -> Result<(), EngineError> {
        preview::prefetch_frame(
            &self.project,
            &self.cache,
            &mut self.decoder_pool,
            &mut self.raster,
            time,
            self.config.color_convert,
        )
    }

    /// Composite enabled video layers at `time` and return an RGBA preview frame.
    pub fn get_frame(&mut self, time: RationalTime) -> Result<RgbaFrame, EngineError> {
        preview::get_frame(
            &self.project,
            &self.cache,
            &mut self.decoder_pool,
            &mut self.raster,
            &self.gpu,
            &mut self.compositor,
            time,
            self.config.color_convert,
            self.transform_override,
            self.generator_override
                .as_ref()
                .map(|(id, generator)| (*id, generator)),
        )
    }

    /// Replace (or clear) the live gesture transform override. Preview frames
    /// render the overridden clip at this transform until cleared; project
    /// state, history, and export are untouched. The drag-release commit
    /// clears it and applies one `SetClipTransform` instead.
    pub fn set_transform_override(&mut self, override_transform: Option<(ClipId, ClipTransform)>) {
        self.transform_override = override_transform;
    }

    /// Replace (or clear) the live generator override. Preview frames raster
    /// the overridden clip from this generator until cleared; project state,
    /// history, and export are untouched. The control-release commit clears it
    /// and applies one `SetGenerator` instead. Used for live inspector edits
    /// (e.g. dragging the font-size slider) without flooding the undo history.
    pub fn set_generator_override(
        &mut self,
        override_generator: Option<(ClipId, cutlass_models::Generator)>,
    ) {
        self.generator_override = override_generator;
    }

    /// Tight size (canvas px) of the content `generator` draws on the current
    /// composite canvas — what a preview selection box should hug, since the
    /// raster itself is canvas-sized and mostly transparent. `None` for
    /// generators the compositor doesn't draw. Served from the raster cache
    /// (`&mut self`: a miss rasterizes once, warming playback too).
    pub fn generator_content_size(
        &mut self,
        generator: &cutlass_models::Generator,
    ) -> Option<(u32, u32)> {
        let (width, height) = crate::composite::composite_canvas_size(&self.project);
        self.raster.content_size(generator, width, height)
    }

    pub fn undo(&mut self) -> bool {
        debug_assert!(
            !self.history.in_group(),
            "undo inside an open history group"
        );
        let Some(action) = self.history.pop_undo() else {
            return false;
        };
        match self.run_action(action) {
            Ok(inverse) => {
                self.history.push_redo(inverse);
                self.revision += 1;
                true
            }
            Err(_) => {
                // Inverse was popped before apply; on failure the action is lost.
                // Inverses are written to be infallible once pushed.
                false
            }
        }
    }

    pub fn redo(&mut self) -> bool {
        debug_assert!(
            !self.history.in_group(),
            "redo inside an open history group"
        );
        let Some(action) = self.history.pop_redo() else {
            return false;
        };
        match self.run_action(action) {
            Ok(inverse) => {
                self.history.push_undo(inverse);
                self.revision += 1;
                true
            }
            Err(_) => false,
        }
    }

    fn run_action(
        &mut self,
        action: Box<dyn crate::action::EditAction>,
    ) -> Result<Box<dyn crate::action::EditAction>, EngineError> {
        let mut ctx = ApplyContext {
            project: &mut self.project,
            cache: &self.cache,
            project_path: &mut self.project_path,
            history: &mut self.history,
        };
        action.apply(&mut ctx)
    }

    /// True when the frame cache has paused writes due to a disk-full I/O error.
    pub fn disk_pressure(&self) -> bool {
        self.cache.disk_pressure()
    }

    /// Resume cache writes after disk space is available again.
    pub fn clear_disk_pressure(&self) {
        self.cache.clear_disk_pressure();
    }
}
