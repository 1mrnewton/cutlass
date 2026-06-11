//! Session-scoped editing runtime.

use std::path::PathBuf;

use cutlass_cache::FrameCache;
use cutlass_commands::{Command, ProjectCommand};
use cutlass_compositor::{Compositor, GpuContext};
use cutlass_models::Project;

use cutlass_models::{ClipId, ClipTransform, RationalTime};

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
        if matches!(outcome, ApplyOutcome::Opened | ApplyOutcome::Loaded) {
            self.decoder_pool.clear();
        }
        if let Some(inverse) = inverse {
            self.history.record_do(inverse);
        }
        Ok(outcome)
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
        )
    }

    /// Replace (or clear) the live gesture transform override. Preview frames
    /// render the overridden clip at this transform until cleared; project
    /// state, history, and export are untouched. The drag-release commit
    /// clears it and applies one `SetClipTransform` instead.
    pub fn set_transform_override(&mut self, override_transform: Option<(ClipId, ClipTransform)>) {
        self.transform_override = override_transform;
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
        debug_assert!(!self.history.in_group(), "undo inside an open history group");
        let Some(action) = self.history.pop_undo() else {
            return false;
        };
        match self.run_action(action) {
            Ok(inverse) => {
                self.history.push_redo(inverse);
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
        debug_assert!(!self.history.in_group(), "redo inside an open history group");
        let Some(action) = self.history.pop_redo() else {
            return false;
        };
        match self.run_action(action) {
            Ok(inverse) => {
                self.history.push_undo(inverse);
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
