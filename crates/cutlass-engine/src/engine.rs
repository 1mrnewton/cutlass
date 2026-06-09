//! Session-scoped editing runtime.

use std::path::PathBuf;

use cutlass_cache::FrameCache;
use cutlass_commands::Command;
use cutlass_models::Project;

use cutlass_models::RationalTime;

use crate::action::{ApplyContext, ApplyOutcome, History, dispatch};
use crate::decoder_pool::DecoderPool;
use crate::error::EngineError;
use crate::frame::RgbaFrame;
use crate::preview;

/// Default on-disk frame cache budget (50 GiB).
pub const DEFAULT_CACHE_BUDGET_BYTES: u64 = 50 * 1024 * 1024 * 1024;

/// Session configuration for [`Engine`].
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Directory for per-source YUV frame blobs and index sidecars.
    pub cache_dir: PathBuf,
    /// Global frame cache byte budget; LRU eviction runs across all sources.
    pub cache_budget_bytes: u64,
    /// Maximum inverse actions retained on the undo stack.
    pub undo_limit: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            cache_dir: PathBuf::from(".cutlass/cache"),
            cache_budget_bytes: DEFAULT_CACHE_BUDGET_BYTES,
            undo_limit: 100,
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
}

impl Engine {
    pub fn new(config: EngineConfig) -> std::io::Result<Self> {
        let undo_limit = config.undo_limit;
        let cache = FrameCache::new(config.cache_dir.clone(), config.cache_budget_bytes)?;
        Ok(Self {
            project: Project::new("untitled", cutlass_models::Rational::FPS_24),
            cache,
            history: History::new(undo_limit),
            project_path: None,
            decoder_pool: DecoderPool::new(),
            config,
        })
    }

    pub fn with_project(config: EngineConfig, project: Project) -> std::io::Result<Self> {
        let undo_limit = config.undo_limit;
        let cache = FrameCache::new(config.cache_dir.clone(), config.cache_budget_bytes)?;
        Ok(Self {
            project,
            cache,
            history: History::new(undo_limit),
            project_path: None,
            decoder_pool: DecoderPool::new(),
            config,
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

    /// Apply a wire command. On success, pushes the inverse action onto the undo stack.
    pub fn apply(&mut self, command: Command) -> Result<ApplyOutcome, EngineError> {
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

    /// Decode the topmost enabled video clip at `time` and return an RGBA preview frame.
    pub fn get_frame(&mut self, time: RationalTime) -> Result<RgbaFrame, EngineError> {
        preview::get_frame(
            &self.project,
            &self.cache,
            &mut self.decoder_pool,
            time,
        )
    }

    pub fn undo(&mut self) -> bool {
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
