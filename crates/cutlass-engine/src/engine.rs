//! Session-scoped editing runtime: project state plus inverse undo/redo.
//!
//! Preview (`get_frame`) and file export delegate to the GPU `cutlass-render`
//! pipeline and the native `cutlass-encoder`; this type owns the session
//! [`Renderer`] and the undo [`History`].

use std::path::PathBuf;

use cutlass_commands::{Command, ProjectCommand};
use cutlass_models::{Project, Rational, RationalTime};
use cutlass_render::{Renderer, RgbaImage, export_to_file};

use crate::action::{ApplyContext, ApplyOutcome, EditAction, History, dispatch};
use crate::error::EngineError;

/// Session configuration for [`Engine`].
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Maximum inverse actions retained on the undo stack.
    pub undo_limit: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self { undo_limit: 100 }
    }
}

/// Cutlass editing engine: project state, inverse undo/redo, and a session
/// renderer for preview and export.
pub struct Engine {
    project: Project,
    config: EngineConfig,
    history: History,
    project_path: Option<PathBuf>,
    renderer: Renderer,
    /// Session revision: bumped on every successful project mutation (edits,
    /// open/load, undo, redo). Never serialized.
    revision: u64,
    /// Revision last written to (or read from) disk; with `revision` this is
    /// the dirty flag (see [`is_dirty`](Self::is_dirty)).
    saved_revision: u64,
}

impl Engine {
    /// Build an engine with a fresh, empty project and a headless renderer.
    pub fn new(config: EngineConfig) -> Result<Self, EngineError> {
        Self::with_project(config, Project::new("untitled", Rational::FPS_24))
    }

    /// Build an engine around an existing project.
    pub fn with_project(config: EngineConfig, project: Project) -> Result<Self, EngineError> {
        let history = History::new(config.undo_limit);
        let renderer = Renderer::new_headless()?;
        Ok(Self {
            project,
            config,
            history,
            project_path: None,
            renderer,
            revision: 0,
            saved_revision: 0,
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

    /// Path last written with `Save` or read with `Open`/`Load`.
    pub fn project_path(&self) -> Option<&PathBuf> {
        self.project_path.as_ref()
    }

    /// Monotonic session revision, bumped by every successful mutation.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// True when the session has mutations not yet saved. Conservative: an undo
    /// back to saved content still reads dirty (revisions only grow).
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
    /// into one history entry, so a multi-command gesture reverts with one undo.
    pub fn begin_group(&mut self) {
        self.history.begin_group();
    }

    /// Close the open group and record it as one undo entry (no-op if empty).
    pub fn commit_group(&mut self) {
        self.history.commit_group();
    }

    /// Abort the open group: revert its commands in reverse, restoring the
    /// pre-group state. History is left untouched.
    pub fn rollback_group(&mut self) {
        for inverse in self.history.take_group().into_iter().rev() {
            if self.run_action(inverse).is_err() {
                tracing::error!("history group rollback failed; state may be partial");
                return;
            }
        }
    }

    /// Replace the session with a fresh, empty, unsaved project (File → New).
    pub fn new_session(&mut self) {
        self.reset_project(Project::new("untitled", Rational::FPS_24));
    }

    /// Replace the session with `project` (e.g. the AI-agent sandbox replaying
    /// a validated plan). Clears history and the project path; rebaselines clean.
    pub fn reset_project(&mut self, project: Project) {
        self.project = project;
        self.history.clear();
        self.project_path = None;
        self.revision += 1;
        self.saved_revision = self.revision;
    }

    /// Apply a wire command. On success, pushes the inverse onto the undo stack.
    pub fn apply(&mut self, command: Command) -> Result<ApplyOutcome, EngineError> {
        if let Command::Project(ProjectCommand::Export { path }) = &command {
            let frames = export_to_file(&mut self.renderer, &self.project, path)?;
            return Ok(ApplyOutcome::Exported { frames });
        }

        let mut ctx = ApplyContext {
            project: &mut self.project,
            project_path: &mut self.project_path,
            history: &mut self.history,
        };
        let (outcome, inverse) = dispatch(command, &mut ctx)?;
        match outcome {
            ApplyOutcome::Opened | ApplyOutcome::Loaded => {
                // The session now mirrors the file it came from: rebaseline as
                // clean (revision still bumps so observers see a change).
                self.revision += 1;
                self.saved_revision = self.revision;
            }
            ApplyOutcome::Saved => self.saved_revision = self.revision,
            ApplyOutcome::Edited(_)
            | ApplyOutcome::RemovedMedia { .. }
            | ApplyOutcome::Imported { .. }
            | ApplyOutcome::Relinked { .. } => self.revision += 1,
            // Unlike Open/Load, a filled template exists nowhere on disk as a
            // project file: bump without rebaselining so the session reads
            // dirty until first saved.
            ApplyOutcome::AppliedTemplate => self.revision += 1,
            // Writing a `.cutlasst` (like exporting an MP4) doesn't touch the
            // session project.
            ApplyOutcome::Exported { .. } | ApplyOutcome::SavedTemplate => {}
        }
        if let Some(inverse) = inverse {
            self.history.record_do(inverse);
        }
        Ok(outcome)
    }

    /// Composite enabled visual layers at `time` into an RGBA preview frame.
    pub fn get_frame(&mut self, time: RationalTime) -> Result<RgbaImage, EngineError> {
        Ok(self.renderer.render_frame(&self.project, time)?)
    }

    pub fn undo(&mut self) -> bool {
        debug_assert!(!self.history.in_group(), "undo inside an open history group");
        let Some(action) = self.history.pop_undo() else {
            return false;
        };
        match self.run_action(action) {
            Ok(inverse) => {
                self.history.push_redo(inverse);
                self.revision += 1;
                true
            }
            // Inverses are written to be infallible once recorded; on failure
            // the action is already popped and lost.
            Err(_) => false,
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
                self.revision += 1;
                true
            }
            Err(_) => false,
        }
    }

    fn run_action(
        &mut self,
        action: Box<dyn EditAction>,
    ) -> Result<Box<dyn EditAction>, EngineError> {
        let mut ctx = ApplyContext {
            project: &mut self.project,
            project_path: &mut self.project_path,
            history: &mut self.history,
        };
        action.apply(&mut ctx)
    }
}
