//! Session-scoped editing runtime: project state plus inverse undo/redo.
//!
//! Preview (`get_frame`) and file export delegate to the GPU `cutlass-render`
//! pipeline and the native `cutlass-encoder`; this type owns the session
//! [`Renderer`] and the undo [`History`].

use std::path::PathBuf;

use cutlass_commands::{Command, ProjectCommand};
use cutlass_models::{
    AnimationRef, AnimationSlot, ClipId, ClipParam, ClipTransform, ColorAdjustments, Filter,
    Generator, LayerStyles, MediaId, MotionBlur, ParamValue, Project, Rational, RationalTime,
    TrackKind,
};
use cutlass_render::{
    FrameSink, ParamOverrides, Renderer, ResolveOverrides, RgbaImage, SeekPolicy, export_to_file,
};

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
    /// Live gesture transform for one clip: preview frames render it instead
    /// of the committed transform until cleared. Session state only — never
    /// in the project, history, or export.
    transform_override: Option<(ClipId, ClipTransform)>,
    /// Live generator content for one clip (inspector slider preview), same
    /// session-only semantics as `transform_override`.
    generator_override: Option<(ClipId, Generator)>,
    /// Live filter/adjustment look for one clip (inspector slider preview),
    /// same session-only semantics as `transform_override`.
    look_override: Option<(ClipId, Option<Filter>, ColorAdjustments)>,
    /// Live layer styles for one clip (inspector slider preview), same
    /// session-only semantics as `transform_override`.
    styles_override: Option<(ClipId, LayerStyles)>,
    /// Live per-(clip, param) values (inspector slider preview), same
    /// session-only semantics as `transform_override`. Multiple params may be
    /// overridden on one clip at once (e.g. crop x+y during a rect drag).
    param_overrides: ParamOverrides,
    /// Live motion-blur settings for one clip (inspector shutter/samples drag).
    motion_blur_override: Option<(ClipId, MotionBlur)>,
    /// Live look-animation knobs for one clip/slot (speed/intensity/stagger).
    animation_override: Option<(ClipId, AnimationSlot, AnimationRef)>,
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
            transform_override: None,
            generator_override: None,
            look_override: None,
            styles_override: None,
            param_overrides: ParamOverrides::new(),
            motion_blur_override: None,
            animation_override: None,
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

    /// Set the project's per-project AI agent rules (`ProjectMetadata`).
    /// Metadata, not timeline state: it bypasses the command layer and is
    /// not undoable by design (like `RelinkMedia`), but it does dirty the
    /// session so the rules save with the project.
    pub fn set_agent_rules(&mut self, rules: String) {
        if self.project.metadata().agent_rules == rules {
            return;
        }
        self.project.metadata_mut().agent_rules = rules;
        self.revision += 1;
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
        let inverses = self.history.take_group();
        if inverses.is_empty() {
            return;
        }
        // The rollback mutates the project again after the group's commands
        // already bumped `revision`: bump once more so revision-keyed
        // observers (the preview frame cache) can never confuse mid-group
        // frames with the restored state.
        self.revision += 1;
        for inverse in inverses.into_iter().rev() {
            if self.run_action(inverse).is_err() {
                tracing::error!("history group rollback failed; state may be partial");
                return;
            }
        }
    }

    /// Replace the session with a fresh, unsaved project (File → New).
    ///
    /// The project starts with the persistent main video track (CapCut's
    /// magnetic lane): it is the only lane that exists without clips, and
    /// the timeline UI renders it even when empty.
    pub fn new_session(&mut self) {
        let mut project = Project::new("untitled", Rational::FPS_24);
        project.add_track(TrackKind::Video, "Main");
        self.reset_project(project);
    }

    /// Replace the session with `project` (e.g. the AI-agent sandbox replaying
    /// a validated plan). Clears history and the project path; rebaselines clean.
    pub fn reset_project(&mut self, project: Project) {
        self.project = project;
        self.history.clear();
        self.project_path = None;
        self.transform_override = None;
        self.generator_override = None;
        self.look_override = None;
        self.styles_override = None;
        self.param_overrides.clear();
        self.motion_blur_override = None;
        self.animation_override = None;
        // Media ids are project-local: the same id can name a different
        // file in the incoming project, so all id-keyed render state is
        // stale (open decoders and still bitmaps, not only proxies).
        self.renderer.reset_media_sources();
        self.revision += 1;
        self.saved_revision = self.revision;
    }

    /// Apply a wire command. On success, pushes the inverse onto the undo stack.
    pub fn apply(&mut self, command: Command) -> Result<ApplyOutcome, EngineError> {
        if let Command::Project(ProjectCommand::Export { path }) = &command {
            // Export must render full quality from the originals; this
            // renderer doubles as the preview's, so suspend its proxy
            // substitutions for the pass.
            self.renderer.set_use_proxies(false);
            let result = export_to_file(&mut self.renderer, &self.project, path);
            self.renderer.set_use_proxies(true);
            return Ok(ApplyOutcome::Exported { frames: result? });
        }

        let mut ctx = ApplyContext {
            project: &mut self.project,
            project_path: &mut self.project_path,
            history: &mut self.history,
        };
        let (outcome, inverse) = dispatch(command, &mut ctx)?;
        match outcome {
            ApplyOutcome::Opened | ApplyOutcome::Loaded => {
                // Every session shows the magnetic main lane; files saved
                // without any video track (audio-only cuts, old versions)
                // gain an empty one on entry. Pre-history, not undoable.
                if self.project.timeline().main_track().is_none() {
                    self.project.add_track(TrackKind::Video, "Main");
                }
                // Same media-id hazard as `reset_project`: the incoming
                // file's ids owe nothing to the outgoing registry.
                self.renderer.reset_media_sources();
                // The session now mirrors the file it came from: rebaseline as
                // clean (revision still bumps so observers see a change).
                self.revision += 1;
                self.saved_revision = self.revision;
            }
            ApplyOutcome::Saved => self.saved_revision = self.revision,
            // The media now names a different file: every id-keyed cache
            // (open decoder, still bitmap, proxy) belongs to the old path.
            ApplyOutcome::Relinked { media } => {
                self.renderer.invalidate_media_source(media);
                self.revision += 1;
            }
            ApplyOutcome::Edited(_)
            | ApplyOutcome::RemovedMedia { .. }
            | ApplyOutcome::Imported { .. } => self.revision += 1,
            // Unlike Open/Load, a filled template exists nowhere on disk as a
            // project file: bump without rebaselining so the session reads
            // dirty until first saved.
            ApplyOutcome::AppliedTemplate => {
                self.renderer.reset_media_sources();
                self.revision += 1;
            }
            // Writing a `.cutlasst` (like exporting an MP4) doesn't touch the
            // session project.
            ApplyOutcome::Exported { .. } | ApplyOutcome::SavedTemplate => {}
        }
        if let Some(inverse) = inverse {
            self.history.record_do(inverse);
        }
        Ok(outcome)
    }

    /// Set (or clear with `None`) the live gesture transform for one clip.
    ///
    /// While set, preview frames render this transform in place of the clip's
    /// committed one; the project, history, and export are untouched. Release
    /// commits a real `SetClipTransform` and clears the override.
    pub fn set_transform_override(&mut self, value: Option<(ClipId, ClipTransform)>) {
        self.transform_override = value;
    }

    /// Set (or clear with `None`) the live generator content for one clip —
    /// the inspector-slider analogue of
    /// [`set_transform_override`](Self::set_transform_override).
    pub fn set_generator_override(&mut self, value: Option<(ClipId, Generator)>) {
        self.generator_override = value;
    }

    /// Set (or clear with `None`) the live look for one clip — the
    /// color-grading analogue of
    /// [`set_transform_override`](Self::set_transform_override).
    pub fn set_look_override(&mut self, value: Option<(ClipId, Option<Filter>, ColorAdjustments)>) {
        self.look_override = value;
    }

    /// Set (or clear with `None`) the live layer styles for one clip — the
    /// shadow/glow/outline/background analogue of
    /// [`set_transform_override`](Self::set_transform_override).
    pub fn set_styles_override(&mut self, value: Option<(ClipId, LayerStyles)>) {
        self.styles_override = value;
    }

    /// Set (or clear with `None`) the live motion-blur settings for one clip —
    /// the shutter/samples analogue of
    /// [`set_transform_override`](Self::set_transform_override).
    pub fn set_motion_blur_override(&mut self, value: Option<(ClipId, MotionBlur)>) {
        self.motion_blur_override = value;
    }

    /// Set (or clear with `None`) the live look-animation knobs for one
    /// clip/slot — the speed/intensity/stagger analogue of
    /// [`set_transform_override`](Self::set_transform_override).
    pub fn set_animation_override(
        &mut self,
        value: Option<(ClipId, AnimationSlot, AnimationRef)>,
    ) {
        self.animation_override = value;
    }

    /// Set (or replace) one live param value for `clip` — the generic
    /// inspector-slider analogue of
    /// [`set_transform_override`](Self::set_transform_override). Uses the same
    /// [`ClipParam`] / [`ParamValue`] addressing as `SetParamConstant`.
    pub fn set_param_override(&mut self, clip: ClipId, param: ClipParam, value: ParamValue) {
        self.param_overrides.set(clip, param, value);
    }

    /// Drop every live param override for `clip` (release / abandon).
    pub fn clear_param_overrides(&mut self, clip: ClipId) {
        self.param_overrides.clear_clip(clip);
    }

    /// Drop one live param override after that param is committed.
    pub fn clear_param_override(&mut self, clip: ClipId, param: ClipParam) {
        self.param_overrides.clear_param(clip, param);
    }

    /// Read-only view of the live param-override map (tests / diagnostics).
    pub fn param_overrides(&self) -> &ParamOverrides {
        &self.param_overrides
    }

    /// True while a live preview override is set: frames rendered now show
    /// session-only state that no project revision describes.
    pub fn has_live_overrides(&self) -> bool {
        self.transform_override.is_some()
            || self.generator_override.is_some()
            || self.look_override.is_some()
            || self.styles_override.is_some()
            || !self.param_overrides.is_empty()
            || self.motion_blur_override.is_some()
            || self.animation_override.is_some()
    }

    /// Stage timings of the most recent successful preview/export render.
    pub fn last_frame_stats(&self) -> cutlass_render::FrameStats {
        self.renderer.last_frame_stats()
    }

    /// Decode `media` from `path` (a preview proxy) instead of the pool file.
    pub fn set_media_proxy(&mut self, media: MediaId, path: PathBuf) {
        self.renderer.set_proxy(media, path);
    }

    /// Remove `media`'s proxy substitution, returning decode to the pool file.
    pub fn clear_media_proxy(&mut self, media: MediaId) {
        self.renderer.clear_proxy(media);
    }

    /// The proxy path registered for `media`, if any.
    pub fn media_proxy(&self, media: MediaId) -> Option<&std::path::Path> {
        self.renderer.proxy_for(media)
    }

    /// Tight size (canvas px, at transform scale 1.0) of the content
    /// `generator` draws on the current canvas — what a preview selection box
    /// should hug, since text/path rasters are mostly transparent padding.
    /// Animated params sample at clip-local `tick`. `None` for generators the
    /// compositor doesn't draw. Served from the renderer's raster caches
    /// (`&mut self`: a miss rasterizes once, warming preview too).
    pub fn generator_content_size(
        &mut self,
        generator: &Generator,
        tick: i64,
    ) -> Option<(u32, u32)> {
        let (width, height) = cutlass_render::canvas_size(&self.project);
        self.renderer
            .generator_content_size(generator, width, height, tick)
    }

    /// Composite enabled visual layers at `time` into an RGBA preview frame.
    pub fn get_frame(&mut self, time: RationalTime) -> Result<RgbaImage, EngineError> {
        // Field-wise construction (not a `&self` helper) so `renderer` stays
        // disjoint from the override borrows under NLL.
        let overrides = ResolveOverrides {
            transform: self.transform_override,
            generator: self.generator_override.as_ref().map(|(id, g)| (*id, g)),
            look: self
                .look_override
                .as_ref()
                .map(|(id, filter, adjust)| (*id, filter.as_ref(), adjust)),
            styles: self.styles_override.as_ref().map(|(id, s)| (*id, s)),
            params: (!self.param_overrides.is_empty()).then_some(&self.param_overrides),
            motion_blur: self.motion_blur_override,
            animation: self
                .animation_override
                .as_ref()
                .map(|(id, slot, anim)| (*id, *slot, anim)),
        };
        Ok(self
            .renderer
            .render_frame_with(&self.project, time, overrides)?)
    }

    /// [`get_frame`](Self::get_frame) scaled to fit within
    /// `max_width`×`max_height` (aspect preserved, never upscaled) — what an
    /// interactive preview should request so scrubbing pays for view-sized
    /// pixels instead of the full canvas.
    pub fn get_frame_fit(
        &mut self,
        time: RationalTime,
        max_width: u32,
        max_height: u32,
    ) -> Result<RgbaImage, EngineError> {
        let overrides = ResolveOverrides {
            transform: self.transform_override,
            generator: self.generator_override.as_ref().map(|(id, g)| (*id, g)),
            look: self
                .look_override
                .as_ref()
                .map(|(id, filter, adjust)| (*id, filter.as_ref(), adjust)),
            styles: self.styles_override.as_ref().map(|(id, s)| (*id, s)),
            params: (!self.param_overrides.is_empty()).then_some(&self.param_overrides),
            motion_blur: self.motion_blur_override,
            animation: self
                .animation_override
                .as_ref()
                .map(|(id, slot, anim)| (*id, *slot, anim)),
        };
        Ok(self.renderer.render_frame_fit_with(
            &self.project,
            time,
            max_width,
            max_height,
            overrides,
        )?)
    }

    /// Partitioned gesture frames for zero-drift preview transform drags.
    /// See [`cutlass_render::Renderer::render_gesture_frames`].
    pub fn get_gesture_frames(
        &mut self,
        time: RationalTime,
        clip_id: ClipId,
        max_width: u32,
        max_height: u32,
    ) -> Result<Option<cutlass_render::GestureFrames>, EngineError> {
        Ok(self.renderer.render_gesture_frames(
            &self.project,
            time,
            clip_id,
            max_width,
            max_height,
        )?)
    }

    /// [`get_frame`](Self::get_frame) writing composited rows directly into `sink`.
    pub fn get_frame_into(
        &mut self,
        time: RationalTime,
        policy: SeekPolicy,
        sink: &mut dyn FrameSink,
    ) -> Result<(), EngineError> {
        let overrides = ResolveOverrides {
            transform: self.transform_override,
            generator: self.generator_override.as_ref().map(|(id, g)| (*id, g)),
            look: self
                .look_override
                .as_ref()
                .map(|(id, filter, adjust)| (*id, filter.as_ref(), adjust)),
            styles: self.styles_override.as_ref().map(|(id, s)| (*id, s)),
            params: (!self.param_overrides.is_empty()).then_some(&self.param_overrides),
            motion_blur: self.motion_blur_override,
            animation: self
                .animation_override
                .as_ref()
                .map(|(id, slot, anim)| (*id, *slot, anim)),
        };
        Ok(self
            .renderer
            .render_frame_into_with(&self.project, time, overrides, policy, sink)?)
    }

    /// [`get_frame_fit`](Self::get_frame_fit) writing composited rows directly into `sink`.
    pub fn get_frame_fit_into(
        &mut self,
        time: RationalTime,
        max_width: u32,
        max_height: u32,
        policy: SeekPolicy,
        sink: &mut dyn FrameSink,
    ) -> Result<(), EngineError> {
        let overrides = ResolveOverrides {
            transform: self.transform_override,
            generator: self.generator_override.as_ref().map(|(id, g)| (*id, g)),
            look: self
                .look_override
                .as_ref()
                .map(|(id, filter, adjust)| (*id, filter.as_ref(), adjust)),
            styles: self.styles_override.as_ref().map(|(id, s)| (*id, s)),
            params: (!self.param_overrides.is_empty()).then_some(&self.param_overrides),
            motion_blur: self.motion_blur_override,
            animation: self
                .animation_override
                .as_ref()
                .map(|(id, slot, anim)| (*id, *slot, anim)),
        };
        Ok(self.renderer.render_frame_fit_into_with(
            &self.project,
            time,
            max_width,
            max_height,
            overrides,
            policy,
            sink,
        )?)
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
            // Inverses are meant to be infallible once recorded. A failure
            // here means the project no longer matches what history expects,
            // so the remaining stacks can't be trusted either — clearing both
            // is safer than silently dropping one entry and marching on (which
            // is what once turned an id collision into permanent data loss).
            Err(e) => {
                tracing::error!("undo failed ({e}); clearing history to avoid corruption");
                self.history.clear();
                self.revision += 1;
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
            // See `undo`: a failed redo means history and project have
            // diverged, so drop both stacks rather than corrupt further.
            Err(e) => {
                tracing::error!("redo failed ({e}); clearing history to avoid corruption");
                self.history.clear();
                self.revision += 1;
                false
            }
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

// The mobile FFI hands a session across threads (calls serialized by the shell,
// e.g. a Swift actor whose executor hops threads), which is only sound if the
// engine is `Send`. Assert it at compile time so a non-Send component (a decoder
// handle, a GPU resource) can never sneak in silently.
const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<Engine>()
};

#[cfg(test)]
mod tests {
    use super::*;

    /// An inverse that always fails, simulating a corrupted/mismatched history
    /// entry.
    struct FailAction;

    impl EditAction for FailAction {
        fn apply(
            self: Box<Self>,
            _ctx: &mut ApplyContext<'_>,
        ) -> Result<Box<dyn EditAction>, EngineError> {
            Err(cutlass_models::ModelError::InvalidRange.into())
        }
    }

    struct NoopAction;

    impl EditAction for NoopAction {
        fn apply(
            self: Box<Self>,
            _ctx: &mut ApplyContext<'_>,
        ) -> Result<Box<dyn EditAction>, EngineError> {
            Ok(Box::new(NoopAction))
        }
    }

    #[test]
    fn failed_undo_clears_both_stacks() {
        let mut engine = Engine::new(EngineConfig::default()).expect("engine");
        // A redo entry that WOULD apply, plus a failing undo entry on top.
        engine.history.push_redo(Box::new(NoopAction));
        engine.history.push_undo(Box::new(FailAction));
        assert!(engine.can_undo() && engine.can_redo());

        assert!(!engine.undo(), "a failing inverse reports no step");
        assert!(
            !engine.can_undo() && !engine.can_redo(),
            "a diverged history is dropped wholesale rather than trusted"
        );
    }

    #[test]
    fn failed_redo_clears_both_stacks() {
        let mut engine = Engine::new(EngineConfig::default()).expect("engine");
        engine.history.push_undo(Box::new(NoopAction));
        engine.history.push_redo(Box::new(FailAction));
        assert!(engine.can_undo() && engine.can_redo());

        assert!(!engine.redo());
        assert!(!engine.can_undo() && !engine.can_redo());
    }
}
