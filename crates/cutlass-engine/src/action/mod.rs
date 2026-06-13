//! Command dispatch: undoable timeline edits and session project commands.

mod dispatch;
mod edit;
pub(crate) mod project;

pub use dispatch::{ApplyOutcome, dispatch};

use cutlass_cache::FrameCache;
use cutlass_models::Project;

use crate::error::EngineError;

/// Session surface passed to every command handler.
pub struct ApplyContext<'a> {
    pub project: &'a mut Project,
    pub cache: &'a FrameCache,
    pub project_path: &'a mut Option<std::path::PathBuf>,
    pub history: &'a mut History,
}

/// A runtime edit action. Consuming `apply` runs the edit and returns its inverse.
pub trait EditAction: Send {
    fn apply(self: Box<Self>, ctx: &mut ApplyContext<'_>) -> Result<Box<dyn EditAction>, EngineError>;
}

/// Inverses for a multi-command gesture, undone/redone as one history entry.
///
/// `actions` holds the inverses in the order their commands were executed;
/// `apply` runs them in reverse (last command reverted first) and returns a
/// compound of the produced inverses, so the entry oscillates like any single
/// action. Also the inverse shape of engine-internal composites
/// (e.g. `RippleInsert` = shift + add).
pub(crate) struct CompoundAction {
    pub(crate) actions: Vec<Box<dyn EditAction>>,
}

impl EditAction for CompoundAction {
    fn apply(self: Box<Self>, ctx: &mut ApplyContext<'_>) -> Result<Box<dyn EditAction>, EngineError> {
        let mut inverses = Vec::with_capacity(self.actions.len());
        for action in self.actions.into_iter().rev() {
            inverses.push(action.apply(ctx)?);
        }
        Ok(Box::new(CompoundAction { actions: inverses }))
    }
}

/// Inverse-command undo/redo stacks.
///
/// A *group* collects the inverses of several dispatched commands into one
/// undo entry (one gesture = one Ctrl+Z): [`begin_group`](Self::begin_group),
/// then dispatch the batch, then [`commit_group`](Self::commit_group). While a
/// group is open, recorded inverses are held aside and the redo stack is left
/// untouched, so an abandoned group (see `Engine::rollback_group`) leaves
/// history exactly as it was.
#[derive(Default)]
pub struct History {
    undo: Vec<Box<dyn EditAction>>,
    redo: Vec<Box<dyn EditAction>>,
    limit: usize,
    pending: Option<Vec<Box<dyn EditAction>>>,
}

impl History {
    pub fn new(limit: usize) -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
            limit: limit.max(1),
            pending: None,
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn in_group(&self) -> bool {
        self.pending.is_some()
    }

    /// Start collecting recorded inverses into a single compound entry.
    /// Groups don't nest; a stray second begin keeps the open group.
    pub fn begin_group(&mut self) {
        debug_assert!(self.pending.is_none(), "history group already open");
        if self.pending.is_none() {
            self.pending = Some(Vec::new());
        }
    }

    /// Close the open group and push it as one undo entry. No-op when the
    /// group is empty (the gesture made no edits) or no group is open.
    pub fn commit_group(&mut self) {
        debug_assert!(self.pending.is_some(), "no history group open");
        let Some(mut actions) = self.pending.take() else {
            return;
        };
        match actions.len() {
            0 => {}
            1 => self.record_do(actions.pop().expect("len checked")),
            _ => self.record_do(Box::new(CompoundAction { actions })),
        }
    }

    /// Detach the open group's collected inverses without recording them
    /// (for rollback: the caller applies them in reverse order).
    pub fn take_group(&mut self) -> Vec<Box<dyn EditAction>> {
        debug_assert!(self.pending.is_some(), "no history group open");
        self.pending.take().unwrap_or_default()
    }

    pub fn record_do(&mut self, inverse: Box<dyn EditAction>) {
        if let Some(pending) = self.pending.as_mut() {
            // Redo clearing waits for commit_group so a rolled-back group
            // leaves the redo stack intact.
            pending.push(inverse);
            return;
        }
        if self.undo.len() >= self.limit {
            self.undo.remove(0);
        }
        self.undo.push(inverse);
        self.redo.clear();
    }

    pub fn pop_undo(&mut self) -> Option<Box<dyn EditAction>> {
        self.undo.pop()
    }

    pub fn push_redo(&mut self, inverse: Box<dyn EditAction>) {
        if self.redo.len() >= self.limit {
            self.redo.remove(0);
        }
        self.redo.push(inverse);
    }

    pub fn pop_redo(&mut self) -> Option<Box<dyn EditAction>> {
        self.redo.pop()
    }

    pub fn push_undo(&mut self, inverse: Box<dyn EditAction>) {
        if self.undo.len() >= self.limit {
            self.undo.remove(0);
        }
        self.undo.push(inverse);
    }

    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
        self.pending = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::edit::{
        add_clip, add_generated, add_track, link_clips, move_clip, remove_media,
        ripple_delete, ripple_insert, set_track_flags, shift_clips, split_clip, trim_clip,
    };
    use cutlass_cache::FrameCache;
    use cutlass_models::{
        Clip, Generator, MediaSource, Rational, RationalTime, TimeRange, TrackKind,
    };

    fn setup() -> (tempfile::TempDir, Project, FrameCache) {
        let dir = tempfile::tempdir().unwrap();
        let cache = FrameCache::new(dir.path().join("cache"), 1024 * 1024).unwrap();
        let project = Project::new("test", Rational::FPS_24);
        (dir, project, cache)
    }

    fn test_ctx<'a>(
        project: &'a mut Project,
        cache: &'a FrameCache,
        project_path: &'a mut Option<std::path::PathBuf>,
        history: &'a mut History,
    ) -> ApplyContext<'a> {
        ApplyContext {
            project,
            cache,
            project_path,
            history,
        }
    }

    #[test]
    fn add_clip_inverse_oscillates() {
        let (_dir, mut project, cache) = setup();
        let media_id = project.add_media(MediaSource::new(
            "/tmp/x.mp4",
            1920,
            1080,
            Rational::FPS_24,
            240,
            true,
        ));
        let track = project.add_track(TrackKind::Video, "V1");

        let mut project_path = None;
        let mut history = History::new(32);
        let mut ctx = test_ctx(&mut project, &cache, &mut project_path, &mut history);

        let (id, inv1) = add_clip::execute(
            &mut ctx,
            track,
            media_id,
            TimeRange::at_rate(0, 48, Rational::FPS_24),
            RationalTime::new(0, Rational::FPS_24),
        )
        .unwrap();
        assert_eq!(ctx.project.timeline().clip_count(), 1);

        let inv2 = inv1.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.timeline().clip_count(), 0);

        let inv3 = inv2.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.timeline().clip_count(), 1);
        assert!(ctx.project.clip(id).is_some());

        let _ = inv3.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.timeline().clip_count(), 0);
    }

    #[test]
    fn remove_media_inverse_restores_snapshot() {
        let (_dir, mut project, cache) = setup();
        let media = MediaSource::new("/tmp/y.mp4", 1280, 720, Rational::FPS_24, 100, false);
        let id = project.add_media(media.clone());

        let mut project_path = None;
        let mut history = History::new(32);
        let mut ctx = test_ctx(&mut project, &cache, &mut project_path, &mut history);

        let inv1 = Box::new(remove_media::RemoveMediaAction { media: id });
        let inv2 = inv1.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.media_count(), 0);

        let inv3 = inv2.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.media_count(), 1);
        assert_eq!(ctx.project.media(id).unwrap(), &media);

        let _ = inv3.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.media_count(), 0);
    }

    #[test]
    fn split_clip_inverse_oscillates() {
        let (_dir, mut project, cache) = setup();
        let media_id = project.add_media(MediaSource::new(
            "/tmp/split.mp4",
            1920,
            1080,
            Rational::FPS_24,
            240,
            false,
        ));
        let track = project.add_track(TrackKind::Video, "V1");
        let clip_id = project
            .add_clip(
                track,
                media_id,
                TimeRange::at_rate(0, 48, Rational::FPS_24),
                RationalTime::new(0, Rational::FPS_24),
            )
            .unwrap();

        let mut project_path = None;
        let mut history = History::new(32);
        let mut ctx = test_ctx(&mut project, &cache, &mut project_path, &mut history);

        let (_tail, inv1) = split_clip::execute(&mut ctx, clip_id, rt(24)).unwrap();
        assert_eq!(ctx.project.timeline().clip_count(), 2);
        assert_eq!(
            ctx.project.clip(clip_id).unwrap().timeline.duration.value,
            24
        );

        let inv2 = inv1.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.timeline().clip_count(), 1);
        assert_eq!(
            ctx.project.clip(clip_id).unwrap().timeline.duration.value,
            48
        );

        let inv3 = inv2.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.timeline().clip_count(), 2);
        assert_eq!(
            ctx.project.clip(clip_id).unwrap().timeline.duration.value,
            24
        );

        let _ = inv3.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.timeline().clip_count(), 1);
    }

    #[test]
    fn ripple_delete_inverse_oscillates() {
        let (_dir, mut project, cache) = setup();
        let track = project.add_track(TrackKind::Adjustment, "FX");
        let first = project
            .timeline_mut()
            .add_clip(
                track,
                Clip::generated(Generator::Adjustment, tr(0, 10)),
            )
            .unwrap();
        let second = project
            .timeline_mut()
            .add_clip(
                track,
                Clip::generated(Generator::Adjustment, tr(20, 10)),
            )
            .unwrap();

        let mut project_path = None;
        let mut history = History::new(32);
        let mut ctx = test_ctx(&mut project, &cache, &mut project_path, &mut history);

        let inv1 = ripple_delete::execute(&mut ctx, first).unwrap();
        assert_eq!(ctx.project.timeline().clip_count(), 1);
        assert_eq!(
            ctx.project.clip(second).unwrap().timeline.start.value,
            10
        );

        let inv2 = inv1.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.timeline().clip_count(), 2);
        assert_eq!(
            ctx.project.clip(second).unwrap().timeline.start.value,
            20
        );

        let _ = inv2.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.timeline().clip_count(), 1);
    }

    #[test]
    fn ripple_delete_middle_of_three_adjacent_oscillates() {
        let (_dir, mut project, cache) = setup();
        let track = project.add_track(TrackKind::Adjustment, "FX");
        let a = project
            .timeline_mut()
            .add_clip(track, Clip::generated(Generator::Adjustment, tr(0, 10)))
            .unwrap();
        let b = project
            .timeline_mut()
            .add_clip(track, Clip::generated(Generator::Adjustment, tr(10, 10)))
            .unwrap();
        let c = project
            .timeline_mut()
            .add_clip(track, Clip::generated(Generator::Adjustment, tr(20, 10)))
            .unwrap();

        let mut project_path = None;
        let mut history = History::new(32);
        let mut ctx = test_ctx(&mut project, &cache, &mut project_path, &mut history);

        let inv1 = ripple_delete::execute(&mut ctx, b).unwrap();
        assert_eq!(ctx.project.clip(c).unwrap().start().value, 10);

        let inv2 = inv1.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.clip(b).unwrap().start().value, 10);
        assert_eq!(ctx.project.clip(c).unwrap().start().value, 20);

        let _ = inv2.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.clip(c).unwrap().start().value, 10);
        assert!(ctx.project.clip(b).is_none());
        let _ = a;
    }

    #[test]
    fn trim_clip_inverse_restores_snapshot() {
        let (_dir, mut project, cache) = setup();
        let media_id = project.add_media(MediaSource::new(
            "/tmp/trim.mp4",
            1280,
            720,
            Rational::FPS_24,
            240,
            false,
        ));
        let track = project.add_track(TrackKind::Video, "V1");
        let clip_id = project
            .add_clip(
                track,
                media_id,
                TimeRange::at_rate(0, 48, Rational::FPS_24),
                RationalTime::new(0, Rational::FPS_24),
            )
            .unwrap();
        let before = ctx_project_clip(&project, clip_id);

        let mut project_path = None;
        let mut history = History::new(32);
        let mut ctx = test_ctx(&mut project, &cache, &mut project_path, &mut history);

        let inv1 = trim_clip::execute(&mut ctx, clip_id, tr(10, 28)).unwrap();
        assert_eq!(ctx.project.clip(clip_id).unwrap().timeline.start.value, 10);

        let inv2 = inv1.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.clip(clip_id).unwrap().timeline, before.timeline);
        assert_eq!(
            ctx.project.clip(clip_id).unwrap().source_range(),
            before.source_range()
        );

        let _ = inv2.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.clip(clip_id).unwrap().timeline.start.value, 10);
    }

    #[test]
    fn move_clip_inverse_oscillates() {
        let (_dir, mut project, cache) = setup();
        let v1 = project.add_track(TrackKind::Text, "T1");
        let v2 = project.add_track(TrackKind::Text, "T2");
        let clip_id = project
            .timeline_mut()
            .add_clip(
                v1,
                Clip::generated(Generator::text("x"), tr(5, 15)),
            )
            .unwrap();

        let mut project_path = None;
        let mut history = History::new(32);
        let mut ctx = test_ctx(&mut project, &cache, &mut project_path, &mut history);

        let inv1 = move_clip::execute(&mut ctx, clip_id, v2, rt(40)).unwrap();
        assert_eq!(ctx.project.timeline().track_of(clip_id), Some(v2));

        let inv2 = inv1.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.timeline().track_of(clip_id), Some(v1));
        assert_eq!(ctx.project.clip(clip_id).unwrap().start().value, 5);

        let _ = inv2.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.timeline().track_of(clip_id), Some(v2));
    }

    #[test]
    fn add_generated_inverse_oscillates() {
        let (_dir, mut project, cache) = setup();
        let track = project.add_track(TrackKind::Sticker, "S1");

        let mut project_path = None;
        let mut history = History::new(32);
        let mut ctx = test_ctx(&mut project, &cache, &mut project_path, &mut history);

        let (id, inv1) = add_generated::execute(
            &mut ctx,
            track,
            Generator::SolidColor {
                rgba: [9, 8, 7, 6],
            },
            tr(0, 12),
        )
        .unwrap();
        assert_eq!(ctx.project.timeline().clip_count(), 1);

        let inv2 = inv1.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.timeline().clip_count(), 0);

        let inv3 = inv2.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.timeline().clip_count(), 1);
        assert!(ctx.project.clip(id).is_some());

        let _ = inv3.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.timeline().clip_count(), 0);
    }

    #[test]
    fn split_generated_clip_inverse_oscillates() {
        let (_dir, mut project, cache) = setup();
        let track = project.add_track(TrackKind::Text, "T1");
        let clip_id = project
            .timeline_mut()
            .add_clip(
                track,
                Clip::generated(
                    Generator::text("split me"),
                    tr(0, 20),
                ),
            )
            .unwrap();

        let mut project_path = None;
        let mut history = History::new(32);
        let mut ctx = test_ctx(&mut project, &cache, &mut project_path, &mut history);

        let (_tail, inv1) = split_clip::execute(&mut ctx, clip_id, rt(10)).unwrap();
        assert_eq!(ctx.project.timeline().clip_count(), 2);

        let inv2 = inv1.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.timeline().clip_count(), 1);
        assert_eq!(
            ctx.project.clip(clip_id).unwrap().timeline.duration.value,
            20
        );

        let _ = inv2.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.timeline().clip_count(), 2);
    }

    #[test]
    fn shift_clips_inverse_selects_exactly_the_shifted_set() {
        let (_dir, mut project, cache) = setup();
        let track = project.add_track(TrackKind::Adjustment, "FX");
        // A [60,80) stays put; B [100,120) shifts left into [80,100). The
        // inverse boundary (B's new start, 80) must re-select only B — a
        // naive `from + delta` boundary (also 80 here, but 60-adjacent cases
        // differ) could drag A along.
        let a = project
            .timeline_mut()
            .add_clip(track, Clip::generated(Generator::Adjustment, tr(60, 20)))
            .unwrap();
        let b = project
            .timeline_mut()
            .add_clip(track, Clip::generated(Generator::Adjustment, tr(100, 20)))
            .unwrap();

        let mut project_path = None;
        let mut history = History::new(32);
        let mut ctx = test_ctx(&mut project, &cache, &mut project_path, &mut history);

        let inv = shift_clips::execute(&mut ctx, track, rt(100), rt(-20)).unwrap();
        assert_eq!(ctx.project.clip(b).unwrap().start().value, 80);

        let redo = inv.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.clip(a).unwrap().start().value, 60, "A untouched");
        assert_eq!(ctx.project.clip(b).unwrap().start().value, 100);

        let _ = redo.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.clip(a).unwrap().start().value, 60);
        assert_eq!(ctx.project.clip(b).unwrap().start().value, 80);
    }

    #[test]
    fn ripple_insert_shifts_then_places_and_oscillates() {
        let (_dir, mut project, cache) = setup();
        let media_id = project.add_media(MediaSource::new(
            "/tmp/ri.mp4",
            1920,
            1080,
            Rational::FPS_24,
            240,
            false,
        ));
        let track = project.add_track(TrackKind::Video, "V1");
        let first = project
            .add_clip(
                track,
                media_id,
                TimeRange::at_rate(0, 48, Rational::FPS_24),
                RationalTime::new(0, Rational::FPS_24),
            )
            .unwrap();
        let second = project
            .add_clip(
                track,
                media_id,
                TimeRange::at_rate(48, 48, Rational::FPS_24),
                RationalTime::new(48, Rational::FPS_24),
            )
            .unwrap();

        let mut project_path = None;
        let mut history = History::new(32);
        let mut ctx = test_ctx(&mut project, &cache, &mut project_path, &mut history);

        // Insert 24 ticks at the boundary between the two clips.
        let (inserted, inv) = ripple_insert::execute(
            &mut ctx,
            track,
            media_id,
            TimeRange::at_rate(100, 24, Rational::FPS_24),
            rt(48),
        )
        .unwrap();
        assert_eq!(ctx.project.clip(first).unwrap().start().value, 0);
        assert_eq!(ctx.project.clip(inserted).unwrap().start().value, 48);
        assert_eq!(ctx.project.clip(second).unwrap().start().value, 72);

        let redo = inv.apply(&mut ctx).unwrap();
        assert!(ctx.project.clip(inserted).is_none());
        assert_eq!(ctx.project.clip(second).unwrap().start().value, 48);

        let _ = redo.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.timeline().clip_count(), 3);
        assert_eq!(ctx.project.clip(second).unwrap().start().value, 72);
    }

    #[test]
    fn ripple_insert_rejected_placement_restores_shift() {
        let (_dir, mut project, cache) = setup();
        let media_id = project.add_media(MediaSource::new(
            "/tmp/ri-bad.mp4",
            1920,
            1080,
            Rational::FPS_24,
            240,
            false,
        ));
        let track = project.add_track(TrackKind::Video, "V1");
        let clip = project
            .add_clip(
                track,
                media_id,
                TimeRange::at_rate(0, 48, Rational::FPS_24),
                RationalTime::new(0, Rational::FPS_24),
            )
            .unwrap();

        let mut project_path = None;
        let mut history = History::new(32);
        let mut ctx = test_ctx(&mut project, &cache, &mut project_path, &mut history);

        // Source range exceeds the media's 240 ticks → add_clip rejects after
        // the shift already ran; the shift must be rolled back atomically.
        let result = ripple_insert::execute(
            &mut ctx,
            track,
            media_id,
            TimeRange::at_rate(200, 100, Rational::FPS_24),
            rt(0),
        );
        assert!(result.is_err());
        assert_eq!(ctx.project.timeline().clip_count(), 1);
        assert_eq!(ctx.project.clip(clip).unwrap().start().value, 0, "shift undone");
    }

    #[test]
    fn compound_action_applies_in_reverse_and_oscillates() {
        let (_dir, mut project, cache) = setup();

        let mut project_path = None;
        let mut history = History::new(32);
        let mut ctx = test_ctx(&mut project, &cache, &mut project_path, &mut history);

        // Gesture: AddTrack + AddGenerated. Inverses are stored in execution
        // order; undo must run them in reverse (remove the clip before its
        // track) or the second inverse hits an unknown clip.
        let (track, inv_track) =
            add_track::execute(&mut ctx, TrackKind::Text, "T1", None).unwrap();
        let (clip, inv_clip) = add_generated::execute(
            &mut ctx,
            track,
            Generator::text("x"),
            tr(0, 10),
        )
        .unwrap();

        let compound: Box<dyn EditAction> = Box::new(CompoundAction {
            actions: vec![inv_track, inv_clip],
        });

        let redo = compound.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.timeline().track_count(), 0);
        assert_eq!(ctx.project.timeline().clip_count(), 0);

        let undo = redo.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.timeline().track_count(), 1);
        assert_eq!(ctx.project.timeline().track_of(clip), Some(track));

        let _ = undo.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.timeline().track_count(), 0);
        assert_eq!(ctx.project.timeline().clip_count(), 0);
    }

    #[test]
    fn add_transition_inverse_oscillates() {
        use crate::action::edit::set_transition;
        let (_dir, mut project, cache) = setup();
        let track = project.add_track(TrackKind::Adjustment, "FX");
        let left = project
            .timeline_mut()
            .add_clip(track, Clip::generated(Generator::Adjustment, tr(0, 24)))
            .unwrap();
        let _right = project
            .timeline_mut()
            .add_clip(track, Clip::generated(Generator::Adjustment, tr(24, 24)))
            .unwrap();

        let mut project_path = None;
        let mut history = History::new(32);
        let mut ctx = test_ctx(&mut project, &cache, &mut project_path, &mut history);

        let inv1 = set_transition::add_transition(&mut ctx, left, "crossfade").unwrap();
        assert!(
            ctx.project.timeline().track(track).unwrap().transition_at(left).is_some(),
            "transition added"
        );

        let inv2 = inv1.apply(&mut ctx).unwrap();
        assert!(ctx.project.timeline().track(track).unwrap().transition_at(left).is_none());

        let _ = inv2.apply(&mut ctx).unwrap();
        assert!(ctx.project.timeline().track(track).unwrap().transition_at(left).is_some());
    }

    #[test]
    fn structural_edit_prunes_dead_junction_and_undo_restores_it() {
        use crate::action::dispatch;
        use cutlass_commands::{Command, EditCommand};
        let (_dir, mut project, cache) = setup();
        let track = project.add_track(TrackKind::Adjustment, "FX");
        let left = project
            .timeline_mut()
            .add_clip(track, Clip::generated(Generator::Adjustment, tr(0, 24)))
            .unwrap();
        let _right = project
            .timeline_mut()
            .add_clip(track, Clip::generated(Generator::Adjustment, tr(24, 24)))
            .unwrap();
        project.add_transition(left, "crossfade").unwrap();

        let mut project_path = None;
        let mut history = History::new(32);
        let mut ctx = test_ctx(&mut project, &cache, &mut project_path, &mut history);

        // Move the left clip away: the junction breaks and the transition is
        // pruned as part of the edit.
        let (_out, inverse) = dispatch(
            Command::Edit(EditCommand::MoveClip {
                clip: left,
                to_track: track,
                start: rt(100),
            }),
            &mut ctx,
        )
        .unwrap();
        let inverse = inverse.expect("structural edit records an inverse");
        assert!(
            ctx.project.timeline().track(track).unwrap().transition_at(left).is_none(),
            "broken junction pruned"
        );

        // Undo restores both the clip position and the pruned transition.
        let redo = inverse.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.clip(left).unwrap().start().value, 0);
        assert!(
            ctx.project.timeline().track(track).unwrap().transition_at(left).is_some(),
            "undo brings the junction back"
        );

        // Redo moves again and re-prunes.
        let _ = redo.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.clip(left).unwrap().start().value, 100);
        assert!(ctx.project.timeline().track(track).unwrap().transition_at(left).is_none());
    }

    #[test]
    fn set_track_flags_inverse_oscillates() {
        let (_dir, mut project, cache) = setup();
        let track = project.add_track(TrackKind::Video, "V1");
        assert!(project.timeline().track(track).unwrap().enabled);

        let mut project_path = None;
        let mut history = History::new(32);
        let mut ctx = test_ctx(&mut project, &cache, &mut project_path, &mut history);

        // Disable the track; only `enabled` changes, the rest stay put.
        let inv1 = set_track_flags::execute(&mut ctx, track, Some(false), None, None, None).unwrap();
        assert!(!ctx.project.timeline().track(track).unwrap().enabled);
        assert!(!ctx.project.timeline().track(track).unwrap().muted);
        assert!(!ctx.project.timeline().track(track).unwrap().locked);

        // Undo restores the full snapshot.
        let inv2 = inv1.apply(&mut ctx).unwrap();
        assert!(ctx.project.timeline().track(track).unwrap().enabled);

        // Redo disables again.
        let _ = inv2.apply(&mut ctx).unwrap();
        assert!(!ctx.project.timeline().track(track).unwrap().enabled);
    }

    #[test]
    fn link_clips_inverse_restores_prior_links_and_oscillates() {
        let (_dir, mut project, cache) = setup();
        let track = project.add_track(TrackKind::Adjustment, "FX");
        let a = project
            .timeline_mut()
            .add_clip(track, Clip::generated(Generator::Adjustment, tr(0, 10)))
            .unwrap();
        let b = project
            .timeline_mut()
            .add_clip(track, Clip::generated(Generator::Adjustment, tr(20, 10)))
            .unwrap();

        let mut project_path = None;
        let mut history = History::new(32);
        let mut ctx = test_ctx(&mut project, &cache, &mut project_path, &mut history);

        let inv1 = link_clips::execute(&mut ctx, &[a, b]).unwrap();
        let group = ctx.project.clip(a).unwrap().link;
        assert!(group.is_some());
        assert_eq!(ctx.project.clip(b).unwrap().link, group);

        // Re-linking `b` into a new pair must let undo restore the old group.
        let c = ctx
            .project
            .timeline_mut()
            .add_clip(track, Clip::generated(Generator::Adjustment, tr(40, 10)))
            .unwrap();
        let inv2 = link_clips::execute(&mut ctx, &[b, c]).unwrap();
        let pair = ctx.project.clip(b).unwrap().link;
        assert_ne!(pair, group);
        assert_eq!(ctx.project.clip(c).unwrap().link, pair);

        let redo2 = inv2.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.clip(b).unwrap().link, group, "old link restored");
        assert_eq!(ctx.project.clip(c).unwrap().link, None);

        let _ = redo2.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.clip(b).unwrap().link, pair);

        // First link's inverse restores `a` to unlinked.
        let inv1b = inv1.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.clip(a).unwrap().link, None);
        let _ = inv1b.apply(&mut ctx).unwrap();
        assert_eq!(ctx.project.clip(a).unwrap().link, group);
    }

    #[test]
    fn link_clips_unknown_clip_mutates_nothing() {
        let (_dir, mut project, cache) = setup();
        let track = project.add_track(TrackKind::Adjustment, "FX");
        let a = project
            .timeline_mut()
            .add_clip(track, Clip::generated(Generator::Adjustment, tr(0, 10)))
            .unwrap();

        let mut project_path = None;
        let mut history = History::new(32);
        let mut ctx = test_ctx(&mut project, &cache, &mut project_path, &mut history);

        let missing = cutlass_models::ClipId::from_raw(999);
        assert!(link_clips::execute(&mut ctx, &[a, missing]).is_err());
        assert_eq!(ctx.project.clip(a).unwrap().link, None, "validated before mutating");
    }

    #[test]
    fn set_track_flags_unknown_track_errors() {
        let (_dir, mut project, cache) = setup();
        let mut project_path = None;
        let mut history = History::new(32);
        let mut ctx = test_ctx(&mut project, &cache, &mut project_path, &mut history);
        let missing = cutlass_models::TrackId::from_raw(999);
        assert!(set_track_flags::execute(&mut ctx, missing, Some(false), None, None, None).is_err());
    }

    /// Inert action for history bookkeeping tests.
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
    fn group_commits_as_single_entry() {
        let mut history = History::new(8);
        history.begin_group();
        history.record_do(Box::new(NoopAction));
        history.record_do(Box::new(NoopAction));
        assert!(!history.can_undo(), "pending group is not undoable yet");
        history.commit_group();
        assert!(history.pop_undo().is_some());
        assert!(history.pop_undo().is_none(), "group collapsed to one entry");
    }

    #[test]
    fn empty_group_records_nothing() {
        let mut history = History::new(8);
        history.begin_group();
        history.commit_group();
        assert!(!history.can_undo());
    }

    #[test]
    fn group_defers_redo_clear_until_commit() {
        let mut history = History::new(8);
        history.push_redo(Box::new(NoopAction));

        history.begin_group();
        history.record_do(Box::new(NoopAction));
        let taken = history.take_group();
        assert_eq!(taken.len(), 1);
        assert!(history.can_redo(), "rolled-back group must not clear redo");
        assert!(!history.can_undo());

        history.begin_group();
        history.record_do(Box::new(NoopAction));
        history.commit_group();
        assert!(!history.can_redo(), "commit clears redo like any new edit");
        assert!(history.can_undo());
    }

    fn ctx_project_clip(project: &Project, id: cutlass_models::ClipId) -> cutlass_models::Clip {
        project.clip(id).unwrap().clone()
    }

    fn rt(value: i64) -> RationalTime {
        RationalTime::new(value, Rational::FPS_24)
    }

    fn tr(start: i64, duration: i64) -> TimeRange {
        TimeRange::at_rate(start, duration, Rational::FPS_24)
    }
}
