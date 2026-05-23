//! Editor — owns the canonical `Project` and the Slint projection of
//! it, and is the only thing in the program allowed to mutate either.
//!
//! Every domain change (UI gesture or AI agent) lands here as a
//! [`Command`], is applied to the domain `Project` first, then
//! reflected into the Slint projection in O(1) via the projector's
//! per-clip patch path. The `EditorStore.project` Slint global is
//! set once at startup; after that, the UI re-renders by reacting to
//! row updates inside the `VecModel`s the projection holds.
//!
//! Undo/redo is not implemented in this slice. The seam is here
//! (commands are pure functions over plain Rust structs, projection
//! updates are diff-driven via `Effect`) so a `Vec<Snapshot>` history
//! drops in with no shape change to the apply path.

use crate::command::{self, Command, CommandError, Effect};
use crate::models::Project;
use crate::projector::Projector;
use crate::Project as SlintProject;

/// Owns the canonical project state and the projection that the UI
/// binds to. All mutations go through [`Editor::apply`].
pub struct Editor {
    project: Project,
    projector: Projector,
}

impl Editor {
    pub fn new(project: Project) -> Self {
        let projector = Projector::new(&project);
        Self { project, projector }
    }

    /// The Slint-facing snapshot of the project. Hand this to
    /// `EditorStore::set_project` exactly once at startup — after
    /// that, `apply` keeps the same `VecModel` allocations alive and
    /// patches rows in place, so the UI stays in sync without ever
    /// being handed a new `Project`.
    #[inline]
    pub fn slint_project(&self) -> &SlintProject {
        self.projector.slint_project()
    }

    /// Borrow of the canonical domain project. Used by Slint-facing
    /// helpers (e.g. snap target computation) that need to walk every
    /// clip without paying for the Slint `Model` indirection. Read-only
    /// — mutation MUST go through [`Editor::apply`].
    #[inline]
    pub fn project(&self) -> &Project {
        &self.project
    }

    /// Apply a structured command. On success the canonical project
    /// is mutated and the Slint projection is patched to match.
    ///
    /// Validation errors leave both the project and the projection
    /// untouched — the domain mutation in [`command::apply`] fails
    /// fast before any projector write, so there's no partial state
    /// to roll back.
    pub fn apply(&mut self, command: &Command) -> Result<(), CommandError> {
        let effect = command::apply(&mut self.project, command)?;
        self.reflect(&effect);
        Ok(())
    }

    /// Push an `Effect` into the Slint projection. Kept separate from
    /// `apply` so the projector path is unit-testable on its own and
    /// so a future undo path can reuse the same projection logic with
    /// an inverted effect.
    ///
    /// `debug_assert!` on every projector call — every reachable
    /// `Effect` produced by `command::apply` corresponds to a domain
    /// mutation we just made, so the projector path can't legitimately
    /// fail. A `false` return means the projector's index drifted from
    /// the domain, which is a logic bug we want to catch in debug.
    fn reflect(&mut self, effect: &Effect) {
        match effect {
            Effect::ClipMoved {
                track_id,
                clip_id,
                new_start_value,
            } => {
                debug_assert!(self.projector.move_clip(track_id, clip_id, *new_start_value));
            }
            Effect::ClipTransferred {
                source_track_id,
                target_track_id,
                clip_id,
                new_start_value,
            } => {
                debug_assert!(self.projector.transfer_clip(
                    source_track_id,
                    target_track_id,
                    clip_id,
                    *new_start_value,
                ));
            }
            Effect::ClipTransferredToNewTrack {
                source_track_id,
                new_track_id,
                new_track_name,
                new_track_kind,
                new_track_color,
                insert_at_index,
                clip_id,
                new_start_value,
            } => {
                debug_assert!(self.projector.insert_track_with_clip(
                    new_track_id,
                    new_track_name,
                    *new_track_kind,
                    *new_track_color,
                    *insert_at_index,
                    source_track_id,
                    clip_id,
                    *new_start_value,
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::sample_project;
    use slint::Model;

    #[test]
    fn apply_move_clip_updates_domain_and_projection_in_step() {
        let mut editor = Editor::new(sample_project());

        editor
            .apply(&Command::MoveClip {
                source_track_id: "1".into(),
                clip_id: "1".into(),
                target_track_id: "1".into(),
                new_start_value: 314,
            })
            .unwrap();

        assert_eq!(
            editor
                .project
                .sequence
                .tracks
                .get("1")
                .unwrap()
                .clips
                .get("1")
                .unwrap()
                .timeline_start
                .value,
            314,
        );

        let slint = editor.slint_project();
        let track = slint.sequence.tracks.row_data(0).unwrap();
        let clip = track.clips.row_data(0).unwrap();
        assert_eq!(clip.id, "1");
        assert_eq!(clip.timeline_start.value, 314);
    }

    #[test]
    fn apply_error_leaves_state_untouched() {
        let mut editor = Editor::new(sample_project());

        let before_value = editor
            .slint_project()
            .sequence
            .tracks
            .row_data(0)
            .unwrap()
            .clips
            .row_data(0)
            .unwrap()
            .timeline_start
            .value;

        let err = editor
            .apply(&Command::MoveClip {
                source_track_id: "1".into(),
                clip_id: "nope".into(),
                target_track_id: "1".into(),
                new_start_value: 999,
            })
            .unwrap_err();
        assert!(matches!(err, CommandError::UnknownClip { .. }));

        let after_value = editor
            .slint_project()
            .sequence
            .tracks
            .row_data(0)
            .unwrap()
            .clips
            .row_data(0)
            .unwrap()
            .timeline_start
            .value;
        assert_eq!(before_value, after_value);
    }

    #[test]
    fn apply_collision_drop_creates_new_lane_in_slint_projection() {
        let mut editor = Editor::new(sample_project());
        let initial_lane_count = editor.slint_project().sequence.tracks.row_count();

        // Drop Clip 3 on top of Clip 2 within V2 → must spawn a new
        // lane and put Clip 3 on it.
        editor
            .apply(&Command::MoveClip {
                source_track_id: "2".into(),
                clip_id: "3".into(),
                target_track_id: "2".into(),
                new_start_value: 10,
            })
            .unwrap();

        let slint = editor.slint_project();
        assert_eq!(slint.sequence.tracks.row_count(), initial_lane_count + 1);

        // V2 (the old row 1) is now at row 2 — and Clip 3 is gone from it.
        let v2 = slint.sequence.tracks.row_data(2).unwrap();
        assert_eq!(v2.id, "2");
        for ci in 0..v2.clips.row_count() {
            assert_ne!(v2.clips.row_data(ci).unwrap().id, "3");
        }

        // The freshly inserted lane carries Clip 3 at value=10.
        let new_lane = slint.sequence.tracks.row_data(1).unwrap();
        assert_eq!(new_lane.clips.row_count(), 1);
        let clip = new_lane.clips.row_data(0).unwrap();
        assert_eq!(clip.id, "3");
        assert_eq!(clip.timeline_start.value, 10);
    }
}
