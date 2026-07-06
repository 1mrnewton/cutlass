//! Compound undo: multi-command gestures grouped into one history entry
//! (Phase 6). Mirrors the worker's gesture batches — new-lane move,
//! delete-that-empties-its-lane — and the rollback path for failed gestures.

mod common;

use common::{add_generated, add_track, rt, temp_engine, tr};
use cutlass_commands::{Command, EditCommand};
use cutlass_models::{Generator, TrackKind};

fn text(content: &str) -> Generator {
    Generator::text(content)
}

#[test]
fn new_lane_move_undoes_as_one_entry() {
    let (_dir, mut engine) = temp_engine();
    let t1 = add_track(&mut engine, TrackKind::Text, "T1");
    let clip = add_generated(&mut engine, t1, text("a"), tr(0, 24));

    // The worker's new-lane move gesture: create the destination lane, move
    // the clip, remove the emptied source lane.
    engine.begin_group();
    let t2 = add_track(&mut engine, TrackKind::Text, "T2");
    engine
        .apply(Command::Edit(EditCommand::MoveClip {
            clip,
            to_track: t2,
            start: rt(10),
        }))
        .expect("move clip");
    engine
        .apply(Command::Edit(EditCommand::RemoveTrack { track: t1 }))
        .expect("remove emptied source lane");
    engine.commit_group();

    assert_eq!(engine.project().timeline().track_count(), 1);
    assert_eq!(engine.project().timeline().track_of(clip), Some(t2));
    assert_eq!(engine.project().clip(clip).unwrap().start().value, 10);

    assert!(engine.undo(), "one undo reverts the whole gesture");
    assert_eq!(engine.project().timeline().track_count(), 1);
    assert!(engine.project().timeline().track(t2).is_none());
    assert_eq!(engine.project().timeline().track_of(clip), Some(t1));
    assert_eq!(engine.project().clip(clip).unwrap().start().value, 0);

    assert!(engine.redo(), "one redo replays the whole gesture");
    assert_eq!(engine.project().timeline().track_count(), 1);
    assert!(engine.project().timeline().track(t1).is_none());
    assert_eq!(engine.project().timeline().track_of(clip), Some(t2));
    assert_eq!(engine.project().clip(clip).unwrap().start().value, 10);

    // And the entry keeps oscillating.
    assert!(engine.undo());
    assert_eq!(engine.project().timeline().track_of(clip), Some(t1));
}

#[test]
fn delete_that_empties_lane_undoes_as_one_entry() {
    let (_dir, mut engine) = temp_engine();
    let t1 = add_track(&mut engine, TrackKind::Text, "T1");
    let clip = add_generated(&mut engine, t1, text("solo"), tr(5, 10));

    // The worker's delete gesture: remove the clip, then the lane it emptied.
    engine.begin_group();
    engine
        .apply(Command::Edit(EditCommand::RemoveClip { clip }))
        .expect("remove clip");
    engine
        .apply(Command::Edit(EditCommand::RemoveTrack { track: t1 }))
        .expect("remove emptied lane");
    engine.commit_group();

    assert_eq!(engine.project().timeline().track_count(), 0);

    assert!(engine.undo(), "one undo restores clip and lane");
    assert_eq!(engine.project().timeline().track_count(), 1);
    assert_eq!(engine.project().timeline().track_of(clip), Some(t1));
    assert_eq!(engine.project().clip(clip).unwrap().start().value, 5);

    assert!(engine.redo(), "one redo removes both again");
    assert_eq!(engine.project().timeline().track_count(), 0);
    assert!(engine.project().clip(clip).is_none());
}

#[test]
fn single_command_group_behaves_like_plain_entry() {
    let (_dir, mut engine) = temp_engine();
    let t1 = add_track(&mut engine, TrackKind::Text, "T1");

    engine.begin_group();
    let clip = add_generated(&mut engine, t1, text("a"), tr(0, 10));
    engine.commit_group();

    assert!(engine.undo());
    assert!(engine.project().clip(clip).is_none());
    assert!(
        engine.project().timeline().track(t1).is_some(),
        "undo stops at the group boundary"
    );
}

#[test]
fn empty_group_is_noop() {
    let (_dir, mut engine) = temp_engine();
    let before_can_undo = engine.can_undo();

    engine.begin_group();
    engine.commit_group();

    assert_eq!(engine.can_undo(), before_can_undo);
    assert!(!engine.undo());
}

/// Every project mutation must move `revision()` forward — it keys the
/// preview worker's composited-frame cache, so a state change under an
/// unchanged revision would serve stale frames.
#[test]
fn revision_advances_with_every_mutation() {
    let (_dir, mut engine) = temp_engine();
    let r0 = engine.revision();

    let t1 = add_track(&mut engine, TrackKind::Text, "T1");
    assert!(engine.revision() > r0, "apply bumps the revision");
    add_generated(&mut engine, t1, text("a"), tr(0, 24));
    let after_edits = engine.revision();

    assert!(engine.undo());
    assert!(engine.revision() > after_edits, "undo bumps the revision");
    let after_undo = engine.revision();
    assert!(engine.redo());
    assert!(engine.revision() > after_undo, "redo bumps the revision");

    // A rolled-back gesture restores pre-group *state* under a *new*
    // revision: frames rendered mid-group must not alias the restored state.
    let before_group = engine.revision();
    engine.begin_group();
    add_track(&mut engine, TrackKind::Text, "T2");
    let mid_group = engine.revision();
    assert!(mid_group > before_group);
    engine.rollback_group();
    assert!(engine.revision() > mid_group, "rollback bumps the revision");

    // An empty group's rollback mutated nothing: no bump.
    let idle = engine.revision();
    engine.begin_group();
    engine.rollback_group();
    assert_eq!(engine.revision(), idle);
}

#[test]
fn rollback_restores_state_and_preserves_redo() {
    let (_dir, mut engine) = temp_engine();
    let t1 = add_track(&mut engine, TrackKind::Text, "T1");
    let clip = add_generated(&mut engine, t1, text("a"), tr(0, 24));

    assert!(engine.undo(), "park the clip add on the redo stack");
    assert!(engine.can_redo());

    // A gesture that mutates and then fails: rollback must revert the
    // mutation and keep the redo stack (a failed gesture is a full no-op).
    engine.begin_group();
    let t2 = add_track(&mut engine, TrackKind::Text, "T2");
    engine.rollback_group();

    assert_eq!(engine.project().timeline().track_count(), 1);
    assert!(engine.project().timeline().track(t2).is_none());
    assert!(engine.can_redo(), "rolled-back gesture preserves redo");

    assert!(engine.redo());
    assert_eq!(engine.project().timeline().track_of(clip), Some(t1));
}
