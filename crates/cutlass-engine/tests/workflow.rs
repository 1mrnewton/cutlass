//! End-to-end engine workflows: import, edit, undo/redo.

mod common;

use common::{import_asset, rt, small_video_asset, temp_engine, tr};
use cutlass_commands::{Command, EditCommand, EditOutcome, ProjectCommand};
use cutlass_engine::ApplyOutcome;
use cutlass_models::TrackKind;

#[test]
fn import_registers_media_and_cache() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let (_dir, mut engine) = temp_engine();
    let media_id = import_asset(&mut engine, &path);

    let media = engine.project().media(media_id).expect("media");
    assert_eq!(media.path(), path.as_path());
    assert!(media.width > 0);
    assert!(media.height > 0);
    assert!(media.duration.value > 0);
}

#[test]
fn edit_session_via_commands() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let (_dir, mut engine) = temp_engine();
    let media_id = import_asset(&mut engine, &path);
    let track = common::add_track(&mut engine, TrackKind::Video, "V1");

    let clip = match engine
        .apply(Command::Edit(EditCommand::AddClip {
            track,
            media: media_id,
            source: tr(0, 48),
            start: rt(0),
        }))
        .expect("add clip")
    {
        ApplyOutcome::Edited(EditOutcome::Created(id)) => id,
        other => panic!("unexpected {other:?}"),
    };

    let tail = match engine
        .apply(Command::Edit(EditCommand::SplitClip {
            clip,
            at: rt(24),
        }))
        .expect("split")
    {
        ApplyOutcome::Edited(EditOutcome::Created(id)) => id,
        other => panic!("unexpected {other:?}"),
    };

    engine
        .apply(Command::Edit(EditCommand::RippleDelete { clip: tail }))
        .expect("ripple delete");

    assert_eq!(engine.project().timeline().clip_count(), 1);
    assert!(engine.project().clip(clip).is_some());
    assert!(engine.project().clip(tail).is_none());
}

#[test]
fn undo_redo_roundtrip_restores_timeline() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let (_dir, mut engine) = temp_engine();
    let media_id = import_asset(&mut engine, &path);
    let track = common::add_track(&mut engine, TrackKind::Video, "V1");

    engine
        .apply(Command::Edit(EditCommand::AddClip {
            track,
            media: media_id,
            source: tr(0, 48),
            start: rt(0),
        }))
        .expect("add");

    assert_eq!(engine.project().timeline().clip_count(), 1);
    assert!(engine.can_undo());

    assert!(engine.undo());
    assert_eq!(engine.project().timeline().clip_count(), 0);
    assert!(engine.can_redo());

    assert!(engine.redo());
    assert_eq!(engine.project().timeline().clip_count(), 1);
}

#[test]
fn failed_command_does_not_mutate_project() {
    let (_dir, mut engine) = temp_engine();
    let track = common::add_track(&mut engine, TrackKind::Video, "V1");
    let fake_media = cutlass_models::MediaId::from_raw(999);

    let err = match engine.apply(Command::Edit(EditCommand::AddClip {
        track,
        media: fake_media,
        source: tr(0, 48),
        start: rt(0),
    })) {
        Err(e) => e,
        Ok(_) => panic!("expected unknown media error"),
    };

    assert!(format!("{err}").contains("unknown media"));
    assert_eq!(engine.project().timeline().clip_count(), 0);
    assert!(engine.can_undo(), "add-track remains undoable after failed add-clip");
}

#[test]
fn import_via_command_from_missing_file_fails_cleanly() {
    let (_dir, mut engine) = temp_engine();
    let err = match engine.apply(Command::Project(ProjectCommand::Import {
        path: "/nonexistent/cutlass-engine.mp4".into(),
    })) {
        Err(e) => e,
        Ok(_) => panic!("expected import error"),
    };
    assert_eq!(engine.project().media_count(), 0);
    assert!(!engine.can_undo());
    assert!(format!("{err}").contains("Open") || format!("{err}").contains("open"));
}
