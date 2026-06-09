//! Inverse-command undo for import and clip edits.

mod common;

use common::{import_asset, rt, small_video_asset, temp_engine, tr};
use cutlass_commands::{Command, EditCommand};
use cutlass_models::TrackKind;

#[test]
fn undo_import_removes_media() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let (_dir, mut engine) = temp_engine();
    let media_id = import_asset(&mut engine, &path);
    assert!(engine.project().media(media_id).is_some());

    assert!(engine.undo());
    assert!(engine.project().media(media_id).is_none());

    assert!(engine.redo());
    assert!(engine.project().media(media_id).is_some());
}

#[test]
fn undo_add_clip_uses_inverse_not_snapshot() {
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
        .expect("add clip");

    assert_eq!(engine.project().timeline().clip_count(), 1);
    assert!(engine.undo());
    assert_eq!(engine.project().timeline().clip_count(), 0);
    assert_eq!(engine.project().media_count(), 1, "import survives clip undo");
}

#[test]
fn undo_import_after_clip_removed() {
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
        .expect("add clip");

    assert!(engine.undo());
    assert_eq!(engine.project().timeline().clip_count(), 0);

    assert!(engine.undo());
    assert_eq!(engine.project().timeline().track_count(), 0);

    assert!(engine.undo());
    assert!(engine.project().media(media_id).is_none());
}

#[test]
fn undo_add_track_removes_track() {
    let (_dir, mut engine) = temp_engine();
    let track = common::add_track(&mut engine, TrackKind::Video, "V1");
    assert_eq!(engine.project().timeline().track_count(), 1);

    assert!(engine.undo());
    assert_eq!(engine.project().timeline().track_count(), 0);
    assert!(engine.project().timeline().track(track).is_none());

    assert!(engine.redo());
    assert_eq!(engine.project().timeline().track_count(), 1);
}
