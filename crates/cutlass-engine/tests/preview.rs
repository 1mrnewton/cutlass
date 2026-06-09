//! Preview: import → add clip → get_frame.

mod common;

use common::{import_asset, rt, small_video_asset, temp_engine, tr};
use cutlass_commands::{Command, EditCommand};
use cutlass_models::TrackKind;

#[test]
fn get_frame_returns_rgba_for_placed_clip() {
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

    let (width, height) = {
        let media = engine.project().media(media_id).expect("media");
        (media.width, media.height)
    };
    let frame = engine.get_frame(rt(0)).expect("get_frame");

    assert_eq!(frame.width, width);
    assert_eq!(frame.height, height);
    assert_eq!(
        frame.bytes.len(),
        usize::try_from(width * height * 4).unwrap()
    );
    assert!(frame.bytes.iter().any(|&b| b != 0), "frame should not be blank");
}

#[test]
fn get_frame_after_split_still_decodes() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let (_dir, mut engine) = temp_engine();
    let media_id = import_asset(&mut engine, &path);
    let track = common::add_track(&mut engine, TrackKind::Video, "V1");

    let clip_id = match engine
        .apply(Command::Edit(EditCommand::AddClip {
            track,
            media: media_id,
            source: tr(0, 48),
            start: rt(0),
        }))
        .expect("add")
    {
        cutlass_engine::ApplyOutcome::Edited(cutlass_commands::EditOutcome::Created(id)) => id,
        other => panic!("unexpected {other:?}"),
    };

    engine
        .apply(Command::Edit(EditCommand::SplitClip {
            clip: clip_id,
            at: rt(24),
        }))
        .expect("split");

    let frame = engine.get_frame(rt(0)).expect("frame at head");
    assert!(frame.bytes.iter().any(|&b| b != 0));
}

#[test]
fn get_frame_errors_when_timeline_empty() {
    let (_dir, mut engine) = temp_engine();
    let err = engine.get_frame(rt(0)).unwrap_err();
    assert!(format!("{err}").contains("no video"));
}
