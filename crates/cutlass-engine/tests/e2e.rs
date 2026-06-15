//! End-to-end session flows: new project → import → edit → save/load/open → export.

mod common;

use std::path::Path;
use std::time::Duration;

use common::{
    add_generated, add_media_clip, add_track, export_to, image_asset, import_asset, load_project,
    open_project, rt, save_project, small_video_asset, temp_engine, tr, video_assets,
};
use cutlass_commands::{Command, EditCommand, ProjectCommand};
use cutlass_decoder::{DecodeOptions, Decoder, HwAccel};
use cutlass_models::{Generator, TimeRange, TrackKind};

fn assert_mp4_playable(path: &Path, min_frames: u64) {
    assert!(path.exists(), "export missing: {}", path.display());
    assert!(
        std::fs::metadata(path).unwrap().len() > 0,
        "export empty: {}",
        path.display()
    );
    let mut dec = Decoder::open_with(path, DecodeOptions::default().hw_accel(HwAccel::None))
        .unwrap_or_else(|e| panic!("open export {}: {e}", path.display()));
    let dur = dec.duration().expect("duration");
    assert!(dur > Duration::ZERO);
    let frame = dec
        .seek_to_frame(Duration::ZERO)
        .expect("seek")
        .expect("first frame");
    assert!(frame.width > 0);
    assert!(frame.height > 0);
    let _ = min_frames;
}

#[test]
fn e2e_solid_project_save_load_export() {
    let (dir, mut engine) = temp_engine();

    assert_eq!(engine.project().name, "untitled");
    assert_eq!(engine.project().timeline().clip_count(), 0);
    assert_eq!(engine.project().media_count(), 0);
    assert!(engine.project_path().is_none());

    let track = add_track(&mut engine, TrackKind::Sticker, "T1");
    add_generated(
        &mut engine,
        track,
        Generator::SolidColor {
            rgba: [20, 40, 60, 255],
        },
        tr(0, 24),
    );

    let frame = engine.get_frame(rt(0)).expect("preview");
    let (pw, ph) = cutlass_engine::preview_scaled_dims(1920, 1080);
    assert_eq!(frame.width, pw);
    assert_eq!(frame.height, ph);

    let project_file = dir.path().join("solid.cutlass");
    save_project(&mut engine, &project_file);
    assert_eq!(engine.project_path(), Some(&project_file));
    assert!(engine.can_undo());

    let (_dir2, mut fresh) = temp_engine();
    load_project(&mut fresh, &project_file);
    assert_eq!(fresh.project().timeline().clip_count(), 1);
    assert_eq!(fresh.project_path(), Some(&project_file));
    assert!(!fresh.can_undo(), "load clears undo");

    let out = dir.path().join("solid_export.mp4");
    let stats = export_to(&mut fresh, &out);
    assert_eq!(stats.frames, 24);
    assert_eq!(stats.width, 1920);
    assert_eq!(stats.height, 1080);
    assert_mp4_playable(&out, 24);
}

#[test]
fn e2e_media_import_edit_save_open_export() {
    let Some(asset) = small_video_asset() else {
        return;
    };
    let (dir, mut engine) = temp_engine();

    let media_id = import_asset(&mut engine, &asset);
    assert_eq!(engine.project().media_count(), 1);
    let track = add_track(&mut engine, TrackKind::Video, "V1");
    let clip = add_media_clip(&mut engine, track, media_id, tr(0, 24), rt(0));

    let (media_w, media_h) = {
        let media = engine.project().media(media_id).expect("media");
        (media.width, media.height)
    };
    let preview = engine.get_frame(rt(0)).expect("preview");
    let (pw, ph) = cutlass_engine::preview_scaled_dims(media_w, media_h);
    assert_eq!(preview.width, pw);
    assert_eq!(preview.height, ph);

    let project_file = dir.path().join("media_session.cutlass");
    save_project(&mut engine, &project_file);

    engine
        .apply(Command::Edit(EditCommand::RippleDelete { clip }))
        .expect("clear timeline");
    assert_eq!(engine.project().timeline().clip_count(), 0);

    open_project(&mut engine, &project_file);
    assert_eq!(engine.project().timeline().clip_count(), 1);
    assert_eq!(engine.project().media_count(), 1);
    // Import canonicalizes the source path (dedupe across spellings).
    assert_eq!(
        engine.project().media_iter().next().expect("media").path(),
        asset.canonicalize().expect("asset exists")
    );
    assert!(!engine.can_undo());

    engine.get_frame(rt(0)).expect("preview after open");

    let out = dir.path().join("media_export.mp4");
    let (w, h) = {
        let m = engine.project().media_iter().next().expect("media");
        (m.width, m.height)
    };
    let stats = export_to(&mut engine, &out);
    assert_eq!(stats.frames, 24);
    assert_eq!(stats.width, w);
    assert_eq!(stats.height, h);
    assert_mp4_playable(&out, 24);
}

#[test]
fn e2e_image_still_import_preview_save_reopen_export() {
    let Some(png) = image_asset() else {
        return;
    };
    let (dir, mut engine) = temp_engine();

    let media_id = import_asset(&mut engine, &png);
    let (source, is_image, duration_ticks) = {
        let media = engine.project().media(media_id).expect("media");
        (media.full_range(), media.is_image, media.duration.value)
    };
    assert!(is_image, "png imports as a still");
    assert_eq!(
        duration_ticks, 5_000,
        "default 5s placement at 1000/1 ticks"
    );

    let track = add_track(&mut engine, TrackKind::Video, "V1");
    add_media_clip(&mut engine, track, media_id, source, rt(0));
    // 5s resampled to the 24fps timeline.
    assert_eq!(engine.project().timeline().duration().value, 120);

    // Stills aspect-fit the canvas; they don't drive its size.
    let frame = engine.get_frame(rt(60)).expect("mid-still preview");
    assert_eq!(
        (frame.width, frame.height),
        cutlass_engine::preview_scaled_dims(1920, 1080)
    );
    assert!(
        frame
            .bytes
            .chunks_exact(4)
            .any(|p| p[0] != 0 || p[1] != 0 || p[2] != 0),
        "still renders visible pixels"
    );

    // The image flag survives a save/reopen roundtrip.
    let project_file = dir.path().join("still.cutlass");
    save_project(&mut engine, &project_file);
    let (_dir2, mut reopened) = temp_engine();
    open_project(&mut reopened, &project_file);
    let media = reopened.project().media_iter().next().expect("media");
    assert!(media.is_image);
    reopened.get_frame(rt(0)).expect("preview after reopen");

    // Export composites the still for every frame of the clip's extent.
    let out = dir.path().join("still_export.mp4");
    let stats = export_to(&mut engine, &out);
    assert_eq!(stats.frames, 120);
    assert_eq!((stats.width, stats.height), (1920, 1080));
    assert_mp4_playable(&out, 120);
}

#[test]
fn e2e_multi_import_save_reopen_export() {
    let assets = video_assets();
    if assets.is_empty() {
        return;
    }

    let (dir, mut engine) = temp_engine();
    let v1 = add_track(&mut engine, TrackKind::Video, "V1");
    let v2 = add_track(&mut engine, TrackKind::Video, "V2");

    let first = import_asset(&mut engine, &assets[0]);
    let first_media = engine.project().media(first).expect("media");
    let first_rate = first_media.frame_rate;
    let first_len = first_media.duration.value.clamp(1, 24);
    add_media_clip(
        &mut engine,
        v1,
        first,
        TimeRange::at_rate(0, first_len, first_rate),
        rt(0),
    );

    if assets.len() >= 2 {
        let second = import_asset(&mut engine, &assets[1]);
        let (second_len, second_rate) = {
            let m = engine.project().media(second).expect("media");
            (m.duration.value.clamp(1, 24), m.frame_rate)
        };
        add_media_clip(
            &mut engine,
            v2,
            second,
            TimeRange::at_rate(0, second_len, second_rate),
            rt(0),
        );
        assert_eq!(engine.project().media_count(), 2);
    } else {
        let second = import_asset(&mut engine, &assets[0]);
        assert_eq!(second, first, "re-import must reuse existing pool entry");
        let second_len = first_len.min(12);
        add_media_clip(
            &mut engine,
            v2,
            second,
            TimeRange::at_rate(0, second_len, first_rate),
            rt(first_len),
        );
        assert_eq!(engine.project().media_count(), 1);
    }

    let expected_frames = engine.project().timeline().duration().value;
    assert!(expected_frames > 0, "timeline empty after edits");

    let project_file = dir.path().join("multi.cutlass");
    save_project(&mut engine, &project_file);

    let (_dir2, mut reopened) = temp_engine();
    open_project(&mut reopened, &project_file);
    let expected_media = if assets.len() >= 2 { 2 } else { 1 };
    assert_eq!(reopened.project().media_count(), expected_media);
    assert_eq!(reopened.project().timeline().clip_count(), 2);
    assert_eq!(
        reopened.project().timeline().duration().value,
        expected_frames
    );

    let out = dir.path().join("multi_export.mp4");
    let stats = export_to(&mut reopened, &out);
    assert_eq!(stats.frames, expected_frames as u64);
    assert_mp4_playable(&out, expected_frames as u64);
}

#[test]
fn e2e_two_layer_stack_save_export() {
    let Some(asset) = small_video_asset() else {
        return;
    };
    let (dir, mut engine) = temp_engine();
    let media_id = import_asset(&mut engine, &asset);
    let v1 = add_track(&mut engine, TrackKind::Video, "V1");
    let overlay = add_track(&mut engine, TrackKind::Sticker, "T1");

    add_media_clip(&mut engine, v1, media_id, tr(0, 24), rt(0));
    add_generated(
        &mut engine,
        overlay,
        Generator::SolidColor {
            rgba: [0, 0, 0, 128],
        },
        tr(0, 24),
    );

    let project_file = dir.path().join("stack.cutlass");
    save_project(&mut engine, &project_file);

    let (_dir2, mut loaded) = temp_engine();
    load_project(&mut loaded, &project_file);
    loaded.get_frame(rt(0)).expect("composite preview");

    let out = dir.path().join("stack_export.mp4");
    export_to(&mut loaded, &out);
    assert_mp4_playable(&out, 24);
}

#[test]
fn e2e_export_errors_on_empty_saved_project() {
    let (dir, mut engine) = temp_engine();
    let project_file = dir.path().join("empty.cutlass");
    save_project(&mut engine, &project_file);

    let (_dir2, mut loaded) = temp_engine();
    load_project(&mut loaded, &project_file);

    let err = loaded
        .apply(Command::Project(ProjectCommand::Export {
            path: dir.path().join("empty.mp4"),
        }))
        .unwrap_err();
    assert!(format!("{err}").contains("no content"));
}

#[test]
fn e2e_undo_survives_save_but_not_open() {
    let (dir, mut engine) = temp_engine();
    let track = add_track(&mut engine, TrackKind::Sticker, "T1");
    add_generated(
        &mut engine,
        track,
        Generator::SolidColor {
            rgba: [1, 2, 3, 255],
        },
        tr(0, 12),
    );
    assert!(engine.can_undo());

    let project_file = dir.path().join("undo.cutlass");
    save_project(&mut engine, &project_file);
    assert!(engine.can_undo(), "save retains undo stack");

    open_project(&mut engine, &project_file);
    assert!(!engine.can_undo(), "open clears undo");
    assert_eq!(engine.project().timeline().clip_count(), 1);
}
