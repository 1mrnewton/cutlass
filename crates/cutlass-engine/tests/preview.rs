//! Preview: import → add clip → get_frame.

mod common;

use common::{import_asset, rt, small_video_asset, temp_engine, tr};
use cutlass_commands::{Command, EditCommand, EditOutcome};
use cutlass_engine::ApplyOutcome;
use cutlass_models::{ClipTransform, Generator, TrackKind};

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

    // Preview composites at the reduced preview resolution, not full source.
    let (width, height) = {
        let media = engine.project().media(media_id).expect("media");
        cutlass_engine::preview_scaled_dims(media.width, media.height)
    };
    let frame = engine.get_frame(rt(0)).expect("get_frame");

    assert_eq!(frame.width, width);
    assert_eq!(frame.height, height);
    assert_eq!(
        frame.bytes.len(),
        usize::try_from(width * height * 4).unwrap()
    );
    assert!(
        frame.bytes.iter().any(|&b| b != 0),
        "frame should not be blank"
    );
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
fn get_frame_returns_black_when_timeline_empty() {
    let (_dir, mut engine) = temp_engine();
    let frame = engine.get_frame(rt(0)).expect("gap frame");
    let (w, h) = cutlass_engine::preview_scaled_dims(1920, 1080);
    assert_eq!(frame.width, w);
    assert_eq!(frame.height, h);
    assert!(frame.bytes.chunks_exact(4).all(|p| p == [0, 0, 0, 255]));
}

#[test]
fn get_frame_renders_solid_generated_clip() {
    let (_dir, mut engine) = temp_engine();
    let track = common::add_track(&mut engine, TrackKind::Sticker, "T1");

    engine
        .apply(Command::Edit(EditCommand::AddGenerated {
            track,
            generator: Generator::SolidColor {
                rgba: [10, 20, 30, 255],
            },
            timeline: tr(0, 48),
        }))
        .expect("add solid");

    let frame = engine.get_frame(rt(0)).expect("solid frame");
    let (w, h) = cutlass_engine::preview_scaled_dims(1920, 1080);
    assert_eq!(frame.width, w);
    assert_eq!(frame.height, h);
    assert!(frame.bytes.chunks_exact(4).all(|p| p == [10, 20, 30, 255]));
}

#[test]
fn get_frame_places_transformed_solid() {
    let (_dir, mut engine) = temp_engine();
    let track = common::add_track(&mut engine, TrackKind::Sticker, "T1");

    let clip_id = match engine
        .apply(Command::Edit(EditCommand::AddGenerated {
            track,
            generator: Generator::SolidColor {
                rgba: [200, 40, 10, 255],
            },
            timeline: tr(0, 48),
        }))
        .expect("add solid")
    {
        ApplyOutcome::Edited(EditOutcome::Created(id)) => id,
        other => panic!("unexpected {other:?}"),
    };

    // Half size, content center moved to the canvas's top-left quadrant
    // center: the solid covers exactly [0, 960) × [0, 540).
    engine
        .apply(Command::Edit(EditCommand::SetClipTransform {
            clip: clip_id,
            transform: ClipTransform {
                position: [-0.25, -0.25],
                scale: 0.5,
                rotation: 0.0,
                opacity: 1.0,
                ..ClipTransform::IDENTITY
            },
            at: None,
        }))
        .expect("set transform");

    // The placement is normalized, so sample points are fractions of the
    // (preview-scaled) frame, not absolute 1080p pixels. The solid covers the
    // top-left quadrant: [0, w/2) × [0, h/2).
    let frame = engine.get_frame(rt(0)).expect("transformed frame");
    let (w, h) = (frame.width, frame.height);

    let pixel = |x: u32, y: u32| {
        let i = ((y * frame.width + x) * 4) as usize;
        [
            frame.bytes[i],
            frame.bytes[i + 1],
            frame.bytes[i + 2],
            frame.bytes[i + 3],
        ]
    };
    assert_eq!(pixel(w / 4, h / 4), [200, 40, 10, 255], "inside placed quad");
    assert_eq!(pixel(10, 10), [200, 40, 10, 255], "top-left corner covered");
    assert_eq!(
        pixel(3 * w / 4, 3 * h / 4),
        [0, 0, 0, 255],
        "rest of canvas stays black"
    );
    assert_eq!(
        pixel(3 * w / 4, h / 4),
        [0, 0, 0, 255],
        "right of the quad is black"
    );
}

#[test]
fn transform_override_previews_without_touching_state() {
    let (_dir, mut engine) = temp_engine();
    let track = common::add_track(&mut engine, TrackKind::Sticker, "T1");

    let clip_id = match engine
        .apply(Command::Edit(EditCommand::AddGenerated {
            track,
            generator: Generator::SolidColor {
                rgba: [200, 40, 10, 255],
            },
            timeline: tr(0, 48),
        }))
        .expect("add solid")
    {
        ApplyOutcome::Edited(EditOutcome::Created(id)) => id,
        other => panic!("unexpected {other:?}"),
    };
    let history_depth_before = engine.can_undo();

    // Live-drag override: half size in the top-left quadrant. Rendering
    // honors it...
    engine.set_transform_override(Some((
        clip_id,
        ClipTransform {
            position: [-0.25, -0.25],
            scale: 0.5,
            rotation: 0.0,
            opacity: 1.0,
            ..ClipTransform::IDENTITY
        },
    )));
    let frame = engine.get_frame(rt(0)).expect("override frame");
    let pixel = |frame: &cutlass_engine::RgbaFrame, x: u32, y: u32| {
        let i = ((y * frame.width + x) * 4) as usize;
        [
            frame.bytes[i],
            frame.bytes[i + 1],
            frame.bytes[i + 2],
            frame.bytes[i + 3],
        ]
    };
    // Normalized sample points (fractions of the preview-scaled frame).
    let (w, h) = (frame.width, frame.height);
    assert_eq!(
        pixel(&frame, w / 4, h / 4),
        [200, 40, 10, 255],
        "override placed quad"
    );
    assert_eq!(
        pixel(&frame, 3 * w / 4, 3 * h / 4),
        [0, 0, 0, 255],
        "rest stays black"
    );

    // ...but the project and history never saw it: session state only.
    let committed = &engine.project().clip(clip_id).expect("clip").transform;
    assert!(committed.is_identity(), "project transform untouched");
    assert_eq!(engine.can_undo(), history_depth_before, "no history entry");

    // Clearing restores the committed (full-canvas) render.
    engine.set_transform_override(None);
    let frame = engine.get_frame(rt(0)).expect("committed frame");
    let (w, h) = (frame.width, frame.height);
    assert_eq!(
        pixel(&frame, 3 * w / 4, 3 * h / 4),
        [200, 40, 10, 255],
        "solid covers canvas again"
    );
}

#[test]
fn get_frame_composites_solid_over_media() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let (_dir, mut engine) = temp_engine();
    let media_id = import_asset(&mut engine, &path);
    let v1 = common::add_track(&mut engine, TrackKind::Video, "V1");
    let v2 = common::add_track(&mut engine, TrackKind::Sticker, "T1");

    engine
        .apply(Command::Edit(EditCommand::AddClip {
            track: v1,
            media: media_id,
            source: tr(0, 48),
            start: rt(0),
        }))
        .expect("add media");

    engine
        .apply(Command::Edit(EditCommand::AddGenerated {
            track: v2,
            generator: Generator::SolidColor {
                rgba: [0, 0, 0, 128],
            },
            timeline: tr(0, 48),
        }))
        .expect("add overlay");

    let frame = engine.get_frame(rt(0)).expect("composite frame");
    let media = engine.project().media(media_id).expect("media");
    let (w, h) = cutlass_engine::preview_scaled_dims(media.width, media.height);
    assert_eq!(frame.width, w);
    assert_eq!(frame.height, h);

    let mut non_zero = 0usize;
    let mut dark = 0usize;
    for px in frame.bytes.chunks_exact(4) {
        if px.iter().any(|&b| b != 0) {
            non_zero += 1;
        }
        if px[0] < 200 && px[1] < 200 && px[2] < 200 {
            dark += 1;
        }
    }
    assert!(non_zero > 0, "frame should have content");
    assert!(
        dark > 0,
        "semi-transparent black overlay should darken pixels"
    );
}
