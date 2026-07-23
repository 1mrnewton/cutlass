//! Preview: import → add clip → get_frame, plus the live-preview overrides
//! (gesture transform, inspector generator) that render without touching
//! project state.

mod common;

use common::{import_asset, rt, small_video_asset, temp_engine, tr};
use cutlass_commands::{Command, EditCommand};
use cutlass_engine::{Engine, EngineConfig};
use cutlass_models::{ClipSource, ClipTransform, Generator, TrackKind};
use cutlass_render::RgbaImage;

/// H.264-style even rounding used by the auto canvas.
fn even(v: u32) -> u32 {
    (v & !1).max(2)
}

fn pixel(frame: &RgbaImage, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * frame.width + x) * 4) as usize;
    [
        frame.pixels[i],
        frame.pixels[i + 1],
        frame.pixels[i + 2],
        frame.pixels[i + 3],
    ]
}

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

    // The auto canvas follows the largest video media (even-rounded).
    let (width, height) = {
        let media = engine.project().media(media_id).expect("media");
        (even(media.width), even(media.height))
    };
    let frame = engine.get_frame(rt(0)).expect("get_frame");

    assert_eq!(frame.width, width);
    assert_eq!(frame.height, height);
    assert_eq!(
        frame.pixels.len(),
        usize::try_from(width * height * 4).unwrap()
    );
    assert!(
        frame.pixels.iter().any(|&b| b != 0),
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

    let clip_id = common::add_media_clip(&mut engine, track, media_id, tr(0, 48), rt(0));
    engine
        .apply(Command::Edit(EditCommand::SplitClip {
            clip: clip_id,
            at: rt(24),
        }))
        .expect("split");

    let frame = engine.get_frame(rt(0)).expect("frame at head");
    assert!(frame.pixels.iter().any(|&b| b != 0));
}

#[test]
fn get_frame_returns_black_when_timeline_empty() {
    let (_dir, mut engine) = temp_engine();
    let frame = engine.get_frame(rt(0)).expect("gap frame");
    // No media anywhere: the auto canvas falls back to 1920×1080.
    assert_eq!(frame.width, 1920);
    assert_eq!(frame.height, 1080);
    assert!(frame.pixels.chunks_exact(4).all(|p| p == [0, 0, 0, 255]));
}

#[test]
fn get_frame_renders_solid_generated_clip() {
    let (_dir, mut engine) = temp_engine();
    let track = common::add_track(&mut engine, TrackKind::Sticker, "T1");

    common::add_generated(
        &mut engine,
        track,
        Generator::SolidColor {
            rgba: [10, 20, 30, 255],
        },
        tr(0, 48),
    );

    let frame = engine.get_frame(rt(0)).expect("solid frame");
    assert_eq!(frame.width, 1920);
    assert_eq!(frame.height, 1080);
    assert!(frame.pixels.chunks_exact(4).all(|p| p == [10, 20, 30, 255]));
}

#[test]
fn get_frame_places_transformed_solid() {
    let (_dir, mut engine) = temp_engine();
    let track = common::add_track(&mut engine, TrackKind::Sticker, "T1");

    let clip_id = common::add_generated(
        &mut engine,
        track,
        Generator::SolidColor {
            rgba: [200, 40, 10, 255],
        },
        tr(0, 48),
    );

    // Half size, content center moved to the canvas's top-left quadrant
    // center: the solid covers exactly [0, w/2) × [0, h/2).
    engine
        .apply(Command::Edit(EditCommand::SetClipTransform {
            clip: clip_id,
            transform: ClipTransform {
                position: [-0.25, -0.25],
                scale: 0.5.into(),
                ..ClipTransform::IDENTITY
            },
            at: None,
        }))
        .expect("set transform");

    let frame = engine.get_frame(rt(0)).expect("transformed frame");
    let (w, h) = (frame.width, frame.height);

    assert_eq!(
        pixel(&frame, w / 4, h / 4),
        [200, 40, 10, 255],
        "inside placed quad"
    );
    assert_eq!(
        pixel(&frame, 10, 10),
        [200, 40, 10, 255],
        "top-left corner covered"
    );
    assert_eq!(
        pixel(&frame, 3 * w / 4, 3 * h / 4),
        [0, 0, 0, 255],
        "rest of canvas stays black"
    );
    assert_eq!(
        pixel(&frame, 3 * w / 4, h / 4),
        [0, 0, 0, 255],
        "right of the quad is black"
    );
}

#[test]
fn transform_override_previews_without_touching_state() {
    let (_dir, mut engine) = temp_engine();
    let track = common::add_track(&mut engine, TrackKind::Sticker, "T1");

    let clip_id = common::add_generated(
        &mut engine,
        track,
        Generator::SolidColor {
            rgba: [200, 40, 10, 255],
        },
        tr(0, 48),
    );
    let could_undo_before = engine.can_undo();

    // Live-drag override: half size in the top-left quadrant. Rendering
    // honors it...
    engine.set_transform_override(Some((
        clip_id,
        ClipTransform {
            position: [-0.25, -0.25],
            scale: 0.5.into(),
            ..ClipTransform::IDENTITY
        },
    )));
    let frame = engine.get_frame(rt(0)).expect("override frame");
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
    assert_eq!(engine.can_undo(), could_undo_before, "no history entry");

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
fn generator_override_previews_without_touching_state() {
    let (_dir, mut engine) = temp_engine();
    let track = common::add_track(&mut engine, TrackKind::Sticker, "T1");

    let clip_id = common::add_generated(
        &mut engine,
        track,
        Generator::SolidColor {
            rgba: [200, 40, 10, 255],
        },
        tr(0, 48),
    );
    let could_undo_before = engine.can_undo();

    // Live inspector edit (e.g. dragging a color slider): render the
    // substituted generator...
    engine.set_generator_override(Some((
        clip_id,
        Generator::SolidColor {
            rgba: [10, 200, 40, 255],
        },
    )));
    let frame = engine.get_frame(rt(0)).expect("override frame");
    assert!(
        frame
            .pixels
            .chunks_exact(4)
            .all(|p| p == [10, 200, 40, 255])
    );

    // ...while the committed generator and history stay untouched.
    let clip = engine.project().clip(clip_id).expect("clip");
    match &clip.content {
        ClipSource::Generated(generator) => assert_eq!(
            *generator,
            Generator::SolidColor {
                rgba: [200, 40, 10, 255],
            },
            "project generator untouched"
        ),
        other => panic!("unexpected content {other:?}"),
    }
    assert_eq!(engine.can_undo(), could_undo_before, "no history entry");

    // Clearing restores the committed render.
    engine.set_generator_override(None);
    let frame = engine.get_frame(rt(0)).expect("committed frame");
    assert!(
        frame
            .pixels
            .chunks_exact(4)
            .all(|p| p == [200, 40, 10, 255])
    );
}

#[test]
fn param_override_previews_without_touching_state() {
    use cutlass_commands::{Command, EditCommand};
    use cutlass_models::{
        ChromaKey, ClipParam, LookParam, MediaSource, ParamValue, Project, Rational, TrackKind,
    };
    use cutlass_render::{ResolveOverrides, resolve, resolve_with};

    let r = Rational::FPS_24;
    let mut project = Project::new("param-override", r);
    let media = project.add_media(MediaSource::new(
        "/tmp/param-override.mp4",
        1920,
        1080,
        r,
        1000,
        true,
    ));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip_id = project
        .add_clip(track, media, tr(0, 48), rt(0))
        .expect("clip");
    project
        .set_clip_chroma_key(
            clip_id,
            Some(ChromaKey {
                rgb: [0, 255, 0],
                strength: 0.25.into(),
                shadow: 0.0.into(),
            }),
        )
        .expect("chroma");
    let mut engine =
        Engine::with_project(EngineConfig { undo_limit: 32 }, project).expect("engine");
    let could_undo_before = engine.can_undo();
    let revision_before = engine.revision();

    let param = ClipParam::Look {
        param: LookParam::ChromaStrength,
    };
    engine.set_param_override(clip_id, param, ParamValue::Scalar(0.8));
    assert!(engine.has_live_overrides());
    assert_eq!(
        engine.param_overrides().get(clip_id, param),
        Some(ParamValue::Scalar(0.8))
    );

    // Resolve path honors the override; project + history stay untouched.
    let scene = resolve_with(
        engine.project(),
        rt(0),
        ResolveOverrides {
            params: Some(engine.param_overrides()),
            ..ResolveOverrides::default()
        },
    )
    .expect("resolve");
    assert!((scene.layers[0].chroma_key.unwrap().strength - 0.8).abs() < 1e-6);
    let committed = engine
        .project()
        .clip(clip_id)
        .expect("clip")
        .chroma_key
        .as_ref()
        .expect("chroma")
        .strength
        .sample(0);
    assert!((committed - 0.25).abs() < 1e-6);
    assert_eq!(engine.can_undo(), could_undo_before);
    assert_eq!(engine.revision(), revision_before);

    // Latest-wins for the same (clip, param).
    engine.set_param_override(clip_id, param, ParamValue::Scalar(0.55));
    assert_eq!(
        engine.param_overrides().get(clip_id, param),
        Some(ParamValue::Scalar(0.55))
    );

    // Multi-param on one clip.
    engine.set_param_override(
        clip_id,
        ClipParam::Crop,
        ParamValue::Rect([0.2, 0.2, 0.6, 0.6]),
    );
    assert!(engine.param_overrides().get(clip_id, param).is_some());
    assert!(
        engine
            .param_overrides()
            .get(clip_id, ClipParam::Crop)
            .is_some()
    );

    // Commit path clears just that param; crop override remains until
    // clear_param_overrides (mirrors release-then-clear).
    engine.clear_param_override(clip_id, param);
    engine
        .apply(Command::Edit(EditCommand::SetParamConstant {
            clip: clip_id,
            param,
            value: ParamValue::Scalar(0.55),
        }))
        .expect("commit");
    assert!(
        engine
            .param_overrides()
            .get(clip_id, ClipParam::Crop)
            .is_some()
    );

    engine.clear_param_overrides(clip_id);
    assert!(!engine.has_live_overrides());
    let plain = resolve(engine.project(), rt(0)).expect("plain");
    assert!((plain.layers[0].chroma_key.unwrap().strength - 0.55).abs() < 1e-6);

    // Fresh session drops leftover overrides (ids can collide).
    engine.set_param_override(clip_id, param, ParamValue::Scalar(0.1));
    engine.new_session();
    assert!(engine.param_overrides().is_empty());
}

#[test]
fn new_session_clears_overrides() {
    let (_dir, mut engine) = temp_engine();
    let track = common::add_track(&mut engine, TrackKind::Sticker, "T1");
    let clip_id = common::add_generated(
        &mut engine,
        track,
        Generator::SolidColor {
            rgba: [200, 40, 10, 255],
        },
        tr(0, 48),
    );
    engine.set_transform_override(Some((
        clip_id,
        ClipTransform {
            scale: 0.5.into(),
            ..ClipTransform::IDENTITY
        },
    )));
    engine.set_generator_override(Some((
        clip_id,
        Generator::SolidColor {
            rgba: [10, 200, 40, 255],
        },
    )));
    engine.set_param_override(
        clip_id,
        cutlass_models::ClipParam::Opacity,
        cutlass_models::ParamValue::Scalar(0.5),
    );

    // A fresh session must not leak the old session's live overrides onto
    // whatever clip is created next (ids restart, so they could collide).
    engine.new_session();
    let track = common::add_track(&mut engine, TrackKind::Sticker, "T1");
    common::add_generated(
        &mut engine,
        track,
        Generator::SolidColor {
            rgba: [10, 20, 30, 255],
        },
        tr(0, 48),
    );

    let frame = engine.get_frame(rt(0)).expect("fresh session frame");
    assert!(
        frame.pixels.chunks_exact(4).all(|p| p == [10, 20, 30, 255]),
        "committed solid covers the whole canvas — no stale override applied"
    );
}

#[test]
fn media_proxy_registry_sets_clears_and_dies_with_the_session() {
    use cutlass_models::MediaId;
    use std::path::{Path, PathBuf};

    let (_dir, mut engine) = temp_engine();
    let media = MediaId::from_raw(7);

    engine.set_media_proxy(media, PathBuf::from("proxy.mp4"));
    assert_eq!(engine.media_proxy(media), Some(Path::new("proxy.mp4")));

    engine.clear_media_proxy(media);
    assert_eq!(engine.media_proxy(media), None);

    // Media ids persist in project files: a fresh session must not inherit
    // the old session's substitutions (id 7 could name a different file).
    engine.set_media_proxy(media, PathBuf::from("proxy.mp4"));
    engine.new_session();
    assert_eq!(engine.media_proxy(media), None);
}
