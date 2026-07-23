//! Engine-level template workflow: authoring markers and music swap through
//! undoable commands, template save/apply as session commands.
//!
//! The fill test synthesizes a real mp4 with the native encoder (Apple-only,
//! like `import_probe`) because `ApplyTemplate` probes each pick from disk.

mod common;

use common::{rt, tr};
use cutlass_commands::{Command, EditCommand, EditOutcome, ProjectCommand, TemplatePick};
use cutlass_engine::{ApplyOutcome, Engine, EngineConfig, EngineError};
use cutlass_models::{
    ClipId, ClipSource, Generator, MediaId, MediaSource, Project, Rational, Replaceable, SlotMedia,
    Template, TemplateMeta, TrackKind,
};

/// A finished-looking session on fake-path media (no test below this line
/// touches the disk for decoding): two 1s clips on video, a title, a music
/// bed, and a spare audio source to swap in.
struct Authoring {
    engine: Engine,
    slot_a: ClipId,
    slot_b: ClipId,
    title: ClipId,
    music: ClipId,
    swap_media: MediaId,
}

fn authoring_engine() -> Authoring {
    let mut project = Project::new("Trendy Intro", Rational::FPS_24);
    let sample = project.add_media(MediaSource::new(
        "sample.mp4",
        1920,
        1080,
        Rational::FPS_24,
        240,
        true,
    ));
    let beat = project.add_media(MediaSource::new(
        "beat.m4a",
        0,
        0,
        Rational::FPS_24,
        480,
        true,
    ));
    let swap_media = project.add_media(MediaSource::new(
        "swap.m4a",
        0,
        0,
        Rational::FPS_24,
        480,
        true,
    ));

    let video = project.add_track(TrackKind::Video, "V1");
    let text = project.add_track(TrackKind::Text, "T1");
    let audio = project.add_track(TrackKind::Audio, "A1");

    let slot_a = project.add_clip(video, sample, tr(0, 24), rt(0)).unwrap();
    let slot_b = project.add_clip(video, sample, tr(0, 24), rt(24)).unwrap();
    let title = project
        .add_generated(text, Generator::text("Your Name"), tr(0, 48))
        .unwrap();
    let music = project.add_clip(audio, beat, tr(0, 48), rt(0)).unwrap();

    Authoring {
        engine: Engine::with_project(EngineConfig { undo_limit: 32 }, project).expect("engine"),
        slot_a,
        slot_b,
        title,
        music,
        swap_media,
    }
}

fn edit(engine: &mut Engine, command: EditCommand) -> ApplyOutcome {
    engine.apply(Command::Edit(command)).expect("edit applies")
}

#[test]
fn mark_replaceable_via_commands_is_undoable() {
    let mut a = authoring_engine();
    let out = edit(
        &mut a.engine,
        EditCommand::SetReplaceable {
            clip: a.slot_a,
            replaceable: Some(Replaceable::new(0)),
        },
    );
    assert!(matches!(out, ApplyOutcome::Edited(EditOutcome::Updated(id)) if id == a.slot_a));
    assert!(
        a.engine
            .project()
            .clip(a.slot_a)
            .unwrap()
            .replaceable
            .is_some()
    );
    assert!(a.engine.is_dirty(), "marking is a project mutation");

    assert!(a.engine.undo());
    assert!(
        a.engine
            .project()
            .clip(a.slot_a)
            .unwrap()
            .replaceable
            .is_none()
    );

    assert!(a.engine.redo());
    assert!(
        a.engine
            .project()
            .clip(a.slot_a)
            .unwrap()
            .replaceable
            .is_some()
    );
}

#[test]
fn mark_replaceable_rejects_generated_clips_and_records_no_history() {
    let mut a = authoring_engine();
    let err = a
        .engine
        .apply(Command::Edit(EditCommand::SetReplaceable {
            clip: a.title,
            replaceable: Some(Replaceable::new(0)),
        }))
        .expect_err("text clips cannot be slots");
    assert!(matches!(err, EngineError::Model(_)), "unexpected: {err}");
    assert!(!a.engine.can_undo(), "rejected command records no history");
    assert!(
        a.engine
            .project()
            .clip(a.title)
            .unwrap()
            .replaceable
            .is_none()
    );
}

#[test]
fn set_text_editable_via_commands_is_undoable() {
    let mut a = authoring_engine();
    edit(
        &mut a.engine,
        EditCommand::SetTextEditable {
            clip: a.title,
            editable: true,
        },
    );
    assert!(a.engine.project().clip(a.title).unwrap().text_editable);

    assert!(a.engine.undo());
    assert!(!a.engine.project().clip(a.title).unwrap().text_editable);

    assert!(a.engine.redo());
    assert!(a.engine.project().clip(a.title).unwrap().text_editable);
}

#[test]
fn music_swap_via_set_clip_media_is_undoable() {
    let mut a = authoring_engine();
    let before = a.engine.project().clip(a.music).unwrap().content.clone();

    edit(
        &mut a.engine,
        EditCommand::SetClipMedia {
            clip: a.music,
            media: a.swap_media,
            source: tr(0, 48),
        },
    );
    match &a.engine.project().clip(a.music).unwrap().content {
        ClipSource::Media { media, .. } => assert_eq!(*media, a.swap_media),
        other => panic!("music not swapped: {other:?}"),
    }

    assert!(a.engine.undo());
    assert_eq!(a.engine.project().clip(a.music).unwrap().content, before);

    assert!(a.engine.redo());
    match &a.engine.project().clip(a.music).unwrap().content {
        ClipSource::Media { media, .. } => assert_eq!(*media, a.swap_media),
        other => panic!("redo lost the swap: {other:?}"),
    }
}

#[test]
fn set_clip_media_rejects_out_of_bounds_window() {
    let mut a = authoring_engine();
    let before = a.engine.project().clip(a.music).unwrap().content.clone();
    let err = a
        .engine
        .apply(Command::Edit(EditCommand::SetClipMedia {
            clip: a.music,
            media: a.swap_media,
            // swap.m4a is 480 ticks long; this window reads past its end.
            source: tr(400, 200),
        }))
        .expect_err("window exceeds the source");
    assert!(matches!(err, EngineError::Model(_)), "unexpected: {err}");
    assert_eq!(a.engine.project().clip(a.music).unwrap().content, before);
}

/// Author the full customization surface through commands: two visual slots
/// (marked out of order on purpose), an editable title, a swappable music bed.
fn author_template_surface(a: &mut Authoring) {
    edit(
        &mut a.engine,
        EditCommand::SetReplaceable {
            clip: a.slot_b,
            replaceable: Some(Replaceable::new(1)),
        },
    );
    edit(
        &mut a.engine,
        EditCommand::SetReplaceable {
            clip: a.slot_a,
            replaceable: Some(Replaceable::new(0)),
        },
    );
    edit(
        &mut a.engine,
        EditCommand::SetTextEditable {
            clip: a.title,
            editable: true,
        },
    );
    edit(
        &mut a.engine,
        EditCommand::SetReplaceable {
            clip: a.music,
            replaceable: Some(Replaceable::new(0).with_accepts(SlotMedia::AudioOnly)),
        },
    );
}

#[test]
fn save_template_writes_a_loadable_cutlasst_without_touching_the_session() {
    let mut a = authoring_engine();
    author_template_surface(&mut a);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("intro.cutlasst");
    let out = a
        .engine
        .apply(Command::Project(ProjectCommand::SaveTemplate {
            path: path.clone(),
            meta: TemplateMeta::new("Trendy Intro").with_author("tester"),
        }))
        .expect("save template");
    assert!(matches!(out, ApplyOutcome::SavedTemplate));
    // A template export is not a project save: the session still has no
    // `.cutlass` path and still reads dirty.
    assert!(a.engine.project_path().is_none());
    assert!(a.engine.is_dirty());

    let template = Template::load_from_file(&path).expect("loadable template");
    assert_eq!(template.meta().name, "Trendy Intro");
    assert_eq!(template.slot_count(), 2);
    assert_eq!(
        template.slots().iter().map(|c| c.id).collect::<Vec<_>>(),
        vec![a.slot_a, a.slot_b],
        "fill order follows `order`, not authoring order"
    );
    assert_eq!(template.editable_texts().len(), 1);
    assert_eq!(template.music().map(|c| c.id), Some(a.music));
}

#[test]
fn apply_template_with_missing_template_file_errors() {
    let (_dir, mut engine) = common::temp_engine();
    let err = engine
        .apply(Command::Project(ProjectCommand::ApplyTemplate {
            path: "/nonexistent/intro.cutlasst".into(),
            picks: Vec::new(),
        }))
        .expect_err("missing template file");
    assert!(matches!(err, EngineError::Model(_)), "unexpected: {err}");
}

#[test]
fn apply_template_with_missing_pick_leaves_session_untouched() {
    // A real template on disk, but a pick path that doesn't exist.
    let mut a = authoring_engine();
    author_template_surface(&mut a);
    let dir = tempfile::tempdir().unwrap();
    let template_path = dir.path().join("intro.cutlasst");
    a.engine
        .apply(Command::Project(ProjectCommand::SaveTemplate {
            path: template_path.clone(),
            meta: TemplateMeta::new("Trendy Intro"),
        }))
        .expect("save template");

    let (_dir2, mut engine) = common::temp_engine();
    common::add_track(&mut engine, TrackKind::Video, "V1");
    assert!(engine.can_undo());

    let err = engine
        .apply(Command::Project(ProjectCommand::ApplyTemplate {
            path: template_path,
            picks: vec![TemplatePick {
                path: "/nonexistent/pick.mp4".into(),
                source_in: None,
            }],
        }))
        .expect_err("missing pick file");
    assert!(matches!(err, EngineError::Io(_)), "unexpected: {err}");

    // The failed apply replaced nothing: same project, history intact.
    assert_eq!(engine.project().name, "untitled");
    assert_eq!(engine.project().timeline().track_count(), 1);
    assert!(engine.can_undo(), "failed apply must not clear history");
}

/// Filling a template probes real files, so this half synthesizes an mp4 with
/// the native encoder first (Apple-only, matching `import_probe`).
#[cfg(target_vendor = "apple")]
mod fill {
    use super::*;
    use cutlass_core::{
        ColorSpace, CpuImage, EncoderConfig, FrameData, PixelFormat, Plane, Rect, Rotation,
        VideoEncoder, VideoFrame,
    };
    use cutlass_encoder::AvfEncoder;
    use cutlass_models::RationalTime;

    fn solid_rgba_frame(width: u32, height: u32, index: i64, rate: Rational) -> VideoFrame {
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for _ in 0..(width * height) {
            pixels.extend_from_slice(&[200, 80, 40, 255]); // opaque rust-orange, RGBA
        }
        let plane = Plane::new(pixels, width as usize * 4, height as usize);
        VideoFrame::new(
            RationalTime::new(index, rate),
            PixelFormat::Rgba8,
            ColorSpace::BT709,
            (width, height),
            Rect::from_size(width, height),
            Rotation::None,
            FrameData::Cpu(CpuImage::new(vec![plane])),
        )
    }

    /// Write a solid-color `frames`-long mp4 with the native encoder.
    fn write_test_mp4(path: &std::path::Path, rate: Rational, frames: i64) {
        let config = EncoderConfig::new((320, 240), rate, ColorSpace::BT709);
        let mut encoder = AvfEncoder::open(path, config).expect("open encoder");
        for i in 0..frames {
            encoder
                .push(&solid_rgba_frame(320, 240, i, rate))
                .expect("push frame");
        }
        encoder.finish().expect("finish");
    }

    #[test]
    fn apply_template_fills_slots_and_replaces_the_session() {
        // Author and save the template.
        let mut a = authoring_engine();
        author_template_surface(&mut a);
        let dir = tempfile::tempdir().unwrap();
        let template_path = dir.path().join("intro.cutlasst");
        a.engine
            .apply(Command::Project(ProjectCommand::SaveTemplate {
                path: template_path.clone(),
                meta: TemplateMeta::new("Trendy Intro"),
            }))
            .expect("save template");

        // A real, probeable pick: 48 frames at the slot rate.
        let pick_path = dir.path().join("pick.mp4");
        write_test_mp4(&pick_path, Rational::FPS_24, 48);

        // A separate session with history that the apply must clear.
        let mut engine = Engine::new(EngineConfig { undo_limit: 8 }).expect("engine");
        common::add_track(&mut engine, TrackKind::Video, "scratch");
        assert!(engine.can_undo());
        // Live overrides are project-local; template fill keeps clip ids, so
        // a leftover override would silently bind to a template slot.
        let sticker = common::add_track(&mut engine, TrackKind::Sticker, "S1");
        let scratch_clip = common::add_generated(
            &mut engine,
            sticker,
            Generator::SolidColor {
                rgba: [1, 2, 3, 255],
            },
            tr(0, 24),
        );
        engine.set_param_override(
            scratch_clip,
            cutlass_models::ClipParam::Opacity,
            cutlass_models::ParamValue::Scalar(0.1),
        );
        assert!(engine.has_live_overrides());

        let out = engine
            .apply(Command::Project(ProjectCommand::ApplyTemplate {
                path: template_path,
                picks: vec![
                    TemplatePick {
                        path: pick_path.clone(),
                        source_in: None,
                    },
                    TemplatePick {
                        path: pick_path.clone(),
                        source_in: Some(rt(12)),
                    },
                ],
            }))
            .expect("apply template");
        assert!(matches!(out, ApplyOutcome::AppliedTemplate));
        assert!(
            !engine.has_live_overrides(),
            "apply-template must drop every session-only override lane"
        );

        // The session is the filled template now: fresh history, no project
        // path, dirty until first saved.
        assert_eq!(engine.project().name, "Trendy Intro");
        assert!(!engine.can_undo(), "a fresh document has no undo past");
        assert!(engine.project_path().is_none());
        assert!(engine.is_dirty(), "a filled template is unsaved content");

        // Clip ids survive apply (only the project id is re-minted): both
        // slots point at the picked file, windowed to their locked 1s, the
        // second from its chosen in-point.
        let canonical = pick_path.canonicalize().expect("pick exists");
        for (slot, source_start) in [(a.slot_a, 0), (a.slot_b, 12)] {
            match &engine.project().clip(slot).unwrap().content {
                ClipSource::Media { media, source } => {
                    assert_eq!(source.duration.value, 24, "locked slot duration");
                    assert_eq!(source.start.value, source_start);
                    assert_eq!(
                        engine.project().media(*media).unwrap().path(),
                        canonical.as_path()
                    );
                }
                other => panic!("slot not filled: {other:?}"),
            }
        }

        // Markers are retained: a filled template stays re-editable.
        assert!(
            engine
                .project()
                .clip(a.slot_a)
                .unwrap()
                .replaceable
                .is_some()
        );
        assert!(engine.project().clip(a.title).unwrap().text_editable);
        assert_eq!(
            engine
                .project()
                .clip(a.music)
                .unwrap()
                .replaceable
                .as_ref()
                .map(|r| r.accepts),
            Some(SlotMedia::AudioOnly)
        );
    }
}
