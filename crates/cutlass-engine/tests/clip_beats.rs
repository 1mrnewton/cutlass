//! Public engine-command coverage for pure beat-grid clearing.

use cutlass_commands::{Command, EditCommand, EditOutcome};
use cutlass_engine::{ApplyOutcome, Engine, EngineConfig, EngineError};
use cutlass_models::{
    AnimationRef, ChromaKey, ClipId, CropRect, EffectInstance, Filter, Generator, Lut, Mask,
    MaskKind, MediaId, MediaSource, ModelError, Param, Project, Rational, RationalTime,
    Replaceable, StabilizeLevel, TimeRange, TrackKind,
};

fn rt(value: i64) -> RationalTime {
    RationalTime::new(value, Rational::FPS_24)
}

fn tr(start: i64, duration: i64) -> TimeRange {
    TimeRange::at_rate(start, duration, Rational::FPS_24)
}

fn media_project(beats: Vec<i64>) -> (Project, MediaId, ClipId) {
    let mut project = Project::new("beats", Rational::FPS_24);
    let media = project.add_media(MediaSource::new(
        "/tmp/beat-grid.mp4",
        1920,
        1080,
        Rational::FPS_24,
        500,
        true,
    ));
    let track = project.add_track(TrackKind::Video, "Main");
    let clip = project
        .add_clip(track, media, tr(10, 100), rt(40))
        .expect("media clip");
    project.set_clip_beats(clip, beats).expect("seed beat grid");
    (project, media, clip)
}

fn engine_with(project: Project) -> Engine {
    Engine::with_project(EngineConfig { undo_limit: 32 }, project).expect("engine")
}

#[test]
fn clear_beats_undo_redo_only_swaps_the_normalized_grid() {
    let (mut project, _media, clip) = media_project(vec![95, 20, 20, 70, 9, 110]);

    // Give unrelated fields non-default values. Whole-clip equality below
    // covers content (including media id + source range) and every property,
    // while the expected snapshots differ only in `beats`.
    {
        let clip = project
            .timeline_mut()
            .clip_mut(clip)
            .expect("clip to decorate");
        clip.transform.position = Param::Constant([0.25, -0.5]);
        clip.transform.rotation = Param::Constant(17.0);
        clip.volume = Param::Constant(0.65);
        clip.fade_in = 3;
        clip.fade_out = 4;
        clip.denoise = true;
        clip.crop = CropRect {
            x: 0.1,
            y: 0.2,
            w: 0.8,
            h: 0.7,
        }
        .into();
        clip.flip_h = true;
        clip.effects.push(EffectInstance::new("gaussian_blur"));
        clip.mask = Some(Mask {
            kind: MaskKind::Circle,
            feather: 0.2.into(),
            invert: true,
            ..Mask::new(MaskKind::Circle)
        });
        clip.chroma_key = Some(ChromaKey {
            rgb: [0, 255, 0],
            strength: 0.6.into(),
            shadow: 0.1.into(),
        });
        clip.stabilize = Some(StabilizeLevel::Smooth);
        clip.filter = Some(Filter::new("mono"));
        clip.lut = Some(Lut::new("/tmp/look.cube"));
        clip.adjust.exposure = 0.25.into();
        clip.animation_in = Some(AnimationRef::new("zoom_in"));
        clip.animation_out = Some(AnimationRef::new("fade_out"));
        clip.replaceable = Some(Replaceable::new(7).with_label("Hero"));
    }

    let before = project.clip(clip).expect("clip").clone();
    assert_eq!(before.beats, vec![20, 70, 95], "model normalized seed");
    let mut cleared = before.clone();
    cleared.beats.clear();
    let mut engine = engine_with(project);

    let outcome = engine
        .apply(Command::Edit(EditCommand::ClearBeats { clip }))
        .expect("clear beats");
    assert_eq!(outcome, ApplyOutcome::Edited(EditOutcome::Updated(clip)));
    assert_eq!(engine.project().clip(clip), Some(&cleared));

    assert!(engine.undo(), "one undo restores the prior grid");
    assert_eq!(engine.project().clip(clip), Some(&before));

    assert!(engine.redo(), "redo restores the exact cleared state");
    assert_eq!(engine.project().clip(clip), Some(&cleared));
}

#[test]
fn clear_beats_rejections_are_atomic_and_record_no_history() {
    let mut project = Project::new("rejections", Rational::FPS_24);
    let track = project.add_track(TrackKind::Text, "Text");
    let generated = project
        .add_generated(track, Generator::text("not media"), tr(0, 48))
        .expect("generated clip");
    let generated_before = project.clip(generated).expect("generated").clone();
    let mut engine = engine_with(project);
    let revision_before = engine.revision();

    let missing = ClipId::from_raw(999_999);
    let err = engine
        .apply(Command::Edit(EditCommand::ClearBeats { clip: missing }))
        .expect_err("unknown clip must fail");
    assert!(matches!(
        err,
        EngineError::Model(ModelError::UnknownClip(id)) if id == missing
    ));

    let err = engine
        .apply(Command::Edit(EditCommand::ClearBeats { clip: generated }))
        .expect_err("generated clip must fail");
    assert!(matches!(
        err,
        EngineError::Model(ModelError::InvalidParam(message))
            if message.contains("media-backed")
    ));

    assert_eq!(
        engine.project().clip(generated),
        Some(&generated_before),
        "both rejections leave clip state exact"
    );
    assert_eq!(engine.project().timeline().clip_count(), 1);
    assert_eq!(engine.revision(), revision_before);
    assert!(!engine.can_undo(), "failed clears push no inverse");
    assert!(!engine.can_redo());
}

#[test]
fn clearing_an_empty_grid_is_one_stable_reversible_step() {
    let (project, _media, clip) = media_project(Vec::new());
    let mut engine = engine_with(project);
    assert!(!engine.can_undo());

    let outcome = engine
        .apply(Command::Edit(EditCommand::ClearBeats { clip }))
        .expect("empty clear is valid");
    assert_eq!(outcome, ApplyOutcome::Edited(EditOutcome::Updated(clip)));
    assert!(engine.project().clip(clip).unwrap().beats.is_empty());
    assert!(
        engine.can_undo(),
        "a successful empty clear intentionally records one no-op inverse"
    );

    assert!(engine.undo());
    assert!(engine.project().clip(clip).unwrap().beats.is_empty());
    assert!(!engine.can_undo());
    assert!(engine.can_redo());

    assert!(engine.redo());
    assert!(engine.project().clip(clip).unwrap().beats.is_empty());
    assert!(engine.can_undo());
    assert!(!engine.can_redo());
}

#[test]
fn detect_beats_remains_unsupported_without_state_or_history_changes() {
    let (project, media, clip) = media_project(vec![20, 50, 80]);
    let mut engine = engine_with(project);
    let clip_before = engine.project().clip(clip).expect("clip").clone();
    let media_before = engine.project().media(media).expect("media").clone();
    let revision_before = engine.revision();

    let err = engine
        .apply(Command::Edit(EditCommand::DetectBeats { clip }))
        .expect_err("pure dispatch cannot analyze audio");
    let EngineError::Unsupported(message) = err else {
        panic!("expected unsupported error, got {err}");
    };
    assert!(message.contains("background job"));
    assert!(message.contains("outside pure dispatch"));
    assert!(message.contains("follow-up edit"));

    assert_eq!(engine.project().clip(clip), Some(&clip_before));
    assert_eq!(engine.project().media(media), Some(&media_before));
    assert_eq!(engine.project().timeline().clip_count(), 1);
    assert_eq!(engine.project().media_count(), 1);
    assert_eq!(engine.revision(), revision_before);
    assert!(!engine.can_undo());
    assert!(!engine.can_redo());
}
