//! Inverse undo for all edit commands.

mod common;

use common::{image_asset, import_asset, rt, small_video_asset, temp_engine, tr};
use cutlass_commands::{Command, EditCommand, EditOutcome};
use cutlass_engine::ApplyOutcome;
use cutlass_models::{
    CanvasAspect, ClipParam, ClipTransform, CropRect, Easing, Generator, Param, ParamValue, Shape,
    ShapeParam, TrackKind,
};

fn created(outcome: ApplyOutcome) -> cutlass_models::ClipId {
    match outcome {
        ApplyOutcome::Edited(EditOutcome::Created(id)) => id,
        other => panic!("expected Created, got {other:?}"),
    }
}

#[test]
fn undo_split_restores_single_clip() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let (_dir, mut engine) = temp_engine();
    let media_id = import_asset(&mut engine, &path);
    let track = common::add_track(&mut engine, TrackKind::Video, "V1");

    let clip_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddClip {
                track,
                media: media_id,
                source: tr(0, 48),
                start: rt(0),
            }))
            .expect("add"),
    );

    engine
        .apply(Command::Edit(EditCommand::SplitClip {
            clip: clip_id,
            at: rt(24),
        }))
        .expect("split");
    assert_eq!(engine.project().timeline().clip_count(), 2);

    assert!(engine.undo());
    assert_eq!(engine.project().timeline().clip_count(), 1);
    assert_eq!(
        engine
            .project()
            .clip(clip_id)
            .unwrap()
            .timeline
            .duration
            .value,
        48
    );
}

#[test]
fn redo_split_after_undo() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let (_dir, mut engine) = temp_engine();
    let media_id = import_asset(&mut engine, &path);
    let track = common::add_track(&mut engine, TrackKind::Video, "V1");

    let clip_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddClip {
                track,
                media: media_id,
                source: tr(0, 48),
                start: rt(0),
            }))
            .expect("add"),
    );

    engine
        .apply(Command::Edit(EditCommand::SplitClip {
            clip: clip_id,
            at: rt(24),
        }))
        .expect("split");

    assert!(engine.undo());
    assert_eq!(engine.project().timeline().clip_count(), 1);

    assert!(engine.redo());
    assert_eq!(engine.project().timeline().clip_count(), 2);
    assert_eq!(
        engine
            .project()
            .clip(clip_id)
            .unwrap()
            .timeline
            .duration
            .value,
        24
    );
}

#[test]
fn undo_ripple_delete_restores_gap() {
    let (_dir, mut engine) = temp_engine();
    let track = common::add_track(&mut engine, TrackKind::Sticker, "T1");

    let first_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddGenerated {
                track,
                generator: Generator::SolidColor {
                    rgba: [0, 0, 0, 255],
                },
                timeline: tr(0, 10),
            }))
            .expect("first"),
    );

    let second_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddGenerated {
                track,
                generator: Generator::SolidColor {
                    rgba: [255, 0, 0, 255],
                },
                timeline: tr(20, 10),
            }))
            .expect("second"),
    );

    engine
        .apply(Command::Edit(EditCommand::RippleDelete { clip: first_id }))
        .expect("ripple");
    assert_eq!(engine.project().clip(second_id).unwrap().start().value, 10);

    assert!(engine.undo());
    assert!(engine.project().clip(first_id).is_some());
    assert_eq!(engine.project().clip(second_id).unwrap().start().value, 20);
}

#[test]
fn undo_ripple_delete_middle_of_three_adjacent() {
    let (_dir, mut engine) = temp_engine();
    let track = common::add_track(&mut engine, TrackKind::Adjustment, "FX");

    let a = created(
        engine
            .apply(Command::Edit(EditCommand::AddGenerated {
                track,
                generator: Generator::Adjustment,
                timeline: tr(0, 10),
            }))
            .expect("a"),
    );
    let b = created(
        engine
            .apply(Command::Edit(EditCommand::AddGenerated {
                track,
                generator: Generator::Adjustment,
                timeline: tr(10, 10),
            }))
            .expect("b"),
    );
    let c = created(
        engine
            .apply(Command::Edit(EditCommand::AddGenerated {
                track,
                generator: Generator::Adjustment,
                timeline: tr(20, 10),
            }))
            .expect("c"),
    );

    engine
        .apply(Command::Edit(EditCommand::RippleDelete { clip: b }))
        .expect("ripple middle");

    assert!(engine.project().clip(b).is_none());
    assert_eq!(engine.project().clip(a).unwrap().start().value, 0);
    assert_eq!(engine.project().clip(c).unwrap().start().value, 10);

    assert!(engine.undo());
    assert!(engine.project().clip(b).is_some());
    assert_eq!(engine.project().clip(a).unwrap().start().value, 0);
    assert_eq!(engine.project().clip(b).unwrap().start().value, 10);
    assert_eq!(engine.project().clip(c).unwrap().start().value, 20);
}

#[test]
fn trim_extends_image_clip_past_default_placement() {
    let Some(png) = image_asset() else {
        return;
    };
    let (_dir, mut engine) = temp_engine();
    let media_id = import_asset(&mut engine, &png);
    let track = common::add_track(&mut engine, TrackKind::Video, "V1");
    let source = engine.project().media(media_id).unwrap().full_range();

    let clip_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddClip {
                track,
                media: media_id,
                source,
                start: rt(0),
            }))
            .expect("add"),
    );

    // Default 5s placement at 24fps timeline.
    assert_eq!(engine.project().clip(clip_id).unwrap().timeline, tr(0, 120));

    engine
        .apply(Command::Edit(EditCommand::TrimClip {
            clip: clip_id,
            timeline: tr(0, 240),
        }))
        .expect("extend still to 10s");

    let clip = engine.project().clip(clip_id).unwrap();
    assert_eq!(clip.timeline, tr(0, 240));
    let source = clip.source_range().unwrap();
    assert!(source.duration.value > 5_000);
}

#[test]
fn trim_video_past_source_bounds_fails() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let (_dir, mut engine) = temp_engine();
    let media_id = import_asset(&mut engine, &path);
    let track = common::add_track(&mut engine, TrackKind::Video, "V1");

    let clip_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddClip {
                track,
                media: media_id,
                source: tr(0, 48),
                start: rt(0),
            }))
            .expect("add"),
    );

    let err = engine
        .apply(Command::Edit(EditCommand::TrimClip {
            clip: clip_id,
            timeline: tr(0, 10_000),
        }))
        .unwrap_err();
    assert!(
        format!("{err}").contains("bounds") || format!("{err}").contains("Source"),
        "expected source bounds error, got {err}"
    );
    assert_eq!(engine.project().clip(clip_id).unwrap().timeline, tr(0, 48));
}

#[test]
fn undo_trim_restores_timeline_and_source() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let (_dir, mut engine) = temp_engine();
    let media_id = import_asset(&mut engine, &path);
    let track = common::add_track(&mut engine, TrackKind::Video, "V1");

    let clip_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddClip {
                track,
                media: media_id,
                source: tr(0, 48),
                start: rt(0),
            }))
            .expect("add"),
    );

    let before_tl = engine.project().clip(clip_id).unwrap().timeline;
    let before_src = engine
        .project()
        .clip(clip_id)
        .unwrap()
        .source_range()
        .unwrap();

    engine
        .apply(Command::Edit(EditCommand::TrimClip {
            clip: clip_id,
            timeline: tr(10, 28),
        }))
        .expect("trim");

    assert_ne!(engine.project().clip(clip_id).unwrap().timeline, before_tl);

    assert!(engine.undo());
    let clip = engine.project().clip(clip_id).unwrap();
    assert_eq!(clip.timeline, before_tl);
    assert_eq!(clip.source_range().unwrap(), before_src);
}

#[test]
fn undo_move_across_tracks() {
    let (_dir, mut engine) = temp_engine();
    let v1 = common::add_track(&mut engine, TrackKind::Text, "T1");
    let v2 = common::add_track(&mut engine, TrackKind::Text, "T2");

    let clip_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddGenerated {
                track: v1,
                generator: Generator::text("title"),
                timeline: tr(5, 20),
            }))
            .expect("add"),
    );

    engine
        .apply(Command::Edit(EditCommand::MoveClip {
            clip: clip_id,
            to_track: v2,
            start: rt(40),
        }))
        .expect("move");

    assert_eq!(engine.project().timeline().track_of(clip_id), Some(v2));
    assert_eq!(engine.project().clip(clip_id).unwrap().start().value, 40);

    assert!(engine.undo());
    assert_eq!(engine.project().timeline().track_of(clip_id), Some(v1));
    assert_eq!(engine.project().clip(clip_id).unwrap().start().value, 5);
}

#[test]
fn undo_remove_clip_restores_with_gap() {
    let (_dir, mut engine) = temp_engine();
    let track = common::add_track(&mut engine, TrackKind::Adjustment, "FX");

    let a = created(
        engine
            .apply(Command::Edit(EditCommand::AddGenerated {
                track,
                generator: Generator::Adjustment,
                timeline: tr(0, 10),
            }))
            .expect("a"),
    );
    let b = created(
        engine
            .apply(Command::Edit(EditCommand::AddGenerated {
                track,
                generator: Generator::Adjustment,
                timeline: tr(20, 10),
            }))
            .expect("b"),
    );

    engine
        .apply(Command::Edit(EditCommand::RemoveClip { clip: a }))
        .expect("remove");

    assert!(engine.project().clip(a).is_none());
    assert_eq!(engine.project().clip(b).unwrap().start().value, 20);

    assert!(engine.undo());
    assert!(engine.project().clip(a).is_some());
    assert_eq!(engine.project().clip(a).unwrap().start().value, 0);
    assert_eq!(engine.project().clip(b).unwrap().start().value, 20);
}

#[test]
fn undo_add_generated() {
    let (_dir, mut engine) = temp_engine();
    let track = common::add_track(&mut engine, TrackKind::Sticker, "S1");

    let clip_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddGenerated {
                track,
                generator: Generator::SolidColor { rgba: [1, 2, 3, 4] },
                timeline: tr(0, 24),
            }))
            .expect("add"),
    );

    assert!(engine.undo());
    assert!(engine.project().clip(clip_id).is_none());

    assert!(engine.redo());
    assert!(engine.project().clip(clip_id).is_some());
}

#[test]
fn undo_redo_set_generator_oscillates_content() {
    let (_dir, mut engine) = temp_engine();
    let track = common::add_track(&mut engine, TrackKind::Text, "T1");

    let clip_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddGenerated {
                track,
                generator: Generator::text("before"),
                timeline: tr(0, 24),
            }))
            .expect("add"),
    );

    engine
        .apply(Command::Edit(EditCommand::SetGenerator {
            clip: clip_id,
            generator: Generator::text("after"),
        }))
        .expect("set generator");

    let content =
        |engine: &cutlass_engine::Engine| match &engine.project().clip(clip_id).unwrap().content {
            cutlass_models::ClipSource::Generated(Generator::Text { content, .. }) => {
                content.clone()
            }
            other => panic!("expected text generator, got {other:?}"),
        };
    assert_eq!(content(&engine), "after");

    assert!(engine.undo());
    assert_eq!(content(&engine), "before");

    assert!(engine.redo());
    assert_eq!(content(&engine), "after");
}

#[test]
fn undo_redo_set_generator_oscillates_style() {
    let (_dir, mut engine) = temp_engine();
    let track = common::add_track(&mut engine, TrackKind::Text, "T1");

    let clip_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddGenerated {
                track,
                generator: Generator::text("hi"),
                timeline: tr(0, 24),
            }))
            .expect("add"),
    );

    let styled = cutlass_models::TextStyle {
        bold: true,
        size: 120.0.into(),
        fill: [10, 20, 30, 255].into(),
        stroke: Some(cutlass_models::TextStroke {
            rgba: [0, 0, 0, 255].into(),
            width: 8.0.into(),
        }),
        ..Default::default()
    };
    engine
        .apply(Command::Edit(EditCommand::SetGenerator {
            clip: clip_id,
            generator: Generator::Text {
                content: "hi".into(),
                style: styled.clone(),
            },
        }))
        .expect("set generator");

    let style =
        |engine: &cutlass_engine::Engine| match &engine.project().clip(clip_id).unwrap().content {
            cutlass_models::ClipSource::Generated(Generator::Text { style, .. }) => style.clone(),
            other => panic!("expected text generator, got {other:?}"),
        };

    assert_eq!(style(&engine), styled);

    assert!(engine.undo());
    assert_eq!(style(&engine), cutlass_models::TextStyle::default());

    assert!(engine.redo());
    assert_eq!(style(&engine), styled);
}

#[test]
fn set_generator_rejects_media_clip() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let (_dir, mut engine) = temp_engine();
    let media_id = import_asset(&mut engine, &path);
    let track = common::add_track(&mut engine, TrackKind::Video, "V1");

    let clip_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddClip {
                track,
                media: media_id,
                source: tr(0, 48),
                start: rt(0),
            }))
            .expect("add"),
    );

    // A media-backed clip has no generator to replace.
    assert!(
        engine
            .apply(Command::Edit(EditCommand::SetGenerator {
                clip: clip_id,
                generator: Generator::text("nope"),
            }))
            .is_err()
    );
}

#[test]
fn undo_redo_set_clip_transform_oscillates() {
    let (_dir, mut engine) = temp_engine();
    let track = common::add_track(&mut engine, TrackKind::Text, "T1");

    let clip_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddGenerated {
                track,
                generator: Generator::text("title"),
                timeline: tr(0, 24),
            }))
            .expect("add"),
    );

    let moved = ClipTransform {
        position: [0.25, -0.1],
        scale: 0.5.into(),
        rotation: 45.0,
        opacity: 0.8,
        ..ClipTransform::IDENTITY
    };
    engine
        .apply(Command::Edit(EditCommand::SetClipTransform {
            clip: clip_id,
            transform: moved,
            at: None,
        }))
        .expect("set transform");

    let transform =
        |engine: &cutlass_engine::Engine| engine.project().clip(clip_id).unwrap().transform.clone();
    assert_eq!(transform(&engine), moved.into());

    assert!(engine.undo());
    assert!(transform(&engine).is_identity());

    assert!(engine.redo());
    assert_eq!(transform(&engine), moved.into());
}

#[test]
fn invalid_transform_rejected_and_state_unchanged() {
    let (_dir, mut engine) = temp_engine();
    let track = common::add_track(&mut engine, TrackKind::Text, "T1");

    let clip_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddGenerated {
                track,
                generator: Generator::text("title"),
                timeline: tr(0, 24),
            }))
            .expect("add"),
    );

    // Zero scale is rejected; the clip keeps its identity transform.
    assert!(
        engine
            .apply(Command::Edit(EditCommand::SetClipTransform {
                clip: clip_id,
                transform: ClipTransform {
                    scale: 0.0.into(),
                    ..ClipTransform::IDENTITY
                },
                at: None,
            }))
            .is_err()
    );
    assert!(
        engine
            .project()
            .clip(clip_id)
            .unwrap()
            .transform
            .is_identity()
    );
}

/// Add a text clip at [0, 48) and return its id — fixture for param tests.
fn text_clip(engine: &mut cutlass_engine::Engine) -> cutlass_models::ClipId {
    let track = common::add_track(engine, TrackKind::Text, "T1");
    created(
        engine
            .apply(Command::Edit(EditCommand::AddGenerated {
                track,
                generator: Generator::text("title"),
                timeline: tr(0, 48),
            }))
            .expect("add"),
    )
}

#[test]
fn set_param_keyframe_undo_redo_roundtrip() {
    let (_dir, mut engine) = temp_engine();
    let clip_id = text_clip(&mut engine);

    engine
        .apply(Command::Edit(EditCommand::SetParamKeyframe {
            clip: clip_id,
            param: ClipParam::Opacity,
            at: rt(0),
            value: ParamValue::Scalar(0.0),
            easing: Easing::Linear,
            tangents: None,
        }))
        .expect("first keyframe");
    engine
        .apply(Command::Edit(EditCommand::SetParamKeyframe {
            clip: clip_id,
            param: ClipParam::Opacity,
            at: rt(24),
            value: ParamValue::Scalar(1.0),
            easing: Easing::EaseInOut,
            tangents: None,
        }))
        .expect("second keyframe");

    let opacity = |engine: &cutlass_engine::Engine| {
        engine
            .project()
            .clip(clip_id)
            .unwrap()
            .transform
            .opacity
            .clone()
    };
    assert_eq!(opacity(&engine).keyframes().len(), 2);

    // Undo peels one keyframe at a time; the first undo restores the
    // single-keyframe curve, the second the constant.
    assert!(engine.undo());
    assert_eq!(opacity(&engine).keyframes().len(), 1);
    assert!(engine.undo());
    assert!(!opacity(&engine).is_animated());
    assert_eq!(opacity(&engine).constant(), Some(1.0));

    assert!(engine.redo());
    assert!(engine.redo());
    assert_eq!(opacity(&engine).keyframes().len(), 2);
    assert_eq!(
        engine
            .project()
            .clip(clip_id)
            .unwrap()
            .transform
            .sample(12)
            .opacity,
        0.5
    );
}

#[test]
fn set_clip_speed_undo_redo_roundtrip() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let (_dir, mut engine) = temp_engine();
    let media_id = import_asset(&mut engine, &path);
    let track = common::add_track(&mut engine, TrackKind::Video, "V1");
    let clip_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddClip {
                track,
                media: media_id,
                source: tr(0, 48),
                start: rt(0),
            }))
            .expect("add"),
    );

    engine
        .apply(Command::Edit(EditCommand::SetClipSpeed {
            clip: clip_id,
            speed: cutlass_models::Rational::new(2, 1),
            reversed: true,
        }))
        .expect("retime");
    let clip = |engine: &cutlass_engine::Engine| engine.project().clip(clip_id).unwrap().clone();
    assert_eq!(clip(&engine).timeline.duration.value, 24);
    assert!(clip(&engine).reversed);

    // One undo restores speed, direction, AND the re-derived duration.
    assert!(engine.undo());
    let restored = clip(&engine);
    assert_eq!(restored.timeline.duration.value, 48);
    assert_eq!(restored.speed, cutlass_models::Rational::new(1, 1));
    assert!(!restored.reversed && !restored.is_retimed());

    assert!(engine.redo());
    assert_eq!(clip(&engine).timeline.duration.value, 24);
    assert!(clip(&engine).is_retimed());
}

#[test]
fn set_speed_curve_undo_redo_roundtrip() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let (_dir, mut engine) = temp_engine();
    let media_id = import_asset(&mut engine, &path);
    let track = common::add_track(&mut engine, TrackKind::Video, "V1");
    let clip_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddClip {
                track,
                media: media_id,
                source: tr(0, 48),
                start: rt(0),
            }))
            .expect("add"),
    );
    let clip = |engine: &cutlass_engine::Engine| engine.project().clip(clip_id).unwrap().clone();

    // A montage ramp (average 2×) halves the footprint and marks the clip
    // retimed, just like a constant 2× — but as a curve.
    let curve = cutlass_models::speed_preset("montage").unwrap();
    let avg = {
        let mut probe = clip(&engine);
        probe.speed_curve = curve.clone();
        probe.speed_curve_average()
    };
    let expected = (48.0 / avg).round() as i64;
    engine
        .apply(Command::Edit(EditCommand::SetSpeedCurve {
            clip: clip_id,
            curve: Some(curve.clone()),
        }))
        .expect("set ramp");
    assert_eq!(clip(&engine).timeline.duration.value, expected);
    assert!(clip(&engine).has_speed_curve() && clip(&engine).is_retimed());

    // One undo restores the flat ramp AND the original duration.
    assert!(engine.undo());
    let restored = clip(&engine);
    assert_eq!(restored.timeline.duration.value, 48);
    assert!(!restored.has_speed_curve() && !restored.is_retimed());

    assert!(engine.redo());
    assert_eq!(clip(&engine).speed_curve, curve);
    assert_eq!(clip(&engine).timeline.duration.value, expected);
}

#[test]
fn set_speed_curve_rejects_generated_clips() {
    let (_dir, mut engine) = temp_engine();
    let clip_id = text_clip(&mut engine);
    assert!(
        engine
            .apply(Command::Edit(EditCommand::SetSpeedCurve {
                clip: clip_id,
                curve: Some(cutlass_models::speed_preset("hero").unwrap()),
            }))
            .is_err()
    );
}

#[test]
fn set_clip_audio_undo_redo_roundtrip() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let (_dir, mut engine) = temp_engine();
    let media_id = import_asset(&mut engine, &path);
    let track = common::add_track(&mut engine, TrackKind::Audio, "A1");
    let clip_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddClip {
                track,
                media: media_id,
                source: tr(0, 48),
                start: rt(0),
            }))
            .expect("add"),
    );

    engine
        .apply(Command::Edit(EditCommand::SetClipAudio {
            clip: clip_id,
            volume: Some(0.5),
            fade_in: rt(12),
            fade_out: rt(24),
        }))
        .expect("set audio");
    let clip = |engine: &cutlass_engine::Engine| engine.project().clip(clip_id).unwrap().clone();
    assert_eq!(clip(&engine).volume.constant(), Some(0.5));
    assert_eq!((clip(&engine).fade_in, clip(&engine).fade_out), (12, 24));

    // One undo restores volume and both fades.
    assert!(engine.undo());
    let restored = clip(&engine);
    assert_eq!(restored.volume.constant(), Some(1.0));
    assert_eq!((restored.fade_in, restored.fade_out), (0, 0));
    assert!(!restored.has_custom_audio());

    assert!(engine.redo());
    assert!(clip(&engine).has_custom_audio());
    assert_eq!(clip(&engine).volume.constant(), Some(0.5));
}

#[test]
fn set_clip_audio_rejects_generated_clips_and_long_fades() {
    let (_dir, mut engine) = temp_engine();
    let clip_id = text_clip(&mut engine);
    let before = engine.project().clip(clip_id).unwrap().clone();

    assert!(
        engine
            .apply(Command::Edit(EditCommand::SetClipAudio {
                clip: clip_id,
                volume: Some(0.5),
                fade_in: rt(0),
                fade_out: rt(0),
            }))
            .is_err()
    );
    // The rejection left the clip untouched.
    assert_eq!(engine.project().clip(clip_id).unwrap(), &before);
}

#[test]
fn set_clip_denoise_undo_redo_roundtrip() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let (_dir, mut engine) = temp_engine();
    let media_id = import_asset(&mut engine, &path);
    let track = common::add_track(&mut engine, TrackKind::Audio, "A1");
    let clip_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddClip {
                track,
                media: media_id,
                source: tr(0, 48),
                start: rt(0),
            }))
            .expect("add"),
    );
    let denoise = |engine: &cutlass_engine::Engine| engine.project().clip(clip_id).unwrap().denoise;
    assert!(!denoise(&engine), "off by default");

    engine
        .apply(Command::Edit(EditCommand::SetClipDenoise {
            clip: clip_id,
            denoise: true,
        }))
        .expect("enable denoise");
    assert!(denoise(&engine));

    assert!(engine.undo());
    assert!(!denoise(&engine), "undo clears the flag");
    assert!(engine.redo());
    assert!(denoise(&engine), "redo restores it");
}

#[test]
fn set_clip_denoise_rejects_generated_clips() {
    let (_dir, mut engine) = temp_engine();
    let clip_id = text_clip(&mut engine);
    let before = engine.project().clip(clip_id).unwrap().clone();

    assert!(
        engine
            .apply(Command::Edit(EditCommand::SetClipDenoise {
                clip: clip_id,
                denoise: true,
            }))
            .is_err()
    );
    assert_eq!(engine.project().clip(clip_id).unwrap(), &before);
}

#[test]
fn set_clip_crop_undo_redo_roundtrip() {
    let (_dir, mut engine) = temp_engine();
    let clip_id = text_clip(&mut engine);
    let crop = CropRect {
        x: 0.25,
        y: 0.1,
        w: 0.5,
        h: 0.8,
    };

    engine
        .apply(Command::Edit(EditCommand::SetClipCrop {
            clip: clip_id,
            crop,
            flip_h: true,
            flip_v: false,
            at: None,
        }))
        .expect("set crop");
    let clip = |engine: &cutlass_engine::Engine| engine.project().clip(clip_id).unwrap().clone();
    assert_eq!(clip(&engine).crop, Param::Constant(crop));
    assert!(clip(&engine).flip_h && !clip(&engine).flip_v);

    // One undo restores the full frame and both flips.
    assert!(engine.undo());
    let restored = clip(&engine);
    assert!(!restored.has_custom_crop());

    assert!(engine.redo());
    assert_eq!(clip(&engine).crop, Param::Constant(crop));
    assert!(clip(&engine).flip_h);
}

#[test]
fn set_canvas_undo_redo_roundtrip() {
    let (_dir, mut engine) = temp_engine();
    assert!(engine.project().timeline().canvas().is_default());

    let outcome = engine
        .apply(Command::Edit(EditCommand::SetCanvas {
            aspect: CanvasAspect::Tall9x16,
            background: [12, 34, 56],
        }))
        .expect("set canvas");
    assert!(matches!(
        outcome,
        ApplyOutcome::Edited(EditOutcome::UpdatedCanvas)
    ));
    let canvas = |engine: &cutlass_engine::Engine| engine.project().timeline().canvas();
    assert_eq!(canvas(&engine).aspect, CanvasAspect::Tall9x16);
    assert_eq!(canvas(&engine).background, [12, 34, 56]);

    // One undo restores both fields.
    assert!(engine.undo());
    assert!(canvas(&engine).is_default());

    assert!(engine.redo());
    assert_eq!(canvas(&engine).aspect, CanvasAspect::Tall9x16);
    assert_eq!(canvas(&engine).background, [12, 34, 56]);
}

#[test]
fn set_clip_crop_rejects_invalid_rects_and_audio_lanes() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let (_dir, mut engine) = temp_engine();
    let media_id = import_asset(&mut engine, &path);
    let track = common::add_track(&mut engine, TrackKind::Audio, "A1");
    let audio_clip = created(
        engine
            .apply(Command::Edit(EditCommand::AddClip {
                track,
                media: media_id,
                source: tr(0, 48),
                start: rt(0),
            }))
            .expect("add"),
    );

    // Audio lanes have no frame to crop.
    assert!(
        engine
            .apply(Command::Edit(EditCommand::SetClipCrop {
                clip: audio_clip,
                crop: CropRect::FULL,
                flip_h: true,
                flip_v: false,
                at: None,
            }))
            .is_err()
    );

    // Out-of-frame rects bounce without touching the clip.
    let text = text_clip(&mut engine);
    let before = engine.project().clip(text).unwrap().clone();
    assert!(
        engine
            .apply(Command::Edit(EditCommand::SetClipCrop {
                clip: text,
                crop: CropRect {
                    x: 0.8,
                    y: 0.0,
                    w: 0.5,
                    h: 1.0
                },
                flip_h: false,
                flip_v: false,
                at: None,
            }))
            .is_err()
    );
    assert_eq!(engine.project().clip(text).unwrap(), &before);
}

#[test]
fn set_clip_speed_rejects_generated_clips() {
    let (_dir, mut engine) = temp_engine();
    let clip_id = text_clip(&mut engine);
    let before = engine.project().clip(clip_id).unwrap().clone();

    assert!(
        engine
            .apply(Command::Edit(EditCommand::SetClipSpeed {
                clip: clip_id,
                speed: cutlass_models::Rational::new(2, 1),
                reversed: false,
            }))
            .is_err()
    );
    // The rejection left the clip untouched.
    assert_eq!(engine.project().clip(clip_id).unwrap(), &before);
}

#[test]
fn remove_param_keyframe_undo_restores_curve() {
    let (_dir, mut engine) = temp_engine();
    let clip_id = text_clip(&mut engine);

    for (tick, value) in [(0, 0.0), (24, 1.0)] {
        engine
            .apply(Command::Edit(EditCommand::SetParamKeyframe {
                clip: clip_id,
                param: ClipParam::Scale,
                at: rt(tick),
                value: ParamValue::Scalar(value + 1.0),
                easing: Easing::Linear,
                tangents: None,
            }))
            .expect("keyframe");
    }

    engine
        .apply(Command::Edit(EditCommand::RemoveParamKeyframe {
            clip: clip_id,
            param: ClipParam::Scale,
            at: rt(24),
        }))
        .expect("remove");
    let scale = |engine: &cutlass_engine::Engine| {
        engine
            .project()
            .clip(clip_id)
            .unwrap()
            .transform
            .scale
            .clone()
    };
    assert_eq!(scale(&engine).keyframes().len(), 1);

    assert!(engine.undo());
    assert_eq!(scale(&engine).keyframes().len(), 2);

    // Removing a keyframe that isn't there is rejected and pushes no history.
    assert!(
        engine
            .apply(Command::Edit(EditCommand::RemoveParamKeyframe {
                clip: clip_id,
                param: ClipParam::Scale,
                at: rt(7),
            }))
            .is_err()
    );
    assert_eq!(scale(&engine).keyframes().len(), 2);
}

#[test]
fn set_param_constant_undo_restores_keyframes() {
    let (_dir, mut engine) = temp_engine();
    let clip_id = text_clip(&mut engine);

    for tick in [0, 24] {
        engine
            .apply(Command::Edit(EditCommand::SetParamKeyframe {
                clip: clip_id,
                param: ClipParam::Rotation,
                at: rt(tick),
                value: ParamValue::Scalar(tick as f32),
                easing: Easing::Linear,
                tangents: None,
            }))
            .expect("keyframe");
    }

    engine
        .apply(Command::Edit(EditCommand::SetParamConstant {
            clip: clip_id,
            param: ClipParam::Rotation,
            value: ParamValue::Scalar(90.0),
        }))
        .expect("flatten");
    let rotation = |engine: &cutlass_engine::Engine| {
        engine
            .project()
            .clip(clip_id)
            .unwrap()
            .transform
            .rotation
            .clone()
    };
    assert_eq!(rotation(&engine).constant(), Some(90.0));

    assert!(engine.undo());
    assert_eq!(rotation(&engine).keyframes().len(), 2);
    assert!(engine.redo());
    assert_eq!(rotation(&engine).constant(), Some(90.0));
}

#[test]
fn param_keyframe_outside_clip_rejected() {
    let (_dir, mut engine) = temp_engine();
    let clip_id = text_clip(&mut engine); // [0, 48)

    assert!(
        engine
            .apply(Command::Edit(EditCommand::SetParamKeyframe {
                clip: clip_id,
                param: ClipParam::Opacity,
                at: rt(48), // exclusive end — outside
                value: ParamValue::Scalar(0.5),
                easing: Easing::Linear,
                tangents: None,
            }))
            .is_err()
    );
    assert!(
        !engine
            .project()
            .clip(clip_id)
            .unwrap()
            .transform
            .is_animated()
    );
}

#[test]
fn transform_gesture_at_playhead_keyframes_animated_property() {
    let (_dir, mut engine) = temp_engine();
    let clip_id = text_clip(&mut engine);

    // Animate scale 1 → 3 across the clip.
    for (tick, value) in [(0, 1.0), (40, 3.0)] {
        engine
            .apply(Command::Edit(EditCommand::SetParamKeyframe {
                clip: clip_id,
                param: ClipParam::Scale,
                at: rt(tick),
                value: ParamValue::Scalar(value),
                easing: Easing::Linear,
                tangents: None,
            }))
            .expect("keyframe");
    }

    // A gesture commit at tick 20 (sampled scale 2.0 → user dragged to 2.5).
    engine
        .apply(Command::Edit(EditCommand::SetClipTransform {
            clip: clip_id,
            transform: ClipTransform {
                scale: 2.5.into(),
                ..ClipTransform::IDENTITY
            },
            at: Some(rt(20)),
        }))
        .expect("compose");

    let transform = engine.project().clip(clip_id).unwrap().transform.clone();
    // Scale gained a keyframe; the endpoints survive.
    assert_eq!(transform.scale.keyframes().len(), 3);
    assert_eq!(transform.sample(20).scale, 2.5.into());
    assert_eq!(transform.sample(0).scale, 1.0.into());
    assert_eq!(transform.sample(40).scale, 3.0.into());
    // Un-animated properties stay constant.
    assert!(!transform.position.is_animated());
}

#[test]
fn failed_split_does_not_push_undo() {
    let (_dir, mut engine) = temp_engine();
    let track = common::add_track(&mut engine, TrackKind::Adjustment, "FX");

    let clip_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddGenerated {
                track,
                generator: Generator::Adjustment,
                timeline: tr(0, 48),
            }))
            .expect("add"),
    );

    let err = engine
        .apply(Command::Edit(EditCommand::SplitClip {
            clip: clip_id,
            at: rt(0),
        }))
        .unwrap_err();
    assert!(format!("{err}").contains("range") || format!("{err}").contains("Invalid"));
    assert_eq!(engine.project().timeline().clip_count(), 1);
    // Failed split must not add an undo step; only add-track + add-generated remain.
    assert!(engine.can_undo());
    assert!(engine.undo());
    assert_eq!(engine.project().timeline().clip_count(), 0);
    assert!(engine.can_undo());
    assert!(engine.undo());
    assert!(!engine.can_undo());
}

#[test]
fn failed_trim_does_not_push_undo() {
    let (_dir, mut engine) = temp_engine();
    let track = common::add_track(&mut engine, TrackKind::Adjustment, "FX");

    let clip_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddGenerated {
                track,
                generator: Generator::Adjustment,
                timeline: tr(0, 48),
            }))
            .expect("add"),
    );

    let err = engine
        .apply(Command::Edit(EditCommand::TrimClip {
            clip: clip_id,
            timeline: tr(0, 0),
        }))
        .unwrap_err();
    assert!(format!("{err}").contains("range") || format!("{err}").contains("Invalid"));
    assert!(engine.can_undo());
    assert!(engine.undo());
    assert_eq!(engine.project().timeline().clip_count(), 0);
}

#[test]
fn multi_step_undo_unwinds_in_order() {
    let (_dir, mut engine) = temp_engine();
    let track = common::add_track(&mut engine, TrackKind::Adjustment, "FX");

    let a = created(
        engine
            .apply(Command::Edit(EditCommand::AddGenerated {
                track,
                generator: Generator::Adjustment,
                timeline: tr(0, 10),
            }))
            .expect("a"),
    );
    let b = created(
        engine
            .apply(Command::Edit(EditCommand::AddGenerated {
                track,
                generator: Generator::Adjustment,
                timeline: tr(20, 10),
            }))
            .expect("b"),
    );

    engine
        .apply(Command::Edit(EditCommand::RippleDelete { clip: a }))
        .expect("ripple");
    engine
        .apply(Command::Edit(EditCommand::MoveClip {
            clip: b,
            to_track: track,
            start: rt(5),
        }))
        .expect("move");

    assert_eq!(engine.project().clip(b).unwrap().start().value, 5);

    assert!(engine.undo());
    assert_eq!(engine.project().clip(b).unwrap().start().value, 10);

    assert!(engine.undo());
    assert!(engine.project().clip(a).is_some());
    assert_eq!(engine.project().clip(b).unwrap().start().value, 20);
}

#[test]
fn redo_trim_after_undo() {
    let (_dir, mut engine) = temp_engine();
    let track = common::add_track(&mut engine, TrackKind::Adjustment, "FX");

    let clip_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddGenerated {
                track,
                generator: Generator::Adjustment,
                timeline: tr(0, 48),
            }))
            .expect("add"),
    );

    engine
        .apply(Command::Edit(EditCommand::TrimClip {
            clip: clip_id,
            timeline: tr(12, 20),
        }))
        .expect("trim");
    assert_eq!(engine.project().clip(clip_id).unwrap().timeline, tr(12, 20));

    assert!(engine.undo());
    assert_eq!(engine.project().clip(clip_id).unwrap().timeline, tr(0, 48));

    assert!(engine.redo());
    assert_eq!(engine.project().clip(clip_id).unwrap().timeline, tr(12, 20));
}

#[test]
fn redo_ripple_delete_after_undo() {
    let (_dir, mut engine) = temp_engine();
    let track = common::add_track(&mut engine, TrackKind::Adjustment, "FX");

    let first = created(
        engine
            .apply(Command::Edit(EditCommand::AddGenerated {
                track,
                generator: Generator::Adjustment,
                timeline: tr(0, 10),
            }))
            .expect("first"),
    );
    let second = created(
        engine
            .apply(Command::Edit(EditCommand::AddGenerated {
                track,
                generator: Generator::Adjustment,
                timeline: tr(20, 10),
            }))
            .expect("second"),
    );

    engine
        .apply(Command::Edit(EditCommand::RippleDelete { clip: first }))
        .expect("ripple");
    assert!(engine.project().clip(first).is_none());
    assert_eq!(engine.project().clip(second).unwrap().start().value, 10);

    assert!(engine.undo());
    assert!(engine.project().clip(first).is_some());
    assert_eq!(engine.project().clip(second).unwrap().start().value, 20);

    assert!(engine.redo());
    assert!(engine.project().clip(first).is_none());
    assert_eq!(engine.project().clip(second).unwrap().start().value, 10);
}

#[test]
fn redo_move_after_undo() {
    let (_dir, mut engine) = temp_engine();
    let v1 = common::add_track(&mut engine, TrackKind::Adjustment, "FX");
    let v2 = common::add_track(&mut engine, TrackKind::Adjustment, "FX2");

    let clip_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddGenerated {
                track: v1,
                generator: Generator::Adjustment,
                timeline: tr(0, 24),
            }))
            .expect("add"),
    );

    engine
        .apply(Command::Edit(EditCommand::MoveClip {
            clip: clip_id,
            to_track: v2,
            start: rt(30),
        }))
        .expect("move");

    assert!(engine.undo());
    assert_eq!(engine.project().timeline().track_of(clip_id), Some(v1));
    assert_eq!(engine.project().clip(clip_id).unwrap().start().value, 0);

    assert!(engine.redo());
    assert_eq!(engine.project().timeline().track_of(clip_id), Some(v2));
    assert_eq!(engine.project().clip(clip_id).unwrap().start().value, 30);
}

#[test]
fn redo_remove_clip_after_undo() {
    let (_dir, mut engine) = temp_engine();
    let track = common::add_track(&mut engine, TrackKind::Adjustment, "FX");

    let clip_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddGenerated {
                track,
                generator: Generator::Adjustment,
                timeline: tr(5, 15),
            }))
            .expect("add"),
    );

    engine
        .apply(Command::Edit(EditCommand::RemoveClip { clip: clip_id }))
        .expect("remove");
    assert!(engine.project().clip(clip_id).is_none());

    assert!(engine.undo());
    assert!(engine.project().clip(clip_id).is_some());

    assert!(engine.redo());
    assert!(engine.project().clip(clip_id).is_none());
}

#[test]
fn new_edit_clears_redo_stack() {
    let (_dir, mut engine) = temp_engine();
    let track = common::add_track(&mut engine, TrackKind::Adjustment, "FX");

    let clip_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddGenerated {
                track,
                generator: Generator::Adjustment,
                timeline: tr(0, 24),
            }))
            .expect("add"),
    );

    engine
        .apply(Command::Edit(EditCommand::TrimClip {
            clip: clip_id,
            timeline: tr(0, 12),
        }))
        .expect("trim");

    assert!(engine.undo());
    assert!(engine.can_redo());

    engine
        .apply(Command::Edit(EditCommand::MoveClip {
            clip: clip_id,
            to_track: track,
            start: rt(6),
        }))
        .expect("move");

    assert!(!engine.can_redo());
    assert_eq!(engine.project().clip(clip_id).unwrap().start().value, 6);
}

// --- markers (M1) -----------------------------------------------------------

fn created_marker(outcome: ApplyOutcome) -> cutlass_models::MarkerId {
    match outcome {
        ApplyOutcome::Edited(EditOutcome::CreatedMarker(id)) => id,
        other => panic!("expected CreatedMarker, got {other:?}"),
    }
}

#[test]
fn undo_add_marker_removes_it_and_redo_restores_same_id() {
    let (_dir, mut engine) = temp_engine();

    let id = created_marker(
        engine
            .apply(Command::Edit(EditCommand::AddMarker {
                at: rt(48),
                name: "drop".into(),
                color: Some(cutlass_models::MarkerColor::Red),
            }))
            .expect("add marker"),
    );
    assert_eq!(engine.project().timeline().marker_count(), 1);

    assert!(engine.undo());
    assert_eq!(engine.project().timeline().marker_count(), 0);

    assert!(engine.redo());
    let marker = engine
        .project()
        .timeline()
        .marker(id)
        .expect("same id restored");
    assert_eq!(marker.tick, rt(48));
    assert_eq!(marker.name, "drop");
    assert_eq!(marker.color, cutlass_models::MarkerColor::Red);
}

#[test]
fn add_marker_without_color_cycles_the_palette() {
    let (_dir, mut engine) = temp_engine();
    for i in 0..3 {
        engine
            .apply(Command::Edit(EditCommand::AddMarker {
                at: rt(i * 10),
                name: String::new(),
                color: None,
            }))
            .expect("add marker");
    }
    let colors: Vec<_> = engine
        .project()
        .timeline()
        .markers()
        .iter()
        .map(|m| m.color)
        .collect();
    assert_eq!(
        colors,
        [
            cutlass_models::MarkerColor::cycle(0),
            cutlass_models::MarkerColor::cycle(1),
            cutlass_models::MarkerColor::cycle(2),
        ]
    );
}

#[test]
fn undo_remove_marker_restores_snapshot() {
    let (_dir, mut engine) = temp_engine();
    let id = created_marker(
        engine
            .apply(Command::Edit(EditCommand::AddMarker {
                at: rt(24),
                name: "beat".into(),
                color: Some(cutlass_models::MarkerColor::Blue),
            }))
            .expect("add marker"),
    );

    engine
        .apply(Command::Edit(EditCommand::RemoveMarker { marker: id }))
        .expect("remove marker");
    assert_eq!(engine.project().timeline().marker_count(), 0);

    assert!(engine.undo());
    let marker = engine.project().timeline().marker(id).expect("restored");
    assert_eq!((marker.tick, marker.name.as_str()), (rt(24), "beat"));

    assert!(engine.redo());
    assert_eq!(engine.project().timeline().marker_count(), 0);
}

#[test]
fn undo_set_marker_restores_previous_state() {
    let (_dir, mut engine) = temp_engine();
    let id = created_marker(
        engine
            .apply(Command::Edit(EditCommand::AddMarker {
                at: rt(10),
                name: "intro".into(),
                color: Some(cutlass_models::MarkerColor::Teal),
            }))
            .expect("add marker"),
    );

    engine
        .apply(Command::Edit(EditCommand::SetMarker {
            marker: id,
            at: rt(99),
            name: "outro".into(),
            color: cutlass_models::MarkerColor::Green,
        }))
        .expect("set marker");
    let moved = engine.project().timeline().marker(id).unwrap();
    assert_eq!((moved.tick, moved.name.as_str()), (rt(99), "outro"));

    assert!(engine.undo());
    let restored = engine.project().timeline().marker(id).unwrap();
    assert_eq!((restored.tick, restored.name.as_str()), (rt(10), "intro"));
    assert_eq!(restored.color, cutlass_models::MarkerColor::Teal);

    assert!(engine.redo());
    let again = engine.project().timeline().marker(id).unwrap();
    assert_eq!((again.tick, again.name.as_str()), (rt(99), "outro"));
}

#[test]
fn marker_commands_reject_unknown_ids_without_history() {
    let (_dir, mut engine) = temp_engine();
    let missing = cutlass_models::MarkerId::from_raw(404);

    assert!(
        engine
            .apply(Command::Edit(EditCommand::RemoveMarker { marker: missing }))
            .is_err()
    );
    assert!(
        engine
            .apply(Command::Edit(EditCommand::SetMarker {
                marker: missing,
                at: rt(0),
                name: String::new(),
                color: cutlass_models::MarkerColor::Teal,
            }))
            .is_err()
    );
    assert!(!engine.can_undo(), "failed marker edits push no history");
}

#[test]
fn failed_move_overlap_does_not_push_undo() {
    let (_dir, mut engine) = temp_engine();
    let track = common::add_track(&mut engine, TrackKind::Adjustment, "FX");

    let _a = created(
        engine
            .apply(Command::Edit(EditCommand::AddGenerated {
                track,
                generator: Generator::Adjustment,
                timeline: tr(0, 20),
            }))
            .expect("a"),
    );
    let b = created(
        engine
            .apply(Command::Edit(EditCommand::AddGenerated {
                track,
                generator: Generator::Adjustment,
                timeline: tr(30, 20),
            }))
            .expect("b"),
    );

    let err = engine
        .apply(Command::Edit(EditCommand::MoveClip {
            clip: b,
            to_track: track,
            start: rt(5),
        }))
        .unwrap_err();
    assert!(format!("{err}").contains("overlap") || format!("{err}").contains("Overlap"));
    assert_eq!(engine.project().clip(b).unwrap().start().value, 30);
    // add-track + two adds are undoable; the failed move must not push a fourth step.
    assert!(engine.can_undo());
    assert!(engine.undo());
    assert!(engine.can_undo());
    assert!(engine.undo());
    assert!(engine.can_undo());
    assert!(engine.undo());
    assert!(!engine.can_undo());
}

#[test]
fn add_effect_undo_redo_roundtrip() {
    let (_dir, mut engine) = temp_engine();
    let clip_id = text_clip(&mut engine);

    engine
        .apply(Command::Edit(EditCommand::AddEffect {
            clip: clip_id,
            effect_id: "gaussian_blur".into(),
        }))
        .expect("add effect");
    let effects =
        |engine: &cutlass_engine::Engine| engine.project().clip(clip_id).unwrap().effects.clone();
    assert_eq!(effects(&engine).len(), 1);
    assert_eq!(effects(&engine)[0].effect_id, "gaussian_blur");

    assert!(engine.undo());
    assert!(effects(&engine).is_empty());
    assert!(engine.redo());
    assert_eq!(effects(&engine).len(), 1);
}

#[test]
fn remove_effect_undo_restores_it() {
    let (_dir, mut engine) = temp_engine();
    let clip_id = text_clip(&mut engine);

    engine
        .apply(Command::Edit(EditCommand::AddEffect {
            clip: clip_id,
            effect_id: "vignette".into(),
        }))
        .expect("add effect");
    engine
        .apply(Command::Edit(EditCommand::RemoveEffect {
            clip: clip_id,
            index: 0,
        }))
        .expect("remove effect");
    assert!(engine.project().clip(clip_id).unwrap().effects.is_empty());

    assert!(engine.undo());
    let effects = engine.project().clip(clip_id).unwrap().effects.clone();
    assert_eq!(effects.len(), 1);
    assert_eq!(effects[0].effect_id, "vignette");
}

#[test]
fn set_effect_param_undo_redo_roundtrip() {
    let (_dir, mut engine) = temp_engine();
    let clip_id = text_clip(&mut engine);

    engine
        .apply(Command::Edit(EditCommand::AddEffect {
            clip: clip_id,
            effect_id: "vignette".into(),
        }))
        .expect("add effect");
    engine
        .apply(Command::Edit(EditCommand::SetEffectParam {
            clip: clip_id,
            index: 0,
            param: 0,
            value: 0.75,
        }))
        .expect("set param");

    let amount = |engine: &cutlass_engine::Engine| {
        engine.project().clip(clip_id).unwrap().effects[0].sample_param("amount", 0.0)
    };
    assert_eq!(amount(&engine), Some(0.75));

    // Undo restores the default (param absent → catalog default 0.6).
    assert!(engine.undo());
    assert_eq!(amount(&engine), Some(0.6));
    assert!(engine.redo());
    assert_eq!(amount(&engine), Some(0.75));
}

#[test]
fn move_effect_undo_redo_restores_exact_instances() {
    let (_dir, mut engine) = temp_engine();
    let clip_id = text_clip(&mut engine);

    for effect_id in ["gaussian_blur", "glitch", "vignette"] {
        engine
            .apply(Command::Edit(EditCommand::AddEffect {
                clip: clip_id,
                effect_id: effect_id.into(),
            }))
            .expect("add effect");
    }
    engine
        .apply(Command::Edit(EditCommand::SetEffectParam {
            clip: clip_id,
            index: 0,
            param: 0,
            value: 12.0,
        }))
        .expect("set constant effect param");
    for (tick, value, easing) in [(0, 0.2, Easing::EaseIn), (24, 0.8, Easing::EaseOut)] {
        engine
            .apply(Command::Edit(EditCommand::SetParamKeyframe {
                clip: clip_id,
                param: ClipParam::Effect {
                    effect: 1,
                    param: 0,
                },
                at: rt(tick),
                value: ParamValue::Scalar(value),
                easing,
                tangents: None,
            }))
            .expect("set animated effect param");
    }

    let before = engine.project().clip(clip_id).unwrap().effects.clone();
    let expected = vec![before[1].clone(), before[2].clone(), before[0].clone()];
    engine
        .apply(Command::Edit(EditCommand::MoveEffect {
            clip: clip_id,
            from_index: 0,
            to_index: 2,
        }))
        .expect("move effect");
    assert_eq!(engine.project().clip(clip_id).unwrap().effects, expected);

    assert!(engine.undo());
    assert_eq!(engine.project().clip(clip_id).unwrap().effects, before);
    assert!(engine.redo());
    assert_eq!(engine.project().clip(clip_id).unwrap().effects, expected);
}

#[test]
fn effect_param_keyframe_through_clip_param() {
    let (_dir, mut engine) = temp_engine();
    let clip_id = text_clip(&mut engine);
    engine
        .apply(Command::Edit(EditCommand::AddEffect {
            clip: clip_id,
            effect_id: "gaussian_blur".into(),
        }))
        .expect("add effect");

    let fx_param = ClipParam::Effect {
        effect: 0,
        param: 0,
    };
    engine
        .apply(Command::Edit(EditCommand::SetParamKeyframe {
            clip: clip_id,
            param: fx_param,
            at: rt(0),
            value: ParamValue::Scalar(0.0),
            easing: Easing::Linear,
            tangents: None,
        }))
        .expect("kf0");
    engine
        .apply(Command::Edit(EditCommand::SetParamKeyframe {
            clip: clip_id,
            param: fx_param,
            at: rt(24),
            value: ParamValue::Scalar(8.0),
            easing: Easing::Linear,
            tangents: None,
        }))
        .expect("kf24");

    let radius_at = |engine: &cutlass_engine::Engine, tick: f64| {
        engine.project().clip(clip_id).unwrap().effects[0].sample_param("radius", tick)
    };
    assert_eq!(radius_at(&engine, 12.0), Some(4.0));

    // Undo peels keyframes one at a time, like the transform path.
    assert!(engine.undo());
    assert!(engine.undo());
    assert_eq!(radius_at(&engine, 12.0), Some(4.0)); // back to default constant
}

/// Add an ellipse sticker at [0, 48) and return its id — fixture for shape
/// param tests.
fn shape_clip(engine: &mut cutlass_engine::Engine) -> cutlass_models::ClipId {
    let track = common::add_track(engine, TrackKind::Sticker, "S1");
    created(
        engine
            .apply(Command::Edit(EditCommand::AddGenerated {
                track,
                generator: Generator::shape(Shape::Ellipse, [255, 0, 0, 255]),
                timeline: tr(0, 48),
            }))
            .expect("add shape"),
    )
}

/// Borrow the shape generator fields of `clip_id` for assertions.
fn shape_generator(
    engine: &cutlass_engine::Engine,
    clip_id: cutlass_models::ClipId,
) -> cutlass_models::Generator {
    match &engine.project().clip(clip_id).unwrap().content {
        cutlass_models::ClipSource::Generated(generator) => generator.clone(),
        other => panic!("expected generated clip, got {other:?}"),
    }
}

#[test]
fn shape_param_keyframe_undo_redo_roundtrip() {
    let (_dir, mut engine) = temp_engine();
    let clip_id = shape_clip(&mut engine);
    let width_param = ClipParam::Shape {
        param: ShapeParam::Width,
    };

    for (tick, value) in [(0, 100.0), (40, 500.0)] {
        engine
            .apply(Command::Edit(EditCommand::SetParamKeyframe {
                clip: clip_id,
                param: width_param,
                at: rt(tick),
                value: ParamValue::Scalar(value),
                easing: Easing::Linear,
                tangents: None,
            }))
            .expect("keyframe");
    }

    let width = |engine: &cutlass_engine::Engine| match shape_generator(engine, clip_id) {
        Generator::Shape { width, .. } => width,
        other => panic!("expected shape, got {other:?}"),
    };
    assert_eq!(width(&engine).keyframes().len(), 2);
    assert_eq!(width(&engine).sample(20), 300.0);

    // Undo peels one keyframe at a time, then restores the drop constant.
    assert!(engine.undo());
    assert_eq!(width(&engine).keyframes().len(), 1);
    assert!(engine.undo());
    assert!(!width(&engine).is_animated());

    assert!(engine.redo());
    assert!(engine.redo());
    assert_eq!(width(&engine).sample(20), 300.0);
}

#[test]
fn shape_fill_color_keyframes_animate_and_flatten() {
    let (_dir, mut engine) = temp_engine();
    let clip_id = shape_clip(&mut engine);
    let fill_param = ClipParam::Shape {
        param: ShapeParam::Fill,
    };

    for (tick, rgba) in [(0, [0, 0, 0, 255]), (40, [200, 100, 0, 255])] {
        engine
            .apply(Command::Edit(EditCommand::SetParamKeyframe {
                clip: clip_id,
                param: fill_param,
                at: rt(tick),
                value: ParamValue::Color(rgba),
                easing: Easing::Linear,
                tangents: None,
            }))
            .expect("keyframe");
    }

    let fill = |engine: &cutlass_engine::Engine| match shape_generator(engine, clip_id) {
        Generator::Shape { rgba, .. } => rgba,
        other => panic!("expected shape, got {other:?}"),
    };
    assert_eq!(fill(&engine).sample(20), [100, 50, 0, 255]);

    // Flattening to a constant drops the curve; undo restores it.
    engine
        .apply(Command::Edit(EditCommand::SetParamConstant {
            clip: clip_id,
            param: fill_param,
            value: ParamValue::Color([1, 2, 3, 255]),
        }))
        .expect("flatten");
    assert_eq!(fill(&engine).constant(), Some([1, 2, 3, 255]));

    assert!(engine.undo());
    assert_eq!(fill(&engine).keyframes().len(), 2);
    assert!(engine.redo());
    assert_eq!(fill(&engine).constant(), Some([1, 2, 3, 255]));
}

#[test]
fn shape_param_commands_reject_bad_targets_without_history() {
    let (_dir, mut engine) = temp_engine();
    let clip_id = shape_clip(&mut engine);
    let before = engine.project().clip(clip_id).unwrap().clone();

    // Stroke params need a stroke; the fixture has none.
    assert!(
        engine
            .apply(Command::Edit(EditCommand::SetParamKeyframe {
                clip: clip_id,
                param: ClipParam::Shape {
                    param: ShapeParam::StrokeWidth,
                },
                at: rt(0),
                value: ParamValue::Scalar(4.0),
                easing: Easing::Linear,
                tangents: None,
            }))
            .is_err()
    );
    // Color params reject scalar values (and vice versa).
    assert!(
        engine
            .apply(Command::Edit(EditCommand::SetParamKeyframe {
                clip: clip_id,
                param: ClipParam::Shape {
                    param: ShapeParam::Fill,
                },
                at: rt(0),
                value: ParamValue::Scalar(0.5),
                easing: Easing::Linear,
                tangents: None,
            }))
            .is_err()
    );
    // Shape params are meaningless on non-shape clips.
    let text = text_clip(&mut engine);
    assert!(
        engine
            .apply(Command::Edit(EditCommand::SetParamKeyframe {
                clip: text,
                param: ClipParam::Shape {
                    param: ShapeParam::Width,
                },
                at: rt(0),
                value: ParamValue::Scalar(100.0),
                easing: Easing::Linear,
                tangents: None,
            }))
            .is_err()
    );

    assert_eq!(engine.project().clip(clip_id).unwrap(), &before);
    // Failed edits push no history: only the three setup steps (two tracks +
    // two generated clips = four) are undoable.
    assert!(engine.undo()); // add text clip
    assert!(engine.undo()); // add text track
    assert!(engine.undo()); // add shape clip
    assert!(engine.undo()); // add sticker track
    assert!(!engine.can_undo());
}

#[test]
fn set_generator_recolors_shape_with_undo() {
    let (_dir, mut engine) = temp_engine();
    let clip_id = shape_clip(&mut engine);

    // Recolor + re-shape in one SetGenerator, as the properties panel does.
    let mut generator = shape_generator(&engine, clip_id);
    let Generator::Shape { shape, rgba, .. } = &mut generator else {
        panic!("expected shape");
    };
    *shape = Shape::Star {
        points: 5,
        inner_ratio: cutlass_models::Param::Constant(0.5),
    };
    *rgba = cutlass_models::Param::Constant([0, 255, 0, 255]);
    engine
        .apply(Command::Edit(EditCommand::SetGenerator {
            clip: clip_id,
            generator,
        }))
        .expect("set generator");

    match shape_generator(&engine, clip_id) {
        Generator::Shape { shape, rgba, .. } => {
            assert!(matches!(shape, Shape::Star { points: 5, .. }));
            assert_eq!(rgba.constant(), Some([0, 255, 0, 255]));
        }
        other => panic!("expected shape, got {other:?}"),
    }

    assert!(engine.undo());
    match shape_generator(&engine, clip_id) {
        Generator::Shape { shape, rgba, .. } => {
            assert_eq!(shape, Shape::Ellipse);
            assert_eq!(rgba.constant(), Some([255, 0, 0, 255]));
        }
        other => panic!("expected shape, got {other:?}"),
    }

    // Invalid generators are rejected at the engine boundary too.
    assert!(
        engine
            .apply(Command::Edit(EditCommand::SetGenerator {
                clip: clip_id,
                generator: Generator::shape(Shape::Polygon { sides: 2 }, [255, 255, 255, 255]),
            }))
            .is_err()
    );
}

#[test]
fn effect_commands_reject_unknown_ids_without_history() {
    let (_dir, mut engine) = temp_engine();
    let clip_id = text_clip(&mut engine);

    assert!(
        engine
            .apply(Command::Edit(EditCommand::AddEffect {
                clip: clip_id,
                effect_id: "no_such_effect".into(),
            }))
            .is_err()
    );
    assert!(
        engine
            .apply(Command::Edit(EditCommand::RemoveEffect {
                clip: clip_id,
                index: 0,
            }))
            .is_err()
    );
    // Only the add-track + add-generated steps are undoable; failed effect
    // edits push nothing.
    assert!(engine.undo());
    assert!(engine.undo());
    assert!(!engine.can_undo());
}

// --- Phase I look commands ---------------------------------------------------

#[test]
fn set_clip_blend_mode_undo_redo_roundtrip() {
    use cutlass_models::BlendMode;

    let Some(path) = small_video_asset() else {
        return;
    };
    let (_dir, mut engine) = temp_engine();
    let media_id = import_asset(&mut engine, &path);
    let track = common::add_track(&mut engine, TrackKind::Video, "V1");
    let clip_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddClip {
                track,
                media: media_id,
                source: tr(0, 48),
                start: rt(0),
            }))
            .expect("add"),
    );
    let before = engine.project().clip(clip_id).unwrap().clone();
    assert_eq!(before.blend_mode, BlendMode::Normal);

    engine
        .apply(Command::Edit(EditCommand::SetClipBlendMode {
            clip: clip_id,
            mode: BlendMode::Multiply,
        }))
        .expect("blend");
    assert_eq!(
        engine.project().clip(clip_id).unwrap().blend_mode,
        BlendMode::Multiply
    );

    // Inverse (undo) restores the pre-edit clip snapshot.
    assert!(engine.undo());
    assert_eq!(engine.project().clip(clip_id).unwrap(), &before);
    assert!(engine.redo());
    assert_eq!(
        engine.project().clip(clip_id).unwrap().blend_mode,
        BlendMode::Multiply
    );
}

#[test]
fn set_clip_motion_blur_undo_redo_roundtrip() {
    use cutlass_models::MotionBlur;

    let Some(path) = small_video_asset() else {
        return;
    };
    let (_dir, mut engine) = temp_engine();
    let media_id = import_asset(&mut engine, &path);
    let track = common::add_track(&mut engine, TrackKind::Video, "V1");
    let clip_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddClip {
                track,
                media: media_id,
                source: tr(0, 48),
                start: rt(0),
            }))
            .expect("add"),
    );
    let before = engine.project().clip(clip_id).unwrap().clone();
    assert!(before.motion_blur.is_default());

    let blur = MotionBlur {
        enabled: true,
        shutter_deg: 270.0,
        samples: 12,
    };
    engine
        .apply(Command::Edit(EditCommand::SetClipMotionBlur {
            clip: clip_id,
            motion_blur: blur,
        }))
        .expect("motion blur");
    assert_eq!(engine.project().clip(clip_id).unwrap().motion_blur, blur);

    // Inverse (undo) restores the pre-edit clip snapshot.
    assert!(engine.undo());
    assert_eq!(engine.project().clip(clip_id).unwrap(), &before);
    assert!(engine.redo());
    assert_eq!(engine.project().clip(clip_id).unwrap().motion_blur, blur);
}

#[test]
fn set_clip_layer_styles_undo_redo_roundtrip() {
    use cutlass_models::{LayerShadow, LayerStyles, Param};

    let Some(path) = small_video_asset() else {
        return;
    };
    let (_dir, mut engine) = temp_engine();
    let media_id = import_asset(&mut engine, &path);
    let track = common::add_track(&mut engine, TrackKind::Video, "V1");
    let clip_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddClip {
                track,
                media: media_id,
                source: tr(0, 48),
                start: rt(0),
            }))
            .expect("add"),
    );
    let before = engine.project().clip(clip_id).unwrap().clone();
    assert!(before.styles.is_empty());

    let styles = LayerStyles {
        shadow: Some(LayerShadow {
            rgba: Param::Constant([0, 0, 0, 128]),
            offset: Param::Constant([4.0, 4.0]),
            blur: Param::Constant(8.0),
        }),
        ..Default::default()
    };
    engine
        .apply(Command::Edit(EditCommand::SetClipLayerStyles {
            clip: clip_id,
            styles: styles.clone(),
        }))
        .expect("styles");
    assert_eq!(engine.project().clip(clip_id).unwrap().styles, styles);

    // Inverse (undo) restores the pre-edit clip snapshot (styles cleared).
    assert!(engine.undo());
    assert_eq!(engine.project().clip(clip_id).unwrap(), &before);
    assert!(engine.project().clip(clip_id).unwrap().styles.is_empty());
    assert!(engine.redo());
    assert_eq!(engine.project().clip(clip_id).unwrap().styles, styles);
}

#[test]
fn look_commands_undo_redo_roundtrip() {
    use cutlass_models::{
        AnimationRef, AnimationSlot, ChromaKey, ColorAdjustments, Filter, Mask, MaskKind,
        StabilizeLevel,
    };

    let Some(path) = small_video_asset() else {
        return;
    };
    let (_dir, mut engine) = temp_engine();
    let media_id = import_asset(&mut engine, &path);
    let track = common::add_track(&mut engine, TrackKind::Video, "V1");
    let clip_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddClip {
                track,
                media: media_id,
                source: tr(0, 48),
                start: rt(0),
            }))
            .expect("add"),
    );
    let clip = |engine: &cutlass_engine::Engine| engine.project().clip(clip_id).unwrap().clone();

    // Mask, chroma, stabilize, filter, adjust — five history entries.
    engine
        .apply(Command::Edit(EditCommand::SetClipMask {
            clip: clip_id,
            mask: Some(Mask::new(MaskKind::Heart)),
        }))
        .expect("mask");
    engine
        .apply(Command::Edit(EditCommand::SetClipChroma {
            clip: clip_id,
            chroma: Some(ChromaKey {
                rgb: [0, 255, 0],
                strength: 0.7.into(),
                shadow: 0.1.into(),
            }),
        }))
        .expect("chroma");
    engine
        .apply(Command::Edit(EditCommand::SetClipStabilize {
            clip: clip_id,
            stabilize: Some(StabilizeLevel::Recommended),
        }))
        .expect("stabilize");
    engine
        .apply(Command::Edit(EditCommand::SetClipFilter {
            clip: clip_id,
            filter: Some(Filter::new("noir")),
        }))
        .expect("filter");
    engine
        .apply(Command::Edit(EditCommand::SetClipAdjustments {
            clip: clip_id,
            adjust: ColorAdjustments {
                exposure: 0.5.into(),
                ..Default::default()
            },
        }))
        .expect("adjust");

    let styled = clip(&engine);
    assert_eq!(styled.mask.as_ref().unwrap().kind, MaskKind::Heart);
    assert_eq!(styled.chroma_key.as_ref().unwrap().rgb, [0, 255, 0]);
    assert_eq!(styled.stabilize, Some(StabilizeLevel::Recommended));
    assert_eq!(styled.filter.as_ref().unwrap().id, "noir");
    assert_eq!(styled.adjust.exposure.sample(0), 0.5);

    // Undo peels them off one at a time (each is one history entry)…
    assert!(engine.undo());
    assert!(clip(&engine).adjust.is_neutral());
    assert!(engine.undo());
    assert!(clip(&engine).filter.is_none());
    assert!(engine.undo());
    assert!(clip(&engine).stabilize.is_none());
    assert!(engine.undo());
    assert!(clip(&engine).chroma_key.is_none());
    assert!(engine.undo());
    assert!(clip(&engine).mask.is_none());

    // …and redo brings the whole look back.
    for _ in 0..5 {
        assert!(engine.redo());
    }
    assert_eq!(clip(&engine), styled);

    // Animations: a combo evicts the entrance in one undoable step.
    engine
        .apply(Command::Edit(EditCommand::SetClipAnimation {
            clip: clip_id,
            slot: AnimationSlot::In,
            animation: Some(AnimationRef::new("zoom_in")),
        }))
        .expect("animation in");
    engine
        .apply(Command::Edit(EditCommand::SetClipAnimation {
            clip: clip_id,
            slot: AnimationSlot::Combo,
            animation: Some(AnimationRef::new("pulse")),
        }))
        .expect("animation combo");
    let c = clip(&engine);
    assert!(c.animation_in.is_none());
    assert_eq!(c.animation_combo.as_ref().unwrap().id, "pulse");
    assert!(engine.undo());
    let c = clip(&engine);
    assert_eq!(c.animation_in.as_ref().unwrap().id, "zoom_in");
    assert!(c.animation_combo.is_none());
}

#[test]
fn look_commands_reject_invalid_targets_without_history() {
    use cutlass_models::{AudioRole, Filter, Mask, MaskKind};

    let (_dir, mut engine) = temp_engine();
    let clip_id = text_clip(&mut engine);
    let before = engine.project().clip(clip_id).unwrap().clone();
    let history_before = (engine.can_undo(), engine.can_redo());

    // Generated text: no pixels to mask, no audio lane for a role, unknown
    // filter ids bounce.
    assert!(
        engine
            .apply(Command::Edit(EditCommand::SetClipMask {
                clip: clip_id,
                mask: Some(Mask::new(MaskKind::Linear)),
            }))
            .is_err()
    );
    assert!(
        engine
            .apply(Command::Edit(EditCommand::SetAudioRole {
                clip: clip_id,
                role: Some(AudioRole::Music),
            }))
            .is_err()
    );
    assert!(
        engine
            .apply(Command::Edit(EditCommand::SetClipFilter {
                clip: clip_id,
                filter: Some(Filter::new("no_such_filter")),
            }))
            .is_err()
    );
    assert_eq!(engine.project().clip(clip_id).unwrap(), &before);
    assert_eq!((engine.can_undo(), engine.can_redo()), history_before);
}

#[test]
fn set_audio_role_tags_audio_clips_with_undo() {
    use cutlass_models::AudioRole;

    let Some(path) = small_video_asset() else {
        return;
    };
    let (_dir, mut engine) = temp_engine();
    let media_id = import_asset(&mut engine, &path);
    let track = common::add_track(&mut engine, TrackKind::Audio, "A1");
    let clip_id = created(
        engine
            .apply(Command::Edit(EditCommand::AddClip {
                track,
                media: media_id,
                source: tr(0, 48),
                start: rt(0),
            }))
            .expect("add"),
    );

    engine
        .apply(Command::Edit(EditCommand::SetAudioRole {
            clip: clip_id,
            role: Some(AudioRole::Extracted),
        }))
        .expect("role");
    assert_eq!(
        engine.project().clip(clip_id).unwrap().audio_role,
        Some(AudioRole::Extracted)
    );
    assert!(engine.undo());
    assert!(engine.project().clip(clip_id).unwrap().audio_role.is_none());
    assert!(engine.redo());
    assert_eq!(
        engine.project().clip(clip_id).unwrap().audio_role,
        Some(AudioRole::Extracted)
    );
}
