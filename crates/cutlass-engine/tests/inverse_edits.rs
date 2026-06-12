//! Inverse undo for all edit commands.

mod common;

use common::{import_asset, rt, small_video_asset, temp_engine, tr};
use cutlass_commands::{Command, EditCommand, EditOutcome};
use cutlass_engine::ApplyOutcome;
use cutlass_models::{ClipParam, ClipTransform, Easing, Generator, ParamValue, TrackKind};

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
        engine.project().clip(clip_id).unwrap().timeline.duration.value,
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
        engine.project().clip(clip_id).unwrap().timeline.duration.value,
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
    assert_eq!(
        engine.project().clip(second_id).unwrap().start().value,
        10
    );

    assert!(engine.undo());
    assert!(engine.project().clip(first_id).is_some());
    assert_eq!(
        engine.project().clip(second_id).unwrap().start().value,
        20
    );
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

    assert_ne!(
        engine.project().clip(clip_id).unwrap().timeline,
        before_tl
    );

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
                generator: Generator::SolidColor {
                    rgba: [1, 2, 3, 4],
                },
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

    let content = |engine: &cutlass_engine::Engine| match &engine.project().clip(clip_id).unwrap().content {
        cutlass_models::ClipSource::Generated(Generator::Text { content, .. }) => content.clone(),
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
        size: 120.0,
        fill: [10, 20, 30, 255],
        stroke: Some(cutlass_models::TextStroke {
            rgba: [0, 0, 0, 255],
            width: 8.0,
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

    let style = |engine: &cutlass_engine::Engine| match &engine.project().clip(clip_id).unwrap().content {
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
        scale: 0.5,
        rotation: 45.0,
        opacity: 0.8,
    };
    engine
        .apply(Command::Edit(EditCommand::SetClipTransform {
            clip: clip_id,
            transform: moved,
            at: None,
        }))
        .expect("set transform");

    let transform = |engine: &cutlass_engine::Engine| {
        engine.project().clip(clip_id).unwrap().transform.clone()
    };
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
                    scale: 0.0,
                    ..ClipTransform::IDENTITY
                },
                at: None,
            }))
            .is_err()
    );
    assert!(engine.project().clip(clip_id).unwrap().transform.is_identity());
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
        }))
        .expect("first keyframe");
    engine
        .apply(Command::Edit(EditCommand::SetParamKeyframe {
            clip: clip_id,
            param: ClipParam::Opacity,
            at: rt(24),
            value: ParamValue::Scalar(1.0),
            easing: Easing::EaseInOut,
        }))
        .expect("second keyframe");

    let opacity = |engine: &cutlass_engine::Engine| {
        engine.project().clip(clip_id).unwrap().transform.opacity.clone()
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
        engine.project().clip(clip_id).unwrap().transform.sample(12).opacity,
        0.5
    );
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
        engine.project().clip(clip_id).unwrap().transform.scale.clone()
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
        engine.project().clip(clip_id).unwrap().transform.rotation.clone()
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
            }))
            .is_err()
    );
    assert!(!engine.project().clip(clip_id).unwrap().transform.is_animated());
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
            }))
            .expect("keyframe");
    }

    // A gesture commit at tick 20 (sampled scale 2.0 → user dragged to 2.5).
    engine
        .apply(Command::Edit(EditCommand::SetClipTransform {
            clip: clip_id,
            transform: ClipTransform {
                scale: 2.5,
                ..ClipTransform::IDENTITY
            },
            at: Some(rt(20)),
        }))
        .expect("compose");

    let transform = engine.project().clip(clip_id).unwrap().transform.clone();
    // Scale gained a keyframe; the endpoints survive.
    assert_eq!(transform.scale.keyframes().len(), 3);
    assert_eq!(transform.sample(20).scale, 2.5);
    assert_eq!(transform.sample(0).scale, 1.0);
    assert_eq!(transform.sample(40).scale, 3.0);
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
    assert_eq!(
        engine.project().clip(clip_id).unwrap().timeline,
        tr(12, 20)
    );

    assert!(engine.undo());
    assert_eq!(
        engine.project().clip(clip_id).unwrap().timeline,
        tr(0, 48)
    );

    assert!(engine.redo());
    assert_eq!(
        engine.project().clip(clip_id).unwrap().timeline,
        tr(12, 20)
    );
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
