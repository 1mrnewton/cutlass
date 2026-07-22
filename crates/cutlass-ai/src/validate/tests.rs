use super::*;
use crate::wire;
use cutlass_models::MediaSource;

const R24: Rational = Rational::FPS_24;

/// 24 fps project: one video track with a 10 s media clip at 0 s, one
/// text track with a title from 2 s to 5 s, and a 60 s media source.
fn fixture() -> (Project, u64, u64, u64, u64, u64) {
    let mut project = Project::new("fixture", R24);
    let media = project.add_media(MediaSource::new(
        "/tmp/agent-fixture.mp4",
        1920,
        1080,
        R24,
        60 * 24,
        true,
    ));
    let video = project.add_track(TrackKind::Video, "V1");
    let text = project.add_track(TrackKind::Text, "Titles");
    let clip = project
        .add_clip(
            video,
            media,
            TimeRange::at_rate(0, 240, R24),
            RationalTime::new(0, R24),
        )
        .unwrap();
    let title = project
        .add_generated(
            text,
            Generator::text("INTRO"),
            TimeRange::at_rate(48, 72, R24),
        )
        .unwrap();
    (
        project,
        media.raw(),
        video.raw(),
        text.raw(),
        clip.raw(),
        title.raw(),
    )
}

fn lower(project: &Project, cmd: WireCommand) -> EditCommand {
    match validate(&cmd, project).expect("command should validate") {
        Command::Edit(edit) => edit,
        other => panic!("expected edit, got {other:?}"),
    }
}

fn reject(project: &Project, cmd: WireCommand) -> String {
    validate(&cmd, project)
        .expect_err("command should be rejected")
        .message
}

#[test]
fn extended_wire_clip_params_lower_to_model_params() {
    let (mut project, _, _, _, clip, title) = fixture();
    project
        .add_effect(ClipId::from_raw(clip), "gaussian_blur")
        .unwrap();
    let sticker_track = project.add_track(TrackKind::Sticker, "Shapes");
    let shape = project
        .add_generated(
            sticker_track,
            Generator::shape(cutlass_models::Shape::Rectangle, [0, 0, 0, 255]),
            TimeRange::at_rate(0, 72, R24),
        )
        .unwrap()
        .raw();

    let cases = [
        (
            clip,
            wire::WireClipParam::AnchorPoint,
            Some([0.25, -0.25]),
            None,
            None,
            ClipParam::AnchorPoint,
            ParamValue::Vec2([0.25, -0.25]),
        ),
        (
            clip,
            wire::WireClipParam::Speed,
            None,
            Some(1.5),
            None,
            ClipParam::Speed,
            ParamValue::Scalar(1.5),
        ),
        (
            clip,
            wire::WireClipParam::Effect {
                index: 0,
                param: "radius".into(),
            },
            None,
            Some(8.0),
            None,
            ClipParam::Effect {
                effect: 0,
                param: 0,
            },
            ParamValue::Scalar(8.0),
        ),
        (
            shape,
            wire::WireClipParam::Shape {
                param: wire::WireShapeParam::Fill,
            },
            None,
            None,
            Some([12, 34, 56, 255]),
            ClipParam::Shape {
                param: cutlass_models::ShapeParam::Fill,
            },
            ParamValue::Color([12, 34, 56, 255]),
        ),
        (
            title,
            wire::WireClipParam::Text {
                param: wire::WireTextParam::Size,
            },
            None,
            Some(72.0),
            None,
            ClipParam::Text {
                param: cutlass_models::TextParam::Size,
            },
            ParamValue::Scalar(72.0),
        ),
        (
            title,
            wire::WireClipParam::Text {
                param: wire::WireTextParam::BackgroundRadius,
            },
            None,
            Some(0.5),
            None,
            ClipParam::Text {
                param: cutlass_models::TextParam::BackgroundRadius,
            },
            ParamValue::Scalar(0.5),
        ),
        (
            title,
            wire::WireClipParam::Text {
                param: wire::WireTextParam::BackgroundColor,
            },
            None,
            None,
            Some([10, 20, 30, 200]),
            ClipParam::Text {
                param: cutlass_models::TextParam::BackgroundColor,
            },
            ParamValue::Color([10, 20, 30, 200]),
        ),
        (
            clip,
            wire::WireClipParam::Look {
                param: wire::WireLookParam::AdjustContrast,
            },
            None,
            Some(0.5),
            None,
            ClipParam::Look {
                param: cutlass_models::LookParam::AdjustContrast,
            },
            ParamValue::Scalar(0.5),
        ),
    ];

    for (target, param, position, value, rgba, expected_param, expected_value) in cases {
        let edit = lower(
            &project,
            WireCommand::SetParamKeyframe(wire::SetParamKeyframe {
                clip: target,
                param,
                at: 2.0,
                value,
                position,
                rgba,
                rect: None,
                easing: Some(wire::WireEasing::EaseOut),
            }),
        );
        assert_eq!(
            edit,
            EditCommand::SetParamKeyframe {
                clip: ClipId::from_raw(target),
                param: expected_param,
                at: RationalTime::new(48, R24),
                value: expected_value,
                easing: Easing::EaseOut,
                tangents: None,
            }
        );
    }
}

#[test]
fn named_and_bezier_easings_lower_on_set_param_keyframe() {
    let (project, _, _, _, clip, _) = fixture();

    let cases = [
        (
            wire::WireEasing::Snappy,
            Easing::from_preset_id("snappy").unwrap(),
        ),
        (
            wire::WireEasing::Overshoot,
            Easing::from_preset_id("overshoot").unwrap(),
        ),
        (
            wire::WireEasing::Anticipate,
            Easing::from_preset_id("anticipate").unwrap(),
        ),
        (wire::WireEasing::Hold, Easing::Hold),
        (
            wire::WireEasing::Bezier {
                points: [0.42, 0.0, 0.58, 1.0],
            },
            Easing::Bezier {
                points: [0.42, 0.0, 0.58, 1.0],
            },
        ),
    ];

    for (wire_easing, expected) in cases {
        let edit = lower(
            &project,
            WireCommand::SetParamKeyframe(wire::SetParamKeyframe {
                clip,
                param: wire::WireClipParam::Opacity,
                at: 1.0,
                value: Some(0.5),
                position: None,
                rgba: None,
                rect: None,
                easing: Some(wire_easing),
            }),
        );
        assert_eq!(
            edit,
            EditCommand::SetParamKeyframe {
                clip: ClipId::from_raw(clip),
                param: ClipParam::Opacity,
                at: RationalTime::new(24, R24),
                value: ParamValue::Scalar(0.5),
                easing: expected,
                tangents: None,
            }
        );
    }

    let err = validate(
        &WireCommand::SetParamKeyframe(wire::SetParamKeyframe {
            clip,
            param: wire::WireClipParam::Opacity,
            at: 1.0,
            value: Some(0.5),
            position: None,
            rgba: None,
            rect: None,
            easing: Some(wire::WireEasing::Bezier {
                points: [1.5, 0.0, 0.5, 1.0],
            }),
        }),
        &project,
    )
    .expect_err("x control point outside 0..=1 must reject");
    assert!(
        err.to_string().contains("invalid easing"),
        "unexpected rejection: {err}"
    );
}

#[test]
fn hold_easing_lowers_from_wire_json() {
    let (project, _, _, _, clip, _) = fixture();
    let cmd: WireCommand = serde_json::from_value(serde_json::json!({
        "command": "set_param_keyframe",
        "clip": clip,
        "param": "opacity",
        "at": 1.0,
        "value": 0.5,
        "easing": "hold",
    }))
    .expect("\"hold\" is a valid wire easing tag");
    assert_eq!(
        lower(&project, cmd),
        EditCommand::SetParamKeyframe {
            clip: ClipId::from_raw(clip),
            param: ClipParam::Opacity,
            at: RationalTime::new(24, R24),
            value: ParamValue::Scalar(0.5),
            easing: Easing::Hold,
            tangents: None,
        }
    );
}

#[test]
fn extended_wire_params_work_for_constant_and_removal() {
    let (project, _, _, _, clip, _) = fixture();

    assert_eq!(
        lower(
            &project,
            WireCommand::SetParamConstant(wire::SetParamConstant {
                clip,
                param: wire::WireClipParam::AnchorPoint,
                value: None,
                position: Some([0.25, 0.5]),
                rgba: None,
                rect: None,
            }),
        ),
        EditCommand::SetParamConstant {
            clip: ClipId::from_raw(clip),
            param: ClipParam::AnchorPoint,
            value: ParamValue::Vec2([0.25, 0.5]),
        }
    );
    assert_eq!(
        lower(
            &project,
            WireCommand::RemoveParamKeyframe(wire::RemoveParamKeyframe {
                clip,
                param: wire::WireClipParam::Speed,
                at: 1.0,
            }),
        ),
        EditCommand::RemoveParamKeyframe {
            clip: ClipId::from_raw(clip),
            param: ClipParam::Speed,
            at: RationalTime::new(24, R24),
        }
    );
}

#[test]
fn add_transition_rejects_canvas_pass_lanes() {
    // Effect/filter/adjustment segments resolve to canvas-wide passes the
    // renderer can't nest inside a transition; the agent gets a rejection
    // instead of a downstream engine error.
    let (mut project, ..) = fixture();
    let lane = project.add_track(TrackKind::Adjustment, "FX");
    let left = project
        .add_generated(lane, Generator::Adjustment, TimeRange::at_rate(0, 24, R24))
        .unwrap();
    project
        .add_generated(lane, Generator::Adjustment, TimeRange::at_rate(24, 24, R24))
        .unwrap();

    let msg = reject(
        &project,
        WireCommand::AddTransition(wire::AddTransition {
            clip: left.raw(),
            transition: "crossfade".into(),
        }),
    );
    assert!(
        msg.contains("Adjustment lane"),
        "message should name the lane kind: {msg}"
    );
}

#[test]
fn trim_clip_converts_seconds_to_frame_ticks() {
    let (project, _, _, _, clip, _) = fixture();
    let edit = lower(
        &project,
        WireCommand::TrimClip(wire::TrimClip {
            clip,
            start: 4.0,
            duration: 6.0,
        }),
    );
    assert_eq!(
        edit,
        EditCommand::TrimClip {
            clip: ClipId::from_raw(clip),
            timeline: TimeRange::at_rate(96, 144, R24),
        }
    );
}

#[test]
fn fractional_seconds_snap_to_nearest_frame() {
    let (project, _, _, _, clip, _) = fixture();
    // 1.02 s at 24 fps = 24.48 frames -> 24; duration 0.01 s -> 0.24
    // frames -> clamps to 1 frame.
    let edit = lower(
        &project,
        WireCommand::TrimClip(wire::TrimClip {
            clip,
            start: 1.02,
            duration: 0.01,
        }),
    );
    assert_eq!(
        edit,
        EditCommand::TrimClip {
            clip: ClipId::from_raw(clip),
            timeline: TimeRange::at_rate(24, 1, R24),
        }
    );
}

#[test]
fn add_clip_uses_media_rate_for_source_and_timeline_rate_for_start() {
    let mut project = Project::new("mixed-rates", R24);
    let media = project.add_media(MediaSource::new(
        "/tmp/30fps.mp4",
        1920,
        1080,
        Rational::FPS_30,
        300,
        true,
    ));
    let track = project.add_track(TrackKind::Video, "V1");

    let edit = lower(
        &project,
        WireCommand::AddClip(wire::AddClip {
            track: track.raw(),
            media: media.raw(),
            source_start: 1.0,
            source_duration: 4.0,
            start: 2.0,
        }),
    );
    assert_eq!(
        edit,
        EditCommand::AddClip {
            track,
            media,
            source: TimeRange::at_rate(30, 120, Rational::FPS_30),
            start: RationalTime::new(48, R24),
        }
    );
}

#[test]
fn add_clip_rejects_out_of_bounds_source_with_media_extent() {
    let (project, media, video, _, _, _) = fixture();
    let msg = reject(
        &project,
        WireCommand::AddClip(wire::AddClip {
            track: video,
            media,
            source_start: 55.0,
            source_duration: 10.0,
            start: 0.0,
        }),
    );
    assert!(msg.contains("exceeds media"), "{msg}");
    assert!(msg.contains("60.000s"), "{msg}");
}

#[test]
fn unknown_ids_list_existing_ones() {
    let (project, _, video, text, clip, title) = fixture();

    let msg = reject(
        &project,
        WireCommand::RemoveClip(wire::RemoveClip { clip: 999 }),
    );
    assert!(msg.contains("clip 999 does not exist"), "{msg}");
    assert!(msg.contains(&clip.to_string()), "{msg}");
    assert!(msg.contains(&title.to_string()), "{msg}");

    let msg = reject(
        &project,
        WireCommand::RemoveTrack(wire::RemoveTrack { track: 999 }),
    );
    assert!(msg.contains("track 999 does not exist"), "{msg}");
    assert!(msg.contains(&video.to_string()), "{msg}");
    assert!(msg.contains(&text.to_string()), "{msg}");

    let msg = reject(
        &project,
        WireCommand::AddClip(wire::AddClip {
            track: video,
            media: 999,
            source_start: 0.0,
            source_duration: 1.0,
            start: 0.0,
        }),
    );
    assert!(msg.contains("media 999 does not exist"), "{msg}");
}

#[test]
fn remove_track_rejects_the_main_track() {
    let (mut project, _, video, text, _, _) = fixture();

    let msg = reject(
        &project,
        WireCommand::RemoveTrack(wire::RemoveTrack { track: video }),
    );
    assert!(msg.contains("main track"), "{msg}");

    // Non-main lanes stay removable.
    let overlay = project.add_track(TrackKind::Video, "V2");
    lower(
        &project,
        WireCommand::RemoveTrack(wire::RemoveTrack {
            track: overlay.raw(),
        }),
    );
    lower(
        &project,
        WireCommand::RemoveTrack(wire::RemoveTrack { track: text }),
    );
}

#[test]
fn generators_must_match_lane_kind() {
    let (project, _, video, text, _, _) = fixture();

    let msg = reject(
        &project,
        WireCommand::AddGenerated(wire::AddGenerated {
            track: video,
            generator: WireGenerator::Text {
                content: "hi".into(),
            },
            start: 0.0,
            duration: 2.0,
        }),
    );
    assert!(msg.contains("needs a text track"), "{msg}");

    let msg = reject(
        &project,
        WireCommand::AddGenerated(wire::AddGenerated {
            track: text,
            generator: WireGenerator::Solid {
                rgba: [0, 0, 0, 255],
            },
            start: 0.0,
            duration: 2.0,
        }),
    );
    assert!(msg.contains("sticker (overlay) track"), "{msg}");
}

#[test]
fn media_clips_cannot_land_on_generator_lanes() {
    let (project, media, _, text, _, _) = fixture();
    let msg = reject(
        &project,
        WireCommand::AddClip(wire::AddClip {
            track: text,
            media,
            source_start: 0.0,
            source_duration: 1.0,
            start: 0.0,
        }),
    );
    assert!(
        msg.contains("media clips need a video or audio track"),
        "{msg}"
    );
}

#[test]
fn set_generator_preserves_text_style_and_rejects_media_clips() {
    let (mut project, _, _, _, clip, title) = fixture();

    // Give the title a non-default style, then replace its content.
    let styled = Generator::Text {
        content: "INTRO".into(),
        style: cutlass_models::TextStyle {
            size: 120.0.into(),
            ..Default::default()
        },
    };
    project
        .set_generator(ClipId::from_raw(title), styled.clone())
        .unwrap();

    let edit = lower(
        &project,
        WireCommand::SetGenerator(wire::SetGenerator {
            clip: title,
            generator: WireGenerator::Text {
                content: "OUTRO".into(),
            },
        }),
    );
    match edit {
        EditCommand::SetGenerator {
            generator: Generator::Text { content, style },
            ..
        } => {
            assert_eq!(content, "OUTRO");
            assert_eq!(
                style.size.sample(0),
                120.0,
                "existing style must be preserved"
            );
        }
        other => panic!("unexpected lowering: {other:?}"),
    }

    let msg = reject(
        &project,
        WireCommand::SetGenerator(wire::SetGenerator {
            clip,
            generator: WireGenerator::Text {
                content: "nope".into(),
            },
        }),
    );
    assert!(msg.contains("is a media clip"), "{msg}");
}

#[test]
fn transform_merges_with_current_values() {
    let (mut project, _, _, _, clip, _) = fixture();
    project
        .set_transform(
            ClipId::from_raw(clip),
            ClipTransform {
                position: [0.25, 0.0],
                scale: 0.5.into(),
                rotation: 10.0,
                opacity: 0.8,
                ..ClipTransform::IDENTITY
            },
            None,
        )
        .unwrap();

    let edit = lower(
        &project,
        WireCommand::SetClipTransform(wire::SetClipTransform {
            clip,
            position_x: None,
            position_y: Some(-0.1),
            anchor_x: None,
            anchor_y: None,
            scale: None,
            rotation: None,
            opacity: Some(1.0),
        }),
    );
    assert_eq!(
        edit,
        EditCommand::SetClipTransform {
            clip: ClipId::from_raw(clip),
            transform: ClipTransform {
                position: [0.25, -0.1],
                scale: 0.5.into(),
                rotation: 10.0,
                opacity: 1.0,
                ..ClipTransform::IDENTITY
            },
            at: None,
        }
    );

    let msg = reject(
        &project,
        WireCommand::SetClipTransform(wire::SetClipTransform {
            clip,
            position_x: None,
            position_y: None,
            anchor_x: None,
            anchor_y: None,
            scale: Some(wire::WireScale::Uniform(0.0)),
            rotation: None,
            opacity: None,
        }),
    );
    assert!(msg.contains("invalid transform"), "{msg}");
}

#[test]
fn transform_scale_accepts_legacy_number_and_axes_array() {
    let (project, _, _, _, clip, _) = fixture();

    // Old agent payloads: bare number → uniform Scale2.
    let legacy = WireCommand::from_tool_call(
        "set_clip_transform",
        serde_json::json!({ "clip": clip, "scale": 1.5 }),
    )
    .unwrap();
    let edit = lower(&project, legacy);
    assert_eq!(
        edit,
        EditCommand::SetClipTransform {
            clip: ClipId::from_raw(clip),
            transform: ClipTransform {
                scale: cutlass_models::Scale2::uniform(1.5),
                ..ClipTransform::IDENTITY
            },
            at: None,
        }
    );

    let split = WireCommand::from_tool_call(
        "set_clip_transform",
        serde_json::json!({ "clip": clip, "scale": [2.0, 1.0] }),
    )
    .unwrap();
    let edit = lower(&project, split);
    assert_eq!(
        edit,
        EditCommand::SetClipTransform {
            clip: ClipId::from_raw(clip),
            transform: ClipTransform {
                scale: cutlass_models::Scale2 { x: 2.0, y: 1.0 },
                ..ClipTransform::IDENTITY
            },
            at: None,
        }
    );
}

#[test]
fn scale_keyframe_accepts_scalar_and_position_vec2() {
    let (project, _, _, _, clip, _) = fixture();

    let uniform = lower(
        &project,
        WireCommand::SetParamKeyframe(wire::SetParamKeyframe {
            clip,
            param: wire::WireClipParam::Scale,
            at: 1.0,
            value: Some(2.0),
            position: None,
            rgba: None,
            rect: None,
            easing: None,
        }),
    );
    assert_eq!(
        uniform,
        EditCommand::SetParamKeyframe {
            clip: ClipId::from_raw(clip),
            param: ClipParam::Scale,
            at: RationalTime::new(24, R24),
            value: ParamValue::Scalar(2.0),
            easing: Easing::Linear,
            tangents: None,
        }
    );

    let axes = lower(
        &project,
        WireCommand::SetParamKeyframe(wire::SetParamKeyframe {
            clip,
            param: wire::WireClipParam::Scale,
            at: 1.0,
            value: None,
            position: Some([2.0, 1.0]),
            rgba: None,
            rect: None,
            easing: None,
        }),
    );
    assert_eq!(
        axes,
        EditCommand::SetParamKeyframe {
            clip: ClipId::from_raw(clip),
            param: ClipParam::Scale,
            at: RationalTime::new(24, R24),
            value: ParamValue::Vec2([2.0, 1.0]),
            easing: Easing::Linear,
            tangents: None,
        }
    );
}

#[test]
fn clip_crop_merges_edges_and_rejects_empty_frames() {
    let (mut project, _, _, _, clip, _) = fixture();

    // Fresh clip: edges lower straight into the kept-region rect.
    let edit = lower(
        &project,
        WireCommand::SetClipCrop(wire::SetClipCrop {
            clip,
            left: Some(0.25),
            top: None,
            right: Some(0.25),
            bottom: None,
            flip_h: Some(true),
            flip_v: None,
        }),
    );
    assert_eq!(
        edit,
        EditCommand::SetClipCrop {
            clip: ClipId::from_raw(clip),
            crop: CropRect {
                x: 0.25,
                y: 0.0,
                w: 0.5,
                h: 1.0
            },
            flip_h: true,
            flip_v: false,
            at: None,
        }
    );

    // Omitted fields keep the stored framing: crop the top, keep the
    // earlier horizontal window and flip.
    project
        .set_clip_crop(
            ClipId::from_raw(clip),
            CropRect {
                x: 0.25,
                y: 0.0,
                w: 0.5,
                h: 1.0,
            },
            true,
            false,
            None,
        )
        .unwrap();
    let edit = lower(
        &project,
        WireCommand::SetClipCrop(wire::SetClipCrop {
            clip,
            left: None,
            top: Some(0.1),
            right: None,
            bottom: None,
            flip_h: None,
            flip_v: None,
        }),
    );
    match edit {
        EditCommand::SetClipCrop { crop, flip_h, .. } => {
            assert_eq!(crop.x, 0.25);
            assert_eq!(crop.w, 0.5);
            assert_eq!(crop.y, 0.1);
            assert!((crop.h - 0.9).abs() < 1e-6);
            assert!(flip_h, "stored flip must be kept");
        }
        other => panic!("unexpected lowering: {other:?}"),
    }

    // Edges that eat the whole frame are rejected with a hint.
    let msg = reject(
        &project,
        WireCommand::SetClipCrop(wire::SetClipCrop {
            clip,
            left: Some(0.6),
            top: None,
            right: Some(0.6),
            bottom: None,
            flip_h: None,
            flip_v: None,
        }),
    );
    assert!(msg.contains("leaves no visible frame"), "{msg}");

    // Out-of-range fractions are rejected by name.
    let msg = reject(
        &project,
        WireCommand::SetClipCrop(wire::SetClipCrop {
            clip,
            left: None,
            top: None,
            right: None,
            bottom: Some(1.5),
            flip_h: None,
            flip_v: None,
        }),
    );
    assert!(msg.contains("bottom must be a fraction"), "{msg}");
}

#[test]
fn pan_keyframe_lowers_and_rejects_out_of_range() {
    let (project, _, _, _, clip, title) = fixture();

    let edit = lower(
        &project,
        WireCommand::SetParamKeyframe(wire::SetParamKeyframe {
            clip,
            param: wire::WireClipParam::Pan,
            at: 1.0,
            value: Some(-0.5),
            position: None,
            rgba: None,
            rect: None,
            easing: Some(wire::WireEasing::Linear),
        }),
    );
    assert_eq!(
        edit,
        EditCommand::SetParamKeyframe {
            clip: ClipId::from_raw(clip),
            param: ClipParam::Pan,
            at: RationalTime::new(24, R24),
            value: ParamValue::Scalar(-0.5),
            easing: Easing::Linear,
            tangents: None,
        }
    );

    let edit = lower(
        &project,
        WireCommand::SetParamConstant(wire::SetParamConstant {
            clip,
            param: wire::WireClipParam::Pan,
            value: Some(0.25),
            position: None,
            rgba: None,
            rect: None,
        }),
    );
    assert_eq!(
        edit,
        EditCommand::SetParamConstant {
            clip: ClipId::from_raw(clip),
            param: ClipParam::Pan,
            value: ParamValue::Scalar(0.25),
        }
    );

    // Range + media-backed gating are enforced by the model when the edit
    // lands (same as volume keyframes — wire lower accepts the scalar shape).
    let mut project = project;
    assert!(
        project
            .set_param_constant(
                ClipId::from_raw(clip),
                ClipParam::Pan,
                ParamValue::Scalar(-1.01),
            )
            .is_err()
    );
    assert!(
        project
            .set_param_constant(
                ClipId::from_raw(title),
                ClipParam::Pan,
                ParamValue::Scalar(0.5),
            )
            .is_err()
    );
}

#[test]
fn crop_keyframe_uses_rect_on_wire() {
    let (project, _, _, _, clip, _) = fixture();

    let edit = lower(
        &project,
        WireCommand::SetParamKeyframe(wire::SetParamKeyframe {
            clip,
            param: wire::WireClipParam::Crop,
            at: 1.0,
            value: None,
            position: None,
            rgba: None,
            rect: Some([0.1, 0.2, 0.5, 0.5]),
            easing: Some(wire::WireEasing::Linear),
        }),
    );
    assert_eq!(
        edit,
        EditCommand::SetParamKeyframe {
            clip: ClipId::from_raw(clip),
            param: ClipParam::Crop,
            at: RationalTime::new(24, R24),
            value: ParamValue::Rect([0.1, 0.2, 0.5, 0.5]),
            easing: Easing::Linear,
            tangents: None,
        }
    );

    // Degenerate width rejected by model validation after lowering path.
    let msg = reject(
        &project,
        WireCommand::SetParamKeyframe(wire::SetParamKeyframe {
            clip,
            param: wire::WireClipParam::Crop,
            at: 1.0,
            value: None,
            position: None,
            rgba: None,
            rect: Some([0.0, 0.0, 0.001, 1.0]),
            easing: None,
        }),
    );
    assert!(
        msg.contains("crop") || msg.contains("0.01") || msg.contains("at least"),
        "{msg}"
    );

    // Scalar value rejected for crop.
    let msg = reject(
        &project,
        WireCommand::SetParamKeyframe(wire::SetParamKeyframe {
            clip,
            param: wire::WireClipParam::Crop,
            at: 1.0,
            value: Some(0.5),
            position: None,
            rgba: None,
            rect: None,
            easing: None,
        }),
    );
    assert!(msg.contains("rect"), "{msg}");
}

#[test]
fn clip_crop_rejects_audio_lane_clips() {
    let (mut project, media, _, _, _, _) = fixture();
    let lane = project.add_track(TrackKind::Audio, "A1");
    let audio_clip = project
        .add_clip(
            lane,
            cutlass_models::MediaId::from_raw(media),
            TimeRange::at_rate(0, 240, R24),
            RationalTime::new(0, R24),
        )
        .unwrap();
    let msg = reject(
        &project,
        WireCommand::SetClipCrop(wire::SetClipCrop {
            clip: audio_clip.raw(),
            left: Some(0.2),
            top: None,
            right: None,
            bottom: None,
            flip_h: None,
            flip_v: None,
        }),
    );
    assert!(msg.contains("no frame to crop"), "{msg}");
}

#[test]
fn move_effect_lowers_and_reports_invalid_requests() {
    let (mut project, _, _, _, clip, _) = fixture();
    let clip_id = ClipId::from_raw(clip);
    for effect in ["gaussian_blur", "glitch", "vignette"] {
        project.add_effect(clip_id, effect).unwrap();
    }

    assert_eq!(
        lower(
            &project,
            WireCommand::MoveEffect(wire::MoveEffect {
                clip,
                from_index: 0,
                to_index: 2,
            }),
        ),
        EditCommand::MoveEffect {
            clip: clip_id,
            from_index: 0,
            to_index: 2,
        }
    );

    for command in [
        wire::MoveEffect {
            clip,
            from_index: 3,
            to_index: 0,
        },
        wire::MoveEffect {
            clip,
            from_index: 0,
            to_index: 3,
        },
    ] {
        let msg = reject(&project, WireCommand::MoveEffect(command));
        assert!(msg.contains("chain length 3"), "{msg}");
    }

    let msg = reject(
        &project,
        WireCommand::MoveEffect(wire::MoveEffect {
            clip,
            from_index: 1,
            to_index: 1,
        }),
    );
    assert!(msg.contains("would make no change"), "{msg}");
}

#[test]
fn clip_speed_lowers_to_exact_rationals() {
    let (project, _, _, _, clip, title) = fixture();

    let edit = lower(
        &project,
        WireCommand::SetClipSpeed(wire::SetClipSpeed {
            clip,
            speed: Some(2.0),
            reversed: None,
        }),
    );
    assert_eq!(
        edit,
        EditCommand::SetClipSpeed {
            clip: ClipId::from_raw(clip),
            speed: Rational::new(2, 1),
            reversed: false,
        }
    );

    // Hundredth snapping: 0.5 → 1/2, 0.75 → 3/4, 0.333 → 33/100.
    assert_eq!(rational_speed(0.5).unwrap(), Rational::new(1, 2));
    assert_eq!(rational_speed(0.75).unwrap(), Rational::new(3, 4));
    assert_eq!(rational_speed(0.333).unwrap(), Rational::new(33, 100));

    // Omitted fields keep the clip's current retiming (reverse-only).
    let edit = lower(
        &project,
        WireCommand::SetClipSpeed(wire::SetClipSpeed {
            clip,
            speed: None,
            reversed: Some(true),
        }),
    );
    assert_eq!(
        edit,
        EditCommand::SetClipSpeed {
            clip: ClipId::from_raw(clip),
            speed: Rational::new(1, 1),
            reversed: true,
        }
    );

    // Out-of-range speeds and generated clips are rejected with names.
    let msg = reject(
        &project,
        WireCommand::SetClipSpeed(wire::SetClipSpeed {
            clip,
            speed: Some(0.0),
            reversed: None,
        }),
    );
    assert!(msg.contains("between 0.05 and 100"), "{msg}");
    let msg = reject(
        &project,
        WireCommand::SetClipSpeed(wire::SetClipSpeed {
            clip: title,
            speed: Some(2.0),
            reversed: None,
        }),
    );
    assert!(msg.contains("generated clip"), "{msg}");
}

#[test]
fn clip_pitch_lowers_and_rejects_generated() {
    let (project, _, _, _, clip, title) = fixture();

    let edit = lower(
        &project,
        WireCommand::SetClipPitch(wire::SetClipPitch {
            clip,
            preserve_pitch: false,
        }),
    );
    assert_eq!(
        edit,
        EditCommand::SetClipPitch {
            clip: ClipId::from_raw(clip),
            preserve_pitch: false,
        }
    );

    // Generated clips have no footage to stretch, so pitch is meaningless.
    let msg = reject(
        &project,
        WireCommand::SetClipPitch(wire::SetClipPitch {
            clip: title,
            preserve_pitch: true,
        }),
    );
    assert!(msg.contains("generated clip"), "{msg}");
}

#[test]
fn set_clip_blend_mode_lowers_and_guards() {
    use cutlass_models::BlendMode;

    let (mut project, media, _, _, clip, _) = fixture();
    let edit = lower(
        &project,
        WireCommand::SetClipBlendMode(wire::SetClipBlendMode {
            clip,
            mode: wire::WireBlendMode::Multiply,
        }),
    );
    assert_eq!(
        edit,
        EditCommand::SetClipBlendMode {
            clip: ClipId::from_raw(clip),
            mode: BlendMode::Multiply,
        }
    );

    let lane = project.add_track(TrackKind::Audio, "A1");
    let audio_clip = project
        .add_clip(
            lane,
            cutlass_models::MediaId::from_raw(media),
            TimeRange::at_rate(0, 240, R24),
            RationalTime::new(0, R24),
        )
        .unwrap();
    let msg = reject(
        &project,
        WireCommand::SetClipBlendMode(wire::SetClipBlendMode {
            clip: audio_clip.raw(),
            mode: wire::WireBlendMode::Screen,
        }),
    );
    assert!(msg.contains("visual frame"), "{msg}");

    let err = WireCommand::from_tool_call(
        "set_clip_blend_mode",
        serde_json::json!({ "clip": clip, "mode": "nope" }),
    )
    .unwrap_err();
    assert!(
        err.contains("nope") || err.contains("unknown variant"),
        "{err}"
    );
}

#[test]
fn set_layer_styles_lowers_and_guards() {
    use cutlass_models::{LayerShadow, LayerStyles, Param};

    let (mut project, media, _, _, clip, _) = fixture();
    let edit = lower(
        &project,
        WireCommand::SetClipLayerStyles(wire::SetClipLayerStyles {
            clip,
            styles: wire::WireLayerStyles {
                shadow: Some(wire::WireLayerShadow {
                    rgba: [0, 0, 0, 128],
                    offset: [4.0, 4.0],
                    blur: 8.0,
                }),
                outline: Some(wire::WireLayerOutline {
                    rgba: [255, 255, 255, 255],
                    width: 2.0,
                }),
                ..Default::default()
            },
        }),
    );
    assert_eq!(
        edit,
        EditCommand::SetClipLayerStyles {
            clip: ClipId::from_raw(clip),
            styles: LayerStyles {
                shadow: Some(LayerShadow {
                    rgba: Param::Constant([0, 0, 0, 128]),
                    offset: Param::Constant([4.0, 4.0]),
                    blur: Param::Constant(8.0),
                }),
                outline: Some(cutlass_models::LayerOutline {
                    rgba: Param::Constant([255, 255, 255, 255]),
                    width: Param::Constant(2.0),
                }),
                ..Default::default()
            },
        }
    );

    let lane = project.add_track(TrackKind::Audio, "A1");
    let audio_clip = project
        .add_clip(
            lane,
            cutlass_models::MediaId::from_raw(media),
            TimeRange::at_rate(0, 240, R24),
            RationalTime::new(0, R24),
        )
        .unwrap();
    let msg = reject(
        &project,
        WireCommand::SetClipLayerStyles(wire::SetClipLayerStyles {
            clip: audio_clip.raw(),
            styles: wire::WireLayerStyles {
                glow: Some(wire::WireLayerGlow {
                    rgba: [255, 255, 255, 255],
                    radius: 12.0,
                    intensity: 1.0,
                }),
                ..Default::default()
            },
        }),
    );
    assert!(msg.contains("visual frame"), "{msg}");
}

#[test]
fn style_param_keyframe_lowers_by_value_kind() {
    let (project, _, _, _, clip, _) = fixture();

    let edit = lower(
        &project,
        WireCommand::SetParamKeyframe(wire::SetParamKeyframe {
            clip,
            param: wire::WireClipParam::Style {
                param: wire::WireStyleParam::ShadowBlur,
            },
            at: 1.0,
            value: Some(12.0),
            position: None,
            rgba: None,
            rect: None,
            easing: None,
        }),
    );
    assert_eq!(
        edit,
        EditCommand::SetParamKeyframe {
            clip: ClipId::from_raw(clip),
            param: ClipParam::Style {
                param: cutlass_models::StyleParam::ShadowBlur,
            },
            at: RationalTime::new(24, R24),
            value: ParamValue::Scalar(12.0),
            easing: Easing::Linear,
            tangents: None,
        }
    );

    let msg = reject(
        &project,
        WireCommand::SetParamKeyframe(wire::SetParamKeyframe {
            clip,
            param: wire::WireClipParam::Style {
                param: wire::WireStyleParam::ShadowOffset,
            },
            at: 1.0,
            value: Some(4.0),
            position: None,
            rgba: None,
            rect: None,
            easing: None,
        }),
    );
    assert!(msg.contains("position"), "{msg}");

    let offset = lower(
        &project,
        WireCommand::SetParamKeyframe(wire::SetParamKeyframe {
            clip,
            param: wire::WireClipParam::Style {
                param: wire::WireStyleParam::ShadowOffset,
            },
            at: 1.0,
            value: None,
            position: Some([4.0, -2.0]),
            rgba: None,
            rect: None,
            easing: None,
        }),
    );
    assert_eq!(
        offset,
        EditCommand::SetParamKeyframe {
            clip: ClipId::from_raw(clip),
            param: ClipParam::Style {
                param: cutlass_models::StyleParam::ShadowOffset,
            },
            at: RationalTime::new(24, R24),
            value: ParamValue::Vec2([4.0, -2.0]),
            easing: Easing::Linear,
            tangents: None,
        }
    );

    let msg = reject(
        &project,
        WireCommand::SetParamKeyframe(wire::SetParamKeyframe {
            clip,
            param: wire::WireClipParam::Style {
                param: wire::WireStyleParam::GlowColor,
            },
            at: 1.0,
            value: Some(1.0),
            position: None,
            rgba: None,
            rect: None,
            easing: None,
        }),
    );
    assert!(msg.contains("rgba"), "{msg}");

    let color = lower(
        &project,
        WireCommand::SetParamKeyframe(wire::SetParamKeyframe {
            clip,
            param: wire::WireClipParam::Style {
                param: wire::WireStyleParam::GlowColor,
            },
            at: 1.0,
            value: None,
            position: None,
            rgba: Some([255, 200, 0, 255]),
            rect: None,
            easing: None,
        }),
    );
    assert_eq!(
        color,
        EditCommand::SetParamKeyframe {
            clip: ClipId::from_raw(clip),
            param: ClipParam::Style {
                param: cutlass_models::StyleParam::GlowColor,
            },
            at: RationalTime::new(24, R24),
            value: ParamValue::Color([255, 200, 0, 255]),
            easing: Easing::Linear,
            tangents: None,
        }
    );

    // Tagged nesting on the wire: {"style":{"param":"shadow_blur"}}
    let from_json = WireCommand::from_tool_call(
        "set_param_keyframe",
        serde_json::json!({
            "clip": clip,
            "param": { "style": { "param": "shadow_blur" } },
            "at": 1.0,
            "value": 8.0,
        }),
    )
    .unwrap();
    let WireCommand::SetParamKeyframe(args) = from_json else {
        panic!("expected SetParamKeyframe");
    };
    assert_eq!(
        args.param,
        wire::WireClipParam::Style {
            param: wire::WireStyleParam::ShadowBlur
        }
    );
}

#[test]
fn look_commands_lower_and_guard() {
    let (project, _, _, _, clip, title) = fixture();

    let edit = lower(
        &project,
        WireCommand::SetClipFilter(wire::SetClipFilter {
            clip,
            filter: Some(wire::WireFilter {
                id: "vivid".to_string(),
                intensity: Some(0.9),
            }),
        }),
    );
    match edit {
        EditCommand::SetClipFilter { filter, .. } => {
            let filter = filter.expect("filter set");
            assert_eq!(filter.id, "vivid");
            assert!((filter.intensity.sample_at(0.0) - 0.9).abs() < 1e-6);
        }
        other => panic!("expected SetClipFilter, got {other:?}"),
    }

    let edit = lower(
        &project,
        WireCommand::SetClipAnimation(wire::SetClipAnimation {
            clip,
            slot: wire::WireAnimationSlot::In,
            animation: Some("fade_in".to_string()),
            speed: None,
            intensity: None,
            stagger: None,
        }),
    );
    assert_eq!(
        edit,
        EditCommand::SetClipAnimation {
            clip: ClipId::from_raw(clip),
            slot: AnimationSlot::In,
            animation: Some(AnimationRef::new("fade_in")),
        }
    );

    let msg = reject(
        &project,
        WireCommand::SetClipAnimation(wire::SetClipAnimation {
            clip,
            slot: wire::WireAnimationSlot::Out,
            animation: Some("fade_in".to_string()),
            speed: None,
            intensity: None,
            stagger: None,
        }),
    );
    assert!(msg.contains("does not fit"), "{msg}");

    let msg = reject(
        &project,
        WireCommand::SetClipMask(wire::SetClipMask {
            clip: title,
            mask: Some(wire::WireMask {
                kind: wire::WireMaskKind::Circle,
                feather: None,
                invert: None,
                center: None,
                size: None,
                rotation: None,
                roundness: None,
            }),
        }),
    );
    assert!(msg.contains("generated clip"), "{msg}");
}

#[test]
fn mask_with_geometry_lowers_to_constants() {
    let (project, _, _, _, clip, _) = fixture();

    let edit = lower(
        &project,
        WireCommand::SetClipMask(wire::SetClipMask {
            clip,
            mask: Some(wire::WireMask {
                kind: wire::WireMaskKind::Rectangle,
                feather: Some(0.25),
                invert: Some(true),
                center: Some([0.1, -0.2]),
                size: Some([0.8, 0.6]),
                rotation: Some(15.0),
                roundness: Some(0.3),
            }),
        }),
    );
    match edit {
        EditCommand::SetClipMask {
            mask: Some(mask), ..
        } => {
            assert_eq!(mask.kind, MaskKind::Rectangle);
            assert_eq!(mask.feather, Param::Constant(0.25));
            assert!(mask.invert);
            assert_eq!(mask.center, Param::Constant([0.1, -0.2]));
            assert_eq!(mask.size, Param::Constant([0.8, 0.6]));
            assert_eq!(mask.rotation, Param::Constant(15.0));
            assert_eq!(mask.roundness, Param::Constant(0.3));
        }
        other => panic!("expected SetClipMask with geometry, got {other:?}"),
    }

    let msg = reject(
        &project,
        WireCommand::SetClipMask(wire::SetClipMask {
            clip,
            mask: Some(wire::WireMask {
                kind: wire::WireMaskKind::Circle,
                feather: None,
                invert: None,
                center: None,
                size: Some([0.0, 1.0]),
                rotation: None,
                roundness: None,
            }),
        }),
    );
    assert!(msg.contains("mask size"), "{msg}");
}

#[test]
fn set_clip_adjustments_lowers_new_sliders_and_rejects_ranges() {
    let (project, _, _, _, clip, _) = fixture();

    let edit = lower(
        &project,
        WireCommand::SetClipAdjustments(wire::SetClipAdjustments {
            clip,
            brightness: None,
            contrast: None,
            saturation: None,
            exposure: None,
            temperature: None,
            tint: Some(-0.5),
            hue: Some(1.0),
            highlights: Some(0.25),
            shadows: Some(-0.25),
            sharpness: Some(0.75),
            vignette: Some(0.4),
        }),
    );
    match edit {
        EditCommand::SetClipAdjustments { adjust, .. } => {
            assert_eq!(adjust.tint, (-0.5f32).into());
            assert_eq!(adjust.hue, 1.0.into());
            assert_eq!(adjust.highlights, 0.25.into());
            assert_eq!(adjust.shadows, (-0.25f32).into());
            assert_eq!(adjust.sharpness, 0.75.into());
            assert_eq!(adjust.vignette, 0.4.into());
        }
        other => panic!("expected SetClipAdjustments, got {other:?}"),
    }

    let msg = reject(
        &project,
        WireCommand::SetClipAdjustments(wire::SetClipAdjustments {
            clip,
            brightness: None,
            contrast: None,
            saturation: None,
            exposure: None,
            temperature: None,
            tint: None,
            hue: None,
            highlights: None,
            shadows: None,
            sharpness: Some(-0.5),
            vignette: None,
        }),
    );
    assert!(msg.contains("sharpness"), "{msg}");

    let edit = lower(
        &project,
        WireCommand::SetParamKeyframe(wire::SetParamKeyframe {
            clip,
            param: wire::WireClipParam::Look {
                param: wire::WireLookParam::AdjustHue,
            },
            at: 0.0,
            value: Some(0.5),
            position: None,
            rgba: None,
            rect: None,
            easing: None,
        }),
    );
    assert_eq!(
        edit,
        EditCommand::SetParamKeyframe {
            clip: ClipId::from_raw(clip),
            param: ClipParam::Look {
                param: cutlass_models::LookParam::AdjustHue,
            },
            at: RationalTime::new(0, R24),
            value: ParamValue::Scalar(0.5),
            easing: Easing::Linear,
            tangents: None,
        }
    );
}

#[test]
fn mask_center_keyframe_uses_position_on_masked_clip() {
    let (mut project, _, _, _, clip, _) = fixture();
    project
        .set_clip_mask(ClipId::from_raw(clip), Some(Mask::new(MaskKind::Circle)))
        .unwrap();

    let msg = reject(
        &project,
        WireCommand::SetParamKeyframe(wire::SetParamKeyframe {
            clip,
            param: wire::WireClipParam::Look {
                param: wire::WireLookParam::MaskCenter,
            },
            at: 1.0,
            value: Some(0.5),
            position: None,
            rgba: None,
            rect: None,
            easing: None,
        }),
    );
    assert!(msg.contains("position"), "{msg}");

    let edit = lower(
        &project,
        WireCommand::SetParamKeyframe(wire::SetParamKeyframe {
            clip,
            param: wire::WireClipParam::Look {
                param: wire::WireLookParam::MaskCenter,
            },
            at: 1.0,
            value: None,
            position: Some([0.25, -0.1]),
            rgba: None,
            rect: None,
            easing: None,
        }),
    );
    assert_eq!(
        edit,
        EditCommand::SetParamKeyframe {
            clip: ClipId::from_raw(clip),
            param: ClipParam::Look {
                param: cutlass_models::LookParam::MaskCenter,
            },
            at: RationalTime::new(24, R24),
            value: ParamValue::Vec2([0.25, -0.1]),
            easing: Easing::Linear,
            tangents: None,
        }
    );

    // Tagged nesting on the wire: {"look":{"param":"mask_center"}}
    let from_json = WireCommand::from_tool_call(
        "set_param_keyframe",
        serde_json::json!({
            "clip": clip,
            "param": { "look": { "param": "mask_center" } },
            "at": 1.0,
            "position": [0.25, -0.1],
        }),
    )
    .unwrap();
    let WireCommand::SetParamKeyframe(args) = from_json else {
        panic!("expected SetParamKeyframe");
    };
    assert_eq!(
        args.param,
        wire::WireClipParam::Look {
            param: wire::WireLookParam::MaskCenter
        }
    );
    assert_eq!(args.position, Some([0.25, -0.1]));
}

#[test]
fn extract_audio_lowers_to_explicit_target_and_preflights_failures() {
    let (mut project, media, video_track, _text, video_clip, title) = fixture();
    let audio = project.add_track(TrackKind::Audio, "A1");
    let edit = lower(
        &project,
        WireCommand::ExtractAudio(wire::ExtractAudio {
            clip: video_clip,
            track: audio.raw(),
        }),
    );
    assert_eq!(
        edit,
        EditCommand::ExtractAudio {
            clip: ClipId::from_raw(video_clip),
            to_track: Some(audio),
        }
    );

    let msg = reject(
        &project,
        WireCommand::ExtractAudio(wire::ExtractAudio {
            clip: title,
            track: audio.raw(),
        }),
    );
    assert!(msg.contains("generated"), "{msg}");

    project
        .timeline_mut()
        .track_mut(TrackId::from_raw(video_track))
        .unwrap()
        .locked = true;
    let msg = reject(
        &project,
        WireCommand::ExtractAudio(wire::ExtractAudio {
            clip: video_clip,
            track: audio.raw(),
        }),
    );
    assert!(msg.contains("video track"), "{msg}");
    assert!(msg.contains("locked"), "{msg}");
    project
        .timeline_mut()
        .track_mut(TrackId::from_raw(video_track))
        .unwrap()
        .locked = false;

    project.timeline_mut().track_mut(audio).unwrap().locked = true;
    let msg = reject(
        &project,
        WireCommand::ExtractAudio(wire::ExtractAudio {
            clip: video_clip,
            track: audio.raw(),
        }),
    );
    assert!(msg.contains("locked"), "{msg}");
    project.timeline_mut().track_mut(audio).unwrap().locked = false;

    project
        .add_clip(
            audio,
            cutlass_models::MediaId::from_raw(media),
            TimeRange::at_rate(0, 240, R24),
            RationalTime::new(0, R24),
        )
        .unwrap();
    let msg = reject(
        &project,
        WireCommand::ExtractAudio(wire::ExtractAudio {
            clip: video_clip,
            track: audio.raw(),
        }),
    );
    assert!(msg.contains("exact timeline range"), "{msg}");
    assert!(msg.contains("choose or add a free audio track"), "{msg}");
}

#[test]
fn duplicate_clip_lowers_at_timeline_rate_without_mutating() {
    let (mut project, _, _, _, clip, _) = fixture();
    let destination = project.add_track(TrackKind::Video, "V2");
    let before = serde_json::to_value(&project).unwrap();

    let edit = lower(
        &project,
        WireCommand::DuplicateClip(wire::DuplicateClip {
            clip,
            to_track: destination.raw(),
            start: 12.25,
        }),
    );

    assert_eq!(
        edit,
        EditCommand::DuplicateClip {
            clip: ClipId::from_raw(clip),
            to_track: destination,
            start: RationalTime::new(294, R24),
        }
    );
    assert_eq!(
        serde_json::to_value(&project).unwrap(),
        before,
        "validation must not mutate the project"
    );
}

#[test]
fn duplicate_clip_rejects_unknown_ids_and_invalid_starts() {
    let (project, _, video, _, clip, _) = fixture();

    let msg = reject(
        &project,
        WireCommand::DuplicateClip(wire::DuplicateClip {
            clip: 999,
            to_track: video,
            start: 12.0,
        }),
    );
    assert!(msg.contains("clip 999 does not exist"), "{msg}");

    let msg = reject(
        &project,
        WireCommand::DuplicateClip(wire::DuplicateClip {
            clip,
            to_track: 999,
            start: 12.0,
        }),
    );
    assert!(msg.contains("track 999 does not exist"), "{msg}");

    for start in [f64::NAN, f64::INFINITY] {
        let msg = reject(
            &project,
            WireCommand::DuplicateClip(wire::DuplicateClip {
                clip,
                to_track: video,
                start,
            }),
        );
        assert!(msg.contains("start must be a finite number"), "{msg}");
    }

    let msg = reject(
        &project,
        WireCommand::DuplicateClip(wire::DuplicateClip {
            clip,
            to_track: video,
            start: -0.5,
        }),
    );
    assert!(msg.contains("start must not be negative"), "{msg}");
}

#[test]
fn duplicate_clip_rejects_incompatible_locked_and_overlapping_destinations() {
    let (mut project, media, _, text, clip, _) = fixture();

    let msg = reject(
        &project,
        WireCommand::DuplicateClip(wire::DuplicateClip {
            clip,
            to_track: text,
            start: 12.0,
        }),
    );
    assert!(msg.contains("cannot be duplicated"), "{msg}");
    assert!(msg.contains("text lane"), "{msg}");

    let destination = project.add_track(TrackKind::Video, "V2");
    project
        .timeline_mut()
        .track_mut(destination)
        .unwrap()
        .locked = true;
    let msg = reject(
        &project,
        WireCommand::DuplicateClip(wire::DuplicateClip {
            clip,
            to_track: destination.raw(),
            start: 12.0,
        }),
    );
    assert!(msg.contains("destination track"), "{msg}");
    assert!(msg.contains("locked"), "{msg}");
    assert!(msg.contains("unlock it or choose another"), "{msg}");

    project
        .timeline_mut()
        .track_mut(destination)
        .unwrap()
        .locked = false;
    project
        .add_clip(
            destination,
            MediaId::from_raw(media),
            TimeRange::at_rate(0, 24, R24),
            RationalTime::new(480, R24),
        )
        .unwrap();
    let msg = reject(
        &project,
        WireCommand::DuplicateClip(wire::DuplicateClip {
            clip,
            to_track: destination.raw(),
            start: 12.0,
        }),
    );
    assert!(
        msg.contains("exact destination range 12.000s to 22.000s"),
        "{msg}"
    );
    assert!(msg.contains("10.000s of free space"), "{msg}");
    assert!(msg.contains("does not ripple or search for space"), "{msg}");
}

#[test]
fn extract_audio_rejections_tell_the_model_how_to_recover() {
    let (mut project, _media, _video, _text, video_clip, title) = fixture();
    let audio = project.add_track(TrackKind::Audio, "A1");
    let link = cutlass_models::LinkId::next();
    for raw in [video_clip, title] {
        project
            .timeline_mut()
            .clip_mut(ClipId::from_raw(raw))
            .unwrap()
            .link = Some(link);
    }
    let msg = reject(
        &project,
        WireCommand::ExtractAudio(wire::ExtractAudio {
            clip: video_clip,
            track: audio.raw(),
        }),
    );
    assert!(msg.contains("unlink_clips first"), "{msg}");
    assert!(msg.contains(&title.to_string()), "{msg}");

    let (mut project, media, _video, _text, video_clip, _) = fixture();
    let audio = project.add_track(TrackKind::Audio, "A1");
    let companion = project
        .add_clip(
            audio,
            cutlass_models::MediaId::from_raw(media),
            TimeRange::at_rate(0, 240, R24),
            RationalTime::new(0, R24),
        )
        .unwrap();
    let link = cutlass_models::LinkId::next();
    for id in [ClipId::from_raw(video_clip), companion] {
        project.timeline_mut().clip_mut(id).unwrap().link = Some(link);
    }
    let msg = reject(
        &project,
        WireCommand::ExtractAudio(wire::ExtractAudio {
            clip: video_clip,
            track: audio.raw(),
        }),
    );
    assert!(msg.contains("already has extracted audio"), "{msg}");

    let (mut project, media, _video, _text, video_clip, _) = fixture();
    project
        .media_mut(cutlass_models::MediaId::from_raw(media))
        .unwrap()
        .is_image = true;
    let audio = project.add_track(TrackKind::Audio, "A1");
    let msg = reject(
        &project,
        WireCommand::ExtractAudio(wire::ExtractAudio {
            clip: video_clip,
            track: audio.raw(),
        }),
    );
    assert!(msg.contains("video media"), "{msg}");

    let (mut project, media, _video, _text, video_clip, _) = fixture();
    project
        .media_mut(cutlass_models::MediaId::from_raw(media))
        .unwrap()
        .has_audio = false;
    let audio = project.add_track(TrackKind::Audio, "A1");
    let msg = reject(
        &project,
        WireCommand::ExtractAudio(wire::ExtractAudio {
            clip: video_clip,
            track: audio.raw(),
        }),
    );
    assert!(msg.contains("no audio stream"), "{msg}");
}

#[test]
fn set_denoise_lowers_steers_and_rejects_generated() {
    let (mut project, media, _, _, video_clip, title) = fixture();
    // An audio lane carrying the linked companion of the video clip.
    let lane = project.add_track(TrackKind::Audio, "A1");
    let audio_clip = project
        .add_clip(
            lane,
            cutlass_models::MediaId::from_raw(media),
            TimeRange::at_rate(0, 240, R24),
            RationalTime::new(0, R24),
        )
        .unwrap();
    let link = cutlass_models::LinkId::next();
    for id in [ClipId::from_raw(video_clip), audio_clip] {
        project.timeline_mut().clip_mut(id).unwrap().link = Some(link);
    }

    // An audio-lane target lowers straight through.
    let edit = lower(
        &project,
        WireCommand::SetDenoise(wire::SetDenoise {
            clip: audio_clip.raw(),
            denoise: true,
        }),
    );
    assert_eq!(
        edit,
        EditCommand::SetClipDenoise {
            clip: audio_clip,
            denoise: true,
        }
    );

    // A video-lane target is steered to its linked audio companion.
    let msg = reject(
        &project,
        WireCommand::SetDenoise(wire::SetDenoise {
            clip: video_clip,
            denoise: true,
        }),
    );
    assert!(
        msg.contains(&format!("linked clip {}", audio_clip.raw())),
        "{msg}"
    );

    // A generated clip has no footage to clean.
    let msg = reject(
        &project,
        WireCommand::SetDenoise(wire::SetDenoise {
            clip: title,
            denoise: true,
        }),
    );
    assert!(msg.contains("generated clip"), "{msg}");
}

#[test]
fn clip_audio_lowers_volume_and_fades() {
    let (mut project, media, _, _, video_clip, title) = fixture();
    // An audio lane carrying the linked companion of the video clip.
    let lane = project.add_track(TrackKind::Audio, "A1");
    let audio_clip = project
        .add_clip(
            lane,
            cutlass_models::MediaId::from_raw(media),
            TimeRange::at_rate(0, 240, R24),
            RationalTime::new(0, R24),
        )
        .unwrap();
    let link = cutlass_models::LinkId::next();
    for id in [ClipId::from_raw(video_clip), audio_clip] {
        project.timeline_mut().clip_mut(id).unwrap().link = Some(link);
    }

    // Volume + fades lower to ticks at the timeline rate (1s = 24).
    let edit = lower(
        &project,
        WireCommand::SetClipAudio(wire::SetClipAudio {
            clip: audio_clip.raw(),
            volume: Some(0.5),
            fade_in: Some(1.0),
            fade_out: Some(0.5),
        }),
    );
    assert_eq!(
        edit,
        EditCommand::SetClipAudio {
            clip: audio_clip,
            volume: Some(0.5),
            fade_in: RationalTime::new(24, R24),
            fade_out: RationalTime::new(12, R24),
        }
    );

    // Omitted fields keep the clip's current mix.
    let edit = lower(
        &project,
        WireCommand::SetClipAudio(wire::SetClipAudio {
            clip: audio_clip.raw(),
            volume: Some(0.0),
            fade_in: None,
            fade_out: None,
        }),
    );
    assert_eq!(
        edit,
        EditCommand::SetClipAudio {
            clip: audio_clip,
            volume: Some(0.0),
            fade_in: RationalTime::new(0, R24),
            fade_out: RationalTime::new(0, R24),
        }
    );

    // Omitting volume lowers to `None`, so the clip's gain (a flat level
    // or an M8 envelope) is preserved — only the fades change.
    let edit = lower(
        &project,
        WireCommand::SetClipAudio(wire::SetClipAudio {
            clip: audio_clip.raw(),
            volume: None,
            fade_in: Some(0.5),
            fade_out: None,
        }),
    );
    assert_eq!(
        edit,
        EditCommand::SetClipAudio {
            clip: audio_clip,
            volume: None,
            fade_in: RationalTime::new(12, R24),
            fade_out: RationalTime::new(0, R24),
        }
    );

    // A video-lane target is steered to its linked audio companion.
    let msg = reject(
        &project,
        WireCommand::SetClipAudio(wire::SetClipAudio {
            clip: video_clip,
            volume: Some(0.5),
            fade_in: None,
            fade_out: None,
        }),
    );
    assert!(
        msg.contains(&format!("linked clip {}", audio_clip.raw())),
        "{msg}"
    );

    // Out-of-range volume, over-long fades, generated clips: rejected.
    let msg = reject(
        &project,
        WireCommand::SetClipAudio(wire::SetClipAudio {
            clip: audio_clip.raw(),
            volume: Some(11.0),
            fade_in: None,
            fade_out: None,
        }),
    );
    assert!(msg.contains("between 0 (mute) and 10"), "{msg}");
    let msg = reject(
        &project,
        WireCommand::SetClipAudio(wire::SetClipAudio {
            clip: audio_clip.raw(),
            volume: None,
            fade_in: Some(60.0),
            fade_out: None,
        }),
    );
    assert!(msg.contains("longer than clip"), "{msg}");
    let msg = reject(
        &project,
        WireCommand::SetClipAudio(wire::SetClipAudio {
            clip: title,
            volume: Some(0.5),
            fade_in: None,
            fade_out: None,
        }),
    );
    assert!(msg.contains("generated clip"), "{msg}");
}

#[test]
fn audio_edits_on_a_video_clip_with_sound_target_the_clip_itself() {
    // CapCut keeps a video's audio on the clip — a drop lands a single
    // clip, not a linked audio companion — so volume/fades and denoise on
    // the video clip adjust it directly, with no steering.
    let (project, _, _, _, video_clip, _) = fixture();

    let edit = lower(
        &project,
        WireCommand::SetClipAudio(wire::SetClipAudio {
            clip: video_clip,
            volume: Some(0.5),
            fade_in: None,
            fade_out: None,
        }),
    );
    assert_eq!(
        edit,
        EditCommand::SetClipAudio {
            clip: ClipId::from_raw(video_clip),
            volume: Some(0.5),
            fade_in: RationalTime::new(0, R24),
            fade_out: RationalTime::new(0, R24),
        }
    );

    let edit = lower(
        &project,
        WireCommand::SetDenoise(wire::SetDenoise {
            clip: video_clip,
            denoise: true,
        }),
    );
    assert_eq!(
        edit,
        EditCommand::SetClipDenoise {
            clip: ClipId::from_raw(video_clip),
            denoise: true,
        }
    );
}

#[test]
fn split_outside_clip_names_its_extent() {
    let (project, _, _, _, clip, _) = fixture();
    let msg = reject(
        &project,
        WireCommand::SplitClip(wire::SplitClip { clip, at: 10.0 }),
    );
    assert!(msg.contains("not strictly inside clip"), "{msg}");
    assert!(msg.contains("0.000s"), "{msg}");
    assert!(msg.contains("10.000s"), "{msg}");
}

#[test]
fn shift_rejects_sub_frame_delta_and_link_lists_are_bounded() {
    let (project, _, video, _, clip, _) = fixture();
    let msg = reject(
        &project,
        WireCommand::ShiftClips(wire::ShiftClips {
            track: video,
            from: 0.0,
            delta: 0.001,
        }),
    );
    assert!(msg.contains("rounds to zero frames"), "{msg}");

    let msg = reject(
        &project,
        WireCommand::LinkClips(wire::LinkClips { clips: vec![clip] }),
    );
    assert!(msg.contains("at least two"), "{msg}");

    let msg = reject(
        &project,
        WireCommand::LinkClips(wire::LinkClips {
            clips: vec![clip; MAX_MULTI_CLIP_REFS + 1],
        }),
    );
    assert!(msg.contains("at most 64"), "{msg}");
}

#[test]
fn unlink_one_member_lowers_to_the_complete_group_command() {
    let (mut project, _, _, _, clip, title) = fixture();
    let link = cutlass_models::LinkId::next();
    for id in [clip, title] {
        project
            .timeline_mut()
            .clip_mut(ClipId::from_raw(id))
            .unwrap()
            .link = Some(link);
    }

    let edit = lower(
        &project,
        WireCommand::UnlinkClips(wire::UnlinkClips { clips: vec![title] }),
    );
    assert_eq!(
        edit,
        EditCommand::UnlinkClips {
            clips: vec![ClipId::from_raw(title)]
        }
    );
    assert_eq!(
        project.clip(ClipId::from_raw(clip)).unwrap().link,
        Some(link)
    );
    assert_eq!(
        project.clip(ClipId::from_raw(title)).unwrap().link,
        Some(link)
    );
}

#[test]
fn unlink_rejects_invalid_lists_before_mutation() {
    let (project, _, _, _, clip, title) = fixture();

    let msg = reject(
        &project,
        WireCommand::UnlinkClips(wire::UnlinkClips { clips: vec![] }),
    );
    assert!(msg.contains("at least one"), "{msg}");

    let msg = reject(
        &project,
        WireCommand::UnlinkClips(wire::UnlinkClips {
            clips: vec![clip, clip],
        }),
    );
    assert!(msg.contains("duplicate clip id"), "{msg}");
    assert!(msg.contains(&clip.to_string()), "{msg}");

    let msg = reject(
        &project,
        WireCommand::UnlinkClips(wire::UnlinkClips {
            clips: vec![clip, 999],
        }),
    );
    assert!(msg.contains("clip 999 does not exist"), "{msg}");

    let msg = reject(
        &project,
        WireCommand::UnlinkClips(wire::UnlinkClips {
            clips: vec![clip, title],
        }),
    );
    assert!(
        msg.contains("all referenced clips are already unlinked"),
        "{msg}"
    );

    let msg = reject(
        &project,
        WireCommand::UnlinkClips(wire::UnlinkClips {
            clips: vec![clip; MAX_MULTI_CLIP_REFS + 1],
        }),
    );
    assert!(msg.contains("at most 64"), "{msg}");

    assert_eq!(project.clip(ClipId::from_raw(clip)).unwrap().link, None);
    assert_eq!(project.clip(ClipId::from_raw(title)).unwrap().link, None);
}

#[test]
fn marker_commands_lower_to_engine() {
    let (project, _, _, _, _, _) = fixture();
    let edit = lower(
        &project,
        WireCommand::AddMarker(wire::AddMarker {
            at: 2.0,
            name: Some("intro".into()),
            color: None,
        }),
    );
    assert_eq!(
        edit,
        EditCommand::AddMarker {
            at: RationalTime::new(48, R24),
            name: "intro".into(),
            color: None,
        }
    );

    // Negative positions are rejected before reaching the engine.
    let msg = reject(
        &project,
        WireCommand::AddMarker(wire::AddMarker {
            at: -1.0,
            name: None,
            color: None,
        }),
    );
    assert!(msg.contains("must not be negative"), "{msg}");

    let mut project = project;
    let id = project
        .timeline_mut()
        .add_marker(cutlass_models::Marker::new(
            RationalTime::new(48, R24),
            "mid",
            MarkerColor::Blue,
        ))
        .unwrap();
    let edit = lower(
        &project,
        WireCommand::SetMarker(wire::SetMarker {
            marker: id.raw(),
            at: Some(3.0),
            name: Some("outro".into()),
            color: Some(WireMarkerColor::Green),
        }),
    );
    assert_eq!(
        edit,
        EditCommand::SetMarker {
            marker: id,
            at: RationalTime::new(72, R24),
            name: "outro".into(),
            color: MarkerColor::Green,
        }
    );

    // Omitted fields keep the marker's current state.
    let edit = lower(
        &project,
        WireCommand::SetMarker(wire::SetMarker {
            marker: id.raw(),
            at: None,
            name: None,
            color: None,
        }),
    );
    assert_eq!(
        edit,
        EditCommand::SetMarker {
            marker: id,
            at: RationalTime::new(48, R24),
            name: "mid".into(),
            color: MarkerColor::Blue,
        }
    );

    let msg = reject(
        &project,
        WireCommand::RemoveMarker(wire::RemoveMarker { marker: 404 }),
    );
    assert!(msg.contains("marker 404 does not exist"), "{msg}");
    assert!(msg.contains(&id.raw().to_string()), "{msg}");
}

#[test]
fn set_canvas_lowers_and_keeps_omitted_fields() {
    let (mut project, _, _, _, _, _) = fixture();
    let edit = lower(
        &project,
        WireCommand::SetCanvas(wire::SetCanvas {
            aspect: Some(wire::WireCanvasAspect::Tall9x16),
            background: Some([20, 20, 28]),
        }),
    );
    assert_eq!(
        edit,
        EditCommand::SetCanvas {
            aspect: CanvasAspect::Tall9x16,
            background: [20, 20, 28],
        }
    );

    // Omitted fields keep the project's current canvas settings.
    project
        .timeline_mut()
        .set_canvas(cutlass_models::CanvasSettings {
            aspect: CanvasAspect::Square1x1,
            background: [255, 0, 0],
        });
    let edit = lower(
        &project,
        WireCommand::SetCanvas(wire::SetCanvas {
            aspect: None,
            background: Some([0, 0, 0]),
        }),
    );
    assert_eq!(
        edit,
        EditCommand::SetCanvas {
            aspect: CanvasAspect::Square1x1,
            background: [0, 0, 0],
        }
    );
    let edit = lower(
        &project,
        WireCommand::SetCanvas(wire::SetCanvas {
            aspect: Some(wire::WireCanvasAspect::Auto),
            background: None,
        }),
    );
    assert_eq!(
        edit,
        EditCommand::SetCanvas {
            aspect: CanvasAspect::Auto,
            background: [255, 0, 0],
        }
    );
}

#[test]
fn non_finite_and_negative_times_are_rejected() {
    let (project, _, _, _, clip, _) = fixture();
    let msg = reject(
        &project,
        WireCommand::TrimClip(wire::TrimClip {
            clip,
            start: f64::NAN,
            duration: 1.0,
        }),
    );
    assert!(msg.contains("finite"), "{msg}");

    let msg = reject(
        &project,
        WireCommand::TrimClip(wire::TrimClip {
            clip,
            start: -1.0,
            duration: 1.0,
        }),
    );
    assert!(msg.contains("must not be negative"), "{msg}");
}

#[test]
fn typed_effect_params_accept_rgba_and_position_on_wire() {
    let (mut project, _, _, _, clip, _) = fixture();
    project
        .add_effect(ClipId::from_raw(clip), "duotone")
        .unwrap();
    project
        .add_effect(ClipId::from_raw(clip), "color_overlay")
        .unwrap();

    let kf = lower(
        &project,
        WireCommand::SetParamKeyframe(wire::SetParamKeyframe {
            clip,
            param: wire::WireClipParam::Effect {
                index: 0,
                param: "shadow_color".into(),
            },
            at: 1.0,
            value: None,
            position: None,
            rgba: Some([20, 16, 60, 255]),
            rect: None,
            easing: None,
        }),
    );
    assert_eq!(
        kf,
        EditCommand::SetParamKeyframe {
            clip: ClipId::from_raw(clip),
            param: ClipParam::Effect {
                effect: 0,
                param: 0,
            },
            at: RationalTime::new(24, R24),
            value: ParamValue::Color([20, 16, 60, 255]),
            easing: Easing::Linear,
            tangents: None,
        }
    );

    let msg = reject(
        &project,
        WireCommand::SetParamKeyframe(wire::SetParamKeyframe {
            clip,
            param: wire::WireClipParam::Effect {
                index: 0,
                param: "shadow_color".into(),
            },
            at: 1.0,
            value: Some(0.5),
            position: None,
            rgba: None,
            rect: None,
            easing: None,
        }),
    );
    assert!(msg.contains("color"), "{msg}");
    assert!(msg.contains("rgba"), "{msg}");

    let offset = lower(
        &project,
        WireCommand::SetEffectParam(wire::SetEffectParam {
            clip,
            index: 1,
            param: "offset".into(),
            value: None,
            position: Some([0.25, -0.5]),
            rgba: None,
        }),
    );
    assert_eq!(
        offset,
        EditCommand::SetParamConstant {
            clip: ClipId::from_raw(clip),
            param: ClipParam::Effect {
                effect: 1,
                param: 1,
            },
            value: ParamValue::Vec2([0.25, -0.5]),
        }
    );

    let color_const = lower(
        &project,
        WireCommand::SetEffectParam(wire::SetEffectParam {
            clip,
            index: 0,
            param: "highlight_color".into(),
            value: None,
            position: None,
            rgba: Some([255, 220, 160, 255]),
        }),
    );
    assert_eq!(
        color_const,
        EditCommand::SetParamConstant {
            clip: ClipId::from_raw(clip),
            param: ClipParam::Effect {
                effect: 0,
                param: 1,
            },
            value: ParamValue::Color([255, 220, 160, 255]),
        }
    );

    let msg = reject(
        &project,
        WireCommand::SetEffectParam(wire::SetEffectParam {
            clip,
            index: 0,
            param: "shadow_color".into(),
            value: Some(0.5),
            position: None,
            rgba: None,
        }),
    );
    assert!(msg.contains("color"), "{msg}");
    assert!(msg.contains("rgba"), "{msg}");
}
