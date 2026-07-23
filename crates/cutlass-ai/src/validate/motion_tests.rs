//! AI-path motion clamp tests (sibling of the legacy oversized `tests.rs`).

use super::*;
use crate::wire;
use cutlass_models::{ClipId, ClipParam, CropRect, Easing, MediaSource, ParamValue};

const R24: Rational = Rational::FPS_24;

fn fixture() -> (Project, u64) {
    let mut project = Project::new("motion-clamp", R24);
    let media = project.add_media(MediaSource::new(
        "/tmp/motion-clamp.mp4",
        1920,
        1080,
        R24,
        60 * 24,
        true,
    ));
    let video = project.add_track(TrackKind::Video, "V1");
    let clip = project
        .add_clip(
            video,
            media,
            TimeRange::at_rate(0, 240, R24),
            RationalTime::new(0, R24),
        )
        .unwrap();
    (project, clip.raw())
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

fn transform(
    clip: u64,
    position_x: Option<f64>,
    position_y: Option<f64>,
    anchor_x: Option<f64>,
    anchor_y: Option<f64>,
    scale: Option<wire::WireScale>,
) -> WireCommand {
    WireCommand::SetClipTransform(wire::SetClipTransform {
        clip,
        position_x,
        position_y,
        anchor_x,
        anchor_y,
        scale,
        rotation: None,
        opacity: None,
    })
}

#[test]
fn position_outside_bound_teaches_anchor_convention() {
    let (project, clip) = fixture();
    let msg = reject(
        &project,
        transform(clip, Some(1.51), None, None, None, None),
    );
    assert!(msg.contains("outside ±1.5"), "{msg}");
    assert!(msg.contains("anchor offset"), "{msg}");
    assert!(msg.contains("[0,0] = centered"), "{msg}");
}

#[test]
fn position_boundary_1_5_passes() {
    let (project, clip) = fixture();
    let edit = lower(
        &project,
        transform(clip, Some(1.5), Some(-1.5), None, None, None),
    );
    match edit {
        EditCommand::SetClipTransform { transform, .. } => {
            assert_eq!(transform.position, [1.5, -1.5]);
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn position_keyframe_outside_bound_rejected() {
    let (project, clip) = fixture();
    let msg = reject(
        &project,
        WireCommand::SetParamKeyframe(wire::SetParamKeyframe {
            clip,
            param: wire::WireClipParam::Position,
            at: 0.0,
            value: None,
            position: Some([0.0, -2.0]),
            rgba: None,
            rect: None,
            easing: None,
            tangent_out: None,
            tangent_in: None,
        }),
    );
    assert!(msg.contains("outside ±1.5"), "{msg}");
    assert!(msg.contains("far off-screen"), "{msg}");
}

#[test]
fn scale_percent_hint_for_150() {
    let (project, clip) = fixture();
    let msg = reject(
        &project,
        transform(
            clip,
            None,
            None,
            None,
            None,
            Some(wire::WireScale::Uniform(150.0)),
        ),
    );
    assert!(msg.contains("exceeds 10"), "{msg}");
    assert!(msg.contains("for 150% send 1.5"), "{msg}");
    assert!(msg.contains("aspect-fit"), "{msg}");
}

#[test]
fn scale_boundary_10_passes() {
    let (project, clip) = fixture();
    let edit = lower(
        &project,
        transform(
            clip,
            None,
            None,
            None,
            None,
            Some(wire::WireScale::Uniform(10.0)),
        ),
    );
    match edit {
        EditCommand::SetClipTransform { transform, .. } => {
            assert_eq!(transform.scale, cutlass_models::Scale2::uniform(10.0));
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn scale_boundary_0_001_passes_and_underflow_is_rejected() {
    let (project, clip) = fixture();
    let edit = lower(
        &project,
        transform(
            clip,
            None,
            None,
            None,
            None,
            Some(wire::WireScale::Uniform(0.001)),
        ),
    );
    match edit {
        EditCommand::SetClipTransform { transform, .. } => {
            assert_eq!(transform.scale, cutlass_models::Scale2::uniform(0.001));
        }
        other => panic!("unexpected {other:?}"),
    }

    let msg = reject(
        &project,
        transform(
            clip,
            None,
            None,
            None,
            None,
            Some(wire::WireScale::Uniform(1e-50)),
        ),
    );
    assert!(msg.contains("below 0.001"), "{msg}");
    assert!(msg.contains("smallest usable scale"), "{msg}");
    assert!(msg.contains("1.0 = 100%"), "{msg}");
}

#[test]
fn scale_keyframe_percent_hint() {
    let (project, clip) = fixture();
    let msg = reject(
        &project,
        WireCommand::SetParamConstant(wire::SetParamConstant {
            clip,
            param: wire::WireClipParam::Scale,
            value: Some(200.0),
            position: None,
            rgba: None,
            rect: None,
        }),
    );
    assert!(msg.contains("for 200% send 2"), "{msg}");
}

#[test]
fn anchor_outside_bound_teaches_content_center() {
    let (project, clip) = fixture();
    let msg = reject(&project, transform(clip, None, None, Some(2.1), None, None));
    assert!(msg.contains("outside [-1, 2]"), "{msg}");
    assert!(msg.contains("[0.5,0.5] = content center"), "{msg}");
}

#[test]
fn anchor_boundaries_pass() {
    let (project, clip) = fixture();
    let edit = lower(
        &project,
        transform(clip, None, None, Some(-1.0), Some(2.0), None),
    );
    match edit {
        EditCommand::SetClipTransform { transform, .. } => {
            assert_eq!(transform.anchor_point, [-1.0, 2.0]);
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn tangent_outside_2_teaches_units() {
    let (project, clip) = fixture();
    let msg = reject(
        &project,
        WireCommand::SetParamKeyframe(wire::SetParamKeyframe {
            clip,
            param: wire::WireClipParam::Position,
            at: 0.0,
            value: None,
            position: Some([0.0, 0.0]),
            rgba: None,
            rect: None,
            easing: None,
            tangent_out: Some([2.01, 0.0]),
            tangent_in: None,
        }),
    );
    assert!(msg.contains("outside ±2"), "{msg}");
    assert!(msg.contains("motion-path"), "{msg}");
}

#[test]
fn tangent_boundary_2_passes() {
    let (project, clip) = fixture();
    let edit = lower(
        &project,
        WireCommand::SetParamKeyframe(wire::SetParamKeyframe {
            clip,
            param: wire::WireClipParam::Position,
            at: 0.0,
            value: None,
            position: Some([0.0, 0.0]),
            rgba: None,
            rect: None,
            easing: None,
            tangent_out: Some([2.0, -2.0]),
            tangent_in: Some([-2.0, 2.0]),
        }),
    );
    match edit {
        EditCommand::SetParamKeyframe {
            tangents: Some(t), ..
        } => {
            assert_eq!(t.out_t, [2.0, -2.0]);
            assert_eq!(t.in_t, [-2.0, 2.0]);
        }
        other => panic!("unexpected {other:?}"),
    }
}

fn animate_position(project: &mut Project, clip: u64) {
    let id = ClipId::from_raw(clip);
    for (tick, xy) in [(0, [0.0, 0.0]), (48, [0.25, 0.0])] {
        project
            .set_param_keyframe(
                id,
                ClipParam::Position,
                RationalTime::new(tick, R24),
                ParamValue::Vec2(xy),
                Easing::Linear,
                None,
            )
            .unwrap();
    }
    assert!(project.clip(id).unwrap().transform.position.is_animated());
}

#[test]
fn set_clip_transform_rejects_animated_position() {
    let (mut project, clip) = fixture();
    animate_position(&mut project, clip);
    let msg = reject(&project, transform(clip, Some(0.1), None, None, None, None));
    assert!(msg.contains("keyframes on position"), "{msg}");
    assert!(msg.contains("set_param_keyframe"), "{msg}");
    assert!(msg.contains("set_param_constant"), "{msg}");
}

#[test]
fn set_clip_transform_rejects_scale_only_when_position_animated() {
    // Lowering uses `at: None` → set_constant on ALL params, so a scale-only
    // edit would also flatten position keyframes. Reject (fallback branch).
    let (mut project, clip) = fixture();
    animate_position(&mut project, clip);
    let before = project
        .clip(ClipId::from_raw(clip))
        .unwrap()
        .transform
        .position
        .keyframes()
        .len();
    let msg = reject(
        &project,
        transform(
            clip,
            None,
            None,
            None,
            None,
            Some(wire::WireScale::Uniform(1.2)),
        ),
    );
    assert!(
        msg.contains("keyframes on position") || msg.contains("keyframed transform params"),
        "{msg}"
    );
    assert!(msg.contains("would erase that animation"), "{msg}");
    // Project unchanged — rejection is validate-only.
    let after = project
        .clip(ClipId::from_raw(clip))
        .unwrap()
        .transform
        .position
        .keyframes()
        .len();
    assert_eq!(before, after);
    assert!(
        project
            .clip(ClipId::from_raw(clip))
            .unwrap()
            .transform
            .position
            .is_animated()
    );
}

#[test]
fn set_clip_transform_still_works_on_non_animated_clip() {
    let (project, clip) = fixture();
    let edit = lower(
        &project,
        transform(
            clip,
            Some(0.1),
            Some(-0.05),
            None,
            None,
            Some(wire::WireScale::Uniform(1.2)),
        ),
    );
    match edit {
        EditCommand::SetClipTransform { transform, at, .. } => {
            assert_eq!(transform.position, [0.1, -0.05]);
            assert_eq!(transform.scale, cutlass_models::Scale2::uniform(1.2));
            assert_eq!(at, None);
        }
        other => panic!("unexpected {other:?}"),
    }
}

fn animate_crop(project: &mut Project, clip: u64) {
    let id = ClipId::from_raw(clip);
    for (tick, crop) in [
        (
            0,
            CropRect {
                x: 0.0,
                y: 0.0,
                w: 1.0,
                h: 1.0,
            },
        ),
        (
            48,
            CropRect {
                x: 0.1,
                y: 0.1,
                w: 0.8,
                h: 0.8,
            },
        ),
    ] {
        project
            .set_param_keyframe(
                id,
                ClipParam::Crop,
                RationalTime::new(tick, R24),
                ParamValue::Rect([crop.x, crop.y, crop.w, crop.h]),
                Easing::Linear,
                None,
            )
            .unwrap();
    }
    assert!(project.clip(id).unwrap().crop.is_animated());
}

#[test]
fn set_clip_crop_rejects_animated_crop_and_preserves_keyframes() {
    let (mut project, clip) = fixture();
    animate_crop(&mut project, clip);
    let before = project
        .clip(ClipId::from_raw(clip))
        .unwrap()
        .crop
        .keyframes()
        .len();
    let msg = reject(
        &project,
        WireCommand::SetClipCrop(wire::SetClipCrop {
            clip,
            left: Some(0.2),
            top: None,
            right: None,
            bottom: None,
            flip_h: None,
            flip_v: None,
        }),
    );
    assert!(msg.contains("keyframes on crop"), "{msg}");
    assert!(msg.contains("set_param_keyframe"), "{msg}");
    assert!(msg.contains("set_param_constant"), "{msg}");
    assert!(msg.contains("would erase that animation"), "{msg}");
    let after = project
        .clip(ClipId::from_raw(clip))
        .unwrap()
        .crop
        .keyframes()
        .len();
    assert_eq!(before, after);
    assert!(
        project
            .clip(ClipId::from_raw(clip))
            .unwrap()
            .crop
            .is_animated()
    );
}

#[test]
fn speed_keyframe_param_is_rejected_with_dedicated_tools_hint() {
    let (project, clip) = fixture();
    for cmd in [
        WireCommand::SetParamKeyframe(wire::SetParamKeyframe {
            clip,
            param: wire::WireClipParam::Speed,
            at: 0.0,
            value: Some(1.5),
            position: None,
            rgba: None,
            rect: None,
            easing: None,
            tangent_out: None,
            tangent_in: None,
        }),
        WireCommand::SetParamConstant(wire::SetParamConstant {
            clip,
            param: wire::WireClipParam::Speed,
            value: Some(1.5),
            position: None,
            rgba: None,
            rect: None,
        }),
        WireCommand::RemoveParamKeyframe(wire::RemoveParamKeyframe {
            clip,
            param: wire::WireClipParam::Speed,
            at: 0.0,
        }),
    ] {
        let msg = reject(&project, cmd);
        assert!(msg.contains("not keyframable"), "{msg}");
        assert!(msg.contains("set_clip_speed"), "{msg}");
        assert!(msg.contains("set_speed_curve"), "{msg}");
    }
}
