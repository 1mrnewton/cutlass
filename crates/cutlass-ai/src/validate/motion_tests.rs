//! AI-path motion clamp tests (sibling of the legacy oversized `tests.rs`).

use super::*;
use crate::wire;
use cutlass_models::MediaSource;

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
