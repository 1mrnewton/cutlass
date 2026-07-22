use super::*;
use crate::{Clip, ParamKeyframe, Rational, RationalTime, TimeRange};
use cutlass_models::{Easing, Keyframe, Param, ParamValue, SpatialTangents};
use slint::{ModelRc, SharedString, VecModel};
use std::rc::Rc;

fn rt(value: i32) -> RationalTime {
    RationalTime {
        value,
        rate: Rational { num: 24, den: 1 },
    }
}

fn kf(tick: i32, x: f32, y: f32) -> ParamKeyframe {
    ParamKeyframe {
        tick,
        value_x: x,
        value_y: y,
        easing: 0,
        bez_x1: 0.0,
        bez_y1: 0.0,
        bez_x2: 0.0,
        bez_y2: 0.0,
        has_tangents: false,
        out_tx: 0.0,
        out_ty: 0.0,
        in_tx: 0.0,
        in_ty: 0.0,
    }
}

fn kfs(items: Vec<ParamKeyframe>) -> ModelRc<ParamKeyframe> {
    ModelRc::from(Rc::new(VecModel::from(items)))
}

fn media_clip(id: &str) -> Clip {
    Clip {
        id: SharedString::from(id),
        name: SharedString::from(id),
        timeline_start: rt(0),
        source_range: TimeRange {
            start: rt(0),
            duration: rt(100),
        },
        media_id: SharedString::from("m1"),
        transform_opacity: 1.0,
        transform_scale: 1.0,
        transform_scale_y: 1.0,
        transform_scale_linked: true,
        transform_anchor_x: 0.5,
        transform_anchor_y: 0.5,
        ..Default::default()
    }
}

#[test]
fn animated_opacity_and_position_list_three_channels() {
    let mut clip = media_clip("A");
    clip.kf_opacity = kfs(vec![kf(0, 0.0, 0.0), kf(40, 1.0, 0.0)]);
    clip.kf_position = kfs(vec![kf(0, -0.25, 0.1), kf(40, 0.25, 0.9)]);
    let channels = animated_channels(&clip);
    assert_eq!(channels.len(), 3);
    assert_eq!(channels[0].key.as_str(), "position");
    assert_eq!(channels[0].channel, 0);
    assert_eq!(channels[0].label.as_str(), "Position X");
    assert_eq!(channels[1].key.as_str(), "position");
    assert_eq!(channels[1].channel, 1);
    assert_eq!(channels[1].label.as_str(), "Position Y");
    assert_eq!(channels[2].key.as_str(), "opacity");
    assert_eq!(channels[2].channel, 0);
}

#[test]
fn linear_two_keyframe_path_is_straight_with_matching_endpoints() {
    let param = Param::Keyframed {
        keyframes: vec![
            Keyframe::new(0, 0.0, Easing::Linear),
            Keyframe::new(40, 1.0, Easing::Linear),
        ],
    };
    let geo = build_geometry(&param, 20, 400.0, 180.0, -1);
    assert!(!geo.path_commands.is_empty());
    assert_eq!(geo.dots.len(), 2);
    let a = &geo.dots[0];
    let b = &geo.dots[1];
    let cmds = geo.path_commands.as_str();
    let start = format!("M {:.2} {:.2}", a.px, a.py);
    let end = format!("L {:.2} {:.2}", b.px, b.py);
    assert!(cmds.starts_with(&start), "path {cmds} missing {start}");
    assert!(cmds.ends_with(&end), "path {cmds} missing {end}");
    let mid_v = param.sample_at(20.0);
    assert!((mid_v - 0.5).abs() < 1e-5);
    let (y0, y1) = padded_y_range(0.0, 1.0);
    let y_span = y1 - y0;
    let samples = sample_curve(&param, 0, 40);
    let plot_w = 400.0 - PAD_L - PAD_R;
    for (t, v) in samples {
        let px = PAD_L + ((t - 0.0) / 40.0) as f32 * plot_w;
        let py = PAD_T + (1.0 - (v - y0) / y_span) * (180.0 - PAD_T - PAD_B);
        let frac = (px - a.px) / (b.px - a.px);
        let chord_y = a.py + frac * (b.py - a.py);
        assert!(
            (py - chord_y).abs() < 0.05,
            "sample off chord at t={t}: py={py} chord={chord_y}"
        );
    }
}

#[test]
fn y_range_padding_expands_by_ten_percent() {
    let (lo, hi) = padded_y_range(0.0, 10.0);
    assert!((lo - (-1.0)).abs() < 1e-5);
    assert!((hi - 11.0).abs() < 1e-5);
    let (lo, hi) = padded_y_range(5.0, 5.0);
    assert!(lo < 5.0 && hi > 5.0);
}

#[test]
fn colors_are_not_enumerated_as_channels() {
    let mut clip = media_clip("A");
    clip.kf_text_fill = kfs(vec![kf(0, 0.0, 0.0), kf(10, 1.0, 1.0)]);
    clip.kf_opacity = kfs(vec![kf(0, 0.0, 0.0), kf(10, 1.0, 0.0)]);
    let channels = animated_channels(&clip);
    assert_eq!(channels.len(), 1);
    assert_eq!(channels[0].key.as_str(), "opacity");
}

#[test]
fn drag_commit_same_tick_preserves_easing() {
    let mut clip = media_clip("A");
    let mut a = kf(0, 0.0, 0.0);
    a.easing = 1; // EaseIn
    clip.kf_opacity = kfs(vec![a, kf(40, 1.0, 0.0)]);
    let plan = plan_drag_commit(&clip, "opacity", 0, 0, 0, 0.25).unwrap();
    assert!(!plan.tick_moved);
    assert_eq!(plan.to_tick, 0);
    assert_eq!(plan.easing, Easing::EaseIn);
    assert_eq!(plan.value, ParamValue::Scalar(0.25));
}

#[test]
fn tick_move_commit_flags_remove_plus_set() {
    let mut clip = media_clip("A");
    clip.kf_opacity = kfs(vec![kf(0, 0.0, 0.0), kf(40, 1.0, 0.0)]);
    let plan = plan_drag_commit(&clip, "opacity", 0, 0, 10, 0.1).unwrap();
    assert!(plan.tick_moved);
    assert_eq!((plan.from_tick, plan.to_tick), (0, 10));
    // Clamp keeps the move off the neighbor at 40.
    let param = channel_param(&clip, "opacity", 0).unwrap();
    assert_eq!(edit::clamp_keyframe_tick(&param, &clip, 0, 40), 39);
    assert_eq!(edit::clamp_keyframe_tick(&param, &clip, 0, -5), 0);
}

#[test]
fn insert_uses_sampled_curve_value() {
    let mut clip = media_clip("A");
    clip.kf_opacity = kfs(vec![kf(0, 0.0, 0.0), kf(40, 1.0, 0.0)]);
    let mapping = PlotMapping {
        t_min: 0,
        t_max: 40,
        y0: -0.1,
        y1: 1.1,
        width: 400.0,
        height: 180.0,
    };
    // Midpoint of the plot X → tick 20 → sample 0.5.
    let mid_x = PAD_L + 0.5 * (400.0 - PAD_L - PAD_R);
    let (tick, value) = plan_insert(&clip, "opacity", 0, mid_x, mapping).unwrap();
    assert_eq!(tick, 20);
    assert!((value - 0.5).abs() < 1e-5);
    let plan = plan_insert_commit(&clip, "opacity", 0, tick, value).unwrap();
    assert_eq!(plan.easing, Easing::Linear);
    assert_eq!(plan.value, ParamValue::Scalar(0.5));
}

#[test]
fn delete_last_keyframe_collapses_to_constant() {
    // Mirrors remove_param_keyframe model semantics used by the graph delete.
    let mut param = Param::Keyframed {
        keyframes: vec![
            Keyframe::new(0, 0.25, Easing::Linear),
            Keyframe::new(40, 1.0, Easing::Linear),
        ],
    };
    assert!(param.remove_keyframe(40));
    assert!(param.remove_keyframe(0));
    assert_eq!(param.constant(), Some(0.25));
}

#[test]
fn vec2_channel_drag_preserves_other_axis_and_tangents() {
    let mut clip = media_clip("A");
    let mut a = kf(0, -0.25, 0.5);
    a.has_tangents = true;
    a.out_tx = 0.1;
    a.out_ty = 0.2;
    a.in_tx = -0.1;
    a.in_ty = -0.2;
    a.easing = 2; // EaseOut
    clip.kf_position = kfs(vec![a, kf(40, 0.25, 0.9)]);
    // Drag Position X only.
    let plan = plan_drag_commit(&clip, "position", 0, 0, 0, 0.0).unwrap();
    assert!(!plan.tick_moved);
    assert_eq!(plan.easing, Easing::EaseOut);
    assert_eq!(plan.value, ParamValue::Vec2([0.0, 0.5]));
    assert_eq!(
        plan.tangents,
        Some(SpatialTangents {
            out_t: [0.1, 0.2],
            in_t: [-0.1, -0.2],
        })
    );
}
