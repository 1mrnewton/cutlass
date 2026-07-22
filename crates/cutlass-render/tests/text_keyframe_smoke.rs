//! End-to-end smoke + tolerance: keyframed text style and tunable animation
//! presets through resolve → GPU glyph path.

use cutlass_core::{Rational, RationalTime};
use cutlass_models::{
    AnimationRef, AnimationSlot, ClipParam, Easing, Generator, ParamValue, Project, TextParam,
    TextStyle as ModelTextStyle, TimeRange, TrackKind,
};
use cutlass_render::{Renderer, resolve};

const FPS_24: Rational = Rational::FPS_24;

fn rt(tick: i64) -> RationalTime {
    RationalTime::new(tick, FPS_24)
}

/// Count near-white ink (ignores the opaque canvas clear).
fn ink(img: &cutlass_core::RgbaImage) -> usize {
    img.pixels
        .chunks_exact(4)
        .filter(|p| p[0] > 200 && p[1] > 200 && p[2] > 200 && p[3] > 200)
        .count()
}

#[test]
fn keyframed_text_size_grows_glyph_coverage() {
    let mut project = Project::new("kf-size", FPS_24);
    let track = project.add_track(TrackKind::Text, "T1");
    let clip = project
        .add_generated(
            track,
            Generator::Text {
                content: "Aa".into(),
                style: ModelTextStyle {
                    size: 24.0.into(),
                    fill: [255, 255, 255, 255].into(),
                    ..ModelTextStyle::default()
                },
            },
            TimeRange::at_rate(0, 49, FPS_24),
        )
        .unwrap();
    project
        .set_param_keyframe(
            clip,
            ClipParam::Text {
                param: TextParam::Size,
            },
            rt(0),
            ParamValue::Scalar(24.0),
            Easing::Linear,
            None,
        )
        .unwrap();
    project
        .set_param_keyframe(
            clip,
            ClipParam::Text {
                param: TextParam::Size,
            },
            rt(48),
            ParamValue::Scalar(96.0),
            Easing::from_preset_id("snappy").unwrap(),
            None,
        )
        .unwrap();

    let mut renderer = match Renderer::new_headless() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("skipping keyframed size smoke: no GPU ({e})");
            return;
        }
    };

    let small = renderer
        .render_frame(&project, rt(0))
        .expect("render small size");
    let mid = renderer
        .render_frame(&project, rt(24))
        .expect("render mid size");
    let large = renderer
        .render_frame(&project, rt(48))
        .expect("render large size");

    let c0 = ink(&small);
    let c1 = ink(&mid);
    let c2 = ink(&large);
    assert!(c0 > 0, "start keyframe should ink something");
    assert!(
        c1 > c0,
        "mid ramp should cover more than start ({c1} <= {c0})"
    );
    assert!(
        c2 >= c1,
        "end keyframe should not shrink coverage ({c2} < {c1})"
    );

    // Resolve samples the curve (tolerance: mid ≈ halfway for linear segment
    // before snappy settle — we only require monotonic size growth).
    let scene0 = resolve(&project, rt(0)).expect("resolve t0");
    let scene1 = resolve(&project, rt(24)).expect("resolve t24");
    let scene2 = resolve(&project, rt(48)).expect("resolve t48");
    let size_at = |scene: &cutlass_render::Scene| -> f32 {
        scene
            .layers
            .iter()
            .find_map(|layer| match &layer.source {
                cutlass_render::LayerSource::Text { style, .. } => Some(style.font_size),
                _ => None,
            })
            .expect("text layer")
    };
    let s0 = size_at(&scene0);
    let s1 = size_at(&scene1);
    let s2 = size_at(&scene2);
    assert!((s0 - 24.0).abs() < 0.5, "t0 size {s0}");
    assert!((s2 - 96.0).abs() < 0.5, "t48 size {s2}");
    assert!(
        s1 > s0 && s1 < s2,
        "mid size {s1} not between {s0} and {s2}"
    );
}

#[test]
fn tunable_typewriter_params_still_render() {
    let mut project = Project::new("anim-params", FPS_24);
    let track = project.add_track(TrackKind::Text, "T1");
    let clip = project
        .add_generated(
            track,
            Generator::Text {
                content: "Hello".into(),
                style: ModelTextStyle {
                    size: 64.0.into(),
                    fill: [255, 255, 255, 255].into(),
                    ..ModelTextStyle::default()
                },
            },
            TimeRange::at_rate(0, 48, FPS_24),
        )
        .unwrap();
    let mut anim = AnimationRef::new("typewriter");
    anim.speed = 1.5;
    anim.intensity = 1.2;
    anim.stagger = 0.6;
    project
        .set_clip_animation(clip, AnimationSlot::Combo, Some(anim))
        .unwrap();

    let mut renderer = match Renderer::new_headless() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("skipping typewriter-params smoke: no GPU ({e})");
            return;
        }
    };

    let early = renderer
        .render_frame(&project, rt(8))
        .expect("early typewriter");
    let late = renderer
        .render_frame(&project, rt(40))
        .expect("late typewriter");
    let c_early = ink(&early);
    let c_late = ink(&late);
    assert!(c_early > 0 || c_late > 0, "tunable typewriter should ink");
    assert!(
        c_late >= c_early,
        "later phase should not lose coverage ({c_late} < {c_early})"
    );

    let scene = resolve(&project, rt(12)).expect("resolve");
    let text_anim = scene
        .layers
        .iter()
        .find_map(|layer| match &layer.source {
            cutlass_render::LayerSource::Text { animation, .. } => animation.as_ref(),
            _ => None,
        })
        .expect("text animation on layer");
    assert_eq!(text_anim.id, "typewriter");
    assert!((text_anim.intensity - 1.2).abs() < 1e-4);
    assert!((text_anim.stagger - 0.6).abs() < 1e-4);
}

#[test]
fn keyframed_text_opacity_matches_tolerance() {
    // Opacity keyframes on the clip transform — compositor path with a
    // tolerance check against a constant half-opacity control frame.
    let mut project = Project::new("kf-opacity", FPS_24);
    let track = project.add_track(TrackKind::Text, "T1");
    let clip = project
        .add_generated(
            track,
            Generator::Text {
                content: "Hi".into(),
                style: ModelTextStyle {
                    size: 72.0.into(),
                    fill: [255, 255, 255, 255].into(),
                    ..ModelTextStyle::default()
                },
            },
            TimeRange::at_rate(0, 25, FPS_24),
        )
        .unwrap();
    project
        .set_param_keyframe(
            clip,
            ClipParam::Opacity,
            rt(0),
            ParamValue::Scalar(0.0),
            Easing::Linear,
            None,
        )
        .unwrap();
    project
        .set_param_keyframe(
            clip,
            ClipParam::Opacity,
            rt(24),
            ParamValue::Scalar(1.0),
            Easing::Linear,
            None,
        )
        .unwrap();

    let mut control = Project::new("opacity-ctrl", FPS_24);
    let ctrack = control.add_track(TrackKind::Text, "T1");
    let cclip = control
        .add_generated(
            ctrack,
            Generator::Text {
                content: "Hi".into(),
                style: ModelTextStyle {
                    size: 72.0.into(),
                    fill: [255, 255, 255, 255].into(),
                    ..ModelTextStyle::default()
                },
            },
            TimeRange::at_rate(0, 25, FPS_24),
        )
        .unwrap();
    control
        .set_param_constant(cclip, ClipParam::Opacity, ParamValue::Scalar(0.5))
        .unwrap();

    let mut renderer = match Renderer::new_headless() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("skipping opacity tolerance: no GPU ({e})");
            return;
        }
    };

    let keyed = renderer
        .render_frame(&project, rt(12))
        .expect("keyframed mid opacity");
    let fixed = renderer
        .render_frame(&control, rt(12))
        .expect("constant half opacity");

    assert_eq!(keyed.width, fixed.width);
    assert_eq!(keyed.height, fixed.height);
    // Max channel delta across all pixels — soft GPU rounding budget.
    let mut max_delta = 0u8;
    for (a, b) in keyed.pixels.iter().zip(fixed.pixels.iter()) {
        max_delta = max_delta.max(a.abs_diff(*b));
    }
    assert!(
        max_delta <= 12,
        "mid-keyframe opacity should match constant 0.5 within tolerance (max Δ={max_delta})"
    );
}
