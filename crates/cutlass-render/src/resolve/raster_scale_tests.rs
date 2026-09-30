//! Resolve-level coverage for text / pen-path scale-aware supersampling.

use cutlass_core::{Rational, RationalTime};
use cutlass_models::{
    ClipTransform, Generator, Project, Scale2, Shape, ShapePath, ShapePathPoint, ShapeStroke,
    TextStyle as ModelTextStyle, TimeRange, TrackKind,
};

use crate::resolve::raster_supersample::{
    clamp_supersample, quantize_supersample, supersample_from_scale,
};
use crate::resolve::resolve;
use crate::scene::{LayerSource, SizeSpec};

const FPS_24: Rational = Rational::FPS_24;

fn rt(value: i64) -> RationalTime {
    RationalTime::new(value, FPS_24)
}

fn tr(start: i64, duration: i64) -> TimeRange {
    TimeRange::at_rate(start, duration, FPS_24)
}

fn approx(a: f32, b: f32) {
    assert!((a - b).abs() < 1e-3, "{a} != {b}");
}

fn approx2(a: [f32; 2], b: [f32; 2]) {
    approx(a[0], b[0]);
    approx(a[1], b[1]);
}

fn add_text(project: &mut Project, content: &str, size: f32) -> cutlass_models::ClipId {
    let track = project.add_track(TrackKind::Text, "T1");
    project
        .add_generated(
            track,
            Generator::Text {
                content: content.into(),
                style: ModelTextStyle {
                    size: size.into(),
                    ..ModelTextStyle::default()
                },
            },
            tr(0, 100),
        )
        .unwrap()
}

#[test]
fn text_scale_2_doubles_font_and_identity_residual() {
    let mut project = Project::new("p", FPS_24);
    let clip = add_text(&mut project, "Hi", 90.0);
    project
        .set_transform(
            clip,
            ClipTransform {
                scale: 2.0.into(),
                ..ClipTransform::IDENTITY
            },
            None,
        )
        .unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    let layer = &scene.layers[0];
    match &layer.source {
        LayerSource::Text { style, .. } => {
            // Reference 90 → × S=2; line_height 90×1.2 → 216.
            approx(style.font_size, 180.0);
            approx(style.line_height, 216.0);
        }
        other => panic!("expected text, got {other:?}"),
    }
    assert_eq!(layer.size, SizeSpec::BitmapScaled([1.0, 1.0]));
}

#[test]
fn text_scale_half_keeps_reference_raster() {
    let mut project = Project::new("p", FPS_24);
    let clip = add_text(&mut project, "Hi", 90.0);
    project
        .set_transform(
            clip,
            ClipTransform {
                scale: 0.5.into(),
                ..ClipTransform::IDENTITY
            },
            None,
        )
        .unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    let layer = &scene.layers[0];
    match &layer.source {
        LayerSource::Text { style, .. } => approx(style.font_size, 90.0),
        other => panic!("expected text, got {other:?}"),
    }
    assert_eq!(layer.size, SizeSpec::BitmapScaled([0.5, 0.5]));
}

#[test]
fn text_nonuniform_scale_uses_max_axis() {
    let mut project = Project::new("p", FPS_24);
    let clip = add_text(&mut project, "Hi", 90.0);
    project
        .set_transform(
            clip,
            ClipTransform {
                scale: Scale2 { x: 1.0, y: 3.0 },
                ..ClipTransform::IDENTITY
            },
            None,
        )
        .unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    let layer = &scene.layers[0];
    match &layer.source {
        LayerSource::Text { style, .. } => approx(style.font_size, 270.0),
        other => panic!("expected text, got {other:?}"),
    }
    let SizeSpec::BitmapScaled(r) = layer.size else {
        panic!("expected BitmapScaled");
    };
    approx2(r, [1.0 / 3.0, 1.0]);
}

#[test]
fn quantization_step_function() {
    // Documented policy: quarter steps. 1.9 and 2.0 share a step.
    approx(quantize_supersample(1.9), 2.0);
    approx(quantize_supersample(2.0), 2.0);
    approx(supersample_from_scale(Scale2 { x: 1.9, y: 1.9 }), 2.0);
    approx(supersample_from_scale(Scale2 { x: 1.12, y: 1.12 }), 1.0);
    approx(supersample_from_scale(Scale2 { x: 1.2, y: 1.2 }), 1.25);
}

/// Adjacent scales inside one quarter-step resolve to identical raster metrics
/// (same `font_size` bit pattern → text memo key hit; no per-tick re-raster).
#[test]
fn scale_drag_within_step_keeps_identical_text_style() {
    let mut project = Project::new("p", FPS_24);
    let clip = add_text(&mut project, "Memo", 90.0);

    let font_at = |project: &mut Project, clip: cutlass_models::ClipId, s: f32| -> u32 {
        project
            .set_transform(
                clip,
                ClipTransform {
                    scale: s.into(),
                    ..ClipTransform::IDENTITY
                },
                None,
            )
            .unwrap();
        let scene = resolve(project, rt(5)).unwrap();
        match &scene.layers[0].source {
            LayerSource::Text { style, .. } => style.font_size.to_bits(),
            other => panic!("expected text, got {other:?}"),
        }
    };

    // [1.125, 1.375) → S = 1.25
    let a = font_at(&mut project, clip, 1.20);
    let b = font_at(&mut project, clip, 1.30);
    assert_eq!(a, b, "within-step drag must not change raster style bits");
    // Crossing into the next quarter step must change the key.
    let c = font_at(&mut project, clip, 1.40);
    assert_ne!(a, c, "crossing a quarter step must change font_size bits");
}

#[test]
fn clamp_allows_residual_above_one() {
    // Huge scale on a large ref edge must back S off; residual may exceed 1.
    let s = clamp_supersample(50.0, 200.0, 1920.0);
    // cap = min(4096, 3840) = 3840 → max S = 3840/200 = 19.2
    approx(s, 19.2);
    let residual = 50.0 / s;
    assert!(residual > 1.0, "residual {residual} should stretch past 1");
}

#[test]
fn text_absurd_scale_clamps_and_stretches_residual() {
    let mut project = Project::new("p", FPS_24);
    // Large font + long wrap width → big ref edge so clamp binds before S=50.
    let track = project.add_track(TrackKind::Text, "T1");
    let clip = project
        .add_generated(
            track,
            Generator::Text {
                content: "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW".into(),
                style: ModelTextStyle {
                    size: 400.0.into(),
                    wrap: true,
                    ..ModelTextStyle::default()
                },
            },
            tr(0, 100),
        )
        .unwrap();
    project
        .set_transform(
            clip,
            ClipTransform {
                scale: 50.0.into(),
                ..ClipTransform::IDENTITY
            },
            None,
        )
        .unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    let layer = &scene.layers[0];
    let SizeSpec::BitmapScaled(r) = layer.size else {
        panic!("expected BitmapScaled");
    };
    assert!(
        r[0] > 1.0 && r[1] > 1.0,
        "clamp should leave residual > 1, got {r:?}"
    );
    match &layer.source {
        LayerSource::Text { style, .. } => {
            // Unclamped would be 400 * 50 = 20000; clamp must keep it far lower.
            assert!(
                style.font_size < 400.0 * 50.0,
                "font_size {} should be clamped",
                style.font_size
            );
            assert!(
                style.font_size > 400.0,
                "still supersampled above reference"
            );
        }
        other => panic!("expected text, got {other:?}"),
    }
}

#[test]
fn path_scale_2_raises_raster_scale_with_identity_residual() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    let path = ShapePath {
        points: vec![
            ShapePathPoint::corner([-40.0, -40.0]),
            ShapePathPoint::corner([40.0, -40.0]),
            ShapePathPoint::corner([0.0, 40.0]),
        ],
        closed: true,
    };
    let mut generator = Generator::shape(Shape::Path(path), [0, 255, 0, 255]);
    if let Generator::Shape { stroke, .. } = &mut generator {
        *stroke = Some(ShapeStroke::new([0, 0, 0, 255], 4.0));
    }
    let clip = project.add_generated(track, generator, tr(0, 100)).unwrap();
    project
        .set_transform(
            clip,
            ClipTransform {
                scale: 2.0.into(),
                ..ClipTransform::IDENTITY
            },
            None,
        )
        .unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    let layer = &scene.layers[0];
    match &layer.source {
        LayerSource::PathShape {
            raster_scale,
            stroke,
            ..
        } => {
            // 1080 canvas → ref_scale 1, S=2 → raster_scale 2.
            approx(*raster_scale, 2.0);
            // Model stroke stays unscaled; PathRaster × raster_scale → 8 px.
            approx(stroke.expect("stroke").width, 4.0);
        }
        other => panic!("expected path, got {other:?}"),
    }
    assert_eq!(layer.size, SizeSpec::BitmapScaled([1.0, 1.0]));
}

#[test]
fn path_scale_half_keeps_ref_raster() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    let path = ShapePath {
        points: vec![
            ShapePathPoint::corner([-40.0, -40.0]),
            ShapePathPoint::corner([40.0, -40.0]),
            ShapePathPoint::corner([0.0, 40.0]),
        ],
        closed: true,
    };
    let clip = project
        .add_generated(
            track,
            Generator::shape(Shape::Path(path), [0, 255, 0, 255]),
            tr(0, 100),
        )
        .unwrap();
    project
        .set_transform(
            clip,
            ClipTransform {
                scale: 0.5.into(),
                ..ClipTransform::IDENTITY
            },
            None,
        )
        .unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    let layer = &scene.layers[0];
    match &layer.source {
        LayerSource::PathShape { raster_scale, .. } => approx(*raster_scale, 1.0),
        other => panic!("expected path, got {other:?}"),
    }
    assert_eq!(layer.size, SizeSpec::BitmapScaled([0.5, 0.5]));
}
