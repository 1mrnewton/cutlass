use super::*;
use crate::grade::effective_grade;
use crate::scene::{LayerSource, SceneShadow, SizeSpec};
use cutlass_models::{
    AnimationRef, AnimationSlot, BlendMode, CanvasAspect, CanvasSettings, ClipParam, ClipTransform,
    ColorAdjustments, CropRect, Easing, Filter, Generator, LayerGlow, LayerOutline, LayerShadow,
    LayerStyles, MediaSource, Param, ParamValue, Project, Rational, RationalTime, Scale2, Shape,
    StyleParam, TextStyle as ModelTextStyle, TimeRange, TrackKind,
};

const FPS_24: Rational = Rational::FPS_24;

fn rt(value: i64) -> RationalTime {
    RationalTime::new(value, FPS_24)
}

fn tr(start: i64, duration: i64) -> TimeRange {
    TimeRange::at_rate(start, duration, FPS_24)
}

fn video(width: u32, height: u32) -> MediaSource {
    MediaSource::new("/tmp/v.mp4", width, height, FPS_24, 600, false)
}

fn approx(a: f32, b: f32) {
    assert!((a - b).abs() < 1e-3, "expected ~{b}, got {a}");
}

fn approx2(a: [f32; 2], b: [f32; 2]) {
    approx(a[0], b[0]);
    approx(a[1], b[1]);
}

#[test]
fn empty_project_uses_default_canvas_and_has_no_layers() {
    let project = Project::new("p", FPS_24);
    let scene = resolve(&project, rt(0)).unwrap();
    assert_eq!((scene.width, scene.height), (1920, 1080));
    assert!(scene.layers.is_empty());
}

#[test]
fn single_video_clip_is_centered_and_aspect_fit() {
    let mut project = Project::new("p", FPS_24);
    let media = project.add_media(video(1920, 1080));
    let track = project.add_track(TrackKind::Video, "V1");
    project.add_clip(track, media, tr(0, 100), rt(0)).unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    assert_eq!((scene.width, scene.height), (1920, 1080));
    assert_eq!(scene.layers.len(), 1);

    let layer = &scene.layers[0];
    approx2(layer.center, [960.0, 540.0]);
    match layer.size {
        SizeSpec::Fixed(size) => approx2(size, [1920.0, 1080.0]),
        other => panic!("expected fixed size, got {other:?}"),
    }
    approx(layer.opacity, 1.0);
    match &layer.source {
        LayerSource::Media {
            media: m,
            source_time,
        } => {
            assert_eq!(*m, media);
            assert_eq!(source_time.value, 5);
        }
        other => panic!("expected media source, got {other:?}"),
    }
}

#[test]
fn clip_inactive_at_time_produces_no_layer() {
    let mut project = Project::new("p", FPS_24);
    let media = project.add_media(video(1920, 1080));
    let track = project.add_track(TrackKind::Video, "V1");
    project.add_clip(track, media, tr(0, 10), rt(0)).unwrap();

    let scene = resolve(&project, rt(50)).unwrap();
    assert!(scene.layers.is_empty());
}

#[test]
fn tracks_stack_bottom_to_top() {
    let mut project = Project::new("p", FPS_24);
    let bottom = project.add_track(TrackKind::Sticker, "S1");
    let top = project.add_track(TrackKind::Sticker, "S2");
    project
        .add_generated(
            bottom,
            Generator::SolidColor {
                rgba: [255, 0, 0, 255],
            },
            tr(0, 100),
        )
        .unwrap();
    project
        .add_generated(
            top,
            Generator::SolidColor {
                rgba: [0, 0, 255, 255],
            },
            tr(0, 100),
        )
        .unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    assert_eq!(scene.layers.len(), 2);
    assert_eq!(scene.layers[0].source, LayerSource::Solid([255, 0, 0, 255]));
    assert_eq!(scene.layers[1].source, LayerSource::Solid([0, 0, 255, 255]));
}

#[test]
fn text_generator_maps_style_and_defers_size() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Text, "T1");
    let style = ModelTextStyle {
        size: 90.0.into(),
        bold: true,
        italic: true,
        underline: true,
        fill: [255, 0, 0, 255].into(),
        letter_spacing: 12.0.into(),
        line_spacing: 1.5.into(),
        align_h: cutlass_models::TextAlignH::Right,
        align_v: cutlass_models::TextAlignV::Bottom,
        stroke: Some(cutlass_models::TextStroke {
            rgba: [0, 0, 0, 255].into(),
            width: 8.0.into(),
        }),
        background: Some(cutlass_models::TextBackground {
            rgba: [0, 0, 255, 200].into(),
            radius: 0.5.into(),
        }),
        shadow: Some(cutlass_models::TextShadow {
            rgba: [0, 0, 0, 230].into(),
            blur: 0.15.into(),
            distance: 10.0.into(),
        }),
        ..ModelTextStyle::default()
    };
    project
        .add_generated(
            track,
            Generator::Text {
                content: "Hi".into(),
                style,
            },
            tr(0, 100),
        )
        .unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    assert_eq!((scene.width, scene.height), (1920, 1080));
    assert_eq!(scene.layers.len(), 1);
    let layer = &scene.layers[0];
    approx2(layer.center, [960.0, 540.0]);
    assert_eq!(layer.size, SizeSpec::BitmapScaled([1.0, 1.0]));
    match &layer.source {
        LayerSource::Text {
            content,
            style,
            animation,
        } => {
            assert!(animation.is_none());
            assert_eq!(content, "Hi");
            approx(style.font_size, 90.0);
            assert_eq!(style.color, [255, 0, 0, 255]);
            assert!(style.bold);
            assert!(style.italic);
            assert!(style.underline);
            approx(style.letter_spacing, 12.0);
            approx(style.line_height, 135.0);
            assert_eq!(style.align, cutlass_text::TextAlign::Right);
            assert_eq!(
                style.vertical_align,
                cutlass_text::TextVerticalAlign::Bottom
            );
            assert_eq!(style.max_width, Some(1920.0));
            let stroke = style.stroke.expect("stroke mapped");
            assert_eq!(stroke.rgba, [0, 0, 0, 255]);
            approx(stroke.width, 8.0);
            let bg = style.background.expect("background mapped");
            assert_eq!(bg.rgba, [0, 0, 255, 200]);
            approx(bg.radius, 0.5);
            let shadow = style.shadow.expect("shadow mapped");
            assert_eq!(shadow.rgba, [0, 0, 0, 230]);
            approx(shadow.blur, 0.15);
            approx(shadow.distance, 10.0);
        }
        other => panic!("expected text source, got {other:?}"),
    }
}

#[test]
fn text_wrap_toggle_removes_the_canvas_width_constraint() {
    let style = ModelTextStyle {
        wrap: false,
        ..ModelTextStyle::default()
    };
    assert_eq!(map_text_style(&style, 1920.0, 1080.0, 0).max_width, None);
}

#[test]
fn keyframed_text_background_samples_card_per_tick() {
    use cutlass_models::{ClipParam, Easing, ParamValue, TextParam};

    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Text, "T1");
    let clip = project
        .add_generated(
            track,
            Generator::Text {
                content: "Card".into(),
                style: ModelTextStyle {
                    background: Some(cutlass_models::TextBackground::default()),
                    ..ModelTextStyle::default()
                },
            },
            tr(0, 100),
        )
        .unwrap();
    let color = ClipParam::Text {
        param: TextParam::BackgroundColor,
    };
    let radius = ClipParam::Text {
        param: TextParam::BackgroundRadius,
    };
    project
        .set_param_keyframe(
            clip,
            color,
            rt(0),
            ParamValue::Color([0, 0, 0, 255]),
            Easing::Linear,
            None,
        )
        .unwrap();
    project
        .set_param_keyframe(
            clip,
            color,
            rt(50),
            ParamValue::Color([200, 100, 0, 255]),
            Easing::Linear,
            None,
        )
        .unwrap();
    project
        .set_param_keyframe(
            clip,
            radius,
            rt(0),
            ParamValue::Scalar(0.0),
            Easing::Linear,
            None,
        )
        .unwrap();
    project
        .set_param_keyframe(
            clip,
            radius,
            rt(50),
            ParamValue::Scalar(1.0),
            Easing::Linear,
            None,
        )
        .unwrap();

    let scene = resolve(&project, rt(25)).unwrap();
    match &scene.layers[0].source {
        LayerSource::Text { style, .. } => {
            let bg = style.background.expect("background");
            assert_eq!(bg.rgba, [100, 50, 0, 255], "card color lerps per channel");
            approx(bg.radius, 0.5);
        }
        other => panic!("expected text source, got {other:?}"),
    }
}

#[test]
fn solid_generator_fills_canvas() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [10, 20, 30, 255],
            },
            tr(0, 100),
        )
        .unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    let layer = &scene.layers[0];
    match layer.size {
        SizeSpec::Fixed(size) => approx2(size, [1920.0, 1080.0]),
        other => panic!("expected fixed size, got {other:?}"),
    }
}

#[test]
fn sticker_generator_resolves_to_a_reference_scaled_layer() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    project
        .add_generated(track, Generator::sticker("heart"), tr(0, 100))
        .unwrap();

    let scene = resolve(&project, rt(12)).unwrap();
    assert_eq!(scene.layers.len(), 1);
    let layer = &scene.layers[0];
    approx2(layer.center, [960.0, 540.0]);
    // The heart asset is 256×256; intrinsic pixels are reference pixels,
    // which map 1:1 on the 1080p canvas.
    match layer.size {
        SizeSpec::Fixed(size) => approx2(size, [256.0, 256.0]),
        other => panic!("expected fixed size, got {other:?}"),
    }
    match &layer.source {
        LayerSource::Sticker { asset, local_time } => {
            assert_eq!(asset, "heart");
            approx(*local_time as f32, 0.5); // tick 12 at 24 fps
        }
        other => panic!("expected sticker source, got {other:?}"),
    }
}

#[test]
fn sticker_local_time_is_clip_relative() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    project
        .add_generated(track, Generator::sticker("star_spin"), tr(24, 100))
        .unwrap();

    let scene = resolve(&project, rt(36)).unwrap();
    match &scene.layers[0].source {
        LayerSource::Sticker { local_time, .. } => approx(*local_time as f32, 0.5),
        other => panic!("expected sticker source, got {other:?}"),
    }
}

#[test]
fn empty_or_unknown_sticker_asset_places_nothing() {
    // The legacy payload-less sticker (empty asset) loads but draws
    // nothing, exactly like before the catalog existed.
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    project
        .add_generated(track, Generator::sticker(""), tr(0, 100))
        .unwrap();
    assert!(resolve(&project, rt(5)).unwrap().layers.is_empty());

    // Unknown ids can't enter via commands (validation), but the
    // resolver stays graceful if one ever reaches it.
    let layer = resolve_generator(
        &Generator::sticker("not-a-sticker"),
        [960.0, 540.0],
        [0.5, 0.5],
        0.0,
        1.0,
        [0.0, 0.0, 1.0, 1.0],
        None,
        None,
        1920.0,
        1080.0,
        Scale2::ONE,
        0,
        0.0,
        Vec::new(),
    );
    assert!(layer.is_none());
}

#[test]
fn crop_and_horizontal_flip_set_uv() {
    let mut project = Project::new("p", FPS_24);
    let media = project.add_media(video(1920, 1080));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip = project.add_clip(track, media, tr(0, 100), rt(0)).unwrap();
    project
        .set_clip_crop(
            clip,
            CropRect {
                x: 0.25,
                y: 0.1,
                w: 0.5,
                h: 0.8,
            },
            true,
            false,
            None,
        )
        .unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    let uv = scene.layers[0].uv;
    // base [0.25, 0.1, 0.75, 0.9]; flip_h swaps the u axis.
    approx(uv[0], 0.75);
    approx(uv[1], 0.1);
    approx(uv[2], 0.25);
    approx(uv[3], 0.9);
}

#[test]
fn keyframed_crop_samples_uv_per_tick() {
    let mut project = Project::new("p", FPS_24);
    let media = project.add_media(video(1920, 1080));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip = project.add_clip(track, media, tr(0, 100), rt(0)).unwrap();
    let a = CropRect {
        x: 0.0,
        y: 0.0,
        w: 1.0,
        h: 1.0,
    };
    let b = CropRect {
        x: 0.4,
        y: 0.2,
        w: 0.5,
        h: 0.6,
    };
    project
        .set_param_keyframe(
            clip,
            ClipParam::Crop,
            rt(0),
            ParamValue::Rect([a.x, a.y, a.w, a.h]),
            Easing::Linear,
            None,
        )
        .unwrap();
    project
        .set_param_keyframe(
            clip,
            ClipParam::Crop,
            rt(10),
            ParamValue::Rect([b.x, b.y, b.w, b.h]),
            Easing::Linear,
            None,
        )
        .unwrap();

    let uv0 = resolve(&project, rt(0)).unwrap().layers[0].uv;
    approx(uv0[0], 0.0);
    approx(uv0[1], 0.0);
    approx(uv0[2], 1.0);
    approx(uv0[3], 1.0);

    let uv5 = resolve(&project, rt(5)).unwrap().layers[0].uv;
    // Midpoint crop [0.2, 0.1, 0.75, 0.8] → UV [u0,v0,u1,v1].
    approx(uv5[0], 0.2);
    approx(uv5[1], 0.1);
    approx(uv5[2], 0.95);
    approx(uv5[3], 0.9);

    let uv10 = resolve(&project, rt(10)).unwrap().layers[0].uv;
    approx(uv10[0], 0.4);
    approx(uv10[1], 0.2);
    approx(uv10[2], 0.9);
    approx(uv10[3], 0.8);
}

#[test]
fn fixed_aspect_presets_resolve_to_1080_baseline() {
    let mut project = Project::new("p", FPS_24);
    project.timeline_mut().set_canvas(CanvasSettings {
        aspect: CanvasAspect::Tall9x16,
        background: [0, 0, 0],
    });
    assert_eq!(canvas_size(&project), (1080, 1920));

    project.timeline_mut().set_canvas(CanvasSettings {
        aspect: CanvasAspect::Wide16x9,
        background: [0, 0, 0],
    });
    assert_eq!(canvas_size(&project), (1920, 1080));

    project.timeline_mut().set_canvas(CanvasSettings {
        aspect: CanvasAspect::Square1x1,
        background: [0, 0, 0],
    });
    assert_eq!(canvas_size(&project), (1080, 1080));
}

#[test]
fn transform_offsets_center_and_scales_size() {
    let mut project = Project::new("p", FPS_24);
    let media = project.add_media(video(1920, 1080));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip = project.add_clip(track, media, tr(0, 100), rt(0)).unwrap();
    project
        .set_transform(
            clip,
            ClipTransform {
                position: [0.25, -0.1],
                scale: 2.0.into(),
                opacity: 0.5,
                ..ClipTransform::IDENTITY
            },
            None,
        )
        .unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    let layer = &scene.layers[0];
    approx2(layer.center, [1920.0 * 0.75, 1080.0 * 0.4]);
    match layer.size {
        SizeSpec::Fixed(size) => approx2(size, [3840.0, 2160.0]),
        other => panic!("expected fixed size, got {other:?}"),
    }
    approx(layer.opacity, 0.5);
}

#[test]
fn per_axis_scale_stretches_media_width_only() {
    let mut project = Project::new("p", FPS_24);
    let media = project.add_media(video(1920, 1080));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip = project.add_clip(track, media, tr(0, 100), rt(0)).unwrap();
    project
        .set_transform(
            clip,
            ClipTransform {
                scale: Scale2 { x: 2.0, y: 1.0 },
                ..ClipTransform::IDENTITY
            },
            None,
        )
        .unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    match scene.layers[0].size {
        // Width doubled, height unchanged.
        SizeSpec::Fixed(size) => approx2(size, [3840.0, 1080.0]),
        other => panic!("expected fixed size, got {other:?}"),
    }
}

#[test]
fn shape_stroke_uses_isotropic_scale() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    let generator = Generator::Shape {
        shape: Shape::Rectangle,
        rgba: Param::Constant([255, 0, 0, 255]),
        width: Param::Constant(100.0),
        height: Param::Constant(50.0),
        corner_radius: Param::Constant(0.0),
        stroke: Some(cutlass_models::ShapeStroke::new([0, 0, 0, 255], 6.0)),
    };
    let clip = project.add_generated(track, generator, tr(0, 100)).unwrap();

    // Uniform [2,2]: stroke doubles (ref_scale = 1 on 1080p canvas).
    project
        .set_transform(
            clip,
            ClipTransform {
                scale: Scale2::uniform(2.0),
                ..ClipTransform::IDENTITY
            },
            None,
        )
        .unwrap();
    let scene = resolve(&project, rt(5)).unwrap();
    match &scene.layers[0].source {
        LayerSource::Shape { stroke, .. } => {
            approx(stroke.expect("stroke").width, 12.0);
        }
        other => panic!("expected shape, got {other:?}"),
    }

    // Split [4,1]: geometric mean is still 2 → stroke still doubles.
    project
        .set_transform(
            clip,
            ClipTransform {
                scale: Scale2 { x: 4.0, y: 1.0 },
                ..ClipTransform::IDENTITY
            },
            None,
        )
        .unwrap();
    let scene = resolve(&project, rt(5)).unwrap();
    match &scene.layers[0].source {
        LayerSource::Shape { stroke, .. } => {
            approx(stroke.expect("stroke").width, 12.0);
        }
        other => panic!("expected shape, got {other:?}"),
    }
    // Placement still stretches per-axis (width ×4, height ×1) plus pad.
    match scene.layers[0].size {
        SizeSpec::Fixed(size) => {
            assert!(size[0] > size[1] * 3.0, "width should dominate under [4,1]");
        }
        other => panic!("expected fixed size, got {other:?}"),
    }
}

#[test]
fn look_animation_fade_in_ramps_opacity_at_clip_start() {
    let mut project = Project::new("p", FPS_24);
    let media = project.add_media(video(1920, 1080));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip = project.add_clip(track, media, tr(0, 100), rt(0)).unwrap();
    project
        .set_clip_animation(clip, AnimationSlot::In, Some(AnimationRef::new("fade_in")))
        .unwrap();

    let start = resolve(&project, rt(0)).unwrap();
    assert!(start.layers[0].opacity < 0.05);
    let mid = resolve(&project, rt(50)).unwrap();
    approx(mid.layers[0].opacity, 1.0);
}

#[test]
fn look_animation_slide_up_offsets_center_at_start() {
    let mut project = Project::new("p", FPS_24);
    let media = project.add_media(video(1920, 1080));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip = project.add_clip(track, media, tr(0, 100), rt(0)).unwrap();
    project
        .set_clip_animation(clip, AnimationSlot::In, Some(AnimationRef::new("slide_up")))
        .unwrap();

    let start = resolve(&project, rt(0)).unwrap();
    assert!(start.layers[0].center[1] > 540.0);
    let mid = resolve(&project, rt(50)).unwrap();
    approx2(mid.layers[0].center, [960.0, 540.0]);
}

#[test]
fn look_animation_zoom_in_scales_size_at_start() {
    let mut project = Project::new("p", FPS_24);
    let media = project.add_media(video(1920, 1080));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip = project.add_clip(track, media, tr(0, 100), rt(0)).unwrap();
    project
        .set_clip_animation(clip, AnimationSlot::In, Some(AnimationRef::new("zoom_in")))
        .unwrap();

    let start = resolve(&project, rt(0)).unwrap();
    match start.layers[0].size {
        SizeSpec::Fixed(size) => assert!(size[0] < 1920.0 * 0.5),
        other => panic!("expected fixed size, got {other:?}"),
    }
}

#[test]
fn look_animation_fade_out_ramps_at_clip_tail() {
    let mut project = Project::new("p", FPS_24);
    let media = project.add_media(video(1920, 1080));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip = project.add_clip(track, media, tr(0, 100), rt(0)).unwrap();
    project
        .set_clip_animation(
            clip,
            AnimationSlot::Out,
            Some(AnimationRef::new("fade_out")),
        )
        .unwrap();

    let tail = resolve(&project, rt(99)).unwrap();
    assert!(tail.layers[0].opacity < 0.05);
    let mid = resolve(&project, rt(40)).unwrap();
    approx(mid.layers[0].opacity, 1.0);
}

#[test]
fn look_animation_combo_loops_and_clears_in_out() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    let clip = project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [255, 0, 0, 255],
            },
            tr(0, 100),
        )
        .unwrap();
    project
        .set_clip_animation(clip, AnimationSlot::In, Some(AnimationRef::new("fade_in")))
        .unwrap();
    project
        .set_clip_animation(clip, AnimationSlot::Combo, Some(AnimationRef::new("pulse")))
        .unwrap();

    let a = resolve(&project, rt(0)).unwrap();
    let b = resolve(&project, rt(6)).unwrap();
    approx(a.layers[0].opacity, 1.0);
    match (a.layers[0].size, b.layers[0].size) {
        (SizeSpec::Fixed(sa), SizeSpec::Fixed(sb)) => {
            assert!((sa[0] - sb[0]).abs() > 1.0);
        }
        other => panic!("expected fixed sizes, got {other:?}"),
    }
}

#[test]
fn non_center_anchor_offsets_the_quad_center() {
    let mut project = Project::new("p", FPS_24);
    let media = project.add_media(video(1920, 1080));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip = project.add_clip(track, media, tr(0, 100), rt(0)).unwrap();
    // Anchor on the content's top-left corner, positioned at the canvas
    // center: the quad center lands half a size down-right of it.
    project
        .set_transform(
            clip,
            ClipTransform {
                anchor_point: [0.0, 0.0],
                ..ClipTransform::IDENTITY
            },
            None,
        )
        .unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    let layer = &scene.layers[0];
    // The layer's `center` is the anchor's canvas position...
    approx2(layer.center, [960.0, 540.0]);
    assert_eq!(layer.anchor_point, [0.0, 0.0]);
    // ...and the derived quad center hangs down-right of it.
    approx2(layer.quad_center([1920.0, 1080.0]), [1920.0, 1080.0]);
}

#[test]
fn quad_center_rotates_about_the_anchor() {
    let layer = SceneLayer {
        clip: None,
        source: LayerSource::Solid([255, 255, 255, 255]),
        center: [960.0, 540.0],
        anchor_point: [0.0, 0.0],
        size: SizeSpec::Fixed([1920.0, 1080.0]),
        rotation: std::f32::consts::FRAC_PI_2, // 90° clockwise
        opacity: 1.0,
        uv: [0.0, 0.0, 1.0, 1.0],
        effects: Vec::new(),
        mask: None,
        chroma_key: None,
        color_grade: None,
        lut: None,
        blend_mode: BlendMode::Normal,
        styles: None,
    };
    // to_center (960, 540) rotated 90° cw (+y down) → (-540, 960).
    approx2(layer.quad_center([1920.0, 1080.0]), [420.0, 1500.0]);
}

#[test]
fn resolve_with_substitutes_transform_and_generator_for_one_clip() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    let clip = project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [255, 0, 0, 255],
            },
            tr(0, 100),
        )
        .unwrap();

    let overrides = ResolveOverrides {
        transform: Some((
            clip,
            ClipTransform {
                position: [0.25, 0.0],
                scale: 0.5.into(),
                ..ClipTransform::IDENTITY
            },
        )),
        generator: Some((
            clip,
            &Generator::SolidColor {
                rgba: [0, 255, 0, 255],
            },
        )),
        look: Some((
            clip,
            None,
            &ColorAdjustments {
                brightness: 0.25.into(),
                ..ColorAdjustments::default()
            },
        )),
        styles: None,
    };
    let scene = resolve_with(&project, rt(5), overrides).unwrap();
    let layer = &scene.layers[0];
    assert_eq!(layer.source, LayerSource::Solid([0, 255, 0, 255]));
    assert_eq!(
        layer.color_grade,
        Some(effective_grade(
            None,
            &ColorAdjustments {
                brightness: 0.25.into(),
                ..ColorAdjustments::default()
            }
        ))
    );
    approx2(layer.center, [1920.0 * 0.75, 540.0]);
    match layer.size {
        SizeSpec::Fixed(size) => approx2(size, [960.0, 540.0]),
        other => panic!("expected fixed size, got {other:?}"),
    }

    // Plain resolve still sees only committed state.
    let scene = resolve(&project, rt(5)).unwrap();
    assert_eq!(scene.layers[0].source, LayerSource::Solid([255, 0, 0, 255]));
    assert_eq!(scene.layers[0].color_grade, None);
    approx2(scene.layers[0].center, [960.0, 540.0]);
}

#[test]
fn image_clip_resolves_to_an_aspect_fit_still_layer() {
    let mut project = Project::new("p", FPS_24);
    // 800×600 still on the default 1920×1080 canvas: fit = 1.8.
    let media = project.add_media(MediaSource::image("/photos/pic.png", 800, 600));
    let window = project.media(media).unwrap().full_range();
    let track = project.add_track(TrackKind::Video, "V1");
    project.add_clip(track, media, window, rt(0)).unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    assert_eq!(scene.layers.len(), 1);
    let layer = &scene.layers[0];
    assert_eq!(layer.source, LayerSource::Still { media });
    approx2(layer.center, [960.0, 540.0]);
    match layer.size {
        SizeSpec::Fixed(size) => approx2(size, [1440.0, 1080.0]),
        other => panic!("expected fixed size, got {other:?}"),
    }
}

#[test]
fn still_layer_ignores_retiming_and_covers_the_whole_clip() {
    let mut project = Project::new("p", FPS_24);
    let media = project.add_media(MediaSource::image("/photos/pic.png", 1920, 1080));
    let window = project.media(media).unwrap().full_range();
    let track = project.add_track(TrackKind::Video, "V1");
    let clip = project.add_clip(track, media, window, rt(0)).unwrap();
    // Reverse would make a video clip walk its source backward; a still
    // must keep producing its one frame at every covered tick.
    project
        .set_clip_speed(clip, Rational::new(1, 1), true)
        .unwrap();
    let duration = project.timeline().duration().value;
    assert!(duration > 0);

    for t in [0, duration / 2, duration - 1] {
        let scene = resolve(&project, rt(t)).unwrap();
        assert_eq!(scene.layers.len(), 1, "tick {t}");
        assert_eq!(scene.layers[0].source, LayerSource::Still { media });
    }
    // Past the clip: nothing.
    let scene = resolve(&project, rt(duration)).unwrap();
    assert!(scene.layers.is_empty());
}

#[test]
fn audio_track_is_skipped() {
    let mut project = Project::new("p", FPS_24);
    let video_track = project.add_track(TrackKind::Video, "V1");
    let media = project.add_media(video(1920, 1080));
    project
        .add_clip(video_track, media, tr(0, 100), rt(0))
        .unwrap();

    let audio_track = project.add_track(TrackKind::Audio, "A1");
    let song = project.add_media(MediaSource::new("/tmp/a.mp3", 0, 0, FPS_24, 600, true));
    project
        .add_clip(audio_track, song, tr(0, 100), rt(0))
        .unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    assert_eq!(scene.layers.len(), 1);
    assert!(matches!(scene.layers[0].source, LayerSource::Media { .. }));
}

#[test]
fn plain_rectangle_keeps_the_solid_fast_path() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    project
        .add_generated(
            track,
            Generator::shape(Shape::Rectangle, [255, 255, 255, 255]),
            tr(0, 100),
        )
        .unwrap();
    let scene = resolve(&project, rt(5)).unwrap();
    assert_eq!(scene.layers.len(), 1);
    assert!(matches!(scene.layers[0].source, LayerSource::Solid(_)));
    // Drop size 200×200 reference px on a 1080 canvas: 1:1.
    assert_eq!(scene.layers[0].size, SizeSpec::Fixed([200.0, 200.0]));
}

#[test]
fn ellipse_resolves_to_a_padded_sdf_layer() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    project
        .add_generated(
            track,
            Generator::shape(Shape::Ellipse, [10, 20, 30, 255]),
            tr(0, 100),
        )
        .unwrap();
    let scene = resolve(&project, rt(5)).unwrap();
    assert_eq!(scene.layers.len(), 1);
    let layer = &scene.layers[0];
    match &layer.source {
        LayerSource::Shape {
            params,
            fill,
            stroke,
            pad,
        } => {
            assert_eq!(*params, cutlass_shapes::SdfParams::Ellipse);
            assert_eq!(*fill, [10, 20, 30, 255]);
            assert!(stroke.is_none());
            // No stroke: pad is the 2px AA margin only.
            approx(*pad, 2.0 * cutlass_shapes::SDF_AA);
        }
        other => panic!("expected shape source, got {other:?}"),
    }
    // Quad = 200×200 shape + pad on each side.
    match layer.size {
        SizeSpec::Fixed(size) => approx2(size, [204.0, 204.0]),
        other => panic!("expected fixed size, got {other:?}"),
    }
}

#[test]
fn rounded_or_stroked_rectangle_leaves_the_fast_path() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    let generator = Generator::Shape {
        shape: Shape::Rectangle,
        rgba: cutlass_models::Param::Constant([255, 0, 0, 255]),
        width: cutlass_models::Param::Constant(100.0),
        height: cutlass_models::Param::Constant(50.0),
        corner_radius: cutlass_models::Param::Constant(8.0),
        stroke: Some(cutlass_models::ShapeStroke::new([0, 0, 0, 255], 6.0)),
    };
    project.add_generated(track, generator, tr(0, 100)).unwrap();
    let scene = resolve(&project, rt(5)).unwrap();
    let layer = &scene.layers[0];
    match &layer.source {
        LayerSource::Shape {
            params,
            stroke,
            pad,
            ..
        } => {
            assert_eq!(
                *params,
                cutlass_shapes::SdfParams::RoundedRect { radius: 8.0 }
            );
            let s = stroke.expect("stroke resolved");
            approx(s.width, 6.0);
            // Pad covers the stroke's outward half plus the AA margin.
            approx(*pad, 3.0 + 2.0 * cutlass_shapes::SDF_AA);
        }
        other => panic!("expected shape source, got {other:?}"),
    }
    match layer.size {
        SizeSpec::Fixed(size) => approx2(size, [110.0, 60.0]),
        other => panic!("expected fixed size, got {other:?}"),
    }
}

#[test]
fn animated_shape_params_sample_at_the_clip_tick() {
    use cutlass_models::{ClipParam, Easing, ParamValue, ShapeParam};

    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    let clip = project
        .add_generated(
            track,
            Generator::shape(
                Shape::Star {
                    points: 5,
                    inner_ratio: cutlass_models::Param::Constant(0.5),
                },
                [255, 255, 255, 255],
            ),
            tr(0, 100),
        )
        .unwrap();
    let width = ClipParam::Shape {
        param: ShapeParam::Width,
    };
    let fill = ClipParam::Shape {
        param: ShapeParam::Fill,
    };
    for (param, at, value) in [
        (width, 0, ParamValue::Scalar(100.0)),
        (width, 50, ParamValue::Scalar(300.0)),
        (fill, 0, ParamValue::Color([0, 0, 0, 255])),
        (fill, 50, ParamValue::Color([200, 100, 0, 255])),
    ] {
        project
            .set_param_keyframe(clip, param, rt(at), value, Easing::Linear, None)
            .unwrap();
    }

    let scene = resolve(&project, rt(25)).unwrap();
    let layer = &scene.layers[0];
    match &layer.source {
        LayerSource::Shape {
            params, fill, pad, ..
        } => {
            assert!(matches!(
                params,
                cutlass_shapes::SdfParams::Star { points: 5, .. }
            ));
            assert_eq!(*fill, [100, 50, 0, 255], "colors lerp per channel");
            // Width halfway between 100 and 300 → 200 + 2·pad quad.
            match layer.size {
                SizeSpec::Fixed(size) => approx(size[0], 200.0 + 2.0 * pad),
                other => panic!("expected fixed size, got {other:?}"),
            }
        }
        other => panic!("expected shape source, got {other:?}"),
    }
}

#[test]
fn pen_path_resolves_to_a_bitmap_scaled_path_layer() {
    use cutlass_models::{ShapePath, ShapePathPoint};

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
    project
        .add_generated(
            track,
            Generator::shape(Shape::Path(path), [0, 255, 0, 255]),
            tr(0, 100),
        )
        .unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    let layer = &scene.layers[0];
    match &layer.source {
        LayerSource::PathShape {
            path,
            fill,
            raster_scale,
            ..
        } => {
            assert_eq!(path.points.len(), 3);
            assert!(path.closed);
            assert_eq!(*fill, [0, 255, 0, 255]);
            approx(*raster_scale, 1.0); // 1080 canvas → reference scale 1
        }
        other => panic!("expected path source, got {other:?}"),
    }
    // Transform scale rides the quad, not the raster.
    assert_eq!(layer.size, SizeSpec::BitmapScaled([1.0, 1.0]));
}

#[test]
fn untouched_clip_resolves_to_identity_grade() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [255, 0, 0, 255],
            },
            tr(0, 100),
        )
        .unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    assert_eq!(scene.layers[0].color_grade, None);
}

#[test]
fn clip_adjust_and_filter_flow_into_scene_layer_grade() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    let adjust_clip = project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [255, 0, 0, 255],
            },
            tr(0, 100),
        )
        .unwrap();
    project
        .set_clip_adjustments(
            adjust_clip,
            ColorAdjustments {
                saturation: (-1.0).into(),
                ..ColorAdjustments::default()
            },
        )
        .unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    let expected = effective_grade(
        None,
        &ColorAdjustments {
            saturation: (-1.0).into(),
            ..ColorAdjustments::default()
        },
    );
    assert_eq!(scene.layers[0].color_grade, Some(expected));
    assert_eq!(scene.layers[0].color_grade.unwrap().saturation, -1.0);

    let filter_track = project.add_track(TrackKind::Sticker, "S2");
    let filter_clip = project
        .add_generated(
            filter_track,
            Generator::SolidColor {
                rgba: [0, 255, 0, 255],
            },
            tr(0, 100),
        )
        .unwrap();
    let filter = Filter {
        id: "mono".into(),
        intensity: 0.8.into(),
    };
    project
        .set_clip_filter(filter_clip, Some(filter.clone()))
        .unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    let filter_layer = scene
        .layers
        .iter()
        .find(|l| matches!(l.source, LayerSource::Solid([0, 255, 0, 255])))
        .expect("filter clip layer");
    assert_eq!(
        filter_layer.color_grade,
        Some(effective_grade(Some(&filter), &ColorAdjustments::default()))
    );
}

/// The model's validation cap and the evaluator's vertex-buffer bound
/// must agree, or a valid project could hold a star the renderer clamps.
#[test]
fn mask_and_chroma_key_reach_media_layer() {
    use cutlass_models::{ChromaKey, Mask, MaskKind};

    let mut project = Project::new("p", FPS_24);
    let media = project.add_media(video(1920, 1080));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip = project.add_clip(track, media, tr(0, 100), rt(0)).unwrap();
    project
        .set_clip_mask(clip, Some(Mask::new(MaskKind::Circle)))
        .unwrap();
    project
        .set_clip_chroma_key(
            clip,
            Some(ChromaKey {
                rgb: [0, 255, 0],
                strength: 0.5.into(),
                shadow: 0.0.into(),
            }),
        )
        .unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    assert_eq!(scene.layers.len(), 1);
    let layer = &scene.layers[0];
    let mask = layer.mask.unwrap();
    assert_eq!(mask.kind, MaskKind::Circle);
    approx2(mask.center, [0.0, 0.0]);
    approx2(mask.size, [1.0, 1.0]);
    approx(mask.rotation_rad, 0.0);
    approx(mask.roundness, 0.0);
    assert_eq!(layer.chroma_key.unwrap().rgb, [0, 255, 0]);
    assert!((layer.chroma_key.unwrap().strength - 0.5).abs() < 1e-6);
}

#[test]
fn keyframed_mask_center_samples_per_tick() {
    use cutlass_models::{LookParam, Mask, MaskKind};

    let mut project = Project::new("p", FPS_24);
    let media = project.add_media(video(1920, 1080));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip = project.add_clip(track, media, tr(0, 100), rt(0)).unwrap();
    project
        .set_clip_mask(clip, Some(Mask::new(MaskKind::Circle)))
        .unwrap();
    let center = ClipParam::Look {
        param: LookParam::MaskCenter,
    };
    project
        .set_param_keyframe(
            clip,
            center,
            rt(0),
            ParamValue::Vec2([0.0, 0.0]),
            Easing::Linear,
            None,
        )
        .unwrap();
    project
        .set_param_keyframe(
            clip,
            center,
            rt(40),
            ParamValue::Vec2([0.5, -0.25]),
            Easing::Linear,
            None,
        )
        .unwrap();

    let start = resolve(&project, rt(0)).unwrap();
    approx2(start.layers[0].mask.unwrap().center, [0.0, 0.0]);
    let mid = resolve(&project, rt(20)).unwrap();
    approx2(mid.layers[0].mask.unwrap().center, [0.25, -0.125]);
    let end = resolve(&project, rt(40)).unwrap();
    approx2(end.layers[0].mask.unwrap().center, [0.5, -0.25]);
}

#[test]
fn mask_rotation_degrees_resolve_to_radians() {
    use cutlass_models::{LookParam, Mask, MaskKind};

    let mut project = Project::new("p", FPS_24);
    let media = project.add_media(video(1920, 1080));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip = project.add_clip(track, media, tr(0, 100), rt(0)).unwrap();
    project
        .set_clip_mask(clip, Some(Mask::new(MaskKind::Rectangle)))
        .unwrap();
    project
        .set_param_constant(
            clip,
            ClipParam::Look {
                param: LookParam::MaskRotation,
            },
            ParamValue::Scalar(90.0),
        )
        .unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    approx(
        scene.layers[0].mask.unwrap().rotation_rad,
        std::f32::consts::FRAC_PI_2,
    );
}

#[test]
fn generated_clip_has_no_look_fields() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [255, 0, 0, 255],
            },
            tr(0, 100),
        )
        .unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    assert_eq!(scene.layers.len(), 1);
    assert!(scene.layers[0].mask.is_none());
    assert!(scene.layers[0].chroma_key.is_none());
}

#[test]
fn cleared_mask_is_none_on_layer() {
    use cutlass_models::{Mask, MaskKind};

    let mut project = Project::new("p", FPS_24);
    let media = project.add_media(video(1920, 1080));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip = project.add_clip(track, media, tr(0, 100), rt(0)).unwrap();
    project
        .set_clip_mask(clip, Some(Mask::new(MaskKind::Rectangle)))
        .unwrap();
    project.set_clip_mask(clip, None).unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    assert!(scene.layers[0].mask.is_none());
}

#[test]
fn star_point_cap_matches_the_shapes_crate() {
    assert_eq!(
        cutlass_models::MAX_STAR_POINTS,
        cutlass_shapes::MAX_STAR_POINTS
    );
}

#[test]
fn clip_with_gaussian_blur_carries_sampled_radius() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    let clip = project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [255, 0, 0, 255],
            },
            tr(0, 100),
        )
        .unwrap();
    project.add_effect(clip, "gaussian_blur").unwrap();
    project
        .set_effect_param(clip, 0, 0, 12.0)
        .expect("set radius");

    let scene = resolve(&project, rt(5)).unwrap();
    assert_eq!(scene.layers.len(), 1);
    assert_eq!(scene.layers[0].effects.len(), 1);
    assert_eq!(scene.layers[0].effects[0].id, "gaussian_blur");
    approx(scene.layers[0].effects[0].params[0], 12.0);
}

#[test]
fn color_overlay_flattens_typed_params_in_catalog_order() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    let clip = project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [255, 0, 0, 255],
            },
            tr(0, 100),
        )
        .unwrap();
    project.add_effect(clip, "color_overlay").unwrap();
    project
        .set_param_constant(
            clip,
            ClipParam::Effect {
                effect: 0,
                param: 0,
            },
            ParamValue::Color([51, 102, 153, 204]),
        )
        .unwrap();
    project
        .set_param_constant(
            clip,
            ClipParam::Effect {
                effect: 0,
                param: 1,
            },
            ParamValue::Vec2([0.25, -0.5]),
        )
        .unwrap();
    project.set_effect_param(clip, 0, 2, 0.75).unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    let params = &scene.layers[0].effects[0].params;
    // color (4) + offset (2) + amount (1)
    assert_eq!(params.len(), 7);
    approx(params[0], 51.0 / 255.0);
    approx(params[1], 102.0 / 255.0);
    approx(params[2], 153.0 / 255.0);
    approx(params[3], 204.0 / 255.0);
    approx(params[4], 0.25);
    approx(params[5], -0.5);
    approx(params[6], 0.75);
}

#[test]
fn adjustment_lane_resolves_to_canvas_pass_above_lower_layers() {
    let mut project = Project::new("p", FPS_24);
    let base = project.add_track(TrackKind::Sticker, "S1");
    project
        .add_generated(
            base,
            Generator::SolidColor {
                rgba: [255, 0, 0, 255],
            },
            tr(0, 100),
        )
        .unwrap();
    let adjustment = project.add_track(TrackKind::Adjustment, "A1");
    let bar = project
        .add_generated(adjustment, Generator::Adjustment, tr(0, 100))
        .unwrap();
    project
        .set_clip_adjustments(
            bar,
            ColorAdjustments {
                saturation: (-1.0).into(),
                ..ColorAdjustments::default()
            },
        )
        .unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    assert_eq!(scene.layers.len(), 2);
    assert!(matches!(scene.layers[0].source, LayerSource::Solid(_)));
    assert_eq!(scene.layers[1].source, LayerSource::CanvasPass);
    assert_eq!(scene.layers[1].clip, Some(bar));
    assert_eq!(scene.layers[1].effects, Vec::<ResolvedPass>::new());
    assert_eq!(
        scene.layers[1].color_grade,
        Some(effective_grade(
            None,
            &ColorAdjustments {
                saturation: (-1.0).into(),
                ..ColorAdjustments::default()
            },
        ))
    );
}

#[test]
fn filter_lane_resolves_to_canvas_pass_grade() {
    let mut project = Project::new("p", FPS_24);
    let base = project.add_track(TrackKind::Sticker, "S1");
    project
        .add_generated(
            base,
            Generator::SolidColor {
                rgba: [255, 0, 0, 255],
            },
            tr(0, 100),
        )
        .unwrap();
    let filter_track = project.add_track(TrackKind::Filter, "F1");
    let bar = project
        .add_generated(filter_track, Generator::Filter, tr(0, 100))
        .unwrap();
    let filter = Filter {
        id: "mono".into(),
        intensity: 1.0.into(),
    };
    project.set_clip_filter(bar, Some(filter.clone())).unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    assert_eq!(scene.layers.len(), 2);
    assert_eq!(scene.layers[1].source, LayerSource::CanvasPass);
    assert_eq!(
        scene.layers[1].color_grade,
        Some(effective_grade(Some(&filter), &ColorAdjustments::default()))
    );
}

#[test]
fn effect_lane_resolves_to_canvas_pass_effect_chain() {
    let mut project = Project::new("p", FPS_24);
    let base = project.add_track(TrackKind::Sticker, "S1");
    project
        .add_generated(
            base,
            Generator::SolidColor {
                rgba: [255, 0, 0, 255],
            },
            tr(0, 100),
        )
        .unwrap();
    let effect_track = project.add_track(TrackKind::Effect, "E1");
    let bar = project
        .add_generated(effect_track, Generator::Effect, tr(0, 100))
        .unwrap();
    project.add_effect(bar, "gaussian_blur").unwrap();
    project.set_effect_param(bar, 0, 0, 8.0).unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    assert_eq!(scene.layers.len(), 2);
    assert_eq!(scene.layers[1].source, LayerSource::CanvasPass);
    assert_eq!(scene.layers[1].color_grade, None);
    assert_eq!(scene.layers[1].effects.len(), 1);
    assert_eq!(scene.layers[1].effects[0].id, "gaussian_blur");
    approx(scene.layers[1].effects[0].params[0], 8.0);
}

#[test]
fn identity_or_inactive_lane_passes_are_elided() {
    let mut project = Project::new("p", FPS_24);
    let base = project.add_track(TrackKind::Sticker, "S1");
    project
        .add_generated(
            base,
            Generator::SolidColor {
                rgba: [255, 0, 0, 255],
            },
            tr(0, 100),
        )
        .unwrap();
    let adjustment = project.add_track(TrackKind::Adjustment, "A1");
    project
        .add_generated(adjustment, Generator::Adjustment, tr(0, 100))
        .unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    assert_eq!(scene.layers.len(), 1, "neutral pass should not render");

    let bar = project
        .add_generated(adjustment, Generator::Adjustment, tr(120, 10))
        .unwrap();
    project
        .set_clip_adjustments(
            bar,
            ColorAdjustments {
                exposure: 1.0.into(),
                ..ColorAdjustments::default()
            },
        )
        .unwrap();
    let scene = resolve(&project, rt(5)).unwrap();
    assert_eq!(scene.layers.len(), 1, "out-of-range pass should not render");

    project
        .timeline_mut()
        .track_mut(adjustment)
        .unwrap()
        .enabled = false;
    let scene = resolve(&project, rt(125)).unwrap();
    assert!(
        scene.layers.is_empty(),
        "disabled pass track should not render"
    );
}

#[test]
fn gesture_partitions_fall_back_when_canvas_pass_sits_above_clip() {
    let mut project = Project::new("p", FPS_24);
    let base = project.add_track(TrackKind::Sticker, "S1");
    let clip = project
        .add_generated(
            base,
            Generator::SolidColor {
                rgba: [255, 0, 0, 255],
            },
            tr(0, 100),
        )
        .unwrap();
    let adjustment = project.add_track(TrackKind::Adjustment, "A1");
    let bar = project
        .add_generated(adjustment, Generator::Adjustment, tr(0, 100))
        .unwrap();
    project
        .set_clip_adjustments(
            bar,
            ColorAdjustments {
                exposure: 1.0.into(),
                ..ColorAdjustments::default()
            },
        )
        .unwrap();

    let partitions = resolve_gesture_partitions(&project, rt(5), clip).unwrap();
    assert!(partitions.is_none());
}

#[test]
fn abutting_clips_crossfade_at_midpoint() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    let left = project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [255, 0, 0, 255],
            },
            tr(0, 24),
        )
        .unwrap();
    project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [0, 0, 255, 255],
            },
            tr(24, 24),
        )
        .unwrap();
    project.add_transition(left, "crossfade").unwrap();
    project.set_transition_duration(left, 24).unwrap();

    // Cut at 24; window [12, 36); midpoint tick 24 → progress 0.5.
    let scene = resolve(&project, rt(24)).unwrap();
    assert_eq!(scene.layers.len(), 1);
    match &scene.layers[0].source {
        LayerSource::Transition {
            transition_id,
            progress,
            ..
        } => {
            assert_eq!(transition_id, "crossfade");
            approx(*progress, 0.5);
        }
        other => panic!("expected transition layer, got {other:?}"),
    }
}

#[test]
fn transition_between_canvas_passes_falls_back_to_plain_clip() {
    // Legacy project files may carry transitions on effect/filter/
    // adjustment lanes (the model now rejects creating them). Both sides
    // resolve to CanvasPass, which the renderer can't nest — the resolver
    // must skip the transition and resolve the track normally, not error.
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Adjustment, "A1");
    let left = project
        .add_generated(track, Generator::Adjustment, tr(0, 24))
        .unwrap();
    let right = project
        .add_generated(track, Generator::Adjustment, tr(24, 24))
        .unwrap();
    for clip in [left, right] {
        project
            .set_clip_adjustments(
                clip,
                ColorAdjustments {
                    exposure: 1.0.into(),
                    ..ColorAdjustments::default()
                },
            )
            .unwrap();
    }
    // Inject the transition through the persistence path, the way a
    // project saved before the model-level guard would carry it.
    let mut doc = serde_json::to_value(&project).unwrap();
    doc["timeline"]["tracks"][0][1]["transitions"] = serde_json::json!([{
        "left": left.raw(),
        "right": right.raw(),
        "transition_id": "crossfade",
        "duration": 24,
    }]);
    let project: Project = serde_json::from_value(doc).unwrap();

    // Midpoint of the window [12, 36): no Transition layer, just the
    // plain canvas pass of the clip under the playhead.
    let scene = resolve(&project, rt(24)).unwrap();
    assert_eq!(scene.layers.len(), 1);
    assert!(matches!(scene.layers[0].source, LayerSource::CanvasPass));
}

#[test]
fn outside_transition_window_is_single_clip() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    let left = project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [255, 0, 0, 255],
            },
            tr(0, 24),
        )
        .unwrap();
    project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [0, 0, 255, 255],
            },
            tr(24, 24),
        )
        .unwrap();
    project.add_transition(left, "crossfade").unwrap();

    let scene = resolve(&project, rt(0)).unwrap();
    assert_eq!(scene.layers.len(), 1);
    assert!(matches!(scene.layers[0].source, LayerSource::Solid(_)));
}

#[test]
fn gesture_partitions_split_stack_and_apply_identity_to_sprite() {
    let mut project = Project::new("p", FPS_24);
    let media_a = project.add_media(video(1920, 1080));
    let media_b = project.add_media(video(1280, 720));
    let bottom_track = project.add_track(TrackKind::Video, "V1");
    let top_track = project.add_track(TrackKind::Video, "V2");
    let bottom = project
        .add_clip(bottom_track, media_a, tr(0, 100), rt(0))
        .unwrap();
    let top = project
        .add_clip(top_track, media_b, tr(0, 100), rt(0))
        .unwrap();

    let parts = resolve_gesture_partitions(&project, rt(5), top)
        .unwrap()
        .expect("top clip should partition");
    assert_eq!(parts.below.layers.len(), 1);
    assert_eq!(parts.below.layers[0].clip, Some(bottom));
    assert_eq!(parts.sprite.layers.len(), 1);
    assert_eq!(parts.sprite.layers[0].clip, Some(top));
    assert!(parts.above.layers.is_empty());
    assert_eq!(parts.below.background[3], 255);
    assert_eq!(parts.sprite.background, [0, 0, 0, 0]);
    assert_eq!(parts.above.background, [0, 0, 0, 0]);
    approx2(parts.sprite.layers[0].center, [960.0, 540.0]);
    match parts.sprite.layers[0].size {
        SizeSpec::Fixed(size) => {
            // 1280×720 aspect-fits into 1920×1080, then × identity scale 1.
            approx(size[0], 1920.0);
            approx(size[1], 1080.0);
        }
        other => panic!("expected fixed size, got {other:?}"),
    }
}

#[test]
fn gesture_partitions_return_none_inside_transition_window() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    let left = project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [255, 0, 0, 255],
            },
            tr(0, 24),
        )
        .unwrap();
    project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [0, 0, 255, 255],
            },
            tr(24, 24),
        )
        .unwrap();
    project.add_transition(left, "crossfade").unwrap();

    assert!(
        resolve_gesture_partitions(&project, rt(12), left)
            .unwrap()
            .is_none()
    );
}

#[test]
fn video_clip_blend_mode_reaches_scene_layer() {
    let mut project = Project::new("p", FPS_24);
    let media = project.add_media(video(1920, 1080));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip = project.add_clip(track, media, tr(0, 100), rt(0)).unwrap();
    project.set_blend_mode(clip, BlendMode::Multiply).unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    assert_eq!(scene.layers.len(), 1);
    assert_eq!(scene.layers[0].blend_mode, BlendMode::Multiply);
}

#[test]
fn text_generator_blend_mode_reaches_scene_layer() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Text, "T1");
    let clip = project
        .add_generated(
            track,
            Generator::Text {
                content: "Hi".into(),
                style: ModelTextStyle::default(),
            },
            tr(0, 100),
        )
        .unwrap();
    project.set_blend_mode(clip, BlendMode::Screen).unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    assert_eq!(scene.layers.len(), 1);
    assert!(matches!(scene.layers[0].source, LayerSource::Text { .. }));
    assert_eq!(scene.layers[0].blend_mode, BlendMode::Screen);
}

#[test]
fn adjustment_lane_blend_mode_stays_normal_on_canvas_pass() {
    let mut project = Project::new("p", FPS_24);
    let base = project.add_track(TrackKind::Sticker, "S1");
    project
        .add_generated(
            base,
            Generator::SolidColor {
                rgba: [255, 0, 0, 255],
            },
            tr(0, 100),
        )
        .unwrap();
    let adjustment = project.add_track(TrackKind::Adjustment, "A1");
    let bar = project
        .add_generated(adjustment, Generator::Adjustment, tr(0, 100))
        .unwrap();
    project
        .set_clip_adjustments(
            bar,
            ColorAdjustments {
                saturation: (-1.0).into(),
                ..ColorAdjustments::default()
            },
        )
        .unwrap();
    // Even if the bar carries a blend mode, canvas passes must stay Normal.
    project.set_blend_mode(bar, BlendMode::Multiply).unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    assert_eq!(scene.layers.len(), 2);
    assert_eq!(scene.layers[1].source, LayerSource::CanvasPass);
    assert_eq!(scene.layers[1].blend_mode, BlendMode::Normal);
}

#[test]
fn shadow_blur_scales_to_canvas_pixels_on_tall_canvas() {
    // Auto canvas follows the 3840×2160 media → ch=2160 → px = 2× at scale 1.
    let mut project = Project::new("p", FPS_24);
    let media = project.add_media(video(3840, 2160));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip = project.add_clip(track, media, tr(0, 100), rt(0)).unwrap();
    project
        .set_layer_styles(
            clip,
            LayerStyles {
                shadow: Some(LayerShadow {
                    rgba: Param::Constant([0, 0, 0, 128]),
                    offset: Param::Constant([4.0, 4.0]),
                    blur: Param::Constant(8.0),
                }),
                ..Default::default()
            },
        )
        .unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    assert_eq!((scene.width, scene.height), (3840, 2160));
    assert_eq!(scene.layers.len(), 1);
    let styles = scene.layers[0].styles.as_ref().expect("styles");
    assert_eq!(
        styles.shadow,
        Some(SceneShadow {
            rgba: [0, 0, 0, 128],
            offset: [8.0, 8.0],
            blur: 16.0,
        })
    );
}

#[test]
fn keyframed_glow_radius_samples_per_tick() {
    let mut project = Project::new("p", FPS_24);
    let media = project.add_media(video(1920, 1080));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip = project.add_clip(track, media, tr(0, 100), rt(0)).unwrap();
    project
        .set_layer_styles(
            clip,
            LayerStyles {
                glow: Some(LayerGlow {
                    rgba: Param::Constant([255, 255, 255, 255]),
                    radius: Param::Constant(0.0),
                    intensity: Param::Constant(1.0),
                }),
                ..Default::default()
            },
        )
        .unwrap();
    let radius = ClipParam::Style {
        param: StyleParam::GlowRadius,
    };
    project
        .set_param_keyframe(
            clip,
            radius,
            rt(0),
            ParamValue::Scalar(4.0),
            Easing::Linear,
            None,
        )
        .unwrap();
    project
        .set_param_keyframe(
            clip,
            radius,
            rt(40),
            ParamValue::Scalar(20.0),
            Easing::Linear,
            None,
        )
        .unwrap();

    // Midpoint of the segment (local tick 20) → radius 12 ref px → 12 canvas px.
    let mid = resolve(&project, rt(20)).unwrap();
    approx(
        mid.layers[0]
            .styles
            .as_ref()
            .unwrap()
            .glow
            .as_ref()
            .unwrap()
            .radius,
        12.0,
    );
    let end = resolve(&project, rt(40)).unwrap();
    approx(
        end.layers[0]
            .styles
            .as_ref()
            .unwrap()
            .glow
            .as_ref()
            .unwrap()
            .radius,
        20.0,
    );
}

#[test]
fn transform_scale_doubles_outline_width() {
    let mut project = Project::new("p", FPS_24);
    let media = project.add_media(video(1920, 1080));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip = project.add_clip(track, media, tr(0, 100), rt(0)).unwrap();
    project
        .set_layer_styles(
            clip,
            LayerStyles {
                outline: Some(LayerOutline {
                    rgba: Param::Constant([255, 255, 255, 255]),
                    width: Param::Constant(4.0),
                }),
                ..Default::default()
            },
        )
        .unwrap();
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
    approx(
        scene.layers[0]
            .styles
            .as_ref()
            .unwrap()
            .outline
            .as_ref()
            .unwrap()
            .width,
        8.0,
    );
}

#[test]
fn clip_without_styles_resolves_styles_none() {
    let mut project = Project::new("p", FPS_24);
    let media = project.add_media(video(1920, 1080));
    let track = project.add_track(TrackKind::Video, "V1");
    project.add_clip(track, media, tr(0, 100), rt(0)).unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    assert_eq!(scene.layers.len(), 1);
    assert!(scene.layers[0].styles.is_none());
}

#[test]
fn resolve_with_substitutes_styles_for_one_clip() {
    let mut project = Project::new("p", FPS_24);
    let media = project.add_media(video(1920, 1080));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip = project.add_clip(track, media, tr(0, 100), rt(0)).unwrap();
    project
        .set_layer_styles(
            clip,
            LayerStyles {
                shadow: Some(LayerShadow {
                    blur: Param::Constant(8.0),
                    ..LayerShadow::default()
                }),
                ..Default::default()
            },
        )
        .unwrap();

    let live = LayerStyles {
        shadow: Some(LayerShadow {
            blur: Param::Constant(24.0),
            ..LayerShadow::default()
        }),
        ..Default::default()
    };
    let overrides = ResolveOverrides {
        transform: None,
        generator: None,
        look: None,
        styles: Some((clip, &live)),
    };
    let scene = resolve_with(&project, rt(5), overrides).unwrap();
    approx(
        scene.layers[0]
            .styles
            .as_ref()
            .unwrap()
            .shadow
            .as_ref()
            .unwrap()
            .blur,
        24.0,
    );

    // Plain resolve still sees only committed state.
    let scene = resolve(&project, rt(5)).unwrap();
    approx(
        scene.layers[0]
            .styles
            .as_ref()
            .unwrap()
            .shadow
            .as_ref()
            .unwrap()
            .blur,
        8.0,
    );
}

#[test]
fn adjustment_lane_styles_stay_none_on_canvas_pass() {
    let mut project = Project::new("p", FPS_24);
    let base = project.add_track(TrackKind::Sticker, "S1");
    project
        .add_generated(
            base,
            Generator::SolidColor {
                rgba: [255, 0, 0, 255],
            },
            tr(0, 100),
        )
        .unwrap();
    let adjustment = project.add_track(TrackKind::Adjustment, "A1");
    let bar = project
        .add_generated(adjustment, Generator::Adjustment, tr(0, 100))
        .unwrap();
    project
        .set_clip_adjustments(
            bar,
            ColorAdjustments {
                saturation: (-1.0).into(),
                ..ColorAdjustments::default()
            },
        )
        .unwrap();
    project
        .set_layer_styles(
            bar,
            LayerStyles {
                shadow: Some(LayerShadow::default()),
                ..Default::default()
            },
        )
        .unwrap();

    let scene = resolve(&project, rt(5)).unwrap();
    assert_eq!(scene.layers.len(), 2);
    assert_eq!(scene.layers[1].source, LayerSource::CanvasPass);
    assert!(scene.layers[1].styles.is_none());
}
