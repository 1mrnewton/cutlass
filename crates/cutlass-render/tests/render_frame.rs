//! End-to-end smoke test: resolve a generator-only project and composite it on
//! a headless GPU. Skips cleanly when no GPU adapter is available (CI).

use std::path::Path;

use cutlass_compositor::{ColorGrade, GpuContext};
use cutlass_models::{
    BlendMode, CanvasAspect, CanvasSettings, ChromaKey, ClipTransform, ColorAdjustments, Filter,
    Generator, LayerBackground, LayerShadow, LayerStyles, Mask, MaskKind, MediaSource, Param,
    Project, Rational, RationalTime, Shape, ShapePath, ShapePathPoint, TextStyle, TimeRange,
    TrackKind,
};
use cutlass_render::Renderer;

const FPS_24: Rational = Rational::FPS_24;

fn rt(value: i64) -> RationalTime {
    RationalTime::new(value, FPS_24)
}

fn write_solid_png(path: &Path, width: u32, height: u32, rgba: [u8; 4]) {
    let file = std::fs::File::create(path).expect("create png");
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().expect("png header");
    let flat: Vec<u8> = rgba
        .iter()
        .copied()
        .cycle()
        .take((width * height * 4) as usize)
        .collect();
    writer.write_image_data(&flat).expect("png data");
}

fn assert_near(actual: [u8; 4], expected: [u8; 4], tolerance: u8, what: &str) {
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!(
            a.abs_diff(*e) <= tolerance,
            "{what}: got {actual:?}, expected ~{expected:?} (±{tolerance})"
        );
    }
}

/// CPU mirror of `grade.wgsl`'s `apply_grade`, for tolerance comparisons.
fn grade_ref_u8(rgba: [u8; 4], grade: ColorGrade) -> [u8; 4] {
    let mut c = [
        f32::from(rgba[0]) / 255.0,
        f32::from(rgba[1]) / 255.0,
        f32::from(rgba[2]) / 255.0,
    ];
    c[0] *= 2f32.powf(2.0 * grade.exposure);
    c[1] *= 2f32.powf(2.0 * grade.exposure);
    c[2] *= 2f32.powf(2.0 * grade.exposure);
    c[0] += 0.25 * grade.temperature;
    c[2] -= 0.25 * grade.temperature;
    c[1] += 0.25 * grade.tint;
    for ch in &mut c {
        *ch += 0.25 * grade.brightness;
    }
    for ch in &mut c {
        *ch = (*ch - 0.5) * (1.0 + grade.contrast) + 0.5;
    }
    let luma = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    let sat = 1.0 + grade.saturation;
    c[0] = luma + (c[0] - luma) * sat;
    c[1] = luma + (c[1] - luma) * sat;
    c[2] = luma + (c[2] - luma) * sat;
    let quant = |x: f32| (x.clamp(0.0, 1.0) * 255.0).round() as u8;
    [quant(c[0]), quant(c[1]), quant(c[2]), rgba[3]]
}

fn pixel_tol(gpu: &GpuContext) -> i32 {
    if gpu.is_software() { 8 } else { 3 }
}

fn assert_px_close(got: [u8; 4], expect: [u8; 4], tol: i32, label: &str) {
    for ch in 0..4 {
        let d = i32::from(got[ch]) - i32::from(expect[ch]);
        assert!(
            d.abs() <= tol,
            "{label}: got {got:?}, expected ~{expect:?} (channel {ch} off by {d}, tol {tol})"
        );
    }
}

#[test]
fn renders_still_with_circle_mask_and_chroma_key() {
    let dir = tempfile::tempdir().unwrap();
    let png_path = dir.path().join("green.png");
    write_solid_png(&png_path, 320, 180, [0, 255, 0, 255]);

    let mut project = Project::new("p", FPS_24);
    let media = project.add_media(MediaSource::image(&png_path, 320, 180));
    let window = project.media(media).unwrap().full_range();
    let track = project.add_track(TrackKind::Video, "V1");
    let clip = project.add_clip(track, media, window, rt(0)).unwrap();
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

    let Ok(mut renderer) = Renderer::new_headless() else {
        eprintln!("skipping: no headless GPU available");
        return;
    };
    let frame = renderer
        .render_frame_fit(&project, rt(0), 640, 640)
        .expect("render masked still");

    // Center is inside the circle but green is keyed out → black background.
    assert_near(frame.pixel(320, 180), [0, 0, 0, 255], 8, "center");
    // Corner is outside the circle → background.
    assert_near(frame.pixel(10, 10), [0, 0, 0, 255], 8, "corner");
}

#[test]
fn renders_still_with_half_size_circle_mask() {
    // Square canvas + square still fills the frame; fit_within keeps the
    // resolve→realize→composite path cheap for CI.
    let dir = tempfile::tempdir().unwrap();
    let png_path = dir.path().join("red.png");
    write_solid_png(&png_path, 200, 200, [255, 0, 0, 255]);

    let mut project = Project::new("p", FPS_24);
    project.timeline_mut().set_canvas(CanvasSettings {
        aspect: CanvasAspect::Square1x1,
        background: [0, 0, 0],
    });
    let media = project.add_media(MediaSource::image(&png_path, 200, 200));
    let window = project.media(media).unwrap().full_range();
    let track = project.add_track(TrackKind::Video, "V1");
    let clip = project.add_clip(track, media, window, rt(0)).unwrap();
    let mut mask = Mask::new(MaskKind::Circle);
    mask.size = Param::Constant([0.5, 0.5]);
    project.set_clip_mask(clip, Some(mask)).unwrap();

    let Ok(mut renderer) = Renderer::new_headless() else {
        eprintln!("skipping: no headless GPU available");
        return;
    };
    let frame = renderer
        .render_frame_fit(&project, rt(0), 200, 200)
        .expect("render half-size mask");

    // Center stays red; ~0.7×half-width falls outside size 0.5 → background.
    let cx = frame.width / 2;
    let cy = frame.height / 2;
    let probe_x = cx + ((0.7 * (frame.width as f32 * 0.5)).round() as u32);
    assert_near(frame.pixel(cx, cy), [255, 0, 0, 255], 8, "center");
    assert_near(frame.pixel(probe_x, cy), [0, 0, 0, 255], 8, "mid-radius");
}

#[test]
fn renders_adjustment_lane_over_still_media() {
    let dir = tempfile::tempdir().unwrap();
    let png_path = dir.path().join("red.png");
    write_solid_png(&png_path, 320, 180, [255, 0, 0, 255]);

    let mut project = Project::new("p", FPS_24);
    let media = project.add_media(MediaSource::image(&png_path, 320, 180));
    let window = project.media(media).unwrap().full_range();
    let video = project.add_track(TrackKind::Video, "V1");
    project.add_clip(video, media, window, rt(0)).unwrap();
    let adjustment = project.add_track(TrackKind::Adjustment, "A1");
    let bar = project
        .add_generated(
            adjustment,
            Generator::Adjustment,
            TimeRange::at_rate(0, 100, FPS_24),
        )
        .unwrap();
    let grade = ColorGrade {
        saturation: -1.0,
        ..ColorGrade::IDENTITY
    };
    project
        .set_clip_adjustments(
            bar,
            ColorAdjustments {
                saturation: (-1.0).into(),
                ..ColorAdjustments::default()
            },
        )
        .unwrap();

    let Ok(mut renderer) = Renderer::new_headless() else {
        eprintln!("skipping: no headless GPU available");
        return;
    };
    let frame = renderer
        .render_frame_fit(&project, rt(0), 320, 180)
        .expect("render adjustment lane");

    assert_px_close(
        frame.pixel(160, 90),
        grade_ref_u8([255, 0, 0, 255], grade),
        8,
        "adjusted center",
    );
}

fn red_solid_project() -> (Project, cutlass_models::ClipId) {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    let clip = project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [255, 0, 0, 255],
            },
            TimeRange::at_rate(0, 100, FPS_24),
        )
        .unwrap();
    (project, clip)
}

#[test]
fn renders_a_solid_generator_to_a_red_canvas() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [255, 0, 0, 255],
            },
            TimeRange::at_rate(0, 100, FPS_24),
        )
        .unwrap();

    let Ok(mut renderer) = Renderer::new_headless() else {
        eprintln!("skipping: no headless GPU available");
        return;
    };

    let image = renderer.render_frame(&project, rt(0)).expect("render");
    assert_eq!((image.width, image.height), (1920, 1080));
    assert_eq!(image.pixels.len(), 1920 * 1080 * 4);

    // The solid fills the whole canvas, so every corner should read red.
    let top_left = &image.pixels[0..4];
    assert!(
        top_left[0] > 240 && top_left[1] < 16 && top_left[2] < 16,
        "expected a red canvas, got {top_left:?}"
    );
}

#[test]
fn gesture_sprite_contains_unwrapped_text_pixels() {
    let mut project = Project::new("text-gesture", FPS_24);
    let track = project.add_track(TrackKind::Text, "T1");
    let clip = project
        .add_generated(
            track,
            Generator::Text {
                content: "hello worldhello worldhello worldhello world".into(),
                style: TextStyle {
                    font: "Micro 5".into(),
                    size: 56.0.into(),
                    letter_spacing: 30.0.into(),
                    wrap: false,
                    ..TextStyle::default()
                },
            },
            TimeRange::at_rate(0, 100, FPS_24),
        )
        .unwrap();

    let Ok(mut renderer) = Renderer::new_headless() else {
        eprintln!("skipping: no headless GPU available");
        return;
    };
    renderer.load_font(include_bytes!("../../cutlass-text/assets/Micro5-Regular.ttf").to_vec());
    let frames = renderer
        .render_gesture_frames(&project, rt(0), clip, 960, 540)
        .expect("render gesture frames")
        .expect("text clip should support sprite partitioning");

    let visible = frames
        .sprite
        .pixels
        .chunks_exact(4)
        .filter(|pixel| pixel[3] > 0)
        .count();
    assert!(
        visible > 100,
        "gesture sprite was transparent ({visible} pixels)"
    );
}

#[test]
fn renders_an_ellipse_through_the_sdf_pipeline() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    project
        .add_generated(
            track,
            Generator::shape(Shape::Ellipse, [0, 255, 0, 255]),
            TimeRange::at_rate(0, 100, FPS_24),
        )
        .unwrap();

    let Ok(mut renderer) = Renderer::new_headless() else {
        eprintln!("skipping: no headless GPU available");
        return;
    };
    let image = renderer.render_frame(&project, rt(0)).expect("render");

    // Drop size 200×200 centered on the 1920×1080 canvas: the center is
    // inside the ellipse, a point 105px right of center is outside it, and
    // the canvas corner is untouched background (black).
    let center = image.pixel(960, 540);
    assert!(
        center[1] > 240 && center[0] < 16,
        "ellipse center should be green, got {center:?}"
    );
    let outside = image.pixel(960 + 105, 540);
    assert!(
        outside[1] < 16,
        "outside the ellipse should be background, got {outside:?}"
    );
    // On the horizontal axis 90px out (inside the 100px semi-axis).
    let on_axis = image.pixel(960 + 90, 540);
    assert!(on_axis[1] > 240, "90px along +x is inside, got {on_axis:?}");
}

#[test]
fn renders_a_pen_path_through_the_bitmap_pipeline() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    // A 160×160 diamond around the origin.
    let path = ShapePath {
        points: vec![
            ShapePathPoint::corner([0.0, -80.0]),
            ShapePathPoint::corner([80.0, 0.0]),
            ShapePathPoint::corner([0.0, 80.0]),
            ShapePathPoint::corner([-80.0, 0.0]),
        ],
        closed: true,
    };
    project
        .add_generated(
            track,
            Generator::shape(Shape::Path(path), [0, 128, 255, 255]),
            TimeRange::at_rate(0, 100, FPS_24),
        )
        .unwrap();

    let Ok(mut renderer) = Renderer::new_headless() else {
        eprintln!("skipping: no headless GPU available");
        return;
    };
    let image = renderer.render_frame(&project, rt(0)).expect("render");

    let center = image.pixel(960, 540);
    assert!(
        center[2] > 240 && center[1] > 100,
        "diamond center should be the fill color, got {center:?}"
    );
    // The diamond's corner region (70, 70) from center is outside the fill.
    let outside = image.pixel(960 + 70, 540 + 70);
    assert!(
        outside[2] < 16,
        "outside the diamond should be background, got {outside:?}"
    );
}

#[test]
fn pixelate_effect_blockifies_a_solid_fill() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    let clip = project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [200, 100, 50, 255],
            },
            TimeRange::at_rate(0, 100, FPS_24),
        )
        .unwrap();
    project.add_effect(clip, "pixelate").unwrap();
    project.set_effect_param(clip, 0, 0, 24.0).unwrap();

    let Ok(mut renderer) = Renderer::new_headless() else {
        eprintln!("skipping: no headless GPU available");
        return;
    };
    let image = renderer.render_frame(&project, rt(0)).expect("render");
    // Blocky pixelate on a uniform fill still renders; center stays the fill color.
    let center = image.pixel(960, 540);
    assert!(center[0] > 180 && center[1] > 80);
}

#[test]
fn crossfade_transition_blends_two_solids_at_midpoint() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    let left = project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [255, 0, 0, 255],
            },
            TimeRange::at_rate(0, 24, FPS_24),
        )
        .unwrap();
    project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [0, 0, 255, 255],
            },
            TimeRange::at_rate(24, 24, FPS_24),
        )
        .unwrap();
    project.add_transition(left, "crossfade").unwrap();
    project.set_transition_duration(left, 24).unwrap();

    let Ok(mut renderer) = Renderer::new_headless() else {
        eprintln!("skipping: no headless GPU available");
        return;
    };
    let image = renderer.render_frame(&project, rt(24)).expect("render");
    let px = image.pixel(960, 540);
    // Mid blend of red + blue → purple-ish, not pure red or blue.
    assert!(
        px[0] > 80 && px[2] > 80,
        "midpoint should blend channels, got {px:?}"
    );
    assert!(px[0] < 240 && px[2] < 240);
}

#[test]
fn saturation_minus_one_desaturates_red_solid() {
    let Ok(gpu) = GpuContext::new_headless_blocking() else {
        eprintln!("skipping: no headless GPU available");
        return;
    };
    let tol = pixel_tol(&gpu);

    let (mut project, clip) = red_solid_project();
    project
        .set_clip_adjustments(
            clip,
            ColorAdjustments {
                saturation: (-1.0).into(),
                ..ColorAdjustments::default()
            },
        )
        .unwrap();

    let Ok(mut renderer) = Renderer::new_headless() else {
        eprintln!("skipping: no headless GPU available");
        return;
    };

    let image = renderer.render_frame(&project, rt(0)).expect("render");
    let grade = ColorGrade {
        saturation: -1.0,
        ..ColorGrade::IDENTITY
    };
    let expect = grade_ref_u8([255, 0, 0, 255], grade);
    let top_left = image.pixel(0, 0);
    assert_px_close(top_left, expect, tol, "desaturated red solid");
}

#[test]
fn transition_with_effected_clip_renders_without_panic() {
    // Regression: packing skipped nested transition-side effect chains, so
    // realizing a transition whose clips carry effects indexed past the
    // instance store and panicked.
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    let left = project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [255, 0, 0, 255],
            },
            TimeRange::at_rate(0, 24, FPS_24),
        )
        .unwrap();
    project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [0, 0, 255, 255],
            },
            TimeRange::at_rate(24, 24, FPS_24),
        )
        .unwrap();
    project.add_effect(left, "gaussian_blur").unwrap();
    project.add_transition(left, "crossfade").unwrap();
    project.set_transition_duration(left, 24).unwrap();

    let Ok(mut renderer) = Renderer::new_headless() else {
        eprintln!("skipping: no headless GPU available");
        return;
    };
    // Midpoint of the transition window [12, 36).
    let frame = renderer
        .render_frame_fit(&project, rt(24), 64, 64)
        .expect("render transition with effected clip");
    assert!(frame.width > 0 && frame.height > 0);
}

#[test]
fn mono_filter_at_zero_intensity_matches_no_filter() {
    let Ok(gpu) = GpuContext::new_headless_blocking() else {
        eprintln!("skipping: no headless GPU available");
        return;
    };
    let tol = pixel_tol(&gpu);

    let (mut project, clip) = red_solid_project();
    project
        .set_clip_filter(
            clip,
            Some(Filter {
                id: "mono".into(),
                intensity: 0.0.into(),
            }),
        )
        .unwrap();

    let Ok(mut renderer) = Renderer::new_headless() else {
        eprintln!("skipping: no headless GPU available");
        return;
    };

    let graded = renderer.render_frame(&project, rt(0)).expect("render");

    let (project, _) = red_solid_project();
    let baseline = renderer.render_frame(&project, rt(0)).expect("render");

    let top_left = graded.pixel(0, 0);
    let baseline_px = baseline.pixel(0, 0);
    assert_px_close(top_left, baseline_px, tol, "mono filter intensity 0");
}

#[test]
fn multiply_red_solid_over_green_solid_is_black() {
    let mut project = Project::new("p", FPS_24);
    let bottom = project.add_track(TrackKind::Sticker, "S1");
    project
        .add_generated(
            bottom,
            Generator::SolidColor {
                rgba: [0, 255, 0, 255],
            },
            TimeRange::at_rate(0, 100, FPS_24),
        )
        .unwrap();
    let top = project.add_track(TrackKind::Sticker, "S2");
    let red = project
        .add_generated(
            top,
            Generator::SolidColor {
                rgba: [255, 0, 0, 255],
            },
            TimeRange::at_rate(0, 100, FPS_24),
        )
        .unwrap();
    project.set_blend_mode(red, BlendMode::Multiply).unwrap();

    let Ok(mut renderer) = Renderer::new_headless() else {
        eprintln!("skipping: no headless GPU available");
        return;
    };
    let image = renderer.render_frame(&project, rt(0)).expect("render");
    let center = image.pixel(960, 540);
    assert!(
        center[0] < 16 && center[1] < 16 && center[2] < 16 && center[3] > 240,
        "multiply red over green should be ~black, got {center:?}"
    );
}

#[test]
fn red_solid_with_white_background_plate_shows_padding() {
    // Default solids fill the canvas; scale down so the padded plate has a
    // probeable band outside the content on a black canvas.
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    let clip = project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [255, 0, 0, 255],
            },
            TimeRange::at_rate(0, 100, FPS_24),
        )
        .unwrap();
    project
        .set_transform(
            clip,
            ClipTransform {
                scale: 0.5,
                ..ClipTransform::IDENTITY
            },
            None,
        )
        .unwrap();
    project
        .set_layer_styles(
            clip,
            LayerStyles {
                background: Some(LayerBackground {
                    rgba: Param::Constant([255, 255, 255, 255]),
                    padding: Param::Constant(20.0),
                    radius: Param::Constant(0.0),
                }),
                ..Default::default()
            },
        )
        .unwrap();

    let Ok(mut renderer) = Renderer::new_headless() else {
        eprintln!("skipping: no headless GPU available");
        return;
    };
    let image = renderer.render_frame(&project, rt(0)).expect("render");

    // Content is 960×540 centered; canvas padding = 20 × 0.5 = 10px.
    // x=1445 sits ~5px into the plate past the content's right edge.
    let plate = image.pixel(1445, 540);
    assert_near(plate, [255, 255, 255, 255], 3, "background plate padding");
    let center = image.pixel(960, 540);
    assert_near(center, [255, 0, 0, 255], 3, "solid content center");
    let corner = image.pixel(10, 10);
    assert_near(corner, [0, 0, 0, 255], 3, "canvas outside plate");
}

#[test]
fn text_clip_with_layer_shadow_composites() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Text, "T1");
    let clip = project
        .add_generated(
            track,
            Generator::Text {
                content: "styles".into(),
                style: TextStyle {
                    font: "Micro 5".into(),
                    size: 72.0.into(),
                    ..TextStyle::default()
                },
            },
            TimeRange::at_rate(0, 100, FPS_24),
        )
        .unwrap();
    project
        .set_layer_styles(
            clip,
            LayerStyles {
                shadow: Some(LayerShadow {
                    rgba: Param::Constant([0, 0, 0, 180]),
                    offset: Param::Constant([8.0, 8.0]),
                    blur: Param::Constant(4.0),
                }),
                ..Default::default()
            },
        )
        .unwrap();

    let Ok(mut renderer) = Renderer::new_headless() else {
        eprintln!("skipping: no headless GPU available");
        return;
    };
    renderer.load_font(include_bytes!("../../cutlass-text/assets/Micro5-Regular.ttf").to_vec());
    let image = renderer
        .render_frame(&project, rt(0))
        .expect("text + layer shadow should composite");
    assert_eq!((image.width, image.height), (1920, 1080));
}
