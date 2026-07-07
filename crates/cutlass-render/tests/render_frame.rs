//! End-to-end smoke test: resolve a generator-only project and composite it on
//! a headless GPU. Skips cleanly when no GPU adapter is available (CI).

use cutlass_models::{
    Generator, Project, Rational, RationalTime, Shape, ShapePath, ShapePathPoint, TimeRange,
    TrackKind,
};
use cutlass_render::Renderer;

const FPS_24: Rational = Rational::FPS_24;

fn rt(value: i64) -> RationalTime {
    RationalTime::new(value, FPS_24)
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
    assert!(px[0] > 80 && px[2] > 80, "midpoint should blend channels, got {px:?}");
    assert!(px[0] < 240 && px[2] < 240);
}
