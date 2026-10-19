//! End-to-end smoke test: resolve a generator-only project and composite it on
//! a headless GPU. Skips cleanly when no GPU adapter is available (CI).

use cutlass_models::{Generator, Project, Rational, RationalTime, TimeRange, TrackKind};
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
