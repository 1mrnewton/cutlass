//! End-to-end `.cube` LUT rendering: a per-clip LUT on a solid, intensity
//! blending, a filter-lane LUT grading the whole canvas, and the missing-file
//! grade-nothing policy — all composited on a headless GPU. GPU tests skip
//! cleanly when no adapter is available (CI).

use cutlass_core::{Rational, RationalTime};
use cutlass_models::{
    CanvasAspect, CanvasSettings, Generator, Lut, Project, TimeRange, TrackKind,
};
use cutlass_render::Renderer;

const FPS_24: Rational = Rational::FPS_24;

fn rt(value: i64) -> RationalTime {
    RationalTime::new(value, FPS_24)
}

fn pixel(image: &cutlass_compositor::RgbaImage, x: u32, y: u32) -> [u8; 4] {
    let idx = ((y * image.width + x) * 4) as usize;
    image.pixels[idx..idx + 4].try_into().unwrap()
}

/// A 2-point LUT that swaps the red and green channels — visually unmistakable
/// and exactly representable at any grid size.
fn swap_rg_cube() -> String {
    cutlass_compositor::CubeLut::from_fn(2, |[r, g, b]| [g, r, b]).to_cube_string("swap-rg")
}

fn temp_cube(source: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "cutlass_lut_test_{}.cube",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(&path, source).unwrap();
    path
}

/// A 16:9 project whose main track is a full-canvas red solid.
fn red_solid_project() -> (Project, cutlass_models::ClipId) {
    let mut project = Project::new("lut", FPS_24);
    project.timeline_mut().set_canvas(CanvasSettings {
        aspect: CanvasAspect::Wide16x9,
        background: [0, 0, 0],
    });
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
fn clip_lut_remaps_layer_colors() {
    let file = temp_cube(&swap_rg_cube());
    let (mut project, clip) = red_solid_project();
    project
        .set_clip_lut(
            clip,
            Some(Lut {
                path: file.to_string_lossy().into_owned(),
                intensity: 1.0,
            }),
        )
        .unwrap();

    let Ok(mut renderer) = Renderer::new_headless() else {
        eprintln!("skipping: no headless GPU available");
        return;
    };
    let frame = renderer
        .render_frame_fit(&project, rt(0), 320, 320)
        .expect("render LUT frame");
    let center = pixel(&frame, 160, 90);
    assert!(
        center[0] < 30 && center[1] > 225,
        "red should become green through the swap LUT, got {center:?}"
    );

    // Half intensity blends the swapped result over the original 50/50.
    project
        .set_clip_lut(
            clip,
            Some(Lut {
                path: file.to_string_lossy().into_owned(),
                intensity: 0.5,
            }),
        )
        .unwrap();
    let blended = renderer
        .render_frame_fit(&project, rt(0), 320, 320)
        .expect("render blended LUT frame");
    let mid = pixel(&blended, 160, 90);
    assert!(
        (100..=155).contains(&mid[0]) && (100..=155).contains(&mid[1]),
        "half intensity should sit between red and green, got {mid:?}"
    );

    let _ = std::fs::remove_file(&file);
}

#[test]
fn filter_lane_lut_grades_the_canvas() {
    let file = temp_cube(&swap_rg_cube());
    let (mut project, _clip) = red_solid_project();
    // A filter lane bar above the solid: its LUT grades everything beneath.
    let lane = project.add_track(TrackKind::Filter, "F1");
    let bar = project
        .add_generated(lane, Generator::Filter, TimeRange::at_rate(0, 100, FPS_24))
        .unwrap();
    project
        .set_clip_lut(
            bar,
            Some(Lut {
                path: file.to_string_lossy().into_owned(),
                intensity: 1.0,
            }),
        )
        .unwrap();

    let Ok(mut renderer) = Renderer::new_headless() else {
        eprintln!("skipping: no headless GPU available");
        return;
    };
    let frame = renderer
        .render_frame_fit(&project, rt(0), 320, 320)
        .expect("render canvas LUT frame");
    let center = pixel(&frame, 160, 90);
    assert!(
        center[0] < 30 && center[1] > 225,
        "the filter-lane LUT should swap the canvas red to green, got {center:?}"
    );

    let _ = std::fs::remove_file(&file);
}

#[test]
fn missing_lut_file_grades_nothing() {
    let (mut project, clip) = red_solid_project();
    project
        .set_clip_lut(clip, Some(Lut::new("/nonexistent/look.cube")))
        .unwrap();

    let Ok(mut renderer) = Renderer::new_headless() else {
        eprintln!("skipping: no headless GPU available");
        return;
    };
    let frame = renderer
        .render_frame_fit(&project, rt(0), 320, 320)
        .expect("a missing LUT file must not fail the frame");
    assert_eq!(
        pixel(&frame, 160, 90),
        [255, 0, 0, 255],
        "the layer renders ungraded when its LUT is unreadable"
    );
}
