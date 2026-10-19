//! Generate the demo `sample.mp4` used by the mobile test apps: a short 16:9
//! clip whose full-canvas color sweeps through the hue wheel, so both the
//! still-frame decode panel and the video scrubber show visible motion.
//!
//! Usage: `cargo run -p cutlass-render --example gen_sample -- <out.mp4>`

use std::path::PathBuf;

use cutlass_models::{
    CanvasAspect, CanvasSettings, Generator, Project, Rational, TimeRange, TrackKind,
};
use cutlass_render::{Renderer, export_to_file};

/// An opaque RGBA sweep through the hue wheel as `t` runs 0->1.
fn hue_sweep(t: f32) -> [u8; 4] {
    let h = (t.rem_euclid(1.0) * 6.0).rem_euclid(6.0);
    let x = 1.0 - (h % 2.0 - 1.0).abs();
    let (r, g, b) = match h as u32 {
        0 => (1.0, x, 0.0),
        1 => (x, 1.0, 0.0),
        2 => (0.0, 1.0, x),
        3 => (0.0, x, 1.0),
        4 => (x, 0.0, 1.0),
        _ => (1.0, 0.0, x),
    };
    [
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
        255,
    ]
}

fn main() {
    let out: PathBuf = std::env::args()
        .nth(1)
        .expect("usage: gen_sample <out.mp4>")
        .into();

    let fps = Rational::FPS_30;
    let total = 90i64; // 3s @ 30fps
    let step = 6i64; // new color every 0.2s

    let mut project = Project::new("sample", fps);
    project.timeline_mut().set_canvas(CanvasSettings {
        aspect: CanvasAspect::Wide16x9,
        background: [10, 10, 10],
    });
    let track = project.add_track(TrackKind::Sticker, "BG");
    let mut start = 0i64;
    while start < total {
        let len = step.min(total - start);
        let rgba = hue_sweep(start as f32 / total as f32);
        project
            .add_generated(track, Generator::SolidColor { rgba }, TimeRange::at_rate(start, len, fps))
            .expect("add generated clip");
        start += step;
    }

    let mut renderer = Renderer::new_headless().expect("headless renderer");
    let frames = export_to_file(&mut renderer, &project, &out).expect("export mp4");
    println!("wrote {frames} frames to {}", out.display());
}
