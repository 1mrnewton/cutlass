//! Build a tiny project (solid background + a centered title) and export it,
//! exercising the whole resolve → render → encode path.
//!
//! With no argument (or an `.mp4` path) it uses the platform-native encoder.
//! With any other path it writes a PNG sequence into that directory instead.
//!
//! ```text
//! cargo run -p cutlass-render --example export_demo                 # -> ./cutlass-demo.mp4
//! cargo run -p cutlass-render --example export_demo clip.mp4        # -> ./clip.mp4
//! cargo run -p cutlass-render --example export_demo frames_dir      # -> ./frames_dir/*.png
//! ```

use std::error::Error;
use std::path::PathBuf;

use cutlass_models::{
    CanvasAspect, CanvasSettings, Generator, Project, Rational, TextStyle, TimeRange, TrackKind,
};
use cutlass_render::{PngSequenceEncoder, Renderer, export, export_to_file};

const FPS: Rational = Rational::FPS_30;
const SECONDS: i64 = 1;

fn main() -> Result<(), Box<dyn Error>> {
    let out = std::env::args()
        .nth(1)
        .map_or_else(|| PathBuf::from("cutlass-demo.mp4"), PathBuf::from);

    let frames = SECONDS * 30;
    let span = TimeRange::at_rate(0, frames, FPS);

    let mut project = Project::new("demo", FPS);
    project.timeline_mut().set_canvas(CanvasSettings {
        aspect: CanvasAspect::Wide16x9,
        background: [20, 20, 30],
    });

    // A solid wash behind the title.
    let bg = project.add_track(TrackKind::Sticker, "BG");
    project.add_generated(
        bg,
        Generator::SolidColor {
            rgba: [38, 42, 64, 255],
        },
        span,
    )?;

    // A centered title on its own text lane.
    let titles = project.add_track(TrackKind::Text, "Titles");
    let style = TextStyle {
        size: 220.0.into(),
        fill: [240, 240, 255, 255].into(),
        ..TextStyle::default()
    };
    project.add_generated(
        titles,
        Generator::Text {
            content: "Cutlass".into(),
            style,
        },
        span,
    )?;

    let mut renderer = Renderer::new_headless()?;

    let is_mp4 = out
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("mp4"));

    let written = if is_mp4 {
        export_to_file(&mut renderer, &project, &out)?
    } else {
        let mut encoder = PngSequenceEncoder::new(&out)?;
        export(&mut renderer, &project, &mut encoder)?
    };

    println!(
        "exported {written} frame(s) at 1920x1080 to {}",
        out.display()
    );
    Ok(())
}
