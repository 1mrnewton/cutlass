//! Time a fixed set of effect recipes on this machine's GPU and report the
//! working texture memory each one keeps.
//!
//! Each case draws a picture with the recipe over it [`REPEAT`] times in
//! one frame, waits for the GPU, and takes the median of many frames. The
//! same picture drawn with no effect is timed the same way and taken off,
//! and what is left, divided by [`REPEAT`], is one effect's own cost. The
//! numbers are wall time, good for comparing one build against another on
//! the same device, not as absolute figures.
//!
//! ```sh
//! cargo run -p cutlass-render --release --example effect_bench
//! cargo run -p cutlass-render --release --example effect_bench -- 1920x1080
//! ```

#![allow(clippy::print_stdout)]

use std::collections::BTreeMap;
use std::error::Error;
use std::time::{Duration, Instant};

use cutlass_effects::{Recipe, Step, bundled_recipe};
use cutlass_model::Micros;
use cutlass_render::{Layer, Placement, Renderer, Rgba};

type Failure = Box<dyn Error>;

/// Frames drawn and thrown away before timing, so pipelines and textures
/// exist.
const WARM: usize = 5;
/// Frames timed per case.
const FRAMES: usize = 40;
/// Times the effect is drawn in one frame, so its cost stands well above
/// the upload of the picture and the wait.
const REPEAT: usize = 8;
/// The moment of each recipe drawn: past the first second, where flashes
/// and reveals have settled.
const AT: Micros = Micros(1_500_000);

/// Controls set over a recipe's defaults, by id.
type Controls = &'static [(&'static str, f32)];

/// What is timed: a label, a bundled recipe's id, and its controls.
const CASES: &[(&str, &str, Controls)] = &[
    ("invert", "invert", &[]),
    ("blur", "blur", &[]),
    ("blur x3", "blur", &[("blur", 3.0)]),
    ("glow", "glow", &[]),
    ("soft-glow", "soft-glow", &[]),
    ("dreamy", "dreamy", &[]),
    ("vhs (5 passes)", "vhs", &[]),
    // The drawn ones: their cost is arithmetic in every pixel, not reads.
    ("film-leak", "film-leak", &[]),
    ("bokeh-lights", "bokeh-lights", &[]),
    ("lens-flare", "lens-flare", &[]),
    ("dust", "dust", &[]),
    ("sparks", "sparks", &[]),
    ("fog", "fog", &[]),
    ("smoke", "smoke", &[]),
    ("real-grain", "real-grain", &[]),
    ("scratched-film", "scratched-film", &[]),
    ("worn-print (4 passes)", "worn-print", &[]),
    ("crumpled-paper", "crumpled-paper", &[]),
    ("ink-reveal", "ink-reveal", &[]),
];

fn main() -> Result<(), Failure> {
    let sizes = match std::env::args().nth(1) {
        Some(size) => vec![parse_size(&size)?],
        None => vec![(1920, 1080), (3840, 2160)],
    };
    let renderer = Renderer::headless()?;
    for (width, height) in sizes {
        println!("{width}x{height}, median of {FRAMES} frames");
        println!(
            "{:<16} {:>10} {:>10} {:>12}",
            "case", "frame ms", "effect ms", "textures MB"
        );
        let picture = scene(width, height);
        let canvas = renderer.canvas(width, height)?;
        let base = time(&renderer, &canvas, &picture, (width, height), &[])?;
        let base_bytes = renderer.texture_bytes();
        println!(
            "{:<16} {:>10.2} {:>10} {:>12.1}",
            "(no effect)",
            millis(base),
            "-",
            megabytes(base_bytes)
        );
        for (label, id, controls) in CASES {
            let recipe: &Recipe = bundled_recipe(&format!("recipe/{id}"))
                .ok_or_else(|| format!("no bundled recipe {id}"))?;
            let controls: BTreeMap<String, f32> = controls
                .iter()
                .map(|(key, value)| ((*key).to_owned(), *value))
                .collect();
            let steps = recipe.steps_with(AT, &controls);
            let total = time(&renderer, &canvas, &picture, (width, height), &steps)?;
            let one = total.saturating_sub(base) / REPEAT as u32;
            println!(
                "{:<16} {:>10.2} {:>10.2} {:>12.1}",
                label,
                millis(total),
                millis(one),
                megabytes(renderer.texture_bytes())
            );
        }
        println!();
    }
    Ok(())
}

/// The median time to draw `picture` with `steps` over it and wait for the
/// GPU to finish.
fn time(
    renderer: &Renderer,
    canvas: &cutlass_render::Canvas,
    picture: &[u8],
    (width, height): (u32, u32),
    steps: &[Step],
) -> Result<Duration, Failure> {
    let rgba = Rgba {
        width,
        height,
        bytes: picture,
    };
    let placement = Placement::canvas(width, height);
    let mut layers = vec![Layer::Rgba(rgba, placement)];
    if !steps.is_empty() {
        for _ in 0..REPEAT {
            layers.push(Layer::Effect(steps, None));
        }
    }
    let mut times = Vec::with_capacity(FRAMES);
    for frame in 0..WARM + FRAMES {
        let start = Instant::now();
        renderer.draw(canvas, &layers)?;
        renderer
            .device()
            .poll(wgpu::PollType::wait_indefinitely())?;
        if frame >= WARM {
            times.push(start.elapsed());
        }
    }
    times.sort_unstable();
    // One more frame with the effect once, so the textures the renderer
    // then keeps are one effect's.
    layers.truncate(2);
    renderer.draw(canvas, &layers)?;
    Ok(times[times.len() / 2])
}

/// A picture with highlights, shadows and edges, so glows and blurs have
/// something to work on: a sky gradient, a bright disc and a grid of bars.
fn scene(width: u32, height: u32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(width as usize * height as usize * 4);
    for y in 0..height {
        for x in 0..width {
            let u = x as f32 / width as f32;
            let v = y as f32 / height as f32;
            let (dx, dy) = (u - 0.7, (v - 0.3) * height as f32 / width as f32);
            let sun = if dx * dx + dy * dy < 0.004 { 1.0 } else { 0.0 };
            let bar = if (x / 40 + y / 40) % 2 == 0 {
                0.15
            } else {
                0.0
            };
            let r = (0.2 + 0.5 * v + bar).max(sun);
            let g = (0.3 + 0.3 * v + bar).max(sun);
            let b = (0.6 - 0.3 * v + bar).max(sun * 0.9);
            for channel in [r, g, b] {
                bytes.push((channel.clamp(0.0, 1.0) * 255.0).round() as u8);
            }
            bytes.push(255);
        }
    }
    bytes
}

fn parse_size(text: &str) -> Result<(u32, u32), Failure> {
    let (width, height) = text
        .split_once('x')
        .ok_or("a size is WIDTHxHEIGHT, such as 1920x1080")?;
    Ok((width.parse()?, height.parse()?))
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn megabytes(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}
