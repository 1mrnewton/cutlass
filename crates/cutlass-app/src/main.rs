//! End-to-end render CLI: decode a video, build a one-clip project, composite a
//! single timeline frame, and write it to a PNG.
//!
//! This exercises the whole pipeline — decode -> engine resolve -> frame cache
//! -> CPU compositor -> image — so a glance at the output confirms the stack is
//! wired correctly. Usage:
//!
//! ```text
//! cutlass-app <video> [frame_index] [output.png]
//! ```

use std::error::Error;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use cutlass_compositor::{CompositeLayer, RgbaImage, composite};
use cutlass_decode::Decoder;
use cutlass_engines::{Engine, RenderedContent, RenderedLayer};
use cutlass_models::{Generator, MediaSource, Rational, TimeRange, TrackKind};
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

fn setup_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
}

/// Probed source facts needed to register media with the engine.
struct Probe {
    width: u32,
    height: u32,
    frame_rate: Rational,
    duration_frames: i64,
}

/// Open the file once to read its dimensions, frame rate, and length.
fn probe(path: &Path) -> Result<Probe, Box<dyn Error>> {
    let decoder = Decoder::open(path)?;
    let info = decoder.info();
    let (num, den) = info.frame_rate_parts();
    let frame_rate = Rational::new(num, den);
    if !frame_rate.is_valid() {
        return Err("source has an invalid frame rate".into());
    }

    // Length in source frames. If the container hides its duration, fall back to
    // a large bound so a single-frame render still has a clip to land on.
    let duration_frames = decoder
        .duration()
        .map(|d| (d.as_secs_f64() * frame_rate.as_f64()).round() as i64)
        .filter(|&n| n > 0)
        .unwrap_or(1_000_000);

    Ok(Probe {
        width: info.width,
        height: info.height,
        frame_rate,
        duration_frames,
    })
}

/// Map the engine's resolved layers onto the compositor's layer type.
///
/// Media frames become sampled `Frame` layers; solid generators become fills.
/// Text/shape/adjustment generators aren't drawable by the CPU compositor yet,
/// so they're skipped with a warning rather than failing the render.
fn to_composite_layers(layers: &[RenderedLayer]) -> Vec<CompositeLayer<'_>> {
    let mut out = Vec::with_capacity(layers.len());
    for layer in layers {
        match &layer.content {
            RenderedContent::Media(frame) => out.push(CompositeLayer::Frame(frame.as_ref())),
            RenderedContent::Generated(Generator::SolidColor { rgba }) => {
                out.push(CompositeLayer::Solid(*rgba))
            }
            RenderedContent::Generated(other) => {
                warn!(
                    ?other,
                    "skipping generator the CPU compositor can't draw yet"
                );
            }
        }
    }
    out
}

fn write_png(path: &Path, image: &RgbaImage) -> Result<(), Box<dyn Error>> {
    let file = BufWriter::new(File::create(path)?);
    let mut encoder = png::Encoder::new(file, image.width, image.height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&image.pixels)?;
    Ok(())
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .unwrap_or_else(|| "assets/13232364_3840_2160_24fps.mp4".to_string());
    let frame: i64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(1000);
    let output = args.next().unwrap_or_else(|| "frame.png".to_string());

    let path = Path::new(&path);
    let probe = probe(path)?;
    info!(
        ?path,
        width = probe.width,
        height = probe.height,
        fps = probe.frame_rate.as_f64(),
        duration_frames = probe.duration_frames,
        "probed source"
    );

    // Timeline runs at the source rate, so timeline frame N == source frame N.
    let mut engine = Engine::new("cli", probe.frame_rate);
    let media = MediaSource::new(
        path,
        probe.width,
        probe.height,
        probe.frame_rate,
        probe.duration_frames,
        false,
    );
    let media_id = engine.import_media(media)?;
    let track = engine.project_mut().add_track(TrackKind::Video, "V1");
    engine
        .project_mut()
        .add_clip(track, media_id, TimeRange::new(0, probe.duration_frames), 0)?;

    let layers = engine.frame_at(frame)?;
    if layers.is_empty() {
        return Err(format!(
            "no layers at frame {frame} (timeline length {})",
            engine.duration()
        )
        .into());
    }

    let composite_layers = to_composite_layers(&layers);
    let image = composite(probe.width, probe.height, &composite_layers);

    let out_path = Path::new(&output);
    write_png(out_path, &image)?;
    let cache = engine.cache_stats();
    info!(
        ?out_path,
        frame,
        width = image.width,
        height = image.height,
        ?cache,
        "wrote composited frame"
    );
    Ok(())
}

fn main() {
    setup_tracing();
    if let Err(e) = run() {
        warn!(error = %e, "render failed");
        std::process::exit(1);
    }
}
