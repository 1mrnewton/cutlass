//! End-to-end render CLI: decode a video, build a one-clip project, then either
//! composite a single timeline frame to a PNG or export the whole timeline to
//! an MP4 (chosen by the output extension).
//!
//! This exercises the whole pipeline — decode -> engine resolve -> frame cache
//! -> CPU compositor -> image/encoder — so a glance at the output confirms the
//! stack is wired correctly. Usage:
//!
//! ```text
//! cutlass-app <video> [frame_index] [output.png]   # single frame -> PNG
//! cutlass-app <video> [frame_index] [output.mp4]   # whole timeline -> MP4
//! ```

use std::error::Error;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use cutlass_compositor::{RgbaImage, composite};
use cutlass_decode::Decoder;
use cutlass_engines::{Engine, ExportSettings, to_composite_layers};
use cutlass_models::{MediaSource, Rational, TimeRange, TrackKind};
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
    let frame: i64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(100);
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

    let out_path = Path::new(&output);
    let is_video = out_path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("mp4") || e.eq_ignore_ascii_case("mov"))
        .unwrap_or(false);

    if is_video {
        export_timeline(&mut engine, out_path, probe.width, probe.height)
    } else {
        render_frame(&mut engine, out_path, frame, probe.width, probe.height)
    }
}

/// Composite a single timeline frame and write it as a PNG.
fn render_frame(
    engine: &mut Engine,
    out_path: &Path,
    frame: i64,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn Error>> {
    let layers = engine.frame_at(frame)?;
    if layers.is_empty() {
        return Err(format!(
            "no layers at frame {frame} (timeline length {})",
            engine.duration()
        )
        .into());
    }

    let composite_layers = to_composite_layers(&layers);
    let image = composite(width, height, &composite_layers);
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

/// Export the whole timeline to an H.264 MP4 at the source resolution.
fn export_timeline(
    engine: &mut Engine,
    out_path: &Path,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn Error>> {
    let total = engine.duration();
    info!(?out_path, total_frames = total, "exporting timeline");

    let mut last_pct = -1i64;
    let mut on_progress = |done: i64, total: i64| {
        let pct = if total > 0 { done * 100 / total } else { 100 };
        if pct != last_pct {
            last_pct = pct;
            info!(done, total, pct, "export progress");
        }
    };

    let stats = engine.export_with(
        out_path,
        ExportSettings::new(width, height),
        Some(&mut on_progress),
    )?;
    info!(
        ?out_path,
        frames = stats.frames,
        width = stats.width,
        height = stats.height,
        "exported timeline"
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
