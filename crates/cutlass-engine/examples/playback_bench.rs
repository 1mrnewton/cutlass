//! Sequential playback benchmark (playback roadmap Phase 2).
//!
//! Simulates what the preview worker does during playback — one frame per
//! sequence tick, in order — and reports per-frame latency for:
//!
//!   1. the decoder alone, seek-per-frame (`seek_to_frame`, the old path);
//!   2. the decoder alone, roll-forward (`frame_at`, the playback path);
//!   3. the engine end to end (`get_frame`: decode + GPU composite +
//!      readback), cache-cold then cache-warm.
//!
//! Run:
//!   `cargo run --release -p cutlass-engine --example playback_bench`
//!   `cargo run --release -p cutlass-engine --example playback_bench -- local-assets/assets/foo.mp4 10`
//!
//! Criterion benches (`benches/preview.rs`) cover steady-state single ticks;
//! this harness exists because playback cost is a *sequence* property (GOP
//! position matters), which criterion's repeated-measurement model hides.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use cutlass_commands::{Command, EditCommand, EditOutcome, ProjectCommand};
use cutlass_decoder::{DecodeOptions, Decoder, HwAccel};
use cutlass_engine::{ApplyOutcome, Engine, EngineConfig};
use cutlass_models::{Rational, RationalTime, TrackKind, resample};

/// Timeline rate the UI uses today (`Project::new` default).
const TL_FPS: i64 = 24;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .map(PathBuf::from)
        .or_else(default_asset)
        .expect("no media found: pass a path or add local-assets/assets/*.mp4");
    let seconds: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);

    println!("media: {}", path.display());
    println!("simulating {seconds}s of {TL_FPS}fps playback\n");

    bench_decoder(&path, seconds);
    bench_engine(&path, seconds);
}

fn default_asset() -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../local-assets/assets");
    for name in [
        "15531444_1920_1080_24fps.mp4",
        "6137050-hd_1920_1080_24fps.mp4",
    ] {
        let path = dir.join(name);
        if path.exists() {
            return Some(path);
        }
    }
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|ext| ext == "mp4"))
}

/// Source-time targets for sequential playback ticks (exact integer micros).
fn targets(seconds: u64) -> Vec<Duration> {
    (0..(seconds as i64 * TL_FPS))
        .map(|tick| Duration::from_micros((tick * 1_000_000 / TL_FPS) as u64))
        .collect()
}

fn bench_decoder(path: &Path, seconds: u64) {
    let opts = DecodeOptions::default().hw_accel(HwAccel::None);
    let targets = targets(seconds);

    let mut seeked = Decoder::open_with(path, opts).expect("open decoder");
    let seek_stats = time_each("decoder seek_to_frame", &targets, |t| {
        seeked.seek_to_frame(*t).expect("seek_to_frame");
    });

    let mut rolled = Decoder::open_with(path, opts).expect("open decoder");
    let roll_stats = time_each("decoder frame_at     ", &targets, |t| {
        rolled.frame_at(*t).expect("frame_at");
    });

    println!(
        "  -> roll-forward speedup: {:.1}x mean, {:.1}x p95\n",
        seek_stats.mean / roll_stats.mean,
        seek_stats.p95 / roll_stats.p95,
    );
}

fn bench_engine(path: &Path, seconds: u64) {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = EngineConfig {
        cache_dir: dir.path().join("cache"),
        cache_budget_bytes: 4 * 1024 * 1024 * 1024,
        undo_limit: 8,
        ..Default::default()
    };
    let mut engine = Engine::new(config).expect("engine");

    let media = match engine
        .apply(Command::Project(ProjectCommand::Import {
            path: path.to_path_buf(),
        }))
        .expect("import")
    {
        ApplyOutcome::Imported { media } => media,
        other => panic!("unexpected import outcome: {other:?}"),
    };
    let track = match engine
        .apply(Command::Edit(EditCommand::AddTrack {
            kind: TrackKind::Video,
            name: "V1".into(),
            index: None,
            pinned: false,
        }))
        .expect("track")
    {
        ApplyOutcome::Edited(EditOutcome::CreatedTrack(id)) => id,
        other => panic!("unexpected add-track outcome: {other:?}"),
    };

    let tl_rate = Rational::new(TL_FPS as i32, 1);
    let source = engine.project().media(media).expect("media").full_range();
    let clip_ticks = resample(source.duration, tl_rate).value.max(1);
    engine
        .apply(Command::Edit(EditCommand::AddClip {
            track,
            media,
            source,
            start: RationalTime::new(0, tl_rate),
        }))
        .expect("clip");

    let frames = (seconds as i64 * TL_FPS).min(clip_ticks);
    let ticks: Vec<i64> = (0..frames).collect();

    let cold = time_each("engine get_frame cold", &ticks, |&tick| {
        engine
            .get_frame(RationalTime::new(tick, tl_rate))
            .expect("frame");
    });
    println!(
        "  -> realtime budget {:.1}ms/frame: {}",
        1000.0 / TL_FPS as f64,
        verdict(&cold)
    );

    // Give the async cache writer a moment to land the cold pass's frames.
    engine.cache().sync();

    let warm = time_each("engine get_frame warm", &ticks, |&tick| {
        engine
            .get_frame(RationalTime::new(tick, tl_rate))
            .expect("frame");
    });
    println!(
        "  -> realtime budget {:.1}ms/frame: {}",
        1000.0 / TL_FPS as f64,
        verdict(&warm)
    );
}

fn verdict(stats: &Stats) -> &'static str {
    let budget = 1000.0 / TL_FPS as f64;
    if stats.p95 <= budget {
        "REALTIME (p95 within budget)"
    } else if stats.p50 <= budget {
        "marginal (p50 within budget, p95 over)"
    } else {
        "NOT realtime"
    }
}

struct Stats {
    mean: f64,
    p50: f64,
    p95: f64,
    max: f64,
}

fn time_each<T>(label: &str, items: &[T], mut f: impl FnMut(&T)) -> Stats {
    let mut samples_ms: Vec<f64> = Vec::with_capacity(items.len());
    let total = Instant::now();
    for item in items {
        let start = Instant::now();
        f(item);
        samples_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let elapsed = total.elapsed().as_secs_f64();

    samples_ms.sort_by(|a, b| a.total_cmp(b));
    let n = samples_ms.len().max(1);
    let stats = Stats {
        mean: samples_ms.iter().sum::<f64>() / n as f64,
        p50: samples_ms[n / 2],
        p95: samples_ms[(n * 95 / 100).min(n - 1)],
        max: samples_ms.last().copied().unwrap_or(0.0),
    };
    println!(
        "{label}: {:>6.2}ms mean | {:>6.2}ms p50 | {:>6.2}ms p95 | {:>7.2}ms max | {:>5.1} fps",
        stats.mean,
        stats.p50,
        stats.p95,
        stats.max,
        items.len() as f64 / elapsed,
    );
    stats
}
