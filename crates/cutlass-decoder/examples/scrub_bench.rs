//! Scrubbing / random-seek benchmark over a real media file.
//!
//! `decode_bench` covers open latency and sequential throughput; this one
//! isolates the **interactive** patterns a timeline produces, per output
//! mode, all through `frame_at` (the decode worker's API):
//!
//! - **forward drag**: playhead dragged right at slow/medium/fast speeds
//!   (steps of +2/+8/+30 frames). Steps inside the 1-second roll window ride
//!   the roll-forward path; larger ones seek.
//! - **backward drag**: same speeds leftward — every target is behind the
//!   decode position, so each pays a seek + GOP prefix re-decode. The worst
//!   case for scrubbing.
//! - **jitter**: small random steps around a hover point (a hand dithering
//!   over one spot).
//! - **random seek**: uniformly random targets across the clip (clicking
//!   around the timeline).
//!
//! The backward-drag and random patterns run twice: exact (`frame_at`, what
//! a settled preview pays) and keyframe-snapped (`frame_at_nearest`, what a
//! mid-drag scrub pays) — the gap between them is the GOP-prefix re-decode
//! that snap-scrubbing skips.
//!
//! Run:
//!   `cargo run --release -p cutlass-decoder --example scrub_bench -- <media> [targets-per-pattern]`

use std::path::PathBuf;
use std::time::Instant;

use cutlass_core::{Rational, RationalTime, VideoDecoder};
use cutlass_decoder::{OutputMode, open_video_decoder, probe};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .map(PathBuf::from)
        .expect("usage: scrub_bench <media> [targets-per-pattern]");
    let targets: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(32);

    let info = probe(&path).expect("probe failed");
    let fps = info.frame_rate;
    let frames = info.frame_count.max(1);

    println!("media: {}", path.display());
    println!(
        "  {}x{} @ {:.3} fps, {} frames, {targets} targets per pattern\n",
        info.width,
        info.height,
        fps.as_f64(),
        frames,
    );

    for mode in [OutputMode::Cpu, OutputMode::Gpu] {
        let label = match mode {
            OutputMode::Cpu => "cpu",
            OutputMode::Gpu => "gpu",
        };
        println!("== output mode: {label} ==");
        let Ok(mut dec) = open_video_decoder(&path, mode) else {
            println!("  open failed; skipping mode\n");
            continue;
        };

        for step in [2i64, 8, 30] {
            let name = format!("forward drag (+{step}/target)");
            bench_pattern(
                &mut *dec,
                &name,
                forward(frames, step, targets),
                fps,
                Seek::Exact,
            );
        }
        for step in [2i64, 8, 30] {
            let name = format!("backward drag (-{step}/target)");
            bench_pattern(
                &mut *dec,
                &name,
                backward(frames, step, targets),
                fps,
                Seek::Exact,
            );
        }
        for step in [2i64, 8, 30] {
            let name = format!("backward snap (-{step}/target)");
            bench_pattern(
                &mut *dec,
                &name,
                backward(frames, step, targets),
                fps,
                Seek::Nearest,
            );
        }
        bench_pattern(
            &mut *dec,
            "jitter around hover point",
            jitter(frames, targets),
            fps,
            Seek::Exact,
        );
        bench_pattern(
            &mut *dec,
            "random seek (whole clip)",
            random(frames, targets),
            fps,
            Seek::Exact,
        );
        bench_pattern(
            &mut *dec,
            "random snap (whole clip)",
            random(frames, targets),
            fps,
            Seek::Nearest,
        );
        println!();
    }
}

/// Which decoder positioning API a pattern exercises.
#[derive(Clone, Copy, PartialEq)]
enum Seek {
    /// `frame_at`: the exact frame (settled preview, export).
    Exact,
    /// `frame_at_nearest`: keyframe-snapped (mid-drag scrubbing).
    Nearest,
}

/// Time exact or snapped positioning over `targets` (frame indices),
/// reporting latency stats.
fn bench_pattern(
    dec: &mut dyn VideoDecoder,
    label: &str,
    targets: Vec<i64>,
    fps: Rational,
    seek: Seek,
) {
    // Park the decoder at a known spot so patterns don't inherit the decode
    // position of the previous one.
    let _ = dec.frame_at(RationalTime::new(0, fps));

    let mut samples = Vec::with_capacity(targets.len());
    let total = Instant::now();
    for frame in targets {
        let start = Instant::now();
        let result = match seek {
            Seek::Exact => dec.frame_at(RationalTime::new(frame, fps)),
            Seek::Nearest => dec.frame_at_nearest(RationalTime::new(frame, fps)),
        };
        match result {
            Ok(Some(_)) => samples.push(start.elapsed().as_secs_f64() * 1000.0),
            Ok(None) => continue, // duration overshoot near EOF: skip sample
            Err(e) => panic!("frame_at({frame}) failed: {e}"),
        }
    }
    Stats::from_samples(samples, total.elapsed().as_secs_f64()).print(label);
}

/// Ascending targets from 0 in `step` increments, wrapping at EOF.
fn forward(frames: i64, step: i64, count: usize) -> Vec<i64> {
    (1..=count as i64).map(|i| (i * step) % frames).collect()
}

/// Descending targets from the end in `step` decrements, wrapping at 0.
fn backward(frames: i64, step: i64, count: usize) -> Vec<i64> {
    (1..=count as i64)
        .map(|i| (frames - 1 - i * step).rem_euclid(frames))
        .collect()
}

/// Small random steps around the clip midpoint (±5 frames), clamped.
fn jitter(frames: i64, count: usize) -> Vec<i64> {
    let mut rng = Lcg::new(0x5EED_5EED_5EED_5EED);
    let mut at = frames / 2;
    (0..count)
        .map(|_| {
            // Non-zero step in [-5, +5].
            let step = loop {
                let s = (rng.next() % 11) as i64 - 5;
                if s != 0 {
                    break s;
                }
            };
            at = (at + step).clamp(0, frames - 1);
            at
        })
        .collect()
}

/// Uniformly random targets across the whole clip.
fn random(frames: i64, count: usize) -> Vec<i64> {
    let mut rng = Lcg::new(0x243F_6A88_85A3_08D3);
    (0..count)
        .map(|_| (rng.next() % frames as u64) as i64)
        .collect()
}

/// Deterministic LCG so runs are comparable; no rand dependency.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

struct Stats {
    mean: f64,
    p50: f64,
    p95: f64,
    max: f64,
    count: usize,
    eff_fps: f64,
}

impl Stats {
    fn from_samples(mut samples_ms: Vec<f64>, total_s: f64) -> Self {
        samples_ms.sort_by(|a, b| a.total_cmp(b));
        let n = samples_ms.len().max(1);
        Self {
            mean: samples_ms.iter().sum::<f64>() / n as f64,
            p50: samples_ms.get(n / 2).copied().unwrap_or(0.0),
            p95: samples_ms
                .get((n * 95 / 100).min(n - 1))
                .copied()
                .unwrap_or(0.0),
            max: samples_ms.last().copied().unwrap_or(0.0),
            count: samples_ms.len(),
            eff_fps: if total_s > 0.0 {
                samples_ms.len() as f64 / total_s
            } else {
                0.0
            },
        }
    }

    fn print(&self, label: &str) {
        println!(
            "{label:<28} {:>4} targets | {:>7.2} ms mean | {:>7.2} ms p50 | {:>7.2} ms p95 | {:>8.2} ms max | {:>6.1} eff fps",
            self.count, self.mean, self.p50, self.p95, self.max, self.eff_fps,
        );
    }
}
