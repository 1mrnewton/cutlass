//! Decode one HEVC stream with this crate and with FFmpeg, after checking
//! that both return the same samples.
//!
//! The stream is `HEVC_BENCH` (an MP4, QuickTime or Annex B file) when that
//! variable is set, else 240 pictures of 1920×1080 that FFmpeg writes from
//! its `testsrc2` source with libx265 (`-preset medium -crf 26`). Both
//! decoders get the machine's thread count (`HEVC_BENCH_THREADS` changes
//! it). Cutlass time is the decode of access units already in memory;
//! FFmpeg time is one `ffmpeg -f null` process, startup and file reading
//! included. Each side runs five times in turn and the best run counts.
//!
//! Without `ffmpeg`, or without libx265 and no `HEVC_BENCH`, the bench
//! prints `skip:` and exits 0. It does not link FFmpeg.

#![allow(clippy::print_stdout)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use cutlass_decode::{Packet, VideoDecoder, VideoOutput};
use cutlass_hevc::HevcDecoder;

#[path = "../examples/common/mod.rs"]
mod common;

const RUNS: usize = 5;

fn main() -> Result<(), String> {
    if !Command::new("ffmpeg")
        .arg("-version")
        .output()
        .is_ok_and(|o| o.status.success())
    {
        println!("skip: ffmpeg is not installed");
        return Ok(());
    }
    let dir = std::env::temp_dir().join(format!("cutlass-hevc-bench-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let outcome = run(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

fn run(dir: &Path) -> Result<(), String> {
    let input = match std::env::var_os("HEVC_BENCH") {
        Some(path) => PathBuf::from(path),
        None => match encode(dir)? {
            Some(path) => path,
            None => {
                println!("skip: ffmpeg has no libx265 and HEVC_BENCH is not set");
                return Ok(());
            }
        },
    };
    let threads = std::env::var("HEVC_BENCH_THREADS")
        .ok()
        .and_then(|t| t.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from));
    let path = input.to_str().ok_or("path is not UTF-8")?;
    let (params, units) = common::read(path)?;
    println!(
        "workload: {} access units from {}, {threads} threads",
        units.len(),
        input.display()
    );

    // The samples first.
    let ours = decode(&params, &units, threads, true)?.1;
    let theirs = Command::new("ffmpeg")
        .args(["-v", "error", "-threads", &threads.to_string(), "-i"])
        .arg(&input)
        .args([
            "-map",
            "0:v:0",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-",
        ])
        .output()
        .map_err(|e| e.to_string())?;
    if !theirs.status.success() {
        return Err(String::from_utf8_lossy(&theirs.stderr).into_owned());
    }
    if ours != theirs.stdout {
        return Err(format!(
            "the samples differ from FFmpeg's ({} bytes against {})",
            ours.len(),
            theirs.stdout.len()
        ));
    }
    println!("samples: the same as FFmpeg's, {} bytes", ours.len());

    let mut best = (f64::MAX, f64::MAX);
    let mut pictures = 0;
    for _ in 0..RUNS {
        let (seconds, _, count) = decode(&params, &units, threads, false)?;
        pictures = count;
        best.0 = best.0.min(seconds);
        let start = Instant::now();
        let status = Command::new("ffmpeg")
            .args(["-v", "error", "-threads", &threads.to_string(), "-i"])
            .arg(&input)
            .args(["-map", "0:v:0", "-f", "null", "-"])
            .status()
            .map_err(|e| e.to_string())?;
        if !status.success() {
            return Err("ffmpeg failed".into());
        }
        best.1 = best.1.min(start.elapsed().as_secs_f64());
    }
    let fps = |s: f64| pictures as f64 / s;
    println!("cutlass: {:.3} s, {:.0} pictures/s", best.0, fps(best.0));
    println!(
        "ffmpeg:  {:.3} s, {:.0} pictures/s (one process)",
        best.1,
        fps(best.1)
    );
    Ok(())
}

/// 240 pictures of 1080p from `testsrc2`, or `None` without libx265.
fn encode(dir: &Path) -> Result<Option<PathBuf>, String> {
    let out = dir.join("bench.hevc");
    let status = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1920x1080:rate=30",
            "-frames:v",
            "240",
            "-c:v",
            "libx265",
            "-preset",
            "medium",
            "-crf",
            "26",
            "-x265-params",
            "log-level=error",
            "-f",
            "hevc",
            "-y",
        ])
        .arg(&out)
        .output()
        .map_err(|e| e.to_string())?;
    Ok(status.status.success().then_some(out))
}

/// Decode every access unit: the time, the samples when `keep`, and the
/// pictures.
fn decode(
    params: &cutlass_decode::CodecParams,
    units: &[Vec<u8>],
    threads: usize,
    keep: bool,
) -> Result<(f64, Vec<u8>, usize), String> {
    let mut decoder =
        HevcDecoder::open_with_threads(params, threads).map_err(|e| format!("{e:?}"))?;
    let mut bytes = Vec::new();
    let mut pictures = 0;
    let mut take = |p: &VideoOutput| {
        pictures += 1;
        if keep {
            for plane in [p.picture.y(), p.picture.u(), p.picture.v()] {
                bytes.extend_from_slice(plane);
            }
        }
    };
    let start = Instant::now();
    for (index, unit) in units.iter().enumerate() {
        let packet = Packet::video(unit.clone(), index as i64, 1, 30, false);
        for p in &decoder
            .push(&packet)
            .map_err(|e| format!("access unit {index}: {e:?}"))?
        {
            take(p);
        }
    }
    for p in &decoder.drain().map_err(|e| format!("drain: {e:?}"))? {
        take(p);
    }
    Ok((start.elapsed().as_secs_f64(), bytes, pictures))
}
