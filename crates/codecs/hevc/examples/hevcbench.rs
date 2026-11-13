//! Times the Rust HEVC decoder on one stream.
//!
//! Usage: hevcbench <in.mp4|in.mov|in.hevc> [runs] [threads] [frames rows]
//!
//! Decodes the whole stream `runs` times (default 3) after reading it into
//! memory, and prints the time and speed of each run and the best one.
//! `threads` defaults to the machine's; `frames rows` sets how many
//! pictures decode at once and how many threads take the rows of a
//! wavefront picture, in place of the split the decoder chooses.

#![allow(clippy::print_stdout, clippy::print_stderr)]

use cutlass_decode::{Packet, VideoDecoder};
use cutlass_hevc::HevcDecoder;

mod common;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1) else {
        return Err(
            "usage: hevcbench <in.mp4|in.mov|in.hevc> [runs] [threads] [frames rows]".into(),
        );
    };
    let number = |i: usize, default: usize| -> Result<usize, String> {
        args.get(i).map_or(Ok(default), |v| {
            v.parse().map_err(|_| format!("not a number: {v}"))
        })
    };
    let runs = number(2, 3)?;
    let threads = number(
        3,
        std::thread::available_parallelism().map_or(1, usize::from),
    )?;
    let split = match (args.get(4), args.get(5)) {
        (Some(_), Some(_)) => Some((number(4, 1)?, number(5, 1)?)),
        _ => None,
    };
    let (params, units) = common::read(path)?;
    let mut best = f64::MAX;
    for run in 0..runs {
        let mut decoder =
            HevcDecoder::open_with_threads(&params, threads).map_err(|e| format!("{e:?}"))?;
        if let Some((frames, rows)) = split {
            decoder.set_threads(frames, rows);
        }
        let start = std::time::Instant::now();
        let mut pictures = 0usize;
        for (index, unit) in units.iter().enumerate() {
            let packet = Packet::video(unit.clone(), index as i64, 1, 30, false);
            pictures += decoder
                .push(&packet)
                .map_err(|e| format!("access unit {index}: {e:?}"))?
                .len();
        }
        pictures += decoder.drain().map_err(|e| format!("drain: {e:?}"))?.len();
        let seconds = start.elapsed().as_secs_f64();
        best = best.min(seconds);
        println!(
            "run {}: {pictures} pictures in {seconds:.3} s, {:.1} pictures/s",
            run + 1,
            pictures as f64 / seconds
        );
    }
    println!("best: {best:.3} s");
    Ok(())
}
