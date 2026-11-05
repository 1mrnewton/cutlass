//! Fold ten seconds of 48 kHz 5.1 and 7.1 into stereo.
//!
//! `cargo bench -p cutlass-audio --bench downmix`
//!
//! The output buffer is kept across runs, as the mix keeps one, so the
//! figure is the matrix pass alone. The reported figure is the median of
//! 15 runs after one warmup, in input frames and input samples per second.

#![allow(clippy::print_stdout)]

use std::time::{Duration, Instant};

use cutlass_audio::downmix_into;
use cutlass_codec::access::{ChannelLayout, Pcm, StandardLayout};

const RATE: u32 = 48_000;
const SECONDS: usize = 10;
const RUNS: usize = 15;

fn main() -> Result<(), String> {
    for (name, layout) in [
        ("5.1", StandardLayout::FiveOne),
        ("7.1", StandardLayout::SevenOne),
    ] {
        let layout = ChannelLayout::standard(layout);
        let channels = layout.len();
        let frames = RATE as usize * SECONDS;
        let samples: Vec<f32> = (0..frames * channels)
            .map(|index| ((index % 977) as f32 / 977.0) - 0.5)
            .collect();
        let input = Pcm::from_f32(RATE, layout, samples).map_err(|error| format!("{error:?}"))?;
        let stereo = ChannelLayout::stereo();
        let mut out = Vec::with_capacity(frames * 2);
        downmix_into(&input, &stereo, &mut out).map_err(|error| error.to_string())?;
        let mut times: Vec<Duration> = (0..RUNS)
            .map(|_| {
                let start = Instant::now();
                let outcome = downmix_into(&input, &stereo, &mut out);
                let elapsed = start.elapsed();
                std::hint::black_box((&outcome, &out));
                elapsed
            })
            .collect();
        times.sort();
        let median = times[RUNS / 2];
        let seconds = median.as_secs_f64();
        let frames_per_second = frames as f64 / seconds;
        println!(
            "{name} -> stereo: {frames} frames in {:.3} ms (min {:.3}, max {:.3}); \
             {:.1} M frames/s, {:.1} M samples/s, {:.0}x real time",
            seconds * 1e3,
            times[0].as_secs_f64() * 1e3,
            times[RUNS - 1].as_secs_f64() * 1e3,
            frames_per_second / 1e6,
            frames_per_second * channels as f64 / 1e6,
            frames_per_second / f64::from(RATE),
        );
    }
    Ok(())
}
