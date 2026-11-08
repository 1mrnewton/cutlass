//! Time ALAC decode and encode against ffmpeg.
//!
//! `cargo bench -p cutlass-alac`
//!
//! The source is the first 60 seconds of a real file from `../local-files`
//! when it and ffmpeg are there, otherwise a synthetic signal. Every encode
//! is decoded back and compared before anything is timed. A figure is the
//! median of 3 runs after one warmup, as real time divided by wall time.
//! ffmpeg rows are a new process each and include process startup.

#![allow(clippy::print_stdout)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use cutlass_alac::{AlacDecoder, AlacEncoder, EncodeMode, EncoderConfig};
use cutlass_decode::ChannelLayout;
use cutlass_encode::Encoder;

const RATE: u32 = 44_100;
const SECONDS: usize = 60;
const SOURCE: &str = "Expo - You Don't Need a Mac to Build iOS Apps Anymore.m4a";

fn main() -> Result<(), String> {
    let ffmpeg = find_ffmpeg();
    let dir = std::env::temp_dir().join(format!("cutlass-alac-bench-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|err| err.to_string())?;
    let outcome = run(ffmpeg.as_deref(), &dir);
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

fn run(ffmpeg: Option<&Path>, dir: &Path) -> Result<(), String> {
    println!("{SECONDS} s stereo at {RATE} Hz; median of 3 after 1 warmup; x real time");
    for depth in [16u8, 24] {
        let (samples, origin) = source(ffmpeg, dir, depth)?;
        println!("{depth}-bit, {origin}");
        for mode in [EncodeMode::Default, EncodeMode::Fast] {
            let config = EncoderConfig {
                mode,
                ..EncoderConfig::new(RATE, 2, depth)
            };
            let (cookie, packets) = encode(config, &samples)?;
            let decoded = decode(&cookie, &packets)?;
            if decoded != samples {
                return Err(format!("{depth}-bit {mode:?} does not round-trip"));
            }
            let bytes: usize = packets.iter().map(Vec::len).sum();
            let encode_time = measure(|| {
                let started = Instant::now();
                std::hint::black_box(encode(config, &samples)?);
                Ok(started.elapsed())
            })?;
            let decode_time = measure(|| {
                let started = Instant::now();
                std::hint::black_box(decode(&cookie, &packets)?);
                Ok(started.elapsed())
            })?;
            println!(
                "  cutlass {mode:?}: {bytes} bytes, encode {}, decode {}",
                speed(encode_time),
                speed(decode_time)
            );
        }
        if let Some(ffmpeg) = ffmpeg {
            let raw = dir.join(format!("source-{depth}.raw"));
            let format = if depth == 16 { "s16le" } else { "s32le" };
            write_raw(&raw, &samples, depth)?;
            let m4a = dir.join(format!("ffmpeg-{depth}.m4a"));
            let encode_args = [
                "-v",
                "error",
                "-y",
                "-f",
                format,
                "-ar",
                "44100",
                "-ac",
                "2",
                "-i",
                path_str(&raw)?,
                "-c:a",
                "alac",
                path_str(&m4a)?,
            ];
            let encode_time = measure(|| process(ffmpeg, &encode_args))?;
            let decode_args = ["-v", "error", "-i", path_str(&m4a)?, "-f", "null", "-"];
            let decode_time = measure(|| process(ffmpeg, &decode_args))?;
            println!(
                "  ffmpeg: encode {}, decode {}",
                speed(encode_time),
                speed(decode_time)
            );
        }
    }
    Ok(())
}

fn speed(time: Duration) -> String {
    format!("{:.0}x", SECONDS as f64 / time.as_secs_f64())
}

fn measure(mut once: impl FnMut() -> Result<Duration, String>) -> Result<Duration, String> {
    let _warmup = once()?;
    let mut times = [once()?, once()?, once()?];
    times.sort();
    Ok(times[1])
}

fn encode(config: EncoderConfig, samples: &[i32]) -> Result<(Vec<u8>, Vec<Vec<u8>>), String> {
    let mut encoder =
        AlacEncoder::new(config, &ChannelLayout::stereo()).map_err(|err| err.to_string())?;
    let mut packets = encoder
        .push_samples(samples)
        .map_err(|err| err.to_string())?;
    packets.extend(encoder.drain().map_err(|err| err.to_string())?);
    let cookie = encoder.cookie().to_bytes();
    Ok((
        cookie,
        packets.into_iter().map(|packet| packet.data).collect(),
    ))
}

fn decode(cookie: &[u8], packets: &[Vec<u8>]) -> Result<Vec<i32>, String> {
    let mut decoder = AlacDecoder::new(cookie).map_err(|err| err.to_string())?;
    let mut out = Vec::new();
    for packet in packets {
        let frame = decoder
            .decode_frame(packet)
            .map_err(|err| err.to_string())?;
        out.extend_from_slice(&frame.samples);
    }
    Ok(out)
}

fn source(
    ffmpeg: Option<&Path>,
    dir: &Path,
    depth: u8,
) -> Result<(Vec<i32>, &'static str), String> {
    let real = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../local-files")
        .join(SOURCE);
    if let Some(ffmpeg) = ffmpeg
        && real.exists()
    {
        let out = dir.join(format!("real-{depth}.raw"));
        let duration = SECONDS.to_string();
        process(
            ffmpeg,
            &[
                "-v",
                "error",
                "-y",
                "-t",
                &duration,
                "-i",
                path_str(&real)?,
                "-ar",
                "44100",
                "-ac",
                "2",
                "-f",
                "s32le",
                path_str(&out)?,
            ],
        )?;
        let bytes = std::fs::read(&out).map_err(|err| err.to_string())?;
        let shift = 32 - u32::from(depth);
        let samples = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b) >> shift)
            .collect();
        return Ok((samples, "real speech and music"));
    }
    let peak = f64::from((1u32 << (depth - 1)) - 1);
    let mut state = 0x2545_f491u32;
    let mut samples = Vec::with_capacity(SECONDS * RATE as usize * 2);
    for i in 0..SECONDS * RATE as usize {
        let t = i as f64 / f64::from(RATE);
        for c in 0..2 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let noise = f64::from(state) / f64::from(u32::MAX) - 0.5;
            let tone = (std::f64::consts::TAU * (220.0 + 110.0 * f64::from(c)) * t).sin();
            samples.push((peak * (0.6 * tone + 0.05 * noise)).round() as i32);
        }
    }
    Ok((samples, "synthetic tones and noise"))
}

fn write_raw(path: &Path, samples: &[i32], depth: u8) -> Result<(), String> {
    let mut bytes = Vec::new();
    for &sample in samples {
        if depth == 16 {
            bytes.extend_from_slice(
                &i16::try_from(sample)
                    .map_err(|err| err.to_string())?
                    .to_le_bytes(),
            );
        } else {
            bytes.extend_from_slice(&(sample << (32 - u32::from(depth))).to_le_bytes());
        }
    }
    std::fs::write(path, bytes).map_err(|err| err.to_string())
}

fn process(program: &Path, args: &[&str]) -> Result<Duration, String> {
    let started = Instant::now();
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|err| err.to_string())?;
    let elapsed = started.elapsed();
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    Ok(elapsed)
}

fn path_str(path: &Path) -> Result<&str, String> {
    path.to_str().ok_or_else(|| "path is not UTF-8".to_string())
}

fn find_ffmpeg() -> Option<PathBuf> {
    [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "ffmpeg",
    ]
    .into_iter()
    .map(PathBuf::from)
    .find(|path| {
        Command::new(path)
            .arg("-version")
            .output()
            .is_ok_and(|output| output.status.success())
    })
}
