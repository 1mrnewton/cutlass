//! Real-time factor of the FLAC encoder at levels 0, 5, and 8 and of the
//! decoder, beside the `flac` binary on the same audio.
//!
//! The audio is the first 60 seconds of `../local-files/videoplayback.m4a`
//! as 16-bit stereo, read through ffmpeg, or a synthetic stereo signal when
//! the folder or ffmpeg is missing. Each time is the median of three runs
//! after one warmup. Ours works on samples and frames already in memory.
//! The `flac` times start at process launch and include file reads and
//! writes, so they are a reference, not a like-for-like race.
//!
//! `cargo bench -p cutlass-flac --bench against_flac`

#![allow(clippy::print_stdout)]

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use cutlass_flac::{EncodedFrame, FlacDecoder, FlacEncoder, Settings};

const RATE: u32 = 44_100;
const CHANNELS: u8 = 2;
const SECONDS: u32 = 60;

fn main() -> Result<(), String> {
    let dir = TempDir::new()?;
    let (samples, source) = audio()?;
    let frames = samples.len() / usize::from(CHANNELS);
    let audio_seconds = frames as f64 / f64::from(RATE);
    println!("audio: {source}, {RATE} Hz, {CHANNELS} channels, 16 bits, {audio_seconds:.1} s");
    println!("each time: median of 3 after 1 warmup; rtf is audio seconds per second");

    let mut level_five = Vec::new();
    for level in [0u8, 5, 8] {
        let encoded = encode(&samples, level)?;
        let decoded = decode(&encoded)?;
        if decoded != samples {
            return Err(format!("level {level}: the round trip differs"));
        }
        let bytes: usize = encoded.iter().map(|frame| frame.data.len()).sum();
        let seconds = median_seconds(|| encode(&samples, level).map(|_| ()))?;
        println!(
            "encode level {level}: {seconds:.4} s, rtf {:.0}x, {bytes} frame bytes",
            audio_seconds / seconds
        );
        if level == 5 {
            level_five = encoded;
        }
    }
    let seconds = median_seconds(|| decode(&level_five).map(|_| ()))?;
    println!(
        "decode level 5 frames: {seconds:.4} s, rtf {:.0}x",
        audio_seconds / seconds
    );

    if !have("flac") {
        println!("skip: the flac binary is not installed");
        return Ok(());
    }
    let wav = dir.path().join("source.wav");
    write_wav(&wav, &samples)?;
    let flac_file = dir.path().join("libflac.flac");
    for level in ["-0", "-5", "-8"] {
        let seconds = median_seconds(|| {
            run_flac(&[
                "-s",
                "-f",
                level,
                path_str(&wav)?,
                "-o",
                path_str(&flac_file)?,
            ])
        })?;
        let bytes = std::fs::metadata(&flac_file)
            .map_err(|err| err.to_string())?
            .len();
        println!(
            "flac {level}: {seconds:.4} s, rtf {:.0}x, {bytes} file bytes",
            audio_seconds / seconds
        );
    }
    run_flac(&[
        "-s",
        "-f",
        "-5",
        path_str(&wav)?,
        "-o",
        path_str(&flac_file)?,
    ])?;
    let decoded = dir.path().join("decoded.wav");
    let seconds = median_seconds(|| {
        run_flac(&[
            "-s",
            "-f",
            "-d",
            path_str(&flac_file)?,
            "-o",
            path_str(&decoded)?,
        ])
    })?;
    println!(
        "flac -d of its -5 file: {seconds:.4} s, rtf {:.0}x",
        audio_seconds / seconds
    );
    Ok(())
}

fn encode(samples: &[i32], level: u8) -> Result<Vec<EncodedFrame>, String> {
    let settings = Settings::level(level).map_err(|err| err.to_string())?;
    let mut encoder =
        FlacEncoder::new(RATE, CHANNELS, 16, settings).map_err(|err| err.to_string())?;
    let mut frames = encoder.encode(samples).map_err(|err| err.to_string())?;
    frames.extend(encoder.finish().map_err(|err| err.to_string())?);
    Ok(frames)
}

fn decode(frames: &[EncodedFrame]) -> Result<Vec<i32>, String> {
    let mut decoder = FlacDecoder::new(None);
    let mut out = Vec::new();
    for frame in frames {
        out.extend(
            decoder
                .decode(&frame.data)
                .map_err(|err| err.to_string())?
                .samples,
        );
    }
    Ok(out)
}

fn median_seconds(mut run: impl FnMut() -> Result<(), String>) -> Result<f64, String> {
    run()?;
    let mut times = Vec::new();
    for _ in 0..3 {
        let started = Instant::now();
        run()?;
        times.push(started.elapsed().as_secs_f64());
    }
    times.sort_by(f64::total_cmp);
    Ok(times[1])
}

/// Real audio when it is here, a synthetic signal otherwise.
fn audio() -> Result<(Vec<i32>, String), String> {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files/videoplayback.m4a");
    if path.is_file() && have("ffmpeg") {
        let output = Command::new("ffmpeg")
            .args(["-v", "error", "-t", &SECONDS.to_string(), "-i"])
            .arg(&path)
            .args(["-ac", "2", "-ar", &RATE.to_string(), "-f", "s16le", "-"])
            .stderr(Stdio::null())
            .output()
            .map_err(|err| err.to_string())?;
        if output.status.success() && !output.stdout.is_empty() {
            let samples = output
                .stdout
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| i32::from(i16::from_le_bytes(*pair)))
                .collect();
            return Ok((samples, "local-files/videoplayback.m4a".to_string()));
        }
    }
    println!("note: local-files or ffmpeg is missing, using a synthetic signal");
    let mut state = 0x00F1_AC00_u64;
    let frames = usize::try_from(RATE * SECONDS).map_err(|err| err.to_string())?;
    let mut samples = Vec::with_capacity(frames * usize::from(CHANNELS));
    for index in 0..frames {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let noise = ((state >> 40) as f64 / f64::from(1u32 << 24)) - 0.5;
        let time = index as f64 / f64::from(RATE);
        let tone = (time * 220.0 * std::f64::consts::TAU).sin() * 0.4
            + (time * 331.0 * std::f64::consts::TAU).sin() * 0.2;
        let left = ((tone + noise * 0.05) * 30_000.0) as i32;
        let right = ((tone * 0.8 - noise * 0.05) * 30_000.0) as i32;
        samples.extend([left, right]);
    }
    Ok((samples, "synthetic".to_string()))
}

fn write_wav(path: &Path, samples: &[i32]) -> Result<(), String> {
    let data_len = u32::try_from(samples.len() * 2).map_err(|err| err.to_string())?;
    let mut bytes = Vec::with_capacity(44 + samples.len() * 2);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&u16::from(CHANNELS).to_le_bytes());
    bytes.extend_from_slice(&RATE.to_le_bytes());
    bytes.extend_from_slice(&(RATE * u32::from(CHANNELS) * 2).to_le_bytes());
    bytes.extend_from_slice(&(u16::from(CHANNELS) * 2).to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        let sample = i16::try_from(*sample).map_err(|err| err.to_string())?;
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    std::fs::write(path, bytes).map_err(|err| err.to_string())
}

fn have(program: &str) -> bool {
    let flag = if program.starts_with("ff") {
        "-version"
    } else {
        "--version"
    };
    Command::new(program)
        .arg(flag)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn run_flac(args: &[&str]) -> Result<(), String> {
    let status = Command::new("flac")
        .args(args)
        .status()
        .map_err(|err| err.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("flac {args:?} failed"))
    }
}

fn path_str(path: &Path) -> Result<&str, String> {
    path.to_str()
        .ok_or_else(|| format!("{} is not UTF-8", path.display()))
}

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new() -> Result<Self, String> {
        let path = std::env::temp_dir().join(format!("cutlass-flac-bench-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).map_err(|err| err.to_string())?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
