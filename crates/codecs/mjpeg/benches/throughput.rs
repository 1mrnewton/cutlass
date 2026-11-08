//! Milliseconds per 1080p frame for Motion JPEG encode and decode, one
//! thread, beside FFmpeg's decoder on the same stream, and resident memory
//! across 1 000 frames.
//!
//! The pictures are real: FFmpeg decodes the first frames of the 1080p clip
//! in `../local-files` to RGBA. Each timing is the best of five passes over
//! `MJPEG_BENCH_FRAMES` pictures (60 by default). The FFmpeg line is one
//! `ffmpeg -threads 1` process decoding the same stream, so it includes
//! startup.
//!
//! Run: `cargo bench -p cutlass-mjpeg --bench throughput`. Without FFmpeg
//! or the clip the bench prints why and exits 0.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use cutlass_decode::{CodecId, CodecParams, Decoder, Packet, PixelFormat};
use cutlass_mjpeg::{MjpegDecoder, MjpegEncoder, MjpegPictureDecoder, RateMode, Sampling};

const CLIP: &str = "16929689_1920_1080_30fps.mp4";
const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const FRAME_BYTES: usize = (WIDTH * HEIGHT * 4) as usize;
const SOURCE_FRAMES: usize = 6;
const PASSES: usize = 5;
const DECODE_BUDGET_MS: f64 = 16.0;
const ENCODE_BUDGET_MS: f64 = 33.0;

fn main() -> Result<(), String> {
    let Some(ffmpeg) = tool("ffmpeg") else {
        say("bench not run: ffmpeg is not installed");
        return Ok(());
    };
    let Some(clip) = local(CLIP) else {
        say(&format!("bench not run: local-files/{CLIP} is missing"));
        return Ok(());
    };
    let count = std::env::var("MJPEG_BENCH_FRAMES")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&count: &usize| count > 0)
        .unwrap_or(60);
    say(&machine());
    say(&format!("ffmpeg: {}", version(&ffmpeg)));
    let frames = source(&ffmpeg, &clip)?;
    say(&format!(
        "{WIDTH}x{HEIGHT}, {count} pictures per pass from {SOURCE_FRAMES} real frames, best of {PASSES} passes, one thread"
    ));

    let (encode_420, stream_420) = encode(&frames, count, Sampling::Yuv420, RateMode::Quality(75))?;
    let (encode_422, stream_422) = encode(&frames, count, Sampling::Yuv422, RateMode::Quality(75))?;
    let (encode_rate, stream_rate) = encode(
        &frames,
        count,
        Sampling::Yuv420,
        RateMode::Bitrate(60_000_000),
    )?;
    let mbps = |stream: &[Vec<u8>]| {
        stream.iter().map(Vec::len).sum::<usize>() as f64 * 8.0 * 30.0 / stream.len() as f64 / 1e6
    };
    say(&format!(
        "encode RGBA 4:2:0 q75: {encode_420:.2} ms/frame, {:.0} KB/frame, {:.1} Mb/s at 30 fps",
        average(&stream_420) / 1000.0,
        mbps(&stream_420)
    ));
    say(&format!(
        "encode RGBA 4:2:2 q75: {encode_422:.2} ms/frame, {:.0} KB/frame",
        average(&stream_422) / 1000.0
    ));
    say(&format!(
        "encode RGBA 4:2:0 at 60 Mb/s: {encode_rate:.2} ms/frame, {:.1} Mb/s delivered",
        mbps(&stream_rate)
    ));

    let ffmpeg_422 = ffmpeg_stream(&ffmpeg, &clip, count)?;
    for (label, stream) in [
        ("this encoder 4:2:0 q75", &stream_420),
        ("this encoder 4:2:2 q75", &stream_422),
        ("ffmpeg 4:2:2 -q:v 3", &ffmpeg_422),
    ] {
        let planar = best(|| decode_planar(stream))?;
        let rgba = best(|| decode_rgba(stream))?;
        let reference = time_ffmpeg(&ffmpeg, stream)?;
        say(&format!(
            "decode {label}: planar {planar:.2} ms/frame, RGBA {rgba:.2} ms/frame; ffmpeg -threads 1 {reference:.2} ms/frame"
        ));
        if label == "this encoder 4:2:0 q75" {
            gate("decode to planar", planar, DECODE_BUDGET_MS);
            gate("decode to RGBA", rgba, DECODE_BUDGET_MS);
        }
    }
    gate("encode q75", encode_420, ENCODE_BUDGET_MS);
    memory(&frames, &stream_420)?;
    Ok(())
}

/// The best pass of `run`, which returns milliseconds per frame.
fn best(mut run: impl FnMut() -> Result<f64, String>) -> Result<f64, String> {
    let mut best = f64::INFINITY;
    for _ in 0..PASSES {
        best = best.min(run()?);
    }
    Ok(best)
}

fn encoder(sampling: Sampling, mode: RateMode) -> Result<MjpegEncoder, String> {
    let params = CodecParams::with_format(CodecId::Mjpeg, WIDTH, HEIGHT, PixelFormat::Rgba8);
    MjpegEncoder::open(&params)
        .map_err(|error| format!("encoder: {error}"))?
        .with_sampling(sampling)
        .with_mode(mode)
        .map_err(|error| format!("mode: {error}"))
}

fn encode(
    frames: &[Vec<u8>],
    count: usize,
    sampling: Sampling,
    mode: RateMode,
) -> Result<(f64, Vec<Vec<u8>>), String> {
    let mut stream = Vec::new();
    let ms = best(|| {
        let mut encoder = encoder(sampling, mode)?;
        stream.clear();
        let start = Instant::now();
        for index in 0..count {
            let rgba = &frames[index % frames.len()];
            stream.push(
                encoder
                    .encode_rgba(WIDTH, HEIGHT, rgba, 1, 30)
                    .map_err(|error| format!("encode: {error}"))?,
            );
        }
        Ok(start.elapsed().as_secs_f64() * 1000.0 / count as f64)
    })?;
    Ok((ms, stream))
}

fn params() -> CodecParams {
    CodecParams::with_format(CodecId::Mjpeg, WIDTH, HEIGHT, PixelFormat::Rgba8)
}

fn decode_planar(stream: &[Vec<u8>]) -> Result<f64, String> {
    let mut decoder = MjpegDecoder::open(&params()).map_err(|error| format!("decoder: {error}"))?;
    let start = Instant::now();
    for bytes in stream {
        let frame = decoder
            .decode(bytes)
            .and_then(cutlass_mjpeg::Picture::into_frame)
            .map_err(|error| format!("decode: {error}"))?;
        std::hint::black_box(frame);
    }
    Ok(start.elapsed().as_secs_f64() * 1000.0 / stream.len() as f64)
}

fn decode_rgba(stream: &[Vec<u8>]) -> Result<f64, String> {
    let mut decoder =
        MjpegPictureDecoder::open(&params()).map_err(|error| format!("decoder: {error}"))?;
    let start = Instant::now();
    for (index, bytes) in stream.iter().enumerate() {
        let packet = Packet::video(bytes.clone(), index as i64, 1, 30, true);
        let out = decoder
            .push(&packet)
            .map_err(|error| format!("decode: {error}"))?;
        std::hint::black_box(out);
    }
    Ok(start.elapsed().as_secs_f64() * 1000.0 / stream.len() as f64)
}

/// Resident memory after 100 and after 1 000 frames of decode, then of
/// encode, from `ps`.
fn memory(frames: &[Vec<u8>], stream: &[Vec<u8>]) -> Result<(), String> {
    let mut decoder =
        MjpegPictureDecoder::open(&params()).map_err(|error| format!("decoder: {error}"))?;
    let mut encoder = encoder(Sampling::Yuv420, RateMode::Bitrate(60_000_000))?;
    let mut decode_marks = Vec::new();
    for index in 0..1_000 {
        let packet = Packet::video(
            stream[index % stream.len()].clone(),
            index as i64,
            1,
            30,
            true,
        );
        std::hint::black_box(
            decoder
                .push(&packet)
                .map_err(|error| format!("decode: {error}"))?,
        );
        if index == 99 || index == 999 {
            decode_marks.push(resident_kib()?);
        }
    }
    let mut encode_marks = Vec::new();
    for index in 0..1_000 {
        std::hint::black_box(
            encoder
                .encode_rgba(WIDTH, HEIGHT, &frames[index % frames.len()], 1, 30)
                .map_err(|error| format!("encode: {error}"))?,
        );
        if index == 99 || index == 999 {
            encode_marks.push(resident_kib()?);
        }
    }
    let growth = |marks: &[u64]| marks[1] as i64 - marks[0] as i64;
    say(&format!(
        "memory: decode resident {} KiB at frame 100, {} KiB at 1000 ({:+} KiB); encode at 60 Mb/s {} KiB, {} KiB ({:+} KiB)",
        decode_marks[0],
        decode_marks[1],
        growth(&decode_marks),
        encode_marks[0],
        encode_marks[1],
        growth(&encode_marks)
    ));
    let flat = growth(&decode_marks) <= 1024 && growth(&encode_marks) <= 1024;
    say(&format!(
        "gate memory: no growth over 1 MiB across 900 frames: {}",
        if flat { "met" } else { "missed" }
    ));
    Ok(())
}

fn resident_kib() -> Result<u64, String> {
    let out = Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .map_err(|error| format!("ps: {error}"))?;
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .map_err(|error| format!("ps: {error}"))
}

fn gate(what: &str, ms: f64, budget: f64) {
    say(&format!(
        "gate {what}: {ms:.2} ms against {budget} ms: {}",
        if ms < budget { "met" } else { "missed" }
    ));
}

fn average(stream: &[Vec<u8>]) -> f64 {
    stream.iter().map(Vec::len).sum::<usize>() as f64 / stream.len().max(1) as f64
}

/// The first frames of the clip, as RGBA.
fn source(ffmpeg: &Path, clip: &Path) -> Result<Vec<Vec<u8>>, String> {
    let out = Command::new(ffmpeg)
        .args(["-v", "error", "-i"])
        .arg(clip)
        .args(["-frames:v", &SOURCE_FRAMES.to_string(), "-an"])
        .args(["-pix_fmt", "rgba", "-f", "rawvideo", "-"])
        .output()
        .map_err(|error| format!("ffmpeg: {error}"))?;
    if !out.status.success() {
        return Err(format!("ffmpeg: {}", String::from_utf8_lossy(&out.stderr)));
    }
    Ok(out
        .stdout
        .as_chunks::<FRAME_BYTES>()
        .0
        .iter()
        .map(|frame| frame.to_vec())
        .collect())
}

/// FFmpeg's own Motion JPEG of the clip: 4:2:2, the layout cameras write.
fn ffmpeg_stream(ffmpeg: &Path, clip: &Path, count: usize) -> Result<Vec<Vec<u8>>, String> {
    let out = Command::new(ffmpeg)
        .args(["-v", "error", "-i"])
        .arg(clip)
        .args(["-frames:v", &count.to_string(), "-an", "-c:v", "mjpeg"])
        .args(["-q:v", "3", "-pix_fmt", "yuvj422p", "-f", "mjpeg", "-"])
        .output()
        .map_err(|error| format!("ffmpeg: {error}"))?;
    if !out.status.success() {
        return Err(format!("ffmpeg: {}", String::from_utf8_lossy(&out.stderr)));
    }
    split(&out.stdout)
}

/// Each image of a concatenated stream: SOI up to the EOI before the next SOI.
fn split(bytes: &[u8]) -> Result<Vec<Vec<u8>>, String> {
    let mut starts: Vec<usize> = bytes
        .windows(3)
        .enumerate()
        .filter(|(at, window)| {
            window == &[0xff, 0xd8, 0xff] && (*at == 0 || bytes[at - 2..*at] == [0xff, 0xd9])
        })
        .map(|(at, _)| at)
        .collect();
    if starts.is_empty() {
        return Err("ffmpeg wrote no images".to_owned());
    }
    starts.push(bytes.len());
    Ok(starts
        .windows(2)
        .map(|pair| bytes[pair[0]..pair[1]].to_vec())
        .collect())
}

fn time_ffmpeg(ffmpeg: &Path, stream: &[Vec<u8>]) -> Result<f64, String> {
    let dir = std::env::temp_dir().join(format!("cutlass-mjpeg-bench-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|error| format!("temp directory: {error}"))?;
    let path = dir.join("stream.mjpeg");
    std::fs::write(&path, stream.concat()).map_err(|error| format!("write: {error}"))?;
    let start = Instant::now();
    let out = Command::new(ffmpeg)
        .args(["-v", "error", "-threads", "1", "-f", "mjpeg", "-i"])
        .arg(&path)
        .args(["-f", "null", "-"])
        .output();
    let elapsed = start.elapsed().as_secs_f64();
    let _ = std::fs::remove_dir_all(&dir);
    let out = out.map_err(|error| format!("ffmpeg: {error}"))?;
    if !out.status.success() {
        return Err(format!("ffmpeg: {}", String::from_utf8_lossy(&out.stderr)));
    }
    Ok(elapsed * 1000.0 / stream.len() as f64)
}

fn local(name: &str) -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = root.ancestors().nth(3)?.join("../local-files").join(name);
    path.is_file().then_some(path)
}

fn tool(name: &str) -> Option<PathBuf> {
    let homebrew = PathBuf::from("/opt/homebrew/bin").join(name);
    if homebrew.is_file() {
        return Some(homebrew);
    }
    let paths = std::env::var_os("PATH")?;
    std::env::split_paths(&paths)
        .map(|dir| dir.join(name))
        .find(|path| path.is_file())
}

fn version(ffmpeg: &Path) -> String {
    Command::new(ffmpeg)
        .arg("-version")
        .output()
        .ok()
        .and_then(|out| {
            String::from_utf8_lossy(&out.stdout)
                .lines()
                .next()
                .map(str::to_owned)
        })
        .unwrap_or_else(|| "unknown".to_owned())
}

fn machine() -> String {
    let read = |key: &str| {
        Command::new("sysctl")
            .args(["-n", key])
            .output()
            .ok()
            .filter(|out| out.status.success())
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
    };
    format!(
        "machine: {}, {} cores; bench profile",
        read("machdep.cpu.brand_string").unwrap_or_else(|| "unknown cpu".to_owned()),
        read("hw.ncpu").unwrap_or_else(|| "?".to_owned())
    )
}

fn say(line: &str) {
    let _ = writeln!(std::io::stdout(), "{line}");
}
