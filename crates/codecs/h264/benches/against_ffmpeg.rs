//! Throughput against ffmpeg on one thread.
//!
//! `cargo bench -p cutlass-h264 --offline`
//!
//! Decode is the same High annex-B stream. Encode is the same `testsrc2`
//! pictures. libx264 is ultrafast; this encoder has no preset. The numbers
//! are wall time, not a quality match.

#![allow(clippy::print_stdout)]

use std::path::Path;
use std::process::Command;
use std::time::Instant;

use cutlass_decode::{CodecId, CodecParams, Packet, VideoDecoder, Yuv420p};
use cutlass_encode::{VideoEncoder, VideoInput};
use cutlass_h264::{H264Decoder, H264Encoder};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const DECODE_FRAMES: usize = 120;
const ENCODE_FRAMES: usize = 30;
const BITRATE: u32 = 2_000_000;
const Y_LEN: usize = 1280 * 720;
const C_LEN: usize = Y_LEN / 4;
const FRAME_LEN: usize = Y_LEN + C_LEN * 2;

fn main() -> Result<(), String> {
    let Ok(ffmpeg) = ffmpeg_bin() else {
        println!("skip: ffmpeg is not installed");
        return Ok(());
    };
    let dir = std::env::temp_dir().join("cutlass-h264-bench");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(|err| err.to_string())?;

    let coded = dir.join("high.h264");
    generate_stream(&ffmpeg, &coded)?;
    let bytes = std::fs::read(&coded).map_err(|err| err.to_string())?;
    let units = access_units(&bytes)?;
    let packets = packets(&units)?;
    let decode_ours = median_seconds(|| time_decode(&packets))?;
    let decode_ffmpeg = median_seconds(|| time_ffmpeg_decode(&ffmpeg, &coded))?;

    let raw_path = dir.join("source.yuv");
    generate_yuv(&ffmpeg, &raw_path)?;
    let raw = std::fs::read(&raw_path).map_err(|err| err.to_string())?;
    let inputs = inputs(&raw)?;
    let _ = time_encode(&inputs)?;
    let encode_ours = time_encode(&inputs)?;
    let encode_ffmpeg = time_ffmpeg_encode(&ffmpeg, &raw_path)?;

    println!("1280x720, one thread");
    println!("cutlass times the codec loop. ffmpeg times include process startup.");
    println!("decode High annex-B, {DECODE_FRAMES} frames, median of 3");
    print_row("cutlass", DECODE_FRAMES, decode_ours);
    print_row("ffmpeg", DECODE_FRAMES, decode_ffmpeg);
    print_ratio(decode_ffmpeg, decode_ours);
    println!("encode testsrc2, {ENCODE_FRAMES} frames, one run, {BITRATE} bits/s");
    println!("libx264 preset ultrafast, profile high");
    print_row("cutlass", ENCODE_FRAMES, encode_ours);
    print_row("ffmpeg", ENCODE_FRAMES, encode_ffmpeg);
    print_ratio(encode_ffmpeg, encode_ours);
    Ok(())
}

fn print_row(name: &str, frames: usize, seconds: f64) {
    let fps = fps(frames, seconds);
    println!("{name:<8} {seconds:>8.2} s   {fps:>8.1} fps");
}

fn print_ratio(ffmpeg_seconds: f64, ours_seconds: f64) {
    if ffmpeg_seconds > 0.0 {
        println!("ffmpeg was {:.1}x as fast", ours_seconds / ffmpeg_seconds);
    }
}

fn fps(frames: usize, seconds: f64) -> f64 {
    let frames = u32::try_from(frames).unwrap_or(u32::MAX);
    f64::from(frames) / seconds
}

fn median_seconds(mut once: impl FnMut() -> Result<f64, String>) -> Result<f64, String> {
    let _ = once()?;
    let mut samples = [once()?, once()?, once()?];
    samples.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    Ok(samples[1])
}

fn time_decode(packets: &[Packet]) -> Result<f64, String> {
    let mut params = CodecParams::video(CodecId::H264, WIDTH, HEIGHT);
    params.nal_length_size = 0;
    let mut decoder = H264Decoder::open(&params).map_err(|err| err.to_string())?;
    let started = Instant::now();
    let mut pictures = 0usize;
    let mut luma = 0usize;
    for packet in packets {
        let frames = decoder.push(packet).map_err(|err| err.to_string())?;
        pictures += frames.len();
        luma += frames
            .iter()
            .map(|frame| frame.picture.y().len())
            .sum::<usize>();
    }
    let tail = decoder.drain().map_err(|err| err.to_string())?;
    pictures += tail.len();
    luma += tail
        .iter()
        .map(|frame| frame.picture.y().len())
        .sum::<usize>();
    let seconds = started.elapsed().as_secs_f64();
    if pictures != DECODE_FRAMES || luma == 0 {
        return Err(format!("decoder returned {pictures} pictures"));
    }
    Ok(seconds)
}

fn time_encode(inputs: &[VideoInput]) -> Result<f64, String> {
    let mut params = CodecParams::video(CodecId::H264, WIDTH, HEIGHT);
    params.nal_length_size = 0;
    params.bitrate = BITRATE;
    let mut encoder = H264Encoder::open(&params).map_err(|err| err.to_string())?;
    let started = Instant::now();
    let mut packets = 0usize;
    let mut bytes = 0usize;
    for input in inputs {
        let produced = encoder.push(input).map_err(|err| err.to_string())?;
        packets += produced.len();
        bytes += produced
            .iter()
            .map(|packet| packet.data.len())
            .sum::<usize>();
    }
    let tail = encoder.drain().map_err(|err| err.to_string())?;
    packets += tail.len();
    bytes += tail.iter().map(|packet| packet.data.len()).sum::<usize>();
    let seconds = started.elapsed().as_secs_f64();
    if packets != ENCODE_FRAMES || bytes == 0 {
        return Err(format!("encoder returned {packets} packets"));
    }
    Ok(seconds)
}

fn time_ffmpeg_decode(ffmpeg: &str, path: &Path) -> Result<f64, String> {
    let started = Instant::now();
    let output = Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-threads", "1", "-i"])
        .arg(path)
        .args(["-f", "null", "-"])
        .output()
        .map_err(|err| err.to_string())?;
    if !output.status.success() {
        return Err(tail(&output.stderr));
    }
    Ok(started.elapsed().as_secs_f64())
}

fn time_ffmpeg_encode(ffmpeg: &str, yuv: &Path) -> Result<f64, String> {
    let started = Instant::now();
    let output = Command::new(ffmpeg)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s",
            "1280x720",
            "-r",
            "30",
            "-i",
        ])
        .arg(yuv)
        .args([
            "-frames:v",
            "30",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-profile:v",
            "high",
            "-threads",
            "1",
            "-b:v",
            "2000k",
            "-f",
            "null",
            "-",
        ])
        .output()
        .map_err(|err| err.to_string())?;
    if !output.status.success() {
        return Err(tail(&output.stderr));
    }
    Ok(started.elapsed().as_secs_f64())
}

fn generate_stream(ffmpeg: &str, path: &Path) -> Result<(), String> {
    let frames = DECODE_FRAMES.to_string();
    let output = Command::new(ffmpeg)
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1280x720:rate=30",
            "-frames:v",
        ])
        .arg(&frames)
        .args([
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            "-x264-params",
            "cabac=1:bframes=2",
            "-f",
            "h264",
        ])
        .arg(path)
        .output()
        .map_err(|err| err.to_string())?;
    if output.status.success() {
        Ok(())
    } else {
        Err(tail(&output.stderr))
    }
}

fn generate_yuv(ffmpeg: &str, path: &Path) -> Result<(), String> {
    let frames = ENCODE_FRAMES.to_string();
    let output = Command::new(ffmpeg)
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1280x720:rate=30",
            "-frames:v",
        ])
        .arg(&frames)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(path)
        .output()
        .map_err(|err| err.to_string())?;
    if output.status.success() {
        Ok(())
    } else {
        Err(tail(&output.stderr))
    }
}

fn packets(units: &[Vec<u8>]) -> Result<Vec<Packet>, String> {
    if units.len() != DECODE_FRAMES {
        return Err(format!("stream has {} access units", units.len()));
    }
    let mut packets = Vec::with_capacity(units.len());
    for (index, unit) in units.iter().enumerate() {
        let pts = i64::try_from(index).unwrap_or(0);
        packets.push(Packet::video(unit.clone(), pts, 1, 30, index == 0));
    }
    Ok(packets)
}

fn inputs(raw: &[u8]) -> Result<Vec<VideoInput>, String> {
    if raw.len() < FRAME_LEN * ENCODE_FRAMES {
        return Err(format!("yuv is {} bytes", raw.len()));
    }
    let mut inputs = Vec::with_capacity(ENCODE_FRAMES);
    for index in 0..ENCODE_FRAMES {
        let start = index * FRAME_LEN;
        let plane = &raw[start..start + FRAME_LEN];
        let picture = Yuv420p::try_new(
            WIDTH,
            HEIGHT,
            plane[..Y_LEN].to_vec(),
            plane[Y_LEN..Y_LEN + C_LEN].to_vec(),
            plane[Y_LEN + C_LEN..].to_vec(),
        )
        .map_err(|err| err.to_string())?;
        inputs.push(VideoInput {
            picture,
            pts: i64::try_from(index).unwrap_or(0),
            duration: 1,
            timescale: 30,
            force_keyframe: index == 0,
        });
    }
    Ok(inputs)
}

fn access_units(data: &[u8]) -> Result<Vec<Vec<u8>>, String> {
    let mut starts = Vec::new();
    let mut index = 0usize;
    while index + 3 < data.len() {
        if data[index] == 0 && data[index + 1] == 0 && data[index + 2] == 1 {
            let header = if index > 0 && data[index - 1] == 0 {
                index - 1
            } else {
                index
            };
            starts.push((header, index + 3));
            index += 3;
        } else {
            index += 1;
        }
    }
    if starts.is_empty() {
        return Err("annex b had no start code".to_string());
    }
    let mut units = Vec::new();
    let mut current = Vec::new();
    let mut saw_vcl = false;
    for (n, &(header, body)) in starts.iter().enumerate() {
        let end = starts
            .get(n + 1)
            .map(|(next, _)| *next)
            .unwrap_or(data.len());
        if body >= data.len() || header >= end {
            continue;
        }
        let kind = data[body] & 31;
        let vcl = kind == 1 || kind == 5;
        if (kind == 9 || (vcl && saw_vcl)) && !current.is_empty() {
            units.push(std::mem::take(&mut current));
            saw_vcl = false;
        }
        current.extend_from_slice(&data[header..end]);
        saw_vcl |= vcl;
    }
    if !current.is_empty() {
        units.push(current);
    }
    Ok(units)
}

fn ffmpeg_bin() -> Result<String, String> {
    for path in [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "ffmpeg",
    ] {
        if let Ok(output) = Command::new(path).arg("-version").output()
            && output.status.success()
        {
            return Ok(path.to_string());
        }
    }
    Err("ffmpeg is not installed".to_string())
}

fn tail(stderr: &[u8]) -> String {
    let text = String::from_utf8_lossy(stderr);
    let start = text.len().saturating_sub(400);
    text[start..].trim().to_string()
}
