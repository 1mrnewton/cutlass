//! Decode a tiny High-profile Annex B file that ffmpeg writes in a temp directory.
//!
//! `cargo run -p cutlass-h264 --example roundtrip`
//!
//! When ffmpeg is absent the process prints `skip: ffmpeg is not installed` and exits 0.

#![allow(clippy::print_stdout)]

use std::path::Path;
use std::process::Command;

use cutlass_decode::{CodecId, CodecParams, Packet, VideoDecoder};
use cutlass_h264::H264Decoder;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;
const FRAMES: usize = 4;

fn main() -> Result<(), String> {
    let Ok(ffmpeg) = ffmpeg_bin() else {
        println!("skip: ffmpeg is not installed");
        return Ok(());
    };
    let dir = std::env::temp_dir().join(format!("cutlass-h264-roundtrip-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(|err| err.to_string())?;
    let outcome = decode_fixture(&ffmpeg, &dir);
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

fn decode_fixture(ffmpeg: &str, dir: &Path) -> Result<(), String> {
    let coded = dir.join("tiny.h264");
    let frames = FRAMES.to_string();
    let output = Command::new(ffmpeg)
        .args([
            "-nostdin",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=30",
            "-frames:v",
        ])
        .arg(&frames)
        .args([
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            "-x264-params",
            "cabac=1:bframes=1:weightp=0:8x8dct=1:keyint=30:scenecut=0:ref=1:threads=1:b-pyramid=0:direct=spatial",
            "-f",
            "h264",
        ])
        .arg(&coded)
        .output()
        .map_err(|err| err.to_string())?;
    if !output.status.success() {
        return Err(tail(&output.stderr));
    }
    let bytes = std::fs::read(&coded).map_err(|err| err.to_string())?;
    let units = access_units(&bytes)?;
    let mut params = CodecParams::video(CodecId::H264, WIDTH, HEIGHT);
    params.nal_length_size = 0;
    let mut decoder = H264Decoder::open(&params).map_err(|err| err.to_string())?;
    let mut pictures = Vec::new();
    for (index, unit) in units.iter().enumerate() {
        let pts = i64::try_from(index).map_err(|_| format!("frame index {index}"))?;
        let produced = decoder
            .push(&Packet::video(unit.clone(), pts, 1, 30, index == 0))
            .map_err(|err| err.to_string())?;
        pictures.extend(produced);
    }
    pictures.extend(decoder.drain().map_err(|err| err.to_string())?);
    let Some(first) = pictures.first() else {
        return Err("decoder returned no pictures".to_string());
    };
    let width = first.picture.width();
    let height = first.picture.height();
    if pictures
        .iter()
        .any(|frame| frame.picture.width() != width || frame.picture.height() != height)
    {
        return Err("decoded pictures do not share one size".to_string());
    }
    println!("frames {}", pictures.len());
    println!("size {width}x{height}");
    Ok(())
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
        let starts_picture = vcl && data.get(body + 1).is_some_and(|byte| byte & 0x80 != 0);
        if (kind == 9 || (starts_picture && saw_vcl)) && !current.is_empty() {
            units.push(std::mem::take(&mut current));
            saw_vcl = false;
        }
        current.extend_from_slice(&data[header..end]);
        saw_vcl |= vcl;
    }
    if !current.is_empty() {
        units.push(current);
    }
    if units.is_empty() {
        return Err("annex b had no access unit".to_string());
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
