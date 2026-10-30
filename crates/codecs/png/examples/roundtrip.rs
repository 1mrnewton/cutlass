//! Decode one small RGBA PNG that ffmpeg writes.
//!
//! The printed width and height come from this crate. `match` means those
//! samples are the same bytes ffmpeg was given.

#![allow(clippy::print_stdout)]

use std::path::{Path, PathBuf};
use std::process::Command;

use cutlass_decode::{BitDepth, ChannelLayout, CodecId, CodecParams, ColorType, Decoder, Packet};
use cutlass_png::PngDecoder;

const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;

fn main() -> Result<(), String> {
    let Some(ffmpeg) = ffmpeg_bin() else {
        println!("skip: ffmpeg is not installed");
        return Ok(());
    };
    let dir = std::env::temp_dir().join(format!("cutlass-png-roundtrip-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|err| err.to_string())?;
    let outcome = decode_written_png(&dir, &ffmpeg);
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

fn decode_written_png(dir: &Path, ffmpeg: &Path) -> Result<(), String> {
    let raw = rgba_pattern();
    let png = dir.join("sample.png");
    write_png(ffmpeg, dir, &raw, &png)?;
    let bytes = std::fs::read(&png).map_err(|err| err.to_string())?;
    let params = CodecParams::audio(CodecId::Png, 1, ChannelLayout::mono());
    let mut decoder = PngDecoder::open(&params).map_err(|err| err.to_string())?;
    let packet = Packet::audio(bytes, 0, 0, 0);
    let pictures = decoder
        .decode_picture(&packet)
        .map_err(|err| err.to_string())?;
    let Some(picture) = pictures.first() else {
        return Err("decoder returned no picture".to_string());
    };
    let image = &picture.image;
    if image.color != ColorType::Rgba || image.depth != BitDepth::Eight {
        return Err("decoded picture is not 8-bit rgba".to_string());
    }
    let decoded = pack8(&image.samples)?;
    if image.width != WIDTH || image.height != HEIGHT || decoded != raw {
        return Err(format!(
            "pixel mismatch: {}x{} {} bytes, source {WIDTH}x{HEIGHT} {} bytes",
            image.width,
            image.height,
            decoded.len(),
            raw.len()
        ));
    }
    println!("width {}", image.width);
    println!("height {}", image.height);
    println!("match");
    Ok(())
}

fn write_png(ffmpeg: &Path, dir: &Path, raw: &[u8], png: &Path) -> Result<(), String> {
    let input = dir.join("sample.raw");
    std::fs::write(&input, raw).map_err(|err| err.to_string())?;
    let output = Command::new(ffmpeg)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            &format!("{WIDTH}x{HEIGHT}"),
            "-i",
        ])
        .arg(&input)
        .args(["-frames:v", "1"])
        .arg(png)
        .output()
        .map_err(|err| err.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "ffmpeg exited {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(())
}

fn rgba_pattern() -> Vec<u8> {
    let mut raw = Vec::with_capacity(8 * 8 * 4);
    for y in 0..8u8 {
        for x in 0..8u8 {
            raw.extend([x.saturating_mul(20), y.saturating_mul(20), 0x40, 0xff]);
        }
    }
    raw
}

fn pack8(samples: &[u16]) -> Result<Vec<u8>, String> {
    let mut packed = Vec::with_capacity(samples.len());
    for sample in samples {
        packed.push(u8::try_from(*sample).map_err(|_| "sample does not fit in a byte")?);
    }
    Ok(packed)
}

fn ffmpeg_bin() -> Option<PathBuf> {
    for candidate in [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "ffmpeg",
    ] {
        let Ok(output) = Command::new(candidate).arg("-version").output() else {
            continue;
        };
        if output.status.success() {
            return resolved(candidate);
        }
    }
    None
}

fn resolved(candidate: &str) -> Option<PathBuf> {
    if candidate.contains('/') {
        return Some(PathBuf::from(candidate));
    }
    let output = Command::new("which").arg(candidate).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let line = text.lines().next()?.trim();
    if line.is_empty() {
        None
    } else {
        Some(PathBuf::from(line))
    }
}
