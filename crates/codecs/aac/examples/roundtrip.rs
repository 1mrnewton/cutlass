//! Decode one short AAC file produced by ffmpeg.

#![allow(clippy::print_stdout)]

use std::path::{Path, PathBuf};
use std::process::Command;

use cutlass_aac::AacDecoder;
use cutlass_decode::{ChannelLayout, CodecId, CodecParams, Packet};

const SAMPLE_RATES: [u32; 13] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350,
];

fn main() -> Result<(), String> {
    let Ok(ffmpeg) = ffmpeg_bin() else {
        println!("skip: ffmpeg is not installed");
        return Ok(());
    };
    let dir = TempDir::new("cutlass-aac-example")?;
    let adts = dir.path().join("tiny.aac");
    encode_aac(&ffmpeg, &adts)?;
    let bytes = std::fs::read(&adts).map_err(|err| err.to_string())?;
    let stream = split_adts(&bytes)?;
    let layout = layout_for(stream.channels)?;
    let params = CodecParams::audio(CodecId::Aac, stream.sample_rate, layout);
    let mut decoder = AacDecoder::open(&params).map_err(|err| err.to_string())?;
    let mut samples = 0usize;
    for unit in &stream.units {
        let packet = Packet::audio(unit.clone(), 0, 1024, stream.sample_rate);
        let pcm = decoder
            .push_packet(&packet)
            .map_err(|err| err.to_string())?;
        samples = add_frames(samples, &pcm, stream.channels)?;
    }
    if let Some(tail) = decoder.drain_pcm().map_err(|err| err.to_string())? {
        samples = add_frames(samples, &tail, stream.channels)?;
    }
    if samples == 0 {
        return Err("decoder produced no samples".to_string());
    }
    let rate = decoder.config().sample_rate;
    let channels = decoder.config().layout.len();
    println!("sample rate: {rate}");
    println!("channels: {channels}");
    println!("samples: {samples}");
    Ok(())
}

struct Adts {
    sample_rate: u32,
    channels: usize,
    units: Vec<Vec<u8>>,
}

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(prefix: &str) -> Result<Self, String> {
        let path = std::env::temp_dir().join(format!("{prefix}-{}", std::process::id()));
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

fn encode_aac(ffmpeg: &Path, output: &Path) -> Result<(), String> {
    let status = Command::new(ffmpeg)
        .args([
            "-nostdin",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:sample_rate=48000:duration=0.1",
            "-ac",
            "1",
            "-c:a",
            "aac",
            "-profile:a",
            "aac_low",
            "-b:a",
            "96k",
            "-f",
            "adts",
        ])
        .arg(output)
        .output()
        .map_err(|err| err.to_string())?;
    if status.status.success() {
        Ok(())
    } else {
        Err(stderr_tail(&status.stderr))
    }
}

fn ffmpeg_bin() -> Result<PathBuf, ()> {
    for candidate in [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "ffmpeg",
    ] {
        let path = PathBuf::from(candidate);
        let ok = Command::new(&path)
            .arg("-version")
            .output()
            .is_ok_and(|output| output.status.success());
        if ok {
            return Ok(path);
        }
    }
    Err(())
}

fn split_adts(data: &[u8]) -> Result<Adts, String> {
    let mut units = Vec::new();
    let mut offset = 0usize;
    let mut sample_rate = None;
    let mut channels = None;
    while offset < data.len() {
        if offset + 7 > data.len() {
            return Err(format!("trailing {} bytes", data.len() - offset));
        }
        let first = byte(data, offset)?;
        let second = byte(data, offset + 1)?;
        if first != 0xFF || second & 0xF0 != 0xF0 {
            return Err(format!("adts sync at {offset}"));
        }
        let header = if second & 0x01 == 1 { 7 } else { 9 };
        let third = byte(data, offset + 2)?;
        let fourth = byte(data, offset + 3)?;
        let profile = third >> 6;
        if profile != 1 {
            return Err(format!("adts profile {profile}"));
        }
        let rate_index = usize::from((third >> 2) & 0x0F);
        let rate = SAMPLE_RATES
            .get(rate_index)
            .copied()
            .ok_or_else(|| format!("sample rate index {rate_index}"))?;
        let channel = usize::from(((third & 0x01) << 2) | (fourth >> 6));
        if sample_rate.get_or_insert(rate) != &rate || channels.get_or_insert(channel) != &channel {
            return Err("adts header changed".to_string());
        }
        let frame_len = (usize::from(fourth & 0x03) << 11)
            | (usize::from(byte(data, offset + 4)?) << 3)
            | (usize::from(byte(data, offset + 5)?) >> 5);
        if frame_len < header || offset + frame_len > data.len() {
            return Err(format!("adts length {frame_len} at {offset}"));
        }
        units.push(data[offset + header..offset + frame_len].to_vec());
        offset += frame_len;
    }
    let sample_rate = sample_rate.ok_or_else(|| "no adts frames".to_string())?;
    let channels = channels.ok_or_else(|| "no adts frames".to_string())?;
    if units.is_empty() {
        return Err("no adts frames".to_string());
    }
    Ok(Adts {
        sample_rate,
        channels,
        units,
    })
}

fn layout_for(channels: usize) -> Result<ChannelLayout, String> {
    match channels {
        1 => Ok(ChannelLayout::mono()),
        2 => Ok(ChannelLayout::stereo()),
        _ => Err(format!("channel config {channels}")),
    }
}

fn add_frames(total: usize, pcm: &[f32], channels: usize) -> Result<usize, String> {
    if channels == 0 || !pcm.len().is_multiple_of(channels) {
        return Err("decoded channel count".to_string());
    }
    Ok(total + pcm.len() / channels)
}

fn byte(data: &[u8], index: usize) -> Result<u8, String> {
    data.get(index)
        .copied()
        .ok_or_else(|| format!("adts truncated at {index}"))
}

fn stderr_tail(stderr: &[u8]) -> String {
    let text = String::from_utf8_lossy(stderr);
    let start = text.len().saturating_sub(400);
    text[start..].trim().to_string()
}
