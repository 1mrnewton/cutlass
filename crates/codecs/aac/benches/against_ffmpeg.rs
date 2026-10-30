//! One short AAC-LC clip timed against the ffmpeg binary.
//!
//! Decode and encode each report the median of three runs after one warmup.
//! Sample comparison happens before those timed runs. ffmpeg's times start
//! when its process is launched.

#![allow(clippy::print_stdout)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use cutlass_aac::{AacDecoder, AacEncoder};
use cutlass_decode::{ChannelLayout, CodecId, CodecParams, Packet};
use cutlass_encode::{AudioInput, Encoder};

const RATE: u32 = 48_000;
const BITRATE: u32 = 96_000;
const SAMPLE_COUNT: usize = 48_000 * 3;
const MAX_LAG: usize = 1024;
const MIN_SNR_DB: f64 = 40.0;
const LENGTH_SLOP: usize = 4096;
const SAMPLE_RATES: [u32; 13] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350,
];

fn main() -> Result<(), String> {
    let Ok(ffmpeg) = ffmpeg_bin() else {
        println!("skip: ffmpeg is not installed");
        return Ok(());
    };
    let dir = TempDir::new()?;
    let samples = sine(SAMPLE_COUNT);
    let pcm_path = dir.path().join("source.f32");
    let aac_path = dir.path().join("source.aac");
    write_f32(&pcm_path, &samples)?;
    ffmpeg_encode(&ffmpeg, &pcm_path, &aac_path)?;
    let adts = std::fs::read(&aac_path).map_err(|err| err.to_string())?;
    let stream = split_adts(&adts)?;
    if stream.sample_rate != RATE || stream.channels != 1 {
        return Err(format!(
            "fixture is {} Hz, {} channels",
            stream.sample_rate, stream.channels
        ));
    }
    let packets = to_packets(&stream.units);
    let ours = decode_packets(&packets)?;
    let foreign_path = dir.path().join("ffmpeg.f32");
    let foreign = ffmpeg_decode(&ffmpeg, &aac_path, &foreign_path)?;
    let compared = compare(&ours, &foreign);
    println!(
        "sample: {RATE} Hz, 1 channel, {SAMPLE_COUNT} samples, 1000 Hz sine, AAC-LC {BITRATE} bps"
    );
    println!(
        "decode gate: snr at least {MIN_SNR_DB} dB, length difference at most {LENGTH_SLOP} samples"
    );
    println!("decode compare: {compared}");
    if !compared.agree {
        return Err(format!("decode samples disagree with ffmpeg: {compared}"));
    }

    let decode_ours = median_seconds(|| time_decode(&packets))?;
    let decode_ffmpeg = median_seconds(|| time_ffmpeg_decode(&ffmpeg, &aac_path))?;
    println!("decode: median of 3 after 1 warmup");
    println!("cutlass {decode_ours:.6} s");
    println!("ffmpeg {decode_ffmpeg:.6} s");

    let (_warmup, encoded) = encode_once(&samples)?;
    let encoded_path = dir.path().join("cutlass.aac");
    std::fs::write(&encoded_path, wrap_adts(&encoded)?).map_err(|err| err.to_string())?;
    let cutlass_pcm = dir.path().join("cutlass.f32");
    let decoded = ffmpeg_decode(&ffmpeg, &encoded_path, &cutlass_pcm)?;
    if decoded.is_empty() || decoded.iter().any(|sample| !sample.is_finite()) {
        return Err("ffmpeg decode of the cutlass bitstream was empty or non-finite".to_string());
    }
    let encode_ours = median_seconds(|| encode_seconds(&samples))?;
    let encode_ffmpeg = median_seconds(|| time_ffmpeg_encode(&ffmpeg, &pcm_path))?;
    println!("encode: median of 3 after 1 warmup");
    println!("cutlass {encode_ours:.6} s");
    println!("ffmpeg {encode_ffmpeg:.6} s");
    let decoded_samples = decoded.len();
    println!("ffmpeg decoded the cutlass bitstream: {decoded_samples} samples");
    println!("cutlass decodes packets already in memory. ffmpeg's time starts at process launch.");
    Ok(())
}

struct Adts {
    sample_rate: u32,
    channels: usize,
    units: Vec<Vec<u8>>,
}

struct Compare {
    agree: bool,
    ours: usize,
    foreign: usize,
    lag: usize,
    snr: f64,
    max_abs: f32,
    compared: usize,
}

impl std::fmt::Display for Compare {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let verdict = if self.agree { "agree" } else { "disagree" };
        let snr = if self.max_abs == 0.0 && self.snr.is_infinite() {
            "bit-exact".to_string()
        } else if self.snr.is_finite() {
            format!("{:.6} dB", self.snr)
        } else {
            "non-finite".to_string()
        };
        let max_abs = format!("{:.8e}", self.max_abs);
        write!(
            formatter,
            "{verdict}, ours {} samples, ffmpeg {} samples, lag {} samples, snr {snr}, max abs error {max_abs}, compared {}",
            self.ours, self.foreign, self.lag, self.compared
        )
    }
}

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new() -> Result<Self, String> {
        let path = std::env::temp_dir().join(format!("cutlass-aac-bench-{}", std::process::id()));
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

fn sine(count: usize) -> Vec<f32> {
    let mut samples = Vec::with_capacity(count);
    let rate = RATE as f32;
    for index in 0..count {
        let n = u32::try_from(index).unwrap_or(u32::MAX);
        let phase = 2.0 * std::f32::consts::PI * 1_000.0 * n as f32 / rate;
        samples.push(0.5 * phase.sin());
    }
    samples
}

fn write_f32(path: &Path, samples: &[f32]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(samples.len().saturating_mul(4));
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    std::fs::write(path, bytes).map_err(|err| err.to_string())
}

fn ffmpeg_encode(ffmpeg: &Path, pcm: &Path, adts: &Path) -> Result<(), String> {
    let output = Command::new(ffmpeg)
        .args([
            "-nostdin",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "f32le",
            "-ar",
            "48000",
            "-ac",
            "1",
            "-i",
        ])
        .arg(pcm)
        .args([
            "-c:a",
            "aac",
            "-profile:a",
            "aac_low",
            "-b:a",
            "96k",
            "-f",
            "adts",
        ])
        .arg(adts)
        .output()
        .map_err(|err| err.to_string())?;
    command_ok(output)
}

fn ffmpeg_decode(ffmpeg: &Path, adts: &Path, pcm: &Path) -> Result<Vec<f32>, String> {
    let output = Command::new(ffmpeg)
        .args(["-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(adts)
        .args(["-f", "f32le", "-ac", "1", "-ar", "48000"])
        .arg(pcm)
        .output()
        .map_err(|err| err.to_string())?;
    command_ok(output)?;
    let bytes = std::fs::read(pcm).map_err(|err| err.to_string())?;
    parse_f32le(&bytes)
}

fn time_ffmpeg_decode(ffmpeg: &Path, adts: &Path) -> Result<f64, String> {
    let started = Instant::now();
    let output = Command::new(ffmpeg)
        .args([
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-i",
        ])
        .arg(adts)
        .args(["-f", "null", "-"])
        .output()
        .map_err(|err| err.to_string())?;
    command_ok(output)?;
    Ok(started.elapsed().as_secs_f64())
}

fn time_ffmpeg_encode(ffmpeg: &Path, pcm: &Path) -> Result<f64, String> {
    let started = Instant::now();
    let output = Command::new(ffmpeg)
        .args([
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-f",
            "f32le",
            "-ar",
            "48000",
            "-ac",
            "1",
            "-i",
        ])
        .arg(pcm)
        .args([
            "-c:a",
            "aac",
            "-profile:a",
            "aac_low",
            "-b:a",
            "96k",
            "-f",
            "null",
            "-",
        ])
        .output()
        .map_err(|err| err.to_string())?;
    command_ok(output)?;
    Ok(started.elapsed().as_secs_f64())
}

fn command_ok(output: std::process::Output) -> Result<(), String> {
    if output.status.success() {
        Ok(())
    } else {
        Err(stderr_tail(&output.stderr))
    }
}

fn to_packets(units: &[Vec<u8>]) -> Vec<Packet> {
    units
        .iter()
        .map(|unit| Packet::audio(unit.clone(), 0, 1024, RATE))
        .collect()
}

fn decode_packets(packets: &[Packet]) -> Result<Vec<f32>, String> {
    let mut decoder = open_decoder()?;
    let mut pcm = Vec::new();
    for packet in packets {
        pcm.extend(decoder.push_packet(packet).map_err(|err| err.to_string())?);
    }
    if let Some(tail) = decoder.drain_pcm().map_err(|err| err.to_string())? {
        pcm.extend(tail);
    }
    Ok(pcm)
}

fn time_decode(packets: &[Packet]) -> Result<f64, String> {
    let mut decoder = open_decoder()?;
    let started = Instant::now();
    let mut count = 0usize;
    for packet in packets {
        count += decoder
            .push_packet(packet)
            .map_err(|err| err.to_string())?
            .len();
    }
    if let Some(tail) = decoder.drain_pcm().map_err(|err| err.to_string())? {
        count += tail.len();
    }
    let seconds = started.elapsed().as_secs_f64();
    if count == 0 {
        return Err("decoder produced no samples".to_string());
    }
    std::hint::black_box(count);
    Ok(seconds)
}

fn open_decoder() -> Result<AacDecoder, String> {
    let params = CodecParams::audio(CodecId::Aac, RATE, ChannelLayout::mono());
    AacDecoder::open(&params).map_err(|err| err.to_string())
}

fn encode_once(samples: &[f32]) -> Result<(f64, Vec<Packet>), String> {
    let mut params = CodecParams::audio(CodecId::Aac, RATE, ChannelLayout::mono());
    params.bitrate = BITRATE;
    let mut encoder = AacEncoder::open(&params).map_err(|err| err.to_string())?;
    let pcm = cutlass_decode::Pcm::from_f32(RATE, ChannelLayout::mono(), samples.to_vec())
        .map_err(|err| format!("pcm: {err:?}"))?;
    let started = Instant::now();
    let mut packets = encoder
        .push(&AudioInput { pcm })
        .map_err(|err| err.to_string())?;
    packets.extend(encoder.drain().map_err(|err| err.to_string())?);
    let seconds = started.elapsed().as_secs_f64();
    let bytes: usize = packets.iter().map(|packet| packet.data.len()).sum();
    if packets.is_empty() || bytes == 0 {
        return Err("encoder produced no bytes".to_string());
    }
    std::hint::black_box(bytes);
    Ok((seconds, packets))
}

fn encode_seconds(samples: &[f32]) -> Result<f64, String> {
    encode_once(samples).map(|(seconds, _packets)| seconds)
}

fn wrap_adts(packets: &[Packet]) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    for packet in packets {
        let header = adts_header(packet.data.len())?;
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&packet.data);
    }
    Ok(bytes)
}

fn adts_header(payload_len: usize) -> Result<[u8; 7], String> {
    let frame_len = payload_len + 7;
    if frame_len >= (1 << 13) {
        return Err("access unit does not fit in an adts header".to_string());
    }
    let frame_len = u32::try_from(frame_len).unwrap_or(0);
    let mut bytes = [0u8; 7];
    bytes[0] = 0xFF;
    bytes[1] = 0xF1;
    bytes[2] = (1 << 6) | (3 << 2);
    bytes[3] = (1 << 6) | u8::try_from((frame_len >> 11) & 0x03).unwrap_or(0);
    bytes[4] = u8::try_from((frame_len >> 3) & 0xFF).unwrap_or(0);
    bytes[5] = u8::try_from((frame_len & 0x07) << 5).unwrap_or(0) | 0x1F;
    bytes[6] = 0xFC;
    Ok(bytes)
}

fn compare(ours: &[f32], foreign: &[f32]) -> Compare {
    if ours.iter().any(|sample| !sample.is_finite())
        || foreign.iter().any(|sample| !sample.is_finite())
    {
        return Compare {
            agree: false,
            ours: ours.len(),
            foreign: foreign.len(),
            lag: 0,
            snr: f64::NAN,
            max_abs: f32::NAN,
            compared: 0,
        };
    }
    let mut best: Option<Align> = None;
    for lag in 0..=MAX_LAG {
        consider(&mut best, ours, foreign, lag, true);
        if lag != 0 {
            consider(&mut best, ours, foreign, lag, false);
        }
    }
    let Some(best) = best else {
        return Compare {
            agree: false,
            ours: ours.len(),
            foreign: foreign.len(),
            lag: 0,
            snr: f64::NAN,
            max_abs: f32::NAN,
            compared: 0,
        };
    };
    let length_ok = ours.len().abs_diff(foreign.len()) <= LENGTH_SLOP;
    let covered = best.compared + LENGTH_SLOP >= ours.len().min(foreign.len());
    Compare {
        agree: length_ok && covered && best.snr >= MIN_SNR_DB,
        ours: ours.len(),
        foreign: foreign.len(),
        lag: best.lag,
        snr: best.snr,
        max_abs: best.max_abs,
        compared: best.compared,
    }
}

struct Align {
    lag: usize,
    snr: f64,
    max_abs: f32,
    compared: usize,
}

fn consider(best: &mut Option<Align>, ours: &[f32], foreign: &[f32], lag: usize, drop_ours: bool) {
    let (left, right) = if drop_ours {
        let Some(left) = ours.get(lag..) else {
            return;
        };
        (left, foreign)
    } else {
        let Some(right) = foreign.get(lag..) else {
            return;
        };
        (ours, right)
    };
    let count = left.len().min(right.len());
    let shorter = ours.len().min(foreign.len());
    if count < 1024 || count + LENGTH_SLOP < shorter {
        return;
    }
    let (snr, max_abs) = error_stats(&left[..count], &right[..count]);
    let replace = best.as_ref().is_none_or(|current| snr > current.snr);
    if replace {
        *best = Some(Align {
            lag,
            snr,
            max_abs,
            compared: count,
        });
    }
}

fn error_stats(left: &[f32], right: &[f32]) -> (f64, f32) {
    let mut signal = 0.0f64;
    let mut noise = 0.0f64;
    let mut max_abs = 0.0f32;
    for (left_sample, right_sample) in left.iter().zip(right) {
        max_abs = max_abs.max((left_sample - right_sample).abs());
        let left_sample = f64::from(*left_sample);
        let right_sample = f64::from(*right_sample);
        signal += left_sample * left_sample;
        let error = left_sample - right_sample;
        noise += error * error;
    }
    let snr = if noise == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (signal / noise).log10()
    };
    (snr, max_abs)
}

fn median_seconds(mut once: impl FnMut() -> Result<f64, String>) -> Result<f64, String> {
    let _warmup = once()?;
    let mut samples = [once()?, once()?, once()?];
    samples.sort_by(f64::total_cmp);
    let median = samples[1];
    if median.is_finite() && median >= 0.0 {
        Ok(median)
    } else {
        Err(format!("non-finite timing {median}"))
    }
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
        if third >> 6 != 1 {
            return Err(format!("adts profile {}", third >> 6));
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
    if units.is_empty() {
        return Err("no adts frames".to_string());
    }
    Ok(Adts {
        sample_rate: sample_rate.ok_or_else(|| "no adts frames".to_string())?,
        channels: channels.ok_or_else(|| "no adts frames".to_string())?,
        units,
    })
}

fn parse_f32le(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let (chunks, rest) = bytes.as_chunks::<4>();
    if !rest.is_empty() {
        return Err(format!("pcm length {} is not a multiple of 4", bytes.len()));
    }
    Ok(chunks.iter().copied().map(f32::from_le_bytes).collect())
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
