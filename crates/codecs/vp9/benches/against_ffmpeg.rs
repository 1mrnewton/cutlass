//! Time profile-0 VP9 against ffmpeg on one thread.
//!
//! `cargo bench -p cutlass-vp9`
//!
//! ffmpeg writes one 320×240 libvpx-vp9 stream. Pixels are compared to
//! ffmpeg's vp9 decoder before either decode is timed. The encoder then
//! times a few frames of the same test pattern, and ffmpeg's vp9 decoder
//! has to read those bytes back.

#![allow(clippy::print_stdout)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use cutlass_decode::video::VideoDecoder;
use cutlass_decode::{CodecId, CodecParams, Packet, Plane, VideoFrame};
use cutlass_encode::video::{VideoEncoder, VideoInput};
use cutlass_vp9::{Vp9Decoder, Vp9Encoder};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const DECODE_FRAMES: usize = 30;
const ENCODE_FRAMES: usize = 8;

struct WorkDir {
    path: PathBuf,
}

impl WorkDir {
    fn new() -> Result<Self, String> {
        let path = std::env::temp_dir().join(format!("cutlass-vp9-bench-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).map_err(|err| err.to_string())?;
        Ok(Self { path })
    }
}

impl Drop for WorkDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

struct Ivf {
    width: u32,
    height: u32,
    frames: Vec<Vec<u8>>,
}

fn main() -> Result<(), String> {
    let Ok(ffmpeg) = ffmpeg_bin() else {
        println!("skip: ffmpeg is not installed");
        return Ok(());
    };
    let dir = WorkDir::new()?;
    let ivf_path = dir.path.join("source.ivf");
    let yuv_path = dir.path.join("source.yuv");
    let raw_path = dir.path.join("source-raw.yuv");

    generate_stream(&ffmpeg, &ivf_path)?;
    let stream = read_ivf(&std::fs::read(&ivf_path).map_err(|err| err.to_string())?)?;
    require_profile_0(&stream)?;
    if stream.width != WIDTH || stream.height != HEIGHT || stream.frames.len() != DECODE_FRAMES {
        return Err(format!(
            "stream is {}x{} frames {}",
            stream.width,
            stream.height,
            stream.frames.len()
        ));
    }
    ffmpeg_raw(&ffmpeg, "vp9", &ivf_path, &yuv_path)?;
    let oracle = std::fs::read(&yuv_path).map_err(|err| err.to_string())?;
    let decoded = decode_frames(&stream.frames, WIDTH, HEIGHT)?;
    let ours = pack(&decoded)?;
    let mismatches = compare_pixels(&ours, &oracle)?;

    let packets = packets_of(&stream.frames);
    let decode_ours = median_seconds(|| time_decode(&packets))?;
    let decode_ffmpeg = median_seconds(|| time_ffmpeg_decode(&ffmpeg, &ivf_path))?;

    generate_yuv(&ffmpeg, &raw_path)?;
    let inputs = inputs_from_yuv(&std::fs::read(&raw_path).map_err(|err| err.to_string())?)?;
    let encoded = encode_frames(&inputs)?;
    let encoded_frames = encoded.len();
    let encoded_mismatches = check_encoded(&ffmpeg, &dir.path, &encoded)?;
    let _ = time_encode(&inputs)?;
    let encode_ours = time_encode(&inputs)?;

    println!("320x240 profile 0 4:2:0, one thread");
    println!("cutlass times the codec loop. ffmpeg times include process startup.");
    println!("decode libvpx-vp9 ivf, {DECODE_FRAMES} frames, median of 3 after 1 warmup");
    println!(
        "pixels: {} bytes, {mismatches} mismatches against ffmpeg vp9",
        ours.len()
    );
    print_row("cutlass", DECODE_FRAMES, decode_ours);
    print_row("ffmpeg", DECODE_FRAMES, decode_ffmpeg);
    println!("encode testsrc2, {ENCODE_FRAMES} frames, quantizer 32, one run after 1 warmup");
    print_row("cutlass", ENCODE_FRAMES, encode_ours);
    println!("ffmpeg vp9 decoded {encoded_frames} frames, {encoded_mismatches} mismatches");
    Ok(())
}

fn print_row(name: &str, frames: usize, seconds: f64) {
    let fps = fps(frames, seconds);
    println!("{name:<8} {seconds:>8.3} s   {fps:>8.1} fps");
}

fn fps(frames: usize, seconds: f64) -> f64 {
    let frames = u32::try_from(frames).unwrap_or(u32::MAX);
    if seconds <= 0.0 {
        return 0.0;
    }
    f64::from(frames) / seconds
}

fn median_seconds(mut once: impl FnMut() -> Result<f64, String>) -> Result<f64, String> {
    let _ = once()?;
    let mut samples = [once()?, once()?, once()?];
    samples.sort_by(f64::total_cmp);
    Ok(samples[1])
}

fn time_decode(packets: &[Packet]) -> Result<f64, String> {
    let params = CodecParams::video(CodecId::Vp9, WIDTH, HEIGHT);
    let mut decoder = Vp9Decoder::open(&params).map_err(|err| err.to_string())?;
    let started = Instant::now();
    let mut pictures = 0usize;
    for packet in packets {
        let shown = decoder.push(packet).map_err(|err| err.to_string())?;
        for output in shown {
            pictures += 1;
            std::hint::black_box(&output.frame);
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    if pictures != DECODE_FRAMES {
        return Err(format!("decoder returned {pictures} pictures"));
    }
    Ok(seconds)
}

fn time_encode(inputs: &[VideoInput]) -> Result<f64, String> {
    let params = CodecParams::video(CodecId::Vp9, WIDTH, HEIGHT);
    let mut encoder = Vp9Encoder::open(&params).map_err(|err| err.to_string())?;
    let started = Instant::now();
    let mut packets = 0usize;
    let mut bytes = 0usize;
    for input in inputs {
        let produced = encoder.push(input).map_err(|err| err.to_string())?;
        for packet in &produced {
            packets += 1;
            bytes += packet.data.len();
            std::hint::black_box(&packet.data);
        }
    }
    let tail = encoder.drain().map_err(|err| err.to_string())?;
    for packet in &tail {
        packets += 1;
        bytes += packet.data.len();
        std::hint::black_box(&packet.data);
    }
    let seconds = started.elapsed().as_secs_f64();
    if packets != ENCODE_FRAMES || bytes == 0 {
        return Err(format!("encoder returned {packets} packets"));
    }
    Ok(seconds)
}

fn time_ffmpeg_decode(ffmpeg: &str, path: &Path) -> Result<f64, String> {
    let started = Instant::now();
    let output = Command::new(ffmpeg)
        .args([
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-c:v",
            "vp9",
            "-i",
        ])
        .arg(path)
        .args(["-f", "null", "-"])
        .output()
        .map_err(|err| err.to_string())?;
    if !output.status.success() {
        return Err(tail(&output.stderr));
    }
    Ok(started.elapsed().as_secs_f64())
}

fn generate_stream(ffmpeg: &str, path: &Path) -> Result<(), String> {
    let frames = DECODE_FRAMES.to_string();
    let spec = format!("testsrc2=size={WIDTH}x{HEIGHT}:rate=30");
    let output = Command::new(ffmpeg)
        .args([
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
        ])
        .arg(&spec)
        .args(["-frames:v"])
        .arg(&frames)
        .args([
            "-an",
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "0",
            "-g",
            &frames,
            "-keyint_min",
            &frames,
            "-lag-in-frames",
            "0",
            "-auto-alt-ref",
            "0",
            "-aq-mode",
            "0",
            "-tile-columns",
            "0",
            "-frame-parallel",
            "0",
            "-row-mt",
            "0",
            "-deadline",
            "realtime",
            "-cpu-used",
            "8",
            "-crf",
            "32",
            "-b:v",
            "0",
            "-threads",
            "1",
            "-f",
            "ivf",
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
    let spec = format!("testsrc2=size={WIDTH}x{HEIGHT}:rate=30");
    let output = Command::new(ffmpeg)
        .args([
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
        ])
        .arg(&spec)
        .args(["-frames:v"])
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

fn ffmpeg_raw(ffmpeg: &str, decoder: &str, ivf: &Path, yuv: &Path) -> Result<(), String> {
    let output = Command::new(ffmpeg)
        .args([
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-c:v",
            decoder,
            "-i",
        ])
        .arg(ivf)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(yuv)
        .output()
        .map_err(|err| err.to_string())?;
    if output.status.success() {
        Ok(())
    } else {
        Err(tail(&output.stderr))
    }
}

fn check_encoded(ffmpeg: &str, dir: &Path, packets: &[Packet]) -> Result<usize, String> {
    let frames: Vec<Vec<u8>> = packets.iter().map(|packet| packet.data.clone()).collect();
    if frames.len() != ENCODE_FRAMES {
        return Err(format!("encoder returned {} frames", frames.len()));
    }
    let ivf = write_ivf(WIDTH, HEIGHT, &frames)?;
    let ivf_path = dir.join("encoded.ivf");
    let yuv_path = dir.join("encoded.yuv");
    std::fs::write(&ivf_path, ivf).map_err(|err| err.to_string())?;
    ffmpeg_raw(ffmpeg, "vp9", &ivf_path, &yuv_path)?;
    let oracle = std::fs::read(&yuv_path).map_err(|err| err.to_string())?;
    let decoded = decode_frames(&frames, WIDTH, HEIGHT)?;
    let ours = pack(&decoded)?;
    compare_pixels(&ours, &oracle)
}

fn encode_frames(inputs: &[VideoInput]) -> Result<Vec<Packet>, String> {
    let params = CodecParams::video(CodecId::Vp9, WIDTH, HEIGHT);
    let mut encoder = Vp9Encoder::open(&params).map_err(|err| err.to_string())?;
    let mut packets = Vec::new();
    for input in inputs {
        packets.extend(encoder.push(input).map_err(|err| err.to_string())?);
    }
    packets.extend(encoder.drain().map_err(|err| err.to_string())?);
    Ok(packets)
}

fn decode_frames(frames: &[Vec<u8>], width: u32, height: u32) -> Result<Vec<VideoFrame>, String> {
    let params = CodecParams::video(CodecId::Vp9, width, height);
    let mut decoder = Vp9Decoder::open(&params).map_err(|err| err.to_string())?;
    let mut pictures = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        let pts = i64::try_from(index).unwrap_or(i64::MAX);
        let packet = Packet::video(frame.clone(), pts, 1, 30, index == 0);
        let shown = decoder
            .push(&packet)
            .map_err(|err| format!("frame {index}: {err}"))?;
        for output in shown {
            if output.frame.width != width || output.frame.height != height {
                return Err(format!(
                    "frame {index} is {}x{}",
                    output.frame.width, output.frame.height
                ));
            }
            pictures.push(output.frame);
        }
    }
    if pictures.len() != frames.len() {
        return Err(format!(
            "decoder returned {} pictures from {} frames",
            pictures.len(),
            frames.len()
        ));
    }
    Ok(pictures)
}

fn pack(frames: &[VideoFrame]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    for frame in frames {
        out.extend(tight(&frame.y)?);
        out.extend(tight(&frame.u)?);
        out.extend(tight(&frame.v)?);
    }
    Ok(out)
}

fn tight(plane: &Plane) -> Result<Vec<u8>, String> {
    let width = usize::try_from(plane.width).map_err(|_| "plane width".to_string())?;
    let height = usize::try_from(plane.height).map_err(|_| "plane height".to_string())?;
    if plane.stride == width {
        return Ok(plane.samples.clone());
    }
    let mut out = Vec::new();
    for row in 0..height {
        let start = row
            .checked_mul(plane.stride)
            .ok_or_else(|| "plane stride".to_string())?;
        let end = start
            .checked_add(width)
            .ok_or_else(|| "plane stride".to_string())?;
        let samples = plane
            .samples
            .get(start..end)
            .ok_or_else(|| "plane stride".to_string())?;
        out.extend_from_slice(samples);
    }
    Ok(out)
}

fn compare_pixels(ours: &[u8], oracle: &[u8]) -> Result<usize, String> {
    if ours.len() != oracle.len() {
        return Err(format!("yuv length {} vs {}", ours.len(), oracle.len()));
    }
    let mismatches = ours
        .iter()
        .zip(oracle.iter())
        .filter(|(left, right)| left != right)
        .count();
    if mismatches != 0 {
        let index = ours
            .iter()
            .zip(oracle.iter())
            .position(|(left, right)| left != right)
            .unwrap_or(0);
        let got = ours.get(index).copied().unwrap_or(0);
        let want = oracle.get(index).copied().unwrap_or(0);
        return Err(format!(
            "{mismatches} mismatches, first at {index}: {got} vs {want}"
        ));
    }
    Ok(mismatches)
}

fn packets_of(frames: &[Vec<u8>]) -> Vec<Packet> {
    frames
        .iter()
        .enumerate()
        .map(|(index, frame)| {
            let pts = i64::try_from(index).unwrap_or(i64::MAX);
            Packet::video(frame.clone(), pts, 1, 30, index == 0)
        })
        .collect()
}

fn inputs_from_yuv(raw: &[u8]) -> Result<Vec<VideoInput>, String> {
    let frame_len = frame_bytes()?;
    let need = frame_len
        .checked_mul(ENCODE_FRAMES)
        .ok_or_else(|| "yuv length".to_string())?;
    if raw.len() < need {
        return Err(format!("yuv is {} bytes", raw.len()));
    }
    let y_len = plane_len(WIDTH, HEIGHT)?;
    let c_len = plane_len(WIDTH / 2, HEIGHT / 2)?;
    let mut inputs = Vec::with_capacity(ENCODE_FRAMES);
    for index in 0..ENCODE_FRAMES {
        let start = index * frame_len;
        let plane = raw
            .get(start..start + frame_len)
            .ok_or_else(|| "yuv frame".to_string())?;
        let y = plane
            .get(..y_len)
            .ok_or_else(|| "luma".to_string())?
            .to_vec();
        let u = plane
            .get(y_len..y_len + c_len)
            .ok_or_else(|| "chroma".to_string())?
            .to_vec();
        let v = plane
            .get(y_len + c_len..y_len + c_len + c_len)
            .ok_or_else(|| "chroma".to_string())?
            .to_vec();
        let frame = VideoFrame::yuv420p(
            WIDTH,
            HEIGHT,
            Plane::tight(WIDTH, HEIGHT, y).map_err(|err| err.to_string())?,
            Plane::tight(WIDTH / 2, HEIGHT / 2, u).map_err(|err| err.to_string())?,
            Plane::tight(WIDTH / 2, HEIGHT / 2, v).map_err(|err| err.to_string())?,
        )
        .map_err(|err| err.to_string())?;
        let pts = i64::try_from(index).unwrap_or(i64::MAX);
        inputs.push(VideoInput {
            frame,
            pts,
            duration: 1,
            timescale: 30,
        });
    }
    Ok(inputs)
}

fn frame_bytes() -> Result<usize, String> {
    let y = plane_len(WIDTH, HEIGHT)?;
    let c = plane_len(WIDTH / 2, HEIGHT / 2)?;
    y.checked_add(c)
        .and_then(|sum| sum.checked_add(c))
        .ok_or_else(|| "frame size".to_string())
}

fn plane_len(width: u32, height: u32) -> Result<usize, String> {
    let width = usize::try_from(width).map_err(|_| "width".to_string())?;
    let height = usize::try_from(height).map_err(|_| "height".to_string())?;
    width
        .checked_mul(height)
        .ok_or_else(|| "plane size".to_string())
}

fn require_profile_0(stream: &Ivf) -> Result<(), String> {
    for (index, frame) in stream.frames.iter().enumerate() {
        let first = frame
            .first()
            .copied()
            .ok_or_else(|| format!("frame {index} is empty"))?;
        if first >> 6 != 2 {
            return Err(format!("frame {index} marker is not vp9"));
        }
        let profile = (first >> 4) & 0x3;
        if profile != 0 {
            return Err(format!(
                "frame {index} is profile {profile}, which this decoder refuses"
            ));
        }
    }
    Ok(())
}

fn read_ivf(data: &[u8]) -> Result<Ivf, String> {
    if data.len() < 32 || !data.starts_with(b"DKIF") {
        return Err("ivf header".to_string());
    }
    let header_len = usize::from(u16::from_le_bytes(read_array(data, 6)?));
    if header_len < 32 || data.len() < header_len {
        return Err("ivf header length".to_string());
    }
    if data.get(8..12) != Some(b"VP90".as_slice()) {
        return Err("ivf is not VP90".to_string());
    }
    let width = u32::from(u16::from_le_bytes(read_array(data, 12)?));
    let height = u32::from(u16::from_le_bytes(read_array(data, 14)?));
    let mut offset = header_len;
    let mut frames = Vec::new();
    while offset < data.len() {
        if data.len() - offset < 12 {
            return Err("truncated ivf frame header".to_string());
        }
        let size = usize::try_from(u32::from_le_bytes(read_array(data, offset)?))
            .map_err(|_| "ivf frame size".to_string())?;
        offset += 12;
        let end = offset
            .checked_add(size)
            .ok_or_else(|| "ivf frame size".to_string())?;
        let frame = data
            .get(offset..end)
            .ok_or_else(|| "truncated ivf frame".to_string())?;
        frames.push(frame.to_vec());
        offset = end;
    }
    if frames.is_empty() {
        return Err("ivf has no frames".to_string());
    }
    Ok(Ivf {
        width,
        height,
        frames,
    })
}

fn write_ivf(width: u32, height: u32, frames: &[Vec<u8>]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    out.extend_from_slice(b"DKIF");
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(b"VP90");
    let width = u16::try_from(width).map_err(|_| "ivf width".to_string())?;
    let height = u16::try_from(height).map_err(|_| "ivf height".to_string())?;
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    out.extend_from_slice(&30u32.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());
    let count = u32::try_from(frames.len()).map_err(|_| "ivf count".to_string())?;
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    for (index, frame) in frames.iter().enumerate() {
        let size = u32::try_from(frame.len()).map_err(|_| "ivf frame size".to_string())?;
        out.extend_from_slice(&size.to_le_bytes());
        let pts = u64::try_from(index).map_err(|_| "ivf pts".to_string())?;
        out.extend_from_slice(&pts.to_le_bytes());
        out.extend_from_slice(frame);
    }
    Ok(out)
}

fn read_array<const N: usize>(data: &[u8], at: usize) -> Result<[u8; N], String> {
    let end = at.checked_add(N).ok_or_else(|| "ivf offset".to_string())?;
    let bytes = data
        .get(at..end)
        .ok_or_else(|| "truncated ivf".to_string())?;
    let mut buf = [0u8; N];
    buf.copy_from_slice(bytes);
    Ok(buf)
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
    let start = stderr.len().saturating_sub(500);
    String::from_utf8_lossy(&stderr[start..]).trim().to_string()
}
