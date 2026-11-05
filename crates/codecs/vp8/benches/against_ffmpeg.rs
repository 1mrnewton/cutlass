//! Time VP8 against ffmpeg on one thread.
//!
//! `cargo bench -p cutlass-vp8`
//!
//! Decode: ffmpeg's libvpx writes a 1920×1080 testsrc2 stream. Our
//! pictures are compared to ffmpeg's `vp8` decoder before anything is
//! timed. Encode: 640×360 testsrc2 at 1 Mb/s, ours against libvpx good
//! cpu-used 4, both one thread, and libvpx has to decode our bytes to the
//! same pictures our decoder makes. ffmpeg's times are its own `-benchmark`
//! real time, without process startup; they include its raw-video reading
//! or writing, which is small next to the codec.

#![allow(clippy::print_stdout)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use cutlass_decode::video::VideoDecoder;
use cutlass_decode::{CodecId, CodecParams, Packet, Plane, VideoFrame};
use cutlass_encode::video::{VideoEncoder, VideoInput};
use cutlass_vp8::{Vp8Decoder, Vp8Encoder, Vp8Options};

const DECODE_W: u32 = 1920;
const DECODE_H: u32 = 1080;
const DECODE_FRAMES: usize = 120;
const ENCODE_W: u32 = 640;
const ENCODE_H: u32 = 360;
const ENCODE_FRAMES: usize = 150;
const BITRATE: u32 = 1_000_000;

struct WorkDir {
    path: PathBuf,
}

impl WorkDir {
    fn new() -> Result<Self, String> {
        let path = std::env::temp_dir().join(format!("cutlass-vp8-bench-{}", std::process::id()));
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

fn main() -> Result<(), String> {
    let Some(ffmpeg) = ffmpeg_bin() else {
        println!("skip: ffmpeg is not installed");
        return Ok(());
    };
    let dir = WorkDir::new()?;
    println!("VP8, one thread. ffmpeg rows are its -benchmark real time.");

    // Decode.
    let ivf = dir.path.join("decode.ivf");
    run(
        &ffmpeg,
        &[
            "-f",
            "lavfi",
            "-i",
            &format!("testsrc2=size={DECODE_W}x{DECODE_H}:rate=30"),
            "-frames:v",
            &DECODE_FRAMES.to_string(),
            "-c:v",
            "libvpx",
            "-b:v",
            "6M",
            "-deadline",
            "good",
            "-cpu-used",
            "4",
            "-threads",
            "1",
            "-auto-alt-ref",
            "0",
            "-f",
            "ivf",
            path_str(&ivf)?,
        ],
    )?;
    let frames = read_ivf(&std::fs::read(&ivf).map_err(|e| e.to_string())?)?;
    let packets: Vec<Packet> = frames
        .iter()
        .enumerate()
        .map(|(i, f)| Packet::video(f.clone(), i as i64, 1, 30, i == 0))
        .collect();
    let ours = pack(&decode(&packets, DECODE_W, DECODE_H)?);
    let theirs = ffmpeg_raw(&ffmpeg, "vp8", &ivf, &dir.path.join("decode.yuv"))?;
    let differ = mismatches(&ours, &theirs);
    let bytes: usize = frames.iter().map(Vec::len).sum();
    println!(
        "decode {DECODE_W}x{DECODE_H} testsrc2, {} frames, {} kB; pixels {} bytes, {differ} mismatches against ffmpeg vp8",
        frames.len(),
        bytes / 1000,
        ours.len()
    );
    let ours_time = median(|| {
        let started = Instant::now();
        let pictures = decode(&packets, DECODE_W, DECODE_H)?;
        std::hint::black_box(&pictures);
        Ok(started.elapsed().as_secs_f64())
    })?;
    row("cutlass", frames.len(), ours_time);
    for decoder in ["vp8", "libvpx"] {
        let t = median(|| {
            ffmpeg_bench(
                &ffmpeg,
                &[
                    "-threads",
                    "1",
                    "-c:v",
                    decoder,
                    "-i",
                    path_str(&ivf)?,
                    "-f",
                    "null",
                    "-",
                ],
            )
        })?;
        row(&format!("ffmpeg {decoder}"), frames.len(), t);
    }

    // A real 1080p WebM, when ../local-files has it.
    let webm = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../local-files/file_example_WEBM_1920_3_7MB.webm");
    if webm.is_file() {
        let real = dir.path.join("real.ivf");
        run(
            &ffmpeg,
            &[
                "-i",
                path_str(&webm)?,
                "-an",
                "-c:v",
                "copy",
                "-f",
                "ivf",
                path_str(&real)?,
            ],
        )?;
        let frames = read_ivf(&std::fs::read(&real).map_err(|e| e.to_string())?)?;
        let packets: Vec<Packet> = frames
            .iter()
            .enumerate()
            .map(|(i, f)| Packet::video(f.clone(), i as i64, 1, 30, i == 0))
            .collect();
        let ours = pack(&decode(&packets, DECODE_W, DECODE_H)?);
        let theirs = ffmpeg_raw(&ffmpeg, "vp8", &real, &dir.path.join("real.yuv"))?;
        let bytes: usize = frames.iter().map(Vec::len).sum();
        println!(
            "decode local 1080p WebM, {} frames, {} kB; {} mismatches against ffmpeg vp8",
            frames.len(),
            bytes / 1000,
            mismatches(&ours, &theirs)
        );
        drop(ours);
        let ours_time = median(|| {
            let started = Instant::now();
            let pictures = decode(&packets, DECODE_W, DECODE_H)?;
            std::hint::black_box(&pictures);
            Ok(started.elapsed().as_secs_f64())
        })?;
        row("cutlass", frames.len(), ours_time);
        for decoder in ["vp8", "libvpx"] {
            let t = median(|| {
                ffmpeg_bench(
                    &ffmpeg,
                    &[
                        "-threads",
                        "1",
                        "-c:v",
                        decoder,
                        "-i",
                        path_str(&real)?,
                        "-f",
                        "null",
                        "-",
                    ],
                )
            })?;
            row(&format!("ffmpeg {decoder}"), frames.len(), t);
        }
    }

    // Encode.
    let raw = dir.path.join("encode.yuv");
    run(
        &ffmpeg,
        &[
            "-f",
            "lavfi",
            "-i",
            &format!("testsrc2=size={ENCODE_W}x{ENCODE_H}:rate=30"),
            "-frames:v",
            &ENCODE_FRAMES.to_string(),
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rawvideo",
            path_str(&raw)?,
        ],
    )?;
    let source = std::fs::read(&raw).map_err(|e| e.to_string())?;
    let inputs = inputs_of(&source)?;
    let opts = Vp8Options {
        bitrate: BITRATE,
        ..Vp8Options::default()
    };
    let (encoded, _) = encode(&inputs, opts)?;
    let encoded_ivf = dir.path.join("ours.ivf");
    write_ivf(&encoded_ivf, ENCODE_W, ENCODE_H, &encoded)?;
    let encoded_packets: Vec<Packet> = encoded
        .iter()
        .enumerate()
        .map(|(i, f)| Packet::video(f.clone(), i as i64, 1, 30, i == 0))
        .collect();
    let ours_decoded = pack(&decode(&encoded_packets, ENCODE_W, ENCODE_H)?);
    let libvpx_decoded = ffmpeg_raw(&ffmpeg, "libvpx", &encoded_ivf, &dir.path.join("ours.yuv"))?;
    let ours_bytes: usize = encoded.iter().map(Vec::len).sum();
    let ours_psnr = psnr_y(&source, &ours_decoded);
    let libvpx_ivf = dir.path.join("libvpx.ivf");
    let size = format!("{ENCODE_W}x{ENCODE_H}");
    let rate = BITRATE.to_string();
    let libvpx_args = [
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-s",
        &size,
        "-r",
        "30",
        "-i",
        path_str(&raw)?,
        "-c:v",
        "libvpx",
        "-b:v",
        &rate,
        "-deadline",
        "good",
        "-cpu-used",
        "4",
        "-threads",
        "1",
        "-auto-alt-ref",
        "0",
        "-lag-in-frames",
        "0",
        "-f",
        "ivf",
        path_str(&libvpx_ivf)?,
    ];
    let libvpx_time = median(|| ffmpeg_bench(&ffmpeg, &libvpx_args))?;
    let libvpx_frames = read_ivf(&std::fs::read(&libvpx_ivf).map_err(|e| e.to_string())?)?;
    let libvpx_bytes: usize = libvpx_frames.iter().map(Vec::len).sum();
    let libvpx_yuv = ffmpeg_raw(&ffmpeg, "libvpx", &libvpx_ivf, &dir.path.join("libvpx.yuv"))?;
    let libvpx_psnr = psnr_y(&source, &libvpx_yuv);
    let ours_time = median(|| encode(&inputs, opts).map(|(_, t)| t))?;
    println!(
        "encode {ENCODE_W}x{ENCODE_H} testsrc2, {ENCODE_FRAMES} frames at {} kb/s; libvpx decodes ours with {} mismatches",
        BITRATE / 1000,
        mismatches(&ours_decoded, &libvpx_decoded)
    );
    println!(
        "cutlass  {:>8.3} s   {:>8.1} fps   {:>7} B  PSNR-Y {ours_psnr:.2} dB",
        ours_time,
        fps(ENCODE_FRAMES, ours_time),
        ours_bytes
    );
    println!(
        "libvpx   {:>8.3} s   {:>8.1} fps   {:>7} B  PSNR-Y {libvpx_psnr:.2} dB",
        libvpx_time,
        fps(ENCODE_FRAMES, libvpx_time),
        libvpx_bytes
    );
    Ok(())
}

fn row(name: &str, frames: usize, seconds: f64) {
    println!(
        "{name:<14} {seconds:>8.3} s   {:>8.1} fps",
        fps(frames, seconds)
    );
}

fn fps(frames: usize, seconds: f64) -> f64 {
    if seconds <= 0.0 {
        return 0.0;
    }
    frames as f64 / seconds
}

fn median(mut once: impl FnMut() -> Result<f64, String>) -> Result<f64, String> {
    let _ = once()?;
    let mut samples = [once()?, once()?, once()?];
    samples.sort_by(f64::total_cmp);
    Ok(samples[1])
}

fn decode(packets: &[Packet], width: u32, height: u32) -> Result<Vec<VideoFrame>, String> {
    let params = CodecParams::video(CodecId::Vp8, width, height);
    let mut decoder = Vp8Decoder::open(&params).map_err(|e| e.to_string())?;
    let mut out = Vec::with_capacity(packets.len());
    for packet in packets {
        for output in decoder.push(packet).map_err(|e| e.to_string())? {
            out.push(output.frame);
        }
    }
    Ok(out)
}

fn encode(inputs: &[VideoInput], opts: Vp8Options) -> Result<(Vec<Vec<u8>>, f64), String> {
    let mut encoder =
        Vp8Encoder::with_options(ENCODE_W, ENCODE_H, opts).map_err(|e| e.to_string())?;
    let started = Instant::now();
    let mut out = Vec::with_capacity(inputs.len());
    for input in inputs {
        for packet in encoder.push(input).map_err(|e| e.to_string())? {
            out.push(packet.data);
        }
    }
    Ok((out, started.elapsed().as_secs_f64()))
}

fn inputs_of(yuv: &[u8]) -> Result<Vec<VideoInput>, String> {
    let (w, h) = (ENCODE_W as usize, ENCODE_H as usize);
    let (cw, ch) = (w / 2, h / 2);
    let size = w * h + 2 * cw * ch;
    yuv.chunks_exact(size)
        .enumerate()
        .map(|(i, f)| {
            let frame = VideoFrame::yuv420p(
                ENCODE_W,
                ENCODE_H,
                Plane::tight(ENCODE_W, ENCODE_H, f[..w * h].to_vec()).map_err(|e| e.to_string())?,
                Plane::tight(
                    ENCODE_W / 2,
                    ENCODE_H / 2,
                    f[w * h..w * h + cw * ch].to_vec(),
                )
                .map_err(|e| e.to_string())?,
                Plane::tight(ENCODE_W / 2, ENCODE_H / 2, f[w * h + cw * ch..].to_vec())
                    .map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            Ok(VideoInput {
                frame,
                pts: i as i64,
                duration: 1,
                timescale: 30,
            })
        })
        .collect()
}

fn pack(frames: &[VideoFrame]) -> Vec<u8> {
    let mut out = Vec::new();
    for f in frames {
        out.extend_from_slice(&f.y.samples);
        out.extend_from_slice(&f.u.samples);
        out.extend_from_slice(&f.v.samples);
    }
    out
}

fn mismatches(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

fn psnr_y(source: &[u8], decoded: &[u8]) -> f64 {
    let (w, h) = (ENCODE_W as usize, ENCODE_H as usize);
    let size = w * h * 3 / 2;
    let mut sse = 0f64;
    let mut frames = 0usize;
    for (a, b) in source.chunks_exact(size).zip(decoded.chunks_exact(size)) {
        for (x, y) in a[..w * h].iter().zip(&b[..w * h]) {
            let d = f64::from(*x) - f64::from(*y);
            sse += d * d;
        }
        frames += 1;
    }
    let mse = sse / (frames.max(1) * w * h) as f64;
    10.0 * (255.0 * 255.0 / mse.max(1e-10)).log10()
}

fn ffmpeg_bin() -> Option<PathBuf> {
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
            .is_ok_and(|o| o.status.success())
    })
}

fn path_str(path: &Path) -> Result<&str, String> {
    path.to_str().ok_or_else(|| "path is not UTF-8".to_owned())
}

fn run(ffmpeg: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new(ffmpeg)
        .args(["-y", "-hide_banner", "-nostats", "-loglevel", "error"])
        .args(args)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Runs ffmpeg with `-benchmark` and returns its real time.
fn ffmpeg_bench(ffmpeg: &Path, args: &[&str]) -> Result<f64, String> {
    let output = Command::new(ffmpeg)
        .args(["-y", "-hide_banner", "-nostats", "-benchmark"])
        .args(args)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    let text = String::from_utf8_lossy(&output.stderr).into_owned()
        + &String::from_utf8_lossy(&output.stdout);
    text.split_whitespace()
        .filter_map(|w| w.strip_prefix("rtime="))
        .filter_map(|v| v.trim_end_matches('s').parse::<f64>().ok())
        .next_back()
        .ok_or_else(|| format!("no rtime in {text}"))
}

fn ffmpeg_raw(ffmpeg: &Path, decoder: &str, ivf: &Path, out: &Path) -> Result<Vec<u8>, String> {
    run(
        ffmpeg,
        &[
            "-threads",
            "1",
            "-c:v",
            decoder,
            "-i",
            path_str(ivf)?,
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            path_str(out)?,
        ],
    )?;
    std::fs::read(out).map_err(|e| e.to_string())
}

fn read_ivf(data: &[u8]) -> Result<Vec<Vec<u8>>, String> {
    if data.len() < 32 || &data[..4] != b"DKIF" {
        return Err("not ivf".to_owned());
    }
    let mut at = usize::from(u16::from_le_bytes([data[6], data[7]]));
    let mut frames = Vec::new();
    while at + 12 <= data.len() {
        let size =
            u32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]]) as usize;
        at += 12;
        let frame = data.get(at..at + size).ok_or("ivf frame")?;
        frames.push(frame.to_vec());
        at += size;
    }
    Ok(frames)
}

fn write_ivf(path: &Path, width: u32, height: u32, frames: &[Vec<u8>]) -> Result<(), String> {
    let mut out = Vec::new();
    out.extend_from_slice(b"DKIF");
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(b"VP80");
    out.extend_from_slice(&(width as u16).to_le_bytes());
    out.extend_from_slice(&(height as u16).to_le_bytes());
    out.extend_from_slice(&30u32.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&(frames.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    for (i, frame) in frames.iter().enumerate() {
        out.extend_from_slice(&(frame.len() as u32).to_le_bytes());
        out.extend_from_slice(&(i as u64).to_le_bytes());
        out.extend_from_slice(frame);
    }
    std::fs::write(path, out).map_err(|e| e.to_string())
}
