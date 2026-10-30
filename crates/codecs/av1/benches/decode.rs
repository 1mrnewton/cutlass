//! Decode one real AV1 keyframe, then the same bytes through FFmpeg.
//!
//! FFmpeg decodes AV1 by calling dav1d. This bench does not link either
//! program. It compares pixels before it times anything. Cutlass time is one
//! `Av1Decoder::push`. FFmpeg time is one process and includes startup.
//!
//! The frame comes from `../local-files` when that folder is present. Without
//! ffmpeg the bench prints `skip: ffmpeg is not installed` and exits 0.

#![allow(clippy::print_stdout)]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use cutlass_av1::{AsView, Av1Decoder};
use cutlass_decode::{CodecId, CodecParams, Decoder, Packet};

fn main() -> Result<(), String> {
    let Some(ffmpeg) = ffmpeg_bin() else {
        println!("skip: ffmpeg is not installed");
        return Ok(());
    };
    match run(&ffmpeg) {
        Ok(()) => Ok(()),
        Err(what) if what == "no AV1 file in ../local-files" => {
            println!("bench not run: {what}");
            Ok(())
        }
        Err(what) => Err(what),
    }
}

fn run(ffmpeg: &Path) -> Result<(), String> {
    let input = production_file().ok_or_else(|| "no AV1 file in ../local-files".to_string())?;
    let dir = std::env::temp_dir().join(format!("cutlass-av1-bench-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|_| "temp directory".to_string())?;
    let ivf = dir.join("frame.ivf");
    let yuv = dir.join("frame.yuv");
    let outcome = compare(ffmpeg, &input, &ivf, &yuv);
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

fn compare(ffmpeg: &Path, input: &Path, ivf: &Path, yuv: &Path) -> Result<(), String> {
    extract_frame(ffmpeg, input, ivf)?;
    require_keyframe(ffmpeg, ivf)?;
    let file = std::fs::read(ivf).map_err(|_| "read ivf".to_string())?;
    let packet = ivf_payload(&file)?;
    let (width, height) = ivf_size(&file)?;
    let name = input
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("file");
    say(&format!(
        "workload: {width}x{height} keyframe, {} bytes, from {name}",
        packet.len()
    ));
    say(&machine());
    say(&format!("ffmpeg: {}", ffmpeg_version(ffmpeg)?));
    say(
        "one sample. cutlass times one Av1Decoder::push in this process. ffmpeg times one process and includes startup. bench profile, no link-time optimization. cutlass decodes tiles in parallel.",
    );

    let ours = decode_once(packet)?;
    let reference = ffmpeg_raw(ffmpeg, ivf, yuv)?;
    if ours != reference {
        let diffs = ours
            .iter()
            .zip(reference.iter())
            .filter(|(left, right)| left != right)
            .count();
        return Err(format!(
            "pixels differ ({diffs} of ours {}, ffmpeg {})",
            ours.len(),
            reference.len()
        ));
    }
    say("pixels: match");

    let cutlass = time_cutlass(packet)?;
    say(&format!("cutlass-av1: {cutlass:.4} s  one push"));
    let ffmpeg_seconds = time_ffmpeg(ffmpeg, ivf)?;
    say(&format!(
        "ffmpeg libdav1d -threads 1: {ffmpeg_seconds:.4} s  one process, includes startup"
    ));
    Ok(())
}

fn time_cutlass(packet: &[u8]) -> Result<f64, String> {
    let params = CodecParams::video(CodecId::Av1, 64, 64);
    let mut decoder = Av1Decoder::open(&params).map_err(|error| format!("open {error:?}"))?;
    let frame = Packet::video(packet.to_vec(), 0, 1, 30, true);
    let start = Instant::now();
    let decoded = decoder
        .push(&frame)
        .map_err(|error| format!("push {error:?}"))?;
    let elapsed = start.elapsed().as_secs_f64();
    let picture = decoded
        .first()
        .and_then(|frame| frame.as_video())
        .ok_or_else(|| "no picture".to_string())?;
    std::hint::black_box(picture.picture.y.samples().len());
    Ok(elapsed)
}

fn decode_once(packet: &[u8]) -> Result<Vec<u8>, String> {
    let params = CodecParams::video(CodecId::Av1, 64, 64);
    let mut decoder = Av1Decoder::open(&params).map_err(|error| format!("open {error:?}"))?;
    let decoded = decoder
        .push(&Packet::video(packet.to_vec(), 0, 1, 30, true))
        .map_err(|error| format!("push {error:?}"))?;
    let picture = decoded
        .first()
        .and_then(|frame| frame.as_video())
        .ok_or_else(|| "no picture".to_string())?;
    let mut packed = picture.picture.y.samples().to_vec();
    packed.extend_from_slice(picture.picture.u.samples());
    packed.extend_from_slice(picture.picture.v.samples());
    Ok(packed)
}

fn time_ffmpeg(ffmpeg: &Path, ivf: &Path) -> Result<f64, String> {
    let started = Instant::now();
    let output = Command::new(ffmpeg)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-c:v",
            "libdav1d",
            "-i",
        ])
        .arg(ivf)
        .args(["-an", "-f", "null", "-"])
        .output()
        .map_err(|err| err.to_string())?;
    let elapsed = started.elapsed().as_secs_f64();
    if !output.status.success() {
        return Err(tail(&output.stderr));
    }
    Ok(elapsed)
}

fn ffmpeg_raw(ffmpeg: &Path, ivf: &Path, yuv: &Path) -> Result<Vec<u8>, String> {
    let output = Command::new(ffmpeg)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-threads",
            "1",
            "-c:v",
            "libdav1d",
            "-i",
        ])
        .arg(ivf)
        .args([
            "-an",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(yuv)
        .output()
        .map_err(|err| err.to_string())?;
    if !output.status.success() {
        return Err(tail(&output.stderr));
    }
    std::fs::read(yuv).map_err(|_| "read yuv".to_string())
}

fn extract_frame(ffmpeg: &Path, input: &Path, ivf: &Path) -> Result<(), String> {
    let output = Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(input)
        .args(["-frames:v", "1", "-c:v", "copy", "-f", "ivf"])
        .arg(ivf)
        .output()
        .map_err(|err| err.to_string())?;
    if output.status.success() {
        Ok(())
    } else {
        Err("ffmpeg could not copy the first frame".to_string())
    }
}

fn require_keyframe(ffmpeg: &Path, ivf: &Path) -> Result<(), String> {
    let output = Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "info", "-i"])
        .arg(ivf)
        .args(["-frames:v", "1", "-vf", "showinfo", "-f", "null", "-"])
        .output()
        .map_err(|err| err.to_string())?;
    if !output.status.success() {
        return Err("ffmpeg could not inspect the frame".to_string());
    }
    let text = String::from_utf8_lossy(&output.stderr);
    let mut saw_key = false;
    let mut saw_other = false;
    for line in text.lines() {
        if !line.contains("iskey:") {
            continue;
        }
        if line.contains("iskey:1") && line.contains("type:I") {
            saw_key = true;
        } else {
            saw_other = true;
        }
    }
    if saw_key && !saw_other {
        Ok(())
    } else {
        Err("the frame is not a keyframe".to_string())
    }
}

fn ffmpeg_version(ffmpeg: &Path) -> Result<String, String> {
    let output = Command::new(ffmpeg)
        .arg("-version")
        .output()
        .map_err(|err| err.to_string())?;
    if !output.status.success() {
        return Err("ffmpeg -version failed".to_string());
    }
    let text = String::from_utf8_lossy(&output.stdout);
    Ok(text.lines().next().unwrap_or("ffmpeg").trim().to_string())
}

fn ivf_payload(file: &[u8]) -> Result<&[u8], String> {
    if file.len() < 44 || &file[..4] != b"DKIF" {
        return Err("not an ivf".to_string());
    }
    let size = u32::from_le_bytes([file[32], file[33], file[34], file[35]]) as usize;
    let end = 44usize
        .checked_add(size)
        .ok_or_else(|| "ivf size".to_string())?;
    file.get(44..end).ok_or_else(|| "truncated ivf".to_string())
}

fn ivf_size(file: &[u8]) -> Result<(u16, u16), String> {
    if file.len() < 16 {
        return Err("short ivf".to_string());
    }
    Ok((
        u16::from_le_bytes([file[12], file[13]]),
        u16::from_le_bytes([file[14], file[15]]),
    ))
}

fn production_file() -> Option<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files");
    let mut entries = std::fs::read_dir(&root).ok()?;
    entries.find_map(|entry| {
        let path = entry.ok()?.path();
        let name = path.file_name()?.to_str()?.to_ascii_lowercase();
        let video = name.ends_with(".mp4") || name.ends_with(".webm") || name.ends_with(".mov");
        (video && name.contains("av1")).then_some(path)
    })
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
            return Some(PathBuf::from(candidate));
        }
    }
    None
}

fn machine() -> String {
    let cpu = Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown cpu".to_string());
    let cores = Command::new("sysctl")
        .args(["-n", "hw.ncpu"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .unwrap_or_else(|| "?".to_string());
    format!("machine: {cpu}, {cores} cores")
}

fn say(line: &str) {
    let _ = writeln!(std::io::stdout(), "{line}");
}

fn tail(stderr: &[u8]) -> String {
    let text = String::from_utf8_lossy(stderr);
    let start = text.len().saturating_sub(400);
    text[start..].trim().to_string()
}
