//! Decode one tiny AV1 keyframe that ffmpeg writes, and compare the picture.
//!
//! ffmpeg is an oracle. This example does not link it. Without the binary it
//! prints `skip: ffmpeg is not installed` and exits 0.

#![allow(clippy::print_stdout)]

use std::path::{Path, PathBuf};
use std::process::Command;

use cutlass_av1::{AsView, Av1Decoder};
use cutlass_decode::{CodecId, CodecParams, Decoder, Packet};

fn main() -> Result<(), String> {
    let Some(ffmpeg) = ffmpeg_bin() else {
        println!("skip: ffmpeg is not installed");
        return Ok(());
    };
    let dir = std::env::temp_dir().join(format!("cutlass-av1-example-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|err| err.to_string())?;
    let outcome = roundtrip(&ffmpeg, &dir);
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

fn roundtrip(ffmpeg: &Path, dir: &Path) -> Result<(), String> {
    let ivf = dir.join("frame.ivf");
    let yuv = dir.join("frame.yuv");
    write_keyframe(ffmpeg, &ivf)?;
    require_keyframe(ffmpeg, &ivf)?;
    let file = std::fs::read(&ivf).map_err(|err| err.to_string())?;
    let packet = ivf_payload(&file)?;
    let reference = ffmpeg_raw(ffmpeg, &ivf, &yuv)?;
    let params = CodecParams::video(CodecId::Av1, 64, 64);
    let mut decoder = Av1Decoder::open(&params).map_err(|err| format!("open {err:?}"))?;
    let decoded = decoder
        .push(&Packet::video(packet.to_vec(), 0, 1, 30, true))
        .map_err(|err| format!("decode {err:?}"))?;
    let picture = decoded
        .first()
        .and_then(|output| output.as_video())
        .ok_or_else(|| "decoder returned no picture".to_string())?;
    let width = picture.picture.y.width();
    let height = picture.picture.y.height();
    let mut packed = picture.picture.y.samples().to_vec();
    packed.extend_from_slice(picture.picture.u.samples());
    packed.extend_from_slice(picture.picture.v.samples());
    if packed == reference {
        println!("{width}x{height} match");
        Ok(())
    } else {
        println!("{width}x{height} mismatch");
        Err("picture mismatch".to_string())
    }
}

fn write_keyframe(ffmpeg: &Path, ivf: &Path) -> Result<(), String> {
    let output = Command::new(ffmpeg)
        .env("SVT_LOG", "0")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=1",
            "-frames:v",
            "1",
            "-an",
            "-c:v",
            "libsvtav1",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "30",
            "-preset",
            "8",
            "-g",
            "1",
            "-svtav1-params",
            "keyint=1:film-grain=0:film-grain-denoise=0:enable-restoration=0:enable-cdef=0:aq-mode=0:superres-mode=0:tune=1",
            "-f",
            "ivf",
        ])
        .arg(ivf)
        .output()
        .map_err(|err| err.to_string())?;
    if output.status.success() {
        Ok(())
    } else {
        Err(tail(&output.stderr))
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
        return Err(tail(&output.stderr));
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
        Err("ffmpeg did not write a keyframe".to_string())
    }
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
    std::fs::read(yuv).map_err(|err| err.to_string())
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

fn tail(stderr: &[u8]) -> String {
    let text = String::from_utf8_lossy(stderr);
    let start = text.len().saturating_sub(400);
    text[start..].trim().to_string()
}
