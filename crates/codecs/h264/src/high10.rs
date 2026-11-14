//! High 10 4:2:0 pictures compared with FFmpeg's software decoder.
//!
//! The stream is produced by libx264 when `ffmpeg` is installed. The test
//! skips when `ffmpeg` or that encoder is missing. The oracle is one software
//! thread, and the compared samples are 10-bit.

use std::path::PathBuf;
use std::process::{Command, Stdio};

use cutlass_decode::{CodecId, CodecParams, Packet, VideoDecoder, Yuv420p10};

use crate::H264Decoder;

const WIDTH: usize = 160;
const HEIGHT: usize = 96;

#[test]
fn high10_420_matches_ffmpeg() {
    matches_ffmpeg("testsrc");
}

/// libx264 changes QP by more than 26 at once on this pattern, which only
/// a High 10 stream may: `mb_qp_delta` there runs from -32 to 31.
#[test]
fn high10_wide_qp_steps_match_ffmpeg() {
    matches_ffmpeg("testsrc2");
}

/// Encodes 0.2 s of the FFmpeg test pattern `source` as High 10 and
/// compares every decoded 10-bit sample with FFmpeg's.
fn matches_ffmpeg(source: &str) {
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg is not installed");
        return;
    };
    let dir = std::env::temp_dir();
    let h264 = dir.join(format!(
        "cutlass-h264-high10-{source}-{}.h264",
        std::process::id()
    ));
    let encoded = Command::new(&ffmpeg)
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            &format!("{source}=size=160x96:rate=30:duration=0.2"),
            "-an",
            "-c:v",
            "libx264",
            "-profile:v",
            "high10",
            "-pix_fmt",
            "yuv420p10le",
            "-bf",
            "2",
            "-g",
            "8",
            "-f",
            "h264",
        ])
        .arg(&h264)
        .status();
    let Ok(status) = encoded else {
        eprintln!("skipped: ffmpeg could not be started");
        return;
    };
    if !status.success() {
        eprintln!("skipped: this ffmpeg cannot encode High 10");
        return;
    }
    let annex = std::fs::read(&h264).expect("annex b");
    let raw = Command::new(&ffmpeg)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-hwaccel",
            "none",
            "-threads",
            "1",
            "-i",
        ])
        .arg(&h264)
        .args(["-an", "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-"])
        .stdout(Stdio::piped())
        .output()
        .expect("ffmpeg raw");
    assert!(
        raw.status.success(),
        "ffmpeg did not decode High 10: {}",
        String::from_utf8_lossy(&raw.stderr)
    );
    let _ = std::fs::remove_file(&h264);

    let units = access_units(&annex);
    assert!(!units.is_empty(), "High 10 produced no access units");
    let mut params = CodecParams::video(CodecId::H264, WIDTH as u32, HEIGHT as u32);
    params.nal_length_size = 0;
    let mut decoder = H264Decoder::open(&params).expect("open");
    let mut frames = Vec::new();
    for (index, unit) in units.iter().enumerate() {
        let pictures = decoder
            .push(&Packet::video(
                unit.clone(),
                index as i64,
                1,
                30,
                index == 0,
            ))
            .unwrap_or_else(|err| panic!("access unit {index}: {err}"));
        frames.extend(pictures);
    }
    frames.extend(decoder.drain().expect("drain"));
    let frame_len = WIDTH * HEIGHT * 2 + (WIDTH / 2) * (HEIGHT / 2) * 2 * 2;
    assert_eq!(
        raw.stdout.len(),
        frames.len() * frame_len,
        "ffmpeg produced a different frame count"
    );
    assert!(frames.len() > 1, "High 10 produced one picture");
    for (index, picture) in frames.iter().enumerate() {
        let Some(ten) = picture.yuv420p10.as_ref() else {
            panic!("picture {index} dropped the 10-bit samples");
        };
        let start = index * frame_len;
        same_ten(index, ten, &raw.stdout[start..start + frame_len]);
        assert_eq!(
            picture.picture.y()[0],
            (ten.y()[0] >> 2) as u8,
            "the 8-bit view is the top 8 bits"
        );
    }
}

fn ffmpeg_bin() -> Option<PathBuf> {
    for candidate in [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "ffmpeg",
    ] {
        let path = PathBuf::from(candidate);
        let Ok(output) = Command::new(candidate).arg("-version").output() else {
            continue;
        };
        if output.status.success() {
            return Some(path);
        }
    }
    None
}

fn same_ten(index: usize, picture: &Yuv420p10, raw: &[u8]) {
    let y_len = WIDTH * HEIGHT;
    let c_len = y_len / 4;
    let y = words(&raw[..y_len * 2]);
    let u = words(&raw[y_len * 2..y_len * 2 + c_len * 2]);
    let v = words(&raw[y_len * 2 + c_len * 2..]);
    same(index, "y", picture.y(), &y);
    same(index, "u", picture.u(), &u);
    same(index, "v", picture.v(), &v);
}

fn words(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect()
}

fn same(index: usize, name: &str, got: &[u16], want: &[u16]) {
    assert_eq!(got.len(), want.len(), "picture {index} {name} length");
    if let Some(at) = got.iter().zip(want).position(|(a, b)| a != b) {
        panic!(
            "picture {index} {name}[{at}] = {} want {}",
            got[at], want[at]
        );
    }
}

fn access_units(data: &[u8]) -> Vec<Vec<u8>> {
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
    units
}
