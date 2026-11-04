//! A real 3840×2160 High picture from `../local-files`, compared with FFmpeg.
//!
//! The folder is not in CI. The test skips when it is missing, and when the
//! named file or `ffmpeg` is missing. The oracle is one software thread.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use cutlass_decode::{CodecId, CodecParams, Packet, VideoDecoder, Yuv420p};

use crate::H264Decoder;

const FILE_NAME: &str = "12471039-uhd_3840_2160_24fps.mp4";
const WIDTH: usize = 3840;
const HEIGHT: usize = 2160;
const Y_LEN: usize = WIDTH * HEIGHT;
const C_LEN: usize = Y_LEN / 4;
const FRAME_LEN: usize = Y_LEN + C_LEN * 2;

#[test]
fn uhd_3840x2160_high_matches_ffmpeg() {
    let Some(path) = uhd_file() else {
        return;
    };
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg is not installed");
        return;
    };
    let annex = annex_b(&ffmpeg, &path);
    let units = access_units(&annex);
    assert!(
        !units.is_empty(),
        "the 3840x2160 file produced no access units"
    );

    let mut params = CodecParams::video(CodecId::H264, WIDTH as u32, HEIGHT as u32);
    params.nal_length_size = 0;
    let mut decoder = H264Decoder::open(&params).expect("open");
    let mut oracle = Oracle::spawn(&ffmpeg, &path);
    let mut raw = vec![0u8; FRAME_LEN];
    let mut frames = 0usize;
    let mut decode_time = Duration::ZERO;

    for (index, unit) in units.iter().enumerate() {
        let packet = Packet::video(unit.clone(), index as i64, 1, 24, index == 0);
        let start = Instant::now();
        let pictures = decoder
            .push(&packet)
            .unwrap_or_else(|err| panic!("access unit {index}: {err}"));
        decode_time += start.elapsed();
        for picture in &pictures {
            expect_frame(&mut oracle, &mut raw, frames);
            same_picture(frames, &picture.picture, &raw);
            frames += 1;
        }
    }
    let start = Instant::now();
    let tail = decoder.drain().expect("drain");
    decode_time += start.elapsed();
    for picture in &tail {
        expect_frame(&mut oracle, &mut raw, frames);
        same_picture(frames, &picture.picture, &raw);
        frames += 1;
    }
    assert!(frames > 0, "the 3840x2160 file produced no pictures");
    assert!(
        oracle.finished_exactly(),
        "ffmpeg produced a different frame count than {frames}"
    );
    let seconds = decode_time.as_secs_f64();
    let fps = frames as f64 / seconds;
    eprintln!(
        "uhd 3840x2160 high: {frames} pictures in {:.0} ms ({fps:.1} fps)",
        seconds * 1_000.0
    );
}

fn uhd_file() -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files");
    if !root.is_dir() {
        eprintln!("skipped: ../local-files is missing");
        return None;
    }
    let path = root.join(FILE_NAME);
    if !path.is_file() {
        eprintln!("skipped: {FILE_NAME} is not in ../local-files");
        return None;
    }
    Some(path)
}

fn ffmpeg_bin() -> Option<String> {
    for candidate in [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "ffmpeg",
    ] {
        let Ok(output) = Command::new(candidate).arg("-version").output() else {
            continue;
        };
        if output.status.success() {
            return Some(candidate.to_string());
        }
    }
    None
}

fn annex_b(ffmpeg: &str, input: &Path) -> Vec<u8> {
    let output = Command::new(ffmpeg)
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
        .arg(input)
        .args([
            "-an",
            "-c:v",
            "copy",
            "-bsf:v",
            "h264_mp4toannexb",
            "-f",
            "h264",
            "-",
        ])
        .output()
        .expect("ffmpeg annex b");
    assert!(
        output.status.success(),
        "ffmpeg did not write annex b: {}",
        tail(&output.stderr)
    );
    assert!(!output.stdout.is_empty(), "annex b was empty");
    output.stdout
}

/// `first_mb_in_slice` 0 is a single 1 bit, so a new picture sets the high bit
/// of the first slice-header byte. An access-unit delimiter does the same split.
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
    assert!(!starts.is_empty(), "annex b had no start code");
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

struct Oracle {
    child: Child,
    stdout: std::process::ChildStdout,
    stderr: PathBuf,
}

impl Oracle {
    fn spawn(ffmpeg: &str, input: &Path) -> Self {
        let stderr_path =
            std::env::temp_dir().join(format!("cutlass-h264-uhd-{}.err", std::process::id()));
        let stderr = std::fs::File::create(&stderr_path).expect("ffmpeg stderr");
        let mut child = Command::new(ffmpeg)
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
            .arg(input)
            .args(["-an", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])
            .stdout(Stdio::piped())
            .stderr(Stdio::from(stderr))
            .spawn()
            .expect("ffmpeg decode");
        let stdout = child.stdout.take().expect("ffmpeg stdout");
        Self {
            child,
            stdout,
            stderr: stderr_path,
        }
    }

    fn finished_exactly(&mut self) -> bool {
        let mut extra = [0u8; 1];
        let more = self.stdout.read(&mut extra).expect("ffmpeg tail");
        let status = self.child.wait().expect("ffmpeg wait");
        let err = std::fs::read_to_string(&self.stderr).unwrap_or_default();
        let _ = std::fs::remove_file(&self.stderr);
        assert!(status.success(), "ffmpeg decode failed: {}", err.trim());
        more == 0
    }
}

impl Drop for Oracle {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_file(&self.stderr);
    }
}

fn expect_frame(oracle: &mut Oracle, raw: &mut [u8], index: usize) {
    oracle
        .stdout
        .read_exact(raw)
        .unwrap_or_else(|err| panic!("ffmpeg frame {index}: {err}"));
}

fn same_picture(index: usize, picture: &Yuv420p, raw: &[u8]) {
    assert_eq!(
        picture.width(),
        WIDTH as u32,
        "picture {index} width {}",
        picture.width()
    );
    assert_eq!(
        picture.height(),
        HEIGHT as u32,
        "picture {index} height {}",
        picture.height()
    );
    same_plane("y", index, picture.y(), &raw[..Y_LEN]);
    same_plane("u", index, picture.u(), &raw[Y_LEN..Y_LEN + C_LEN]);
    same_plane("v", index, picture.v(), &raw[Y_LEN + C_LEN..]);
}

fn same_plane(name: &str, index: usize, got: &[u8], want: &[u8]) {
    assert_eq!(got.len(), want.len(), "picture {index} {name} length");
    let mut first = None;
    let mut count = 0usize;
    let mut max = 0u8;
    for (at, (got_byte, want_byte)) in got.iter().zip(want.iter()).enumerate() {
        let delta = got_byte.abs_diff(*want_byte);
        if delta == 0 {
            continue;
        }
        count += 1;
        if first.is_none() {
            first = Some((at, *got_byte, *want_byte));
        }
        if delta > max {
            max = delta;
        }
    }
    if let Some((at, got_byte, want_byte)) = first {
        panic!(
            "picture {index} {name} disagrees at sample {at}: decoder {got_byte}, ffmpeg {want_byte}, {count} samples, max delta {max}"
        );
    }
}

fn tail(stderr: &[u8]) -> String {
    let text = String::from_utf8_lossy(stderr);
    let start = text.len().saturating_sub(400);
    text[start..].trim().to_string()
}
