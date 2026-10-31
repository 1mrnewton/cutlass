//! Real profile-0 pictures from `../local-files`, compared with ffmpeg's software vp9 decoder.
//!
//! The folder is absent in CI. The test skips in that case. The oracle is
//! `ffmpeg -hwaccel none -threads 1 -c:v vp9`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use cutlass_decode::video::VideoDecoder;
use cutlass_decode::{CodecId, CodecParams, Packet, PixelFormat};

use crate::Vp9Decoder;

#[test]
fn local_profile0_matches_ffmpeg_software() {
    let Some(source) = local_video() else {
        eprintln!("skipped: ../local-files has no profile 0 source");
        return;
    };
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg was not found");
        return;
    };
    let dir = temp_dir("profile0");
    let ivf = dir.join("in.ivf");
    encode_profile0(&ffmpeg, &source, &ivf, &[]);
    let stream = read_ivf(&ivf);
    assert!(!stream.frames.is_empty(), "encoder produced no frames");
    let started = Instant::now();
    let (ours, temporal) = decode_shown(&stream);
    let elapsed = started.elapsed();
    let oracle = ffmpeg_yuv(&ffmpeg, &ivf, "yuv420p");
    assert_eq!(ours.len(), oracle.len(), "shown bytes differ");
    let mismatch = ours.iter().zip(oracle.iter()).position(|(a, b)| a != b);
    assert_eq!(mismatch, None, "first mismatch at {mismatch:?}");
    eprintln!(
        "profile 0 {} frames {} bytes {} temporal {} wall {:.3} s",
        source
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("file"),
        stream.frames.len(),
        ours.len(),
        temporal,
        elapsed.as_secs_f64()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn segmentation_temporal_update_matches_ffmpeg() {
    let Some(source) = local_video() else {
        eprintln!("skipped: ../local-files has no profile 0 source");
        return;
    };
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg was not found");
        return;
    };
    let dir = temp_dir("temporal");
    let ivf = dir.join("in.ivf");
    let mut saw = 0u32;
    for aq in ["3", "2", "1"] {
        encode_profile0(
            &ffmpeg,
            &source,
            &ivf,
            &[
                "-frames:v",
                "24",
                "-vf",
                "scale=320:240",
                "-aq-mode",
                aq,
                "-deadline",
                "good",
                "-cpu-used",
                "2",
                "-lag-in-frames",
                "8",
                "-auto-alt-ref",
                "1",
            ],
        );
        let stream = read_ivf(&ivf);
        let (ours, temporal) = decode_shown(&stream);
        if temporal == 0 {
            continue;
        }
        saw = temporal;
        let oracle = ffmpeg_yuv(&ffmpeg, &ivf, "yuv420p");
        assert_eq!(ours.len(), oracle.len(), "aq-mode {aq} shown bytes differ");
        let mismatch = ours.iter().zip(oracle.iter()).position(|(a, b)| a != b);
        assert_eq!(
            mismatch, None,
            "aq-mode {aq} first mismatch at {mismatch:?}"
        );
        eprintln!(
            "segmentation temporal update aq-mode {aq} frames {} bytes {} temporal frames {temporal}",
            stream.frames.len(),
            ours.len()
        );
        break;
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        saw > 0,
        "no encoded frame set the segmentation temporal-update bit"
    );
}

fn decode_shown(stream: &Ivf) -> (Vec<u8>, u32) {
    let params = CodecParams::with_format(
        CodecId::Vp9,
        stream.width,
        stream.height,
        PixelFormat::Yuv420p,
    );
    let mut decoder = Vp9Decoder::open(&params).expect("open");
    let mut packed = Vec::new();
    for (index, frame) in stream.frames.iter().enumerate() {
        let pts = i64::try_from(index).unwrap_or(0);
        let packet = Packet::video(frame.clone(), pts, 1, 30, index == 0);
        let outputs = decoder
            .push(&packet)
            .unwrap_or_else(|err| panic!("frame {index} {err}"));
        for output in outputs {
            packed.extend_from_slice(&output.frame.y.samples);
            packed.extend_from_slice(&output.frame.u.samples);
            packed.extend_from_slice(&output.frame.v.samples);
        }
    }
    (packed, decoder.temporal_frames())
}

fn local_video() -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files");
    if !root.is_dir() {
        return None;
    }
    const NAMES: &[&str] = &[
        "snaptik_7282700172428086530_v3.mp4",
        "IMG_8146.MP4",
        "7150172-hd_720_1280_50fps.mp4",
    ];
    for name in NAMES {
        let path = root.join(name);
        if path.is_file() {
            return Some(path);
        }
    }
    None
}

fn encode_profile0(ffmpeg: &Path, source: &Path, ivf: &Path, extra: &[&str]) {
    let mut command = Command::new(ffmpeg);
    command.args(["-y", "-hide_banner", "-loglevel", "error", "-i"]);
    command.arg(source);
    command.args([
        "-an",
        "-c:v",
        "libvpx-vp9",
        "-profile:v",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-b:v",
        "400k",
        "-deadline",
        "realtime",
        "-cpu-used",
        "8",
    ]);
    command.args(extra);
    command.args(["-f", "ivf"]);
    command.arg(ivf);
    let status = command.output().expect("encode");
    assert!(
        status.status.success(),
        "encode {}",
        String::from_utf8_lossy(&status.stderr)
    );
}

fn ffmpeg_yuv(ffmpeg: &Path, ivf: &Path, pix_fmt: &str) -> Vec<u8> {
    let output = ivf.with_extension("yuv");
    let status = Command::new(ffmpeg)
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-hwaccel",
            "none",
            "-threads",
            "1",
            "-c:v",
            "vp9",
            "-i",
        ])
        .arg(ivf)
        .args(["-f", "rawvideo", "-pix_fmt", pix_fmt])
        .arg(&output)
        .output()
        .expect("ffmpeg");
    assert!(
        status.status.success(),
        "vp9 {}",
        String::from_utf8_lossy(&status.stderr)
    );
    std::fs::read(&output).expect("yuv")
}

fn ffmpeg_bin() -> Option<PathBuf> {
    for candidate in [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "ffmpeg",
    ] {
        let path = PathBuf::from(candidate);
        if Command::new(&path)
            .arg("-version")
            .output()
            .is_ok_and(|output| output.status.success())
        {
            return Some(path);
        }
    }
    None
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "cutlass-vp9-{name}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("dir");
    dir
}

struct Ivf {
    width: u32,
    height: u32,
    frames: Vec<Vec<u8>>,
}

fn read_ivf(path: &Path) -> Ivf {
    let data = std::fs::read(path).expect("ivf");
    parse_ivf(&data)
}

fn parse_ivf(data: &[u8]) -> Ivf {
    assert!(data.len() >= 32, "ivf header");
    assert_eq!(&data[..4], b"DKIF");
    let header = u16::from_le_bytes([data[6], data[7]]) as usize;
    assert!(header >= 32 && data.len() >= header, "ivf header length");
    let width = u32::from(u16::from_le_bytes([data[12], data[13]]));
    let height = u32::from(u16::from_le_bytes([data[14], data[15]]));
    assert!(width > 0 && height > 0, "ivf size {width}x{height}");
    let mut at = header;
    let mut frames = Vec::new();
    while at + 12 <= data.len() {
        let size = u32::from_le_bytes(data[at..at + 4].try_into().expect("size")) as usize;
        at += 12;
        assert!(at + size <= data.len(), "ivf frame");
        frames.push(data[at..at + size].to_vec());
        at += size;
    }
    Ivf {
        width,
        height,
        frames,
    }
}
