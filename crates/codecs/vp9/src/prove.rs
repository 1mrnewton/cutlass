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
    let (ours, temporal) = decode_shown(&stream, PixelFormat::Yuv420p);
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
        let (ours, temporal) = decode_shown(&stream, PixelFormat::Yuv420p);
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

#[test]
fn profile1_chroma_matches_ffmpeg_software() {
    let Some(source) = local_video() else {
        eprintln!("skipped: ../local-files has no profile 1 source");
        return;
    };
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg was not found");
        return;
    };
    let cases = [
        ("yuv422p", PixelFormat::Yuv422p),
        ("yuv440p", PixelFormat::Yuv440p),
        ("yuv444p", PixelFormat::Yuv444p),
        ("gbrp", PixelFormat::Gbrp),
    ];
    for (pix_fmt, format) in cases {
        let dir = temp_dir(pix_fmt);
        let ivf = dir.join("in.ivf");
        encode_profile1(&ffmpeg, &source, &ivf, pix_fmt);
        let stream = read_ivf(&ivf);
        assert!(!stream.frames.is_empty(), "{pix_fmt} produced no frames");
        let (ours, _) = decode_shown(&stream, format);
        let oracle = ffmpeg_yuv(&ffmpeg, &ivf, pix_fmt);
        assert_eq!(
            ours.len(),
            oracle.len(),
            "{pix_fmt} shown bytes differ {} vs {}",
            ours.len(),
            oracle.len()
        );
        let mismatch = ours.iter().zip(oracle.iter()).position(|(a, b)| a != b);
        assert_eq!(mismatch, None, "{pix_fmt} first mismatch at {mismatch:?}");
        eprintln!(
            "profile 1 {pix_fmt} {} frames {} bytes",
            stream.frames.len(),
            ours.len()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
fn profile2_depth_matches_ffmpeg_software() {
    let Some(source) = local_video() else {
        eprintln!("skipped: ../local-files has no profile 2 source");
        return;
    };
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg was not found");
        return;
    };
    let cases = [
        ("yuv420p10le", PixelFormat::Yuv420p10),
        ("yuv420p12le", PixelFormat::Yuv420p12),
    ];
    for (pix_fmt, format) in cases {
        let dir = temp_dir(pix_fmt);
        let ivf = dir.join("in.ivf");
        encode_profile2(&ffmpeg, &source, &ivf, pix_fmt);
        let stream = read_ivf(&ivf);
        assert!(!stream.frames.is_empty(), "{pix_fmt} produced no frames");
        let (ours, _) = decode_shown(&stream, format);
        let oracle = ffmpeg_yuv(&ffmpeg, &ivf, pix_fmt);
        assert_eq!(
            ours.len(),
            oracle.len(),
            "{pix_fmt} shown bytes differ {} vs {}",
            ours.len(),
            oracle.len()
        );
        let mismatch = ours.iter().zip(oracle.iter()).position(|(a, b)| a != b);
        assert_eq!(mismatch, None, "{pix_fmt} first mismatch at {mismatch:?}");
        eprintln!(
            "profile 2 {pix_fmt} {} frames {} bytes",
            stream.frames.len(),
            ours.len()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
fn profile3_chroma_matches_ffmpeg_software() {
    let Some(source) = local_video() else {
        eprintln!("skipped: ../local-files has no profile 3 source");
        return;
    };
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg was not found");
        return;
    };
    let cases = [
        ("yuv422p10le", PixelFormat::Yuv422p10),
        ("yuv422p12le", PixelFormat::Yuv422p12),
        ("yuv440p10le", PixelFormat::Yuv440p10),
        ("yuv440p12le", PixelFormat::Yuv440p12),
        ("yuv444p10le", PixelFormat::Yuv444p10),
        ("yuv444p12le", PixelFormat::Yuv444p12),
        ("gbrp10le", PixelFormat::Gbrp10),
        ("gbrp12le", PixelFormat::Gbrp12),
    ];
    for (pix_fmt, format) in cases {
        let dir = temp_dir(pix_fmt);
        let ivf = dir.join("in.ivf");
        encode_profile3(&ffmpeg, &source, &ivf, pix_fmt);
        let stream = read_ivf(&ivf);
        assert!(!stream.frames.is_empty(), "{pix_fmt} produced no frames");
        let (ours, _) = decode_shown(&stream, format);
        let oracle = ffmpeg_yuv(&ffmpeg, &ivf, pix_fmt);
        assert_eq!(
            ours.len(),
            oracle.len(),
            "{pix_fmt} shown bytes differ {} vs {}",
            ours.len(),
            oracle.len()
        );
        let mismatch = ours.iter().zip(oracle.iter()).position(|(a, b)| a != b);
        assert_eq!(mismatch, None, "{pix_fmt} first mismatch at {mismatch:?}");
        eprintln!(
            "profile 3 {pix_fmt} {} frames {} bytes",
            stream.frames.len(),
            ours.len()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
fn level_width_above_4096_matches_ffmpeg_software() {
    let Some(source) = local_video() else {
        eprintln!("skipped: ../local-files has no level source");
        return;
    };
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg was not found");
        return;
    };
    let dir = temp_dir("level-4128");
    let ivf = dir.join("in.ivf");
    let status = Command::new(&ffmpeg)
        .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(&source)
        .args([
            "-an",
            "-frames:v",
            "2",
            "-vf",
            "scale=4128:16",
            "-c:v",
            "libvpx-vp9",
            "-profile:v",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            "200k",
            "-deadline",
            "realtime",
            "-cpu-used",
            "8",
            "-f",
            "ivf",
        ])
        .arg(&ivf)
        .output()
        .expect("encode");
    assert!(
        status.status.success(),
        "encode {}",
        String::from_utf8_lossy(&status.stderr)
    );
    let stream = read_ivf(&ivf);
    assert!(stream.width > 4096, "coded width {}", stream.width);
    let (ours, _) = decode_shown(&stream, PixelFormat::Yuv420p);
    let oracle = ffmpeg_yuv(&ffmpeg, &ivf, "yuv420p");
    assert_eq!(ours.len(), oracle.len());
    let mismatch = ours.iter().zip(oracle.iter()).position(|(a, b)| a != b);
    assert_eq!(mismatch, None, "first mismatch at {mismatch:?}");
    eprintln!(
        "level width {}x{} {} frames {} bytes",
        stream.width,
        stream.height,
        stream.frames.len(),
        ours.len()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Official level bitstreams from the levels document. The WebM is copied to
/// IVF without re-encoding. The test skips when those files are not on disk.
#[test]
fn official_level_stream_matches_ffmpeg_software() {
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg was not found");
        return;
    };
    let cases = [
        (
            "crowd_run_256X144_fr15_bd8_gf_dist_4_l1.webm",
            "yuv420p",
            PixelFormat::Yuv420p,
            "0",
            "gf_dist",
        ),
        (
            "crowd_run_256X144_fr15_bd8_sub8X8_l1.webm",
            "yuv420p",
            PixelFormat::Yuv420p,
            "0",
            "sub8x8",
        ),
        (
            "crowd_run_256X144_fr15_bd8_8buf_l1.webm",
            "yuv420p",
            PixelFormat::Yuv420p,
            "0",
            "buf",
        ),
        (
            "crowd_run_248X144_fr15_bd8_odd_size_l1.webm",
            "yuv420p",
            PixelFormat::Yuv420p,
            "0",
            "odd_size",
        ),
        (
            "crowd_run_256X144_fr15_bd8_frm_resize_l1.webm",
            "yuv420p",
            PixelFormat::Yuv420p,
            "0",
            "frm_resize",
        ),
        (
            "crowd_run_256X144_fr15_bd8_sub8x8_sf_l1.webm",
            "yuv420p",
            PixelFormat::Yuv420p,
            "0",
            "sub8x8_sf",
        ),
        (
            "crowd_run_256X144_fr15_bd10_gf_dist_4_l1.webm",
            "yuv420p10le",
            PixelFormat::Yuv420p10,
            "2",
            "gf_dist",
        ),
        (
            "crowd_run_256X144_fr15_bd10_sub8X8_l1.webm",
            "yuv420p10le",
            PixelFormat::Yuv420p10,
            "2",
            "sub8x8",
        ),
        (
            "crowd_run_256X144_fr15_bd10_8buf_l1.webm",
            "yuv420p10le",
            PixelFormat::Yuv420p10,
            "2",
            "buf",
        ),
        (
            "crowd_run_248X144_fr15_bd10_odd_size_l1.webm",
            "yuv420p10le",
            PixelFormat::Yuv420p10,
            "2",
            "odd_size",
        ),
        (
            "crowd_run_256X144_fr15_bd10_frm_resize_l1.webm",
            "yuv420p10le",
            PixelFormat::Yuv420p10,
            "2",
            "frm_resize",
        ),
        (
            "crowd_run_256X144_fr15_bd10_sub8x8_sf_l1.webm",
            "yuv420p10le",
            PixelFormat::Yuv420p10,
            "2",
            "sub8x8_sf",
        ),
    ];
    let mut ran = 0u32;
    for (name, pix_fmt, format, profile, kind) in cases {
        let Some(webm) = local_level_stream(name) else {
            continue;
        };
        ran += 1;
        let dir = temp_dir("level-stream");
        let ivf = dir.join("in.ivf");
        copy_webm_to_ivf(&ffmpeg, &webm, &ivf);
        let stream = read_ivf(&ivf);
        assert!(!stream.frames.is_empty(), "{name} produced no frames");
        let (ours, _) = decode_shown(&stream, format);
        let oracle = ffmpeg_yuv(&ffmpeg, &ivf, pix_fmt);
        assert_eq!(
            ours.len(),
            oracle.len(),
            "{name} shown bytes differ {} vs {}",
            ours.len(),
            oracle.len()
        );
        let mismatch = ours.iter().zip(oracle.iter()).position(|(a, b)| a != b);
        assert_eq!(mismatch, None, "{name} first mismatch at {mismatch:?}");
        eprintln!(
            "official profile {profile} level 1 {kind} {}x{} {} frames {} bytes",
            stream.width,
            stream.height,
            stream.frames.len(),
            ours.len()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
    if ran == 0 {
        eprintln!("skipped: official level streams are not in ../local-files/vp9-levels");
    }
}

#[test]
fn uhd_3840x2160_matches_ffmpeg_software() {
    let Some(source) = local_uhd() else {
        eprintln!("skipped: ../local-files has no 3840x2160 source");
        return;
    };
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg was not found");
        return;
    };
    let dir = temp_dir("uhd-3840");
    let ivf = dir.join("in.ivf");
    encode_profile0(
        &ffmpeg,
        &source,
        &ivf,
        &["-frames:v", "2", "-vf", "scale=3840:2160"],
    );
    let stream = read_ivf(&ivf);
    assert_eq!(stream.width, 3840, "coded width {}", stream.width);
    assert_eq!(stream.height, 2160, "coded height {}", stream.height);
    let (ours, _) = decode_shown(&stream, PixelFormat::Yuv420p);
    let oracle = ffmpeg_yuv(&ffmpeg, &ivf, "yuv420p");
    assert_eq!(ours.len(), oracle.len());
    let mismatch = ours.iter().zip(oracle.iter()).position(|(a, b)| a != b);
    assert_eq!(mismatch, None, "first mismatch at {mismatch:?}");
    eprintln!(
        "uhd {}x{} {} frames {} bytes",
        stream.width,
        stream.height,
        stream.frames.len(),
        ours.len()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Four tile columns and two tile rows. Level 1 streams are one column, so they
/// do not exercise this path.
#[test]
fn tiled_picture_matches_ffmpeg_software() {
    let Some(source) = local_video() else {
        eprintln!("skipped: ../local-files has no tile source");
        return;
    };
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg was not found");
        return;
    };
    let dir = temp_dir("tiles");
    let ivf = dir.join("in.ivf");
    encode_profile0(
        &ffmpeg,
        &source,
        &ivf,
        &[
            "-frames:v",
            "4",
            "-vf",
            "scale=1024:256",
            "-tile-columns",
            "2",
            "-tile-rows",
            "1",
            "-threads",
            "1",
        ],
    );
    let stream = read_ivf(&ivf);
    assert!(stream.frames.len() > 1, "tiles need an inter frame");
    let parsed = crate::header::parse(&crate::header::Persist::new(), &stream.frames[0])
        .expect("keyframe header");
    assert_eq!(parsed.header.log2_tile_cols, 2, "tile columns");
    assert_eq!(parsed.header.log2_tile_rows, 1, "tile rows");
    let (ours, _) = decode_shown(&stream, PixelFormat::Yuv420p);
    let oracle = ffmpeg_yuv(&ffmpeg, &ivf, "yuv420p");
    assert_eq!(ours.len(), oracle.len());
    let mismatch = ours.iter().zip(oracle.iter()).position(|(a, b)| a != b);
    assert_eq!(mismatch, None, "first mismatch at {mismatch:?}");
    eprintln!(
        "tiles 4x2 {}x{} {} frames {} bytes",
        stream.width,
        stream.height,
        stream.frames.len(),
        ours.len()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

fn decode_shown(stream: &Ivf, pixel: PixelFormat) -> (Vec<u8>, u32) {
    let params = CodecParams::with_format(CodecId::Vp9, stream.width, stream.height, pixel);
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

fn local_level_stream(name: &str) -> Option<PathBuf> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../local-files/vp9-levels")
        .join(name);
    path.is_file().then_some(path)
}

fn copy_webm_to_ivf(ffmpeg: &Path, webm: &Path, ivf: &Path) {
    let status = Command::new(ffmpeg)
        .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(webm)
        .args(["-an", "-c:v", "copy", "-f", "ivf"])
        .arg(ivf)
        .output()
        .expect("copy");
    assert!(
        status.status.success(),
        "copy {} {}",
        webm.display(),
        String::from_utf8_lossy(&status.stderr)
    );
}

fn local_uhd() -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files");
    if !root.is_dir() {
        return None;
    }
    const NAMES: &[&str] = &[
        "14703872_3840_2160_60fps.mp4",
        "12471039-uhd_3840_2160_24fps.mp4",
        "12129324_3840_2160_30fps.mp4",
    ];
    for name in NAMES {
        let path = root.join(name);
        if path.is_file() {
            return Some(path);
        }
    }
    None
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

fn encode_profile1(ffmpeg: &Path, source: &Path, ivf: &Path, pix_fmt: &str) {
    let status = Command::new(ffmpeg)
        .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(source)
        .args([
            "-an",
            "-frames:v",
            "8",
            "-vf",
            "scale=320:240",
            "-c:v",
            "libvpx-vp9",
            "-profile:v",
            "1",
            "-pix_fmt",
            pix_fmt,
            "-b:v",
            "400k",
            "-deadline",
            "realtime",
            "-cpu-used",
            "8",
            "-f",
            "ivf",
        ])
        .arg(ivf)
        .output()
        .expect("encode");
    assert!(
        status.status.success(),
        "encode {pix_fmt} {}",
        String::from_utf8_lossy(&status.stderr)
    );
}

fn encode_profile2(ffmpeg: &Path, source: &Path, ivf: &Path, pix_fmt: &str) {
    let status = Command::new(ffmpeg)
        .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(source)
        .args([
            "-an",
            "-frames:v",
            "4",
            "-vf",
            "scale=160:96",
            "-c:v",
            "libvpx-vp9",
            "-profile:v",
            "2",
            "-pix_fmt",
            pix_fmt,
            "-b:v",
            "400k",
            "-deadline",
            "realtime",
            "-cpu-used",
            "8",
            "-f",
            "ivf",
        ])
        .arg(ivf)
        .output()
        .expect("encode");
    assert!(
        status.status.success(),
        "encode {pix_fmt} {}",
        String::from_utf8_lossy(&status.stderr)
    );
}

fn encode_profile3(ffmpeg: &Path, source: &Path, ivf: &Path, pix_fmt: &str) {
    let status = Command::new(ffmpeg)
        .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(source)
        .args([
            "-an",
            "-frames:v",
            "4",
            "-vf",
            "scale=160:96",
            "-c:v",
            "libvpx-vp9",
            "-profile:v",
            "3",
            "-pix_fmt",
            pix_fmt,
            "-b:v",
            "400k",
            "-deadline",
            "realtime",
            "-cpu-used",
            "8",
            "-f",
            "ivf",
        ])
        .arg(ivf)
        .output()
        .expect("encode");
    assert!(
        status.status.success(),
        "encode {pix_fmt} {}",
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
        .args(["-noautoscale", "-f", "rawvideo", "-pix_fmt", pix_fmt])
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
