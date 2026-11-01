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

/// Quantizer 0 uses the Walsh-Hadamard transform. Profile 0 already matches.
/// These pictures are 10-bit, 12-bit, or 4:4:4.
#[test]
fn lossless_high_depth_matches_ffmpeg_software() {
    let Some(source) = local_video() else {
        eprintln!("skipped: ../local-files has no lossless source");
        return;
    };
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg was not found");
        return;
    };
    let cases = [
        ("1", "yuv444p", PixelFormat::Yuv444p, 8u8),
        ("2", "yuv420p10le", PixelFormat::Yuv420p10, 10),
        ("2", "yuv420p12le", PixelFormat::Yuv420p12, 12),
        ("3", "yuv444p12le", PixelFormat::Yuv444p12, 12),
    ];
    for (profile, pix_fmt, format, depth) in cases {
        let dir = temp_dir(pix_fmt);
        let ivf = dir.join("in.ivf");
        encode_lossless(&ffmpeg, &source, &ivf, profile, pix_fmt, "96:64");
        let stream = read_ivf(&ivf);
        assert_eq!(stream.width, 96, "{pix_fmt} width");
        assert_eq!(stream.height, 64, "{pix_fmt} height");
        assert!(stream.frames.len() > 1, "{pix_fmt} needs an inter frame");
        assert_lossless(&stream, depth, pix_fmt);
        let (ours, _) = decode_shown(&stream, format);
        let oracle = ffmpeg_yuv(&ffmpeg, &ivf, pix_fmt);
        assert_eq!(ours.len(), oracle.len(), "{pix_fmt} shown bytes differ");
        let mismatch = ours.iter().zip(oracle.iter()).position(|(a, b)| a != b);
        assert_eq!(mismatch, None, "{pix_fmt} first mismatch at {mismatch:?}");
        assert!(
            ours.iter().any(|sample| *sample != ours[0]),
            "{pix_fmt} picture is a single sample"
        );
        eprintln!(
            "lossless {pix_fmt} {}x{} {} frames {} bytes",
            stream.width,
            stream.height,
            stream.frames.len(),
            ours.len()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// Sharpness above 4 shifts the loop-filter limit and clamps it.
/// The field is three bits, so rewriting it leaves the rest of the frame aligned.
#[test]
fn sharp_loop_filter_matches_ffmpeg_software() {
    let Some(source) = local_video() else {
        eprintln!("skipped: ../local-files has no loop-filter source");
        return;
    };
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg was not found");
        return;
    };
    let dir = temp_dir("sharpness");
    let ivf = dir.join("in.ivf");
    encode_profile0(
        &ffmpeg,
        &source,
        &ivf,
        &["-frames:v", "4", "-vf", "scale=320:240", "-b:v", "80k"],
    );
    let stream = read_ivf(&ivf);
    assert_eq!(stream.width, 320);
    assert_eq!(stream.height, 240);
    assert!(stream.frames.len() > 1, "sharpness needs an inter frame");
    let rewritten: Vec<Vec<u8>> = stream
        .frames
        .iter()
        .map(|frame| force_sharpness(frame, 7))
        .collect();
    assert!(
        rewritten_sharpness_filters(&rewritten),
        "no frame combined sharpness 7 with a loop-filter level"
    );
    let (plain_pic, _) = decode_shown(&stream, PixelFormat::Yuv420p);
    let sharp_ivf = dir.join("sharp7.ivf");
    write_same_ivf(&ivf, &sharp_ivf, &rewritten);
    let sharp_stream = read_ivf(&sharp_ivf);
    let (sharp_pic, _) = decode_shown(&sharp_stream, PixelFormat::Yuv420p);
    assert_ne!(plain_pic, sharp_pic, "sharpness 7 did not change a sample");
    let oracle = ffmpeg_yuv(&ffmpeg, &sharp_ivf, "yuv420p");
    assert_eq!(sharp_pic.len(), oracle.len());
    let mismatch = sharp_pic
        .iter()
        .zip(oracle.iter())
        .position(|(a, b)| a != b);
    assert_eq!(mismatch, None, "first mismatch at {mismatch:?}");
    eprintln!(
        "sharpness 7 {}x{} {} frames {} bytes",
        sharp_stream.width,
        sharp_stream.height,
        sharp_stream.frames.len(),
        sharp_pic.len()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The same sharpness at 10-bit and 12-bit. The filter thresholds scale with the depth.
/// 4:2:2 uses the chroma walk that 4:2:0 does not.
#[test]
fn sharp_high_depth_matches_ffmpeg_software() {
    let Some(source) = local_video() else {
        eprintln!("skipped: ../local-files has no loop-filter source");
        return;
    };
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg was not found");
        return;
    };
    let cases = [
        ("2", "yuv420p10le", PixelFormat::Yuv420p10),
        ("2", "yuv420p12le", PixelFormat::Yuv420p12),
        ("3", "yuv422p10le", PixelFormat::Yuv422p10),
    ];
    for (profile, pix_fmt, format) in cases {
        let dir = temp_dir(pix_fmt);
        let ivf = dir.join("in.ivf");
        encode_bitrate(&ffmpeg, &source, &ivf, profile, pix_fmt, "160:96", "40k");
        let stream = read_ivf(&ivf);
        assert_eq!(stream.width, 160, "{pix_fmt} width");
        assert_eq!(stream.height, 96, "{pix_fmt} height");
        assert!(stream.frames.len() > 1, "{pix_fmt} needs an inter frame");
        let rewritten: Vec<Vec<u8>> = stream
            .frames
            .iter()
            .map(|frame| force_sharpness(frame, 7))
            .collect();
        assert!(
            rewritten_sharpness_filters(&rewritten),
            "{pix_fmt} never combined sharpness 7 with a loop-filter level"
        );
        let (plain_pic, _) = decode_shown(&stream, format);
        let sharp_ivf = dir.join("sharp7.ivf");
        write_same_ivf(&ivf, &sharp_ivf, &rewritten);
        let sharp_stream = read_ivf(&sharp_ivf);
        let (sharp_pic, _) = decode_shown(&sharp_stream, format);
        assert_ne!(
            plain_pic, sharp_pic,
            "{pix_fmt} sharpness 7 did not change a sample"
        );
        let oracle = ffmpeg_yuv(&ffmpeg, &sharp_ivf, pix_fmt);
        assert_eq!(
            sharp_pic.len(),
            oracle.len(),
            "{pix_fmt} shown bytes differ"
        );
        let mismatch = sharp_pic
            .iter()
            .zip(oracle.iter())
            .position(|(a, b)| a != b);
        assert_eq!(mismatch, None, "{pix_fmt} first mismatch at {mismatch:?}");
        eprintln!(
            "sharpness 7 {pix_fmt} {}x{} {} frames {} bytes",
            sharp_stream.width,
            sharp_stream.height,
            sharp_stream.frames.len(),
            sharp_pic.len()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// Parallel mode keeps the probability context from being adapted for the next frame.
#[test]
fn frame_parallel_matches_ffmpeg_software() {
    let Some(source) = local_video() else {
        eprintln!("skipped: ../local-files has no parallel source");
        return;
    };
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg was not found");
        return;
    };
    let dir = temp_dir("parallel");
    let ivf = dir.join("in.ivf");
    encode_profile0(
        &ffmpeg,
        &source,
        &ivf,
        &[
            "-frames:v",
            "8",
            "-vf",
            "scale=320:240",
            "-frame-parallel",
            "1",
            "-threads",
            "1",
        ],
    );
    let stream = read_ivf(&ivf);
    assert_eq!(stream.width, 320);
    assert_eq!(stream.height, 240);
    let parallel = parallel_inter_frames(&stream);
    assert!(
        parallel >= 2,
        "need two inter frames with parallel mode, found {parallel}"
    );
    let (ours, _) = decode_shown(&stream, PixelFormat::Yuv420p);
    let oracle = ffmpeg_yuv(&ffmpeg, &ivf, "yuv420p");
    assert_eq!(ours.len(), oracle.len());
    let mismatch = ours.iter().zip(oracle.iter()).position(|(a, b)| a != b);
    assert_eq!(mismatch, None, "first mismatch at {mismatch:?}");
    eprintln!(
        "frame parallel {}x{} {} frames {} bytes",
        stream.width,
        stream.height,
        stream.frames.len(),
        ours.len()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

fn parallel_inter_frames(stream: &Ivf) -> usize {
    let mut state = crate::header::Persist::new();
    let mut count = 0usize;
    for (index, frame) in stream.frames.iter().enumerate() {
        let parsed = crate::header::parse(&state, frame)
            .unwrap_or_else(|err| panic!("parallel frame {index} header: {err}"));
        if !parsed.header.key_frame && !parsed.header.intra_only && parsed.header.frame_parallel {
            count += 1;
        }
        remember_header(&mut state, &parsed);
    }
    count
}

/// An intra reference delta of -63 drives the intra filter level to 0.
/// The magnitude and sign are fixed width, so the rest of the frame stays aligned.
#[test]
fn loop_filter_delta_matches_ffmpeg_software() {
    let Some(source) = local_video() else {
        eprintln!("skipped: ../local-files has no loop-filter source");
        return;
    };
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg was not found");
        return;
    };
    let dir = temp_dir("lf-delta");
    let ivf = dir.join("in.ivf");
    encode_profile0(
        &ffmpeg,
        &source,
        &ivf,
        &["-frames:v", "4", "-vf", "scale=320:240", "-b:v", "80k"],
    );
    let stream = read_ivf(&ivf);
    assert_eq!(stream.width, 320);
    assert_eq!(stream.height, 240);
    assert!(stream.frames.len() > 1, "deltas need an inter frame");
    let rewritten: Vec<Vec<u8>> = stream
        .frames
        .iter()
        .map(|frame| force_intra_delta(frame))
        .collect();
    assert!(
        intra_delta_is_extreme(&rewritten),
        "no frame applied an intra loop-filter delta of -63"
    );
    let (plain_pic, _) = decode_shown(&stream, PixelFormat::Yuv420p);
    let delta_ivf = dir.join("delta.ivf");
    write_same_ivf(&ivf, &delta_ivf, &rewritten);
    let delta_stream = read_ivf(&delta_ivf);
    let (delta_pic, _) = decode_shown(&delta_stream, PixelFormat::Yuv420p);
    assert_ne!(
        plain_pic, delta_pic,
        "intra delta -63 did not change a sample"
    );
    let oracle = ffmpeg_yuv(&ffmpeg, &delta_ivf, "yuv420p");
    assert_eq!(delta_pic.len(), oracle.len());
    let mismatch = delta_pic
        .iter()
        .zip(oracle.iter())
        .position(|(a, b)| a != b);
    assert_eq!(mismatch, None, "first mismatch at {mismatch:?}");
    eprintln!(
        "loop filter delta {}x{} {} frames {} bytes",
        delta_stream.width,
        delta_stream.height,
        delta_stream.frames.len(),
        delta_pic.len()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Segment 1 is skip: those blocks have no residual and a zero motion vector.
#[test]
fn segment_skip_matches_ffmpeg_software() {
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg was not found");
        return;
    };
    let Some(flags) = vpx_compile_flags() else {
        eprintln!("skipped: libvpx headers were not found");
        return;
    };
    let dir = temp_dir("seg-skip");
    let ivf = dir.join("in.ivf");
    if !encode_roi_map(&dir, &ivf, &flags, Some("skip")) {
        eprintln!("skipped: the segment-map encoder did not build");
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }
    let stream = read_ivf(&ivf);
    assert_eq!((stream.width, stream.height), (160, 96));
    assert!(
        segment_skip_is_set(&stream),
        "no inter frame marked segment 1 as skip"
    );
    let (ours, _) = decode_shown(&stream, PixelFormat::Yuv420p);
    let oracle = ffmpeg_yuv(&ffmpeg, &ivf, "yuv420p");
    assert_eq!(ours.len(), oracle.len());
    let mismatch = ours.iter().zip(oracle.iter()).position(|(a, b)| a != b);
    assert_eq!(mismatch, None, "first mismatch at {mismatch:?}");
    assert_eq!(ours.len(), 160 * 96 * 3 / 2 * 4);
    eprintln!(
        "segment skip {}x{} {} frames {} bytes",
        stream.width,
        stream.height,
        stream.frames.len(),
        ours.len()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A segment loop-filter step of -63 and a segment reference of the last frame.
#[test]
fn segment_tools_match_ffmpeg_software() {
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg was not found");
        return;
    };
    let Some(flags) = vpx_compile_flags() else {
        eprintln!("skipped: libvpx headers were not found");
        return;
    };
    let dir = temp_dir("seg-tools");
    let ivf = dir.join("in.ivf");
    if !encode_roi_map(&dir, &ivf, &flags, None) {
        eprintln!("skipped: the segment-map encoder did not build");
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }
    let stream = read_ivf(&ivf);
    assert_eq!((stream.width, stream.height), (160, 96));
    assert!(
        segment_tools_are_set(&stream),
        "no inter frame carried the loop-filter step and the reference"
    );
    let (ours, _) = decode_shown(&stream, PixelFormat::Yuv420p);
    let oracle = ffmpeg_yuv(&ffmpeg, &ivf, "yuv420p");
    assert_eq!(ours.len(), oracle.len());
    let mismatch = ours.iter().zip(oracle.iter()).position(|(a, b)| a != b);
    assert_eq!(mismatch, None, "first mismatch at {mismatch:?}");
    assert_eq!(ours.len(), 160 * 96 * 3 / 2 * 4);
    eprintln!(
        "segment tools {}x{} {} frames {} bytes",
        stream.width,
        stream.height,
        stream.frames.len(),
        ours.len()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

fn vpx_compile_flags() -> Option<Vec<String>> {
    let output = Command::new("pkg-config")
        .args(["--cflags", "--libs", "vpx"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let flags: Vec<String> = text.split_whitespace().map(str::to_string).collect();
    (!flags.is_empty()).then_some(flags)
}

fn encode_roi_map(dir: &Path, ivf: &Path, flags: &[String], mode: Option<&str>) -> bool {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/roi_map.c");
    let bin = dir.join("roi_map");
    let Ok(built) = Command::new("cc")
        .arg("-O2")
        .arg(&source)
        .arg("-o")
        .arg(&bin)
        .args(flags)
        .output()
    else {
        return false;
    };
    if !built.status.success() {
        return false;
    }
    let mut command = Command::new(&bin);
    command.arg(ivf);
    if let Some(mode) = mode {
        command.arg(mode);
    }
    let encoded = command.output().expect("roi map");
    assert!(
        encoded.status.success(),
        "roi map {}",
        String::from_utf8_lossy(&encoded.stderr)
    );
    true
}

fn segment_skip_is_set(stream: &Ivf) -> bool {
    let mut state = crate::header::Persist::new();
    let mut found = false;
    for (index, frame) in stream.frames.iter().enumerate() {
        let parsed = crate::header::parse(&state, frame)
            .unwrap_or_else(|err| panic!("segment skip frame {index} header: {err}"));
        let header = &parsed.header;
        let seg = &header.seg;
        found |= !header.key_frame && seg.enabled && seg.feature_on[1][3] && !seg.feature_on[1][1];
        remember_header(&mut state, &parsed);
    }
    found
}

fn segment_tools_are_set(stream: &Ivf) -> bool {
    let mut state = crate::header::Persist::new();
    let mut found = false;
    for (index, frame) in stream.frames.iter().enumerate() {
        let parsed = crate::header::parse(&state, frame)
            .unwrap_or_else(|err| panic!("segment tool frame {index} header: {err}"));
        let header = &parsed.header;
        let seg = &header.seg;
        found |= !header.key_frame
            && header.filter_level > 0
            && seg.enabled
            && !seg.abs_delta
            && seg.feature_on[1][1]
            && seg.feature_data[1][1] == -63
            && seg.feature_on[2][2]
            && seg.feature_data[2][2] == 1;
        remember_header(&mut state, &parsed);
    }
    found
}

/// A hidden alternate reference is coded with context 1, and later frames
/// mark that reference as backward.
#[test]
fn altref_context_matches_ffmpeg_software() {
    let Some(source) = local_video() else {
        eprintln!("skipped: ../local-files has no alternate-reference source");
        return;
    };
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg was not found");
        return;
    };
    let dir = temp_dir("altref-ctx");
    let ivf = dir.join("in.ivf");
    encode_two_pass(&ffmpeg, &source, &dir, &ivf);
    let stream = read_ivf(&ivf);
    assert_eq!((stream.width, stream.height), (160, 96));
    let (context, backward) = altref_marks(&stream);
    assert!(context, "no hidden frame used context 1");
    assert!(
        backward,
        "no shown frame marked the alternate reference backward"
    );
    let (ours, _) = decode_shown(&stream, PixelFormat::Yuv420p);
    let oracle = ffmpeg_yuv(&ffmpeg, &ivf, "yuv420p");
    assert_eq!(ours.len(), oracle.len());
    let mismatch = ours.iter().zip(oracle.iter()).position(|(a, b)| a != b);
    assert_eq!(mismatch, None, "first mismatch at {mismatch:?}");
    assert_eq!(ours.len(), 160 * 96 * 3 / 2 * 16);
    eprintln!(
        "alternate reference context {}x{} {} frames {} bytes",
        stream.width,
        stream.height,
        stream.frames.len(),
        ours.len()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

fn encode_two_pass(ffmpeg: &Path, source: &Path, dir: &Path, ivf: &Path) {
    let common = [
        "-frames:v",
        "16",
        "-vf",
        "scale=160:96",
        "-deadline",
        "good",
        "-cpu-used",
        "1",
        "-lag-in-frames",
        "16",
        "-auto-alt-ref",
        "1",
        "-passlogfile",
        "pass",
    ];
    for (pass, output) in [("1", None), ("2", Some(ivf))] {
        let mut command = Command::new(ffmpeg);
        command
            .current_dir(dir)
            .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
            .arg(source)
            .args(["-an", "-c:v", "libvpx-vp9", "-b:v", "200k"])
            .args(common)
            .args(["-pass", pass]);
        if let Some(ivf) = output {
            command.args(["-f", "ivf"]).arg(ivf);
        } else {
            command.args(["-f", "null", "-"]);
        }
        let status = command.output().expect("two-pass encode");
        assert!(
            status.status.success(),
            "pass {pass} {}",
            String::from_utf8_lossy(&status.stderr)
        );
    }
}

fn altref_marks(stream: &Ivf) -> (bool, bool) {
    let mut state = crate::header::Persist::new();
    let mut context = false;
    let mut backward = false;
    for (index, frame) in stream.frames.iter().enumerate() {
        let parsed = crate::header::parse(&state, frame)
            .unwrap_or_else(|err| panic!("alt-ref frame {index} header: {err}"));
        let header = &parsed.header;
        context |= !header.show_frame && header.context_idx == 1;
        backward |= header.show_frame && !header.key_frame && header.sign_bias[3] == 1;
        remember_header(&mut state, &parsed);
    }
    (context, backward)
}

/// A relative segment quantizer step above 63 is added to the base index and clamped.
#[test]
fn segment_quant_delta_matches_ffmpeg_software() {
    let Some(source) = local_video() else {
        eprintln!("skipped: ../local-files has no segmentation source");
        return;
    };
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg was not found");
        return;
    };
    let dir = temp_dir("seg-q");
    let ivf = dir.join("in.ivf");
    encode_profile0(
        &ffmpeg,
        &source,
        &ivf,
        &[
            "-frames:v",
            "6",
            "-vf",
            "scale=320:240",
            "-aq-mode",
            "4",
            "-deadline",
            "good",
            "-cpu-used",
            "2",
        ],
    );
    let stream = read_ivf(&ivf);
    assert_eq!(stream.width, 320);
    assert_eq!(stream.height, 240);
    assert!(
        stream.frames.len() > 1,
        "segment quant needs an inter frame"
    );
    assert!(
        large_relative_quant(&stream),
        "no relative segment quantizer step above 63"
    );
    let (ours, _) = decode_shown(&stream, PixelFormat::Yuv420p);
    let oracle = ffmpeg_yuv(&ffmpeg, &ivf, "yuv420p");
    assert_eq!(ours.len(), oracle.len());
    let mismatch = ours.iter().zip(oracle.iter()).position(|(a, b)| a != b);
    assert_eq!(mismatch, None, "first mismatch at {mismatch:?}");
    eprintln!(
        "segment quant {}x{} {} frames {} bytes",
        stream.width,
        stream.height,
        stream.frames.len(),
        ours.len()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

fn large_relative_quant(stream: &Ivf) -> bool {
    let mut state = crate::header::Persist::new();
    let mut found = false;
    for (index, frame) in stream.frames.iter().enumerate() {
        let parsed = crate::header::parse(&state, frame)
            .unwrap_or_else(|err| panic!("segment frame {index} header: {err}"));
        let seg = &parsed.header.seg;
        if seg.enabled && !seg.abs_delta {
            found |= seg
                .feature_data
                .iter()
                .enumerate()
                .any(|(id, data)| seg.feature_on[id][0] && data[0] > 63_i16);
        }
        remember_header(&mut state, &parsed);
    }
    found
}

fn intra_delta_is_extreme(frames: &[Vec<u8>]) -> bool {
    let mut state = crate::header::Persist::new();
    let mut active = false;
    for (index, frame) in frames.iter().enumerate() {
        let parsed = crate::header::parse(&state, frame)
            .unwrap_or_else(|err| panic!("delta frame {index} header: {err}"));
        active |= parsed.header.ref_deltas[0] == -63 && parsed.header.filter_level > 0;
        remember_header(&mut state, &parsed);
    }
    active
}

#[derive(Clone, Copy)]
struct DeltaBits {
    magnitude: usize,
    sign: usize,
}

fn force_intra_delta(frame: &[u8]) -> Vec<u8> {
    let Some(site) = intra_delta_bits(frame) else {
        return frame.to_vec();
    };
    let mut out = frame.to_vec();
    for shift in 0..6 {
        set_msb_bit(&mut out, site.magnitude + shift, true);
    }
    set_msb_bit(&mut out, site.sign, true);
    out
}

fn intra_delta_bits(frame: &[u8]) -> Option<DeltaBits> {
    let sharp = sharpness_bit(frame)?;
    let mut bits = crate::bits::BitReader::new(frame);
    skip_bits(&mut bits, sharp.saturating_add(3))?;
    if bits.read(1).ok()? == 0 || bits.read(1).ok()? == 0 || bits.read(1).ok()? == 0 {
        return None;
    }
    let magnitude = bits.position();
    bits.read(6).ok()?;
    let sign = bits.position();
    Some(DeltaBits { magnitude, sign })
}

fn skip_bits(bits: &mut crate::bits::BitReader<'_>, mut count: usize) -> Option<()> {
    while count > 0 {
        let take = count.min(32);
        bits.read(u32::try_from(take).ok()?).ok()?;
        count -= take;
    }
    Some(())
}

fn rewritten_sharpness_filters(frames: &[Vec<u8>]) -> bool {
    let mut state = crate::header::Persist::new();
    let mut active = false;
    for (index, frame) in frames.iter().enumerate() {
        let parsed = crate::header::parse(&state, frame)
            .unwrap_or_else(|err| panic!("sharpness frame {index} header: {err}"));
        assert_eq!(parsed.header.sharpness, 7, "frame {index} sharpness");
        active |= parsed.header.filter_level > 0;
        remember_header(&mut state, &parsed);
    }
    active
}

fn force_sharpness(frame: &[u8], sharpness: u8) -> Vec<u8> {
    let Some(at) = sharpness_bit(frame) else {
        return frame.to_vec();
    };
    let mut out = frame.to_vec();
    for shift in 0..3 {
        let one = (u32::from(sharpness) >> (2 - shift)) & 1 == 1;
        set_msb_bit(&mut out, at + shift, one);
    }
    out
}

fn sharpness_bit(frame: &[u8]) -> Option<usize> {
    let mut bits = crate::bits::BitReader::new(frame);
    if bits.read(2).ok()? != 2 {
        return None;
    }
    let mut profile = bits.read(1).ok()?;
    profile |= bits.read(1).ok()? << 1;
    if profile > 2 {
        bits.read(1).ok()?;
    }
    if bits.read(1).ok()? == 1 {
        return None;
    }
    let key_frame = bits.read(1).ok()? == 0;
    let show_frame = bits.read(1).ok()? == 1;
    let error_resilient = bits.read(1).ok()? == 1;
    if key_frame {
        bits.read(24).ok()?;
        if profile >= 2 {
            bits.read(1).ok()?;
        }
        let space = bits.read(3).ok()?;
        if space == 7 {
            if profile & 1 == 0 {
                return None;
            }
            bits.read(1).ok()?;
        } else {
            bits.read(1).ok()?;
            if profile & 1 == 1 {
                bits.read(1).ok()?;
                bits.read(1).ok()?;
                bits.read(1).ok()?;
            }
        }
        bits.read(16).ok()?;
        bits.read(16).ok()?;
        if bits.read(1).ok()? == 1 {
            bits.read(16).ok()?;
            bits.read(16).ok()?;
        }
    } else {
        if !show_frame && bits.read(1).ok()? == 1 {
            return None;
        }
        if !error_resilient {
            bits.read(2).ok()?;
        }
        bits.read(8).ok()?;
        for _ in 0..3 {
            bits.read(3).ok()?;
            bits.read(1).ok()?;
        }
        let mut found = false;
        for _ in 0..3 {
            if bits.read(1).ok()? == 1 {
                found = true;
                break;
            }
        }
        if !found {
            bits.read(16).ok()?;
            bits.read(16).ok()?;
        }
        if bits.read(1).ok()? == 1 {
            bits.read(16).ok()?;
            bits.read(16).ok()?;
        }
        bits.read(1).ok()?;
        if bits.read(1).ok()? == 0 {
            bits.read(2).ok()?;
        }
    }
    if !error_resilient {
        bits.read(1).ok()?;
        bits.read(1).ok()?;
    }
    bits.read(2).ok()?;
    bits.read(6).ok()?;
    Some(bits.position())
}

fn assert_lossless(stream: &Ivf, depth: u8, pix_fmt: &str) {
    let mut state = crate::header::Persist::new();
    for (index, frame) in stream.frames.iter().enumerate() {
        let parsed = crate::header::parse(&state, frame).unwrap_or_else(|err| {
            panic!("{pix_fmt} frame {index} header: {err}");
        });
        assert!(
            parsed.header.lossless,
            "{pix_fmt} frame {index} is not lossless"
        );
        assert_eq!(parsed.header.bit_depth, depth, "{pix_fmt} bit depth");
        remember_header(&mut state, &parsed);
    }
}

fn remember_header(state: &mut crate::header::Persist, parsed: &crate::header::Parsed<'_>) {
    state.contexts = parsed.contexts.clone();
    if parsed.header.refresh_context {
        let index = usize::from(parsed.header.context_idx);
        if let Some(slot) = state.contexts.get_mut(index) {
            *slot = parsed.fc.clone();
        }
    }
    state.key_frame = parsed.header.key_frame;
    state.intra_only = parsed.header.intra_only;
    state.width = parsed.header.width;
    state.height = parsed.header.height;
    state.last_show_frame = parsed.header.show_frame;
    state.last_width = parsed.header.width;
    state.last_height = parsed.header.height;
    state.need_resync = parsed.header.need_resync;
    state.seg = parsed.header.seg.clone();
    state.ref_deltas = parsed.header.ref_deltas;
    state.mode_deltas = parsed.header.mode_deltas;
    state.sign_bias = parsed.header.sign_bias;
    state.ss_x = parsed.header.ss_x;
    state.ss_y = parsed.header.ss_y;
    state.bit_depth = parsed.header.bit_depth;
    state.color_space = parsed.header.color_space;
    state.color_range = parsed.header.color_range;
    for slot in 0..state.ref_ok.len() {
        if parsed.header.refresh_flags & (1 << slot) != 0 {
            state.ref_ok[slot] = true;
            state.ref_w[slot] = parsed.header.width;
            state.ref_h[slot] = parsed.header.height;
        }
    }
}

/// 52 is not a multiple of 8, so the loop filter runs on a partial superblock.
/// 4:2:2, 4:4:0, and 10-bit change the chroma and sample width of that edge.
#[test]
fn odd_chroma_matches_ffmpeg_software() {
    let Some(source) = local_video() else {
        eprintln!("skipped: ../local-files has no odd-size source");
        return;
    };
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg was not found");
        return;
    };
    let cases = [
        ("1", "yuv422p", PixelFormat::Yuv422p),
        ("1", "yuv440p", PixelFormat::Yuv440p),
        ("1", "yuv444p", PixelFormat::Yuv444p),
        ("2", "yuv420p10le", PixelFormat::Yuv420p10),
        ("2", "yuv420p12le", PixelFormat::Yuv420p12),
        ("3", "yuv422p10le", PixelFormat::Yuv422p10),
    ];
    for (profile, pix_fmt, format) in cases {
        let dir = temp_dir(pix_fmt);
        let ivf = dir.join("in.ivf");
        encode_scaled(&ffmpeg, &source, &ivf, profile, pix_fmt, "52:52");
        let stream = read_ivf(&ivf);
        assert_eq!(stream.width, 52, "{pix_fmt} width");
        assert_eq!(stream.height, 52, "{pix_fmt} height");
        assert!(stream.frames.len() > 1, "{pix_fmt} needs an inter frame");
        let (ours, _) = decode_shown(&stream, format);
        let oracle = ffmpeg_yuv(&ffmpeg, &ivf, pix_fmt);
        assert_eq!(ours.len(), oracle.len(), "{pix_fmt} shown bytes differ");
        let mismatch = ours.iter().zip(oracle.iter()).position(|(a, b)| a != b);
        assert_eq!(mismatch, None, "{pix_fmt} first mismatch at {mismatch:?}");
        eprintln!(
            "odd {pix_fmt} {}x{} {} frames {} bytes",
            stream.width,
            stream.height,
            stream.frames.len(),
            ours.len()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// 53 is odd, so a subsampled chroma plane is 27 samples on that axis.
#[test]
fn odd_edge_chroma_matches_ffmpeg_software() {
    let Some(source) = local_video() else {
        eprintln!("skipped: ../local-files has no odd-size source");
        return;
    };
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg was not found");
        return;
    };
    let cases = [
        ("0", "yuv420p", PixelFormat::Yuv420p),
        ("1", "yuv422p", PixelFormat::Yuv422p),
        ("1", "yuv440p", PixelFormat::Yuv440p),
        ("1", "yuv444p", PixelFormat::Yuv444p),
        ("2", "yuv420p10le", PixelFormat::Yuv420p10),
        ("2", "yuv420p12le", PixelFormat::Yuv420p12),
        ("3", "yuv422p10le", PixelFormat::Yuv422p10),
        ("3", "yuv422p12le", PixelFormat::Yuv422p12),
    ];
    for (profile, pix_fmt, format) in cases {
        let dir = temp_dir(pix_fmt);
        let ivf = dir.join("in.ivf");
        encode_scaled(&ffmpeg, &source, &ivf, profile, pix_fmt, "53:53");
        let stream = read_ivf(&ivf);
        assert_eq!(stream.width, 53, "{pix_fmt} width");
        assert_eq!(stream.height, 53, "{pix_fmt} height");
        assert!(stream.frames.len() > 1, "{pix_fmt} needs an inter frame");
        let (ours, _) = decode_shown(&stream, format);
        let oracle = ffmpeg_yuv(&ffmpeg, &ivf, pix_fmt);
        assert_eq!(ours.len(), oracle.len(), "{pix_fmt} shown bytes differ");
        let mismatch = ours.iter().zip(oracle.iter()).position(|(a, b)| a != b);
        assert_eq!(mismatch, None, "{pix_fmt} first mismatch at {mismatch:?}");
        eprintln!(
            "odd edge {pix_fmt} {}x{} {} frames {} bytes",
            stream.width,
            stream.height,
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

/// Four tile rows is the largest row split the header can name.
#[test]
fn four_tile_rows_match_ffmpeg_software() {
    let Some(source) = local_video() else {
        eprintln!("skipped: ../local-files has no tile source");
        return;
    };
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg was not found");
        return;
    };
    let dir = temp_dir("tile-rows");
    let ivf = dir.join("in.ivf");
    encode_profile0(
        &ffmpeg,
        &source,
        &ivf,
        &[
            "-frames:v",
            "4",
            "-vf",
            "scale=256:512",
            "-tile-columns",
            "0",
            "-tile-rows",
            "2",
            "-threads",
            "1",
        ],
    );
    let stream = read_ivf(&ivf);
    assert!(stream.frames.len() > 1, "tile rows need an inter frame");
    let parsed = crate::header::parse(&crate::header::Persist::new(), &stream.frames[0])
        .expect("keyframe header");
    assert_eq!(parsed.header.log2_tile_rows, 2, "tile rows");
    let (ours, _) = decode_shown(&stream, PixelFormat::Yuv420p);
    let oracle = ffmpeg_yuv(&ffmpeg, &ivf, "yuv420p");
    assert_eq!(ours.len(), oracle.len());
    let mismatch = ours.iter().zip(oracle.iter()).position(|(a, b)| a != b);
    assert_eq!(mismatch, None, "first mismatch at {mismatch:?}");
    eprintln!(
        "tiles 1x4 {}x{} {} frames {} bytes",
        stream.width,
        stream.height,
        stream.frames.len(),
        ours.len()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Four columns and four rows is the largest grid one frame header can name.
#[test]
fn four_by_four_tiles_match_ffmpeg_software() {
    let Some(source) = local_video() else {
        eprintln!("skipped: ../local-files has no tile source");
        return;
    };
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg was not found");
        return;
    };
    let dir = temp_dir("tile-grid");
    let ivf = dir.join("in.ivf");
    encode_profile0(
        &ffmpeg,
        &source,
        &ivf,
        &[
            "-frames:v",
            "4",
            "-vf",
            "scale=1024:512",
            "-tile-columns",
            "2",
            "-tile-rows",
            "2",
            "-threads",
            "1",
        ],
    );
    let stream = read_ivf(&ivf);
    assert_eq!(stream.width, 1024);
    assert_eq!(stream.height, 512);
    assert!(
        stream.frames.len() > 1,
        "the tile grid needs an inter frame"
    );
    let parsed = crate::header::parse(&crate::header::Persist::new(), &stream.frames[0])
        .expect("keyframe header");
    assert_eq!(parsed.header.log2_tile_cols, 2, "tile columns");
    assert_eq!(parsed.header.log2_tile_rows, 2, "tile rows");
    let (ours, _) = decode_shown(&stream, PixelFormat::Yuv420p);
    let oracle = ffmpeg_yuv(&ffmpeg, &ivf, "yuv420p");
    assert_eq!(ours.len(), oracle.len());
    let mismatch = ours.iter().zip(oracle.iter()).position(|(a, b)| a != b);
    assert_eq!(mismatch, None, "first mismatch at {mismatch:?}");
    eprintln!(
        "tiles 4x4 {}x{} {} frames {} bytes",
        stream.width,
        stream.height,
        stream.frames.len(),
        ours.len()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The literal filter values are smooth, regular, sharp, and bilinear.
/// A fixed filter is two bits, so rewriting it leaves the rest of the frame aligned.
#[test]
fn interpolation_filters_match_ffmpeg_software() {
    let Some(source) = local_video() else {
        eprintln!("skipped: ../local-files has no filter source");
        return;
    };
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg was not found");
        return;
    };
    let dir = temp_dir("interp");
    let ivf = dir.join("in.ivf");
    encode_profile0(
        &ffmpeg,
        &source,
        &ivf,
        &[
            "-frames:v",
            "4",
            "-vf",
            "scale=320:240",
            "-lag-in-frames",
            "0",
            "-auto-alt-ref",
            "0",
            "-g",
            "4",
            "-threads",
            "1",
        ],
    );
    let stream = read_ivf(&ivf);
    assert!(stream.frames.len() > 1, "filters need an inter frame");
    let switchable = stream
        .frames
        .iter()
        .filter(|frame| locate_interp(frame).is_some_and(|site| site.switchable))
        .count();
    assert!(switchable > 0, "the source frames are not switchable");
    let (ours, _) = decode_shown(&stream, PixelFormat::Yuv420p);
    let oracle = ffmpeg_yuv(&ffmpeg, &ivf, "yuv420p");
    assert_eq!(ours.len(), oracle.len(), "switchable shown bytes differ");
    let mismatch = ours.iter().zip(oracle.iter()).position(|(a, b)| a != b);
    assert_eq!(mismatch, None, "switchable first mismatch at {mismatch:?}");
    eprintln!(
        "interp switchable {} frames {} bytes",
        stream.frames.len(),
        ours.len()
    );
    let fixed_ivf = dir.join("fixed.ivf");
    encode_profile0(
        &ffmpeg,
        &source,
        &fixed_ivf,
        &[
            "-frames:v",
            "4",
            "-vf",
            "scale=320:240",
            "-lag-in-frames",
            "0",
            "-auto-alt-ref",
            "0",
            "-g",
            "4",
            "-cpu-used",
            "0",
            "-threads",
            "1",
        ],
    );
    let stream = read_ivf(&fixed_ivf);
    let fixed = stream
        .frames
        .iter()
        .filter(|frame| fixed_interp_bit(frame).is_some())
        .count();
    assert!(
        fixed > 0,
        "cpu-used 0 did not emit a fixed interpolation filter"
    );
    let mut packed = Vec::new();
    for (literal, name) in [
        (0u32, "smooth"),
        (1, "regular"),
        (2, "sharp"),
        (3, "bilinear"),
    ] {
        let frames = stream
            .frames
            .iter()
            .map(|frame| force_interp(frame, literal))
            .collect::<Vec<_>>();
        let rewritten = dir.join(format!("{name}.ivf"));
        write_same_ivf(&fixed_ivf, &rewritten, &frames);
        let shown = read_ivf(&rewritten);
        let (ours, _) = decode_shown(&shown, PixelFormat::Yuv420p);
        let oracle = ffmpeg_yuv(&ffmpeg, &rewritten, "yuv420p");
        assert_eq!(ours.len(), oracle.len(), "{name} shown bytes differ");
        let mismatch = ours.iter().zip(oracle.iter()).position(|(a, b)| a != b);
        assert_eq!(mismatch, None, "{name} first mismatch at {mismatch:?}");
        eprintln!(
            "interp {name} {} frames {} bytes",
            shown.frames.len(),
            ours.len()
        );
        packed.push(ours);
    }
    assert!(
        packed.windows(2).any(|pair| pair[0] != pair[1]),
        "interpolation filters did not change a sample"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

fn force_interp(frame: &[u8], literal: u32) -> Vec<u8> {
    let Some(at) = fixed_interp_bit(frame) else {
        return frame.to_vec();
    };
    let mut out = frame.to_vec();
    set_msb_bit(&mut out, at + 1, (literal >> 1) & 1 == 1);
    set_msb_bit(&mut out, at + 2, literal & 1 == 1);
    let read = (u32::from(bit_at(&out, at + 1)) << 1) | u32::from(bit_at(&out, at + 2));
    assert_eq!(read, literal, "filter literal");
    out
}

fn fixed_interp_bit(frame: &[u8]) -> Option<usize> {
    let site = locate_interp(frame)?;
    if site.switchable { None } else { Some(site.at) }
}

struct InterpSite {
    at: usize,
    switchable: bool,
}

fn locate_interp(frame: &[u8]) -> Option<InterpSite> {
    let mut bits = crate::bits::BitReader::new(frame);
    if bits.read(2).ok()? != 2 {
        return None;
    }
    let mut profile = bits.read(1).ok()?;
    profile |= bits.read(1).ok()? << 1;
    if profile > 2 {
        bits.read(1).ok()?;
    }
    let show_existing = bits.read(1).ok()? == 1;
    let key_frame = bits.read(1).ok()? == 0;
    if show_existing || key_frame {
        return None;
    }
    let show_frame = bits.read(1).ok()? == 1;
    let error_resilient = bits.read(1).ok()? == 1;
    if !show_frame && bits.read(1).ok()? == 1 {
        return None;
    }
    if !error_resilient {
        bits.read(2).ok()?;
    }
    bits.read(8).ok()?;
    for _ in 0..3 {
        bits.read(3).ok()?;
        bits.read(1).ok()?;
    }
    let mut found = false;
    for _ in 0..3 {
        if bits.read(1).ok()? == 1 {
            found = true;
            break;
        }
    }
    if !found {
        bits.read(16).ok()?;
        bits.read(16).ok()?;
    }
    if bits.read(1).ok()? == 1 {
        bits.read(16).ok()?;
        bits.read(16).ok()?;
    }
    bits.read(1).ok()?;
    let at = bits.position();
    let switchable = bits.read(1).ok()? == 1;
    Some(InterpSite { at, switchable })
}

fn bit_at(data: &[u8], index: usize) -> bool {
    let shift = 7 - (index % 8);
    data[index / 8] & (1 << shift) != 0
}

fn set_msb_bit(data: &mut [u8], index: usize, one: bool) {
    let shift = 7 - (index % 8);
    if one {
        data[index / 8] |= 1 << shift;
    } else {
        data[index / 8] &= !(1 << shift);
    }
}

fn write_same_ivf(src: &Path, dst: &Path, frames: &[Vec<u8>]) {
    let data = std::fs::read(src).expect("ivf");
    let header = u16::from_le_bytes([data[6], data[7]]) as usize;
    let mut out = data[..header].to_vec();
    out[24..28].copy_from_slice(&u32::try_from(frames.len()).unwrap_or(0).to_le_bytes());
    for (index, frame) in frames.iter().enumerate() {
        let size = u32::try_from(frame.len()).unwrap_or(0);
        out.extend_from_slice(&size.to_le_bytes());
        out.extend_from_slice(&(u64::try_from(index).unwrap_or(0)).to_le_bytes());
        out.extend_from_slice(frame);
    }
    std::fs::write(dst, out).expect("ivf");
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

fn encode_scaled(
    ffmpeg: &Path,
    source: &Path,
    ivf: &Path,
    profile: &str,
    pix_fmt: &str,
    scale: &str,
) {
    let vf = format!("scale={scale}");
    let status = Command::new(ffmpeg)
        .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(source)
        .args([
            "-an",
            "-frames:v",
            "4",
            "-vf",
            &vf,
            "-c:v",
            "libvpx-vp9",
            "-profile:v",
            profile,
            "-pix_fmt",
            pix_fmt,
            "-b:v",
            "200k",
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

fn encode_bitrate(
    ffmpeg: &Path,
    source: &Path,
    ivf: &Path,
    profile: &str,
    pix_fmt: &str,
    scale: &str,
    bitrate: &str,
) {
    let vf = format!("scale={scale}");
    let status = Command::new(ffmpeg)
        .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(source)
        .args([
            "-an",
            "-frames:v",
            "4",
            "-vf",
            &vf,
            "-c:v",
            "libvpx-vp9",
            "-profile:v",
            profile,
            "-pix_fmt",
            pix_fmt,
            "-b:v",
            bitrate,
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

fn encode_lossless(
    ffmpeg: &Path,
    source: &Path,
    ivf: &Path,
    profile: &str,
    pix_fmt: &str,
    scale: &str,
) {
    let vf = format!("scale={scale}");
    let status = Command::new(ffmpeg)
        .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(source)
        .args([
            "-an",
            "-frames:v",
            "4",
            "-vf",
            &vf,
            "-c:v",
            "libvpx-vp9",
            "-profile:v",
            profile,
            "-pix_fmt",
            pix_fmt,
            "-lossless",
            "1",
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
