//! Streams from libvpx, decoded here and by two independent decoders.
//!
//! The sources come from `../local-files`, which CI does not have; the
//! tests skip without it. The oracles are ffmpeg's native `vp8` decoder
//! and libvpx (`-c:v libvpx`), both single-threaded.

use std::path::{Path, PathBuf};
use std::process::Command;

use cutlass_decode::video::VideoDecoder;
use cutlass_decode::{CodecId, CodecParams, Packet};

use crate::Vp8Decoder;
use crate::header::{self, Persist};

pub(crate) struct Ivf {
    pub width: u32,
    pub height: u32,
    pub frames: Vec<Vec<u8>>,
}

pub(crate) fn parse_ivf(data: &[u8]) -> Ivf {
    assert!(data.len() >= 32, "ivf header");
    assert_eq!(&data[..4], b"DKIF");
    let header = usize::from(u16::from_le_bytes([data[6], data[7]]));
    let width = u32::from(u16::from_le_bytes([data[12], data[13]]));
    let height = u32::from(u16::from_le_bytes([data[14], data[15]]));
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

pub(crate) fn write_ivf(path: &Path, width: u32, height: u32, frames: &[Vec<u8>]) {
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
    std::fs::write(path, out).expect("write ivf");
}

pub(crate) fn ffmpeg_bin() -> Option<PathBuf> {
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
            .is_ok_and(|output| output.status.success())
    })
}

pub(crate) fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "cutlass-vp8-{name}-{}-{}",
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

pub(crate) fn local_root() -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files");
    root.is_dir().then_some(root)
}

pub(crate) fn local_video() -> Option<PathBuf> {
    let root = local_root()?;
    [
        "snaptik_7282700172428086530_v3.mp4",
        "IMG_8146.MP4",
        "7150172-hd_720_1280_50fps.mp4",
        "16929689_1920_1080_30fps.mp4",
    ]
    .iter()
    .map(|name| root.join(name))
    .find(|path| path.is_file())
}

/// Decodes a raw yuv420p stream with one of ffmpeg's decoders.
pub(crate) fn ffmpeg_yuv(ffmpeg: &Path, ivf: &Path, decoder: &str) -> Vec<u8> {
    let output = ivf.with_extension(format!("{decoder}.yuv"));
    let status = Command::new(ffmpeg)
        .args([
            "-y",
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
        .args([
            "-fps_mode",
            "passthrough",
            "-noautoscale",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&output)
        .output()
        .expect("ffmpeg");
    assert!(
        status.status.success(),
        "{decoder} {}",
        String::from_utf8_lossy(&status.stderr)
    );
    std::fs::read(&output).expect("yuv")
}

/// Encodes `frames` frames of the source with libvpx.
pub(crate) fn libvpx_encode(
    ffmpeg: &Path,
    source: &Path,
    ivf: &Path,
    vf: &str,
    frames: u32,
    extra: &[&str],
) {
    let mut command = Command::new(ffmpeg);
    command.args(["-y", "-hide_banner", "-loglevel", "error", "-i"]);
    command.arg(source);
    let count = frames.to_string();
    command.args([
        "-an",
        "-vf",
        vf,
        "-frames:v",
        &count,
        "-c:v",
        "libvpx",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "1",
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

/// Two-pass libvpx, the mode that makes hidden alt-ref frames.
pub(crate) fn libvpx_two_pass(
    ffmpeg: &Path,
    source: &Path,
    ivf: &Path,
    vf: &str,
    frames: u32,
    extra: &[&str],
) {
    let log = ivf.with_extension("log");
    let count = frames.to_string();
    for pass in ["1", "2"] {
        let mut command = Command::new(ffmpeg);
        command.args(["-y", "-hide_banner", "-loglevel", "error", "-i"]);
        command.arg(source);
        command.args([
            "-an",
            "-vf",
            vf,
            "-frames:v",
            &count,
            "-c:v",
            "libvpx",
            "-pix_fmt",
            "yuv420p",
        ]);
        command.args(["-threads", "1", "-pass", pass, "-passlogfile"]);
        command.arg(&log);
        command.args(extra);
        if pass == "1" {
            command.args(["-f", "null", "-"]);
        } else {
            command.args(["-f", "ivf"]);
            command.arg(ivf);
        }
        let status = command.output().expect("encode");
        assert!(
            status.status.success(),
            "pass {pass} {}",
            String::from_utf8_lossy(&status.stderr)
        );
    }
}

/// Decodes every frame and returns the shown pictures packed as yuv420p.
pub(crate) fn decode_shown(stream: &Ivf) -> Vec<u8> {
    decode_counted(stream).0
}

/// Like [`decode_shown`], with macroblocks per luma mode.
pub(crate) fn decode_counted(stream: &Ivf) -> (Vec<u8>, [u64; 10]) {
    let params = CodecParams::video(CodecId::Vp8, stream.width, stream.height);
    let mut decoder = Vp8Decoder::open(&params).expect("open");
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
    (packed, decoder.mode_counts())
}

/// Header features a stream uses, to show what a case covered.
#[derive(Debug, Default)]
pub(crate) struct Coverage {
    pub frames: usize,
    pub hidden: usize,
    pub keys: usize,
    pub segmented: usize,
    pub lf_deltas: usize,
    pub partitions: usize,
    pub golden_refresh: usize,
    pub altref_refresh: usize,
    pub copies: usize,
    pub sign_bias: usize,
    pub no_entropy_refresh: usize,
    pub simple_filter: usize,
    pub versions: [usize; 4],
    /// Macroblocks by luma mode: DC, V, H, TM, B, NEAREST, NEAR, ZERO, NEW, SPLIT.
    pub modes: [u64; 10],
}

pub(crate) fn coverage(stream: &Ivf) -> Coverage {
    let mut c = Coverage::default();
    let mut state = Persist::default();
    for frame in &stream.frames {
        let Ok(parsed) = header::parse(frame, &state) else {
            continue;
        };
        let h = &parsed.header;
        c.frames += 1;
        c.hidden += usize::from(!h.tag.show);
        c.keys += usize::from(h.tag.key);
        c.segmented += usize::from(h.segmentation.enabled);
        c.lf_deltas += usize::from(h.deltas.enabled);
        c.partitions = c.partitions.max(h.partitions);
        c.golden_refresh += usize::from(!h.tag.key && h.refresh_golden);
        c.altref_refresh += usize::from(!h.tag.key && h.refresh_altref);
        c.copies += usize::from(h.copy_to_golden != 0 || h.copy_to_altref != 0);
        c.sign_bias += usize::from(h.sign_bias_golden || h.sign_bias_altref);
        c.no_entropy_refresh += usize::from(!h.refresh_entropy);
        c.simple_filter += usize::from(h.filter_simple);
        c.versions[usize::from(h.tag.version & 3)] += 1;
        state = parsed.next;
    }
    c
}

/// Describes where two yuv420p streams first differ.
pub(crate) fn first_difference(a: &[u8], b: &[u8], width: u32, height: u32) -> Option<String> {
    let (w, h) = (width as usize, height as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let frame = w * h + 2 * cw * ch;
    let at = a.iter().zip(b.iter()).position(|(x, y)| x != y)?;
    let index = at / frame;
    let mut offset = at % frame;
    let (plane, pw) = if offset < w * h {
        ("y", w)
    } else if offset < w * h + cw * ch {
        offset -= w * h;
        ("u", cw)
    } else {
        offset -= w * h + cw * ch;
        ("v", cw)
    };
    let (x, y) = (offset % pw, offset / pw);
    let scale = if plane == "y" { 16 } else { 8 };
    Some(format!(
        "frame {index} plane {plane} x {x} y {y} (mb {} {}) ours {} theirs {}",
        x / scale,
        y / scale,
        a[at],
        b[at]
    ))
}

/// A source without scene cuts, where libvpx places alt-ref frames.
pub(crate) fn local_smooth() -> Option<PathBuf> {
    let root = local_root()?;
    [
        "IMG_8146.MP4",
        "16929689_1920_1080_30fps.mp4",
        "7150172-hd_720_1280_50fps.mp4",
    ]
    .iter()
    .map(|name| root.join(name))
    .find(|path| path.is_file())
}

/// Encodes with libvpx, decodes here and with both oracles, and compares.
/// A name starting with `smooth` uses [`local_smooth`].
pub(crate) fn check_case(name: &str, vf: &str, frames: u32, extra: &[&str]) -> Option<Coverage> {
    let source = if name.starts_with("smooth") {
        local_smooth()?
    } else {
        local_video()?
    };
    let ffmpeg = ffmpeg_bin()?;
    let dir = temp_dir(name);
    let ivf = dir.join("in.ivf");
    if extra.first() == Some(&"two-pass") {
        libvpx_two_pass(&ffmpeg, &source, &ivf, vf, frames, &extra[1..]);
    } else {
        libvpx_encode(&ffmpeg, &source, &ivf, vf, frames, extra);
    }
    let stream = parse_ivf(&std::fs::read(&ivf).expect("ivf"));
    let mut cover = coverage(&stream);
    let (ours, modes) = decode_counted(&stream);
    cover.modes = modes;
    for decoder in ["vp8", "libvpx"] {
        let theirs = ffmpeg_yuv(&ffmpeg, &ivf, decoder);
        if let Some(diff) = first_difference(&ours, &theirs, stream.width, stream.height) {
            panic!("{name}: {decoder} differs at {diff}; {cover:?}");
        }
        assert_eq!(ours.len(), theirs.len(), "{name}: {decoder} shown bytes");
    }
    eprintln!("{name}: {} bytes identical; {cover:?}", ours.len());
    let _ = std::fs::remove_dir_all(&dir);
    Some(cover)
}

#[test]
fn libvpx_realtime_matches_both_decoders() {
    if check_case(
        "realtime",
        "scale=320:240",
        30,
        &["-b:v", "300k", "-deadline", "realtime", "-cpu-used", "8"],
    )
    .is_none()
    {
        eprintln!("skipped: no local source or ffmpeg");
    }
}

/// Runs cases and checks that, between them, each feature was exercised.
fn run_cases(cases: &[(&str, &str, u32, &[&str])]) -> Vec<Coverage> {
    let mut out = Vec::new();
    for (name, vf, frames, extra) in cases {
        match check_case(name, vf, *frames, extra) {
            Some(cover) => out.push(cover),
            None => {
                eprintln!("skipped: no local source or ffmpeg");
                return Vec::new();
            }
        }
    }
    out
}

#[test]
fn libvpx_speeds_and_quality_match_both_decoders() {
    let covers = run_cases(&[
        (
            "good0",
            "scale=352:288",
            20,
            &["-b:v", "600k", "-deadline", "good", "-cpu-used", "0"],
        ),
        (
            "best",
            "scale=176:144",
            12,
            &["-b:v", "200k", "-deadline", "best"],
        ),
        (
            "rt16",
            "scale=320:240",
            30,
            &["-b:v", "200k", "-deadline", "realtime", "-cpu-used", "16"],
        ),
        (
            "high_q",
            "scale=320:240",
            10,
            &["-qmin", "0", "-qmax", "2", "-b:v", "20M"],
        ),
        (
            "best_split",
            "scale=320:240",
            20,
            &["two-pass", "-deadline", "best", "-b:v", "500k"],
        ),
        (
            "low_q",
            "scale=320:240",
            20,
            &["-qmin", "60", "-qmax", "63", "-b:v", "50k"],
        ),
    ]);
    if !covers.is_empty() {
        for (mode, name) in [
            "DC", "V", "H", "TM", "B", "NEAREST", "NEAR", "ZERO", "NEW", "SPLIT",
        ]
        .iter()
        .enumerate()
        {
            assert!(
                covers.iter().any(|c| c.modes[mode] > 0),
                "no {name} macroblock"
            );
        }
    }
}

#[test]
fn libvpx_partitions_and_resilience_match_both_decoders() {
    let covers = run_cases(&[
        (
            "slices2",
            "scale=320:240",
            15,
            &["-slices", "2", "-b:v", "400k"],
        ),
        (
            "slices4",
            "scale=320:240",
            15,
            &["-slices", "4", "-b:v", "400k"],
        ),
        (
            "slices8",
            "scale=640:360",
            10,
            &["-slices", "8", "-b:v", "800k"],
        ),
        (
            "resilient",
            "scale=320:240",
            20,
            &["-error-resilient", "default", "-b:v", "300k"],
        ),
    ]);
    if !covers.is_empty() {
        assert_eq!(covers.iter().map(|c| c.partitions).max(), Some(8));
        assert!(covers.iter().any(|c| c.no_entropy_refresh > 0));
    }
}

#[test]
fn libvpx_altref_and_golden_match_both_decoders() {
    let covers = run_cases(&[
        (
            "altref",
            "scale=320:240",
            40,
            &[
                "-deadline",
                "good",
                "-cpu-used",
                "1",
                "-auto-alt-ref",
                "1",
                "-lag-in-frames",
                "16",
                "-arnr-maxframes",
                "5",
                "-arnr-strength",
                "3",
                "-b:v",
                "300k",
            ],
        ),
        (
            "smooth_two_pass_altref",
            "scale=320:240",
            60,
            &[
                "two-pass",
                "-deadline",
                "good",
                "-cpu-used",
                "0",
                "-auto-alt-ref",
                "1",
                "-lag-in-frames",
                "25",
                "-arnr-maxframes",
                "7",
                "-b:v",
                "250k",
            ],
        ),
        (
            "layers",
            "scale=320:240",
            24,
            &[
                "-deadline",
                "realtime",
                "-cpu-used",
                "8",
                "-ts-parameters",
                "ts_number_layers=3:ts_target_bitrate=100,200,300:ts_rate_decimator=4,2,1:ts_periodicity=4:ts_layer_id=0,2,1,2",
            ],
        ),
        ("keys", "scale=320:240", 30, &["-g", "7", "-b:v", "300k"]),
    ]);
    if !covers.is_empty() {
        assert!(covers.iter().any(|c| c.hidden > 0), "no hidden frame");
        assert!(
            covers.iter().any(|c| c.altref_refresh > 0),
            "no alt-ref refresh"
        );
        assert!(covers.iter().any(|c| c.sign_bias > 0), "no sign bias");
        assert!(
            covers.iter().any(|c| c.golden_refresh > 0),
            "no golden refresh"
        );
        assert!(covers.iter().any(|c| c.keys > 2), "no repeated key frame");
    }
}

#[test]
fn libvpx_filter_options_match_both_decoders() {
    run_cases(&[
        (
            "sharp5",
            "scale=320:240",
            15,
            &["-sharpness", "5", "-b:v", "200k"],
        ),
        (
            "sharp7",
            "scale=320:240",
            15,
            &["-sharpness", "7", "-b:v", "150k"],
        ),
        (
            "static",
            "scale=320:240",
            20,
            &["-static-thresh", "800", "-b:v", "300k"],
        ),
        (
            "noise",
            "scale=320:240",
            20,
            &["-noise-sensitivity", "3", "-b:v", "300k"],
        ),
        (
            "screen",
            "scale=320:240",
            20,
            &[
                "-deadline",
                "realtime",
                "-screen-content-mode",
                "1",
                "-b:v",
                "300k",
            ],
        ),
    ]);
}

#[test]
fn libvpx_versions_match_both_decoders() {
    let covers = run_cases(&[
        (
            "profile1",
            "scale=320:240",
            15,
            &["-profile:v", "1", "-b:v", "300k"],
        ),
        (
            "profile2",
            "scale=320:240",
            15,
            &["-profile:v", "2", "-b:v", "300k"],
        ),
        (
            "profile3",
            "scale=320:240",
            15,
            &["-profile:v", "3", "-b:v", "300k"],
        ),
    ]);
    if !covers.is_empty() {
        for v in 1..4 {
            assert!(covers.iter().any(|c| c.versions[v] > 0), "no version {v}");
        }
        assert!(
            covers.iter().any(|c| c.simple_filter > 0),
            "no simple filter"
        );
    }
}

#[test]
fn libvpx_odd_sizes_match_both_decoders() {
    run_cases(&[
        ("odd193", "scale=193:97", 12, &["-b:v", "150k"]),
        ("odd17", "scale=17:9", 8, &["-b:v", "50k"]),
        ("tall", "scale=33:301", 8, &["-b:v", "100k"]),
        ("one", "scale=1:1", 4, &["-b:v", "10k"]),
    ]);
}

#[test]
fn independent_webm_matches_ffmpeg() {
    let Some(webm) = local_root().map(|r| r.join("file_example_WEBM_1920_3_7MB.webm")) else {
        eprintln!("skipped: no ../local-files");
        return;
    };
    let Some(ffmpeg) = ffmpeg_bin() else {
        eprintln!("skipped: ffmpeg was not found");
        return;
    };
    if !webm.is_file() {
        eprintln!("skipped: no 1080p WebM");
        return;
    }
    let dir = temp_dir("webm");
    let ivf = dir.join("in.ivf");
    let status = Command::new(&ffmpeg)
        .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(&webm)
        .args(["-an", "-c:v", "copy", "-f", "ivf"])
        .arg(&ivf)
        .output()
        .expect("copy");
    assert!(
        status.status.success(),
        "{}",
        String::from_utf8_lossy(&status.stderr)
    );
    let stream = parse_ivf(&std::fs::read(&ivf).expect("ivf"));
    let mut cover = coverage(&stream);
    let started = std::time::Instant::now();
    let (ours, modes) = decode_counted(&stream);
    let elapsed = started.elapsed();
    cover.modes = modes;
    let theirs = ffmpeg_yuv(&ffmpeg, &ivf, "vp8");
    if let Some(diff) = first_difference(&ours, &theirs, stream.width, stream.height) {
        panic!("webm differs at {diff}");
    }
    assert_eq!(ours.len(), theirs.len());
    eprintln!(
        "webm {}x{} {} frames identical in {:.2} s; {cover:?}",
        stream.width,
        stream.height,
        stream.frames.len(),
        elapsed.as_secs_f64()
    );
    let _ = std::fs::remove_dir_all(&dir);
}
