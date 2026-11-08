//! Checks against independent implementations: FFmpeg's `mjpeg` encoder and
//! decoder, and libjpeg-turbo's `cjpeg` and `djpeg`. They run as separate
//! programs in tests only. A missing program skips the test, unless
//! `MJPEG_REQUIRE_ORACLES=1`, which makes it fail.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

use cutlass_decode::{ColorRange, PixelFormat, VideoFrame};
use cutlass_jpeg::{Sampling, Tables, decode_image};

use crate::tests::{detailed_rgb, field_rows, interleave, params, strip};
use crate::{MjpegDecoder, MjpegEncoder};

fn tool(name: &str) -> Option<PathBuf> {
    let homebrew = Path::new("/opt/homebrew/bin").join(name);
    let found = if homebrew.is_file() {
        Some(homebrew)
    } else {
        std::env::var_os("PATH").and_then(|paths| {
            std::env::split_paths(&paths)
                .map(|dir| dir.join(name))
                .find(|path| path.is_file())
        })
    };
    if found.is_none() && std::env::var_os("MJPEG_REQUIRE_ORACLES").is_some() {
        panic!("{name} is required by MJPEG_REQUIRE_ORACLES");
    }
    found
}

struct Tools {
    ffmpeg: PathBuf,
    cjpeg: PathBuf,
    djpeg: PathBuf,
}

fn tools() -> Option<Tools> {
    Some(Tools {
        ffmpeg: tool("ffmpeg")?,
        cjpeg: tool("cjpeg")?,
        djpeg: tool("djpeg")?,
    })
}

/// A temporary folder, removed when the test ends.
struct Scratch(PathBuf);

impl std::ops::Deref for Scratch {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scratch() -> Scratch {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "cutlass-mjpeg-oracle-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    Scratch(dir)
}

fn run(command: &mut Command) -> Output {
    let output = command.output().expect("run tool");
    assert!(
        output.status.success(),
        "{command:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

/// Width, height, channels, and samples of a binary PPM or PGM.
fn pnm(bytes: &[u8]) -> (u32, u32, usize, Vec<u8>) {
    let mut fields = Vec::new();
    let mut at = 0;
    while fields.len() < 4 {
        while bytes[at].is_ascii_whitespace() {
            at += 1;
        }
        let start = at;
        while !bytes[at].is_ascii_whitespace() {
            at += 1;
        }
        fields.push(String::from_utf8_lossy(&bytes[start..at]).into_owned());
    }
    let channels = if fields[0] == "P6" { 3 } else { 1 };
    let width = fields[1].parse().expect("width");
    let height = fields[2].parse().expect("height");
    (width, height, channels, bytes[at + 1..].to_vec())
}

fn write_ppm(path: &Path, rgb: &[u8], width: u32, height: u32) {
    let mut bytes = format!("P6\n{width} {height}\n255\n").into_bytes();
    bytes.extend_from_slice(rgb);
    std::fs::write(path, bytes).expect("ppm");
}

/// djpeg's integer decode of `jpeg`, as RGB or grey samples. Its stderr must
/// be empty: a warning is a failure.
fn djpeg(tools: &Tools, dir: &Path, jpeg: &[u8]) -> (u32, u32, usize, Vec<u8>) {
    let input = dir.join("djpeg-in.jpg");
    std::fs::write(&input, jpeg).expect("write");
    let output = run(Command::new(&tools.djpeg)
        .args(["-dct", "int", "-pnm"])
        .arg(&input));
    assert!(
        output.stderr.is_empty(),
        "djpeg: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    pnm(&output.stdout)
}

/// Each JPEG image in a concatenated stream.
fn split_stream(stream: &[u8]) -> Vec<Vec<u8>> {
    let mut frames = Vec::new();
    let mut at = 0;
    while at + 2 <= stream.len() {
        let decoded =
            decode_image(&stream[at..], &mut Tables::standard_huffman()).expect("stream frame");
        frames.push(stream[at..at + decoded.len].to_vec());
        at += decoded.len;
    }
    frames
}

fn our_rgb(decoder: &mut MjpegDecoder, jpeg: &[u8], channels: usize) -> Vec<u8> {
    let mut rgba = Vec::new();
    decoder
        .decode(jpeg)
        .expect("decode")
        .to_rgba(&mut rgba)
        .expect("rgba");
    rgba.as_chunks::<4>()
        .0
        .iter()
        .flat_map(|pixel| pixel[..channels].to_vec())
        .collect()
}

/// The source of the oracle clips: the first video in `../local-files`
/// when it is there, else FFmpeg's own test pattern.
fn source_args(width: u32, height: u32) -> Vec<String> {
    let scale = format!("scale={width}:{height}");
    match first_local_video() {
        Some(path) => vec![
            "-ss".into(),
            "1".into(),
            "-i".into(),
            path.to_string_lossy().into_owned(),
            "-vf".into(),
            scale,
        ],
        None => vec![
            "-f".into(),
            "lavfi".into(),
            "-i".into(),
            format!("testsrc2=size={width}x{height}:rate=10"),
        ],
    }
}

fn local_files() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files");
    dir.is_dir().then_some(dir)
}

/// Natural footage first; any other video only when those are absent.
const PREFERRED: [&str; 2] = [
    "16929689_1920_1080_30fps.mp4",
    "7150172-hd_720_1280_50fps.mp4",
];

fn local_videos() -> Vec<PathBuf> {
    let Some(dir) = local_files() else {
        return Vec::new();
    };
    let preferred: Vec<PathBuf> = PREFERRED
        .iter()
        .map(|name| dir.join(name))
        .filter(|path| path.is_file())
        .collect();
    if !preferred.is_empty() {
        return preferred;
    }
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut videos: Vec<(u64, PathBuf)> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| {
                    matches!(ext.to_ascii_lowercase().as_str(), "mp4" | "mov" | "webm")
                })
        })
        .filter_map(|path| Some((std::fs::metadata(&path).ok()?.len(), path)))
        .collect();
    videos.sort();
    videos.into_iter().take(1).map(|(_, path)| path).collect()
}

fn first_local_video() -> Option<PathBuf> {
    local_videos().into_iter().next()
}

/// FFmpeg's raw planes for one JPEG, decoded with its integer IDCT, in the
/// file's own pixel format.
fn ffmpeg_planes(tools: &Tools, dir: &Path, jpeg: &[u8], pix_fmt: &str) -> Vec<u8> {
    let input = dir.join("ffmpeg-in.jpg");
    std::fs::write(&input, jpeg).expect("write");
    run(Command::new(&tools.ffmpeg)
        .args(["-v", "error", "-xerror", "-idct", "int", "-i"])
        .arg(&input)
        .args(["-f", "rawvideo", "-pix_fmt", pix_fmt, "-y"])
        .arg(dir.join("planes.raw")));
    std::fs::read(dir.join("planes.raw")).expect("planes")
}

fn assert_planes_within_one(
    tools: &Tools,
    dir: &Path,
    jpeg: &[u8],
    pix_fmt: &str,
    frame: &VideoFrame,
) {
    let reference = ffmpeg_planes(tools, dir, jpeg, pix_fmt);
    let ours: Vec<u8> = [&frame.y, &frame.u, &frame.v]
        .into_iter()
        .flat_map(|plane| plane.samples.clone())
        .collect();
    assert_eq!(ours.len(), reference.len(), "{pix_fmt}");
    let worst = ours
        .iter()
        .zip(&reference)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    assert!(worst <= 1, "{pix_fmt}: differs by {worst}");
}

#[test]
fn ffmpeg_motion_jpeg_matches_djpeg_exactly_and_ffmpeg_within_one() {
    let Some(tools) = tools() else { return };
    let dir = scratch();
    let (width, height) = (99, 67);
    // FFmpeg writes `yuv420p` as studio range, marked by a `CS=ITU601`
    // comment. Its encoder has no grey; cjpeg covers that below.
    for (pix_fmt, format, range) in [
        ("yuvj420p", PixelFormat::Yuv420p, ColorRange::Full),
        ("yuvj422p", PixelFormat::Yuv422p, ColorRange::Full),
        ("yuvj444p", PixelFormat::Yuv444p, ColorRange::Full),
        ("yuv420p", PixelFormat::Yuv420p, ColorRange::Limited),
    ] {
        let stream = dir.join(format!("{pix_fmt}.mjpeg"));
        run(Command::new(&tools.ffmpeg)
            .args(["-v", "error"])
            .args(source_args(width, height))
            .args([
                "-frames:v",
                "4",
                "-c:v",
                "mjpeg",
                "-q:v",
                "3",
                "-pix_fmt",
                pix_fmt,
            ])
            .args(["-strict", "-1", "-f", "mjpeg", "-y"])
            .arg(&stream));
        let frames = split_stream(&std::fs::read(&stream).expect("stream"));
        assert_eq!(frames.len(), 4, "{pix_fmt}");
        let mut decoder = MjpegDecoder::open(&params(width, height, b"jpeg")).expect("open");
        for (index, frame) in frames.iter().enumerate() {
            let (w, h, channels, want) = djpeg(&tools, &dir, frame);
            assert_eq!((w, h), (width, height));
            let ours = our_rgb(&mut decoder, frame, channels);
            if range == ColorRange::Full {
                assert_eq!(ours, want, "{pix_fmt} frame {index}");
            } else {
                // djpeg ignores the studio-range comment. FFmpeg honours it
                // but upsamples chroma its own way, so compare on average.
                let reference = ffmpeg_planes(&tools, &dir, frame, "rgb24");
                let mean = |a: &[u8], b: &[u8]| {
                    a.iter()
                        .zip(b)
                        .map(|(x, y)| f64::from(x.abs_diff(*y)))
                        .sum::<f64>()
                        / a.len() as f64
                };
                let studio = mean(&ours, &reference);
                let full = mean(&want, &reference);
                assert!(studio < 2.0 && full > 2.0 * studio, "{studio} vs {full}");
            }
            let planar = decoder
                .decode(frame)
                .expect("decode")
                .into_frame()
                .expect("frame");
            assert_eq!((planar.format, planar.range), (format, range), "{pix_fmt}");
            assert_planes_within_one(&tools, &dir, frame, pix_fmt, &planar);
        }
    }
}

#[test]
fn frames_without_dht_decode_like_djpeg_with_its_default_tables() {
    let Some(tools) = tools() else { return };
    let dir = scratch();
    let (width, height) = (64, 48);
    let stream = dir.join("default.mjpeg");
    run(Command::new(&tools.ffmpeg)
        .args(["-v", "error"])
        .args(source_args(width, height))
        .args([
            "-frames:v",
            "3",
            "-c:v",
            "mjpeg",
            "-huffman",
            "default",
            "-q:v",
            "4",
        ])
        .args(["-pix_fmt", "yuvj422p", "-f", "mjpeg", "-y"])
        .arg(&stream));
    let mut decoder = MjpegDecoder::open(&params(width, height, &[])).expect("open");
    for frame in split_stream(&std::fs::read(&stream).expect("stream")) {
        let bare = strip(&frame, &[0xc4]);
        assert!(bare.len() < frame.len(), "ffmpeg wrote DHT");
        let (_, _, channels, want) = djpeg(&tools, &dir, &bare);
        assert_eq!(djpeg(&tools, &dir, &frame).3, want);
        assert_eq!(our_rgb(&mut decoder, &bare, channels), want);
    }
}

#[test]
fn cjpeg_440_rgb_and_restart_frames_match_djpeg() {
    let Some(tools) = tools() else { return };
    let dir = scratch();
    let (width, height) = (53, 38);
    let source = dir.join("source.ppm");
    write_ppm(&source, &detailed_rgb(width, height, 21), width, height);
    for args in [
        &["-sample", "1x2"][..],
        &["-grayscale"],
        &["-rgb"],
        &["-sample", "2x2", "-restart", "1"],
        &["-sample", "2x1", "-progressive"],
        &["-sample", "2x1", "-quality", "5"],
    ] {
        let jpeg = run(Command::new(&tools.cjpeg)
            .args(["-dct", "int", "-quality", "88"])
            .args(args)
            .arg(&source))
        .stdout;
        if args.contains(&"5") {
            let coarse = crate::tests::segments(&jpeg);
            assert!(coarse.iter().any(|(marker, ..)| *marker == 0xc1), "SOF1");
            assert!(
                coarse
                    .iter()
                    .any(|(marker, start, _)| *marker == 0xdb && jpeg[start + 4] >> 4 == 1),
                "16-bit DQT"
            );
        }
        let (_, _, channels, want) = djpeg(&tools, &dir, &jpeg);
        let mut decoder = MjpegDecoder::open(&params(width, height, &[])).expect("open");
        assert_eq!(our_rgb(&mut decoder, &jpeg, channels), want, "{args:?}");
        let frame = decoder
            .decode(&jpeg)
            .expect("decode")
            .into_frame()
            .expect("frame");
        let want_format = match args[0] {
            "-rgb" => PixelFormat::Gbrp,
            "-grayscale" => PixelFormat::Gray8,
            _ if args[1] == "1x2" => PixelFormat::Yuv440p,
            _ if args[1] == "2x1" => PixelFormat::Yuv422p,
            _ => PixelFormat::Yuv420p,
        };
        assert_eq!(frame.format, want_format, "{args:?}");
        match want_format {
            PixelFormat::Gray8 => assert_planes_within_one(&tools, &dir, &jpeg, "gray", &frame),
            PixelFormat::Yuv440p => {
                assert_planes_within_one(&tools, &dir, &jpeg, "yuvj440p", &frame);
            }
            _ => {}
        }
        if want_format == PixelFormat::Gbrp {
            let rgb: Vec<u8> = (0..(width * height) as usize)
                .flat_map(|i| [frame.v.samples[i], frame.y.samples[i], frame.u.samples[i]])
                .collect();
            assert_eq!(rgb, want);
        }
    }
}

#[test]
fn cjpeg_field_pairs_weave_like_the_djpeg_fields_interleaved() {
    let Some(tools) = tools() else { return };
    let dir = scratch();
    for (width, height, sample) in [(72, 48, "2x2"), (72, 45, "2x1"), (40, 34, "1x1")] {
        let rgb = detailed_rgb(width, height, 22);
        let (top, rows) = field_rows(&rgb, width, height, 0);
        let (mut bottom, bottom_rows) = field_rows(&rgb, width, height, 1);
        if bottom_rows < rows {
            let row = (width * 3) as usize;
            let last = bottom[bottom.len() - row..].to_vec();
            bottom.extend_from_slice(&last);
        }
        let mut fields = Vec::new();
        for (name, samples) in [("top", &top), ("bottom", &bottom)] {
            let path = dir.join(format!("{name}.ppm"));
            write_ppm(&path, samples, width, rows);
            fields.push(
                run(Command::new(&tools.cjpeg)
                    .args(["-dct", "int", "-quality", "90", "-sample", sample])
                    .arg(&path))
                .stdout,
            );
        }
        let want = interleave(
            &djpeg(&tools, &dir, &fields[0]).3,
            &djpeg(&tools, &dir, &fields[1]).3,
            width as usize * 3,
            height as usize,
        );
        let mut packet = fields[0].clone();
        packet.extend_from_slice(&fields[1]);
        let mut decoder = MjpegDecoder::open(&params(width, height, &[])).expect("open");
        assert_eq!(
            our_rgb(&mut decoder, &packet, 3),
            want,
            "{width}x{height} {sample}"
        );

        let mut swapped = fields[1].clone();
        swapped.extend_from_slice(&fields[0]);
        let mut decoder =
            MjpegDecoder::open(&params(width, height, &[b'j', b'p', b'e', b'g', 2, 14]))
                .expect("open");
        assert_eq!(our_rgb(&mut decoder, &swapped, 3), want, "bottom first");
    }
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse = a
        .iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    10.0 * (255.0 * 255.0 / mse.max(1e-9)).log10()
}

#[test]
fn this_encoder_opens_cleanly_and_matches_cjpeg_size_and_quality() {
    let Some(tools) = tools() else { return };
    let dir = scratch();
    let (width, height) = (160, 90);
    let source = dir.join("frame.ppm");
    run(Command::new(&tools.ffmpeg)
        .args(["-v", "error"])
        .args(source_args(width, height))
        .args(["-frames:v", "1", "-pix_fmt", "rgb24", "-y"])
        .arg(&source));
    let (_, _, _, rgb) = pnm(&std::fs::read(&source).expect("source"));
    let cjpeg_len = |quality: &str| {
        run(Command::new(&tools.cjpeg)
            .args(["-dct", "int", "-quality", quality])
            .arg(&source))
        .stdout
        .len()
    };
    assert!(
        cjpeg_len("95") > 2 * cjpeg_len("30"),
        "the source frame has too little detail to measure quality"
    );
    for (sampling, args) in [
        (Sampling::Yuv420, &["-sample", "2x2"][..]),
        (Sampling::Yuv422, &["-sample", "2x1"]),
        (Sampling::Yuv444, &["-sample", "1x1"]),
        (Sampling::Gray, &["-grayscale"]),
    ] {
        for quality in [30u8, 75, 95] {
            let mut encoder = MjpegEncoder::open(&params(width, height, &[quality]))
                .expect("open")
                .with_sampling(sampling);
            let rgba: Vec<u8> = rgb
                .as_chunks::<3>()
                .0
                .iter()
                .flat_map(|p| [p[0], p[1], p[2], 255])
                .collect();
            let ours = encoder
                .encode_rgba(width, height, &rgba, 1, 25)
                .expect("encode");
            let path = dir.join("ours.jpg");
            std::fs::write(&path, &ours).expect("write");
            let checked = run(Command::new(&tools.ffmpeg)
                .args(["-v", "warning", "-xerror", "-i"])
                .arg(&path)
                .args(["-f", "null", "-"]));
            assert!(
                checked.stderr.is_empty(),
                "ffmpeg: {}",
                String::from_utf8_lossy(&checked.stderr)
            );

            let theirs = run(Command::new(&tools.cjpeg)
                .args(["-dct", "int", "-quality", &quality.to_string()])
                .args(args)
                .arg(&source))
            .stdout;
            let size_ratio = ours.len() as f64 / theirs.len() as f64;
            assert!(
                (size_ratio - 1.0).abs() <= 0.02,
                "{sampling:?} q{quality}: {} vs {} bytes",
                ours.len(),
                theirs.len()
            );
            let (_, _, channels, ours_rgb) = djpeg(&tools, &dir, &ours);
            let theirs_rgb = djpeg(&tools, &dir, &theirs).3;
            // The same samples as cjpeg's: only the table segments differ.
            assert_eq!(ours_rgb, theirs_rgb, "{sampling:?} q{quality}");
            if channels == 3 {
                let ours_psnr = psnr(&ours_rgb, &rgb);
                let theirs_psnr = psnr(&theirs_rgb, &rgb);
                assert!(
                    (ours_psnr - theirs_psnr).abs() <= 0.2,
                    "{sampling:?} q{quality}: {ours_psnr} vs {theirs_psnr} dB"
                );
            }
        }
    }
}

#[test]
fn real_footage_frames_from_local_files_match_djpeg() {
    let Some(tools) = tools() else { return };
    let dir = scratch();
    for video in local_videos() {
        for (pix_fmt, size) in [("yuvj420p", "-2:720"), ("yuvj422p", "641:361")] {
            let stream = dir.join(format!("real-{pix_fmt}.mjpeg"));
            run(Command::new(&tools.ffmpeg)
                .args(["-v", "error", "-ss", "0.5", "-i"])
                .arg(&video)
                .args(["-vf", &format!("scale={size}"), "-frames:v", "5"])
                .args([
                    "-c:v", "mjpeg", "-q:v", "2", "-pix_fmt", pix_fmt, "-f", "mjpeg", "-y",
                ])
                .arg(&stream));
            let frames = split_stream(&std::fs::read(&stream).expect("stream"));
            assert_eq!(frames.len(), 5);
            let mut decoder = MjpegDecoder::open(&params(0, 0, &[])).expect("open");
            for (index, frame) in frames.iter().enumerate() {
                let (_, _, channels, want) = djpeg(&tools, &dir, frame);
                assert_eq!(
                    our_rgb(&mut decoder, frame, channels),
                    want,
                    "{} {pix_fmt} {index}",
                    video.display()
                );
            }
        }
    }
}
