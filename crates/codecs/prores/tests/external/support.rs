//! Tools, clips, scratch folders, and picture metrics.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

pub fn tool(name: &str) -> Option<PathBuf> {
    let out = Command::new("which").arg(name).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    let line = text.lines().next()?.trim();
    (!line.is_empty()).then(|| PathBuf::from(line))
}

/// Panics when oracles are required, else returns so the test can skip.
pub fn skip(what: &str) {
    if std::env::var_os("PRORES_REQUIRE_ORACLES").is_some() {
        panic!("{what} is required");
    }
    eprintln!("skipped: {what} is missing");
}

pub fn ffmpeg() -> Option<PathBuf> {
    let found = tool("ffmpeg");
    if found.is_none() {
        skip("ffmpeg");
    }
    found
}

/// `../local-files/<name>` beside the checkout, when it is there.
pub fn local(name: &str) -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = root.ancestors().nth(3)?.join("../local-files").join(name);
    if path.is_file() {
        Some(path)
    } else {
        skip(&format!("local-files/{name}"));
        None
    }
}

/// The 1080p BT.709 clip most tests start from.
pub const HD: &str = "16929689_1920_1080_30fps.mp4";
/// The 2160p BT.709 clip.
pub const UHD: &str = "12129324_3840_2160_30fps.mp4";

pub struct Scratch {
    pub dir: PathBuf,
}

impl Scratch {
    pub fn new(label: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("cutlass-prores-{label}-{n}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch");
        Self { dir }
    }

    pub fn path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Runs a tool; true when it exits cleanly.
pub fn run(program: &Path, args: &[&str]) -> bool {
    match Command::new(program).args(args).output() {
        Ok(out) if out.status.success() => true,
        Ok(out) => {
            eprintln!(
                "{} {:?} failed: {}",
                program.display(),
                args,
                String::from_utf8_lossy(&out.stderr)
            );
            false
        }
        Err(err) => {
            eprintln!("{} did not start: {err}", program.display());
            false
        }
    }
}

/// The ProRes frames of a movie's video track, as ffmpeg's stream copy writes them.
pub fn frames(ffmpeg: &Path, movie: &Path, scratch: &Scratch) -> Vec<Vec<u8>> {
    let out = scratch.path("frames.bin");
    let movie = movie.to_str().expect("path");
    let out_s = out.to_str().expect("path");
    assert!(run(
        ffmpeg,
        &[
            "-v", "error", "-y", "-i", movie, "-map", "0:v:0", "-c", "copy", "-f", "data", out_s
        ]
    ));
    let bytes = std::fs::read(&out).expect("frames");
    split_frames(&bytes)
}

/// Splits back-to-back frames by their `frame_size` fields.
pub fn split_frames(mut bytes: &[u8]) -> Vec<Vec<u8>> {
    let mut frames = Vec::new();
    while bytes.len() >= 8 {
        let size = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        assert!(size >= 8 && size <= bytes.len(), "frame size {size}");
        frames.push(bytes[..size].to_vec());
        bytes = &bytes[size..];
    }
    frames
}

/// ffmpeg's decode of a movie as raw little-endian planes in `pix_fmt`.
/// With `toolbox`, VideoToolbox decodes instead (Apple's decoder, through ffmpeg).
pub fn reference(
    ffmpeg: &Path,
    movie: &Path,
    pix_fmt: &str,
    scratch: &Scratch,
    toolbox: bool,
) -> Vec<u8> {
    let out = scratch.path(if toolbox {
        "toolbox.raw"
    } else {
        "reference.raw"
    });
    let movie = movie.to_str().expect("path");
    let out_s = out.to_str().expect("path");
    let mut args = vec!["-v", "error", "-y"];
    if toolbox {
        args.extend(["-hwaccel", "videotoolbox"]);
    }
    args.extend([
        "-i", movie, "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", pix_fmt, out_s,
    ]);
    assert!(run(ffmpeg, &args));
    std::fs::read(&out).expect("reference")
}

/// Little-endian 16-bit samples.
pub fn samples16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect()
}

/// How far one plane is from another.
#[derive(Clone, Copy, Debug, Default)]
pub struct Diff {
    pub max: u32,
    pub mismatched: usize,
    pub count: usize,
    pub psnr: f64,
}

impl Diff {
    pub fn between(test: &[u16], reference: &[u16], peak: f64) -> Self {
        assert_eq!(test.len(), reference.len(), "plane lengths");
        let mut max = 0;
        let mut mismatched = 0;
        let mut sq = 0f64;
        for (&a, &b) in test.iter().zip(reference) {
            let d = u32::from(a.abs_diff(b));
            if d > 0 {
                mismatched += 1;
            }
            max = max.max(d);
            sq += f64::from(d * d);
        }
        let mse = sq / test.len().max(1) as f64;
        let psnr = if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (peak * peak / mse).log10()
        };
        Self {
            max,
            mismatched,
            count: test.len(),
            psnr,
        }
    }
}

impl std::fmt::Display for Diff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let psnr = if self.psnr.is_infinite() {
            String::from("inf")
        } else {
            format!("{:.2}", self.psnr)
        };
        write!(
            f,
            "max {} mismatched {}/{} ({:.4}%) psnr {psnr}",
            self.max,
            self.mismatched,
            self.count,
            100.0 * self.mismatched as f64 / self.count.max(1) as f64
        )
    }
}

/// Frames per second the tests code at: 30000/1001.
pub const RATE: (u32, u32) = (30_000, 1_001);

/// Writes ProRes frames into a QuickTime movie with this repository's muxer.
pub fn mux_mov(frames: &[Vec<u8>], fourcc: &[u8; 4], width: u32, height: u32) -> Vec<u8> {
    use cutlass_codec::{
        CodecId, FrameSize, Packet, StreamCodec, StreamParams, Timescale, Timestamp,
    };
    use cutlass_container::{Mux, NewStream, TrackDuration};
    use cutlass_isobmff::{FileBrand, Mp4Mux};
    let mut mux = Mp4Mux::with_brand(std::io::Cursor::new(Vec::new()), FileBrand::Mov);
    let scale = Timescale::new(RATE.0);
    let track = mux
        .add_stream(NewStream {
            params: StreamParams::video(
                StreamCodec::Known(CodecId::Prores),
                FrameSize::try_new(width, height).expect("size"),
                fourcc.to_vec(),
            ),
            timescale: scale,
            language: None,
        })
        .expect("stream");
    let step = i64::from(RATE.1);
    for (index, frame) in frames.iter().enumerate() {
        let pts = index as i64 * step;
        let time = Timestamp::try_new(Some(pts), Some(pts), Some(step)).expect("time");
        let packet =
            Packet::try_new(track.get(), frame.clone(), time, scale, true).expect("packet");
        mux.write_packet(&packet).expect("write");
    }
    mux.finish(&[TrackDuration {
        track,
        timescale: scale,
        ticks: frames.len() as i64 * step,
    }])
    .expect("finish");
    mux.into_writer().expect("writer").into_inner()
}

/// ffmpeg's SSIM of a movie against raw planes: (Y, all).
pub fn ssim(
    ffmpeg: &Path,
    movie: &Path,
    raw: &Path,
    pix_fmt: &str,
    width: u32,
    height: u32,
) -> (f64, f64) {
    let size = format!("{width}x{height}");
    let rate = format!("{}/{}", RATE.0, RATE.1);
    let out = Command::new(ffmpeg)
        .args(["-v", "info", "-i"])
        .arg(movie)
        .args([
            "-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", &size, "-r", &rate, "-i",
        ])
        .arg(raw)
        .args(["-lavfi", "[0:v][1:v]ssim", "-f", "null", "-"])
        .output()
        .expect("ssim");
    let text = String::from_utf8_lossy(&out.stderr);
    let line = text
        .lines()
        .rev()
        .find(|l| l.contains("SSIM Y:"))
        .expect("ssim line");
    let field = |key: &str| -> f64 {
        let at = line.find(key).expect("field") + key.len();
        line[at..]
            .split_whitespace()
            .next()
            .expect("value")
            .parse()
            .expect("number")
    };
    (field("Y:"), field("All:"))
}
