//! Helpers that run the independent tools.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use cutlass_decode::{
    BitDepth, CodecError, CodecId, CodecParams, ColorType, Decoder, Packet, StillImage,
};
use cutlass_encode::{Encoder, PictureInput};
use cutlass_png::{PngDecoder, PngEncoder};

use crate::picture::{self, Picture};
use crate::riff;

pub(crate) fn tool(name: &str) -> Option<PathBuf> {
    let output = Command::new("which").arg(name).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let line = text.lines().next()?.trim();
    (!line.is_empty()).then(|| PathBuf::from(line))
}

/// Skip, or fail when `WEBP_REQUIRE_ORACLES` asks for every oracle.
pub(crate) fn require_or_skip(name: &str) {
    assert!(
        std::env::var_os("WEBP_REQUIRE_ORACLES").is_none(),
        "{name} is required"
    );
}

/// A fresh folder under the system temporary folder.
pub(crate) fn scratch(label: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("cutlass-webp-{label}-{n}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch folder");
    dir
}

/// `../local-files` from the workspace root, when this machine has it.
pub(crate) fn local_files() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files");
    dir.is_dir().then_some(dir)
}

/// RGBA from a PAM that `dwebp -pam` wrote.
pub(crate) fn read_pam(bytes: &[u8]) -> (u32, u32, Vec<u8>) {
    let end = bytes
        .windows(7)
        .position(|window| window == b"ENDHDR\n")
        .expect("pam header")
        + 7;
    let header = std::str::from_utf8(&bytes[..end]).expect("pam text");
    let mut width = 0;
    let mut height = 0;
    let mut depth = 0;
    for line in header.lines() {
        let mut words = line.split_whitespace();
        match (words.next(), words.next()) {
            (Some("WIDTH"), Some(value)) => width = value.parse().expect("width"),
            (Some("HEIGHT"), Some(value)) => height = value.parse().expect("height"),
            (Some("DEPTH"), Some(value)) => depth = value.parse().expect("depth"),
            _ => {}
        }
    }
    assert_eq!(depth, 4, "pam depth");
    (width, height, bytes[end..].to_vec())
}

/// `dwebp -pam` of `webp`: straight RGBA.
pub(crate) fn dwebp_rgba(dwebp: &Path, webp: &Path) -> (u32, u32, Vec<u8>) {
    let output = Command::new(dwebp)
        .arg(webp)
        .args(["-pam", "-o", "-"])
        .output()
        .expect("dwebp runs");
    assert!(
        output.status.success(),
        "dwebp failed on {}: {}",
        webp.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    read_pam(&output.stdout)
}

/// `dwebp -yuv` of `webp`: the Y, U and V planes, each cropped and packed.
pub(crate) fn dwebp_yuv(dwebp: &Path, webp: &Path) -> Vec<u8> {
    let output = Command::new(dwebp)
        .arg(webp)
        .args(["-yuv", "-o", "-"])
        .output()
        .expect("dwebp runs");
    assert!(
        output.status.success(),
        "dwebp -yuv failed on {}",
        webp.display()
    );
    output.stdout
}

/// ffmpeg's own WebP decoder: packed 4:2:0 planes, or `None` when this
/// ffmpeg has no such decoder or refuses the file.
pub(crate) fn ffmpeg_yuv(ffmpeg: &Path, webp: &Path) -> Option<Vec<u8>> {
    let output = Command::new(ffmpeg)
        .args(["-v", "error", "-c:v", "webp", "-i"])
        .arg(webp)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])
        .output()
        .ok()?;
    output.status.success().then_some(output.stdout)
}

/// `cwebp` of `source` with `args`.
pub(crate) fn cwebp(cwebp: &Path, source: &Path, args: &[&str], out: &Path) -> Vec<u8> {
    let status = Command::new(cwebp)
        .arg("-quiet")
        .args(args)
        .arg(source)
        .arg("-o")
        .arg(out)
        .status()
        .expect("cwebp runs");
    assert!(status.success(), "cwebp {args:?} failed");
    std::fs::read(out).expect("cwebp output")
}

/// This crate's still decode of a whole file.
pub(crate) fn decode_still(bytes: &[u8]) -> Picture {
    try_decode_still(bytes).expect("decode")
}

pub(crate) fn try_decode_still(bytes: &[u8]) -> Result<Picture, CodecError> {
    let still = riff::parse(bytes)?;
    picture::decode(still.bitstream)
}

/// First differing pixel, for a readable failure.
pub(crate) fn first_difference(
    ours: &[u8],
    theirs: &[u8],
    width: u32,
) -> Option<(usize, usize, [u8; 4], [u8; 4])> {
    let width = width as usize;
    ours.as_chunks::<4>()
        .0
        .iter()
        .zip(theirs.as_chunks::<4>().0)
        .enumerate()
        .find(|(_, (a, b))| a != b)
        .map(|(index, (a, b))| {
            (
                index % width,
                index / width,
                [a[0], a[1], a[2], a[3]],
                [b[0], b[1], b[2], b[3]],
            )
        })
}

/// Every `.webp` below `dir`.
pub(crate) fn webp_files(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(next) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&next) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|ext| ext == "webp") {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

/// Write straight RGBA as a PNG with `cutlass-png`.
pub(crate) fn write_png(path: &Path, width: u32, height: u32, rgba: &[u8]) {
    let samples = rgba.iter().map(|&byte| u16::from(byte)).collect();
    let image = StillImage::from_samples(width, height, ColorType::Rgba, BitDepth::Eight, samples);
    let mut encoder =
        PngEncoder::open(&CodecParams::video(CodecId::Png, width, height)).expect("png encoder");
    let packets = encoder
        .push_picture(&PictureInput { image })
        .expect("png encode");
    std::fs::write(path, &packets[0].data).expect("write png");
}

/// Straight RGBA of a PNG, through `cutlass-png`.
pub(crate) fn read_png(bytes: &[u8]) -> (u32, u32, Vec<u8>) {
    let mut decoder =
        PngDecoder::open(&CodecParams::video(CodecId::Png, 1, 1)).expect("png decoder");
    let outputs = decoder
        .decode_picture(&Packet::video(bytes.to_vec(), 0, 0, 0, true))
        .expect("png decode");
    let image = &outputs[0].image;
    assert_eq!(image.depth, BitDepth::Eight);
    let rgba = match image.color {
        ColorType::Rgba => image.samples.iter().map(|&s| s as u8).collect(),
        ColorType::Rgb => image
            .samples
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|p| [p[0] as u8, p[1] as u8, p[2] as u8, 255])
            .collect(),
        other => panic!("png color {other:?}"),
    };
    (image.width, image.height, rgba)
}

/// ImageIO's decode of `webp`, through `sips` to a PNG.
pub(crate) fn sips_rgba(sips: &Path, webp: &Path) -> (u32, u32, Vec<u8>) {
    let out = webp.with_extension("sips.png");
    let output = Command::new(sips)
        .args(["-s", "format", "png"])
        .arg(webp)
        .arg("--out")
        .arg(&out)
        .output()
        .expect("sips runs");
    assert!(output.status.success(), "sips failed on {}", webp.display());
    read_png(&std::fs::read(&out).expect("sips output"))
}

/// A first frame of a video in `local-files`, as a PNG, through ffmpeg.
pub(crate) fn video_frame(ffmpeg: &Path, video: &Path, out: &Path, filter: &str) -> bool {
    Command::new(ffmpeg)
        .args(["-v", "error", "-y", "-ss", "1", "-i"])
        .arg(video)
        .args(["-frames:v", "1", "-vf", filter])
        .arg(out)
        .status()
        .is_ok_and(|status| status.success())
        && out.is_file()
}

/// Synthetic RGBA with every alpha level, gradients, and hard edges.
pub(crate) fn synthetic_rgba(width: u32, height: u32) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let red = (x * 255 / width.max(1)) as u8;
            let green = (y * 255 / height.max(1)) as u8;
            let blue = if (x / 7 + y / 5) % 2 == 0 { 30 } else { 220 };
            let alpha = ((x + y * 3) % 256) as u8;
            rgba.extend_from_slice(&[red, green, blue, alpha]);
        }
    }
    rgba
}
