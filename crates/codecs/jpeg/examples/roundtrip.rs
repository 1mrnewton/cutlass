//! Encode and decode one 16×16 gray frame and compare those pixels to ffmpeg.
//!
//! The frame is gray level 128. ffmpeg writes a baseline 4:2:0 JPEG of it.
//! This crate decodes that file and encodes the same frame at quality 75.
//! `matched` is true only when both results equal ffmpeg's raw pixels.

#![allow(clippy::print_stdout)]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use cutlass_decode::{
    CodecId, CodecParams, ColorPrimaries, ColorRange, ColorType, Decoder, MatrixCoefficients,
    Packet, PixelFormat, Transfer,
};
use cutlass_encode::{BitDepth, Encoder, PictureInput, StillImage};
use cutlass_jpeg::{JpegDecoder, JpegEncoder, read_jpeg_info};

const WIDTH: u32 = 16;
const HEIGHT: u32 = 16;
const LEVEL: u16 = 128;

fn main() -> Result<(), String> {
    let Some(ffmpeg) = locate_ffmpeg() else {
        return Err("ffmpeg is not installed".to_string());
    };
    let image = gray_frame()?;
    let outcome = compare(&ffmpeg, &image)?;
    println!(
        "width={} height={} matched={}",
        outcome.width, outcome.height, outcome.matched
    );
    if !outcome.matched {
        return Err("pixels differ".to_string());
    }
    Ok(())
}

struct Outcome {
    width: u32,
    height: u32,
    matched: bool,
}

fn compare(ffmpeg: &Path, image: &StillImage) -> Result<Outcome, String> {
    let raw = pack(&image.samples)?;
    let jpeg = ffmpeg_jpeg(ffmpeg, &raw, image.width, image.height, "gray")?;
    ensure_baseline(&jpeg)?;
    let decoded = decode(&jpeg)?;
    let foreign = ffmpeg_raw(ffmpeg, &jpeg, raw_pix(decoded.color)?)?;
    let file_match = same(&decoded.samples, &foreign);

    let encoded = encode(image, 75)?;
    ensure_baseline(&encoded)?;
    let back = decode(&encoded)?;
    let foreign = ffmpeg_raw(ffmpeg, &encoded, raw_pix(back.color)?)?;
    let round_match = same(&back.samples, &foreign) && same_samples(&back.samples, &image.samples);
    let matched = file_match
        && round_match
        && decoded.width == image.width
        && decoded.height == image.height
        && back.width == image.width
        && back.height == image.height;
    Ok(Outcome {
        width: decoded.width,
        height: decoded.height,
        matched,
    })
}

fn gray_frame() -> Result<StillImage, String> {
    let count = usize::try_from(WIDTH * HEIGHT).map_err(|_| "frame size".to_string())?;
    StillImage::from_samples(
        WIDTH,
        HEIGHT,
        ColorType::Gray,
        BitDepth::Eight,
        vec![LEVEL; count],
    )
    .checked()
    .map_err(|err| err.to_string())
}

fn params(quality: Option<u8>) -> CodecParams {
    CodecParams {
        codec: CodecId::Jpeg,
        extra: quality.map(|value| vec![value]).unwrap_or_default(),
        sample_rate: 0,
        layout: None,
        sample_format: None,
        bitrate: 0,
        width: 0,
        height: 0,
        pixel: PixelFormat::None,
        primaries: ColorPrimaries::Unspecified,
        transfer: Transfer::Unspecified,
        matrix: MatrixCoefficients::Unspecified,
        range: ColorRange::Unspecified,
        nal_length_size: 0,
    }
}

fn decode(bytes: &[u8]) -> Result<StillImage, String> {
    let mut decoder = JpegDecoder::open(&params(None)).map_err(|err| err.to_string())?;
    let packet = Packet::audio(bytes.to_vec(), 0, 0, 0);
    let pictures = decoder
        .decode_picture(&packet)
        .map_err(|err| err.to_string())?;
    let Some(picture) = pictures.into_iter().next() else {
        return Err("decoder returned no picture".to_string());
    };
    Ok(picture.image)
}

fn encode(image: &StillImage, quality: u8) -> Result<Vec<u8>, String> {
    let mut encoder = JpegEncoder::open(&params(Some(quality))).map_err(|err| err.to_string())?;
    let packets = encoder
        .push_picture(&PictureInput {
            image: image.clone(),
        })
        .map_err(|err| err.to_string())?;
    let Some(packet) = packets.into_iter().next() else {
        return Err("encoder returned no packet".to_string());
    };
    Ok(packet.data)
}

fn ensure_baseline(bytes: &[u8]) -> Result<(), String> {
    let info = read_jpeg_info(bytes).map_err(|err| err.to_string())?;
    if info.progressive || !bytes.windows(2).any(|marker| marker == [0xff, 0xc0]) {
        return Err("not a baseline jpeg".to_string());
    }
    Ok(())
}

fn raw_pix(color: ColorType) -> Result<&'static str, String> {
    match color {
        ColorType::Gray => Ok("gray"),
        ColorType::Rgb => Ok("rgb24"),
        _ => Err("unexpected jpeg color".to_string()),
    }
}

fn pack(samples: &[u16]) -> Result<Vec<u8>, String> {
    samples
        .iter()
        .map(|sample| u8::try_from(*sample).map_err(|_| "sample".to_string()))
        .collect()
}

fn same(samples: &[u16], raw: &[u8]) -> bool {
    samples.len() == raw.len()
        && samples
            .iter()
            .zip(raw.iter())
            .all(|(sample, byte)| *sample == u16::from(*byte))
}

fn same_samples(left: &[u16], right: &[u16]) -> bool {
    left == right
}

fn locate_ffmpeg() -> Option<PathBuf> {
    let homebrew = PathBuf::from("/opt/homebrew/bin/ffmpeg");
    if homebrew.is_file() {
        return Some(homebrew);
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).find_map(|dir| {
        let candidate = dir.join("ffmpeg");
        candidate.is_file().then_some(candidate)
    })
}

fn ffmpeg_jpeg(
    bin: &Path,
    raw: &[u8],
    width: u32,
    height: u32,
    pix: &str,
) -> Result<Vec<u8>, String> {
    let size = format!("{width}x{height}");
    ffmpeg_bytes(
        bin,
        &[
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix,
            "-s",
            &size,
            "-i",
            "pipe:0",
            "-frames:v",
            "1",
            "-pix_fmt",
            "yuvj420p",
            "-c:v",
            "mjpeg",
            "-q:v",
            "2",
            "-f",
            "image2pipe",
            "pipe:1",
        ],
        raw,
    )
}

fn ffmpeg_raw(bin: &Path, jpeg: &[u8], pix: &str) -> Result<Vec<u8>, String> {
    ffmpeg_bytes(
        bin,
        &[
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "image2pipe",
            "-i",
            "pipe:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix,
            "pipe:1",
        ],
        jpeg,
    )
}

fn ffmpeg_bytes(bin: &Path, args: &[&str], input: &[u8]) -> Result<Vec<u8>, String> {
    let mut child = Command::new(bin)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| err.to_string())?;
    let Some(mut stdin) = child.stdin.take() else {
        return Err("ffmpeg stdin".to_string());
    };
    stdin.write_all(input).map_err(|err| err.to_string())?;
    drop(stdin);
    let output = child.wait_with_output().map_err(|err| err.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "ffmpeg exited {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(output.stdout)
}
