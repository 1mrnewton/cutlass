//! Times the encoder alone on raw pictures, with no container and no file
//! output.
//!
//! `cargo run --release -p cutlass-prores --example encode_speed -- raw width height format profile [threads] [passes]`
//!
//! `format` is `yuv422p10`, `yuv444p12`, `yuv420p` or `rgba`, as ffmpeg
//! writes them with `-f rawvideo` (16-bit samples little-endian). `profile`
//! is a sample-entry code such as `apch`. Zero threads means the shared pool.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::process::ExitCode;
use std::time::Instant;

use cutlass_decode::{CodecId, CodecParams, PixelFormat, Plane, VideoFrame};
use cutlass_prores::ProresEncoder;

fn main() -> ExitCode {
    match run() {
        Ok(line) => {
            println!("{line}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<String, String> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: encode_speed raw width height yuv422p10|yuv444p12|yuv420p|rgba profile [threads] [passes]";
    let [_, input, width, height, format, profile, rest @ ..] = args.as_slice() else {
        return Err(usage.to_owned());
    };
    let width: u32 = width.parse().map_err(|_| usage.to_owned())?;
    let height: u32 = height.parse().map_err(|_| usage.to_owned())?;
    let number =
        |at: usize, default: usize| rest.get(at).and_then(|t| t.parse().ok()).unwrap_or(default);
    let threads = number(0, 0);
    let passes = number(1, 1).max(1);
    let bytes = std::fs::read(input).map_err(|error| format!("{input}: {error}"))?;
    let mut params = CodecParams::with_format(CodecId::Prores, 0, 0, PixelFormat::None);
    params.extra = profile.as_bytes().to_vec();
    let mut encoder = ProresEncoder::open(&params).map_err(|error| error.to_string())?;
    if threads > 0 {
        encoder = encoder.with_threads(threads);
    }
    let planar = match format.as_str() {
        "yuv422p10" => Some((PixelFormat::Yuv422p10, 2)),
        "yuv444p12" => Some((PixelFormat::Yuv444p12, 2)),
        "yuv420p" => Some((PixelFormat::Yuv420p, 1)),
        "rgba" => None,
        _ => return Err(usage.to_owned()),
    };
    let mut frames = Vec::new();
    let mut pictures = Vec::new();
    if let Some((pixel, sample)) = planar {
        let (cw, ch) = pixel.chroma_wh(width, height).ok_or("chroma size")?;
        let luma = (width * height) as usize * sample;
        let chroma = (cw * ch) as usize * sample;
        for frame in bytes.chunks_exact(luma + 2 * chroma) {
            let plane = |at: usize, len: usize, w: u32, h: u32| {
                Plane::bytes(w, h, sample, frame[at..at + len].to_vec())
                    .map_err(|error| error.to_string())
            };
            frames.push(
                VideoFrame::planar(
                    pixel,
                    width,
                    height,
                    plane(0, luma, width, height)?,
                    plane(luma, chroma, cw, ch)?,
                    plane(luma + chroma, chroma, cw, ch)?,
                )
                .map_err(|error| error.to_string())?,
            );
        }
    } else {
        pictures = bytes
            .chunks_exact((width * height * 4) as usize)
            .collect::<Vec<_>>();
    }
    let count = (frames.len() + pictures.len()) * passes;
    if count == 0 {
        return Err("no whole picture in the file".to_owned());
    }
    let start = Instant::now();
    let mut coded = 0usize;
    for _ in 0..passes {
        for frame in &frames {
            coded += encoder
                .encode(frame)
                .map_err(|error| error.to_string())?
                .len();
        }
        for picture in &pictures {
            coded += encoder
                .encode_rgba(width, height, picture)
                .map_err(|error| error.to_string())?
                .len();
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    Ok(format!(
        "{count} frames in {seconds:.3} s: {:.1} frames/s, {:.3} ms a frame, {} bytes a frame",
        count as f64 / seconds,
        seconds * 1000.0 / count as f64,
        coded / count
    ))
}
