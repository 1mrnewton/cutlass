//! Times the decoder alone on back-to-back ProRes frames (ffmpeg's
//! `-map 0:v -c copy -f data` output), with no container and no file output.
//!
//! `cargo run --release -p cutlass-prores --example decode_speed -- frames.bin [threads] [passes] [picture|wide]`
//!
//! `picture` is the app's path, 8-bit 4:2:0; `wide` is the full-depth
//! frame, handed back to the decoder after each one. Zero threads means
//! the shared pool.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::process::ExitCode;
use std::time::Instant;

use cutlass_decode::{CodecId, CodecParams, Decoder, Packet, PixelFormat};
use cutlass_prores::{ProresDecoder, ProresPictureDecoder};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let Some(input) = args.get(1) else {
        eprintln!("usage: decode_speed frames.bin [threads] [passes] [picture|wide]");
        return ExitCode::FAILURE;
    };
    let number = |at: usize, default: usize| {
        args.get(at)
            .and_then(|text| text.parse().ok())
            .unwrap_or(default)
    };
    let threads = number(2, 0);
    let passes = number(3, 1).max(1);
    let wide = args.get(4).is_some_and(|mode| mode == "wide");
    let Ok(bytes) = std::fs::read(input) else {
        eprintln!("cannot read {input}");
        return ExitCode::FAILURE;
    };
    let mut frames = Vec::new();
    let mut rest = &bytes[..];
    while rest.len() >= 8 {
        let size = u32::from_be_bytes([rest[0], rest[1], rest[2], rest[3]]) as usize;
        if size < 8 || size > rest.len() {
            eprintln!("bad frame size {size}");
            return ExitCode::FAILURE;
        }
        frames.push(Packet::video(rest[..size].to_vec(), 0, 1, 30, true));
        rest = &rest[size..];
    }
    let params = CodecParams::with_format(CodecId::Prores, 0, 0, PixelFormat::None);
    let opened = ProresDecoder::open(&params).map(|decoder| {
        if threads == 0 {
            decoder
        } else {
            decoder.with_threads(threads)
        }
    });
    let Ok(mut decoder) = opened else {
        return ExitCode::FAILURE;
    };
    let mut picture = ProresPictureDecoder::from(match ProresDecoder::open(&params) {
        Ok(decoder) if threads == 0 => decoder,
        Ok(decoder) => decoder.with_threads(threads),
        Err(_) => return ExitCode::FAILURE,
    });
    let start = Instant::now();
    let mut seen = 0u64;
    for _ in 0..passes {
        for packet in &frames {
            if wide {
                match decoder.decode(&packet.data) {
                    Ok(decoded) => {
                        seen += u64::from(decoded.frame.y.samples[0]);
                        decoder.recycle(decoded.frame);
                    }
                    Err(error) => {
                        eprintln!("{error}");
                        return ExitCode::FAILURE;
                    }
                }
            } else {
                match picture.push(packet) {
                    Ok(outputs) => seen += outputs.len() as u64,
                    Err(error) => {
                        eprintln!("{error}");
                        return ExitCode::FAILURE;
                    }
                }
            }
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    let count = frames.len() * passes;
    println!(
        "{count} frames in {seconds:.3} s: {:.1} frames/s, {:.3} ms a frame ({seen})",
        count as f64 / seconds,
        seconds * 1000.0 / count as f64
    );
    ExitCode::SUCCESS
}
