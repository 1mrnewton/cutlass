//! Decodes back-to-back ProRes frames (ffmpeg's `-c copy -f data` output)
//! to raw planes: Y, Cb, Cr, then alpha, 16-bit little-endian.
//!
//! `cargo run --release -p cutlass-prores --example decode_raw -- frames.bin out.raw [10|12]`
#![allow(clippy::print_stderr)]

use std::process::ExitCode;

use cutlass_decode::{CodecId, CodecParams, PixelFormat};
use cutlass_prores::ProresDecoder;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let (Some(input), Some(output)) = (args.get(1), args.get(2)) else {
        eprintln!("usage: decode_raw frames.bin out.raw [10|12]");
        return ExitCode::FAILURE;
    };
    let Ok(bytes) = std::fs::read(input) else {
        eprintln!("cannot read {input}");
        return ExitCode::FAILURE;
    };
    let params = CodecParams::with_format(CodecId::Prores, 0, 0, PixelFormat::None);
    let Ok(mut decoder) = ProresDecoder::open(&params) else {
        return ExitCode::FAILURE;
    };
    if let Some(depth) = args.get(3).and_then(|d| d.parse::<u8>().ok()) {
        decoder = match decoder.with_depth(depth) {
            Ok(decoder) => decoder,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::FAILURE;
            }
        };
    }
    let mut out = Vec::new();
    let mut rest = &bytes[..];
    let mut count = 0;
    while rest.len() >= 8 {
        let size = u32::from_be_bytes([rest[0], rest[1], rest[2], rest[3]]) as usize;
        if size < 8 || size > rest.len() {
            eprintln!("bad frame size {size}");
            return ExitCode::FAILURE;
        }
        match decoder.decode(&rest[..size]) {
            Ok(decoded) => {
                let frame = decoded.frame;
                for plane in [&frame.y, &frame.u, &frame.v] {
                    out.extend_from_slice(&plane.samples);
                }
                if let Some(alpha) = frame.alpha {
                    if alpha.bit_depth > 8 {
                        out.extend_from_slice(&alpha.plane.samples);
                    } else {
                        out.extend(
                            alpha
                                .plane
                                .samples
                                .iter()
                                .flat_map(|&a| u16::from(a).to_le_bytes()),
                        );
                    }
                }
            }
            Err(err) => {
                eprintln!("frame {count}: {err}");
                return ExitCode::FAILURE;
            }
        }
        rest = &rest[size..];
        count += 1;
    }
    if std::fs::write(output, out).is_err() {
        return ExitCode::FAILURE;
    }
    eprintln!("{count} frames");
    ExitCode::SUCCESS
}
