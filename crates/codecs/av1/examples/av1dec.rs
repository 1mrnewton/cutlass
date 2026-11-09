//! Decodes an IVF file with the Rust AV1 decoder and writes raw planes.
//!
//! Usage: av1dec <in.ivf> <out.yuv>
//! 8-bit samples are written as bytes, higher bit depths as little-endian u16.
//! Monochrome output writes only luma.

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::io::Write;

use cutlass_av1::dec::decoder::{AnyPicture, Av1Core};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return Err("usage: av1dec <in.ivf> <out.yuv>".into());
    }
    let data = std::fs::read(&args[1]).map_err(|e| e.to_string())?;
    if data.len() < 32 || &data[0..4] != b"DKIF" {
        return Err("not an IVF file".into());
    }
    let hdr_len = u16::from_le_bytes([data[6], data[7]]) as usize;
    let mut pos = hdr_len;
    let mut out =
        std::io::BufWriter::new(std::fs::File::create(&args[2]).map_err(|e| e.to_string())?);
    let mut core = Av1Core::new();
    let mut n = 0;
    let start = std::time::Instant::now();
    while pos + 12 <= data.len() {
        let size =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 12;
        let frame = &data[pos..(pos + size).min(data.len())];
        pos += size;
        let frames = core.decode(frame).map_err(|e| format!("frame {n}: {e}"))?;
        for f in frames {
            match &*f.picture {
                AnyPicture::U8(p) => {
                    for pl in &p.planes {
                        for y in 0..pl.h {
                            out.write_all(&pl.data[y * pl.stride..y * pl.stride + pl.w])
                                .map_err(|e| e.to_string())?;
                        }
                    }
                }
                AnyPicture::U16(p) => {
                    for pl in &p.planes {
                        for y in 0..pl.h {
                            for v in &pl.data[y * pl.stride..y * pl.stride + pl.w] {
                                out.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
                            }
                        }
                    }
                }
            }
            n += 1;
        }
    }
    eprintln!("{n} frames in {:.3}s", start.elapsed().as_secs_f64());
    #[cfg(feature = "stats")]
    eprintln!("{}", cutlass_av1::dec::stats::report());
    Ok(())
}
