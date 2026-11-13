//! Decodes an HEVC stream with the Rust decoder and writes raw planes.
//!
//! Usage: hevcdec <in.mp4|in.mov|in.hevc> <out.yuv> [threads]
//!
//! The input is the first HEVC track of an MP4 or QuickTime file, or an
//! Annex B byte stream. Pictures are written in output order, cropped,
//! as planar 4:2:0: bytes for 8-bit streams, little-endian 16-bit values
//! for deeper ones (as `ffmpeg -f rawvideo` writes `yuv420p` or
//! `yuv420p10le`).

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::io::Write;

use cutlass_decode::{Packet, VideoDecoder, VideoOutput};
use cutlass_hevc::HevcDecoder;

mod common;

fn write(out: &mut impl Write, picture: &VideoOutput) -> std::io::Result<()> {
    if let Some(p) = &picture.yuv420p10 {
        for plane in [p.y(), p.u(), p.v()] {
            for v in plane {
                out.write_all(&v.to_le_bytes())?;
            }
        }
        return Ok(());
    }
    if let Some(p) = &picture.yuv420_high {
        for plane in [p.y(), p.u(), p.v()] {
            for v in plane {
                out.write_all(&v.to_le_bytes())?;
            }
        }
        return Ok(());
    }
    let p = &picture.picture;
    for plane in [p.y(), p.u(), p.v()] {
        out.write_all(plane)?;
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return Err("usage: hevcdec <in.mp4|in.mov|in.hevc> <out.yuv> [threads]".into());
    }
    let (params, units) = common::read(&args[1])?;
    let mut decoder = match args.get(3) {
        Some(t) => {
            let threads = t.parse().map_err(|_| format!("threads: {t}"))?;
            HevcDecoder::open_with_threads(&params, threads)
        }
        None => HevcDecoder::open(&params),
    }
    .map_err(|e| format!("{e:?}"))?;
    let mut out =
        std::io::BufWriter::new(std::fs::File::create(&args[2]).map_err(|e| e.to_string())?);
    let start = std::time::Instant::now();
    let mut pictures = 0usize;
    for (index, unit) in units.iter().enumerate() {
        let packet = Packet::video(unit.clone(), index as i64, 1, 30, false);
        let ready = decoder
            .push(&packet)
            .map_err(|e| format!("access unit {index}: {e:?}"))?;
        for picture in &ready {
            write(&mut out, picture).map_err(|e| e.to_string())?;
        }
        pictures += ready.len();
    }
    for picture in &decoder.drain().map_err(|e| format!("drain: {e:?}"))? {
        write(&mut out, picture).map_err(|e| e.to_string())?;
        pictures += 1;
    }
    out.flush().map_err(|e| e.to_string())?;
    eprintln!(
        "{pictures} pictures from {} access units in {:.3} s",
        units.len(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
