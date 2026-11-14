//! Encodes raw 4:2:0 video to an IVF file:
//! `vp9enc <in.yuv> <width> <height> <out.ivf> [q=N | rate=BITS] [speed=N] [frames=N] [fps=N] [keyint=N]`.

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::io::{Read, Write};
use std::time::Instant;

use cutlass_vp9::{Vp9Encoder, Vp9Options};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        return Err("usage: vp9enc <in.yuv> <width> <height> <out.ivf> [q=N|rate=BITS] [speed=N] [frames=N] [fps=N] [keyint=N]".into());
    }
    let w: usize = args[2].parse().map_err(|_| "width")?;
    let h: usize = args[3].parse().map_err(|_| "height")?;
    let mut opts = Vp9Options::default();
    let mut frames = usize::MAX;
    let mut fps = 30u32;
    for a in &args[5..] {
        let (k, v) = a.split_once('=').ok_or("option")?;
        let n: u32 = v.parse().map_err(|_| "option value")?;
        match k {
            "q" => opts.qindex = n as u8,
            "rate" => opts.bitrate = n,
            "speed" => opts.speed = n as u8,
            "frames" => frames = n as usize,
            "fps" => fps = n,
            "keyint" => opts.keyint = n,
            _ => return Err(format!("unknown option {k}")),
        }
    }
    let mut input = std::fs::File::open(&args[1]).map_err(|e| e.to_string())?;
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args[4]).map_err(|e| e.to_string())?);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let size = w * h + 2 * cw * ch;
    let mut enc = Vp9Encoder::with_options(w as u32, h as u32, opts).map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; size];
    let mut packets = Vec::new();
    let start = Instant::now();
    let mut n = 0;
    while n < frames && input.read_exact(&mut buf).is_ok() {
        let (y, rest) = buf.split_at(w * h);
        let (u, v) = rest.split_at(cw * ch);
        let (data, _) = enc.encode_planes([(y, w), (u, cw), (v, cw)], 1.0 / f64::from(fps)).map_err(|e| e.to_string())?;
        packets.push(data);
        n += 1;
    }
    let secs = start.elapsed().as_secs_f64();
    cutlass_vp9::enc_stats();
    let mut hdr = Vec::new();
    hdr.extend_from_slice(b"DKIF");
    hdr.extend_from_slice(&0u16.to_le_bytes());
    hdr.extend_from_slice(&32u16.to_le_bytes());
    hdr.extend_from_slice(b"VP90");
    hdr.extend_from_slice(&(w as u16).to_le_bytes());
    hdr.extend_from_slice(&(h as u16).to_le_bytes());
    hdr.extend_from_slice(&fps.to_le_bytes());
    hdr.extend_from_slice(&1u32.to_le_bytes());
    hdr.extend_from_slice(&(packets.len() as u32).to_le_bytes());
    hdr.extend_from_slice(&0u32.to_le_bytes());
    out.write_all(&hdr).map_err(|e| e.to_string())?;
    let mut total = 0;
    for (i, p) in packets.iter().enumerate() {
        out.write_all(&(p.len() as u32).to_le_bytes()).map_err(|e| e.to_string())?;
        out.write_all(&(i as u64).to_le_bytes()).map_err(|e| e.to_string())?;
        out.write_all(p).map_err(|e| e.to_string())?;
        total += p.len();
    }
    let kbps = total as f64 * 8.0 / (n as f64 / f64::from(fps)) / 1000.0;
    println!("{n} frames in {secs:.3}s ({:.1} fps), {total} bytes, {kbps:.0} kb/s", n as f64 / secs);
    Ok(())
}
