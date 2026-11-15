//! Encodes raw video to an IVF file:
//! `vp9enc <in.yuv> <width> <height> <out.ivf> [q=N | rate=BITS] [speed=N] [frames=N] [fps=N] [keyint=N] [threads=N] [lag=N] [aq=0|1] [tile_rows=0..2] [cbr=BUFFER_MS] [twopass=1] [chroma=420|422|440|444] [bits=8|10|12]`.
//! With `bits=10` or `12` the input has little-endian 16-bit samples
//! (FFmpeg's `yuv420p10le` and the like).

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::io::{Read, Write};
use std::time::Instant;

use cutlass_vp9::{Chroma, Vp9Encoder, Vp9Options};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        return Err("usage: vp9enc <in.yuv> <width> <height> <out.ivf> [q=N|rate=BITS] [speed=N] [frames=N] [fps=N] [keyint=N] [threads=N] [lag=N] [aq=0|1] [tile_rows=0..2] [cbr=BUFFER_MS] [twopass=1] [chroma=420|422|440|444] [bits=8|10|12]".into());
    }
    let w: usize = args[2].parse().map_err(|_| "width")?;
    let h: usize = args[3].parse().map_err(|_| "height")?;
    let mut opts = Vp9Options::default();
    let mut frames = usize::MAX;
    let mut fps = 30u32;
    let mut twopass = false;
    let mut first_speed = 10u8;
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
            "threads" => opts.threads = n as usize,
            "lag" => opts.lag = Some(n),
            "aq" => opts.aq = n != 0,
            "tile_rows" => opts.tile_rows = n as u8,
            "cbr" => opts.cbr_buffer_ms = n,
            "twopass" => twopass = n != 0,
            "chroma" => {
                opts.chroma = match n {
                    422 => Chroma::Yuv422,
                    440 => Chroma::Yuv440,
                    444 => Chroma::Yuv444,
                    _ => Chroma::Yuv420,
                }
            }
            "first_speed" => first_speed = n as u8,
            "bits" => opts.bit_depth = n as u8,
            _ => return Err(format!("unknown option {k}")),
        }
    }
    let mut input = std::fs::File::open(&args[1]).map_err(|e| e.to_string())?;
    let mut out =
        std::io::BufWriter::new(std::fs::File::create(&args[4]).map_err(|e| e.to_string())?);
    let (cw, ch) = match opts.chroma {
        Chroma::Yuv420 => (w.div_ceil(2), h.div_ceil(2)),
        Chroma::Yuv422 => (w.div_ceil(2), h),
        Chroma::Yuv440 => (w, h.div_ceil(2)),
        Chroma::Yuv444 => (w, h),
    };
    let bytes = if opts.bit_depth > 8 { 2 } else { 1 };
    let size = (w * h + 2 * cw * ch) * bytes;
    let mut buf = vec![0u8; size];
    let mut wide = Vec::new();
    let mut enc = Vp9Encoder::with_options(w as u32, h as u32, opts).map_err(|e| e.to_string())?;
    if twopass {
        // The first pass: the same pictures at a fixed quantizer, fast.
        const FIRST_Q: u8 = 120;
        let first_opts = Vp9Options {
            bitrate: 0,
            qindex: FIRST_Q,
            speed: first_speed,
            ..opts
        };
        let mut first =
            Vp9Encoder::with_options(w as u32, h as u32, first_opts).map_err(|e| e.to_string())?;
        let (mut bits, mut n) = (Vec::new(), 0u32);
        while (n as usize) < frames && input.read_exact(&mut buf).is_ok() {
            let clock = (i64::from(n), 1, fps);
            for (data, _, _) in push(&mut first, &buf, &mut wide, (w, h, cw, ch), fps, clock)? {
                bits.push(data.len() as u64 * 8);
            }
            n += 1;
        }
        for (data, _, _) in first.flush().map_err(|e| e.to_string())? {
            bits.push(data.len() as u64 * 8);
        }
        enc.set_second_pass(&bits, FIRST_Q);
        input = std::fs::File::open(&args[1]).map_err(|e| e.to_string())?;
    }
    let mut packets = Vec::new();
    let start = Instant::now();
    let mut n = 0;
    while n < frames && input.read_exact(&mut buf).is_ok() {
        let clock = (n as i64, 1, fps);
        for (data, _, _) in push(&mut enc, &buf, &mut wide, (w, h, cw, ch), fps, clock)? {
            packets.push(data);
        }
        n += 1;
    }
    for (data, _, _) in enc.flush().map_err(|e| e.to_string())? {
        packets.push(data);
    }
    let secs = start.elapsed().as_secs_f64();
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
        out.write_all(&(p.len() as u32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        out.write_all(&(i as u64).to_le_bytes())
            .map_err(|e| e.to_string())?;
        out.write_all(p).map_err(|e| e.to_string())?;
        total += p.len();
    }
    let kbps = total as f64 * 8.0 / (n as f64 / f64::from(fps)) / 1000.0;
    println!(
        "{n} frames in {secs:.3}s ({:.1} fps), {total} bytes, {kbps:.0} kb/s",
        n as f64 / secs
    );
    Ok(())
}

/// The pictures coded on pushing one raw picture: 8-bit samples as they
/// are, 16-bit ones read little-endian into `wide`.
#[allow(clippy::type_complexity)]
fn push(
    enc: &mut Vp9Encoder,
    buf: &[u8],
    wide: &mut Vec<u16>,
    (w, h, cw, ch): (usize, usize, usize, usize),
    fps: u32,
    clock: (i64, i64, u32),
) -> Result<Vec<(Vec<u8>, bool, (i64, i64, u32))>, String> {
    let seconds = 1.0 / f64::from(fps);
    if buf.len() == w * h + 2 * cw * ch {
        let (y, rest) = buf.split_at(w * h);
        let (u, v) = rest.split_at(cw * ch);
        return enc
            .push_planes([(y, w), (u, cw), (v, cw)], seconds, clock)
            .map_err(|e| e.to_string());
    }
    wide.clear();
    wide.extend(
        buf.as_chunks::<2>()
            .0
            .iter()
            .map(|&b| u16::from_le_bytes(b)),
    );
    let (y, rest) = wide.split_at(w * h);
    let (u, v) = rest.split_at(cw * ch);
    enc.push_planes16([(y, w), (u, cw), (v, cw)], seconds, clock)
        .map_err(|e| e.to_string())
}
