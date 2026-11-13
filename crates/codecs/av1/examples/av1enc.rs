//! Encodes a Y4M file with the Rust AV1 encoder and writes IVF.
//!
//! Usage: av1enc <in.y4m> <out.ivf> [--q N | --kbps N] [--speed N] [--keyint N] [--frames N]
//!        [--tiles COLSLOG2xROWSLOG2] [--recon out.yuv] [--gop N] [--loop-restoration]
//!        [--global-motion] [--temporal-filter] [--threads N]
//!        [--force-key N[,N...]] [--color PRIMARIES:TRANSFER:MATRIX[:full]] [--av1c out.bin]
//!
//! `--force-key` makes the pictures at those display indices key frames,
//! `--color` writes those ISO/IEC 23091-2 codes into the sequence header, and
//! `--av1c` writes the stream's `av1C` record.

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::io::Write;

use cutlass_av1::dec::decoder::AnyPicture;
use cutlass_av1::enc::encoder::{Av1Enc, EncConfig, EncodedUnit, InputFrame, RateControl};

struct Y4m {
    w: usize,
    h: usize,
    bd: u32,
    sub_x: bool,
    sub_y: bool,
    mono: bool,
    fps: f64,
    /// The `F` ratio as written, for the IVF time base.
    rate: (u32, u32),
    data: Vec<u8>,
    pos: usize,
}

fn parse_y4m(data: Vec<u8>) -> Result<Y4m, String> {
    let end = data.iter().position(|&b| b == b'\n').ok_or("no header")?;
    let hdr = String::from_utf8_lossy(&data[..end]).to_string();
    if !hdr.starts_with("YUV4MPEG2") {
        return Err("not y4m".into());
    }
    let mut y = Y4m {
        w: 0,
        h: 0,
        bd: 8,
        sub_x: true,
        sub_y: true,
        mono: false,
        fps: 30.0,
        rate: (30, 1),
        data,
        pos: end + 1,
    };
    for tok in hdr.split_whitespace().skip(1) {
        let (k, v) = tok.split_at(1);
        match k {
            "W" => y.w = v.parse().map_err(|_| "W")?,
            "H" => y.h = v.parse().map_err(|_| "H")?,
            "F" => {
                let mut it = v.split(':');
                let n: u32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(30);
                let d: u32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(1);
                if n > 0 && d > 0 {
                    y.rate = (n, d);
                    y.fps = f64::from(n) / f64::from(d);
                }
            }
            "C" => {
                let c = v.to_string();
                y.bd = if c.contains("p10") {
                    10
                } else if c.contains("p12") {
                    12
                } else {
                    8
                };
                if c.starts_with("444") {
                    y.sub_x = false;
                    y.sub_y = false;
                } else if c.starts_with("422") {
                    y.sub_y = false;
                } else if c.starts_with("mono") {
                    y.mono = true;
                }
            }
            _ => {}
        }
    }
    Ok(y)
}

type Recon = Option<std::io::Result<std::io::BufWriter<std::fs::File>>>;

fn write_units(
    ivf: &mut Vec<u8>,
    recon: &mut Recon,
    units: Vec<EncodedUnit>,
    total: &mut usize,
) -> Result<(), String> {
    for u in units {
        ivf.extend_from_slice(&(u.data.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&u.display.to_le_bytes());
        ivf.extend_from_slice(&u.data);
        *total += u.data.len();
        if let Some(Ok(f)) = recon.as_mut()
            && let Some(pic) = u.recon.as_deref()
        {
            match pic {
                AnyPicture::U8(p) => {
                    for pl in &p.planes {
                        for r in 0..pl.h {
                            f.write_all(&pl.data[r * pl.stride..r * pl.stride + pl.w])
                                .map_err(|e| e.to_string())?;
                        }
                    }
                }
                AnyPicture::U16(p) => {
                    for pl in &p.planes {
                        for r in 0..pl.h {
                            for v in &pl.data[r * pl.stride..r * pl.stride + pl.w] {
                                f.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return Err("usage: av1enc <in.y4m> <out.ivf> [options]".into());
    }
    let mut y = parse_y4m(std::fs::read(&args[1]).map_err(|e| e.to_string())?)?;
    let mut cfg = EncConfig::new(y.w, y.h);
    cfg.bit_depth = y.bd;
    cfg.sub_x = y.sub_x;
    cfg.sub_y = y.sub_y;
    cfg.mono = y.mono;
    if let Some(q) = arg(&args, "--q") {
        cfg.rc = RateControl::Qindex(q.parse().map_err(|_| "q")?);
    }
    if let Some(k) = arg(&args, "--kbps") {
        let kbps: u64 = k.parse().map_err(|_| "kbps")?;
        cfg.rc = RateControl::Bitrate {
            bps: kbps * 1000,
            fps: y.fps,
        };
    }
    if let Some(s) = arg(&args, "--speed") {
        cfg.speed = s.parse().map_err(|_| "speed")?;
    }
    if args.iter().any(|a| a == "--global-motion") {
        cfg.global_motion = true;
    }
    if args.iter().any(|a| a == "--temporal-filter") {
        cfg.temporal_filter = true;
    }
    if args.iter().any(|a| a == "--loop-restoration") {
        cfg.loop_restoration = true;
    }
    if let Some(s) = arg(&args, "--threads") {
        cfg.threads = s.parse().map_err(|_| "threads")?;
    }
    if let Some(s) = arg(&args, "--gop") {
        cfg.gop = s.parse().map_err(|_| "gop")?;
    }
    if let Some(s) = arg(&args, "--keyint") {
        cfg.keyint = s.parse().map_err(|_| "keyint")?;
    }
    if let Some(t) = arg(&args, "--tiles") {
        let mut it = t.split('x');
        cfg.tile_cols_log2 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        cfg.tile_rows_log2 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    }
    if let Some(c) = arg(&args, "--color") {
        let mut it = c.split(':');
        let mut code = || it.next().and_then(|s| s.parse::<u8>().ok()).ok_or("color");
        cfg.color_primaries = code()?;
        cfg.transfer = code()?;
        cfg.matrix = code()?;
        cfg.full_range = it.next() == Some("full");
    }
    let forced: Vec<u64> = arg(&args, "--force-key")
        .map(|s| s.split(',').filter_map(|v| v.parse().ok()).collect())
        .unwrap_or_default();
    let max_frames: usize = arg(&args, "--frames")
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);
    let mut enc = Av1Enc::new(cfg).map_err(|e| e.to_string())?;
    if let Some(path) = arg(&args, "--av1c") {
        std::fs::write(path, enc.config_record()).map_err(|e| e.to_string())?;
    }
    let (cw, ch) = if y.mono {
        (0, 0)
    } else {
        (
            (y.w + usize::from(y.sub_x)) >> u32::from(y.sub_x),
            (y.h + usize::from(y.sub_y)) >> u32::from(y.sub_y),
        )
    };
    let bps = if y.bd > 8 { 2 } else { 1 };
    let frame_bytes = (y.w * y.h + 2 * cw * ch) * bps;
    let mut ivf = Vec::new();
    ivf.extend_from_slice(b"DKIF");
    ivf.extend_from_slice(&0u16.to_le_bytes());
    ivf.extend_from_slice(&32u16.to_le_bytes());
    ivf.extend_from_slice(b"AV01");
    ivf.extend_from_slice(&(y.w as u16).to_le_bytes());
    ivf.extend_from_slice(&(y.h as u16).to_le_bytes());
    // Packet timestamps count frames, so the time base is one frame.
    ivf.extend_from_slice(&y.rate.0.to_le_bytes());
    ivf.extend_from_slice(&y.rate.1.to_le_bytes());
    ivf.extend_from_slice(&0u32.to_le_bytes());
    ivf.extend_from_slice(&0u32.to_le_bytes());
    let mut recon =
        arg(&args, "--recon").map(|p| std::fs::File::create(p).map(std::io::BufWriter::new));
    let mut n = 0u64;
    let mut total = 0usize;
    let start = std::time::Instant::now();
    while y.pos < y.data.len() && (n as usize) < max_frames {
        let fstart = y.data[y.pos..]
            .iter()
            .position(|&b| b == b'\n')
            .ok_or("frame header")?
            + y.pos
            + 1;
        if fstart + frame_bytes > y.data.len() {
            break;
        }
        let buf = &y.data[fstart..fstart + frame_bytes];
        y.pos = fstart + frame_bytes;
        if forced.contains(&n) {
            enc.force_keyframe();
        }
        let pkt = if y.bd > 8 {
            let s: Vec<u16> = buf
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes(*c))
                .collect();
            let (py, rest) = s.split_at(y.w * y.h);
            let (pu, pv) = rest.split_at(cw * ch);
            enc.encode(&InputFrame {
                planes: [py, pu, pv],
                strides: [y.w, cw, cw],
            })
        } else {
            let (py, rest) = buf.split_at(y.w * y.h);
            let (pu, pv) = rest.split_at(cw * ch);
            enc.encode(&InputFrame {
                planes: [py, pu, pv],
                strides: [y.w, cw, cw],
            })
        }
        .map_err(|e| format!("frame {n}: {e}"))?;
        write_units(&mut ivf, &mut recon, pkt, &mut total)?;
        n += 1;
    }
    let rest = if y.bd > 8 {
        enc.flush::<u16>()
    } else {
        enc.flush::<u8>()
    }
    .map_err(|e| e.to_string())?;
    write_units(&mut ivf, &mut recon, rest, &mut total)?;
    ivf[24..28].copy_from_slice(&(n as u32).to_le_bytes());
    std::fs::write(&args[2], &ivf).map_err(|e| e.to_string())?;
    let secs = start.elapsed().as_secs_f64();
    eprintln!(
        "{n} frames, {total} bytes, {:.1} kbit/s at {:.2} fps input rate, {secs:.3}s ({:.2} fps)",
        total as f64 * 8.0 / 1000.0 / (n as f64 / y.fps),
        y.fps,
        n as f64 / secs
    );
    Ok(())
}
