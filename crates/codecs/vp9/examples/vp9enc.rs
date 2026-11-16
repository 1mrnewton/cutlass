//! Encodes raw video to an IVF file:
//! `vp9enc <in.yuv> <width> <height> <out.ivf> [q=N | rate=BITS] [speed=N] [frames=N] [fps=N] [keyint=N] [threads=N] [lag=N] [aq=0|1|3|4] [tile_rows=0..2] [cbr=BUFFER_MS] [twopass=1] [chroma=420|422|440|444] [bits=8|10|12]`
//! and the libvpx-like options `min_q=` `max_q=` (quantizer indices),
//! `crf=0..63` (a quality level: constant quality, or with `rate=`
//! constrained quality), `cq=QINDEX`, `sharpness=0..7`,
//! `colorspace=0..7` (VP9's codes; 7 is sRGB), `range=1` (full),
//! `render=WxH`, `error_resilient=1`, `frame_parallel=1`,
//! `static_thresh=N`, `alt_ref=0`, `arnr_strength=0..6`,
//! `arnr_frames=1..15`, `undershoot=PCT`, `overshoot=PCT`,
//! `keyint_min=N`, `max_intra=PCT`, `tile_columns=0..6`, `level=N`
//! (10 × the level) and `lossless=1`; `drop=PCT` (FFmpeg's
//! `-drop-threshold`: dropped frames leave gaps in the IVF timestamps),
//! `screen=1` (`-tune-content screen`), `ssim=1` (`-tune ssim`) and
//! `noise=0..6` (`-noise-sensitivity`).
//! With `bits=10` or `12` the input has little-endian 16-bit samples
//! (FFmpeg's `yuv420p10le` and the like).

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::io::{Read, Write};
use std::time::Instant;

use cutlass_vp9::{AqMode, Chroma, ColorSpace, Tune, TuneContent, Vp9Encoder, Vp9Options};

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
    let mut crf = None;
    for a in &args[5..] {
        let (k, v) = a.split_once('=').ok_or("option")?;
        if k == "render" {
            let (rw, rh) = v.split_once('x').ok_or("render=WxH")?;
            opts.render_size = Some((
                rw.parse().map_err(|_| "render width")?,
                rh.parse().map_err(|_| "render height")?,
            ));
            continue;
        }
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
            "aq" => match n {
                0 => opts.aq = false,
                1 => opts.aq = true,
                3 => opts.aq_mode = AqMode::CyclicRefresh,
                4 => opts.aq_mode = AqMode::Equator360,
                _ => return Err("aq=0|1|3|4".into()),
            },
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
            "min_q" => opts.min_q = n as u8,
            "max_q" => opts.max_q = n as u8,
            "crf" => crf = Some(Vp9Encoder::qindex_of_level(n as u8)),
            "cq" => opts.cq_level = Some(n as u8),
            "sharpness" => opts.sharpness = n as u8,
            "colorspace" => {
                opts.color_space = match n {
                    0 => ColorSpace::Unknown,
                    2 => ColorSpace::Bt709,
                    3 => ColorSpace::Smpte170,
                    4 => ColorSpace::Smpte240,
                    5 => ColorSpace::Bt2020,
                    7 => ColorSpace::Srgb,
                    _ => ColorSpace::Bt601,
                }
            }
            "range" => opts.full_range = n != 0,
            "error_resilient" => opts.error_resilient = n != 0,
            "frame_parallel" => opts.frame_parallel = n != 0,
            "static_thresh" => opts.static_thresh = n,
            "alt_ref" => opts.alt_ref = n != 0,
            "arnr_strength" => opts.arnr_strength = Some(n as u8),
            "arnr_frames" => opts.arnr_max_frames = Some(n as u8),
            "undershoot" => opts.undershoot_pct = n,
            "overshoot" => opts.overshoot_pct = n,
            "keyint_min" => opts.keyint_min = n,
            "max_intra" => opts.max_intra_pct = n,
            "tile_columns" => opts.tile_columns = Some(n as u8),
            "level" => opts.level = n as u8,
            "drop" => opts.drop_frame_pct = n,
            "noise" => opts.noise_sensitivity = n as u8,
            "ssim" => {
                if n != 0 {
                    opts.tune = Tune::Ssim;
                }
            }
            "screen" => {
                if n != 0 {
                    opts.tune_content = TuneContent::Screen;
                }
            }
            "lossless" => {
                if n != 0 {
                    opts.qindex = 0;
                }
            }
            _ => return Err(format!("unknown option {k}")),
        }
    }
    // FFmpeg's `-crf`: constant quality, or with a bitrate constrained.
    if let Some(q) = crf {
        if opts.bitrate > 0 {
            opts.cq_level = Some(q);
        } else {
            opts.qindex = q;
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
        // The first pass: the same pictures at a fixed quantizer near the
        // one the bitrate will take, fast.
        let first_q = enc.first_pass_qindex(1.0 / f64::from(fps));
        let first_opts = Vp9Options {
            bitrate: 0,
            qindex: first_q,
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
        enc.set_second_pass(&bits, first_q);
        input = std::fs::File::open(&args[1]).map_err(|e| e.to_string())?;
    }
    let mut packets = Vec::new();
    let start = Instant::now();
    let mut n = 0;
    // The pictures are read on a thread of their own, a few ahead of the
    // encoder (as FFmpeg's demuxer runs on its own thread), their buffers
    // handed back to be filled again.
    std::thread::scope(|sc| -> Result<(), String> {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(2);
        let (back_tx, back_rx) = std::sync::mpsc::channel::<Vec<u8>>();
        sc.spawn(move || {
            for _ in 0..frames {
                let mut b = back_rx.try_recv().unwrap_or_else(|_| vec![0u8; size]);
                if input.read_exact(&mut b).is_err() || tx.send(b).is_err() {
                    break;
                }
            }
        });
        for buf in rx {
            let clock = (n as i64, 1, fps);
            for (data, _, (pts, _, _)) in
                push(&mut enc, &buf, &mut wide, (w, h, cw, ch), fps, clock)?
            {
                packets.push((data, pts));
            }
            n += 1;
            let _ = back_tx.send(buf);
        }
        Ok(())
    })?;
    for (data, _, (pts, _, _)) in enc.flush().map_err(|e| e.to_string())? {
        packets.push((data, pts));
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
    for (p, pts) in &packets {
        out.write_all(&(p.len() as u32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        out.write_all(&(*pts as u64).to_le_bytes())
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
