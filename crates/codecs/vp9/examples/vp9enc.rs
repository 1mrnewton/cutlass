//! Encodes raw video to an IVF file:
//! `vp9enc <in.yuv> <width> <height> <out.ivf> [q=N | rate=BITS] [speed=N] [frames=N] [fps=N] [keyint=N] [threads=N] [lag=N] [aq=0|1|2|3|4] [tile_rows=0..2] [cbr=BUFFER_MS] [twopass=1] [chroma=420|422|440|444] [bits=8|10|12]`
//! and the libvpx-like options `min_q=` `max_q=` (quantizer indices),
//! `crf=0..63` (a quality level: constant quality, or with `rate=`
//! constrained quality), `cq=QINDEX`, `sharpness=0..7`,
//! `colorspace=0..7` (VP9's codes; 7 is sRGB), `range=1` (full),
//! `render=WxH`, `error_resilient=1`, `frame_parallel=1`,
//! `static_thresh=N`, `alt_ref=0`, `arf_layers=1..3` (layered alt-refs,
//! `-auto-alt-ref` past 1), `arnr_strength=0..6`,
//! `arnr_frames=1..15`, `undershoot=PCT`, `overshoot=PCT`,
//! `keyint_min=N`, `max_intra=PCT`, `tile_columns=0..6`, `level=N`
//! (10 × the level) and `lossless=1`; `drop=PCT` (FFmpeg's
//! `-drop-threshold`: dropped frames leave gaps in the IVF timestamps),
//! `screen=1` (`-tune-content screen`), `film=1` (`-tune-content film`),
//! `ssim=1` (`-tune ssim`) and
//! `noise=0..6` (`-noise-sensitivity`); `layers=2|3` (temporal layers),
//! with `keep_layers=N` writing only the packets of the lowest N; and a
//! region of interest map: `roi_q=N` (the middle half N quantizer
//! indices finer), `roi_ref=0..3` (its one reference), `roi_skip=1` (the
//! top eighth of the rows skipped); with `twopass=1`, `corpus=N` (the
//! corpus's complexity; the clip's own is printed).
//! With `bits=10` or `12` the input has little-endian 16-bit samples
//! (FFmpeg's `yuv420p10le` and the like).
//! `alpha=FILE` codes alpha (FFmpeg's `-pix_fmt yuva420p`): FILE holds one
//! 8-bit plane of the picture's size a picture (`-pix_fmt gray`), coded as
//! a second stream. An output that ends in `.webm` or `.mkv` is written as
//! WebM or Matroska, the alpha packets as each block's BlockAdditional
//! (BlockAddID 1, AlphaMode 1); an IVF output gets the alpha stream as a
//! second IVF beside it, `<out>.alpha.ivf`.

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::io::{Read, Write};
use std::time::Instant;

use cutlass_codec::{
    BlockAddition, CodecId, FrameSize, Packet, StreamCodec, StreamParams, Timescale, Timestamp,
};
use cutlass_container::{Mux, NewStream, TrackDuration};
use cutlass_matroska::MatroskaMux;
use cutlass_vp9::{AqMode, Chroma, ColorSpace, RoiMap, Tune, TuneContent, Vp9Encoder, Vp9Options};

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
    let mut keep_layers = u8::MAX;
    // A region of interest map: the middle half of the picture finer by
    // `roi_q` (segment 1, on reference `roi_ref` when given), the top
    // eighth of its rows skipped (segment 2) with `roi_skip=1`.
    let mut roi: (Option<i16>, Option<u8>, bool) = (None, None, false);
    let mut alpha_path = None;
    for a in &args[5..] {
        let (k, v) = a.split_once('=').ok_or("option")?;
        if k == "alpha" {
            alpha_path = Some(v.to_string());
            opts.alpha = true;
            continue;
        }
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
                2 => opts.aq_mode = AqMode::Complexity,
                3 => opts.aq_mode = AqMode::CyclicRefresh,
                4 => opts.aq_mode = AqMode::Equator360,
                _ => return Err("aq=0|1|2|3|4".into()),
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
            "arf_layers" => opts.alt_ref_layers = n as u8,
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
            "layers" => opts.temporal_layers = n as u8,
            "corpus" => opts.corpus_complexity = n,
            "keep_layers" => keep_layers = n as u8,
            "roi_q" => roi.0 = Some(n as i16),
            "roi_ref" => roi.1 = Some(n as u8),
            "roi_skip" => roi.2 = n != 0,
            "ssim" => {
                if n != 0 {
                    opts.tune = Tune::Ssim;
                }
            }
            "film" => {
                if n != 0 {
                    opts.tune_content = TuneContent::Film;
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
    if roi != (None, None, false) {
        let (mc, mr) = (w.div_ceil(8), h.div_ceil(8));
        let mut map = RoiMap {
            map: vec![0; mc * mr],
            ..RoiMap::default()
        };
        for r in 0..mr {
            for c in 0..mc {
                map.map[r * mc + c] = if roi.2 && r < mr / 8 {
                    2
                } else if (mr / 4..mr - mr / 4).contains(&r) && (mc / 4..mc - mc / 4).contains(&c) {
                    1
                } else {
                    0
                };
            }
        }
        map.delta_q[1] = -roi.0.unwrap_or(0);
        map.ref_frame[1] = roi.1;
        map.skip[2] = roi.2;
        enc.set_roi_map(Some(map)).map_err(|e| e.to_string())?;
    }
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
        // (The clip's complexity, for a corpus's `corpus=`.)
        eprintln!("complexity {}", enc.first_pass_complexity(&bits, first_q));
        enc.set_second_pass(&bits, first_q);
        input = std::fs::File::open(&args[1]).map_err(|e| e.to_string())?;
    }
    let mut alpha_in = match &alpha_path {
        Some(path) => Some(std::fs::File::open(path).map_err(|e| e.to_string())?),
        None => None,
    };
    let mut alpha_buf = vec![0u8; w * h];
    // (Each packet's alpha packet, with `alpha=`.)
    let mut alphas: Vec<Vec<u8>> = Vec::new();
    let mut keys: Vec<bool> = Vec::new();
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
            if let Some(file) = alpha_in.as_mut() {
                file.read_exact(&mut alpha_buf)
                    .map_err(|e| format!("alpha: {e}"))?;
                let (y, rest) = buf.split_at(w * h);
                let (u, v) = rest.split_at(cw * ch);
                let planes = [(y, w), (u, cw), (v, cw), (&alpha_buf[..], w)];
                for c in enc
                    .push_planes_alpha(planes, 1.0 / f64::from(fps), clock)
                    .map_err(|e| e.to_string())?
                {
                    packets.push((c.data, c.clock.0));
                    alphas.push(c.alpha);
                    keys.push(c.keyframe);
                }
                n += 1;
                let _ = back_tx.send(buf);
                continue;
            }
            for (data, key, (pts, _, _)) in
                push(&mut enc, &buf, &mut wide, (w, h, cw, ch), fps, clock)?
            {
                if enc.temporal_layer() < keep_layers {
                    packets.push((data, pts));
                    keys.push(key);
                }
            }
            n += 1;
            let _ = back_tx.send(buf);
        }
        Ok(())
    })?;
    if opts.alpha {
        for c in enc.flush_alpha().map_err(|e| e.to_string())? {
            packets.push((c.data, c.clock.0));
            alphas.push(c.alpha);
            keys.push(c.keyframe);
        }
    } else {
        for (data, key, (pts, _, _)) in enc.flush().map_err(|e| e.to_string())? {
            packets.push((data, pts));
            keys.push(key);
        }
    }
    let secs = start.elapsed().as_secs_f64();
    let out_path = args[4].to_ascii_lowercase();
    if out_path.ends_with(".webm") || out_path.ends_with(".mkv") {
        drop(out);
        let file = std::fs::File::create(&args[4]).map_err(|e| e.to_string())?;
        let total = write_matroska(
            file,
            out_path.ends_with(".webm"),
            (w, h, fps),
            &packets,
            &keys,
            &alphas,
        )?;
        report(n, secs, total, &alphas, fps);
        return Ok(());
    }
    if !alphas.is_empty() {
        let pairs: Vec<(Vec<u8>, i64)> = alphas
            .iter()
            .zip(&packets)
            .map(|(a, (_, pts))| (a.clone(), *pts))
            .collect();
        let mut file = std::io::BufWriter::new(
            std::fs::File::create(format!("{}.alpha.ivf", args[4])).map_err(|e| e.to_string())?,
        );
        write_ivf(&mut file, (w, h, fps), &pairs)?;
    }
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
    report(n, secs, total, &alphas, fps);
    Ok(())
}

/// Prints the frames, time, bytes and rate (and the alpha stream's).
fn report(n: usize, secs: f64, total: usize, alphas: &[Vec<u8>], fps: u32) {
    let kbps = total as f64 * 8.0 / (n as f64 / f64::from(fps)) / 1000.0;
    println!(
        "{n} frames in {secs:.3}s ({:.1} fps), {total} bytes, {kbps:.0} kb/s",
        n as f64 / secs
    );
    if !alphas.is_empty() {
        let bytes: usize = alphas.iter().map(Vec::len).sum();
        let kbps = bytes as f64 * 8.0 / (n as f64 / f64::from(fps)) / 1000.0;
        println!("alpha: {bytes} bytes, {kbps:.0} kb/s");
    }
}

/// An IVF file of `packets` (data, pts).
fn write_ivf(
    out: &mut impl Write,
    (w, h, fps): (usize, usize, u32),
    packets: &[(Vec<u8>, i64)],
) -> Result<(), String> {
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
    for (p, pts) in packets {
        out.write_all(&(p.len() as u32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        out.write_all(&(*pts as u64).to_le_bytes())
            .map_err(|e| e.to_string())?;
        out.write_all(p).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// A WebM or Matroska file of the packets (data, pts at `fps`), each with
/// its alpha packet as BlockAdditional 1 when there are alpha packets;
/// returns the packets' bytes.
fn write_matroska(
    file: std::fs::File,
    webm: bool,
    (w, h, fps): (usize, usize, u32),
    packets: &[(Vec<u8>, i64)],
    keys: &[bool],
    alphas: &[Vec<u8>],
) -> Result<usize, String> {
    let writer = std::io::BufWriter::with_capacity(1 << 20, file);
    let mut mux = if webm {
        MatroskaMux::webm(writer)
    } else {
        MatroskaMux::new(writer)
    };
    let size = FrameSize::try_new(w as u32, h as u32).map_err(|e| e.to_string())?;
    let track = mux
        .add_stream(NewStream {
            params: StreamParams::video(StreamCodec::Known(CodecId::Vp9), size, Vec::new()),
            timescale: Timescale::new(fps),
            language: None,
        })
        .map_err(|e| e.to_string())?;
    let mut total = 0;
    for (i, (data, pts)) in packets.iter().enumerate() {
        let time =
            Timestamp::try_new(Some(*pts), Some(*pts), Some(1)).map_err(|e| e.to_string())?;
        let mut packet = Packet::try_new(
            track.get(),
            data.clone(),
            time,
            Timescale::new(fps),
            keys[i],
        )
        .map_err(|e| e.to_string())?;
        if let Some(alpha) = alphas.get(i) {
            packet = packet.with_additions(vec![BlockAddition {
                id: 1,
                data: alpha.clone(),
            }]);
        }
        mux.write_packet(&packet).map_err(|e| e.to_string())?;
        total += data.len();
    }
    mux.finish(&[TrackDuration {
        track,
        timescale: Timescale::new(fps),
        ticks: packets.len() as i64,
    }])
    .map_err(|e| format!("{e:?}"))?;
    Ok(total)
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
