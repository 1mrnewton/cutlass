//! Decodes the VP9 track of a WebM or Matroska file to raw video:
//! `vp9dec <in.webm> <out.yuv> [threads=N]`.
//! A track with alpha (AlphaMode 1: each block's BlockAdditional of
//! BlockAddID 1, as FFmpeg's `libvpx-vp9` writes `yuva420p`) is written as
//! `yuva420p` (the Y, U, V and alpha planes of each picture); one without
//! as `yuv420p`, as FFmpeg's `-f rawvideo` writes them. 8-bit 4:2:0 only.

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::io::Write;

use cutlass_container::Demux;
use cutlass_decode::video::{VideoDecoder, VideoOutput};
use cutlass_decode::{CodecId, CodecParams, Packet, PixelFormat, SideData};
use cutlass_matroska::MatroskaDemux;
use cutlass_vp9::Vp9Decoder;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return Err("usage: vp9dec <in.webm> <out.yuv> [threads=N]".into());
    }
    let mut threads = 1;
    for a in &args[3..] {
        match a.split_once('=') {
            Some(("threads", n)) => threads = n.parse().map_err(|_| "threads=N")?,
            _ => return Err(format!("unknown option {a}")),
        }
    }
    let file = std::fs::File::open(&args[1]).map_err(|e| e.to_string())?;
    let length = file.metadata().map_err(|e| e.to_string())?.len();
    let mut demux = MatroskaDemux::open(file, length).map_err(|e| e.to_string())?;
    let entry = demux
        .track_entries()
        .iter()
        .find(|t| t.codec_id == b"V_VP9")
        .ok_or("no VP9 track")?;
    let (track, alpha) = (entry.number, entry.alpha_mode == 1);
    let (w, h) = (entry.pixel_width as u32, entry.pixel_height as u32);
    let params = CodecParams::with_format(CodecId::Vp9, w, h, PixelFormat::Yuv420p);
    let mut decoder = Vp9Decoder::open_with_threads(&params, threads).map_err(|e| e.to_string())?;
    let mut out =
        std::io::BufWriter::new(std::fs::File::create(&args[2]).map_err(|e| e.to_string())?);
    let mut frames = 0;
    let mut write = |shown: Vec<VideoOutput>| -> Result<(), String> {
        for o in shown {
            let f = &o.frame;
            if f.format != PixelFormat::Yuv420p {
                return Err("not 8-bit 4:2:0".into());
            }
            for plane in [&f.y, &f.u, &f.v] {
                out.write_all(&plane.packed(1)).map_err(|e| e.to_string())?;
            }
            if alpha {
                // (A picture whose alpha did not decode is opaque.)
                let a = match &f.alpha {
                    Some(a) => a.plane.packed(1),
                    None => vec![255; f.width as usize * f.height as usize],
                };
                out.write_all(&a).map_err(|e| e.to_string())?;
            }
            frames += 1;
        }
        Ok(())
    };
    while let Some(p) = demux.read_packet().map_err(|e| e.to_string())? {
        if u64::from(p.stream()) != track {
            continue;
        }
        let time = p.timestamp();
        let mut packet = Packet::video(
            p.bytes().to_vec(),
            time.presentation_or_decode().unwrap_or(0),
            time.duration().unwrap_or(0),
            p.timescale().ticks_per_second(),
            p.keyframe(),
        );
        packet
            .side
            .extend(p.additions().iter().map(|a| SideData::BlockAdditional {
                id: a.id,
                data: a.data.clone(),
            }));
        write(decoder.push(&packet).map_err(|e| e.to_string())?)?;
    }
    write(decoder.drain().map_err(|e| e.to_string())?)?;
    println!(
        "{frames} frames, {w}x{h}, {}",
        if alpha { "yuva420p" } else { "yuv420p" }
    );
    Ok(())
}
