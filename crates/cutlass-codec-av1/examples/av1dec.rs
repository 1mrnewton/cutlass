//! Decode a low-overhead or Annex B AV1 file and print frame geometry.
//!
//! IVF is accepted only as a length prefix around each temporal unit. The
//! decoder itself never parses IVF. `--dump` writes tight planar samples in
//! plane order: one byte per sample at 8-bit, or little-endian `u16` per
//! sample at 10/12-bit. Allocation padding is not written.

use std::env;
use std::fs;
use std::process::ExitCode;

use cutlass_codec_av1::{
    Av1Decoder, DecodedFrame, DecoderConfig, FilmGrainMode, InputFraming, PacketTiming,
    ReceiveStatus, SubmitStatus,
};
use cutlass_core::{FrameData, Plane};

fn main() -> ExitCode {
    let mut framing = InputFraming::LowOverhead;
    let mut grain = FilmGrainMode::Apply;
    let mut dump: Option<String> = None;
    let mut file: Option<String> = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--annexb" => framing = InputFraming::AnnexB,
            "--grain-off" => grain = FilmGrainMode::Reconstruction,
            "--dump" => {
                dump = match args.next() {
                    Some(path) => Some(path),
                    None => {
                        eprintln!("usage: av1dec [--annexb] [--grain-off] [--dump out.raw] <file>");
                        return ExitCode::from(2);
                    }
                };
            }
            other if file.is_none() => file = Some(other.to_string()),
            _ => {
                eprintln!("usage: av1dec [--annexb] [--grain-off] [--dump out.raw] <file>");
                return ExitCode::from(2);
            }
        }
    }
    let Some(file) = file else {
        eprintln!("usage: av1dec [--annexb] [--grain-off] [--dump out.raw] <file>");
        return ExitCode::from(2);
    };
    let bytes = match fs::read(&file) {
        Ok(bytes) => bytes,
        Err(err) => {
            eprintln!("read {file}: {err}");
            return ExitCode::from(1);
        }
    };
    let packets: Vec<&[u8]> = if bytes.len() >= 32 && &bytes[..4] == b"DKIF" {
        match ivf_frames(&bytes) {
            Ok(frames) => frames,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        }
    } else {
        vec![bytes.as_slice()]
    };
    let config = DecoderConfig {
        framing,
        grain,
        ..DecoderConfig::default()
    };
    let mut dec = match Av1Decoder::new(config) {
        Ok(dec) => dec,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let mut dumped = Vec::new();
    let mut frame_index = 0usize;
    for packet in packets {
        match dec.submit(packet, PacketTiming::default()) {
            Ok(SubmitStatus::Accepted) => {}
            Ok(SubmitStatus::Blocked) => {
                eprintln!("output queue is full");
                return ExitCode::from(1);
            }
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        }
        loop {
            match dec.receive() {
                Ok(ReceiveStatus::Frame(frame)) => {
                    let info = &frame.info;
                    println!(
                        "frame {frame_index} {}x{} coded {}x{} render {}x{} depth {} grain {}",
                        frame.frame.width(),
                        frame.frame.height(),
                        info.coded_size.0,
                        info.coded_size.1,
                        info.render_size.0,
                        info.render_size.1,
                        info.bit_depth,
                        info.grain_applied
                    );
                    if dump.is_some() {
                        match tight_planes(&frame) {
                            Ok(bytes) => dumped.extend(bytes),
                            Err(err) => {
                                eprintln!("{err}");
                                return ExitCode::from(1);
                            }
                        }
                    }
                    frame_index += 1;
                }
                Ok(ReceiveStatus::NeedInput) => break,
                Ok(ReceiveStatus::Drained) => break,
                Err(err) => {
                    eprintln!("{err}");
                    return ExitCode::from(1);
                }
            }
        }
    }
    if let Some(path) = dump
        && let Err(err) = fs::write(&path, dumped)
    {
        eprintln!("write {path}: {err}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

/// Visible samples only. 8-bit is one byte per sample. 10/12-bit storage is
/// native-endian `u16`; the file is little-endian `u16` so a byte compare does
/// not depend on the host.
fn tight_planes(decoded: &DecodedFrame) -> Result<Vec<u8>, String> {
    let FrameData::Cpu(image) = &decoded.frame.data else {
        return Err("frame is not a CPU image".into());
    };
    let bps = if decoded.info.bit_depth == 8 { 1 } else { 2 };
    let mut out = Vec::new();
    for (index, plane) in image.planes.iter().enumerate() {
        let row_samples = if index == 0 {
            decoded.frame.width() as usize
        } else {
            (decoded.frame.width() as usize + usize::from(decoded.info.subsampling_x))
                >> decoded.info.subsampling_x
        };
        append_plane(&mut out, plane, row_samples, bps)?;
    }
    Ok(out)
}

fn append_plane(
    out: &mut Vec<u8>,
    plane: &Plane,
    row_samples: usize,
    bps: usize,
) -> Result<(), String> {
    let row_bytes = row_samples
        .checked_mul(bps)
        .ok_or("plane row size overflow")?;
    for row in 0..plane.rows {
        let start = row
            .checked_mul(plane.stride)
            .ok_or("plane stride overflow")?;
        let end = start.checked_add(row_bytes).ok_or("plane row overflow")?;
        let src = plane
            .data
            .get(start..end)
            .ok_or("plane row exceeds the allocation")?;
        if bps == 1 {
            out.extend_from_slice(src);
        } else {
            for chunk in src.as_chunks::<2>().0 {
                let sample = u16::from_ne_bytes(*chunk);
                out.extend_from_slice(&sample.to_le_bytes());
            }
        }
    }
    Ok(())
}

fn ivf_frames(bytes: &[u8]) -> Result<Vec<&[u8]>, String> {
    if bytes.len() < 32 {
        return Err("ivf header truncated".into());
    }
    let mut out = Vec::new();
    let mut at = 32usize;
    while at + 12 <= bytes.len() {
        let size = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        at += 12;
        if at + size > bytes.len() {
            return Err("ivf frame truncated".into());
        }
        out.push(&bytes[at..at + size]);
        at += size;
    }
    Ok(out)
}
