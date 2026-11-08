//! Encodes raw 4:2:0 planar video to an MPEG-2 video elementary stream.
//!
//! `cargo run --release -p cutlass-mpeg2 --example yuv_m2v -- in.yuv WIDTH HEIGHT RATE_NUM RATE_DEN BITRATE [cbr] out.m2v`
//!
//! `BITRATE` 0 picks the default. `cbr` asks for a constant rate.

use std::io::{Read, Write};

use cutlass_decode::{CodecId, CodecParams, Yuv420p};
use cutlass_encode::{VideoEncoder, VideoInput};
use cutlass_mpeg2::Mpeg2Encoder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 8 {
        return Err(
            "usage: yuv_m2v in.yuv WIDTH HEIGHT RATE_NUM RATE_DEN BITRATE [cbr] out.m2v".into(),
        );
    }
    let width: u32 = args[2].parse()?;
    let height: u32 = args[3].parse()?;
    let rate_num: u32 = args[4].parse()?;
    let rate_den: i64 = args[5].parse()?;
    let mut params = CodecParams::video(CodecId::Mpeg2, width, height);
    params.bitrate = args[6].parse()?;
    params.constant_rate = args.get(7).is_some_and(|a| a == "cbr");
    let Some(output) = args.last() else {
        return Err("missing output".into());
    };
    let mut encoder = Mpeg2Encoder::open(&params)?;
    let mut input = std::io::BufReader::new(std::fs::File::open(&args[1])?);
    let mut out = std::io::BufWriter::new(std::fs::File::create(output)?);
    let luma = (width * height) as usize;
    let chroma = luma / 4;
    let mut frame = vec![0u8; luma + 2 * chroma];
    let mut index = 0i64;
    let started = std::time::Instant::now();
    loop {
        match input.read_exact(&mut frame) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let picture = Yuv420p::try_new(
            width,
            height,
            frame[..luma].to_vec(),
            frame[luma..luma + chroma].to_vec(),
            frame[luma + chroma..].to_vec(),
        )?;
        let packets = encoder.push(&VideoInput {
            picture,
            pts: index * rate_den,
            duration: rate_den,
            timescale: rate_num,
            force_keyframe: false,
        })?;
        for packet in packets {
            out.write_all(&packet.data)?;
        }
        index += 1;
    }
    for packet in encoder.drain()? {
        out.write_all(&packet.data)?;
    }
    out.flush()?;
    let seconds = started.elapsed().as_secs_f64();
    writeln!(
        std::io::stderr(),
        "{index} frames in {seconds:.3} s ({:.1} fps)",
        index as f64 / seconds
    )?;
    Ok(())
}
