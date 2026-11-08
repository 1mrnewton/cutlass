//! Decodes an MPEG-1 or MPEG-2 video elementary stream to raw YUV.
//!
//! `cargo run --release -p cutlass-mpeg2 --example m2v_yuv -- in.m2v out.yuv`
//!
//! Frames are written in display order, each in the stream's own chroma
//! format (4:2:0, 4:2:2, or 4:4:4 planar), cropped to the coded size.

use std::io::Write;

use cutlass_decode::{CodecError, Packet, VideoOutput};
use cutlass_mpeg2::Mpeg2Decoder;
use cutlass_mpeg2::es::split;

fn write(out: &mut impl Write, frame: &VideoOutput) -> std::io::Result<()> {
    if let Some(full) = &frame.yuv422 {
        out.write_all(full.y())?;
        out.write_all(full.u())?;
        out.write_all(full.v())
    } else if let Some(full) = &frame.yuv444 {
        out.write_all(full.y())?;
        out.write_all(full.u())?;
        out.write_all(full.v())
    } else {
        out.write_all(frame.picture.y())?;
        out.write_all(frame.picture.u())?;
        out.write_all(frame.picture.v())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let (Some(input), Some(output)) = (args.get(1), args.get(2)) else {
        return Err("usage: m2v_yuv in.m2v out.yuv".into());
    };
    let stream = std::fs::read(input)?;
    let mut out = std::io::BufWriter::new(std::fs::File::create(output)?);
    let mut decoder = Mpeg2Decoder::default();
    for unit in split(&stream)? {
        let packet = Packet {
            stream: 0,
            data: unit.data,
            pts: i64::try_from(unit.display_index)?,
            duration: 1,
            timescale: 25,
            keyframe: unit.coding_type == 1,
            side: Vec::new(),
        };
        let frames = decoder
            .push(&packet)
            .map_err(|e: CodecError| format!("frame {}: {e}", unit.decode_index))?;
        for frame in frames {
            write(&mut out, &frame)?;
        }
    }
    for frame in decoder.drain()? {
        write(&mut out, &frame)?;
    }
    out.flush()?;
    Ok(())
}
