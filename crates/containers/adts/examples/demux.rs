//! Demux two ADTS frames and print the frame count and first timestamp.
//!
//! [`AdtsMuxer`] writes the file. [`AdtsFile`] reads it through the shared
//! demux trait. The access units are raw bytes, not decoded PCM.

#![allow(clippy::print_stdout)]

use std::io::Cursor;

use cutlass_adts::{AdtsFile, AdtsMuxer, Muxer};
use cutlass_codec::CodecId;
use cutlass_codec::access::{ChannelLayout, CodecParams, Packet};
use cutlass_container::Demux;

fn main() -> Result<(), String> {
    let bytes = framed()?;
    let mut file = AdtsFile::open(Cursor::new(bytes)).map_err(|err| err.to_string())?;
    let mut packets = Vec::new();
    while let Some(packet) = file.read_packet().map_err(|err| err.to_string())? {
        packets.push(packet);
    }
    let first = packets
        .first()
        .ok_or_else(|| "adts file has no frames".to_owned())?;
    let pts = first
        .timestamp()
        .presentation()
        .ok_or_else(|| "first frame has no presentation time".to_owned())?;
    let timescale = first.timescale().ticks_per_second();
    println!("frames {}", packets.len());
    println!("first {pts}/{timescale}");
    Ok(())
}

fn framed() -> Result<Vec<u8>, String> {
    let mut mux = AdtsMuxer::new();
    mux.add_stream(CodecParams::audio(
        CodecId::Aac,
        48_000,
        ChannelLayout::mono(),
    ))
    .map_err(|err| err.to_string())?;
    mux.write_packet(&Packet::audio(
        vec![0x21, 0x10, 0x04, 0x60],
        0,
        1024,
        48_000,
    ))
    .map_err(|err| err.to_string())?;
    mux.write_packet(&Packet::audio(
        vec![0x21, 0x10, 0x04, 0x61],
        1024,
        1024,
        48_000,
    ))
    .map_err(|err| err.to_string())?;
    mux.finish().map_err(|err| err.to_string())
}
