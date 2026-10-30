//! Read one MP3 and print the frame count, duration, and timestamps.
//!
//! With no path, the example writes eight stereo frames through `Mp3Mux`
//! and opens that buffer with `Mp3Demux`. A path argument is read instead.
//!
//! ```text
//! cargo run --release -p cutlass-mp3-container --example demux
//! cargo run --release -p cutlass-mp3-container --example demux -- file.mp3
//! ```

#![allow(clippy::print_stdout)]

use std::env;
use std::io::Cursor;

use cutlass_mp3_container::{CodecId, CodecParams, Demux, Mp3Demux, Mp3Mux, Mux, Packet};

const SAMPLE_RATE: u32 = 48_000;
const CHANNELS: u16 = 2;
const BITRATE_KBPS: u32 = 128;
const SAMPLES_PER_FRAME: i64 = 1_152;
const FRAMES: u16 = 8;

fn main() -> Result<(), String> {
    let bytes = match env::args().nth(1) {
        Some(path) => std::fs::read(&path).map_err(|err| err.to_string())?,
        None => sample_file()?,
    };
    let mut demux = Mp3Demux::open(Cursor::new(bytes)).map_err(|err| err.to_string())?;
    let timescale = demux
        .streams()
        .first()
        .map(|stream| stream.timescale)
        .ok_or_else(|| "open published no stream".to_owned())?;
    let duration = demux.duration();
    let mut count = 0u64;
    let mut first = None;
    let mut last = None;
    while let Some(packet) = demux.next_packet().map_err(|err| err.to_string())? {
        if first.is_none() {
            first = Some(packet.pts);
        }
        last = Some(packet.pts);
        count += 1;
    }
    let first = first.ok_or_else(|| "file has no frames".to_owned())?;
    let last = last.ok_or_else(|| "file has no frames".to_owned())?;
    println!("packets: {count}");
    match duration {
        Some(samples) => println!("duration_samples: {samples}"),
        None => println!("duration_samples: unknown"),
    }
    println!("timescale: {timescale}");
    println!("first_pts: {first}");
    println!("last_pts: {last}");
    Ok(())
}

/// Eight MPEG-1 Layer III frames at 128 kbit/s, 48 kHz, stereo.
fn sample_file() -> Result<Vec<u8>, String> {
    let mut mux = Mp3Mux::new(Cursor::new(Vec::new()));
    mux.add_stream(CodecParams::audio(CodecId::Mp3, SAMPLE_RATE, CHANNELS))
        .map_err(|err| err.to_string())?;
    for index in 0..FRAMES {
        let frame = layer3_frame(index)?;
        let pts = i64::from(index) * SAMPLES_PER_FRAME;
        let packet = Packet::audio(frame, pts, SAMPLES_PER_FRAME, SAMPLE_RATE);
        mux.write_packet(&packet).map_err(|err| err.to_string())?;
    }
    mux.finish().map_err(|err| err.to_string())?;
    let cursor = mux.into_inner().map_err(|err| err.to_string())?;
    Ok(cursor.into_inner())
}

/// ISO/IEC 11172-3 Layer III header. Bitrate index 9 is 128 kbit/s.
/// Sample-rate index 1 is 48 kHz. The frame length is 384 bytes.
fn layer3_frame(index: u16) -> Result<Vec<u8>, String> {
    let bitrate_index = 9u32;
    let rate_index = 1u32;
    let raw = (0x7FFu32 << 21)
        | (0b11 << 19)
        | (0b01 << 17)
        | (1 << 16)
        | (bitrate_index << 12)
        | (rate_index << 10);
    let size = usize::try_from(BITRATE_KBPS * 144_000 / SAMPLE_RATE)
        .map_err(|_| "frame size does not fit".to_owned())?;
    if size < 8 {
        return Err("frame shorter than a header".to_owned());
    }
    let mut frame = vec![0u8; size];
    frame[..4].copy_from_slice(&raw.to_be_bytes());
    let mark = index.to_be_bytes();
    frame[5] = mark[0];
    frame[6] = mark[1];
    Ok(frame)
}
