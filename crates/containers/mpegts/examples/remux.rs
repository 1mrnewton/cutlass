//! Copy the streams of an MP4, MOV or transport stream into a new
//! transport stream.
//!
//! `cargo run -p cutlass-mpegts --example remux -- IN OUT [--m2ts]`. Streams
//! the muxer does not carry are left out and named. MP4 input exercises
//! the `avcC`/`hvcC` to Annex B conversion and raw AAC to ADTS.
//!
//! The input is read twice. The first pass finds how far each stream's
//! decode times must move earlier so that no sample shows before it
//! decodes: an MP4 with negative composition offsets (macOS screen
//! recordings have them) gives such samples, and a transport stream
//! cannot carry them. FFmpeg's MOV demuxer moves its DTS the same way.

#![allow(clippy::print_stdout)]

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek};

use cutlass_codec::{CodecId, StreamCodec, Timestamp};
use cutlass_container::{Demux, Mux, NewStream, TrackDuration};
use cutlass_isobmff::IsoBmffFile;
use cutlass_mpegts::{TsDemux, TsMux};

const CARRIED: [CodecId; 8] = [
    CodecId::H264,
    CodecId::Hevc,
    CodecId::Mpeg2,
    CodecId::Aac,
    CodecId::Mp3,
    CodecId::Ac3,
    CodecId::Eac3,
    CodecId::Opus,
];

fn open(input: &str) -> Result<Box<dyn Demux>, String> {
    let file = File::open(input).map_err(|error| error.to_string())?;
    let length = file.metadata().map_err(|error| error.to_string())?.len();
    let mut reader = BufReader::new(file);
    let mut head = [0_u8; 1024];
    let got = reader.read(&mut head).map_err(|error| error.to_string())?;
    reader.rewind().map_err(|error| error.to_string())?;
    let is_ts = matches!(
        cutlass_mpegts::probe(&head[..got]),
        cutlass_container::ProbeResult::Accept { .. }
    );
    Ok(if is_ts {
        Box::new(TsDemux::open(reader, length).map_err(|error| error.to_string())?)
    } else {
        Box::new(IsoBmffFile::open(reader).map_err(|error| error.to_string())?)
    })
}

/// Per stream, the most any sample's DTS is past its PTS.
fn decode_shifts(input: &str) -> Result<Vec<(u32, i64)>, String> {
    let mut demux = open(input)?;
    let mut shifts: Vec<(u32, i64)> = demux
        .streams()
        .iter()
        .map(|stream| (stream.id().get(), 0))
        .collect();
    while let Some(packet) = demux.read_packet().map_err(|error| error.to_string())? {
        let time = packet.timestamp();
        if let (Some(dts), Some(pts)) = (time.decode(), time.presentation())
            && let Some(slot) = shifts.iter_mut().find(|(id, _)| *id == packet.stream())
        {
            slot.1 = slot.1.max(dts.saturating_sub(pts));
        }
    }
    Ok(shifts)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [input, output, rest @ ..] = args.as_slice() else {
        return Err("usage: remux IN OUT [--m2ts]".to_owned());
    };
    let m2ts = rest.iter().any(|arg| arg == "--m2ts");
    let shifts = decode_shifts(input)?;
    let mut demux = open(input)?;
    let out = File::create(output).map_err(|error| error.to_string())?;
    let writer = BufWriter::new(out);
    let mut mux = if m2ts {
        TsMux::m2ts(writer)
    } else {
        TsMux::new(writer)
    };
    let mut map = Vec::new();
    for stream in demux.streams() {
        let carried = matches!(
            stream.params().codec(),
            StreamCodec::Known(codec) if CARRIED.contains(codec)
        );
        if !carried {
            println!(
                "left out stream {}: {:?}",
                stream.id().get(),
                stream.params().codec()
            );
            continue;
        }
        let track = mux
            .add_stream(NewStream {
                params: stream.params().clone(),
                timescale: stream.timescale(),
                language: stream.language().cloned(),
            })
            .map_err(|error| format!("stream {}: {error}", stream.id().get()))?;
        let shift = shifts
            .iter()
            .find(|(id, _)| *id == stream.id().get())
            .map_or(0, |slot| slot.1);
        if shift > 0 {
            println!(
                "stream {}: decode times moved {shift} ticks earlier",
                stream.id().get()
            );
        }
        map.push((stream.id().get(), track, shift));
    }
    let mut packets = 0_u64;
    while let Some(packet) = demux.read_packet().map_err(|error| error.to_string())? {
        let Some((_, track, shift)) = map.iter().find(|(id, ..)| *id == packet.stream()) else {
            continue;
        };
        let time = packet.timestamp();
        let time = Timestamp::try_new(
            time.decode().map(|dts| dts - shift),
            time.presentation(),
            time.duration(),
        )
        .map_err(|error| format!("{error:?}"))?;
        let packet = cutlass_codec::Packet::try_new(
            track.get(),
            packet.bytes().to_vec(),
            time,
            packet.timescale(),
            packet.keyframe(),
        )
        .map_err(|error| error.to_string())?;
        mux.write_packet(&packet)
            .map_err(|error| error.to_string())?;
        packets += 1;
    }
    let durations: Vec<TrackDuration> = Vec::new();
    mux.finish(&durations)
        .map_err(|error| format!("{error:?}"))?;
    println!("wrote {packets} packets to {output}");
    Ok(())
}
