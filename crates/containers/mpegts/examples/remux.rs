//! Copy the streams of an MP4, MOV or transport stream into a new
//! transport stream.
//!
//! `cargo run -p cutlass-mpegts --example remux -- IN OUT [--m2ts]`. Streams
//! the muxer does not carry are left out and named. MP4 input exercises
//! the `avcC`/`hvcC` to Annex B conversion and raw AAC to ADTS.

#![allow(clippy::print_stdout)]

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek};

use cutlass_container::{Demux, Mux, NewStream, TrackDuration};
use cutlass_isobmff::IsoBmffFile;
use cutlass_mpegts::{TsDemux, TsMux};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [input, output, rest @ ..] = args.as_slice() else {
        return Err("usage: remux IN OUT [--m2ts]".to_owned());
    };
    let m2ts = rest.iter().any(|arg| arg == "--m2ts");
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
    let mut demux: Box<dyn Demux> = if is_ts {
        Box::new(TsDemux::open(reader, length).map_err(|error| error.to_string())?)
    } else {
        Box::new(IsoBmffFile::open(reader).map_err(|error| error.to_string())?)
    };
    let out = File::create(output).map_err(|error| error.to_string())?;
    let writer = BufWriter::new(out);
    let mut mux = if m2ts {
        TsMux::m2ts(writer)
    } else {
        TsMux::new(writer)
    };
    let mut map = Vec::new();
    for stream in demux.streams() {
        let added = mux.add_stream(NewStream {
            params: stream.params().clone(),
            timescale: stream.timescale(),
            language: stream.language().cloned(),
        });
        match added {
            Ok(track) => map.push((stream.id().get(), track)),
            Err(error) => {
                println!("left out stream {}: {error}", stream.id().get());
                // A refused stream fails the mux; start again without it.
                return Err("remove the unsupported stream first".to_owned());
            }
        }
    }
    let mut packets = 0_u64;
    while let Some(packet) = demux.read_packet().map_err(|error| error.to_string())? {
        let Some((_, track)) = map.iter().find(|(id, _)| *id == packet.stream()) else {
            continue;
        };
        let packet = cutlass_codec::Packet::try_new(
            track.get(),
            packet.bytes().to_vec(),
            packet.timestamp(),
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
