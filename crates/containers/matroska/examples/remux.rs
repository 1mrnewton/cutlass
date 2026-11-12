//! Copy the tracks of a Matroska or WebM file into a new one.
//!
//! `cargo run -p cutlass-matroska --example remux -- in.mkv out.mkv` reads
//! every packet with [`MatroskaDemux`] and writes them with [`MatroskaMux`];
//! an output name ending in `.webm` writes WebM. Packets the input left
//! without a duration get the time to the next packet of their track (the
//! last one gets the one before it), since the muxer wants every duration.

#![allow(clippy::print_stdout)]

use std::fs::File;
use std::io::{BufReader, BufWriter};

use cutlass_codec::{Packet, Timestamp};
use cutlass_container::{Demux, Mux, NewStream, TrackDuration, TrackId};
use cutlass_matroska::{MatroskaDemux, MatroskaMux};

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let input = args.next().ok_or("usage: remux <in> <out>")?;
    let output = args.next().ok_or("usage: remux <in> <out>")?;
    let file = File::open(&input).map_err(|error| error.to_string())?;
    let length = file.metadata().map_err(|error| error.to_string())?.len();
    let mut demux =
        MatroskaDemux::open(BufReader::new(file), length).map_err(|error| error.to_string())?;
    let streams = demux.streams().to_vec();
    let mut packets = Vec::new();
    while let Some(packet) = demux.read_packet().map_err(|error| error.to_string())? {
        if packet.timestamp().presentation().is_some() {
            packets.push(packet);
        }
    }
    let packets = with_durations(packets)?;

    let out = BufWriter::new(File::create(&output).map_err(|error| error.to_string())?);
    let mut mux = if output.ends_with(".webm") {
        MatroskaMux::webm(out)
    } else {
        MatroskaMux::new(out)
    };
    let mut ids = Vec::new();
    for stream in &streams {
        let id = mux
            .add_stream(NewStream {
                params: stream.params().clone(),
                timescale: stream.timescale(),
                language: stream.language().cloned(),
            })
            .map_err(|error| error.to_string())?;
        ids.push((stream.id().get(), id, stream.timescale(), 0_i64));
    }
    for packet in &packets {
        let Some(entry) = ids.iter_mut().find(|entry| entry.0 == packet.stream()) else {
            continue;
        };
        let renumbered = Packet::try_new(
            entry.1.get(),
            packet.bytes().to_vec(),
            packet.timestamp(),
            packet.timescale(),
            packet.keyframe(),
        )
        .map_err(|error| error.to_string())?;
        entry.3 += packet.timestamp().duration().unwrap_or(0);
        mux.write_packet(&renumbered)
            .map_err(|error| error.to_string())?;
    }
    let durations: Vec<TrackDuration> = ids
        .iter()
        .map(|(_, id, timescale, ticks)| TrackDuration {
            track: TrackId::new(id.get()),
            timescale: *timescale,
            ticks: *ticks,
        })
        .collect();
    mux.finish(&durations)
        .map_err(|failed| failed.reason().to_string())?;
    println!("{} packets", packets.len());
    Ok(())
}

/// Fills missing durations from the next packet of the same track.
fn with_durations(packets: Vec<Packet>) -> Result<Vec<Packet>, String> {
    let mut out = Vec::with_capacity(packets.len());
    for (index, packet) in packets.iter().enumerate() {
        let time = packet.timestamp();
        let duration = match time.duration() {
            Some(duration) => duration,
            None => {
                let pts = time.presentation().unwrap_or(0);
                let next = packets[index + 1..]
                    .iter()
                    .find(|other| other.stream() == packet.stream())
                    .and_then(|other| other.timestamp().presentation());
                let previous = packets[..index]
                    .iter()
                    .rev()
                    .find(|other| other.stream() == packet.stream())
                    .and_then(|other| other.timestamp().presentation());
                match (next, previous) {
                    (Some(next), _) if next > pts => next - pts,
                    (_, Some(previous)) if pts > previous => pts - previous,
                    _ => 0,
                }
            }
        };
        let timestamp = Timestamp::try_new(time.decode(), time.presentation(), Some(duration))
            .map_err(|error| error.to_string())?;
        out.push(
            Packet::try_new(
                packet.stream(),
                packet.bytes().to_vec(),
                timestamp,
                packet.timescale(),
                packet.keyframe(),
            )
            .map_err(|error| error.to_string())?,
        );
    }
    Ok(out)
}
