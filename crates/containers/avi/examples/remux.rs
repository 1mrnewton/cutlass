//! Copy every stream of an AVI file into a new OpenDML AVI:
//!
//! `cargo run -p cutlass-avi --example remux -- in.avi out.avi`
//!
//! Packets go from [`cutlass_avi::AviDemux`] to [`cutlass_avi::AviMux`]
//! unchanged, in the order the demuxer returns them. A stream the muxer
//! does not store (raw video, an unknown codec) fails the copy.

#![allow(clippy::print_stdout)]

use std::fs::File;
use std::io::BufReader;

use cutlass_avi::{AviDemux, AviMux};
use cutlass_container::{Demux, Mux, NewStream, TrackDuration};

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let (Some(input), Some(output)) = (args.next(), args.next()) else {
        return Err("usage: remux <in.avi> <out.avi>".to_owned());
    };
    let file = File::open(&input).map_err(|err| err.to_string())?;
    let length = file.metadata().map_err(|err| err.to_string())?.len();
    let mut demux = AviDemux::open(BufReader::new(file), length).map_err(|err| err.to_string())?;
    let out = File::create(&output).map_err(|err| err.to_string())?;
    let mut mux = AviMux::new(out);
    let mut durations = Vec::new();
    for stream in demux.streams() {
        let track = mux
            .add_stream(NewStream {
                params: stream.params().clone(),
                timescale: stream.timescale(),
                language: None,
            })
            .map_err(|err| err.to_string())?;
        durations.push(TrackDuration {
            track,
            timescale: stream.timescale(),
            ticks: 0,
        });
    }
    let mut packets = 0u64;
    while let Some(packet) = demux.read_packet().map_err(|err| err.to_string())? {
        let index = usize::try_from(packet.stream()).map_err(|err| err.to_string())?;
        let span = packet.timestamp().duration().unwrap_or(0);
        let entry = durations
            .get_mut(index)
            .ok_or_else(|| "packet of an unknown stream".to_owned())?;
        entry.ticks = entry
            .ticks
            .checked_add(span)
            .ok_or_else(|| "duration overflow".to_owned())?;
        mux.write_packet(&packet).map_err(|err| err.to_string())?;
        packets += 1;
    }
    mux.finish(&durations)
        .map_err(|err| err.reason().to_string())?;
    println!(
        "{packets} packets in {} streams, {} RIFF segments",
        durations.len(),
        mux.riff_count()
    );
    Ok(())
}
