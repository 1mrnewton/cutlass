//! Copies the packets of a CAF file into a new CAF file with this crate's
//! muxer, keeping the codec configuration, priming, remainder, channel
//! layout and info strings.
//!
//! `cargo run --release -p cutlass-caf --example remux -- <in.caf> <out.caf>`

#![allow(clippy::print_stdout)]

use std::fs::File;
use std::io::BufWriter;

use cutlass_caf::{CafDemux, CafMux};
use cutlass_container::{Demux, Mux, NewStream, TrackDuration};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [from, to] = args.as_slice() else {
        println!("usage: remux <in.caf> <out.caf>");
        return Ok(());
    };
    let file = File::open(from)?;
    let length = file.metadata()?.len();
    let mut demux = CafDemux::open(file, length)?;
    let stream = demux.streams()[0].clone();
    let mut mux = CafMux::new(BufWriter::new(File::create(to)?));
    if let Some(layout) = demux.channel_layout() {
        mux.set_channel_layout(layout.clone())?;
    }
    if !demux.info().is_empty() {
        mux.set_info(demux.info().to_vec())?;
    }
    let track = mux.add_stream(NewStream {
        params: stream.params().clone(),
        timescale: stream.timescale(),
        language: None,
    })?;
    let trim = stream.trim();
    if trim.encoder_delay().is_some() || trim.trailing_padding().is_some() {
        mux.set_trim(track, trim)?;
    }
    let mut ticks = 0;
    let mut count = 0_u64;
    while let Some(packet) = demux.read_packet()? {
        ticks += packet.timestamp().duration().unwrap_or(0);
        count += 1;
        mux.write_packet(&packet)?;
    }
    let finished = mux.finish(&[TrackDuration {
        track,
        timescale: stream.timescale(),
        ticks,
    }]);
    let finished = finished.map_err(|failed| failed.reason().to_string())?;
    println!(
        "{count} packets, {ticks} frames, {} payload bytes, packet table {:?}",
        finished.payload_bytes(),
        finished.index()
    );
    Ok(())
}
