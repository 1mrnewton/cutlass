//! Print every packet of an AVI file, one line each, in the order the
//! demuxer returns them:
//!
//! `cargo run -p cutlass-avi --example packets -- movie.avi`
//!
//! Each line is `stream dts pts duration size key position`, with times in
//! the stream's own ticks and `-` for a missing time, so it lines up with
//! `ffprobe -show_packets`.

#![allow(clippy::print_stdout)]

use std::fs::File;

use cutlass_avi::AviDemux;
use cutlass_container::Demux;

fn main() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| "usage: packets <file.avi>".to_owned())?;
    let file = File::open(&path).map_err(|err| err.to_string())?;
    let length = file.metadata().map_err(|err| err.to_string())?.len();
    let mut demux = AviDemux::open(file, length).map_err(|err| err.to_string())?;
    println!(
        "# index {:?} interleaved {} riffs {} truncated {} fallbacks {:?}",
        demux.index_source(),
        demux.interleaved(),
        demux.riff_count(),
        demux.truncated(),
        demux.fallbacks()
    );
    for stream in demux.streams() {
        println!(
            "# stream {} {:?} timescale {} size {:?} rate {:?} channels {:?} extradata {}",
            stream.id().get(),
            stream.params().codec(),
            stream.timescale().ticks_per_second(),
            stream.params().size(),
            stream.params().sample_rate(),
            stream.params().channels(),
            stream.params().extradata().len(),
        );
    }
    let show = |time: Option<i64>| time.map_or_else(|| "-".to_owned(), |time| time.to_string());
    while let Some(packet) = demux.read_packet().map_err(|err| err.to_string())? {
        let stamp = packet.timestamp();
        println!(
            "{} {} {} {} {} {}",
            packet.stream(),
            show(stamp.decode()),
            show(stamp.presentation()),
            show(stamp.duration()),
            packet.bytes().len(),
            u8::from(packet.keyframe()),
        );
    }
    Ok(())
}
