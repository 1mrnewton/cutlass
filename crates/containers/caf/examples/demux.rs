//! Prints what a CAF file holds: the stream, the packet table, and the
//! first and last packets.
//!
//! `cargo run --release -p cutlass-caf --example demux -- <file.caf> [--packets]`
//!
//! `--packets` prints one line per packet: index, time, duration, size.

#![allow(clippy::print_stdout)]

use std::fs::File;

use cutlass_caf::CafDemux;
use cutlass_container::Demux;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        println!("usage: demux <file.caf> [--packets]");
        return Ok(());
    };
    let every = args.next().as_deref() == Some("--packets");
    let file = File::open(&path)?;
    let length = file.metadata()?.len();
    let mut demux = CafDemux::open(file, length)?;
    let desc = *demux.description();
    let stream = demux.streams()[0].clone();
    println!(
        "format {:?} rate {} channels {} bytes/packet {} frames/packet {} bits {} flags {}",
        String::from_utf8_lossy(&desc.format),
        desc.sample_rate,
        desc.channels,
        desc.bytes_per_packet,
        desc.frames_per_packet,
        desc.bits_per_channel,
        desc.flags
    );
    println!(
        "codec {:?} extradata {} bytes",
        stream.params().codec(),
        stream.params().extradata().len()
    );
    if let Some(table) = demux.packet_table() {
        println!(
            "pakt packets {} valid {} priming {} remainder {}",
            table.packets, table.valid_frames, table.priming_frames, table.remainder_frames
        );
    }
    println!("trim {:?} edits {:?}", stream.trim(), stream.edits());
    let chunks: Vec<String> = demux
        .chunks()
        .iter()
        .map(|chunk| String::from_utf8_lossy(&chunk.kind).into_owned())
        .collect();
    println!("chunks {chunks:?} stopped at {:?}", demux.stopped_at());
    let mut count = 0_u64;
    let mut bytes = 0_u64;
    let mut frames = 0_i64;
    while let Some(packet) = demux.read_packet()? {
        let stamp = packet.timestamp();
        if every {
            println!(
                "{count} {} {} {}",
                stamp.presentation().unwrap_or(-1),
                stamp.duration().unwrap_or(-1),
                packet.bytes().len()
            );
        }
        count += 1;
        bytes += u64::try_from(packet.bytes().len())?;
        frames += stamp.duration().unwrap_or(0);
    }
    println!("read {count} packets, {bytes} bytes, {frames} frames");
    Ok(())
}
