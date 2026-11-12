//! Print the streams and packets of a Matroska or WebM file.
//!
//! `cargo run -p cutlass-matroska --example packets -- movie.mkv` prints one
//! line per stream, then one line per packet: track, presentation time and
//! duration in the file's ticks, size, and `K` for a keyframe. With a time
//! after the path (seconds), it seeks the first track there first and
//! prints the landing.

#![allow(clippy::print_stdout)]

use std::fs::File;
use std::io::BufReader;

use cutlass_container::{Demux, SeekTo};
use cutlass_matroska::MatroskaDemux;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or("usage: packets <file> [seek seconds]")?;
    let file = File::open(&path).map_err(|error| error.to_string())?;
    let length = file.metadata().map_err(|error| error.to_string())?.len();
    let mut demux = MatroskaDemux::open(BufReader::new(file), length).map_err(|e| e.to_string())?;
    println!(
        "doctype {} scale {}",
        demux.ebml_header().doc_type.as_str(),
        demux.info().timestamp_scale
    );
    for stream in demux.streams() {
        println!(
            "stream {} {:?} timescale {} edits {:?} trim {:?}",
            stream.id().get(),
            stream.params().codec(),
            stream.timescale().ticks_per_second(),
            stream.edits(),
            stream.trim(),
        );
    }
    if let Some(seconds) = args.next() {
        let seconds: f64 = seconds.parse().map_err(|_| "seek time is not a number")?;
        let stream = demux.streams().first().ok_or("no streams")?;
        let ticks = seconds * f64::from(stream.timescale().ticks_per_second());
        let to = SeekTo {
            track: stream.id().get(),
            #[allow(clippy::cast_possible_truncation)]
            time: ticks as i64,
            end: None,
        };
        let landing = demux.seek(to).map_err(|error| error.to_string())?;
        println!("seek {landing:?}");
    }
    while let Some(packet) = demux.read_packet().map_err(|error| error.to_string())? {
        let time = packet.timestamp();
        println!(
            "{} {} {} {} {}",
            packet.stream(),
            time.presentation()
                .map_or("-".to_owned(), |t| t.to_string()),
            time.duration().map_or("-".to_owned(), |t| t.to_string()),
            packet.bytes().len(),
            if packet.keyframe() { "K" } else { "_" },
        );
    }
    Ok(())
}
