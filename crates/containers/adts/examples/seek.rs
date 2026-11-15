//! Index a raw AAC file, then seek through it.
//!
//! `cargo run -p cutlass-adts --example seek -- file.aac 1.5 30`
//! prints the frame count, the stream length in samples, each packet's
//! size, and where each asked second lands.

#![allow(clippy::print_stdout)]

use std::fs::File;

use cutlass_adts::AdtsSeekFile;
use cutlass_container::{Demux, Landing, SeekTo};

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or("usage: seek FILE [SECONDS...]")?;
    let file = File::open(&path).map_err(|err| err.to_string())?;
    let length = file.metadata().map_err(|err| err.to_string())?.len();
    let mut demux = AdtsSeekFile::open(file, length).map_err(|err| err.to_string())?;
    let rate = demux.streams()[0].timescale().ticks_per_second();
    println!(
        "frames {} samples {} rate {rate}",
        demux.frames(),
        demux.samples()
    );
    let mut sizes = Vec::new();
    while let Some(packet) = demux.read_packet().map_err(|err| err.to_string())? {
        sizes.push(packet.bytes().len());
    }
    println!("packets {}", sizes.len());
    if std::env::var_os("SIZES").is_some() {
        for size in &sizes {
            println!("{size}");
        }
    }
    for second in args {
        let seconds: f64 = second
            .parse()
            .map_err(|_| format!("not seconds: {second}"))?;
        #[allow(clippy::cast_possible_truncation)]
        let time = (seconds * f64::from(rate)) as i64;
        let landing = demux
            .seek(SeekTo {
                track: 0,
                time,
                end: None,
            })
            .map_err(|err| err.to_string())?;
        let next = demux.read_packet().map_err(|err| err.to_string())?;
        match (landing.landing, next) {
            (Landing::Packet { time, .. }, Some(packet)) => {
                println!("seek {second}s -> {time} size {}", packet.bytes().len())
            }
            (Landing::End, None) => println!("seek {second}s -> end"),
            (landing, next) => println!(
                "seek {second}s -> {landing:?} then {:?}",
                next.map(|p| p.bytes().len())
            ),
        }
    }
    Ok(())
}
