//! The Rust side of the AVI benchmark; `benches/ffmpeg_avi.c` does the same
//! work with libavformat.
//!
//! ```text
//! cargo run --release -p cutlass-avi --example bench -- open <file> <repeat>
//! cargo run --release -p cutlass-avi --example bench -- read <file> <repeat>
//! cargo run --release -p cutlass-avi --example bench -- seek <file> <seeks>
//! ```
//!
//! `open` opens the file and drops it; `read` also reads every packet of
//! every stream; `seek` opens once, then seeks stream 0 to a spread of times
//! and reads one packet after each, the targets walking the stream's length
//! (`dwLength`) the same way the C harness walks it. Nothing is decoded.
//! Time it with `/usr/bin/time -l` and compare instructions retired.

#![allow(clippy::print_stdout)]

use std::fs::File;
use std::hint::black_box;

use cutlass_avi::AviDemux;
use cutlass_container::{Demux, SeekTo};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [mode, path, count] = args.as_slice() else {
        return Err("usage: bench <open|read|seek> <file> <repeat|seeks>".to_owned());
    };
    let count: u32 = count.parse().map_err(|_| "the count is not a number")?;
    match mode.as_str() {
        "open" => {
            for _ in 0..count {
                black_box(open(path)?);
            }
            println!("opens {count}");
        }
        "read" => {
            let (mut packets, mut bytes) = (0_u64, 0_u64);
            for _ in 0..count {
                (packets, bytes) = (0, 0);
                let mut demux = open(path)?;
                while let Some(packet) = demux.read_packet().map_err(|err| err.to_string())? {
                    packets += 1;
                    bytes += packet.bytes().len() as u64;
                    black_box(packet.bytes());
                }
            }
            println!("packets {packets} bytes {bytes}");
        }
        "seek" => seek(path, count)?,
        _ => return Err(format!("no mode {mode}")),
    }
    Ok(())
}

fn open(path: &str) -> Result<AviDemux<File>, String> {
    let file = File::open(path).map_err(|err| err.to_string())?;
    let length = file.metadata().map_err(|err| err.to_string())?.len();
    AviDemux::open(file, length).map_err(|err| err.to_string())
}

fn seek(path: &str, seeks: u32) -> Result<(), String> {
    let mut demux = open(path)?;
    let header = demux.stream_header(0).ok_or("no stream 0")?;
    let length = u64::from(header.length);
    if length == 0 {
        return Err("stream 0 has no length".to_owned());
    }
    // Ticks of the stream's timescale in one dwScale/dwRate unit.
    let unit = i64::from(header.scale / gcd(header.scale, header.rate).max(1));
    let mut state = 1_u64;
    let mut landed = 0_u64;
    for _ in 0..seeks {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let target = i64::try_from((state >> 33) % length).map_err(|err| err.to_string())?;
        demux
            .seek(SeekTo {
                track: 0,
                time: target * unit,
                end: None,
            })
            .map_err(|err| err.to_string())?;
        let packet = demux
            .read_packet()
            .map_err(|err| err.to_string())?
            .ok_or("nothing after a seek")?;
        landed += packet.bytes().len() as u64;
    }
    println!("seeks {seeks} bytes {landed}");
    Ok(())
}

const fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}
