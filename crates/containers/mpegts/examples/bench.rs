//! The Rust side of the speed comparison with FFmpeg's demuxer
//! (`scripts/ffmpeg_decode.c`), one job per process so each is measured
//! whole with `/usr/bin/time -l`:
//!
//! ```text
//! bench open FILE REPEAT          open REPEAT times: tables, parameters, end
//! bench read FILE                 open, then every packet of every stream
//! bench seek FILE N FIRST LAST    open, then N seeks of the first picture
//!                                 stream and a read to its next packet
//! ```
//!
//! Seek targets are 90 kHz times from FIRST to LAST drawn by a 64-bit
//! xorshift from seed 1, the same sequence `ffmpeg_decode seek:N:FIRST:LAST`
//! uses. The file is read through a `BufReader`, as the desktop opens it.

#![allow(clippy::print_stdout)]

use std::fs::File;
use std::hint::black_box;
use std::io::BufReader;

use cutlass_container::{Demux, Landing, SeekTo};
use cutlass_mpegts::TsDemux;

fn open(path: &str) -> Result<TsDemux<BufReader<File>>, String> {
    let file = File::open(path).map_err(|error| error.to_string())?;
    let length = file.metadata().map_err(|error| error.to_string())?.len();
    TsDemux::open(BufReader::new(file), length).map_err(|error| error.to_string())
}

fn number(text: &str) -> Result<i64, String> {
    text.parse().map_err(|_| format!("not a number: {text}"))
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [mode, path, repeat] if mode == "open" => {
            let mut streams = 0;
            for _ in 0..number(repeat)? {
                streams += black_box(open(path)?).streams().len();
            }
            println!("streams {streams}");
        }
        [mode, path] if mode == "read" => {
            let mut demux = open(path)?;
            let (mut packets, mut bytes) = (0_u64, 0_u64);
            while let Some(packet) = demux.read_packet().map_err(|error| error.to_string())? {
                packets += 1;
                bytes += packet.bytes().len() as u64;
                black_box(packet.bytes());
            }
            println!("packets {packets} bytes {bytes}");
        }
        [mode, path, times, first, last] if mode == "seek" => {
            let (times, first, last) = (number(times)?, number(first)?, number(last)?);
            let span = u64::try_from(last - first + 1).map_err(|_| "LAST is before FIRST")?;
            let mut demux = open(path)?;
            let track = demux
                .streams()
                .iter()
                .find(|stream| stream.params().size().is_some())
                .ok_or("no picture stream")?
                .id()
                .get();
            let end = demux.end();
            let mut state = 1_u64;
            let (mut landed, mut read) = (0_i64, 0_u64);
            for _ in 0..times {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let target = first + i64::try_from(state % span).map_err(|e| e.to_string())?;
                let landing = demux
                    .seek(SeekTo {
                        track,
                        time: target,
                        end,
                    })
                    .map_err(|error| error.to_string())?;
                read += demux.last_seek().bytes;
                if let Landing::Packet { .. } = landing.landing {
                    while let Some(packet) =
                        demux.read_packet().map_err(|error| error.to_string())?
                    {
                        if packet.stream() == track {
                            landed += packet.timestamp().presentation().unwrap_or(0);
                            break;
                        }
                    }
                }
            }
            println!("seeks {times} landed {landed} seek_bytes {read}");
        }
        _ => {
            return Err(
                "usage: bench open FILE REPEAT | read FILE | seek FILE N FIRST LAST".to_owned(),
            );
        }
    }
    Ok(())
}
