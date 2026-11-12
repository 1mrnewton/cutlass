//! Open a transport stream and print what the demuxer finds.
//!
//! `cargo run -p cutlass-mpegts --example demux -- FILE` prints the programs,
//! the streams and a packet summary. With `--packets` it also prints one
//! line per packet: stream, DTS, PTS, duration, keyframe and size, the
//! columns `ffprobe -show_packets` gives. With `--seek SECONDS` it seeks
//! the first stream there first. Without a file it writes a short sample
//! with ffmpeg, when ffmpeg is installed.

#![allow(clippy::print_stdout)]

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::process::Command;

use cutlass_container::{Demux, SeekTo};
use cutlass_mpegts::{TIMESCALE, TsDemux};

const FFMPEG: &str = "/opt/homebrew/bin/ffmpeg";

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut path = None;
    let mut packets = false;
    let mut seek = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--packets" => packets = true,
            "--seek" => {
                let seconds: f64 = args
                    .next()
                    .and_then(|value| value.parse().ok())
                    .ok_or("--seek needs seconds")?;
                seek = Some(seconds);
            }
            _ => path = Some(PathBuf::from(arg)),
        }
    }
    let path = match path {
        Some(path) => path,
        None => match sample()? {
            Some(path) => path,
            None => {
                println!("skip: no file given and ffmpeg is not installed");
                return Ok(());
            }
        },
    };
    let file = File::open(&path).map_err(|error| error.to_string())?;
    let length = file.metadata().map_err(|error| error.to_string())?.len();
    let mut demux =
        TsDemux::open(BufReader::new(file), length).map_err(|error| error.to_string())?;
    for program in demux.programs() {
        println!(
            "program {} pmt 0x{:04x} streams {}",
            program.number,
            program.pmt_pid,
            program.pmt.as_ref().map_or(0, |pmt| pmt.streams.len())
        );
    }
    println!(
        "reading program {} ({:?} packets), start {} end {:?}",
        demux.program(),
        demux.packet_size(),
        demux.start(),
        demux.end()
    );
    for (stream, elementary) in demux.streams().iter().zip(demux.elementary_streams()) {
        let params = stream.params();
        println!(
            "stream 0x{:04x} type 0x{:02x} {:?} size {:?} rate {:?} channels {:?} extradata {} language {:?}",
            stream.id().get(),
            elementary.stream_type,
            params.codec(),
            params.size(),
            params.sample_rate(),
            params.channels(),
            params.extradata().len(),
            stream
                .language()
                .map(|language| language.as_str().to_owned()),
        );
    }
    if let Some(seconds) = seek {
        let track = demux.streams().first().ok_or("no streams")?.id().get();
        let time = demux.start() + (seconds * f64::from(TIMESCALE)) as i64;
        let landing = demux
            .seek(SeekTo {
                track,
                time,
                end: None,
            })
            .map_err(|error| error.to_string())?;
        println!("seek {time}: {landing:?} cost {:?}", demux.last_seek());
    }
    let mut count = 0_u64;
    let mut bytes = 0_u64;
    let mut keyframes = 0_u64;
    while let Some(packet) = demux.read_packet().map_err(|error| error.to_string())? {
        count += 1;
        bytes += packet.bytes().len() as u64;
        keyframes += u64::from(packet.keyframe());
        if packets {
            let time = packet.timestamp();
            println!(
                "packet 0x{:04x} dts {:?} pts {:?} duration {:?} key {} size {}",
                packet.stream(),
                time.decode(),
                time.presentation(),
                time.duration(),
                u8::from(packet.keyframe()),
                packet.bytes().len()
            );
        }
    }
    println!("packets {count} keyframes {keyframes} bytes {bytes}");
    println!("counters {:?}", demux.counters());
    println!(
        "resyncs {} skipped {} read {}",
        demux.resyncs(),
        demux.skipped_bytes(),
        demux.total_read()
    );
    Ok(())
}

fn sample() -> Result<Option<PathBuf>, String> {
    if !PathBuf::from(FFMPEG).is_file() {
        return Ok(None);
    }
    let dir = std::env::temp_dir().join("cutlass-mpegts-demux-example");
    std::fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
    let path = dir.join("sample.ts");
    let status = Command::new(FFMPEG)
        .args(["-y", "-hide_banner", "-loglevel", "error"])
        .args([
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=25:duration=1",
        ])
        .args([
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:sample_rate=48000:duration=1",
        ])
        .args(["-c:v", "libx264", "-preset", "ultrafast", "-g", "10"])
        .args(["-c:a", "aac", "-b:a", "64k"])
        .arg(&path)
        .status()
        .map_err(|error| error.to_string())?;
    if !status.success() {
        return Err("ffmpeg failed".to_owned());
    }
    Ok(Some(path))
}
