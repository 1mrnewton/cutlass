//! The Rust side of the Matroska benchmark: the same four jobs
//! `bench/ffmpeg_matroska.c` does with libavformat.
//!
//! ```text
//! cargo run --release -p cutlass-matroska --example bench -- open  movie.mkv
//! cargo run --release -p cutlass-matroska --example bench -- read  movie.mkv
//! cargo run --release -p cutlass-matroska --example bench -- seek  movie.mkv 200
//! cargo run --release -p cutlass-matroska --example bench -- remux movie.mkv out.mkv
//! ```
//!
//! `open` opens the file and reads no packet: the start-up and header
//! cost the other jobs include. `read` reads every packet of every stream.
//! `seek` seeks the first stream `count` times, to a fixed shuffle of times
//! across its duration, and reads one packet after each. `remux` copies
//! every packet into a new file (WebM when the name ends in `.webm`),
//! streaming: each track holds back one packet, so a packet the input left
//! without a duration gets the time to the next one of its track.
//! `bench/against_ffmpeg.sh` times both sides as whole processes under
//! `/usr/bin/time -l`.

#![allow(clippy::print_stdout)]

use std::fs::File;
use std::io::BufWriter;

use cutlass_codec::{Packet, Timestamp};
use cutlass_container::{Demux, Mux, NewStream, SeekTo, TrackDuration, TrackId, Trim};
use cutlass_matroska::{MatroskaDemux, MatroskaMux};

/// The write buffer the muxer is given.
const WRITE_BUFFER: usize = 1 << 20;

const USAGE: &str = "usage: bench <open|read|seek|remux> <file> [count|out]";

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (command, path) = match args.as_slice() {
        [command, path, ..] => (command.as_str(), path),
        _ => return Err(USAGE.to_owned()),
    };
    let file = File::open(path).map_err(|error| error.to_string())?;
    let length = file.metadata().map_err(|error| error.to_string())?.len();
    // The demuxer reads ahead itself, so the file goes in unbuffered.
    let demux = MatroskaDemux::open(file, length).map_err(|error| error.to_string())?;
    match (command, args.get(2)) {
        ("open", None) => {
            println!("streams {}", demux.streams().len());
            Ok(())
        }
        ("read", None) => read(demux),
        ("seek", Some(count)) => seek(demux, count.parse().map_err(|_| USAGE.to_owned())?),
        ("remux", Some(out)) => remux(demux, out),
        _ => Err(USAGE.to_owned()),
    }
}

fn read(mut demux: MatroskaDemux<File>) -> Result<(), String> {
    let (mut packets, mut bytes) = (0_u64, 0_u64);
    while let Some(packet) = demux.read_packet().map_err(|error| error.to_string())? {
        packets += 1;
        bytes += u64::try_from(packet.bytes().len()).unwrap_or(0);
        std::hint::black_box(packet.bytes());
    }
    println!(
        "packets {packets} bytes {bytes} streams {}",
        demux.streams().len()
    );
    Ok(())
}

/// The `index`th of `count` seek times across `duration`: a fixed shuffle,
/// the same as the C side's, so the seeks jump back and forth.
fn spread(duration: i64, index: i64, count: i64) -> i64 {
    let share = i128::from((index * 7919) % count);
    i64::try_from(i128::from(duration) * share / i128::from(count)).unwrap_or(0)
}

fn seek(mut demux: MatroskaDemux<File>, count: i64) -> Result<(), String> {
    let stream = demux.streams().first().ok_or("no streams")?;
    let track = stream.id().get();
    let duration = demux
        .duration_ticks()
        .ok_or("the file states no duration")?;
    let (mut landed, mut sum) = (0_u64, 0_i64);
    for index in 0..count.max(0) {
        let time = spread(duration, index, count);
        demux
            .seek(SeekTo {
                track,
                time,
                end: None,
            })
            .map_err(|error| error.to_string())?;
        if let Some(packet) = demux.read_packet().map_err(|error| error.to_string())? {
            landed += 1;
            sum += packet.timestamp().presentation().unwrap_or(0);
        }
    }
    println!("seeks {count} landed {landed} pts {sum}");
    Ok(())
}

struct Track {
    input: u32,
    output: TrackId,
    held: Option<Packet>,
    /// The duration of the last packet written, for the last one.
    last: i64,
    ticks: i64,
}

fn remux(mut demux: MatroskaDemux<File>, path: &str) -> Result<(), String> {
    // Frames go to the writer one by one; a large buffer turns them into
    // few writes. libavformat's muxer buffers a whole Cluster in memory.
    let file = File::create(path).map_err(|error| error.to_string())?;
    let out = BufWriter::with_capacity(WRITE_BUFFER, file);
    let mut mux = if path.ends_with(".webm") {
        MatroskaMux::webm(out)
    } else {
        MatroskaMux::new(out)
    };
    let mut tracks = Vec::new();
    for stream in demux.streams() {
        let output = mux
            .add_stream(NewStream {
                params: stream.params().clone(),
                timescale: stream.timescale(),
                language: stream.language().cloned(),
            })
            .map_err(|error| error.to_string())?;
        // The encoder delay a CodecDelay stated, so that the copy keeps it.
        if stream.trim() != Trim::default() {
            mux.set_trim(output, stream.trim())
                .map_err(|error| error.to_string())?;
        }
        tracks.push(Track {
            input: stream.id().get(),
            output,
            held: None,
            last: 0,
            ticks: 0,
        });
    }
    let timescale = demux.clock().timescale();
    let mut packets = 0_u64;
    while let Some(packet) = demux.read_packet().map_err(|error| error.to_string())? {
        let Some(pts) = packet.timestamp().presentation() else {
            continue;
        };
        let Some(track) = tracks
            .iter_mut()
            .find(|track| track.input == packet.stream())
        else {
            continue;
        };
        if let Some(held) = track.held.take() {
            let start = held.timestamp().presentation().unwrap_or(0);
            let gap = if pts > start { pts - start } else { track.last };
            write(&mut mux, track, held, gap)?;
            packets += 1;
        }
        track.held = Some(packet);
    }
    for track in &mut tracks {
        if let Some(held) = track.held.take() {
            let last = track.last;
            write(&mut mux, track, held, last)?;
            packets += 1;
        }
    }
    let durations: Vec<TrackDuration> = tracks
        .iter()
        .map(|track| TrackDuration {
            track: track.output,
            timescale,
            ticks: track.ticks,
        })
        .collect();
    mux.finish(&durations)
        .map_err(|failed| failed.reason().to_string())?;
    println!("packets {packets}");
    Ok(())
}

/// Writes `packet` on `track`'s output with its own duration, or `gap`
/// when it has none.
fn write(
    mux: &mut MatroskaMux<BufWriter<File>>,
    track: &mut Track,
    packet: Packet,
    gap: i64,
) -> Result<(), String> {
    let time = packet.timestamp();
    let duration = time.duration().unwrap_or(gap);
    let timestamp = Timestamp::try_new(time.decode(), time.presentation(), Some(duration))
        .map_err(|error| error.to_string())?;
    let (timescale, keyframe) = (packet.timescale(), packet.keyframe());
    let out = Packet::try_new(
        track.output.get(),
        packet.into_bytes(),
        timestamp,
        timescale,
        keyframe,
    )
    .map_err(|error| error.to_string())?;
    mux.write_packet(&out).map_err(|error| error.to_string())?;
    track.last = duration;
    track.ticks += duration;
    Ok(())
}
