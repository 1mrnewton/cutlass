//! Measures what each MP4 layout costs to write.
//!
//! ```text
//! mux_cost write <layout> <out.mp4> <seconds>
//! mux_cost remux <layout> <in.mp4> <out.mp4>
//! mux_cost hash <in.mp4>
//! ```
//!
//! `layout` is `progressive`, `fast-start`, `recoverable` or
//! `recoverable-fast-start`. `write` makes a synthetic movie: 59.94 fps
//! video at about 40 Mbit/s with a keyframe a second, and 44.1 kHz sound.
//! The payload is noise; only its size and timing matter to a muxer.
//! `remux` copies a file's packets into a new one, the work
//! `ffmpeg -i in -c copy out` does. `hash` prints a digest of each track's
//! packets, to show two layouts hold the same ones. Each prints the time to
//! write the packets, the time `finish` took, the time to sync the file to
//! disk, and the process's peak resident memory where the system tells.
//! With `MUX_COST_NO_SYNC` set the file is not synced, as FFmpeg's is not.

#![allow(clippy::print_stdout)]

use cutlass_isobmff::{
    Demux, FileSource, IsoBmff, IsoBmffMux, MovieFile, Mux, MuxOptions, Packet, StreamParams,
    Timescale,
};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let words: Vec<&str> = args.iter().map(String::as_str).collect();
    match words.as_slice() {
        ["write", layout, out, seconds] => {
            let seconds: u64 = seconds.parse().map_err(|_| "seconds is not a number")?;
            write(options(layout)?, Path::new(out), seconds)
        }
        ["remux", layout, input, out] => remux(options(layout)?, Path::new(input), Path::new(out)),
        ["hash", input] => hash(Path::new(input)),
        _ => Err("usage: mux_cost write|remux|hash ... (see the file's header)".to_owned()),
    }
}

fn options(layout: &str) -> Result<MuxOptions, String> {
    let (moov_at_end, recoverable) = match layout {
        "progressive" => (true, false),
        "fast-start" => (false, false),
        "recoverable" => (true, true),
        "recoverable-fast-start" => (false, true),
        _ => return Err(format!("unknown layout {layout}")),
    };
    Ok(MuxOptions {
        moov_at_end,
        recoverable,
        ..MuxOptions::default()
    })
}

fn text(error: impl std::fmt::Display) -> String {
    error.to_string()
}

/// A stream of bytes that repeats only after 2^64 steps.
struct Noise(u64);

impl Noise {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn write(options: MuxOptions, out: &Path, seconds: u64) -> Result<(), String> {
    let mut noise = Noise(0x9E37_79B9_7F4A_7C15);
    let pool: Vec<u8> = (0..1 << 20)
        .map(|_| noise.next().to_le_bytes()[0])
        .collect();
    let mut video = StreamParams::video(
        cutlass_isobmff::CodecId::H264,
        Timescale(60_000),
        3840,
        2160,
    );
    video.extradata = vec![1, 100, 0, 51, 0xFF, 0xE0, 0];
    let mut audio = StreamParams::audio(
        cutlass_isobmff::CodecId::Aac,
        Timescale(44_100),
        44_100,
        2,
        16,
        cutlass_isobmff::SampleFormat::S16Le,
    );
    audio.extradata = vec![0x12, 0x10];

    let started = Instant::now();
    let mut mux = IsoBmffMux::with_reader(MovieFile::create(out).map_err(text)?, options);
    let video_id = mux.add_stream(&video).map_err(text)?;
    let audio_id = mux.add_stream(&audio).map_err(text)?;
    let frames = seconds * 60_000 / 1_001;
    let sound = seconds * 44_100 / 1_024;
    let mut heard = 0u64;
    let mut packet = Packet {
        track_id: video_id,
        payload: Vec::new(),
        dts: 0,
        pts: 0,
        duration: 0,
        timescale: Timescale(60_000),
        keyframe: false,
    };
    let fill = |packet: &mut Packet, len: usize, noise: &mut Noise| {
        let at = usize::try_from(noise.next() % 4096).unwrap_or(0);
        packet.payload.clear();
        packet.payload.extend_from_slice(&pool[at..at + len]);
    };
    for frame in 0..frames {
        let key = frame % 60 == 0;
        // About 83 KiB a frame on average: a keyframe is five times a
        // dependent one.
        let base: u64 = if key { 360_000 } else { 72_000 };
        let len = usize::try_from(base + noise.next() % 16_000).unwrap_or(0);
        fill(&mut packet, len, &mut noise);
        packet.track_id = video_id;
        packet.timescale = Timescale(60_000);
        packet.dts = i64::try_from(frame * 1_001).unwrap_or(0);
        packet.pts = packet.dts;
        packet.duration = 1_001;
        packet.keyframe = key;
        mux.write_packet(&packet).map_err(text)?;
        while heard < sound && heard * 1_024 * 60_000 < (frame + 1) * 1_001 * 44_100 {
            let len = usize::try_from(300 + noise.next() % 140).unwrap_or(0);
            fill(&mut packet, len, &mut noise);
            packet.track_id = audio_id;
            packet.timescale = Timescale(44_100);
            packet.dts = i64::try_from(heard * 1_024).unwrap_or(0);
            packet.pts = packet.dts;
            packet.duration = 1_024;
            packet.keyframe = true;
            mux.write_packet(&packet).map_err(text)?;
            heard += 1;
        }
    }
    finish(mux, started, out)
}

fn remux(options: MuxOptions, input: &Path, out: &Path) -> Result<(), String> {
    let started = Instant::now();
    let mut source = FileSource::open(input).map_err(text)?;
    let mut movie = IsoBmff::open(&mut source).map_err(text)?;
    // The edits are copied as they are, so the movie timescale they are
    // counted in is kept too.
    let options = MuxOptions {
        movie_timescale: Some(movie.movie_timescale().get()),
        ..options
    };
    let mut mux = IsoBmffMux::with_reader(MovieFile::create(out).map_err(text)?, options);
    for stream in movie.streams() {
        let id = mux.add_stream(&stream.params).map_err(text)?;
        mux.set_edits(id, stream.edits.clone()).map_err(text)?;
    }
    while let Some(packet) = movie.read_packet(&mut source).map_err(text)? {
        mux.write_packet(&packet).map_err(text)?;
    }
    finish(mux, started, out)
}

fn finish(mux: IsoBmffMux<MovieFile>, started: Instant, out: &Path) -> Result<(), String> {
    let written = started.elapsed();
    let finishing = Instant::now();
    let mut file = mux.finish_into().map_err(text)?.into_inner();
    file.flush().map_err(text)?;
    let finished = finishing.elapsed();
    let syncing = Instant::now();
    let file = file
        .into_inner()
        .map_err(|error| text(error.into_error()))?;
    if std::env::var_os("MUX_COST_NO_SYNC").is_none() {
        file.sync_all().map_err(text)?;
    }
    let synced = syncing.elapsed();
    let bytes = std::fs::metadata(out).map_err(text)?.len();
    println!("bytes {bytes}");
    println!("packets_seconds {:.3}", written.as_secs_f64());
    println!("finish_seconds {:.3}", finished.as_secs_f64());
    println!("sync_seconds {:.3}", synced.as_secs_f64());
    println!("total_seconds {:.3}", started.elapsed().as_secs_f64());
    match peak_resident_kib() {
        Some(kib) => println!("peak_resident_kib {kib}"),
        None => println!("peak_resident_kib unknown"),
    }
    Ok(())
}

/// The most memory this process has had resident, from Linux's
/// `/proc/self/status`. `None` on a system without it.
fn peak_resident_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

/// FNV-1a over each track's packets: times, duration, sync flag and bytes.
fn hash(input: &Path) -> Result<(), String> {
    let mut source = FileSource::open(input).map_err(text)?;
    let mut movie = IsoBmff::open(&mut source).map_err(text)?;
    let ids: Vec<u32> = movie.streams().iter().map(|s| s.params.track_id).collect();
    let durations: Vec<u64> = movie
        .streams()
        .iter()
        .map(|s| s.params.duration_ticks)
        .collect();
    let mut digests = vec![0xcbf2_9ce4_8422_2325u64; ids.len()];
    let mut counts = vec![0u64; ids.len()];
    let mix = |digest: &mut u64, bytes: &[u8]| {
        for byte in bytes {
            *digest ^= u64::from(*byte);
            *digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    while let Some(packet) = movie.read_packet(&mut source).map_err(text)? {
        let Some(track) = ids.iter().position(|id| *id == packet.track_id) else {
            return Err("a packet of no track".to_owned());
        };
        let digest = &mut digests[track];
        mix(digest, &packet.dts.to_le_bytes());
        mix(digest, &packet.pts.to_le_bytes());
        mix(digest, &packet.duration.to_le_bytes());
        mix(digest, &[u8::from(packet.keyframe)]);
        mix(digest, &packet.payload);
        counts[track] += 1;
    }
    for (index, id) in ids.iter().enumerate() {
        println!(
            "track {id} packets {} duration {} digest {:016x}",
            counts[index], durations[index], digests[index]
        );
    }
    Ok(())
}
