//! Release timing for open, first packet, full read, and seek on long Opus
//! and Vorbis files, with seek landings compared to the packet list and to
//! libavformat's own seek.
//!
//! Run with
//! `cargo test --release -p cutlass-ogg --test external measure -- --ignored --nocapture`.
//! The files are `rain_bedroom_night_30min.mp3` encoded into a temporary
//! folder by libopus at 96 kb/s, and by libvorbis at quality 0.4 through
//! GStreamer (FFmpeg's native Vorbis encoder when GStreamer is missing).

use std::fs::File;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use cutlass_container::{Demux, Landing, SeekTo};
use cutlass_ogg::OggDemux;

use crate::oracle::{duration, pts};
use crate::support::{environment, ffmpeg, gstreamer_on_path, scratch};

fn open_file(path: &Path) -> OggDemux<File> {
    let file = File::open(path).expect("open");
    let len = file.metadata().expect("meta").len();
    OggDemux::open(file, len).expect("ogg open")
}

/// The first packet libavformat returns after seeking to `seconds`.
fn ffmpeg_landing(path: &Path, seconds: f64) -> Option<i64> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-read_intervals",
            &format!("{seconds}%+#1"),
            "-show_entries",
            "packet=pts",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .trim()
        .parse()
        .ok()
}

/// A stream's rate, sample span, seek pre-roll, and the offset from
/// libavformat's `pts` to this crate's time.
struct Facts {
    rate: i64,
    start: i64,
    end: i64,
    preroll: i64,
    offset: i64,
}

impl Facts {
    fn of(demux: &OggDemux<File>) -> Self {
        if let Some(info) = demux.opus(0) {
            Self {
                rate: 48_000,
                start: info.start.expect("start"),
                end: info.end.expect("end"),
                preroll: 3_840,
                offset: i64::from(info.head.pre_skip),
            }
        } else {
            let info = demux.vorbis(0).expect("vorbis");
            Self {
                rate: i64::from(info.id.sample_rate),
                start: info.start.expect("start"),
                end: info.end.expect("end"),
                preroll: i64::from(info.id.blocksizes[1] / 2),
                offset: 0,
            }
        }
    }
}

#[test]
#[ignore = "release measurement; needs ../local-files and ffmpeg"]
fn measure_a_thirty_minute_opus_file() {
    let Some(root) = environment("measure_a_thirty_minute_opus_file") else {
        return;
    };
    let source = root.join("rain_bedroom_night_30min.mp3");
    if !source.is_file() {
        eprintln!("skipped: {} is missing", source.display());
        return;
    }
    let dir = scratch("measure");
    let path = dir.join("rain.opus");
    let started = Instant::now();
    ffmpeg(&[
        "-i",
        source.to_str().expect("utf8"),
        "-c:a",
        "libopus",
        "-b:a",
        "96k",
        path.to_str().expect("utf8"),
    ]);
    eprintln!("encoded in {:.1} s", started.elapsed().as_secs_f64());
    measure_file(&path);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "release measurement; needs ../local-files and ffmpeg"]
fn measure_a_thirty_minute_vorbis_file() {
    let Some(root) = environment("measure_a_thirty_minute_vorbis_file") else {
        return;
    };
    let source = root.join("rain_bedroom_night_30min.mp3");
    if !source.is_file() {
        eprintln!("skipped: {} is missing", source.display());
        return;
    }
    let dir = scratch("measure-vorbis");
    let path = dir.join("rain.ogg");
    let started = Instant::now();
    if gstreamer_on_path() {
        let status = Command::new("gst-launch-1.0")
            .args(["-q", "filesrc"])
            .arg(format!("location={}", source.display()))
            .args([
                "!",
                "decodebin",
                "!",
                "audioconvert",
                "!",
                "vorbisenc",
                "quality=0.4",
                "!",
                "oggmux",
                "!",
                "filesink",
            ])
            .arg(format!("location={}", path.display()))
            .status()
            .expect("gst-launch-1.0");
        assert!(status.success(), "libvorbis encode");
        eprintln!("libvorbis through GStreamer");
    } else {
        ffmpeg(&[
            "-i",
            source.to_str().expect("utf8"),
            "-c:a",
            "vorbis",
            "-strict",
            "-2",
            "-q:a",
            "4",
            path.to_str().expect("utf8"),
        ]);
        eprintln!("FFmpeg's native Vorbis encoder: gst-launch-1.0 is not on PATH");
    }
    eprintln!("encoded in {:.1} s", started.elapsed().as_secs_f64());
    measure_file(&path);
    let _ = std::fs::remove_dir_all(&dir);
}

fn measure_file(path: &Path) {
    let file_len = std::fs::metadata(path).expect("meta").len();
    let facts = Facts::of(&open_file(path));
    eprintln!("fixture: {file_len} bytes, rate {}", facts.rate);

    // Open and first packet, best of five (the file is in the page cache).
    let mut open_best = Duration::MAX;
    let mut first_best = Duration::MAX;
    let mut open_bytes = 0;
    for _ in 0..5 {
        let started = Instant::now();
        let mut demux = open_file(path);
        let opened = started.elapsed();
        open_bytes = demux.consumed();
        let first = demux.read_packet().expect("read").expect("packet");
        let to_first = started.elapsed();
        assert_eq!(pts(&first), facts.start);
        open_best = open_best.min(opened);
        first_best = first_best.min(to_first);
    }
    eprintln!(
        "open {:.3} ms reading {open_bytes} bytes; time to first packet {:.3} ms",
        open_best.as_secs_f64() * 1e3,
        first_best.as_secs_f64() * 1e3
    );

    // Full read.
    let started = Instant::now();
    let mut demux = open_file(path);
    let mut list = Vec::new();
    while let Some(packet) = demux.read_packet().expect("read") {
        list.push((pts(&packet), duration(&packet), packet.bytes().len()));
    }
    let read = started.elapsed();
    let end = facts.end;
    let seconds = (end - facts.start) as f64 / facts.rate as f64;
    eprintln!(
        "full read: {} packets, {:.1} s of audio in {:.1} ms, {:.0} MB/s, {:.0}x real time",
        list.len(),
        seconds,
        read.as_secs_f64() * 1e3,
        f64::from(u32::try_from(file_len).expect("len")) / read.as_secs_f64() / 1e6,
        seconds / read.as_secs_f64()
    );

    // Seeks.
    let mut worst_probes = 0;
    let mut worst_bytes = 0;
    let mut total = Duration::ZERO;
    let fractions = [0.0, 0.01, 0.1, 0.25, 0.333, 0.5, 0.75, 0.9, 0.99, 0.999];
    for fraction in fractions {
        let target = (end as f64 * fraction) as i64;
        let started = Instant::now();
        let landing = demux
            .seek(SeekTo {
                track: 0,
                time: target,
                end: None,
            })
            .expect("seek");
        let next = demux.read_packet().expect("read").expect("packet");
        let took = started.elapsed();
        total += took;
        let Landing::Packet { time, preroll, .. } = landing.landing else {
            panic!("seek to {target} reached the end");
        };
        assert!(!landing.index_used);
        assert_eq!(pts(&next), time);
        // The landing is the packet holding target less the pre-roll, or
        // the start.
        let aim = (target - facts.preroll).max(facts.start);
        let expected = list
            .iter()
            .find(|(start, length, _)| start + (*length).max(1) > aim)
            .expect("packet");
        assert_eq!(time, expected.0, "landing for {target}");
        assert_eq!(next.bytes().len(), expected.2);
        let cost = demux.last_seek();
        worst_probes = worst_probes.max(cost.probes);
        worst_bytes = worst_bytes.max(cost.bytes);
        let theirs = ffmpeg_landing(
            path,
            (target - facts.offset).max(0) as f64 / facts.rate as f64,
        )
        .map(|pts| pts + facts.offset);
        eprintln!(
            "seek {target:>9}: landed {time:>9} preroll {preroll:>3} in {:.3} ms, {} bytes, {} probes, {} pages, {} payload scans; libavformat lands {theirs:?}",
            took.as_secs_f64() * 1e3,
            cost.bytes,
            cost.probes,
            cost.pages,
            landing.payload_scans
        );
    }
    let past = demux
        .seek(SeekTo {
            track: 0,
            time: end + facts.rate,
            end: None,
        })
        .expect("seek past");
    assert_eq!(past.landing, Landing::End);
    eprintln!(
        "{} seeks: mean {:.3} ms to first packet, worst {worst_probes} probes and {worst_bytes} bytes",
        fractions.len(),
        total.as_secs_f64() * 1e3 / fractions.len() as f64
    );
}
