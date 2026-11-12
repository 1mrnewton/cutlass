//! Helpers for tests that shell out to independent tools: FFmpeg's
//! libavformat (ffprobe) and GStreamer's matroskademux.

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use cutlass_codec::Packet;
use cutlass_container::{Demux, Mux, NewStream, TrackDuration, TrackId};
use cutlass_matroska::{MatroskaDemux, MatroskaMux};

pub fn tool_on_path(name: &str) -> bool {
    Command::new("which")
        .arg(name)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

pub fn ffmpeg_tools() -> bool {
    let found = tool_on_path("ffmpeg") && tool_on_path("ffprobe");
    if !found {
        eprintln!("skipped: ffmpeg and ffprobe are not on PATH");
    }
    found
}

/// A fresh directory for this test under Cargo's per-target temp dir. The
/// next run replaces it.
pub fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("matroska")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

pub fn output_timeout(program: &str, args: &[&str], limit: Duration) -> Option<Output> {
    let mut child = Command::new(program)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;
    // Read the pipes on threads so a chatty tool does not block on a full pipe.
    let mut stdout = child.stdout.take()?;
    let mut stderr = child.stderr.take()?;
    let out = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = std::io::Read::read_to_end(&mut stdout, &mut bytes);
        bytes
    });
    let err = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = std::io::Read::read_to_end(&mut stderr, &mut bytes);
        bytes
    });
    let started = Instant::now();
    let status = loop {
        if started.elapsed() > limit {
            let _ = child.kill();
            let _ = child.wait();
            return None;
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => std::thread::sleep(Duration::from_millis(10)),
            Err(_) => return None,
        }
    };
    Some(Output {
        status,
        stdout: out.join().ok()?,
        stderr: err.join().ok()?,
    })
}

/// Runs ffmpeg quietly; `false` when it fails (an encoder missing here).
pub fn ffmpeg(args: &[&str]) -> bool {
    let mut all = vec!["-hide_banner", "-loglevel", "error", "-y"];
    all.extend_from_slice(args);
    output_timeout("ffmpeg", &all, Duration::from_secs(120)).is_some_and(|out| out.status.success())
}

/// One packet as ffprobe (libavformat) reads it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Probed {
    pub stream: usize,
    pub pts: Option<i64>,
    pub duration: Option<i64>,
    pub size: usize,
    pub key: bool,
    /// The packet carries DiscardPadding (skip-samples side data with padding).
    pub discard: bool,
}

/// Packets in file order, timestamps in each stream's time base (1/1000
/// for Matroska's default scale).
pub fn ffprobe_packets(path: &Path) -> Vec<Probed> {
    let out = output_timeout(
        "ffprobe",
        &[
            "-v",
            "error",
            "-show_entries",
            "packet=stream_index,pts,duration,size,flags:packet_side_data=discard_padding",
            "-of",
            "compact=p=0:nk=0",
            &path.display().to_string(),
        ],
        Duration::from_secs(60),
    )
    .expect("ffprobe");
    assert!(out.status.success(), "ffprobe failed on {}", path.display());
    let text = String::from_utf8_lossy(&out.stdout);
    let mut packets: Vec<Probed> = Vec::new();
    for line in text.lines() {
        let field = |name: &str| {
            line.split('|')
                .find_map(|part| part.strip_prefix(&format!("{name}=")))
                .map(str::to_owned)
        };
        if line.starts_with("stream_index=") {
            packets.push(Probed {
                stream: field("stream_index")
                    .and_then(|v| v.parse().ok())
                    .expect("index"),
                pts: field("pts").and_then(|v| v.parse().ok()),
                duration: field("duration").and_then(|v| v.parse().ok()),
                size: field("size").and_then(|v| v.parse().ok()).expect("size"),
                key: field("flags").is_some_and(|flags| flags.starts_with('K')),
                discard: line
                    .split("discard_padding=")
                    .nth(1)
                    .and_then(|tail| tail.split('|').next())
                    .is_some_and(|value| value != "0"),
            });
        }
    }
    packets
}

/// The concatenated payloads of stream `index`, as libavformat returns them.
pub fn ffmpeg_payload(path: &Path, index: usize, dir: &Path) -> Vec<u8> {
    let raw = dir.join(format!("stream{index}.raw"));
    let map = format!("0:{index}");
    assert!(
        ffmpeg(&[
            "-i",
            &path.display().to_string(),
            "-map",
            &map,
            "-c",
            "copy",
            "-f",
            "data",
            &raw.display().to_string(),
        ]),
        "ffmpeg -f data"
    );
    std::fs::read(&raw).expect("raw")
}

/// Packets GStreamer's matroskademux puts out on one pad: (size, pts ns).
/// `None` when gst-launch is missing or does not finish in time.
pub fn gstreamer_packets(path: &Path, pad: &str) -> Option<Vec<(usize, Option<u64>)>> {
    if !tool_on_path("gst-launch-1.0") {
        return None;
    }
    let location = format!("location={}", path.display());
    let pad = format!("d.{pad}");
    let out = output_timeout(
        "gst-launch-1.0",
        &[
            "-v",
            "filesrc",
            &location,
            "!",
            "matroskademux",
            "name=d",
            &pad,
            "!",
            "fakesink",
            "silent=false",
        ],
        Duration::from_secs(60),
    )?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut packets = Vec::new();
    for line in text.lines() {
        let Some(at) = line.find("chain   ******* (fakesink0:sink) (") else {
            continue;
        };
        let rest = &line[at + "chain   ******* (fakesink0:sink) (".len()..];
        let size = rest.split(' ').next()?.parse().ok()?;
        let pts = rest
            .split("pts: ")
            .nth(1)
            .and_then(|tail| tail.split(',').next())
            .and_then(parse_clock);
        packets.push((size, pts));
    }
    Some(packets)
}

/// `0:00:01.500000000` in nanoseconds.
fn parse_clock(text: &str) -> Option<u64> {
    let mut parts = text.trim().split(':');
    let hours: u64 = parts.next()?.parse().ok()?;
    let minutes: u64 = parts.next()?.parse().ok()?;
    let seconds = parts.next()?;
    let (whole, fraction) = seconds.split_once('.')?;
    let whole: u64 = whole.parse().ok()?;
    let fraction: u64 = fraction.parse().ok()?;
    Some(((hours * 60 + minutes) * 60 + whole) * 1_000_000_000 + fraction)
}

pub type Opened = MatroskaDemux<BufReader<File>>;

pub fn open(path: &Path) -> Opened {
    let file = File::open(path).expect("file");
    let length = file.metadata().expect("metadata").len();
    MatroskaDemux::open(BufReader::new(file), length).expect("open")
}

pub fn read_all(demux: &mut Opened) -> Vec<Packet> {
    let mut out = Vec::new();
    while let Some(packet) = demux.read_packet().expect("packet") {
        out.push(packet);
    }
    out
}

/// The CodecDelay edit of each stream, in its ticks (zero when none).
pub fn delays(demux: &Opened) -> Vec<i64> {
    demux
        .streams()
        .iter()
        .map(|stream| {
            stream
                .edits()
                .first()
                .and_then(|edit| edit.media_time())
                .unwrap_or(0)
        })
        .collect()
}

/// Copies every packet of `input` into a new file at `output` with
/// [`MatroskaMux`]. Missing durations become the time to the next packet
/// of the track.
pub fn remux(input: &Path, output: &Path, webm: bool) {
    let mut demux = open(input);
    let streams = demux.streams().to_vec();
    let packets: Vec<Packet> = read_all(&mut demux)
        .into_iter()
        .filter(|packet| packet.timestamp().presentation().is_some())
        .collect();
    let file = std::io::BufWriter::new(File::create(output).expect("create"));
    let mut mux = if webm {
        MatroskaMux::webm(file)
    } else {
        MatroskaMux::new(file)
    };
    let mut tracks = Vec::new();
    for stream in &streams {
        let id = mux
            .add_stream(NewStream {
                params: stream.params().clone(),
                timescale: stream.timescale(),
                language: stream.language().cloned(),
            })
            .expect("add stream");
        tracks.push((stream.id().get(), id, stream.timescale(), 0_i64));
    }
    for (index, packet) in packets.iter().enumerate() {
        let time = packet.timestamp();
        let pts = time.presentation().expect("pts");
        let duration = time.duration().unwrap_or_else(|| {
            packets[index + 1..]
                .iter()
                .find(|other| other.stream() == packet.stream())
                .and_then(|other| other.timestamp().presentation())
                .map_or(0, |next| (next - pts).max(0))
        });
        let entry = tracks
            .iter_mut()
            .find(|entry| entry.0 == packet.stream())
            .expect("track");
        entry.3 += duration;
        let stamped =
            cutlass_codec::Timestamp::try_new(None, Some(pts), Some(duration)).expect("timestamp");
        let copy = Packet::try_new(
            entry.1.get(),
            packet.bytes().to_vec(),
            stamped,
            packet.timescale(),
            packet.keyframe(),
        )
        .expect("packet");
        mux.write_packet(&copy).expect("write");
    }
    let durations: Vec<TrackDuration> = tracks
        .iter()
        .map(|(_, id, timescale, ticks)| TrackDuration {
            track: TrackId::new(id.get()),
            timescale: *timescale,
            ticks: *ticks,
        })
        .collect();
    mux.finish(&durations).expect("finish");
}
