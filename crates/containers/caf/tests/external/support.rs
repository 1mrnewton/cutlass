//! Helpers for tests that run independent tools on files.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use cutlass_caf::{CafDemux, CafMux};
use cutlass_codec::Packet;
use cutlass_container::{Demux, Mux, MuxError, NewStream, TrackDuration, TrackId};

pub fn tool(name: &str) -> bool {
    Command::new("which")
        .arg(name)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// Runs a tool and returns stdout, or `None` with stderr shown when it
/// fails.
pub fn run(program: &str, args: &[&str]) -> Option<Vec<u8>> {
    let out = Command::new(program)
        .args(args)
        .output()
        .unwrap_or_else(|error| panic!("{program} failed to start: {error}"));
    if out.status.success() {
        Some(out.stdout)
    } else {
        eprintln!(
            "{program} {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        );
        None
    }
}

pub fn path(path: &Path) -> &str {
    path.to_str().expect("utf-8 path")
}

pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("cutlass-caf-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        Self(dir)
    }

    pub fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    /// `CUTLASS_CAF_KEEP=1` keeps the files for a look after a failure.
    fn drop(&mut self) {
        if std::env::var_os("CUTLASS_CAF_KEEP").is_some() {
            eprintln!("kept {}", self.0.display());
        } else {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

/// Stereo 16-bit frames: a rising tone on the left and quiet noise on the
/// right, the same on every run.
pub fn source_frames(frames: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(frames * 4);
    let mut noise = 0x1234_5678_u32;
    for index in 0..frames {
        let t = index as f64 / 44_100.0;
        let left =
            (0.6 * (std::f64::consts::TAU * (220.0 + 400.0 * t) * t).sin() * 32_767.0) as i16;
        noise ^= noise << 13;
        noise ^= noise >> 17;
        noise ^= noise << 5;
        let right = ((noise >> 20) as i16).wrapping_sub(2048);
        out.extend(left.to_le_bytes());
        out.extend(right.to_le_bytes());
    }
    out
}

/// A 16-bit WAVE file of `payload` at `rate`.
pub fn wave(payload: &[u8], rate: u32, channels: u16) -> Vec<u8> {
    let len = u32::try_from(payload.len()).expect("len");
    let mut out = b"RIFF".to_vec();
    out.extend((36 + len).to_le_bytes());
    out.extend(b"WAVEfmt ");
    out.extend(16_u32.to_le_bytes());
    out.extend(1_u16.to_le_bytes());
    out.extend(channels.to_le_bytes());
    out.extend(rate.to_le_bytes());
    out.extend((rate * u32::from(channels) * 2).to_le_bytes());
    out.extend((channels * 2).to_le_bytes());
    out.extend(16_u16.to_le_bytes());
    out.extend(b"data");
    out.extend(len.to_le_bytes());
    out.extend(payload);
    out
}

/// The payload of the `data` chunk of a WAVE file.
pub fn wave_data(bytes: &[u8]) -> Vec<u8> {
    let mut at = 12;
    while at + 8 <= bytes.len() {
        let size = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().expect("size"));
        let size = usize::try_from(size).expect("size");
        if &bytes[at..at + 4] == b"data" {
            return bytes[at + 8..(at + 8 + size).min(bytes.len())].to_vec();
        }
        at += 8 + size + (size & 1);
    }
    panic!("no data chunk");
}

pub fn open(file: &Path) -> CafDemux<File> {
    let handle = File::open(file).expect("open");
    let len = handle.metadata().expect("meta").len();
    CafDemux::open(handle, len).unwrap_or_else(|error| panic!("{}: {error}", file.display()))
}

pub fn packets(file: &Path) -> Vec<Packet> {
    let mut demux = open(file);
    let mut out = Vec::new();
    while let Some(packet) = demux.read_packet().expect("read") {
        out.push(packet);
    }
    out
}

/// Packet times, durations and sizes as ffprobe reports them, and the
/// payload ffmpeg copies out.
pub struct Probed {
    pub packets: Vec<(i64, i64, usize)>,
    pub payload: Vec<u8>,
}

pub fn ffprobe(file: &Path) -> Option<Probed> {
    let text = run(
        "ffprobe",
        &[
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "packet=pts,duration,size",
            "-of",
            "csv=p=0",
            path(file),
        ],
    )?;
    let mut packets = Vec::new();
    for line in String::from_utf8(text).expect("utf-8").lines() {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 3 {
            continue;
        }
        packets.push((
            fields[0].parse().expect("pts"),
            fields[1].parse().expect("duration"),
            fields[2].parse().expect("size"),
        ));
    }
    let payload = run(
        "ffmpeg",
        &[
            "-v",
            "error",
            "-i",
            path(file),
            "-map",
            "0:a",
            "-c",
            "copy",
            "-f",
            "data",
            "-",
        ],
    )?;
    Some(Probed { packets, payload })
}

/// The packet, frame and byte counts afinfo prints.
#[derive(Debug, PartialEq, Eq)]
pub struct AfInfo {
    pub packets: u64,
    pub bytes: u64,
    pub valid: u64,
    pub priming: u64,
    pub remainder: u64,
}

pub fn afinfo(file: &Path) -> Option<AfInfo> {
    let text = String::from_utf8(run("afinfo", &[path(file)])?).expect("utf-8");
    let field = |prefix: &str| -> Option<u64> {
        text.lines()
            .find_map(|line| line.trim().strip_prefix(prefix))
            .and_then(|rest| rest.trim().parse().ok())
    };
    let line = text
        .lines()
        .find(|line| line.contains("valid frames"))?
        .split_whitespace()
        .collect::<Vec<_>>();
    // "audio 44100 valid frames + 2112 priming + 892 remainder = 47104"
    Some(AfInfo {
        packets: field("audio packets:")?,
        bytes: field("audio bytes:")?,
        valid: line.get(1)?.parse().ok()?,
        priming: line.get(5)?.parse().ok()?,
        remainder: line.get(8)?.parse().ok()?,
    })
}

/// Copies every packet of `from` into a new file at `to` with this crate's
/// muxer, carrying the trim and the layout over.
pub fn remux(from: &Path, to: &Path) -> Result<(), MuxError> {
    let mut demux = open(from);
    let stream = demux.streams()[0].clone();
    let layout = demux.channel_layout().cloned();
    let out = File::create(to).expect("create");
    let mut mux = CafMux::new(out);
    if let Some(layout) = layout {
        mux.set_channel_layout(layout).expect("layout");
    }
    let track = mux
        .add_stream(NewStream {
            params: stream.params().clone(),
            timescale: stream.timescale(),
            language: None,
        })
        .expect("add");
    let trim = stream.trim();
    if trim.encoder_delay().is_some() || trim.trailing_padding().is_some() {
        mux.set_trim(track, trim).expect("trim");
    }
    let mut ticks = 0;
    while let Some(packet) = demux.read_packet().expect("read") {
        ticks += packet.timestamp().duration().expect("duration");
        mux.write_packet(&packet)?;
    }
    mux.finish(&[TrackDuration {
        track: TrackId::new(0),
        timescale: stream.timescale(),
        ticks,
    }])
    .map_err(|failed| failed.reason().clone())?;
    Ok(())
}
