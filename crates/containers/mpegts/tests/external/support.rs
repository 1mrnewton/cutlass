//! Tools, scratch folders, ffprobe parsing, and reading this crate's
//! packets in the same shape.

use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use cutlass_container::Demux;
use cutlass_mpegts::TsDemux;

pub fn tool_on_path(name: &str) -> bool {
    Command::new(name)
        .arg(if name.starts_with("gst-") {
            "--version"
        } else {
            "-version"
        })
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// Whether ffmpeg and ffprobe are both here; prints why a test skips.
pub fn ffmpeg_present(test: &str) -> bool {
    let present = tool_on_path("ffmpeg") && tool_on_path("ffprobe");
    if !present {
        eprintln!("skipped {test}: ffmpeg or ffprobe is not on PATH");
    }
    present
}

/// `../local-files` beside the checkout, or `CUTLASS_LOCAL_FILES`.
pub fn local_root() -> Option<PathBuf> {
    let path = std::env::var_os("CUTLASS_LOCAL_FILES").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files"),
        PathBuf::from,
    );
    path.is_dir().then_some(path)
}

/// A fresh folder under the system temp dir for this test.
pub fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cutlass-mpegts-{}-{test}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// Runs ffmpeg with `args` and panics with its stderr on failure.
pub fn ffmpeg(args: &[&str]) {
    let mut all = vec!["-v", "error", "-y", "-nostdin"];
    all.extend_from_slice(args);
    let out = Command::new("ffmpeg").args(&all).output().expect("ffmpeg");
    assert!(
        out.status.success(),
        "ffmpeg {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// What ffmpeg prints while decoding every stream of `path`, which is
/// empty for a clean file.
pub fn ffmpeg_decode_errors(path: &Path) -> String {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-i"])
        .arg(path)
        .args(["-f", "null", "-"])
        .output()
        .expect("ffmpeg");
    let mut text = String::from_utf8_lossy(&out.stderr).trim().to_owned();
    if !out.status.success() && text.is_empty() {
        text = format!("ffmpeg exited with {}", out.status);
    }
    text
}

/// GStreamer's `tsdemux` reads `path` to the end into fake sinks; `None`
/// when GStreamer is missing, else its complaint or an empty string.
pub fn gstreamer_errors(path: &Path) -> Option<String> {
    if !tool_on_path("gst-launch-1.0") {
        return None;
    }
    let location = format!("location={}", path.display());
    let out = Command::new("gst-launch-1.0")
        .args(["-q", "filesrc", &location, "!", "tsdemux", "name=d"])
        .args(["d.", "!", "queue", "!", "fakesink"])
        .args(["d.", "!", "queue", "!", "fakesink"])
        .output()
        .expect("gst-launch-1.0");
    Some(if out.status.success() {
        String::new()
    } else {
        format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    })
}

/// One packet, from ffprobe or from this crate, times in the stream's
/// time base.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packet {
    pub pts: Option<i64>,
    pub dts: Option<i64>,
    pub duration: Option<i64>,
    pub key: bool,
    pub size: usize,
}

/// One stream as ffprobe reports it, with its packets.
#[derive(Clone, Debug)]
pub struct Stream {
    /// The transport PID, or the track number in other containers.
    pub id: u32,
    pub codec: String,
    /// The time base as numerator and denominator.
    pub time_base: (i64, i64),
    pub packets: Vec<Packet>,
}

fn field(value: &str) -> Option<i64> {
    value.parse().ok()
}

/// Every stream and packet of `path` as ffprobe reads it. `extra` goes
/// before the input, such as `-ignore_editlist 1`.
pub fn ffprobe(path: &Path, extra: &[&str]) -> Vec<Stream> {
    let run = |entries: &str| {
        let out = Command::new("ffprobe")
            .args(["-v", "error"])
            .args(extra)
            .args(["-show_entries", entries, "-of", "csv=p=0"])
            .arg(path)
            .output()
            .expect("ffprobe");
        assert!(
            out.status.success(),
            "ffprobe {}: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    let mut streams: Vec<Stream> = run("stream=index,id,codec_name,time_base")
        .lines()
        .filter(|line| line.contains(','))
        // A transport stream's streams are listed under its programs too.
        .filter({
            let mut seen = Vec::new();
            move |line| {
                let index = line.split(',').next().unwrap_or_default().to_owned();
                let fresh = !seen.contains(&index);
                seen.push(index);
                fresh
            }
        })
        .map(|line| {
            // ffprobe prints the fields in its own order: index,
            // codec_name, id, time_base.
            let parts: Vec<&str> = line.split(',').collect();
            let index: u32 = parts[0].parse().expect("index");
            let codec = parts[1].to_owned();
            let id = parts[2]
                .strip_prefix("0x")
                .and_then(|hex| u32::from_str_radix(hex, 16).ok())
                .unwrap_or(index);
            let (num, den) = parts[3].split_once('/').expect("time base");
            Stream {
                id,
                codec,
                time_base: (num.parse().expect("num"), den.parse().expect("den")),
                packets: Vec::new(),
            }
        })
        .collect();
    for line in run("packet=stream_index,pts,dts,duration,size,flags").lines() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 6 {
            continue;
        }
        let index: usize = parts[0].parse().expect("stream index");
        streams[index].packets.push(Packet {
            pts: field(parts[1]),
            dts: field(parts[2]),
            duration: field(parts[3]),
            size: parts[4].parse().expect("size"),
            key: parts[5].starts_with('K'),
        });
    }
    streams
}

/// The demuxer over a file, and every packet it read, by PID.
pub type Read = (TsDemux<BufReader<std::fs::File>>, Vec<(u32, Vec<Packet>)>);

/// Every packet this crate reads from `path`, by PID, with the demuxer.
pub fn ours(path: &Path) -> Read {
    let file = std::fs::File::open(path).expect("open");
    let length = file.metadata().expect("metadata").len();
    let mut demux = TsDemux::open(BufReader::new(file), length).expect("demux open");
    let mut tracks: Vec<(u32, Vec<Packet>)> = demux
        .streams()
        .iter()
        .map(|stream| (stream.id().get(), Vec::new()))
        .collect();
    while let Some(packet) = demux.read_packet().expect("read") {
        let time = packet.timestamp();
        let slot = tracks
            .iter_mut()
            .find(|(id, _)| *id == packet.stream())
            .expect("a known track");
        slot.1.push(Packet {
            pts: time.presentation(),
            dts: time.decode(),
            duration: time.duration(),
            key: packet.keyframe(),
            size: packet.bytes().len(),
        });
    }
    (demux, tracks)
}

/// The provider and service name ffprobe reads for each program, against
/// those this crate reads from the SDT.
pub fn check_services(path: &Path, demux: &TsDemux<BufReader<std::fs::File>>) -> Vec<String> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-show_entries"])
        .arg("program=program_id:program_tags=service_name,service_provider")
        .args(["-of", "default=nw=1"])
        .arg(path)
        .output()
        .expect("ffprobe");
    let text = String::from_utf8_lossy(&out.stdout);
    let mut theirs: Vec<(u16, String, String)> = Vec::new();
    for line in text.lines() {
        if let Some(id) = line.strip_prefix("program_id=") {
            theirs.push((id.parse().unwrap_or(0), String::new(), String::new()));
        } else if let Some(last) = theirs.last_mut() {
            if let Some(name) = line.strip_prefix("TAG:service_name=") {
                name.clone_into(&mut last.2);
            } else if let Some(provider) = line.strip_prefix("TAG:service_provider=") {
                provider.clone_into(&mut last.1);
            }
        }
    }
    let mut problems = Vec::new();
    for (number, provider, name) in theirs {
        let Some(program) = demux
            .programs()
            .iter()
            .find(|program| program.number == number)
        else {
            problems.push(format!("program {number} is not in this crate's programs"));
            continue;
        };
        let mine = program
            .service
            .as_ref()
            .map_or((String::new(), String::new()), |service| {
                (service.provider.clone(), service.name.clone())
            });
        if mine != (provider.clone(), name.clone()) {
            problems.push(format!(
                "program {number}: service {mine:?}, ffprobe {:?}",
                (provider, name)
            ));
        }
    }
    problems
}

/// Codecs whose frames FFmpeg times from the PES timestamp plus the
/// frames before, rounding each length; this crate rounds the sum, so the
/// two may differ by a tick.
pub fn audio(codec: &str) -> bool {
    matches!(
        codec,
        "aac" | "aac_latm" | "mp3" | "mp2" | "ac3" | "eac3" | "opus"
    )
}
