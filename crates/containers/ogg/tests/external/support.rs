//! Fixture generation and ffprobe parsing for tests that shell out.
//!
//! Fixtures are encoded from `../local-files` into a temporary folder on
//! every run and never committed. `CUTLASS_LOCAL_FILES` names that folder
//! when the checkout is not next to it. A test skips, with a message, when
//! ffmpeg, ffprobe, or the folder is missing.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub fn tool_on_path(name: &str) -> bool {
    Command::new(name)
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

pub fn local_root() -> Option<PathBuf> {
    let path = std::env::var_os("CUTLASS_LOCAL_FILES").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files"),
        PathBuf::from,
    );
    path.is_dir().then_some(path)
}

/// The local folder and both tools, or a printed reason to skip.
pub fn environment(test: &str) -> Option<PathBuf> {
    if !tool_on_path("ffmpeg") || !tool_on_path("ffprobe") {
        eprintln!("skipped {test}: ffmpeg or ffprobe is not on PATH");
        return None;
    }
    let Some(root) = local_root() else {
        eprintln!("skipped {test}: ../local-files is missing (set CUTLASS_LOCAL_FILES)");
        return None;
    };
    Some(root)
}

/// A fresh folder under the system temp dir for this test.
pub fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cutlass-ogg-{}-{test}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// The first local file, by name, that ends with one of `suffixes` and has
/// an audio stream.
pub fn pick(root: &Path, suffixes: &[&str]) -> Option<PathBuf> {
    let mut names: Vec<PathBuf> = std::fs::read_dir(root)
        .ok()?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file())
        .collect();
    names.sort();
    names.into_iter().find(|path| {
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        suffixes.iter().any(|suffix| name.ends_with(suffix)) && has_audio(path)
    })
}

fn has_audio(path: &Path) -> bool {
    Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .is_ok_and(|out| !out.stdout.is_empty())
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

/// One packet as ffprobe reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProbePacket {
    pub pts: i64,
    pub duration: i64,
    pub size: usize,
    pub hash: String,
}

pub fn ffprobe_packets(path: &Path) -> Vec<ProbePacket> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_packets",
            "-show_data_hash",
            "sha256",
            "-show_entries",
            "packet=pts,duration,size,data_hash",
            "-of",
            "compact",
        ])
        .arg(path)
        .output()
        .expect("ffprobe");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    let mut packets = Vec::new();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("packet|") else {
            continue;
        };
        let mut packet = ProbePacket {
            pts: i64::MIN,
            duration: -1,
            size: 0,
            hash: String::new(),
        };
        for part in rest.split('|') {
            if let Some(value) = part.strip_prefix("pts=") {
                packet.pts = value.parse().unwrap_or(i64::MIN);
            } else if let Some(value) = part.strip_prefix("duration=") {
                packet.duration = value.parse().unwrap_or(-1);
            } else if let Some(value) = part.strip_prefix("size=") {
                packet.size = value.parse().unwrap_or(0);
            } else if let Some(value) = part.strip_prefix("data_hash=SHA256:") {
                packet.hash = value.to_owned();
            }
        }
        packets.push(packet);
    }
    packets
}

/// Stream facts ffprobe reports.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProbeStream {
    pub codec: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub initial_padding: i64,
    pub extradata_size: usize,
}

pub fn ffprobe_stream(path: &Path) -> ProbeStream {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,initial_padding,extradata_size",
            "-of",
            "compact",
        ])
        .arg(path)
        .output()
        .expect("ffprobe");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    let mut stream = ProbeStream::default();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("stream|") else {
            continue;
        };
        for part in rest.split('|') {
            if let Some(value) = part.strip_prefix("codec_name=") {
                value.clone_into(&mut stream.codec);
            } else if let Some(value) = part.strip_prefix("sample_rate=") {
                stream.sample_rate = value.parse().unwrap_or(0);
            } else if let Some(value) = part.strip_prefix("channels=") {
                stream.channels = value.parse().unwrap_or(0);
            } else if let Some(value) = part.strip_prefix("initial_padding=") {
                stream.initial_padding = value.parse().unwrap_or(-1);
            } else if let Some(value) = part.strip_prefix("extradata_size=") {
                stream.extradata_size = value.parse().unwrap_or(0);
            }
        }
    }
    stream
}

/// Samples per channel ffmpeg decodes, with pre-skip and end trimming applied.
pub fn ffmpeg_decoded_samples(path: &Path) -> u64 {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-i"])
        .arg(path)
        .args(["-map", "0:a:0", "-ac", "1", "-f", "s16le", "-"])
        .output()
        .expect("ffmpeg decode");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    u64::try_from(out.stdout.len() / 2).expect("count")
}

pub fn sha256_hex(bytes: &[u8], scratch: &Path) -> String {
    // shasum is not libavformat; it hashes the bytes this crate returned.
    let file = scratch.join("hash.bin");
    std::fs::write(&file, bytes).expect("write");
    let out = Command::new("shasum")
        .args(["-a", "256"])
        .arg(&file)
        .output()
        .expect("shasum");
    String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_owned()
}

/// Every audio packet payload ffmpeg's demuxer returns, concatenated.
pub fn ffmpeg_payload(path: &Path) -> Vec<u8> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-i"])
        .arg(path)
        .args(["-map", "0:a:0", "-c", "copy", "-f", "data", "-"])
        .output()
        .expect("ffmpeg data");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

pub fn gstreamer_on_path() -> bool {
    Command::new("gst-launch-1.0")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// Audio packets GStreamer's `oggdemux` returns: (size, granule end).
///
/// GStreamer's Ogg demuxer is a second implementation, independent of
/// libavformat. `None` when `gst-launch-1.0` is not installed.
pub fn gstreamer_packets(path: &Path) -> Option<Vec<(usize, i64)>> {
    if !gstreamer_on_path() {
        return None;
    }
    let location = format!("location={}", path.display());
    let out = Command::new("gst-launch-1.0")
        .args([
            "-v",
            "filesrc",
            &location,
            "!",
            "oggdemux",
            "!",
            "fakesink",
            "silent=false",
        ])
        .output()
        .ok()?;
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
        if rest.contains(" header ") {
            continue;
        }
        let size = rest.split(' ').next()?.parse().ok()?;
        let end = rest
            .split(", ")
            .find_map(|part| part.strip_prefix("offset_end: "))?
            .parse()
            .ok()?;
        packets.push((size, end));
    }
    Some(packets)
}

/// Samples per channel GStreamer's `oggdemux ! decoder` decodes at `rate`.
pub fn gstreamer_decoded_samples(
    path: &Path,
    scratch: &Path,
    decoder: &str,
    rate: u32,
) -> Option<u64> {
    if !gstreamer_on_path() {
        return None;
    }
    let raw = scratch.join("gst.raw");
    let status = Command::new("gst-launch-1.0")
        .args([
            "-q",
            "filesrc",
            &format!("location={}", path.display()),
            "!",
            "oggdemux",
            "!",
            decoder,
            "!",
            "audioconvert",
            "!",
            &format!("audio/x-raw,format=S16LE,channels=1,rate={rate}"),
            "!",
            "filesink",
            &format!("location={}", raw.display()),
        ])
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    let len = std::fs::metadata(&raw).ok()?.len();
    Some(len / 2)
}
