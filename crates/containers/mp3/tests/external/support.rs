//! Helpers for tests that shell out to independent tools.
//! Frame bytes are built from the ISO length formula, not from the crate.

use std::path::Path;
use std::process::{Command, Output, Stdio};

pub fn tool_on_path(name: &str) -> bool {
    Command::new("which")
        .arg(name)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

pub fn output(program: &str, args: &[&str]) -> Output {
    Command::new(program)
        .args(args)
        .output()
        .unwrap_or_else(|err| panic!("{program} failed to start: {err}"))
}

pub fn mpeg1_frame(bitrate_index: u32, rate_index: u32, padding: bool, marker: u16) -> Vec<u8> {
    const KBPS: [u32; 15] = [
        0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
    ];
    const RATES: [u32; 3] = [44_100, 48_000, 32_000];
    let kbps = KBPS[usize::try_from(bitrate_index).expect("bitrate")];
    let rate = RATES[usize::try_from(rate_index).expect("rate")];
    let pad = u32::from(padding);
    let size = usize::try_from(kbps * 144_000 / rate + pad).expect("size");
    let raw = (0x7FF << 21)
        | (0b11 << 19)
        | (0b01 << 17)
        | (1 << 16)
        | (bitrate_index << 12)
        | (rate_index << 10)
        | (pad << 9);
    let mut frame = vec![0u8; size];
    frame[..4].copy_from_slice(&raw.to_be_bytes());
    frame[5] = u8::try_from(marker >> 8).unwrap_or(0);
    frame[6] = u8::try_from(marker & 0x00FF).unwrap_or(0);
    frame
}

pub fn marker(frame: &[u8]) -> u16 {
    (u16::from(frame[5]) << 8) | u16::from(frame[6])
}

/// ffprobe packet time base for MPEG audio is 1/14112000.
pub fn ticks_to_samples(ticks: i64, sample_rate: u32) -> i64 {
    let ticks = i128::from(ticks);
    let rate = i128::from(sample_rate);
    i64::try_from(ticks * rate / 14_112_000).unwrap_or(0)
}

pub struct ProbePacket {
    pub size: u64,
    pub pts_samples: i64,
}

pub struct ProbeStream {
    pub sample_rate: u32,
    pub channels: u16,
    pub packets: Vec<ProbePacket>,
}

pub fn ffprobe(path: &Path) -> ProbeStream {
    let out = output(
        "ffprobe",
        &[
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-show_entries",
            "packet=size,pts",
            "-of",
            "compact",
            &path.display().to_string(),
        ],
    );
    assert!(
        out.status.success(),
        "ffprobe: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    let mut raw = Vec::new();
    for line in text.lines() {
        if line.starts_with("stream|") {
            for part in line.split('|').skip(1) {
                if let Some(value) = part.strip_prefix("sample_rate=") {
                    sample_rate = value.parse().unwrap_or(0);
                } else if let Some(value) = part.strip_prefix("channels=") {
                    channels = value.parse().unwrap_or(0);
                }
            }
        } else if line.starts_with("packet|") {
            let mut size = 0u64;
            let mut pts = 0i64;
            for part in line.split('|').skip(1) {
                if let Some(value) = part.strip_prefix("size=") {
                    size = value.parse().unwrap_or(0);
                } else if let Some(value) = part.strip_prefix("pts=") {
                    pts = value.parse().unwrap_or(0);
                }
            }
            raw.push((size, pts));
        }
    }
    assert!(sample_rate > 0, "ffprobe stream missing in {text}");
    let packets = raw
        .into_iter()
        .map(|(size, pts)| ProbePacket {
            size,
            pts_samples: ticks_to_samples(pts, sample_rate),
        })
        .collect();
    ProbeStream {
        sample_rate,
        channels,
        packets,
    }
}
