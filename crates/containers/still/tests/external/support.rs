//! Helpers for tests that shell out to independent tools.

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
        .unwrap_or_else(|error| panic!("{program} failed to start: {error}"))
}

pub struct ProbePacket {
    pub size: u64,
    pub pos: Option<u64>,
    pub pts: Option<i64>,
    pub duration: Option<i64>,
}

pub struct ProbeFile {
    pub codec: String,
    pub width: u32,
    pub height: u32,
    pub time_base: Option<String>,
    pub packets: Vec<ProbePacket>,
}

pub fn ffprobe(path: &Path) -> ProbeFile {
    let out = output(
        "ffprobe",
        &[
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,width,height,time_base",
            "-show_entries",
            "packet=size,pts,duration,pos",
            "-of",
            "compact",
            path.to_str().expect("path"),
        ],
    );
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "ffprobe {}\n{text}\n{}",
        path.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    let mut file = ProbeFile {
        codec: String::new(),
        width: 0,
        height: 0,
        time_base: None,
        packets: Vec::new(),
    };
    for line in text.lines() {
        let fields = fields(line);
        if line.starts_with("packet|") {
            file.packets.push(ProbePacket {
                size: fields
                    .get("size")
                    .and_then(|value| value.parse().ok())
                    .expect("size"),
                pos: fields.get("pos").and_then(|value| value.parse().ok()),
                pts: fields.get("pts").and_then(|value| value.parse().ok()),
                duration: fields.get("duration").and_then(|value| value.parse().ok()),
            });
        } else if line.starts_with("stream|") {
            file.codec = fields.get("codec_name").cloned().unwrap_or_default();
            file.width = fields
                .get("width")
                .and_then(|value| value.parse().ok())
                .unwrap_or(0);
            file.height = fields
                .get("height")
                .and_then(|value| value.parse().ok())
                .unwrap_or(0);
            file.time_base = fields.get("time_base").cloned();
        }
    }
    file
}

pub struct WebpFrame {
    pub offset: usize,
    pub duration: i64,
}

pub fn webpinfo(path: &Path) -> (Vec<WebpFrame>, Option<u16>) {
    let out = output("webpinfo", &[path.to_str().expect("path")]);
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "webpinfo {}\n{text}\n{}",
        path.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    let mut frames = Vec::new();
    let mut loop_count = None;
    let mut pending_offset = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("Chunk ANMF at offset") {
            let offset = rest
                .split(',')
                .next()
                .and_then(|value| value.trim().parse().ok())
                .expect("anmf offset");
            pending_offset = Some(offset);
        } else if let Some(rest) = line.strip_prefix("Duration:") {
            let duration = rest.trim().parse().expect("duration");
            let offset = pending_offset.take().expect("duration follows anmf");
            frames.push(WebpFrame { offset, duration });
        } else if let Some(rest) = line.strip_prefix("Loop count") {
            let value = rest.trim().trim_start_matches(':').trim();
            loop_count = value.parse().ok();
        }
    }
    (frames, loop_count)
}

fn fields(line: &str) -> std::collections::HashMap<String, String> {
    line.split('|')
        .filter_map(|part| {
            let (key, value) = part.split_once('=')?;
            Some((key.to_owned(), value.to_owned()))
        })
        .collect()
}
