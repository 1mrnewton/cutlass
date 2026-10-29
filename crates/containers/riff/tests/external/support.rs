//! Helpers for tests that shell out to independent tools.

use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

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

pub fn output_timeout(program: &str, args: &[&str], limit: Duration) -> Option<Output> {
    let mut child = Command::new(program)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;
    let started = Instant::now();
    loop {
        if started.elapsed() > limit {
            let _ = child.kill();
            let _ = child.wait();
            return None;
        }
        match child.try_wait() {
            Ok(Some(_)) => return child.wait_with_output().ok(),
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(_) => return None,
        }
    }
}

pub struct ProbeStream {
    pub sample_rate: u32,
    pub channels: u16,
    pub duration_ts: Option<u64>,
    pub codec_name: String,
}

pub fn ffprobe_stream(path: &Path) -> ProbeStream {
    let out = output(
        "ffprobe",
        &[
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,duration_ts,codec_name",
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
    let mut duration_ts = None;
    let mut codec_name = String::new();
    for line in text.lines() {
        if !line.starts_with("stream|") {
            continue;
        }
        for part in line.split('|').skip(1) {
            if let Some(value) = part.strip_prefix("sample_rate=") {
                sample_rate = value.parse().unwrap_or(0);
            } else if let Some(value) = part.strip_prefix("channels=") {
                channels = value.parse().unwrap_or(0);
            } else if let Some(value) = part.strip_prefix("duration_ts=") {
                duration_ts = value.parse().ok();
            } else if let Some(value) = part.strip_prefix("codec_name=") {
                codec_name = value.to_owned();
            }
        }
    }
    ProbeStream {
        sample_rate,
        channels,
        duration_ts,
        codec_name,
    }
}

/// Python's `wave` module. It is not libavformat.
pub fn python_wave(path: &Path, dest: &Path) -> Option<(u16, u32, u16, u64)> {
    if !tool_on_path("python3") {
        return None;
    }
    let script = "import wave,sys\nw=wave.open(sys.argv[1],'rb')\nframes=w.readframes(w.getnframes())\nopen(sys.argv[2],'wb').write(frames)\nprint(w.getnchannels(), w.getframerate(), w.getsampwidth(), w.getnframes())\n";
    let out = output(
        "python3",
        &[
            "-c",
            script,
            &path.display().to_string(),
            &dest.display().to_string(),
        ],
    );
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut parts = text.split_whitespace();
    let channels = parts.next()?.parse().ok()?;
    let rate = parts.next()?.parse().ok()?;
    let width = parts.next()?.parse().ok()?;
    let frames = parts.next()?.parse().ok()?;
    Some((channels, rate, width, frames))
}
