//! Open one PNG that ffmpeg writes and print its picture size.
//!
//! `cargo run --release -p cutlass-still --example open`

#![allow(clippy::print_stdout)]

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use cutlass_container::{Demux, Input};
use cutlass_still::{StillDemux, StillKind, still_limits};

fn main() -> Result<(), String> {
    let Some(ffmpeg) = find_ffmpeg() else {
        return Err("ffmpeg is not installed".to_string());
    };
    let dir = std::env::temp_dir().join(format!("cutlass-still-open-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|err| err.to_string())?;
    let outcome = print_size(&ffmpeg, &dir);
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

fn print_size(ffmpeg: &Path, dir: &Path) -> Result<(), String> {
    let png = dir.join("sample.png");
    write_png(ffmpeg, &png)?;
    let file = std::fs::File::open(&png).map_err(|err| err.to_string())?;
    let length = file.metadata().map_err(|err| err.to_string())?.len();
    let demux = StillDemux::open(Input::random(file, length, still_limits()))
        .map_err(|err| err.to_string())?;
    if demux.kind() != StillKind::Png {
        return Err("opened file is not a png".to_string());
    }
    let size = demux
        .streams()
        .first()
        .and_then(|stream| stream.params().size())
        .ok_or_else(|| "opened png has no picture size".to_string())?;
    println!("{}x{}", size.width(), size.height());
    Ok(())
}

fn write_png(ffmpeg: &Path, png: &Path) -> Result<(), String> {
    let output = Command::new(ffmpeg)
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=32x32:r=1:d=1",
            "-frames:v",
            "1",
        ])
        .arg(png)
        .output()
        .map_err(|err| err.to_string())?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "ffmpeg did not write the png: {}",
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

fn find_ffmpeg() -> Option<PathBuf> {
    let homebrew = PathBuf::from("/opt/homebrew/bin/ffmpeg");
    if homebrew.is_file() && runs(&homebrew) {
        return Some(homebrew);
    }
    let output = Command::new("which").arg("ffmpeg").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let line = text.lines().next()?.trim();
    if line.is_empty() {
        return None;
    }
    let path = PathBuf::from(line);
    runs(&path).then_some(path)
}

fn runs(path: &Path) -> bool {
    Command::new(path)
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}
