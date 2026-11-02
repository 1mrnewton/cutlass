//! Open one short MP4 that ffmpeg writes and print the demux result.
//!
//! The file is a few frames of `testsrc2` plus a sine. The timestamp is the
//! first packet's presentation time in that track's file timescale.

#![allow(clippy::print_stdout)]

use cutlass_isobmff::{Demux, FileSource, IsoBmff};
use std::path::{Path, PathBuf};
use std::process::Command;

const FFMPEG: &str = "/opt/homebrew/bin/ffmpeg";

fn main() -> Result<(), String> {
    let ffmpeg = PathBuf::from(FFMPEG);
    if !ffmpeg.is_file() {
        println!("skip: ffmpeg is not installed");
        return Ok(());
    }
    let dir = std::env::temp_dir().join("cutlass-isobmff-demux-example");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(|err| err.to_string())?;
    let path = dir.join("sample.mp4");
    write_sample(&ffmpeg, &path)?;
    let (streams, packets, first_pts) = read_summary(&path)?;
    println!("streams {streams}");
    println!("packets {packets}");
    println!("first timestamp {first_pts}");
    Ok(())
}

fn write_sample(ffmpeg: &Path, path: &Path) -> Result<(), String> {
    let path = path.to_str().ok_or("sample path is not utf-8")?;
    let output = Command::new(ffmpeg)
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=10:duration=0.4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:sample_rate=48000:duration=0.4",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "10",
            "-bf",
            "0",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            "-shortest",
            "-use_editlist",
            "0",
            path,
        ])
        .output()
        .map_err(|err| err.to_string())?;
    if output.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).into_owned())
    }
}

fn read_summary(path: &Path) -> Result<(usize, usize, i64), String> {
    let mut source = FileSource::open(path).map_err(|err| err.to_string())?;
    let mut movie = IsoBmff::open(&mut source).map_err(|err| err.to_string())?;
    let streams = movie.streams().len();
    let mut packets = 0usize;
    let mut first_pts = None;
    while let Some(packet) = movie
        .read_packet(&mut source)
        .map_err(|err| err.to_string())?
    {
        if first_pts.is_none() {
            first_pts = Some(packet.pts);
        }
        packets += 1;
    }
    let Some(first_pts) = first_pts else {
        return Err("the sample has no packets".to_string());
    };
    Ok((streams, packets, first_pts))
}
