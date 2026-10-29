//! Decode one real AV1 keyframe, then the same bytes through FFmpeg.
//!
//! FFmpeg 8 decodes AV1 by calling dav1d. This bench does not link either
//! program. It times `Av1Decoder::push` in this process and FFmpeg's wall
//! time on a loop of the same frame. The FFmpeg figure subtracts a one-frame
//! run from a longer run so process startup is not counted as decode time.
//!
//! The frame comes from `../local-files` when that folder is present. Without
//! it, the bench reports that the comparison was not run.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use cutlass_av1::{AsView, Av1Decoder};
use cutlass_decode::{CodecId, CodecParams, Decoder, Packet};

const FFMPEG_LOOPS: u32 = 20;
const CUTLASS_SAMPLES: usize = 5;

fn main() {
    match run() {
        Ok(()) => {}
        Err(what) => {
            let _ = writeln!(std::io::stdout(), "bench not run: {what}");
        }
    }
}

fn run() -> Result<(), String> {
    let input = production_file().ok_or("no AV1 file in ../local-files")?;
    let dir = std::env::temp_dir().join(format!("cutlass-av1-bench-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|_| "temp directory".to_string())?;
    let ivf = dir.join("frame.ivf");
    let yuv = dir.join("frame.yuv");
    let outcome = compare(&input, &ivf, &yuv);
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

fn compare(input: &Path, ivf: &Path, yuv: &Path) -> Result<(), String> {
    extract_frame(input, ivf)?;
    let file = std::fs::read(ivf).map_err(|_| "read ivf".to_string())?;
    let packet = ivf_payload(&file)?;
    let (width, height) = ivf_size(&file)?;
    let name = input
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("file");
    say(&format!(
        "workload: {width}x{height} keyframe, {} bytes, from {name}",
        packet.len()
    ));
    say(&machine());
    say(
        "ffmpeg is dav1d through libavcodec. cutlass decodes tiles in parallel and filters rows in parallel, bench profile, no link-time optimization.",
    );

    let ours = decode_once(packet)?;
    let reference = ffmpeg_raw(ivf, yuv)?;
    if ours == reference {
        say("pixels: match");
    } else {
        let diffs = ours
            .iter()
            .zip(reference.iter())
            .filter(|(left, right)| left != right)
            .count();
        say(&format!(
            "pixels: differ ({} of ours {}, ffmpeg {})",
            diffs,
            ours.len(),
            reference.len()
        ));
    }

    let mut cutlass = Vec::new();
    let _ = decode_once(packet)?;
    for _ in 0..CUTLASS_SAMPLES {
        cutlass.push(time_cutlass(packet)?);
    }
    report("cutlass-av1", &mut cutlass);

    let ffmpeg_one = time_ffmpeg(ivf, 1, Some(1))?;
    let ffmpeg_many = time_ffmpeg(ivf, FFMPEG_LOOPS, Some(1))?;
    report_ffmpeg("ffmpeg -threads 1", ffmpeg_one, ffmpeg_many)?;

    let default_one = time_ffmpeg(ivf, 1, None)?;
    let default_many = time_ffmpeg(ivf, FFMPEG_LOOPS, None)?;
    report_ffmpeg("ffmpeg default threads", default_one, default_many)?;
    Ok(())
}

fn report(name: &str, samples: &mut [f64]) {
    let Some(mid) = median(samples) else {
        return;
    };
    let min = samples.first().copied().unwrap_or(mid);
    let max = samples.last().copied().unwrap_or(mid);
    say(&format!(
        "{name}: {mid:.4} s/frame  min {min:.4}  max {max:.4}  n={}",
        samples.len()
    ));
}

fn report_ffmpeg(name: &str, once: f64, many: f64) -> Result<(), String> {
    if many <= once {
        return Err(format!("{name}: longer run was not slower than one frame"));
    }
    let per = (many - once) / f64::from(FFMPEG_LOOPS - 1);
    say(&format!(
        "{name}: {per:.4} s/frame  (one process {once:.4} s, {FFMPEG_LOOPS} frames {many:.4} s)"
    ));
    Ok(())
}

fn time_cutlass(packet: &[u8]) -> Result<f64, String> {
    let params = CodecParams::video(CodecId::Av1, 64, 64);
    let mut decoder = Av1Decoder::open(&params).map_err(|error| format!("open {error:?}"))?;
    let frame = Packet {
        stream: 0,
        data: packet.to_vec(),
        pts: 0,
        duration: 1,
        timescale: 30,
        keyframe: true,
        side: Vec::new(),
    };
    let start = Instant::now();
    let decoded = decoder
        .push(&frame)
        .map_err(|error| format!("push {error:?}"))?;
    let elapsed = start.elapsed().as_secs_f64();
    let picture = decoded
        .first()
        .and_then(|frame| frame.as_video())
        .ok_or("no picture")?;
    std::hint::black_box(picture.picture.y.samples().len());
    Ok(elapsed)
}

fn decode_once(packet: &[u8]) -> Result<Vec<u8>, String> {
    let params = CodecParams::video(CodecId::Av1, 64, 64);
    let mut decoder = Av1Decoder::open(&params).map_err(|error| format!("open {error:?}"))?;
    let decoded = decoder
        .push(&Packet {
            stream: 0,
            data: packet.to_vec(),
            pts: 0,
            duration: 1,
            timescale: 30,
            keyframe: true,
            side: Vec::new(),
        })
        .map_err(|error| format!("push {error:?}"))?;
    let picture = decoded
        .first()
        .and_then(|frame| frame.as_video())
        .ok_or("no picture")?;
    let mut packed = picture.picture.y.samples().to_vec();
    packed.extend_from_slice(picture.picture.u.samples());
    packed.extend_from_slice(picture.picture.v.samples());
    Ok(packed)
}

fn time_ffmpeg(ivf: &Path, frames: u32, threads: Option<u32>) -> Result<f64, String> {
    let mut command = Command::new("ffmpeg");
    command.arg("-hide_banner").arg("-nostats");
    command.args(["-c:v", "libdav1d"]);
    if let Some(threads) = threads {
        command.arg("-threads").arg(threads.to_string());
    }
    if frames > 1 {
        command.arg("-stream_loop").arg((frames - 1).to_string());
    }
    command
        .arg("-i")
        .arg(ivf)
        .args(["-an", "-benchmark", "-f", "null", "-"]);
    let output = command
        .output()
        .map_err(|_| "ffmpeg is not runnable".to_string())?;
    if !output.status.success() {
        return Err("ffmpeg decode failed".to_string());
    }
    let text = String::from_utf8_lossy(&output.stderr);
    rtime(&text).ok_or("ffmpeg did not report rtime".to_string())
}

fn ffmpeg_raw(ivf: &Path, yuv: &Path) -> Result<Vec<u8>, String> {
    let output = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-threads",
            "1",
            "-c:v",
            "libdav1d",
            "-i",
        ])
        .arg(ivf)
        .args([
            "-an",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(yuv)
        .output()
        .map_err(|_| "ffmpeg is not runnable".to_string())?;
    if !output.status.success() {
        return Err("ffmpeg raw decode failed".to_string());
    }
    std::fs::read(yuv).map_err(|_| "read yuv".to_string())
}

fn extract_frame(input: &Path, ivf: &Path) -> Result<(), String> {
    let output = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(input)
        .args(["-frames:v", "1", "-c:v", "copy", "-f", "ivf"])
        .arg(ivf)
        .output()
        .map_err(|_| "ffmpeg is not runnable".to_string())?;
    if output.status.success() {
        Ok(())
    } else {
        Err("ffmpeg could not copy the first frame".to_string())
    }
}

fn rtime(text: &str) -> Option<f64> {
    for line in text.lines() {
        let Some(rest) = line.split("rtime=").nth(1) else {
            continue;
        };
        let number = rest.trim_end_matches('s').trim();
        return number.parse().ok();
    }
    None
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        Some((values[mid - 1] + values[mid]) / 2.0)
    } else {
        values.get(mid).copied()
    }
}

fn production_file() -> Option<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files");
    let mut entries = std::fs::read_dir(&root).ok()?;
    entries.find_map(|entry| {
        let path = entry.ok()?.path();
        let name = path.file_name()?.to_str()?.to_ascii_lowercase();
        let video = name.ends_with(".mp4") || name.ends_with(".webm") || name.ends_with(".mov");
        (video && name.contains("av1")).then_some(path)
    })
}

fn ivf_payload(file: &[u8]) -> Result<&[u8], String> {
    if file.len() < 44 || &file[..4] != b"DKIF" {
        return Err("not an ivf".to_string());
    }
    let size = u32::from_le_bytes([file[32], file[33], file[34], file[35]]) as usize;
    let end = 44usize.checked_add(size).ok_or("ivf size")?;
    file.get(44..end).ok_or("truncated ivf".to_string())
}

fn ivf_size(file: &[u8]) -> Result<(u16, u16), String> {
    if file.len() < 16 {
        return Err("short ivf".to_string());
    }
    Ok((
        u16::from_le_bytes([file[12], file[13]]),
        u16::from_le_bytes([file[14], file[15]]),
    ))
}

fn machine() -> String {
    let cpu = Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown cpu".to_string());
    let cores = Command::new("sysctl")
        .args(["-n", "hw.ncpu"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .unwrap_or_else(|| "?".to_string());
    format!("machine: {cpu}, {cores} cores")
}

fn say(line: &str) {
    let _ = writeln!(std::io::stdout(), "{line}");
}
