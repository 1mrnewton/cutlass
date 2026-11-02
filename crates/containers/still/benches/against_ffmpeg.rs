//! Open one still picture per format beside ffmpeg.
//!
//! `cargo bench -p cutlass-still`
//!
//! Each sample is one 32×32 picture. Packet bytes are compared to
//! `ffmpeg -c copy` before anything is timed. The timed cutlass work is
//! `StillDemux::open` plus the packet read, in this process. The timed
//! ffmpeg work is `-c copy -f null` and includes process startup. The
//! reported time is the median of 3 after one warmup.

#![allow(clippy::print_stdout)]

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use cutlass_container::{Demux, Input};
use cutlass_still::{StillDemux, still_limits};

const WIDTH: u32 = 32;
const HEIGHT: u32 = 32;
const PICTURE: &str = "color=c=red:s=32x32:r=1:d=1";

struct Sample {
    name: &'static str,
    path: PathBuf,
    muxer: &'static str,
    writer: String,
}

struct Opened {
    bytes: Vec<u8>,
    width: u32,
    height: u32,
    packets: usize,
}

struct Span {
    median: Duration,
    min: Duration,
    max: Duration,
}

struct Measured {
    bytes: usize,
    width: u32,
    height: u32,
    cutlass: Span,
    ffmpeg: Span,
}

fn main() -> Result<(), String> {
    let Some(ffmpeg) = find_ffmpeg() else {
        println!("skip: ffmpeg is not installed");
        return Ok(());
    };
    let dir = std::env::temp_dir().join(format!("cutlass-still-bench-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|err| err.to_string())?;
    let outcome = run(&ffmpeg, &dir);
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

fn run(ffmpeg: &Path, dir: &Path) -> Result<(), String> {
    let version = ffmpeg_version(ffmpeg)?;
    let samples = write_samples(ffmpeg, dir)?;
    let mut measured = Vec::with_capacity(samples.len());
    for sample in &samples {
        measured.push(measure(ffmpeg, sample)?);
    }
    print_report(ffmpeg, &version, &samples, &measured);
    Ok(())
}

fn write_samples(ffmpeg: &Path, dir: &Path) -> Result<Vec<Sample>, String> {
    let jpeg = write_lavfi(ffmpeg, &dir.join("sample.jpg"))?;
    let png = write_lavfi(ffmpeg, &dir.join("sample.png"))?;
    let gif = write_lavfi(ffmpeg, &dir.join("sample.gif"))?;
    let webp_path = dir.join("sample.webp");
    let webp_writer = write_webp(ffmpeg, &png, &webp_path)?;
    Ok(vec![
        Sample {
            name: "jpeg",
            path: jpeg,
            muxer: "mjpeg",
            writer: "ffmpeg".to_string(),
        },
        Sample {
            name: "png",
            path: png,
            muxer: "image2pipe",
            writer: "ffmpeg".to_string(),
        },
        Sample {
            name: "webp",
            path: webp_path,
            muxer: "webp",
            writer: webp_writer,
        },
        Sample {
            name: "gif",
            path: gif,
            muxer: "gif",
            writer: "ffmpeg".to_string(),
        },
    ])
}

fn write_lavfi(ffmpeg: &Path, output: &Path) -> Result<PathBuf, String> {
    let command = Command::new(ffmpeg)
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            PICTURE,
            "-frames:v",
            "1",
        ])
        .arg(output)
        .output()
        .map_err(|err| err.to_string())?;
    if !command.status.success() {
        return Err(format!(
            "ffmpeg did not write {}: {}",
            output.display(),
            String::from_utf8_lossy(&command.stderr)
        ));
    }
    Ok(output.to_path_buf())
}

fn write_webp(ffmpeg: &Path, png: &Path, webp: &Path) -> Result<String, String> {
    if let Some(encoder) = webp_encoder(ffmpeg)? {
        let command = Command::new(ffmpeg)
            .args([
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                PICTURE,
                "-frames:v",
                "1",
                "-c:v",
                &encoder,
            ])
            .arg(webp)
            .output()
            .map_err(|err| err.to_string())?;
        if !command.status.success() {
            return Err(format!(
                "ffmpeg did not write the webp: {}",
                String::from_utf8_lossy(&command.stderr)
            ));
        }
        return Ok("ffmpeg".to_string());
    }
    let Some(cwebp) = find_cwebp() else {
        return Err("ffmpeg has no WebP encoder and cwebp is not installed".to_string());
    };
    let command = Command::new(&cwebp)
        .args(["-quiet", "-q", "80"])
        .arg(png)
        .arg("-o")
        .arg(webp)
        .output()
        .map_err(|err| err.to_string())?;
    if !command.status.success() {
        return Err(format!(
            "cwebp did not write the webp: {}",
            String::from_utf8_lossy(&command.stderr)
        ));
    }
    Ok(format!("cwebp {}", tool_version(&cwebp)?))
}

fn webp_encoder(ffmpeg: &Path) -> Result<Option<String>, String> {
    let output = Command::new(ffmpeg)
        .args(["-hide_banner", "-encoders"])
        .stderr(Stdio::null())
        .output()
        .map_err(|err| err.to_string())?;
    if !output.status.success() {
        return Err("ffmpeg -encoders failed".to_string());
    }
    let text = String::from_utf8_lossy(&output.stdout);
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        let flags = fields.next().unwrap_or("");
        let name = fields.next().unwrap_or("");
        if flags.contains('V') && (name == "libwebp" || name == "webp") {
            return Ok(Some(name.to_string()));
        }
    }
    Ok(None)
}

fn measure(ffmpeg: &Path, sample: &Sample) -> Result<Measured, String> {
    let opened = open_still(&sample.path)?;
    let copied = ffmpeg_copy(ffmpeg, &sample.path, sample.muxer)?;
    if opened.bytes != copied {
        return Err(format!(
            "{} packet bytes differ from ffmpeg: cutlass {} bytes, ffmpeg {} bytes",
            sample.name,
            opened.bytes.len(),
            copied.len()
        ));
    }
    if opened.width != WIDTH || opened.height != HEIGHT {
        return Err(format!(
            "{} picture is {}x{}",
            sample.name, opened.width, opened.height
        ));
    }
    if opened.packets != 1 {
        return Err(format!(
            "{} opened as {} packets",
            sample.name, opened.packets
        ));
    }
    let bytes = opened.bytes.len();
    let cutlass = time_runs(|| {
        let again = open_still(&sample.path)?;
        let len = std::hint::black_box(again.bytes.len());
        if len != bytes {
            return Err(format!(
                "{} opened {len} bytes during timing, compared {bytes}",
                sample.name
            ));
        }
        Ok(())
    })?;
    let ffmpeg_span = time_runs(|| ffmpeg_null(ffmpeg, &sample.path))?;
    Ok(Measured {
        bytes,
        width: opened.width,
        height: opened.height,
        cutlass,
        ffmpeg: ffmpeg_span,
    })
}

fn open_still(path: &Path) -> Result<Opened, String> {
    let file = std::fs::File::open(path).map_err(|err| format!("{}: {err}", path.display()))?;
    let length = file.metadata().map_err(|err| err.to_string())?.len();
    let mut demux = StillDemux::open(Input::random(file, length, still_limits()))
        .map_err(|err| format!("{}: {err}", path.display()))?;
    let size = demux
        .streams()
        .first()
        .and_then(|stream| stream.params().size())
        .ok_or_else(|| format!("{} has no picture size", path.display()))?;
    let mut bytes = Vec::new();
    let mut packets = 0usize;
    while let Some(packet) = demux.read_packet().map_err(|err| err.to_string())? {
        bytes.extend_from_slice(packet.bytes());
        packets += 1;
    }
    if bytes.is_empty() {
        return Err(format!("{} opened with no packet bytes", path.display()));
    }
    Ok(Opened {
        bytes,
        width: size.width(),
        height: size.height(),
        packets,
    })
}

fn ffmpeg_copy(ffmpeg: &Path, input: &Path, muxer: &str) -> Result<Vec<u8>, String> {
    let output = Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(input)
        .args(["-c", "copy", "-f", muxer, "pipe:1"])
        .output()
        .map_err(|err| err.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "ffmpeg copy of {} failed: {}",
            input.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(output.stdout)
}

fn ffmpeg_null(ffmpeg: &Path, input: &Path) -> Result<(), String> {
    let output = Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(input)
        .args(["-c", "copy", "-f", "null", "-"])
        .output()
        .map_err(|err| err.to_string())?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "ffmpeg open of {} failed: {}",
            input.display(),
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

fn time_runs(mut once: impl FnMut() -> Result<(), String>) -> Result<Span, String> {
    once()?;
    let mut samples = [Duration::ZERO; 3];
    for sample in &mut samples {
        let started = Instant::now();
        once()?;
        *sample = started.elapsed();
    }
    Ok(Span::from_samples(samples))
}

impl Span {
    fn from_samples(mut samples: [Duration; 3]) -> Self {
        let mut min = samples[0];
        let mut max = samples[0];
        for sample in samples {
            if sample < min {
                min = sample;
            }
            if sample > max {
                max = sample;
            }
        }
        samples.sort();
        Self {
            median: samples[1],
            min,
            max,
        }
    }
}

fn print_report(ffmpeg: &Path, version: &str, samples: &[Sample], measured: &[Measured]) {
    println!("cutlass-still open versus ffmpeg");
    println!("{version}");
    println!("ffmpeg: {}", ffmpeg.display());
    println!("sample: one {WIDTH}x{HEIGHT} picture");
    println!("reps: median of 3 after 1 warmup");
    println!("cutlass times StillDemux::open and the packet read in this process");
    println!("ffmpeg times -c copy -f null and includes process startup");
    println!("packet bytes were compared to ffmpeg -c copy before timing");
    for (sample, result) in samples.iter().zip(measured) {
        println!(
            "{} {}x{} {} bytes match ffmpeg, written by {}",
            sample.name, result.width, result.height, result.bytes, sample.writer
        );
    }
    println!();
    println!(
        "{:<8} {:<8} {:>12} {:>12} {:>12} {:>8}",
        "format", "side", "median", "min", "max", "bytes"
    );
    for (sample, result) in samples.iter().zip(measured) {
        print_row(sample.name, "cutlass", &result.cutlass, result.bytes);
        print_row(sample.name, "ffmpeg", &result.ffmpeg, result.bytes);
    }
}

fn print_row(format: &str, side: &str, span: &Span, bytes: usize) {
    println!(
        "{format:<8} {side:<8} {:>12} {:>12} {:>12} {bytes:>8}",
        show(span.median),
        show(span.min),
        show(span.max),
    );
}

fn show(duration: Duration) -> String {
    format!("{:.3} ms", duration.as_secs_f64() * 1000.0)
}

fn ffmpeg_version(ffmpeg: &Path) -> Result<String, String> {
    let output = Command::new(ffmpeg)
        .arg("-version")
        .stderr(Stdio::null())
        .output()
        .map_err(|err| err.to_string())?;
    if !output.status.success() {
        return Err("ffmpeg -version failed".to_string());
    }
    let text = String::from_utf8_lossy(&output.stdout);
    text.lines()
        .next()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToString::to_string)
        .ok_or_else(|| "ffmpeg -version printed nothing".to_string())
}

fn tool_version(path: &Path) -> Result<String, String> {
    let output = Command::new(path)
        .arg("-version")
        .output()
        .map_err(|err| err.to_string())?;
    let text = String::from_utf8_lossy(&output.stdout);
    let err = String::from_utf8_lossy(&output.stderr);
    text.lines()
        .chain(err.lines())
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(ToString::to_string)
        .ok_or_else(|| format!("{} -version printed nothing", path.display()))
}

fn find_ffmpeg() -> Option<PathBuf> {
    let homebrew = PathBuf::from("/opt/homebrew/bin/ffmpeg");
    if homebrew.is_file() && runs(&homebrew) {
        return Some(homebrew);
    }
    which("ffmpeg").filter(|path| runs(path))
}

fn find_cwebp() -> Option<PathBuf> {
    let homebrew = PathBuf::from("/opt/homebrew/bin/cwebp");
    if homebrew.is_file() {
        return Some(homebrew);
    }
    which("cwebp")
}

fn which(name: &str) -> Option<PathBuf> {
    let output = Command::new("which").arg(name).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let line = text.lines().next()?.trim();
    if line.is_empty() {
        None
    } else {
        Some(PathBuf::from(line))
    }
}

fn runs(path: &Path) -> bool {
    Command::new(path)
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}
