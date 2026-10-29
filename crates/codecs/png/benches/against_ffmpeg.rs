//! Release timing of this PNG codec beside ffmpeg on the same pictures.
//!
//! `cargo bench -p cutlass-png` runs it. The numbers are wall time, median of
//! several passes after one warmup. Cutlass stays in this process. ffmpeg is
//! a new process each pass, so its rows include process start. A separate
//! `ffmpeg -version` row is that start cost.
//!
//! Encode compared here is RGBA8. ffmpeg's default prediction is Paeth and
//! its deflate level is zlib's default. A second ffmpeg row uses `-pred mixed`,
//! which searches filters the way this encoder does. This encoder still uses
//! its own fixed-Huffman deflate. Decode rows share one PNG file.

#![allow(clippy::print_stdout)]

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use cutlass_decode::{CodecId, CodecParams, Decoder, Packet};
use cutlass_encode::{BitDepth, ColorType, Encoder, PictureInput, StillImage};
use cutlass_png::{PngDecoder, PngEncoder};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        self.0
    }
}

struct Sample {
    median: Duration,
    min: Duration,
    max: Duration,
    bytes: usize,
}

struct Row {
    picture: String,
    side: String,
    op: &'static str,
    sample: Sample,
    note: String,
}

fn main() -> Result<(), String> {
    let reps = reps();
    let dir = std::env::temp_dir().join(format!("cutlass-png-bench-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|err| err.to_string())?;
    let outcome = run(&dir, reps);
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

fn run(dir: &Path, reps: usize) -> Result<(), String> {
    let ffmpeg = find_ffmpeg();
    let mut rows = Vec::new();
    if let Some(bin) = &ffmpeg {
        rows.push(version_row(bin, reps)?);
    }

    let gradient = gradient();
    measure_picture(
        dir,
        reps,
        ffmpeg.as_deref(),
        "1920x1080 gradient",
        &gradient,
        &mut rows,
    )?;

    let noise = noise();
    measure_picture(
        dir,
        reps,
        ffmpeg.as_deref(),
        "1920x1080 noise",
        &noise,
        &mut rows,
    )?;

    if let Some(path) = local_png() {
        measure_file(dir, reps, ffmpeg.as_deref(), &path, &mut rows)?;
    } else {
        rows.push(missing(
            "local png",
            "04-motion-graphics.png is not beside this checkout",
        ));
    }

    print_report(reps, ffmpeg.as_deref(), &rows);
    Ok(())
}

fn measure_picture(
    dir: &Path,
    reps: usize,
    ffmpeg: Option<&Path>,
    name: &str,
    image: &StillImage,
    rows: &mut Vec<Row>,
) -> Result<(), String> {
    let raw = dir.join(format!("{name}.raw"));
    let png = dir.join(format!("{name}.png"));
    std::fs::write(&raw, pack8(&image.samples)?).map_err(|err| err.to_string())?;

    let input = PictureInput {
        image: image.clone(),
    };
    let mut encoder = PngEncoder::open(&png_params()).map_err(|err| format!("{err:?}"))?;
    rows.push(cutlass_encode(name, &mut encoder, &input, reps)?);
    let packets = encoder
        .push_picture(&input)
        .map_err(|err| format!("{err:?}"))?;
    let encoded = packets
        .first()
        .map(|packet| packet.data.clone())
        .ok_or_else(|| "encoder returned no packet".to_string())?;
    std::fs::write(&png, &encoded).map_err(|err| err.to_string())?;

    let mut decoder = PngDecoder::open(&png_params()).map_err(|err| format!("{err:?}"))?;
    let packet = Packet::audio(encoded, 0, 0, 0);
    rows.push(cutlass_decode(
        name,
        &mut decoder,
        &packet,
        reps,
        "png written by cutlass",
    )?);

    let Some(bin) = ffmpeg else {
        rows.push(missing(name, "ffmpeg was not found"));
        return Ok(());
    };
    confirm_same_pixels(bin, &png, &image.samples)?;
    rows.push(ffmpeg_encode(
        bin,
        name,
        &raw,
        image.width,
        image.height,
        reps,
        None,
    )?);
    rows.push(ffmpeg_encode(
        bin,
        name,
        &raw,
        image.width,
        image.height,
        reps,
        Some("mixed"),
    )?);
    rows.push(ffmpeg_decode(
        bin,
        name,
        &png,
        reps,
        "png written by cutlass",
    )?);
    Ok(())
}

fn measure_file(
    dir: &Path,
    reps: usize,
    ffmpeg: Option<&Path>,
    path: &Path,
    rows: &mut Vec<Row>,
) -> Result<(), String> {
    let name = "motion-graphics.png";
    let bytes = std::fs::read(path).map_err(|err| err.to_string())?;
    let mut decoder = PngDecoder::open(&png_params()).map_err(|err| format!("{err:?}"))?;
    let packet = Packet::audio(bytes, 0, 0, 0);
    rows.push(cutlass_decode(
        name,
        &mut decoder,
        &packet,
        reps,
        "the file on disk",
    )?);
    let pictures = decoder
        .decode_picture(&packet)
        .map_err(|err| format!("{err:?}"))?;
    let image = pictures
        .first()
        .map(|picture| picture.image.clone())
        .ok_or_else(|| "decoder returned no picture".to_string())?;

    let Some(bin) = ffmpeg else {
        rows.push(missing(name, "ffmpeg was not found"));
        return Ok(());
    };
    confirm_same_pixels(bin, path, &image.samples)?;
    rows.push(ffmpeg_decode(bin, name, path, reps, "the file on disk")?);

    if image.color != ColorType::Rgba || image.depth != BitDepth::Eight {
        return Ok(());
    }
    let raw = dir.join("motion-graphics.raw");
    std::fs::write(&raw, pack8(&image.samples)?).map_err(|err| err.to_string())?;
    let input = PictureInput { image };
    let mut encoder = PngEncoder::open(&png_params()).map_err(|err| format!("{err:?}"))?;
    rows.push(cutlass_encode(name, &mut encoder, &input, reps)?);
    rows.push(ffmpeg_encode(
        bin,
        name,
        &raw,
        input.image.width,
        input.image.height,
        reps,
        None,
    )?);
    rows.push(ffmpeg_encode(
        bin,
        name,
        &raw,
        input.image.width,
        input.image.height,
        reps,
        Some("mixed"),
    )?);
    Ok(())
}

fn cutlass_encode(
    picture: &str,
    encoder: &mut PngEncoder,
    input: &PictureInput,
    reps: usize,
) -> Result<Row, String> {
    let sample = time(reps, || {
        let packets = encoder
            .push_picture(input)
            .map_err(|err| format!("{err:?}"))?;
        let bytes = packets.iter().map(|packet| packet.data.len()).sum();
        Ok(std::hint::black_box(bytes))
    })?;
    Ok(Row {
        picture: picture.to_string(),
        side: "cutlass".to_string(),
        op: "encode",
        sample,
        note: "filter search, fixed Huffman".to_string(),
    })
}

fn cutlass_decode(
    picture: &str,
    decoder: &mut PngDecoder,
    packet: &Packet,
    reps: usize,
    note: &str,
) -> Result<Row, String> {
    let sample = time(reps, || {
        let pictures = decoder
            .decode_picture(packet)
            .map_err(|err| format!("{err:?}"))?;
        let samples = pictures
            .iter()
            .map(|picture| picture.image.samples.len())
            .sum();
        Ok(std::hint::black_box(samples))
    })?;
    Ok(Row {
        picture: picture.to_string(),
        side: "cutlass".to_string(),
        op: "decode",
        sample,
        note: note.to_string(),
    })
}

fn ffmpeg_encode(
    bin: &Path,
    picture: &str,
    raw: &Path,
    width: u32,
    height: u32,
    reps: usize,
    pred: Option<&str>,
) -> Result<Row, String> {
    let sample = time(reps, || {
        let mut command = Command::new(bin);
        command
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgba",
                "-s",
                &format!("{width}x{height}"),
                "-i",
            ])
            .arg(raw);
        command.args(["-frames:v", "1", "-c:v", "png"]);
        if let Some(pred) = pred {
            command.args(["-pred", pred]);
        }
        command.args(["-f", "image2pipe", "pipe:1"]);
        let (_elapsed, stdout) = ffmpeg_output(command)?;
        Ok(stdout.len())
    })?;
    let note = match pred {
        Some(pred) => format!("-pred {pred}, zlib default"),
        None => "default pred paeth, zlib default".to_string(),
    };
    Ok(Row {
        picture: picture.to_string(),
        side: "ffmpeg".to_string(),
        op: "encode",
        sample,
        note,
    })
}

fn ffmpeg_decode(
    bin: &Path,
    picture: &str,
    png: &Path,
    reps: usize,
    note: &str,
) -> Result<Row, String> {
    let sample = time(reps, || {
        let mut command = Command::new(bin);
        command
            .args(["-hide_banner", "-loglevel", "error", "-i"])
            .arg(png)
            .args(["-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"]);
        let (_elapsed, stdout) = ffmpeg_output(command)?;
        Ok(stdout.len())
    })?;
    Ok(Row {
        picture: picture.to_string(),
        side: "ffmpeg".to_string(),
        op: "decode",
        sample,
        note: note.to_string(),
    })
}

fn version_row(bin: &Path, reps: usize) -> Result<Row, String> {
    let sample = time(reps, || {
        let mut command = Command::new(bin);
        command.arg("-version").stderr(Stdio::null());
        let (_elapsed, _stdout) = ffmpeg_output(command)?;
        Ok(0)
    })?;
    Ok(Row {
        picture: "process start".to_string(),
        side: "ffmpeg".to_string(),
        op: "version",
        sample,
        note: ffmpeg_version(bin)?,
    })
}

fn confirm_same_pixels(bin: &Path, png: &Path, samples: &[u16]) -> Result<(), String> {
    let mut command = Command::new(bin);
    command
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(png)
        .args(["-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"]);
    let (_elapsed, foreign) = ffmpeg_output(command)?;
    let ours = pack8(samples)?;
    if foreign != ours {
        return Err(format!(
            "pixel mismatch on {}: cutlass {} bytes, ffmpeg {} bytes",
            png.display(),
            ours.len(),
            foreign.len()
        ));
    }
    Ok(())
}

fn time(reps: usize, mut once: impl FnMut() -> Result<usize, String>) -> Result<Sample, String> {
    let _ = once()?;
    let mut times = Vec::with_capacity(reps);
    let mut bytes = 0;
    for _ in 0..reps {
        let started = Instant::now();
        bytes = once()?;
        times.push(started.elapsed());
    }
    let min = times.iter().copied().min().unwrap_or_default();
    let max = times.iter().copied().max().unwrap_or_default();
    Ok(Sample {
        median: median(&mut times),
        min,
        max,
        bytes,
    })
}

fn ffmpeg_output(mut command: Command) -> Result<(Duration, Vec<u8>), String> {
    command.stdin(Stdio::null());
    let started = Instant::now();
    let output = command.output().map_err(|err| err.to_string())?;
    let elapsed = started.elapsed();
    if !output.status.success() {
        return Err(format!(
            "ffmpeg exited {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok((elapsed, output.stdout))
}

fn median(times: &mut [Duration]) -> Duration {
    times.sort();
    times.get(times.len() / 2).copied().unwrap_or_default()
}

fn print_report(reps: usize, ffmpeg: Option<&Path>, rows: &[Row]) {
    println!("PNG wall time, release, one thread");
    println!("reps: {reps} after 1 warmup, median of the timed passes");
    println!(
        "encode out is the PNG size. decode out is the raw sample count, the same number as ffmpeg's rgba bytes."
    );
    println!(
        "cutlass decode stores u16 samples. An ffmpeg row includes process start. That start is the version row."
    );
    match ffmpeg {
        Some(path) => println!("ffmpeg: {}", path.display()),
        None => println!("ffmpeg: not found, cutlass rows only"),
    }
    println!();
    let header = format!(
        "{:<24} {:<8} {:<8} {:>8} {:>8} {:>8} {:>12}  {}",
        "picture", "side", "op", "median", "min", "max", "out", "note"
    );
    println!("{header}");
    for row in rows {
        println!(
            "{:<24} {:<8} {:<8} {:>8} {:>8} {:>8} {:>12}  {}",
            row.picture,
            row.side,
            row.op,
            ms(row.sample.median),
            ms(row.sample.min),
            ms(row.sample.max),
            row.sample.bytes,
            row.note
        );
    }
}

fn missing(picture: &str, note: &str) -> Row {
    Row {
        picture: picture.to_string(),
        side: "-".to_string(),
        op: "skip",
        sample: Sample {
            median: Duration::ZERO,
            min: Duration::ZERO,
            max: Duration::ZERO,
            bytes: 0,
        },
        note: note.to_string(),
    }
}

fn ms(duration: Duration) -> String {
    format!("{:.1} ms", duration.as_secs_f64() * 1000.0)
}

fn reps() -> usize {
    std::env::var("PNG_BENCH_REPS")
        .ok()
        .and_then(|text| text.parse().ok())
        .filter(|reps| *reps > 0)
        .unwrap_or(5)
}

fn png_params() -> CodecParams {
    CodecParams::audio(CodecId::Png, 1, cutlass_decode::ChannelLayout::mono())
}

fn gradient() -> StillImage {
    let mut samples = Vec::with_capacity(1920 * 1080 * 4);
    for y in 0..1080u32 {
        for x in 0..1920u32 {
            let value = (x.wrapping_mul(3).wrapping_add(y) & 255) as u16;
            samples.extend([
                value,
                value.wrapping_add(20) & 255,
                value.wrapping_add(40) & 255,
                255,
            ]);
        }
    }
    StillImage::from_samples(1920, 1080, ColorType::Rgba, BitDepth::Eight, samples)
}

fn noise() -> StillImage {
    let mut rng = Rng(1);
    let samples = (0..1920 * 1080 * 4)
        .map(|_| ((rng.next() >> 33) & 255) as u16)
        .collect();
    StillImage::from_samples(1920, 1080, ColorType::Rgba, BitDepth::Eight, samples)
}

fn pack8(samples: &[u16]) -> Result<Vec<u8>, String> {
    let mut packed = Vec::with_capacity(samples.len());
    for sample in samples {
        packed.push(u8::try_from(*sample).map_err(|_| "sample does not fit in a byte")?);
    }
    Ok(packed)
}

fn local_png() -> Option<PathBuf> {
    let checkout = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)?
        .to_path_buf();
    let path = checkout.join("../local-files/04-motion-graphics.png");
    path.is_file().then_some(path)
}

fn find_ffmpeg() -> Option<PathBuf> {
    for candidate in [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "ffmpeg",
    ] {
        let Ok(output) = Command::new(candidate).arg("-version").output() else {
            continue;
        };
        if output.status.success() {
            return which(candidate);
        }
    }
    None
}

fn which(candidate: &str) -> Option<PathBuf> {
    if candidate.contains('/') {
        return Some(PathBuf::from(candidate));
    }
    let output = Command::new("which").arg(candidate).output().ok()?;
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

fn ffmpeg_version(bin: &Path) -> Result<String, String> {
    let output = Command::new(bin)
        .arg("-version")
        .output()
        .map_err(|err| err.to_string())?;
    if !output.status.success() {
        return Err("ffmpeg -version failed".to_string());
    }
    let text = String::from_utf8_lossy(&output.stdout);
    Ok(text
        .lines()
        .next()
        .unwrap_or("ffmpeg")
        .split_whitespace()
        .take(3)
        .collect::<Vec<_>>()
        .join(" "))
}
