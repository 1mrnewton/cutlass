//! Time this JPEG codec against ffmpeg on one 16×16 gray frame.
//!
//! `cargo bench -p cutlass-jpeg` runs it. Pixels are compared first. A
//! mismatch fails the process. Timing is the median of three passes after
//! one warmup. Each ffmpeg pass is a new process, so that time includes
//! startup. A missing ffmpeg prints a skip line and exits 0.

#![allow(clippy::print_stdout)]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use cutlass_decode::{
    CodecId, CodecParams, ColorPrimaries, ColorRange, ColorType, Decoder, MatrixCoefficients,
    Packet, PixelFormat, Transfer,
};
use cutlass_encode::{BitDepth, Encoder, PictureInput, StillImage};
use cutlass_jpeg::{JpegDecoder, JpegEncoder, read_jpeg_info};

const WIDTH: u32 = 16;
const HEIGHT: u32 = 16;
const LEVEL: u16 = 128;
const REPS: usize = 3;

fn main() -> Result<(), String> {
    let Some(ffmpeg) = locate_ffmpeg() else {
        println!("skip: ffmpeg is not installed");
        return Ok(());
    };
    let checked = check(&ffmpeg)?;
    let timed = time(&ffmpeg, &checked)?;
    print_report(&ffmpeg, &checked, &timed);
    Ok(())
}

struct Checked {
    jpeg: Vec<u8>,
    encoded: Vec<u8>,
    decoded: StillImage,
    back: StillImage,
    image: StillImage,
}

struct Timed {
    cutlass_decode: Span,
    ffmpeg_decode: Span,
    cutlass_encode: Span,
    ffmpeg_encoded: Span,
}

struct Span {
    median: Duration,
    min: Duration,
    max: Duration,
}

fn check(ffmpeg: &Path) -> Result<Checked, String> {
    let image = gray_frame()?;
    let raw = pack(&image.samples)?;
    let jpeg = ffmpeg_jpeg(ffmpeg, &raw, image.width, image.height, "gray")?;
    ensure_baseline(&jpeg)?;
    let decoded = decode(&jpeg)?;
    let foreign = ffmpeg_raw(ffmpeg, &jpeg, raw_pix(decoded.color)?)?;
    if !same(&decoded.samples, &foreign) {
        return Err("decode pixels differ".to_string());
    }
    let encoded = encode(&image, 75)?;
    ensure_baseline(&encoded)?;
    let back = decode(&encoded)?;
    let foreign = ffmpeg_raw(ffmpeg, &encoded, raw_pix(back.color)?)?;
    if !same(&back.samples, &foreign) || back.samples != image.samples {
        return Err("encode pixels differ".to_string());
    }
    if decoded.width != image.width
        || decoded.height != image.height
        || back.width != image.width
        || back.height != image.height
    {
        return Err("frame size differs".to_string());
    }
    Ok(Checked {
        jpeg,
        encoded,
        decoded,
        back,
        image,
    })
}

fn time(ffmpeg: &Path, checked: &Checked) -> Result<Timed, String> {
    let mut decoder = JpegDecoder::open(&params(None)).map_err(|err| err.to_string())?;
    let packet = Packet::audio(checked.jpeg.clone(), 0, 0, 0);
    let cutlass_decode = measure(|| {
        let pictures = decoder
            .decode_picture(&packet)
            .map_err(|err| err.to_string())?;
        let count = pictures
            .iter()
            .map(|picture| picture.image.samples.len())
            .sum();
        Ok(count)
    })?;

    let decode_pix = raw_pix(checked.decoded.color)?;
    let ffmpeg_decode = measure(|| {
        let raw = ffmpeg_raw(ffmpeg, &checked.jpeg, decode_pix)?;
        Ok(raw.len())
    })?;

    let input = PictureInput {
        image: checked.image.clone(),
    };
    let mut encoder = JpegEncoder::open(&params(Some(75))).map_err(|err| err.to_string())?;
    let cutlass_encode = measure(|| {
        let packets = encoder
            .push_picture(&input)
            .map_err(|err| err.to_string())?;
        let count = packets.iter().map(|packet| packet.data.len()).sum();
        Ok(count)
    })?;

    let encode_pix = raw_pix(checked.back.color)?;
    let ffmpeg_encoded = measure(|| {
        let raw = ffmpeg_raw(ffmpeg, &checked.encoded, encode_pix)?;
        Ok(raw.len())
    })?;

    Ok(Timed {
        cutlass_decode,
        ffmpeg_decode,
        cutlass_encode,
        ffmpeg_encoded,
    })
}

fn print_report(ffmpeg: &Path, checked: &Checked, timed: &Timed) {
    println!("sample: {WIDTH}x{HEIGHT} gray {LEVEL}");
    println!("ffmpeg: {}", ffmpeg.display());
    println!("ffmpeg jpeg: {} bytes baseline", checked.jpeg.len());
    println!("cutlass jpeg: {} bytes baseline", checked.encoded.len());
    println!(
        "decode pixels: match ({}x{} {:?}, {} samples)",
        checked.decoded.width,
        checked.decoded.height,
        checked.decoded.color,
        checked.decoded.samples.len()
    );
    println!(
        "encode pixels: match ({}x{} {:?}, {} samples)",
        checked.back.width,
        checked.back.height,
        checked.back.color,
        checked.back.samples.len()
    );
    println!("median of {REPS} after 1 warmup");
    print_span("cutlass decode", &timed.cutlass_decode);
    print_span("ffmpeg decode", &timed.ffmpeg_decode);
    print_span("cutlass encode", &timed.cutlass_encode);
    print_span("ffmpeg decode of cutlass jpeg", &timed.ffmpeg_encoded);
}

fn print_span(label: &str, span: &Span) {
    println!(
        "{label}: median {} us min {} us max {} us",
        format_us(span.median),
        format_us(span.min),
        format_us(span.max)
    );
}

fn format_us(duration: Duration) -> String {
    let nanos = duration.as_nanos();
    let whole = nanos / 1_000;
    let frac = nanos % 1_000;
    format!("{whole}.{frac:03}")
}

fn measure(mut run: impl FnMut() -> Result<usize, String>) -> Result<Span, String> {
    let warmed = run()?;
    std::hint::black_box(warmed);
    let mut times = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let started = Instant::now();
        let value = run()?;
        let elapsed = started.elapsed();
        std::hint::black_box(value);
        times.push(elapsed);
    }
    let min = times
        .iter()
        .copied()
        .min()
        .ok_or_else(|| "no samples".to_string())?;
    let max = times
        .iter()
        .copied()
        .max()
        .ok_or_else(|| "no samples".to_string())?;
    times.sort();
    let median = times
        .get(times.len() / 2)
        .copied()
        .ok_or_else(|| "no samples".to_string())?;
    Ok(Span { median, min, max })
}

fn gray_frame() -> Result<StillImage, String> {
    let count = usize::try_from(WIDTH * HEIGHT).map_err(|_| "frame size".to_string())?;
    StillImage::from_samples(
        WIDTH,
        HEIGHT,
        ColorType::Gray,
        BitDepth::Eight,
        vec![LEVEL; count],
    )
    .checked()
    .map_err(|err| err.to_string())
}

fn params(quality: Option<u8>) -> CodecParams {
    CodecParams {
        codec: CodecId::Jpeg,
        extra: quality.map(|value| vec![value]).unwrap_or_default(),
        sample_rate: 0,
        layout: None,
        sample_format: None,
        bitrate: 0,
        width: 0,
        height: 0,
        pixel: PixelFormat::None,
        primaries: ColorPrimaries::Unspecified,
        transfer: Transfer::Unspecified,
        matrix: MatrixCoefficients::Unspecified,
        range: ColorRange::Unspecified,
        nal_length_size: 0,
    }
}

fn decode(bytes: &[u8]) -> Result<StillImage, String> {
    let mut decoder = JpegDecoder::open(&params(None)).map_err(|err| err.to_string())?;
    let packet = Packet::audio(bytes.to_vec(), 0, 0, 0);
    let pictures = decoder
        .decode_picture(&packet)
        .map_err(|err| err.to_string())?;
    let Some(picture) = pictures.into_iter().next() else {
        return Err("decoder returned no picture".to_string());
    };
    Ok(picture.image)
}

fn encode(image: &StillImage, quality: u8) -> Result<Vec<u8>, String> {
    let mut encoder = JpegEncoder::open(&params(Some(quality))).map_err(|err| err.to_string())?;
    let packets = encoder
        .push_picture(&PictureInput {
            image: image.clone(),
        })
        .map_err(|err| err.to_string())?;
    let Some(packet) = packets.into_iter().next() else {
        return Err("encoder returned no packet".to_string());
    };
    Ok(packet.data)
}

fn ensure_baseline(bytes: &[u8]) -> Result<(), String> {
    let info = read_jpeg_info(bytes).map_err(|err| err.to_string())?;
    if info.progressive || !bytes.windows(2).any(|marker| marker == [0xff, 0xc0]) {
        return Err("not a baseline jpeg".to_string());
    }
    Ok(())
}

fn raw_pix(color: ColorType) -> Result<&'static str, String> {
    match color {
        ColorType::Gray => Ok("gray"),
        ColorType::Rgb => Ok("rgb24"),
        _ => Err("unexpected jpeg color".to_string()),
    }
}

fn pack(samples: &[u16]) -> Result<Vec<u8>, String> {
    samples
        .iter()
        .map(|sample| u8::try_from(*sample).map_err(|_| "sample".to_string()))
        .collect()
}

fn same(samples: &[u16], raw: &[u8]) -> bool {
    samples.len() == raw.len()
        && samples
            .iter()
            .zip(raw.iter())
            .all(|(sample, byte)| *sample == u16::from(*byte))
}

fn locate_ffmpeg() -> Option<PathBuf> {
    let homebrew = PathBuf::from("/opt/homebrew/bin/ffmpeg");
    if homebrew.is_file() {
        return Some(homebrew);
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).find_map(|dir| {
        let candidate = dir.join("ffmpeg");
        candidate.is_file().then_some(candidate)
    })
}

fn ffmpeg_jpeg(
    bin: &Path,
    raw: &[u8],
    width: u32,
    height: u32,
    pix: &str,
) -> Result<Vec<u8>, String> {
    let size = format!("{width}x{height}");
    ffmpeg_bytes(
        bin,
        &[
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix,
            "-s",
            &size,
            "-i",
            "pipe:0",
            "-frames:v",
            "1",
            "-pix_fmt",
            "yuvj420p",
            "-c:v",
            "mjpeg",
            "-q:v",
            "2",
            "-f",
            "image2pipe",
            "pipe:1",
        ],
        raw,
    )
}

fn ffmpeg_raw(bin: &Path, jpeg: &[u8], pix: &str) -> Result<Vec<u8>, String> {
    ffmpeg_bytes(
        bin,
        &[
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "image2pipe",
            "-i",
            "pipe:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix,
            "pipe:1",
        ],
        jpeg,
    )
}

fn ffmpeg_bytes(bin: &Path, args: &[&str], input: &[u8]) -> Result<Vec<u8>, String> {
    let mut child = Command::new(bin)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| err.to_string())?;
    let Some(mut stdin) = child.stdin.take() else {
        return Err("ffmpeg stdin".to_string());
    };
    stdin.write_all(input).map_err(|err| err.to_string())?;
    drop(stdin);
    let output = child.wait_with_output().map_err(|err| err.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "ffmpeg exited {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(output.stdout)
}
