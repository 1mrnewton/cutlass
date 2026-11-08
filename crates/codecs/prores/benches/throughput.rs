//! Frames per second for every ProRes profile at 1080p and 2160p, encode
//! and decode, on all cores and on one, beside ffmpeg's ProRes decoder.
//!
//! The pictures are real: ffmpeg decodes the first frames of the 1080p and
//! 2160p clips in `../local-files` to 10-bit 4:2:2 and 12-bit 4:4:4. Each
//! measurement codes `PRORES_BENCH_FRAMES` pictures (30 by default),
//! cycling through those frames, inside this process. The ffmpeg line is
//! one `ffmpeg -benchmark` process decoding the same 2160p HQ stream, so it
//! includes startup. The budget line asks whether 2160p HQ decodes at 29.97
//! frames per second or faster.
//!
//! Run: `cargo bench -p cutlass-prores --bench throughput`. Without ffmpeg
//! or the clips the bench prints why and exits 0.

#![allow(clippy::print_stdout)]

use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use cutlass_decode::{CodecId, CodecParams, PixelFormat, Plane, VideoFrame};
use cutlass_prores::{Profile, ProresDecoder, ProresEncoder};

const SOURCE_FRAMES: usize = 6;
const CLIPS: [(&str, &str, u32, u32); 2] = [
    ("1080p", "16929689_1920_1080_30fps.mp4", 1920, 1080),
    ("2160p", "12129324_3840_2160_30fps.mp4", 3840, 2160),
];

fn main() -> Result<(), String> {
    let Some(ffmpeg) = tool("ffmpeg") else {
        say("bench not run: ffmpeg is not installed");
        return Ok(());
    };
    let count = std::env::var("PRORES_BENCH_FRAMES")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&count: &usize| count > 0)
        .unwrap_or(30);
    let cores = std::thread::available_parallelism().map_or(1, usize::from);
    say(&machine());
    say(&format!(
        "{count} pictures per measurement from {SOURCE_FRAMES} real frames; all cores is {cores} workers"
    ));
    say("size   profile  encode all  encode one  decode all  decode one   Mb/s");
    let mut uhd_hq = None;
    for (label, clip, width, height) in CLIPS {
        let Some(path) = local(clip) else {
            say(&format!(
                "bench not run for {label}: local-files/{clip} is missing"
            ));
            continue;
        };
        let narrow = source(&ffmpeg, &path, PixelFormat::Yuv422p10, width, height)?;
        let wide = source(&ffmpeg, &path, PixelFormat::Yuv444p12, width, height)?;
        for profile in Profile::ALL {
            let frames = if profile.is_444() { &wide } else { &narrow };
            let row = measure(profile, frames, count, cores)?;
            let mbps = row.bytes as f64 * 8.0 / count as f64 * 30000.0 / 1001.0 / 1e6;
            say(&format!(
                "{label}  {:<7}  {:>10.1}  {:>10.1}  {:>10.1}  {:>10.1}  {mbps:>5.0}",
                profile.label(),
                row.encode_all,
                row.encode_one,
                row.decode_all,
                row.decode_one
            ));
            if label == "2160p" && profile == Profile::Hq {
                uhd_hq = Some((row, width, height));
            }
        }
    }
    if let Some((row, width, height)) = uhd_hq {
        let budget = 30000.0 / 1001.0;
        say(&format!(
            "budget: 2160p HQ decodes at {:.1} fps on all cores, {:.1} times real time at 29.97 fps ({})",
            row.decode_all,
            row.decode_all / budget,
            if row.decode_all >= budget {
                "met"
            } else {
                "missed"
            }
        ));
        say(&format!(
            "budget: one core decodes at {:.1} fps, {:.2} times real time",
            row.decode_one,
            row.decode_one / budget
        ));
        against_ffmpeg(&ffmpeg, &row.stream, width, height, count)?;
    }
    Ok(())
}

struct Row {
    encode_all: f64,
    encode_one: f64,
    decode_all: f64,
    decode_one: f64,
    bytes: usize,
    stream: Vec<Vec<u8>>,
}

fn measure(
    profile: Profile,
    frames: &[VideoFrame],
    count: usize,
    cores: usize,
) -> Result<Row, String> {
    let mut params = CodecParams::with_format(CodecId::Prores, 0, 0, PixelFormat::None);
    params.extra = profile.fourcc().to_vec();
    let open_encoder = |threads| {
        ProresEncoder::open(&params)
            .map(|encoder| encoder.with_threads(threads))
            .map_err(|error| format!("encoder: {error}"))
    };
    let open_decoder = |threads| {
        ProresDecoder::open(&params)
            .map(|decoder| decoder.with_threads(threads))
            .map_err(|error| format!("decoder: {error}"))
    };

    let mut stream = Vec::with_capacity(count);
    let mut encoder = open_encoder(cores)?;
    let start = Instant::now();
    for index in 0..count {
        let frame = &frames[index % frames.len()];
        stream.push(
            encoder
                .encode(frame)
                .map_err(|error| format!("encode: {error}"))?,
        );
    }
    let encode_all = count as f64 / start.elapsed().as_secs_f64();

    let mut encoder = open_encoder(1)?;
    let start = Instant::now();
    for index in 0..count {
        let bytes = encoder
            .encode(&frames[index % frames.len()])
            .map_err(|error| format!("encode: {error}"))?;
        if bytes != stream[index] {
            return Err(format!(
                "{}: one worker coded different bytes",
                profile.label()
            ));
        }
    }
    let encode_one = count as f64 / start.elapsed().as_secs_f64();

    let decode_rate = |threads| -> Result<f64, String> {
        let mut decoder = open_decoder(threads)?;
        let start = Instant::now();
        for bytes in &stream {
            decoder
                .decode(bytes)
                .map_err(|error| format!("decode: {error}"))?;
        }
        Ok(count as f64 / start.elapsed().as_secs_f64())
    };
    let decode_all = decode_rate(cores)?;
    let decode_one = decode_rate(1)?;
    let bytes = stream.iter().map(Vec::len).sum();
    Ok(Row {
        encode_all,
        encode_one,
        decode_all,
        decode_one,
        bytes,
        stream,
    })
}

/// The first frames of `clip`, converted by ffmpeg to `format`.
fn source(
    ffmpeg: &Path,
    clip: &Path,
    format: PixelFormat,
    width: u32,
    height: u32,
) -> Result<Vec<VideoFrame>, String> {
    let name = match format {
        PixelFormat::Yuv422p10 => "yuv422p10le",
        _ => "yuv444p12le",
    };
    let frames = SOURCE_FRAMES.to_string();
    let out = Command::new(ffmpeg)
        .args(["-v", "error", "-i"])
        .arg(clip)
        .args([
            "-frames:v",
            &frames,
            "-an",
            "-pix_fmt",
            name,
            "-f",
            "rawvideo",
            "-",
        ])
        .output()
        .map_err(|error| format!("ffmpeg: {error}"))?;
    if !out.status.success() {
        return Err(format!("ffmpeg: {}", String::from_utf8_lossy(&out.stderr)));
    }
    let (chroma_w, chroma_h) = format
        .chroma_wh(width, height)
        .ok_or_else(|| "chroma size".to_string())?;
    let luma = (width * height * 2) as usize;
    let chroma = (chroma_w * chroma_h * 2) as usize;
    let frame_bytes = luma + 2 * chroma;
    out.stdout
        .chunks_exact(frame_bytes)
        .map(|frame| {
            let plane = |at: usize, w: u32, h: u32, len: usize| {
                Plane::bytes(w, h, 2, frame[at..at + len].to_vec())
                    .map_err(|error| format!("plane: {error}"))
            };
            VideoFrame::planar(
                format,
                width,
                height,
                plane(0, width, height, luma)?,
                plane(luma, chroma_w, chroma_h, chroma)?,
                plane(luma + chroma, chroma_w, chroma_h, chroma)?,
            )
            .map_err(|error| format!("frame: {error}"))
        })
        .collect()
}

/// ffmpeg's own ProRes decoder, and VideoToolbox through ffmpeg on a Mac,
/// on the same 2160p HQ pictures in a movie.
fn against_ffmpeg(
    ffmpeg: &Path,
    stream: &[Vec<u8>],
    width: u32,
    height: u32,
    count: usize,
) -> Result<(), String> {
    let dir = std::env::temp_dir().join(format!("cutlass-prores-bench-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|error| format!("temp directory: {error}"))?;
    let movie = dir.join("hq.mov");
    let outcome = mux(stream, width, height)
        .and_then(|bytes| std::fs::write(&movie, bytes).map_err(|error| format!("write: {error}")))
        .and_then(|()| {
            say(&format!("ffmpeg: {}", version(ffmpeg)));
            let software = time_ffmpeg(ffmpeg, &movie, &[])?;
            say(&format!(
                "ffmpeg prores decoder, all threads, 2160p HQ: {:.1} fps over {count} pictures (one process, startup included)",
                count as f64 / software
            ));
            if cfg!(target_vendor = "apple") {
                let hardware = time_ffmpeg(ffmpeg, &movie, &["-hwaccel", "videotoolbox"])?;
                say(&format!(
                    "ffmpeg with VideoToolbox, 2160p HQ: {:.1} fps (one process, startup included)",
                    count as f64 / hardware
                ));
            }
            Ok(())
        });
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

fn time_ffmpeg(ffmpeg: &Path, movie: &Path, before: &[&str]) -> Result<f64, String> {
    let start = Instant::now();
    let out = Command::new(ffmpeg)
        .args(["-v", "error", "-threads", "0"])
        .args(before)
        .arg("-i")
        .arg(movie)
        .args(["-f", "null", "-"])
        .output()
        .map_err(|error| format!("ffmpeg: {error}"))?;
    if !out.status.success() {
        return Err(format!(
            "ffmpeg decode: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(start.elapsed().as_secs_f64())
}

fn mux(stream: &[Vec<u8>], width: u32, height: u32) -> Result<Vec<u8>, String> {
    use cutlass_codec::{FrameSize, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
    use cutlass_container::{Mux, NewStream, TrackDuration};
    use cutlass_isobmff::{FileBrand, Mp4Mux};
    let fail = |what: &str| format!("mux: {what}");
    let mut mux = Mp4Mux::with_brand(Cursor::new(Vec::new()), FileBrand::Mov);
    let scale = Timescale::new(30000);
    let size = FrameSize::try_new(width, height).map_err(|_| fail("size"))?;
    let track = mux
        .add_stream(NewStream {
            params: StreamParams::video(
                StreamCodec::Known(CodecId::Prores),
                size,
                Profile::Hq.fourcc().to_vec(),
            ),
            timescale: scale,
            language: None,
        })
        .map_err(|_| fail("stream"))?;
    let mut end = 0;
    for (index, bytes) in stream.iter().enumerate() {
        let pts = index as i64 * 1001;
        let time =
            Timestamp::try_new(Some(pts), Some(pts), Some(1001)).map_err(|_| fail("time"))?;
        let packet = Packet::try_new(track.get(), bytes.clone(), time, scale, true)
            .map_err(|_| fail("packet"))?;
        mux.write_packet(&packet).map_err(|_| fail("write"))?;
        end = pts + 1001;
    }
    mux.finish(&[TrackDuration {
        track,
        timescale: scale,
        ticks: end,
    }])
    .map_err(|_| fail("finish"))?;
    Ok(mux.into_writer().map_err(|_| fail("writer"))?.into_inner())
}

fn local(name: &str) -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = root.ancestors().nth(3)?.join("../local-files").join(name);
    path.is_file().then_some(path)
}

fn tool(name: &str) -> Option<PathBuf> {
    let paths = std::env::var_os("PATH")?;
    std::env::split_paths(&paths)
        .map(|dir| dir.join(name))
        .find(|path| path.is_file())
}

fn version(ffmpeg: &Path) -> String {
    Command::new(ffmpeg)
        .arg("-version")
        .output()
        .ok()
        .and_then(|out| {
            String::from_utf8_lossy(&out.stdout)
                .lines()
                .next()
                .map(str::to_owned)
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn machine() -> String {
    let read = |key: &str| {
        Command::new("sysctl")
            .args(["-n", key])
            .output()
            .ok()
            .filter(|out| out.status.success())
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
    };
    format!(
        "machine: {}, {} cores; bench profile",
        read("machdep.cpu.brand_string").unwrap_or_else(|| "unknown cpu".to_string()),
        read("hw.ncpu").unwrap_or_else(|| "?".to_string())
    )
}

fn say(line: &str) {
    let _ = writeln!(std::io::stdout(), "{line}");
}
