//! Throughput against ffmpeg on the same machine.
//!
//! `cargo bench -p cutlass-h264`
//!
//! Decode runs two High annex-B streams: an easy one (clean `testsrc2`,
//! mostly skipped macroblocks) and a hard one (the same pictures with noise
//! at 5 Mb/s, so most macroblocks carry a residual). Both sides are timed
//! over the same work: split the file's bytes into access units, decode
//! every picture, and drop it. This decoder is timed around that loop.
//! ffmpeg is timed by its own `-benchmark` clock, which starts after the
//! process has loaded and probed the input, so neither side pays for
//! process startup. ffmpeg's whole-process time is printed beside it.
//! This decoder uses one thread; ffmpeg is shown on one thread and on all.
//!
//! Encode is the same `testsrc2` pictures. libx264 is ultrafast; this
//! encoder has no preset, and it searches far more than ultrafast does. The
//! encode numbers are wall time, not a quality match.
//!
//! Before any speed is printed, decode samples must match ffmpeg with zero
//! difference, and this encoder's bitstream must reconstruct to the same
//! samples in this decoder and in ffmpeg. A mismatch fails the process.

#![allow(clippy::print_stdout)]

use std::path::Path;
use std::process::Command;
use std::time::Instant;

use cutlass_decode::{CodecId, CodecParams, Packet, VideoDecoder, VideoOutput, Yuv420p};
use cutlass_encode::{VideoEncoder, VideoInput};
use cutlass_h264::{H264Decoder, H264Encoder};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const DECODE_FRAMES: usize = 300;
const RUNS: usize = 5;
const ENCODE_FRAMES: usize = 30;
const BITRATE: u32 = 2_000_000;
const Y_LEN: usize = 1280 * 720;
const C_LEN: usize = Y_LEN / 4;
const FRAME_LEN: usize = Y_LEN + C_LEN * 2;

fn main() -> Result<(), String> {
    let Ok(ffmpeg) = ffmpeg_bin() else {
        println!("skip: ffmpeg is not installed");
        return Ok(());
    };
    let dir = std::env::temp_dir().join("cutlass-h264-bench");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(|err| err.to_string())?;

    println!("1280x720, High profile, CABAC, {DECODE_FRAMES} frames, median of {RUNS} runs");
    println!("both sides: split the stream, decode every picture, drop it");
    println!("ffmpeg's time is its own -benchmark clock, without process startup");
    for stream in STREAMS {
        let coded = dir.join(stream.file);
        generate_stream(&ffmpeg, &coded, stream)?;
        let bytes = std::fs::read(&coded).map_err(|err| err.to_string())?;
        let packets = packets(&access_units(&bytes)?)?;
        let decoded = reconstruct(&packets, DECODE_FRAMES)?;
        drop(packets);
        let reference = ffmpeg_yuv(&ffmpeg, &coded, &dir.join("decoded.yuv"), DECODE_FRAMES)?;
        same_samples(stream.name, &decoded, &reference)?;
        drop(decoded);
        drop(reference);
        let _ = time_decode(&bytes)?;
        let mut ours = Vec::new();
        let mut one = Vec::new();
        let mut one_process = Vec::new();
        let mut all = Vec::new();
        // The three are run in turn so a busy moment on the machine hits
        // them alike.
        for _ in 0..RUNS {
            ours.push(time_decode(&bytes)?);
            let (clock, process) = time_ffmpeg_decode(&ffmpeg, &coded, Some(1))?;
            one.push(clock);
            one_process.push(process);
            all.push(time_ffmpeg_decode(&ffmpeg, &coded, None)?.0);
        }
        let ours = median(ours);
        let one = median(one);
        let megabits = bytes.len() as f64 * 8.0 * 30.0 / DECODE_FRAMES as f64 / 1e6;
        println!();
        println!("decode {}, {megabits:.1} Mb/s", stream.name);
        println!("pixels match ffmpeg, zero sample difference");
        print_row("cutlass, 1 thread", DECODE_FRAMES, ours);
        print_row("ffmpeg, 1 thread", DECODE_FRAMES, one);
        print_row("  whole process", DECODE_FRAMES, median(one_process));
        print_row("ffmpeg, all threads", DECODE_FRAMES, median(all));
        print_ratio(one, ours);
    }

    let raw_path = dir.join("source.yuv");
    generate_yuv(&ffmpeg, &raw_path)?;
    let raw = std::fs::read(&raw_path).map_err(|err| err.to_string())?;
    let inputs = inputs(&raw)?;
    drop(raw);
    let encoded = encode_packets(&inputs)?;
    let encoded_path = dir.join("encoded.h264");
    write_annex(&encoded, &encoded_path)?;
    let encoded_ours = reconstruct(&encoded, ENCODE_FRAMES)?;
    let encoded_ffmpeg = ffmpeg_yuv(
        &ffmpeg,
        &encoded_path,
        &dir.join("encoded.yuv"),
        ENCODE_FRAMES,
    )?;
    same_samples("encode", &encoded_ours, &encoded_ffmpeg)?;
    drop(encoded);
    drop(encoded_ours);
    drop(encoded_ffmpeg);
    let _ = time_encode(&inputs)?;
    let mut encode_ours = Vec::new();
    let mut encode_ffmpeg = Vec::new();
    for _ in 0..RUNS {
        encode_ours.push(time_encode(&inputs)?);
        encode_ffmpeg.push(time_ffmpeg_encode(&ffmpeg, &raw_path)?);
    }
    let encode_ours = median(encode_ours);
    let encode_ffmpeg = median(encode_ffmpeg);

    println!();
    println!("encode testsrc2, {ENCODE_FRAMES} frames, one thread, {BITRATE} bits/s");
    println!("reconstruction matches ffmpeg, zero sample difference");
    println!("libx264 is preset ultrafast and searches far less than this encoder");
    print_row("cutlass, 1 thread", ENCODE_FRAMES, encode_ours);
    print_row("ffmpeg, 1 thread", ENCODE_FRAMES, encode_ffmpeg);
    print_ratio(encode_ffmpeg, encode_ours);
    Ok(())
}

/// One decode test stream and how libx264 makes it.
struct Stream {
    name: &'static str,
    file: &'static str,
    /// A filter over `testsrc2`, if any.
    filter: Option<&'static str>,
    /// libx264 options after the profile.
    coding: &'static [&'static str],
}

const STREAMS: &[Stream] = &[
    Stream {
        name: "easy: clean testsrc2, preset veryfast, 2 B-frames",
        file: "easy.h264",
        filter: None,
        coding: &["-preset", "veryfast", "-x264-params", "cabac=1:bframes=2"],
    },
    Stream {
        name: "hard: noisy testsrc2, preset medium",
        file: "hard.h264",
        filter: Some("noise=alls=12:allf=t"),
        coding: &["-preset", "medium", "-b:v", "5M"],
    },
];

fn same_samples(what: &str, got: &[u8], want: &[u8]) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!(
            "{what} pixels disagree: {} bytes vs {}",
            got.len(),
            want.len()
        ));
    }
    let mut count = 0usize;
    let mut first = None;
    let mut max = 0u8;
    for (index, (left, right)) in got.iter().zip(want.iter()).enumerate() {
        if left == right {
            continue;
        }
        count += 1;
        let delta = left.abs_diff(*right);
        if delta > max {
            max = delta;
        }
        if first.is_none() {
            first = Some((index, *left, *right));
        }
    }
    let Some((index, got_byte, want_byte)) = first else {
        return Ok(());
    };
    Err(format!(
        "{what} pixels disagree: {count} samples, first byte {index} (this decoder {got_byte}, ffmpeg {want_byte}), max delta {max}"
    ))
}

fn reconstruct(packets: &[Packet], frames: usize) -> Result<Vec<u8>, String> {
    let mut params = CodecParams::video(CodecId::H264, WIDTH, HEIGHT);
    params.nal_length_size = 0;
    let mut decoder = H264Decoder::open(&params).map_err(|err| err.to_string())?;
    let mut packed = Vec::new();
    let mut pictures = 0usize;
    for packet in packets {
        let produced = decoder.push(packet).map_err(|err| err.to_string())?;
        pictures += produced.len();
        append_frames(&produced, &mut packed)?;
    }
    let tail = decoder.drain().map_err(|err| err.to_string())?;
    pictures += tail.len();
    append_frames(&tail, &mut packed)?;
    if pictures != frames {
        return Err(format!(
            "decoder returned {pictures} pictures, wanted {frames}"
        ));
    }
    Ok(packed)
}

fn append_frames(frames: &[VideoOutput], packed: &mut Vec<u8>) -> Result<(), String> {
    for frame in frames {
        if frame.picture.width() != WIDTH || frame.picture.height() != HEIGHT {
            return Err(format!(
                "picture is {}x{}",
                frame.picture.width(),
                frame.picture.height()
            ));
        }
        packed.extend_from_slice(frame.picture.y());
        packed.extend_from_slice(frame.picture.u());
        packed.extend_from_slice(frame.picture.v());
    }
    Ok(())
}

fn encode_packets(inputs: &[VideoInput]) -> Result<Vec<Packet>, String> {
    let mut params = CodecParams::video(CodecId::H264, WIDTH, HEIGHT);
    params.nal_length_size = 0;
    params.bitrate = BITRATE;
    let mut encoder = H264Encoder::open(&params).map_err(|err| err.to_string())?;
    let mut packets = Vec::new();
    for input in inputs {
        packets.extend(encoder.push(input).map_err(|err| err.to_string())?);
    }
    packets.extend(encoder.drain().map_err(|err| err.to_string())?);
    if packets.len() != ENCODE_FRAMES {
        return Err(format!("encoder returned {} packets", packets.len()));
    }
    Ok(packets)
}

fn write_annex(packets: &[Packet], path: &Path) -> Result<(), String> {
    let mut bytes = Vec::new();
    for packet in packets {
        bytes.extend_from_slice(&packet.data);
    }
    if bytes.is_empty() {
        return Err("encoder returned no bytes".to_string());
    }
    std::fs::write(path, bytes).map_err(|err| err.to_string())
}

fn ffmpeg_yuv(ffmpeg: &str, input: &Path, output: &Path, frames: usize) -> Result<Vec<u8>, String> {
    let count = frames.to_string();
    let result = Command::new(ffmpeg)
        .args([
            "-nostdin",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-i",
        ])
        .arg(input)
        .args(["-frames:v", &count, "-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(output)
        .output()
        .map_err(|err| err.to_string())?;
    if !result.status.success() {
        return Err(tail(&result.stderr));
    }
    let bytes = std::fs::read(output).map_err(|err| err.to_string())?;
    let _ = std::fs::remove_file(output);
    Ok(bytes)
}

fn print_row(name: &str, frames: usize, seconds: f64) {
    let fps = fps(frames, seconds);
    println!("{name:<20} {seconds:>8.3} s   {fps:>8.1} fps");
}

fn print_ratio(ffmpeg_seconds: f64, ours_seconds: f64) {
    if ffmpeg_seconds > 0.0 {
        println!("ffmpeg was {:.1}x as fast", ours_seconds / ffmpeg_seconds);
    }
}

fn fps(frames: usize, seconds: f64) -> f64 {
    let frames = u32::try_from(frames).unwrap_or(u32::MAX);
    f64::from(frames) / seconds
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    samples.get(samples.len() / 2).copied().unwrap_or(0.0)
}

/// Seconds to split `bytes` into access units and decode them all.
fn time_decode(bytes: &[u8]) -> Result<f64, String> {
    let mut params = CodecParams::video(CodecId::H264, WIDTH, HEIGHT);
    params.nal_length_size = 0;
    let mut decoder = H264Decoder::open(&params).map_err(|err| err.to_string())?;
    let started = Instant::now();
    let packets = packets(&access_units(bytes)?)?;
    let mut pictures = 0usize;
    let mut luma = 0usize;
    for packet in &packets {
        let frames = decoder.push(packet).map_err(|err| err.to_string())?;
        pictures += frames.len();
        luma += frames
            .iter()
            .map(|frame| frame.picture.y().len())
            .sum::<usize>();
    }
    let tail = decoder.drain().map_err(|err| err.to_string())?;
    pictures += tail.len();
    luma += tail
        .iter()
        .map(|frame| frame.picture.y().len())
        .sum::<usize>();
    let seconds = started.elapsed().as_secs_f64();
    if pictures != DECODE_FRAMES || luma == 0 {
        return Err(format!("decoder returned {pictures} pictures"));
    }
    Ok(seconds)
}

fn time_encode(inputs: &[VideoInput]) -> Result<f64, String> {
    let mut params = CodecParams::video(CodecId::H264, WIDTH, HEIGHT);
    params.nal_length_size = 0;
    params.bitrate = BITRATE;
    let mut encoder = H264Encoder::open(&params).map_err(|err| err.to_string())?;
    let started = Instant::now();
    let mut packets = 0usize;
    let mut bytes = 0usize;
    for input in inputs {
        let produced = encoder.push(input).map_err(|err| err.to_string())?;
        packets += produced.len();
        bytes += produced
            .iter()
            .map(|packet| packet.data.len())
            .sum::<usize>();
    }
    let tail = encoder.drain().map_err(|err| err.to_string())?;
    packets += tail.len();
    bytes += tail.iter().map(|packet| packet.data.len()).sum::<usize>();
    let seconds = started.elapsed().as_secs_f64();
    if packets != ENCODE_FRAMES || bytes == 0 {
        return Err(format!("encoder returned {packets} packets"));
    }
    Ok(seconds)
}

/// ffmpeg's own clock and the whole process's wall time, in seconds, to
/// decode `path` on `threads` threads (all of them when `None`).
fn time_ffmpeg_decode(
    ffmpeg: &str,
    path: &Path,
    threads: Option<u32>,
) -> Result<(f64, f64), String> {
    let mut command = Command::new(ffmpeg);
    command.args(BENCHMARK);
    if let Some(threads) = threads {
        command.args(["-threads", &threads.to_string()]);
    }
    command.arg("-i").arg(path).args(["-f", "null", "-"]);
    let started = Instant::now();
    let output = command.output().map_err(|err| err.to_string())?;
    let process = started.elapsed().as_secs_f64();
    if !output.status.success() {
        return Err(tail(&output.stderr));
    }
    let pictures = format!("frame={DECODE_FRAMES:>5} ");
    if !String::from_utf8_lossy(&output.stderr).contains(&pictures) {
        return Err(format!("ffmpeg did not decode {DECODE_FRAMES} frames"));
    }
    Ok((benchmark_seconds(&output.stderr)?, process))
}

/// Options that make ffmpeg print its own timing and nothing per frame.
const BENCHMARK: [&str; 6] = [
    "-nostdin",
    "-hide_banner",
    "-loglevel",
    "info",
    "-nostats",
    "-benchmark",
];

/// The real time ffmpeg's `-benchmark` reports: from the first packet read
/// to the last frame written, after the process has started and opened its
/// input.
fn benchmark_seconds(stderr: &[u8]) -> Result<f64, String> {
    let text = String::from_utf8_lossy(stderr);
    text.split("rtime=")
        .nth(1)
        .and_then(|rest| rest.split('s').next())
        .and_then(|seconds| seconds.parse().ok())
        .ok_or_else(|| format!("ffmpeg printed no benchmark time: {}", tail(stderr)))
}

fn time_ffmpeg_encode(ffmpeg: &str, yuv: &Path) -> Result<f64, String> {
    let output = Command::new(ffmpeg)
        .args(BENCHMARK)
        .args([
            "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", "1280x720", "-r", "30", "-i",
        ])
        .arg(yuv)
        .args([
            "-frames:v",
            "30",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-profile:v",
            "high",
            "-threads",
            "1",
            "-b:v",
            "2000k",
            "-f",
            "null",
            "-",
        ])
        .output()
        .map_err(|err| err.to_string())?;
    if !output.status.success() {
        return Err(tail(&output.stderr));
    }
    benchmark_seconds(&output.stderr)
}

fn generate_stream(ffmpeg: &str, path: &Path, stream: &Stream) -> Result<(), String> {
    let frames = DECODE_FRAMES.to_string();
    let mut command = Command::new(ffmpeg);
    command.args([
        "-nostdin",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=1280x720:rate=30",
    ]);
    if let Some(filter) = stream.filter {
        command.args(["-vf", filter]);
    }
    let output = command
        .args([
            "-frames:v",
            &frames,
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
        ])
        .args(stream.coding)
        .args(["-pix_fmt", "yuv420p", "-threads", "1", "-f", "h264"])
        .arg(path)
        .output()
        .map_err(|err| err.to_string())?;
    if output.status.success() {
        Ok(())
    } else {
        Err(tail(&output.stderr))
    }
}

fn generate_yuv(ffmpeg: &str, path: &Path) -> Result<(), String> {
    let frames = ENCODE_FRAMES.to_string();
    let output = Command::new(ffmpeg)
        .args([
            "-nostdin",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1280x720:rate=30",
            "-frames:v",
        ])
        .arg(&frames)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(path)
        .output()
        .map_err(|err| err.to_string())?;
    if output.status.success() {
        Ok(())
    } else {
        Err(tail(&output.stderr))
    }
}

fn packets(units: &[Vec<u8>]) -> Result<Vec<Packet>, String> {
    if units.len() != DECODE_FRAMES {
        return Err(format!("stream has {} access units", units.len()));
    }
    let mut packets = Vec::with_capacity(units.len());
    for (index, unit) in units.iter().enumerate() {
        let pts = i64::try_from(index).unwrap_or(0);
        packets.push(Packet::video(unit.clone(), pts, 1, 30, index == 0));
    }
    Ok(packets)
}

fn inputs(raw: &[u8]) -> Result<Vec<VideoInput>, String> {
    if raw.len() < FRAME_LEN * ENCODE_FRAMES {
        return Err(format!("yuv is {} bytes", raw.len()));
    }
    let mut inputs = Vec::with_capacity(ENCODE_FRAMES);
    for index in 0..ENCODE_FRAMES {
        let start = index * FRAME_LEN;
        let plane = &raw[start..start + FRAME_LEN];
        let picture = Yuv420p::try_new(
            WIDTH,
            HEIGHT,
            plane[..Y_LEN].to_vec(),
            plane[Y_LEN..Y_LEN + C_LEN].to_vec(),
            plane[Y_LEN + C_LEN..].to_vec(),
        )
        .map_err(|err| err.to_string())?;
        inputs.push(VideoInput {
            picture,
            pts: i64::try_from(index).unwrap_or(0),
            duration: 1,
            timescale: 30,
            force_keyframe: index == 0,
        });
    }
    Ok(inputs)
}

fn access_units(data: &[u8]) -> Result<Vec<Vec<u8>>, String> {
    let mut starts = Vec::new();
    let mut index = 0usize;
    while index + 3 < data.len() {
        if data[index] == 0 && data[index + 1] == 0 && data[index + 2] == 1 {
            let header = if index > 0 && data[index - 1] == 0 {
                index - 1
            } else {
                index
            };
            starts.push((header, index + 3));
            index += 3;
        } else {
            index += 1;
        }
    }
    if starts.is_empty() {
        return Err("annex b had no start code".to_string());
    }
    let mut units = Vec::new();
    let mut current = Vec::new();
    let mut saw_vcl = false;
    for (n, &(header, body)) in starts.iter().enumerate() {
        let end = starts
            .get(n + 1)
            .map(|(next, _)| *next)
            .unwrap_or(data.len());
        if body >= data.len() || header >= end {
            continue;
        }
        let kind = data[body] & 31;
        let vcl = kind == 1 || kind == 5;
        // first_mb_in_slice 0 is one 1 bit, so the next byte's high bit is set.
        let starts_picture = vcl && data.get(body + 1).is_some_and(|byte| byte & 0x80 != 0);
        if (kind == 9 || (starts_picture && saw_vcl)) && !current.is_empty() {
            units.push(std::mem::take(&mut current));
            saw_vcl = false;
        }
        current.extend_from_slice(&data[header..end]);
        saw_vcl |= vcl;
    }
    if !current.is_empty() {
        units.push(current);
    }
    Ok(units)
}

fn ffmpeg_bin() -> Result<String, String> {
    for path in [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "ffmpeg",
    ] {
        if let Ok(output) = Command::new(path).arg("-version").output()
            && output.status.success()
        {
            return Ok(path.to_string());
        }
    }
    Err("ffmpeg is not installed".to_string())
}

fn tail(stderr: &[u8]) -> String {
    let text = String::from_utf8_lossy(stderr);
    let start = text.len().saturating_sub(400);
    text[start..].trim().to_string()
}
