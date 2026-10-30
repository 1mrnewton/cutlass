//! Time one s16le stereo buffer against ffmpeg.
//!
//! `cargo bench -p cutlass-pcm`
//!
//! ffmpeg writes the payload. Samples are compared before anything is timed.
//! The reported figure is the median of 3 runs after one warmup. An ffmpeg
//! row is a new process and includes process startup. When ffmpeg is missing
//! this prints `skip: ffmpeg is not installed` and exits 0.

#![allow(clippy::print_stdout)]

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use cutlass_decode::{
    ChannelLayout, CodecId, CodecParams, Decoder, Output, Packet, Pcm, SampleFormat,
};
use cutlass_encode::{AudioInput, Encoder};
use cutlass_pcm::{Arrangement, Coding, Endian, Layout, PcmDecoder, PcmEncoder};

const RATE: u32 = 48_000;
const CHANNELS: usize = 2;
const FRAME_BYTES: usize = 4;

struct Times {
    min: Duration,
    median: Duration,
    max: Duration,
}

struct Row {
    name: &'static str,
    times: Times,
}

fn main() -> Result<(), String> {
    let Some(ffmpeg) = find_ffmpeg() else {
        println!("skip: ffmpeg is not installed");
        return Ok(());
    };
    let dir = std::env::temp_dir().join(format!("cutlass-pcm-bench-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|err| err.to_string())?;
    let outcome = run(&ffmpeg, &dir);
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

fn run(ffmpeg: &Path, dir: &Path) -> Result<(), String> {
    let source_path = dir.join("source.s16le");
    write_source(ffmpeg, &source_path)?;
    let source = std::fs::read(&source_path).map_err(|err| err.to_string())?;
    let frames = frame_count(&source)?;
    let reference = s16le_samples(&source)?;
    let duration = i64::try_from(frames).map_err(|_| "frame count does not fit".to_string())?;
    let packet = Packet::audio(source.clone(), 0, duration, RATE);

    let mut decoder = PcmDecoder::open(&params()).map_err(|err| err.to_string())?;
    let pcm = decode_pcm(&mut decoder, &packet)?;
    let decoded = pcm
        .as_i16()
        .ok_or_else(|| "decoded PCM is not i16".to_string())?;
    let delta = max_abs_delta(decoded, &reference)?;
    if delta != 0 {
        return Err(format!("max abs sample delta {delta}"));
    }
    let frames_u32 = u32::try_from(frames).map_err(|_| "frame count does not fit".to_string())?;
    if pcm.samples != frames_u32 {
        return Err(format!("decoder returned {} frames", pcm.samples));
    }

    let mut encoder = PcmEncoder::open(&params()).map_err(|err| err.to_string())?;
    let input = AudioInput { pcm };
    let encoded = encode_bytes(&mut encoder, &input)?;
    if encoded.as_slice() != source.as_slice() {
        return Err("encoder bytes differ from the ffmpeg payload".to_string());
    }
    let encoded_path = dir.join("encoded.s16le");
    std::fs::write(&encoded_path, &encoded).map_err(|err| err.to_string())?;
    ffmpeg_accepts(ffmpeg, &encoded_path, &encoded)?;

    let decode_cutlass = measure(|| time_decode(&mut decoder, &packet, frames_u32))?;
    let decode_ffmpeg = measure(|| time_ffmpeg(ffmpeg, &source_path, source.len()))?;
    let encode_cutlass = measure(|| time_encode(&mut encoder, &input, source.len()))?;
    let encode_ffmpeg = measure(|| time_ffmpeg(ffmpeg, &source_path, source.len()))?;

    print_report(
        frames,
        reference.len(),
        source.len(),
        delta,
        &[
            Row {
                name: "cutlass decode",
                times: decode_cutlass,
            },
            Row {
                name: "ffmpeg decode",
                times: decode_ffmpeg,
            },
            Row {
                name: "cutlass encode",
                times: encode_cutlass,
            },
            Row {
                name: "ffmpeg encode",
                times: encode_ffmpeg,
            },
        ],
    );
    Ok(())
}

fn print_report(frames: usize, samples: usize, bytes: usize, delta: u32, rows: &[Row]) {
    println!("s16le stereo, {RATE} Hz, {frames} frames, {samples} samples, {bytes} bytes");
    println!("max abs sample delta {delta}");
    println!("ffmpeg accepted the encoded bytes");
    println!("median of 3 after 1 warmup");
    println!("cutlass times bytes in memory");
    println!("ffmpeg time includes process startup");
    println!("ffmpeg rows convert pcm_s16le to pcm_s16le");
    for row in rows {
        let name = row.name;
        let median = seconds(row.times.median);
        let min = seconds(row.times.min);
        let max = seconds(row.times.max);
        println!("{name} median {median} min {min} max {max}");
    }
}

fn seconds(duration: Duration) -> String {
    format!("{:.6} s", duration.as_secs_f64())
}

fn measure(mut once: impl FnMut() -> Result<Duration, String>) -> Result<Times, String> {
    let _warmup = once()?;
    let mut times = [once()?, once()?, once()?];
    times.sort();
    Ok(Times {
        min: times[0],
        median: times[1],
        max: times[2],
    })
}

fn time_decode(decoder: &mut PcmDecoder, packet: &Packet, frames: u32) -> Result<Duration, String> {
    let started = Instant::now();
    let outputs = decoder.push(packet).map_err(|err| err.to_string())?;
    let count = std::hint::black_box(output_frames(&outputs)?);
    let elapsed = started.elapsed();
    if count != frames {
        return Err(format!("decoder returned {count} frames"));
    }
    Ok(elapsed)
}

fn time_encode(
    encoder: &mut PcmEncoder,
    input: &AudioInput,
    expected_len: usize,
) -> Result<Duration, String> {
    let started = Instant::now();
    let packets = encoder.push(input).map_err(|err| err.to_string())?;
    let written = std::hint::black_box(encoded_len(&packets)?);
    let elapsed = started.elapsed();
    if written != expected_len {
        return Err(format!("encoder wrote {written} bytes"));
    }
    Ok(elapsed)
}

fn time_ffmpeg(ffmpeg: &Path, input: &Path, expected_len: usize) -> Result<Duration, String> {
    let mut command = s16le_command(ffmpeg, input);
    let started = Instant::now();
    let output = command.output().map_err(|err| err.to_string())?;
    let elapsed = started.elapsed();
    if !output.status.success() {
        return Err(stderr_line(&output.stderr));
    }
    let written = std::hint::black_box(output.stdout.len());
    if written != expected_len {
        return Err(format!("ffmpeg wrote {written} bytes"));
    }
    Ok(elapsed)
}

fn output_frames(outputs: &[Output]) -> Result<u32, String> {
    let [Output::Audio(audio)] = outputs else {
        return Err("decoder returned no audio".to_string());
    };
    Ok(audio.pcm.samples)
}

fn encoded_len(packets: &[Packet]) -> Result<usize, String> {
    let [packet] = packets else {
        return Err("encoder returned no packet".to_string());
    };
    Ok(packet.data.len())
}

fn decode_pcm(decoder: &mut PcmDecoder, packet: &Packet) -> Result<Pcm, String> {
    let outputs = decoder.push(packet).map_err(|err| err.to_string())?;
    let [Output::Audio(audio)] = outputs.as_slice() else {
        return Err("decoder returned no audio".to_string());
    };
    if audio.pcm.rate != RATE || audio.pcm.layout.len() != CHANNELS {
        return Err("decoded rate or channel count differed".to_string());
    }
    if audio.pcm.format != SampleFormat::I16 {
        return Err("decoded PCM is not i16".to_string());
    }
    Ok(audio.pcm.clone())
}

fn encode_bytes(encoder: &mut PcmEncoder, input: &AudioInput) -> Result<Vec<u8>, String> {
    let packets = encoder.push(input).map_err(|err| err.to_string())?;
    let [packet] = packets.as_slice() else {
        return Err("encoder returned no packet".to_string());
    };
    Ok(packet.data.clone())
}

fn write_source(ffmpeg: &Path, path: &Path) -> Result<(), String> {
    let rate = RATE.to_string();
    let channels = CHANNELS.to_string();
    let mut command = Command::new(ffmpeg);
    command.args([
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000:duration=1",
        "-ar",
    ]);
    command.arg(&rate);
    command.arg("-ac");
    command.arg(&channels);
    command.args(["-c:a", "pcm_s16le", "-f", "s16le"]);
    command.arg(path);
    command.stdin(Stdio::null());
    let output = command.output().map_err(|err| err.to_string())?;
    if output.status.success() {
        Ok(())
    } else {
        Err(stderr_line(&output.stderr))
    }
}

fn ffmpeg_accepts(ffmpeg: &Path, input: &Path, expected: &[u8]) -> Result<(), String> {
    let mut command = s16le_command(ffmpeg, input);
    let output = command.output().map_err(|err| err.to_string())?;
    if !output.status.success() {
        return Err(stderr_line(&output.stderr));
    }
    if output.stdout.as_slice() == expected {
        Ok(())
    } else {
        Err(format!(
            "ffmpeg read {} bytes back, encoder wrote {}",
            output.stdout.len(),
            expected.len()
        ))
    }
}

fn s16le_command(ffmpeg: &Path, input: &Path) -> Command {
    let rate = RATE.to_string();
    let channels = CHANNELS.to_string();
    let mut command = Command::new(ffmpeg);
    command.args([
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "1",
        "-f",
        "s16le",
        "-ar",
    ]);
    command.arg(&rate);
    command.arg("-ac");
    command.arg(&channels);
    command.arg("-i");
    command.arg(input);
    command.args(["-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"]);
    command.stdin(Stdio::null());
    command
}

fn frame_count(bytes: &[u8]) -> Result<usize, String> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(FRAME_BYTES) {
        return Err(format!("payload is {} bytes", bytes.len()));
    }
    Ok(bytes.len() / FRAME_BYTES)
}

fn s16le_samples(bytes: &[u8]) -> Result<Vec<i16>, String> {
    let (chunks, rest) = bytes.as_chunks::<2>();
    if !rest.is_empty() {
        return Err(format!("payload is {} bytes", bytes.len()));
    }
    Ok(chunks
        .iter()
        .map(|pair| i16::from_le_bytes(*pair))
        .collect())
}

fn max_abs_delta(decoded: &[i16], reference: &[i16]) -> Result<u32, String> {
    if decoded.len() != reference.len() {
        return Err(format!(
            "decoded {} samples, ffmpeg wrote {}",
            decoded.len(),
            reference.len()
        ));
    }
    let mut max = 0u32;
    for (left, right) in decoded.iter().zip(reference) {
        max = max.max(i32::from(*left).abs_diff(i32::from(*right)));
    }
    Ok(max)
}

fn params() -> CodecParams {
    let spec = Layout::new(Coding::S16, Endian::Little, Arrangement::Interleaved);
    let mut opened = CodecParams::audio(CodecId::Pcm, RATE, ChannelLayout::stereo());
    opened.extra = spec.to_extra().to_vec();
    opened.sample_format = Some(SampleFormat::I16);
    opened
}

fn find_ffmpeg() -> Option<PathBuf> {
    for candidate in [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "ffmpeg",
    ] {
        let mut command = Command::new(candidate);
        command.arg("-version");
        command.stdin(Stdio::null());
        command.stdout(Stdio::null());
        command.stderr(Stdio::null());
        if command.status().is_ok_and(|status| status.success()) {
            return Some(PathBuf::from(candidate));
        }
    }
    None
}

fn stderr_line(stderr: &[u8]) -> String {
    let text = String::from_utf8_lossy(stderr);
    let line = text
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("ffmpeg failed");
    format!("ffmpeg failed: {line}")
}
