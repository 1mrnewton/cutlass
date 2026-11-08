//! Release throughput, frame latency, and resident memory of both codecs.
//!
//! Times are the median of three runs after one warmup. Memory is the
//! process resident set, sampled with `ps`, around a ten-second encode and
//! a minute of decoding the same packets. ffmpeg is timed when it is
//! installed; a missing binary is reported and is not a failure.

#![allow(clippy::print_stdout)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use cutlass_ac3::{Ac3Decoder, Ac3Encoder, ENCODER_DELAY};
use cutlass_decode::{Decoder, Output};
use cutlass_encode::{
    AudioInput, Channel, ChannelLayout, CodecError, CodecId, CodecParams, Encoder, Packet, Pcm,
};

const RATE: u32 = 48_000;
const SECONDS: usize = 10;

struct Case {
    name: &'static str,
    codec: CodecId,
    layout: ChannelLayout,
    bitrate: u32,
}

fn main() -> Result<(), String> {
    let cases = [
        Case {
            name: "AC-3 stereo 192 kbps",
            codec: CodecId::Ac3,
            layout: ChannelLayout::stereo(),
            bitrate: 192_000,
        },
        Case {
            name: "E-AC-3 stereo 192 kbps",
            codec: CodecId::Eac3,
            layout: ChannelLayout::stereo(),
            bitrate: 192_000,
        },
        Case {
            name: "AC-3 5.1 448 kbps",
            codec: CodecId::Ac3,
            layout: five_one()?,
            bitrate: 448_000,
        },
        Case {
            name: "E-AC-3 5.1 384 kbps",
            codec: CodecId::Eac3,
            layout: five_one()?,
            bitrate: 384_000,
        },
    ];
    println!("{SECONDS} s at {RATE} Hz; median of 3 after 1 warmup; times are this process");
    println!(
        "encoder delay {ENCODER_DELAY} samples ({:.3} ms at {RATE} Hz)",
        f64::from(ENCODER_DELAY) * 1_000.0 / f64::from(RATE)
    );
    if let Some(rss) = rss_kib() {
        println!("rss at start {rss} KiB");
    }
    for case in &cases {
        measure(case)?;
    }
    cancel_latency()?;
    if let Some(ffmpeg) = ffmpeg_bin() {
        ffmpeg_side(&ffmpeg);
    } else {
        println!("ffmpeg: not installed, comparison not run");
    }
    Ok(())
}

fn five_one() -> Result<ChannelLayout, String> {
    ChannelLayout::new(vec![
        Channel::FrontLeft,
        Channel::FrontRight,
        Channel::FrontCenter,
        Channel::Lfe,
        Channel::SideLeft,
        Channel::SideRight,
    ])
    .map_err(|error| format!("{error:?}"))
}

fn tone(frames: usize, channels: usize) -> Vec<f32> {
    let mut samples = Vec::with_capacity(frames * channels);
    for index in 0..frames {
        let t = index as f32 / RATE as f32;
        for channel in 0..channels {
            let freq = 180.0 + 90.0 * channel as f32;
            samples.push((t * freq * std::f32::consts::TAU).sin() * 0.2);
        }
    }
    samples
}

fn encode(case: &Case, samples: &[f32]) -> Result<(Duration, Vec<Packet>), String> {
    let mut params = CodecParams::audio(case.codec, RATE, case.layout.clone());
    params.bitrate = case.bitrate;
    let mut encoder = Ac3Encoder::open(&params).map_err(|error| error.to_string())?;
    let pcm = Pcm::from_f32(RATE, case.layout.clone(), samples.to_vec())
        .map_err(|error| format!("{error:?}"))?;
    let started = Instant::now();
    let mut packets = encoder
        .push(&AudioInput { pcm })
        .map_err(|error| error.to_string())?;
    packets.extend(encoder.drain().map_err(|error| error.to_string())?);
    Ok((started.elapsed(), packets))
}

fn decode(case: &Case, packets: &[Packet]) -> Result<Duration, String> {
    let mut decoder = Ac3Decoder::open(&CodecParams::audio(case.codec, RATE, case.layout.clone()))
        .map_err(|error| error.to_string())?;
    let started = Instant::now();
    for packet in packets {
        let outputs = decoder.push(packet).map_err(|error| error.to_string())?;
        for output in outputs {
            let Output::Audio(audio) = output else {
                continue;
            };
            let pcm = audio.pcm.as_f32().ok_or("decoded pcm is not f32")?;
            if pcm.iter().any(|sample| !sample.is_finite()) {
                return Err("decoded a non-finite sample".to_owned());
            }
        }
    }
    decoder.drain().map_err(|error| error.to_string())?;
    Ok(started.elapsed())
}

fn median(samples: [Duration; 3]) -> Duration {
    let mut ordered = samples;
    ordered.sort();
    ordered[1]
}

fn realtime(elapsed: Duration) -> f64 {
    (SECONDS as f64) / elapsed.as_secs_f64()
}

fn measure(case: &Case) -> Result<(), String> {
    let frames = RATE as usize * SECONDS;
    let samples = tone(frames, case.layout.len());
    let _ = encode(case, &samples)?;
    let mut encode_times = [Duration::ZERO; 3];
    let mut packets = Vec::new();
    for slot in &mut encode_times {
        let (elapsed, encoded) = encode(case, &samples)?;
        *slot = elapsed;
        packets = encoded;
    }
    let _ = decode(case, &packets)?;
    let mut decode_times = [Duration::ZERO; 3];
    for slot in &mut decode_times {
        *slot = decode(case, &packets)?;
    }
    let encode_median = median(encode_times);
    let decode_median = median(decode_times);
    let latency = frame_latency(case, &packets)?;
    let before = rss_kib();
    for _ in 0..6 {
        let _ = decode(case, &packets)?;
    }
    let after = rss_kib();
    println!("{}", case.name);
    println!(
        "  encode {:.3} ms, {:.0}x realtime",
        encode_median.as_secs_f64() * 1_000.0,
        realtime(encode_median)
    );
    println!(
        "  decode {:.3} ms, {:.0}x realtime",
        decode_median.as_secs_f64() * 1_000.0,
        realtime(decode_median)
    );
    println!(
        "  decode frame latency p50 {:.1} us, p95 {:.1} us, p99 {:.1} us over {} frames",
        latency.0,
        latency.1,
        latency.2,
        packets.len()
    );
    match (before, after) {
        (Some(before), Some(after)) => {
            println!("  rss before a further 60 s of decode {before} KiB, after {after} KiB");
        }
        _ => println!("  rss: ps did not report it"),
    }
    Ok(())
}

fn frame_latency(case: &Case, packets: &[Packet]) -> Result<(f64, f64, f64), String> {
    if packets.is_empty() {
        return Err("no frames to time".to_owned());
    }
    let mut decoder = Ac3Decoder::open(&CodecParams::audio(case.codec, RATE, case.layout.clone()))
        .map_err(|error| error.to_string())?;
    let mut times = Vec::with_capacity(packets.len());
    for packet in packets {
        let started = Instant::now();
        decoder.push(packet).map_err(|error| error.to_string())?;
        times.push(started.elapsed().as_secs_f64() * 1_000_000.0);
    }
    times.sort_by(f64::total_cmp);
    let at = |fraction: usize| times[times.len() * fraction / 100];
    Ok((at(50), at(95), at(99)))
}

fn cancel_latency() -> Result<(), String> {
    let layout = ChannelLayout::stereo();
    let mut params = CodecParams::audio(CodecId::Ac3, RATE, layout.clone());
    params.bitrate = 192_000;
    let mut encoder = Ac3Encoder::open(&params).map_err(|error| error.to_string())?;
    encoder.cancel_flag().store(true, Ordering::Relaxed);
    let samples = tone(RATE as usize * 5, layout.len());
    let pcm = Pcm::from_f32(RATE, layout, samples).map_err(|error| format!("{error:?}"))?;
    let started = Instant::now();
    let error = encoder.push(&AudioInput { pcm });
    let elapsed = started.elapsed();
    if !matches!(error, Err(CodecError::Cancelled)) {
        return Err(format!("cancel returned {error:?}"));
    }
    println!(
        "cancel of a 5 s stereo push, flag already set: {:.3} ms (sample check, then the flag, no frame coded)",
        elapsed.as_secs_f64() * 1_000.0
    );
    Ok(())
}

fn rss_kib() -> Option<u64> {
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok()?.trim().parse().ok()
}

fn ffmpeg_bin() -> Option<PathBuf> {
    for candidate in ["/opt/homebrew/bin/ffmpeg", "ffmpeg"] {
        let path = PathBuf::from(candidate);
        let ok = Command::new(&path)
            .arg("-version")
            .output()
            .ok()
            .is_some_and(|output| output.status.success());
        if ok {
            return Some(path);
        }
    }
    None
}

fn ffmpeg_side(ffmpeg: &PathBuf) {
    let dir = std::env::temp_dir().join(format!("cutlass-ac3-bench-{}", std::process::id()));
    if std::fs::create_dir_all(&dir).is_err() {
        println!("ffmpeg: could not make a temporary directory, comparison not run");
        return;
    }
    let pcm_path = dir.join("stereo.f32");
    let samples = tone(RATE as usize * SECONDS, 2);
    let bytes: Vec<u8> = samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect();
    if std::fs::write(&pcm_path, bytes).is_err() {
        println!("ffmpeg: could not write pcm, comparison not run");
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }
    let pcm = pcm_path.to_string_lossy().into_owned();
    for (codec, rate) in [("ac3", "192k"), ("eac3", "192k")] {
        let coded = dir.join(format!("stereo.{codec}"));
        let decoded = dir.join(format!("{codec}.f32"));
        let coded_text = coded.to_string_lossy().into_owned();
        let decoded_text = decoded.to_string_lossy().into_owned();
        let encoded = time_ffmpeg(
            ffmpeg,
            &[
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "f32le",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-i",
                &pcm,
                "-c:a",
                codec,
                "-b:a",
                rate,
                "-y",
                &coded_text,
            ],
        );
        let decoded = time_ffmpeg(
            ffmpeg,
            &[
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                &coded_text,
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-y",
                &decoded_text,
            ],
        );
        match (encoded, decoded) {
            (Some(encoded), Some(decoded)) => println!(
                "ffmpeg {codec} stereo, process time including startup: encode {:.3} ms ({:.0}x), decode {:.3} ms ({:.0}x)",
                encoded.as_secs_f64() * 1_000.0,
                realtime(encoded),
                decoded.as_secs_f64() * 1_000.0,
                realtime(decoded)
            ),
            _ => println!("ffmpeg {codec}: the command failed, comparison not run"),
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

fn time_ffmpeg(ffmpeg: &PathBuf, args: &[&str]) -> Option<Duration> {
    let warmup = Command::new(ffmpeg).args(args).output().ok()?;
    if !warmup.status.success() {
        return None;
    }
    let mut times = [Duration::ZERO; 3];
    for slot in &mut times {
        let started = Instant::now();
        let output = Command::new(ffmpeg).args(args).output().ok()?;
        if !output.status.success() {
            return None;
        }
        *slot = started.elapsed();
    }
    Some(median(times))
}
