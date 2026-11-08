//! Release throughput, delay, and cancel latency.
//!
//! ```text
//! cargo test --release -p cutlass-vorbis --test measure -- --ignored --nocapture
//! ```
//!
//! The decode gate is 100× real time and the encode gate is 20×, on a
//! dense stereo 44.1 kHz item. Times are the median of three runs after
//! one warmup.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::print_stderr
)]

use std::sync::atomic::Ordering;
use std::thread;
use std::time::Instant;

use cutlass_codec::CodecId;
use cutlass_codec::access::{Channel, ChannelLayout, SideData};
use cutlass_encode::{AudioInput, CodecParams, Encoder, Pcm};
use cutlass_vorbis::{Options, VorbisDecoder, VorbisEncoder};

const RATE: u32 = 44_100;
const SECONDS: usize = 10;
const CHANNELS: usize = 2;

fn layout() -> ChannelLayout {
    ChannelLayout::new(vec![Channel::FrontLeft, Channel::FrontRight]).expect("layout")
}

fn dense() -> Vec<f32> {
    let frames = RATE as usize * SECONDS;
    let mut samples = Vec::with_capacity(frames * CHANNELS);
    let mut noise = 0x1234_5678_u32;
    for frame in 0..frames {
        noise = noise.wrapping_mul(1664525).wrapping_add(1013904223);
        let n = (noise >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
        let t = frame as f32 / RATE as f32;
        let left = (t * 440.0 * std::f32::consts::TAU).sin() * 0.2
            + (t * 1250.0 * std::f32::consts::TAU).sin() * 0.1
            + n * 0.05;
        let right = (t * 660.0 * std::f32::consts::TAU).sin() * 0.2
            + (t * 3000.0 * std::f32::consts::TAU).sin() * 0.1
            + n * 0.05;
        samples.push(left);
        samples.push(right);
    }
    samples
}

fn median(mut times: Vec<f64>) -> f64 {
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

fn rss_kb() -> u64 {
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .expect("ps");
    String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .next()
        .unwrap_or("0")
        .parse()
        .unwrap_or(0)
}

fn encode(pcm: &[f32]) -> (Vec<Vec<u8>>, Vec<u8>, i64, i64) {
    let speakers = layout();
    let mut params = CodecParams::audio(CodecId::Vorbis, RATE, speakers.clone());
    params.bitrate = 128_000;
    let mut encoder = VorbisEncoder::open_with(&params, Options { quality: 0.4 }).expect("open");
    let pcm = Pcm::from_f32(RATE, speakers, pcm.to_vec()).expect("pcm");
    let mut packets = encoder.push(&AudioInput { pcm }).expect("push");
    packets.extend(encoder.drain().expect("drain"));
    let config = packets
        .iter()
        .find_map(|packet| {
            packet.side.iter().find_map(|side| match side {
                SideData::Config(config) => Some(config.clone()),
                _ => None,
            })
        })
        .expect("config");
    let delay = packets.iter().find_map(|packet| {
        packet.side.iter().find_map(|side| match side {
            SideData::Gapless {
                encoder_delay,
                trailing_padding,
                ..
            } => Some((*encoder_delay, *trailing_padding)),
            _ => None,
        })
    });
    let (encoder_delay, trailing) = delay.expect("gapless");
    let bytes = packets.into_iter().map(|packet| packet.data).collect();
    (bytes, config, i64::from(encoder_delay), i64::from(trailing))
}

fn percentile(times: &[f64], fraction: f64) -> f64 {
    let mut sorted = times.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() - 1) as f64 * fraction) as usize]
}

fn decode_all(config: &[u8], packets: &[Vec<u8>]) -> usize {
    let mut decoder = VorbisDecoder::from_config(config).expect("decoder");
    let mut frames = 0;
    for packet in packets {
        let samples = decoder.decode(packet).expect("decode");
        frames += samples.len() / CHANNELS;
    }
    frames
}

#[test]
#[ignore = "release measurement"]
fn release_throughput_delay_and_cancel() {
    let pcm = dense();
    let audio = SECONDS as f64;
    let _ = encode(&pcm);
    let encode_times: Vec<f64> = (0..3)
        .map(|_| {
            let started = Instant::now();
            let _ = encode(&pcm);
            started.elapsed().as_secs_f64()
        })
        .collect();
    let (packets, config, encoder_delay, trailing) = encode(&pcm);
    assert_eq!(encoder_delay, 0, "the encoder adds no leading delay");
    assert!(!packets[0].is_empty());
    let played = decode_all(&config, &packets) - usize::try_from(trailing).unwrap();
    assert_eq!(played, RATE as usize * SECONDS);
    let decode_times: Vec<f64> = (0..3)
        .map(|_| {
            let started = Instant::now();
            let frames = decode_all(&config, &packets);
            assert_eq!(
                frames - usize::try_from(trailing).unwrap(),
                RATE as usize * SECONDS
            );
            started.elapsed().as_secs_f64()
        })
        .collect();
    let encode_s = median(encode_times);
    let decode_s = median(decode_times);
    let before = rss_kb();
    let _ = encode(&pcm);
    let after_encode = rss_kb();
    let _ = decode_all(&config, &packets);
    let after_decode = rss_kb();

    let mut decoder = VorbisDecoder::from_config(&config).expect("decoder");
    let mut packet_s = Vec::with_capacity(packets.len());
    for packet in &packets {
        let started = Instant::now();
        decoder.decode(packet).expect("packet");
        packet_s.push(started.elapsed().as_secs_f64());
    }
    let first_packet_s = packet_s[0];

    let speakers = layout();
    let mut params = CodecParams::audio(CodecId::Vorbis, RATE, speakers.clone());
    params.bitrate = 128_000;
    let mut encoder = VorbisEncoder::open_with(&params, Options { quality: 0.4 }).expect("open");
    let flag = encoder.cancel_flag();
    let long_frames = RATE as usize * 60;
    let long: Vec<f32> = (0..long_frames * CHANNELS)
        .map(|i| ((i as f32) * 0.01).sin() * 0.2)
        .collect();
    let pcm = Pcm::from_f32(RATE, speakers, long).expect("pcm");
    let started = Instant::now();
    let worker = thread::spawn(move || encoder.push(&AudioInput { pcm }));
    thread::sleep(std::time::Duration::from_millis(30));
    let marked = Instant::now();
    flag.store(true, Ordering::Relaxed);
    let result = worker.join().expect("join");
    let cancel_s = marked.elapsed().as_secs_f64();
    let total_s = started.elapsed().as_secs_f64();
    assert!(
        result.is_err(),
        "cancel did not stop a 60 s encode in {total_s:.3} s"
    );

    eprintln!(
        "vorbis measure: encode {encode_s:.4} s ({:.1}×), decode {decode_s:.4} s ({:.1}×), packet p50 {:.6} p95 {:.6} p99 {:.6} first {first_packet_s:.6} s, encoder delay {encoder_delay}, trailing {trailing}, rss kb before {before} after encode {after_encode} after decode {after_decode}, cancel {cancel_s:.4} s",
        audio / encode_s,
        audio / decode_s,
        percentile(&packet_s, 0.50),
        percentile(&packet_s, 0.95),
        percentile(&packet_s, 0.99),
    );
    assert!(
        audio / decode_s >= 100.0,
        "decode {:.1}× is under 100×",
        audio / decode_s
    );
    assert!(
        audio / encode_s >= 20.0,
        "encode {:.1}× is under 20×",
        audio / encode_s
    );
}
