//! In-process fuzz of the Vorbis decoder and encoder.
//!
//! cargo-fuzz is not used. It is not installed here, and libfuzzer-sys
//! builds C, which this crate does not allow. `fuzz_seeds_do_not_panic`
//! runs on every `cargo test`. The timed campaigns are ignored:
//!
//! ```text
//! VORBIS_DECODE_FUZZ_SECS=1800 VORBIS_ENCODE_FUZZ_SECS=600 cargo test --release -p cutlass-vorbis --test fuzz -- --ignored --nocapture --test-threads=2
//! ```

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

mod common;

use std::time::Instant;

use common::split_stream;
use cutlass_codec::CodecId;
use cutlass_codec::access::{Channel, ChannelLayout, Packet, SideData};
use cutlass_decode::Decoder;
use cutlass_encode::{AudioInput, CodecParams, Encoder, Pcm};
use cutlass_vorbis::{Options, VorbisDecoder, VorbisEncoder};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() as usize) % n.max(1)
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next() & 0xff) as u8).collect()
    }
}

struct Stream {
    config: Vec<u8>,
    packets: Vec<Vec<u8>>,
}

struct Stats {
    cases: u64,
    refused: u64,
    nonfinite: u64,
}

impl Stats {
    fn add(&mut self, other: Stats) {
        self.cases += other.cases;
        self.refused += other.refused;
        self.nonfinite += other.nonfinite;
    }
}

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/tests/fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap_or_else(|error| panic!("fixture {name}: {error}"))
}

fn layout(channels: usize) -> ChannelLayout {
    let speakers = match channels {
        1 => vec![Channel::FrontCenter],
        2 => vec![Channel::FrontLeft, Channel::FrontRight],
        6 => vec![
            Channel::FrontLeft,
            Channel::FrontCenter,
            Channel::FrontRight,
            Channel::BackLeft,
            Channel::BackRight,
            Channel::Lfe,
        ],
        _ => panic!("channels {channels}"),
    };
    ChannelLayout::new(speakers).expect("layout")
}

fn encode_stream(channels: usize, rate: u32, frames: usize) -> Stream {
    let speakers = layout(channels);
    let params = CodecParams::audio(CodecId::Vorbis, rate, speakers.clone());
    let mut encoder = VorbisEncoder::open_with(&params, Options { quality: 0.4 }).expect("encode");
    let pcm: Vec<f32> = (0..frames * channels)
        .map(|i| ((i as f32) * 0.017).sin() * 0.4)
        .collect();
    let pcm = Pcm::from_f32(rate, speakers, pcm).expect("pcm");
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
    Stream {
        config,
        packets: packets.into_iter().map(|packet| packet.data).collect(),
    }
}

fn corpus() -> Vec<Stream> {
    let mut streams = Vec::new();
    for name in [
        "stereo_44100_q4.ogg",
        "clicks_48000_q9.ogg",
        "sweep_8000_mono_q0.ogg",
        "surround51_48000_q4.ogg",
        "ffmpeg_stereo_44100.ogg",
    ] {
        let (config, audio) = split_stream(&fixture(name));
        streams.push(Stream {
            config,
            packets: audio.into_iter().map(|packet| packet.data).collect(),
        });
    }
    streams.push(encode_stream(2, 44_100, 4096));
    streams.push(encode_stream(1, 8_000, 2048));
    streams
}

fn note_samples(samples: &[f32], stats: &mut Stats) {
    if samples.iter().any(|sample| !sample.is_finite()) {
        stats.nonfinite += 1;
    }
}

fn decode_case(rng: &mut Rng, corpus: &[Stream]) -> Stats {
    let mut stats = Stats {
        cases: 1,
        refused: 0,
        nonfinite: 0,
    };
    let stream = &corpus[rng.below(corpus.len())];
    match rng.below(6) {
        0 => {
            let n = rng.below(4096);
            let bytes = rng.bytes(n);
            if VorbisDecoder::from_config(&bytes).is_err() {
                stats.refused += 1;
            }
        }
        1 => {
            let end = rng.below(stream.config.len().max(1));
            if VorbisDecoder::from_config(&stream.config[..end]).is_err() {
                stats.refused += 1;
            }
        }
        _ => {
            let Ok(mut decoder) = VorbisDecoder::from_config(&stream.config) else {
                stats.refused += 1;
                return stats;
            };
            let packet = mutated(rng, stream);
            match decoder.decode(&packet) {
                Ok(samples) => note_samples(&samples, &mut stats),
                Err(_) => stats.refused += 1,
            }
            decoder.reset();
            if let Some(first) = stream
                .packets
                .iter()
                .find(|packet| packet.first().is_some_and(|b| b & 1 == 0))
            {
                match decoder.decode(first) {
                    Ok(samples) => note_samples(&samples, &mut stats),
                    Err(_) => stats.refused += 1,
                }
            }
        }
    }
    stats
}

fn mutated(rng: &mut Rng, stream: &Stream) -> Vec<u8> {
    if stream.packets.is_empty() || rng.below(5) == 0 {
        let n = rng.below(2048);
        return rng.bytes(n);
    }
    let mut packet = stream.packets[rng.below(stream.packets.len())].clone();
    if packet.is_empty() {
        return packet;
    }
    match rng.below(4) {
        0 => {
            for _ in 0..1 + rng.below(8) {
                let at = rng.below(packet.len());
                packet[at] ^= 1 << rng.below(8);
            }
        }
        1 => packet.truncate(rng.below(packet.len())),
        2 => {
            let at = rng.below(packet.len());
            let n = 1 + rng.below(32);
            let extra = rng.bytes(n);
            packet.splice(at..at, extra);
        }
        _ => {
            let at = rng.below(packet.len());
            let end = (at + 1 + rng.below(24)).min(packet.len());
            for byte in &mut packet[at..end] {
                *byte = (rng.next() & 0xff) as u8;
            }
        }
    }
    packet
}

fn encode_case(rng: &mut Rng) -> Stats {
    let mut stats = Stats {
        cases: 1,
        refused: 0,
        nonfinite: 0,
    };
    let channels = rng.below(10);
    let rates = [0, 1, 7_999, 8_000, 44_100, 48_000, 192_000, 192_001];
    let rate = rates[rng.below(rates.len())];
    let qualities = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -1.0,
        0.0,
        0.4,
        1.0,
        3.0,
    ];
    let quality = qualities[rng.below(qualities.len())];
    let bitrates = [0, 1, 64_000, 128_000, u32::MAX];
    let Some(speakers) = vorbis_speakers(channels) else {
        stats.refused += 1;
        return stats;
    };
    let mut params = CodecParams::audio(CodecId::Vorbis, rate, speakers.clone());
    params.bitrate = bitrates[rng.below(bitrates.len())];
    params.constant_rate = rng.below(2) == 0;
    let mut encoder = match VorbisEncoder::open_with(&params, Options { quality }) {
        Ok(encoder) => encoder,
        Err(_) => {
            stats.refused += 1;
            return stats;
        }
    };
    let frames = rng.below(400);
    let mut samples = Vec::with_capacity(frames * channels);
    for _ in 0..frames * channels {
        let unit = (rng.next() % 10_000) as f32 / 5_000.0 - 1.0;
        let sample = if rng.below(32) == 0 {
            f32::NAN
        } else {
            unit * if rng.below(8) == 0 { 2.0 } else { 0.8 }
        };
        samples.push(sample);
    }
    let Ok(pcm) = Pcm::from_f32(rate, speakers, samples) else {
        stats.refused += 1;
        return stats;
    };
    let pushed = encoder.push(&AudioInput { pcm });
    let drained = pushed.as_ref().ok().and_then(|_| encoder.drain().ok());
    match (pushed, drained) {
        (Ok(mut packets), Some(more)) => {
            packets.extend(more);
            play_back(&packets, &mut stats);
        }
        _ => stats.refused += 1,
    }
    stats
}

fn play_back(packets: &[Packet], stats: &mut Stats) {
    let Some(config) = packets.iter().find_map(|packet| {
        packet.side.iter().find_map(|side| match side {
            SideData::Config(config) => Some(config.as_slice()),
            _ => None,
        })
    }) else {
        stats.refused += 1;
        return;
    };
    let Ok(mut decoder) = VorbisDecoder::from_config(config) else {
        stats.refused += 1;
        return;
    };
    for packet in packets {
        match decoder.decode(&packet.data) {
            Ok(samples) => note_samples(&samples, stats),
            Err(_) => stats.refused += 1,
        }
    }
    decoder.reset();
}

fn vorbis_speakers(channels: usize) -> Option<ChannelLayout> {
    let speakers = match channels {
        1 => vec![Channel::FrontCenter],
        2 => vec![Channel::FrontLeft, Channel::FrontRight],
        3 => vec![
            Channel::FrontLeft,
            Channel::FrontCenter,
            Channel::FrontRight,
        ],
        4 => vec![
            Channel::FrontLeft,
            Channel::FrontRight,
            Channel::BackLeft,
            Channel::BackRight,
        ],
        5 => vec![
            Channel::FrontLeft,
            Channel::FrontCenter,
            Channel::FrontRight,
            Channel::BackLeft,
            Channel::BackRight,
        ],
        6 => vec![
            Channel::FrontLeft,
            Channel::FrontCenter,
            Channel::FrontRight,
            Channel::BackLeft,
            Channel::BackRight,
            Channel::Lfe,
        ],
        7 => vec![
            Channel::FrontLeft,
            Channel::FrontCenter,
            Channel::FrontRight,
            Channel::SideLeft,
            Channel::SideRight,
            Channel::BackCenter,
            Channel::Lfe,
        ],
        8 => vec![
            Channel::FrontLeft,
            Channel::FrontCenter,
            Channel::FrontRight,
            Channel::SideLeft,
            Channel::SideRight,
            Channel::BackLeft,
            Channel::BackRight,
            Channel::Lfe,
        ],
        _ => return None,
    };
    ChannelLayout::new(speakers).ok()
}

fn seconds(name: &str) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(60)
}

#[test]
fn fuzz_seeds_do_not_panic() {
    let corpus = corpus();
    let mut rng = Rng(0x00c0_ffee);
    let mut stats = Stats {
        cases: 0,
        refused: 0,
        nonfinite: 0,
    };
    for _ in 0..96 {
        stats.add(decode_case(&mut rng, &corpus));
    }
    for _ in 0..32 {
        stats.add(encode_case(&mut rng));
    }
    assert!(stats.cases >= 128, "cases {}", stats.cases);
    assert_eq!(stats.nonfinite, 0, "a decoded sample was not finite");
    assert!(stats.refused > 0, "nothing was refused");
}

/// Decode mutants until `VORBIS_DECODE_FUZZ_SECS` (default 60).
#[test]
#[ignore = "timed campaign; set VORBIS_DECODE_FUZZ_SECS"]
fn fuzz_campaign_decode() {
    let corpus = corpus();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let budget = seconds("VORBIS_DECODE_FUZZ_SECS");
    let started = Instant::now();
    let mut stats = Stats {
        cases: 0,
        refused: 0,
        nonfinite: 0,
    };
    while started.elapsed().as_secs() < budget {
        stats.add(decode_case(&mut rng, &corpus));
    }
    eprintln!(
        "vorbis decode fuzz: {} cases, {} refused, {} non-finite, {} s, seed {:#x}",
        stats.cases, stats.refused, stats.nonfinite, budget, rng.0
    );
    assert_eq!(stats.nonfinite, 0);
}

/// Encoder opens and short pushes until `VORBIS_ENCODE_FUZZ_SECS` (default 60).
#[test]
#[ignore = "timed campaign; set VORBIS_ENCODE_FUZZ_SECS"]
fn fuzz_campaign_encode() {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let budget = seconds("VORBIS_ENCODE_FUZZ_SECS");
    let started = Instant::now();
    let mut stats = Stats {
        cases: 0,
        refused: 0,
        nonfinite: 0,
    };
    while started.elapsed().as_secs() < budget {
        stats.add(encode_case(&mut rng));
    }
    eprintln!(
        "vorbis encode fuzz: {} cases, {} refused, {} non-finite, {} s, seed {:#x}",
        stats.cases, stats.refused, stats.nonfinite, budget, rng.0
    );
    assert_eq!(stats.nonfinite, 0);
}
