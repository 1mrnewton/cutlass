//! In-process fuzz campaigns, run on every test run. cargo-fuzz is not
//! installed here, so these stand in for it: seeded, reproducible, and
//! bounded by `CUTLASS_OPUS_CAMPAIGN_SECONDS` (default 2 seconds each).
//!
//! - Decoder: random bytes, and bit flips, truncations, splices, and TOC
//!   rewrites of real packets (this crate's encoder; with
//!   `CUTLASS_OPUS_VECTORS` also every SILK, hybrid, and CELT packet of the
//!   RFC 8251 vectors), through mono, stereo, and 5.1 decoders at every API
//!   rate, mixed with concealment and FEC calls. Nothing panics, every
//!   sample is finite, a failed push leaves the decoder as it was, and
//!   after a reset it decodes like a fresh one.
//! - Encoder: random bitrates, rate modes, channel counts, push sizes, and
//!   hostile samples (NaN, infinities, over full scale, full-scale square
//!   waves, denormals). Nothing panics; NaN, infinities, and samples over
//!   full scale are refused without changing the encoder; every packet
//!   parses and decodes to finite samples; and the durations add up to the
//!   pre-skip plus the accepted input.

#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::unwrap_used
)]

use std::time::{Duration, Instant};

use cutlass_decode::{CodecError, Decoder};
use cutlass_encode::{AudioInput, ChannelLayout, CodecId, CodecParams, Encoder, Pcm};
use cutlass_opus::{OpusDecoder, OpusEncoder, OpusHead, parse_packet, vorbis_mapping};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

fn budget() -> Duration {
    let seconds = std::env::var("CUTLASS_OPUS_CAMPAIGN_SECONDS")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(2.0);
    Duration::from_secs_f64(seconds)
}

fn layout(channels: usize) -> ChannelLayout {
    if channels == 1 {
        ChannelLayout::mono()
    } else {
        ChannelLayout::stereo()
    }
}

fn encode(channels: usize, bitrate: u32, cbr: bool, samples: &[f32]) -> Vec<Vec<u8>> {
    let mut params = CodecParams::audio(CodecId::Opus, 48_000, layout(channels));
    params.bitrate = bitrate;
    params.constant_rate = cbr;
    let mut encoder = OpusEncoder::open(&params).expect("encoder");
    let pcm = Pcm::from_f32(48_000, layout(channels), samples.to_vec()).expect("pcm");
    let mut packets = encoder.push(&AudioInput { pcm }).expect("push");
    packets.extend(encoder.drain().expect("drain"));
    packets.into_iter().map(|packet| packet.data).collect()
}

/// Packets to mutate: this crate's encoder at a few rates, and the vectors
/// when they are there.
fn corpus() -> Vec<Vec<u8>> {
    let tone: Vec<f32> = (0..2 * 48_000)
        .map(|i| ((i as f32) * 0.021).sin() * 0.5 + ((i * 7 % 13) as f32 - 6.0) * 0.01)
        .collect();
    let mut out = Vec::new();
    for (channels, bitrate, cbr) in [(1, 6_000, false), (2, 64_000, true), (2, 510_000, false)] {
        let samples: Vec<f32> = if channels == 1 {
            tone.iter().step_by(2).copied().collect()
        } else {
            tone.clone()
        };
        out.extend(encode(channels, bitrate, cbr, &samples));
    }
    if let Some(dir) = std::env::var_os("CUTLASS_OPUS_VECTORS") {
        for index in 1..=12 {
            let path = std::path::Path::new(&dir).join(format!("testvector{index:02}.bit"));
            let bytes = std::fs::read(path).unwrap_or_default();
            let mut at = 0;
            while at + 8 <= bytes.len() {
                let len = u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
                at += 8;
                let end = (at + len).min(bytes.len());
                if end > at {
                    out.push(bytes[at..end].to_vec());
                }
                at = end;
            }
        }
    }
    out
}

fn mutate(rng: &mut Rng, corpus: &[Vec<u8>]) -> Vec<u8> {
    let mut data = corpus[rng.below(corpus.len())].clone();
    match rng.below(6) {
        0 => {
            let len = rng.below(400) + 1;
            data = (0..len).map(|_| rng.next() as u8).collect();
        }
        1 => {
            for _ in 0..=rng.below(8) {
                let bit = rng.below(data.len() * 8);
                data[bit / 8] ^= 1 << (bit % 8);
            }
        }
        2 => data.truncate(rng.below(data.len()) + 1),
        3 => {
            let other = &corpus[rng.below(corpus.len())];
            let cut = rng.below(other.len());
            data.extend_from_slice(&other[cut..]);
        }
        4 => data[0] = rng.next() as u8,
        _ => {
            // A code 3 count byte with random flags and frame count.
            data[0] |= 3;
            if data.len() > 1 {
                data[1] = rng.next() as u8;
            }
        }
    }
    data
}

/// A code 0 packet in self-delimited framing (RFC 6716 appendix B): the
/// TOC, then the frame length, then the frame.
fn delimit(part: &[u8]) -> Vec<u8> {
    let toc = part[0] & !3;
    let frame = &part[1..];
    let mut out = vec![toc];
    let len = frame.len().min(1275);
    if len < 252 {
        out.push(len as u8);
    } else {
        let first = 252 + (len - 252) % 4;
        out.push(first as u8);
        out.push(((len - first) / 4) as u8);
    }
    out.extend_from_slice(&frame[..len]);
    out
}

/// Four streams (two coupled) from mutated parts, all but the last
/// self-delimited; sometimes one is missing or one more is left over.
fn multistream(rng: &mut Rng, corpus: &[Vec<u8>]) -> Vec<u8> {
    let count = match rng.below(8) {
        0 => 3,
        1 => 5,
        _ => 4,
    };
    let mut out = Vec::new();
    for index in 0..count {
        let part = mutate(rng, corpus);
        if index + 1 < count {
            out.extend(delimit(&part));
        } else {
            out.extend(part);
        }
    }
    out
}

fn surround() -> OpusHead {
    let (streams, coupled, mapping) = vorbis_mapping(6).expect("5.1");
    OpusHead {
        channels: 6,
        pre_skip: 312,
        input_rate: 48_000,
        output_gain: 0,
        family: 1,
        streams,
        coupled,
        mapping,
    }
}

#[test]
fn the_decoder_survives_hostile_packets() {
    let corpus = corpus();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let heads = [
        OpusHead::simple(1, 0, 48_000),
        OpusHead::simple(2, 0, 48_000),
        surround(),
    ];
    let rates = [8_000, 12_000, 16_000, 24_000, 48_000];
    let started = Instant::now();
    let (mut pushes, mut refused) = (0u64, 0u64);
    while started.elapsed() < budget() {
        let head = heads[rng.below(heads.len())].clone();
        let rate = rates[rng.below(rates.len())];
        let mut decoder = OpusDecoder::with_head(head.clone(), rate).expect("decoder");
        for _ in 0..64 {
            let data = if head.channels == 6 && rng.below(2) == 0 {
                multistream(&mut rng, &corpus)
            } else {
                mutate(&mut rng, &corpus)
            };
            pushes += 1;
            let before = decoder.final_range();
            match rng.below(10) {
                0 => {
                    let samples = (rng.below(48) + 1) * rate as usize / 400;
                    if let Ok(out) = decoder.conceal(samples) {
                        assert!(out.iter().all(|x| x.is_finite()));
                    }
                }
                1 => {
                    let samples = (rng.below(48) + 1) * rate as usize / 400;
                    if let Ok(out) = decoder.decode_fec(&data, samples) {
                        assert!(out.iter().all(|x| x.is_finite()));
                    }
                }
                _ => match decoder.decode(&data) {
                    Ok(out) => {
                        assert!(out.iter().all(|x| x.is_finite()));
                        assert_eq!(out.len() % usize::from(head.channels), 0);
                    }
                    Err(_) => {
                        refused += 1;
                        assert_eq!(decoder.final_range(), before);
                    }
                },
            }
        }
        decoder.reset();
        let mut fresh = OpusDecoder::with_head(head.clone(), rate).expect("fresh");
        if head.channels <= 2 {
            let good = &corpus[rng.below(3)];
            if parse_packet(good, false).is_ok() {
                assert_eq!(decoder.decode(good).ok(), fresh.decode(good).ok());
            }
        }
    }
    eprintln!(
        "decoder campaign: {pushes} calls, {refused} packets refused, {:.1} s",
        started.elapsed().as_secs_f64()
    );
}

#[test]
fn the_encoder_survives_hostile_input_and_settings() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let started = Instant::now();
    let mut runs = 0u64;
    while started.elapsed() < budget() {
        runs += 1;
        let channels = rng.below(2) + 1;
        let mut params = CodecParams::audio(CodecId::Opus, 48_000, layout(channels));
        params.bitrate = match rng.below(4) {
            0 => 0,
            1 => rng.below(600_000) as u32,
            _ => 6_000 + rng.below(504_000) as u32,
        };
        params.constant_rate = rng.below(2) == 0;
        let mut encoder = OpusEncoder::open(&params).expect("encoder");
        let frames = rng.below(48_000) + 1;
        let kind = rng.below(6);
        let samples: Vec<f32> = (0..frames * channels)
            .map(|i| match kind {
                0 => f32::NAN,
                1 => {
                    if i % 2 == 0 {
                        f32::INFINITY
                    } else {
                        f32::NEG_INFINITY
                    }
                }
                2 => {
                    if (i / 37) % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                }
                3 => f32::MIN_POSITIVE / 4.0,
                4 => 1.5,
                _ => (rng.next() as u32 as f32 / u32::MAX as f32 - 0.5) * 2.0,
            })
            .collect();
        let mut packets = Vec::new();
        let mut at = 0;
        let mut accepted = 0;
        while at < samples.len() {
            let take = ((rng.below(4000) + 1) * channels).min(samples.len() - at);
            let pcm = Pcm::from_f32(48_000, layout(channels), samples[at..at + take].to_vec())
                .expect("pcm");
            match encoder.push(&AudioInput { pcm }) {
                Ok(out) => {
                    packets.extend(out);
                    accepted += take / channels;
                }
                // Non-finite and over-full-scale samples are refused, as
                // the AAC encoder refuses them, and change nothing.
                Err(CodecError::NonFinite) => assert!(kind <= 1),
                Err(CodecError::OutOfRange) => assert_eq!(kind, 4),
                Err(error) => panic!("{error:?}"),
            }
            at += take;
        }
        // After refusals the encoder still takes good samples.
        let tail = vec![0.25f32; 960 * channels];
        let pcm = Pcm::from_f32(48_000, layout(channels), tail).expect("pcm");
        packets.extend(encoder.push(&AudioInput { pcm }).expect("push"));
        accepted += 960;
        packets.extend(encoder.drain().expect("drain"));
        let mut decoder =
            OpusDecoder::with_head(OpusHead::simple(channels as u8, 0, 48_000), 48_000)
                .expect("decoder");
        let mut total = 0;
        for packet in &packets {
            parse_packet(&packet.data, false).expect("packet parses");
            let out = decoder.decode(&packet.data).expect("decodes");
            assert!(out.iter().all(|x| x.is_finite()));
            total += packet.duration;
        }
        assert_eq!(total, 120 + accepted as i64);
    }
    eprintln!(
        "encoder campaign: {runs} runs, {:.1} s",
        started.elapsed().as_secs_f64()
    );
}
