//! In-process decoder and encoder campaigns. This is not libFuzzer.
//!
//! The short runs are part of the suite. The long ones are:
//! `FLAC_FUZZ_SECS=1800 cargo test --release -p cutlass-flac --test external -- --ignored --nocapture`
//!
//! The decoder campaign damages frames this crate encoded and, half the
//! time, refreshes the CRC-16 so the damage reaches the subframe readers.
//! The encoder campaign draws random settings and signals and requires an
//! exact round trip.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::{Duration, Instant};

use cutlass_decode::{ChannelLayout, CodecId, CodecParams, Decoder, Packet};
use cutlass_flac::{Apodization, FlacDecoder, FlacEncoder, Md5Check, Settings, Stereo, crc};

use crate::support::{Kind, Rng, signal};

fn below(rng: &mut Rng, n: usize) -> usize {
    usize::try_from(rng.next() % n.max(1) as u64).unwrap_or(0)
}

fn random_settings(rng: &mut Rng) -> Settings {
    let mut settings = Settings::level(u8::try_from(below(rng, 9)).unwrap_or(0)).expect("level");
    if below(rng, 2) == 0 {
        settings.block_size = u16::try_from(16 + below(rng, 8192)).unwrap_or(4096);
        settings.max_lpc_order = u8::try_from(below(rng, 33)).unwrap_or(0);
        settings.max_partition_order = u8::try_from(below(rng, 16)).unwrap_or(0);
        settings.min_partition_order =
            u8::try_from(below(rng, usize::from(settings.max_partition_order) + 1)).unwrap_or(0);
        settings.exhaustive_order = below(rng, 4) == 0;
        settings.stereo =
            [Stereo::Independent, Stereo::Estimate, Stereo::Exhaustive][below(rng, 3)];
        settings.apodizations = vec![match below(rng, 3) {
            0 => Apodization::Tukey((rng.next() % 101) as f64 / 100.0),
            1 => Apodization::PartialTukey(1 + u8::try_from(below(rng, 8)).unwrap_or(0)),
            _ => Apodization::PunchoutTukey(1 + u8::try_from(below(rng, 8)).unwrap_or(0)),
        }];
    }
    settings
}

fn random_signal(rng: &mut Rng, channels: u8, bits: u8, frames: usize) -> Vec<i32> {
    let kind = [
        Kind::Music,
        Kind::Noise,
        Kind::SilentFirst,
        Kind::Wasted,
        Kind::Correlated,
        Kind::Extremes,
    ][below(rng, 6)];
    signal(
        kind,
        usize::from(channels),
        u32::from(bits),
        frames,
        rng.next(),
    )
}

/// Encodes a random stream and requires the decoder to return it exactly.
fn encode_case(seed: u64) {
    let mut rng = Rng(seed | 1);
    let channels = 1 + u8::try_from(below(&mut rng, 8)).unwrap_or(0);
    let bits = [4u8, 8, 12, 16, 20, 24, 32][below(&mut rng, 7)];
    let rate = [8000u32, 44_100, 48_000, 96_000, 192_000, 700_000][below(&mut rng, 6)];
    let frames = below(&mut rng, 20_000);
    let samples = random_signal(&mut rng, channels, bits, frames);
    let settings = random_settings(&mut rng);
    let mut encoder = FlacEncoder::new(rate, channels, bits, settings).expect("encoder");
    let mut encoded = Vec::new();
    let mut at = 0;
    while at < samples.len() {
        let take = (1 + below(&mut rng, 9000)) * usize::from(channels);
        let end = (at + take).min(samples.len());
        encoded.extend(encoder.encode(&samples[at..end]).expect("encode"));
        at = end;
    }
    encoded.extend(encoder.finish().expect("finish"));
    let mut decoder = FlacDecoder::new(Some(encoder.stream_info()));
    let mut decoded = Vec::with_capacity(samples.len());
    for frame in &encoded {
        decoded.extend(decoder.decode(&frame.data).expect("decode").samples);
    }
    assert!(decoded == samples, "seed {seed:#x}: round trip differs");
    if !samples.is_empty() {
        assert_eq!(decoder.md5_check(), Md5Check::Match, "seed {seed:#x}");
    }
}

/// Decodes damaged frames; only a panic fails.
fn decode_case(seed: u64) {
    let mut rng = Rng(seed | 1);
    let channels = 1 + u8::try_from(below(&mut rng, 8)).unwrap_or(0);
    let bits = [4u8, 8, 16, 24, 32][below(&mut rng, 5)];
    let frames = 1 + below(&mut rng, 3000);
    let samples = random_signal(&mut rng, channels, bits, frames);
    let mut settings = random_settings(&mut rng);
    settings.block_size = settings.block_size.min(2048);
    let mut encoder = FlacEncoder::new(44_100, channels, bits, settings).expect("encoder");
    let mut frames = encoder.encode(&samples).expect("encode");
    frames.extend(encoder.finish().expect("finish"));
    let info = encoder.stream_info();
    let layout = match channels {
        1 => ChannelLayout::mono(),
        2 => ChannelLayout::stereo(),
        _ => cutlass_flac::layout(channels).expect("layout"),
    };
    let mut params = CodecParams::audio(CodecId::Flac, 44_100, layout);
    if below(&mut rng, 2) == 0 {
        params.extra = info.to_bytes().to_vec();
    }
    let mut decoder = FlacDecoder::open(&params).expect("decoder");
    let mut bare = FlacDecoder::new(None);
    for frame in &frames {
        let mut data = frame.data.clone();
        match below(&mut rng, 5) {
            0 => {
                for _ in 0..1 + below(&mut rng, 6) {
                    let at = below(&mut rng, data.len());
                    data[at] ^= 1 << below(&mut rng, 8);
                }
            }
            1 => data.truncate(below(&mut rng, data.len())),
            2 => {
                let at = below(&mut rng, data.len());
                for _ in 0..below(&mut rng, 32) {
                    data.insert(at, (rng.next() & 0xFF) as u8);
                }
            }
            3 => {
                let len = data.len();
                data = (0..len).map(|_| (rng.next() & 0xFF) as u8).collect();
                data[0] = 0xFF;
                if len > 1 {
                    data[1] = 0xF8;
                }
            }
            _ => {}
        }
        if data.len() > 2 && below(&mut rng, 2) == 0 {
            let end = data.len() - 2;
            let crc = crc::crc16(&data[..end]);
            data[end..].copy_from_slice(&crc.to_be_bytes());
        }
        let _ = bare.decode(&data);
        let _ = decoder.push(&Packet::audio(data, 0, 0, 44_100));
    }
    let _ = decoder.drain();
    let _ = decoder.md5_check();
}

fn campaign(name: &str, run: fn(u64), budget: Duration, seeds: u64) {
    let started = Instant::now();
    let mut seed = 0x00F1_AC5E_u64;
    let mut exercised = 0u64;
    while started.elapsed() < budget && exercised < seeds {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| run(seed))) {
            panic!("{name} seed {seed:#x} panicked: {payload:?}");
        }
        exercised += 1;
    }
    assert!(exercised > 0);
    eprintln!(
        "flac {name} campaign: {exercised} seeds in {:.1} s",
        started.elapsed().as_secs_f64()
    );
}

fn long_budget() -> Duration {
    Duration::from_secs(
        std::env::var("FLAC_FUZZ_SECS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(1800),
    )
}

#[test]
fn a_short_decoder_campaign() {
    campaign("decoder", decode_case, Duration::from_secs(20), 4000);
}

#[test]
fn a_short_encoder_campaign() {
    campaign("encoder", encode_case, Duration::from_secs(20), 600);
}

#[test]
#[ignore = "long decoder campaign; set FLAC_FUZZ_SECS, default 1800"]
fn decoder_campaign() {
    campaign("decoder", decode_case, long_budget(), u64::MAX);
}

#[test]
#[ignore = "long encoder campaign; set FLAC_FUZZ_SECS, default 1800"]
fn encoder_campaign() {
    campaign("encoder", encode_case, long_budget(), u64::MAX);
}
