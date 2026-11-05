//! In-process decoder and encoder campaigns.
//!
//! This is not libFuzzer (cargo-fuzz is not installed here): a seeded
//! generator mutates real and encoder-made streams and drives the decoder
//! and encoder through their whole lifecycle, under `catch_unwind`. A short
//! campaign runs with every test run; the long one is
//! `MP3_FUZZ_SECS=600 cargo test --release -p cutlass-mp3 --test mp3 -- --ignored --nocapture campaign`.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use cutlass_decode::{Decoder, Output, Packet, Pcm, SideData};
use cutlass_encode::{AudioInput, Encoder};
use cutlass_mp3::{DECODER_DELAY, Mp3Decoder, Mp3Encoder, Mp3EncoderConfig, Mp3Rate};

use crate::common::{demux, fixture};
use crate::encoder::{decode, layout, music};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

/// A stream to mutate: its frames and sample rate.
struct Source {
    frames: Vec<Vec<u8>>,
    rate: u32,
}

const RATES: [u32; 9] = [
    8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000,
];

/// The fixtures, and encoder output at every rate in both layouts, with
/// reservoir use, short blocks and mid/side.
fn sources() -> &'static [Source] {
    static SOURCES: OnceLock<Vec<Source>> = OnceLock::new();
    SOURCES.get_or_init(|| {
        let mut out = Vec::new();
        for name in [
            "sine_44100_j128_crc.mp3",
            "clicks_48000_v2.mp3",
            "sweep_22050_mono64.mp3",
            "sine_8000_mono16.mp3",
            "stereo_32000_ms96.mp3",
            "sine_44100_free400.mp3",
        ] {
            let file = demux(&fixture(name));
            out.push(Source {
                frames: file.packets,
                rate: file.rate,
            });
        }
        for (index, rate) in RATES.into_iter().enumerate() {
            let channels = 1 + index % 2;
            let config = Mp3EncoderConfig {
                sample_rate: rate,
                channels,
                rate: Mp3Rate::Variable((index % 10) as u8),
            };
            let pcm = music(rate, channels, 0.5);
            let packets = crate::encoder::encode(config, &pcm, 1000);
            out.push(Source {
                frames: packets.into_iter().map(|packet| packet.data).collect(),
                rate,
            });
        }
        out
    })
}

/// Check the samples are finite; return how many audible frames (per
/// channel, outside the skips) the outputs hold.
fn check_outputs(outputs: &[Output]) -> usize {
    let mut samples = 0;
    for output in outputs {
        if let Output::Audio(audio) = output {
            let pcm = audio.pcm.as_f32().unwrap_or(&[]);
            assert!(pcm.iter().all(|s| s.is_finite()), "non-finite output");
            assert!(audio.skip_leading + audio.skip_trailing <= audio.pcm.samples);
            samples += (audio.pcm.samples - audio.skip_leading - audio.skip_trailing) as usize;
        }
    }
    samples
}

/// Mutate one run of consecutive frames and feed it to one decoder.
fn exercise_decoder(seed: u64) {
    let mut rng = Rng(seed | 1);
    let all = sources();
    let source = &all[rng.below(all.len())];
    let mut decoder = Mp3Decoder::open(&cutlass_decode::CodecParams::audio(
        cutlass_decode::CodecId::Mp3,
        if rng.below(2) == 0 { 0 } else { source.rate },
        layout(1 + rng.below(2)),
    ))
    .expect("open");
    let start = rng.below(source.frames.len());
    let count = 1 + rng.below(12);
    let mut pts = 0i64;
    let (mut accepted, mut produced) = (0usize, 0usize);
    for step in 0..count {
        let index = match rng.below(8) {
            0 => rng.below(source.frames.len()),
            _ => (start + step) % source.frames.len(),
        };
        let mut data = source.frames[index].clone();
        match rng.below(7) {
            0 => {}
            1 => {
                for _ in 0..1 + rng.below(16) {
                    let at = 4 + rng.below(data.len().saturating_sub(4));
                    if at < data.len() {
                        data[at] ^= 1 << rng.below(8);
                    }
                }
            }
            2 => data.truncate(rng.below(data.len()) + 1),
            3 => {
                // Header fields, keeping the sync word.
                if data.len() >= 4 {
                    data[1] = 0xE0 | (rng.next() as u8 & 0x1F);
                    data[2] = rng.next() as u8;
                    data[3] = rng.next() as u8;
                }
            }
            4 => {
                // Side information and main_data_begin.
                let end = data.len().min(4 + 32);
                for byte in data.iter_mut().take(end).skip(4) {
                    if rng.below(3) == 0 {
                        *byte = rng.next() as u8;
                    }
                }
            }
            5 => {
                let other = &source.frames[rng.below(source.frames.len())];
                let cut = rng.below(data.len());
                data.truncate(cut);
                data.extend_from_slice(&other[rng.below(other.len())..]);
            }
            _ => {
                let len = rng.below(2000);
                data = (0..len).map(|_| rng.next() as u8).collect();
                if data.len() >= 2 {
                    data[0] = 0xFF;
                    data[1] |= 0xE0;
                }
            }
        }
        if data.is_empty() {
            continue;
        }
        let mut packet = Packet::audio(data, pts, 1152, source.rate);
        if rng.below(6) == 0 {
            packet.side.push(SideData::Gapless {
                encoder_delay: rng.next() as u32 % 5000,
                trailing_padding: rng.next() as u32 % 5000,
                sample_rate: if rng.below(2) == 0 { 0 } else { source.rate },
            });
        }
        if rng.below(10) == 0 {
            packet.timescale = rng.next() as u32 % 100_000;
            packet.pts = rng.next() as i64;
        }
        pts = pts.wrapping_add(1152);
        if let Ok(outputs) = decoder.push(&packet) {
            accepted += 1152;
            produced += check_outputs(&outputs);
        }
        match rng.below(20) {
            0 => decoder.reset(),
            1 => {
                // Drain ends a stream; reset starts the next one.
                if let Ok(outputs) = decoder.drain() {
                    produced += check_outputs(&outputs);
                }
                accepted += DECODER_DELAY as usize;
                decoder.reset();
            }
            _ => {}
        }
        // Nothing is invented: at most a frame of audible samples per
        // accepted push, and the decoder delay's tail at a drain (all of it
        // audible when the gapless trim arrives after the first frame).
        assert!(produced <= accepted, "{produced} > {accepted}");
    }
    if let Ok(outputs) = decoder.drain() {
        produced += check_outputs(&outputs);
    }
    accepted += DECODER_DELAY as usize;
    assert!(produced <= accepted, "{produced} > {accepted}");
}

/// Random settings, signals, chunking and lifecycle calls; every packet
/// must decode, and a whole stream must decode to its accepted length.
fn exercise_encoder(seed: u64) {
    let mut rng = Rng(seed | 1);
    let sample_rate = RATES[rng.below(RATES.len())];
    let channels = 1 + rng.below(2);
    let lsf = sample_rate < 32_000;
    let legal: &[u32] = if lsf {
        &[8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160]
    } else {
        &[
            32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
        ]
    };
    let rate = if rng.below(2) == 0 {
        Mp3Rate::Constant(legal[rng.below(legal.len())])
    } else {
        Mp3Rate::Variable(rng.below(10) as u8)
    };
    let config = Mp3EncoderConfig {
        sample_rate,
        channels,
        rate,
    };
    let Ok(mut encoder) = Mp3Encoder::new(config) else {
        return;
    };
    let frames = rng.below(sample_rate as usize / 3);
    let kind = rng.below(6);
    let mut level = 0.5f32;
    let mut pcm = Vec::with_capacity(frames * channels);
    for i in 0..frames {
        for ch in 0..channels {
            let noise = (rng.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0;
            let value = match kind {
                0 => 0.0,
                1 => noise,
                2 => {
                    if i % (1 + rng.below(3000)) == 0 {
                        1.0
                    } else {
                        0.0
                    }
                }
                3 => {
                    if (i / 37 + ch) % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                }
                4 => {
                    if rng.below(500) == 0 {
                        level = (rng.next() >> 40) as f32 / (1u64 << 24) as f32;
                    }
                    level * noise
                }
                _ => (i as f32 * (0.01 + ch as f32 * 0.37)).sin() * 0.9,
            };
            pcm.push(value);
        }
    }
    let mut packets = Vec::new();
    let mut at = 0;
    while at < pcm.len() {
        let take = (1 + rng.below(3000)) * channels;
        let end = (at + take).min(pcm.len());
        let chunk = pcm[at..end].to_vec();
        at = end;
        if rng.below(30) == 0 {
            // A refused push changes nothing.
            let mut bad = chunk.clone();
            let slot = rng.below(bad.len().max(1));
            if let Some(sample) = bad.get_mut(slot) {
                *sample = if rng.below(2) == 0 { f32::NAN } else { 3.0 };
            }
            let input = AudioInput {
                pcm: Pcm::from_f32(sample_rate, layout(channels), bad).expect("pcm"),
            };
            assert!(encoder.push(&input).is_err(), "bad sample accepted");
        }
        let input = AudioInput {
            pcm: Pcm::from_f32(sample_rate, layout(channels), chunk).expect("pcm"),
        };
        packets.extend(encoder.push(&input).expect("push"));
    }
    packets.extend(encoder.drain().expect("drain"));
    assert!(encoder.drain().expect("second drain").is_empty());
    for packet in &packets {
        assert!(packet.data.len() >= 4, "short packet");
    }
    let decoded = decode(config, &packets);
    assert_eq!(decoded.len(), frames * channels, "seed {seed:#x}");
    assert!(decoded.iter().all(|s| s.is_finite()));
    if rng.below(4) == 0 {
        encoder.reset();
        let input = AudioInput {
            pcm: Pcm::from_f32(sample_rate, layout(channels), vec![0.0; 1152 * channels])
                .expect("pcm"),
        };
        let mut again = encoder.push(&input).expect("push after reset");
        again.extend(encoder.drain().expect("drain after reset"));
        assert!(!again.is_empty());
    }
}

/// Run both campaigns for `budget`; return the seeds each exercised.
fn campaign(budget: Duration) -> (u64, u64) {
    let started = Instant::now();
    let mut seed = 0x00C0_FFEE_u64;
    let (mut decoded, mut encoded) = (0u64, 0u64);
    while started.elapsed() < budget {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let result = catch_unwind(AssertUnwindSafe(|| exercise_decoder(seed)));
        assert!(result.is_ok(), "decoder seed {seed:#x} panicked");
        decoded += 1;
        // The encoder is far slower per seed; one in sixteen keeps both busy.
        if decoded % 16 == 0 {
            let result = catch_unwind(AssertUnwindSafe(|| exercise_encoder(seed)));
            assert!(result.is_ok(), "encoder seed {seed:#x} panicked");
            encoded += 1;
        }
    }
    (decoded, encoded)
}

#[test]
fn a_short_campaign_never_panics() {
    let (decoded, encoded) = campaign(Duration::from_secs(2));
    assert!(decoded > 0 && encoded > 0, "{decoded} {encoded}");
}

#[test]
#[ignore = "long campaign; set MP3_FUZZ_SECS, default 600"]
fn campaign_for_the_set_time() {
    let secs = std::env::var("MP3_FUZZ_SECS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(600u64);
    let (decoded, encoded) = campaign(Duration::from_secs(secs));
    eprintln!("mp3 fuzz: {decoded} decoder seeds, {encoded} encoder seeds in {secs} s");
}
