//! Long seeded campaigns, ignored by default. cargo-fuzz is not usable in
//! this workspace (libfuzzer-sys builds C, which `deny.toml` bans), so
//! these stand in for it:
//!
//! `ALAC_FUZZ_SECS=1800 cargo test -p cutlass-alac --release -- --ignored campaign`
//!
//! Each runs for `ALAC_FUZZ_SECS` seconds (default 30 minutes) and prints
//! how many inputs it tried. A finding is a panic, which names the seed
//! state and the input.

use cutlass_decode::{ChannelLayout, Packet};

use super::{encode_all, synth};
use crate::{AlacConfig, AlacDecoder, Cookie, EncodeMode, EncoderConfig, default_layout};

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

fn budget() -> std::time::Duration {
    let seconds = std::env::var("ALAC_FUZZ_SECS")
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(30 * 60);
    std::time::Duration::from_secs(seconds)
}

/// One real stream per shape, as mutation seeds.
fn seeds() -> Vec<(Vec<u8>, Vec<Packet>, usize)> {
    let shapes = [
        (1u8, 16u8, EncodeMode::Default, 4096u32),
        (2, 16, EncodeMode::Default, 4096),
        (2, 16, EncodeMode::Fast, 1152),
        (2, 20, EncodeMode::Default, 1001),
        (2, 24, EncodeMode::Default, 4096),
        (2, 32, EncodeMode::Fast, 512),
        (3, 24, EncodeMode::Default, 777),
        (6, 16, EncodeMode::Default, 4096),
        (8, 32, EncodeMode::Default, 256),
    ];
    shapes
        .into_iter()
        .map(|(channels, depth, mode, frame_length)| {
            let samples = synth(
                usize::from(channels),
                frame_length as usize * 3 + 17,
                u32::from(depth),
                u32::from(channels) * 31 + u32::from(depth),
            );
            let config = EncoderConfig {
                frame_length,
                mode,
                ..EncoderConfig::new(44_100, channels, depth)
            };
            let (cookie, packets) = encode_all(config, &default_layout(channels), &samples);
            (
                cookie,
                packets,
                usize::from(channels) * frame_length as usize,
            )
        })
        .collect()
}

#[test]
#[ignore = "alac decoder fuzz; runs ALAC_FUZZ_SECS, default 30 minutes"]
fn decoder_campaign() {
    let seeds = seeds();
    let mut decoders: Vec<AlacDecoder> = seeds
        .iter()
        .map(|(cookie, _, _)| AlacDecoder::new(cookie).expect("decoder"))
        .collect();
    let mut rng = Rng(0x414c_4143_4655_5a5a);
    let started = std::time::Instant::now();
    let (mut inputs, mut errors) = (0u64, 0u64);
    while started.elapsed() < budget() {
        let which = rng.below(seeds.len());
        let (_, packets, limit) = &seeds[which];
        let mut data = packets[rng.below(packets.len())].data.clone();
        let state = rng.0;
        match rng.below(5) {
            0 => {
                for _ in 0..1 + rng.below(8) {
                    let at = rng.below(data.len());
                    data[at] ^= 1 << rng.below(8);
                }
            }
            1 => data.truncate(rng.below(data.len())),
            2 => {
                let from = rng.below(data.len());
                for byte in &mut data[from..] {
                    *byte = rng.next() as u8;
                }
            }
            3 => {
                let at = rng.below(data.len());
                let extra = rng.below(64);
                data.splice(at..at, (0..extra).map(|_| rng.next() as u8));
            }
            _ => {
                data = (0..1 + rng.below(400)).map(|_| rng.next() as u8).collect();
            }
        }
        if data.is_empty() {
            continue;
        }
        inputs += 1;
        match decoders[which].decode_frame(&data) {
            Ok(frame) => assert!(
                frame.samples.len() <= *limit,
                "finding: seed {which}, state {state:#x}, {} samples",
                frame.samples.len()
            ),
            Err(_) => errors += 1,
        }
    }
    // The decoders are still whole.
    for ((_, packets, _), decoder) in seeds.iter().zip(&mut decoders) {
        decoder.decode_frame(&packets[0].data).expect("seed frame");
    }
    eprintln!("alac decoder campaign: {inputs} inputs, {errors} errors, no panics");
}

#[test]
#[ignore = "alac cookie fuzz; runs ALAC_FUZZ_SECS, default 30 minutes"]
fn cookie_campaign() {
    let base = Cookie::new(AlacConfig::new(48_000, 6, 24, 4096)).to_bytes();
    let mut rng = Rng(0x434f_4f4b_4945_5a5a);
    let started = std::time::Instant::now();
    let (mut inputs, mut accepted) = (0u64, 0u64);
    while started.elapsed() < budget() {
        let mut bytes = base.clone();
        if rng.below(2) == 0 {
            bytes.splice(0..0, [0, 0, 0, 0]);
        }
        for _ in 0..1 + rng.below(6) {
            let at = rng.below(bytes.len());
            bytes[at] = rng.next() as u8;
        }
        bytes.truncate(bytes.len() - rng.below(bytes.len() / 4));
        inputs += 1;
        let Ok(cookie) = Cookie::parse(&bytes) else {
            continue;
        };
        accepted += 1;
        // Whatever parses must decode random frames without a panic.
        if let Ok(mut decoder) = AlacDecoder::new(&cookie.to_bytes()) {
            let data: Vec<u8> = (0..1 + rng.below(200)).map(|_| rng.next() as u8).collect();
            let _ = decoder.decode_frame(&data);
        }
    }
    eprintln!("alac cookie campaign: {inputs} inputs, {accepted} accepted, no panics");
}

/// Random shapes and signals through the encoder and back: every one must
/// be lossless.
#[test]
#[ignore = "alac round-trip campaign; runs ALAC_FUZZ_SECS, default 30 minutes"]
fn round_trip_campaign() {
    let mut rng = Rng(0x524f_554e_4454_5250);
    let started = std::time::Instant::now();
    let mut streams = 0u64;
    while started.elapsed() < budget() {
        let channels = 1 + rng.below(8) as u8;
        let depth = [16u8, 20, 24, 32][rng.below(4)];
        let frame_length = 1 + rng.below(5000) as u32;
        let mode = if rng.below(2) == 0 {
            EncodeMode::Default
        } else {
            EncodeMode::Fast
        };
        let frames = rng.below(3 * frame_length as usize + 50);
        let peak = (1i64 << (depth - 1)) - 1;
        let samples: Vec<i32> = match rng.below(3) {
            0 => synth(
                usize::from(channels),
                frames,
                u32::from(depth),
                rng.next() as u32,
            ),
            // White noise at a random level, the worst case for prediction.
            1 => {
                let level = 1 + rng.below(usize::from(depth) - 1);
                (0..frames * usize::from(channels))
                    .map(|_| ((rng.next() as i64) >> (64 - level)) as i32)
                    .collect()
            }
            // Full-scale steps and silence.
            _ => (0..frames * usize::from(channels))
                .map(|i| match (i / 97) % 3 {
                    0 => peak as i32,
                    1 => (-peak - 1) as i32,
                    _ => 0,
                })
                .collect(),
        };
        let config = EncoderConfig {
            frame_length,
            mode,
            ..EncoderConfig::new(44_100, channels, depth)
        };
        let layout = if channels <= 2 {
            if channels == 1 {
                ChannelLayout::mono()
            } else {
                ChannelLayout::stereo()
            }
        } else {
            default_layout(channels)
        };
        if samples.is_empty() {
            continue;
        }
        let (cookie, packets) = encode_all(config, &layout, &samples);
        let mut decoder = AlacDecoder::new(&cookie).expect("decoder");
        let decoded: Vec<i32> = packets
            .iter()
            .flat_map(|packet| decoder.decode_frame(&packet.data).expect("frame").samples)
            .collect();
        assert!(
            decoded == samples,
            "finding: {channels} ch, {depth} bit, {frame_length} frame, {mode:?}"
        );
        streams += 1;
    }
    eprintln!("alac round-trip campaign: {streams} streams, all lossless");
}
