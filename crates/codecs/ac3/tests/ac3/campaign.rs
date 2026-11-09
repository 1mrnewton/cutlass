//! Seeded mutation campaigns for the decoder and the encoder.
//!
//! cargo-fuzz is not used: `libfuzzer-sys` builds C through `cc`, which
//! `deny.toml` bans. These campaigns are an xorshift generator, so a
//! finding names the seed and the iteration that reproduce it.
//!
//! `bounded_mutations_do_not_panic` runs on every `cargo test`. The long
//! run is ignored:
//!
//! `AC3_FUZZ_SECS=1800 cargo test -p cutlass-ac3 --release --test ac3 -- --ignored campaign --nocapture`
//!
//! The default budget is 30 minutes, shared by the two campaigns.

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use cutlass_ac3::{Ac3Decoder, Ac3Encoder, ENCODER_DELAY, EncoderConfig, Format};
use cutlass_decode::{Channel, ChannelLayout, CodecError, CodecId, Decoder, Output, Packet, Pcm};
use cutlass_encode::{AudioInput, Encoder};

use crate::common::{FIXTURES, fixture, units};

const DECODER_SEED: u64 = 0xAC3D_EC0D;
const ENCODER_SEED: u64 = 0xAC3E_0C0D;
const BOUNDED_DECODER: u64 = 2_048;
const BOUNDED_ENCODER: u64 = 128;
const MAX_INPUT: usize = 16 * 1024;

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

    fn byte(&mut self) -> u8 {
        (self.next() & 0xff) as u8
    }
}

struct Counts {
    inputs: u64,
    errors: u64,
    accepted: u64,
}

fn budget() -> Duration {
    let seconds = std::env::var("AC3_FUZZ_SECS")
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(30 * 60);
    Duration::from_secs(seconds)
}

fn corpus() -> Vec<Vec<u8>> {
    let mut seeds = Vec::new();
    for name in FIXTURES {
        let data = fixture(name);
        seeds.extend(units(&data));
        seeds.push(data);
    }
    seeds.extend(encoded_seeds());
    seeds
}

fn encoded_seeds() -> Vec<Vec<u8>> {
    let mut seeds = Vec::new();
    let push = |seeds: &mut Vec<Vec<u8>>, packets: Vec<Packet>| {
        let mut bytes = Vec::new();
        for packet in packets {
            bytes.extend_from_slice(&packet.data);
        }
        seeds.push(bytes);
    };
    for codec in [CodecId::Ac3, CodecId::Eac3] {
        let layout = ChannelLayout::stereo();
        let frames = 1536 * 2;
        let samples = tone(frames, layout.len(), 48_000, 0.2);
        push(
            &mut seeds,
            encode(codec, 48_000, layout, 192_000, 6, &samples).expect("seed"),
        );
    }
    let mono = ChannelLayout::mono();
    let frames = 1536 * 2;
    push(
        &mut seeds,
        encode(
            CodecId::Eac3,
            48_000,
            mono.clone(),
            40_000,
            6,
            &tone(frames, 1, 48_000, 0.2),
        )
        .expect("spectral extension seed"),
    );
    push(
        &mut seeds,
        encode(
            CodecId::Eac3,
            48_000,
            mono,
            96_000,
            6,
            &tone(frames, 1, 48_000, 0.85),
        )
        .expect("hybrid transform seed"),
    );
    push(
        &mut seeds,
        encode(
            CodecId::Eac3,
            48_000,
            ChannelLayout::seven_one(),
            0,
            6,
            &tone(frames, 8, 48_000, 0.2),
        )
        .expect("7.1 seed"),
    );
    for blocks in [1usize, 2, 3] {
        push(
            &mut seeds,
            encode(
                CodecId::Eac3,
                48_000,
                ChannelLayout::stereo(),
                192_000,
                blocks,
                &tone(frames, 2, 48_000, 0.2),
            )
            .expect("short frame seed"),
        );
    }
    seeds
}

fn mutate(rng: &mut Rng, seeds: &[Vec<u8>]) -> Vec<u8> {
    let seed = &seeds[rng.below(seeds.len())];
    let mut out = match rng.below(8) {
        0 => {
            let mut bytes = seed.clone();
            if !bytes.is_empty() {
                let at = rng.below(bytes.len());
                bytes[at] ^= 1u8 << rng.below(8);
            }
            bytes
        }
        1 => {
            let mut bytes = seed.clone();
            for _ in 0..rng.below(8) + 1 {
                if bytes.is_empty() {
                    break;
                }
                let at = rng.below(bytes.len());
                bytes[at] = rng.byte();
            }
            bytes
        }
        2 => {
            let end = rng.below(seed.len().saturating_add(1));
            seed[..end].to_vec()
        }
        3 => {
            let mut bytes = seed.clone();
            for _ in 0..rng.below(64) + 1 {
                bytes.push(rng.byte());
            }
            bytes
        }
        4 => {
            let other = &seeds[rng.below(seeds.len())];
            let mut bytes = seed.clone();
            let take = other.len().min(64);
            bytes.extend_from_slice(&other[..take]);
            bytes
        }
        5 => (0..rng.below(4_096) + 1).map(|_| rng.byte()).collect(),
        6 => {
            let mut bytes = vec![0x0B, 0x77];
            for _ in 0..rng.below(200) + 4 {
                bytes.push(rng.byte());
            }
            bytes
        }
        _ => {
            let mut bytes = seed.clone();
            if bytes.len() > 8 {
                let at = rng.below(bytes.len() - 4);
                bytes[at] = 0;
                bytes[at + 1] = 0;
            }
            bytes
        }
    };
    out.truncate(MAX_INPUT);
    if out.is_empty() {
        out.push(rng.byte());
    }
    out
}

fn check_outputs(outputs: &[Output]) -> Result<(), &'static str> {
    for output in outputs {
        let Output::Audio(audio) = output else {
            continue;
        };
        let pcm = audio.pcm.as_f32().ok_or("not f32")?;
        let channels = audio.pcm.layout.len().max(1);
        if !pcm.len().is_multiple_of(channels) {
            return Err("uneven decoded frame");
        }
        if pcm.iter().any(|sample| !sample.is_finite()) {
            return Err("non-finite sample");
        }
    }
    Ok(())
}

/// `Ok(true)` when a unit decoded, `Ok(false)` when the decoder refused it.
fn push_bytes(codec: CodecId, data: &[u8]) -> Result<bool, &'static str> {
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let params = cutlass_decode::CodecParams::audio(codec, 0, ChannelLayout::stereo());
        let mut decoder = Ac3Decoder::open(&params).expect("open");
        let packet = Packet::audio(data.to_vec(), 0, 1, 48_000);
        match decoder.push(&packet) {
            Ok(outputs) => {
                check_outputs(&outputs)?;
                let drained = decoder.drain().expect("drain");
                check_outputs(&drained)?;
                decoder.reset();
                Ok(true)
            }
            Err(_) => Ok(false),
        }
    }));
    match caught {
        Ok(inner) => inner,
        Err(_) => Err("panic"),
    }
}

fn decoder_campaign(seed: u64, rounds: Option<u64>, limit: Option<Duration>) -> Counts {
    let seeds = corpus();
    let mut rng = Rng(seed);
    let mut counts = Counts {
        inputs: 0,
        errors: 0,
        accepted: 0,
    };
    let started = Instant::now();
    while rounds.is_none_or(|rounds| counts.inputs < rounds)
        && limit.is_none_or(|limit| started.elapsed() < limit)
    {
        let data = mutate(&mut rng, &seeds);
        let codec = if rng.below(2) == 0 {
            CodecId::Ac3
        } else {
            CodecId::Eac3
        };
        match push_bytes(codec, &data) {
            Ok(true) => counts.accepted += 1,
            Ok(false) => counts.errors += 1,
            Err(what) => panic!(
                "decoder {what} at input {} seed {seed:#x} codec {codec:?}",
                counts.inputs
            ),
        }
        counts.inputs += 1;
    }
    counts
}

fn layouts() -> Vec<ChannelLayout> {
    vec![
        ChannelLayout::mono(),
        ChannelLayout::stereo(),
        ChannelLayout::new(vec![
            Channel::FrontLeft,
            Channel::FrontRight,
            Channel::FrontCenter,
            Channel::Lfe,
            Channel::SideLeft,
            Channel::SideRight,
        ])
        .expect("5.1"),
        ChannelLayout::seven_one(),
    ]
}

fn tone(frames: usize, channels: usize, rate: u32, amplitude: f32) -> Vec<f32> {
    let mut samples = Vec::with_capacity(frames * channels);
    for index in 0..frames {
        let t = index as f32 / rate as f32;
        for channel in 0..channels {
            let freq = 180.0 + 90.0 * channel as f32;
            samples.push((t * freq * std::f32::consts::TAU).sin() * amplitude);
        }
    }
    samples
}

/// A rate the open check accepts, chosen so the new tools are reached.
/// Zero is the layout's default.
fn legal_bitrate(codec: CodecId, channels: usize, rng: &mut Rng) -> u32 {
    if codec != CodecId::Eac3 {
        return 0;
    }
    match (channels, rng.below(3)) {
        (1, 0) => 40_000,
        (1, 1) => 96_000,
        (2, 0) => 80_000,
        _ => 0,
    }
}

fn encode(
    codec: CodecId,
    rate: u32,
    layout: ChannelLayout,
    bitrate: u32,
    blocks: usize,
    samples: &[f32],
) -> Result<Vec<Packet>, CodecError> {
    let format = match codec {
        CodecId::Eac3 => Format::Eac3,
        _ => Format::Ac3,
    };
    let mut config = EncoderConfig::new(format, rate, layout.clone());
    config.bit_rate = bitrate;
    config.blocks = if format == Format::Eac3 { blocks } else { 6 };
    let mut encoder = Ac3Encoder::new(config)?;
    let pcm = Pcm::from_f32(rate, layout, samples.to_vec()).expect("pcm");
    let mut packets = encoder.push(&AudioInput { pcm })?;
    packets.extend(encoder.drain()?);
    Ok(packets)
}

fn encoder_round(rng: &mut Rng) -> Result<bool, &'static str> {
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let good = rng.below(2) == 0;
        if good {
            let codec = if rng.below(2) == 0 {
                CodecId::Ac3
            } else {
                CodecId::Eac3
            };
            let rate = [32_000, 44_100, 48_000][rng.below(3)];
            let layouts = layouts();
            let mut layout = layouts[rng.below(layouts.len())].clone();
            if layout.len() == 8 && codec == CodecId::Ac3 {
                layout = ChannelLayout::stereo();
            }
            let mut bitrate = legal_bitrate(codec, layout.len(), rng);
            // One, two, and three blocks are E-AC-3 only, and a short
            // frame at 40 kbit/s does not fit. 7.1 stays at six blocks.
            let blocks = if codec == CodecId::Eac3 && layout.len() <= 6 {
                [1, 2, 3, 6][rng.below(4)]
            } else {
                6
            };
            if blocks < 6 && bitrate < 96_000 {
                bitrate = 0;
            }
            let frames = 1_536 * (rng.below(2) + 1);
            let amplitude = if rng.below(5) == 0 { 0.85 } else { 0.2 };
            let samples = tone(frames, layout.len(), rate, amplitude);
            let packets = match encode(codec, rate, layout, bitrate, blocks, &samples) {
                Ok(packets) => packets,
                Err(_) => return Err("legal encode failed"),
            };
            if packets.is_empty() || packets[0].data.len() < 2 {
                return Err("empty encode");
            }
            if packets[0].data[0] != 0x0B || packets[0].data[1] != 0x77 {
                return Err("syncword");
            }
            let mut decoder = match Ac3Decoder::open(&cutlass_decode::CodecParams::audio(
                codec,
                rate,
                ChannelLayout::stereo(),
            )) {
                Ok(decoder) => decoder,
                Err(_) => return Err("decoder open failed"),
            };
            for packet in &packets {
                let audio = Packet::audio(packet.data.clone(), packet.pts, packet.duration, rate);
                let outputs = match decoder.push(&audio) {
                    Ok(outputs) => outputs,
                    Err(_) => return Err("our frame did not decode"),
                };
                check_outputs(&outputs)?;
            }
            let drained = match decoder.drain() {
                Ok(outputs) => outputs,
                Err(_) => return Err("drain failed"),
            };
            check_outputs(&drained)?;
            return Ok(true);
        }
        let layouts = layouts();
        let layout = layouts[rng.below(layouts.len())].clone();
        let codec = if rng.below(2) == 0 {
            CodecId::Ac3
        } else {
            CodecId::Eac3
        };
        let rate = [0, 8_000, 22_050, 32_000, 48_000, 96_000][rng.below(6)];
        let bitrate = [0, 1, 32_000, 192_000, u32::MAX][rng.below(5)];
        let mut params = cutlass_encode::CodecParams::audio(codec, rate, layout.clone());
        params.bitrate = bitrate;
        let Ok(mut encoder) = Ac3Encoder::open(&params) else {
            return Ok(false);
        };
        let channels = layout.len();
        let frames = rng.below(3_000);
        let mut samples = Vec::with_capacity(frames.saturating_mul(channels));
        for _ in 0..frames * channels {
            let sample = match rng.below(5) {
                0 => f32::NAN,
                1 => f32::INFINITY,
                2 => 2.0,
                3 => -2.0,
                _ => (rng.byte() as f32 / 255.0) * 2.0 - 1.0,
            };
            samples.push(sample);
        }
        if rng.below(4) == 0 && channels > 1 {
            samples.pop();
        }
        let Ok(pcm) = Pcm::from_f32(rate, layout, samples) else {
            return Ok(false);
        };
        let pushed = encoder.push(&AudioInput { pcm });
        if let Ok(packets) = &pushed {
            for packet in packets {
                if packet.data.len() >= 2 && (packet.data[0] != 0x0B || packet.data[1] != 0x77) {
                    return Err("syncword");
                }
            }
        }
        let _ = encoder.drain();
        let _ = encoder.drain();
        let _ = encoder.push(&AudioInput {
            pcm: Pcm::from_f32(rate, ChannelLayout::mono(), vec![0.0]).expect("pcm"),
        });
        encoder.reset();
        Ok(false)
    }));
    match caught {
        Ok(inner) => inner,
        Err(_) => Err("panic"),
    }
}

fn encoder_campaign(seed: u64, rounds: Option<u64>, limit: Option<Duration>) -> Counts {
    let mut rng = Rng(seed);
    let mut counts = Counts {
        inputs: 0,
        errors: 0,
        accepted: 0,
    };
    let started = Instant::now();
    while rounds.is_none_or(|rounds| counts.inputs < rounds)
        && limit.is_none_or(|limit| started.elapsed() < limit)
    {
        match encoder_round(&mut rng) {
            Ok(true) => counts.accepted += 1,
            Ok(false) => counts.errors += 1,
            Err(what) => panic!("encoder {what} at round {} seed {seed:#x}", counts.inputs),
        }
        counts.inputs += 1;
    }
    counts
}

#[test]
fn bounded_mutations_do_not_panic() {
    let decoded = decoder_campaign(DECODER_SEED, Some(BOUNDED_DECODER), None);
    assert!(decoded.inputs == BOUNDED_DECODER);
    assert!(decoded.errors > 0, "mutations never failed");
    let encoded = encoder_campaign(ENCODER_SEED, Some(BOUNDED_ENCODER), None);
    assert!(encoded.inputs == BOUNDED_ENCODER);
    assert!(encoded.accepted > 0, "no legal encode");
    assert!(encoded.errors > 0, "hostile input was never refused");
}

#[test]
fn cancel_stops_before_the_next_frame_and_reset_encodes_again() {
    let layout = ChannelLayout::stereo();
    let mut params = cutlass_encode::CodecParams::audio(CodecId::Ac3, 48_000, layout.clone());
    params.bitrate = 192_000;
    let mut encoder = Ac3Encoder::open(&params).expect("encoder");
    encoder.cancel_flag().store(true, Ordering::Relaxed);
    let samples = tone(48_000, layout.len(), 48_000, 0.2);
    let pcm = Pcm::from_f32(48_000, layout.clone(), samples).expect("pcm");
    let started = Instant::now();
    let error = encoder.push(&AudioInput { pcm }).expect_err("cancel");
    let elapsed = started.elapsed();
    assert!(matches!(error, CodecError::Cancelled), "{error:?}");
    assert!(
        elapsed < Duration::from_millis(200),
        "cancel took {elapsed:?}"
    );
    encoder.cancel_flag().store(false, Ordering::Relaxed);
    encoder.reset();
    let again = tone(1536, 2, 48_000, 0.2);
    let pcm = Pcm::from_f32(48_000, layout, again).expect("pcm");
    let packets = encoder.push(&AudioInput { pcm }).expect("after reset");
    assert!(!packets.is_empty());
    assert!(packets[0].duration > i64::from(ENCODER_DELAY));
}

#[test]
#[ignore = "thirty minute campaign; AC3_FUZZ_SECS overrides the seconds"]
fn decoder_and_encoder_campaigns() {
    let limit = budget();
    let decoded = std::thread::scope(|scope| {
        let decoder = scope.spawn(move || decoder_campaign(DECODER_SEED, None, Some(limit)));
        let encoder = scope.spawn(move || encoder_campaign(ENCODER_SEED, None, Some(limit)));
        (
            decoder.join().expect("decoder"),
            encoder.join().expect("encoder"),
        )
    });
    let (decoder, encoder) = decoded;
    eprintln!(
        "ac3 decoder campaign: seed {DECODER_SEED:#x} seconds {} inputs {} errors {} accepted {} findings 0",
        limit.as_secs(),
        decoder.inputs,
        decoder.errors,
        decoder.accepted
    );
    eprintln!(
        "ac3 encoder campaign: seed {ENCODER_SEED:#x} seconds {} rounds {} refused {} encoded {} findings 0",
        limit.as_secs(),
        encoder.inputs,
        encoder.errors,
        encoder.accepted
    );
}
