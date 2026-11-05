//! Encoder round trips through this crate's decoder.

use cutlass_decode::{ChannelLayout, Decoder, Packet, Pcm, SideData};
use cutlass_encode::{AudioInput, Encoder};
use cutlass_mp3::{ENCODER_DELAY, Mp3Encoder, Mp3EncoderConfig, Mp3Rate};

use crate::common::{audible, open_decoder};

pub fn layout(channels: usize) -> ChannelLayout {
    if channels == 1 {
        ChannelLayout::mono()
    } else {
        ChannelLayout::stereo()
    }
}

/// Encode interleaved PCM in pushes of `chunk` frames.
pub fn encode(config: Mp3EncoderConfig, pcm: &[f32], chunk: usize) -> Vec<Packet> {
    let mut encoder = Mp3Encoder::new(config).expect("encoder");
    let mut packets = Vec::new();
    for part in pcm.chunks(chunk * config.channels) {
        let input = AudioInput {
            pcm: Pcm::from_f32(config.sample_rate, layout(config.channels), part.to_vec())
                .expect("pcm"),
        };
        packets.extend(encoder.push(&input).expect("push"));
    }
    packets.extend(encoder.drain().expect("drain"));
    packets
}

/// Decode packets with the side data they carry.
pub fn decode(config: Mp3EncoderConfig, packets: &[Packet]) -> Vec<f32> {
    let mut decoder = open_decoder(config.sample_rate, config.channels);
    let mut pcm = Vec::new();
    for packet in packets {
        pcm.extend(audible(decoder.push(packet).expect("push")));
    }
    pcm.extend(audible(decoder.drain().expect("drain")));
    pcm
}

/// A test signal: two tones, noise, and clicks every quarter second.
pub fn music(rate: u32, channels: usize, seconds: f32) -> Vec<f32> {
    let frames = (rate as f32 * seconds) as usize;
    let mut seed = 0x1234_5678u32;
    let mut out = Vec::with_capacity(frames * channels);
    for i in 0..frames {
        let t = i as f32 / rate as f32;
        for ch in 0..channels {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5;
            let click = if i % (rate as usize / 4) < 64 {
                0.5 * (1.0 - (i % (rate as usize / 4)) as f32 / 64.0)
            } else {
                0.0
            };
            let tone = 0.25 * (2.0 * std::f32::consts::PI * (440.0 + 110.0 * ch as f32) * t).sin()
                + 0.1 * (2.0 * std::f32::consts::PI * 3_000.0 * t).sin();
            out.push(tone + 0.02 * noise + click);
        }
    }
    out
}

/// Signal-to-noise ratio of `decoded` against `input`, in dB.
pub fn snr(input: &[f32], decoded: &[f32]) -> f64 {
    let mut signal = 0f64;
    let mut noise = 0f64;
    for (a, b) in input.iter().zip(decoded) {
        signal += f64::from(*a) * f64::from(*a);
        noise += (f64::from(*a) - f64::from(*b)).powi(2);
    }
    10.0 * (signal / noise.max(1e-30)).log10()
}

fn config(sample_rate: u32, channels: usize, rate: Mp3Rate) -> Mp3EncoderConfig {
    Mp3EncoderConfig {
        sample_rate,
        channels,
        rate,
    }
}

#[test]
fn a_constant_rate_stream_decodes_to_the_input_length_and_shape() {
    let config = config(44_100, 2, Mp3Rate::Constant(128));
    let input = music(44_100, 2, 2.0);
    let packets = encode(config, &input, 1000);
    let decoded = decode(config, &packets);
    assert_eq!(decoded.len(), input.len());
    let snr = snr(&input, &decoded);
    eprintln!("cbr128 snr {snr:.1} dB, {} packets", packets.len());
    assert!(snr > 15.0, "{snr}");
    for packet in &packets {
        // 128 kbit/s at 44.1 kHz is 417.96 bytes a frame: 417 or 418.
        assert!(
            matches!(packet.data.len(), 417 | 418),
            "{}",
            packet.data.len()
        );
        assert!(packet.side.iter().any(|side| matches!(
            side,
            SideData::Gapless { encoder_delay, .. } if *encoder_delay == ENCODER_DELAY
        )));
    }
}

#[test]
fn every_sample_rate_and_layout_round_trips() {
    for rate in [
        8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000,
    ] {
        for channels in [1, 2] {
            for mode in [Mp3Rate::Constant(64), Mp3Rate::Variable(4)] {
                let config = config(rate, channels, mode);
                let input = music(rate, channels, 1.0);
                let packets = encode(config, &input, 777);
                let decoded = decode(config, &packets);
                assert_eq!(decoded.len(), input.len(), "{config:?}");
                let snr = snr(&input, &decoded);
                eprintln!("{rate} {channels} {mode:?}: {snr:.1} dB");
                assert!(snr > 8.0, "{config:?}: {snr}");
            }
        }
    }
}
