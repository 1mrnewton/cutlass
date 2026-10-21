//! A sine encoded as AAC-LC comes back with the same pitch.

#![cfg(target_vendor = "apple")]

use cutlass_audiotoolbox::{Decoder, Encoder};
use cutlass_core::{AudioCodec, AudioSamples, AudioStreamConfig};

#[test]
fn a_sine_survives_aac() {
    let rate = 48_000;
    let channels = 2u16;
    let frames = 48_000usize;
    let mut planar = vec![0f32; frames * usize::from(channels)];
    for frame in 0..frames {
        let t = frame as f32 / rate as f32;
        let sample = (t * 440.0 * std::f32::consts::TAU).sin() * 0.5;
        planar[frame] = sample;
        planar[frames + frame] = sample;
    }
    let source = AudioSamples::new(rate, channels, planar).expect("source");
    let mut encoder = Encoder::new(rate, channels).expect("encoder");
    let cookie = encoder.config_record().to_vec();
    assert_eq!(cookie, [0x11, 0x90]);
    let mut packets = encoder.encode(&source).expect("encode");
    packets.extend(encoder.finish().expect("finish"));
    assert!(packets.len() > 40, "got {} packets", packets.len());

    let mut decoder = Decoder::new(&AudioStreamConfig {
        codec: AudioCodec::Aac,
        sample_rate: rate,
        channels,
        config_record: cookie,
    })
    .expect("decoder");
    let mut decoded = vec![Vec::new(); usize::from(channels)];
    for packet in &packets {
        let frame = decoder.decode(&packet.data).expect("decode");
        for channel in 0..channels {
            decoded[usize::from(channel)].extend_from_slice(frame.plane(channel).expect("plane"));
        }
    }
    let tail = decoder.flush().expect("flush");
    for channel in 0..channels {
        decoded[usize::from(channel)].extend_from_slice(tail.plane(channel).expect("tail plane"));
    }
    assert!(
        decoded[0].len() > 40_000,
        "decoded {} samples from {} packets",
        decoded[0].len(),
        packets.len()
    );
    let (lag, error) = best_lag(&source_plane(&source, 0), &decoded[0]);
    assert!(error < 0.03, "left channel mean error {error} at lag {lag}");
    let (_, right) = best_lag(&source_plane(&source, 1), &decoded[1]);
    assert!(right < 0.03, "right channel mean error {right}");
}

#[test]
fn a_non_aac_stream_is_rejected() {
    match Decoder::new(&AudioStreamConfig {
        codec: AudioCodec::Mp3,
        sample_rate: 48_000,
        channels: 2,
        config_record: vec![1],
    }) {
        Ok(_) => panic!("mp3 should be rejected"),
        Err(error) => assert!(error.is_unsupported()),
    }
}

fn source_plane(samples: &AudioSamples, channel: u16) -> Vec<f32> {
    samples.plane(channel).expect("source plane").to_vec()
}

fn best_lag(source: &[f32], decoded: &[f32]) -> (usize, f32) {
    let window = 8_000.min(source.len());
    let max_lag = decoded.len().saturating_sub(window).min(8_192);
    let mut best = (0, f32::MAX);
    for lag in 0..=max_lag {
        let mut error = 0f32;
        for index in 0..window {
            error += (decoded[lag + index] - source[index]).abs();
        }
        let mean = error / window as f32;
        if mean < best.1 {
            best = (lag, mean);
        }
    }
    best
}
