//! Encoder round trips through this crate's CELT decoder.

use super::{CeltConfig, CeltEncoder, RateMode};
use crate::celt::{CeltDecoder, CeltFrame};
use crate::range::RangeDecoder;

fn lcg(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *seed
}

/// Music-like test signal: a few partials with vibrato, a noise floor,
/// and a click every 400 ms.
fn signal(len: usize, channels: usize, seed: &mut u32) -> Vec<f32> {
    let mut out = Vec::with_capacity(len * channels);
    for i in 0..len {
        let t = i as f32 / 48_000.0;
        let vib = 1.0 + 0.003 * (t * 5.0 * std::f32::consts::TAU).sin();
        let mut v = 0.0;
        for (k, amp) in [(1.0f32, 0.3f32), (2.0, 0.15), (3.0, 0.08), (5.0, 0.04)] {
            v += amp * (t * 220.0 * k * vib * std::f32::consts::TAU).sin();
        }
        if i % 19_200 < 48 {
            v += 0.4 * (1.0 - (i % 19_200) as f32 / 48.0);
        }
        for ch in 0..channels {
            let noise = ((lcg(seed) >> 9) as f32 / 8_388_608.0 - 1.0) * 0.01;
            let pan = if ch == 0 { 1.0 } else { 0.7 };
            out.push(v * pan + noise);
        }
    }
    out
}

struct Trip {
    sizes: Vec<usize>,
    /// Decoded output, delayed by the CELT overlap.
    decoded: Vec<f32>,
}

/// Encodes `pcm` frame by frame and decodes each frame, asserting the
/// decoder ends every frame on the encoder's final range.
fn round_trip(cfg: CeltConfig, pcm: &[f32]) -> Trip {
    let c = cfg.channels;
    let mut enc = CeltEncoder::new(cfg).expect("encoder");
    let mut dec = CeltDecoder::new(c, 48_000).expect("decoder");
    let n = enc.frame_size();
    let mut out = vec![0f32; n * c];
    let mut trip = Trip {
        sizes: Vec::new(),
        decoded: Vec::new(),
    };
    for (index, frame) in pcm.chunks_exact(n * c).enumerate() {
        let bytes = enc.encode(frame).expect("encode");
        trip.sizes.push(bytes.len());
        let mut rd = RangeDecoder::new(&bytes);
        let f = CeltFrame {
            rd: Some(&mut rd),
            frame_size: n,
            stream_channels: c,
            start_band: 0,
            end_band: cfg.end_band,
        };
        dec.decode(f, &mut out).expect("decode");
        assert_eq!(
            rd.rng(),
            enc.rng(),
            "final range of frame {index} ({cfg:?})"
        );
        trip.decoded.extend_from_slice(&out);
    }
    trip
}

/// SNR in dB of `decoded` against `pcm` after the 120-sample delay,
/// skipping the first 100 ms.
fn snr(pcm: &[f32], decoded: &[f32], channels: usize) -> f32 {
    let delay = 120 * channels;
    let skip = 4800 * channels;
    let mut sig = 0f64;
    let mut err = 0f64;
    for i in skip..decoded.len().min(pcm.len() + delay) {
        let a = f64::from(pcm[i - delay]);
        let b = f64::from(decoded[i]);
        sig += a * a;
        err += (a - b) * (a - b);
    }
    (10.0 * (sig / err.max(1e-20)).log10()) as f32
}

#[test]
fn every_layout_round_trips_with_matching_ranges() {
    let mut seed = 1u32;
    for channels in [1usize, 2] {
        let pcm = signal(48_000, channels, &mut seed);
        for lm in 0..4 {
            for (end_band, bitrate) in [(21usize, 96_000u32), (17, 24_000), (13, 8_000)] {
                for mode in [RateMode::Cbr, RateMode::Vbr, RateMode::Constrained] {
                    let cfg = CeltConfig {
                        channels,
                        lm,
                        end_band,
                        bitrate,
                        mode,
                    };
                    let trip = round_trip(cfg, &pcm);
                    if mode == RateMode::Cbr {
                        let want = (bitrate as usize * (120 << lm) + 4 * 48_000) / (8 * 48_000);
                        assert!(trip.sizes.iter().all(|&s| s + 1 == want), "{cfg:?}");
                    }
                    assert!(trip.decoded.iter().all(|v| v.is_finite()));
                }
            }
        }
    }
}

#[test]
fn full_band_output_follows_the_input() {
    let mut seed = 5u32;
    for (channels, bitrate, floor) in [(1usize, 64_000u32, 12.0f32), (2, 128_000, 12.0)] {
        let pcm = signal(96_000, channels, &mut seed);
        for mode in [RateMode::Cbr, RateMode::Vbr] {
            let cfg = CeltConfig {
                channels,
                lm: 3,
                end_band: 21,
                bitrate,
                mode,
            };
            let trip = round_trip(cfg, &pcm);
            let s = snr(&pcm, &trip.decoded, channels);
            eprintln!("{cfg:?}: SNR {s:.1} dB");
            assert!(s > floor, "{cfg:?}: SNR {s} dB");
        }
    }
}

#[test]
fn vbr_holds_the_requested_mean() {
    let mut seed = 9u32;
    for (channels, bitrate) in [(1usize, 32_000u32), (2, 96_000)] {
        let pcm = signal(48_000 * 8, channels, &mut seed);
        for mode in [RateMode::Vbr, RateMode::Constrained] {
            let cfg = CeltConfig {
                channels,
                lm: 3,
                end_band: 21,
                bitrate,
                mode,
            };
            let trip = round_trip(cfg, &pcm);
            let bytes: usize = trip.sizes.iter().map(|s| s + 1).sum();
            let rate = bytes as f64 * 8.0 / (trip.sizes.len() as f64 * 0.02);
            let ratio = rate / f64::from(bitrate);
            eprintln!("{cfg:?}: {rate:.0} b/s");
            assert!((0.9..=1.1).contains(&ratio), "{cfg:?}: {rate} b/s");
        }
    }
}

#[test]
fn hostile_signals_encode_and_decode() {
    let mut seed = 3u32;
    let n = 960;
    let shapes: Vec<Vec<f32>> = vec![
        vec![0.0; n * 2 * 6],
        (0..n * 2 * 6)
            .map(|_| (lcg(&mut seed) >> 8) as f32 / 8_388_608.0 - 1.0)
            .collect(),
        (0..n * 2 * 6)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect(),
        vec![1.0; n * 2 * 6],
        (0..n * 2 * 6)
            .map(|i| if i == 5000 { 1.0 } else { 0.0 })
            .collect(),
        (0..n * 2 * 6).map(|i| 1e-30 * (i % 7) as f32).collect(),
        (0..n * 2 * 6)
            .map(|i| if (i / 1440) % 2 == 0 { 0.0 } else { 0.9 })
            .collect(),
    ];
    for shape in &shapes {
        for channels in [1usize, 2] {
            for lm in [0usize, 3] {
                for mode in [RateMode::Cbr, RateMode::Vbr, RateMode::Constrained] {
                    for bitrate in [6_000u32, 510_000] {
                        let cfg = CeltConfig {
                            channels,
                            lm,
                            end_band: 21,
                            bitrate,
                            mode,
                        };
                        let frames = shape.len() / (n * channels) * n / (120 << lm);
                        let len = frames * (120 << lm) * channels;
                        let trip = round_trip(cfg, &shape[..len]);
                        assert!(trip.decoded.iter().all(|v| v.is_finite()));
                    }
                }
            }
        }
    }
}

#[test]
fn reset_matches_a_fresh_encoder() {
    let mut seed = 11u32;
    let pcm = signal(9600, 2, &mut seed);
    let cfg = CeltConfig {
        channels: 2,
        lm: 3,
        end_band: 21,
        bitrate: 96_000,
        mode: RateMode::Vbr,
    };
    let run = |enc: &mut CeltEncoder| -> Vec<Vec<u8>> {
        let n = enc.frame_size() * 2;
        pcm.chunks_exact(n)
            .map(|f| enc.encode(f).expect("encode"))
            .collect()
    };
    let mut fresh = CeltEncoder::new(cfg).expect("encoder");
    let expected = run(&mut fresh);
    let mut used = CeltEncoder::new(cfg).expect("encoder");
    run(&mut used);
    used.reset();
    assert_eq!(run(&mut used), expected);
}
