//! The encoders: our decoder, FFmpeg and Apple read what they write, at
//! every layout, rate and block count, with the bit rate, timing and
//! metadata they were asked for.

use std::f64::consts::PI;

use cutlass_ac3::{
    Ac3Decoder, Ac3Encoder, ENCODER_DELAY, EncoderConfig, EncoderMetadata, Format, MixCodes,
    Options, ac3_bit_rate, probe,
};
use cutlass_decode::{Channel, ChannelLayout, CodecError, CodecId, Decoder, Packet, Pcm, SideData};
use cutlass_encode::{AudioInput, Encoder};

use crate::common::{apple_decode, audible, deinterleave, ffmpeg_decode, oracle, power, snr, tone};

fn layout(channels: &[Channel]) -> ChannelLayout {
    ChannelLayout::new(channels.to_vec()).expect("layout")
}

use Channel::{
    BackCenter as Cs, BackLeft as Bl, BackRight as Br, FrontCenter as C, FrontLeft as L,
    FrontRight as R, Lfe, SideLeft as Sl, SideRight as Sr,
};

/// Every layout the encoders take, with its `acmod` and LFE.
const LAYOUTS: [(&[Channel], u8, bool); 12] = [
    (&[C], 1, false),
    (&[L, R], 2, false),
    (&[L, R, C], 3, false),
    (&[L, R, Cs], 4, false),
    (&[L, R, C, Cs], 5, false),
    (&[L, R, Sl, Sr], 6, false),
    (&[L, R, C, Sl, Sr], 7, false),
    (&[L, R, Lfe], 2, true),
    (&[C, L, R, Bl, Br, Lfe], 7, true),
    (&[L, R, C, Lfe, Sl, Sr], 7, true),
    (&[L, R, Cs, Lfe], 4, true),
    (&[L, R, C, Lfe], 3, true),
];

/// A tone per speaker at `amplitude`, interleaved in `channels` order.
fn tones(channels: &[Channel], rate: u32, frames: usize, amplitude: f64) -> Vec<f32> {
    let mut pcm = Vec::with_capacity(frames * channels.len());
    for n in 0..frames {
        for &channel in channels {
            let t = n as f64 / f64::from(rate);
            pcm.push((amplitude * (2.0 * PI * tone(channel) * t).sin()) as f32);
        }
    }
    pcm
}

fn input(channels: &[Channel], rate: u32, pcm: Vec<f32>) -> AudioInput {
    AudioInput {
        pcm: Pcm::from_f32(rate, layout(channels), pcm).expect("pcm"),
    }
}

/// Encode `pcm` in uneven pushes, then drain.
fn encode(config: EncoderConfig, pcm: &[f32]) -> Vec<Packet> {
    let channels = config.layout.channels().to_vec();
    let rate = config.sample_rate;
    let mut encoder = Ac3Encoder::new(config).expect("open");
    let n = channels.len();
    let mut packets = Vec::new();
    let mut at = 0;
    for chunk in [1usize, 700, 4096, 333, 2000].iter().cycle() {
        if at >= pcm.len() {
            break;
        }
        let end = (at + chunk * n).min(pcm.len());
        packets.extend(
            encoder
                .push(&input(&channels, rate, pcm[at..end].to_vec()))
                .expect("push"),
        );
        at = end;
    }
    packets.extend(encoder.drain().expect("drain"));
    packets
}

fn stream(packets: &[Packet]) -> Vec<u8> {
    packets
        .iter()
        .flat_map(|p| p.data.iter().copied())
        .collect()
}

struct Decoded {
    channels: Vec<Channel>,
    /// Gapless-trimmed, interleaved.
    pcm: Vec<f32>,
    decoder: Ac3Decoder,
}

fn decode(codec: CodecId, packets: &[Packet]) -> Decoded {
    let params = cutlass_decode::CodecParams::audio(codec, 0, ChannelLayout::stereo());
    let mut decoder = Ac3Decoder::with_options(&params, Options::default()).expect("decoder");
    let mut outputs = Vec::new();
    for packet in packets {
        outputs.extend(decoder.push(packet).expect("decode"));
    }
    outputs.extend(decoder.drain().expect("drain"));
    let channels = decoder.stream().expect("stream").1.channels().to_vec();
    Decoded {
        channels,
        pcm: audible(outputs),
        decoder,
    }
}

fn config(format: Format, rate: u32, channels: &[Channel], bit_rate: u32) -> EncoderConfig {
    EncoderConfig {
        bit_rate,
        ..EncoderConfig::new(format, rate, layout(channels))
    }
}

fn codec(format: Format) -> CodecId {
    match format {
        Format::Ac3 => CodecId::Ac3,
        Format::Eac3 => CodecId::Eac3,
    }
}

/// The input channel each decoded channel came from: the surround pair
/// decodes as the side pair whichever pair went in.
fn source(channels: &[Channel], decoded: Channel) -> Option<usize> {
    let alias = |c: Channel| match c {
        Bl => Sl,
        Br => Sr,
        other => other,
    };
    channels.iter().position(|&c| alias(c) == alias(decoded))
}

#[test]
fn every_layout_decodes_in_order_with_its_lfe() {
    for format in [Format::Ac3, Format::Eac3] {
        for (channels, acmod, lfe) in LAYOUTS {
            let rate = 48_000;
            let pcm = tones(channels, rate, 48_000, 0.25);
            let packets = encode(config(format, rate, channels, 0), &pcm);
            let decoded = decode(codec(format), &packets);
            let metadata = decoded.decoder.metadata().expect("metadata");
            assert_eq!(
                (metadata.acmod, metadata.lfe),
                (acmod, lfe),
                "{format:?} {channels:?}"
            );
            assert_eq!(
                decoded.channels.len(),
                channels.len(),
                "{format:?} {channels:?}"
            );
            let n = channels.len();
            for (index, &channel) in decoded.channels.iter().enumerate() {
                let from = source(channels, channel)
                    .unwrap_or_else(|| panic!("{format:?} {channels:?}: {channel:?} not sent"));
                let ours = deinterleave(&decoded.pcm, n, index);
                let sent = deinterleave(&pcm, n, from);
                let middle = 4_096..ours.len() - 4_096;
                let quality = snr(&ours[middle.clone()], &sent[middle.clone()]);
                let floor = if channel == Lfe { 20.0 } else { 25.0 };
                assert!(
                    quality > floor,
                    "{format:?} {channels:?} {channel:?}: {quality:.1} dB"
                );
                let own = power(&ours[middle.clone()], rate, tone(channel));
                // Side and back surrounds share a fixture tone, so a
                // decoded side channel is not a leak of the back channel.
                for &other in channels
                    .iter()
                    .filter(|&&c| c != channel && c != Lfe && tone(c) != tone(channel))
                {
                    let leak = power(&ours[middle.clone()], rate, tone(other));
                    assert!(
                        own > leak * 100.0,
                        "{format:?} {channels:?}: {channel:?} carries {other:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn decoded_length_matches_the_input_exactly() {
    for format in [Format::Ac3, Format::Eac3] {
        for frames in [1usize, 255, 256, 1_280, 1_536, 1_537, 10_000] {
            let channels = [L, R];
            let pcm = tones(&channels, 48_000, frames, 0.3);
            let packets = encode(config(format, 48_000, &channels, 0), &pcm);
            let spf = 1536i64;
            for (index, packet) in packets.iter().enumerate() {
                assert_eq!(packet.pts, index as i64 * spf);
                assert_eq!(packet.duration, spf);
                assert_eq!(packet.timescale, 48_000);
            }
            let total = packets.len() as u64 * 1536;
            assert_eq!(total, (frames as u64 + 256).div_ceil(1536) * 1536);
            let Some(SideData::Gapless {
                encoder_delay,
                trailing_padding,
                ..
            }) = packets.last().and_then(|p| {
                p.side
                    .iter()
                    .find(|s| matches!(s, SideData::Gapless { .. }))
            })
            else {
                panic!("gapless side data");
            };
            assert_eq!(*encoder_delay, ENCODER_DELAY);
            assert_eq!(
                u64::from(*trailing_padding),
                total - frames as u64 - u64::from(ENCODER_DELAY)
            );
            let decoded = decode(codec(format), &packets);
            assert_eq!(decoded.pcm.len(), pcm.len(), "{format:?} {frames}");
        }
    }
}

#[test]
fn the_decoded_signal_lines_up_with_the_input() {
    // A click at a known sample comes back at the same sample.
    for format in [Format::Ac3, Format::Eac3] {
        let mut pcm = vec![0.0f32; 9_000];
        pcm[5_000] = 0.9;
        let packets = encode(config(format, 48_000, &[C], 0), &pcm);
        let decoded = decode(codec(format), &packets);
        let peak = decoded
            .pcm
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .map(|(i, _)| i);
        assert_eq!(peak, Some(5_000), "{format:?}");
    }
}

#[test]
fn ac3_syncframes_follow_table_5_18() {
    for rate in [48_000u32, 44_100, 32_000] {
        for kbps in [64u32, 192, 448, 640] {
            let channels = [L, R];
            let pcm = tones(&channels, rate, rate as usize * 2, 0.3);
            let packets = encode(config(Format::Ac3, rate, &channels, kbps * 1000), &pcm);
            let mut bytes = 0usize;
            for packet in &packets {
                let info = probe(&packet.data).expect("probe");
                assert_eq!(info.frame_bytes, packet.data.len());
                assert_eq!(ac3_bit_rate(packet.data[4] & 0x3f), Some(kbps * 1000));
                bytes += packet.data.len();
            }
            let seconds = packets.len() as f64 * 1536.0 / f64::from(rate);
            let actual = bytes as f64 * 8.0 / seconds;
            let target = f64::from(kbps * 1000);
            assert!(
                (actual - target).abs() / target < 0.002,
                "{rate} {kbps}: {actual:.0} bit/s"
            );
        }
    }
}

#[test]
fn eac3_holds_any_bit_rate_at_any_block_count() {
    for blocks in [1usize, 2, 3, 6] {
        for bit_rate in [96_000u32, 127_000, 200_000, 333_333] {
            let channels = [L, R];
            let rate = 48_000;
            let pcm = tones(&channels, rate, rate as usize * 2, 0.3);
            let mut cfg = config(Format::Eac3, rate, &channels, bit_rate);
            cfg.blocks = blocks;
            let packets = encode(cfg, &pcm);
            let spf = blocks * 256;
            let bytes: usize = packets.iter().map(|p| p.data.len()).sum();
            for packet in &packets {
                assert_eq!(packet.duration, spf as i64);
                let info = probe(&packet.data).expect("probe");
                assert_eq!(info.frame_bytes, packet.data.len());
            }
            let seconds = (packets.len() * spf) as f64 / f64::from(rate);
            let actual = bytes as f64 * 8.0 / seconds;
            let target = f64::from(bit_rate);
            // One syncframe of rounding over the whole stream.
            let slack = 16.0 / seconds + target * 0.001;
            assert!(
                (actual - target).abs() <= slack,
                "{blocks} blocks {bit_rate}: {actual:.0}"
            );
            let decoded = decode(CodecId::Eac3, &packets);
            assert_eq!(decoded.pcm.len(), pcm.len());
            let tools = decoded.decoder.tools();
            assert_eq!(tools.short_frames > 0, blocks < 6);
            let middle = 4_096..pcm.len() / 2 - 4_096;
            let left = deinterleave(&decoded.pcm, 2, 0);
            let sent = deinterleave(&pcm, 2, 0);
            let quality = snr(&left[middle.clone()], &sent[middle]);
            assert!(
                quality > 25.0,
                "{blocks} blocks {bit_rate}: {quality:.1} dB"
            );
        }
    }
}

#[test]
fn quality_rises_with_the_bit_rate() {
    for format in [Format::Ac3, Format::Eac3] {
        let channels = [L, R];
        let rate = 48_000;
        // Noise-like program: a chord with a slow sweep.
        let pcm: Vec<f32> = (0..rate as usize * 2)
            .flat_map(|n| {
                let t = n as f64 / f64::from(rate);
                let sweep = 200.0 + 4_000.0 * t;
                let l = 0.2 * (2.0 * PI * sweep * t).sin() + 0.1 * (2.0 * PI * 3_100.0 * t).sin();
                let r = 0.2 * (2.0 * PI * 440.0 * t).sin() + 0.1 * (2.0 * PI * 7_500.0 * t).sin();
                [l as f32, r as f32]
            })
            .collect();
        let mut last = f64::NEG_INFINITY;
        for kbps in [96u32, 192, 384] {
            let packets = encode(config(format, rate, &channels, kbps * 1000), &pcm);
            let decoded = decode(codec(format), &packets);
            let quality = snr(&decoded.pcm, &pcm);
            assert!(
                quality > last,
                "{format:?} {kbps}: {quality:.1} after {last:.1}"
            );
            last = quality;
        }
    }
}

#[test]
fn a_transient_stays_quiet_before_the_attack() {
    for format in [Format::Ac3, Format::Eac3] {
        let channels = [L, R];
        let rate = 48_000;
        let mut pcm = vec![0.0f32; rate as usize * 2];
        // Silence, then a burst of noise-like tone at full level.
        for n in 30_000..pcm.len() / 2 {
            let t = n as f64 / f64::from(rate);
            let v = (0.8 * (2.0 * PI * 9_000.0 * t).sin()) as f32;
            pcm[2 * n] = v;
            pcm[2 * n + 1] = v * 0.5;
        }
        let packets = encode(config(format, rate, &channels, 0), &pcm);
        let decoded = decode(codec(format), &packets);
        // E-AC-3 does not switch length: a short block would change
        // coupling inside the syncframe, which the reference decoders
        // reject. AC-3 still switches, and that is what holds pre-echo down.
        let switched = decoded.decoder.tools().short_blocks > 0;
        assert_eq!(switched, format == Format::Ac3, "{format:?}");
        let left = deinterleave(&decoded.pcm, 2, 0);
        let before = &left[29_000..29_900];
        let peak = before.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(peak < 0.05, "{format:?}: pre-echo {peak}");
    }
}

#[test]
fn coupling_and_rematrixing_carry_the_low_rate_programs() {
    for format in [Format::Ac3, Format::Eac3] {
        let channels = [L, R];
        let rate = 48_000;
        let pcm: Vec<f32> = (0..rate as usize)
            .flat_map(|n| {
                let t = n as f64 / f64::from(rate);
                let common = 0.3 * (2.0 * PI * 600.0 * t).sin();
                let high = 0.05 * (2.0 * PI * 9_000.0 * t).sin();
                [(common + high) as f32, (common * 0.95 + high * 0.8) as f32]
            })
            .collect();
        let packets = encode(config(format, rate, &channels, 96_000), &pcm);
        let decoded = decode(codec(format), &packets);
        let tools = decoded.decoder.tools();
        assert!(tools.coupling_blocks > 0, "{format:?}");
        assert!(tools.rematrix_blocks > 0, "{format:?}");
    }
}

#[test]
fn metadata_round_trips() {
    let mix = MixCodes {
        dmixmod: 2,
        ltrt_center: 3,
        ltrt_surround: 5,
        loro_center: 6,
        loro_surround: 7,
    };
    let channels = [L, R, C, Lfe, Sl, Sr];
    for format in [Format::Ac3, Format::Eac3] {
        let mut cfg = config(format, 48_000, &channels, 0);
        cfg.metadata = EncoderMetadata {
            dialnorm: 24,
            bsmod: 3,
            dsurmod: 0,
            mix: Some(mix),
            lfe_mix: (format == Format::Eac3).then_some(12),
        };
        let pcm = tones(&channels, 48_000, 6_000, 0.2);
        let packets = encode(cfg, &pcm);
        let decoded = decode(codec(format), &packets);
        let metadata = decoded.decoder.metadata().expect("metadata");
        assert_eq!(metadata.dialnorm, 24);
        assert_eq!(metadata.bsmod, 3);
        assert_eq!(metadata.dmixmod, Some(2));
        let levels = metadata.mix_levels.expect("mix levels");
        // Annex D codes are powers of two: code 3 is 2^-1/4 (−1.5 dB),
        // code 5 is 2^-3/4 (−4.5 dB).
        let gain = |quarters: i32| 2f32.powf(quarters as f32 / 4.0);
        assert!((levels.ltrt_center - gain(-1)).abs() < 1e-6);
        assert!((levels.ltrt_surround - gain(-3)).abs() < 1e-6);
        assert!((levels.loro_center - 0.5).abs() < 1e-3);
        assert_eq!(levels.loro_surround, 0.0);
        if format == Format::Eac3 {
            assert_eq!(metadata.lfe_mix_db, Some(-2));
        }
    }
    let channels = [L, R];
    for format in [Format::Ac3, Format::Eac3] {
        let mut cfg = config(format, 44_100, &channels, 0);
        cfg.metadata.dsurmod = 2;
        let packets = encode(cfg, &tones(&channels, 44_100, 3_000, 0.2));
        let decoded = decode(codec(format), &packets);
        assert_eq!(decoded.decoder.metadata().expect("metadata").dsurmod, 2);
    }
}

#[test]
fn the_first_packet_carries_the_config_record() {
    let channels = [L, R, C, Lfe, Sl, Sr];
    for format in [Format::Ac3, Format::Eac3] {
        let cfg = config(format, 48_000, &channels, 0);
        let encoder = Ac3Encoder::new(cfg.clone()).expect("open");
        let record = encoder.config_record();
        let packets = encode(cfg, &tones(&channels, 48_000, 4_000, 0.2));
        let configs: Vec<_> = packets
            .iter()
            .enumerate()
            .flat_map(|(i, p)| {
                p.side.iter().filter_map(move |s| match s {
                    SideData::Config(bytes) => Some((i, bytes.clone())),
                    _ => None,
                })
            })
            .collect();
        assert_eq!(configs, vec![(0, record.clone())]);
        match format {
            Format::Ac3 => {
                let dac3 = cutlass_ac3::Dac3::parse(&record).expect("dac3");
                assert_eq!((dac3.acmod, dac3.lfeon, dac3.fscod), (7, true, 0));
                assert_eq!(dac3.bit_rate_code, 15);
            }
            Format::Eac3 => {
                let dec3 = cutlass_ac3::Dec3::parse(&record).expect("dec3");
                assert_eq!(dec3.data_rate, 384);
                assert_eq!(dec3.substreams.len(), 1);
                assert_eq!(dec3.substreams[0].acmod, 7);
                assert!(dec3.substreams[0].lfeon);
            }
        }
    }
}

#[test]
fn reset_starts_a_fresh_stream_and_drain_ends_it() {
    let channels = [L, R];
    for format in [Format::Ac3, Format::Eac3] {
        let cfg = config(format, 48_000, &channels, 0);
        let pcm = tones(&channels, 48_000, 5_000, 0.3);
        let fresh = encode(cfg.clone(), &pcm);
        let mut encoder = Ac3Encoder::new(cfg).expect("open");
        encoder
            .push(&input(
                &channels,
                48_000,
                tones(&channels, 48_000, 3_333, 0.7),
            ))
            .expect("push");
        encoder.reset();
        let mut again = encoder.push(&input(&channels, 48_000, pcm)).expect("push");
        again.extend(encoder.drain().expect("drain"));
        assert_eq!(again, fresh, "{format:?}");
        assert!(encoder.drain().expect("second drain").is_empty());
        assert!(matches!(
            encoder.push(&input(&channels, 48_000, vec![0.0; 2])),
            Err(CodecError::InvalidApi { .. })
        ));
        encoder.reset();
        assert!(encoder.drain().expect("empty drain").is_empty());
    }
}

#[test]
fn illegal_input_fails_and_adds_nothing() {
    let channels = [L, R];
    let cfg = config(Format::Ac3, 48_000, &channels, 0);
    let pcm = tones(&channels, 48_000, 4_000, 0.3);
    let clean = encode(cfg.clone(), &pcm);
    let mut encoder = Ac3Encoder::new(cfg).expect("open");
    let mut packets = encoder
        .push(&input(&channels, 48_000, pcm[..2_000].to_vec()))
        .expect("push");
    let mut bad = vec![0.1f32; 600];
    bad[301] = f32::NAN;
    assert_eq!(
        encoder.push(&input(&channels, 48_000, bad)),
        Err(CodecError::NonFinite)
    );
    assert_eq!(
        encoder.push(&input(&channels, 48_000, vec![1.5; 4])),
        Err(CodecError::OutOfRange)
    );
    assert!(matches!(
        encoder.push(&input(&channels, 44_100, vec![0.0; 4])),
        Err(CodecError::InvalidApi { .. })
    ));
    assert!(matches!(
        encoder.push(&input(&[C], 48_000, vec![0.0; 4])),
        Err(CodecError::InvalidApi { .. })
    ));
    packets.extend(
        encoder
            .push(&input(&channels, 48_000, pcm[2_000..].to_vec()))
            .expect("push"),
    );
    packets.extend(encoder.drain().expect("drain"));
    assert_eq!(packets, clean);
}

#[test]
fn a_raised_cancel_flag_stops_the_next_frame() {
    let channels = [L, R];
    let mut encoder = Ac3Encoder::new(config(Format::Eac3, 48_000, &channels, 0)).expect("open");
    encoder
        .cancel_flag()
        .store(true, std::sync::atomic::Ordering::Relaxed);
    assert_eq!(
        encoder.push(&input(
            &channels,
            48_000,
            tones(&channels, 48_000, 4_000, 0.2)
        )),
        Err(CodecError::Cancelled)
    );
}

#[test]
fn unsupported_configurations_are_refused() {
    let unsupported = |cfg: EncoderConfig| {
        matches!(
            Ac3Encoder::new(cfg),
            Err(CodecError::Unsupported { .. } | CodecError::InvalidApi { .. })
        )
    };
    assert!(unsupported(config(Format::Ac3, 22_050, &[L, R], 0)));
    assert!(unsupported(config(Format::Ac3, 48_000, &[L, R], 100_000)));
    assert!(unsupported(EncoderConfig {
        blocks: 4,
        ..config(Format::Eac3, 48_000, &[L, R], 0)
    }));
    assert!(unsupported(config(
        Format::Eac3,
        48_000,
        &[L, R, C, Lfe, Bl, Br, Sl, Sr],
        0
    )));
    assert!(unsupported(config(Format::Ac3, 48_000, &[L, C], 0)));
    assert!(unsupported(config(Format::Eac3, 48_000, &[L, R], 4_000)));
    assert!(unsupported(config(
        Format::Eac3,
        48_000,
        &[L, R],
        2_000_000
    )));
    let mut cfg = config(Format::Ac3, 48_000, &[L, R, Lfe], 0);
    cfg.metadata.lfe_mix = Some(10);
    assert!(unsupported(cfg));
    let mut cfg = config(Format::Ac3, 48_000, &[L, R], 0);
    cfg.metadata.dialnorm = 0;
    assert!(unsupported(cfg));
}

#[test]
fn the_lowest_rates_still_code_every_layout() {
    for (format, kbps) in [(Format::Ac3, 32u32), (Format::Eac3, 32)] {
        for (channels, _, _) in LAYOUTS {
            let rate = 48_000;
            let bit_rate = kbps * 1000 * channels.len().div_ceil(2) as u32;
            let bit_rate = match format {
                Format::Ac3 => [32u32, 64, 96, 128]
                    .iter()
                    .map(|k| k * 1000)
                    .find(|&r| r >= bit_rate)
                    .unwrap_or(128_000),
                Format::Eac3 => bit_rate,
            };
            // Full-band noise is the hardest program for the rate control.
            let mut seed = 1u32;
            let pcm: Vec<f32> = (0..rate as usize * channels.len())
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
                })
                .collect();
            let packets = encode(config(format, rate, channels, bit_rate), &pcm);
            let decoded = decode(codec(format), &packets);
            assert_eq!(decoded.pcm.len(), pcm.len(), "{format:?} {channels:?}");
        }
    }
}

/// FFmpeg reads our streams and hears what our decoder hears.
#[test]
fn ffmpeg_decodes_our_streams() {
    let Some(ffmpeg) = oracle("ffmpeg", "-version") else {
        return;
    };
    for format in [Format::Ac3, Format::Eac3] {
        for (channels, _, _) in LAYOUTS {
            for rate in [48_000u32, 44_100, 32_000] {
                let pcm = tones(channels, rate, rate as usize, 0.25);
                let packets = encode(config(format, rate, channels, 0), &pcm);
                let ours = decode(codec(format), &packets);
                let theirs = ffmpeg_decode(&ffmpeg, &stream(&packets), &[]);
                let n = channels.len();
                // FFmpeg writes WAVEFORMATEXTENSIBLE order and no trim.
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by_key(|&i| ours.channels[i].wave_bit());
                let skip = ENCODER_DELAY as usize;
                for (theirs_index, &our_index) in order.iter().enumerate() {
                    let a = deinterleave(&ours.pcm, n, our_index);
                    let b: Vec<f32> = deinterleave(&theirs, n, theirs_index)
                        .into_iter()
                        .skip(skip)
                        .collect();
                    let n = a.len().min(b.len());
                    // The first and last frames are the transform warmup and
                    // the flush. The program itself is the span between them.
                    let margin = 1536.min(n / 4);
                    let quality = snr(&a[margin..n - margin], &b[margin..n - margin]);
                    assert!(
                        quality > 60.0,
                        "{format:?} {channels:?} {rate}: {:?} {quality:.1} dB",
                        ours.channels[our_index]
                    );
                }
            }
        }
    }
}

/// One, two, three and six blocks per syncframe. Each packet lasts that
/// many blocks, and FFmpeg hears the same program our decoder hears.
#[test]
fn every_eac3_block_count_decodes_with_ffmpeg() {
    let Some(ffmpeg) = oracle("ffmpeg", "-version") else {
        return;
    };
    let rate = 48_000u32;
    let layouts: [&[Channel]; 3] = [&[C], &[L, R], &[L, R, C, Lfe, Sl, Sr]];
    for blocks in [1usize, 2, 3, 6] {
        for channels in layouts {
            let mut cfg = config(Format::Eac3, rate, channels, 0);
            cfg.blocks = blocks;
            let pcm = tones(channels, rate, rate as usize, 0.25);
            let packets = encode(cfg, &pcm);
            let spf = (blocks * 256) as i64;
            assert!(!packets.is_empty(), "{blocks} {channels:?}");
            for packet in &packets {
                assert_eq!(packet.duration, spf, "{blocks} {channels:?}");
                let info = probe(&packet.data).expect("probe");
                assert_eq!(info.blocks, blocks, "{blocks} {channels:?}");
            }
            let ours = decode(CodecId::Eac3, &packets);
            assert_eq!(
                ours.pcm.len(),
                pcm.len(),
                "{blocks} blocks {channels:?}: sample count"
            );
            let theirs = ffmpeg_decode(&ffmpeg, &stream(&packets), &["-f", "eac3"]);
            let nch = channels.len();
            let mut order: Vec<usize> = (0..nch).collect();
            order.sort_by_key(|&i| ours.channels[i].wave_bit());
            let skip = ENCODER_DELAY as usize;
            for (theirs_index, &our_index) in order.iter().enumerate() {
                let a = deinterleave(&ours.pcm, nch, our_index);
                let b: Vec<f32> = deinterleave(&theirs, nch, theirs_index)
                    .into_iter()
                    .skip(skip)
                    .collect();
                let n = a.len().min(b.len());
                let margin = 1536.min(n / 4);
                let quality = snr(&a[margin..n - margin], &b[margin..n - margin]);
                let src = deinterleave(&pcm, nch, our_index);
                let against = snr(&a[margin..n - margin], &src[margin..n - margin]);
                assert!(
                    quality > 60.0,
                    "{blocks} blocks {channels:?} {:?}: {quality:.1} dB against FFmpeg",
                    ours.channels[our_index]
                );
                let floor = if blocks == 1 { 40.0 } else { 60.0 };
                assert!(
                    against > floor,
                    "{blocks} blocks {channels:?} {:?}: {against:.1} dB against the source",
                    ours.channels[our_index]
                );
            }
        }
    }
}

/// A short block used to turn coupling on or off inside an E-AC-3
/// syncframe, and reference decoders reject that frame. The onset of a
/// tone stays in the comparison. Full-band noise at the default rate is
/// mostly dither, so it only has to be a syncframe FFmpeg accepts.
#[test]
fn ffmpeg_decodes_an_eac3_frame_that_starts_with_a_transient() {
    let Some(ffmpeg) = oracle("ffmpeg", "-version") else {
        return;
    };
    let channels = &[L, R];
    let rate = 48_000u32;
    let mut seed = 1u32;
    let noise: Vec<f32> = (0..1536 * 2)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect();
    let packets = encode(config(Format::Eac3, rate, channels, 0), &noise);
    let _ours = decode(CodecId::Eac3, &packets);
    let _theirs = ffmpeg_decode(&ffmpeg, &stream(&packets), &["-f", "eac3"]);

    let mut impulse = vec![0.0f32; 1536 * 2];
    impulse[768 * 2] = 0.8;
    impulse[768 * 2 + 1] = 0.8;
    let packets = encode(config(Format::Eac3, rate, channels, 0), &impulse);
    let ours = decode(CodecId::Eac3, &packets);
    let theirs = ffmpeg_decode(&ffmpeg, &stream(&packets), &["-f", "eac3"]);
    let skip = ENCODER_DELAY as usize * 2;
    let theirs = &theirs[skip.min(theirs.len())..];
    let n = ours.pcm.len().min(theirs.len());
    let quality = snr(&ours.pcm[..n], &theirs[..n]);
    assert!(quality > 50.0, "impulse {quality:.1} dB");

    let pcm = tones(channels, rate, rate as usize, 0.25);
    let packets = encode(config(Format::Eac3, rate, channels, 0), &pcm);
    let ours = decode(CodecId::Eac3, &packets);
    let theirs = ffmpeg_decode(&ffmpeg, &stream(&packets), &[]);
    let a = deinterleave(&ours.pcm, 2, 0);
    let b: Vec<f32> = deinterleave(&theirs, 2, 0)
        .into_iter()
        .skip(ENCODER_DELAY as usize)
        .collect();
    let n = a.len().min(b.len());
    let tail = 1536.min(n / 8);
    let quality = snr(&a[..n - tail], &b[..n - tail]);
    assert!(quality > 50.0, "onset {quality:.1} dB");
}

/// Apple's decoder reads our streams.
#[test]
fn apple_decodes_our_streams() {
    let Some(afconvert) = oracle("afconvert", "-h") else {
        return;
    };
    for format in [Format::Ac3, Format::Eac3] {
        for (channels, _, _) in LAYOUTS {
            let rate = 48_000;
            let pcm = tones(channels, rate, rate as usize, 0.25);
            let packets = encode(config(format, rate, channels, 0), &pcm);
            let ours = decode(codec(format), &packets);
            let (count, theirs) = apple_decode(&afconvert, &stream(&packets));
            let n = channels.len();
            // Apple folds a center into the fronts and drops LFE for 3/1
            // and 2/1 once an LFE is present, and it drops the single
            // surround of 2/1 and 3/1. Speakers it does render are matched
            // by the signal, in whichever slot it chose.
            let folded =
                channels.contains(&Lfe) && (channels.contains(&C) || channels.contains(&Cs));
            for index in 0..n {
                if folded || ours.channels[index] == Cs {
                    continue;
                }
                let a = deinterleave(&ours.pcm, n, index);
                let mut best = f64::NEG_INFINITY;
                for apple in 0..count {
                    let b = deinterleave(&theirs, count, apple);
                    let len = a.len().min(b.len()).saturating_sub(2_048);
                    if len < 256 {
                        continue;
                    }
                    best = best.max(snr(&a[..len], &b[..len]));
                }
                assert!(
                    best > 40.0,
                    "{format:?} {channels:?}: {:?} {best:.1} dB",
                    ours.channels[index]
                );
            }
        }
    }
}
