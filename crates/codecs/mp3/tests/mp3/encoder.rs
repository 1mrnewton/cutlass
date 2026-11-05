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

/// Packets in an MP3 file with the Info/Xing frame carrying their trim.
pub fn mux(config: Mp3EncoderConfig, packets: &[Packet]) -> Vec<u8> {
    use cutlass_mp3_container::{CodecId, CodecParams, Mp3Mux, Mux, Packet as FramePacket};
    let mut params = CodecParams::audio(CodecId::Mp3, config.sample_rate, config.channels as u16);
    for side in packets.iter().flat_map(|packet| &packet.side) {
        if let SideData::Gapless {
            encoder_delay,
            trailing_padding,
            ..
        } = side
        {
            params.encoder_delay = *encoder_delay;
            params.trailing_padding = params.trailing_padding.max(*trailing_padding);
        }
    }
    let mut mux = Mp3Mux::new(std::io::Cursor::new(Vec::new()));
    mux.add_stream(params).expect("add stream");
    for packet in packets {
        mux.write_packet(&FramePacket::audio(
            packet.data.clone(),
            packet.pts,
            packet.duration,
            packet.timescale,
        ))
        .expect("write");
    }
    mux.finish().expect("finish");
    mux.into_inner().expect("bytes").into_inner()
}

/// The trailing padding the last packet carries.
fn padding(packets: &[Packet]) -> u32 {
    packets
        .last()
        .and_then(|packet| {
            packet.side.iter().find_map(|side| match side {
                SideData::Gapless {
                    trailing_padding, ..
                } => Some(*trailing_padding),
                _ => None,
            })
        })
        .expect("gapless")
}

const ORACLE_CONFIGS: [(u32, usize, Mp3Rate); 7] = [
    (44_100, 2, Mp3Rate::Constant(128)),
    (44_100, 2, Mp3Rate::Variable(2)),
    (48_000, 1, Mp3Rate::Constant(64)),
    (32_000, 2, Mp3Rate::Constant(320)),
    (22_050, 2, Mp3Rate::Variable(5)),
    (16_000, 1, Mp3Rate::Constant(32)),
    (8_000, 1, Mp3Rate::Constant(8)),
];

#[test]
fn the_reference_decoders_read_what_the_encoder_writes() {
    use crate::common::{
        ISO_MAX, ISO_RMS, decode_packets, demux, difference, oracle_compare, oracle_ffmpeg,
        oracle_mpg123, reference_pcm,
    };
    let ffmpeg = oracle_ffmpeg();
    let mpg123 = oracle_mpg123();
    for (rate, channels, mode) in ORACLE_CONFIGS {
        let config = config(rate, channels, mode);
        let input = music(rate, channels, 1.5);
        let packets = encode(config, &input, 4096);
        let bytes = mux(config, &packets);
        let label = format!("{config:?}");
        // mpg123 trims as the LAME tag says, as this crate does.
        if let Some(oracle) = &mpg123 {
            oracle_compare(oracle, &bytes, &label);
        }
        // ffmpeg trims only behind a LAME, Lavf or Lavc tag name: compare
        // every decoded sample.
        if let Some(oracle) = &ffmpeg {
            let ours = decode_packets(&demux(&bytes), false).pcm;
            let reference = reference_pcm(oracle, &bytes);
            assert_eq!(ours.len(), reference.len(), "{label}");
            let (rms, max) = difference(&ours, &reference);
            assert!(rms < ISO_RMS && max <= ISO_MAX, "{label}: {rms:e} {max:e}");
        }
    }
}

#[test]
fn lame_decodes_the_encoder_output_to_the_input_length() {
    use std::process::Command;
    let Ok(output) = Command::new("lame").arg("--version").output() else {
        eprintln!("skipped the lame reference: lame was not found");
        return;
    };
    if !output.status.success() {
        eprintln!("skipped the lame reference: lame did not run");
        return;
    }
    let dir = crate::common::temp_dir("lame-decode");
    for (rate, channels, mode) in ORACLE_CONFIGS {
        let config = config(rate, channels, mode);
        let input = music(rate, channels, 1.5);
        let packets = encode(config, &input, 4096);
        let ours = decode(config, &packets);
        let path = dir.join("in.mp3");
        std::fs::write(&path, mux(config, &packets)).expect("write");
        let raw = dir.join("out.raw");
        let status = Command::new("lame")
            .args(["--quiet", "--decode", "-t"])
            .arg(&path)
            .arg(&raw)
            .status()
            .expect("lame");
        assert!(status.success());
        let pcm: Vec<f32> = std::fs::read(&raw)
            .expect("raw")
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from(i16::from_le_bytes(*b)) / 32_768.0)
            .collect();
        assert_eq!(pcm.len(), input.len(), "{config:?}");
        // lame writes 16-bit: the same samples within rounding.
        let worst = pcm
            .iter()
            .zip(&ours)
            .map(|(a, b)| (a - b.clamp(-1.0, 32_767.0 / 32_768.0)).abs())
            .fold(0f32, f32::max);
        assert!(worst <= 1.5 / 32_768.0, "{config:?}: {worst}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Bits per second over the coded frames.
fn kbps(packets: &[Packet], rate: u32) -> f64 {
    let bytes: usize = packets.iter().map(|packet| packet.data.len()).sum();
    let samples: i64 = packets.iter().map(|packet| packet.duration).sum();
    bytes as f64 * 8.0 * f64::from(rate) / samples as f64 / 1000.0
}

/// Gates on the synthetic signal, set from the first measurements of the
/// tuned encoder with a margin; a change that drops below them is a
/// regression to look at, not a threshold to move.
#[test]
fn constant_and_variable_rates_meet_their_quality_and_rate_gates() {
    let input = music(44_100, 2, 3.0);
    for (kbps_nominal, min_snr) in [(64, 20.0), (128, 26.0), (192, 30.0), (320, 32.0)] {
        let config = config(44_100, 2, Mp3Rate::Constant(kbps_nominal));
        let packets = encode(config, &input, 4096);
        let rate = kbps(&packets, 44_100);
        assert!(
            (rate - f64::from(kbps_nominal)).abs() < f64::from(kbps_nominal) * 0.002,
            "{kbps_nominal}: {rate}"
        );
        let snr = snr(&input, &decode(config, &packets));
        eprintln!("cbr {kbps_nominal}: {rate:.2} kbps, {snr:.1} dB");
        assert!(snr >= min_snr, "cbr {kbps_nominal}: {snr}");
    }
    let mut last_rate = f64::MAX;
    for (quality, min_snr) in [(0, 30.0), (2, 28.0), (4, 25.0), (6, 22.0), (9, 15.0)] {
        let config = config(44_100, 2, Mp3Rate::Variable(quality));
        let packets = encode(config, &input, 4096);
        let rate = kbps(&packets, 44_100);
        let snr = snr(&input, &decode(config, &packets));
        eprintln!("vbr {quality}: {rate:.1} kbps, {snr:.1} dB");
        assert!(snr >= min_snr, "vbr {quality}: {snr}");
        assert!(rate < last_rate, "vbr {quality}: {rate} after {last_rate}");
        assert!((32.0..=320.0).contains(&rate), "vbr {quality}: {rate}");
        last_rate = rate;
    }
}

#[test]
fn silence_codes_to_silence_and_the_smallest_variable_frames() {
    let config = config(44_100, 2, Mp3Rate::Variable(2));
    let input = vec![0f32; 44_100 * 2];
    let packets = encode(config, &input, 4096);
    assert!(
        packets.iter().all(|packet| packet.data.len() <= 105),
        "32 kbit/s frames"
    );
    let decoded = decode(config, &packets);
    assert_eq!(decoded.len(), input.len());
    assert!(decoded.iter().all(|sample| *sample == 0.0));
}

#[test]
fn chunking_does_not_change_the_stream() {
    for mode in [Mp3Rate::Constant(128), Mp3Rate::Variable(3)] {
        let config = config(44_100, 2, mode);
        let input = music(44_100, 2, 1.0);
        let whole = encode(config, &input, input.len());
        for chunk in [1, 333, 1152, 5000] {
            assert_eq!(encode(config, &input, chunk), whole, "{mode:?} by {chunk}");
        }
    }
}

#[test]
fn short_inputs_keep_their_length() {
    for frames in [1usize, 2, 575, 576, 1151, 1152, 1153, 2304] {
        for (rate, channels) in [(44_100, 2), (22_050, 1), (8_000, 2)] {
            let config = config(rate, channels, Mp3Rate::Constant(64));
            let input = music(rate, channels, frames as f32 / rate as f32 + 1e-6);
            let input = &input[..frames * channels];
            let packets = encode(config, input, 4096);
            let spf = packets[0].duration as u32;
            let total = packets.len() as u32 * spf;
            assert_eq!(total - ENCODER_DELAY - padding(&packets), frames as u32);
            assert_eq!(
                decode(config, &packets).len(),
                input.len(),
                "{frames} {config:?}"
            );
        }
    }
}

#[test]
fn an_empty_stream_drains_to_nothing() {
    let mut encoder = Mp3Encoder::new(config(44_100, 2, Mp3Rate::Constant(128))).expect("open");
    assert!(encoder.drain().expect("drain").is_empty());
    assert!(encoder.drain().expect("drain").is_empty());
}

fn input(rate: u32, channels: usize, samples: Vec<f32>) -> AudioInput {
    AudioInput {
        pcm: Pcm::from_f32(rate, layout(channels), samples).expect("pcm"),
    }
}

#[test]
fn the_lifecycle_follows_the_encoder_contract() {
    use cutlass_decode::CodecError;
    let config = config(44_100, 2, Mp3Rate::Constant(128));
    let signal = music(44_100, 2, 0.5);
    let reference = encode(config, &signal, 4096);

    let mut encoder = Mp3Encoder::new(config).expect("open");
    let mut packets = encoder
        .push(&input(44_100, 2, signal.clone()))
        .expect("push");
    packets.extend(encoder.drain().expect("drain"));
    assert_eq!(packets, reference);
    assert!(encoder.drain().expect("second drain").is_empty());
    assert!(matches!(
        encoder.push(&input(44_100, 2, signal.clone())),
        Err(CodecError::InvalidApi { .. })
    ));

    // Reset after drain, and reset mid-stream, start a fresh stream.
    encoder.reset();
    let _ = encoder
        .push(&input(44_100, 2, signal[..20_000].to_vec()))
        .expect("push");
    encoder.reset();
    let mut again = encoder
        .push(&input(44_100, 2, signal.clone()))
        .expect("push");
    again.extend(encoder.drain().expect("drain"));
    assert_eq!(again, reference);
}

#[test]
fn a_refused_push_adds_nothing() {
    use cutlass_decode::CodecError;
    let config = config(44_100, 2, Mp3Rate::Variable(4));
    let signal = music(44_100, 2, 0.5);
    let reference = encode(config, &signal, 4096);
    let mut encoder = Mp3Encoder::new(config).expect("open");
    let (head, tail) = signal.split_at(10_000);
    let mut packets = encoder
        .push(&input(44_100, 2, head.to_vec()))
        .expect("push");

    let mut bad = tail[..100].to_vec();
    bad[37] = f32::NAN;
    assert_eq!(
        encoder.push(&input(44_100, 2, bad.clone())),
        Err(CodecError::NonFinite)
    );
    bad[37] = 1.5;
    assert_eq!(
        encoder.push(&input(44_100, 2, bad)),
        Err(CodecError::OutOfRange)
    );
    assert!(matches!(
        encoder.push(&input(48_000, 2, tail[..100].to_vec())),
        Err(CodecError::InvalidApi { .. })
    ));
    assert!(matches!(
        encoder.push(&input(44_100, 1, tail[..100].to_vec())),
        Err(CodecError::InvalidApi { .. })
    ));
    let frame = cutlass_encode::FrameInput::Platform(
        cutlass_encode::PlatformFrame::new(
            16,
            16,
            0,
            cutlass_encode::FrameTime::new(0, 1, 30).expect("time"),
        )
        .expect("frame"),
    );
    assert!(matches!(
        encoder.push_frame(&frame),
        Err(CodecError::Unsupported { .. })
    ));

    packets.extend(
        encoder
            .push(&input(44_100, 2, tail.to_vec()))
            .expect("push"),
    );
    packets.extend(encoder.drain().expect("drain"));
    assert_eq!(packets, reference);
}

#[test]
fn sixteen_bit_input_codes_as_its_float_value() {
    let config = config(32_000, 2, Mp3Rate::Constant(96));
    let ints: Vec<i16> = music(32_000, 2, 0.5)
        .iter()
        .map(|s| (s.clamp(-1.0, 1.0) * 32_767.0) as i16)
        .collect();
    let floats: Vec<f32> = ints.iter().map(|s| f32::from(*s) / 32_768.0).collect();
    let mut encoder = Mp3Encoder::new(config).expect("open");
    let pcm = Pcm::from_i16(32_000, layout(2), ints).expect("pcm");
    let mut packets = encoder.push(&AudioInput { pcm }).expect("push");
    packets.extend(encoder.drain().expect("drain"));
    assert_eq!(packets, encode(config, &floats, 1 << 20));
}

#[test]
fn a_raised_cancel_flag_stops_the_encoder_between_frames() {
    use cutlass_decode::CodecError;
    let mut encoder = Mp3Encoder::new(config(44_100, 2, Mp3Rate::Constant(128))).expect("open");
    encoder
        .cancel_flag()
        .store(true, std::sync::atomic::Ordering::Relaxed);
    assert_eq!(
        encoder.push(&input(44_100, 2, music(44_100, 2, 0.2))),
        Err(CodecError::Cancelled)
    );
}

#[test]
fn opening_from_codec_parameters_picks_the_rate_mode() {
    use cutlass_decode::{CodecError, CodecId, CodecParams};
    let mut factory = cutlass_encode::EncodeFactory::new();
    cutlass_mp3::register_encoders(&mut factory);
    let signal = music(44_100, 2, 1.0);
    let run = |params: &CodecParams| {
        let mut encoder = factory.open(params).expect("open");
        let mut packets = encoder
            .push(&input(44_100, 2, signal.clone()))
            .expect("push");
        packets.extend(encoder.drain().expect("drain"));
        packets
    };
    let mut params = CodecParams::audio(CodecId::Mp3, 44_100, ChannelLayout::stereo());
    params.bitrate = 130_000;
    params.constant_rate = true;
    let constant = run(&params);
    assert!(constant.iter().all(|p| matches!(p.data.len(), 417 | 418)));
    params.constant_rate = false;
    params.bitrate = 0;
    let default = run(&params);
    assert_eq!(
        default,
        encode(config(44_100, 2, Mp3Rate::Variable(2)), &signal, 4096)
    );

    for (rate, layout, codec) in [
        (44_000, ChannelLayout::stereo(), CodecId::Mp3),
        (44_100, ChannelLayout::five_one(), CodecId::Mp3),
        (44_100, ChannelLayout::stereo(), CodecId::Aac),
    ] {
        let params = CodecParams::audio(codec, rate, layout);
        assert!(matches!(
            Mp3Encoder::open(&params),
            Err(CodecError::Unsupported { .. })
        ));
    }
    assert!(matches!(
        Mp3Encoder::new(config(44_100, 2, Mp3Rate::Constant(130))),
        Err(CodecError::Unsupported { .. })
    ));
    assert!(matches!(
        Mp3Encoder::new(config(22_050, 2, Mp3Rate::Constant(192))),
        Err(CodecError::Unsupported { .. })
    ));
    assert!(matches!(
        Mp3Encoder::new(config(44_100, 2, Mp3Rate::Variable(10))),
        Err(CodecError::Unsupported { .. })
    ));
}

#[test]
fn full_scale_square_waves_and_clicks_stay_in_range() {
    for mode in [
        Mp3Rate::Constant(32),
        Mp3Rate::Constant(320),
        Mp3Rate::Variable(0),
    ] {
        let config = config(48_000, 2, mode);
        let input: Vec<f32> = (0..48_000 * 2)
            .map(|i| if (i / 2 / 37) % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let packets = encode(config, &input, 4096);
        let decoded = decode(config, &packets);
        assert_eq!(decoded.len(), input.len());
        assert!(
            decoded.iter().all(|s| s.is_finite() && s.abs() < 2.0),
            "{mode:?}"
        );
    }
}
