//! Encoder round trips through this crate's decoder, and the lifecycle.

use std::sync::atomic::Ordering;

use cutlass_decode::{
    Channel, ChannelLayout, CodecError, CodecId, CodecParams, Decoder, Output, Packet, Pcm,
    SampleFormat, SideData,
};
use cutlass_encode::{AudioInput, Encoder};

use super::{EncodedFrame, FlacEncoder, Settings, Stereo};
use crate::decode::{FlacDecoder, Md5Check};
use crate::streaminfo::StreamInfo;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn signed(&mut self, bits: u8) -> i64 {
        (self.next() >> (64 - u32::from(bits))) as i64 - (1i64 << (bits - 1))
    }
}

fn music(channels: u8, bits: u8, frames: usize, seed: u64) -> Vec<i32> {
    let mut rng = Rng(seed | 1);
    let high = (1i64 << (bits - 1)) - 1;
    let low = -(1i64 << (bits - 1));
    let amplitude = high as f64 * 0.7;
    let mut out = Vec::with_capacity(frames * usize::from(channels));
    for i in 0..frames {
        let t = i as f64;
        for channel in 0..channels {
            let c = f64::from(channel);
            let tone = (t * (0.011 + 0.003 * c)).sin() * 0.7 + (t * 0.093 + c).sin() * 0.2;
            let noise = rng.signed(bits.min(10)) as f64 / 16.0;
            let value = ((tone * amplitude) + noise) as i64;
            out.push(value.clamp(low, high) as i32);
        }
    }
    out
}

fn roundtrip(samples: &[i32], channels: u8, bits: u8, settings: Settings) -> (usize, StreamInfo) {
    let mut encoder = FlacEncoder::new(44_100, channels, bits, settings).expect("encoder");
    let mut frames: Vec<EncodedFrame> = Vec::new();
    // Uneven pushes exercise the buffer.
    let step = 1000 * usize::from(channels);
    for chunk in samples.chunks(step) {
        frames.extend(encoder.encode(chunk).expect("encode"));
    }
    frames.extend(encoder.finish().expect("finish"));
    let info = encoder.stream_info();
    let mut decoder = FlacDecoder::new(Some(info));
    let mut decoded = Vec::with_capacity(samples.len());
    let mut bytes = 0;
    for (index, frame) in frames.iter().enumerate() {
        bytes += frame.data.len();
        let out = decoder
            .decode(&frame.data)
            .unwrap_or_else(|error| panic!("frame {index}: {error}"));
        assert_eq!(out.header.number, index as u64);
        assert_eq!(u64::from(out.block_size()), u64::from(frame.samples));
        decoded.extend_from_slice(&out.samples);
    }
    assert!(decoded == samples, "round trip differs");
    if !samples.is_empty() {
        assert_eq!(decoder.md5_check(), Md5Check::Match);
    }
    assert_eq!(
        info.total_samples,
        (samples.len() / usize::from(channels)) as u64
    );
    (bytes, info)
}

#[test]
fn every_level_round_trips_and_higher_levels_are_not_larger() {
    let samples = music(2, 16, 30_000, 7);
    let mut sizes = Vec::new();
    for level in 0..=8 {
        let (bytes, _) = roundtrip(&samples, 2, 16, Settings::level(level).expect("level"));
        sizes.push(bytes);
    }
    let raw = samples.len() * 2;
    assert!(sizes[5] * 10 < raw * 9, "level 5 {} of {raw}", sizes[5]);
    assert!(sizes[8] <= sizes[5] + sizes[5] / 100, "{sizes:?}");
    assert!(sizes[5] <= sizes[0], "{sizes:?}");
}

#[test]
fn depths_channel_counts_and_signals_round_trip() {
    for bits in [4u8, 8, 12, 16, 20, 24, 32] {
        for channels in [1u8, 2, 3, 6, 8] {
            let samples = music(
                channels,
                bits,
                5000,
                u64::from(bits) * 31 + u64::from(channels),
            );
            for level in [0, 5, 8] {
                roundtrip(
                    &samples,
                    channels,
                    bits,
                    Settings::level(level).expect("level"),
                );
            }
        }
    }
}

#[test]
fn extremes_noise_silence_and_wasted_bits_round_trip() {
    let mut rng = Rng(99);
    for bits in [8u8, 16, 24, 32] {
        let high = (1i64 << (bits - 1)) - 1;
        let low = -(1i64 << (bits - 1));
        let extremes: Vec<i32> = (0..9000)
            .map(|i| match (i / 5) % 4 {
                0 => high as i32,
                1 => low as i32,
                2 => 0,
                _ => {
                    if i % 2 == 0 {
                        high as i32
                    } else {
                        low as i32
                    }
                }
            })
            .collect();
        let noise: Vec<i32> = (0..9000).map(|_| rng.signed(bits) as i32).collect();
        let silence = vec![0i32; 9000];
        let wasted: Vec<i32> = music(2, bits, 4500, 3)
            .iter()
            .map(|s| (s >> (bits / 3)) << (bits / 3))
            .collect();
        for samples in [&extremes, &noise, &silence, &wasted] {
            for level in [0, 2, 5, 8] {
                roundtrip(samples, 2, bits, Settings::level(level).expect("level"));
            }
        }
    }
}

#[test]
fn incompressible_noise_costs_no_more_than_verbatim() {
    let mut rng = Rng(5);
    let noise: Vec<i32> = (0..8192).map(|_| rng.signed(16) as i32).collect();
    let (bytes, _) = roundtrip(&noise, 1, 16, Settings::level(5).expect("five"));
    // Two frames of verbatim subframes plus headers and footers.
    assert!(bytes <= 8192 * 2 + 64, "{bytes}");
}

#[test]
fn short_streams_and_odd_blocks_round_trip() {
    for frames in [1usize, 2, 15, 16, 17, 4095, 4096, 4097, 8192] {
        let samples = music(2, 16, frames, frames as u64);
        roundtrip(&samples, 2, 16, Settings::level(5).expect("five"));
    }
    let mut settings = Settings::level(8).expect("eight");
    settings.block_size = 16;
    roundtrip(&music(1, 16, 1000, 1), 1, 16, settings.clone());
    settings.block_size = 65_535;
    settings.max_lpc_order = 32;
    settings.max_partition_order = 15;
    settings.exhaustive_order = true;
    roundtrip(&music(1, 16, 70_000, 2), 1, 16, settings);
    roundtrip(&[], 2, 16, Settings::level(5).expect("five"));
}

#[test]
fn every_stereo_mode_round_trips_correlated_channels() {
    let base = music(1, 16, 20_000, 11);
    let samples: Vec<i32> = base
        .iter()
        .enumerate()
        .flat_map(|(i, s)| [*s, (*s + (i as i32 % 3) - 1).clamp(-32_768, 32_767)])
        .collect();
    let mut sizes = Vec::new();
    for stereo in [Stereo::Independent, Stereo::Estimate, Stereo::Exhaustive] {
        let mut settings = Settings::level(5).expect("five");
        settings.stereo = stereo;
        sizes.push(roundtrip(&samples, 2, 16, settings).0);
    }
    assert!(sizes[2] < sizes[0], "{sizes:?}");
    assert!(sizes[1] < sizes[0], "{sizes:?}");
}

#[test]
fn out_of_range_and_uneven_input_add_nothing() {
    let mut encoder = FlacEncoder::new(48_000, 2, 16, Settings::level(0).expect("0")).expect("e");
    assert_eq!(encoder.encode(&[1, 2, 3]), Err(CodecError::UnevenChannels));
    assert_eq!(encoder.encode(&[0, 40_000]), Err(CodecError::OutOfRange));
    assert!(encoder.encode(&[5, 5]).expect("ok").is_empty());
    let frames = encoder.finish().expect("finish");
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].samples, 1);
    assert!(FlacEncoder::new(0, 2, 16, Settings::level(0).expect("0")).is_err());
    assert!(FlacEncoder::new(48_000, 9, 16, Settings::level(0).expect("0")).is_err());
    assert!(FlacEncoder::new(48_000, 2, 3, Settings::level(0).expect("0")).is_err());
    assert!(FlacEncoder::new(48_000, 2, 33, Settings::level(0).expect("0")).is_err());
}

fn params(layout: ChannelLayout, format: Option<SampleFormat>) -> CodecParams {
    let mut params = CodecParams::audio(CodecId::Flac, 48_000, layout);
    params.sample_format = format;
    params
}

fn decode_packets(packets: &[Packet], format: SampleFormat, channels: u8) -> (Pcm, FlacDecoder) {
    let layout = crate::pcm::layout(channels).expect("layout");
    let mut decoder = FlacDecoder::open(&params(layout.clone(), Some(format))).expect("decoder");
    let mut i16s = Vec::new();
    let mut f32s = Vec::new();
    for packet in packets {
        for output in decoder.push(packet).expect("push") {
            let Output::Audio(audio) = output else {
                panic!("audio");
            };
            if let Some(data) = audio.pcm.as_i16() {
                i16s.extend_from_slice(data);
            }
            if let Some(data) = audio.pcm.as_f32() {
                f32s.extend_from_slice(data);
            }
        }
    }
    let pcm = match format {
        SampleFormat::I16 => Pcm::from_i16(48_000, layout, i16s),
        SampleFormat::F32 => Pcm::from_f32(48_000, layout, f32s),
    }
    .expect("pcm");
    (pcm, decoder)
}

#[test]
fn the_trait_carries_streaminfo_and_round_trips_i16() {
    let samples: Vec<i16> = music(2, 16, 10_000, 4).iter().map(|s| *s as i16).collect();
    let mut encoder =
        FlacEncoder::open(&params(ChannelLayout::stereo(), Some(SampleFormat::I16))).expect("e");
    let pcm = Pcm::from_i16(48_000, ChannelLayout::stereo(), samples.clone()).expect("pcm");
    let mut packets = encoder.push(&AudioInput { pcm }).expect("push");
    assert_eq!(packets.len(), 2, "4096-sample frames, the rest buffered");
    packets.extend(encoder.drain().expect("drain"));
    assert!(encoder.drain().expect("second drain").is_empty());
    assert_eq!(packets.len(), 3);
    let configs: Vec<StreamInfo> = packets
        .iter()
        .flat_map(|packet| &packet.side)
        .map(|side| match side {
            SideData::Config(bytes) => StreamInfo::parse(bytes).expect("streaminfo"),
            other => panic!("unexpected side data {other:?}"),
        })
        .collect();
    assert_eq!(configs.len(), 2);
    assert_eq!(configs[0].total_samples, 0);
    assert_eq!(configs[1].total_samples, 10_000);
    assert_ne!(configs[1].md5, [0; 16]);
    let mut clock = 0;
    for packet in &packets {
        assert_eq!(packet.pts, clock);
        assert_eq!(packet.timescale, 48_000);
        clock += packet.duration;
    }
    assert_eq!(clock, 10_000);
    let (pcm, decoder) = decode_packets(&packets, SampleFormat::I16, 2);
    assert_eq!(pcm.as_i16().expect("i16"), samples.as_slice());
    assert_eq!(decoder.md5_check(), Md5Check::Match);
}

#[test]
fn f32_input_is_exact_at_24_bits_and_out_of_range_adds_nothing() {
    let ints = music(1, 24, 9000, 8);
    let floats: Vec<f32> = ints
        .iter()
        .map(|s| crate::pcm::sample_to_f32(*s, 24))
        .collect();
    let mut encoder = FlacEncoder::open(&params(ChannelLayout::mono(), None)).expect("e");
    let bad = Pcm::from_f32(48_000, ChannelLayout::mono(), vec![0.5, 1.5]).expect("pcm");
    assert_eq!(
        encoder.push(&AudioInput { pcm: bad }),
        Err(CodecError::OutOfRange)
    );
    let nan = Pcm::from_f32(48_000, ChannelLayout::mono(), vec![f32::NAN]).expect("pcm");
    assert_eq!(
        encoder.push(&AudioInput { pcm: nan }),
        Err(CodecError::NonFinite)
    );
    let pcm = Pcm::from_f32(48_000, ChannelLayout::mono(), floats.clone()).expect("pcm");
    let mut packets = encoder.push(&AudioInput { pcm }).expect("push");
    packets.extend(encoder.drain().expect("drain"));
    let (pcm, decoder) = decode_packets(&packets, SampleFormat::F32, 1);
    assert_eq!(pcm.as_f32().expect("f32"), floats.as_slice());
    assert_eq!(decoder.md5_check(), Md5Check::Match);
    assert_eq!(decoder.stream_info().map(|info| info.bits), Some(24));
}

#[test]
fn a_layout_in_another_order_is_reordered_to_the_flac_order() {
    // 5.1 in AAC order: C, L, R, BL, BR, LFE. FLAC order: L, R, C, LFE, BL, BR.
    let aac = ChannelLayout::five_one();
    let mut encoder =
        FlacEncoder::open(&params(aac.clone(), Some(SampleFormat::I16))).expect("encoder");
    let frame: [i16; 6] = [3, 1, 2, 5, 6, 4];
    let samples: Vec<i16> = frame.iter().copied().cycle().take(6 * 100).collect();
    let pcm = Pcm::from_i16(48_000, aac, samples).expect("pcm");
    let mut packets = encoder.push(&AudioInput { pcm }).expect("push");
    packets.extend(encoder.drain().expect("drain"));
    let (pcm, _) = decode_packets(&packets, SampleFormat::I16, 6);
    assert_eq!(&pcm.as_i16().expect("i16")[..6], &[1, 2, 3, 4, 5, 6]);
    let odd =
        ChannelLayout::new(vec![Channel::FrontLeft, Channel::FrontLeftOfCenter]).expect("layout");
    assert!(FlacEncoder::open(&params(odd, Some(SampleFormat::I16))).is_err());
}

#[test]
fn reset_starts_a_new_stream_and_push_after_drain_continues() {
    let samples: Vec<i16> = music(1, 16, 5000, 2).iter().map(|s| *s as i16).collect();
    let pcm = Pcm::from_i16(48_000, ChannelLayout::mono(), samples).expect("pcm");
    let input = AudioInput { pcm };
    let mut encoder =
        FlacEncoder::open(&params(ChannelLayout::mono(), Some(SampleFormat::I16))).expect("e");
    let mut first = encoder.push(&input).expect("push");
    first.extend(encoder.drain().expect("drain"));
    // Pushing after drain continues the same stream.
    let more = encoder.push(&input).expect("push");
    assert_eq!(more.first().map(|p| p.pts), Some(5000));
    encoder.reset();
    let mut again = encoder.push(&input).expect("push");
    again.extend(encoder.drain().expect("drain"));
    assert_eq!(first, again);
}

#[test]
fn a_raised_cancel_flag_stops_push_and_drain() {
    let mut encoder =
        FlacEncoder::open(&params(ChannelLayout::mono(), Some(SampleFormat::I16))).expect("e");
    encoder.cancel_flag().store(true, Ordering::Relaxed);
    let pcm = Pcm::from_i16(48_000, ChannelLayout::mono(), vec![0; 10]).expect("pcm");
    assert_eq!(
        encoder.push(&AudioInput { pcm }),
        Err(CodecError::Cancelled)
    );
    assert_eq!(encoder.drain(), Err(CodecError::Cancelled));
}

#[test]
fn open_refuses_mismatched_parameters() {
    let mut wrong = params(ChannelLayout::stereo(), None);
    wrong.codec = CodecId::Pcm;
    assert!(FlacEncoder::open(&wrong).is_err());
    let mut deep = params(ChannelLayout::stereo(), Some(SampleFormat::I16));
    let info = StreamInfo {
        min_block: 4096,
        max_block: 4096,
        min_frame: 0,
        max_frame: 0,
        sample_rate: 48_000,
        channels: 2,
        bits: 24,
        total_samples: 0,
        md5: [0; 16],
    };
    deep.extra = info.to_bytes().to_vec();
    assert!(FlacEncoder::open(&deep).is_err());
    let mut no_layout = params(ChannelLayout::stereo(), None);
    no_layout.layout = None;
    assert!(FlacEncoder::open(&no_layout).is_err());
    let mut i16_input =
        FlacEncoder::open(&params(ChannelLayout::stereo(), Some(SampleFormat::I16))).expect("e");
    let floats = Pcm::from_f32(48_000, ChannelLayout::stereo(), vec![0.0; 2]).expect("pcm");
    assert!(matches!(
        i16_input.push(&AudioInput { pcm: floats }),
        Err(CodecError::InvalidApi { .. })
    ));
}
