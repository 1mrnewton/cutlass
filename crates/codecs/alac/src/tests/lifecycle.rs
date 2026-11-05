//! The decoder and encoder through the workspace traits: opening, every
//! call in order and out of order, errors that must leave state intact,
//! and the conversions between integers and mix PCM.

use std::sync::atomic::Ordering;

use cutlass_decode::{
    ChannelLayout, CodecError, CodecId, DecodeFactory, Decoder, Output, Packet, Pcm, SampleFormat,
    SideData,
};
use cutlass_encode::{AudioInput, EncodeFactory, Encoder};

use super::{encode_all, params, synth};
use crate::{
    AlacConfig, AlacDecoder, AlacEncoder, Cookie, EncodeMode, EncoderConfig, default_layout,
};

const RATE: u32 = 44_100;

fn stream(depth: u8, frames: usize) -> (Vec<i32>, Vec<u8>, Vec<Packet>) {
    let samples = synth(2, frames, u32::from(depth), 11);
    let config = EncoderConfig::new(RATE, 2, depth);
    let (cookie, packets) = encode_all(config, &ChannelLayout::stereo(), &samples);
    (samples, cookie, packets)
}

fn audio(outputs: Vec<Output>) -> cutlass_decode::AudioOutput {
    let mut outputs = outputs.into_iter();
    let Some(Output::Audio(audio)) = outputs.next() else {
        panic!("not audio");
    };
    assert!(outputs.next().is_none());
    audio
}

fn decoder(cookie: &[u8]) -> Box<dyn Decoder> {
    let mut factory = DecodeFactory::new();
    crate::register_decoders(&mut factory);
    factory
        .open(&params(cookie, ChannelLayout::stereo()))
        .expect("open")
}

#[test]
fn the_registered_decoder_gives_exact_sixteen_bit_pcm_at_each_packet_time() {
    let (samples, cookie, packets) = stream(16, 10_000);
    let mut decoder = decoder(&cookie);
    assert_eq!(decoder.codec(), CodecId::Alac);
    let mut out = Vec::new();
    let mut clock = 0;
    for packet in &packets {
        let audio = audio(decoder.push(packet).expect("push"));
        assert_eq!(audio.pts, clock);
        assert_eq!(audio.timescale, RATE);
        assert_eq!(audio.pcm.format, SampleFormat::I16);
        assert_eq!(audio.pcm.rate, RATE);
        let pcm = audio.pcm.as_i16().expect("i16");
        clock += i64::try_from(pcm.len() / 2).expect("len");
        out.extend(pcm.iter().map(|&v| i32::from(v)));
    }
    assert_eq!(out, samples);
    assert!(decoder.drain().expect("drain").is_empty());
}

#[test]
fn a_padded_last_frame_is_trimmed_to_its_packet_duration() {
    let (samples, cookie, packets) = stream(16, 4096);
    let mut decoder = decoder(&cookie);
    let mut short = packets[0].clone();
    short.duration = 1000;
    let audio = audio(decoder.push(&short).expect("push"));
    let pcm = audio.pcm.as_i16().expect("i16");
    assert_eq!(pcm.len(), 2000);
    assert!(pcm.iter().zip(&samples).all(|(&a, &b)| i32::from(a) == b));
    // Another clock, or no duration, keeps the whole frame.
    let mut other_clock = short.clone();
    other_clock.timescale = 48_000;
    let audio = self::audio(decoder.push(&other_clock).expect("push"));
    assert_eq!(audio.pcm.samples, 4096);
    short.duration = 0;
    let audio = self::audio(decoder.push(&short).expect("push"));
    assert_eq!(audio.pcm.samples, 4096);
}

#[test]
fn twenty_four_bit_pcm_is_exact_in_float() {
    let (samples, cookie, packets) = stream(24, 5000);
    let mut decoder = decoder(&cookie);
    let mut out = Vec::new();
    for packet in &packets {
        let audio = audio(decoder.push(packet).expect("push"));
        assert_eq!(audio.pcm.format, SampleFormat::F32);
        out.extend(
            audio
                .pcm
                .as_f32()
                .expect("f32")
                .iter()
                .map(|&v| (f64::from(v) * f64::from(1u32 << 23)) as i32),
        );
    }
    assert_eq!(out, samples);
}

#[test]
fn a_failed_packet_changes_nothing() {
    let (_, cookie, packets) = stream(16, 9000);
    let mut decoder = AlacDecoder::new(&cookie).expect("decoder");
    let first = decoder.decode_frame(&packets[0].data).expect("first");
    assert!(matches!(
        decoder.decode_frame(&[]),
        Err(CodecError::InvalidApi { .. })
    ));
    let half = &packets[1].data[..packets[1].data.len() / 2];
    assert!(decoder.decode_frame(half).is_err());
    let mut flipped = packets[1].data.clone();
    flipped[0] ^= 0xe0;
    assert!(decoder.decode_frame(&flipped).is_err());
    assert_eq!(
        decoder.decode_frame(&packets[0].data).expect("again"),
        first
    );
    decoder.reset();
    assert_eq!(
        decoder.decode_frame(&packets[0].data).expect("reset"),
        first
    );

    // The same through the trait, with an empty packet.
    let mut boxed = self::decoder(&cookie);
    let empty = Packet::audio(Vec::new(), 0, 0, RATE);
    assert!(matches!(
        boxed.push(&empty),
        Err(CodecError::InvalidApi { .. })
    ));
    let audio = audio(boxed.push(&packets[0]).expect("push"));
    let ints: Vec<i32> = audio
        .pcm
        .as_i16()
        .expect("i16")
        .iter()
        .map(|&v| i32::from(v))
        .collect();
    assert_eq!(ints, first.samples);
}

#[test]
fn configuration_side_data_must_describe_the_same_stream() {
    let (_, cookie, packets) = stream(16, 5000);
    let mut decoder = decoder(&cookie);
    let mut same = packets[0].clone();
    same.side = vec![SideData::Config(cookie.clone())];
    decoder.push(&same).expect("same cookie");

    let mut other = Cookie::parse(&cookie).expect("cookie");
    other.config.bit_depth = 24;
    let mut changed = packets[0].clone();
    changed.side = vec![SideData::Config(other.to_bytes())];
    assert!(matches!(
        decoder.push(&changed),
        Err(CodecError::Unsupported { .. })
    ));
    let mut broken = packets[0].clone();
    broken.side = vec![SideData::Config(vec![0; 5])];
    assert!(decoder.push(&broken).is_err());
    decoder.push(&packets[0]).expect("still decodes");
}

#[test]
fn opening_checks_the_parameters() {
    let (_, cookie, _) = stream(16, 4096);
    let mut wrong = params(&cookie, ChannelLayout::stereo());
    wrong.codec = CodecId::Flac;
    assert!(matches!(
        AlacDecoder::open(&wrong),
        Err(CodecError::Unsupported { .. })
    ));
    let mut empty = params(&cookie, ChannelLayout::stereo());
    empty.extra.clear();
    assert!(matches!(
        AlacDecoder::open(&empty),
        Err(CodecError::InvalidApi { .. })
    ));
    assert!(matches!(
        AlacDecoder::open(&params(&cookie, ChannelLayout::five_one())),
        Err(CodecError::Malformed { .. })
    ));

    // A cookie without a rate takes the stream's.
    let mut zero = Cookie::parse(&cookie).expect("cookie");
    zero.config.sample_rate = 0;
    let mut stream_rate = params(&zero.to_bytes(), ChannelLayout::stereo());
    stream_rate.sample_rate = 32_000;
    let decoder = AlacDecoder::open(&stream_rate).expect("open");
    assert_eq!(decoder.cookie().config.sample_rate, 0);
    assert!(AlacDecoder::new(&zero.to_bytes()).is_err());

    // The MP4 box form opens too, and gives the same frames.
    let mut boxed = vec![0, 0, 0, 0];
    boxed.extend_from_slice(&cookie);
    let decoder = AlacDecoder::new(&boxed).expect("boxed");
    assert_eq!(
        decoder.cookie().config,
        Cookie::parse(&cookie).expect("cookie").config
    );
}

#[test]
fn five_one_comes_out_labelled_in_frame_order() {
    let cookie = Cookie::new(AlacConfig::new(48_000, 6, 16, 4096)).to_bytes();
    let decoder = AlacDecoder::new(&cookie).expect("decoder");
    assert_eq!(decoder.layout(), &default_layout(6));
    assert_eq!(decoder.cookie().layout_tag, Some(crate::LAYOUT_TAGS[5]));
}

fn encoder(sample_format: SampleFormat, layout: ChannelLayout) -> Box<dyn Encoder> {
    let mut factory = EncodeFactory::new();
    crate::register_encoders(&mut factory);
    let mut params = cutlass_decode::CodecParams::audio(CodecId::Alac, RATE, layout);
    params.sample_format = Some(sample_format);
    factory.open(&params).expect("open")
}

fn i16_input(samples: &[i32], layout: ChannelLayout) -> AudioInput {
    let pcm = Pcm::from_i16(
        RATE,
        layout,
        samples
            .iter()
            .map(|&v| i16::try_from(v).expect("i16"))
            .collect(),
    )
    .expect("pcm");
    AudioInput { pcm }
}

#[test]
fn the_registered_encoder_writes_a_stream_with_its_cookie_first_and_last() {
    let samples = synth(2, 10_000, 16, 5);
    let mut encoder = encoder(SampleFormat::I16, ChannelLayout::stereo());
    assert_eq!(encoder.codec(), CodecId::Alac);
    let mut packets = Vec::new();
    for piece in samples.chunks(2 * 1500) {
        packets.extend(
            encoder
                .push(&i16_input(piece, ChannelLayout::stereo()))
                .expect("push"),
        );
    }
    packets.extend(encoder.drain().expect("drain"));
    assert!(encoder.drain().expect("drain twice").is_empty());
    assert_eq!(packets.len(), 3);
    let mut clock = 0;
    for packet in &packets {
        assert_eq!(packet.pts, clock);
        assert_eq!(packet.timescale, RATE);
        assert!(packet.keyframe);
        clock += packet.duration;
    }
    assert_eq!(clock, 10_000);
    let configs: Vec<&Vec<u8>> = packets
        .iter()
        .flat_map(|packet| &packet.side)
        .filter_map(|side| match side {
            SideData::Config(cookie) => Some(cookie),
            _ => None,
        })
        .collect();
    assert_eq!(configs.len(), 2, "first packet and the drained one");
    let first = Cookie::parse(configs[0]).expect("first");
    let last = Cookie::parse(configs[1]).expect("last");
    assert_eq!(first.config.bit_depth, 16);
    assert_eq!(first.config.max_frame_bytes, 0);
    let largest = packets.iter().map(|p| p.data.len()).max().expect("max");
    assert_eq!(last.config.max_frame_bytes as usize, largest);
    assert!(last.config.avg_bit_rate > 0);
    assert!(last.config.decodes_like(&first.config));

    let mut decoder = AlacDecoder::new(configs[1]).expect("decoder");
    let decoded: Vec<i32> = packets
        .iter()
        .flat_map(|packet| decoder.decode_frame(&packet.data).expect("frame").samples)
        .collect();
    assert_eq!(decoded, samples);
}

#[test]
fn reset_starts_a_new_stream_that_encodes_as_the_first() {
    let samples = synth(2, 9000, 16, 8);
    let mut encoder = encoder(SampleFormat::I16, ChannelLayout::stereo());
    let input = i16_input(&samples, ChannelLayout::stereo());
    let mut first = encoder.push(&input).expect("push");
    first.extend(encoder.drain().expect("drain"));
    // Leave samples buffered, then reset.
    encoder
        .push(&i16_input(&samples[..200], ChannelLayout::stereo()))
        .expect("push");
    encoder.reset();
    let mut second = encoder.push(&input).expect("push");
    second.extend(encoder.drain().expect("drain"));
    assert_eq!(first, second);
}

#[test]
fn float_input_is_checked_and_rounds_to_the_grid() {
    let mut encoder = encoder(SampleFormat::F32, ChannelLayout::stereo());
    let float = |samples: Vec<f32>| AudioInput {
        pcm: Pcm::from_f32(RATE, ChannelLayout::stereo(), samples).expect("pcm"),
    };
    assert!(matches!(
        encoder.push(&float(vec![0.0, f32::NAN])),
        Err(CodecError::NonFinite)
    ));
    assert!(matches!(
        encoder.push(&float(vec![0.0, 1.5])),
        Err(CodecError::OutOfRange)
    ));
    let other_rate = AudioInput {
        pcm: Pcm::from_f32(48_000, ChannelLayout::stereo(), vec![0.0; 2]).expect("pcm"),
    };
    assert!(matches!(
        encoder.push(&other_rate),
        Err(CodecError::InvalidApi { .. })
    ));
    let mono = AudioInput {
        pcm: Pcm::from_f32(RATE, ChannelLayout::mono(), vec![0.0; 2]).expect("pcm"),
    };
    assert!(matches!(
        encoder.push(&mono),
        Err(CodecError::InvalidApi { .. })
    ));

    // 24-bit floats the decoder wrote encode back to the same integers,
    // and the extremes clamp to the 24-bit range.
    let samples = synth(2, 5000, 24, 3);
    let scale = 1.0 / f64::from(1u32 << 23);
    let mut floats: Vec<f32> = samples
        .iter()
        .map(|&v| (f64::from(v) * scale) as f32)
        .collect();
    floats[0] = 1.0;
    floats[1] = -1.0;
    let mut expected = samples.clone();
    expected[0] = (1 << 23) - 1;
    expected[1] = -(1 << 23);
    let mut packets = encoder.push(&float(floats)).expect("push");
    packets.extend(encoder.drain().expect("drain"));
    let cookie = packets[0]
        .side
        .iter()
        .find_map(|side| match side {
            SideData::Config(cookie) => Some(cookie.clone()),
            _ => None,
        })
        .expect("cookie");
    assert_eq!(Cookie::parse(&cookie).expect("cookie").config.bit_depth, 24);
    let mut decoder = AlacDecoder::new(&cookie).expect("decoder");
    let decoded: Vec<i32> = packets
        .iter()
        .flat_map(|packet| decoder.decode_frame(&packet.data).expect("frame").samples)
        .collect();
    assert_eq!(decoded, expected);
}

#[test]
fn integer_input_is_checked() {
    let mut encoder =
        AlacEncoder::new(EncoderConfig::new(RATE, 2, 20), &ChannelLayout::stereo()).expect("new");
    assert!(matches!(
        encoder.push_samples(&[0, 0, 0]),
        Err(CodecError::UnevenChannels)
    ));
    assert!(matches!(
        encoder.push_samples(&[0, 1 << 19]),
        Err(CodecError::OutOfRange)
    ));
    assert!(encoder.push_samples(&[-(1 << 19), (1 << 19) - 1]).is_ok());
}

#[test]
fn bad_configurations_are_refused() {
    let stereo = ChannelLayout::stereo();
    for config in [
        EncoderConfig::new(0, 2, 16),
        EncoderConfig::new(RATE, 2, 18),
        EncoderConfig {
            frame_length: 0,
            ..EncoderConfig::new(RATE, 2, 16)
        },
        EncoderConfig {
            frame_length: crate::MAX_FRAME_LENGTH + 1,
            ..EncoderConfig::new(RATE, 2, 16)
        },
    ] {
        assert!(AlacEncoder::new(config, &stereo).is_err(), "{config:?}");
    }
    assert!(AlacEncoder::new(EncoderConfig::new(RATE, 9, 16), &stereo).is_err());
    assert!(AlacEncoder::new(EncoderConfig::new(RATE, 3, 16), &stereo).is_err());
    // Three channels need the labels of the reference layout.
    let wrong = ChannelLayout::new(vec![
        cutlass_decode::Channel::FrontLeft,
        cutlass_decode::Channel::FrontRight,
        cutlass_decode::Channel::Lfe,
    ])
    .expect("layout");
    assert!(matches!(
        AlacEncoder::new(EncoderConfig::new(RATE, 3, 16), &wrong),
        Err(CodecError::Unsupported { .. })
    ));
    assert!(AlacEncoder::new(EncoderConfig::new(RATE, 3, 16), &default_layout(3)).is_ok());
}

#[test]
fn a_cookie_in_the_parameters_sets_depth_and_frame_length() {
    let cookie = Cookie::new(AlacConfig::new(RATE, 2, 20, 1152)).to_bytes();
    let encoder = AlacEncoder::open(&params(&cookie, ChannelLayout::stereo())).expect("open");
    assert_eq!(encoder.config().bit_depth, 20);
    assert_eq!(encoder.config().frame_length, 1152);
    assert!(matches!(
        AlacEncoder::open(&params(&cookie, ChannelLayout::mono())),
        Err(CodecError::InvalidApi { .. })
    ));
}

#[test]
fn cancel_stops_between_frames() {
    let samples = synth(2, 3 * 4096, 16, 2);
    let mut encoder = encoder(SampleFormat::I16, ChannelLayout::stereo());
    encoder.cancel_flag().store(true, Ordering::Relaxed);
    assert!(matches!(
        encoder.push(&i16_input(&samples, ChannelLayout::stereo())),
        Err(CodecError::Cancelled)
    ));
    encoder.cancel_flag().store(false, Ordering::Relaxed);
    encoder.reset();
    assert_eq!(
        encoder
            .push(&i16_input(&samples, ChannelLayout::stereo()))
            .expect("push")
            .len(),
        3
    );
}

#[test]
fn fast_mode_is_lossless_for_every_depth_and_count() {
    for depth in [16u8, 20, 24, 32] {
        for channels in 1..=8u8 {
            let samples = synth(usize::from(channels), 3000, u32::from(depth), 21);
            let config = EncoderConfig {
                frame_length: 1024,
                mode: EncodeMode::Fast,
                ..EncoderConfig::new(RATE, channels, depth)
            };
            let layout = default_layout(channels);
            let (cookie, packets) = encode_all(config, &layout, &samples);
            let mut decoder = AlacDecoder::new(&cookie).expect("decoder");
            let decoded: Vec<i32> = packets
                .iter()
                .flat_map(|packet| decoder.decode_frame(&packet.data).expect("frame").samples)
                .collect();
            assert_eq!(decoded, samples, "{depth}-bit, {channels} channels");
        }
    }
}

#[test]
fn silence_full_scale_and_a_single_sample_round_trip() {
    for depth in [16u8, 24, 32] {
        let top = i32::try_from((1i64 << (depth - 1)) - 1).expect("top");
        let bottom = -top - 1;
        let square: Vec<i32> = (0..8192)
            .map(|i| if (i / 7) % 2 == 0 { top } else { bottom })
            .collect();
        for samples in [vec![0; 9000], square, vec![top, bottom]] {
            for mode in [EncodeMode::Default, EncodeMode::Fast] {
                let config = EncoderConfig {
                    mode,
                    ..EncoderConfig::new(RATE, 2, depth)
                };
                let (cookie, packets) = encode_all(config, &ChannelLayout::stereo(), &samples);
                let mut decoder = AlacDecoder::new(&cookie).expect("decoder");
                let decoded: Vec<i32> = packets
                    .iter()
                    .flat_map(|packet| decoder.decode_frame(&packet.data).expect("frame").samples)
                    .collect();
                assert_eq!(decoded, samples, "{depth}-bit {mode:?}");
            }
        }
    }
}
