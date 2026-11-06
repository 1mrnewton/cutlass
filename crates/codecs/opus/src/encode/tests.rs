//! Encoder lifecycle and packet framing.

use std::sync::atomic::Ordering;

use cutlass_encode::{
    AudioInput, ChannelLayout, CodecError, CodecId, CodecParams, Encoder, FrameInput, FrameTime,
    Packet, Pcm, PlatformFrame, SideData,
};

use super::{FrameDuration, Options, OpusEncoder, PRE_SKIP};
use crate::celt::{CeltDecoder, CeltFrame};
use crate::header::OpusHead;
use crate::packet;
use crate::range::RangeDecoder;

fn layout(channels: usize) -> ChannelLayout {
    if channels == 1 {
        ChannelLayout::mono()
    } else {
        ChannelLayout::stereo()
    }
}

fn params(channels: usize, bitrate: u32, cbr: bool) -> CodecParams {
    let mut p = CodecParams::audio(CodecId::Opus, 48_000, layout(channels));
    p.bitrate = bitrate;
    p.constant_rate = cbr;
    p
}

fn tone(samples: usize, channels: usize, offset: usize) -> Vec<f32> {
    (0..samples)
        .flat_map(|i| {
            let t = (i + offset) as f32 / 48_000.0;
            let v = 0.5 * (t * 440.0 * std::f32::consts::TAU).sin();
            std::iter::repeat_n(v, channels)
        })
        .collect()
}

fn input(channels: usize, samples: Vec<f32>) -> AudioInput {
    AudioInput {
        pcm: Pcm::from_f32(48_000, layout(channels), samples).expect("pcm"),
    }
}

fn encode_all(enc: &mut OpusEncoder, channels: usize, total: usize, chunk: usize) -> Vec<Packet> {
    let mut packets = Vec::new();
    let mut done = 0;
    while done < total {
        let take = chunk.min(total - done);
        packets.extend(
            enc.push(&input(channels, tone(take, channels, done)))
                .expect("push"),
        );
        done += take;
    }
    packets.extend(enc.drain().expect("drain"));
    packets
}

#[test]
fn open_checks_the_parameters() {
    assert!(OpusEncoder::open(&params(1, 0, false)).is_ok());
    assert!(OpusEncoder::open(&params(2, 1, false)).is_ok());
    let mut p = params(2, 0, false);
    p.sample_rate = 44_100;
    assert!(matches!(
        OpusEncoder::open(&p),
        Err(CodecError::Unsupported { .. })
    ));
    let mut p = params(2, 0, false);
    p.layout = Some(ChannelLayout::five_one());
    assert!(matches!(
        OpusEncoder::open(&p),
        Err(CodecError::Unsupported { .. })
    ));
    let mut p = params(2, 0, false);
    p.codec = CodecId::Aac;
    assert!(matches!(
        OpusEncoder::open(&p),
        Err(CodecError::Unsupported { .. })
    ));
    let mut p = params(2, 0, false);
    p.layout = None;
    assert!(OpusEncoder::open(&p).is_err());
}

#[test]
fn packets_carry_the_header_times_and_end_trim() {
    for channels in [1usize, 2] {
        for total in [1usize, 959, 960, 961, 48_000, 48_123] {
            let mut enc = OpusEncoder::open(&params(channels, 0, false)).expect("open");
            let packets = encode_all(&mut enc, channels, total, 777);
            let sum: i64 = packets.iter().map(|p| p.duration).sum();
            assert_eq!(sum, total as i64 + i64::from(PRE_SKIP));
            for (i, p) in packets.iter().enumerate() {
                assert_eq!(p.pts, i as i64 * 960);
                assert_eq!(p.timescale, 48_000);
                assert!(p.keyframe);
                let frames = packet::parse(&p.data, false).expect("parse");
                assert_eq!(frames.toc.frame_samples, 960);
                assert_eq!(frames.toc.stereo, channels == 2);
                assert_eq!(frames.toc.code, 0);
                assert!(p.duration > 0 && p.duration <= 960);
            }
            let head = packets[0]
                .side
                .iter()
                .find_map(|s| match s {
                    SideData::Config(c) => Some(c.clone()),
                    _ => None,
                })
                .expect("config");
            assert_eq!(
                head,
                OpusHead::simple(channels as u8, PRE_SKIP, 48_000).to_ogg()
            );
            let last = packets.last().expect("packets");
            let padding = 960 * packets.len() as u32 - total as u32 - u32::from(PRE_SKIP);
            assert!(last.side.contains(&SideData::Gapless {
                encoder_delay: u32::from(PRE_SKIP),
                trailing_padding: padding,
                sample_rate: 48_000,
            }));
            assert!(enc.drain().expect("second drain").is_empty());
            assert!(enc.push(&input(channels, tone(10, channels, 0))).is_err());
        }
    }
}

#[test]
fn buffering_stays_under_two_frames() {
    let mut enc = OpusEncoder::open(&params(2, 0, false)).expect("open");
    let mut pushed = 0usize;
    let mut emitted = 0usize;
    for chunk in [1usize, 100, 959, 2000, 5000, 3, 960] {
        emitted += enc
            .push(&input(2, tone(chunk, 2, pushed)))
            .expect("push")
            .len();
        pushed += chunk;
        assert!(pushed - emitted * 960 < 2 * 960);
    }
    assert!(enc.push(&input(2, Vec::new())).expect("empty").is_empty());
}

#[test]
fn bad_input_is_refused_and_consumes_nothing() {
    let clean = {
        let mut enc = OpusEncoder::open(&params(2, 0, false)).expect("open");
        encode_all(&mut enc, 2, 5000, 5000)
    };
    let mut enc = OpusEncoder::open(&params(2, 0, false)).expect("open");
    let mut nan = tone(500, 2, 0);
    nan[333] = f32::NAN;
    assert_eq!(
        enc.push(&input(2, nan)).expect_err("nan"),
        CodecError::NonFinite
    );
    let mut loud = tone(500, 2, 0);
    loud[3] = 1.5;
    assert_eq!(
        enc.push(&input(2, loud)).expect_err("loud"),
        CodecError::OutOfRange
    );
    let mut uneven = input(2, tone(500, 2, 0));
    uneven.pcm.samples = 499;
    assert_eq!(
        enc.push(&uneven).expect_err("uneven"),
        CodecError::UnevenChannels
    );
    let mono = input(1, tone(500, 1, 0));
    assert!(enc.push(&mono).is_err());
    let picture = FrameInput::Platform(
        PlatformFrame::new(16, 16, 1, FrameTime::new(0, 1, 30).expect("time")).expect("frame"),
    );
    assert!(matches!(
        enc.push_frame(&picture),
        Err(CodecError::Unsupported { .. })
    ));
    let i16_input = AudioInput {
        pcm: Pcm::from_i16(48_000, layout(2), vec![0; 20]).expect("pcm"),
    };
    let mut other = OpusEncoder::open(&params(2, 0, false)).expect("open");
    assert!(other.push(&i16_input).is_ok());
    assert_eq!(encode_all(&mut enc, 2, 5000, 5000), clean);
}

#[test]
fn reset_and_cancel_leave_a_clean_encoder() {
    let clean = {
        let mut enc = OpusEncoder::open(&params(1, 32_000, false)).expect("open");
        encode_all(&mut enc, 1, 20_000, 3000)
    };
    let mut enc = OpusEncoder::open(&params(1, 32_000, false)).expect("open");
    encode_all(&mut enc, 1, 7000, 3000);
    enc.reset();
    assert_eq!(encode_all(&mut enc, 1, 20_000, 3000), clean);

    let mut enc = OpusEncoder::open(&params(1, 32_000, false)).expect("open");
    let flag = enc.cancel_flag();
    flag.store(true, Ordering::Relaxed);
    assert_eq!(
        enc.push(&input(1, tone(5000, 1, 0)))
            .expect_err("cancelled"),
        CodecError::Cancelled
    );
    assert_eq!(enc.drain().expect_err("cancelled"), CodecError::Cancelled);
    flag.store(false, Ordering::Relaxed);
    assert_eq!(encode_all(&mut enc, 1, 20_000, 3000), clean);
}

#[test]
fn cbr_packets_have_the_exact_size() {
    for (channels, bitrate) in [(1usize, 6_000u32), (1, 32_000), (2, 64_000), (2, 510_000)] {
        let mut enc = OpusEncoder::open(&params(channels, bitrate, true)).expect("open");
        let packets = encode_all(&mut enc, channels, 30_000, 4096);
        let want = (bitrate * 20 / 1000 / 8) as usize;
        assert!(packets.iter().all(|p| p.data.len() == want), "{bitrate}");
    }
}

/// Decodes packets with this crate's CELT decoder, without the pre-skip.
fn decode(packets: &[Packet], channels: usize) -> Vec<f32> {
    let mut dec = CeltDecoder::new(channels, 48_000).expect("decoder");
    let mut out = vec![0f32; 960 * channels];
    let mut all = Vec::new();
    for p in packets {
        let frames = packet::parse(&p.data, false).expect("parse");
        let toc = frames.toc;
        for &(off, len) in &frames.frames {
            let mut rd = RangeDecoder::new(&p.data[off..off + len]);
            let frame = CeltFrame {
                rd: Some(&mut rd),
                frame_size: toc.frame_samples as usize,
                stream_channels: if toc.stereo { 2 } else { 1 },
                start_band: 0,
                end_band: toc.bandwidth.celt_end_band(),
            };
            dec.decode(frame, &mut out).expect("decode");
            all.extend_from_slice(&out[..toc.frame_samples as usize * channels]);
        }
    }
    all.drain(..usize::from(PRE_SKIP) * channels);
    all
}

#[test]
fn every_frame_duration_decodes_aligned() {
    for frame in [
        FrameDuration::Ms2_5,
        FrameDuration::Ms5,
        FrameDuration::Ms10,
        FrameDuration::Ms20,
    ] {
        let options = Options {
            frame,
            constrained: true,
        };
        let mut enc = OpusEncoder::open_with(&params(2, 128_000, false), options).expect("open");
        let total = 24_000;
        let packets = encode_all(&mut enc, 2, total, 1000);
        let sum: i64 = packets.iter().map(|p| p.duration).sum();
        assert_eq!(sum, total as i64 + i64::from(PRE_SKIP));
        let decoded = decode(&packets, 2);
        let reference = tone(total, 2, 0);
        let mut sig = 0f64;
        let mut err = 0f64;
        for i in 9600..total * 2 {
            sig += f64::from(reference[i]).powi(2);
            err += f64::from(reference[i] - decoded[i]).powi(2);
        }
        let snr = 10.0 * (sig / err).log10();
        assert!(snr > 20.0, "{frame:?}: {snr} dB");
    }
}
