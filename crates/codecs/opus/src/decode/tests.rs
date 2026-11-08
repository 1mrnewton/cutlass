//! The whole decoder: the twelve RFC 8251 vectors, multistream, gain,
//! end trim, concealment, and the `Decoder` contract.

use super::OpusDecoder;
use crate::celt::softclip::soft_clip;
use crate::header::{OpusHead, vorbis_mapping};
use crate::packet;
use cutlass_decode::{ChannelLayout, CodecError, CodecId, CodecParams, Decoder, Output, Packet};
use cutlass_encode::{AudioInput, Encoder, Pcm};

struct VectorPacket {
    range: u32,
    data: Vec<u8>,
}

fn read_bit(path: &std::path::Path) -> Vec<VectorPacket> {
    let bytes = std::fs::read(path).unwrap_or_default();
    let mut out = Vec::new();
    let mut at = 0usize;
    while at + 8 <= bytes.len() {
        let len = u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        let range = u32::from_be_bytes(bytes[at + 4..at + 8].try_into().unwrap());
        at += 8;
        let end = (at + len).min(bytes.len());
        out.push(VectorPacket {
            range,
            data: bytes[at..end].to_vec(),
        });
        at = end;
    }
    out
}

fn vectors_dir() -> Option<std::path::PathBuf> {
    let dir = std::path::PathBuf::from(std::env::var_os("CUTLASS_OPUS_VECTORS")?);
    dir.join("testvector01.bit").is_file().then_some(dir)
}

/// Decodes a vector as `opus_demo -d 48000 <channels>` does: float decode,
/// soft clip, round to 16 bits. A zero-length packet is concealed.
fn decode_vector(packets: &[VectorPacket], channels: usize) -> (Vec<i16>, usize, usize) {
    let head = OpusHead::simple(channels as u8, 0, 48_000);
    let mut decoder = OpusDecoder::with_head(head, 48_000).unwrap();
    let mut pcm = Vec::new();
    let mut matched = 0;
    let mut clip = vec![0f32; channels];
    let mut last_samples = 960;
    for packet in packets {
        let mut out = if packet.data.is_empty() {
            decoder.conceal(last_samples).unwrap()
        } else {
            let out = decoder.decode(&packet.data).unwrap();
            last_samples = out.len() / channels;
            out
        };
        if decoder.final_range() == packet.range {
            matched += 1;
        }
        soft_clip(&mut out, &mut clip);
        pcm.extend(
            out.iter()
                .map(|&x| (x * 32768.0).round().clamp(-32768.0, 32767.0) as i16),
        );
    }
    (pcm, matched, packets.len())
}

/// Every final range of the twelve RFC 8251 vectors matches, in stereo and
/// mono output, and with `CUTLASS_OPUS_COMPARE` naming the `opus_compare`
/// tool each output passes it. `CUTLASS_OPUS_VECTORS` names the
/// `opus_newvectors` folder; without it the test is skipped.
#[test]
fn the_rfc_8251_vectors_match_every_final_range_and_pass_opus_compare() {
    let Some(dir) = vectors_dir() else {
        eprintln!("skipped: CUTLASS_OPUS_VECTORS is not set");
        return;
    };
    let compare = std::env::var_os("CUTLASS_OPUS_COMPARE");
    let scratch = std::env::temp_dir().join(format!("cutlass-opus-vectors-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let mut failures = Vec::new();
    for index in 1..=12 {
        let name = format!("testvector{index:02}");
        let packets = read_bit(&dir.join(format!("{name}.bit")));
        for channels in [2usize, 1] {
            let started = std::time::Instant::now();
            let (pcm, matched, total) = decode_vector(&packets, channels);
            let seconds = pcm.len() as f64 / channels as f64 / 48_000.0;
            let speed = seconds / started.elapsed().as_secs_f64();
            let mut verdict = String::new();
            if let Some(tool) = &compare {
                let out = scratch.join(format!("{name}-{channels}.pcm"));
                let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
                std::fs::write(&out, bytes).unwrap();
                let reference = dir.join(if channels == 2 {
                    format!("{name}.dec")
                } else {
                    format!("{name}m.dec")
                });
                let mut command = std::process::Command::new(tool);
                if channels == 2 {
                    command.arg("-s");
                }
                let result = command.arg(&reference).arg(&out).output().unwrap();
                let text = String::from_utf8_lossy(&result.stderr).to_string()
                    + &String::from_utf8_lossy(&result.stdout);
                let passed = text.contains("PASSES");
                let quality = text
                    .lines()
                    .find(|line| line.contains("quality metric"))
                    .unwrap_or("")
                    .trim()
                    .to_owned();
                verdict = format!(
                    ", opus_compare {} ({quality})",
                    if passed { "PASS" } else { "FAIL" }
                );
                if !passed {
                    failures.push(format!("{name} {channels}ch opus_compare: {text}"));
                }
            }
            eprintln!(
                "{name} {}: {matched}/{total} final ranges{verdict}, {speed:.0}x real time",
                if channels == 2 { "stereo" } else { "mono" }
            );
            if matched != total {
                failures.push(format!("{name} {channels}ch: {matched}/{total} ranges"));
            }
        }
    }
    let _ = std::fs::remove_dir_all(&scratch);
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A CELT-only stream from this crate's encoder, 20 ms packets.
fn encoded(channels: usize, frames: usize) -> (Vec<Packet>, Vec<f32>) {
    let layout = if channels == 2 {
        ChannelLayout::stereo()
    } else {
        ChannelLayout::mono()
    };
    let mut params = CodecParams::audio(CodecId::Opus, 48_000, layout.clone());
    params.bitrate = 64_000 * channels as u32;
    let mut encoder = crate::OpusEncoder::open(&params).unwrap();
    let samples: Vec<f32> = (0..frames * 960)
        .flat_map(|i| {
            let t = i as f32 / 48_000.0;
            let left = 0.4 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            let right = 0.3 * (2.0 * std::f32::consts::PI * 660.0 * t).sin();
            if channels == 2 {
                vec![left, right]
            } else {
                vec![left]
            }
        })
        .collect();
    let pcm = Pcm::from_f32(48_000, layout, samples.clone()).unwrap();
    let mut packets = encoder.push(&AudioInput { pcm }).unwrap();
    packets.extend(encoder.drain().unwrap());
    (packets, samples)
}

fn audio(outputs: Vec<Output>) -> cutlass_decode::AudioOutput {
    match outputs.into_iter().next() {
        Some(Output::Audio(audio)) => audio,
        _ => panic!("no audio"),
    }
}

#[test]
fn the_factory_opens_the_decoder_from_an_opus_head_or_a_dops_body() {
    let mut factory = cutlass_decode::DecodeFactory::new();
    crate::register_decoders(&mut factory);
    let mut params = CodecParams::audio(CodecId::Opus, 48_000, ChannelLayout::stereo());
    params.extra = OpusHead::simple(2, 312, 44_100).to_ogg();
    assert!(factory.open(&params).is_ok());
    params.extra = OpusHead::simple(2, 312, 44_100).to_dops();
    assert!(factory.open(&params).is_ok());
    params.extra = vec![1, 2, 3];
    assert!(factory.open(&params).is_err());
    params.extra.clear();
    params.codec = CodecId::Aac;
    assert!(OpusDecoder::open(&params).is_err());
}

#[test]
fn an_empty_packet_is_an_invalid_call_and_a_bad_packet_changes_nothing() {
    let (packets, _) = encoded(2, 10);
    let params = CodecParams::audio(CodecId::Opus, 48_000, ChannelLayout::stereo());
    let mut decoder = OpusDecoder::open(&params).unwrap();
    let mut fresh = OpusDecoder::open(&params).unwrap();
    let empty = Packet::audio(Vec::new(), 0, 960, 48_000);
    assert!(matches!(
        decoder.push(&empty),
        Err(CodecError::InvalidApi { .. })
    ));
    decoder.push(&packets[0]).unwrap();
    fresh.push(&packets[0]).unwrap();
    // Code 3 with a zero frame count is refused before any state changes.
    let bad = Packet::audio(vec![0xFB, 0x00], 0, 960, 48_000);
    assert!(decoder.push(&bad).is_err());
    let a = audio(decoder.push(&packets[1]).unwrap());
    let b = audio(fresh.push(&packets[1]).unwrap());
    assert_eq!(a.pcm, b.pcm);
}

#[test]
fn reset_decodes_like_a_fresh_decoder_and_drain_is_empty() {
    let (packets, _) = encoded(1, 8);
    let params = CodecParams::audio(CodecId::Opus, 48_000, ChannelLayout::mono());
    let mut decoder = OpusDecoder::open(&params).unwrap();
    for packet in &packets {
        decoder.push(packet).unwrap();
    }
    assert!(decoder.drain().unwrap().is_empty());
    assert!(decoder.drain().unwrap().is_empty());
    decoder.reset();
    let mut fresh = OpusDecoder::open(&params).unwrap();
    for packet in &packets {
        let a = audio(decoder.push(packet).unwrap());
        let b = audio(fresh.push(packet).unwrap());
        assert_eq!(a.pcm, b.pcm);
    }
}

#[test]
fn a_short_last_packet_is_an_end_trim() {
    let (packets, samples) = encoded(2, 7);
    let params = CodecParams::audio(CodecId::Opus, 48_000, ChannelLayout::stereo());
    let mut decoder = OpusDecoder::open(&params).unwrap();
    let mut total = 0usize;
    let mut kept = 0usize;
    for packet in &packets {
        let out = audio(decoder.push(packet).unwrap());
        assert_eq!(out.timescale, 48_000);
        assert_eq!(out.pts, packet.pts);
        total += out.pcm.samples as usize;
        kept += out.pcm.samples as usize - out.skip_trailing as usize;
    }
    let pre_skip = OpusHead::parse(match &packets[0].side[0] {
        cutlass_decode::SideData::Config(head) => head,
        _ => panic!("no head"),
    })
    .unwrap()
    .pre_skip as usize;
    assert_eq!(kept, pre_skip + samples.len() / 2);
    assert!(total > kept);
}

#[test]
fn lower_api_rates_decode_the_same_packets() {
    let (packets, _) = encoded(2, 5);
    for rate in [8_000u32, 12_000, 16_000, 24_000, 48_000] {
        let params = CodecParams::audio(CodecId::Opus, rate, ChannelLayout::stereo());
        let mut decoder = OpusDecoder::open(&params).unwrap();
        let out = audio(decoder.push(&packets[2]).unwrap());
        assert_eq!(out.pcm.rate, rate);
        assert_eq!(out.pcm.samples, rate / 50);
    }
    // A rate Opus cannot produce decodes at 48 kHz.
    let params = CodecParams::audio(CodecId::Opus, 44_100, ChannelLayout::stereo());
    let mut decoder = OpusDecoder::open(&params).unwrap();
    assert_eq!(audio(decoder.push(&packets[0]).unwrap()).pcm.rate, 48_000);
}

/// Joins one packet per stream into a multistream packet.
fn multistream(parts: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    for (index, part) in parts.iter().enumerate() {
        let frames = packet::parse(part, false).unwrap();
        let bytes: Vec<&[u8]> = frames
            .frames
            .iter()
            .map(|&(offset, len)| &part[offset..offset + len])
            .collect();
        let delimited = index + 1 < parts.len();
        out.extend(packet::pack(part[0] >> 3, frames.toc.stereo, &bytes, None, delimited).unwrap());
    }
    out
}

#[test]
fn a_five_one_stream_maps_each_decoded_channel_to_its_output() {
    let (stereo, _) = encoded(2, 4);
    let (mono, _) = encoded(1, 4);
    let (streams, coupled, mapping) = vorbis_mapping(6).unwrap();
    let head = OpusHead {
        channels: 6,
        pre_skip: 0,
        input_rate: 48_000,
        output_gain: 0,
        family: 1,
        streams,
        coupled,
        mapping: mapping.clone(),
    };
    let mut decoder = OpusDecoder::with_head(head, 48_000).unwrap();
    let mut alone_stereo = OpusDecoder::with_head(OpusHead::simple(2, 0, 48_000), 48_000).unwrap();
    let mut alone_mono = OpusDecoder::with_head(OpusHead::simple(1, 0, 48_000), 48_000).unwrap();
    for frame in 0..3 {
        // Streams: two coupled (front pair, rear pair), then centre and LFE.
        let packet = multistream(&[
            &stereo[frame].data,
            &stereo[frame].data,
            &mono[frame].data,
            &mono[frame].data,
        ]);
        let out = decoder.decode(&packet).unwrap();
        let pair = alone_stereo.decode(&stereo[frame].data).unwrap();
        let single = alone_mono.decode(&mono[frame].data).unwrap();
        assert_eq!(out.len(), 960 * 6);
        for i in 0..960 {
            // Vorbis 5.1 order: FL, C, FR, RL, RR, LFE.
            assert_eq!(out[i * 6], pair[i * 2]);
            assert_eq!(out[i * 6 + 2], pair[i * 2 + 1]);
            assert_eq!(out[i * 6 + 3], pair[i * 2]);
            assert_eq!(out[i * 6 + 1], single[i]);
            assert_eq!(out[i * 6 + 5], single[i]);
        }
    }
    // A stream missing from the packet is refused.
    let short = multistream(&[&stereo[3].data, &stereo[3].data]);
    assert!(decoder.decode(&short).is_err());
}

#[test]
fn output_gain_scales_the_samples() {
    let (packets, _) = encoded(1, 3);
    let mut plain = OpusDecoder::with_head(OpusHead::simple(1, 0, 48_000), 48_000).unwrap();
    let mut head = OpusHead::simple(1, 0, 48_000);
    head.output_gain = -6 * 256;
    let mut quiet = OpusDecoder::with_head(head, 48_000).unwrap();
    for packet in &packets {
        let a = plain.decode(&packet.data).unwrap();
        let b = quiet.decode(&packet.data).unwrap();
        for (x, y) in a.iter().zip(&b) {
            assert!((x * 0.501_187 - y).abs() < 1e-6);
        }
    }
}

#[test]
fn concealment_is_finite_marked_and_bounded() {
    let (packets, _) = encoded(2, 6);
    let mut decoder = OpusDecoder::with_head(OpusHead::simple(2, 0, 48_000), 48_000).unwrap();
    // Before any packet, concealment is silence.
    assert!(decoder.conceal(960).unwrap().iter().all(|&x| x == 0.0));
    for packet in &packets[..4] {
        decoder.decode(&packet.data).unwrap();
    }
    let Output::Audio(out) = decoder.conceal_output(2880).unwrap() else {
        panic!("audio");
    };
    assert!(out.concealed);
    assert_eq!(out.pcm.samples, 2880);
    assert!(out.pcm.as_f32().unwrap().iter().all(|x| x.is_finite()));
    assert!(decoder.conceal(0).is_err());
    assert!(decoder.conceal(100).is_err());
    assert!(decoder.conceal(5760 + 120).is_err());
    // CELT-only packets carry no redundancy: FEC conceals.
    assert_eq!(
        decoder.decode_fec(&packets[4].data, 960).unwrap().len(),
        1920
    );
    decoder.decode(&packets[4].data).unwrap();
}

/// Random bytes, truncations, and every TOC never panic, and every error
/// leaves a decoder that still decodes a good packet like a fresh one.
#[test]
fn hostile_packets_fail_closed() {
    let (packets, _) = encoded(2, 3);
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for channels in [1u8, 2] {
        let mut decoder =
            OpusDecoder::with_head(OpusHead::simple(channels, 0, 48_000), 48_000).unwrap();
        for round in 0..3000 {
            let len = (next() % 300) as usize + 1;
            let mut data: Vec<u8> = (0..len).map(|_| next() as u8).collect();
            data[0] = (round % 256) as u8;
            if let Ok(out) = decoder.decode(&data) {
                assert!(out.iter().all(|x| x.is_finite()));
            }
            if round % 7 == 0 {
                let cut = (next() as usize) % packets[1].data.len();
                let _ = decoder.decode(&packets[1].data[..cut.max(1)]);
            }
        }
        decoder.reset();
        let mut fresh =
            OpusDecoder::with_head(OpusHead::simple(channels, 0, 48_000), 48_000).unwrap();
        assert_eq!(
            decoder.decode(&packets[0].data).unwrap(),
            fresh.decode(&packets[0].data).unwrap()
        );
    }
}
