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
    encoded_at(channels, &vec![1.0; frames])
}

/// As [`encoded`], one 20 ms frame per entry of `levels`, each frame's
/// tones scaled by its level.
fn encoded_at(channels: usize, levels: &[f32]) -> (Vec<Packet>, Vec<f32>) {
    let layout = if channels == 2 {
        ChannelLayout::stereo()
    } else {
        ChannelLayout::mono()
    };
    let mut params = CodecParams::audio(CodecId::Opus, 48_000, layout.clone());
    params.bitrate = 64_000 * channels as u32;
    let mut encoder = crate::OpusEncoder::open(&params).unwrap();
    let samples: Vec<f32> = (0..levels.len() * 960)
        .flat_map(|i| {
            let t = i as f32 / 48_000.0;
            let level = levels[i / 960];
            let left = level * 0.4 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            let right = level * 0.3 * (2.0 * std::f32::consts::PI * 660.0 * t).sin();
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

/// A packet of `toc` and `len` fixed pseudo-random payload bytes. Any
/// payload is a valid SILK, hybrid, or CELT frame.
fn noise_packet(toc: u8, seed: u32, len: usize) -> Vec<u8> {
    let mut state = seed;
    let mut data = vec![toc];
    for _ in 0..len {
        state = (state.wrapping_mul(1_103_515_245).wrapping_add(12_345)) & 0x7fff_ffff;
        data.push((state >> 16) as u8);
    }
    data
}

/// The switch from hybrid to CELT conceals 5 ms of the hybrid stream to
/// fade from. The new packet's narrower bandwidth must not reach that
/// concealment. The final ranges are libopus 1.6.1's for these packets.
#[test]
fn a_narrower_celt_packet_after_a_hybrid_packet_decodes() {
    // Hybrid fullband 20 ms, then CELT narrowband 20 ms twice, all mono.
    let packets = [
        (noise_packet(0x78, 1, 40), 0x022c_a610),
        (noise_packet(0x98, 2, 40), 0x6835_5100),
        (noise_packet(0x98, 3, 40), 0x008d_e41d),
    ];
    let mut decoder = OpusDecoder::with_head(OpusHead::simple(1, 0, 48_000), 48_000).unwrap();
    for (data, range) in &packets {
        let out = decoder.decode(data).unwrap();
        assert_eq!(out.len(), 960);
        assert!(out.iter().all(|x| x.is_finite()));
        assert_eq!(decoder.final_range(), *range);
    }
    // Hybrid, then CELT wideband.
    let mut decoder = OpusDecoder::with_head(OpusHead::simple(1, 0, 48_000), 48_000).unwrap();
    decoder.decode(&noise_packet(0x78, 1, 40)).unwrap();
    decoder.decode(&noise_packet(0xb8, 2, 40)).unwrap();
    assert_eq!(decoder.final_range(), 0x00f8_a600);
}

/// A lost frame is concealed in the mode of the frame before it, with
/// that frame's bands, whatever bandwidth the lost frame's TOC names.
#[test]
fn a_lost_narrowband_frame_after_a_hybrid_packet_is_concealed() {
    let mut decoder = OpusDecoder::with_head(OpusHead::simple(1, 0, 48_000), 48_000).unwrap();
    decoder.decode(&noise_packet(0x78, 1, 40)).unwrap();
    // SILK narrowband 20 ms with a one-byte frame: lost.
    let out = decoder.decode(&[0x08, 0x00]).unwrap();
    assert_eq!(out.len(), 960);
    assert!(out.iter().all(|x| x.is_finite()));
    assert_eq!(decoder.final_range(), 0);
    decoder.decode(&noise_packet(0x08, 4, 20)).unwrap();
    assert_eq!(decoder.final_range(), 0x026c_360a);
}

/// A gap is any multiple of 2.5 ms, but concealment runs on 2.5, 5, 10, or
/// 20 ms at a time: 12.5 ms is concealed as 10 ms and then 2.5 ms.
#[test]
fn a_gap_between_frame_lengths_is_concealed_in_pieces() {
    let after_one_packet = |toc: u8| {
        let mut decoder = OpusDecoder::with_head(OpusHead::simple(1, 0, 48_000), 48_000).unwrap();
        decoder.decode(&noise_packet(toc, 1, 40)).unwrap();
        decoder
    };
    // SILK wideband, hybrid fullband, and CELT fullband, 20 ms mono.
    for toc in [0x48, 0x78, 0xf8] {
        for (gap, rest) in [(600, 120), (720, 240)] {
            let whole = after_one_packet(toc).conceal(gap).unwrap();
            let mut decoder = after_one_packet(toc);
            let mut pieces = decoder.conceal(480).unwrap();
            pieces.extend(decoder.conceal(rest).unwrap());
            assert!(pieces.iter().any(|x| *x != 0.0));
            assert!(whole == pieces, "TOC {toc:#x}, {gap} samples");
        }
        for steps in 1..=48 {
            let out = after_one_packet(toc).conceal(steps * 120).unwrap();
            assert_eq!(out.len(), steps * 120);
        }
    }
}

/// Whether a CELT-only packet's frame codes its energies intra, not
/// predicted from the frame before (the header of RFC 6716 section 4.3).
fn intra_energy(packet: &[u8]) -> bool {
    let mut rd = crate::range::RangeDecoder::new(&packet[1..]);
    let total = (packet.len() as i32 - 1) * 8;
    if rd.bit_logp(15) {
        return true;
    }
    if rd.tell() + 16 <= total && rd.bit_logp(1) {
        let octave = rd.dec_uint(6);
        rd.dec_bits(4 + octave);
        rd.dec_bits(3);
        if rd.tell() + 2 <= total {
            rd.dec_icdf(&[2, 1, 0], 2);
        }
    }
    if rd.tell() + 3 <= total {
        rd.bit_logp(3);
    }
    rd.tell() + 3 <= total && rd.bit_logp(3)
}

/// The packets that carry a drop in level are lost, sent as their TOC
/// alone (as ffmpeg's native encoder sends silence). The encoder predicts
/// the next packet's energies from its own quiet state, and the decoder
/// must not add them to the loud state it last decoded: as libopus 1.6.1
/// does, it decays its energies through the loss and makes the first
/// prediction after it safe. The audio after the gap is never louder than
/// the concealment before it, and comes back to the source's level.
#[test]
fn audio_after_a_long_loss_does_not_burst() {
    // 0.4 s loud, then the same tones 40 dB down.
    let levels: Vec<f32> = (0..60).map(|k| if k < 20 { 1.0 } else { 0.01 }).collect();
    let mean_square = |x: &[f32]| x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    for channels in [1usize, 2] {
        let (packets, samples) = encoded_at(channels, &levels);
        let quiet = mean_square(&samples[30 * 960 * channels..]);
        // Packets 20 on are lost, up to one that predicts its energies.
        let resume = (40..50).find(|&k| !intra_energy(&packets[k].data)).unwrap();
        let head = OpusHead::simple(channels as u8, 0, 48_000);
        let mut decoder = OpusDecoder::with_head(head, 48_000).unwrap();
        // Each packet's level against the quiet source, in dB.
        let mut db = Vec::new();
        for (k, packet) in packets.iter().enumerate().take(60) {
            let data = if (20..resume).contains(&k) {
                &packet.data[..1]
            } else {
                &packet.data[..]
            };
            let out = decoder.decode(data).unwrap();
            assert!(out.iter().all(|x| x.is_finite()));
            db.push(10.0 * (mean_square(&out) / quiet).log10());
        }
        let shown: Vec<String> = db[resume - 1..].iter().map(|d| format!("{d:.1}")).collect();
        let shown = format!(
            "{channels} ch, from the last concealed packet {}: {shown:?}",
            resume - 1
        );
        eprintln!("{shown}");
        let concealed = db[resume - 1];
        assert!(
            db[resume..].iter().all(|&d| d <= concealed + 1.0),
            "{shown}"
        );
        assert!(db[55..].iter().all(|&d| d.abs() <= 1.0), "{shown}");
    }
}

/// Three lost packets in a steady tone are concealed by repeating its last
/// pitch period, which libopus 1.6.1 fades by how much the last two periods
/// decayed, and by 0.8 a frame after the first: about 0, 2 and 4 dB down
/// on these tones. The concealment stays near the tone's level instead of
/// dying out.
#[test]
fn pitch_concealment_holds_a_steady_tone() {
    let mean_square = |x: &[f32]| x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    for channels in [1usize, 2] {
        let (packets, _) = encoded(channels, 30);
        let head = OpusHead::simple(channels as u8, 0, 48_000);
        let mut decoder = OpusDecoder::with_head(head, 48_000).unwrap();
        let mut energies = Vec::new();
        for (k, packet) in packets.iter().enumerate().take(23) {
            let lost = (20..23).contains(&k);
            let data = if lost { &packet.data[..1] } else { &packet.data[..] };
            energies.push(mean_square(&decoder.decode(data).unwrap()));
        }
        let last = energies[19];
        let db: Vec<f32> = energies[20..]
            .iter()
            .map(|e| 10.0 * (e / last).log10())
            .collect();
        let all = 10.0 * (energies[20..].iter().sum::<f32>() / (3.0 * last)).log10();
        let shown = format!("{channels} ch: lost packets {db:.1?} dB, together {all:.1} dB");
        eprintln!("{shown}");
        assert!(db.iter().all(|d| (-6.0..=1.0).contains(d)), "{shown}");
        assert!(all >= -3.0, "{shown}");
    }
}
