//! Decoder tests over the committed LAME fixtures.

use crate::common::{
    decode_packets, demux, fixture, open_decoder as open, oracle_compare, oracle_ffmpeg,
    oracle_mpg123, set_bits,
};
use cutlass_decode::{
    ChannelLayout, CodecError, CodecId, CodecParams, Decoder, Output, Packet, SideData,
};
use cutlass_mp3::Mp3Decoder;
use cutlass_mp3::header::FrameHeader;

/// Each fixture, its rate, channels, and the length of the source it was
/// encoded from (`None`: no LAME tag fits the frame, so no trim).
const FIXTURES: [(&str, u32, usize, Option<usize>); 6] = [
    ("sine_44100_j128_crc.mp3", 44_100, 2, Some(22_050)),
    ("clicks_48000_v2.mp3", 48_000, 2, Some(24_000)),
    ("sweep_22050_mono64.mp3", 22_050, 1, Some(11_025)),
    ("sine_8000_mono16.mp3", 8_000, 1, None),
    ("stereo_32000_ms96.mp3", 32_000, 2, Some(16_000)),
    ("sine_44100_free400.mp3", 44_100, 2, Some(11_025)),
];

#[test]
fn every_fixture_decodes_to_the_length_of_its_source() {
    for (name, rate, channels, source) in FIXTURES {
        let file = demux(&fixture(name));
        assert_eq!(file.rate, rate, "{name}");
        assert_eq!(file.channels, channels, "{name}");
        let decoded = decode_packets(&file, true);
        let frames = decoded.pcm.len() / channels;
        let expected = source.unwrap_or(file.packets.len() * file.samples_per_frame);
        assert_eq!(frames, expected, "{name}");
        assert!(decoded.pcm.iter().all(|s| s.is_finite()), "{name}");
    }
}

#[test]
fn the_container_and_the_decoder_read_the_same_headers() {
    for (name, ..) in FIXTURES {
        let file = demux(&fixture(name));
        for packet in &file.packets {
            let header =
                FrameHeader::parse([packet[0], packet[1], packet[2], packet[3]]).expect("header");
            assert_eq!(header.sample_rate, file.rate, "{name}");
            assert_eq!(header.channels(), file.channels, "{name}");
            assert_eq!(header.samples(), file.samples_per_frame, "{name}");
            if let Some(size) = header.frame_bytes() {
                assert_eq!(size, packet.len(), "{name}");
            }
        }
    }
}

/// Fit `a sin(wn) + b cos(wn)` to the first channel: amplitude, phase lag
/// in samples, and the SNR of what the sine does not explain.
fn fit_sine(pcm: &[f32], channels: usize, rate: u32, offset: usize) -> (f64, f64, f64) {
    let w = 2.0 * std::f64::consts::PI * 1000.0 / f64::from(rate);
    let samples: Vec<f64> = pcm.iter().step_by(channels).map(|s| f64::from(*s)).collect();
    let range = 200..samples.len() - 200;
    let (mut xs, mut xc, mut ss, mut cc) = (0.0, 0.0, 0.0, 0.0);
    for i in range.clone() {
        let t = (i + offset) as f64 * w;
        xs += samples[i] * t.sin();
        xc += samples[i] * t.cos();
        ss += t.sin() * t.sin();
        cc += t.cos() * t.cos();
    }
    let (a, b) = (xs / ss, xc / cc);
    let (mut signal, mut residual) = (0.0, 0.0);
    for i in range {
        let t = (i + offset) as f64 * w;
        let model = a * t.sin() + b * t.cos();
        signal += model * model;
        residual += (samples[i] - model).powi(2);
    }
    (
        a.hypot(b),
        -b.atan2(a) / w,
        10.0 * (signal / residual).log10(),
    )
}

#[test]
fn a_gapless_sine_lines_up_with_the_source_it_was_encoded_from() {
    // The source is a 1 kHz sine at half scale starting at phase zero. With
    // the trim applied, the decoded sine starts there too.
    let file = demux(&fixture("sine_44100_j128_crc.mp3"));
    let decoded = decode_packets(&file, true);
    let (amplitude, lag, snr) = fit_sine(&decoded.pcm, 2, 44_100, 0);
    assert!(lag.abs() < 0.01, "lag {lag}");
    assert!((0.45..0.52).contains(&amplitude), "amplitude {amplitude}");
    assert!(snr > 40.0, "snr {snr}");

    // No tag at 16 kbps and 8 kHz: the stream starts with LAME's 576-sample
    // delay and the 529-sample decoder delay.
    let file = demux(&fixture("sine_8000_mono16.mp3"));
    let decoded = decode_packets(&file, false);
    let lead = 576 + 529;
    let (amplitude, lag, snr) = fit_sine(&decoded.pcm[lead..lead + 3000], 1, 8_000, 0);
    assert!(lag.abs() < 0.05, "lag {lag}");
    assert!((0.4..0.55).contains(&amplitude), "amplitude {amplitude}");
    assert!(snr > 15.0, "snr {snr}");
}

#[test]
fn every_fixture_matches_ffmpeg_within_the_iso_11172_4_bounds() {
    let Some(ffmpeg) = oracle_ffmpeg() else {
        return;
    };
    for (name, ..) in FIXTURES {
        // FFmpeg does not decode free format above 320 kbps.
        if name.contains("free") {
            continue;
        }
        oracle_compare(&ffmpeg, &fixture(name), name);
    }
}

#[test]
fn every_fixture_matches_mpg123_within_the_iso_11172_4_bounds() {
    let Some(mpg123) = oracle_mpg123() else {
        return;
    };
    for (name, ..) in FIXTURES {
        oracle_compare(&mpg123, &fixture(name), name);
    }
}

#[test]
fn local_media_matches_ffmpeg_within_the_iso_11172_4_bounds() {
    let Some(files) = crate::common::local_mp3s(2_000_000) else {
        return;
    };
    let Some(ffmpeg) = oracle_ffmpeg() else {
        return;
    };
    for path in files {
        let bytes = std::fs::read(&path).expect("read");
        oracle_compare(&ffmpeg, &bytes, &path.display().to_string());
    }
}

/// Thirty minutes of rain against FFmpeg. Run with `--ignored`.
#[test]
#[ignore = "decodes the thirty-minute local file"]
fn the_thirty_minute_local_file_matches_ffmpeg() {
    let Some(files) = crate::common::local_mp3s(u64::MAX) else {
        return;
    };
    let Some(ffmpeg) = oracle_ffmpeg() else {
        return;
    };
    let files = files
        .into_iter()
        .filter(|path| std::fs::metadata(path).is_ok_and(|meta| meta.len() > 2_000_000));
    for path in files {
        let bytes = std::fs::read(&path).expect("read");
        oracle_compare(&ffmpeg, &bytes, &path.display().to_string());
    }
}

// ---- lifecycle ----

#[test]
fn an_empty_packet_is_an_api_error_and_changes_nothing() {
    let file = demux(&fixture("stereo_32000_ms96.mp3"));
    let clean = decode_packets(&file, false);
    let mut decoder = open(file.rate, file.channels);
    let mut pcm = Vec::new();
    for index in 0..file.packets.len() {
        if index == 3 {
            let empty = Packet::audio(Vec::new(), 0, 0, file.rate);
            assert!(matches!(
                decoder.push(&empty),
                Err(CodecError::InvalidApi { .. })
            ));
        }
        pcm.extend(crate::common::samples(
            decoder.push(&file.packet(index)).expect("push"),
        ));
    }
    assert_eq!(pcm, clean.pcm);
}

#[test]
fn a_failed_push_leaves_the_decoder_as_it_was() {
    let file = demux(&fixture("stereo_32000_ms96.mp3"));
    let clean = decode_packets(&file, true);
    let mut decoder = open(file.rate, file.channels);
    let mut pcm = Vec::new();
    for index in 0..file.packets.len() {
        if index == 5 {
            // A frame cut short, then one whose part2_3_length overruns.
            let mut cut = file.packet(index);
            cut.data.truncate(cut.data.len() / 2);
            assert!(decoder.push(&cut).is_err());
            let mut bad = file.packet(index);
            set_bits(&mut bad.data, 52, 12, 4095);
            assert!(decoder.push(&bad).is_err());
        }
        pcm.extend(crate::common::audible(
            decoder.push(&file.packet_gapless(index)).expect("push"),
        ));
    }
    pcm.extend(crate::common::audible(decoder.drain().expect("drain")));
    assert_eq!(pcm, clean.pcm);
}

#[test]
fn drain_twice_returns_nothing_the_second_time() {
    let file = demux(&fixture("sine_44100_j128_crc.mp3"));
    let mut decoder = open(file.rate, file.channels);
    for index in 0..file.packets.len() {
        decoder.push(&file.packet_gapless(index)).expect("push");
    }
    assert!(!decoder.drain().expect("drain").is_empty());
    assert!(decoder.drain().expect("again").is_empty());
}

#[test]
fn without_gapless_side_data_every_sample_is_returned_and_drain_is_empty() {
    let file = demux(&fixture("clicks_48000_v2.mp3"));
    let mut decoder = open(file.rate, file.channels);
    let mut frames = 0;
    for index in 0..file.packets.len() {
        for output in decoder.push(&file.packet(index)).expect("push") {
            let Output::Audio(audio) = output else {
                panic!("audio")
            };
            assert_eq!(audio.pts, (index * file.samples_per_frame) as i64);
            assert_eq!(audio.skip_trailing, 0);
            frames += audio.pcm.samples as usize;
        }
    }
    assert_eq!(frames, file.packets.len() * file.samples_per_frame);
    assert!(decoder.drain().expect("drain").is_empty());
}

#[test]
fn reset_starts_over_and_decodes_the_same_samples() {
    let file = demux(&fixture("sweep_22050_mono64.mp3"));
    let clean = decode_packets(&file, true);
    let mut decoder = open(file.rate, file.channels);
    for index in 0..7 {
        decoder.push(&file.packet_gapless(index)).expect("push");
    }
    decoder.reset();
    let mut pcm = Vec::new();
    for index in 0..file.packets.len() {
        pcm.extend(crate::common::audible(
            decoder.push(&file.packet_gapless(index)).expect("push"),
        ));
    }
    pcm.extend(crate::common::audible(decoder.drain().expect("drain")));
    assert_eq!(pcm, clean.pcm);
}

#[test]
fn after_a_seek_the_preroll_frames_restore_identical_output() {
    for name in ["clicks_48000_v2.mp3", "sweep_22050_mono64.mp3"] {
        let file = demux(&fixture(name));
        let clean = decode_packets(&file, false);
        let spf = file.samples_per_frame * file.channels;
        let target = 12;
        let preroll = file.reservoir_preroll.min(target);
        let mut decoder = open(file.rate, file.channels);
        // Land somewhere else first, then seek.
        decoder.push(&file.packet(2)).expect("push");
        decoder.reset();
        let mut pcm = Vec::new();
        for index in target - preroll..file.packets.len() {
            let samples = crate::common::samples(decoder.push(&file.packet(index)).expect("push"));
            if index >= target {
                pcm.extend(samples);
            }
        }
        assert_eq!(pcm, clean.pcm[target * spf..], "{name}");
    }
}

#[test]
fn a_frame_after_reset_whose_main_data_is_gone_is_skipped_not_invented() {
    let file = demux(&fixture("clicks_48000_v2.mp3"));
    // Find a frame that borrows from the reservoir.
    let index = (1..file.packets.len())
        .find(|&i| crate::common::get_bits(&file.packets[i], 32, 9) > 0)
        .expect("a frame with main_data_begin");
    let mut decoder = open(file.rate, file.channels);
    let outputs = decoder.push(&file.packet(index)).expect("push");
    let Output::Audio(audio) = &outputs[0] else {
        panic!("audio")
    };
    assert!(audio.skip_leading >= 576, "{}", audio.skip_leading);
}

#[test]
fn gapless_timestamps_count_from_the_first_valid_sample() {
    let file = demux(&fixture("sine_44100_j128_crc.mp3"));
    let (delay, _) = file.trim.expect("trim");
    let mut decoder = open(file.rate, file.channels);
    let lead = i64::from(delay) + i64::from(cutlass_mp3::DECODER_DELAY);
    let mut skipped = 0;
    let mut next_pts = -lead;
    for index in 0..3 {
        for output in decoder.push(&file.packet_gapless(index)).expect("push") {
            let Output::Audio(audio) = output else {
                panic!("audio")
            };
            assert_eq!(audio.pts, next_pts);
            next_pts += i64::from(audio.pcm.samples);
            skipped += i64::from(audio.skip_leading);
        }
    }
    assert_eq!(skipped, lead);
    let mut decoder = open(file.rate, file.channels);
    let mut packet = file.packet(0);
    packet.side.push(SideData::Gapless {
        encoder_delay: delay,
        trailing_padding: 0,
        sample_rate: 48_000,
    });
    // Gapless data for another rate is ignored.
    let outputs = decoder.push(&packet).expect("push");
    let Output::Audio(audio) = &outputs[0] else {
        panic!("audio")
    };
    assert_eq!(audio.skip_leading, 0);
}

// ---- malformed input ----

#[test]
fn every_truncation_of_every_frame_fails_without_harm() {
    for name in ["sine_44100_j128_crc.mp3", "sweep_22050_mono64.mp3"] {
        let file = demux(&fixture(name));
        let clean = decode_packets(&file, false);
        let mut decoder = open(file.rate, file.channels);
        let mut pcm = Vec::new();
        for index in 0..file.packets.len() {
            let data = &file.packets[index];
            for cut in 1..data.len() {
                let packet = Packet::audio(data[..cut].to_vec(), 0, 0, file.rate);
                assert!(decoder.push(&packet).is_err(), "{name} {index} {cut}");
            }
            pcm.extend(crate::common::samples(
                decoder.push(&file.packet(index)).expect("push"),
            ));
        }
        assert_eq!(pcm, clean.pcm, "{name}");
    }
}

#[test]
fn reserved_and_unsupported_header_fields_are_refused() {
    let file = demux(&fixture("stereo_32000_ms96.mp3"));
    let frame = &file.packets[0];
    let mut decoder = open(0, 0);
    let mut try_header = |bit: usize, count: u32, value: u32| {
        let mut data = frame.clone();
        set_bits(&mut data, bit, count, value);
        decoder.push(&Packet::audio(data, 0, 1152, 32_000))
    };
    // Sync, version 01, layer 00, bitrate 15, rate 3, emphasis 2.
    assert!(matches!(
        try_header(0, 11, 0x7FE),
        Err(CodecError::Malformed { .. })
    ));
    assert!(matches!(
        try_header(11, 2, 1),
        Err(CodecError::Malformed { .. })
    ));
    assert!(matches!(
        try_header(13, 2, 0),
        Err(CodecError::Malformed { .. })
    ));
    assert!(matches!(
        try_header(16, 4, 15),
        Err(CodecError::Malformed { .. })
    ));
    assert!(matches!(
        try_header(20, 2, 3),
        Err(CodecError::Malformed { .. })
    ));
    assert!(matches!(
        try_header(30, 2, 2),
        Err(CodecError::Malformed { .. })
    ));
    // Layers I and II.
    for layer in [2, 3] {
        assert!(matches!(
            try_header(13, 2, layer),
            Err(CodecError::Unsupported { what: "layer" })
        ));
    }
    // A free-format frame longer than any legal frame.
    let mut data = frame.clone();
    set_bits(&mut data, 16, 4, 0);
    data.resize(4097, 0);
    assert!(decoder.push(&Packet::audio(data, 0, 1152, 32_000)).is_err());
}

#[test]
fn a_crc_mismatch_is_malformed_and_main_data_is_not_covered() {
    let file = demux(&fixture("sine_44100_j128_crc.mp3"));
    let mut decoder = open(file.rate, file.channels);
    let mut data = file.packets[0].clone();
    // global_gain of granule 0, channel 0: header 32, crc 16, then 9 + 3 + 8 + 12 + 9.
    let at = 32 + 16 + 9 + 3 + 8 + 12 + 9;
    let gain = crate::common::get_bits(&data, at, 8);
    set_bits(&mut data, at, 8, gain ^ 1);
    assert!(matches!(
        decoder.push(&Packet::audio(data, 0, 1152, file.rate)),
        Err(CodecError::Malformed { what: "crc", .. })
    ));
    let mut data = file.packets[0].clone();
    let last = data.len() - 1;
    data[last] ^= 0x55;
    assert!(
        decoder
            .push(&Packet::audio(data, 0, 1152, file.rate))
            .is_ok()
    );
}

/// Bit offsets inside an unprotected MPEG-1 stereo frame.
mod mpeg1_stereo {
    pub const MAIN_DATA_BEGIN: usize = 32;
    /// First bit of granule 0, channel 0.
    pub const GRANULE: usize = 32 + 9 + 3 + 8;
    pub const BIG_VALUES: usize = GRANULE + 12;
    pub const WINDOW_SWITCHING: usize = GRANULE + 12 + 9 + 8 + 4;
    pub const TABLE_SELECT: usize = WINDOW_SWITCHING + 1;
}

#[test]
fn side_information_out_of_range_is_malformed() {
    use mpeg1_stereo::{BIG_VALUES, MAIN_DATA_BEGIN, TABLE_SELECT, WINDOW_SWITCHING};
    let file = demux(&fixture("stereo_32000_ms96.mp3"));
    let mut decoder = open(file.rate, file.channels);
    decoder.push(&file.packet(0)).expect("first");
    let push = |decoder: &mut Mp3Decoder, data: Vec<u8>| {
        decoder.push(&Packet::audio(data, 1152, 1152, file.rate))
    };

    let mut data = file.packets[1].clone();
    set_bits(&mut data, BIG_VALUES, 9, 289);
    assert!(matches!(
        push(&mut decoder, data),
        Err(CodecError::Malformed {
            what: "big_values",
            ..
        })
    ));

    // main_data_begin past the bytes the first frame left behind.
    let mut data = file.packets[1].clone();
    set_bits(&mut data, MAIN_DATA_BEGIN, 9, 511);
    assert!(matches!(
        push(&mut decoder, data),
        Err(CodecError::Malformed {
            what: "main_data_begin",
            ..
        })
    ));

    // Table 4 does not exist.
    let index = (1..file.packets.len())
        .find(|&i| {
            let data = &file.packets[i];
            crate::common::get_bits(data, WINDOW_SWITCHING, 1) == 0
                && crate::common::get_bits(data, BIG_VALUES, 9) > 8
        })
        .expect("a long-block granule with big values");
    let mut decoder = open(file.rate, file.channels);
    for i in 0..index {
        decoder.push(&file.packet(i)).expect("push");
    }
    let mut data = file.packets[index].clone();
    set_bits(&mut data, TABLE_SELECT, 5, 4);
    assert!(matches!(
        push(&mut decoder, data),
        Err(CodecError::Malformed {
            what: "table_select",
            ..
        })
    ));

    // A Huffman codeword cut by part2_3_length.
    let mut data = file.packets[index].clone();
    set_bits(&mut data, mpeg1_stereo::GRANULE, 12, 40);
    assert!(matches!(
        push(&mut decoder, data),
        Err(CodecError::Malformed { .. })
    ));
    // The decoder still continues the stream.
    decoder.push(&file.packet(index)).expect("clean frame");
}

#[test]
fn a_sample_rate_or_channel_change_mid_stream_is_malformed() {
    let stereo = demux(&fixture("stereo_32000_ms96.mp3"));
    let other = demux(&fixture("sine_44100_j128_crc.mp3"));
    let mut decoder = open(0, 0);
    decoder.push(&stereo.packet(0)).expect("first");
    assert!(matches!(
        decoder.push(&other.packet(0)),
        Err(CodecError::Malformed {
            what: "sample rate change",
            ..
        })
    ));
    let mut mono = stereo.packets[1].clone();
    set_bits(&mut mono, 24, 2, 3);
    assert!(matches!(
        decoder.push(&Packet::audio(mono, 1152, 1152, 32_000)),
        Err(CodecError::Malformed {
            what: "channel change",
            ..
        })
    ));
    decoder.push(&stereo.packet(1)).expect("the stream goes on");
    assert!(matches!(
        Mp3Decoder::open(&CodecParams::audio(
            CodecId::Aac,
            44_100,
            ChannelLayout::stereo()
        )),
        Err(CodecError::Unsupported { what: "codec" })
    ));
    assert!(matches!(
        Mp3Decoder::open(&CodecParams::audio(
            CodecId::Mp3,
            44_000,
            ChannelLayout::stereo()
        )),
        Err(CodecError::Unsupported {
            what: "sample rate"
        })
    ));
}

#[test]
fn random_and_mutated_frames_never_panic() {
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for (name, ..) in FIXTURES {
        let file = demux(&fixture(name));
        let mut decoder = open(0, 0);
        for round in 0..3000 {
            let index = (next() as usize) % file.packets.len();
            let mut data = file.packets[index].clone();
            for _ in 0..1 + next() % 8 {
                let at = (next() as usize) % data.len();
                // Keep the sync word most of the time so the payload is read.
                if at >= 4 || round % 4 == 0 {
                    data[at] ^= 1 << (next() % 8);
                }
            }
            let packet = Packet::audio(data, 0, 0, file.rate);
            if let Ok(outputs) = decoder.push(&packet) {
                for sample in crate::common::samples(outputs) {
                    assert!(sample.is_finite(), "{name}");
                }
            }
            if round % 500 == 0 {
                decoder.reset();
            }
        }
        for _ in 0..500 {
            let len = (next() as usize) % 2000;
            let mut data: Vec<u8> = (0..len).map(|_| next() as u8).collect();
            if data.len() >= 2 {
                data[0] = 0xFF;
                data[1] |= 0xE0;
            }
            let _ = decoder.push(&Packet::audio(data, 0, 0, file.rate));
        }
    }
}
