//! Decoder conformance, channel order, metadata, timing and lifecycle.

use cutlass_ac3::{Downmix, DynamicRange, Options, Syntax};
use cutlass_decode::{Channel, CodecError, CodecId, Decoder, Output, Packet, SideData};

use crate::common::{
    FIXTURES, apple_decode, audible, codec_of, decode, decode_with, deinterleave, ffmpeg_decode,
    fixture, open, oracle, packet, power, samples, snr, tone, units,
};

use Channel::{
    BackCenter as Bc, FrontCenter as C, FrontLeft as L, FrontRight as R, Lfe, SideLeft as Ls,
    SideRight as Rs,
};

/// Each fixture's rate, WAVEFORMATEXTENSIBLE layout and syncframes.
fn expected(name: &str) -> (u32, Vec<Channel>, usize) {
    match name {
        "ac3_mono_48000.ac3" => (48_000, vec![C], 8),
        "ac3_stereo_44100.ac3" => (44_100, vec![L, R], 8),
        "ac3_3.0_32000.ac3" => (32_000, vec![L, R, C], 5),
        "ac3_3.0-back_48000.ac3" => (48_000, vec![L, R, Bc], 8),
        "ac3_quad_48000.ac3" => (48_000, vec![L, R, Ls, Rs], 8),
        "ac3_5.1_48000_xbsi.ac3" => (48_000, vec![L, R, C, Lfe, Ls, Rs], 8),
        "ac3_2.1_44100.ac3" => (44_100, vec![L, R, Lfe], 8),
        "eac3_stereo_48000.eac3" => (48_000, vec![L, R], 8),
        "eac3_5.1_44100.eac3" => (44_100, vec![L, R, C, Lfe, Ls, Rs], 7),
        "eac3_mono_32000.eac3" => (32_000, vec![C], 6),
        "eac3_5.0_48000.eac3" => (48_000, vec![L, R, C, Ls, Rs], 8),
        _ => panic!("{name}"),
    }
}

#[test]
fn every_fixture_decodes_with_its_rate_layout_and_length() {
    for name in FIXTURES {
        let decoded = decode(&fixture(name));
        let (rate, layout, frames) = expected(name);
        assert_eq!(decoded.rate, rate, "{name}");
        assert_eq!(decoded.channels, layout, "{name}");
        assert_eq!(decoded.frames(), frames * 1536, "{name}");
        assert!(decoded.pcm.iter().all(|s| s.is_finite()), "{name}");
    }
}

#[test]
fn each_channel_carries_its_own_tone_in_speaker_order() {
    let candidates = [60.0, 300.0, 500.0, 700.0, 900.0, 1100.0, 1300.0];
    for name in FIXTURES {
        let decoded = decode(&fixture(name));
        for (index, &speaker) in decoded.channels.iter().enumerate() {
            // Skip the first block: the encoder's priming.
            let signal = &decoded.channel(index)[1024..];
            let want = power(signal, decoded.rate, tone(speaker));
            for &other in candidates.iter().filter(|&&f| f != tone(speaker)) {
                let got = power(signal, decoded.rate, other);
                assert!(
                    want > 1000.0 * got,
                    "{name} {speaker:?}: {} Hz {want:e} vs {other} Hz {got:e}",
                    tone(speaker)
                );
            }
            assert!(want > 1e-4, "{name} {speaker:?} is silent");
        }
    }
}

#[test]
fn every_fixture_matches_ffmpeg_to_the_dither_floor() {
    let Some(ffmpeg) = oracle("ffmpeg", "-version") else {
        return;
    };
    for name in FIXTURES {
        let data = fixture(name);
        let ours = decode(&data);
        let reference = ffmpeg_decode(&ffmpeg, &data, &[]);
        let n = ours.channels.len();
        assert_eq!(reference.len(), ours.pcm.len(), "{name}");
        for (index, speaker) in ours.channels.iter().enumerate() {
            let quality = snr(&ours.channel(index), &deinterleave(&reference, n, index));
            // LFE is never dithered: it matches to float rounding.
            let floor = if *speaker == Lfe { 120.0 } else { 55.0 };
            assert!(quality > floor, "{name} {speaker:?}: {quality:.1} dB");
        }
    }
}

#[test]
fn every_fixture_matches_apple_to_the_dither_floor() {
    let Some(afconvert) = oracle("afconvert", "-h") else {
        return;
    };
    // Apple renders 2/1 as L, a silent C and R, and drops the surround;
    // FFmpeg and the tone test cover that fixture.
    for name in FIXTURES
        .into_iter()
        .filter(|n| *n != "ac3_3.0-back_48000.ac3")
    {
        let data = fixture(name);
        let ours = decode(&data);
        let (channels, reference) = apple_decode(&afconvert, &data);
        assert_eq!(channels, ours.channels.len(), "{name}");
        let mut used = vec![false; channels];
        for (index, speaker) in ours.channels.iter().enumerate() {
            let mine = &ours.channel(index)[256..];
            let (best, quality) = (0..channels)
                .map(|c| (c, snr(mine, &deinterleave(&reference, channels, c))))
                .fold((0, f64::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a });
            assert!(!used[best], "{name} {speaker:?} matched twice");
            used[best] = true;
            let floor = if *speaker == Lfe { 80.0 } else { 55.0 };
            assert!(quality > floor, "{name} {speaker:?}: {quality:.1} dB");
        }
    }
}

#[test]
fn metadata_reports_what_the_encoder_wrote() {
    let decoder = |name: &str| {
        let data = fixture(name);
        let mut decoder = open(codec_of(&data), Options::default());
        for (index, unit) in units(&data).iter().enumerate() {
            decoder.push(&packet(unit, index, 0)).expect("push");
        }
        decoder
    };
    let xbsi = decoder("ac3_5.1_48000_xbsi.ac3");
    let m = xbsi.metadata().expect("metadata");
    assert_eq!(
        (m.syntax, m.bsid, m.acmod, m.lfe),
        (Syntax::Ac3, 6, 7, true)
    );
    assert_eq!(m.dialnorm, 20);
    assert_eq!(m.dmixmod, Some(1));
    let levels = m.mix_levels.expect("mix levels");
    assert!((levels.ltrt_center - 0.595).abs() < 1e-3);
    assert!((levels.ltrt_surround - 0.5).abs() < 1e-6);
    assert!((levels.loro_center - 0.841).abs() < 1e-3);
    assert_eq!(levels.loro_surround, 0.0);
    assert_eq!((m.dsurexmod, m.adconvtyp), (Some(2), Some(1)));

    let apinfo = decoder("ac3_2.1_44100.ac3");
    let m = apinfo.metadata().expect("metadata");
    assert_eq!((m.bsid, m.dsurmod, m.mixing_room), (8, 2, Some((100, 2))));
    assert_eq!(m.dialnorm, 31);

    let eac3 = decoder("eac3_5.1_44100.eac3");
    let m = eac3.metadata().expect("metadata");
    assert_eq!((m.syntax, m.bsid, m.dialnorm), (Syntax::Eac3, 16, 24));
    assert_eq!(m.dmixmod, Some(2));
    let levels = m.mix_levels.expect("mix levels");
    assert!((levels.loro_center - 0.595).abs() < 1e-3);
    assert!((levels.loro_surround - 0.5).abs() < 1e-6);
}

#[test]
fn lo_ro_downmix_matches_ffmpeg() {
    let Some(ffmpeg) = oracle("ffmpeg", "-version") else {
        return;
    };
    // FFmpeg downmixes Lo/Ro only, with the `cmixlev`/`surmixlev` levels
    // (E-AC-3: the Lo/Ro levels); it ignores the extended levels of the
    // alternate bit stream syntax, which the equation test covers.
    for name in [
        "eac3_5.1_44100.eac3",
        "ac3_quad_48000.ac3",
        "eac3_5.0_48000.eac3",
    ] {
        let data = fixture(name);
        let ours = decode_with(
            &data,
            Options {
                downmix: Downmix::LoRo,
                ..Options::default()
            },
        );
        assert_eq!(ours.channels, vec![L, R], "{name}");
        let reference = ffmpeg_decode(&ffmpeg, &data, &["-downmix", "stereo"]);
        for index in 0..2 {
            let quality = snr(&ours.channel(index), &deinterleave(&reference, 2, index));
            assert!(quality > 55.0, "{name} {index}: {quality:.1} dB");
        }
    }
}

#[test]
fn downmix_follows_the_a52_equations_with_the_stream_levels() {
    // 3/2 with LFE, Lt/Rt center 0.595 and surround 0.5, Lo/Ro center
    // 0.841 and surround 0 (the extended levels of `bsid` 6).
    let data = fixture("ac3_5.1_48000_xbsi.ac3");
    let full = decode(&data);
    let [l, r, c, _lfe, ls, rs] = [0, 1, 2, 3, 4, 5].map(|i| full.channel(i));
    let check = |mode: Downmix, left: [f32; 5], right: [f32; 5]| {
        let mixed = decode_with(
            &data,
            Options {
                downmix: mode,
                ..Options::default()
            },
        );
        for (out, coefs) in [(0, left), (1, right)] {
            let norm: f32 = coefs.iter().map(|g| g.abs()).sum();
            let got = mixed.channel(out);
            for i in 0..got.len() {
                let want = (coefs[0] * l[i]
                    + coefs[1] * r[i]
                    + coefs[2] * c[i]
                    + coefs[3] * ls[i]
                    + coefs[4] * rs[i])
                    / norm;
                assert!((got[i] - want).abs() < 1e-5, "{mode:?} {out} {i}");
            }
        }
    };
    let cl = 0.840_896_4;
    check(
        Downmix::LoRo,
        [1.0, 0.0, cl, 0.0, 0.0],
        [0.0, 1.0, cl, 0.0, 0.0],
    );
    let (ct, st) = (0.594_603_6, 0.5);
    check(
        Downmix::LtRt,
        [1.0, 0.0, ct, -st, -st],
        [0.0, 1.0, ct, st, st],
    );
}

#[test]
fn mono_downmix_is_half_the_lo_ro_sum() {
    let data = fixture("ac3_5.1_48000_xbsi.ac3");
    let stereo = decode_with(
        &data,
        Options {
            downmix: Downmix::LoRo,
            ..Options::default()
        },
    );
    let mono = decode_with(
        &data,
        Options {
            downmix: Downmix::Mono,
            ..Options::default()
        },
    );
    assert_eq!(mono.channels, vec![C]);
    for (i, &m) in mono.pcm.iter().enumerate() {
        let expect = 0.5 * (stereo.pcm[2 * i] + stereo.pcm[2 * i + 1]);
        assert!((m - expect).abs() < 1e-6);
    }
}

#[test]
fn dynamic_range_off_ignores_the_words_and_line_mode_applies_them() {
    // The fixtures carry no compression, so every mode decodes the same.
    let data = fixture("ac3_stereo_44100.ac3");
    let off = decode(&data);
    let line = decode_with(
        &data,
        Options {
            dynamic_range: DynamicRange::Line {
                cut: 1.0,
                boost: 1.0,
            },
            ..Options::default()
        },
    );
    let heavy = decode_with(
        &data,
        Options {
            dynamic_range: DynamicRange::Heavy,
            ..Options::default()
        },
    );
    assert_eq!(off.pcm, line.pcm);
    assert_eq!(off.pcm, heavy.pcm);
}

// ---- timing ----

#[test]
fn timestamps_pass_through_and_every_sample_is_returned() {
    for name in ["ac3_stereo_44100.ac3", "eac3_5.1_44100.eac3"] {
        let data = fixture(name);
        let mut decoder = open(codec_of(&data), Options::default());
        let mut frames = 0u64;
        for (index, unit) in units(&data).iter().enumerate() {
            // A 90 kHz container clock.
            let pts = (index as i64) * 1536 * 90_000 / 44_100 + 7;
            let packet = Packet::audio(unit.clone(), pts, 3135, 90_000);
            for output in decoder.push(&packet).expect("push") {
                let Output::Audio(audio) = output else {
                    panic!("audio")
                };
                assert_eq!((audio.pts, audio.timescale), (pts, 90_000), "{name}");
                assert_eq!((audio.skip_leading, audio.skip_trailing), (0, 0));
                assert!(!audio.concealed);
                frames += u64::from(audio.pcm.samples);
            }
        }
        assert_eq!(frames as usize, units(&data).len() * 1536);
        assert!(decoder.drain().expect("drain").is_empty());
    }
}

#[test]
fn gapless_side_data_trims_the_delay_and_padding_exactly() {
    let data = fixture("eac3_stereo_48000.eac3");
    let all = units(&data);
    let (delay, padding) = (256u32, 1000u32);
    let mut decoder = open(CodecId::Eac3, Options::default());
    let mut audible_pcm = Vec::new();
    let mut next_pts = -i64::from(delay);
    let mut skipped = 0;
    for (index, unit) in all.iter().enumerate() {
        let mut packet = packet(unit, index, 48_000);
        packet.side.push(SideData::Gapless {
            encoder_delay: delay,
            trailing_padding: padding,
            sample_rate: 48_000,
        });
        let outputs = decoder.push(&packet).expect("push");
        for output in &outputs {
            let Output::Audio(audio) = output else {
                panic!("audio")
            };
            assert_eq!(audio.pts, next_pts);
            next_pts += i64::from(audio.pcm.samples);
            skipped += audio.skip_leading;
        }
        audible_pcm.extend(audible(outputs));
    }
    let tail = decoder.drain().expect("drain");
    let Some(Output::Audio(last)) = tail.first() else {
        panic!("the padding is held until drain")
    };
    assert_eq!(last.skip_trailing, padding);
    audible_pcm.extend(audible(tail));
    assert_eq!(skipped, delay);
    assert_eq!(
        audible_pcm.len() / 2,
        all.len() * 1536 - (delay + padding) as usize
    );
    // The audible samples are the full decode with both ends cut.
    let full = decode(&data);
    let from = delay as usize * 2;
    assert_eq!(audible_pcm, full.pcm[from..from + audible_pcm.len()]);
}

// ---- lifecycle ----

#[test]
fn an_empty_packet_is_an_api_error_and_changes_nothing() {
    let data = fixture("ac3_stereo_44100.ac3");
    let clean = decode(&data);
    let mut decoder = open(CodecId::Ac3, Options::default());
    let mut pcm = Vec::new();
    for (index, unit) in units(&data).iter().enumerate() {
        if index == 3 {
            let empty = Packet::audio(Vec::new(), 0, 0, 44_100);
            assert!(matches!(
                decoder.push(&empty),
                Err(CodecError::InvalidApi { .. })
            ));
        }
        pcm.extend(samples(
            decoder.push(&packet(unit, index, 0)).expect("push"),
        ));
    }
    assert_eq!(pcm, clean.pcm);
}

#[test]
fn a_failed_push_leaves_the_decoder_as_it_was() {
    for name in ["ac3_5.1_48000_xbsi.ac3", "eac3_5.1_44100.eac3"] {
        let data = fixture(name);
        let clean = decode(&data);
        let mut decoder = open(codec_of(&data), Options::default());
        let mut pcm = Vec::new();
        for (index, unit) in units(&data).iter().enumerate() {
            if index == 4 {
                let mut cut = unit.clone();
                cut.truncate(cut.len() / 2);
                assert!(decoder.push(&packet(&cut, index, 0)).is_err());
                let mut flipped = unit.clone();
                flipped[unit.len() / 3] ^= 0x10;
                assert!(matches!(
                    decoder.push(&packet(&flipped, index, 0)),
                    Err(CodecError::Malformed { .. })
                ));
            }
            pcm.extend(samples(
                decoder.push(&packet(unit, index, 0)).expect("push"),
            ));
        }
        assert_eq!(pcm, clean.pcm, "{name}");
    }
}

#[test]
fn drain_twice_returns_nothing_the_second_time() {
    let data = fixture("eac3_stereo_48000.eac3");
    let mut decoder = open(CodecId::Eac3, Options::default());
    for (index, unit) in units(&data).iter().enumerate() {
        let mut packet = packet(unit, index, 48_000);
        packet.side.push(SideData::Gapless {
            encoder_delay: 256,
            trailing_padding: 100,
            sample_rate: 48_000,
        });
        decoder.push(&packet).expect("push");
    }
    assert!(!decoder.drain().expect("drain").is_empty());
    assert!(decoder.drain().expect("again").is_empty());
}

#[test]
fn reset_starts_over_and_decodes_the_same_samples() {
    for name in ["ac3_quad_48000.ac3", "eac3_5.0_48000.eac3"] {
        let data = fixture(name);
        let clean = decode(&data);
        let mut decoder = open(codec_of(&data), Options::default());
        let all = units(&data);
        for (index, unit) in all.iter().enumerate().take(5) {
            decoder.push(&packet(unit, index, 0)).expect("push");
        }
        decoder.reset();
        let mut pcm = Vec::new();
        for (index, unit) in all.iter().enumerate() {
            pcm.extend(samples(
                decoder.push(&packet(unit, index, 0)).expect("push"),
            ));
        }
        assert_eq!(pcm, clean.pcm, "{name}");
        // The decoder is reusable after drain as well.
        decoder.drain().expect("drain");
        decoder.reset();
        let again = samples(decoder.push(&packet(&all[0], 0, 0)).expect("push"));
        assert_eq!(again, clean.pcm[..again.len()], "{name}");
    }
}

#[test]
fn after_a_seek_one_preroll_frame_restores_the_overlap() {
    for name in ["ac3_5.1_48000_xbsi.ac3", "eac3_stereo_48000.eac3"] {
        let data = fixture(name);
        let clean = decode(&data);
        let n = clean.channels.len();
        let all = units(&data);
        let target = 5;
        let mut decoder = open(codec_of(&data), Options::default());
        decoder.push(&packet(&all[1], 1, 0)).expect("push");
        decoder.reset();
        decoder
            .push(&packet(&all[target - 1], target - 1, 0))
            .expect("preroll");
        let pcm = samples(
            decoder
                .push(&packet(&all[target], target, 0))
                .expect("push"),
        );
        let want = &clean.pcm[target * 1536 * n..(target + 1) * 1536 * n];
        // Only the dither sequence differs from a continuous decode.
        let worst = pcm
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-3, "{name}: {worst}");
        assert!(snr(&pcm, want) > 55.0, "{name}");
    }
}

#[test]
fn the_sample_rate_is_locked_and_a_change_is_refused() {
    let a = units(&fixture("ac3_stereo_44100.ac3"));
    let b = units(&fixture("ac3_3.0-back_48000.ac3"));
    let mut decoder = open(CodecId::Ac3, Options::default());
    decoder.push(&packet(&a[0], 0, 0)).expect("push");
    assert!(matches!(
        decoder.push(&packet(&b[0], 1, 0)),
        Err(CodecError::Malformed {
            what: "sample rate change",
            ..
        })
    ));
    // The stream goes on.
    decoder.push(&packet(&a[1], 1, 0)).expect("push");
}

#[test]
fn a_layout_change_keeps_the_first_layout() {
    let stereo = units(&fixture("ac3_3.0-back_48000.ac3"));
    let five = units(&fixture("ac3_5.1_48000_xbsi.ac3"));
    let mut decoder = open(CodecId::Ac3, Options::default());
    decoder.push(&packet(&stereo[0], 0, 0)).expect("push");
    let outputs = decoder.push(&packet(&five[3], 1, 0)).expect("push");
    let Some(Output::Audio(audio)) = outputs.first() else {
        panic!("audio")
    };
    assert_eq!(audio.pcm.layout.channels(), &[L, R, Bc]);
}

#[test]
fn open_refuses_other_codecs_rates_and_programs() {
    use cutlass_ac3::Ac3Decoder;
    use cutlass_decode::{ChannelLayout, CodecParams};
    let params = |codec, rate| CodecParams::audio(codec, rate, ChannelLayout::stereo());
    assert!(Ac3Decoder::open(&params(CodecId::Aac, 48_000)).is_err());
    assert!(Ac3Decoder::open(&params(CodecId::Ac3, 22_050)).is_err());
    assert!(
        Ac3Decoder::with_options(
            &params(CodecId::Eac3, 48_000),
            Options {
                program: 8,
                ..Options::default()
            }
        )
        .is_err()
    );
    let mut decoder = Ac3Decoder::open(&params(CodecId::Ac3, 48_000)).expect("open");
    assert_eq!(decoder.codec(), CodecId::Ac3);
    let unit = &units(&fixture("ac3_stereo_44100.ac3"))[0];
    assert!(decoder.push(&packet(unit, 0, 0)).is_err());
}

// ---- malformed input ----

#[test]
fn every_truncation_of_a_frame_fails_without_harm() {
    for name in ["ac3_mono_48000.ac3", "eac3_mono_32000.eac3"] {
        let data = fixture(name);
        let all = units(&data);
        let clean = decode(&data);
        let mut decoder = open(codec_of(&data), Options::default());
        let mut pcm = samples(decoder.push(&packet(&all[0], 0, 0)).expect("push"));
        for len in 1..all[1].len() {
            assert!(
                decoder.push(&packet(&all[1][..len], 1, 0)).is_err(),
                "{name} {len}"
            );
        }
        for (index, unit) in all.iter().enumerate().skip(1) {
            pcm.extend(samples(
                decoder.push(&packet(unit, index, 0)).expect("push"),
            ));
        }
        assert_eq!(pcm, clean.pcm, "{name}");
    }
}

#[test]
fn bytes_after_the_last_frame_are_malformed() {
    let unit = &units(&fixture("ac3_stereo_44100.ac3"))[0];
    let mut decoder = open(CodecId::Ac3, Options::default());
    let mut extra = unit.clone();
    extra.extend_from_slice(&[0, 0, 0]);
    assert!(matches!(
        decoder.push(&packet(&extra, 0, 0)),
        Err(CodecError::Malformed { .. })
    ));
    let mut two = unit.clone();
    two.extend_from_slice(&[0x0B, 0x77, 0, 0, 0, 0, 0, 0, 0]);
    assert!(decoder.push(&packet(&two, 0, 0)).is_err());
}

#[test]
fn a_flipped_bit_in_any_frame_is_caught_by_the_crc() {
    for name in ["ac3_stereo_44100.ac3", "eac3_stereo_48000.eac3"] {
        let unit = &units(&fixture(name))[2];
        let mut decoder = open(codec_of(unit), Options::default());
        // Past the syncword, every bit is covered by a CRC.
        for bit in (16..unit.len() * 8).step_by(37) {
            let mut bad = unit.clone();
            bad[bit / 8] ^= 0x80 >> (bit % 8);
            assert!(
                decoder.push(&packet(&bad, 0, 0)).is_err(),
                "{name} bit {bit}"
            );
        }
    }
}
