//! The FFmpeg FATE samples: the tools FFmpeg's encoders do not write.
//!
//! Each sample is checked for the tools it must exercise, and against
//! Apple's decoder per channel. FFmpeg is the reference for the dynamic
//! range modes; on these samples it departs from Apple and from A/52 in
//! block switched blocks and in E-AC-3 LFE, so it is not the reference
//! for the audio.

use cutlass_ac3::{DynamicRange, Extension, Options, Tools};
use cutlass_decode::{Channel, CodecId, Decoder};

use crate::common::{
    Decoded, apple_decode, codec_of, decode_with, deinterleave, fate_dir, ffmpeg_decode, open,
    oracle, packet, samples, snr, units,
};

use Channel::{
    BackCenter as Bc, BackLeft as Lb, BackRight as Rb, FrontCenter as C, FrontLeft as L,
    FrontRight as R, Lfe, SideLeft as Ls, SideRight as Rs,
};

struct Sample {
    path: &'static str,
    layout: &'static [Channel],
    check: fn(&Tools) -> bool,
}

const SAMPLES: [Sample; 8] = [
    Sample {
        path: "ac3/monsters_inc_2.0_192_small.ac3",
        layout: &[L, R],
        check: |t| t.coupling_blocks > 0 && t.rematrix_blocks > 0 && t.short_blocks > 0,
    },
    Sample {
        path: "ac3/monsters_inc_5.1_448_small.ac3",
        layout: &[L, R, C, Lfe, Ls, Rs],
        check: |t| t.coupling_blocks > 0,
    },
    Sample {
        path: "ac3/millers_crossing_4.0.ac3",
        layout: &[L, R, C, Bc],
        check: |t| t.short_blocks > 0,
    },
    Sample {
        path: "eac3/csi_miami_5.1_256_spx_small.eac3",
        layout: &[L, R, C, Lfe, Ls, Rs],
        check: |t| t.spx_blocks > 0 && t.aht_channels > 0,
    },
    Sample {
        path: "eac3/csi_miami_stereo_128_spx_small.eac3",
        layout: &[L, R],
        check: |t| t.spx_blocks > 0 && t.aht_channels > 0 && t.rematrix_blocks > 0,
    },
    Sample {
        path: "eac3/matrix2_commentary1_stereo_192_small.eac3",
        layout: &[L, R],
        check: |t| t.coupling_blocks > 0 && t.rematrix_blocks > 0,
    },
    Sample {
        path: "eac3/serenity_english_5.1_1536_small.eac3",
        layout: &[L, R, C, Lfe, Ls, Rs],
        check: |t| t.short_frames == t.frames && t.blocks == t.frames,
    },
    Sample {
        path: "eac3/the_great_wall_7.1.eac3",
        layout: &[L, R, C, Lfe, Lb, Rb, Ls, Rs],
        check: |t| t.dependent_frames * 2 == t.frames,
    },
];

fn load(path: &str) -> Option<Vec<u8>> {
    let dir = fate_dir()?;
    let full = dir.join(path);
    match std::fs::read(&full) {
        Ok(bytes) => Some(bytes),
        Err(error) => panic!("{}: {error}", full.display()),
    }
}

fn decode_tools(data: &[u8], options: Options) -> (Decoded, Tools) {
    let mut decoder = open(codec_of(data), options);
    let mut pcm = Vec::new();
    for (index, unit) in units(data).iter().enumerate() {
        pcm.extend(samples(
            decoder.push(&packet(unit, index, 0)).expect("push"),
        ));
    }
    let (rate, layout) = decoder.stream().expect("stream");
    let tools = *decoder.tools();
    (
        Decoded {
            rate,
            channels: layout.channels().to_vec(),
            pcm,
        },
        tools,
    )
}

#[test]
fn each_sample_exercises_its_tools_in_its_layout() {
    for sample in &SAMPLES {
        let Some(data) = load(sample.path) else {
            return;
        };
        let (decoded, tools) = decode_tools(&data, Options::default());
        assert_eq!(decoded.channels, sample.layout, "{}", sample.path);
        assert!((sample.check)(&tools), "{}: {tools:?}", sample.path);
        assert!(decoded.pcm.iter().all(|s| s.is_finite()));
    }
}

#[test]
fn each_sample_matches_apple_per_channel() {
    let Some(afconvert) = oracle("afconvert", "-h") else {
        return;
    };
    for sample in &SAMPLES {
        let Some(data) = load(sample.path) else {
            return;
        };
        let (ours, _) = decode_tools(&data, Options::default());
        let (channels, reference) = apple_decode(&afconvert, &data);
        assert_eq!(channels, ours.channels.len(), "{}", sample.path);
        let mut used = vec![false; channels];
        for (index, speaker) in ours.channels.iter().enumerate() {
            let mine = &ours.channel(index)[256..];
            let rms = (mine.iter().map(|s| f64::from(*s).powi(2)).sum::<f64>() / mine.len() as f64)
                .sqrt();
            let (best, quality) = (0..channels)
                .map(|c| (c, snr(mine, &deinterleave(&reference, channels, c))))
                .fold((0, f64::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a });
            // Below -60 dBFS the content is dither and the match is noise.
            if rms < 1e-3 && *speaker != Lfe {
                continue;
            }
            assert!(!used[best], "{} {speaker:?} matched twice", sample.path);
            used[best] = true;
            let floor = if *speaker == Lfe { 80.0 } else { 25.0 };
            assert!(
                quality > floor,
                "{} {speaker:?}: {quality:.1} dB",
                sample.path
            );
        }
    }
}

/// Annex E 3.8.2: a core decode is the independent substream. It matches
/// a stream with the dependent frames removed, and it counts every
/// dependent frame it left out.
#[test]
fn a_core_decode_drops_the_dependent_substream() {
    let Some(data) = load("eac3/the_great_wall_7.1.eac3") else {
        return;
    };
    let (full, tools) = decode_tools(&data, Options::default());
    assert!(tools.dependent_frames > 0);
    let mut alone = Vec::new();
    for unit in units(&data) {
        let info = cutlass_ac3::probe(&unit).expect("independent frame");
        assert_ne!(info.stream_type, cutlass_ac3::StreamType::Dependent);
        alone.push(unit[..info.frame_bytes].to_vec());
    }
    let (core_only, _) = decode_tools(&alone.concat(), Options::default());
    let mut decoder = open(
        codec_of(&data),
        Options {
            extension: Extension::Core,
            ..Options::default()
        },
    );
    let mut pcm = Vec::new();
    for (index, unit) in units(&data).iter().enumerate() {
        pcm.extend(samples(
            decoder.push(&packet(unit, index, 0)).expect("push"),
        ));
    }
    assert_eq!(decoder.skipped_dependents(), tools.dependent_frames);
    let (_, layout) = decoder.stream().expect("stream");
    assert_eq!(layout.channels(), core_only.channels.as_slice());
    assert!(layout.len() < full.channels.len());
    assert_eq!(pcm.len(), core_only.pcm.len());
    let mut worst = 0.0_f32;
    let mut at = 0_usize;
    for (index, (left, right)) in pcm.iter().zip(&core_only.pcm).enumerate() {
        let delta = (left - right).abs();
        if delta > worst {
            worst = delta;
            at = index;
        }
    }
    assert!(
        worst == 0.0,
        "core differs from the independent frames at sample {at} by {worst}"
    );
}

#[test]
fn line_and_heavy_compression_match_ffmpeg() {
    let Some(ffmpeg) = oracle("ffmpeg", "-version") else {
        return;
    };
    let line = (
        DynamicRange::Line {
            cut: 1.0,
            boost: 1.0,
        },
        &["-drc_scale", "1"][..],
    );
    let heavy = (
        DynamicRange::Heavy,
        &["-drc_scale", "1", "-heavy_compr", "1"][..],
    );
    // FFmpeg's E-AC-3 header path never sets `compression_exists`, so its
    // heavy mode plays `dynrng` there; that mode is checked against the
    // stream's own `compr` words below.
    for (path, modes) in [
        ("ac3/monsters_inc_2.0_192_small.ac3", &[line, heavy][..]),
        (
            "eac3/matrix2_commentary1_stereo_192_small.eac3",
            &[line][..],
        ),
    ] {
        let Some(data) = load(path) else {
            return;
        };
        let off = decode_with(&data, Options::default());
        let off_ref = ffmpeg_decode(&ffmpeg, &data, &[]);
        let n = off.channels.len();
        // Root mean square of one channel's block.
        let level = |pcm: &[f32], block: usize, channel: usize| {
            let sum: f64 = (block * 256..block * 256 + 256)
                .map(|i| f64::from(pcm[i * n + channel]).powi(2))
                .sum();
            (sum / 256.0).sqrt()
        };
        for &(mode, flags) in modes {
            let ours = decode_with(
                &data,
                Options {
                    dynamic_range: mode,
                    ..Options::default()
                },
            );
            assert!(
                snr(&ours.pcm, &off.pcm) < 40.0,
                "{path}: {mode:?} is a no-op"
            );
            let reference = ffmpeg_decode(&ffmpeg, &data, flags);
            // The gain each decoder applied, block by block, agrees. A
            // whole-file SNR would weigh the boosted quiet blocks, where
            // dither dominates, and say less.
            let mut compared = 0;
            for block in 0..off.frames() / 256 {
                for channel in 0..n {
                    let base = level(&off.pcm, block, channel);
                    if base < 1e-3 {
                        continue;
                    }
                    let ours_gain = level(&ours.pcm, block, channel) / base;
                    let ref_gain =
                        level(&reference, block, channel) / level(&off_ref, block, channel);
                    let db = 20.0 * (ours_gain / ref_gain).log10();
                    assert!(db.abs() < 0.5, "{path} {mode:?} block {block}: {db:.2} dB");
                    compared += 1;
                }
            }
            assert!(compared > 100, "{path}: {compared} blocks");
        }
    }
}

#[test]
fn eac3_heavy_compression_applies_each_frames_compr_word() {
    let path = "eac3/matrix2_commentary1_stereo_192_small.eac3";
    let Some(data) = load(path) else {
        return;
    };
    let units = units(&data);
    let run = |mode| {
        let mut decoder = open(
            CodecId::Eac3,
            Options {
                dynamic_range: mode,
                ..Options::default()
            },
        );
        units
            .iter()
            .enumerate()
            .map(|(index, unit)| {
                let pcm = samples(decoder.push(&packet(unit, index, 0)).expect("push"));
                let compr = decoder.metadata().and_then(|m| m.compr_db);
                (pcm, compr)
            })
            .collect::<Vec<_>>()
    };
    let off = run(DynamicRange::Off);
    let heavy = run(DynamicRange::Heavy);
    let mut compared = 0;
    for ((base, _), (pcm, compr)) in off.iter().zip(&heavy) {
        let Some(db) = compr else {
            continue;
        };
        assert_eq!(base.len(), pcm.len());
        // Block 0 overlaps the previous frame, which has its own word.
        for block in 1..base.len() / 512 {
            let range = block * 512..block * 512 + 512;
            let energy = |x: &[f32]| {
                x[range.clone()]
                    .iter()
                    .map(|s| f64::from(*s).powi(2))
                    .sum::<f64>()
            };
            let e = energy(base);
            if e / 512.0 < 1e-6 {
                continue;
            }
            let gain = 10.0 * (energy(pcm) / e).log10();
            assert!(
                (gain - f64::from(*db)).abs() < 0.05,
                "{path} {gain:.3} dB against compr {db:.3} dB"
            );
            compared += 1;
        }
    }
    assert!(compared > 100, "{compared} blocks");
}
