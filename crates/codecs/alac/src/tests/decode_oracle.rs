//! Streams from two independent encoders, decoded here and by two
//! independent decoders, compared sample for sample.

use std::path::{Path, PathBuf};

use super::{
    afconvert, afconvert_samples, channel_map, decode_all, demux_mp4, ffmpeg, ffmpeg_samples,
    local, path_str, run, scratch, skip,
};
use crate::Cookie;

/// Where libavcodec puts each ALAC channel, by channel count
/// (`ff_alac_channel_layout_offsets`).
pub(crate) const FFMPEG_ORDER: [&[usize]; 8] = [
    &[0],
    &[0, 1],
    &[2, 0, 1],
    &[2, 0, 1, 3],
    &[2, 0, 1, 3, 4],
    &[2, 0, 1, 4, 5, 3],
    &[2, 0, 1, 4, 5, 6, 3],
    &[2, 6, 7, 0, 1, 4, 5, 3],
];

pub(crate) const EXPO: &str = "Expo - You Don't Need a Mac to Build iOS Apps Anymore.m4a";
pub(crate) const SPEECH: &str = "GOOGLE IS BACK! (Gemini 3.8 Flash).wav";

/// The first `seconds` of a real file as PCM WAV, `codec` one of ffmpeg's
/// `pcm_s16le`, `pcm_s24le`, `pcm_s32le`. `filter` reshapes the channels.
pub(crate) fn real_source(
    ffmpeg: &Path,
    dir: &Path,
    name: &str,
    seconds: &str,
    codec: &str,
    filter: Option<&str>,
) -> Option<PathBuf> {
    let Some(input) = local(name) else {
        skip("../local-files");
        return None;
    };
    let out = dir.join(format!("source-{codec}.wav"));
    let mut args = vec!["-v", "error", "-y", "-t", seconds, "-i", path_str(&input)];
    if let Some(filter) = filter {
        args.extend(["-af", filter]);
    }
    args.extend(["-c:a", codec, path_str(&out)]);
    run(ffmpeg, &args);
    Some(out)
}

/// Decode `file` here and with each available oracle, and require the same
/// integers. Returns the decoded samples and the channel count.
pub(crate) fn agrees(file: &Path, af: Option<&Path>, ff: &Path) -> (Vec<i32>, usize) {
    let track = demux_mp4(file);
    let cookie = Cookie::parse(&track.cookie).expect("cookie");
    let depth = u32::from(cookie.config.bit_depth);
    let channels = usize::from(cookie.config.channels);
    assert_eq!(usize::from(track.channels), channels);
    let (ours, counts) = decode_all(&track.cookie, &track.packets);
    let expected: i64 = track.durations.iter().sum();
    for (count, duration) in counts.iter().zip(&track.durations) {
        assert!(
            i64::try_from(*count).expect("count") >= *duration,
            "a frame holds fewer samples than its packet lasts"
        );
    }
    let total = ours.len() / channels;
    let reference = ffmpeg_samples(ff, file, depth);
    let ours_trimmed = &ours[..usize::try_from(expected).expect("duration") * channels];
    if total as i64 == expected {
        assert_eq!(reference.len(), ours.len(), "{}", file.display());
    }
    let reference = &reference[..ours_trimmed.len().min(reference.len())];
    assert_eq!(reference.len(), ours_trimmed.len(), "{}", file.display());
    let map = channel_map(ours_trimmed, reference, channels);
    assert_eq!(map, FFMPEG_ORDER[channels - 1], "{}", file.display());
    if let Some(af) = af {
        let apple = afconvert_samples(af, file, depth);
        assert_eq!(apple.len(), ours_trimmed.len(), "{}", file.display());
        assert!(
            apple == ours_trimmed,
            "afconvert differs on {}",
            file.display()
        );
    }
    (ours, channels)
}

fn oracles() -> Option<(PathBuf, Option<PathBuf>)> {
    let Some(ff) = ffmpeg() else {
        skip("ffmpeg");
        return None;
    };
    let af = afconvert();
    if af.is_none() {
        skip("afconvert");
    }
    Some((ff, af))
}

#[test]
fn real_stereo_from_both_encoders_decodes_like_both_decoders() {
    let Some((ff, af)) = oracles() else { return };
    let dir = scratch("stereo");
    for (codec, flag) in [("pcm_s16le", "alac/1"), ("pcm_s24le", "alac/3")] {
        let Some(source) = real_source(&ff, &dir, EXPO, "12", codec, None) else {
            return;
        };
        let by_ffmpeg = dir.join(format!("ff-{codec}.m4a"));
        run(
            &ff,
            &[
                "-v",
                "error",
                "-y",
                "-i",
                path_str(&source),
                "-c:a",
                "alac",
                path_str(&by_ffmpeg),
            ],
        );
        agrees(&by_ffmpeg, af.as_deref(), &ff);
        if let Some(af) = &af {
            let by_apple = dir.join(format!("af-{codec}.m4a"));
            run(
                af,
                &[
                    "-f",
                    "m4af",
                    "-d",
                    flag,
                    path_str(&source),
                    path_str(&by_apple),
                ],
            );
            agrees(&by_apple, Some(af), &ff);
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn real_mono_speech_decodes_like_both_decoders() {
    let Some((ff, af)) = oracles() else { return };
    let dir = scratch("mono");
    let Some(source) = real_source(&ff, &dir, SPEECH, "20", "pcm_s16le", None) else {
        return;
    };
    let by_ffmpeg = dir.join("ff.m4a");
    run(
        &ff,
        &[
            "-v",
            "error",
            "-y",
            "-i",
            path_str(&source),
            "-c:a",
            "alac",
            path_str(&by_ffmpeg),
        ],
    );
    let (_, channels) = agrees(&by_ffmpeg, af.as_deref(), &ff);
    assert_eq!(channels, 1);
    if let Some(af) = &af {
        let by_apple = dir.join("af.m4a");
        run(
            af,
            &[
                "-f",
                "m4af",
                "-d",
                "alac",
                path_str(&source),
                path_str(&by_apple),
            ],
        );
        agrees(&by_apple, Some(af), &ff);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

const FIVE_ONE: &str =
    "pan=5.1|FL=c0|FR=c1|FC=0.5*c0+0.5*c1|LFE=0.3*c0|BL=0.7*c0-0.2*c1|BR=0.6*c1-0.1*c0";

#[test]
fn real_five_one_keeps_every_channel_in_place() {
    let Some((ff, af)) = oracles() else { return };
    let dir = scratch("five-one");
    for (codec, flag) in [("pcm_s16le", "alac/1"), ("pcm_s24le", "alac/3")] {
        let Some(source) = real_source(&ff, &dir, EXPO, "6", codec, Some(FIVE_ONE)) else {
            return;
        };
        let by_ffmpeg = dir.join(format!("ff-{codec}.m4a"));
        run(
            &ff,
            &[
                "-v",
                "error",
                "-y",
                "-i",
                path_str(&source),
                "-c:a",
                "alac",
                path_str(&by_ffmpeg),
            ],
        );
        agrees(&by_ffmpeg, af.as_deref(), &ff);
        if let Some(af) = &af {
            let by_apple = dir.join(format!("af-{codec}.m4a"));
            run(
                af,
                &[
                    "-f",
                    "m4af",
                    "-d",
                    flag,
                    path_str(&source),
                    path_str(&by_apple),
                ],
            );
            let track = demux_mp4(&by_apple);
            let cookie = Cookie::parse(&track.cookie).expect("cookie");
            assert_eq!(cookie.config.channels, 6);
            agrees(&by_apple, Some(af), &ff);
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn every_channel_count_decodes_in_reference_order() {
    let Some((ff, af)) = oracles() else { return };
    let dir = scratch("counts");
    for channels in 1..=8usize {
        for (fmt, depth) in [("s16p", 16u8), ("s32p", 24)] {
            // A different tone and noise level in each channel, so the map
            // can tell them apart. 0.7 s ends in a partial frame.
            let exprs: Vec<String> = (0..channels)
                .map(|c| {
                    format!(
                        "{}*sin(2*PI*{}*t)+0.0{}*(random({c})-0.5)",
                        0.3 + 0.05 * c as f64,
                        150 + 97 * c,
                        c + 1
                    )
                })
                .collect();
            // ffmpeg's encoder takes only its own layouts.
            let layout = [
                "mono",
                "stereo",
                "3.0",
                "4.0",
                "5.0",
                "5.1",
                "6.1(back)",
                "7.1(wide)",
            ][channels - 1];
            let source = format!(
                "aevalsrc=exprs='{}':c={layout}:s=44100:d=0.7",
                exprs.join("|")
            );
            let out = dir.join(format!("ch{channels}-{depth}.m4a"));
            run(
                &ff,
                &[
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    &source,
                    "-c:a",
                    "alac",
                    "-sample_fmt",
                    fmt,
                    path_str(&out),
                ],
            );
            let track = demux_mp4(&out);
            assert_eq!(
                Cookie::parse(&track.cookie)
                    .expect("cookie")
                    .config
                    .bit_depth,
                depth
            );
            let (_, decoded) = agrees(&out, af.as_deref(), &ff);
            assert_eq!(decoded, channels);
            if let Some(af) = &af {
                // The same signal through AudioToolbox's encoder, which
                // writes the layout in a `chan` box.
                let wav = dir.join(format!("ch{channels}-{depth}.wav"));
                let pcm = if depth == 16 {
                    "pcm_s16le"
                } else {
                    "pcm_s24le"
                };
                run(
                    &ff,
                    &[
                        "-v",
                        "error",
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        &source,
                        "-c:a",
                        pcm,
                        path_str(&wav),
                    ],
                );
                let by_apple = dir.join(format!("af-ch{channels}-{depth}.m4a"));
                let flag = if depth == 16 { "alac/1" } else { "alac/3" };
                run(
                    af,
                    &[
                        "-f",
                        "m4af",
                        "-d",
                        flag,
                        path_str(&wav),
                        path_str(&by_apple),
                    ],
                );
                let (_, decoded) = agrees(&by_apple, Some(af), &ff);
                assert_eq!(decoded, channels);
                let cookie = Cookie::parse(&demux_mp4(&by_apple).cookie).expect("cookie");
                if channels > 2 {
                    assert_eq!(cookie.layout_tag, Some(crate::LAYOUT_TAGS[channels - 1]));
                }
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn twenty_and_thirty_two_bit_streams_decode_like_both_decoders() {
    let Some((ff, af)) = oracles() else { return };
    let Some(af) = af else { return };
    let dir = scratch("wide");
    let Some(source) = real_source(&ff, &dir, EXPO, "8", "pcm_s32le", None) else {
        return;
    };
    for (flag, depth) in [("alac/2", 20u8), ("alac/4", 32)] {
        let file = dir.join(format!("af-{depth}.m4a"));
        run(
            &af,
            &["-f", "m4af", "-d", flag, path_str(&source), path_str(&file)],
        );
        let track = demux_mp4(&file);
        assert_eq!(
            Cookie::parse(&track.cookie)
                .expect("cookie")
                .config
                .bit_depth,
            depth
        );
        agrees(&file, Some(&af), &ff);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
