//! Streams from this encoder, written to MP4 by this workspace, decoded by
//! ffmpeg, AudioToolbox, and this crate, and compared with the PCM that
//! went in.

use std::path::{Path, PathBuf};

use super::decode_oracle::{EXPO, SPEECH, real_source};
use super::{
    Wav, afconvert, afconvert_samples, decode_all, demux_mp4, encode_all, ffmpeg, ffmpeg_layout,
    ffmpeg_samples, path_str, run, scratch, skip, to_frame_order, wav, write_m4a,
};
use crate::{Cookie, EncodeMode, EncoderConfig};

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

/// Encode `samples` (interleaved in ffmpeg's order) with `config`, write
/// `file`, and require every decoder to give them back. Returns the bytes
/// of all packets.
fn round_trip(
    file: &Path,
    config: EncoderConfig,
    samples: &[i32],
    ff: &Path,
    af: Option<&Path>,
) -> usize {
    let channels = usize::from(config.channels);
    let depth = u32::from(config.bit_depth);
    let (cookie, packets) = encode_all(config, &ffmpeg_layout(config.channels), samples);
    let parsed = Cookie::parse(&cookie).expect("cookie");
    assert_eq!(parsed.config.frame_length, config.frame_length);
    assert_eq!(parsed.config.bit_depth, config.bit_depth);
    write_m4a(file, &cookie, &packets);

    let track = demux_mp4(file);
    assert_eq!(track.cookie, cookie, "the cookie survives the file");
    assert_eq!(usize::from(track.channels), channels);
    assert_eq!(track.rate, config.sample_rate);
    let frame_order = to_frame_order(samples, channels);
    let (ours, _) = decode_all(&track.cookie, &track.packets);
    assert!(
        ours == frame_order,
        "this decoder differs on {}",
        file.display()
    );

    let theirs = ffmpeg_samples(ff, file, depth);
    assert_eq!(
        theirs.len(),
        samples.len(),
        "ffmpeg length {}",
        file.display()
    );
    assert!(theirs == samples, "ffmpeg differs on {}", file.display());
    if let Some(af) = af {
        let apple = afconvert_samples(af, file, depth);
        assert_eq!(
            apple.len(),
            samples.len(),
            "afconvert length {}",
            file.display()
        );
        if apple != frame_order {
            let map = super::channel_map(&apple, &frame_order, channels);
            panic!("afconvert differs on {}: map {map:?}", file.display());
        }
    }
    packets.iter().map(|packet| packet.data.len()).sum()
}

fn source(ff: &Path, dir: &Path, name: &str, seconds: &str, codec: &str) -> Option<Wav> {
    source_with(ff, dir, name, seconds, codec, None)
}

fn source_with(
    ff: &Path,
    dir: &Path,
    name: &str,
    seconds: &str,
    codec: &str,
    filter: Option<&str>,
) -> Option<Wav> {
    let path = real_source(ff, dir, name, seconds, codec, filter)?;
    Some(wav(&std::fs::read(path).expect("wav")))
}

fn config(source: &Wav, depth: u8, frame_length: u32, mode: EncodeMode) -> EncoderConfig {
    EncoderConfig {
        sample_rate: source.rate,
        channels: u8::try_from(source.channels).expect("channels"),
        bit_depth: depth,
        frame_length,
        mode,
    }
}

#[test]
fn real_stereo_in_both_modes_decodes_back_exactly() {
    let Some((ff, af)) = oracles() else { return };
    let dir = scratch("enc-stereo");
    for (codec, depth) in [("pcm_s16le", 16u8), ("pcm_s24le", 24)] {
        let Some(source) = source(&ff, &dir, EXPO, "12", codec) else {
            return;
        };
        assert_eq!(source.bits, u16::from(depth));
        for mode in [EncodeMode::Default, EncodeMode::Fast] {
            for frame_length in [4096, 1001] {
                let file = dir.join(format!("{depth}-{mode:?}-{frame_length}.m4a"));
                let config = config(&source, depth, frame_length, mode);
                round_trip(&file, config, &source.samples, &ff, af.as_deref());
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn real_mono_speech_decodes_back_exactly() {
    let Some((ff, af)) = oracles() else { return };
    let dir = scratch("enc-mono");
    let Some(source) = source(&ff, &dir, SPEECH, "20", "pcm_s16le") else {
        return;
    };
    assert_eq!(source.channels, 1);
    for frame_length in [4096, 4095] {
        let file = dir.join(format!("mono-{frame_length}.m4a"));
        let config = config(&source, 16, frame_length, EncodeMode::Default);
        round_trip(&file, config, &source.samples, &ff, af.as_deref());
    }
    let _ = std::fs::remove_dir_all(&dir);
}

const FIVE_ONE: &str =
    "pan=5.1|FL=c0|FR=c1|FC=0.5*c0+0.5*c1|LFE=0.3*c0|BL=0.7*c0-0.2*c1|BR=0.6*c1-0.1*c0";

#[test]
fn real_five_one_decodes_back_with_every_channel_in_place() {
    let Some((ff, af)) = oracles() else { return };
    let dir = scratch("enc-five-one");
    for (codec, depth) in [("pcm_s16le", 16u8), ("pcm_s24le", 24)] {
        let Some(source) = source_with(&ff, &dir, EXPO, "6", codec, Some(FIVE_ONE)) else {
            return;
        };
        assert_eq!(source.channels, 6);
        let file = dir.join(format!("five-one-{depth}.m4a"));
        let config = config(&source, depth, 4096, EncodeMode::Default);
        round_trip(&file, config, &source.samples, &ff, af.as_deref());
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn twenty_and_thirty_two_bit_decode_back_exactly() {
    let Some((ff, af)) = oracles() else { return };
    let dir = scratch("enc-wide");
    let Some(source) = source(&ff, &dir, EXPO, "8", "pcm_s32le") else {
        return;
    };
    for depth in [20u8, 32] {
        let samples: Vec<i32> = source
            .samples
            .iter()
            .map(|&v| v >> (32 - u32::from(depth)))
            .collect();
        for mode in [EncodeMode::Default, EncodeMode::Fast] {
            let file = dir.join(format!("{depth}-{mode:?}.m4a"));
            let config = config(&source, depth, 4096, mode);
            round_trip(&file, config, &samples, &ff, af.as_deref());
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn every_channel_count_decodes_back_in_place() {
    let Some((ff, af)) = oracles() else { return };
    let dir = scratch("enc-counts");
    for channels in 1..=8usize {
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
        let graph = format!(
            "aevalsrc=exprs='{}':c={layout}:s=48000:d=0.9",
            exprs.join("|")
        );
        for (codec, depth) in [("pcm_s16le", 16u8), ("pcm_s24le", 24)] {
            let wav_path = dir.join(format!("ch{channels}-{depth}.wav"));
            run(
                &ff,
                &[
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    &graph,
                    "-c:a",
                    codec,
                    path_str(&wav_path),
                ],
            );
            let source = wav(&std::fs::read(&wav_path).expect("wav"));
            assert_eq!(usize::from(source.channels), channels);
            // An odd frame length, so frames and the last partial frame
            // fall at different places for every count.
            let file = dir.join(format!("ch{channels}-{depth}.m4a"));
            let frame_length = 1001 + 2 * u32::try_from(channels).expect("count");
            let config = config(&source, depth, frame_length, EncodeMode::Default);
            round_trip(&file, config, &source.samples, &ff, af.as_deref());
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Sum of the ALAC packet bytes in an MP4.
fn payload_bytes(file: &Path) -> usize {
    demux_mp4(file).packets.iter().map(Vec::len).sum()
}

/// Compression against both other encoders on the same real audio. Both
/// modes follow Apple's encoder, so their frames must equal AudioToolbox's
/// at its default and at its lowest quality.
#[test]
fn compression_matches_the_reference_encoders() {
    let Some((ff, af)) = oracles() else { return };
    let dir = scratch("enc-ratio");
    for (codec, depth, flag) in [("pcm_s16le", 16u8, "alac/1"), ("pcm_s24le", 24, "alac/3")] {
        let Some(source_path) = real_source(&ff, &dir, EXPO, "10", codec, None) else {
            return;
        };
        let source = wav(&std::fs::read(&source_path).expect("wav"));
        let raw = source.samples.len() * usize::from(depth) / 8;
        let ours = round_trip(
            &dir.join(format!("ours-{depth}.m4a")),
            config(&source, depth, 4096, EncodeMode::Default),
            &source.samples,
            &ff,
            None,
        );
        let fast = round_trip(
            &dir.join(format!("fast-{depth}.m4a")),
            config(&source, depth, 4096, EncodeMode::Fast),
            &source.samples,
            &ff,
            None,
        );
        let by_ffmpeg = dir.join(format!("ff-{depth}.m4a"));
        run(
            &ff,
            &[
                "-v",
                "error",
                "-y",
                "-i",
                path_str(&source_path),
                "-c:a",
                "alac",
                path_str(&by_ffmpeg),
            ],
        );
        let ffmpeg_bytes = payload_bytes(&by_ffmpeg);
        let ratio = |bytes: usize| raw as f64 / bytes as f64;
        eprintln!(
            "{depth}-bit stereo, {raw} PCM bytes: ours {ours} ({:.4}), fast {fast} ({:.4}), ffmpeg {ffmpeg_bytes} ({:.4})",
            ratio(ours),
            ratio(fast),
            ratio(ffmpeg_bytes)
        );
        assert!(
            fast >= ours,
            "fast mode compresses no better than the search"
        );
        if let Some(af) = &af {
            let by_apple = dir.join(format!("af-{depth}.m4a"));
            run(
                af,
                &[
                    "-f",
                    "m4af",
                    "-d",
                    flag,
                    path_str(&source_path),
                    path_str(&by_apple),
                ],
            );
            // AudioToolbox's lowest qualities take its fast path.
            let apple_fast = dir.join(format!("af-fast-{depth}.m4a"));
            run(
                af,
                &[
                    "-f",
                    "m4af",
                    "-d",
                    flag,
                    "-q",
                    "0",
                    path_str(&source_path),
                    path_str(&apple_fast),
                ],
            );
            let apple = payload_bytes(&by_apple);
            let apple_fast_bytes = payload_bytes(&apple_fast);
            eprintln!(
                "{depth}-bit stereo: afconvert {apple} ({:.4}), afconvert -q 0 {apple_fast_bytes} ({:.4})",
                ratio(apple),
                ratio(apple_fast_bytes)
            );
            // Not only the same size: the same frames, byte for byte.
            let packets = |name: String| demux_mp4(&dir.join(name)).packets;
            assert!(
                packets(format!("ours-{depth}.m4a")) == demux_mp4(&by_apple).packets,
                "default mode differs from afconvert"
            );
            assert!(
                packets(format!("fast-{depth}.m4a")) == demux_mp4(&apple_fast).packets,
                "fast mode differs from afconvert -q 0"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
