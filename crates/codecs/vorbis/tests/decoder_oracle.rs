//! The decoder against libvorbis and FFmpeg's own decoder.
//!
//! Files come from libvorbis (GStreamer `vorbisenc`) and FFmpeg's native
//! encoder, made from `local-files` audio (a synthetic source without
//! it). Each is decoded here (packets read by a test-only Ogg reader, cut
//! to the granules) and by both references. Gates: the end granule's
//! sample count (every file starts at granule 0), and at least
//! `MIN_SNR_DB` against each reference over their common length.
//! libvorbis through GStreamer can give one sample more, and FFmpeg skips
//! the end trim when the first audio page is also the last.
//!
//! Without GStreamer's Vorbis plugin or ffmpeg the tests skip.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::print_stderr
)]

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use common::{
    decode_file_with_layout, ffmpeg_decode, has_ffmpeg, has_gst_vorbis, libvorbis_decode, reorder,
    run, scratch, snr_db, wav_order,
};

/// Both references decode in float; this crate differs by rounding only.
const MIN_SNR_DB: f64 = 90.0;

fn source(dir: &Path, rate: u32, channels: u32, seconds: u32) -> PathBuf {
    let out = dir.join(format!("source-{rate}-{channels}.wav"));
    let mut command = Command::new("ffmpeg");
    command.args(["-v", "error", "-y"]);
    match common::local_files() {
        Some(root) => {
            command
                .args(["-ss", "30", "-t", &seconds.to_string(), "-i"])
                .arg(root.join("06 First Person Shooter (feat. J. Cole).m4a"));
        }
        None => {
            command.args([
                "-f",
                "lavfi",
                "-i",
                &format!("aevalsrc=0.4*sin(440*2*PI*t)|0.3*sin(660*2*PI*t)+0.1*sin(5000*2*PI*t):d={seconds}"),
            ]);
        }
    }
    // Every channel gets its own mix, so none is silent or a copy.
    let pan = match channels {
        1 => "pan=mono|c0=0.5*c0+0.5*c1",
        2 => "pan=stereo|c0=c0|c1=c1",
        6 => "pan=5.1|FL=c0|FR=c1|FC=0.5*c0+0.5*c1|LFE=0.3*c0|BL=0.7*c0-0.3*c1|BR=0.7*c1-0.3*c0",
        8 => {
            "pan=7.1|FL=c0|FR=c1|FC=0.5*c0+0.5*c1|LFE=0.3*c0|BL=0.7*c0-0.3*c1|BR=0.7*c1-0.3*c0|SL=0.6*c0+0.2*c1|SR=0.6*c1+0.2*c0"
        }
        _ => "anull",
    };
    command
        .args(["-ar", &rate.to_string(), "-af"])
        .arg(pan)
        .args(["-c:a", "pcm_s16le"])
        .arg(&out);
    run(&mut command);
    out
}

fn libvorbis_encode(wav: &Path, out: &Path, setting: &str) {
    let mut command = Command::new("gst-launch-1.0");
    command
        .arg("-q")
        .arg("filesrc")
        .arg(format!("location={}", wav.display()))
        .args(["!", "wavparse", "!", "audioconvert", "!", "vorbisenc"]);
    for part in setting.split_whitespace() {
        command.arg(part);
    }
    command
        .args(["!", "oggmux", "!", "filesink"])
        .arg(format!("location={}", out.display()));
    run(&mut command);
}

struct Case {
    name: &'static str,
    rate: u32,
    channels: u32,
    setting: &'static str,
}

fn check(dir: &Path, file: &Path, name: &str) {
    let bytes = std::fs::read(file).unwrap();
    let (rate, layout, ours) = decode_file_with_layout(&bytes);
    let channels = layout.len();
    // The references give WAV speaker order; this decoder the Vorbis order,
    // labelled.
    let ours = reorder(&ours, &layout, &wav_order(channels));
    let ffmpeg = ffmpeg_decode(file, dir);
    let libvorbis = libvorbis_decode(file, dir);
    let frames = ours.len() / channels;
    let ffmpeg_snr = snr_db(&ffmpeg, &ours, channels);
    let libvorbis_snr = snr_db(&libvorbis, &ours, channels);
    eprintln!(
        "{name}: {rate} Hz, {channels} ch, {frames} samples (ffmpeg {}, libvorbis {}); SNR {ffmpeg_snr:.1} dB against ffmpeg, {libvorbis_snr:.1} dB against libvorbis",
        ffmpeg.len() / channels,
        libvorbis.len() / channels,
    );
    let end = common::ogg_packets(&bytes)
        .iter()
        .rev()
        .find_map(|p| p.granule)
        .unwrap();
    assert_eq!(frames as i64, end, "{name}: the end granule's sample count");
    assert!(
        ffmpeg_snr >= MIN_SNR_DB,
        "{name}: {ffmpeg_snr} dB against ffmpeg"
    );
    assert!(
        libvorbis_snr >= MIN_SNR_DB,
        "{name}: {libvorbis_snr} dB against libvorbis"
    );
}

#[test]
fn libvorbis_files_decode_like_libvorbis_and_ffmpeg() {
    if !has_gst_vorbis() || !has_ffmpeg() {
        eprintln!("skipped: needs GStreamer vorbisenc/vorbisdec and ffmpeg");
        return;
    }
    let dir = scratch("decoder-oracle");
    let cases = [
        Case {
            name: "q-0.1 stereo 44.1k",
            rate: 44_100,
            channels: 2,
            setting: "quality=-0.1",
        },
        Case {
            name: "q0.4 stereo 44.1k",
            rate: 44_100,
            channels: 2,
            setting: "quality=0.4",
        },
        Case {
            name: "q1.0 stereo 48k",
            rate: 48_000,
            channels: 2,
            setting: "quality=1.0",
        },
        Case {
            name: "q0.6 mono 22.05k",
            rate: 22_050,
            channels: 1,
            setting: "quality=0.6",
        },
        Case {
            name: "q0.1 mono 8k",
            rate: 8_000,
            channels: 1,
            setting: "quality=0.1",
        },
        Case {
            name: "q0.5 stereo 96k",
            rate: 96_000,
            channels: 2,
            setting: "quality=0.5",
        },
        Case {
            name: "managed 64k stereo 44.1k",
            rate: 44_100,
            channels: 2,
            setting: "managed=true bitrate=64000",
        },
        Case {
            name: "q0.4 5.1 48k",
            rate: 48_000,
            channels: 6,
            setting: "quality=0.4",
        },
        Case {
            name: "q0.4 7.1 48k",
            rate: 48_000,
            channels: 8,
            setting: "quality=0.4",
        },
    ];
    for case in &cases {
        let wav = source(&dir, case.rate, case.channels, 10);
        let file = dir.join(format!("{}.ogg", case.name.replace(' ', "_")));
        libvorbis_encode(&wav, &file, case.setting);
        check(&dir, &file, case.name);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn fixtures_decode_like_libvorbis_and_ffmpeg() {
    if !has_gst_vorbis() || !has_ffmpeg() {
        eprintln!("skipped: needs GStreamer vorbisdec and ffmpeg");
        return;
    }
    let dir = scratch("decoder-fixtures");
    for name in [
        "stereo_44100_q4.ogg",
        "clicks_48000_q9.ogg",
        "sweep_8000_mono_q0.ogg",
        "surround51_48000_q4.ogg",
        "ffmpeg_stereo_44100.ogg",
    ] {
        let file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name);
        check(&dir, &file, name);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
