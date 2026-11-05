//! Opus decode and encode speed beside the ffmpeg binary.
//!
//! Ten seconds of stereo (speech from `local-files` when it is there, else
//! a synthetic chord with noise). Decode: three libopus files (CELT 96k,
//! voip 24k with SILK and hybrid packets, SILK wideband mono 16k) read by
//! `OggDemux` and decoded here, against ffmpeg decoding the same file with
//! libopus and with its native decoder. Encode: this crate at 96k stereo
//! VBR and 128k CBR against ffmpeg's libopus. Each time is the median of
//! three runs after a warmup; ffmpeg's times start when its process is
//! launched, so they include its start-up and file reading.
//!
//! `cargo bench -p cutlass-opus --bench against_ffmpeg`

#![allow(clippy::print_stdout, clippy::expect_used, clippy::unwrap_used)]

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use cutlass_container::Demux;
use cutlass_encode::{AudioInput, ChannelLayout, CodecId, CodecParams, Encoder, Pcm};
use cutlass_ogg::OggDemux;
use cutlass_opus::{OpusDecoder, OpusEncoder, OpusHead};

const SECONDS: f64 = 10.0;

fn median(mut times: Vec<f64>) -> f64 {
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

fn timed(mut run: impl FnMut()) -> f64 {
    run();
    median(
        (0..3)
            .map(|_| {
                let started = Instant::now();
                run();
                started.elapsed().as_secs_f64()
            })
            .collect(),
    )
}

fn ffmpeg(args: &[&str]) {
    let status = Command::new("ffmpeg")
        .args(["-v", "error", "-y"])
        .args(args)
        .stdout(Stdio::null())
        .status()
        .expect("ffmpeg");
    assert!(status.success(), "ffmpeg {args:?}");
}

fn has_libopus() -> bool {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-encoders"])
        .output()
        .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).contains("libopus"))
}

fn source(dir: &Path) -> PathBuf {
    let raw = dir.join("source.f32");
    let speech = std::env::var_os("CUTLASS_LOCAL_FILES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/Users/mrnewton/dev/cutlass/local-files"))
        .join("GOOGLE IS BACK! (Gemini 3.8 Flash).wav");
    let raw_arg = raw.to_str().expect("path");
    if speech.is_file() {
        ffmpeg(&[
            "-ss",
            "30",
            "-t",
            "10",
            "-i",
            speech.to_str().expect("path"),
            "-ac",
            "2",
            "-ar",
            "48000",
            "-f",
            "f32le",
            raw_arg,
        ]);
    } else {
        ffmpeg(&[
            "-f",
            "lavfi",
            "-i",
            "aevalsrc=0.3*sin(2*PI*220*t)+0.05*(random(0)-0.5)|0.3*sin(2*PI*330*t):s=48000:d=10",
            "-f",
            "f32le",
            raw_arg,
        ]);
    }
    raw
}

fn packets(file: &Path) -> (OpusHead, Vec<Vec<u8>>) {
    let reader = File::open(file).expect("open");
    let length = reader.metadata().expect("metadata").len();
    let mut demux = OggDemux::open(reader, length).expect("ogg");
    let head = OpusHead::parse(demux.streams()[0].params().extradata()).expect("head");
    let mut out = Vec::new();
    while let Some(packet) = demux.read_packet().expect("read") {
        out.push(packet.bytes().to_vec());
    }
    (head, out)
}

fn main() {
    if !has_libopus() {
        println!("skip: ffmpeg with libopus is not installed");
        return;
    }
    let dir = std::env::temp_dir().join(format!("cutlass-opus-bench-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("dir");
    let raw = source(&dir);
    let raw_arg = raw.to_str().expect("path");
    let input = ["-f", "f32le", "-ar", "48000", "-ac", "2", "-i", raw_arg];
    println!("ten seconds, 48 kHz; real-time factor = audio seconds / wall seconds");
    println!("decode:");
    for (name, args) in [
        ("celt 96k stereo", vec!["-c:a", "libopus", "-b:a", "96k"]),
        (
            "voip 24k stereo (silk + hybrid)",
            vec!["-c:a", "libopus", "-b:a", "24k", "-application", "voip"],
        ),
        (
            "silk wb 16k mono",
            vec![
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "libopus",
                "-b:a",
                "16k",
                "-application",
                "voip",
            ],
        ),
    ] {
        let file = dir.join(format!("{}.opus", name.split(' ').next().unwrap_or("x")));
        let mut all = input.to_vec();
        all.extend(&args);
        all.push(file.to_str().expect("path"));
        ffmpeg(&all);
        let (head, packets) = packets(&file);
        let ours = timed(|| {
            let mut decoder = OpusDecoder::with_head(head.clone(), 48_000).expect("decoder");
            for packet in &packets {
                std::hint::black_box(decoder.decode(packet).expect("decode"));
            }
        });
        let file_arg = file.to_str().expect("path");
        let libopus = timed(|| ffmpeg(&["-c:a", "libopus", "-i", file_arg, "-f", "f32le", "-"]));
        let native = timed(|| ffmpeg(&["-c:a", "opus", "-i", file_arg, "-f", "f32le", "-"]));
        println!(
            "  {name}: ours {:.0}x, ffmpeg libopus {:.0}x, ffmpeg native {:.0}x",
            SECONDS / ours,
            SECONDS / libopus,
            SECONDS / native
        );
    }
    println!("encode:");
    let samples: Vec<f32> = std::fs::read(&raw)
        .expect("raw")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b).clamp(-1.0, 1.0))
        .collect();
    for (bitrate, cbr) in [(96_000u32, false), (128_000, true)] {
        let ours = timed(|| {
            let mut params = CodecParams::audio(CodecId::Opus, 48_000, ChannelLayout::stereo());
            params.bitrate = bitrate;
            params.constant_rate = cbr;
            let mut encoder = OpusEncoder::open(&params).expect("encoder");
            for chunk in samples.chunks(2 * 4800) {
                let pcm =
                    Pcm::from_f32(48_000, ChannelLayout::stereo(), chunk.to_vec()).expect("pcm");
                std::hint::black_box(encoder.push(&AudioInput { pcm }).expect("push"));
            }
            std::hint::black_box(encoder.drain().expect("drain"));
        });
        let out = dir.join("libopus.opus");
        let rate = format!("{}", bitrate);
        let mut all = input.to_vec();
        all.extend([
            "-c:a",
            "libopus",
            "-b:a",
            &rate,
            "-vbr",
            if cbr { "off" } else { "on" },
            out.to_str().expect("path"),
        ]);
        let libopus = timed(|| ffmpeg(&all));
        println!(
            "  {} kb/s {}: ours {:.0}x, ffmpeg libopus {:.0}x",
            bitrate / 1000,
            if cbr { "CBR" } else { "VBR" },
            SECONDS / ours,
            SECONDS / libopus
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
