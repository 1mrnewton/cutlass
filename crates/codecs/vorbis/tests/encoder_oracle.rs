//! The Vorbis encoder against libvorbis and FFmpeg's native decoder.
//!
//! Gates, fixed in `STATUS.md` and not lowered after a failure:
//!
//! 1. Every item, muxed by `OggMux`, decodes in libvorbis and in FFmpeg's
//!    native Vorbis decoder to exactly the input's sample count. Both agree
//!    with this crate's decoder at 90 dB. The decoder here reads the Ogg
//!    granules, so a trim that only this crate applies does not pass.
//! 2. Quality mode spends more bits at a higher quality. Silence stays at
//!    or under 2 kb/s of audio packets.
//! 3. With a bitrate, the mean over the dense item is within 10% of the
//!    request.
//! 4. At libvorbis's rate for quality 0.4 and 0.6 (stereo 44.1 kHz), this
//!    encoder's mean per-band log-spectral distance (critical bands,
//!    decoded by libvorbis) is at most 1.5 dB above libvorbis at that rate,
//!    and the rate is within 10%.
//!
//! Without GStreamer's Vorbis plugin or ffmpeg the test skips.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::print_stderr
)]

mod common;

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;

use common::{
    decode_file_with_layout, ffmpeg_decode, has_ffmpeg, has_gst_vorbis, libvorbis_decode, reorder,
    run, scratch, snr_db, wav_order,
};
use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
use cutlass_container::{Mux, NewStream, TrackDuration};
use cutlass_encode::{AudioInput, Channel, ChannelLayout, CodecParams, Encoder, Pcm, SideData};
use cutlass_ogg::OggMux;
use cutlass_vorbis::{Options, VorbisEncoder};

const MIN_SNR_DB: f64 = 90.0;
const RATE_TOLERANCE: f64 = 0.10;
const LSD_MARGIN_DB: f64 = 1.5;
const SILENCE_BITS_PER_SECOND: f64 = 2_000.0;

fn layout(channels: usize) -> ChannelLayout {
    let speakers = match channels {
        1 => vec![Channel::FrontCenter],
        2 => vec![Channel::FrontLeft, Channel::FrontRight],
        6 => vec![
            Channel::FrontLeft,
            Channel::FrontCenter,
            Channel::FrontRight,
            Channel::BackLeft,
            Channel::BackRight,
            Channel::Lfe,
        ],
        _ => panic!("no Vorbis layout for {channels}"),
    };
    ChannelLayout::new(speakers).expect("layout")
}

struct Job {
    pcm: Vec<f32>,
    channels: usize,
    rate: u32,
    quality: f32,
    bitrate: u32,
}

/// Encodes `job` and muxes it. Returns the audio-packet bytes, headers aside.
fn encode_ogg(job: &Job, out: &Path) -> usize {
    let mut params = CodecParams::audio(CodecId::Vorbis, job.rate, layout(job.channels));
    params.bitrate = job.bitrate;
    let mut encoder = VorbisEncoder::open_with(
        &params,
        Options {
            quality: job.quality,
        },
    )
    .expect("open");
    let pcm = Pcm::from_f32(job.rate, layout(job.channels), job.pcm.clone()).expect("pcm");
    let mut packets = encoder.push(&AudioInput { pcm }).expect("push");
    packets.extend(encoder.drain().expect("drain"));
    let config = packets[0]
        .side
        .iter()
        .find_map(|side| match side {
            SideData::Config(config) => Some(config.clone()),
            _ => None,
        })
        .expect("headers");
    let mut mux = OggMux::new(File::create(out).expect("create"));
    let track = mux
        .add_stream(NewStream {
            params: StreamParams::try_audio(
                StreamCodec::Known(CodecId::Vorbis),
                job.rate,
                u16::try_from(job.channels).expect("channels"),
                config,
            )
            .expect("params"),
            timescale: Timescale::new(job.rate),
            language: None,
        })
        .expect("stream");
    let mut ticks = 0_i64;
    for packet in &packets {
        let stamp = Timestamp::try_new(Some(packet.pts), Some(packet.pts), Some(packet.duration))
            .expect("time");
        let written = Packet::try_new(
            track.get(),
            packet.data.clone(),
            stamp,
            Timescale::new(job.rate),
            true,
        )
        .expect("packet");
        mux.write_packet(&written).expect("write");
        mux.interleave().expect("interleave");
        ticks += packet.duration;
    }
    let finished = mux
        .finish(&[TrackDuration {
            track,
            timescale: Timescale::new(job.rate),
            ticks,
        }])
        .expect("finish");
    assert!(finished.is_complete());
    packets.iter().map(|packet| packet.data.len()).sum()
}

fn write_wav(dir: &Path, pcm: &[f32], rate: u32, channels: usize) -> PathBuf {
    let raw = dir.join("source.f32");
    let wav = dir.join("source.wav");
    let bytes: Vec<u8> = pcm.iter().flat_map(|sample| sample.to_le_bytes()).collect();
    std::fs::write(&raw, bytes).unwrap();
    run(Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "f32le", "-ar"])
        .arg(rate.to_string())
        .args(["-ac"])
        .arg(channels.to_string())
        .arg("-i")
        .arg(&raw)
        .args(["-c:a", "pcm_s16le"])
        .arg(&wav));
    wav
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

fn audio_bytes(file: &Path) -> usize {
    let bytes = std::fs::read(file).unwrap();
    common::ogg_packets(&bytes)
        .iter()
        .skip(3)
        .map(|packet| packet.data.len())
        .sum()
}

fn kbps(bytes: usize, frames: usize, rate: u32) -> f64 {
    bytes as f64 * 8.0 * f64::from(rate) / frames as f64 / 1000.0
}

/// Both references, and this decoder, on one muxed file.
fn decoded(dir: &Path, file: &Path, channels: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let bytes = std::fs::read(file).unwrap();
    let (_, layout, ours) = decode_file_with_layout(&bytes);
    let ours = reorder(&ours, &layout, &wav_order(channels));
    let ffmpeg = ffmpeg_decode(file, dir);
    let libvorbis = libvorbis_decode(file, dir);
    (ours, ffmpeg, libvorbis)
}

fn check_interop(
    dir: &Path,
    file: &Path,
    frames: usize,
    channels: usize,
    name: &str,
    failures: &mut Vec<String>,
) {
    let want = frames * channels;
    let (ours, ffmpeg, libvorbis) = decoded(dir, file, channels);
    let ffmpeg_snr = snr_db(&ffmpeg, &ours, channels);
    let libvorbis_snr = snr_db(&libvorbis, &ours, channels);
    eprintln!(
        "{name}: ours {} ffmpeg {} libvorbis {} samples; {ffmpeg_snr:.1} dB / {libvorbis_snr:.1} dB",
        ours.len() / channels,
        ffmpeg.len() / channels,
        libvorbis.len() / channels,
    );
    // GStreamer `vorbisdec` emits one sample past the end granule.
    let libvorbis_frames = libvorbis.len() / channels;
    if ours.len() != want
        || ffmpeg.len() != want
        || (libvorbis_frames != frames && libvorbis_frames != frames + 1)
    {
        failures.push(format!(
            "{name}: sample count ours {} ffmpeg {} libvorbis {libvorbis_frames}, want {frames}",
            ours.len() / channels,
            ffmpeg.len() / channels,
        ));
    }
    if ffmpeg_snr < MIN_SNR_DB {
        failures.push(format!("{name}: {ffmpeg_snr:.1} dB against ffmpeg"));
    }
    if libvorbis_snr < MIN_SNR_DB {
        failures.push(format!("{name}: {libvorbis_snr:.1} dB against libvorbis"));
    }
}

fn tone(frames: usize, rate: u32, channels: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames * channels);
    for i in 0..frames {
        for channel in 0..channels {
            let freq = 220.0 * (channel as f32 + 1.0) * 1.5;
            let phase = 2.0 * std::f32::consts::PI * freq * i as f32 / rate as f32;
            out.push(0.4 * phase.sin());
        }
    }
    out
}

/// Tones plus noise, loud enough that a bitrate target has something to spend on.
fn dense(frames: usize, rate: u32) -> Vec<f32> {
    let mut state = 0x1234_5678_u32;
    let mut out = Vec::with_capacity(frames * 2);
    for i in 0..frames {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let noise = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
        let t = i as f32 / rate as f32;
        let left = 0.25 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            + 0.15 * (2.0 * std::f32::consts::PI * 1_250.0 * t).sin()
            + 0.12 * noise;
        let right = 0.2 * (2.0 * std::f32::consts::PI * 660.0 * t).sin()
            + 0.12 * (2.0 * std::f32::consts::PI * 3_000.0 * t).sin()
            + 0.1 * noise;
        out.push(left);
        out.push(right);
    }
    out
}

fn bark(hz: f64) -> f64 {
    13.0 * (0.000_76 * hz).atan() + 3.5 * (hz / 7_500.0).powi(2).atan()
}

fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (s, c) = (ang * k as f64).sin_cos();
                let (a, b) = (start + k, start + k + len / 2);
                let tr = re[b] * c - im[b] * s;
                let ti = re[b] * s + im[b] * c;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len <<= 1;
    }
}

/// One-Bark band edges, as FFT bins of an `n`-point transform at `rate`.
fn critical_edges(rate: u32, n: usize) -> Vec<usize> {
    let bins = n / 2;
    let hz = |bin: usize| bin as f64 * f64::from(rate) / n as f64;
    let mut edges = vec![1_usize];
    let mut start = bark(hz(1));
    for bin in 2..bins {
        if bark(hz(bin)) - start >= 1.0 {
            edges.push(bin);
            start = bark(hz(bin));
        }
    }
    if *edges.last().unwrap() != bins {
        edges.push(bins);
    }
    edges
}

fn band_powers(frame: &[f64], edges: &[usize]) -> Vec<f64> {
    let n = 2048;
    let mut re = vec![0.0; n];
    let mut im = vec![0.0; n];
    for (i, &sample) in frame.iter().take(n).enumerate() {
        let window = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * (i as f64 + 0.5) / n as f64).cos();
        re[i] = sample * window;
    }
    fft(&mut re, &mut im);
    edges
        .windows(2)
        .map(|pair| {
            (pair[0]..pair[1])
                .map(|bin| re[bin] * re[bin] + im[bin] * im[bin])
                .sum::<f64>()
        })
        .collect()
}

/// Mean per-frame log-spectral distance, in dB, over critical bands.
/// Frames quieter than −70 dBFS are left out.
fn lsd(reference: &[f32], test: &[f32], channels: usize, rate: u32) -> f64 {
    let n = 2048;
    let hop = 1024;
    let edges = critical_edges(rate, n);
    let bands = edges.len() - 1;
    let frames = reference.len() / channels;
    let mut distances = Vec::new();
    let mut start = 0;
    while start + n <= frames {
        for channel in 0..channels {
            let sample = |pcm: &[f32], i: usize| f64::from(pcm[(start + i) * channels + channel]);
            let energy = (0..n).map(|i| sample(reference, i).powi(2)).sum::<f64>() / n as f64;
            if energy < 1e-7 {
                continue;
            }
            let reference_frame: Vec<f64> = (0..n).map(|i| sample(reference, i)).collect();
            let test_frame: Vec<f64> = (0..n).map(|i| sample(test, i)).collect();
            let left = band_powers(&reference_frame, &edges);
            let right = band_powers(&test_frame, &edges);
            let mean = left
                .iter()
                .zip(&right)
                .map(|(a, b)| (10.0 * ((a + 1e-12) / (b + 1e-12)).log10()).powi(2))
                .sum::<f64>()
                / bands as f64;
            distances.push(mean.sqrt());
        }
        start += hop;
    }
    if distances.is_empty() {
        return 0.0;
    }
    distances.iter().sum::<f64>() / distances.len() as f64
}

#[test]
fn the_encoder_meets_the_independent_decoder_gates() {
    if !has_ffmpeg() || !has_gst_vorbis() {
        eprintln!("skipped: needs GStreamer vorbisenc/vorbisdec and ffmpeg");
        return;
    }
    let dir = scratch("encoder");
    let mut failures = Vec::new();

    let interop = [
        (
            "mono 44.1 kHz q0.4",
            1_usize,
            44_100_u32,
            132_300_usize,
            0.4_f32,
            0_u32,
        ),
        ("stereo 44.1 kHz q0.3", 2, 44_100, 88_200, 0.3, 0),
        ("stereo 48 kHz q0.7", 2, 48_000, 96_000, 0.7, 0),
        ("5.1 48 kHz q0.5", 6, 48_000, 48_000, 0.5, 0),
        ("mono 8 kHz q0.2", 1, 8_000, 24_000, 0.2, 0),
        ("stereo 96 kHz q0.4", 2, 96_000, 192_000, 0.4, 0),
    ];
    for (name, channels, rate, frames, quality, bitrate) in interop {
        let pcm = tone(frames, rate, channels);
        let file = dir.join(format!("{name}.ogg"));
        encode_ogg(
            &Job {
                pcm: pcm.clone(),
                channels,
                rate,
                quality,
                bitrate,
            },
            &file,
        );
        check_interop(&dir, &file, frames, channels, name, &mut failures);
    }

    let silent_frames = 88_200;
    let silence = Job {
        pcm: vec![0.0; silent_frames * 2],
        channels: 2,
        rate: 44_100,
        quality: 0.5,
        bitrate: 0,
    };
    let silence_file = dir.join("silence.ogg");
    let silence_bytes = encode_ogg(&silence, &silence_file);
    let silence_rate = kbps(silence_bytes, silent_frames, 44_100);
    eprintln!("silence: {silence_rate:.2} kb/s of audio packets");
    check_interop(
        &dir,
        &silence_file,
        silent_frames,
        2,
        "silence",
        &mut failures,
    );
    if silence_rate > SILENCE_BITS_PER_SECOND / 1000.0 {
        failures.push(format!("silence: {silence_rate:.2} kb/s"));
    }

    let climb_frames = 44_100;
    let climb = tone(climb_frames, 44_100, 2);
    let low = dir.join("q-low.ogg");
    let high = dir.join("q-high.ogg");
    let low_bytes = encode_ogg(
        &Job {
            pcm: climb.clone(),
            channels: 2,
            rate: 44_100,
            quality: 0.2,
            bitrate: 0,
        },
        &low,
    );
    let high_bytes = encode_ogg(
        &Job {
            pcm: climb,
            channels: 2,
            rate: 44_100,
            quality: 0.8,
            bitrate: 0,
        },
        &high,
    );
    eprintln!(
        "quality: q0.2 {:.1} kb/s, q0.8 {:.1} kb/s",
        kbps(low_bytes, climb_frames, 44_100),
        kbps(high_bytes, climb_frames, 44_100),
    );
    if high_bytes <= low_bytes {
        failures.push(format!(
            "quality: q0.8 used {high_bytes} bytes, q0.2 used {low_bytes}"
        ));
    }

    let dense_frames = 44_100 * 4;
    let dense = dense(dense_frames, 44_100);
    for bitrate in [64_000_u32, 128_000] {
        let file = dir.join(format!("abr-{bitrate}.ogg"));
        let bytes = encode_ogg(
            &Job {
                pcm: dense.clone(),
                channels: 2,
                rate: 44_100,
                quality: 0.4,
                bitrate,
            },
            &file,
        );
        let rate = kbps(bytes, dense_frames, 44_100);
        let ratio = rate * 1000.0 / f64::from(bitrate);
        eprintln!("bitrate {bitrate}: {rate:.1} kb/s ({ratio:.3} of the request)");
        check_interop(
            &dir,
            &file,
            dense_frames,
            2,
            &format!("abr {bitrate}"),
            &mut failures,
        );
        if (ratio - 1.0).abs() > RATE_TOLERANCE {
            failures.push(format!("bitrate {bitrate}: {rate:.1} kb/s"));
        }
    }

    let wav = write_wav(&dir, &dense, 44_100, 2);
    for quality in [0.4_f32, 0.6] {
        let reference = dir.join(format!("libvorbis-q{quality}.ogg"));
        libvorbis_encode(&wav, &reference, &format!("quality={quality}"));
        let reference_bytes = audio_bytes(&reference);
        let reference_rate = kbps(reference_bytes, dense_frames, 44_100);
        let bitrate = (reference_rate * 1000.0).round() as u32;
        let ours = dir.join(format!("matched-q{quality}.ogg"));
        let our_bytes = encode_ogg(
            &Job {
                pcm: dense.clone(),
                channels: 2,
                rate: 44_100,
                quality,
                bitrate,
            },
            &ours,
        );
        let our_rate = kbps(our_bytes, dense_frames, 44_100);
        let (_, _, reference_pcm) = decoded(&dir, &reference, 2);
        let (_, _, our_pcm) = decoded(&dir, &ours, 2);
        let reference_lsd = lsd(&dense, &reference_pcm, 2, 44_100);
        let our_lsd = lsd(&dense, &our_pcm, 2, 44_100);
        eprintln!(
            "q{quality}: libvorbis {reference_rate:.1} kb/s LSD {reference_lsd:.2} dB; ours {our_rate:.1} kb/s LSD {our_lsd:.2} dB"
        );
        let ratio = our_rate / reference_rate;
        if (ratio - 1.0).abs() > RATE_TOLERANCE {
            failures.push(format!(
                "matched q{quality}: {our_rate:.1} kb/s vs libvorbis {reference_rate:.1}"
            ));
        }
        if our_lsd > reference_lsd + LSD_MARGIN_DB {
            failures.push(format!(
                "LSD q{quality}: {our_lsd:.2} dB vs libvorbis {reference_lsd:.2}"
            ));
        }
    }

    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
