//! The Opus encoder against ffmpeg and libopus.
//!
//! `clippy.toml` allows `expect`, `panic`, and printing in `#[cfg(test)]`
//! modules. This file is a harness, so the same exception is named here.
//!
//! Gates, fixed before any tuning and never lowered after a failure:
//!
//! 1. Interop: every item, muxed by `OggMux`, decodes with ffmpeg's native
//!    Opus decoder and with libopus to exactly the input's sample count.
//!    (The in-crate tests decode every frame with this crate's decoder and
//!    compare final ranges.)
//! 2. Rate: on every non-sparse item, the VBR mean is within 10% of the
//!    request at 32 and 64 kb/s mono and 64, 96, and 128 kb/s stereo, and
//!    every CBR packet is exactly `bitrate * 20 ms / 8` bytes. Sparse items
//!    (silence, clicks, a pure sine) are reported, not gated: VBR rightly
//!    spends less there.
//! 3. Quality: at 64, 96, and 128 kb/s stereo VBR, on every item, the mean
//!    per-band log-spectral distance (21 CELT bands, 20 ms frames) of our
//!    decoded output is at most 1.5 dB above libopus's
//!    (`-c:a libopus -b:a <rate> -vbr on -application audio`), both decoded
//!    by libopus. p95 LSD and segmental SNR are reported. Perceptual
//!    quality is pending a human listening test.
//!
//! Items: speech, rain, thunder, sound effects, and film soundtracks from
//! `local-files` (10 s each), plus synthetic sine, sweep, clicks, white and
//! pink noise, silence, and stereo anti-phase. Media is extracted into a
//! temporary folder and never committed. Without ffmpeg (with libopus)
//! the test skips; without `local-files` it runs the synthetic items only.

#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::unwrap_used
)]

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
use cutlass_container::{Mux, NewStream, TrackDuration};
use cutlass_encode::{AudioInput, ChannelLayout, CodecParams, Encoder, Pcm};
use cutlass_ogg::OggMux;
use cutlass_opus::OpusEncoder;

const LSD_MARGIN_DB: f64 = 1.5;
const VBR_TOLERANCE: f64 = 0.10;
const QUALITY_RATES: [u32; 3] = [64_000, 96_000, 128_000];
const RATE_CASES: [(usize, u32); 5] = [
    (1, 32_000),
    (1, 64_000),
    (2, 64_000),
    (2, 96_000),
    (2, 128_000),
];
const SECONDS: usize = 10;
const RATE: usize = 48_000;

/// Recordings from `local-files`: name, start second, and a label.
const RECORDINGS: [(&str, u32, &str); 10] = [
    (
        "Expo - You Don't Need a Mac to Build iOS Apps Anymore.m4a",
        30,
        "speech-expo",
    ),
    (
        "GOOGLE IS BACK! (Gemini 3.8 Flash).wav",
        20,
        "speech-gemini",
    ),
    (
        "Seamless_gentle_rain_#1-1784676090807.mp3",
        5,
        "rain-gentle",
    ),
    ("rain_bedroom_night_30min.mp3", 120, "rain-bedroom"),
    ("thunder_distant_10s.mp3", 0, "thunder"),
    ("sfx_A_sin_20260721_231942.mp3", 0, "sfx-a"),
    ("sfx_Seaml_20260721_231840.mp3", 0, "sfx-seaml-a"),
    ("sfx_Seaml_20260721_231938.mp3", 0, "sfx-seaml-b"),
    (
        "Fireship - 5 open source tools that replaced my $320 mo AI stack.mp4",
        10,
        "film-fireship",
    ),
    (
        "DBX Is Almost the Perfect Free Database Client - Better Stack (1080p, h264).mp4",
        5,
        "film-dbx",
    ),
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Dense,
    Sparse,
}

struct Item {
    name: String,
    kind: Kind,
    /// Interleaved stereo, `SECONDS` long.
    stereo: Vec<f32>,
}

fn tool_works(name: &str, args: &[&str]) -> bool {
    Command::new(name)
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

fn has_libopus() -> bool {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-encoders"])
        .output()
        .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).contains("libopus"))
}

fn local_root() -> Option<PathBuf> {
    let candidates = [
        std::env::var_os("CUTLASS_LOCAL_FILES").map(PathBuf::from),
        Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files")),
        Some(PathBuf::from("/Users/mrnewton/dev/cutlass/local-files")),
    ];
    candidates.into_iter().flatten().find(|p| p.is_dir())
}

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cutlass-opus-oracle-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch");
    dir
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn write_raw(path: &Path, pcm: &[f32]) {
    let bytes: Vec<u8> = pcm.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes).expect("write raw");
}

/// Ten seconds of a recording as 48 kHz stereo floats, looped when the
/// file is shorter (the sound effects last five).
fn extract(path: &Path, start: u32) -> Option<Vec<f32>> {
    let out = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-stream_loop",
            "-1",
            "-ss",
            &start.to_string(),
            "-t",
            &SECONDS.to_string(),
            "-i",
        ])
        .arg(path)
        .args(["-vn", "-ac", "2", "-ar", "48000", "-f", "f32le", "-"])
        .output()
        .ok()?;
    let mut pcm = floats(&out.stdout);
    let want = SECONDS * RATE * 2;
    // A looped file can come up a few milliseconds short at the seam;
    // pad that with silence, but refuse anything clearly incomplete.
    if !out.status.success() || pcm.len() < want * 9 / 10 {
        return None;
    }
    pcm.resize(want, 0.0);
    Some(pcm.into_iter().map(|v| v.clamp(-1.0, 1.0)).collect())
}

struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 8) as f32 / 8_388_608.0 - 1.0
    }
}

fn synthetic() -> Vec<Item> {
    let len = SECONDS * RATE;
    let tau = std::f32::consts::TAU;
    let both = |mono: Vec<f32>| -> Vec<f32> { mono.iter().flat_map(|&v| [v, v]).collect() };
    let mut items = Vec::new();
    let sine: Vec<f32> = (0..len)
        .map(|i| 0.5 * (i as f32 / 48_000.0 * 1000.0 * tau).sin())
        .collect();
    items.push(("sine-1k", Kind::Sparse, both(sine)));
    let mut phase = 0f32;
    let sweep: Vec<f32> = (0..len)
        .map(|i| {
            let f = 20.0 * (1000f32).powf(i as f32 / len as f32);
            phase = (phase + tau * f / 48_000.0) % tau;
            0.4 * phase.sin()
        })
        .collect();
    items.push(("sweep", Kind::Dense, both(sweep)));
    let clicks: Vec<f32> = (0..len)
        .map(|i| {
            if i % 12_000 < 8 {
                0.8 * (1.0 - (i % 12_000) as f32 / 8.0)
            } else {
                0.0
            }
        })
        .collect();
    items.push(("clicks", Kind::Sparse, both(clicks)));
    let mut rng = Lcg(7);
    let white: Vec<f32> = (0..len * 2).map(|_| 0.3 * rng.next()).collect();
    items.push(("white-noise", Kind::Dense, white));
    // Paul Kellet's pink filter.
    let mut b = [0f32; 7];
    let pink: Vec<f32> = (0..len)
        .map(|_| {
            let w = rng.next();
            b[0] = 0.998_86 * b[0] + w * 0.055_517_9;
            b[1] = 0.993_32 * b[1] + w * 0.075_075_9;
            b[2] = 0.969 * b[2] + w * 0.153_852;
            b[3] = 0.8665 * b[3] + w * 0.310_485_6;
            b[4] = 0.55 * b[4] + w * 0.532_952_4;
            b[5] = -0.7616 * b[5] - w * 0.016_898;
            let out = b.iter().sum::<f32>() + w * 0.5362;
            b[6] = w * 0.115_926;
            0.06 * out
        })
        .collect();
    items.push(("pink-noise", Kind::Dense, both(pink)));
    items.push(("silence", Kind::Sparse, vec![0.0; len * 2]));
    let music: Vec<f32> = (0..len)
        .map(|i| {
            let t = i as f32 / 48_000.0;
            [
                (220.0f32, 0.3f32),
                (330.0, 0.2),
                (495.0, 0.1),
                (880.0, 0.05),
            ]
            .iter()
            .map(|(f, a)| a * (t * f * tau).sin())
            .sum::<f32>()
                * (0.6 + 0.4 * (t * 2.0 * tau).sin())
        })
        .collect();
    let anti: Vec<f32> = music.iter().flat_map(|&v| [v, -v]).collect();
    items.push(("anti-phase", Kind::Dense, anti));
    items
        .into_iter()
        .map(|(name, kind, stereo)| Item {
            name: name.to_owned(),
            kind,
            stereo,
        })
        .collect()
}

fn mono(stereo: &[f32]) -> Vec<f32> {
    stereo
        .chunks_exact(2)
        .map(|p| 0.5 * (p[0] + p[1]))
        .collect()
}

fn layout(channels: usize) -> ChannelLayout {
    if channels == 1 {
        ChannelLayout::mono()
    } else {
        ChannelLayout::stereo()
    }
}

/// Our encoder into an Ogg Opus file; returns the packet sizes and the
/// time spent in the encoder alone.
fn encode_ours(
    pcm: &[f32],
    channels: usize,
    bitrate: u32,
    cbr: bool,
    out: &Path,
) -> (Vec<usize>, Duration) {
    let mut params = CodecParams::audio(CodecId::Opus, 48_000, layout(channels));
    params.bitrate = bitrate;
    params.constant_rate = cbr;
    let started = Instant::now();
    let mut enc = OpusEncoder::open(&params).expect("open");
    let mut packets = Vec::new();
    for chunk in pcm.chunks(4800 * channels) {
        let pcm = Pcm::from_f32(48_000, layout(channels), chunk.to_vec()).expect("pcm");
        packets.extend(enc.push(&AudioInput { pcm }).expect("push"));
    }
    packets.extend(enc.drain().expect("drain"));
    let spent = started.elapsed();
    let head = packets[0]
        .side
        .iter()
        .find_map(|s| match s {
            cutlass_encode::SideData::Config(c) => Some(c.clone()),
            _ => None,
        })
        .expect("head");
    let mut mux = OggMux::new(File::create(out).expect("create"));
    let track = mux
        .add_stream(NewStream {
            params: StreamParams::try_audio(
                StreamCodec::Known(CodecId::Opus),
                48_000,
                channels as u16,
                head,
            )
            .expect("params"),
            timescale: Timescale::new(48_000),
            language: None,
        })
        .expect("stream");
    let mut ticks = 0;
    for p in &packets {
        let stamp = Timestamp::try_new(Some(p.pts), Some(p.pts), Some(p.duration)).expect("time");
        let packet = Packet::try_new(
            track.get(),
            p.data.clone(),
            stamp,
            Timescale::new(48_000),
            true,
        )
        .expect("packet");
        mux.write_packet(&packet).expect("write");
        mux.interleave().expect("interleave");
        ticks += p.duration;
    }
    let finished = mux
        .finish(&[TrackDuration {
            track,
            timescale: Timescale::new(48_000),
            ticks,
        }])
        .expect("finish");
    assert!(finished.is_complete());
    (packets.iter().map(|p| p.data.len()).collect(), spent)
}

fn encode_libopus(raw: &Path, channels: usize, bitrate: u32, out: &Path) {
    let status = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "f32le", "-ar", "48000", "-ac"])
        .arg(channels.to_string())
        .arg("-i")
        .arg(raw)
        .args(["-c:a", "libopus", "-b:a"])
        .arg(bitrate.to_string())
        .args(["-vbr", "on", "-application", "audio"])
        .arg(out)
        .status()
        .expect("ffmpeg");
    assert!(status.success(), "libopus encode");
}

/// Decodes with ffmpeg's `decoder` (`opus` or `libopus`) to floats.
fn decode(file: &Path, decoder: &str, channels: usize) -> Vec<f32> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-c:a", decoder, "-i"])
        .arg(file)
        .args([
            "-ac",
            &channels.to_string(),
            "-ar",
            "48000",
            "-f",
            "f32le",
            "-",
        ])
        .output()
        .expect("ffmpeg");
    assert!(
        out.status.success(),
        "{decoder} decode of {}: {}",
        file.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    floats(&out.stdout)
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

const EBANDS: [usize; 22] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100,
];

/// Band powers of one 20 ms frame of one channel (Hann window, 2048 FFT).
fn band_powers(x: &[f64]) -> [f64; 21] {
    let mut re = vec![0f64; 2048];
    let mut im = vec![0f64; 2048];
    for (i, v) in x.iter().enumerate() {
        let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * (i as f64 + 0.5) / x.len() as f64).cos();
        re[i] = v * w;
    }
    fft(&mut re, &mut im);
    let mut out = [0f64; 21];
    for (b, slot) in out.iter_mut().enumerate() {
        let lo = (EBANDS[b] as f64 * 200.0 / (48_000.0 / 2048.0)).round() as usize;
        let hi =
            ((EBANDS[b + 1] as f64 * 200.0 / (48_000.0 / 2048.0)).round() as usize).max(lo + 1);
        *slot = (lo..hi).map(|k| re[k] * re[k] + im[k] * im[k]).sum::<f64>();
    }
    out
}

struct Score {
    lsd_mean: f64,
    lsd_p95: f64,
    seg_snr: f64,
}

/// Per-band log-spectral distance and segmental SNR over the frames where
/// the reference is above -70 dBFS.
fn score(reference: &[f32], test: &[f32], channels: usize) -> Score {
    let frame = 960;
    let frames = reference.len() / channels / frame;
    let mut lsd = Vec::new();
    let mut snr = Vec::new();
    for f in 0..frames {
        for ch in 0..channels {
            let at = |s: &[f32], i: usize| {
                f64::from(
                    s.get((f * frame + i) * channels + ch)
                        .copied()
                        .unwrap_or(0.0),
                )
            };
            let r: Vec<f64> = (0..frame).map(|i| at(reference, i)).collect();
            let t: Vec<f64> = (0..frame).map(|i| at(test, i)).collect();
            let energy: f64 = r.iter().map(|v| v * v).sum::<f64>() / frame as f64;
            if energy < 1e-7 {
                continue;
            }
            let pr = band_powers(&r);
            let pt = band_powers(&t);
            let floor = 1e-9;
            let d2: f64 = pr
                .iter()
                .zip(&pt)
                .map(|(a, b)| (10.0 * ((a + floor) / (b + floor)).log10()).powi(2))
                .sum::<f64>()
                / 21.0;
            lsd.push(d2.sqrt());
            let err: f64 = r.iter().zip(&t).map(|(a, b)| (a - b) * (a - b)).sum();
            let s: f64 = r.iter().map(|v| v * v).sum();
            snr.push((10.0 * (s / err.max(1e-20)).log10()).clamp(-10.0, 35.0));
        }
    }
    if lsd.is_empty() {
        return Score {
            lsd_mean: 0.0,
            lsd_p95: 0.0,
            seg_snr: 0.0,
        };
    }
    let mean = lsd.iter().sum::<f64>() / lsd.len() as f64;
    let mut sorted = lsd.clone();
    sorted.sort_by(f64::total_cmp);
    let p95 = sorted[(sorted.len() * 95 / 100).min(sorted.len() - 1)];
    Score {
        lsd_mean: mean,
        lsd_p95: p95,
        seg_snr: snr.iter().sum::<f64>() / snr.len() as f64,
    }
}

#[test]
fn encoder_meets_the_interop_rate_and_quality_gates() {
    if !tool_works("ffmpeg", &["-version"]) || !has_libopus() {
        eprintln!("skipped: ffmpeg with libopus is not on PATH");
        return;
    }
    let mut items = synthetic();
    match local_root() {
        Some(root) => {
            for (file, start, label) in RECORDINGS {
                match extract(&root.join(file), start) {
                    Some(stereo) => items.push(Item {
                        name: label.to_owned(),
                        kind: Kind::Dense,
                        stereo,
                    }),
                    None => eprintln!("no audio from {file}; skipped"),
                }
            }
        }
        None => eprintln!("local-files missing: synthetic items only"),
    }
    let dir = scratch();
    let mut failures = Vec::new();
    let input_len = SECONDS * RATE;

    eprintln!("rate (kb/s): item, then ours for each of {RATE_CASES:?}");
    for item in &items {
        let mut row = format!("{:<14}", item.name);
        for (channels, bitrate) in RATE_CASES {
            let pcm = if channels == 1 {
                mono(&item.stereo)
            } else {
                item.stereo.clone()
            };
            let file = dir.join(format!("{}-{channels}-{bitrate}.opus", item.name));
            let (sizes, _) = encode_ours(&pcm, channels, bitrate, false, &file);
            let kbps = sizes.iter().sum::<usize>() as f64 * 8.0 / SECONDS as f64 / 1000.0;
            row.push_str(&format!(" {kbps:7.1}"));
            let ratio = kbps * 1000.0 / f64::from(bitrate);
            // The mono downmix of an anti-phase item is digital silence,
            // which the gate counts as sparse.
            let silent = pcm.iter().all(|&v| v == 0.0);
            if item.kind == Kind::Dense && !silent && (ratio - 1.0).abs() > VBR_TOLERANCE {
                failures.push(format!(
                    "rate {} {channels}ch {bitrate}: {kbps:.1}",
                    item.name
                ));
            }
            // Interop: both ffmpeg decoders give back exactly the input.
            for decoder in ["opus", "libopus"] {
                let got = decode(&file, decoder, channels).len() / channels;
                if got != input_len {
                    failures.push(format!(
                        "interop {} {channels}ch {bitrate} {decoder}: {got} samples",
                        item.name
                    ));
                }
            }
            let cbr = dir.join(format!("{}-{channels}-{bitrate}-cbr.opus", item.name));
            let (sizes, _) = encode_ours(&pcm, channels, bitrate, true, &cbr);
            let want = (bitrate / 400) as usize;
            if sizes.iter().any(|&s| s != want) {
                failures.push(format!("cbr {} {channels}ch {bitrate}", item.name));
            }
            if decode(&cbr, "libopus", channels).len() / channels != input_len {
                failures.push(format!("interop cbr {} {channels}ch {bitrate}", item.name));
            }
        }
        eprintln!("{row}");
    }

    eprintln!(
        "quality, stereo VBR: item rate | ours LSD mean p95 segSNR | libopus LSD mean p95 segSNR | ours kb/s"
    );
    let mut spent = [Duration::ZERO; QUALITY_RATES.len()];
    for item in &items {
        let raw = dir.join(format!("{}.raw", item.name));
        write_raw(&raw, &item.stereo);
        for (slot, bitrate) in QUALITY_RATES.into_iter().enumerate() {
            let ours = dir.join(format!("{}-q-{bitrate}.opus", item.name));
            let (sizes, time) = encode_ours(&item.stereo, 2, bitrate, false, &ours);
            spent[slot] += time;
            let theirs = dir.join(format!("{}-lib-{bitrate}.opus", item.name));
            encode_libopus(&raw, 2, bitrate, &theirs);
            let a = decode(&ours, "libopus", 2);
            let b = decode(&theirs, "libopus", 2);
            let sa = score(&item.stereo, &a, 2);
            let sb = score(&item.stereo, &b, 2);
            let kbps = sizes.iter().sum::<usize>() as f64 * 8.0 / SECONDS as f64 / 1000.0;
            eprintln!(
                "{:<14} {:>3}k | {:5.2} {:5.2} {:6.2} | {:5.2} {:5.2} {:6.2} | {kbps:6.1}",
                item.name,
                bitrate / 1000,
                sa.lsd_mean,
                sa.lsd_p95,
                sa.seg_snr,
                sb.lsd_mean,
                sb.lsd_p95,
                sb.seg_snr
            );
            if sa.lsd_mean > sb.lsd_mean + LSD_MARGIN_DB {
                failures.push(format!(
                    "quality {} {bitrate}: LSD {:.2} vs libopus {:.2}",
                    item.name, sa.lsd_mean, sb.lsd_mean
                ));
            }
        }
    }
    // Speed, reported only: audio seconds coded per second of encoder time.
    let audio = (items.len() * SECONDS) as f64;
    for (bitrate, time) in QUALITY_RATES.into_iter().zip(spent) {
        let rtf = audio / time.as_secs_f64().max(1e-9);
        eprintln!(
            "speed, stereo VBR {:>3}k: {rtf:.0}x real time over {audio:.0} s",
            bitrate / 1000
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(failures.is_empty(), "{failures:#?}");
}
