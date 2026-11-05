//! Signals, WAV files, and the independent tools the oracle tests call.
//!
//! Frame boundaries come from `flac --analyze`, not from this crate or the
//! container crate. Expected samples come from the generated signal, from
//! `flac -d`, and from `ffmpeg`.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use cutlass_flac::StreamInfo;

/// `true` when `name` runs. Missing tools skip a test unless
/// `FLAC_REQUIRE_ORACLES` is set, which turns the skip into a failure.
pub fn have(name: &str) -> bool {
    let flag = if name.starts_with("ff") {
        "-version"
    } else {
        "--version"
    };
    let found = Command::new(name)
        .arg(flag)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success());
    if !found {
        assert!(
            std::env::var_os("FLAC_REQUIRE_ORACLES").is_none(),
            "{name} is required by FLAC_REQUIRE_ORACLES"
        );
        eprintln!("skipped: {name} was not found");
    }
    found
}

pub fn run(program: &str, args: &[&str]) -> Vec<u8> {
    let out = Command::new(program)
        .args(args)
        .output()
        .unwrap_or_else(|error| panic!("{program} did not start: {error}"));
    assert!(
        out.status.success(),
        "{program} {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

/// Runs and reports success without asserting.
pub fn succeeds(program: &str, args: &[&str]) -> (bool, String) {
    let out = Command::new(program)
        .args(args)
        .output()
        .unwrap_or_else(|error| panic!("{program} did not start: {error}"));
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// A fresh scratch folder under the system temp directory.
pub fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cutlass-flac-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch");
    dir
}

pub fn path_str(path: &Path) -> &str {
    path.to_str().expect("utf-8 path")
}

/// A deterministic generator.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `-half..half`.
    pub fn signed(&mut self, bits: u32) -> i64 {
        let raw = self.next() >> (64 - bits);
        raw as i64 - (1i64 << (bits - 1))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Tones and a little noise, a different mix per channel.
    Music,
    /// Uniform noise over the whole range: incompressible.
    Noise,
    /// Channel 0 is silent; the rest are music.
    SilentFirst,
    /// Music with the low `bits / 3` bits zero.
    Wasted,
    /// Left and right nearly equal.
    Correlated,
    /// Alternating extremes and steps.
    Extremes,
}

/// Interleaved samples at `bits` depth.
pub fn signal(kind: Kind, channels: usize, bits: u32, frames: usize, seed: u64) -> Vec<i32> {
    let mut rng = Rng(seed | 1);
    let high = (1i64 << (bits - 1)) - 1;
    let low = -(1i64 << (bits - 1));
    let amplitude = (high as f64) * 0.6;
    let mut out = Vec::with_capacity(frames * channels);
    for i in 0..frames {
        let t = i as f64;
        let common = (t * 0.031).sin() * 0.5 + (t * 0.0071).sin() * 0.3;
        for channel in 0..channels {
            let value = match kind {
                Kind::Noise => rng.signed(bits),
                Kind::Extremes => match (i / 7 + channel) % 4 {
                    0 => high,
                    1 => low,
                    2 => 0,
                    _ => {
                        if i % 2 == 0 {
                            high
                        } else {
                            low
                        }
                    }
                },
                Kind::SilentFirst if channel == 0 => 0,
                Kind::Correlated => {
                    let base = common * amplitude;
                    let wobble = (rng.signed(bits.min(8)) as f64) * 0.5;
                    (base + if channel == 1 { wobble } else { 0.0 }) as i64
                }
                _ => {
                    let phase = channel as f64 * 0.7;
                    let tone = (t * (0.013 + 0.004 * channel as f64) + phase).sin() * 0.6
                        + (t * 0.19 + phase).sin() * 0.1
                        + common * 0.25;
                    let noise = rng.signed(bits.min(12)) as f64 / 64.0;
                    (tone * amplitude + noise) as i64
                }
            };
            let mut value = value.clamp(low, high);
            if kind == Kind::Wasted {
                let zeros = bits / 3;
                value = (value >> zeros) << zeros;
            }
            out.push(i32::try_from(value).expect("fits"));
        }
    }
    out
}

/// Writes WAVE_FORMAT_EXTENSIBLE with left-justified samples.
pub fn write_wav(path: &Path, samples: &[i32], channels: usize, rate: u32, bits: u32) {
    let container = bits.div_ceil(8) * 8;
    let width = (container / 8) as usize;
    let mut data = Vec::with_capacity(samples.len() * width);
    for sample in samples {
        let shifted = (i64::from(*sample) << (container - bits)) as i32;
        if container == 8 {
            // 8-bit WAV is unsigned offset binary.
            data.push((shifted + 128) as u8);
        } else {
            data.extend_from_slice(&shifted.to_le_bytes()[..width]);
        }
    }
    let channels16 = u16::try_from(channels).expect("channels");
    let block = channels16 * u16::try_from(width).expect("width");
    let mut fmt = Vec::new();
    fmt.extend_from_slice(&0xFFFEu16.to_le_bytes());
    fmt.extend_from_slice(&channels16.to_le_bytes());
    fmt.extend_from_slice(&rate.to_le_bytes());
    fmt.extend_from_slice(&(rate * u32::from(block)).to_le_bytes());
    fmt.extend_from_slice(&block.to_le_bytes());
    fmt.extend_from_slice(&u16::try_from(container).expect("container").to_le_bytes());
    fmt.extend_from_slice(&22u16.to_le_bytes());
    fmt.extend_from_slice(&u16::try_from(bits).expect("bits").to_le_bytes());
    fmt.extend_from_slice(&0u32.to_le_bytes());
    fmt.extend_from_slice(&[
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B,
        0x71,
    ]);
    let mut body = b"WAVEfmt ".to_vec();
    body.extend_from_slice(&u32::try_from(fmt.len()).expect("fmt").to_le_bytes());
    body.extend_from_slice(&fmt);
    body.extend_from_slice(b"data");
    body.extend_from_slice(&u32::try_from(data.len()).expect("data").to_le_bytes());
    body.extend_from_slice(&data);
    let mut file = b"RIFF".to_vec();
    file.extend_from_slice(&u32::try_from(body.len()).expect("riff").to_le_bytes());
    file.extend_from_slice(&body);
    std::fs::write(path, file).expect("write wav");
}

/// Reads integer PCM from a WAV file: samples right-justified, plus the
/// valid bit depth and channel count.
pub fn read_wav(path: &Path) -> (Vec<i32>, u32, usize) {
    let bytes = std::fs::read(path).expect("read wav");
    assert_eq!(&bytes[..4], b"RIFF");
    let mut at = 12;
    let mut format = None;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let len = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().expect("len")) as usize;
        let body = &bytes[at + 8..(at + 8 + len).min(bytes.len())];
        if id == b"fmt " {
            let channels = usize::from(u16::from_le_bytes([body[2], body[3]]));
            let container = u32::from(u16::from_le_bytes([body[14], body[15]]));
            let valid = if body.len() >= 20 && u16::from_le_bytes([body[0], body[1]]) == 0xFFFE {
                u32::from(u16::from_le_bytes([body[18], body[19]]))
            } else {
                container
            };
            format = Some((channels, container, valid));
        } else if id == b"data" {
            let (channels, container, valid) = format.expect("fmt before data");
            let width = (container / 8) as usize;
            let mut samples = Vec::with_capacity(body.len() / width);
            for chunk in body.chunks_exact(width) {
                let mut raw = [0u8; 4];
                raw[4 - width..].copy_from_slice(chunk);
                let mut value = i32::from_le_bytes(raw) >> (32 - container);
                if container == 8 {
                    value = i32::from(chunk[0]) - 128;
                }
                samples.push(value >> (container - valid));
            }
            return (samples, valid, channels);
        }
        at += 8 + len + (len & 1);
    }
    panic!("no data chunk in {}", path.display());
}

/// One frame's byte range from `flac --analyze`.
pub struct AnalyzedFrame {
    pub offset: usize,
    pub bits: usize,
}

/// Frame offsets and sizes from `flac --analyze`.
pub fn analyze(flac: &Path) -> Vec<AnalyzedFrame> {
    let ana = flac.with_extension("ana");
    run(
        "flac",
        &["-s", "-f", "-a", path_str(flac), "-o", path_str(&ana)],
    );
    let text = std::fs::read_to_string(&ana).expect("ana");
    let mut frames = Vec::new();
    for line in text.lines() {
        if !line.starts_with("frame=") {
            continue;
        }
        let mut offset = None;
        let mut bits = None;
        for field in line.split('\t') {
            if let Some(value) = field.strip_prefix("offset=") {
                offset = value.parse().ok();
            } else if let Some(value) = field.strip_prefix("bits=") {
                bits = value.parse().ok();
            }
        }
        frames.push(AnalyzedFrame {
            offset: offset.expect("offset"),
            bits: bits.expect("bits"),
        });
    }
    frames
}

/// STREAMINFO and frame byte slices of a native FLAC file, split at the
/// offsets `flac --analyze` reports.
pub fn split(flac: &Path) -> (StreamInfo, Vec<Vec<u8>>) {
    let bytes = std::fs::read(flac).expect("flac");
    assert_eq!(&bytes[..4], b"fLaC");
    assert_eq!(bytes[4] & 0x7F, 0, "streaminfo first");
    let info = StreamInfo::parse(&bytes[8..42]).expect("streaminfo");
    let frames = analyze(flac)
        .iter()
        .map(|frame| bytes[frame.offset..frame.offset + frame.bits / 8].to_vec())
        .collect();
    (info, frames)
}

/// ffmpeg's decode of `path` as signed 32-bit little-endian, interleaved,
/// shifted right back to `bits`.
pub fn ffmpeg_samples(path: &Path, bits: u32) -> Vec<i32> {
    let raw = run(
        "ffmpeg",
        &["-v", "error", "-i", path_str(path), "-f", "s32le", "-"],
    );
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| i32::from_le_bytes(*chunk) >> (32 - bits))
        .collect()
}

/// `flac -d` of `path` to a WAV file, read back.
pub fn flac_samples(path: &Path) -> (Vec<i32>, u32, usize) {
    let wav = path.with_extension("decoded.wav");
    run(
        "flac",
        &["-s", "-f", "-d", path_str(path), "-o", path_str(&wav)],
    );
    read_wav(&wav)
}
