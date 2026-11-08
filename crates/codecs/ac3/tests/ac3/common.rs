//! Shared helpers: fixtures, framing, decode, tone analysis, and the
//! reference decoders.

use std::path::{Path, PathBuf};
use std::process::Command;

use cutlass_ac3::{Ac3Decoder, Options, StreamType, Syntax, probe};
use cutlass_decode::{Channel, ChannelLayout, CodecId, CodecParams, Decoder, Output, Packet};

pub fn fixture(name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

pub const FIXTURES: [&str; 11] = [
    "ac3_mono_48000.ac3",
    "ac3_stereo_44100.ac3",
    "ac3_3.0_32000.ac3",
    "ac3_3.0-back_48000.ac3",
    "ac3_quad_48000.ac3",
    "ac3_5.1_48000_xbsi.ac3",
    "ac3_2.1_44100.ac3",
    "eac3_stereo_48000.eac3",
    "eac3_5.1_44100.eac3",
    "eac3_mono_32000.eac3",
    "eac3_5.0_48000.eac3",
];

/// Split an elementary stream into access units: every independent
/// substream 0 frame opens one, and the frames after it join it. Bytes
/// before the first syncword and a cut-off last frame are dropped.
pub fn units(data: &[u8]) -> Vec<Vec<u8>> {
    let mut units = Vec::new();
    let mut start = None;
    let mut at = 0;
    while at + 6 <= data.len() {
        let Ok(info) = probe(&data[at..]) else {
            at += 1;
            continue;
        };
        if at + info.frame_bytes > data.len() {
            break;
        }
        let opens = info.stream_type != StreamType::Dependent && info.substream_id == 0;
        match start {
            Some(from) if opens => {
                units.push(data[from..at].to_vec());
                start = Some(at);
            }
            None => start = Some(at),
            _ => {}
        }
        at += info.frame_bytes;
    }
    if let Some(from) = start {
        units.push(data[from..at].to_vec());
    }
    units
}

/// E-AC-3 when any frame is, since an E-AC-3 stream may open with an
/// AC-3 core frame.
pub fn codec_of(data: &[u8]) -> CodecId {
    let eac3 = units(data).iter().any(|unit| {
        let mut at = 0;
        while let Ok(info) = probe(&unit[at..]) {
            if info.syntax == Syntax::Eac3 {
                return true;
            }
            at += info.frame_bytes;
        }
        false
    });
    if eac3 { CodecId::Eac3 } else { CodecId::Ac3 }
}

pub fn open(codec: CodecId, options: Options) -> Ac3Decoder {
    let params = CodecParams::audio(codec, 0, ChannelLayout::stereo());
    Ac3Decoder::with_options(&params, options).expect("open")
}

pub struct Decoded {
    pub rate: u32,
    pub channels: Vec<Channel>,
    pub pcm: Vec<f32>,
}

impl Decoded {
    pub fn frames(&self) -> usize {
        self.pcm.len() / self.channels.len().max(1)
    }

    pub fn channel(&self, index: usize) -> Vec<f32> {
        let n = self.channels.len();
        self.pcm.iter().skip(index).step_by(n).copied().collect()
    }
}

/// Every sample of every output.
pub fn samples(outputs: Vec<Output>) -> Vec<f32> {
    let mut pcm = Vec::new();
    for output in outputs {
        if let Output::Audio(audio) = output {
            pcm.extend_from_slice(audio.pcm.as_f32().expect("f32"));
        }
    }
    pcm
}

/// Samples outside `skip_leading` and `skip_trailing`.
pub fn audible(outputs: Vec<Output>) -> Vec<f32> {
    let mut pcm = Vec::new();
    for output in outputs {
        if let Output::Audio(audio) = output {
            let channels = audio.pcm.layout.len();
            let all = audio.pcm.as_f32().expect("f32");
            let from = audio.skip_leading as usize * channels;
            let to = all.len() - audio.skip_trailing as usize * channels;
            if from < to {
                pcm.extend_from_slice(&all[from..to]);
            }
        }
    }
    pcm
}

pub fn packet(unit: &[u8], index: usize, rate: u32) -> Packet {
    Packet::audio(unit.to_vec(), (index * 1536) as i64, 1536, rate)
}

pub fn decode_with(data: &[u8], options: Options) -> Decoded {
    let mut decoder = open(codec_of(data), options);
    let mut pcm = Vec::new();
    for (index, unit) in units(data).iter().enumerate() {
        pcm.extend(samples(
            decoder.push(&packet(unit, index, 0)).expect("push"),
        ));
    }
    pcm.extend(samples(decoder.drain().expect("drain")));
    let (rate, layout) = decoder.stream().expect("stream");
    Decoded {
        rate,
        channels: layout.channels().to_vec(),
        pcm,
    }
}

pub fn decode(data: &[u8]) -> Decoded {
    decode_with(data, Options::default())
}

/// The fixtures' tone for each speaker.
pub fn tone(channel: Channel) -> f64 {
    match channel {
        Channel::FrontLeft => 300.0,
        Channel::FrontRight => 500.0,
        Channel::FrontCenter => 700.0,
        Channel::Lfe => 60.0,
        Channel::SideLeft | Channel::BackLeft => 900.0,
        Channel::SideRight | Channel::BackRight => 1100.0,
        Channel::BackCenter => 1300.0,
        _ => 0.0,
    }
}

/// Power of `signal` at `freq`, by a Goertzel filter.
pub fn power(signal: &[f32], rate: u32, freq: f64) -> f64 {
    let w = 2.0 * std::f64::consts::PI * freq / f64::from(rate);
    let coeff = 2.0 * w.cos();
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    for &x in signal {
        let s = f64::from(x) + coeff * s1 - s2;
        s2 = s1;
        s1 = s;
    }
    (s1 * s1 + s2 * s2 - coeff * s1 * s2) / (signal.len().max(1) as f64).powi(2)
}

/// Signal to noise ratio of `ours` against `reference`, in dB, over the
/// samples both hold.
pub fn snr(ours: &[f32], reference: &[f32]) -> f64 {
    let (mut signal, mut noise) = (0.0f64, 0.0f64);
    for (&a, &b) in ours.iter().zip(reference) {
        signal += f64::from(b) * f64::from(b);
        noise += (f64::from(a) - f64::from(b)).powi(2);
    }
    if noise == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (signal / noise).log10()
}

pub fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cutlass-ac3-tests-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

fn require_oracles() -> bool {
    std::env::var_os("AC3_REQUIRE_ORACLES").is_some_and(|v| v == "1")
}

/// A tool on `PATH`, or `None` with a line on stderr (a failure under
/// `AC3_REQUIRE_ORACLES=1`).
pub fn oracle(tool: &str, probe_arg: &str) -> Option<String> {
    let found = Command::new(tool)
        .arg(probe_arg)
        .output()
        .is_ok_and(|out| out.status.success() || !out.stdout.is_empty() || !out.stderr.is_empty());
    if found {
        return Some(tool.to_string());
    }
    assert!(!require_oracles(), "{tool} is required and missing");
    eprintln!("skipped: {tool} is not installed");
    None
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("reference output");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// FFmpeg's interleaved full-range decode, in WAVEFORMATEXTENSIBLE order.
pub fn ffmpeg_decode(ffmpeg: &str, data: &[u8], extra: &[&str]) -> Vec<f32> {
    let dir = temp_dir("ffmpeg");
    let input = dir.join(format!("in-{:x}.ac3", fxhash(data)));
    let output = dir.join(format!("out-{:x}-{}.f32", fxhash(data), extra.join("_")));
    std::fs::write(&input, data).expect("write");
    let status = Command::new(ffmpeg)
        .args(["-v", "error", "-y", "-drc_scale", "0"])
        .args(extra)
        .arg("-i")
        .arg(&input)
        .args(["-f", "f32le", "-c:a", "pcm_f32le"])
        .arg(&output)
        .status()
        .expect("ffmpeg");
    assert!(status.success(), "ffmpeg failed");
    read_f32(&output)
}

/// Apple's decode through `afconvert`: coded channel order, and the first
/// 256 samples dropped.
pub fn apple_decode(afconvert: &str, data: &[u8]) -> (usize, Vec<f32>) {
    let dir = temp_dir("apple");
    // `afconvert` reads an `.ac3` file as AC-3 only and drops dependent
    // substreams.
    let extension = if codec_of(data) == CodecId::Eac3 {
        "ec3"
    } else {
        "ac3"
    };
    let input = dir.join(format!("in-{:x}.{extension}", fxhash(data)));
    let output = dir.join(format!("out-{:x}.wav", fxhash(data)));
    std::fs::write(&input, data).expect("write");
    let status = Command::new(afconvert)
        .args(["-f", "WAVE", "-d", "LEF32"])
        .arg(&input)
        .arg(&output)
        .status()
        .expect("afconvert");
    assert!(status.success(), "afconvert failed");
    let wav = std::fs::read(&output).expect("wav");
    let (mut channels, mut pcm) = (0, Vec::new());
    let mut at = 12;
    while at + 8 <= wav.len() {
        let id = &wav[at..at + 4];
        let len = u32::from_le_bytes([wav[at + 4], wav[at + 5], wav[at + 6], wav[at + 7]]) as usize;
        let body = &wav[at + 8..(at + 8 + len).min(wav.len())];
        if id == b"fmt " {
            channels = usize::from(u16::from_le_bytes([body[2], body[3]]));
        } else if id == b"data" {
            pcm = body
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
        }
        at += 8 + len + (len & 1);
    }
    (channels, pcm)
}

pub fn deinterleave(pcm: &[f32], channels: usize, index: usize) -> Vec<f32> {
    pcm.iter().skip(index).step_by(channels).copied().collect()
}

fn fxhash(data: &[u8]) -> u64 {
    data.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    })
}

/// The FATE sample folder named by `AC3_FATE_SAMPLES`, or `None` with a
/// line on stderr.
pub fn fate_dir() -> Option<PathBuf> {
    match std::env::var_os("AC3_FATE_SAMPLES") {
        Some(dir) if Path::new(&dir).is_dir() => Some(PathBuf::from(dir)),
        _ => {
            eprintln!("skipped: AC3_FATE_SAMPLES does not name the FATE sample folder");
            None
        }
    }
}
