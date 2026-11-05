//! Shared helpers: fixtures, demux, decode, bit edits, and reference decoders.

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;

use cutlass_decode::{ChannelLayout, CodecId, CodecParams, Decoder, Output, Packet, SideData};
use cutlass_mp3::Mp3Decoder;
use cutlass_mp3_container::{Demux, Mp3Demux};

/// ISO/IEC 11172-4 full-accuracy bounds on the difference to a reference
/// decoder, for output at full scale 1.0: RMS below `2^-15 / sqrt(12)` and
/// no sample further than `2^-14`.
pub const ISO_RMS: f64 = 8.809_142_651_444_272e-6;
pub const ISO_MAX: f64 = 6.103_515_625e-5;

/// An MP3 file split by `cutlass-mp3-container`.
pub struct File {
    pub rate: u32,
    pub channels: usize,
    pub samples_per_frame: usize,
    pub packets: Vec<Vec<u8>>,
    pub pts: Vec<i64>,
    /// LAME encoder delay and padding.
    pub trim: Option<(u32, u32)>,
    pub reservoir_preroll: usize,
}

impl File {
    pub fn packet(&self, index: usize) -> Packet {
        Packet::audio(
            self.packets[index].clone(),
            self.pts[index],
            self.samples_per_frame as i64,
            self.rate,
        )
    }

    /// The packet with the container's trim attached.
    pub fn packet_gapless(&self, index: usize) -> Packet {
        let mut packet = self.packet(index);
        if let Some((encoder_delay, trailing_padding)) = self.trim {
            packet.side.push(SideData::Gapless {
                encoder_delay,
                trailing_padding,
                sample_rate: self.rate,
            });
        }
        packet
    }
}

pub fn fixture(name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

pub fn demux(bytes: &[u8]) -> File {
    let mut demux = Mp3Demux::open(Cursor::new(bytes.to_vec())).expect("demux");
    let stream = demux.streams()[0].clone();
    let mut packets = Vec::new();
    let mut pts = Vec::new();
    let mut samples_per_frame = 0;
    while let Some(packet) = demux.next_packet().expect("packet") {
        samples_per_frame = packet.duration as usize;
        pts.push(packet.pts);
        packets.push(packet.data);
    }
    File {
        rate: stream.params.sample_rate,
        channels: usize::from(stream.params.channels),
        samples_per_frame,
        packets,
        pts,
        trim: stream
            .trim
            .map(|trim| (trim.encoder_delay, trim.trailing_padding)),
        reservoir_preroll: stream.reservoir_preroll as usize,
    }
}

pub fn open_decoder(rate: u32, channels: usize) -> Mp3Decoder {
    let layout = match channels {
        0 => None,
        1 => Some(ChannelLayout::mono()),
        _ => Some(ChannelLayout::stereo()),
    };
    let mut params = CodecParams::audio(CodecId::Mp3, rate, ChannelLayout::stereo());
    params.layout = layout;
    Mp3Decoder::open(&params).expect("open")
}

/// Every sample of every output, trimmed or not.
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

pub struct Decoded {
    pub pcm: Vec<f32>,
}

/// Decode a whole file. `gapless` attaches the trim and keeps only
/// audible samples; otherwise every decoded sample is returned.
pub fn decode_packets(file: &File, gapless: bool) -> Decoded {
    let mut decoder = open_decoder(file.rate, file.channels);
    let mut pcm = Vec::new();
    for index in 0..file.packets.len() {
        if gapless {
            pcm.extend(audible(
                decoder.push(&file.packet_gapless(index)).expect("push"),
            ));
        } else {
            pcm.extend(samples(decoder.push(&file.packet(index)).expect("push")));
        }
    }
    pcm.extend(audible(decoder.drain().expect("drain")));
    Decoded { pcm }
}

pub fn get_bits(data: &[u8], bit: usize, count: u32) -> u32 {
    let mut value = 0;
    for i in 0..count as usize {
        let at = bit + i;
        value = (value << 1) | u32::from((data[at / 8] >> (7 - at % 8)) & 1);
    }
    value
}

pub fn set_bits(data: &mut [u8], bit: usize, count: u32, value: u32) {
    for i in 0..count as usize {
        let at = bit + i;
        let one = (value >> (count as usize - 1 - i)) & 1 == 1;
        let mask = 1u8 << (7 - at % 8);
        if one {
            data[at / 8] |= mask;
        } else {
            data[at / 8] &= !mask;
        }
    }
}

/// A reference decoder that writes interleaved `f32` on stdout.
pub struct Oracle {
    pub name: &'static str,
    pub path: PathBuf,
}

fn find_tool(name: &'static str, version_arg: &str) -> Option<Oracle> {
    for dir in ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"] {
        let path = Path::new(dir).join(name);
        if path.exists() {
            return Some(Oracle { name, path });
        }
    }
    let path = PathBuf::from(name);
    if Command::new(&path)
        .arg(version_arg)
        .output()
        .is_ok_and(|output| output.status.success())
    {
        return Some(Oracle { name, path });
    }
    if std::env::var_os("MP3_REQUIRE_ORACLES").is_some() {
        panic!("{name} is required");
    }
    eprintln!("skipped the {name} reference: {name} was not found");
    None
}

pub fn oracle_ffmpeg() -> Option<Oracle> {
    find_tool("ffmpeg", "-version")
}

pub fn oracle_mpg123() -> Option<Oracle> {
    find_tool("mpg123", "--version")
}

pub fn temp_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "cutlass-mp3-{label}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

/// Decode `bytes` with the reference, gapless as that decoder does it.
pub fn reference_pcm(oracle: &Oracle, bytes: &[u8]) -> Vec<f32> {
    let dir = temp_dir(oracle.name);
    let input = dir.join("in.mp3");
    std::fs::write(&input, bytes).expect("write");
    let output = match oracle.name {
        "ffmpeg" => Command::new(&oracle.path)
            .args(["-v", "error", "-i"])
            .arg(&input)
            .args(["-f", "f32le", "-"])
            .output(),
        _ => Command::new(&oracle.path)
            .args(["-q", "-e", "f32", "-s"])
            .arg(&input)
            .output(),
    }
    .expect("run reference");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(output.status.success(), "{}: {output:?}", oracle.name);
    output
        .stdout
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

/// RMS and largest absolute difference.
pub fn difference(ours: &[f32], reference: &[f32]) -> (f64, f64) {
    let mut sum = 0f64;
    let mut max = 0f64;
    for (a, b) in ours.iter().zip(reference) {
        let d = f64::from(*a) - f64::from(*b);
        sum += d * d;
        max = max.max(d.abs());
    }
    ((sum / ours.len().max(1) as f64).sqrt(), max)
}

/// Decode `bytes` gapless with this crate and with `oracle`; require the
/// same sample count and the ISO/IEC 11172-4 bounds.
pub fn oracle_compare(oracle: &Oracle, bytes: &[u8], label: &str) {
    let file = demux(bytes);
    let ours = decode_packets(&file, true).pcm;
    let reference = reference_pcm(oracle, bytes);
    assert_eq!(ours.len(), reference.len(), "{label} vs {}", oracle.name);
    let (rms, max) = difference(&ours, &reference);
    assert!(
        rms < ISO_RMS && max <= ISO_MAX,
        "{label} vs {}: rms {rms:e} max {max:e}",
        oracle.name
    );
}

/// The MP3 files in the local media folder no larger than `max_bytes`, or
/// `None` (with a note) when the folder is missing.
pub fn local_mp3s(max_bytes: u64) -> Option<Vec<PathBuf>> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!("skipped local media: {} is missing", dir.display());
        return None;
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "mp3"))
        .filter(|path| std::fs::metadata(path).is_ok_and(|meta| meta.len() <= max_bytes))
        .collect();
    files.sort();
    Some(files)
}
