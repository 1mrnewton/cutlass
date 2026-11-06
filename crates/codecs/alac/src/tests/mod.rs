//! Tests that need more than one module: independent oracles, real files,
//! malformed input, lifecycle, and campaigns.
//!
//! The oracles are `ffmpeg` (libavcodec's ALAC) and, on macOS, `afconvert`
//! (AudioToolbox's ALAC). Both are found at run time. A missing tool skips
//! its test with a message, unless `ALAC_REQUIRE_ORACLES` is set, which
//! turns the skip into a failure. `../local-files` is the real media; a
//! missing folder skips too.

mod campaign;
mod decode_oracle;
mod encode_oracle;
mod end_to_end;
mod hostile;
mod lifecycle;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

use cutlass_codec::access::{ChannelLayout, Packet, SideData};
use cutlass_container::{Demux, Mux, NewStream, TrackDuration};

use cutlass_encode::Encoder;

use crate::{AlacDecoder, AlacEncoder, Cookie, EncoderConfig, default_layout};

static SCRATCH: AtomicU32 = AtomicU32::new(0);

pub(crate) fn ffmpeg() -> Option<PathBuf> {
    [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "ffmpeg",
    ]
    .into_iter()
    .map(PathBuf::from)
    .find(|path| {
        Command::new(path)
            .arg("-version")
            .output()
            .is_ok_and(|output| output.status.success())
    })
}

pub(crate) fn afconvert() -> Option<PathBuf> {
    let path = PathBuf::from("/usr/bin/afconvert");
    let info = PathBuf::from("/usr/bin/afinfo");
    (path.exists() && info.exists()).then_some(path)
}

/// `false` after printing why, or a panic when oracles are required.
pub(crate) fn skip(what: &str) -> bool {
    assert!(
        std::env::var_os("ALAC_REQUIRE_ORACLES").is_none(),
        "{what} is required"
    );
    eprintln!("skipped: {what} is not available");
    false
}

pub(crate) fn local_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files")
}

pub(crate) fn local(name: &str) -> Option<PathBuf> {
    let path = local_root().join(name);
    path.exists().then_some(path)
}

/// A fresh folder under the system temp folder.
pub(crate) fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "cutlass-alac-{name}-{}-{}",
        std::process::id(),
        SCRATCH.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch folder");
    dir
}

pub(crate) fn run(program: &Path, args: &[&str]) {
    let output = Command::new(program).args(args).output().expect("spawn");
    assert!(
        output.status.success(),
        "{} {:?}: {}",
        program.display(),
        args,
        String::from_utf8_lossy(&output.stderr)
    );
}

pub(crate) fn path_str(path: &Path) -> &str {
    path.to_str().expect("utf-8 path")
}

/// The ALAC track of an MP4: its cookie, rate, channel count, and packets.
pub(crate) struct Track {
    pub cookie: Vec<u8>,
    pub rate: u32,
    pub channels: u16,
    pub packets: Vec<Vec<u8>>,
    pub durations: Vec<i64>,
}

pub(crate) fn demux_mp4(path: &Path) -> Track {
    let file = std::fs::File::open(path).expect("open");
    let mut demux = cutlass_isobmff::IsoBmffFile::open(file).expect("mp4");
    let stream = demux
        .streams()
        .iter()
        .find(|stream| {
            stream.params().codec()
                == &cutlass_codec::StreamCodec::Known(cutlass_codec::CodecId::Alac)
        })
        .expect("alac track")
        .clone();
    let mut track = Track {
        cookie: stream.params().extradata().to_vec(),
        rate: stream.params().sample_rate().expect("rate"),
        channels: stream.params().channels().expect("channels"),
        packets: Vec::new(),
        durations: Vec::new(),
    };
    while let Some(packet) = demux.read_packet().expect("packet") {
        if packet.stream() == stream.id().get() {
            track
                .durations
                .push(packet.timestamp().duration().expect("duration"));
            track.packets.push(packet.into_bytes());
        }
    }
    track
}

/// Every packet decoded to integers, interleaved in frame order.
pub(crate) fn decode_all(cookie: &[u8], packets: &[Vec<u8>]) -> (Vec<i32>, Vec<usize>) {
    let mut decoder = AlacDecoder::new(cookie).expect("cookie");
    let mut samples = Vec::new();
    let mut counts = Vec::new();
    for packet in packets {
        let frame = decoder.decode_frame(packet).expect("frame");
        counts.push(frame.frames);
        samples.extend(frame.samples);
    }
    (samples, counts)
}

/// ffmpeg's decode as signed 32-bit, shifted down to `depth`, in ffmpeg's
/// channel order.
pub(crate) fn ffmpeg_samples(ffmpeg: &Path, input: &Path, depth: u32) -> Vec<i32> {
    let output = Command::new(ffmpeg)
        .args(["-v", "error", "-i"])
        .arg(input)
        .args(["-map", "0:a:0", "-f", "s32le", "-c:a", "pcm_s32le", "-"])
        .output()
        .expect("ffmpeg");
    assert!(
        output.status.success(),
        "ffmpeg decode {}: {}",
        input.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    output
        .stdout
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b) >> (32 - depth))
        .collect()
}

/// AudioToolbox's decode, through a big-endian 32-bit CAF that keeps the
/// ALAC channel order, shifted down to `depth`.
pub(crate) fn afconvert_samples(afconvert: &Path, input: &Path, depth: u32) -> Vec<i32> {
    let dir = scratch("afdecode");
    let out = dir.join("out.caf");
    run(
        afconvert,
        &["-f", "caff", "-d", "BEI32", path_str(input), path_str(&out)],
    );
    let bytes = std::fs::read(&out).expect("caf");
    let _ = std::fs::remove_dir_all(&dir);
    caf_data(&bytes)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_be_bytes(*b) >> (32 - depth))
        .collect()
}

/// The audio bytes of a CAF `data` chunk, after its edit count.
pub(crate) fn caf_data(bytes: &[u8]) -> &[u8] {
    assert_eq!(&bytes[..4], b"caff");
    let mut at = 8;
    while at + 12 <= bytes.len() {
        let kind = &bytes[at..at + 4];
        let size = i64::from_be_bytes(bytes[at + 4..at + 12].try_into().expect("size"));
        let body = at + 12;
        if kind == b"data" {
            let end = if size < 0 {
                bytes.len()
            } else {
                body + usize::try_from(size).expect("size")
            };
            return &bytes[body + 4..end];
        }
        at = body + usize::try_from(size).expect("chunk size");
    }
    panic!("caf has no data chunk");
}

/// Interleaved integer PCM from a WAV ffmpeg wrote.
pub(crate) struct Wav {
    pub channels: u16,
    pub rate: u32,
    pub bits: u16,
    pub samples: Vec<i32>,
}

/// The samples of a 16, 24, or 32-bit integer WAV, plain or extensible.
pub(crate) fn wav(bytes: &[u8]) -> Wav {
    assert_eq!(&bytes[..4], b"RIFF");
    let mut at = 12;
    let mut out = Wav {
        channels: 0,
        rate: 0,
        bits: 0,
        samples: Vec::new(),
    };
    while at + 8 <= bytes.len() {
        let kind = &bytes[at..at + 8][..4];
        let size = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().expect("size")) as usize;
        let body = &bytes[at + 8..(at + 8 + size).min(bytes.len())];
        if kind == b"fmt " {
            out.channels = u16::from_le_bytes([body[2], body[3]]);
            out.rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
            out.bits = u16::from_le_bytes([body[14], body[15]]);
        }
        if kind == b"data" {
            let width = usize::from(out.bits / 8);
            out.samples = body
                .chunks_exact(width)
                .map(|b| match width {
                    2 => i32::from(i16::from_le_bytes([b[0], b[1]])),
                    3 => i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8,
                    _ => i32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                })
                .collect();
            return out;
        }
        at += 8 + size + (size & 1);
    }
    panic!("wav has no data chunk");
}

/// Everything `samples` encodes to, pushed in uneven pieces, and the final
/// cookie.
pub(crate) fn encode_all(
    config: EncoderConfig,
    layout: &ChannelLayout,
    samples: &[i32],
) -> (Vec<u8>, Vec<Packet>) {
    let mut encoder = AlacEncoder::new(config, layout).expect("encoder");
    let channels = usize::from(config.channels);
    let mut packets = Vec::new();
    let mut at = 0;
    let mut piece = 777;
    while at < samples.len() {
        let end = (at + piece * channels).min(samples.len());
        packets.extend(encoder.push_samples(&samples[at..end]).expect("push"));
        at = end;
        piece = piece * 3 % 5003 + 1;
    }
    packets.extend(encoder.drain().expect("drain"));
    let cookie = packets
        .iter()
        .flat_map(|packet| &packet.side)
        .rev()
        .find_map(|side| match side {
            SideData::Config(cookie) => Some(cookie.clone()),
            _ => None,
        })
        .expect("cookie");
    (cookie, packets)
}

/// An MP4 of one ALAC track, written by this workspace's muxer.
pub(crate) fn write_m4a(path: &Path, cookie: &[u8], packets: &[Packet]) {
    let parsed = Cookie::parse(cookie).expect("cookie");
    let rate = parsed.config.sample_rate;
    let file = std::fs::File::create(path).expect("create");
    let mut mux = cutlass_isobmff::Mp4Mux::new(std::io::BufWriter::new(file));
    let params = cutlass_codec::StreamParams::try_audio(
        cutlass_codec::StreamCodec::Known(cutlass_codec::CodecId::Alac),
        rate,
        u16::from(parsed.config.channels),
        Vec::new(),
    )
    .expect("params");
    let track = mux
        .add_stream(NewStream {
            params,
            timescale: cutlass_codec::Timescale::new(rate),
            language: None,
        })
        .expect("track");
    let mut ticks = 0;
    for packet in packets {
        for side in &packet.side {
            if let SideData::Config(config) = side {
                mux.set_config(track, config).expect("config");
            }
        }
        let time = cutlass_codec::Timestamp::try_new(
            Some(packet.pts),
            Some(packet.pts),
            Some(packet.duration),
        )
        .expect("time");
        let file_packet = cutlass_codec::Packet::try_new(
            track.get(),
            packet.data.clone(),
            time,
            cutlass_codec::Timescale::new(packet.timescale),
            true,
        )
        .expect("packet");
        mux.write_packet(&file_packet).expect("write");
        ticks += packet.duration;
    }
    mux.finish(&[TrackDuration {
        track,
        timescale: cutlass_codec::Timescale::new(rate),
        ticks,
    }])
    .expect("finish");
}

/// The labels of ffmpeg's layout for `channels`, in ffmpeg's order: the
/// order a WAV from ffmpeg holds.
pub(crate) fn ffmpeg_layout(channels: u8) -> ChannelLayout {
    let alac = default_layout(channels);
    let order = decode_oracle::FFMPEG_ORDER[usize::from(channels) - 1];
    let mut labels = alac.channels().to_vec();
    for (frame, &theirs) in order.iter().enumerate() {
        labels[theirs] = alac.channels()[frame];
    }
    ChannelLayout::new(labels).expect("layout")
}

/// `samples` in ffmpeg's order, moved to ALAC frame order.
pub(crate) fn to_frame_order(samples: &[i32], channels: usize) -> Vec<i32> {
    let order = decode_oracle::FFMPEG_ORDER[channels - 1];
    samples
        .chunks_exact(channels)
        .flat_map(|frame| order.iter().map(|&at| frame[at]))
        .collect()
}

/// For each of our channels, the oracle channel that holds the same samples.
pub(crate) fn channel_map(ours: &[i32], theirs: &[i32], channels: usize) -> Vec<usize> {
    let frames = ours.len() / channels;
    let column = |data: &[i32], c: usize| -> Vec<i32> {
        (0..frames).map(|i| data[i * channels + c]).collect()
    };
    // A channel the encoder's own matrix duplicated matches twice; each
    // oracle channel is taken once, in order.
    let mut taken = vec![false; channels];
    (0..channels)
        .map(|c| {
            let mine = column(ours, c);
            let found = (0..channels)
                .find(|&t| !taken[t] && column(theirs, t) == mine)
                .unwrap_or_else(|| panic!("channel {c} matches no oracle channel"));
            taken[found] = true;
            found
        })
        .collect()
}

/// A deterministic signal: a tone per channel, a slow sweep, and noise,
/// at `depth` bits. Interleaved, `frames` samples per channel.
pub(crate) fn synth(channels: usize, frames: usize, depth: u32, seed: u32) -> Vec<i32> {
    let mut state = seed | 1;
    let peak = f64::from((1u32 << (depth - 1)) - 1);
    let mut out = Vec::with_capacity(channels * frames);
    for i in 0..frames {
        let t = i as f64 / 44_100.0;
        for c in 0..channels {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let noise = f64::from(state) / f64::from(u32::MAX) - 0.5;
            let tone = (std::f64::consts::TAU * (220.0 + 110.0 * c as f64) * t).sin();
            let sweep = (std::f64::consts::TAU * (50.0 + 2000.0 * t) * t).sin();
            let value = peak * (0.5 * tone + 0.3 * sweep + 0.05 * noise);
            out.push(value.round() as i32);
        }
    }
    out
}

/// Interleaved stereo or mono parameters for the trait openers.
pub(crate) fn params(cookie: &[u8], layout: ChannelLayout) -> cutlass_codec::access::CodecParams {
    let parsed = Cookie::parse(cookie).expect("cookie");
    let mut params = cutlass_codec::access::CodecParams::audio(
        cutlass_codec::CodecId::Alac,
        parsed.config.sample_rate,
        layout,
    );
    params.extra = cookie.to_vec();
    params
}
