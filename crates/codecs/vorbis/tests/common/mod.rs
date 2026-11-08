//! Test helpers: a minimal Ogg page reader (so the decoder's oracle tests
//! do not depend on `cutlass-ogg`), the external tools, and comparisons.

#![allow(dead_code, clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::path::{Path, PathBuf};
use std::process::Command;

use cutlass_decode::Channel;
use cutlass_vorbis::{Headers, VorbisDecoder};

/// One packet and the granule of the page it ends on, when it is the last
/// packet to end there.
pub struct OggPacket {
    pub data: Vec<u8>,
    pub granule: Option<i64>,
}

/// The packets of the first logical stream of an Ogg file, and of each
/// link chained after it.
pub fn ogg_packets(bytes: &[u8]) -> Vec<OggPacket> {
    let mut packets = Vec::new();
    let mut partial: Vec<u8> = Vec::new();
    let mut at = 0;
    let mut serial = None;
    let mut ended = false;
    while at + 27 <= bytes.len() {
        assert_eq!(&bytes[at..at + 4], b"OggS", "page at {at}");
        let flags = bytes[at + 5];
        let granule = i64::from_le_bytes(bytes[at + 6..at + 14].try_into().unwrap());
        let page_serial = u32::from_le_bytes(bytes[at + 14..at + 18].try_into().unwrap());
        let segments = bytes[at + 26] as usize;
        let lacing = &bytes[at + 27..at + 27 + segments];
        let mut body = at + 27 + segments;
        let size: usize = lacing.iter().map(|&l| l as usize).sum();
        if ended && flags & 2 == 2 {
            serial = Some(page_serial);
            ended = false;
        }
        let mine = *serial.get_or_insert(page_serial) == page_serial;
        if mine && flags & 4 == 4 {
            ended = true;
        }
        let first_new = packets.len();
        if mine {
            for &len in lacing {
                partial.extend_from_slice(&bytes[body..body + len as usize]);
                body += len as usize;
                if len < 255 {
                    packets.push(OggPacket {
                        data: std::mem::take(&mut partial),
                        granule: None,
                    });
                }
            }
            if packets.len() > first_new && granule != -1 {
                packets.last_mut().unwrap().granule = Some(granule);
            }
        }
        at += 27 + segments + size;
    }
    packets
}

/// The configuration and audio packets of an Ogg Vorbis file.
pub fn split_stream(bytes: &[u8]) -> (Vec<u8>, Vec<OggPacket>) {
    let mut packets = ogg_packets(bytes);
    let audio = packets.split_off(3);
    let headers = Headers {
        identification: packets[0].data.clone(),
        comment: packets[1].data.clone(),
        setup: packets[2].data.clone(),
    };
    (headers.join(), audio)
}

/// Decodes every packet of an Ogg Vorbis file and trims to its granules
/// (Vorbis I appendix A.2): the end at the last granule, and a start
/// below zero.
pub fn decode_file(bytes: &[u8]) -> (u32, usize, Vec<f32>) {
    let (rate, layout, samples) = decode_file_with_layout(bytes);
    (rate, layout.len(), samples)
}

/// [`decode_file`], with the speaker of each channel.
pub fn decode_file_with_layout(bytes: &[u8]) -> (u32, Vec<Channel>, Vec<f32>) {
    let (config, audio) = split_stream(bytes);
    let mut decoder = VorbisDecoder::from_config(&config).unwrap();
    let layout: Vec<Channel> = decoder.layout().channels().to_vec();
    let id = decoder.identification();
    let channels = usize::from(id.channels);
    let mut samples = Vec::new();
    let mut granules = Vec::new();
    for packet in &audio {
        let out = decoder.decode(&packet.data).unwrap();
        samples.extend_from_slice(&out);
        if let Some(granule) = packet.granule {
            granules.push((granule, samples.len() / channels));
        }
    }
    let decoded = samples.len() / channels;
    // The first page's granule counts the samples to its end; fewer
    // decoded there is a negative start, cut from the front. On the last
    // page the granule is the end instead (an end trim).
    let lead = match granules.as_slice() {
        [(first, before), _, ..] => (*before as i64 - first).max(0) as usize,
        _ => 0,
    };
    let last = granules
        .last()
        .map_or(decoded as i64, |&(granule, _)| granule);
    let total = (last.max(0) as usize).min(decoded - lead);
    let mut trimmed: Vec<f32> = samples.into_iter().skip(lead * channels).collect();
    trimmed.truncate(total * channels);
    (id.sample_rate, layout, trimmed)
}

/// The speaker order FFmpeg and GStreamer give decoded audio (the WAV
/// channel-mask order) for the layouts the tests use.
pub fn wav_order(channels: usize) -> Vec<Channel> {
    use Channel::{
        BackLeft, BackRight, FrontCenter, FrontLeft, FrontRight, Lfe, SideLeft, SideRight,
    };
    match channels {
        1 => vec![FrontCenter],
        2 => vec![FrontLeft, FrontRight],
        6 => vec![FrontLeft, FrontRight, FrontCenter, Lfe, BackLeft, BackRight],
        8 => vec![
            FrontLeft,
            FrontRight,
            FrontCenter,
            Lfe,
            BackLeft,
            BackRight,
            SideLeft,
            SideRight,
        ],
        _ => panic!("no WAV order for {channels} channels"),
    }
}

/// Interleaved samples in `from` speaker order, put in `to` order.
pub fn reorder(samples: &[f32], from: &[Channel], to: &[Channel]) -> Vec<f32> {
    let map: Vec<usize> = to
        .iter()
        .map(|speaker| {
            from.iter()
                .position(|s| s == speaker)
                .expect("same speakers")
        })
        .collect();
    let channels = from.len();
    let mut out = Vec::with_capacity(samples.len());
    for frame in samples.chunks_exact(channels) {
        for &index in &map {
            out.push(frame[index]);
        }
    }
    out
}

pub fn has_tool(tool: &str, args: &[&str]) -> bool {
    Command::new(tool)
        .args(args)
        .output()
        .is_ok_and(|o| o.status.success())
}

pub fn has_gst_vorbis() -> bool {
    has_tool("gst-inspect-1.0", &["vorbisenc"]) && has_tool("gst-inspect-1.0", &["vorbisdec"])
}

pub fn has_ffmpeg() -> bool {
    has_tool("ffmpeg", &["-hide_banner", "-version"])
}

pub fn run(command: &mut Command) {
    let output = command.output().expect("tool runs");
    assert!(
        output.status.success(),
        "{command:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

pub fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cutlass-vorbis-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

pub fn local_files() -> Option<PathBuf> {
    let root = std::env::var_os("CUTLASS_LOCAL_FILES")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files")
        });
    root.is_dir().then_some(root)
}

pub fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect()
}

/// FFmpeg's own Vorbis decoder (not libvorbis) on a file, as `f32`.
///
/// Its C path: the aarch64 NEON inverse coupling branches on the
/// magnitude's sign bit, so a +0 magnitude takes the positive branch where
/// the spec's `magnitude > 0` does not, which changes the angle channel of
/// streams with exact zeros (FFmpeg's own encoder writes them).
pub fn ffmpeg_decode(file: &Path, dir: &Path) -> Vec<f32> {
    let out = dir.join("ffmpeg.f32");
    run(Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-cpuflags",
            "0",
            "-c:a",
            "vorbis",
            "-i",
        ])
        .arg(file)
        .args(["-f", "f32le", "-c:a", "pcm_f32le"])
        .arg(&out));
    read_f32(&out)
}

/// libvorbis (GStreamer `vorbisdec`) on a file, every decoded sample, as
/// `f32`.
pub fn libvorbis_decode(file: &Path, dir: &Path) -> Vec<f32> {
    let out = dir.join("libvorbis.f32");
    run(Command::new("gst-launch-1.0")
        .arg("-q")
        .arg("filesrc")
        .arg(format!("location={}", file.display()))
        .args([
            "!",
            "oggdemux",
            "!",
            "vorbisdec",
            "!",
            "audio/x-raw,format=F32LE",
            "!",
            "filesink",
        ])
        .arg(format!("location={}", out.display())));
    read_f32(&out)
}

/// Signal-to-noise ratio of `got` against `want` over their common
/// length, per channel; the worst channel. A silent reference channel is
/// measured against full scale instead.
pub fn snr_db(want: &[f32], got: &[f32], channels: usize) -> f64 {
    let len = want.len().min(got.len()) / channels * channels;
    let mut worst = f64::INFINITY;
    for c in 0..channels {
        let mut signal = 0.0_f64;
        let mut noise = 0.0_f64;
        for i in (c..len).step_by(channels) {
            let w = f64::from(want[i]);
            let d = w - f64::from(got[i]);
            signal += w * w;
            noise += d * d;
        }
        let frames = (len / channels) as f64;
        let reference = if signal < frames * 1e-10 {
            frames
        } else {
            signal
        };
        let snr = if noise == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (reference / noise).log10()
        };
        worst = worst.min(snr);
    }
    worst
}
