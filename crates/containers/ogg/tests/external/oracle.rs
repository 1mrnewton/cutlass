//! This demuxer against files ffmpeg wrote, read back by ffprobe and ffmpeg.
//!
//! Producers: libopus through ffmpeg, and ffmpeg's native Opus encoder,
//! both muxed by libavformat's Ogg muxer. Oracle: libavformat's Ogg demuxer
//! (ffprobe packets, `-f data` payloads) and ffmpeg's decode for the
//! sample count after pre-skip and end trimming. ffmpeg stamps a packet at
//! granule minus pre-skip; this crate stamps the granule-domain time, so
//! every comparison adds the pre-skip to ffprobe's time.

use std::fs::File;
use std::path::{Path, PathBuf};

use cutlass_codec::{CodecId, Packet, StreamCodec};
use cutlass_container::{Demux, Landing, SeekTo};
use cutlass_ogg::{OggDemux, packet_samples};

use crate::support::{
    environment, ffmpeg, ffmpeg_decoded_samples, ffmpeg_payload, ffprobe_packets, ffprobe_stream,
    gstreamer_decoded_samples, gstreamer_packets, pick, scratch, sha256_hex,
};

pub struct Variant {
    pub name: &'static str,
    source: &'static [&'static str],
    args: &'static [&'static str],
}

pub const VARIANTS: &[Variant] = &[
    Variant {
        name: "libopus-stereo-96k",
        source: &[".m4a"],
        args: &["-t", "20", "-vn", "-c:a", "libopus", "-b:a", "96k"],
    },
    Variant {
        name: "libopus-mono-32k-60ms",
        source: &[".wav"],
        args: &[
            "-t",
            "20",
            "-ac",
            "1",
            "-c:a",
            "libopus",
            "-b:a",
            "32k",
            "-frame_duration",
            "60",
        ],
    },
    Variant {
        name: "libopus-stereo-2.5ms",
        source: &["thunder_distant_10s.mp3", ".mp3"],
        args: &[
            "-t",
            "10",
            "-c:a",
            "libopus",
            "-b:a",
            "128k",
            "-frame_duration",
            "2.5",
        ],
    },
    Variant {
        name: "libopus-120ms-24k",
        source: &[".mp3"],
        args: &[
            "-t",
            "12",
            "-c:a",
            "libopus",
            "-b:a",
            "24k",
            "-frame_duration",
            "120",
        ],
    },
    Variant {
        name: "libopus-voip-10ms",
        source: &[".wav"],
        args: &[
            "-t",
            "8",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "libopus",
            "-application",
            "voip",
            "-b:a",
            "16k",
            "-frame_duration",
            "10",
        ],
    },
    Variant {
        name: "native-opus-stereo-128k",
        source: &[".mp4"],
        args: &[
            "-t", "20", "-vn", "-ar", "48000", "-ac", "2", "-c:a", "opus", "-strict", "-2", "-b:a",
            "128k",
        ],
    },
    Variant {
        name: "native-opus-mono-64k",
        source: &[".m4a"],
        args: &[
            "-t", "15", "-vn", "-ar", "48000", "-ac", "1", "-c:a", "opus", "-strict", "-2", "-b:a",
            "64k",
        ],
    },
    Variant {
        name: "libopus-5.1-family-1",
        source: &[".m4a"],
        args: &[
            "-t",
            "10",
            "-vn",
            "-af",
            "pan=5.1|FL=c0|FR=c1|FC=0.5*c0+0.5*c1|LFE=0.1*c0|BL=c0|BR=c1",
            "-c:a",
            "libopus",
            "-mapping_family",
            "1",
            "-b:a",
            "256k",
        ],
    },
];

pub fn encode(root: &Path, dir: &Path, variant: &Variant) -> Option<PathBuf> {
    let source = variant
        .source
        .iter()
        .find_map(|suffix| pick(root, &[suffix]))?;
    let out = dir.join(format!("{}.opus", variant.name));
    let mut args = vec!["-i", source.to_str()?];
    args.extend_from_slice(variant.args);
    args.push(out.to_str()?);
    ffmpeg(&args);
    Some(out)
}

pub fn open(path: &Path) -> OggDemux<File> {
    let file = File::open(path).expect("open");
    let len = file.metadata().expect("meta").len();
    OggDemux::open(file, len).expect("ogg open")
}

pub fn read_all(demux: &mut OggDemux<File>) -> Vec<Packet> {
    let mut packets = Vec::new();
    while let Some(packet) = demux.read_packet().expect("read") {
        packets.push(packet);
    }
    packets
}

pub fn pts(packet: &Packet) -> i64 {
    packet.timestamp().presentation().expect("pts")
}

pub fn duration(packet: &Packet) -> i64 {
    packet.timestamp().duration().expect("duration")
}

#[test]
fn ffmpeg_files_demux_to_the_same_packets_times_and_trim() {
    let Some(root) = environment("ffmpeg_files_demux_to_the_same_packets_times_and_trim") else {
        return;
    };
    let dir = scratch("oracle");
    let mut checked = 0;
    for variant in VARIANTS {
        let Some(path) = encode(&root, &dir, variant) else {
            eprintln!("skipped {}: no source file", variant.name);
            continue;
        };
        check_against_ffmpeg(&path, variant.name, &dir);
        check_seeks(&path, variant.name);
        checked += 1;
    }
    assert!(checked >= 6, "only {checked} variants ran");
    let _ = std::fs::remove_dir_all(&dir);
}

fn check_against_ffmpeg(path: &Path, name: &str, dir: &Path) {
    let mut demux = open(path);
    let probe = ffprobe_stream(path);
    let stream = demux.streams()[0].clone();
    assert_eq!(probe.codec, "opus", "{name}");
    assert_eq!(stream.params().codec(), &StreamCodec::Known(CodecId::Opus));
    assert_eq!(stream.params().sample_rate(), Some(48_000));
    assert_eq!(stream.timescale().ticks_per_second(), 48_000);
    assert_eq!(stream.params().channels(), Some(probe.channels), "{name}");
    assert_eq!(
        stream.params().extradata().len(),
        probe.extradata_size,
        "{name}"
    );
    assert!(stream.params().extradata().starts_with(b"OpusHead"));
    let pre_skip = stream.trim().encoder_delay().expect("pre-skip");
    assert_eq!(
        i64::try_from(pre_skip).expect("pre"),
        probe.initial_padding,
        "{name}"
    );
    let pre_skip = probe.initial_padding;

    let packets = read_all(&mut demux);
    let expected = ffprobe_packets(path);
    assert_eq!(packets.len(), expected.len(), "{name} packet count");
    let payload = ffmpeg_payload(path);
    let mut at = 0;
    for (index, (mine, theirs)) in packets.iter().zip(&expected).enumerate() {
        assert_eq!(
            pts(mine),
            theirs.pts + pre_skip,
            "{name} packet {index} pts"
        );
        assert_eq!(
            mine.timestamp().decode(),
            mine.timestamp().presentation(),
            "{name}"
        );
        assert_eq!(
            mine.bytes().len(),
            theirs.size,
            "{name} packet {index} size"
        );
        assert_eq!(
            mine.bytes(),
            &payload[at..at + theirs.size],
            "{name} packet {index} bytes"
        );
        at += theirs.size;
        // FFmpeg keeps one sample on a packet that end trimming removes
        // whole; this crate reports zero.
        if !(theirs.duration == 1 && duration(mine) == 0) {
            assert_eq!(
                duration(mine),
                theirs.duration,
                "{name} packet {index} duration"
            );
        }
        assert!(mine.keyframe());
        assert_eq!(mine.timescale().ticks_per_second(), 48_000);
    }
    assert_eq!(at, payload.len(), "{name} payload length");
    for index in [0, packets.len() - 1] {
        assert_eq!(
            sha256_hex(packets[index].bytes(), dir),
            expected[index].hash,
            "{name} packet {index} sha256"
        );
    }
    for pair in packets.windows(2) {
        assert_eq!(
            pts(&pair[0]) + duration(&pair[0]),
            pts(&pair[1]),
            "{name} contiguous"
        );
    }

    // Gapless metadata against ffmpeg's decode.
    let info = demux.opus(0).expect("opus info").clone();
    let start = info.start.expect("start");
    let end = info.end.expect("end");
    assert_eq!(pts(&packets[0]), start);
    let last = packets.last().expect("last");
    assert_eq!(pts(last) + duration(last), end, "{name} end trim");
    let samples = i64::from(packet_samples(last.bytes()).expect("toc"));
    assert_eq!(
        info.trailing,
        Some(samples - duration(last)),
        "{name} trailing"
    );
    assert_eq!(
        stream.trim().trailing_padding(),
        u64::try_from(samples - duration(last)).ok()
    );
    let edit = stream.edits()[0];
    assert_eq!(edit.media_time(), Some(start + pre_skip));
    let decoded = ffmpeg_decoded_samples(path);
    assert_eq!(
        u64::try_from(edit.duration()).expect("edit"),
        decoded,
        "{name} playable samples"
    );
    assert!(demux.gaps().is_empty());
    assert_eq!(demux.crc_failures(), 0);
    assert_eq!(demux.lost_pages(), 0);
    check_against_gstreamer(path, name, dir, &packets, decoded);
    eprintln!(
        "{name}: {} packets, pre-skip {pre_skip}, start {start}, end {end}, trailing {:?}, decoded {decoded}",
        packets.len(),
        info.trailing
    );
}

/// GStreamer's `oggdemux` is a second demuxer, independent of libavformat:
/// same packet sizes, each packet ending at the same granule, and
/// `opusdec` decoding the same number of samples.
pub fn check_against_gstreamer(
    path: &Path,
    name: &str,
    dir: &Path,
    packets: &[Packet],
    decoded: u64,
) {
    let Some(theirs) = gstreamer_packets(path) else {
        eprintln!("{name}: GStreamer comparison skipped, gst-launch-1.0 is not on PATH");
        return;
    };
    assert_eq!(packets.len(), theirs.len(), "{name} GStreamer packet count");
    for (index, (mine, (size, end))) in packets.iter().zip(&theirs).enumerate() {
        assert_eq!(
            mine.bytes().len(),
            *size,
            "{name} GStreamer packet {index} size"
        );
        assert_eq!(
            pts(mine) + duration(mine),
            *end,
            "{name} GStreamer packet {index} end"
        );
    }
    let samples = gstreamer_decoded_samples(path, dir).expect("gst decode");
    assert_eq!(samples, decoded, "{name} GStreamer decoded samples");
}

fn check_seeks(path: &Path, name: &str) {
    let mut demux = open(path);
    let all = read_all(&mut demux);
    let start = pts(&all[0]);
    let last = all.last().expect("last");
    let end = pts(last) + duration(last);
    let targets = [
        start - 1000,
        start,
        start + 1,
        start + 3_840,
        start + (end - start) / 3,
        start + (end - start) / 2,
        end - 4_000,
        end - 1,
    ];
    for &target in &targets {
        let landing = demux
            .seek(SeekTo {
                track: 0,
                time: target,
                end: None,
            })
            .expect("seek");
        let Landing::Packet { time, preroll, .. } = landing.landing else {
            panic!("{name} seek {target} landed at the end");
        };
        assert!(!landing.index_used);
        let aim = (target - 3_840).max(start);
        let index = all
            .iter()
            .position(|packet| pts(packet) == time)
            .expect("landing is a packet start");
        let held = &all[index];
        let span = i64::from(packet_samples(held.bytes()).unwrap_or(0));
        assert!(
            time <= aim && aim < time + span.max(1),
            "{name} {target}: {time}"
        );
        let expected_preroll = all[index..]
            .iter()
            .take_while(|packet| pts(packet) < target)
            .count();
        assert_eq!(
            usize::try_from(preroll).expect("preroll"),
            expected_preroll,
            "{name} preroll at {target}"
        );
        let next = demux.read_packet().expect("read").expect("packet");
        assert_eq!(pts(&next), time, "{name} no stale packet at {target}");
        assert_eq!(next.bytes(), held.bytes());
        let after = demux.read_packet().expect("read");
        if let Some(after) = after {
            assert_eq!(after.bytes(), all[index + 1].bytes());
        }
    }
    let past = demux
        .seek(SeekTo {
            track: 0,
            time: end,
            end: None,
        })
        .expect("seek past end");
    assert_eq!(past.landing, Landing::End, "{name}");
    assert!(demux.read_packet().expect("read").is_none());
}

#[test]
fn a_chained_file_returns_its_first_link() {
    let Some(root) = environment("a_chained_file_returns_its_first_link") else {
        return;
    };
    let Some(source) = pick(&root, &[".m4a"]) else {
        eprintln!("skipped: no m4a");
        return;
    };
    let dir = scratch("chain");
    let first = dir.join("first.opus");
    let second = dir.join("second.opus");
    let src = source.to_str().expect("utf8");
    ffmpeg(&[
        "-i",
        src,
        "-t",
        "3",
        "-vn",
        "-c:a",
        "libopus",
        first.to_str().expect("p"),
    ]);
    ffmpeg(&[
        "-ss",
        "5",
        "-i",
        src,
        "-t",
        "2",
        "-vn",
        "-ac",
        "1",
        "-c:a",
        "libopus",
        second.to_str().expect("p"),
    ]);
    let mut chained = std::fs::read(&first).expect("first");
    chained.extend(std::fs::read(&second).expect("second"));
    let chain = dir.join("chain.opus");
    std::fs::write(&chain, &chained).expect("chain");

    let mut demux = open(&chain);
    assert!(demux.chained());
    assert_eq!(
        demux.link_end(),
        Some(std::fs::metadata(&first).expect("meta").len())
    );
    let packets = read_all(&mut demux);
    let expected = ffprobe_packets(&first);
    assert_eq!(packets.len(), expected.len());
    let decoded = ffmpeg_decoded_samples(&first);
    assert_eq!(
        u64::try_from(demux.streams()[0].edits()[0].duration()).expect("edit"),
        decoded
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_multiplexed_file_keeps_each_opus_stream_and_lists_flac_as_unknown() {
    let Some(root) = environment("a_multiplexed_file") else {
        return;
    };
    let Some(source) = pick(&root, &[".m4a"]) else {
        eprintln!("skipped: no m4a");
        return;
    };
    let dir = scratch("multi");
    let path = dir.join("multi.ogg");
    ffmpeg(&[
        "-i",
        source.to_str().expect("utf8"),
        "-t",
        "6",
        "-map",
        "0:a:0",
        "-map",
        "0:a:0",
        "-map",
        "0:a:0",
        "-c:a:0",
        "libopus",
        "-c:a:1",
        "flac",
        "-c:a:2",
        "libopus",
        "-ac:a:2",
        "1",
        "-b:a:2",
        "24k",
        path.to_str().expect("utf8"),
    ]);
    let mut demux = open(&path);
    assert_eq!(demux.streams().len(), 3);
    assert_eq!(
        demux.streams()[1].params().codec(),
        &StreamCodec::Unknown(cutlass_codec::UnknownCodec::new(b"ogg:flac".to_vec()).expect("t"))
    );
    assert_eq!(demux.streams()[2].params().channels(), Some(1));
    let packets = read_all(&mut demux);
    for (track, selector) in [(0_u32, "a:0"), (2, "a:2")] {
        let mine: Vec<&Packet> = packets.iter().filter(|p| p.stream() == track).collect();
        let theirs = probe_stream_packets(&path, selector);
        assert_eq!(mine.len(), theirs.len(), "track {track}");
        let pre_skip = i64::try_from(
            demux.streams()[track as usize]
                .trim()
                .encoder_delay()
                .expect("pre"),
        )
        .expect("pre");
        for (left, right) in mine.iter().zip(&theirs) {
            assert_eq!(pts(left), right.0 + pre_skip);
            assert_eq!(left.bytes().len(), right.1);
        }
    }
    assert!(packets.iter().all(|packet| packet.stream() != 1));
    let landing = demux
        .seek(SeekTo {
            track: 2,
            time: 96_000,
            end: None,
        })
        .expect("seek");
    let Landing::Packet { time, .. } = landing.landing else {
        panic!("end");
    };
    let next = loop {
        let packet = demux.read_packet().expect("read").expect("packet");
        if packet.stream() == 2 {
            break packet;
        }
    };
    assert_eq!(pts(&next), time);
    let _ = std::fs::remove_dir_all(&dir);
}

fn probe_stream_packets(path: &Path, selector: &str) -> Vec<(i64, usize)> {
    let out = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            selector,
            "-show_entries",
            "packet=pts,size",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .expect("ffprobe");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| {
            let mut parts = line.split(',');
            let pts = parts.next()?.parse().ok()?;
            let size = parts.next()?.parse().ok()?;
            Some((pts, size))
        })
        .collect()
}
