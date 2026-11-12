//! The committed files from other muxers read as FFmpeg 8.1.1's ffprobe
//! reads them: per track the packet count, keyframes, first and last PTS.
//! Those numbers were checked packet by packet against `ffprobe
//! -show_packets` (sizes too, with ADTS headers taken off); see
//! `tests/fixtures/README.md`.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::io::Cursor;
use std::path::Path;

use cutlass_codec::{CodecId, StreamCodec};
use cutlass_container::{Demux, Landing, SeekTo};
use cutlass_mpegts::TsDemux;
use cutlass_mpegts::packet::PacketSize;

/// One track as it should read: PID, codec (None for an unknown one),
/// packets, keyframes, first PTS, last PTS.
type Track = (u32, Option<CodecId>, usize, usize, i64, i64);

fn read(name: &str) -> (TsDemux<Cursor<Vec<u8>>>, Vec<u8>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    let bytes = std::fs::read(path).expect("fixture");
    let demux = TsDemux::open(Cursor::new(bytes.clone()), bytes.len() as u64).expect("open");
    (demux, bytes)
}

fn check(name: &str, expected: &[Track]) {
    let (mut demux, _) = read(name);
    let codecs: Vec<(u32, Option<CodecId>)> = demux
        .streams()
        .iter()
        .map(|stream| {
            let codec = match stream.params().codec() {
                StreamCodec::Known(id) => Some(*id),
                StreamCodec::Unknown(_) => None,
            };
            (stream.id().get(), codec)
        })
        .collect();
    let wanted: Vec<(u32, Option<CodecId>)> = expected.iter().map(|t| (t.0, t.1)).collect();
    assert_eq!(codecs, wanted, "{name}: streams");

    let mut seen: Vec<(usize, usize, Option<i64>, Option<i64>)> =
        vec![(0, 0, None, None); expected.len()];
    while let Some(packet) = demux.read_packet().expect("read") {
        let slot = expected
            .iter()
            .position(|t| t.0 == packet.stream())
            .expect("known track");
        let entry = &mut seen[slot];
        entry.0 += 1;
        entry.1 += usize::from(packet.keyframe());
        if let Some(pts) = packet.timestamp().presentation() {
            entry.2 = Some(entry.2.map_or(pts, |low: i64| low.min(pts)));
            entry.3 = Some(entry.3.map_or(pts, |high: i64| high.max(pts)));
        }
    }
    for (track, got) in expected.iter().zip(&seen) {
        assert_eq!(
            (got.0, got.1, got.2, got.3),
            (track.2, track.3, Some(track.4), Some(track.5)),
            "{name}: track {:#x}",
            track.0
        );
    }
    let counters = demux.counters();
    assert_eq!(counters.cc_errors, 0, "{name}");
    assert_eq!(counters.dropped_units, 0, "{name}");
    assert_eq!(demux.resyncs(), 0, "{name}");
}

#[test]
fn ffmpeg_h264_with_b_frames_and_adts() {
    check(
        "h264_aac.ts",
        &[
            (0x100, Some(CodecId::H264), 30, 3, 133_200, 237_600),
            (0x101, Some(CodecId::Aac), 58, 58, 131_280, 240_720),
        ],
    );
}

#[test]
fn ffmpeg_m2ts_with_hdmv_eac3() {
    let (demux, _) = read("h264_eac3.m2ts");
    assert_eq!(demux.packet_size(), PacketSize::M2ts);
    check(
        "h264_eac3.m2ts",
        &[
            (0x1011, Some(CodecId::H264), 30, 1, 126_480, 230_880),
            (0x1100, Some(CodecId::Eac3), 38, 38, 126_000, 232_560),
        ],
    );
}

#[test]
fn ffmpeg_latm() {
    check(
        "h264_latm.ts",
        &[
            (0x100, Some(CodecId::H264), 30, 1, 127_920, 232_320),
            (0x101, Some(CodecId::Aac), 58, 58, 126_000, 235_440),
        ],
    );
}

#[test]
fn ffmpeg_opus() {
    check(
        "h264_opus.ts",
        &[
            (0x100, Some(CodecId::H264), 30, 1, 126_585, 230_985),
            (0x101, Some(CodecId::Opus), 61, 61, 126_000, 234_000),
        ],
    );
}

#[test]
fn ffmpeg_hevc_and_ac3() {
    check(
        "hevc_ac3.ts",
        &[
            (0x100, Some(CodecId::Hevc), 30, 3, 133_200, 237_600),
            (0x101, Some(CodecId::Ac3), 38, 38, 132_720, 239_280),
        ],
    );
}

#[test]
fn ffmpeg_mpeg2_video_and_layer_ii() {
    // Layer II has no codec id, so it reads as an unknown codec.
    check(
        "mpeg2_mp2.ts",
        &[
            (0x100, Some(CodecId::Mpeg2), 30, 6, 129_600, 234_000),
            (0x101, None, 50, 50, 128_698, 234_538),
        ],
    );
}

#[test]
fn ffmpeg_two_programs_opens_the_first() {
    let (demux, _) = read("two_programs.ts");
    assert_eq!(demux.programs().len(), 2);
    assert_eq!(demux.program(), 1);
    check(
        "two_programs.ts",
        &[
            (0x100, Some(CodecId::H264), 30, 1, 128_255, 232_655),
            (0x101, Some(CodecId::Mp3), 51, 51, 126_183, 234_183),
        ],
    );
}

#[test]
fn gstreamer_mpegtsmux() {
    check(
        "gst_h264_aac.ts",
        &[
            (0x41, Some(CodecId::H264), 30, 3, 324_000_000, 324_104_400),
            (0x42, Some(CodecId::Aac), 52, 52, 324_000_000, 324_097_920),
        ],
    );
}

/// Every fixture seeks on its first track to its start and two later
/// points, and lands on a keyframe at or before the target (or the first
/// keyframe when the target is before it).
#[test]
fn every_fixture_seeks() {
    for name in [
        "gst_h264_aac.ts",
        "h264_aac.ts",
        "h264_eac3.m2ts",
        "h264_latm.ts",
        "h264_opus.ts",
        "hevc_ac3.ts",
        "mpeg2_mp2.ts",
        "two_programs.ts",
    ] {
        let (mut demux, _) = read(name);
        let track = demux.streams()[0].id().get();
        let start = demux.start();
        let mut first_key = None;
        for offset in [0_i64, 45_000, 90_000] {
            let target = start + offset;
            let landing = demux
                .seek(SeekTo {
                    track,
                    time: target,
                    end: demux.end(),
                })
                .unwrap_or_else(|err| panic!("{name} seek {offset}: {err:?}"));
            assert!(matches!(landing.landing, Landing::Packet { .. }), "{name}");
            let first = demux.read_packet().expect("read").expect("a packet");
            assert_eq!(first.stream(), track, "{name} {offset}");
            assert!(first.keyframe(), "{name} {offset}");
            let pts = first.timestamp().presentation().unwrap();
            let key = *first_key.get_or_insert(pts);
            assert!(pts <= target.max(key), "{name} {offset}: {pts} > {target}");
        }
    }
}
