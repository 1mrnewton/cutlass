//! Files in `tests/fixtures` that stop early or state a fractional rate.
//! Each opens and plays the whole packets it holds, as libavformat plays
//! it; the decoded samples are compared with `ffmpeg -f s16le` when
//! `ffmpeg` is on the `PATH`.

use std::io::Cursor;

use cutlass_caf::CafDemux;
use cutlass_container::{Demux, Landing, SeekTo, TrackId};

use crate::support::{run, tool};

fn fixture_path(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(fixture_path(name)).unwrap_or_else(|error| panic!("{name}: {error}"))
}

fn open(bytes: &[u8]) -> CafDemux<Cursor<Vec<u8>>> {
    let len = u64::try_from(bytes.len()).expect("len");
    CafDemux::open(Cursor::new(bytes.to_vec()), len).expect("open")
}

/// The 100 samples the PCM fixtures were written from.
fn sequence() -> Vec<i16> {
    let mut state = 12_345_u64;
    (0..100)
        .map(|_| {
            state = (state * 1_103_515_245 + 12_345) & 0x7FFF_FFFF;
            let sample = i32::try_from((state >> 8) & 0xFFFF).expect("sample") - 32_768;
            i16::try_from(sample).expect("i16")
        })
        .collect()
}

fn s16_be(bytes: &[u8]) -> Vec<i16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| i16::from_be_bytes(*pair))
        .collect()
}

fn s16_le(bytes: &[u8]) -> Vec<i16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| i16::from_le_bytes(*pair))
        .collect()
}

fn ffmpeg_s16(name: &str) -> Option<Vec<i16>> {
    if !tool("ffmpeg") {
        eprintln!("skipped: ffmpeg is not on PATH");
        return None;
    }
    let file = fixture_path(name);
    let file = file.to_str().expect("path");
    // ffmpeg reports the packet the file ends inside and exits non-zero
    // on some builds; the samples before it are still written.
    let out = std::process::Command::new("ffmpeg")
        .args(["-v", "quiet", "-i", file, "-f", "s16le", "-"])
        .output()
        .expect("ffmpeg");
    Some(s16_le(&out.stdout))
}

/// `data` states 100 frames of 16-bit mono and the file stops inside the
/// seventieth.
#[test]
fn pcm_data_that_runs_past_the_file_plays_the_whole_frames() {
    let bytes = fixture("data-past-end.caf");
    let want = &sequence()[..69];
    let mut demux = open(&bytes);
    assert!(demux.is_cut());
    assert_eq!(demux.total_frames(), Some(69));
    assert_eq!(demux.packet_count(), Some(69));
    assert_eq!(demux.indexed_end(TrackId::new(0)), Some(69));
    let packet = demux.read_packet().expect("read").expect("packet");
    assert_eq!(packet.timestamp().duration(), Some(69));
    assert_eq!(s16_be(packet.bytes()), want);
    assert!(demux.read_packet().expect("end").is_none());
    let last = demux
        .seek(SeekTo {
            track: 0,
            time: 68,
            end: None,
        })
        .expect("seek");
    assert!(matches!(last.landing, Landing::Packet { time: 68, .. }));
    let packet = demux.read_packet().expect("read").expect("last frame");
    assert_eq!(s16_be(packet.bytes()), &want[68..]);

    // A forward reader returns the same frames and then knows.
    let mut forward = CafDemux::open_forward(Cursor::new(bytes)).expect("forward");
    assert!(!forward.is_cut());
    let packet = forward.read_packet().expect("read").expect("packet");
    assert_eq!(s16_be(packet.bytes()), want);
    assert!(forward.is_cut());
    assert!(forward.read_packet().expect("end").is_none());

    if let Some(theirs) = ffmpeg_s16("data-past-end.caf") {
        assert_eq!(theirs, want);
    }
}

/// ALAC, four packets in the table, the file cut inside the third. The
/// two whole packets are returned and the rows after them dropped.
#[test]
fn packet_table_rows_past_the_end_of_the_file_are_dropped() {
    let bytes = fixture("pakt-past-end.caf");
    let mut demux = open(&bytes);
    assert!(demux.is_cut());
    assert_eq!(demux.packet_table().expect("pakt").packets, 4);
    assert_eq!(demux.packet_count(), Some(2));
    assert_eq!(demux.total_frames(), Some(8_192));
    assert_eq!(demux.indexed_end(TrackId::new(0)), Some(8_192));
    let mut seen = Vec::new();
    while let Some(packet) = demux.read_packet().expect("read") {
        seen.push((
            packet.timestamp().presentation(),
            packet.timestamp().duration(),
            packet.bytes().len(),
        ));
    }
    assert_eq!(
        seen,
        [
            (Some(0), Some(4_096), 3_003),
            (Some(4_096), Some(4_096), 3_003)
        ]
    );
    // The dropped packets are past the end for a seek as well.
    let end = demux
        .seek(SeekTo {
            track: 0,
            time: 8_192,
            end: None,
        })
        .expect("seek");
    assert!(matches!(end.landing, Landing::End));
    let second = demux
        .seek(SeekTo {
            track: 0,
            time: 5_000,
            end: None,
        })
        .expect("seek");
    assert!(matches!(
        second.landing,
        Landing::Packet {
            ordinal: 1,
            time: 4_096,
            ..
        }
    ));

    // ffmpeg decodes the same two packets: 8,192 samples, the triangle
    // wave the file was encoded from.
    if let Some(theirs) = ffmpeg_s16("pakt-past-end.caf") {
        let wave: Vec<i16> = (0..8_192_i16)
            .map(|index| (((index % 64) - 32).abs() - 16) * 256)
            .collect();
        assert_eq!(theirs.len(), 8_192);
        assert!(theirs == wave);
    }
}

/// 244800 / 11 Hz opens at 22255, the nearest whole rate. ffmpeg's CAF
/// demuxer drops the fraction and says 22254; the samples are the same.
#[test]
fn a_fractional_sample_rate_is_the_nearest_hertz() {
    let bytes = fixture("rate-22254.caf");
    let mut demux = open(&bytes);
    assert_eq!(demux.sample_rate(), 22_255);
    assert_eq!(demux.streams()[0].timescale().ticks_per_second(), 22_255);
    assert!(!demux.is_cut());
    let packet = demux.read_packet().expect("read").expect("packet");
    assert_eq!(packet.timescale().ticks_per_second(), 22_255);
    assert_eq!(s16_be(packet.bytes()), sequence());
    if let Some(theirs) = ffmpeg_s16("rate-22254.caf") {
        assert_eq!(theirs, sequence());
    }
    if tool("ffprobe") {
        let file = fixture_path("rate-22254.caf");
        let rate = run(
            "ffprobe",
            &[
                "-v",
                "error",
                "-show_entries",
                "stream=sample_rate",
                "-of",
                "csv=p=0",
                file.to_str().expect("path"),
            ],
        );
        if let Some(rate) = rate {
            assert_eq!(String::from_utf8_lossy(&rate).trim(), "22254");
        }
    }
}
