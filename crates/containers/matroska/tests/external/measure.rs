//! What a seek costs: no frame is inspected, and the bytes read stay
//! within the GOP the walk crosses, with Cues and without.
//!
//! The file is 20 seconds of 640x360 H.264 at 4 Mb/s with a keyframe every
//! two seconds, and AAC, written by FFmpeg (about 10 MB). The copy without
//! Cues is the same packets written by this crate's muxer and cut before
//! its Cues.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use cutlass_container::{Demux, Landing, SeekTo};
use cutlass_matroska::MatroskaDemux;

use crate::support::{ffmpeg, ffmpeg_tools, scratch};

fn seek_costs(path: &Path) -> (u64, Vec<(i64, u64, bool, u32)>) {
    let file = File::open(path).expect("file");
    let length = file.metadata().expect("metadata").len();
    let mut demux = MatroskaDemux::open(BufReader::new(file), length).expect("open");
    let opened = demux.consumed();
    let mut costs = Vec::new();
    for time in [15_300, 2_100, 9_999, 19_000, 600, 12_345] {
        let before = demux.consumed();
        let landing = demux
            .seek(SeekTo {
                track: 1,
                time,
                end: None,
            })
            .expect("seek");
        assert_eq!(landing.payload_scans, 0);
        let Landing::Packet {
            time: landed,
            keyframe,
            ..
        } = landing.landing
        else {
            panic!("end at {time}");
        };
        assert!(keyframe);
        assert!(
            landed <= time && time - landed < 2_100,
            "{time} landed {landed}"
        );
        costs.push((
            time,
            demux.consumed() - before,
            landing.index_used,
            landing.entries_visited,
        ));
    }
    (opened, costs)
}

#[test]
fn seeks_read_no_frames_and_a_small_part_of_the_file() {
    if !ffmpeg_tools() {
        return;
    }
    let dir = scratch("measure");
    let path = dir.join("long.mkv");
    assert!(ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=640x360:rate=25:duration=20",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000:duration=20",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-b:v",
        "4M",
        "-g",
        "50",
        "-c:a",
        "aac",
        &path.display().to_string(),
    ]));
    let length = std::fs::metadata(&path).expect("metadata").len();
    let (opened, costs) = seek_costs(&path);
    eprintln!("file {length} bytes, open read {opened} bytes");
    for (time, bytes, index, visited) in &costs {
        eprintln!("  seek {time} ms: {bytes} bytes read, index {index}, {visited} entries");
        assert!(*index);
        // Each seek reads across at most the GOP it walks (a tenth of the
        // file) and one buffer past it: the walk reads whole buffers, since
        // a read call costs far more than the bytes it brings.
        assert!(
            *bytes < length / 10 + 64 * 1024,
            "seek to {time} read {bytes} of {length}"
        );
    }
    assert!(opened <= 64 * 1024);

    // The same packets without Cues: the first seek walks Cluster headers.
    let copy = dir.join("copy.mkv");
    crate::support::remux(&path, &copy, false);
    let bytes = std::fs::read(&copy).expect("copy");
    let cues = bytes
        .windows(4)
        .rposition(|window| window == [0x1C, 0x53, 0xBB, 0x6B])
        .expect("Cues");
    let cut = dir.join("no-cues.mkv");
    std::fs::write(&cut, &bytes[..cues]).expect("write");
    let (_, costs) = seek_costs(&cut);
    let first = costs.first().expect("first").1;
    eprintln!("no Cues: first seek {first} bytes (builds the Cluster list)");
    for (time, bytes, index, visited) in &costs {
        eprintln!("  seek {time} ms: {bytes} bytes read, index {index}, {visited} entries");
        assert!(!*index);
        assert!(
            bytes * 4 < length,
            "seek to {time} read {bytes} of {length}"
        );
    }
}
