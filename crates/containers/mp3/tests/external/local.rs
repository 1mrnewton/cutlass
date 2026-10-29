//! Real files under `../local-files` relative to the checkout.
//! The test skips when that folder is missing, which is the CI case.

use std::fs::File;
use std::io::Cursor;
use std::path::PathBuf;

use crate::support::{ffprobe, tool_on_path};
use cutlass_mp3_container::Demux;
use cutlass_mp3_container::Mp3Demux;

fn local_root() -> Option<PathBuf> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files");
    if path.is_dir() { Some(path) } else { None }
}

#[test]
fn local_mp3_files_keep_a_monotonic_sample_clock() {
    let Some(root) = local_root() else {
        eprintln!("skipped: ../local-files is missing");
        return;
    };
    let names = [
        "thunder_distant_10s.mp3",
        "sfx_A_sin_20260721_231942.mp3",
        "Seamless_gentle_rain_#1-1784676090807.mp3",
    ];
    let mut opened = 0;
    for name in names {
        let path = root.join(name);
        if !path.is_file() {
            continue;
        }
        opened += 1;
        let bytes = std::fs::read(&path).expect("read");
        check_file(&path, &bytes);
    }
    assert!(opened >= 2, "expected at least two local mp3 files");

    let long = root.join("rain_bedroom_night_30min.mp3");
    if long.is_file() {
        seek_without_reading_the_whole_file(&long);
    }
}

fn check_file(path: &std::path::Path, bytes: &[u8]) {
    let mut demux = Mp3Demux::open(Cursor::new(bytes.to_vec())).expect("open");
    let stream = demux.streams()[0].clone();
    assert_eq!(stream.timescale, stream.params.sample_rate);
    assert_ne!(stream.timescale, 1_000_000);
    assert!(stream.params.channels == 1 || stream.params.channels == 2);
    let mut count = 0u64;
    let mut clock = 0i64;
    while let Some(packet) = demux.next_packet().expect("packet") {
        assert_eq!(packet.pts, clock);
        assert!(packet.duration > 0);
        assert_eq!(packet.timescale, stream.timescale);
        assert!(packet.keyframe);
        assert!(packet.data.len() >= 4);
        clock += packet.duration;
        count += 1;
    }
    assert!(count > 1, "{}", path.display());
    if demux.duration_exact() {
        assert_eq!(clock, demux.duration().expect("duration"));
    }
    if tool_on_path("ffprobe") {
        let probed = ffprobe(path);
        assert_eq!(probed.sample_rate, stream.params.sample_rate);
        assert_eq!(probed.channels, stream.params.channels);
        assert!(
            probed.packets.len() == usize::try_from(count).expect("count")
                || probed.packets.len() == usize::try_from(count).expect("count") + 1,
            "{} ffprobe {} ours {count}",
            path.display(),
            probed.packets.len()
        );
    }
}

fn seek_without_reading_the_whole_file(path: &std::path::Path) {
    let file = File::open(path).expect("open");
    let mut demux = Mp3Demux::open(file).expect("demux");
    let duration = demux.duration().expect("duration");
    assert!(duration > 1_000_000);
    let landing = demux.seek(duration / 2).expect("seek");
    assert!(landing.indexed);
    let packet = demux.next_packet().expect("packet").expect("frame");
    assert_eq!(packet.pts, landing.pts);
    assert!(packet.pts > 0);
    assert_eq!(packet.timescale, demux.streams()[0].timescale);
}
