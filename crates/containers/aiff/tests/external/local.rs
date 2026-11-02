//! Real AIFF files under `../local-files`. The test skips when that folder,
//! or a file of this type, is missing.

use std::fs::File;
use std::path::{Path, PathBuf};

use cutlass_aiff::AiffDemux;
use cutlass_container::{Demux, Landing, SeekTo};

fn local_root() -> Option<PathBuf> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files");
    path.is_dir().then_some(path)
}

#[test]
fn local_aiff_files_open_and_seek_without_reading_the_sound() {
    let Some(root) = local_root() else {
        eprintln!("skipped: ../local-files is missing");
        return;
    };
    let mut opened = 0u32;
    let entries = std::fs::read_dir(&root).expect("local-files");
    for entry in entries {
        let path = entry.expect("entry").path();
        let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
            continue;
        };
        if !ext.eq_ignore_ascii_case("aiff") && !ext.eq_ignore_ascii_case("aif") {
            continue;
        }
        opened += 1;
        check_aiff(&path);
    }
    if opened == 0 {
        eprintln!("skipped: ../local-files has no aiff files");
    }
}

fn check_aiff(path: &Path) {
    let file = File::open(path).expect("open");
    let len = file.metadata().expect("meta").len();
    let mut demux = AiffDemux::open(file, len).unwrap_or_else(|error| {
        panic!("{}: {error}", path.display());
    });
    let samples = demux.duration_samples();
    assert!(samples > 0, "{}", path.display());
    let header = demux.consumed();
    assert!(header < 64 * 1024, "open read {header} of {len}");
    let first = demux
        .seek(SeekTo {
            track: 0,
            time: 0,
            end: None,
        })
        .expect("first");
    assert!(matches!(first.landing, Landing::Packet { ordinal: 0, .. }));
    assert_eq!(first.payload_scans, 0);
    assert_eq!(demux.consumed(), header);
    let packet = demux.read_packet().expect("read").expect("frame");
    assert_eq!(packet.timestamp().presentation(), Some(0));
    assert!(!packet.bytes().is_empty());
    if demux.streams()[0].params().extradata().len() == 3 && samples > 1 {
        let middle = i64::try_from(samples / 2).expect("middle");
        let mid = demux
            .seek(SeekTo {
                track: 0,
                time: middle,
                end: None,
            })
            .expect("middle");
        match mid.landing {
            Landing::Packet { time, .. } => assert_eq!(time, middle),
            Landing::End => panic!("middle landed past the end"),
        }
        assert_eq!(mid.payload_scans, 0);
        let last = i64::try_from(samples - 1).expect("last");
        let end = demux
            .seek(SeekTo {
                track: 0,
                time: last,
                end: None,
            })
            .expect("last");
        match end.landing {
            Landing::Packet { time, .. } => assert_eq!(time, last),
            Landing::End => panic!("last frame was past the end"),
        }
    }
}
