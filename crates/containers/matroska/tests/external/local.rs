//! Media on this machine that is not checked in: read when present,
//! skipped when not.
//!
//! `CUTLASS_LOCAL_MEDIA` names the directory; the default is the
//! `local-files` folder beside the checkout. Every `.mkv`, `.mka` and
//! `.webm` in it is read to the end and checked against libavformat, and
//! seeked through.

use std::path::PathBuf;

use cutlass_container::{Demux, Landing, SeekTo};

use crate::oracle::agrees_with_libavformat;
use crate::support::{ffmpeg_tools, open, read_all, scratch};

fn local_dir() -> PathBuf {
    std::env::var_os("CUTLASS_LOCAL_MEDIA").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files"),
        PathBuf::from,
    )
}

#[test]
fn local_matroska_files_read_like_libavformat_and_seek() {
    let dir = local_dir();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!("skipped: no local media at {}", dir.display());
        return;
    };
    if !ffmpeg_tools() {
        return;
    }
    let scratch = scratch("local");
    let mut seen = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        let wanted = path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| matches!(ext, "mkv" | "mka" | "webm"));
        if !wanted {
            continue;
        }
        seen += 1;
        agrees_with_libavformat(&path, &scratch, None);
        let mut demux = open(&path);
        let packets = read_all(&mut demux);
        let streams = demux.streams().to_vec();
        for stream in &streams {
            let track = stream.id().get();
            let keys: Vec<i64> = packets
                .iter()
                .filter(|p| p.stream() == track && p.keyframe())
                .filter_map(|p| p.timestamp().presentation())
                .collect();
            let Some(last) = keys.last().copied() else {
                continue;
            };
            for step in 0..=10 {
                let time = last * step / 10;
                let landing = demux
                    .seek(SeekTo {
                        track,
                        time,
                        end: None,
                    })
                    .expect("seek");
                let want = keys
                    .iter()
                    .copied()
                    .filter(|key| *key <= time)
                    .max()
                    .or(keys.first().copied());
                match landing.landing {
                    Landing::Packet { time: got, .. } => {
                        assert_eq!(
                            Some(got),
                            want,
                            "{} track {track} at {time}",
                            path.display()
                        );
                    }
                    Landing::End => panic!("{}: end at {time}", path.display()),
                }
            }
        }
    }
    eprintln!(
        "local media: {seen} Matroska files read from {}",
        dir.display()
    );
}
