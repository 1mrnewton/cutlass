//! Real WAV files under `../local-files`. The test skips when that folder is missing.

use std::fs::File;
use std::path::PathBuf;

use crate::support::{ffprobe_stream, output, tool_on_path};
use cutlass_container::{Demux, Landing, SeekTo};
use cutlass_riff::WavDemux;

fn local_root() -> Option<PathBuf> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files");
    if path.is_dir() { Some(path) } else { None }
}

#[test]
fn local_wav_files_open_and_seek_without_reading_every_frame() {
    let Some(root) = local_root() else {
        eprintln!("skipped: ../local-files is missing");
        return;
    };
    let mut opened = 0u32;
    let entries = std::fs::read_dir(&root).expect("local-files");
    for entry in entries {
        let path = entry.expect("entry").path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("wav") {
            continue;
        }
        opened += 1;
        check_wav(&path);
    }
    if opened == 0 {
        eprintln!("skipped: ../local-files has no wav files");
    }
}

fn check_wav(path: &std::path::Path) {
    let file = File::open(path).expect("open");
    let len = file.metadata().expect("meta").len();
    let mut demux = WavDemux::open(file, len).expect("demux");
    let rate = demux.format().sample_rate();
    let channels = demux.format().channels();
    assert_eq!(demux.streams()[0].timescale().ticks_per_second(), rate);
    assert_ne!(rate, 1_000_000);
    let samples = demux.duration_samples().expect("duration");
    assert!(samples > 1, "{}", path.display());
    let header = demux.consumed();
    assert!(header < 64 * 1024, "open read {header} of {len}");

    let first = demux.seek(at(0)).expect("first");
    assert!(matches!(first.landing, Landing::Packet { ordinal: 0, .. }));
    assert_eq!(first.payload_scans, 0);
    assert_eq!(demux.consumed(), header);
    let packet = demux.read_packet().expect("read").expect("frame");
    assert_eq!(packet.timestamp().presentation(), Some(0));
    assert_eq!(
        packet.bytes().len(),
        usize::from(demux.format().block_align())
    );
    if tool_on_path("python3")
        && let Some(frame) = python_first_frame(path)
    {
        assert_eq!(frame, packet.bytes());
    }

    let middle = i64::try_from(samples / 2).expect("middle");
    let mid = demux.seek(at(middle)).expect("middle");
    match mid.landing {
        Landing::Packet { time, .. } => assert_eq!(time, middle),
        Landing::End => panic!("middle landed past the end"),
    }
    assert_eq!(mid.payload_scans, 0);

    let last = i64::try_from(samples - 1).expect("last");
    let end = demux.seek(at(last)).expect("last");
    match end.landing {
        Landing::Packet { time, .. } => assert_eq!(time, last),
        Landing::End => panic!("last frame was past the end"),
    }
    let past = demux
        .seek(at(i64::try_from(samples).expect("end")))
        .expect("past");
    assert!(matches!(past.landing, Landing::End));

    if tool_on_path("ffprobe") {
        let probed = ffprobe_stream(path);
        assert_eq!(probed.sample_rate, rate);
        assert_eq!(probed.channels, channels);
        assert_eq!(probed.duration_ts, Some(samples));
    }
}

fn at(time: i64) -> SeekTo {
    SeekTo {
        track: 0,
        time,
        end: None,
    }
}

fn python_first_frame(path: &std::path::Path) -> Option<Vec<u8>> {
    let script = "import wave,sys\nw=wave.open(sys.argv[1],'rb')\nsys.stdout.buffer.write(w.readframes(1))\n";
    let out = output("python3", &["-c", script, &path.display().to_string()]);
    if out.status.success() && !out.stdout.is_empty() {
        Some(out.stdout)
    } else {
        None
    }
}
