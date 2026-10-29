//! Release timing for open and seek.
//!
//! Debug builds still check that a PCM seek does not read the payload.
//! Wall-clock budgets apply when debug assertions are off.

use std::io::Cursor;
use std::process::Command;
use std::time::Instant;

use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
use cutlass_container::{Demux, Mux, NewStream, SeekTo, TrackDuration, TrackId};
use cutlass_riff::{WavDemux, WavMux};

#[test]
fn pcm_open_and_seek_stay_in_budget() {
    const FRAMES: i64 = 8_000;
    let pcm = vec![0u8; usize::try_from(FRAMES).expect("frames") * 4];
    let bytes = mux_pcm(&pcm, FRAMES);
    let file_len = bytes.len();
    time_file(&bytes, FRAMES, file_len, "generated");

    let local =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files");
    if local.is_dir()
        && let Ok(entries) = std::fs::read_dir(&local)
    {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("wav") {
                continue;
            }
            let file = std::fs::read(&path).expect("local");
            let len = file.len();
            let started = Instant::now();
            let mut demux = WavDemux::open(Cursor::new(file), u64::try_from(len).unwrap_or(0))
                .expect("local open");
            let opened = started.elapsed();
            eprintln!("wav local open {opened:?}");
            let samples = demux.duration_samples().unwrap_or(0);
            if samples > 1 {
                time_open_seek(&mut demux, samples, len, &path.display().to_string());
            }
            if !cfg!(debug_assertions) {
                assert!(
                    opened.as_millis() < 50,
                    "local open {opened:?} exceeded 50ms"
                );
            }
        }
    }
}

fn time_file(bytes: &[u8], frames: i64, file_len: usize, label: &str) {
    let started = Instant::now();
    let mut demux = WavDemux::open(
        Cursor::new(bytes.to_vec()),
        u64::try_from(file_len).unwrap_or(0),
    )
    .expect("open");
    let opened = started.elapsed();
    time_open_seek(
        &mut demux,
        u64::try_from(frames).unwrap_or(0),
        file_len,
        label,
    );
    eprintln!("wav {label} open {opened:?}");
    if !cfg!(debug_assertions) {
        assert!(opened.as_millis() < 50, "open {opened:?} exceeded 50ms");
    }
}

fn time_open_seek<R: std::io::Read + std::io::Seek>(
    demux: &mut WavDemux<R>,
    samples: u64,
    file_len: usize,
    label: &str,
) {
    let header = demux.consumed();
    let middle = i64::try_from(samples / 2).unwrap_or(0);
    let started = Instant::now();
    let landing = demux
        .seek(SeekTo {
            track: 0,
            time: middle,
            end: None,
        })
        .expect("seek");
    let seek_elapsed = started.elapsed();
    assert_eq!(demux.consumed(), header, "{label} seek read payload");
    assert_eq!(landing.payload_scans, 0);
    let machine = machine_line();
    eprintln!(
        "wav release sample: {label} {file_len} bytes, samples {samples}, seek {seek_elapsed:?}, header bytes {header}, {machine}"
    );
    if !cfg!(debug_assertions) {
        assert!(
            seek_elapsed.as_millis() < 20,
            "{label} seek {seek_elapsed:?} exceeded 20ms"
        );
    }
}

fn machine_line() -> String {
    let brand = Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
        .unwrap_or_default();
    let mem = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
        .unwrap_or_default();
    format!("{brand}, memsize {mem}")
}

fn mux_pcm(pcm: &[u8], frames: i64) -> Vec<u8> {
    let params = StreamParams::try_audio(StreamCodec::Known(CodecId::Pcm), 44_100, 2, Vec::new())
        .expect("pcm");
    let timestamp = Timestamp::try_new(Some(0), Some(0), Some(frames)).expect("time");
    let packet =
        Packet::try_new(0, pcm.to_vec(), timestamp, Timescale::new(44_100), true).expect("packet");
    let mut mux = WavMux::new(Cursor::new(Vec::new()));
    mux.add_stream(NewStream {
        params,
        timescale: Timescale::new(44_100),
        language: None,
    })
    .expect("stream");
    mux.write_packet(&packet).expect("write");
    mux.finish(&[TrackDuration {
        track: TrackId::new(0),
        timescale: Timescale::new(44_100),
        ticks: frames,
    }])
    .expect("finish");
    mux.into_inner().into_inner()
}
