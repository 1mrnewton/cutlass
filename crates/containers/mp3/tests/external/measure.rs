//! Release timing on an in-memory constant-bitrate file.
//!
//! Debug builds still check the seek window. Wall-clock budgets apply when
//! debug assertions are off. Hardware for the recorded run is in SUPPORT.md.

use std::io::{Cursor, Read, Seek, SeekFrom};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use crate::support::mpeg1_frame;
use cutlass_mp3_container::{CodecId, CodecParams, Packet};
use cutlass_mp3_container::{Demux, Mux};
use cutlass_mp3_container::{Mp3Demux, Mp3Mux};

struct Meter {
    inner: Cursor<Vec<u8>>,
    bytes: Arc<AtomicU64>,
    armed: Arc<AtomicBool>,
}

impl Read for Meter {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        if self.armed.load(Ordering::Relaxed) {
            self.bytes
                .fetch_add(u64::try_from(n).unwrap_or(0), Ordering::Relaxed);
        }
        Ok(n)
    }
}

impl Seek for Meter {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(pos)
    }
}

#[test]
fn indexed_seek_does_not_scan_and_release_time_stays_in_budget() {
    const FRAMES: u16 = 4_000;
    let mut mux = Mp3Mux::new(Cursor::new(Vec::new()));
    mux.add_stream(CodecParams::audio(CodecId::Mp3, 48_000, 2))
        .expect("stream");
    for index in 0..FRAMES {
        let frame = mpeg1_frame(9, 1, false, index);
        let pts = i64::from(index) * 1152;
        mux.write_packet(&Packet::audio(frame, pts, 1152, 48_000))
            .expect("write");
    }
    mux.finish().expect("finish");
    let bytes = mux.into_inner().expect("inner").into_inner();
    let file_len = bytes.len();

    let started = Instant::now();
    let mut demux = Mp3Demux::open(Cursor::new(bytes.clone())).expect("open");
    let opened = started.elapsed();
    assert!(demux.next_packet().expect("pkt").is_some());
    let first = started.elapsed();
    let mut count = 1u32;
    while demux.next_packet().expect("pkt").is_some() {
        count += 1;
    }
    let sequential = started.elapsed();
    assert_eq!(count, u32::from(FRAMES));

    let read = Arc::new(AtomicU64::new(0));
    let armed = Arc::new(AtomicBool::new(false));
    let mut demux = Mp3Demux::open(Meter {
        inner: Cursor::new(bytes),
        bytes: Arc::clone(&read),
        armed: Arc::clone(&armed),
    })
    .expect("reopen");
    armed.store(true, Ordering::Relaxed);
    let seek_started = Instant::now();
    let landing = demux.seek(2_000 * 1152).expect("seek");
    let seek_elapsed = seek_started.elapsed();
    let packet = demux.next_packet().expect("pkt").expect("frame");
    let scanned = read.load(Ordering::Relaxed);
    assert_eq!(landing.pts, packet.pts);
    assert!(landing.indexed);
    assert!(
        scanned < 128 * 1024,
        "seek read {scanned} bytes of {file_len}"
    );

    eprintln!(
        "mp3 release sample: {file_len} bytes, open {opened:?}, first packet {first:?}, sequential {sequential:?}, seek {seek_elapsed:?}, seek bytes {scanned}"
    );
    if !cfg!(debug_assertions) {
        assert!(
            first.as_millis() < 30,
            "first packet {first:?} exceeded 30ms"
        );
        assert!(
            sequential.as_millis() < 80,
            "sequential {sequential:?} exceeded 80ms"
        );
        assert!(
            seek_elapsed.as_millis() < 20,
            "seek {seek_elapsed:?} exceeded 20ms"
        );
    }
}
