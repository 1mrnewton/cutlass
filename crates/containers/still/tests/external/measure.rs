//! Release timing for open and seek of a large JPEG.
//!
//! The file is one packet. Open scans markers. Seek answers from the index
//! built at open and does not read the payload again. Wall-clock budgets
//! apply when debug assertions are off. The measured numbers are in
//! `SUPPORT.md`.

use std::io::{Cursor, Read, Seek, SeekFrom};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use cutlass_container::{Demux, Input, Landing, SeekTo};
use cutlass_still::{StillDemux, still_limits};

struct Meter {
    inner: Cursor<Vec<u8>>,
    bytes: Arc<AtomicU64>,
}

impl Read for Meter {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let amount = self.inner.read(buf)?;
        self.bytes
            .fetch_add(u64::try_from(amount).unwrap_or(0), Ordering::Relaxed);
        Ok(amount)
    }
}

impl Seek for Meter {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(pos)
    }
}

#[test]
fn open_scans_a_large_jpeg_and_seek_does_not_read_it_again() {
    let file = large_jpeg(8 * 1024 * 1024);
    let file_len = file.len();
    let reads = Arc::new(AtomicU64::new(0));
    let started = Instant::now();
    let mut demux = StillDemux::open(Input::random(
        Meter {
            inner: Cursor::new(file),
            bytes: Arc::clone(&reads),
        },
        u64::try_from(file_len).expect("len"),
        still_limits(),
    ))
    .expect("open");
    let opened = started.elapsed();
    assert_eq!(
        reads.load(Ordering::Relaxed),
        u64::try_from(file_len).expect("len")
    );
    assert_eq!(demux.consumed(), u64::try_from(file_len).expect("len"));
    reads.store(0, Ordering::Relaxed);
    let seek_started = Instant::now();
    let landing = demux
        .seek(SeekTo {
            track: 0,
            time: 0,
            end: None,
        })
        .expect("seek");
    let seek_elapsed = seek_started.elapsed();
    assert_eq!(landing.payload_scans, 0);
    assert!(landing.index_used);
    assert!(matches!(
        landing.landing,
        Landing::Packet { ordinal: 0, .. }
    ));
    assert_eq!(reads.load(Ordering::Relaxed), 0);
    assert_eq!(demux.consumed(), u64::try_from(file_len).expect("len"));
    let packet = demux.read_packet().expect("read").expect("packet");
    assert_eq!(packet.bytes().len(), file_len);
    assert!(packet.timestamp().is_missing());
    eprintln!(
        "still release sample: {file_len} bytes, open {opened:?}, seek {seek_elapsed:?}, seek bytes 0"
    );
    if !cfg!(debug_assertions) {
        assert!(opened.as_millis() < 100, "open {opened:?} exceeded 100ms");
        assert!(
            seek_elapsed.as_millis() < 5,
            "seek {seek_elapsed:?} exceeded 5ms"
        );
    }
}

fn large_jpeg(entropy: usize) -> Vec<u8> {
    let mut bytes = vec![
        0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00,
        0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00,
    ];
    bytes.resize(bytes.len() + entropy, 0x00);
    bytes.extend_from_slice(&[0xFF, 0xD9]);
    bytes
}
