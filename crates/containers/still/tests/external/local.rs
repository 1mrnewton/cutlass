//! Real files under `../local-files` relative to the checkout.
//! The test skips when that folder is missing, which is the CI case.

use std::fs::File;
use std::io::{Cursor, Read, Seek};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use cutlass_codec::Timescale;
use cutlass_container::{Demux, Input, OpenError};
use cutlass_still::{StillDemux, StillKind, still_limits};

fn local_root() -> Option<PathBuf> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files");
    if path.is_dir() { Some(path) } else { None }
}

#[test]
fn local_png_is_one_packet_and_a_movie_is_refused_from_the_signature() {
    let Some(root) = local_root() else {
        eprintln!("skipped: ../local-files is missing");
        return;
    };
    let png = root.join("04-motion-graphics.png");
    if png.is_file() {
        let bytes = std::fs::read(&png).expect("png");
        let mut demux = StillDemux::open(Input::random(
            Cursor::new(bytes.clone()),
            u64::try_from(bytes.len()).expect("len"),
            still_limits(),
        ))
        .expect("open png");
        assert_eq!(demux.kind(), StillKind::Png);
        assert!(!demux.animated());
        assert_eq!(demux.media_end(), None);
        assert_eq!(demux.streams()[0].timescale(), Timescale::new(0));
        assert!(demux.streams()[0].params().extradata().is_empty());
        let packet = demux.read_packet().expect("read").expect("packet");
        assert_eq!(packet.bytes(), bytes.as_slice());
        assert!(packet.timestamp().is_missing());
    } else {
        eprintln!("skipped: 04-motion-graphics.png is not in ../local-files");
    }

    let mut refused = 0;
    for name in [
        "12129324_3840_2160_30fps.mp4",
        "file_example_WEBM_1920_3_7MB.webm",
    ] {
        let path = root.join(name);
        if !path.is_file() {
            eprintln!("skipped: {name} is not in ../local-files");
            continue;
        }
        refuse_movie(&path);
        refused += 1;
    }
    if refused == 0 {
        eprintln!("skipped: no movie file in ../local-files");
    }
}

fn refuse_movie(path: &Path) {
    let file = File::open(path).expect("open");
    let length = file.metadata().expect("meta").len();
    let reads = std::sync::Arc::new(AtomicU64::new(0));
    let counted = Counted {
        inner: file,
        reads: std::sync::Arc::clone(&reads),
    };
    let error =
        StillDemux::open(Input::random(counted, length, still_limits())).expect_err("movie");
    assert!(
        matches!(error, OpenError::Unrecognized { .. }),
        "{} {error}",
        path.display()
    );
    assert!(!error.streams_published());
    let taken = reads.load(Ordering::Relaxed);
    assert!(
        taken <= 12,
        "{} read {taken} bytes of {length}",
        path.display()
    );
}

#[derive(Debug)]
struct Counted<R> {
    inner: R,
    reads: std::sync::Arc<AtomicU64>,
}

impl<R: Read> Read for Counted<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let amount = self.inner.read(buf)?;
        self.reads
            .fetch_add(u64::try_from(amount).unwrap_or(0), Ordering::Relaxed);
        Ok(amount)
    }
}

impl<R: Seek> Seek for Counted<R> {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(pos)
    }
}
