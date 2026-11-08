use std::path::PathBuf;

use cutlass_codec::ContainerId;
use cutlass_container::{Demux, OpenError, ProbeResult};
use cutlass_engine::decoded_picture;
use cutlass_eval::ReaderFactory;
use cutlass_isobmff::IsoBmffFile;
use cutlass_model::{Micros, TimeRange};

use super::*;

struct Temp(PathBuf);

impl Temp {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("cutlass-pictures-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn encoders() -> EncodeFactory {
    let mut encoders = EncodeFactory::new();
    cutlass_h264::register_encoders(&mut encoders);
    encoders
}

fn probe(header: &[u8]) -> ProbeResult {
    if header.get(4..8) == Some(b"ftyp") {
        ProbeResult::Accept {
            family: ContainerId::Isobmff,
            examined: 8,
        }
    } else {
        ProbeResult::Refuse { examined: 8 }
    }
}

fn open(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let file = File::open(path).map_err(|error| OpenError::Source {
        message: error.to_string(),
    })?;
    Ok(Box::new(IsoBmffFile::open(file)?))
}

fn readers() -> ReaderFactory {
    let mut readers = ReaderFactory::new();
    readers.register_container(probe, open);
    cutlass_h264::register_decoders(readers.decoders_mut());
    readers
}

const RATE: FrameRate = FrameRate {
    numerator: 10,
    denominator: 1,
};

fn movie(frames: u64) -> PictureMovie {
    PictureMovie {
        width: 64,
        height: 48,
        frame_rate: RATE,
        bitrate: 4_000_000,
        frames,
    }
}

/// Picture `k`: flat grey, lighter each frame.
fn grey(k: u64) -> Vec<u8> {
    let level = 40 + 50 * k as u8;
    [level, level, level, 255].repeat(64 * 48)
}

#[test]
fn pictures_handed_in_come_back_in_order_at_their_size_and_rate() {
    let temp = Temp::new("round-trip");
    let path = temp.0.join("enhanced.mp4");
    let mut heard = Vec::new();
    write_pictures(
        &encoders(),
        &movie(4),
        &path,
        &mut |k| Ok(grey(k)),
        &mut |done, all| {
            heard.push((done, all));
            true
        },
    )
    .unwrap();
    assert_eq!(heard, vec![(1, 4), (2, 4), (3, 4), (4, 4)]);
    assert!(!part_path(&path).unwrap().exists());
    let readers = readers();
    let mut media = readers.open(&path).unwrap();
    assert_eq!(media.info().duration, Some(Micros(400_000)));
    let range = TimeRange::new(0, 400_000);
    for k in 0..4_u64 {
        let held = media.decode_at(Micros(k as i64 * 100_000), &range).unwrap();
        let Ok(cutlass_picture::Picture::Cpu(picture)) = decoded_picture(&held.frame) else {
            panic!("frame {k} is not in memory");
        };
        assert_eq!((picture.width(), picture.height()), (64, 48));
        let want = i32::from(grey(k)[0]);
        let got = i32::from(picture.rgba()[4 * (24 * 64 + 32)]);
        assert!((got - want).abs() <= 3, "frame {k}: {got} for {want}");
    }
}

#[test]
fn a_stopped_or_failed_movie_leaves_nothing_behind() {
    let temp = Temp::new("stopped");
    let path = temp.0.join("enhanced.mp4");
    let stopped = write_pictures(
        &encoders(),
        &movie(5),
        &path,
        &mut |k| Ok(grey(k)),
        &mut |done, _| done < 2,
    );
    assert_eq!(stopped, Err(ExportError::Cancelled));
    let failed = write_pictures(
        &encoders(),
        &movie(5),
        &path,
        &mut |k| {
            if k == 3 {
                Err(ExportError::Picture("the model failed".into()))
            } else {
                Ok(grey(k))
            }
        },
        &mut |_, _| true,
    );
    assert!(matches!(failed, Err(ExportError::Picture(_))));
    let short = write_pictures(
        &encoders(),
        &movie(2),
        &path,
        &mut |_| Ok(vec![0; 10]),
        &mut |_, _| true,
    );
    assert!(matches!(short, Err(ExportError::Picture(_))));
    assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
}

#[test]
fn a_shape_h264_cannot_hold_is_refused_before_any_picture() {
    let temp = Temp::new("refused");
    let path = temp.0.join("enhanced.mp4");
    let mut asked = 0;
    for bad in [
        PictureMovie {
            width: 63,
            ..movie(1)
        },
        PictureMovie {
            height: 0,
            ..movie(1)
        },
        PictureMovie {
            width: MAX_EDGE + 2,
            ..movie(1)
        },
        PictureMovie {
            frame_rate: FrameRate {
                numerator: 0,
                denominator: 1,
            },
            ..movie(1)
        },
        movie(0),
        movie(MAX_FRAMES + 1),
    ] {
        assert!(
            write_pictures(
                &encoders(),
                &bad,
                &path,
                &mut |k| {
                    asked += 1;
                    Ok(grey(k))
                },
                &mut |_, _| true
            )
            .is_err(),
            "{bad:?}"
        );
    }
    assert_eq!(asked, 0);
    assert!(
        write_pictures(
            &EncodeFactory::new(),
            &movie(1),
            &path,
            &mut |k| Ok(grey(k)),
            &mut |_, _| true
        )
        .is_err(),
        "no H.264 encoder"
    );
    assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
}
