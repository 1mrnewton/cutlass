use std::fs::{self, File};
use std::sync::atomic::{AtomicU64, Ordering};

use cutlass_codec::ContainerId;
use cutlass_container::{Demux, OpenError, ProbeResult};
use cutlass_export::{PictureMovie, write_pictures};
use cutlass_isobmff::IsoBmffFile;
use cutlass_model::{Draft, FrameRate, MaterialVideo, Micros, Segment, TimeRange};
use std::path::{Path, PathBuf};

use super::*;
use crate::Compute;

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "cutlass-reframe-job-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
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

fn encoders() -> cutlass_encode::EncodeFactory {
    let mut encoders = cutlass_encode::EncodeFactory::new();
    cutlass_h264::register_encoders(&mut encoders);
    encoders
}

const W: u32 = 32;
const H: u32 = 24;

/// A project holding a ten-frame 32 by 24 movie at 10 a second, and a
/// draft whose clip shows 0.2 s to 0.5 s of it.
fn project() -> (Temp, Draft) {
    let temp = Temp::new();
    let movie = PictureMovie {
        width: W,
        height: H,
        frame_rate: FrameRate {
            numerator: 10,
            denominator: 1,
        },
        bitrate: 2_000_000,
        frames: 10,
    };
    write_pictures(
        &encoders(),
        &movie,
        &temp.0.join("walk.mp4"),
        &mut |k| {
            Ok((0..W * H)
                .flat_map(|i| [(i % W * 8) as u8, 20 * k as u8, 100, 255])
                .collect())
        },
        &mut |_, _| true,
    )
    .unwrap();
    let mut draft = Draft::empty("D");
    draft.fps = 10.0;
    draft.materials.videos.push(MaterialVideo {
        id: "m".into(),
        kind: "video".into(),
        path: "walk.mp4".into(),
        width: W as i32,
        height: H as i32,
        duration: Micros(1_000_000),
        ..MaterialVideo::default()
    });
    let mut segment = Segment::new("v", "m", TimeRange::new(0, 300_000));
    segment.source_range = Some(TimeRange::new(200_000, 300_000));
    draft.tracks[0].segments.push(segment);
    (temp, draft)
}

fn model() -> DepthModel {
    DepthModel::untrained(Compute::Cpu).unwrap()
}

/// A depth `w` by `h`: a floor nearing the camera row by row, and where
/// `post` says, a post standing out of it.
fn floor(w: u32, h: u32, post: Option<std::ops::Range<u32>>) -> DepthMap {
    let values = (0..h)
        .flat_map(|y| {
            let post = post.clone();
            (0..w).map(move |x| {
                let ground = y as f32 / h as f32;
                match &post {
                    Some(columns) if columns.contains(&x) && y > h / 4 => ground + 0.6,
                    _ => ground,
                }
            })
        })
        .collect();
    DepthMap {
        width: w,
        height: h,
        values,
    }
}

#[test]
fn what_stands_out_nearest_is_the_subject_not_the_floor() {
    let found = subject_of(&floor(64, 36, Some(44..52)));
    assert!(found.x > 0.65 && found.x < 0.85, "{found:?}");
    assert!(found.y > 0.4, "{found:?}");
    assert!(found.weight > 0.5, "{found:?}");
    let left = subject_of(&floor(64, 36, Some(4..12)));
    assert!(left.x < 0.3, "{left:?}");
    // A bare floor or a flat wall has nothing standing out.
    assert_eq!(subject_of(&floor(64, 36, None)).weight, 0.0);
    let wall = DepthMap {
        width: 8,
        height: 8,
        values: vec![3.0; 64],
    };
    assert_eq!(subject_of(&wall), Subject::NONE);
    let none = DepthMap {
        width: 0,
        height: 0,
        values: Vec::new(),
    };
    assert_eq!(subject_of(&none), Subject::NONE);
    let short = DepthMap {
        width: 8,
        height: 8,
        values: vec![1.0; 10],
    };
    assert_eq!(subject_of(&short), Subject::NONE);
    let mut broken = floor(64, 36, Some(44..52));
    broken.values[5] = f32::NAN;
    let found = subject_of(&broken);
    assert!(found.x.is_finite() && found.x > 0.65, "{found:?}");
}

#[test]
fn every_sample_of_the_whole_file_is_searched() {
    let (temp, draft) = project();
    let ask = cutlass_eval::reframe_ask(&draft, Some(&temp.0), &"v".into()).unwrap();
    let mut heard = Vec::new();
    let path = find_subjects(&readers(), &model(), &ask, &mut |done| {
        heard.push(done);
        true
    })
    .unwrap();
    assert_eq!(path.subjects.len(), 5, "a second at 5 a second");
    assert_eq!((path.width, path.height), (W, H));
    assert_eq!(heard.last(), Some(&1.0));
    assert!(
        path.subjects
            .iter()
            .all(|s| s.x.is_finite() && (0.0..=1.0).contains(&s.weight))
    );
}

#[test]
fn a_stopped_job_or_a_missing_file_finds_nothing() {
    let (temp, draft) = project();
    let ask = cutlass_eval::reframe_ask(&draft, Some(&temp.0), &"v".into()).unwrap();
    let model = model();
    assert_eq!(
        find_subjects(&readers(), &model, &ask, &mut |done| done < 0.5),
        Err(EnhanceJobError::Stopped)
    );
    let gone = ReframeAsk {
        source: temp.0.join("gone.mp4"),
        ..ask
    };
    assert!(matches!(
        find_subjects(&readers(), &model, &gone, &mut |_| true),
        Err(EnhanceJobError::Clip(_))
    ));
}
