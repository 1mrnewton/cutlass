use std::fs::{self, File};
use std::sync::atomic::{AtomicU64, Ordering};

use cutlass_codec::ContainerId;
use cutlass_container::{Demux, OpenError, ProbeResult};
use cutlass_engine::decoded_picture;
use cutlass_isobmff::IsoBmffFile;
use cutlass_model::{Draft, MaterialVideo, Micros, Segment, TimeRange};
use cutlass_store::find_relight;

use super::*;
use crate::Compute;

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "cutlass-relight-job-{}-{}",
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

#[test]
fn a_clip_is_read_into_a_grey_depth_video_its_stretch_finds() {
    let (temp, draft) = project();
    let ask = cutlass_eval::relight_ask(&draft, Some(&temp.0), &"v".into()).unwrap();
    let mut heard = Vec::new();
    let file = relight_clip(
        &readers(),
        &encoders(),
        &model(),
        &ask,
        &temp.0,
        &mut |done| {
            heard.push(done);
            true
        },
    )
    .unwrap();
    assert_eq!(heard, vec![1.0 / 3.0, 2.0 / 3.0, 1.0]);
    assert!(file.parent().unwrap().ends_with("Resources/video_relight"));
    assert_eq!(
        find_relight(&temp.0, &ask.key(), &ask.range),
        Some((file.clone(), ask.range.clone()))
    );
    let mut media = readers().open(&file).unwrap();
    assert_eq!(media.info().duration, Some(Micros(300_000)));
    let held = media
        .decode_at(Micros(0), &TimeRange::new(0, 300_000))
        .unwrap();
    let Ok(cutlass_picture::Picture::Cpu(picture)) = decoded_picture(&held.frame) else {
        panic!("not in memory");
    };
    // Read at the model's short side, whole patches, even.
    let (pw, ph) = DepthModel::grid_at(W, H, RELIGHT_SIDE);
    assert_eq!(
        (picture.width(), picture.height()),
        ((pw * 14) & !1, (ph * 14) & !1)
    );
    assert!(
        picture
            .rgba()
            .chunks(4)
            .all(|p| p[0].abs_diff(p[1]) <= 3 && p[1].abs_diff(p[2]) <= 3),
        "grey"
    );
}

#[test]
fn a_stopped_job_or_a_missing_file_or_encoder_leaves_no_depth() {
    let (temp, draft) = project();
    let ask = cutlass_eval::relight_ask(&draft, Some(&temp.0), &"v".into()).unwrap();
    let model = model();
    let stopped = relight_clip(
        &readers(),
        &encoders(),
        &model,
        &ask,
        &temp.0,
        &mut |done| done < 0.5,
    );
    assert_eq!(stopped, Err(EnhanceJobError::Stopped));
    let no_encoder = relight_clip(
        &readers(),
        &cutlass_encode::EncodeFactory::new(),
        &model,
        &ask,
        &temp.0,
        &mut |_| true,
    );
    assert!(matches!(no_encoder, Err(EnhanceJobError::Movie(_))));
    assert_eq!(find_relight(&temp.0, &ask.key(), &ask.range), None);
    let folder = temp.0.join("Resources/video_relight");
    assert_eq!(fs::read_dir(folder).unwrap().count(), 0);
    let gone = RelightAsk {
        source: temp.0.join("gone.mp4"),
        ..ask
    };
    assert!(matches!(
        relight_clip(&readers(), &encoders(), &model, &gone, &temp.0, &mut |_| {
            true
        }),
        Err(EnhanceJobError::Clip(_))
    ));
}

#[test]
fn each_frame_is_spread_between_its_tails_and_a_flat_or_broken_one_stays_calm() {
    let values: Vec<f32> = (0..100).map(|i| i as f32).collect();
    assert_eq!(tails(&values), (2.0, 97.0));
    let mut broken = values.clone();
    broken.extend([f32::NAN, f32::INFINITY]);
    assert_eq!(tails(&broken), (2.0, 97.0));
    let (lo, hi) = tails(&[5.0; 50]);
    assert!(lo == 5.0 && hi > lo);
    assert_eq!(tails(&[]), (0.0, 1.0));
    let map = DepthMap {
        width: 3,
        height: 1,
        values: vec![0.0, 10.0, f32::NAN],
    };
    assert_eq!(
        grey(&map, (2, 1), (0.0, 10.0)),
        vec![0, 0, 0, 255, 255, 255, 255, 255]
    );
    let flat = DepthMap {
        width: 2,
        height: 1,
        values: vec![5.0, 5.0],
    };
    assert!(
        grey(&flat, (2, 1), tails(&flat.values))
            .iter()
            .all(|b| *b == 0 || *b == 255)
    );
}
