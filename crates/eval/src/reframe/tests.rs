use std::fs::{self, File};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use cutlass_container::{Demux, Input, OpenError, Probe, ProbeResult};
use cutlass_engine::ReaderFactory;
use cutlass_model::{Algorithm, MaterialVideo, TrackType, VideoAlgorithm};
use cutlass_motion::Subject;
use cutlass_renderer_software::SoftwareDevice;
use cutlass_still::{StillDemux, StillProbe, still_limits};
use cutlass_store::{SubjectPath, save_subject_path};

use super::*;
use crate::{OpenClips, Snapshot, frame_in};

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "cutlass-eval-reframe-{}-{}",
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

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../engine/fixtures")
        .join(name)
}

fn open_still(path: &Path) -> Result<Box<dyn Demux>, OpenError> {
    let file = File::open(path).map_err(|error| OpenError::Source {
        message: error.to_string(),
    })?;
    let len = file
        .metadata()
        .map_err(|error| OpenError::Source {
            message: error.to_string(),
        })?
        .len();
    Ok(Box::new(StillDemux::open(Input::random(
        file,
        len,
        still_limits(),
    ))?))
}

fn stills() -> ReaderFactory {
    let mut factory = ReaderFactory::new();
    factory.register_container(
        |header: &[u8]| -> ProbeResult { StillProbe.probe(header) },
        open_still,
    );
    cutlass_png::register_decoders(factory.decoders_mut());
    cutlass_jpeg::register_decoders(factory.decoders_mut());
    factory
}

/// A project holding `pattern.png` as `walk.png`, and a draft whose one
/// clip shows a second of it from 1 s on a 16 by 16 canvas, cut to `ratio`
/// when there is one.
fn project(ratio: Option<&str>) -> (Temp, Draft) {
    let temp = Temp::new();
    fs::copy(fixture("pattern.png"), temp.0.join("walk.png")).unwrap();
    let mut draft = Draft::empty("D");
    draft.fps = 10.0;
    draft.canvas_config.width = 16;
    draft.canvas_config.height = 16;
    draft.materials.videos.push(MaterialVideo {
        id: "m".into(),
        kind: "video".into(),
        path: "walk.png".into(),
        width: 16,
        height: 16,
        duration: Micros(5_000_000),
        video_algorithm: ratio.map(|ratio| VideoAlgorithm {
            algorithms: vec![Algorithm {
                kind: Algorithm::AUTO_REFRAME.into(),
                level: 1,
                ratio: ratio.into(),
                ..Algorithm::default()
            }],
            ..VideoAlgorithm::default()
        }),
        ..MaterialVideo::default()
    });
    let mut segment = Segment::new("v", "m", TimeRange::new(2_000_000, 1_000_000));
    segment.source_range = Some(TimeRange::new(1_000_000, 1_000_000));
    draft.tracks[0].segments.push(segment);
    (temp, draft)
}

fn draw(draft: &Draft, root: &Path, at: i64) -> Vec<u8> {
    let mut clips = OpenClips::new();
    frame_in(
        &Snapshot::new(draft),
        Micros(at),
        &mut SoftwareDevice::new(),
        &stills(),
        &mut clips,
        0,
        Some(root),
    )
    .unwrap()
    .rgba
}

/// Keeps a track with the subject at `x` across in every sample, as the
/// job would.
fn keep_track(temp: &Temp, draft: &Draft, x: f64) {
    let ask = reframe_ask(draft, Some(&temp.0), &"v".into()).unwrap();
    let samples = (ask.range.duration.0 as f64 * ask.rate / 1e6).ceil() as usize;
    let path = SubjectPath {
        width: 16,
        height: 16,
        subjects: vec![
            Subject {
                x,
                y: 0.5,
                weight: 1.0
            };
            samples
        ],
    };
    save_subject_path(&temp.0, &ask.key(), &path).unwrap();
}

#[test]
fn the_ask_is_the_whole_file_at_the_sample_rate() {
    let (temp, draft) = project(None);
    let ask = reframe_ask(&draft, Some(&temp.0), &"v".into()).unwrap();
    assert_eq!(ask.source, temp.0.join("walk.png").canonicalize().unwrap());
    assert_eq!(ask.range, TimeRange::new(0, 5_000_000));
    assert_eq!(ask.rate, REFRAME_RATE);
    assert_eq!(ask.key().model, REFRAME_MODEL);
    assert_eq!(ask.frames().range, ask.range);
    let mut text = draft.clone();
    text.tracks[0].kind = TrackType::Text;
    let Err(EnhanceError::Clip { reason }) = reframe_ask(&text, Some(&temp.0), &"v".into()) else {
        panic!("a graphic was reframed");
    };
    assert!(reason.contains("reframed"), "{reason}");
    let mut escaping = draft;
    escaping.materials.videos[0].path = "../../etc/passwd".into();
    assert!(reframe_ask(&escaping, Some(&temp.0), &"v".into()).is_err());
}

#[test]
fn a_reframed_clip_is_cut_around_its_subject_once_there_is_a_track() {
    let (temp, draft) = project(Some("9:16"));
    let (_, plain) = project(None);
    let own = draw(&plain, &temp.0, 2_500_000);
    let segment = draft.tracks[0].segments[0].clone();
    let mut clips = OpenClips::new();
    clips.set_project(Some(&temp.0));
    assert_eq!(
        crop_at(&mut clips, &draft, &segment, Micros(2_500_000), (16, 16)),
        None,
        "no track yet"
    );
    assert_eq!(draw(&draft, &temp.0, 2_500_000), own);
    keep_track(&temp, &draft, 0.95);
    let mut clips = OpenClips::new();
    clips.set_project(Some(&temp.0));
    let crop = crop_at(&mut clips, &draft, &segment, Micros(2_500_000), (16, 16)).unwrap();
    // A 9:16 window in a square frame: full height, as far right as it goes.
    assert!((crop.upper_right_x - 1.0).abs() < 1e-6, "{crop:?}");
    assert!(
        (crop.upper_left_x - (1.0 - 9.0 / 16.0)).abs() < 1e-6,
        "{crop:?}"
    );
    assert_eq!((crop.upper_left_y, crop.lower_left_y), (0.0, 1.0));
    assert_ne!(draw(&draft, &temp.0, 2_500_000), own);
    // No project folder, or off, keeps the clip's own crop.
    let mut loose = OpenClips::new();
    assert_eq!(
        crop_at(&mut loose, &draft, &segment, Micros(2_500_000), (16, 16)),
        None
    );
    assert_eq!(
        crop_at(
            &mut clips,
            &plain,
            &plain.tracks[0].segments[0],
            Micros(2_500_000),
            (16, 16)
        ),
        None
    );
}

#[test]
fn another_shape_is_planned_again_from_the_same_track() {
    let (temp, draft) = project(Some("9:16"));
    keep_track(&temp, &draft, 0.05);
    let segment = draft.tracks[0].segments[0].clone();
    let mut clips = OpenClips::new();
    clips.set_project(Some(&temp.0));
    let tall = crop_at(&mut clips, &draft, &segment, Micros(2_200_000), (16, 16)).unwrap();
    assert!(tall.upper_left_x.abs() < 1e-6, "{tall:?}");
    let mut wide = draft.clone();
    wide.materials.videos[0]
        .video_algorithm
        .as_mut()
        .unwrap()
        .algorithms[0]
        .ratio = "16:9".into();
    let cut = crop_at(&mut clips, &wide, &segment, Micros(2_200_000), (16, 16)).unwrap();
    assert_eq!((cut.upper_left_x, cut.upper_right_x), (0.0, 1.0));
    assert!(
        (cut.lower_left_y - cut.upper_left_y - 9.0 / 16.0).abs() < 1e-6,
        "{cut:?}"
    );
    let square = crop_at(&mut clips, &wide, &segment, Micros(2_200_000), (16, 9));
    assert!(square.is_some(), "the canvas is part of what is planned");
}

#[test]
fn a_window_between_samples_lies_between_them() {
    let reframed = Reframed {
        ask: ReframeAsk {
            source: "a".into(),
            range: TimeRange::new(1_000_000, 1_000_000),
            rate: 5.0,
        },
        reframe: Reframe {
            ratio: "9:16".into(),
            stability: 2,
            speed: 2,
        },
        canvas: (16, 16),
        size: (0.5, 1.0),
        middles: vec![(0.3, 0.5), (0.5, 0.5)],
    };
    assert_eq!(reframed.at(Micros(1_100_000)), Some((0.4, 0.5)));
    assert_eq!(reframed.at(Micros(0)), Some((0.3, 0.5)));
    assert_eq!(reframed.at(Micros(9_000_000)), Some((0.5, 0.5)));
    let empty = Reframed {
        middles: Vec::new(),
        ..reframed
    };
    assert_eq!(empty.at(Micros(1_000_000)), None);
}
