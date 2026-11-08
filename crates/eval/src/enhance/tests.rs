use std::fs::{self, File};
use std::sync::atomic::{AtomicU64, Ordering};

use cutlass_container::{Demux, Input, OpenError, Probe, ProbeResult};
use cutlass_model::{Algorithm, MaterialVideo, Segment, TrackType, VideoAlgorithm};
use cutlass_renderer_software::SoftwareDevice;
use cutlass_still::{StillDemux, StillProbe, still_limits};
use cutlass_store::enhanced_file;

use super::*;
use crate::{OpenClips, Snapshot, frame_in};

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "cutlass-eval-enhance-{}-{}",
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
/// clip shows a second of it from 1 s, on a 16 by 16 canvas.
fn project(level: Option<i64>) -> (Temp, Draft) {
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
        video_algorithm: level.map(|level| VideoAlgorithm {
            algorithms: vec![Algorithm {
                kind: Algorithm::QUALITY_ENHANCE.into(),
                level,
                ..Algorithm::default()
            }],
            ..VideoAlgorithm::default()
        }),
        ..MaterialVideo::default()
    });
    let mut segment = Segment::new("v", "m", TimeRange::new(0, 1_000_000));
    segment.source_range = Some(TimeRange::new(1_000_000, 1_000_000));
    draft.tracks[0].segments.push(segment);
    (temp, draft)
}

#[test]
fn the_ask_is_the_stretch_the_clip_shows_at_the_project_rate() {
    let (temp, draft) = project(None);
    let ask = enhance_ask(&draft, Some(&temp.0), &"v".into(), Algorithm::HD).unwrap();
    assert_eq!(ask.source, temp.0.join("walk.png").canonicalize().unwrap());
    assert_eq!(ask.range, TimeRange::new(1_000_000, 1_000_000));
    assert_eq!(ask.rate, 10.0);
    assert_eq!(ask.level, Algorithm::HD);
    let key = ask.key();
    assert_eq!((key.level, key.rate), (Algorithm::HD, 10.0));
    assert_eq!(key.model, ENHANCE_MODEL);
}

#[test]
fn an_off_level_a_graphic_no_stretch_or_a_file_outside_is_refused() {
    let (temp, draft) = project(None);
    let root = Some(temp.0.as_path());
    assert!(enhance_ask(&draft, root, &"v".into(), 0).is_err());
    assert!(enhance_ask(&draft, root, &"nope".into(), 1).is_err());
    let mut text = draft.clone();
    text.tracks[0].kind = TrackType::Text;
    let Err(EnhanceError::Clip { reason }) = enhance_ask(&text, root, &"v".into(), 1) else {
        panic!("a graphic was enhanced");
    };
    assert!(reason.contains("enhanced"), "{reason}");
    let mut none = draft.clone();
    none.tracks[0].segments[0].source_range = Some(TimeRange::new(0, 0));
    assert!(enhance_ask(&none, root, &"v".into(), 1).is_err());
    let mut escaping = draft;
    escaping.materials.videos[0].path = "../../etc/passwd".into();
    assert!(enhance_ask(&escaping, root, &"v".into(), 1).is_err());
}

#[test]
fn a_clip_draws_from_its_copy_once_there_is_one() {
    let (temp, draft) = project(Some(Algorithm::HD));
    let draw = |draft: &Draft| {
        let mut clips = OpenClips::new();
        frame_in(
            &Snapshot::new(draft),
            Micros(500_000),
            &mut SoftwareDevice::new(),
            &stills(),
            &mut clips,
            0,
            Some(&temp.0),
        )
        .unwrap()
        .rgba
    };
    let own = draw(&draft);
    assert_eq!(enhanced_copy(&draft, Some(&temp.0), &"v".into()), None);
    // The job's copy: here the same pattern as a JPEG, which decodes to
    // other pixels, holding 0.5 s to 3 s of the file.
    let ask = enhance_ask(&draft, Some(&temp.0), &"v".into(), Algorithm::HD).unwrap();
    let file = enhanced_file(&temp.0, &ask.key(), &TimeRange::new(500_000, 2_500_000)).unwrap();
    fs::copy(fixture("pattern.jpg"), &file).unwrap();
    assert_eq!(
        enhanced_copy(&draft, Some(&temp.0), &"v".into()),
        Some((file, Micros(500_000)))
    );
    let enhanced = draw(&draft);
    assert_ne!(enhanced, own);
    let mut jpeg = draft.clone();
    fs::copy(fixture("pattern.jpg"), temp.0.join("walk.jpg")).unwrap();
    jpeg.materials.videos[0].path = "walk.jpg".into();
    jpeg.materials.videos[0].video_algorithm = None;
    assert_eq!(enhanced, draw(&jpeg), "drawn from the copy");
    // Off, at another level, or with no project folder, the clip draws its own file.
    for level in [0, Algorithm::UHD] {
        let mut other = draft.clone();
        other.materials.videos[0]
            .video_algorithm
            .as_mut()
            .unwrap()
            .algorithms[0]
            .level = level;
        assert_eq!(draw(&other), own, "level {level}");
    }
    assert_eq!(enhanced_copy(&draft, None, &"v".into()), None);
}

#[test]
fn a_frame_is_turned_upright() {
    // Two pixels side by side, red then blue.
    let picture = CpuBytes::new(2, 1, vec![255, 0, 0, 255, 0, 0, 255, 255]).unwrap();
    let red = [255, 0, 0, 255];
    let blue = [0, 0, 255, 255];
    let turned = |display| upright_rgba(&picture, display).unwrap();
    let zero = turned(DisplayQuarter::Zero);
    assert_eq!(
        (zero.width(), zero.rgba()),
        (2, [red, blue].concat().as_slice())
    );
    let quarter = turned(DisplayQuarter::Clockwise90);
    assert_eq!((quarter.width(), quarter.height()), (1, 2));
    assert_eq!(quarter.rgba(), [red, blue].concat().as_slice());
    let half = turned(DisplayQuarter::Half);
    assert_eq!(half.rgba(), [blue, red].concat().as_slice());
    let three = turned(DisplayQuarter::Clockwise270);
    assert_eq!(three.rgba(), [blue, red].concat().as_slice());
}

#[test]
fn a_still_or_missing_file_is_not_read_as_frames() {
    let (temp, draft) = project(None);
    let ask = enhance_ask(&draft, Some(&temp.0), &"v".into(), 1).unwrap();
    assert!(matches!(
        EnhanceFrames::open(&stills(), &ask),
        Err(EnhanceError::Clip { .. })
    ));
    let gone = EnhanceAsk {
        source: temp.0.join("gone.mp4"),
        ..ask
    };
    assert!(matches!(
        EnhanceFrames::open(&stills(), &gone),
        Err(EnhanceError::Frame { .. })
    ));
}
