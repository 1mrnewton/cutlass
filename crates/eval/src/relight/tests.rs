use std::fs::{self, File};
use std::sync::atomic::{AtomicU64, Ordering};

use cutlass_container::{Demux, Input, OpenError, Probe, ProbeResult};
use cutlass_engine::ReaderFactory;
use cutlass_model::{Algorithm, MaterialVideo, TrackType, VideoAlgorithm};
use cutlass_renderer_software::SoftwareDevice;
use cutlass_still::{StillDemux, StillProbe, still_limits};
use cutlass_store::relight_file;

use super::*;
use crate::{OpenClips, Snapshot, frame_in};

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "cutlass-eval-relight-{}-{}",
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
/// clip shows a second of it from 1 s on a 16 by 16 canvas, lit by
/// `preset` at `strength` when there is one.
fn project(preset: Option<(&str, f64)>) -> (Temp, Draft) {
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
        video_algorithm: preset.map(|(preset, strength)| VideoAlgorithm {
            algorithms: vec![Algorithm {
                kind: Algorithm::RELIGHT.into(),
                level: 1,
                preset: preset.into(),
                strength: Some(strength),
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

/// Writes the job's depth video for `draft`'s clip: here a still JPEG,
/// whose red varies across it.
fn keep_depth(temp: &Temp, draft: &Draft) -> PathBuf {
    let ask = relight_ask(draft, Some(&temp.0), &"v".into()).unwrap();
    let file = relight_file(&temp.0, &ask.key(), &TimeRange::new(500_000, 2_500_000)).unwrap();
    fs::copy(fixture("pattern.jpg"), &file).unwrap();
    file
}

#[test]
fn the_ask_is_the_stretch_the_clip_shows_at_the_project_rate() {
    let (temp, draft) = project(None);
    let ask = relight_ask(&draft, Some(&temp.0), &"v".into()).unwrap();
    assert_eq!(ask.source, temp.0.join("walk.png").canonicalize().unwrap());
    assert_eq!(ask.range, TimeRange::new(1_000_000, 1_000_000));
    assert_eq!(ask.rate, 10.0);
    let key = ask.key();
    assert_eq!((key.model.as_str(), key.rate), (RELIGHT_MODEL, 10.0));
    let frames = ask.frames();
    assert_eq!(
        (frames.level, frames.denoise, frames.interpolate),
        (0, 0, 0)
    );
    assert_eq!(frames.range, ask.range);
}

#[test]
fn a_graphic_no_stretch_or_a_file_outside_is_refused() {
    let (temp, draft) = project(None);
    let root = Some(temp.0.as_path());
    assert!(relight_ask(&draft, root, &"nope".into()).is_err());
    let mut text = draft.clone();
    text.tracks[0].kind = TrackType::Text;
    let Err(EnhanceError::Clip { reason }) = relight_ask(&text, root, &"v".into()) else {
        panic!("a graphic was relit");
    };
    assert!(reason.contains("relit"), "{reason}");
    let mut none = draft.clone();
    none.tracks[0].segments[0].source_range = Some(TimeRange::new(0, 0));
    assert!(relight_ask(&none, root, &"v".into()).is_err());
    let mut escaping = draft;
    escaping.materials.videos[0].path = "../../etc/passwd".into();
    assert!(relight_ask(&escaping, root, &"v".into()).is_err());
}

#[test]
fn a_relit_clip_is_lit_from_its_depth_once_there_is_one() {
    let (temp, draft) = project(Some(("warm_key", 1.0)));
    let (_, plain) = project(None);
    let own = draw(&plain, &temp.0, 2_500_000);
    assert_eq!(draw(&draft, &temp.0, 2_500_000), own, "no depth yet: unlit");
    assert_eq!(relight_depth(&draft, Some(&temp.0), &"v".into()), None);
    let file = keep_depth(&temp, &draft);
    assert_eq!(
        relight_depth(&draft, Some(&temp.0), &"v".into()),
        Some((file, Micros(500_000)))
    );
    let lit = draw(&draft, &temp.0, 2_500_000);
    assert_ne!(lit, own);
    // No strength, no preset, or no project folder draws it unlit.
    let (_, none) = project(Some(("warm_key", 0.0)));
    assert_eq!(draw(&none, &temp.0, 2_500_000), own);
    assert_eq!(relight_depth(&plain, Some(&temp.0), &"v".into()), None);
    assert_eq!(relight_depth(&draft, None, &"v".into()), None);
    // A preset this build does not know draws unlit too.
    let (_, unknown) = project(Some(("from_the_future", 1.0)));
    assert_eq!(draw(&unknown, &temp.0, 2_500_000), own);
}

#[test]
fn a_moving_light_is_timed_from_the_clip_start() {
    let (_, draft) = project(Some(("rainbow_spots", 1.0)));
    let segment = &draft.tracks[0].segments[0];
    let first = light_at(&draft, segment, Micros(2_000_000)).unwrap();
    let relight = draft.materials.videos[0].relight().unwrap();
    assert_eq!(first, lighting(&relight, 0.0).unwrap());
    assert_eq!(
        light_at(&draft, segment, Micros(2_400_000)).unwrap(),
        lighting(&relight, 0.4).unwrap()
    );
    let (_, plain) = project(None);
    assert_eq!(
        light_at(&plain, &plain.tracks[0].segments[0], Micros(2_000_000)),
        None
    );
}

#[test]
fn a_depth_is_stretched_to_the_picture_with_its_centres_matched() {
    let depth = CpuBytes::new(2, 1, vec![0, 0, 0, 255, 200, 0, 0, 255]).unwrap();
    assert_eq!(stretched_grey(&depth, 2, 1).unwrap(), vec![0, 200]);
    assert_eq!(
        stretched_grey(&depth, 4, 2).unwrap(),
        vec![0, 50, 150, 200, 0, 50, 150, 200]
    );
    assert_eq!(stretched_grey(&depth, 1, 1).unwrap(), vec![100]);
    assert!(stretched_grey(&depth, 0, 3).is_none());
    // Lighting a picture from a depth of another size keeps its size.
    let picture = CpuBytes::new(4, 2, [90, 90, 90, 255].repeat(8)).unwrap();
    let relight = cutlass_model::Relight {
        preset: "warm_key".into(),
        strength: 1.0,
        color: String::new(),
        angle: cutlass_model::Relight::DEFAULT_ANGLE,
    };
    let lit = relit(picture.clone(), &depth, &lighting(&relight, 0.0).unwrap());
    assert_eq!((lit.width(), lit.height()), (4, 2));
    assert_ne!(lit.rgba(), picture.rgba());
}

#[test]
fn a_keyed_strength_lights_the_clip_as_keyed_and_none_at_zero() {
    let (_, mut draft) = project(Some(("rainbow_spots", 1.0)));
    let segment = &mut draft.tracks[0].segments[0];
    let strength = cutlass_model::RELIGHT_STRENGTH_KEYS[0];
    cutlass_model::put_key(segment, strength, Micros(0), 0.0);
    cutlass_model::put_key(segment, strength, Micros(1_000_000), 1.0);
    let segment = &draft.tracks[0].segments[0];
    // Off where the key is 0, and half way along at half strength.
    assert_eq!(light_at(&draft, segment, Micros(2_000_000)), None);
    let half = light_at(&draft, segment, Micros(2_500_000)).unwrap();
    assert!((half.strength - 0.5).abs() < 1e-6);
}
