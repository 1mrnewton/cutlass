use std::fs::{self, File};
use std::sync::atomic::{AtomicU64, Ordering};

use cutlass_codec::ContainerId;
use cutlass_container::{Demux, OpenError, ProbeResult};
use cutlass_engine::decoded_picture;
use cutlass_isobmff::IsoBmffFile;
use cutlass_model::{Draft, MaterialVideo, Micros, Segment, TimeRange};
use cutlass_store::find_enhanced;

use super::*;
use crate::{Compute, INTERPOLATE_WEIGHTS, MAX_NOISE, MIN_NOISE};

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "cutlass-enhance-job-{}-{}",
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

fn enhancing(enhancer: &Enhancer) -> CopyModels<'_> {
    CopyModels {
        enhancer: Some(enhancer),
        denoiser: None,
        interpolator: None,
    }
}

/// The copy's first frame and its size.
fn first_frame(file: &Path) -> (u32, u32, Vec<u8>) {
    let mut media = readers().open(file).unwrap();
    assert_eq!(media.info().duration, Some(Micros(300_000)));
    let held = media
        .decode_at(Micros(0), &TimeRange::new(0, 300_000))
        .unwrap();
    let Ok(cutlass_picture::Picture::Cpu(picture)) = decoded_picture(&held.frame) else {
        panic!("not in memory");
    };
    (picture.width(), picture.height(), picture.rgba().to_vec())
}

const W: u32 = 32;
const H: u32 = 24;

/// A project holding a ten-frame 32 by 24 movie at 10 a second whose
/// frames grow lighter, and a draft whose clip shows 0.2 s to 0.5 s of it.
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
        &mut |k| Ok([20 + 20 * k as u8, 60, 100, 255].repeat((W * H) as usize)),
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

#[test]
fn each_level_takes_its_short_side_and_a_large_clip_keeps_its_own() {
    assert_eq!(enhanced_size(640, 360, Algorithm::HD), (1920, 1080));
    assert_eq!(enhanced_size(360, 640, Algorithm::HD), (1080, 1920));
    assert_eq!(enhanced_size(640, 360, Algorithm::UHD), (3840, 2160));
    assert_eq!(enhanced_size(1920, 1080, Algorithm::HD), (1920, 1080));
    assert_eq!(enhanced_size(3840, 2160, Algorithm::HD), (3840, 2160));
    // Even, as H.264 codes it.
    assert_eq!(enhanced_size(101, 75, Algorithm::HD), (1454, 1080));
    assert_eq!(enhanced_size(1921, 1081, Algorithm::HD), (1920, 1080));
    // Only denoised: the clip's own size, even.
    assert_eq!(enhanced_size(640, 360, 0), (640, 360));
    assert_eq!(enhanced_size(641, 361, 0), (640, 360));
}

#[test]
fn each_noise_level_is_one_the_model_learned_and_strong_takes_out_more() {
    let (weak, strong) = (
        noise_for_level(Algorithm::WEAK),
        noise_for_level(Algorithm::STRONG),
    );
    assert!(weak < strong);
    for noise in [weak, strong] {
        assert!((MIN_NOISE..=MAX_NOISE).contains(&noise), "{noise}");
    }
}

#[test]
fn an_odd_last_row_and_column_are_left_off_a_denoised_frame() {
    let rgb: Vec<u8> = (0..3 * 3 * 3).map(|v| v as u8).collect();
    let out = cropped_rgba((3, 3), &rgb, (2, 2));
    assert_eq!(
        out,
        [0, 1, 2, 255, 3, 4, 5, 255, 9, 10, 11, 255, 12, 13, 14, 255]
    );
}

#[test]
fn a_clip_is_denoised_into_a_copy_at_its_own_size() {
    let (temp, draft) = project();
    let ask =
        cutlass_eval::enhance_ask(&draft, Some(&temp.0), &"v".into(), 0, Algorithm::STRONG, 0)
            .unwrap();
    let denoiser = Denoiser::untrained(Compute::Cpu).unwrap();
    let mut heard = Vec::new();
    let file = enhance_clip(
        &readers(),
        &encoders(),
        CopyModels {
            enhancer: None,
            denoiser: Some(&denoiser),
            interpolator: None,
        },
        &ask,
        &temp.0,
        &mut |done| {
            heard.push(done);
            true
        },
    )
    .unwrap();
    assert_eq!(heard, vec![1.0 / 3.0, 2.0 / 3.0, 1.0]);
    assert!(file.parent().unwrap().ends_with("Resources/video_denoise"));
    assert_eq!(
        find_enhanced(&temp.0, &ask.key(), &ask.range),
        Some((file.clone(), ask.range.clone()))
    );
    let (w, h, _) = first_frame(&file);
    assert_eq!((w, h), (W, H));
}

#[test]
fn a_clip_with_both_is_denoised_then_enhanced_into_one_copy() {
    let (temp, draft) = project();
    let ask = cutlass_eval::enhance_ask(
        &draft,
        Some(&temp.0),
        &"v".into(),
        Algorithm::HD,
        Algorithm::WEAK,
        0,
    )
    .unwrap();
    let enhancer = Enhancer::untrained(Compute::Cpu).unwrap();
    let denoiser = Denoiser::untrained(Compute::Cpu).unwrap();
    let both = CopyModels {
        enhancer: Some(&enhancer),
        denoiser: Some(&denoiser),
        interpolator: None,
    };
    let file = enhance_clip(&readers(), &encoders(), both, &ask, &temp.0, &mut |_| true).unwrap();
    assert!(file.parent().unwrap().ends_with("Resources/video_enhance"));
    let (w, h, together) = first_frame(&file);
    assert_eq!((w, h), (1440, 1080));
    // Not the clip enhanced alone: the denoiser changed what the enhancer saw.
    let alone =
        cutlass_eval::enhance_ask(&draft, Some(&temp.0), &"v".into(), Algorithm::HD, 0, 0).unwrap();
    let file = enhance_clip(
        &readers(),
        &encoders(),
        enhancing(&enhancer),
        &alone,
        &temp.0,
        &mut |_| true,
    )
    .unwrap();
    assert_ne!(first_frame(&file).2, together);
}

#[test]
fn a_level_whose_model_is_not_loaded_is_refused_and_leaves_no_copy() {
    let (temp, draft) = project();
    let enhancer = Enhancer::untrained(Compute::Cpu).unwrap();
    let ask = cutlass_eval::enhance_ask(
        &draft,
        Some(&temp.0),
        &"v".into(),
        Algorithm::HD,
        Algorithm::WEAK,
        0,
    )
    .unwrap();
    for models in [CopyModels::default(), enhancing(&enhancer)] {
        assert!(matches!(
            enhance_clip(&readers(), &encoders(), models, &ask, &temp.0, &mut |_| {
                true
            }),
            Err(EnhanceJobError::Vision(_))
        ));
    }
    let off = EnhanceAsk {
        level: 0,
        denoise: 0,
        ..ask.clone()
    };
    assert!(matches!(
        enhance_clip(
            &readers(),
            &encoders(),
            enhancing(&enhancer),
            &off,
            &temp.0,
            &mut |_| true
        ),
        Err(EnhanceJobError::Clip(_))
    ));
    assert_eq!(find_enhanced(&temp.0, &ask.key(), &ask.range), None);
    assert!(!temp.0.join("Resources").exists());
}

#[test]
fn a_clip_is_enhanced_into_a_copy_its_stretch_finds() {
    let (temp, draft) = project();
    let ask =
        cutlass_eval::enhance_ask(&draft, Some(&temp.0), &"v".into(), Algorithm::HD, 0, 0).unwrap();
    let enhancer = Enhancer::untrained(Compute::Cpu).unwrap();
    let mut heard = Vec::new();
    let file = enhance_clip(
        &readers(),
        &encoders(),
        enhancing(&enhancer),
        &ask,
        &temp.0,
        &mut |done| {
            heard.push(done);
            true
        },
    )
    .unwrap();
    assert_eq!(heard, vec![1.0 / 3.0, 2.0 / 3.0, 1.0]);
    assert_eq!(
        find_enhanced(&temp.0, &ask.key(), &ask.range),
        Some((file.clone(), ask.range.clone()))
    );
    let (w, h, _) = first_frame(&file);
    assert_eq!((w, h), (1440, 1080));
}

#[test]
fn a_stopped_job_leaves_no_copy() {
    let (temp, draft) = project();
    let ask =
        cutlass_eval::enhance_ask(&draft, Some(&temp.0), &"v".into(), Algorithm::HD, 0, 0).unwrap();
    let enhancer = Enhancer::untrained(Compute::Cpu).unwrap();
    let stopped = enhance_clip(
        &readers(),
        &encoders(),
        enhancing(&enhancer),
        &ask,
        &temp.0,
        &mut |done| done < 0.5,
    );
    assert_eq!(stopped, Err(EnhanceJobError::Stopped));
    assert_eq!(find_enhanced(&temp.0, &ask.key(), &ask.range), None);
    let folder = temp.0.join("Resources/video_enhance");
    assert_eq!(fs::read_dir(folder).unwrap().count(), 0);
}

#[test]
fn a_missing_file_or_no_encoder_is_an_error_and_leaves_no_copy() {
    let (temp, draft) = project();
    let ask =
        cutlass_eval::enhance_ask(&draft, Some(&temp.0), &"v".into(), Algorithm::HD, 0, 0).unwrap();
    let enhancer = Enhancer::untrained(Compute::Cpu).unwrap();
    let no_encoder = enhance_clip(
        &readers(),
        &cutlass_encode::EncodeFactory::new(),
        enhancing(&enhancer),
        &ask,
        &temp.0,
        &mut |_| true,
    );
    assert!(matches!(no_encoder, Err(EnhanceJobError::Movie(_))));
    assert_eq!(find_enhanced(&temp.0, &ask.key(), &ask.range), None);
    let gone = EnhanceAsk {
        source: temp.0.join("gone.mp4"),
        ..ask
    };
    assert!(matches!(
        enhance_clip(
            &readers(),
            &encoders(),
            enhancing(&enhancer),
            &gone,
            &temp.0,
            &mut |_| true
        ),
        Err(EnhanceJobError::Clip(_))
    ));
}

fn interpolating(interpolator: &Interpolator) -> CopyModels<'_> {
    CopyModels {
        interpolator: Some(interpolator),
        ..CopyModels::default()
    }
}

/// The red of the copy's frames, at its `rate`, over its 0.3 s.
fn reds(file: &Path, rate: f64) -> Vec<u8> {
    let mut media = readers().open(file).unwrap();
    let range = TimeRange::new(0, 300_000);
    (0..(0.3 * rate).round() as i64)
        .map(|k| {
            let at = Micros((k as f64 * 1_000_000.0 / rate).round() as i64);
            let held = media.decode_at(at, &range).unwrap();
            let Ok(cutlass_picture::Picture::Cpu(picture)) = decoded_picture(&held.frame) else {
                panic!("not in memory");
            };
            picture.rgba()[(W as usize * H as usize / 2 + W as usize / 2) * 4]
        })
        .collect()
}

#[test]
fn a_clip_is_filled_with_frames_made_between_its_own() {
    let (temp, draft) = project();
    let ask = cutlass_eval::enhance_ask(&draft, Some(&temp.0), &"v".into(), 0, 0, 30).unwrap();
    assert_eq!(ask.rate, 30.0);
    let interpolator = Interpolator::untrained(Compute::Cpu).unwrap();
    let mut heard = Vec::new();
    let file = enhance_clip(
        &readers(),
        &encoders(),
        interpolating(&interpolator),
        &ask,
        &temp.0,
        &mut |done| {
            heard.push(done);
            true
        },
    )
    .unwrap();
    // Nine frames for 0.3 s at 30 a second, three of them the clip's own.
    assert_eq!(heard.len(), 9);
    assert_eq!(heard.last(), Some(&1.0));
    assert!(
        file.parent()
            .unwrap()
            .ends_with("Resources/video_interpolate")
    );
    assert_eq!(
        find_enhanced(&temp.0, &ask.key(), &ask.range),
        Some((file.clone(), ask.range.clone()))
    );
    let (w, h, _) = first_frame(&file);
    assert_eq!((w, h), (W, H));
    let reds = reds(&file, 30.0);
    // The clip's own frames 2, 3 and 4 land where they were.
    for (k, own) in [(0, 60), (3, 80), (6, 100)] {
        assert!(reds[k].abs_diff(own) <= 4, "{k}: {reds:?}");
    }
}

#[test]
fn trained_frames_between_two_lie_between_them() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../local-files/models/interpolate");
    if !dir.join(INTERPOLATE_WEIGHTS).is_file() {
        assert!(
            std::env::var_os("CUTLASS_REQUIRE_INTERPOLATE_MODEL").is_none(),
            "the interpolate weights are missing"
        );
        return;
    }
    let (temp, draft) = project();
    let ask = cutlass_eval::enhance_ask(&draft, Some(&temp.0), &"v".into(), 0, 0, 30).unwrap();
    let interpolator = Interpolator::load(&dir, Compute::Cpu).unwrap();
    let file = enhance_clip(
        &readers(),
        &encoders(),
        interpolating(&interpolator),
        &ask,
        &temp.0,
        &mut |_| true,
    )
    .unwrap();
    // The file's frames go 60, 80, 100 and then 120, so the copy climbs
    // through the ones made between them too.
    let reds = reds(&file, 30.0);
    assert!(reds.windows(2).all(|pair| pair[0] < pair[1]), "{reds:?}");
    assert!(reds[8] > 100 + 4 && reds[8] < 120, "{reds:?}");
}

#[test]
fn optical_flow_without_its_model_is_refused_and_leaves_no_copy() {
    let (temp, draft) = project();
    let ask = cutlass_eval::enhance_ask(&draft, Some(&temp.0), &"v".into(), 0, 0, 60).unwrap();
    let result = enhance_clip(
        &readers(),
        &encoders(),
        CopyModels::default(),
        &ask,
        &temp.0,
        &mut |_| true,
    );
    let Err(EnhanceJobError::Vision(VisionError::Model { message })) = result else {
        panic!("made a copy without the model");
    };
    assert!(message.contains("Optical flow"), "{message}");
    assert_eq!(find_enhanced(&temp.0, &ask.key(), &ask.range), None);
    // Stopped part way, nothing is kept either.
    let interpolator = Interpolator::untrained(Compute::Cpu).unwrap();
    let mut seen = 0;
    let stopped = enhance_clip(
        &readers(),
        &encoders(),
        interpolating(&interpolator),
        &ask,
        &temp.0,
        &mut |_| {
            seen += 1;
            seen < 4
        },
    );
    assert_eq!(stopped, Err(EnhanceJobError::Stopped));
    assert_eq!(find_enhanced(&temp.0, &ask.key(), &ask.range), None);
}

#[test]
fn a_slowed_clip_with_every_level_on_is_made_in_one_copy() {
    let (temp, mut draft) = project();
    draft.tracks[0].segments[0].speed = 0.5;
    draft.tracks[0].segments[0].target = TimeRange::new(0, 600_000);
    let ask = cutlass_eval::enhance_ask(
        &draft,
        Some(&temp.0),
        &"v".into(),
        Algorithm::HD,
        Algorithm::WEAK,
        30,
    )
    .unwrap();
    assert_eq!(ask.rate, 60.0);
    let enhancer = Enhancer::untrained(Compute::Cpu).unwrap();
    let denoiser = Denoiser::untrained(Compute::Cpu).unwrap();
    let interpolator = Interpolator::untrained(Compute::Cpu).unwrap();
    let mut frames = 0;
    let file = enhance_clip(
        &readers(),
        &encoders(),
        CopyModels {
            enhancer: Some(&enhancer),
            denoiser: Some(&denoiser),
            interpolator: Some(&interpolator),
        },
        &ask,
        &temp.0,
        &mut |_| {
            frames += 1;
            true
        },
    )
    .unwrap();
    assert_eq!(frames, 18);
    assert!(file.parent().unwrap().ends_with("Resources/video_enhance"));
    assert_eq!(first_frame(&file).0, 1440);
}
