use std::fs::{self, File};
use std::sync::atomic::{AtomicU64, Ordering};

use cutlass_codec::ContainerId;
use cutlass_container::{Demux, OpenError, Probe, ProbeResult};
use cutlass_effects::{MATTING, Matting, MattingMode, PickClick};
use cutlass_engine::decoded_picture;
use cutlass_isobmff::IsoBmffFile;
use cutlass_model::{Draft, MaterialVideo, Micros, RawMaterial, Segment, TimeRange};
use cutlass_store::find_matte;

use super::*;

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "cutlass-matte-job-{}-{}",
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

/// Frame `k` of the movie is flat grey `GREYS[k]`, so a stub can tell which
/// frame it was handed.
const GREYS: [u8; 10] = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180];

/// A project holding a ten-frame 32 by 24 movie at 10 a second, and a
/// draft whose clip shows 0.2 s to 0.6 s of it (frames 2 to 5), with
/// `matting`.
fn project(matting: &Matting) -> (Temp, Draft) {
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
            let g = GREYS[k as usize];
            Ok((0..W * H).flat_map(|_| [g, g, g, 255]).collect())
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
    draft.materials.common_mask.push(RawMaterial {
        id: "r".into(),
        kind: MATTING.into(),
        fields: matting.to_fields(),
    });
    let mut segment = Segment::new("v", "m", TimeRange::new(0, 400_000));
    segment.source_range = Some(TimeRange::new(200_000, 400_000));
    segment.extra_material_refs.push("r".into());
    draft.tracks[0].segments.push(segment);
    (temp, draft)
}

fn ask(temp: &Temp, draft: &Draft) -> MatteAsk {
    cutlass_eval::matte_ask(draft, Some(&temp.0), &"v".into()).unwrap()
}

/// Which movie frame a flat grey picture is, within the codec's error.
fn frame_of(rgb: &[u8]) -> usize {
    let g = rgb[rgb.len() / 2];
    GREYS
        .iter()
        .enumerate()
        .min_by_key(|(_, grey)| grey.abs_diff(g))
        .map(|(k, _)| k)
        .unwrap()
}

/// Keeps the left half, or `width` pixels of it when set; can be told to
/// answer at the wrong size.
struct LeftHalf {
    wrong_size: bool,
}

impl Matter for LeftHalf {
    fn id(&self) -> &str {
        "left-half"
    }

    fn matte(&mut self, image: &Image<'_>) -> Result<Matte, VisionError> {
        let width = if self.wrong_size {
            image.width + 2
        } else {
            image.width
        };
        let alpha = (0..width * image.height)
            .map(|i| if i % width < width / 2 { 255 } else { 0 })
            .collect();
        Ok(Matte::new(width, image.height, alpha).unwrap())
    }
}

/// Answers each step with a matte whose kept share is the frame's own
/// number over ten, keeping the left columns, and records what it saw.
#[derive(Default)]
struct Recorder {
    steps: Vec<(usize, Option<Prompt>)>,
    resets: usize,
}

impl Follower for Recorder {
    fn id(&self) -> &str {
        "recorder"
    }

    fn reset(&mut self) {
        self.resets += 1;
    }

    fn step(&mut self, image: &Image<'_>, prompt: Option<&Prompt>) -> Result<Matte, VisionError> {
        let k = frame_of(image.rgb);
        self.steps.push((k, prompt.cloned()));
        if let Some(mask) = prompt.and_then(|prompt| prompt.mask.clone()) {
            return Ok(mask);
        }
        let kept = image.width * k as u32 / 10;
        let alpha = (0..image.width * image.height)
            .map(|i| if i % image.width < kept { 255 } else { 0 })
            .collect();
        Ok(Matte::new(image.width, image.height, alpha).unwrap())
    }
}

/// The written matte video's frames' first-row greys, decoded.
fn read_back(file: &Path, frames: usize) -> Vec<Vec<u8>> {
    let mut media = readers().open(file).unwrap();
    (0..frames)
        .map(|k| {
            let at = Micros(k as i64 * 100_000);
            let held = media
                .decode_at(at, &TimeRange::new(0, frames as i64 * 100_000))
                .unwrap();
            let Ok(cutlass_picture::Picture::Cpu(picture)) = decoded_picture(&held.frame) else {
                panic!("not in memory");
            };
            let row = picture.width() as usize;
            picture.rgba()[..row * 4].chunks(4).map(|p| p[0]).collect()
        })
        .collect()
}

#[test]
fn auto_removal_writes_a_grey_matte_video_its_stretch_finds() {
    let (temp, draft) = project(&Matting::auto());
    let ask = ask(&temp, &draft);
    let mut heard = Vec::new();
    let file = matte_clip(
        &readers(),
        &encoders(),
        &mut LeftHalf { wrong_size: false },
        &ask,
        &temp.0,
        &mut |done| {
            heard.push(done);
            true
        },
    )
    .unwrap();
    assert_eq!(heard, vec![0.25, 0.5, 0.75, 1.0]);
    assert!(file.parent().unwrap().ends_with("Resources/video_matte"));
    assert_eq!(
        find_matte(&temp.0, &ask.key(), &ask.range),
        Some((file.clone(), ask.range.clone()))
    );
    for row in read_back(&file, 4) {
        assert_eq!(row.len(), W as usize);
        assert!(row[2] > 230, "{row:?}");
        assert!(row[W as usize - 3] < 25, "{row:?}");
    }
}

/// Movies and still pictures.
fn readers_with_stills() -> ReaderFactory {
    let mut readers = readers();
    readers.register_container(
        |header: &[u8]| -> ProbeResult { cutlass_still::StillProbe.probe(header) },
        |path: &Path| -> Result<Box<dyn Demux>, OpenError> {
            let file = File::open(path).map_err(|error| OpenError::Source {
                message: error.to_string(),
            })?;
            let len = file
                .metadata()
                .map_err(|error| OpenError::Source {
                    message: error.to_string(),
                })?
                .len();
            Ok(Box::new(cutlass_still::StillDemux::open(
                cutlass_container::Input::random(file, len, cutlass_still::still_limits()),
            )?))
        },
    );
    cutlass_png::register_decoders(readers.decoders_mut());
    readers
}

#[test]
fn a_photos_matte_is_one_grey_picture_found_however_the_clip_is_trimmed() {
    let (temp, mut draft) = project(&Matting::auto());
    let (w, h) = (W as usize, H as usize);
    let rgb: Vec<u8> = (0..w * h).flat_map(|_| [90, 120, 150]).collect();
    let mut png = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(&rgb, W, H, ExtendedColorType::Rgb8)
        .unwrap();
    fs::write(temp.0.join("photo.png"), png).unwrap();
    let video = &mut draft.materials.videos[0];
    video.kind = "photo".into();
    video.path = "photo.png".into();
    video.duration = Micros(10_800_000_000);
    let ask = ask(&temp, &draft);
    assert!(ask.still);
    let mut heard = Vec::new();
    let file = matte_clip(
        &readers_with_stills(),
        &encoders(),
        &mut LeftHalf { wrong_size: false },
        &ask,
        &temp.0,
        &mut |done| {
            heard.push(done);
            true
        },
    )
    .unwrap();
    assert_eq!(heard, vec![1.0], "one picture");
    assert_eq!(
        find_matte(&temp.0, &ask.key(), &ask.range).map(|(path, _)| path),
        Some(file.clone())
    );
    let mut media = readers_with_stills().open(&file).unwrap();
    let held = media.decode_at(Micros(0), &ask.range).unwrap();
    let Ok(cutlass_picture::Picture::Cpu(picture)) = decoded_picture(&held.frame) else {
        panic!("not in memory");
    };
    assert_eq!((picture.width(), picture.height()), (W, H));
    let row: Vec<u8> = picture.rgba()[..w * 4].chunks(4).map(|p| p[0]).collect();
    assert_eq!(row[2], 255, "{row:?}");
    assert_eq!(row[w - 3], 0, "{row:?}");
    let leftovers: Vec<_> = fs::read_dir(file.parent().unwrap())
        .unwrap()
        .flatten()
        .map(|entry| entry.file_name())
        .collect();
    assert_eq!(leftovers.len(), 1, "{leftovers:?}");

    draft.tracks[0].segments[0].source_range = Some(TimeRange::new(5_000_000, 2_000_000));
    assert_eq!(
        cutlass_eval::matte_ask(&draft, Some(&temp.0), &"v".into())
            .unwrap()
            .key(),
        ask.key()
    );
}

#[test]
fn a_stopped_or_failed_job_leaves_no_matte() {
    let (temp, draft) = project(&Matting::auto());
    let ask = ask(&temp, &draft);
    let stopped = matte_clip(
        &readers(),
        &encoders(),
        &mut LeftHalf { wrong_size: false },
        &ask,
        &temp.0,
        &mut |done| done < 0.5,
    );
    assert_eq!(stopped, Err(EnhanceJobError::Stopped));
    let wrong = matte_clip(
        &readers(),
        &encoders(),
        &mut LeftHalf { wrong_size: true },
        &ask,
        &temp.0,
        &mut |_| true,
    );
    assert!(
        matches!(wrong, Err(EnhanceJobError::Vision(_))),
        "{wrong:?}"
    );
    let no_encoder = matte_clip(
        &readers(),
        &cutlass_encode::EncodeFactory::new(),
        &mut LeftHalf { wrong_size: false },
        &ask,
        &temp.0,
        &mut |_| true,
    );
    assert!(matches!(no_encoder, Err(EnhanceJobError::Movie(_))));
    assert_eq!(find_matte(&temp.0, &ask.key(), &ask.range), None);
    let folder = temp.0.join("Resources/video_matte");
    assert_eq!(fs::read_dir(folder).unwrap().count(), 0);
}

fn picked(frames: &[(i64, f64)]) -> Matting {
    Matting {
        mode: MattingMode::Custom,
        picks: frames
            .iter()
            .map(|(at, x)| Pick {
                frame: Micros(*at),
                clicks: vec![PickClick {
                    x: *x,
                    y: 0.5,
                    keep: true,
                }],
                brushes: Vec::new(),
            })
            .collect(),
        ..Matting::auto()
    }
}

#[test]
fn custom_removal_follows_back_from_the_first_pick_then_forward_correcting_on_each() {
    // Picks on movie frames 3 and 5: the stretch's frames 1 and 3.
    let matting = picked(&[(300_000, 0.25), (500_000, 0.75)]);
    let (temp, draft) = project(&matting);
    let ask = ask(&temp, &draft);
    let mut follower = Recorder::default();
    let mut heard = Vec::new();
    let file = follow_clip(
        &readers(),
        &encoders(),
        &mut follower,
        &ask,
        &temp.0,
        &mut |done| {
            heard.push(done);
            true
        },
    )
    .unwrap();
    assert_eq!(follower.resets, 2);
    let order: Vec<(usize, bool)> = follower
        .steps
        .iter()
        .map(|(k, prompt)| (*k, prompt.is_some()))
        .collect();
    assert_eq!(
        order,
        vec![
            // Back from the first pick.
            (3, true),
            (2, false),
            // Forward from it, corrected on the second.
            (3, true),
            (4, false),
            (5, true),
        ]
    );
    // The clicks are in the frame's own pixels.
    let first = follower.steps[0].1.as_ref().unwrap();
    assert_eq!((first.clicks[0].x, first.clicks[0].y), (8.0, 12.0));
    // Every frame counted once; the share rises to the whole.
    assert_eq!(heard.len(), 5);
    assert_eq!(heard.last(), Some(&1.0));
    assert!(heard.windows(2).all(|pair| pair[0] < pair[1]));
    // Frame 2 came from the backward run: 2 tenths kept, and so on.
    let rows = read_back(&file, 4);
    let kept = |row: &Vec<u8>| row.iter().filter(|v| **v > 128).count();
    assert_eq!(
        rows.iter().map(kept).collect::<Vec<_>>(),
        vec![6, 9, 12, 16]
    );
    assert_eq!(
        find_matte(&temp.0, &ask.key(), &ask.range),
        Some((file, ask.range.clone()))
    );
}

#[test]
fn plain_brush_strokes_paint_the_picked_frame_exactly_and_are_carried_on() {
    let matting = Matting {
        mode: MattingMode::Custom,
        picks: vec![Pick {
            frame: Micros(200_000),
            clicks: Vec::new(),
            brushes: vec![Brush {
                points: vec![[0.0, 0.5], [1.0, 0.5]],
                radius: 0.5,
                keep: true,
                smart: false,
            }],
        }],
        ..Matting::auto()
    };
    let (temp, draft) = project(&matting);
    let ask = ask(&temp, &draft);
    let mut follower = Recorder::default();
    let file = follow_clip(
        &readers(),
        &encoders(),
        &mut follower,
        &ask,
        &temp.0,
        &mut |_| true,
    )
    .unwrap();
    // No clicks: the painted matte is the prompt itself, once a run.
    let painted = follower.steps[0].1.as_ref().unwrap();
    assert!(painted.clicks.is_empty());
    assert!(
        painted
            .mask
            .as_ref()
            .unwrap()
            .alpha
            .iter()
            .all(|a| *a == 255)
    );
    let rows = read_back(&file, 1);
    assert!(rows[0].iter().all(|v| *v > 230), "{:?}", rows[0]);
}

#[test]
fn custom_removal_with_no_pick_is_refused() {
    let unpicked = Matting {
        mode: MattingMode::Custom,
        ..Matting::auto()
    };
    let (temp, draft) = project(&picked(&[(200_000, 0.5)]));
    let mut ask = ask(&temp, &draft);
    ask.matting = unpicked;
    let refused = follow_clip(
        &readers(),
        &encoders(),
        &mut Recorder::default(),
        &ask,
        &temp.0,
        &mut |_| true,
    );
    assert!(matches!(refused, Err(EnhanceJobError::Clip(_))));
}

#[test]
fn a_smart_brush_becomes_clicks_along_its_path() {
    let prompt = PickPrompt {
        pick: Pick {
            frame: Micros(0),
            clicks: vec![PickClick {
                x: 0.5,
                y: 0.5,
                keep: true,
            }],
            brushes: vec![Brush {
                points: (0..=10).map(|i| [f64::from(i) / 10.0, 0.5]).collect(),
                radius: 0.05,
                keep: false,
                smart: true,
            }],
        },
    };
    // 100 by 100: a click every 10 pixels of path, from the first point.
    let clicks = prompt.clicks(100, 100);
    assert_eq!(
        clicks[0],
        Click {
            x: 50.0,
            y: 50.0,
            keep: true
        }
    );
    let smart: Vec<f32> = clicks[1..].iter().map(|c| c.x).collect();
    assert_eq!(
        smart,
        vec![
            0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0
        ]
    );
    assert!(clicks[1..].iter().all(|c| !c.keep));
    assert!(prompt.painted().is_empty());
}

#[test]
fn painting_keeps_and_erases_round_strokes() {
    let mut matte = Matte::new(10, 10, vec![0; 100]).unwrap();
    let keep = Brush {
        points: vec![[0.5, 0.5]],
        radius: 0.2,
        keep: true,
        smart: false,
    };
    let erase = Brush {
        points: vec![[0.5, 0.5]],
        radius: 0.1,
        keep: false,
        smart: false,
    };
    paint(&mut matte, &[&keep, &erase]);
    let at = |x: usize, y: usize| matte.alpha[y * 10 + x];
    assert_eq!(at(5, 5), 0, "erased in the middle");
    assert_eq!(at(6, 4), 255, "kept around it");
    assert_eq!(at(0, 0), 0, "outside the stroke");
}

#[test]
fn easing_settles_small_changes_and_follows_big_ones() {
    let last = [100, 100, 0];
    let mut now = [110, 200, 40];
    ease(&last, &mut now);
    assert_eq!(now, [105, 200, 20]);
}

#[test]
fn a_matte_video_is_at_most_1080_on_its_short_side_and_even() {
    assert_eq!(matte_size(1920, 1080), (1920, 1080));
    assert_eq!(matte_size(3840, 2160), (1920, 1080));
    assert_eq!(matte_size(2160, 3840), (1080, 1920));
    assert_eq!(matte_size(33, 25), (32, 24));
    assert_eq!(matte_size(1, 1), (2, 2));
}

#[test]
fn packed_mattes_unpack_to_the_same_bytes() {
    for alpha in [vec![], vec![0; 1000], vec![0, 0, 255, 255, 128, 0], vec![7]] {
        let packed = Packed::pack(&alpha);
        assert_eq!(packed.unpack(), alpha);
    }
    assert_eq!(Packed::pack(&[0; 1000]).0.len(), 1);
}

#[test]
fn picks_land_on_the_nearest_frame_and_two_on_one_keep_the_later() {
    let (temp, draft) = project(&picked(&[(200_000, 0.1), (240_000, 0.2), (900_000, 0.3)]));
    let ask = ask(&temp, &draft);
    let frames = picked_frames(&ask.matting.picks, &ask, 4);
    let at: Vec<usize> = frames.iter().map(|(k, _)| *k).collect();
    assert_eq!(at, vec![0, 3]);
    let kept = &frames[0].1.pick.clicks[0];
    assert!((kept.x - 0.2).abs() < 1e-9);
}

/// BiRefNet_lite on a second of a real clip: three children in a busy room.
/// Slow, so run by hand with `--ignored`; set `CUTLASS_MATTE_OUT` to a
/// folder to keep the matte video and look at it.
#[test]
#[ignore = "reads a real clip and runs the real model"]
fn birefnet_mattes_a_real_clip() {
    let local = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
    let clip = local.join("7150172-hd_720_1280_50fps.mp4");
    let weights = local.join("models/matting");
    if !clip.is_file() || !weights.join(crate::MATTING_WEIGHTS).is_file() {
        return;
    }
    let temp = Temp::new();
    fs::copy(&clip, temp.0.join("party.mp4")).unwrap();
    let mut draft = Draft::empty("D");
    draft.fps = 25.0;
    draft.materials.videos.push(MaterialVideo {
        id: "m".into(),
        kind: "video".into(),
        path: "party.mp4".into(),
        width: 720,
        height: 1280,
        duration: Micros(7_820_000),
        ..MaterialVideo::default()
    });
    draft.materials.common_mask.push(RawMaterial {
        id: "r".into(),
        kind: MATTING.into(),
        fields: Matting::auto().to_fields(),
    });
    let mut segment = Segment::new("v", "m", TimeRange::new(0, 1_000_000));
    segment.source_range = Some(TimeRange::new(2_000_000, 1_000_000));
    segment.extra_material_refs.push("r".into());
    draft.tracks[0].segments.push(segment);
    let ask = cutlass_eval::matte_ask(&draft, Some(&temp.0), &"v".into()).unwrap();
    let compute = if crate::gpu_available() {
        crate::Compute::Gpu
    } else {
        crate::Compute::Cpu
    };
    let mut model = crate::Matting::load(&weights, compute).unwrap();
    assert_eq!(model.id(), ask.model());
    let start = std::time::Instant::now();
    let file = matte_clip(
        &readers(),
        &encoders(),
        &mut model,
        &ask,
        &temp.0,
        &mut |_| true,
    )
    .unwrap();
    let took = start.elapsed();
    // Every tenth of a second's first row.
    let frames = read_back(&file, 10);
    // The children fill the middle and lower frame; the wall above them
    // does not.
    let kept = |row: &[u8]| row.iter().filter(|g| **g > 128).count();
    let first = &frames[0];
    assert!(
        kept(first) < first.len() / 4,
        "the wall's top row is dropped"
    );
    eprintln!(
        "25 frames in {took:.2?} on {compute:?}, {:.0} ms a frame",
        took.as_secs_f64() * 40.0
    );
    if let Some(out) = std::env::var_os("CUTLASS_MATTE_OUT") {
        fs::copy(&file, Path::new(&out).join("party-matte.mp4")).unwrap();
    }
}

/// Answers every step with the same soft greys, as SAM 2 does where it is
/// unsure.
struct Hazy;

impl Follower for Hazy {
    fn id(&self) -> &str {
        "hazy"
    }

    fn reset(&mut self) {}

    fn step(&mut self, image: &Image<'_>, _: Option<&Prompt>) -> Result<Matte, VisionError> {
        let alpha = (0..image.width * image.height)
            .map(|i| [90, 127, 128, 170][i as usize % 4])
            .collect();
        Ok(Matte::new(image.width, image.height, alpha).unwrap())
    }
}

#[test]
fn a_followers_unsure_greys_are_cut_at_half() {
    let rgb = vec![50_u8; 8 * 2 * 3];
    let image = Image {
        rgb: &rgb,
        width: 8,
        height: 2,
    };
    let expected: Vec<u8> = (0..16).map(|i| [0, 0, 255, 255][i % 4]).collect();
    let followed = prompted_step(&mut Hazy, &image, None, 0).unwrap();
    assert_eq!(followed.alpha, expected);
    let mut matte = Matte::new(4, 1, vec![0, 127, 128, 255]).unwrap();
    harden(&mut matte);
    assert_eq!(matte.alpha, vec![0, 0, 255, 255]);
}
