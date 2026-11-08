use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use cutlass_model::{MaterialVideo, TimeRange};
use cutlass_motion::{Move, Path as CameraPath};
use cutlass_store::save_camera_path;

use super::*;
use crate::media::OpenClips;

fn close(a: (f64, f64), b: (f64, f64)) -> bool {
    (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9
}

#[test]
fn no_correction_leaves_every_pixel_in_place() {
    let map = steady_map(&Correction::NONE, (64, 36)).unwrap();
    for p in [(0.5, 0.5), (63.5, 35.5), (10.0, 20.0)] {
        assert!(close(map.apply(p.0, p.1), p));
    }
}

#[test]
fn a_slide_reads_from_where_the_picture_was() {
    // Slid right a quarter of its width and down a tenth of its height.
    let map = steady_map(
        &Correction {
            x: 0.25,
            y: 0.1,
            ..Correction::NONE
        },
        (8, 10),
    )
    .unwrap();
    assert!(close(map.apply(4.5, 5.5), (2.5, 4.5)));
}

#[test]
fn the_map_undoes_the_turn_growth_and_slide() {
    let correction = Correction {
        x: -0.03,
        y: 0.02,
        turn: 0.07,
        scale: 1.08,
    };
    let (w, h) = (320.0, 180.0);
    let map = steady_map(&correction, (320, 180)).unwrap();
    for p in [(10.0, 15.0), (160.0, 90.0), (300.0, 170.0)] {
        // Forward, as the plan defines it: about the middle, then slid.
        let (u, v) = (p.0 - w / 2.0, p.1 - h / 2.0);
        let (c, s) = (correction.turn.cos(), correction.turn.sin());
        let q = (
            w / 2.0 + correction.scale * (c * u - s * v) + correction.x * w,
            h / 2.0 + correction.scale * (s * u + c * v) + correction.y * h,
        );
        let back = map.apply(q.0, q.1);
        assert!(close(back, p), "{back:?} {p:?}");
    }
}

#[test]
fn a_bad_correction_has_no_map() {
    for bad in [
        Correction {
            scale: 0.0,
            ..Correction::NONE
        },
        Correction {
            x: f64::NAN,
            ..Correction::NONE
        },
    ] {
        assert!(steady_map(&bad, (8, 8)).is_none());
    }
    assert!(steady_map(&Correction::NONE, (0, 8)).is_none());
}

#[test]
fn maps_compose_inner_first() {
    let slide = Affine([1.0, 0.0, 3.0, 0.0, 1.0, 0.0]);
    let double = Affine([2.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
    assert!(close(then(slide, double).apply(1.0, 1.0), (8.0, 2.0)));
    assert!(close(then(double, slide).apply(1.0, 1.0), (5.0, 2.0)));
}

#[test]
fn a_resample_moves_pixels_and_clears_what_falls_outside() {
    let row: Vec<u8> = (1..=4).flat_map(|v| [v, 0, 0, 255]).collect();
    let picture = CpuBytes::new(4, 1, row).unwrap();
    // Each pixel reads the one to its left.
    let left = Affine([1.0, 0.0, -1.0, 0.0, 1.0, 0.0]);
    let moved = resample(&picture, left, (4, 1)).unwrap();
    assert_eq!(
        moved.rgba(),
        &[0, 0, 0, 0, 1, 0, 0, 255, 2, 0, 0, 255, 3, 0, 0, 255]
    );
}

fn steady(frames: Vec<Correction>, start: i64, rate: f64) -> Steady {
    Steady {
        ask: StableAsk {
            source: PathBuf::from("/a.mp4"),
            range: TimeRange::new(start, 10_000_000),
            rate,
        },
        strength: Strength::Recommended,
        plan: Plan { zoom: 1.0, frames },
    }
}

#[test]
fn a_time_between_frames_mixes_them_and_the_ends_hold() {
    let shift = |x: f64| Correction {
        x,
        ..Correction::NONE
    };
    let kept = steady(vec![shift(0.0), shift(0.1), shift(0.3)], 1_000_000, 10.0);
    assert!((kept.at(Micros(1_100_000)).x - 0.1).abs() < 1e-12);
    assert!((kept.at(Micros(1_150_000)).x - 0.2).abs() < 1e-12);
    assert_eq!(kept.at(Micros(0)).x, 0.0);
    assert!((kept.at(Micros(9_000_000)).x - 0.3).abs() < 1e-12);
    assert_eq!(steady(Vec::new(), 0, 10.0).at(Micros(5)), Correction::NONE);
}

#[test]
fn a_stored_level_names_its_strength() {
    assert_eq!(strength_of(1), Strength::MinimumCut);
    assert_eq!(strength_of(2), Strength::Recommended);
    assert_eq!(strength_of(3), Strength::MostStable);
    assert_eq!(strength_of(7), Strength::Recommended);
}

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "cutlass-steady-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A project folder holding a movie, and a draft whose one clip shows a
/// second of it from 1 s, at 2 s on the timeline.
fn project(level: Option<i64>) -> (Temp, Draft) {
    let temp = Temp::new();
    fs::write(temp.0.join("walk.mp4"), b"a movie").unwrap();
    let mut draft = Draft::empty("D");
    draft.fps = 10.0;
    draft.materials.videos.push(MaterialVideo {
        id: "m".into(),
        kind: "video".into(),
        path: "walk.mp4".into(),
        width: 320,
        height: 180,
        duration: Micros(3_000_000),
        stable: level.map(|level| VideoStable {
            level,
            ..VideoStable::default()
        }),
        ..MaterialVideo::default()
    });
    let mut segment = Segment::new("v", "m", TimeRange::new(2_000_000, 1_000_000));
    segment.source_range = Some(TimeRange::new(1_000_000, 1_000_000));
    draft.tracks[0].segments.push(segment);
    (temp, draft)
}

/// A shaking camera's path through the whole movie at 10 frames a second.
fn shaky() -> CameraPath {
    CameraPath {
        width: 320,
        height: 180,
        rate: 10.0,
        moves: (0..30)
            .map(|k| Move {
                dx: 4.0 * (f64::from(k) * 1.7).sin(),
                dy: 3.0 * (f64::from(k) * 2.3).cos(),
                confidence: 1.0,
                ..Move::STILL
            })
            .collect(),
    }
}

fn keep(temp: &Temp, draft: &Draft) {
    let ask = stable_ask(draft, Some(&temp.0), &"v".into()).unwrap();
    let key = StableKey {
        source: ask.source,
        start: ask.range.start,
        duration: ask.range.duration,
        rate: ask.rate,
    };
    save_camera_path(&temp.0, &key, &shaky()).unwrap();
}

fn clips_in(temp: &Temp) -> OpenClips {
    let mut clips = OpenClips::new();
    clips.set_project(Some(&temp.0));
    clips
}

#[test]
fn a_steadied_clip_reads_its_planned_correction_at_the_source_frame() {
    let (temp, draft) = project(Some(VideoStable::RECOMMENDED));
    keep(&temp, &draft);
    let mut clips = clips_in(&temp);
    let segment = draft.tracks[0].segments[0].clone();
    // Timeline 2.5 s is source 1.5 s: frame 15 of the movie.
    let got = steady_at(&mut clips, &draft, &segment, Micros(2_500_000)).unwrap();
    let want = plan(&shaky(), Strength::Recommended).unwrap().frames[15];
    assert_eq!(got, want);
    assert!(want.scale > 1.0 && want != Correction::NONE);
    // Another level plans again from the same kept path.
    let mut most = draft.clone();
    most.materials.videos[0].stable.as_mut().unwrap().level = VideoStable::MOST_STABLE;
    let again = steady_at(&mut clips, &most, &segment, Micros(2_500_000)).unwrap();
    assert_eq!(
        again,
        plan(&shaky(), Strength::MostStable).unwrap().frames[15]
    );
}

#[test]
fn an_unsteadied_clip_or_one_without_a_project_is_drawn_as_it_is() {
    let (temp, draft) = project(None);
    keep(&temp, &draft);
    let segment = draft.tracks[0].segments[0].clone();
    assert_eq!(
        steady_at(&mut clips_in(&temp), &draft, &segment, Micros(2_500_000)),
        None
    );
    let (temp, draft) = project(Some(0));
    keep(&temp, &draft);
    assert_eq!(
        steady_at(&mut clips_in(&temp), &draft, &segment, Micros(2_500_000)),
        None
    );
    let (_temp, draft) = project(Some(VideoStable::RECOMMENDED));
    assert_eq!(
        steady_at(&mut OpenClips::new(), &draft, &segment, Micros(2_500_000)),
        None
    );
}

#[test]
fn a_clip_still_measuring_is_steadied_once_its_path_is_kept() {
    let (temp, draft) = project(Some(VideoStable::RECOMMENDED));
    let mut clips = clips_in(&temp);
    let segment = draft.tracks[0].segments[0].clone();
    assert_eq!(
        steady_at(&mut clips, &draft, &segment, Micros(2_500_000)),
        None
    );
    keep(&temp, &draft);
    assert!(steady_at(&mut clips, &draft, &segment, Micros(2_500_000)).is_some());
}

#[test]
fn the_cpu_picture_is_slid_as_the_gpu_maps_it() {
    // An 8 by 2 picture whose red channel counts across.
    let rgba: Vec<u8> = (0..2)
        .flat_map(|_| (0..8).flat_map(|x| [x * 10 + 10, 0, 0, 255]))
        .collect();
    let picture = CpuBytes::new(8, 2, rgba).unwrap();
    let slide = Correction {
        x: 0.25,
        ..Correction::NONE
    };
    let map = steady_map(&slide, (8, 2)).unwrap();
    let cpu = resample(&picture, map, (8, 2)).unwrap();
    let reds: Vec<u8> = cpu.rgba().chunks(4).take(8).map(|p| p[0]).collect();
    assert_eq!(reds, vec![0, 0, 10, 20, 30, 40, 50, 60]);
}
