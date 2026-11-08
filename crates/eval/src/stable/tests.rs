use cutlass_model::{MaterialVideo, Segment, TrackType};

use super::*;

const W: u32 = 200;
const H: u32 = 150;

/// Smooth value noise, 0 to 255.
fn noise(x: f64, y: f64) -> f32 {
    let cell = 6.0;
    let (gx, gy) = (x / cell, y / cell);
    let (ix, iy) = (gx.floor(), gy.floor());
    let (fx, fy) = (gx - ix, gy - iy);
    let smooth = |t: f64| t * t * (3.0 - 2.0 * t);
    let corner = |dx: f64, dy: f64| {
        let mut z = 7
            ^ ((ix + dx) as i64 as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ ((iy + dy) as i64 as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        (z >> 56) as f64
    };
    let (sx, sy) = (smooth(fx), smooth(fy));
    let top = corner(0.0, 0.0) * (1.0 - sx) + corner(1.0, 0.0) * sx;
    let bottom = corner(0.0, 1.0) * (1.0 - sx) + corner(1.0, 1.0) * sx;
    (top * (1.0 - sy) + bottom * sy) as f32
}

/// How far the picture slid by frame `k`: a hand's shake.
fn shake(k: usize) -> (f64, f64) {
    let k = k as f64;
    (4.0 * (1.3 * k).sin(), 3.0 * (2.1 * k).cos())
}

/// Frame `k`: a textured world seen through the shaking camera.
fn frame(k: usize) -> Gray {
    let (dx, dy) = shake(k);
    let data = (0..H)
        .flat_map(|y| (0..W).map(move |x| noise(f64::from(x) - dx, f64::from(y) - dy)))
        .collect();
    Gray::new(W, H, data).unwrap()
}

fn times(frames: usize) -> Vec<Micros> {
    (0..frames)
        .map(|k| Micros(k as i64 * 1_000_000 / 30))
        .collect()
}

fn index(time: Micros) -> usize {
    (time.0 as f64 * 30.0 / 1_000_000.0).round() as usize
}

#[test]
fn a_shaking_clip_is_measured_frame_by_frame() {
    let mut heard = Vec::new();
    let path = measure_frames(
        &times(16),
        30.0,
        &mut |time| Ok(frame(index(time))),
        &mut |done| {
            heard.push(done);
            true
        },
    )
    .unwrap();
    assert_eq!(path.moves.len(), 15);
    assert_eq!(path.rate, 30.0);
    for (k, found) in path.moves.iter().enumerate() {
        let (before, after) = (shake(k), shake(k + 1));
        let want = (after.0 - before.0, after.1 - before.1);
        let off = (found.dx - want.0).hypot(found.dy - want.1);
        assert!(off < 0.3, "frame {}: {off:.3} pixels off", k + 1);
    }
    assert_eq!(heard.len(), 16);
    assert!(heard.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(heard.last(), Some(&1.0));
}

#[test]
fn the_callback_stops_the_job() {
    let mut calls = 0;
    let result = measure_frames(
        &times(10),
        30.0,
        &mut |time| Ok(frame(index(time))),
        &mut |_| {
            calls += 1;
            calls < 3
        },
    );
    assert_eq!(result, Err(StableError::Stopped));
    assert_eq!(calls, 3);
}

#[test]
fn an_unreadable_or_resized_frame_or_no_frames_is_an_error() {
    let unreadable = measure_frames(
        &times(5),
        30.0,
        &mut |time| {
            if index(time) == 3 {
                Err(StableError::Frame {
                    reason: "broken".into(),
                })
            } else {
                Ok(frame(index(time)))
            }
        },
        &mut |_| true,
    );
    assert!(matches!(unreadable, Err(StableError::Frame { .. })));
    let resized = measure_frames(
        &times(5),
        30.0,
        &mut |time| {
            if index(time) == 2 {
                Ok(Gray::new(W / 2, H / 2, vec![0.0; (W * H / 4) as usize]).unwrap())
            } else {
                Ok(frame(index(time)))
            }
        },
        &mut |_| true,
    );
    assert!(matches!(
        resized,
        Err(StableError::Motion(MotionError::Frame { .. }))
    ));
    let none = measure_frames(&[], 30.0, &mut |_| Ok(frame(0)), &mut |_| true);
    assert!(matches!(none, Err(StableError::Clip { .. })));
}

#[test]
fn frames_are_read_at_the_rate_through_the_source_range() {
    let range = TimeRange::new(1_000_000, 1_000_000);
    let times = source_times(&range, 30.0).unwrap();
    assert_eq!(times.len(), 30);
    assert_eq!(times[0], Micros(1_000_000));
    assert_eq!(times[1], Micros(1_033_333));
    assert!(times.iter().all(|t| (1_000_000..2_000_000).contains(&t.0)));
    for bad in [TimeRange::new(0, 0), TimeRange::new(-5, 1_000)] {
        assert!(matches!(
            source_times(&bad, 30.0),
            Err(StableError::Clip { .. })
        ));
    }
    for rate in [0.0, f64::NAN] {
        assert!(source_times(&range, rate).is_err());
    }
    // A clip far too long is refused before anything is read.
    let endless = TimeRange::new(0, i64::MAX / 2);
    assert!(matches!(
        source_times(&endless, 60.0),
        Err(StableError::Clip { .. })
    ));
}

fn clip_draft(path: &str) -> Draft {
    let mut draft = Draft::empty("D");
    draft.fps = 25.0;
    draft.materials.videos.push(MaterialVideo {
        id: "m".into(),
        kind: "video".into(),
        path: path.into(),
        width: 1920,
        height: 1080,
        ..MaterialVideo::default()
    });
    let mut segment = Segment::new("v", "m", TimeRange::new(0, 2_000_000));
    segment.source_range = Some(TimeRange::new(500_000, 2_000_000));
    draft.tracks[0].segments.push(segment);
    draft
}

#[test]
fn the_ask_names_the_file_the_source_range_and_the_project_rate() {
    let draft = clip_draft("/movies/walk.mp4");
    let ask = stable_ask(&draft, None, &"v".into()).unwrap();
    assert_eq!(
        ask,
        StableAsk {
            source: PathBuf::from("/movies/walk.mp4"),
            range: TimeRange::new(500_000, 2_000_000),
            rate: 25.0,
        }
    );
    let mut odd = draft.clone();
    odd.fps = f64::NAN;
    assert_eq!(stable_ask(&odd, None, &"v".into()).unwrap().rate, 30.0);
}

#[test]
fn a_missing_clip_a_graphic_or_a_file_outside_the_project_is_refused() {
    let draft = clip_draft("/movies/walk.mp4");
    assert!(matches!(
        stable_ask(&draft, None, &"nope".into()),
        Err(StableError::Clip { .. })
    ));
    let mut text = draft.clone();
    text.tracks[0].kind = TrackType::Text;
    assert!(matches!(
        stable_ask(&text, None, &"v".into()),
        Err(StableError::Clip { .. })
    ));
    let mut no_file = draft.clone();
    no_file.materials.videos[0].path.clear();
    assert!(stable_ask(&no_file, None, &"v".into()).is_err());
    let root = std::env::temp_dir();
    let escaping = clip_draft("../../etc/passwd");
    assert!(matches!(
        stable_ask(&escaping, Some(&root), &"v".into()),
        Err(StableError::Clip { .. })
    ));
    let mut empty = draft;
    empty.tracks[0].segments[0].source_range = Some(TimeRange::new(0, 0));
    assert!(stable_ask(&empty, None, &"v".into()).is_err());
    empty.tracks[0].segments[0].source_range = None;
    assert!(stable_ask(&empty, None, &"v".into()).is_err());
}

#[test]
fn a_missing_file_is_a_frame_error() {
    let ask = StableAsk {
        source: std::env::temp_dir().join("cutlass-no-such-movie.mp4"),
        range: TimeRange::new(0, 1_000_000),
        rate: 30.0,
    };
    let result = measure_camera(&ReaderFactory::new(), &ask, &mut |_| true);
    assert!(matches!(result, Err(StableError::Frame { .. })));
}
