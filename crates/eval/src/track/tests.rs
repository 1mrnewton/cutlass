use cutlass_model::{Draft, MaterialVideo, Segment, TimeRange};

use super::*;

const W: u32 = 200;
const H: u32 = 150;

/// Smooth value noise, 0 to 255.
fn noise(x: f64, y: f64, seed: u64) -> f32 {
    let cell = 5.0;
    let (gx, gy) = (x / cell, y / cell);
    let (ix, iy) = (gx.floor(), gy.floor());
    let (fx, fy) = (gx - ix, gy - iy);
    let smooth = |t: f64| t * t * (3.0 - 2.0 * t);
    let corner = |dx: f64, dy: f64| {
        let mut z = seed
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

/// The square's top left at frame `k`: 40 pixels wide, sliding right.
fn square_at(k: i64) -> (f64, f64) {
    (60.0 + 1.5 * k as f64, 50.0 + 0.5 * k as f64)
}

/// Frame `k`: the textured square over a still background. In `gone`,
/// the square is behind the background.
fn frame(k: i64, gone: &std::ops::Range<i64>) -> Gray {
    let (sx, sy) = square_at(k);
    let mut data = Vec::with_capacity((W * H) as usize);
    for y in 0..H {
        for x in 0..W {
            let (px, py) = (f64::from(x), f64::from(y));
            let value = if gone.contains(&k) {
                noise(px, py, 1)
            } else if (sx..sx + 40.0).contains(&px) && (sy..sy + 40.0).contains(&py) {
                noise(px - sx, py - sy, 2)
            } else {
                noise(px, py, 1)
            };
            data.push(value);
        }
    }
    Gray::new(W, H, data).unwrap()
}

/// Thirty-one frames a thirtieth of a second apart, the box drawn on
/// frame 10.
fn times() -> FrameTimes {
    let at = |k: i64| Micros(k * 1_000_000 / 30);
    FrameTimes {
        before: (0..10).rev().map(at).collect(),
        at: at(10),
        after: (11..31).map(at).collect(),
    }
}

fn index(time: Micros) -> i64 {
    (time.0 * 30 + 500_000) / 1_000_000
}

/// The square's box at frame `k` in the clip's own frame.
fn box_at(k: i64) -> [(f64, f64); 4] {
    let (sx, sy) = square_at(k);
    [
        (sx, sy),
        (sx + 40.0, sy),
        (sx + 40.0, sy + 40.0),
        (sx, sy + 40.0),
    ]
    .map(|(x, y)| (2.0 * x / f64::from(W) - 1.0, 1.0 - 2.0 * y / f64::from(H)))
}

/// The unit picture placed on the canvas as the clip's own frame.
fn same(_: Micros, quad: Quad) -> Option<[(f64, f64); 4]> {
    Some(quad.0.map(|(x, y)| (2.0 * x - 1.0, 1.0 - 2.0 * y)))
}

fn run(gone: std::ops::Range<i64>, stop_after: usize) -> Result<TrackPath, TrackError> {
    let mut calls = 0;
    track_frames(
        &times(),
        &mut |time| Ok(frame(index(time), &gone)),
        &same,
        box_at(10),
        TrackMode::Object,
        &mut |_| {
            calls += 1;
            calls < stop_after
        },
    )
}

#[test]
fn a_box_is_followed_both_ways_from_the_frame_it_was_drawn_on() {
    let path = run(0..0, usize::MAX).unwrap();
    assert_eq!(path.samples.len(), 31);
    for (k, sample) in path.samples.iter().enumerate() {
        let k = k as i64;
        assert_eq!(index(sample.time), k);
        let want = box_at(k);
        for (got, want) in sample.corners.iter().zip(want) {
            let off = ((got.0 - want.0) * f64::from(W) / 2.0)
                .hypot((got.1 - want.1) * f64::from(H) / 2.0);
            // The square is drawn with hard pixel edges at fractional places,
            // so its own box is only good to about a pixel.
            assert!(off < 2.5, "frame {k}: {off:.2} pixels off");
        }
    }
}

fn pixels_off(got: [(f64, f64); 4], want: [(f64, f64); 4]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(got, want)| {
            ((got.0 - want.0) * f64::from(W) / 2.0).hypot((got.1 - want.1) * f64::from(H) / 2.0)
        })
        .fold(0.0, f64::max)
}

#[test]
fn frames_where_the_target_hides_leave_a_gap_and_it_is_found_after() {
    let path = run(14..18, usize::MAX).unwrap();
    let frames: Vec<i64> = path.samples.iter().map(|s| index(s.time)).collect();
    assert!(frames.iter().all(|k| !(14..18).contains(k)), "{frames:?}");
    assert_eq!(frames.first(), Some(&0));
    assert_eq!(frames.last(), Some(&30));
    for sample in &path.samples {
        let k = index(sample.time);
        let off = pixels_off(sample.corners, box_at(k));
        assert!(off < 3.0, "frame {k}: {off:.2} pixels off");
    }
}

#[test]
fn a_span_keeps_one_side_of_the_drawn_frame() {
    let only = |span| times().only(span);
    assert_eq!(only(TrackSpan::After).before, Vec::new());
    assert_eq!(only(TrackSpan::After).after.len(), 20);
    assert_eq!(only(TrackSpan::Before).after, Vec::new());
    assert_eq!(only(TrackSpan::Before).before.len(), 10);
    assert_eq!(only(TrackSpan::Both).before.len(), 10);
}

#[test]
fn the_path_ends_where_the_target_is_lost() {
    let path = run(20..i64::MAX, usize::MAX).unwrap();
    let last = path.samples.last().map(|sample| index(sample.time));
    assert_eq!(last, Some(19));
    assert_eq!(
        path.samples.first().map(|sample| index(sample.time)),
        Some(0)
    );
}

#[test]
fn the_callback_stops_the_job() {
    assert_eq!(run(0..0, 4), Err(TrackError::Stopped));
}

#[test]
fn a_bad_box_or_frame_is_an_error() {
    let flat = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)];
    let result = track_frames(
        &times(),
        &mut |time| Ok(frame(index(time), &(0..0))),
        &same,
        flat,
        TrackMode::Plane,
        &mut |_| true,
    );
    assert!(matches!(result, Err(TrackError::Motion(_))));
    let result = track_frames(
        &times(),
        &mut |_| {
            Err(TrackError::Frame {
                reason: "unreadable".into(),
            })
        },
        &same,
        box_at(10),
        TrackMode::Object,
        &mut |_| true,
    );
    assert!(matches!(result, Err(TrackError::Frame { .. })));
}

fn clip_draft() -> Draft {
    let mut draft = Draft::empty("D");
    draft.canvas_config.width = 1920;
    draft.canvas_config.height = 1080;
    draft.materials.videos.push(MaterialVideo {
        id: "m".into(),
        width: 1920,
        height: 1080,
        ..MaterialVideo::default()
    });
    let mut segment = Segment::new("v", "m", TimeRange::new(0, 2_000_000));
    segment.clip.transform.x = 0.5;
    segment.uniform_scale.value = 0.5;
    draft.tracks[0].segments.push(segment);
    draft
}

#[test]
fn frames_are_read_at_the_project_rate_on_the_clip() {
    let draft = clip_draft();
    let segment = &draft.tracks[0].segments[0];
    let times = frame_times(&draft, segment, Micros(1_000_000)).unwrap();
    assert_eq!(times.before.len(), 30);
    assert_eq!(times.after.len(), 29);
    assert_eq!(times.before.first(), Some(&Micros(966_667)));
    assert!(times.before.iter().all(|t| t.0 >= 0));
    assert!(matches!(
        frame_times(&draft, segment, Micros(2_000_000)),
        Err(TrackError::Clip { .. })
    ));
}

#[test]
fn a_box_lands_on_the_canvas_where_the_clip_draws_it() {
    let draft = clip_draft();
    let segment = &draft.tracks[0].segments[0];
    // The whole picture, half size, moved right a quarter of the canvas.
    let whole = Quad([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
    let placed = on_canvas(&draft, segment, Micros(0), (1920, 1080), whole).unwrap();
    let want = [(0.0, 0.5), (1.0, 0.5), (1.0, -0.5), (0.0, -0.5)];
    for (got, want) in placed.iter().zip(want) {
        assert!(
            (got.0 - want.0).abs() < 1e-3 && (got.1 - want.1).abs() < 1e-3,
            "{got:?} {want:?}"
        );
    }
}

#[test]
fn a_turned_file_is_read_upright() {
    let mut data = vec![0.0; 16 * 20];
    data[0] = 9.0; // the top left of a 16 by 20 picture
    let gray = Gray::new(16, 20, data).unwrap();
    let turned = upright(gray.clone(), DisplayQuarter::Clockwise90).unwrap();
    assert_eq!((turned.width(), turned.height()), (20, 16));
    // Turned clockwise, the top left goes to the top right.
    assert_eq!(turned.data()[19], 9.0);
    assert_eq!(upright(gray.clone(), DisplayQuarter::Zero).unwrap(), gray);
}

#[test]
fn a_path_reads_in_the_clips_own_frame() {
    let mut draft = Draft::empty("D");
    draft.canvas_config.width = 1920;
    draft.canvas_config.height = 1080;
    draft.materials.videos.push(MaterialVideo {
        id: "m".into(),
        width: 1920,
        height: 1080,
        ..MaterialVideo::default()
    });
    let mut segment = Segment::new("v", "m", TimeRange::new(0, 1_000_000));
    // Half size, its middle half way to the right edge.
    segment.uniform_scale.value = 0.5;
    segment.clip.transform.x = 0.5;
    draft.tracks[0].segments.push(segment.clone());
    let path = TrackPath {
        mode: TrackMode::Object,
        samples: vec![TrackSample {
            time: Micros(0),
            corners: [(0.25, 0.5), (0.75, 0.5), (0.75, 0.0), (0.25, 0.0)],
        }],
    };
    let own = path_in_clip(&draft, &segment, &path).unwrap();
    let corners = own.samples[0].corners;
    let near = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).abs() < 1e-3 && (a.1 - b.1).abs() < 1e-3;
    assert!(near(corners[0], (-0.5, 1.0)), "{corners:?}");
    assert!(near(corners[2], (0.5, 0.0)), "{corners:?}");
}
