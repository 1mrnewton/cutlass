//! Steadies paths whose true camera is known, without drawing frames.

use super::*;

const W: u32 = 240;
const H: u32 = 180;
const MIDDLE: (f64, f64) = (W as f64 / 2.0, H as f64 / 2.0);
const RATE: f64 = 30.0;

/// The camera in frame `t`: a slow pan right with a hand's shake of
/// `shake` pixels on top, and a small turn.
fn camera(t: usize, shake: f64) -> Sim {
    let t = t as f64;
    Sim::about(
        MIDDLE,
        0.4 * t + shake * ((1.7 * t).sin() + 0.6 * (3.1 * t + 1.0).sin()),
        shake * (0.7 * (2.3 * t).cos() - 0.7 + 0.5 * (4.1 * t).sin()),
        0.004 * shake * (2.9 * t).sin(),
        1.0,
    )
}

/// The moves between `frames` frames of [`camera`].
fn path(frames: usize, shake: f64) -> Path {
    let moves = (1..frames)
        .map(|t| {
            let step = camera(t - 1, shake)
                .inverse()
                .unwrap()
                .then(&camera(t, shake));
            let to = step.apply(MIDDLE);
            Move {
                dx: to.0 - MIDDLE.0,
                dy: to.1 - MIDDLE.1,
                turn: step.b.atan2(step.a),
                scale: step.a.hypot(step.b),
                confidence: 1.0,
            }
        })
        .collect();
    Path {
        width: W,
        height: H,
        rate: RATE,
        moves,
    }
}

/// `correction` as a similarity in grey pixels.
fn sim(correction: &Correction) -> Sim {
    Sim::about(
        MIDDLE,
        correction.x * f64::from(W),
        correction.y * f64::from(H),
        correction.turn,
        correction.scale,
    )
}

/// Where the first frame's middle shows on screen in each frame once
/// steadied by `plan`.
fn steadied(plan: &Plan, shake: f64) -> Vec<(f64, f64)> {
    plan.frames
        .iter()
        .enumerate()
        .map(|(t, c)| {
            let at = camera(t, shake).then(&sim(c)).apply(MIDDLE);
            (at.0 / plan.zoom, at.1 / plan.zoom)
        })
        .collect()
}

fn raw(frames: usize, shake: f64) -> Vec<(f64, f64)> {
    (0..frames)
        .map(|t| camera(t, shake).apply(MIDDLE))
        .collect()
}

/// How much a path jerks: the summed size of its second differences.
fn jitter(path: &[(f64, f64)]) -> f64 {
    path.windows(3)
        .map(|w| (w[0].0 - 2.0 * w[1].0 + w[2].0).hypot(w[0].1 - 2.0 * w[1].1 + w[2].1))
        .sum()
}

/// Whether every frame, steadied, still covers the whole canvas.
fn covers(plan: &Plan) -> bool {
    plan.frames.iter().all(|c| {
        let back = sim(c).inverse().unwrap();
        [(0.0, 0.0), (240.0, 0.0), (240.0, 180.0), (0.0, 180.0)]
            .iter()
            .all(|&q| {
                let p = back.apply(q);
                (-1e-6..=f64::from(W) + 1e-6).contains(&p.0)
                    && (-1e-6..=f64::from(H) + 1e-6).contains(&p.1)
            })
    })
}

#[test]
fn a_shaky_pan_comes_out_smooth_and_still_pans() {
    let frames = 90;
    let plan = plan(&path(frames, 3.0), Strength::Recommended).unwrap();
    assert_eq!(plan.frames.len(), frames);
    let (before, after) = (raw(frames, 3.0), steadied(&plan, 3.0));
    let (shaky, smooth) = (jitter(&before), jitter(&after));
    assert!(smooth < shaky * 0.15, "jitter {smooth:.1} from {shaky:.1}");
    // The pan is kept, not locked off.
    let panned = after[frames - 1].0 - after[0].0;
    let truth = before[frames - 1].0 - before[0].0;
    assert!(
        (panned - truth).abs() < truth * 0.25,
        "{panned:.1} of {truth:.1}"
    );
    assert!(plan.zoom > 1.0 && plan.zoom <= Strength::Recommended.max_zoom());
    assert!(covers(&plan));
}

#[test]
fn a_still_camera_needs_no_correction() {
    let still = Path {
        width: W,
        height: H,
        rate: RATE,
        moves: vec![
            Move {
                confidence: 1.0,
                ..Move::STILL
            };
            40
        ],
    };
    for strength in [
        Strength::MinimumCut,
        Strength::Recommended,
        Strength::MostStable,
    ] {
        let plan = plan(&still, strength).unwrap();
        assert_eq!(plan.zoom, 1.0);
        assert!(plan.frames.iter().all(|c| {
            c.x.abs() < 1e-9
                && c.y.abs() < 1e-9
                && c.turn.abs() < 1e-9
                && (c.scale - 1.0).abs() < 1e-9
        }));
    }
}

#[test]
fn the_zoom_never_passes_the_strengths_cap_and_hides_every_edge() {
    // Shake far past what any cap can hide.
    let frames = 120;
    let wild = path(frames, 25.0);
    for strength in [
        Strength::MinimumCut,
        Strength::Recommended,
        Strength::MostStable,
    ] {
        let plan = plan(&wild, strength).unwrap();
        assert!(
            plan.zoom <= strength.max_zoom() + 1e-9,
            "{strength:?} zoomed {}",
            plan.zoom
        );
        assert!(covers(&plan), "{strength:?} shows an edge");
        let smooth = jitter(&steadied(&plan, 25.0));
        assert!(smooth < jitter(&raw(frames, 25.0)), "{strength:?}");
    }
}

#[test]
fn more_stable_is_smoother_and_cuts_more() {
    let shaky = path(150, 2.0);
    let light = plan(&shaky, Strength::MinimumCut).unwrap();
    let heavy = plan(&shaky, Strength::MostStable).unwrap();
    assert!(jitter(&steadied(&heavy, 2.0)) < jitter(&steadied(&light, 2.0)));
    assert!(heavy.zoom > light.zoom);
}

#[test]
fn a_new_strength_needs_no_new_measure_and_repeats_exactly() {
    let shaky = path(60, 3.0);
    let first = plan(&shaky, Strength::Recommended).unwrap();
    assert_eq!(plan(&shaky, Strength::Recommended).unwrap(), first);
    assert_ne!(plan(&shaky, Strength::MinimumCut).unwrap(), first);
}

#[test]
fn an_empty_path_is_one_still_frame() {
    let empty = Path {
        width: W,
        height: H,
        rate: RATE,
        moves: Vec::new(),
    };
    let plan = plan(&empty, Strength::default()).unwrap();
    assert_eq!(plan.zoom, 1.0);
    assert_eq!(plan.frames, vec![Correction::NONE]);
}

#[test]
fn a_hostile_path_is_refused_or_read_as_still() {
    let mut bad = path(10, 3.0);
    bad.rate = f64::NAN;
    assert!(matches!(
        plan(&bad, Strength::default()),
        Err(MotionError::Path { .. })
    ));
    let mut bad = path(10, 3.0);
    bad.width = 0;
    assert!(matches!(
        plan(&bad, Strength::default()),
        Err(MotionError::Path { .. })
    ));
    // Moves that are not numbers or no camera could make count as still.
    let wild = [f64::NAN, f64::INFINITY, 1e9];
    let mut odd = path(4, 0.0);
    let mut still = odd.clone();
    for (i, value) in wild.into_iter().enumerate() {
        odd.moves[i].dx = value;
        still.moves[i] = Move::STILL;
    }
    assert_eq!(
        plan(&odd, Strength::default()).unwrap(),
        plan(&still, Strength::default()).unwrap()
    );
    let mut flipped = path(4, 0.0);
    flipped.moves[0].scale = -1.0;
    flipped.moves[1].turn = 3.0;
    let mut still = flipped.clone();
    still.moves[0] = Move::STILL;
    still.moves[1] = Move::STILL;
    assert_eq!(
        plan(&flipped, Strength::default()).unwrap(),
        plan(&still, Strength::default()).unwrap()
    );
}

#[test]
fn a_long_clip_plans_in_one_pass_per_window() {
    // Ten minutes at 60 frames a second: the blur and the least are the
    // window's width per frame, never the clip's.
    let long = Path {
        width: W,
        height: H,
        rate: 60.0,
        moves: vec![
            Move {
                dx: 0.5,
                confidence: 1.0,
                ..Move::STILL
            };
            36_000
        ],
    };
    let started = std::time::Instant::now();
    let plan = plan(&long, Strength::MostStable).unwrap();
    assert_eq!(plan.frames.len(), 36_001);
    assert!(started.elapsed().as_secs() < 20, "{:?}", started.elapsed());
}

#[test]
fn the_blur_keeps_a_straight_line_and_the_least_takes_the_window() {
    let line: Vec<f64> = (0..20).map(|i| 2.0 * f64::from(i) + 1.0).collect();
    let blurred = blur(&line, 3.0, 9, true);
    for (a, b) in line.iter().zip(&blurred) {
        assert!((a - b).abs() < 1e-9, "{a} {b}");
    }
    assert_eq!(
        least(&[5.0, 1.0, 4.0, 4.0, 3.0, 9.0], 1),
        vec![1.0, 1.0, 1.0, 3.0, 3.0, 3.0]
    );
    // Repeating the ends never lifts a value above the least near it.
    let shares = [1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0];
    let eased = blur(&least(&shares, 2), 1.0, 2, false);
    for (s, e) in shares.iter().zip(&eased) {
        assert!(e <= s);
    }
}
