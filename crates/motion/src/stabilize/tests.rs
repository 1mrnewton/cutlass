//! Measures and steadies drawn scenes whose true camera path is known: a
//! textured world seen through a camera that slides and turns.

use super::*;
use crate::tests::noise;

pub(super) const W: u32 = 240;
pub(super) const H: u32 = 180;
const MIDDLE: (f64, f64) = (W as f64 / 2.0, H as f64 / 2.0);

/// The camera in frame `t`: how far the picture slid from frame 0, and how
/// far it turned. A slow pan right with a hand's shake on top.
pub(super) fn shaky(t: usize) -> ((f64, f64), f64) {
    let t = t as f64;
    (
        (
            0.6 * t + 3.0 * (1.7 * t).sin() + 2.0 * (3.1 * t + 1.0).sin(),
            2.0 * (2.3 * t).cos() - 2.0 + 1.5 * (4.1 * t).sin(),
        ),
        0.01 * (2.9 * t).sin(),
    )
}

/// The picture in frame `t` as a similarity from frame 0's pixels.
fn pose(camera: impl Fn(usize) -> ((f64, f64), f64), t: usize) -> Sim {
    let ((dx, dy), turn) = camera(t);
    Sim::about(MIDDLE, dx, dy, turn, 1.0)
}

/// Frame `t`: the world seen through the camera, and where `object` gives
/// a place, a second textured thing there instead.
fn frame(
    camera: impl Fn(usize) -> ((f64, f64), f64),
    t: usize,
    object: impl Fn(usize, f64, f64) -> Option<(f64, f64)>,
) -> Gray {
    let back = pose(&camera, t).inverse().unwrap();
    let mut data = Vec::with_capacity((W * H) as usize);
    for y in 0..H {
        for x in 0..W {
            let (px, py) = (f64::from(x), f64::from(y));
            let value = match object(t, px, py) {
                Some((u, v)) => noise(u, v, 9, 5.0),
                None => {
                    let w = back.apply((px, py));
                    noise(w.0, w.1, 1, 6.0)
                }
            };
            data.push(value);
        }
    }
    Gray::new(W, H, data).unwrap()
}

fn nothing(_: usize, _: f64, _: f64) -> Option<(f64, f64)> {
    None
}

/// The true move from frame `t - 1` to `t`.
fn truth(camera: impl Fn(usize) -> ((f64, f64), f64), t: usize) -> Move {
    let step = pose(&camera, t - 1)
        .inverse()
        .unwrap()
        .then(&pose(&camera, t));
    let to = step.apply(MIDDLE);
    Move {
        dx: to.0 - MIDDLE.0,
        dy: to.1 - MIDDLE.1,
        turn: step.b.atan2(step.a),
        scale: step.a.hypot(step.b),
        confidence: 1.0,
    }
}

fn assert_close(got: Move, want: Move, t: usize) {
    let off = (got.dx - want.dx).hypot(got.dy - want.dy);
    assert!(off < 0.25, "frame {t}: slide {off:.3} pixels off");
    assert!(
        (got.turn - want.turn).abs() < 0.002,
        "frame {t}: turn {} against {}",
        got.turn,
        want.turn
    );
    assert!(
        (got.scale - 1.0).abs() < 0.003,
        "frame {t}: scale {}",
        got.scale
    );
}

#[test]
fn a_shaking_camera_is_measured_frame_to_frame() {
    let mut camera = Camera::start(&frame(shaky, 0, nothing), 30.0).unwrap();
    for t in 1..=20 {
        let got = camera.step(&frame(shaky, t, nothing)).unwrap();
        assert_close(got, truth(shaky, t), t);
        assert!(
            got.confidence > 0.8,
            "frame {t}: confidence {}",
            got.confidence
        );
    }
    let path = camera.finish();
    assert_eq!((path.width, path.height, path.rate), (W, H, 30.0));
    assert_eq!(path.moves.len(), 20);
}

#[test]
fn someone_walking_through_does_not_move_the_camera() {
    // A box a fifth of the frame wide crossing left to right on its own.
    let walker = |t: usize, x: f64, y: f64| {
        let left = 30.0 + 6.0 * t as f64;
        ((left..left + 50.0).contains(&x) && (50.0..140.0).contains(&y)).then_some((x - left, y))
    };
    let mut camera = Camera::start(&frame(shaky, 0, walker), 30.0).unwrap();
    for t in 1..=12 {
        let got = camera.step(&frame(shaky, t, walker)).unwrap();
        assert_close(got, truth(shaky, t), t);
    }
}

#[test]
fn a_flat_frame_is_taken_as_still_and_the_path_goes_on() {
    let flat = Gray::new(W, H, vec![128.0; (W * H) as usize]).unwrap();
    let mut camera = Camera::start(&frame(shaky, 0, nothing), 30.0).unwrap();
    assert_eq!(camera.step(&flat).unwrap(), Move::STILL);
    assert_eq!(camera.step(&frame(shaky, 2, nothing)).unwrap(), Move::STILL);
    let got = camera.step(&frame(shaky, 3, nothing)).unwrap();
    assert_close(got, truth(shaky, 3), 3);
    assert_eq!(camera.path().moves.len(), 3);
}

#[test]
fn a_frame_of_another_size_or_a_bad_rate_is_refused() {
    let first = frame(shaky, 0, nothing);
    for rate in [0.0, -30.0, f64::NAN, f64::INFINITY, 5000.0] {
        assert!(matches!(
            Camera::start(&first, rate),
            Err(MotionError::Path { .. })
        ));
    }
    let mut camera = Camera::start(&first, 30.0).unwrap();
    let small = Gray::new(W / 2, H / 2, vec![0.0; (W * H / 4) as usize]).unwrap();
    assert!(matches!(
        camera.step(&small),
        Err(MotionError::Frame { .. })
    ));
    assert!(camera.path().moves.is_empty());
}

#[test]
fn points_are_spread_over_the_frame_and_capped() {
    let first = Pyramid::new(&frame(shaky, 0, nothing));
    let picked = spread(first.base());
    assert!(picked.len() <= MAX_SPREAD);
    assert!(picked.len() > MAX_SPREAD / 2, "{} points", picked.len());
    // Every quarter of the frame gets some.
    for (left, top) in [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)] {
        let inside = picked.iter().filter(|p| {
            let (x, y) = (f64::from(p.0) / f64::from(W), f64::from(p.1) / f64::from(H));
            (left..left + 0.5).contains(&x) && (top..top + 0.5).contains(&y)
        });
        assert!(inside.count() >= MAX_SPREAD / 8);
    }
}

#[test]
fn a_similarity_composes_and_inverts() {
    let a = Sim::about(MIDDLE, 3.0, -2.0, 0.1, 1.05);
    let b = Sim::about(MIDDLE, -1.0, 4.0, -0.05, 0.98);
    let p = (17.0, 140.0);
    let both = a.then(&b).apply(p);
    let step = b.apply(a.apply(p));
    assert!((both.0 - step.0).abs() < 1e-9 && (both.1 - step.1).abs() < 1e-9);
    let back = a.inverse().unwrap().apply(a.apply(p));
    assert!((back.0 - p.0).abs() < 1e-9 && (back.1 - p.1).abs() < 1e-9);
    let flat = Sim {
        a: 0.0,
        b: 0.0,
        tx: 1.0,
        ty: 1.0,
    };
    assert!(flat.inverse().is_none());
    assert_eq!(Sim::IDENTITY.apply(p), p);
}
