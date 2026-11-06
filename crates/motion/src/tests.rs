//! Tracks on drawn scenes whose true motion is known: a textured object
//! moved, scaled, turned or tilted over a textured background that stays
//! still.

use super::*;
use crate::fit::{Model, homography};

const W: u32 = 240;
const H: u32 = 180;

/// Smooth value noise, 0 to 255, with cells `cell` pixels wide.
fn noise(x: f64, y: f64, seed: u64, cell: f64) -> f32 {
    let (gx, gy) = (x / cell, y / cell);
    let (ix, iy) = (gx.floor(), gy.floor());
    let (fx, fy) = (gx - ix, gy - iy);
    let smooth = |t: f64| t * t * (3.0 - 2.0 * t);
    let (sx, sy) = (smooth(fx), smooth(fy));
    let corner = |dx: f64, dy: f64| {
        let mut z = seed
            ^ ((ix + dx) as i64 as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ ((iy + dy) as i64 as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        (z >> 56) as f64
    };
    let top = corner(0.0, 0.0) * (1.0 - sx) + corner(1.0, 0.0) * sx;
    let bottom = corner(0.0, 1.0) * (1.0 - sx) + corner(1.0, 1.0) * sx;
    (top * (1.0 - sy) + bottom * sy) as f32
}

/// A frame: the background, and the object wherever `to_object` maps a
/// pixel inside its `size`.
fn frame(size: (f64, f64), to_object: impl Fn(f64, f64) -> Option<(f64, f64)>) -> Gray {
    let mut data = Vec::with_capacity((W * H) as usize);
    for y in 0..H {
        for x in 0..W {
            let (px, py) = (f64::from(x), f64::from(y));
            let value = match to_object(px, py) {
                Some((u, v)) if (0.0..size.0).contains(&u) && (0.0..size.1).contains(&v) => {
                    noise(u, v, 2, 4.0)
                }
                _ => noise(px, py, 1, 6.0),
            };
            data.push(value);
        }
    }
    Gray::new(W, H, data).unwrap()
}

const OBJECT: (f64, f64) = (70.0, 60.0);

/// The object's pose in frame `t`: middle, scale, turn.
fn pose(t: usize) -> ((f64, f64), f64, f64) {
    let t = t as f64;
    ((80.0 + 2.0 * t, 80.0 + 0.8 * t), 1.0 + 0.008 * t, 0.012 * t)
}

/// Where object point `p` is in frame `t`.
fn place(t: usize, p: (f64, f64)) -> (f64, f64) {
    let (c, s, a) = pose(t);
    let (u, v) = (p.0 - OBJECT.0 / 2.0, p.1 - OBJECT.1 / 2.0);
    (
        c.0 + s * (a.cos() * u - a.sin() * v),
        c.1 + s * (a.sin() * u + a.cos() * v),
    )
}

fn object_frame(t: usize) -> Gray {
    let (c, s, a) = pose(t);
    frame(OBJECT, |x, y| {
        let (dx, dy) = ((x - c.0) / s, (y - c.1) / s);
        Some((
            a.cos() * dx + a.sin() * dy + OBJECT.0 / 2.0,
            -a.sin() * dx + a.cos() * dy + OBJECT.1 / 2.0,
        ))
    })
}

/// The object's box, `pad` pixels bigger each side, in frame `t`.
fn object_quad(t: usize, pad: f64) -> Quad {
    let corners = [
        (-pad, -pad),
        (OBJECT.0 + pad, -pad),
        (OBJECT.0 + pad, OBJECT.1 + pad),
        (-pad, OBJECT.1 + pad),
    ];
    Quad(corners.map(|p| {
        let q = place(t, p);
        (q.0 / f64::from(W), q.1 / f64::from(H))
    }))
}

/// The farthest any corner is from the truth, in grey pixels.
fn error(a: Quad, b: Quad) -> f64 {
    a.0.iter()
        .zip(&b.0)
        .map(|(p, q)| ((p.0 - q.0) * f64::from(W)).hypot((p.1 - q.1) * f64::from(H)))
        .fold(0.0, f64::max)
}

fn tracked(step: Step) -> (Quad, f32) {
    match step {
        Step::Tracked { quad, confidence } => (quad, confidence),
        Step::Hidden => panic!("the target hid"),
        Step::Lost => panic!("lost the target"),
    }
}

#[test]
fn an_object_that_moves_grows_and_turns_is_followed() {
    let mut tracker = Tracker::start(&object_frame(0), object_quad(0, 0.0), Mode::Object).unwrap();
    for t in 1..=25 {
        let (quad, confidence) = tracked(tracker.step(&object_frame(t)).unwrap());
        let off = error(quad, object_quad(t, 0.0));
        assert!(off < 1.0, "frame {t}: {off:.2} pixels off");
        assert!(confidence > 0.5, "frame {t}: confidence {confidence}");
    }
}

#[test]
fn a_box_with_background_in_it_follows_the_object_not_the_background() {
    let mut tracker = Tracker::start(&object_frame(0), object_quad(0, 8.0), Mode::Object).unwrap();
    for t in 1..=20 {
        let (quad, _) = tracked(tracker.step(&object_frame(t)).unwrap());
        let off = error(quad, object_quad(t, 8.0));
        assert!(off < 2.0, "frame {t}: {off:.2} pixels off");
    }
}

#[test]
fn tracking_back_through_reversed_frames_returns_to_the_start() {
    let last = 15;
    let mut tracker =
        Tracker::start(&object_frame(last), object_quad(last, 0.0), Mode::Object).unwrap();
    for t in (0..last).rev() {
        tracked(tracker.step(&object_frame(t)).unwrap());
    }
    let off = error(tracker.quad(), object_quad(0, 0.0));
    assert!(off < 1.0, "{off:.2} pixels off");
}

/// A plane's four corners in frame `t`: it tilts back, its top edge
/// narrowing and dropping, while it slides right.
fn plane_corners(t: usize) -> [(f64, f64); 4] {
    let t = t as f64;
    [
        (60.0 + 1.2 * t, 40.0 + 0.8 * t),
        (170.0 - 0.4 * t, 45.0 + 0.6 * t),
        (175.0 + 0.6 * t, 140.0),
        (55.0 + 0.6 * t, 135.0 + 0.2 * t),
    ]
}

const PLANE: (f64, f64) = (100.0, 80.0);

fn plane_frame(t: usize) -> Gray {
    let rect = [
        (0.0, 0.0),
        (PLANE.0, 0.0),
        (PLANE.0, PLANE.1),
        (0.0, PLANE.1),
    ];
    let pairs: Vec<_> = plane_corners(t).into_iter().zip(rect).collect();
    let to_plane = homography(&pairs).unwrap();
    frame(PLANE, |x, y| to_plane.apply((x, y)))
}

fn plane_quad(t: usize) -> Quad {
    Quad(plane_corners(t).map(|(x, y)| (x / f64::from(W), y / f64::from(H))))
}

#[test]
fn a_tilting_plane_keeps_all_four_corners_in_plane_mode() {
    let mut plane = Tracker::start(&plane_frame(0), plane_quad(0), Mode::Plane).unwrap();
    let mut object = Tracker::start(&plane_frame(0), plane_quad(0), Mode::Object).unwrap();
    let (mut plane_off, mut object_off) = (0.0, 0.0);
    for t in 1..=20 {
        let frame = plane_frame(t);
        let (quad, confidence) = tracked(plane.step(&frame).unwrap());
        plane_off = error(quad, plane_quad(t));
        assert!(plane_off < 1.25, "frame {t}: {plane_off:.2} pixels off");
        assert!(confidence > 0.5, "frame {t}: confidence {confidence}");
        if let Step::Tracked { quad, .. } = object.step(&frame).unwrap() {
            object_off = error(quad, plane_quad(t));
        }
    }
    // A move, scale and turn cannot follow a tilt.
    assert!(
        object_off > 3.0 * plane_off.max(1.0),
        "{object_off} vs {plane_off}"
    );
}

/// A frame of only background, unlike the object's.
fn elsewhere() -> Gray {
    let blank: Vec<f32> = (0..(W * H) as usize)
        .map(|i| noise((i % W as usize) as f64, (i / W as usize) as f64, 9, 5.0))
        .collect();
    Gray::new(W, H, blank).unwrap()
}

#[test]
fn a_target_that_stays_away_is_hidden_then_lost_for_good() {
    let mut tracker = Tracker::start(&object_frame(0), object_quad(0, 0.0), Mode::Object).unwrap();
    tracked(tracker.step(&object_frame(1)).unwrap());
    let other = elsewhere();
    for _ in 0..MAX_HIDDEN {
        assert_eq!(tracker.step(&other).unwrap(), Step::Hidden);
    }
    assert_eq!(tracker.step(&other).unwrap(), Step::Lost);
    assert_eq!(tracker.step(&object_frame(2)).unwrap(), Step::Lost);
}

/// The object slid to `x`, `y` with no turn or scale, over the
/// background; `None` hides it behind the background.
fn slid_frame(at: Option<(f64, f64)>) -> Gray {
    frame(OBJECT, |x, y| at.map(|(ax, ay)| (x - ax, y - ay)))
}

fn slid_quad(at: (f64, f64)) -> Quad {
    Quad::from_box(
        at.0 / f64::from(W),
        at.1 / f64::from(H),
        OBJECT.0 / f64::from(W),
        OBJECT.1 / f64::from(H),
    )
}

#[test]
fn a_target_hidden_for_a_while_is_found_again_where_it_came_out() {
    let at = |t: usize| (40.0 + 3.0 * t as f64, 50.0 + 1.5 * t as f64);
    let mut tracker =
        Tracker::start(&slid_frame(Some(at(0))), slid_quad(at(0)), Mode::Object).unwrap();
    for t in 1..=3 {
        tracked(tracker.step(&slid_frame(Some(at(t)))).unwrap());
    }
    // Hidden behind the background for eight frames, still moving.
    for _ in 4..12 {
        assert_eq!(tracker.step(&slid_frame(None)).unwrap(), Step::Hidden);
    }
    // Found again about as close as tracking alone keeps it here.
    for t in 12..=18 {
        let (quad, confidence) = tracked(tracker.step(&slid_frame(Some(at(t)))).unwrap());
        let off = error(quad, slid_quad(at(t)));
        assert!(off < 3.0, "frame {t}: {off:.2} pixels off");
        assert!(confidence > 0.5, "frame {t}: confidence {confidence}");
    }
}

#[test]
fn a_turning_target_found_again_stays_close() {
    let mut tracker = Tracker::start(&object_frame(0), object_quad(0, 0.0), Mode::Object).unwrap();
    for t in 1..=3 {
        tracked(tracker.step(&object_frame(t)).unwrap());
    }
    let other = elsewhere();
    for _ in 0..4 {
        assert_eq!(tracker.step(&other).unwrap(), Step::Hidden);
    }
    for t in 8..=14 {
        let (quad, _) = tracked(tracker.step(&object_frame(t)).unwrap());
        let off = error(quad, object_quad(t, 0.0));
        assert!(off < 5.0, "frame {t}: {off:.2} pixels off");
    }
}

#[test]
fn a_track_is_the_same_every_run() {
    let run = || {
        let mut tracker = Tracker::start(&plane_frame(0), plane_quad(0), Mode::Plane).unwrap();
        (1..=6)
            .map(|t| tracker.step(&plane_frame(t)).unwrap())
            .collect::<Vec<_>>()
    };
    assert_eq!(run(), run());
}

#[test]
fn a_bad_target_or_frame_is_refused() {
    let first = object_frame(0);
    let flat = Gray::new(W, H, vec![128.0; (W * H) as usize]).unwrap();
    assert_eq!(
        Tracker::start(&flat, object_quad(0, 0.0), Mode::Object).err(),
        Some(MotionError::Featureless)
    );
    let refused = |quad: Quad| {
        matches!(
            Tracker::start(&first, quad, Mode::Object),
            Err(MotionError::Target { .. })
        )
    };
    assert!(refused(Quad::from_box(f64::NAN, 0.1, 0.2, 0.2)));
    assert!(refused(Quad::from_box(0.5, 0.5, 0.01, 0.01)), "too small");
    assert!(refused(Quad::from_box(1.5, 1.5, 0.3, 0.3)), "off the frame");
    assert!(
        refused(Quad([(0.1, 0.1), (0.5, 0.5), (0.5, 0.1), (0.1, 0.5)])),
        "folded"
    );
    let mut tracker = Tracker::start(&first, object_quad(0, 0.0), Mode::Object).unwrap();
    let small = Gray::new(W / 2, H / 2, vec![0.0; (W * H / 4) as usize]).unwrap();
    assert!(matches!(
        tracker.step(&small),
        Err(MotionError::Frame { .. })
    ));
    assert_eq!(tracker.mode(), Mode::Object);
    let _ = Model::Similarity {
        a: 1.0,
        b: 0.0,
        tx: 0.0,
        ty: 0.0,
    };
}
