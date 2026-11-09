//! Pyramidal Lucas-Kanade: where a point in one frame went in the next,
//! from the coarsest level down, and the check that it comes back.

use crate::gray::{Level, Pyramid};

/// The half width of the window matched around a point.
const HALF: i32 = 7;

/// The most refinements on one level, and the step that ends them.
const ITERATIONS: usize = 20;
const SETTLED: f32 = 0.01;

/// The farthest, in pixels, a point may land from where it started when
/// followed forward and back.
pub(crate) const ROUND_TRIP: f32 = 1.0;

/// The points that came back: each one's index in `points`, and where it
/// went.
pub(crate) fn follow_all(
    prev: &Pyramid,
    next: &Pyramid,
    points: &[(f32, f32)],
) -> Vec<(usize, (f32, f32))> {
    points
        .iter()
        .enumerate()
        .filter_map(|(i, &p)| {
            let q = follow(prev, next, p)?;
            let back = follow(next, prev, q)?;
            let error = ((back.0 - p.0).powi(2) + (back.1 - p.1).powi(2)).sqrt();
            (error <= ROUND_TRIP).then_some((i, q))
        })
        .collect()
}

/// Where `p` in `prev` is in `next`, or `None` when its window has too
/// little texture or it lands outside the frame.
pub(crate) fn follow(prev: &Pyramid, next: &Pyramid, p: (f32, f32)) -> Option<(f32, f32)> {
    let levels = prev.levels.len().min(next.levels.len());
    let mut guess = (0.0_f32, 0.0_f32);
    let mut found = (0.0_f32, 0.0_f32);
    for level in (0..levels).rev() {
        let scale = (1_u32 << level) as f32;
        let at = (p.0 / scale, p.1 / scale);
        let step = refine(&prev.levels[level], &next.levels[level], at, guess)?;
        let d = (guess.0 + step.0, guess.1 + step.1);
        if level == 0 {
            found = d;
        } else {
            guess = (2.0 * d.0, 2.0 * d.1);
        }
    }
    let q = (p.0 + found.0, p.1 + found.1);
    let base = next.base();
    let inside = q.0.is_finite()
        && q.1.is_finite()
        && q.0 >= 0.0
        && q.1 >= 0.0
        && q.0 <= (base.width - 1) as f32
        && q.1 <= (base.height - 1) as f32;
    inside.then_some(q)
}

/// The move, beyond `guess`, that best lines `next` up with the window
/// around `at` in `prev`.
fn refine(prev: &Level, next: &Level, at: (f32, f32), guess: (f32, f32)) -> Option<(f32, f32)> {
    let side = (2 * HALF + 1) as usize;
    let mut template = Vec::with_capacity(side * side);
    let (mut gxx, mut gxy, mut gyy) = (0.0_f32, 0.0_f32, 0.0_f32);
    for dy in -HALF..=HALF {
        for dx in -HALF..=HALF {
            let (x, y) = (at.0 + dx as f32, at.1 + dy as f32);
            let (ix, iy) = (prev.sample_gx(x, y), prev.sample_gy(x, y));
            template.push((prev.sample(x, y), ix, iy));
            gxx += ix * ix;
            gxy += ix * iy;
            gyy += iy * iy;
        }
    }
    let det = gxx * gyy - gxy * gxy;
    let area = (side * side) as f32;
    // The smaller eigenvalue per pixel: too little texture to lock onto.
    let min_eigen = ((gxx + gyy) - ((gxx - gyy).powi(2) + 4.0 * gxy * gxy).sqrt()) / (2.0 * area);
    if !det.is_finite() || det.abs() < f32::EPSILON || min_eigen < 1e-2 {
        return None;
    }
    let mut v = (0.0_f32, 0.0_f32);
    for _ in 0..ITERATIONS {
        let (mut bx, mut by) = (0.0_f32, 0.0_f32);
        let mut i = 0;
        for dy in -HALF..=HALF {
            for dx in -HALF..=HALF {
                let (value, ix, iy) = template[i];
                i += 1;
                let x = at.0 + guess.0 + v.0 + dx as f32;
                let y = at.1 + guess.1 + v.1 + dy as f32;
                let diff = value - next.sample(x, y);
                bx += diff * ix;
                by += diff * iy;
            }
        }
        let eta = ((gyy * bx - gxy * by) / det, (gxx * by - gxy * bx) / det);
        if !eta.0.is_finite() || !eta.1.is_finite() {
            return None;
        }
        v = (v.0 + eta.0, v.1 + eta.1);
        if eta.0.abs() < SETTLED && eta.1.abs() < SETTLED {
            break;
        }
    }
    Some(v)
}
