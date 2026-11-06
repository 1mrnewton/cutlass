//! Picking corner points inside the target: places with texture in both
//! directions, which the point tracker can follow.

use crate::gray::Level;
use crate::quad::Corners;

/// The most points followed at once.
pub(crate) const MAX_POINTS: usize = 160;

/// Points stay this far from the frame's edge, so their windows fit.
const EDGE: usize = 8;

/// The half width of the window the corner score sums over.
const HALF: usize = 2;

/// The weakest corner kept, against the strongest in the target.
const QUALITY: f32 = 0.03;

/// A corner score below this is flat picture, however strong the rest is.
const MIN_SCORE: f32 = 25.0;

/// Corner points inside `corners`, strongest first, each at least `gap`
/// pixels from the others and from `taken`, until there are `limit`
/// points in all.
pub(crate) fn pick(
    level: &Level,
    corners: &Corners,
    taken: &[(f32, f32)],
    limit: usize,
) -> Vec<(f32, f32)> {
    let (w, h) = (level.width, level.height);
    if w <= 2 * EDGE || h <= 2 * EDGE || taken.len() >= limit {
        return Vec::new();
    }
    let (min_x, min_y, max_x, max_y) = corners.bounds();
    let x0 = (min_x.floor().max(EDGE as f64) as usize).min(w - EDGE);
    let x1 = (max_x.ceil().max(0.0) as usize).clamp(x0, w - EDGE);
    let y0 = (min_y.floor().max(EDGE as f64) as usize).min(h - EDGE);
    let y1 = (max_y.ceil().max(0.0) as usize).clamp(y0, h - EDGE);
    if x1 <= x0 || y1 <= y0 {
        return Vec::new();
    }

    // Sum the gradient products over each pixel's window, with summed
    // tables over the box around the target.
    let (bw, bh) = (x1 - x0 + 2 * HALF + 1, y1 - y0 + 2 * HALF + 1);
    let (ox, oy) = (x0 - HALF, y0 - HALF);
    let mut sums = [
        vec![0.0_f64; (bw + 1) * (bh + 1)],
        vec![0.0; (bw + 1) * (bh + 1)],
        vec![0.0; (bw + 1) * (bh + 1)],
    ];
    for y in 0..bh {
        for x in 0..bw {
            let i = (oy + y) * w + ox + x;
            let (gx, gy) = (f64::from(level.gx[i]), f64::from(level.gy[i]));
            for (table, value) in sums.iter_mut().zip([gx * gx, gx * gy, gy * gy]) {
                let at = (y + 1) * (bw + 1) + x + 1;
                table[at] = value + table[at - 1] + table[at - (bw + 1)] - table[at - bw - 2];
            }
        }
    }
    let window = |table: &[f64], x: usize, y: usize| {
        // The window around box pixel (x + HALF, y + HALF).
        let (left, top, right, bottom) = (x, y, x + 2 * HALF + 1, y + 2 * HALF + 1);
        table[bottom * (bw + 1) + right]
            - table[top * (bw + 1) + right]
            - table[bottom * (bw + 1) + left]
            + table[top * (bw + 1) + left]
    };

    let mut scored = Vec::new();
    let mut best = 0.0_f32;
    for y in y0..y1 {
        for x in x0..x1 {
            if !corners.contains(x as f64, y as f64) {
                continue;
            }
            let (bx, by) = (x - x0, y - y0);
            let a = window(&sums[0], bx, by);
            let b = window(&sums[1], bx, by);
            let c = window(&sums[2], bx, by);
            // The smaller eigenvalue of the structure tensor.
            let score = ((a + c) / 2.0 - (((a - c) / 2.0).powi(2) + b * b).sqrt()) as f32;
            if score > MIN_SCORE {
                best = best.max(score);
                scored.push((score, x, y));
            }
        }
    }
    scored.retain(|&(score, _, _)| score >= best * QUALITY);
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.2.cmp(&b.2)).then(a.1.cmp(&b.1)));

    // Spread the points: the gap shrinks with the target, so a small one
    // still gets enough.
    let area = corners.area().max(1.0);
    let gap = ((area / limit as f64).sqrt() * 0.7).clamp(3.0, 10.0) as f32;
    let mut picked: Vec<(f32, f32)> = Vec::new();
    let room = limit - taken.len();
    for (_, x, y) in scored {
        if picked.len() >= room {
            break;
        }
        let p = (x as f32, y as f32);
        let near = |q: &(f32, f32)| (q.0 - p.0).powi(2) + (q.1 - p.1).powi(2) < gap * gap;
        if taken.iter().any(near) || picked.iter().any(near) {
            continue;
        }
        picked.push(p);
    }
    picked
}
