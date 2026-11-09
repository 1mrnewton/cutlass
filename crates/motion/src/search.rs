//! The target's picture, and finding it again by normalised
//! cross-correlation after it was hidden.

use crate::gray::Pyramid;
use crate::quad::Corners;

/// The widest a kept picture is, on the level it is kept at.
const SIDE: f64 = 32.0;

/// The smallest kept picture side.
const MIN_SIDE: usize = 4;

/// How far around the last place to look, in box widths.
const REACH: f64 = 1.5;

/// The weakest correlation that counts as finding the target.
pub(crate) const MATCH: f32 = 0.6;

/// A picture flatter than this cannot be found again.
const MIN_SPREAD: f32 = 1.0;

/// How far a finer level may move from the coarser level's answer, in
/// that level's pixels.
const REFINE: usize = 2;

/// The target's picture on one pyramid level.
#[derive(Clone, Debug)]
struct Patch {
    level: usize,
    /// Its top left on that level.
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    /// The values less their mean, and their length.
    values: Vec<f32>,
    norm: f32,
}

/// The target's picture on the level where its box is at most [`SIDE`]
/// across and on every finer one, coarsest first: the coarse one finds
/// it, the finer ones place it to the pixel.
#[derive(Clone, Debug)]
pub(crate) struct Template {
    patches: Vec<Patch>,
}

/// The target's picture inside `corners`. `None` when it is too small or
/// flat.
pub(crate) fn capture(pyramid: &Pyramid, corners: &Corners) -> Option<Template> {
    let (left, top, right, bottom) = corners.bounds();
    let side = (right - left).max(bottom - top);
    if !side.is_finite() || side <= 0.0 {
        return None;
    }
    let mut level = 0;
    while level + 1 < pyramid.levels.len() && side / f64::from(1_u32 << level) > SIDE {
        level += 1;
    }
    let patches = (0..=level)
        .rev()
        .map(|level| patch(pyramid, level, (left, top, right, bottom)))
        .collect::<Option<Vec<_>>>()?;
    Some(Template { patches })
}

fn patch(pyramid: &Pyramid, level: usize, bounds: (f64, f64, f64, f64)) -> Option<Patch> {
    let (left, top, right, bottom) = bounds;
    let image = pyramid.levels.get(level)?;
    let scale = f64::from(1_u32 << level);
    let x0 = (left / scale).floor().max(0.0) as usize;
    let y0 = (top / scale).floor().max(0.0) as usize;
    let x1 = ((right / scale).ceil().max(0.0) as usize).min(image.width);
    let y1 = ((bottom / scale).ceil().max(0.0) as usize).min(image.height);
    if x1 < x0 + MIN_SIDE || y1 < y0 + MIN_SIDE {
        return None;
    }
    let (width, height) = (x1 - x0, y1 - y0);
    let mut values = Vec::with_capacity(width * height);
    for y in y0..y1 {
        values.extend_from_slice(&image.image[y * image.width + x0..y * image.width + x1]);
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    for v in &mut values {
        *v -= mean;
    }
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm / (values.len() as f32).sqrt() < MIN_SPREAD {
        return None;
    }
    Some(Patch {
        level,
        x: x0,
        y: y0,
        width,
        height,
        values,
        norm,
    })
}

/// Where `template` best matches in `pyramid` near where it was kept: the
/// move from there in full-size pixels, and the correlation on the
/// coarsest level. `None` when nothing matches [`MATCH`] or better.
pub(crate) fn find(pyramid: &Pyramid, template: &Template) -> Option<((f64, f64), f32)> {
    let coarse = template.patches.first()?;
    let reach = (coarse.width.max(coarse.height) as f64 * REACH).ceil() as isize;
    let ((mut dx, mut dy), score) = best(pyramid, coarse, (0, 0), reach)?;
    if score < MATCH {
        return None;
    }
    let mut fine = coarse;
    for next in &template.patches[1..] {
        let guess = (dx * 2, dy * 2);
        ((dx, dy), _) = best(pyramid, next, guess, REFINE as isize)?;
        fine = next;
    }
    // A parabola through the scores either side puts it between pixels.
    let around = |ox: isize, oy: isize| score_at(pyramid, fine, (dx + ox, dy + oy));
    let between = |minus: Option<f32>, middle: Option<f32>, plus: Option<f32>| {
        let (Some(m), Some(c), Some(p)) = (minus, middle, plus) else {
            return 0.0;
        };
        let curve = m - 2.0 * c + p;
        if curve >= -f32::EPSILON {
            return 0.0;
        }
        f64::from(((m - p) / (2.0 * curve)).clamp(-0.5, 0.5))
    };
    let middle = around(0, 0);
    let sub_x = between(around(-1, 0), middle, around(1, 0));
    let sub_y = between(around(0, -1), middle, around(0, 1));
    let scale = f64::from(1_u32 << fine.level);
    Some((
        ((dx as f64 + sub_x) * scale, (dy as f64 + sub_y) * scale),
        score,
    ))
}

/// The move within `reach` of `guess` where `patch` matches best, in its
/// level's pixels, and the score there.
fn best(
    pyramid: &Pyramid,
    patch: &Patch,
    guess: (isize, isize),
    reach: isize,
) -> Option<((isize, isize), f32)> {
    let mut best: Option<((isize, isize), f32)> = None;
    for dy in guess.1 - reach..=guess.1 + reach {
        for dx in guess.0 - reach..=guess.0 + reach {
            let Some(score) = score_at(pyramid, patch, (dx, dy)) else {
                continue;
            };
            if best.is_none_or(|(_, b)| score > b) {
                best = Some(((dx, dy), score));
            }
        }
    }
    best
}

/// The correlation of `patch` moved by `shift` on its level, or `None`
/// off the picture or on a flat window.
fn score_at(pyramid: &Pyramid, patch: &Patch, shift: (isize, isize)) -> Option<f32> {
    let image = pyramid.levels.get(patch.level)?;
    let x = usize::try_from(patch.x as isize + shift.0).ok()?;
    let y = usize::try_from(patch.y as isize + shift.1).ok()?;
    if x + patch.width > image.width || y + patch.height > image.height {
        return None;
    }
    let count = (patch.width * patch.height) as f32;
    let (mut sum, mut squares, mut cross) = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut values = patch.values.iter();
    for row in y..y + patch.height {
        let start = row * image.width + x;
        for (&value, &kept) in image.image[start..start + patch.width]
            .iter()
            .zip(&mut values)
        {
            sum += value;
            squares += value * value;
            cross += value * kept;
        }
    }
    // The patch has zero mean, so the window's mean drops out of the
    // cross term.
    let spread = (squares - sum * sum / count).max(0.0).sqrt();
    if spread / count.sqrt() < MIN_SPREAD {
        return None;
    }
    let score = cross / (spread * patch.norm);
    score.is_finite().then_some(score)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gray;

    fn noise(w: u32, h: u32, seed: u32) -> Vec<f32> {
        (0..w * h)
            .map(|i| {
                let (x, y) = (i % w / 3, i / w / 3);
                let mut z = x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663) ^ seed;
                z ^= z >> 13;
                z = z.wrapping_mul(0x5bd1_e995);
                (z >> 24) as f32
            })
            .collect()
    }

    #[test]
    fn a_kept_picture_is_found_where_it_moved() {
        let (w, h) = (160_u32, 120_u32);
        let first = Gray::new(w, h, noise(w, h, 1)).unwrap();
        let box_at =
            |x: f64, y: f64| Corners([(x, y), (x + 30.0, y), (x + 30.0, y + 30.0), (x, y + 30.0)]);
        let template = capture(&Pyramid::new(&first), &box_at(50.0, 40.0)).unwrap();
        // The whole picture slides 9 right and 6 down.
        let moved: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                if x >= 9 && y >= 6 {
                    first.data()[((y - 6) * w + x - 9) as usize]
                } else {
                    0.0
                }
            })
            .collect();
        let next = Gray::new(w, h, moved).unwrap();
        let ((dx, dy), score) = find(&Pyramid::new(&next), &template).unwrap();
        assert!(
            (dx - 9.0).abs() < 0.3 && (dy - 6.0).abs() < 0.3,
            "{dx} {dy}"
        );
        assert!(score > 0.99);
        // Another picture has no match.
        let other = Gray::new(w, h, noise(w, h, 2)).unwrap();
        assert!(find(&Pyramid::new(&other), &template).is_none());
    }

    #[test]
    fn a_flat_or_tiny_box_keeps_no_picture() {
        let flat = Gray::new(64, 64, vec![50.0; 64 * 64]).unwrap();
        let square = Corners([(10.0, 10.0), (40.0, 10.0), (40.0, 40.0), (10.0, 40.0)]);
        assert!(capture(&Pyramid::new(&flat), &square).is_none());
        let textured = Gray::new(64, 64, noise(64, 64, 3)).unwrap();
        let tiny = Corners([(10.0, 10.0), (12.0, 10.0), (12.0, 12.0), (10.0, 12.0)]);
        assert!(capture(&Pyramid::new(&textured), &tiny).is_none());
    }
}
