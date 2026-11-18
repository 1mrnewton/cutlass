//! A soft blur for one-channel masks: a title's shadow and glow, and a
//! mask's feather.

/// The most a mask is made smaller by, each way, to be blurred.
const MAX_SHRINK: usize = 8;
/// The widest box summed pixel by pixel; a wider one keeps a running sum.
const SHORT_REACH: usize = 4;

/// Three box blurs, close to a gaussian `radius` pixels wide. A wide blur
/// is made on a mask a half, a quarter or an eighth the size and drawn
/// back up: what it gives is smooth enough that the two differ by about
/// a hundredth of full cover at most, and a title whose letters move has
/// its shadow and its glow blurred again at every frame.
pub fn blur_mask(mask: &[f32], width: usize, height: usize, radius: f32) -> Vec<f32> {
    let whole = width.checked_mul(height);
    if radius.is_nan()
        || radius < 0.5
        || width == 0
        || height == 0
        || whole.is_none_or(|whole| mask.len() < whole)
    {
        return mask.to_vec();
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let reach = ((radius / 2.0).round() as usize).clamp(1, width.max(height));
    // Three boxes reaching `reach` spread a pixel as a gaussian whose
    // width squared is this.
    let spread = pixel(reach) * pixel(reach + 1);
    // Drawn back up between pixels `shrink` apart, a curve this wide is
    // off by about `shrink` squared over thirty-three times `spread`.
    let mut shrink = 1;
    while shrink < MAX_SHRINK && pixel(shrink * shrink * 4) <= 0.2 * spread {
        shrink *= 2;
    }
    if shrink == 1 {
        let mut out = mask.to_vec();
        boxes(&mut out, width, height, reach, 0.0);
        return out;
    }
    let (small_w, small_h) = (width.div_ceil(shrink), height.div_ceil(shrink));
    let mut small = smaller(mask, width, height, shrink);
    // Making the mask smaller and drawing it back up each spread it a
    // little; the boxes make up the rest, in the small mask's pixels.
    let across = pixel(shrink * shrink);
    let left = ((spread - (across - 1.0) / 12.0 - across / 6.0) / across).max(0.0);
    let (reach, edge) = fitted(left / 3.0);
    boxes(&mut small, small_w, small_h, reach, edge);
    larger(&small, small_w, small_h, shrink, width, height)
}

/// The box that spreads a pixel by `spread`, a third of what three of
/// them do: how far it reaches in whole pixels, and how much it counts
/// of the pixel one further at each end, from nothing to all of it. A
/// box of whole pixels alone could only come near.
fn fitted(spread: f32) -> (usize, f32) {
    if spread.is_nan() || spread <= 0.0 {
        return (0, 0.0);
    }
    // A box reaching `reach` spreads by a third of `reach` times
    // `reach + 1`.
    let mut reach = 0_usize;
    while reach < 4_096 && pixel((reach + 1) * (reach + 2)) / 3.0 <= spread {
        reach += 1;
    }
    let whole = pixel(reach * (reach + 1)) / 3.0;
    let next = pixel((reach + 1) * (reach + 1));
    let edge = pixel(reach * 2 + 1) * (spread - whole) / (2.0 * (next - spread));
    (
        reach,
        if edge.is_finite() {
            edge.clamp(0.0, 1.0)
        } else {
            0.0
        },
    )
}

/// Blur `mask` in place with three boxes, each reaching `reach` pixels
/// each way and counting `edge` of the pixel one further, along the rows
/// and then down the columns. Past the mask's edge is empty.
fn boxes(mask: &mut [f32], width: usize, height: usize, reach: usize, edge: f32) {
    if reach == 0 && edge <= 0.0 {
        return;
    }
    let scale = 1.0 / (pixel(reach * 2 + 1) + edge * 2.0);
    let mut other = vec![0.0_f32; mask.len()];
    let mut sums = vec![0.0_f32; width];
    let row = |at: usize| at * width..(at + 1) * width;
    for _ in 0..3 {
        for (from, to) in mask
            .chunks_exact(width)
            .zip(other.chunks_exact_mut(width))
            .take(height)
        {
            across(from, to, reach, edge, scale);
        }
        // Down the columns, every column's sum kept side by side so a
        // row is added and taken away whole.
        sums.fill(0.0);
        for at in 0..reach.min(height) {
            for (sum, value) in sums.iter_mut().zip(&other[row(at)]) {
                *sum += value;
            }
        }
        for at in 0..height {
            if at + reach < height {
                for (sum, value) in sums.iter_mut().zip(&other[row(at + reach)]) {
                    *sum += value;
                }
            }
            for (slot, sum) in mask[row(at)].iter_mut().zip(&sums) {
                *slot = *sum;
            }
            if edge > 0.0 {
                // The rows one past the box, each counted in part.
                let ends = [at.checked_sub(reach + 1), Some(at + reach + 1)];
                for end in ends.into_iter().flatten().filter(|end| *end < height) {
                    for (slot, value) in mask[row(at)].iter_mut().zip(&other[row(end)]) {
                        *slot += value * edge;
                    }
                }
            }
            for slot in &mut mask[row(at)] {
                *slot = (*slot * scale).clamp(0.0, 1.0);
            }
            if let Some(leaving) = at.checked_sub(reach) {
                for (sum, value) in sums.iter_mut().zip(&other[row(leaving)]) {
                    *sum -= value;
                }
            }
        }
    }
}

/// One row through one box: each pixel of `to` the mean of the pixels of
/// `from` within `reach` of it, with `edge` of the pixel one further at
/// each end, counting what is past either end of the row as empty.
fn across(from: &[f32], to: &mut [f32], reach: usize, edge: f32, scale: f32) {
    let width = from.len();
    // How far the box reaches with its part pixels.
    let outer = reach + usize::from(edge > 0.0);
    let at = |index: Option<usize>| index.and_then(|index| from.get(index)).copied();
    let ends = |index: usize| {
        if edge > 0.0 {
            let before = at(index.checked_sub(reach + 1)).unwrap_or(0.0);
            (before + at(Some(index + reach + 1)).unwrap_or(0.0)) * edge
        } else {
            0.0
        }
    };
    if outer <= SHORT_REACH && width > outer * 2 {
        // A narrow box is summed a neighbour at a time, every pixel of
        // the row at once.
        let inner = width - outer * 2;
        let (head, rest) = to.split_at_mut(outer);
        let (middle, tail) = rest.split_at_mut(inner);
        middle.fill(0.0);
        let first = outer - reach;
        for offset in first..=first + reach * 2 {
            for (slot, value) in middle.iter_mut().zip(&from[offset..offset + inner]) {
                *slot += value;
            }
        }
        if edge > 0.0 {
            for offset in [0, outer * 2] {
                for (slot, value) in middle.iter_mut().zip(&from[offset..offset + inner]) {
                    *slot += value * edge;
                }
            }
        }
        for slot in middle.iter_mut() {
            *slot = (*slot * scale).clamp(0.0, 1.0);
        }
        let whole = |index: usize| {
            let low = index.saturating_sub(reach);
            let high = (index + reach + 1).min(width);
            from[low..high].iter().sum::<f32>() + ends(index)
        };
        for (index, slot) in head.iter_mut().enumerate() {
            *slot = (whole(index) * scale).clamp(0.0, 1.0);
        }
        for (index, slot) in tail.iter_mut().enumerate() {
            *slot = (whole(outer + inner + index) * scale).clamp(0.0, 1.0);
        }
        return;
    }
    let mut sum: f32 = from.iter().take(reach).sum();
    for (index, slot) in to.iter_mut().enumerate() {
        if let Some(entering) = from.get(index + reach) {
            sum += entering;
        }
        *slot = ((sum + ends(index)) * scale).clamp(0.0, 1.0);
        if let Some(leaving) = at(index.checked_sub(reach)) {
            sum -= leaving;
        }
    }
}

/// `mask` with each square of `shrink` by `shrink` pixels as one, their
/// mean; a square that hangs over the edge counts what is past it as
/// empty.
pub fn smaller(mask: &[f32], width: usize, height: usize, shrink: usize) -> Vec<f32> {
    let (small_w, small_h) = (width.div_ceil(shrink), height.div_ceil(shrink));
    let scale = 1.0 / pixel(shrink * shrink);
    let mut small = vec![0.0_f32; small_w * small_h];
    for (y, row) in mask.chunks_exact(width).take(height).enumerate() {
        let to = &mut small[(y / shrink) * small_w..][..small_w];
        for (slot, square) in to.iter_mut().zip(row.chunks(shrink)) {
            *slot += square.iter().sum::<f32>();
        }
    }
    for value in &mut small {
        *value *= scale;
    }
    small
}

/// A mask `small_w` by `small_h` drawn `shrink` times the size, into one
/// `width` by `height`, each pixel mixed from the four nearest. A pixel
/// past the middle of the last small one, at any side, carries on the
/// slope of the two nearest.
pub fn larger(
    small: &[f32],
    small_w: usize,
    small_h: usize,
    shrink: usize,
    width: usize,
    height: usize,
) -> Vec<f32> {
    // Where a pixel's middle falls among the small mask's: the one
    // before it, the one after, and how far it is toward the second.
    let place = |at: usize, count: usize| {
        let along = (pixel(at) + 0.5) / pixel(shrink) - 0.5;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let first = (along.max(0.0) as usize).min(count.saturating_sub(2));
        (first, (first + 1).min(count - 1), along - pixel(first))
    };
    // Each small row drawn out across the width, then the rows mixed.
    let columns: Vec<(usize, usize, f32)> = (0..width).map(|x| place(x, small_w)).collect();
    let mut wide = vec![0.0_f32; width * small_h];
    for (from, to) in small
        .chunks_exact(small_w)
        .zip(wide.chunks_exact_mut(width))
    {
        for (slot, (first, second, toward)) in to.iter_mut().zip(&columns) {
            *slot = from[*first] + (from[*second] - from[*first]) * toward;
        }
    }
    let mut out = vec![0.0_f32; width * height];
    for (y, to) in out.chunks_exact_mut(width).enumerate() {
        let (first, second, toward) = place(y, small_h);
        let (above, below) = (
            &wide[first * width..][..width],
            &wide[second * width..][..width],
        );
        for ((slot, above), below) in to.iter_mut().zip(above).zip(below) {
            *slot = (above + (below - above) * toward).clamp(0.0, 1.0);
        }
    }
    out
}

fn pixel(value: usize) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    {
        value as f32
    }
}

#[cfg(test)]
mod tests {
    use super::{blur_mask as blur, boxes, fitted};

    /// A mask of letter-like shapes: bars, a ring and a few dots, hard
    /// edged, with room around them.
    fn shapes(width: usize, height: usize) -> Vec<f32> {
        let mut mask = vec![0.0_f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let bar = (x / 37) % 2 == 0 && (20..height.saturating_sub(20)).contains(&y);
                let (cx, cy) = (x.abs_diff(width / 2), y.abs_diff(height / 2));
                let ring = (900..1600).contains(&(cx * cx + cy * cy));
                let dot = x % 53 < 3 && y % 29 < 3;
                if (bar && x > 30 && x + 30 < width) || ring || dot {
                    mask[y * width + x] = 1.0;
                }
            }
        }
        mask
    }

    /// The blur made the plain way: three boxes on the mask at its size.
    fn plain(mask: &[f32], width: usize, height: usize, radius: f32) -> Vec<f32> {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let reach = ((radius / 2.0).round() as usize).max(1);
        let mut out = mask.to_vec();
        boxes(&mut out, width, height, reach, 0.0);
        out
    }

    #[test]
    fn a_wide_blur_made_small_is_the_blur_made_at_full_size() {
        let (width, height) = (301, 157);
        let mask = shapes(width, height);
        let mut widest = 0.0_f32;
        for radius in [
            1.0_f32, 3.0, 6.0, 9.0, 13.0, 18.0, 21.6, 30.0, 47.0, 64.0, 120.0,
        ] {
            let quick = blur(&mask, width, height, radius);
            let slow = plain(&mask, width, height, radius);
            assert_eq!(quick.len(), slow.len());
            let off = quick
                .iter()
                .zip(&slow)
                .map(|(quick, slow)| (quick - slow).abs())
                .fold(0.0_f32, f32::max);
            // By about a hundredth of full cover at most, at the picture's
            // edge; less than that inside it.
            assert!(off < 0.012, "radius {radius}: off by {off}");
            widest = widest.max(off);
            // What was there is still there: a blur moves cover about
            // and loses only what goes past the edge.
            let (kept, had): (f32, f32) = (quick.iter().sum(), slow.iter().sum());
            assert!(
                (kept - had).abs() < 0.02 * had,
                "radius {radius}: {kept} of {had}"
            );
            assert!(quick.iter().all(|cover| (0.0..=1.0).contains(cover)));
        }
        assert!(widest > 0.0, "no blur was made small");
        // A narrow blur is made at full size, the same to the bit.
        for radius in [0.6_f32, 1.0, 2.0, 3.0] {
            assert_eq!(
                blur(&mask, width, height, radius),
                plain(&mask, width, height, radius)
            );
        }
    }

    #[test]
    fn a_narrow_box_summed_by_neighbours_is_the_running_sum() {
        // Reaches up to four are summed a neighbour at a time; the same
        // row through the running sum gives the same means.
        let row: Vec<f32> = (0..97_u8).map(|at| f32::from(at % 7) / 6.0).collect();
        for reach in 1..=6_usize {
            #[allow(clippy::cast_precision_loss)]
            let scale = 1.0 / (reach * 2 + 1) as f32;
            let mut quick = vec![0.0; row.len()];
            super::across(&row, &mut quick, reach, 0.0, scale);
            for (at, mean) in quick.iter().enumerate() {
                let from = at.saturating_sub(reach);
                let to = (at + reach + 1).min(row.len());
                let wanted = row[from..to].iter().sum::<f32>() * scale;
                assert!((mean - wanted).abs() < 1e-5, "reach {reach} at {at}");
            }
        }
        // A row shorter than the box.
        let mut short = vec![0.0; 3];
        super::across(&[1.0, 1.0, 1.0], &mut short, 4, 0.0, 1.0 / 9.0);
        assert!(short.iter().all(|mean| (mean - 3.0 / 9.0).abs() < 1e-6));
    }

    #[test]
    fn a_box_is_fitted_to_spread_as_far_as_asked() {
        // Whole boxes: reaching one spreads by two thirds, two by two.
        assert_eq!(fitted(0.0), (0, 0.0));
        let (reach, edge) = fitted(2.0 / 3.0);
        assert_eq!(reach, 1);
        assert!(edge.abs() < 1e-5);
        let (reach, edge) = fitted(2.0);
        assert_eq!(reach, 2);
        assert!(edge.abs() < 1e-5);
        // Between two, the pixel one further counts in part, and the box
        // spreads by what was asked: the mean square of how far its
        // pixels are from its middle.
        for spread in [0.1_f32, 0.5, 1.0, 1.5, 3.0, 7.7, 40.0] {
            let (reach, edge) = fitted(spread);
            assert!((0.0..=1.0).contains(&edge), "{spread}");
            let mut squares = 0.0_f32;
            let mut weight = 1.0_f32;
            for step in 1..=reach + 1 {
                #[allow(clippy::cast_precision_loss)]
                let far = step as f32;
                let counted = if step > reach { edge } else { 1.0 };
                squares += 2.0 * counted * far * far;
                weight += 2.0 * counted;
            }
            assert!(
                (squares / weight - spread).abs() < 1e-3 * spread.max(1.0),
                "{spread}"
            );
        }
        // No spread that is no number, and one no mask is wide enough for.
        assert_eq!(fitted(f32::NAN), (0, 0.0));
        assert_eq!(fitted(-1.0), (0, 0.0));
        assert!(fitted(f32::MAX).0 <= 4_096);
    }

    #[test]
    fn an_odd_mask_or_radius_is_blurred_or_left_and_a_large_one_is_quick() {
        // Nothing to blur, a mask shorter than its size says, and radii
        // that are no radius: the mask comes back as it was.
        assert!(blur(&[], 0, 0, 8.0).is_empty());
        assert_eq!(blur(&[1.0, 0.0], 4, 4, 8.0), [1.0, 0.0]);
        assert_eq!(blur(&[1.0; 4], usize::MAX, 2, 8.0), [1.0; 4]);
        for radius in [f32::NAN, -3.0, 0.0, 0.4] {
            assert_eq!(
                blur(&[1.0, 0.0, 0.0, 1.0], 2, 2, radius),
                [1.0, 0.0, 0.0, 1.0]
            );
        }
        // A radius far wider than the mask, and an endless one.
        for radius in [1.0e6_f32, f32::MAX, f32::INFINITY] {
            let blurred = blur(&[1.0; 15], 5, 3, radius);
            assert_eq!(blurred.len(), 15);
            assert!(blurred.iter().all(|cover| (0.0..=1.0).contains(cover)));
        }
        // One pixel wide, one tall, and sizes no square divides.
        for (width, height) in [(1, 9), (9, 1), (7, 5), (17, 3), (33, 31)] {
            let mask = vec![1.0; width * height];
            for radius in [2.0_f32, 12.0, 40.0] {
                assert_eq!(blur(&mask, width, height, radius).len(), width * height);
            }
        }
        // A picture as large as a title in a 4K frame, blurred wide.
        let (width, height) = (3_840, 600);
        let mask = shapes(width, height);
        let started = std::time::Instant::now();
        let blurred = blur(&mask, width, height, 60.0);
        assert_eq!(blurred.len(), mask.len());
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
    }
}
