//! Intra predictors used by the reconstruction path.
//!
//! DC, vertical, and horizontal follow the decoding-process integer definitions.
//! Directional, smooth, Paeth, filter-intra, and CfL are not invoked from the
//! tile decoder until they have fixtures; calling them is an explicit error
//! at the mode switch in `tiledec`.

use crate::math::clip1;

/// Fill `pred` (row-major, `stride` samples) with the DC predictor.
///
/// `left` / `above` are the `h` / `w` edge samples. They are read only when
/// the corresponding `have_*` flag is set. The both-available average is
/// truncating division, matching the specification. One-sided averages use
/// the shift form and `Clip1`.
pub fn predict_dc(
    bit_depth: u8,
    have_left: bool,
    have_above: bool,
    left: &[u16],
    above: &[u16],
    w: usize,
    h: usize,
    pred: &mut [u16],
    stride: usize,
) {
    let avg = if have_left && have_above {
        let mut sum = 0i32;
        for &s in left.iter().take(h) {
            sum += i32::from(s);
        }
        for &s in above.iter().take(w) {
            sum += i32::from(s);
        }
        sum += ((w + h) >> 1) as i32;
        sum / (w + h) as i32
    } else if have_left {
        let mut sum = 0i32;
        for &s in left.iter().take(h) {
            sum += i32::from(s);
        }
        let log2_h = h.trailing_zeros();
        clip1(bit_depth, (sum + ((h as i32) >> 1)) >> log2_h)
    } else if have_above {
        let mut sum = 0i32;
        for &s in above.iter().take(w) {
            sum += i32::from(s);
        }
        let log2_w = w.trailing_zeros();
        clip1(bit_depth, (sum + ((w as i32) >> 1)) >> log2_w)
    } else {
        1i32 << (bit_depth - 1)
    };
    let sample = avg as u16;
    for row in 0..h {
        let dst = &mut pred[row * stride..row * stride + w];
        dst.fill(sample);
    }
}

/// Vertical predictor: every row copies `above[0..w]`.
pub fn predict_vertical(above: &[u16], w: usize, h: usize, pred: &mut [u16], stride: usize) {
    for row in 0..h {
        pred[row * stride..row * stride + w].copy_from_slice(&above[..w]);
    }
}

/// Horizontal predictor: every column copies `left[row]`.
pub fn predict_horizontal(left: &[u16], w: usize, h: usize, pred: &mut [u16], stride: usize) {
    for row in 0..h {
        let sample = left[row];
        pred[row * stride..row * stride + w].fill(sample);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_with_no_neighbors_is_midpoint() {
        let mut pred = [0u16; 16];
        predict_dc(8, false, false, &[], &[], 4, 4, &mut pred, 4);
        assert!(pred.iter().all(|&s| s == 128));
    }

    #[test]
    fn dc_above_only_uses_the_shift_average() {
        let above = [10u16, 20, 30, 40];
        let mut pred = [0u16; 16];
        predict_dc(8, false, true, &[], &above, 4, 4, &mut pred, 4);
        // sum = 100, (100 + 2) >> 2 = 25.
        assert!(pred.iter().all(|&s| s == 25));
    }

    #[test]
    fn dc_both_sides_truncates() {
        let edge = [1u16, 1, 1, 1];
        let mut pred = [0u16; 16];
        predict_dc(8, true, true, &edge, &edge, 4, 4, &mut pred, 4);
        // sum = 8, plus (4+4)>>1 = 4, total 12, 12/8 = 1.
        assert!(pred.iter().all(|&s| s == 1));
    }

    #[test]
    fn vertical_copies_the_above_row() {
        let above = [1u16, 2, 3, 4];
        let mut pred = [0u16; 16];
        predict_vertical(&above, 4, 4, &mut pred, 4);
        for row in 0..4 {
            assert_eq!(&pred[row * 4..row * 4 + 4], &above);
        }
    }

    #[test]
    fn horizontal_replicates_each_left_sample() {
        let left = [5u16, 6, 7, 8];
        let mut pred = [0u16; 16];
        predict_horizontal(&left, 4, 4, &mut pred, 4);
        assert_eq!(&pred[0..4], &[5, 5, 5, 5]);
        assert_eq!(&pred[12..16], &[8, 8, 8, 8]);
    }
}
