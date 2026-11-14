//! Intra prediction one sample at a time, as the specification has it.

use super::FilterTaps;

/// The inputs of the 4x2 block at row `r`, column `c` of
/// [`super::filter_intra`]: above-left, the four above, the two left.
#[cfg_attr(
    all(target_arch = "aarch64", target_feature = "neon", not(test)),
    allow(dead_code)
)]
pub(super) fn inputs(
    dst: &[u8],
    stride: usize,
    r: usize,
    c: usize,
    above: &[u8],
    left: &[u8],
) -> [u8; 7] {
    let mut p = [0u8; 7];
    for (i, v) in p[..5].iter_mut().enumerate() {
        // Column c + i - 1 of the row above.
        *v = match (r, c + i) {
            (0, x) => above[x],
            (_, 0) => left[r - 1],
            (_, x) => dst[(r - 1) * stride + x - 1],
        };
    }
    for k in 0..2 {
        p[5 + k] = if c == 0 {
            left[r + k]
        } else {
            dst[(r + k) * stride + c - 1]
        };
    }
    p
}

#[cfg_attr(
    all(target_arch = "aarch64", target_feature = "neon", not(test)),
    allow(dead_code)
)]
pub(super) fn filter_intra(
    dst: &mut [u8],
    stride: usize,
    w: usize,
    h: usize,
    taps: &FilterTaps,
    above: &[u8],
    left: &[u8],
) {
    for r in (0..h).step_by(2) {
        for c in (0..w).step_by(4) {
            let p = inputs(dst, stride, r, c, above, left);
            for (k, t) in taps.iter().enumerate() {
                let sum: i32 = t
                    .iter()
                    .zip(&p)
                    .map(|(&t, &p)| i32::from(t) * i32::from(p))
                    .sum();
                // Round2Signed(sum, 4) clipped to 8 bits.
                let v = if sum < 0 {
                    0
                } else {
                    ((sum + 8) >> 4).min(255)
                };
                dst[(r + k / 4) * stride + c + k % 4] = v as u8;
            }
        }
    }
}
