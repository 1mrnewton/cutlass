//! The Wiener filter as the specification writes it (7.17.4).

use super::Rounding;

/// `Round2(x, n)`.
fn round2(x: i32, n: u32) -> i32 {
    (x + (1 << (n - 1))) >> n
}

/// The seven taps of a row from its first four.
fn row(taps: &[i16; 4]) -> [i32; 7] {
    let t = taps.map(i32::from);
    [t[0], t[1], t[2], t[3], t[2], t[1], t[0]]
}

#[cfg_attr(
    all(target_arch = "aarch64", target_feature = "neon", not(test)),
    allow(dead_code)
)]
pub(super) fn horizontal(dst: &mut [i16], src: &[i16], taps: &[i16; 4], r: &Rounding) {
    let f = row(taps);
    for (c, d) in dst.iter_mut().enumerate() {
        let sum: i32 = (0..7).map(|t| f[t] * i32::from(src[c + t])).sum();
        *d = round2(sum, r.round0).clamp(i32::from(r.low), i32::from(r.high)) as i16;
    }
}

/// The output sample of column `c`.
#[cfg_attr(
    all(target_arch = "aarch64", target_feature = "neon", not(test)),
    allow(dead_code)
)]
pub(super) fn vertical(rows: &[&[i16]; 7], c: usize, taps: &[i16; 4], r: &Rounding) -> u16 {
    // The clip is to the largest sample of the depth of `r`.
    let f = row(taps);
    let sum: i32 = (0..7).map(|t| f[t] * i32::from(rows[t][c])).sum();
    round2(sum, r.round1).clamp(0, i32::from(r.max)) as u16
}
