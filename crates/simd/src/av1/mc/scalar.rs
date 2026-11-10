//! The two-pass subpel filter as the specification writes it.

use super::Block;

/// `Round2(x, n)`.
fn round2(x: i32, n: u32) -> i32 {
    (x + (1 << (n - 1))) >> n
}

/// The horizontal pass: `mid[r * w + c]` for the `h + 7` rows.
fn horizontal(b: &Block<'_>, mid: &mut [i16]) {
    for r in 0..b.h + 7 {
        for c in 0..b.w {
            let at = b.start + r * b.src_stride + c;
            let sum: i32 = (0..8)
                .map(|t| i32::from(b.fh[t]) * i32::from(b.src[at + t]))
                .sum();
            mid[r * b.w + c] = round2(sum, 3) as i16;
        }
    }
}

/// The vertical sum for output `(r, c)`.
fn vertical(b: &Block<'_>, mid: &[i16], r: usize, c: usize) -> i32 {
    (0..8)
        .map(|t| i32::from(b.fv[t]) * i32::from(mid[(r + t) * b.w + c]))
        .sum()
}

#[cfg_attr(
    all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "dotprod",
        not(test)
    ),
    allow(dead_code)
)]
pub(super) fn prep_hv(dst: &mut [i16], dst_stride: usize, b: &Block<'_>, mid: &mut [i16]) {
    horizontal(b, mid);
    for r in 0..b.h {
        for c in 0..b.w {
            dst[r * dst_stride + c] = round2(vertical(b, mid, r, c), 7) as i16;
        }
    }
}

#[cfg_attr(
    all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "dotprod",
        not(test)
    ),
    allow(dead_code)
)]
pub(super) fn put_hv(dst: &mut [u8], dst_stride: usize, b: &Block<'_>, mid: &mut [i16]) {
    horizontal(b, mid);
    for r in 0..b.h {
        for c in 0..b.w {
            dst[r * dst_stride + c] = round2(vertical(b, mid, r, c), 11).clamp(0, 255) as u8;
        }
    }
}
