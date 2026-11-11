//! The subpel filters as the specification writes them. With a fraction
//! of 0 the filter is the identity (one tap of 128): the horizontal pass
//! then gives `16 x` for a sample `x`, and the vertical one returns its
//! input times 128.

use super::Block;

/// `Round2(x, n)`.
fn round2(x: i32, n: u32) -> i32 {
    (x + (1 << (n - 1))) >> n
}

/// The horizontal pass: `mid[r * w + c]` for the `h + 7` rows.
fn horizontal(b: &Block<'_>) -> Vec<i16> {
    let mut mid = vec![0i16; (b.h + 7) * b.w];
    for r in 0..b.h + 7 {
        for c in 0..b.w {
            let at = b.start + r * b.src_stride + c;
            let sum: i32 = (0..8)
                .map(|t| i32::from(b.fh[t]) * i32::from(b.src[at + t]))
                .sum();
            mid[r * b.w + c] = round2(sum, 3) as i16;
        }
    }
    mid
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
pub(super) fn prep_hv(dst: &mut [i16], dst_stride: usize, b: &Block<'_>) {
    let mid = horizontal(b);
    for r in 0..b.h {
        for c in 0..b.w {
            dst[r * dst_stride + c] = round2(vertical(b, &mid, r, c), 7) as i16;
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
pub(super) fn put_hv(dst: &mut [u8], dst_stride: usize, b: &Block<'_>) {
    let mid = horizontal(b);
    for r in 0..b.h {
        for c in 0..b.w {
            dst[r * dst_stride + c] = round2(vertical(b, &mid, r, c), 11).clamp(0, 255) as u8;
        }
    }
}

/// The vertical sum of the samples for output `(r, c)`.
fn vertical_samples(b: &Block<'_>, r: usize, c: usize) -> i32 {
    (0..8)
        .map(|t| i32::from(b.fv[t]) * i32::from(b.src[b.start + 3 + (r + t) * b.src_stride + c]))
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
pub(super) fn prep_h(dst: &mut [i16], dst_stride: usize, b: &Block<'_>) {
    let mid = horizontal(b);
    for r in 0..b.h {
        for c in 0..b.w {
            // Round2(128 * mid, 7).
            dst[r * dst_stride + c] = mid[(r + 3) * b.w + c];
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
pub(super) fn put_h(dst: &mut [u8], dst_stride: usize, b: &Block<'_>) {
    let mid = horizontal(b);
    for r in 0..b.h {
        for c in 0..b.w {
            let v = round2(128 * i32::from(mid[(r + 3) * b.w + c]), 11);
            dst[r * dst_stride + c] = v.clamp(0, 255) as u8;
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
pub(super) fn prep_v(dst: &mut [i16], dst_stride: usize, b: &Block<'_>) {
    for r in 0..b.h {
        for c in 0..b.w {
            // Round2(sum * 16, 7).
            dst[r * dst_stride + c] = round2(16 * vertical_samples(b, r, c), 7) as i16;
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
pub(super) fn put_v(dst: &mut [u8], dst_stride: usize, b: &Block<'_>) {
    for r in 0..b.h {
        for c in 0..b.w {
            let v = round2(16 * vertical_samples(b, r, c), 11);
            dst[r * dst_stride + c] = v.clamp(0, 255) as u8;
        }
    }
}
