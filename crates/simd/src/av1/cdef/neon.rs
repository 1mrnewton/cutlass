//! The CDEF kernels on NEON.

use core::arch::aarch64::{
    int16x8_t, vaddq_s16, vaddq_s32, vaddvq_s16, vaddvq_s32, vdupq_n_s16, vextq_s16, vget_low_s16,
    vmovl_u8, vmull_high_s16, vmull_s16, vmulq_s32, vnegq_s16, vpaddq_s16, vreinterpretq_s16_u16,
    vrev64q_s16, vshlq_u16, vsubq_s16,
};

use super::scalar::best;
use crate::neon::{i32x4, u8x8, u16x8};

/// Weights of the diagonal sums: a lane summing `n` samples gets
/// `Div_Table[n]`. Lane 15 sums nothing.
const W_DIAG: [i32; 16] = [
    840, 420, 280, 210, 168, 140, 120, 105, 120, 140, 168, 210, 280, 420, 840, 0,
];
/// Weights of the sums along the four in-between directions, 11 lanes.
const W_ALT: [i32; 16] = [
    420, 210, 140, 105, 105, 105, 105, 105, 140, 210, 420, 0, 0, 0, 0, 0,
];

#[target_feature(enable = "neon")]
pub(super) fn direction_u8(px: &[[u8; 8]; 8]) -> (u8, i32) {
    let mid = vdupq_n_s16(128);
    direction(
        px.each_ref()
            .map(|r| vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u8x8(r))), mid)),
    )
}

#[target_feature(enable = "neon")]
pub(super) fn direction_u16(px: &[[u16; 8]; 8], bd: u32) -> (u8, i32) {
    let mid = vdupq_n_s16(128);
    // A shift left by a negative count shifts right.
    let down = vnegq_s16(vdupq_n_s16(bd as i16 - 8));
    direction(
        px.each_ref()
            .map(|r| vsubq_s16(vreinterpretq_s16_u16(vshlq_u16(u16x8(r), down)), mid)),
    )
}

/// Sixteen lanes of partial sums.
type Sums = (int16x8_t, int16x8_t);

/// Sum of `lane^2 * weight` over the sixteen lanes.
#[inline]
#[target_feature(enable = "neon")]
fn weighted_squares(v: Sums, w: &[i32; 16]) -> i32 {
    let w = |i: usize| i32x4([w[i], w[i + 1], w[i + 2], w[i + 3]]);
    let a = vmulq_s32(vmull_s16(vget_low_s16(v.0), vget_low_s16(v.0)), w(0));
    let b = vmulq_s32(vmull_high_s16(v.0, v.0), w(4));
    let c = vmulq_s32(vmull_s16(vget_low_s16(v.1), vget_low_s16(v.1)), w(8));
    let d = vmulq_s32(vmull_high_s16(v.1, v.1), w(12));
    vaddvq_s32(vaddq_s32(vaddq_s32(a, b), vaddq_s32(c, d)))
}

/// The direction process on rows of `sample - 128`.
///
/// A sum along a diagonal is a sum of whole vectors once row `i` is
/// shifted `i` lanes up: the eight lanes of a row land in sixteen.
#[inline]
#[target_feature(enable = "neon")]
fn direction(r: [int16x8_t; 8]) -> (u8, i32) {
    let z = vdupq_n_s16(0);
    // Adds `v` to lanes `i..i + 8` of the sums.
    macro_rules! add_at {
        ($acc:ident, $v:expr, 0) => {
            $acc.0 = vaddq_s16($acc.0, $v)
        };
        ($acc:ident, $v:expr, $i:literal) => {{
            let v = $v;
            $acc.0 = vaddq_s16($acc.0, vextq_s16::<{ 8 - $i }>(z, v));
            $acc.1 = vaddq_s16($acc.1, vextq_s16::<{ 8 - $i }>(v, z));
        }};
    }
    // partial[0], [4]: the diagonals. partial[1], [3], [5], [7]: between.
    let (mut d0, mut d1, mut a0, mut a1, mut a2, mut a3): (Sums, Sums, Sums, Sums, Sums, Sums) =
        ((z, z), (z, z), (z, z), (z, z), (z, z), (z, z));
    // partial[6]: the columns. partial[2]: the rows.
    let mut cols = z;
    let mut rows = [0i16; 8];
    macro_rules! row {
        ($i:tt) => {{
            let v = r[$i];
            let rev = vrev64q_s16(v);
            let rev = vextq_s16::<4>(rev, rev);
            cols = vaddq_s16(cols, v);
            rows[$i] = vaddvq_s16(v);
            add_at!(d0, v, $i);
            add_at!(d1, rev, $i);
            // Sums of column pairs in lanes 0..4.
            add_at!(a0, vpaddq_s16(v, z), $i);
            add_at!(a1, vpaddq_s16(rev, z), $i);
        }};
    }
    row!(0);
    row!(1);
    row!(2);
    row!(3);
    row!(4);
    row!(5);
    row!(6);
    row!(7);
    // Sums of row pairs.
    let p0 = vaddq_s16(r[0], r[1]);
    let p1 = vaddq_s16(r[2], r[3]);
    let p2 = vaddq_s16(r[4], r[5]);
    let p3 = vaddq_s16(r[6], r[7]);
    add_at!(a2, p0, 3);
    add_at!(a2, p1, 2);
    add_at!(a2, p2, 1);
    add_at!(a2, p3, 0);
    add_at!(a3, p0, 0);
    add_at!(a3, p1, 1);
    add_at!(a3, p2, 2);
    add_at!(a3, p3, 3);
    let flat = [105i32; 16];
    let rows = crate::neon::i16x8(rows);
    let cost = [
        weighted_squares(d0, &W_DIAG),
        weighted_squares(a0, &W_ALT),
        weighted_squares((rows, z), &flat),
        weighted_squares(a1, &W_ALT),
        weighted_squares(d1, &W_DIAG),
        weighted_squares(a2, &W_ALT),
        weighted_squares((cols, z), &flat),
        weighted_squares(a3, &W_ALT),
    ];
    best(&cost)
}
