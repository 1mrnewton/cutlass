//! The CDEF kernels on NEON.

use core::arch::aarch64::{
    int16x8_t, vaddq_s16, vaddq_s32, vaddvq_s16, vaddvq_s32, vdupq_n_s16, vextq_s16, vget_low_s16,
    vmovl_u8, vmull_high_s16, vmull_s16, vmulq_s32, vnegq_s16, vpaddq_s16, vreinterpretq_s16_u16,
    vrev64q_s16, vshlq_u16, vsubq_s16,
};
use core::arch::aarch64::{
    uint16x8_t, vabdq_u16, vandq_u16, vdupq_n_u16, vmaxq_s16, vminq_s16, vmulq_s16,
    vnegq_s16 as neg, vqsubq_u16, vreinterpretq_s16_u16 as signed, vreinterpretq_u16_s16,
    vshlq_u16 as shift_u16, vshrq_n_s16,
};

use super::scalar::best;
use super::{CENTRE, Filter, STRIDE, WINDOW};
use crate::neon::{i32x4, u8x8, u16x8, u16x8_array};

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

/// Eight samples of the window from `at`.
#[inline]
#[target_feature(enable = "neon")]
fn window(win: &[u16; WINDOW], at: usize) -> uint16x8_t {
    // `at + 8 <= WINDOW` for every row and tap: the entry clamps both.
    let at = at.min(WINDOW - 8);
    let mut a = [0u16; 8];
    a.copy_from_slice(&win[at..at + 8]);
    u16x8(&a)
}

/// What one row's taps add up to, and the range of the samples they read.
struct Row {
    x: int16x8_t,
    sum: int16x8_t,
    min: int16x8_t,
    max: int16x8_t,
}

/// Adds `weight * constrain(p - x)` for the tap at `at` and the one
/// opposite it.
///
/// `constrain(d)` is `d` clamped to `+-max(0, strength - (|d| >> shift))`,
/// the bound a saturating subtraction. A sample that is not available
/// holds 0x7000: its distance from any real sample, shifted, is still above
/// every strength, so it adds 0. It never wins the minimum, and `p & 0xfff`
/// keeps every real sample and clears the mark, so it never wins the
/// maximum.
#[inline]
#[target_feature(enable = "neon")]
fn tap<const RANGE: bool>(
    win: &[u16; WINDOW],
    base: usize,
    at: usize,
    weight: i16,
    strength: u16,
    shift: u32,
    r: &mut Row,
) {
    let at = at.min(2 * CENTRE);
    let down = neg(vdupq_n_s16(shift.min(15) as i16));
    let strength = vdupq_n_u16(strength);
    for at in [at, 2 * CENTRE - at] {
        let p = window(win, base + at);
        let d = vsubq_s16(signed(p), r.x);
        let far = shift_u16(vabdq_u16(p, vreinterpretq_u16_s16(r.x)), down);
        let bound = signed(vqsubq_u16(strength, far));
        let v = vminq_s16(vmaxq_s16(d, neg(bound)), bound);
        r.sum = vaddq_s16(r.sum, vmulq_s16(v, vdupq_n_s16(weight)));
        if RANGE {
            r.min = vminq_s16(r.min, signed(p));
            r.max = vmaxq_s16(r.max, signed(vandq_u16(p, vdupq_n_u16(0xfff))));
        }
    }
}

/// The filter with the primary taps (`PRI`), the secondary taps (`SEC`),
/// or both. With one of them the result cannot leave the range of the
/// samples read (the weights add up to 12 of 16), so only both clamp.
#[inline]
#[target_feature(enable = "neon")]
fn filter_with<const PRI: bool, const SEC: bool, const BOTH: bool>(
    win: &[u16; WINDOW],
    f: &Filter,
    rows: usize,
) -> [[u16; 8]; 8] {
    let mut out = [[0u16; 8]; 8];
    for (i, o) in out.iter_mut().enumerate().take(rows) {
        let base = i * STRIDE;
        let x = signed(window(win, base + CENTRE));
        let mut r = Row {
            x,
            sum: vdupq_n_s16(0),
            min: x,
            max: x,
        };
        for k in 0..2 {
            if PRI {
                let w = f.pri_taps[k];
                tap::<BOTH>(win, base, f.pri_at[k], w, f.pri, f.pri_shift, &mut r);
            }
            if SEC {
                tap::<BOTH>(win, base, f.sec_at[k], 2, f.sec, f.sec_shift, &mut r);
                tap::<BOTH>(win, base, f.sec_at[2 + k], 1, f.sec, f.sec_shift, &mut r);
            }
        }
        // (8 + sum - (sum < 0)) >> 4
        let round = vaddq_s16(vaddq_s16(r.sum, vdupq_n_s16(8)), vshrq_n_s16::<15>(r.sum));
        let mut y = vaddq_s16(x, vshrq_n_s16::<4>(round));
        if BOTH {
            y = vminq_s16(vmaxq_s16(y, r.min), r.max);
        }
        *o = u16x8_array(vreinterpretq_u16_s16(y));
    }
    out
}

#[target_feature(enable = "neon")]
pub(super) fn filter(win: &[u16; WINDOW], f: &Filter, rows: usize) -> [[u16; 8]; 8] {
    match (f.pri != 0, f.sec != 0) {
        (true, true) => filter_with::<true, true, true>(win, f, rows),
        (true, false) => filter_with::<true, false, false>(win, f, rows),
        _ => filter_with::<false, true, false>(win, f, rows),
    }
}
