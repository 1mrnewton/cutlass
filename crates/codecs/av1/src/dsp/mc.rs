//! Inter prediction kernels (AV1 specification section 7.11.3): sub-sample
//! interpolation (unscaled and scaled), compound combination, wedge,
//! difference-weighted and inter-intra masks, mask blending, overlapped motion
//! blending, and warped motion.
//!
//! # Rounding
//!
//! The kernels apply the rounding variables of section 7.11.3.2:
//! `InterRound0` is 3 (5 at 12 bits), `InterRound1` is 7 for compound
//! prediction and 11 (9 at 12 bits) otherwise, and `InterPostRound` is
//! `14 - InterRound0 - InterRound1`.
//!
//! # Compound intermediate
//!
//! [`prep`], [`prep_scaled`] and [`warp_8x8_prep`] produce the specification's
//! `preds[ refList ]` values for `isCompound = 1`, stored as `i16` minus
//! [`prep_bias`]`(bd)` (0 at 8 bits, 8192 at 10 and 12 bits). The bias keeps
//! every possible value inside `i16`: at 10 and 12 bits the unbiased range is
//! `-20602..=36982`. Only the combining functions of this module ([`avg`],
//! [`w_avg`], [`mask_blend`], [`diffwtd_mask`]) read this format, and they
//! remove the bias exactly. Intermediate buffers have a stride of `w`.
//!
//! # Limits
//!
//! Block widths and heights are at most 128 (the largest AV1 block).

use std::sync::OnceLock;

use super::{Pixel, clip_px, round2};
use crate::spec::tables::{
    DIV_LUT, DIV_LUT_BITS, DIV_LUT_PREC_BITS, II_H_PRED, II_SMOOTH_PRED, II_V_PRED, II_WEIGHTS_1D,
    MASK_MASTER_SIZE, MAX_SB_SIZE, NUM_4X4_BLOCKS_HIGH, NUM_4X4_BLOCKS_WIDE, OBMC_MASK_2,
    OBMC_MASK_4, OBMC_MASK_8, OBMC_MASK_16, OBMC_MASK_32, SCALE_SUBPEL_BITS, SUBPEL_FILTERS,
    SUBPEL_MASK, WARP_PARAM_REDUCE_BITS, WARPED_FILTERS, WARPEDDIFF_PREC_BITS,
    WARPEDMODEL_PREC_BITS, WEDGE_BITS, WEDGE_CODEBOOK, WEDGE_HORIZONTAL, WEDGE_MASTER_OBLIQUE_EVEN,
    WEDGE_MASTER_OBLIQUE_ODD, WEDGE_MASTER_VERTICAL, WEDGE_OBLIQUE27, WEDGE_OBLIQUE63,
    WEDGE_OBLIQUE117, WEDGE_OBLIQUE153, WEDGE_TYPES, WEDGE_VERTICAL,
};

/// `WARPEDPIXEL_PREC_SHIFTS` from the specification.
const WARPEDPIXEL_PREC_SHIFTS: i32 = 1 << 6;

// ---------------------------------------------------------------------------
// Rounding variables (7.11.3.2)
// ---------------------------------------------------------------------------

#[inline(always)]
fn inter_round0(bd: u32) -> u32 {
    if bd == 12 { 5 } else { 3 }
}

#[inline(always)]
fn inter_round1(bd: u32, compound: bool) -> u32 {
    if compound {
        7
    } else if bd == 12 {
        9
    } else {
        11
    }
}

/// `InterPostRound` of section 7.11.3.2.
#[inline]
pub fn inter_post_round(bd: u32, compound: bool) -> u32 {
    14 - inter_round0(bd) - inter_round1(bd, compound)
}

/// The value subtracted from compound intermediates before they are stored
/// as `i16` (see the module documentation).
#[inline]
pub fn prep_bias(bd: u32) -> i32 {
    if bd > 8 { 8192 } else { 0 }
}

// ---------------------------------------------------------------------------
// Scratch buffers
// ---------------------------------------------------------------------------

/// Runs `f` on a scratch buffer of at least `need` samples, kept per thread
/// between calls. Its contents are stale: every caller writes the samples
/// it reads first.
#[inline(always)]
fn with_mid<R>(need: usize, f: impl FnOnce(&mut [i16]) -> R) -> R {
    thread_local! {
        static MID: core::cell::RefCell<Vec<i16>> = const { core::cell::RefCell::new(Vec::new()) };
    }
    MID.with(|m| match m.try_borrow_mut() {
        Ok(mut m) => {
            if m.len() < need {
                m.resize(need, 0);
            }
            f(&mut m[..need])
        }
        // Not reentered today; a nested call gets a buffer of its own.
        Err(_) => f(&mut vec![0i16; need]),
    })
}

// ---------------------------------------------------------------------------
// Separable 8-tap filters
// ---------------------------------------------------------------------------

/// The span of non-zero taps of a filter row. Every row of
/// `SUBPEL_FILTERS` is zero outside one of these spans.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Taps {
    Eight,
    Six,
    Four,
    Two,
}

impl Taps {
    #[inline]
    fn of(f: &[i16; 8]) -> Self {
        if f[0] != 0 || f[7] != 0 {
            Taps::Eight
        } else if f[1] != 0 || f[6] != 0 {
            Taps::Six
        } else if f[2] != 0 || f[5] != 0 {
            Taps::Four
        } else {
            Taps::Two
        }
    }

    /// First and one-past-last tap index.
    #[inline]
    fn range(self) -> (usize, usize) {
        match self {
            Taps::Eight => (0, 8),
            Taps::Six => (1, 7),
            Taps::Four => (2, 6),
            Taps::Two => (3, 5),
        }
    }
}

// The unscaled filters run entirely on 16-bit lanes (which vectorize well
// even on baseline x86-64) while staying bit exact:
//
// * Every `SUBPEL_FILTERS` coefficient is even, so the kernels use the
//   halved coefficients `fh = f / 2` and fold the factor 2 into the
//   rounding shift: `Round2(2 * s, n) == Round2(s, n - 1)`.
// * A 16-bit sum `sum(fh[t] * x[t])` cannot overflow when `x` lies in
//   `0..=255` (the positive halved taps sum to at most 92, the negative to
//   at most -28). Wider inputs `x` are split as `x = 256 * (x >> 8) +
//   (x & 255)`: `B = sum(fh * (x & 255))` is exact in 16 bits and
//   `A = sum(fh * (x >> 8))` is either small enough to be exact or only
//   enters the result through wrapping multiply/add, so the 16-bit result
//   is correct modulo 2^16 and exact because the final value fits.
// * `Round2(256 * A + B, n)` with `n >= 8` equals
//   `(A + ((B + (1 << (n - 1))) >> 8)) >> (n - 8)`, and with `n <= 8`
//   equals `(A << (8 - n)) + ((B + (1 << (n - 1))) >> n)`.

/// Halves a filter row (all coefficients are even).
#[inline(always)]
fn half(f: &[i16; 8]) -> [i16; 8] {
    let mut h = [0i16; 8];
    for t in 0..8 {
        h[t] = f[t] >> 1;
    }
    h
}

/// Horizontal filter of one row with halved coefficients `fh`. `src` starts
/// 3 samples left of the first output sample. `wr` receives the
/// intermediate `Round2(sum, InterRound0)`; `sh` is `InterRound0 - 1`.
#[inline(always)]
fn h_row<P: Pixel, D, const T0: usize, const T1: usize>(
    dst: &mut [D],
    src: &[P],
    fh: &[i16; 8],
    sh: u32,
    wr: impl Fn(i16) -> D,
) {
    let w = dst.len();
    let src = &src[..w + 7];
    let rnd = 1i16 << (sh - 1);
    if !P::HBD {
        for c in 0..w {
            let mut s = rnd;
            for t in T0..T1 {
                s += fh[t] * src[c + t].to_i32() as i16;
            }
            dst[c] = wr(s >> sh);
        }
    } else {
        let up = 8 - sh;
        for c in 0..w {
            let mut a = 0i16;
            let mut b = rnd;
            for t in T0..T1 {
                let p = src[c + t].to_i32() as i16;
                a = a.wrapping_add(fh[t].wrapping_mul(p >> 8));
                b += fh[t] * (p & 255);
            }
            dst[c] = wr((a << up).wrapping_add(b >> sh));
        }
    }
}

/// `W` is the block width when it is a compile-time constant, or 0.
#[inline(always)]
fn h_pass<P: Pixel, D, const T0: usize, const T1: usize, const W: usize>(
    dst: &mut [D],
    dst_stride: usize,
    src: &[P],
    src_start: usize,
    src_stride: usize,
    w: usize,
    rows: usize,
    fh: &[i16; 8],
    sh: u32,
    wr: impl Fn(i16) -> D + Copy,
) {
    let w = if W != 0 { W } else { w };
    for r in 0..rows {
        h_row::<P, D, T0, T1>(
            &mut dst[r * dst_stride..][..w],
            &src[src_start + r * src_stride..],
            fh,
            sh,
            wr,
        );
    }
}

/// Calls `$f::<$pre.., T0, T1, W $post..>` for the tap span of `$filter`
/// and the width `$w` (constant for the small widths, where a runtime trip
/// count keeps the loops from vectorizing).
macro_rules! dispatch {
    ($f:ident, [$($pre:ty),*], [$($post:expr),*], $filter:expr, $w:expr, ($($arg:expr),*)) => {
        match (Taps::of($filter), $w) {
            (Taps::Eight, 2) => $f::<$($pre,)* 0, 8, 2 $(, $post)*>($($arg),*),
            (Taps::Eight, 4) => $f::<$($pre,)* 0, 8, 4 $(, $post)*>($($arg),*),
            (Taps::Eight, 8) => $f::<$($pre,)* 0, 8, 8 $(, $post)*>($($arg),*),
            (Taps::Eight, 16) => $f::<$($pre,)* 0, 8, 16 $(, $post)*>($($arg),*),
            (Taps::Eight, _) => $f::<$($pre,)* 0, 8, 0 $(, $post)*>($($arg),*),
            (Taps::Six, 2) => $f::<$($pre,)* 1, 7, 2 $(, $post)*>($($arg),*),
            (Taps::Six, 4) => $f::<$($pre,)* 1, 7, 4 $(, $post)*>($($arg),*),
            (Taps::Six, 8) => $f::<$($pre,)* 1, 7, 8 $(, $post)*>($($arg),*),
            (Taps::Six, 16) => $f::<$($pre,)* 1, 7, 16 $(, $post)*>($($arg),*),
            (Taps::Six, _) => $f::<$($pre,)* 1, 7, 0 $(, $post)*>($($arg),*),
            (Taps::Four, 2) => $f::<$($pre,)* 2, 6, 2 $(, $post)*>($($arg),*),
            (Taps::Four, 4) => $f::<$($pre,)* 2, 6, 4 $(, $post)*>($($arg),*),
            (Taps::Four, 8) => $f::<$($pre,)* 2, 6, 8 $(, $post)*>($($arg),*),
            (Taps::Four, 16) => $f::<$($pre,)* 2, 6, 16 $(, $post)*>($($arg),*),
            (Taps::Four, _) => $f::<$($pre,)* 2, 6, 0 $(, $post)*>($($arg),*),
            (Taps::Two, 2) => $f::<$($pre,)* 3, 5, 2 $(, $post)*>($($arg),*),
            (Taps::Two, 4) => $f::<$($pre,)* 3, 5, 4 $(, $post)*>($($arg),*),
            (Taps::Two, 8) => $f::<$($pre,)* 3, 5, 8 $(, $post)*>($($arg),*),
            (Taps::Two, 16) => $f::<$($pre,)* 3, 5, 16 $(, $post)*>($($arg),*),
            (Taps::Two, _) => $f::<$($pre,)* 3, 5, 0 $(, $post)*>($($arg),*),
        }
    };
}

/// Horizontal pass over `rows` rows. `src_start` is the index of the sample
/// 3 left of the first output sample of the first row.
#[inline(always)]
fn h_pass_dyn<P: Pixel, D>(
    dst: &mut [D],
    dst_stride: usize,
    src: &[P],
    src_start: usize,
    src_stride: usize,
    w: usize,
    rows: usize,
    fh: &[i16; 8],
    sh: u32,
    wr: impl Fn(i16) -> D + Copy,
) {
    dispatch!(
        h_pass,
        [P, D],
        [],
        fh,
        w,
        (
            dst, dst_stride, src, src_start, src_stride, w, rows, fh, sh, wr
        )
    )
}

/// Plain sums: inputs lie in `0..=255`.
const V_PLAIN: u8 = 0;
/// Split sums, the high parts computed from the inputs.
const V_SPLIT: u8 = 1;
/// Split sums, the high parts `x >> 8` read from the plane `hi`.
const V_SPLIT_HI: u8 = 2;

/// Vertical filter with halved coefficients. `wr(A, B)` receives the split
/// sums `A = sum(fh * (x >> 8))`, `B = sum(fh * (x & 255))`; in `V_PLAIN`
/// mode `A` is 0 and `B` the plain sum. `B` is recovered from the wrapping
/// full sum as `S - 256 * A`, which is exact because `B` fits in 16 bits.
#[inline(always)]
fn v_pass<S: Copy, D, const T0: usize, const T1: usize, const W: usize, const MODE: u8>(
    dst: &mut [D],
    dst_stride: usize,
    src: &[S],
    hi: &[i16],
    top: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    fh: &[i16; 8],
    rd: impl Fn(S) -> i16 + Copy,
    wr: impl Fn(i16, i16) -> D + Copy,
) {
    let w = if W != 0 { W } else { w };
    let fh = *fh;
    // A window of the rows the taps read, moved down one row per output
    // row: every row is sliced (and bounds checked) once.
    let mut rows: [&[S]; 8] = [&[]; 8];
    let mut hrows: [&[i16]; 8] = [&[]; 8];
    for t in T0..T1 - 1 {
        rows[t] = &src[top + t * src_stride..][..w];
        if MODE == V_SPLIT_HI {
            hrows[t] = &hi[top + t * src_stride..][..w];
        }
    }
    for r in 0..h {
        let d = &mut dst[r * dst_stride..][..w];
        let last = top + (r + T1 - 1) * src_stride;
        rows[T1 - 1] = &src[last..][..w];
        if MODE == V_SPLIT_HI {
            hrows[T1 - 1] = &hi[last..][..w];
        }
        for c in 0..w {
            let mut a = 0i16;
            let mut s = 0i16;
            for t in T0..T1 {
                let x = rd(rows[t][c]);
                s = s.wrapping_add(fh[t].wrapping_mul(x));
                if MODE == V_SPLIT {
                    a = a.wrapping_add(fh[t].wrapping_mul(x >> 8));
                } else if MODE == V_SPLIT_HI {
                    a = a.wrapping_add(fh[t].wrapping_mul(hrows[t][c]));
                }
            }
            d[c] = wr(a, s.wrapping_sub(a << 8));
        }
        for t in T0..T1 - 1 {
            rows[t] = rows[t + 1];
            if MODE == V_SPLIT_HI {
                hrows[t] = hrows[t + 1];
            }
        }
    }
}

/// Vertical pass. `top` is the index of the sample 3 rows above the first
/// output sample.
#[inline(always)]
fn v_pass_dyn<S: Copy, D, const MODE: u8>(
    dst: &mut [D],
    dst_stride: usize,
    src: &[S],
    hi: &[i16],
    top: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    fh: &[i16; 8],
    rd: impl Fn(S) -> i16 + Copy,
    wr: impl Fn(i16, i16) -> D + Copy,
) {
    dispatch!(
        v_pass,
        [S, D],
        [MODE],
        fh,
        w,
        (dst, dst_stride, src, hi, top, src_stride, w, h, fh, rd, wr)
    )
}

/// Clips a non-compound prediction held in 16 bits to the pixel range.
#[inline(always)]
fn clip16<P: Pixel>(v: i16, max: i16) -> P {
    P::from_i32(i32::from(v.clamp(0, max)))
}

/// Rounds a non-compound 32-bit prediction sum by `sh` and clips it.
/// Non-compound predictions lie inside `-5150..=9245`, so the clamp runs on
/// 16-bit lanes.
#[inline(always)]
fn put_px<P: Pixel>(s: i32, rnd: i32, sh: u32, max: i16) -> P {
    clip16::<P>(((s + rnd) >> sh) as i16, max)
}

/// `(1 << sh) >> 1`: the `Round2` offset, also for `sh == 0`.
#[inline(always)]
fn rnd_of(sh: u32) -> i32 {
    (1 << sh) >> 1
}

/// Horizontal then vertical filtering through an `i16` intermediate.
/// `wr(A, B)` receives the split vertical sums.
#[inline]
fn hv<P: Pixel, D>(
    dst: &mut [D],
    dst_stride: usize,
    src: &[P],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    fh: &[i16; 8],
    fv: &[i16; 8],
    r0: u32,
    wr: impl Fn(i16, i16) -> D + Copy,
) {
    let (t0, t1) = Taps::of(fv).range();
    // Intermediate row k holds source row k - 3; only the rows the vertical
    // taps reach are computed.
    let n = w * (h + 7);
    with_mid(2 * n, |buf| {
        let (mid, hi) = buf.split_at_mut(n);
        let rows = h + t1 - t0 - 1;
        let start = src_off - (3 - t0) * src_stride - 3;
        macro_rules! fixed {
            ($w:literal, $r:literal, $l:literal) => {{
                let m = &mut mid[t0 * $w..];
                let a = (m, $w, src, start, src_stride, $w, rows, fh, r0 - 1);
                match Taps::of(fh) {
                    Taps::Eight => h_pass::<P, i16, 0, 8, $w>(
                        a.0,
                        a.1,
                        a.2,
                        a.3,
                        a.4,
                        a.5,
                        a.6,
                        a.7,
                        a.8,
                        |v| v,
                    ),
                    Taps::Six => h_pass::<P, i16, 1, 7, $w>(
                        a.0,
                        a.1,
                        a.2,
                        a.3,
                        a.4,
                        a.5,
                        a.6,
                        a.7,
                        a.8,
                        |v| v,
                    ),
                    Taps::Four => h_pass::<P, i16, 2, 6, $w>(
                        a.0,
                        a.1,
                        a.2,
                        a.3,
                        a.4,
                        a.5,
                        a.6,
                        a.7,
                        a.8,
                        |v| v,
                    ),
                    Taps::Two => h_pass::<P, i16, 3, 5, $w>(
                        a.0,
                        a.1,
                        a.2,
                        a.3,
                        a.4,
                        a.5,
                        a.6,
                        a.7,
                        a.8,
                        |v| v,
                    ),
                }
                let (lo, end) = (t0 * $w, (t0 + rows) * $w);
                for (h8, &m) in hi[lo..end].iter_mut().zip(&mid[lo..end]) {
                    *h8 = m >> 8;
                }
                match Taps::of(fv) {
                    Taps::Eight => {
                        v_mid::<D, 0, 8, $w, $r, $l>(dst, dst_stride, mid, hi, h, fv, wr)
                    }
                    Taps::Six => v_mid::<D, 1, 7, $w, $r, $l>(dst, dst_stride, mid, hi, h, fv, wr),
                    Taps::Four => v_mid::<D, 2, 6, $w, $r, $l>(dst, dst_stride, mid, hi, h, fv, wr),
                    Taps::Two => v_mid::<D, 3, 5, $w, $r, $l>(dst, dst_stride, mid, hi, h, fv, wr),
                }
            }};
        }
        match (w, h % 4) {
            (2, 0) => return fixed!(2, 4, 8),
            (4, 0 | 2) => return fixed!(4, 2, 8),
            (8, _) => return fixed!(8, 1, 8),
            (16, _) => return fixed!(16, 1, 16),
            _ => {}
        }
        h_pass_dyn::<P, i16>(
            &mut mid[t0 * w..],
            w,
            src,
            start,
            src_stride,
            w,
            rows,
            fh,
            r0 - 1,
            |v| v,
        );
        let (lo, end) = (t0 * w, (t0 + rows) * w);
        for (h8, &m) in hi[lo..end].iter_mut().zip(&mid[lo..end]) {
            *h8 = m >> 8;
        }
        v_pass_dyn::<i16, D, V_SPLIT_HI>(dst, dst_stride, mid, hi, 0, w, w, h, fv, |v| v, wr);
    });
}

/// The vertical pass of [`hv`] (`V_SPLIT_HI` sums) for blocks `W` wide
/// (2, 4, 8 or 16), `R` rows (`L = R * W` lanes, at least 8) at a time;
/// `h` is a multiple of `R`. Row `k` of `mid` and `hi` is intermediate row
/// `k` (stride `W`).
#[inline(always)]
fn v_mid<D, const T0: usize, const T1: usize, const W: usize, const R: usize, const L: usize>(
    dst: &mut [D],
    dst_stride: usize,
    mid: &[i16],
    hi: &[i16],
    h: usize,
    fv: &[i16; 8],
    wr: impl Fn(i16, i16) -> D + Copy,
) {
    // The taps splat to whole vectors once (instead of once per step).
    let fv: [[i16; L]; 8] = fv.map(|f| [f; L]);
    // Samples the taps of one step read, from the first tap's row.
    let span = (T1 - T0 - 1) * W + L;
    for i in 0..h / R {
        let b = (i * R + T0) * W;
        let (Some(m), Some(hh)) = (mid.get(b..b + span), hi.get(b..b + span)) else {
            return;
        };
        let mut a = [0i16; L];
        let mut s = [0i16; L];
        for t in T0..T1 {
            let o = (t - T0) * W;
            for l in 0..L {
                s[l] = s[l].wrapping_add(fv[t][l].wrapping_mul(m[o + l]));
                a[l] = a[l].wrapping_add(fv[t][l].wrapping_mul(hh[o + l]));
            }
        }
        for r in 0..R {
            let Some(d) = dst.get_mut((i * R + r) * dst_stride..(i * R + r) * dst_stride + W)
            else {
                return;
            };
            for c in 0..W {
                let l = r * W + c;
                d[c] = wr(a[l], s[l].wrapping_sub(a[l] << 8));
            }
        }
    }
}

/// Unscaled prediction (`xStep == yStep == 1024`, spec 7.11.3.4) written as
/// final pixels (`isCompound = 0`).
///
/// `fx` / `fy` select the horizontal / vertical row of `SUBPEL_FILTERS`
/// (0 regular, 1 smooth, 2 sharp, 3 bilinear, 4 four-tap regular,
/// 5 four-tap smooth); the caller has already applied the spec rule that
/// picks 4 or 5 when the block dimension is at most 4. `mx`, `my` are the
/// 1/16-sample fractions `(startX >> 6) & 15` and `(startY >> 6) & 15`.
///
/// `src_off` indexes the integer sample `(startX >> 10, startY >> 10)`;
/// `src` must be readable 3 samples left/up and 4 right/down of the
/// `w` x `h` block (when the block reaches outside the reference plane, pass
/// a copy made by [`emu_edge`] at `(ix - 3, iy - 3)` of size
/// `(w + 7) x (h + 7)` and `src_off = 3 * stride + 3`).
pub fn put<P: Pixel>(
    dst: &mut [P],
    dst_stride: usize,
    src: &[P],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: usize,
    my: usize,
    fx: usize,
    fy: usize,
    bd: u32,
) {
    let r0 = inter_round0(bd);
    let r1 = inter_round1(bd, false);
    // A constant for 8 bits so the clamp becomes a saturating pack.
    let max = if P::HBD {
        ((1i32 << bd) - 1) as i16
    } else {
        255
    };
    let fh = half(&SUBPEL_FILTERS[fx][mx]);
    let fv = half(&SUBPEL_FILTERS[fy][my]);
    match (mx != 0, my != 0) {
        (false, false) => match w {
            4 => copy_w::<P, 4>(dst, dst_stride, src, src_off, src_stride, w, h),
            8 => copy_w::<P, 8>(dst, dst_stride, src, src_off, src_stride, w, h),
            16 => copy_w::<P, 16>(dst, dst_stride, src, src_off, src_stride, w, h),
            32 => copy_w::<P, 32>(dst, dst_stride, src, src_off, src_stride, w, h),
            64 => copy_w::<P, 64>(dst, dst_stride, src, src_off, src_stride, w, h),
            _ => copy_w::<P, 0>(dst, dst_stride, src, src_off, src_stride, w, h),
        },
        (true, false) => {
            // The vertical filter is the identity: Round2(i * 128, r1).
            let sh = r1 - 7;
            let rnd = 1i16 << (sh - 1);
            h_pass_dyn::<P, P>(
                dst,
                dst_stride,
                src,
                src_off - 3,
                src_stride,
                w,
                h,
                &fh,
                r0 - 1,
                move |v| clip16::<P>((v + rnd) >> sh, max),
            );
        }
        (false, true) => {
            // The horizontal filter is exact (intermediate = px << (7 - r0)),
            // so pred = Round2(sum(f * px), r1 + r0 - 7) = Round2(2 * S, 7).
            let top = src_off - 3 * src_stride;
            if P::HBD {
                v_pass_dyn::<P, P, V_SPLIT>(
                    dst,
                    dst_stride,
                    src,
                    &[],
                    top,
                    src_stride,
                    w,
                    h,
                    &fv,
                    |p: P| p.to_i32() as i16,
                    move |a, b| clip16::<P>(a.wrapping_mul(4).wrapping_add((b + 32) >> 6), max),
                );
            } else {
                v_pass_dyn::<P, P, V_PLAIN>(
                    dst,
                    dst_stride,
                    src,
                    &[],
                    top,
                    src_stride,
                    w,
                    h,
                    &fv,
                    |p: P| p.to_i32() as i16,
                    move |_, b| clip16::<P>((b + 32) >> 6, max),
                );
            }
        }
        (true, true) => {
            // Round2(2 * (256 * A + B), r1) with n = r1 - 1 >= 8.
            let n = r1 - 1;
            let rnd = 1i16 << (n - 1);
            let sh = n - 8;
            hv::<P, P>(
                dst,
                dst_stride,
                src,
                src_off,
                src_stride,
                w,
                h,
                &fh,
                &fv,
                r0,
                move |a, b| clip16::<P>((a + ((b + rnd) >> 8)) >> sh, max),
            );
        }
    }
}

/// The horizontal pass of [`put`] with both fractions nonzero, for sharing
/// between predictions with the same `(startX >> 10, mx)` and filter `fx`:
/// writes the intermediate of `rows` source rows to `mid` and its high
/// parts (`>> 8`) to `hi` (both with stride `w`). `src_start` indexes the
/// sample 3 left of the first output sample of the first row.
pub fn put_h_rows<P: Pixel>(
    mid: &mut [i16],
    hi: &mut [i16],
    src: &[P],
    src_start: usize,
    src_stride: usize,
    w: usize,
    rows: usize,
    mx: usize,
    fx: usize,
    bd: u32,
) {
    let fh = half(&SUBPEL_FILTERS[fx][mx]);
    h_pass_dyn::<P, i16>(
        mid,
        w,
        src,
        src_start,
        src_stride,
        w,
        rows,
        &fh,
        inter_round0(bd) - 1,
        |v| v,
    );
    for (h8, &m) in hi[..rows * w].iter_mut().zip(&mid[..rows * w]) {
        *h8 = m >> 8;
    }
}

/// The vertical pass of [`put`] with both fractions nonzero, from the rows
/// [`put_h_rows`] made: `top` indexes (in `mid` and `hi`, stride `w`) the
/// intermediate row 3 above the block. Equals [`put`] of the same block.
pub fn put_v_rows<P: Pixel>(
    dst: &mut [P],
    dst_stride: usize,
    mid: &[i16],
    hi: &[i16],
    top: usize,
    w: usize,
    h: usize,
    my: usize,
    fy: usize,
    bd: u32,
) {
    let n = inter_round1(bd, false) - 1;
    let rnd = 1i16 << (n - 1);
    let sh = n - 8;
    let max = if P::HBD {
        ((1i32 << bd) - 1) as i16
    } else {
        255
    };
    let fv = half(&SUBPEL_FILTERS[fy][my]);
    v_pass_dyn::<i16, P, V_SPLIT_HI>(
        dst,
        dst_stride,
        mid,
        hi,
        top,
        w,
        w,
        h,
        &fv,
        |v| v,
        move |a, b| clip16::<P>((a + ((b + rnd) >> 8)) >> sh, max),
    );
}

/// Like [`put`], but writes the compound intermediate (`isCompound = 1`,
/// minus [`prep_bias`]) to `tmp` with stride `w`.
pub fn prep<P: Pixel>(
    tmp: &mut [i16],
    src: &[P],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: usize,
    my: usize,
    fx: usize,
    fy: usize,
    bd: u32,
) {
    let r0 = inter_round0(bd);
    let bias = prep_bias(bd) as i16;
    let fh = half(&SUBPEL_FILTERS[fx][mx]);
    let fv = half(&SUBPEL_FILTERS[fy][my]);
    match (mx != 0, my != 0) {
        (false, false) => match w {
            4 => prep_copy::<P, 4>(tmp, src, src_off, src_stride, w, h, 7 - r0, bias),
            8 => prep_copy::<P, 8>(tmp, src, src_off, src_stride, w, h, 7 - r0, bias),
            16 => prep_copy::<P, 16>(tmp, src, src_off, src_stride, w, h, 7 - r0, bias),
            32 => prep_copy::<P, 32>(tmp, src, src_off, src_stride, w, h, 7 - r0, bias),
            64 => prep_copy::<P, 64>(tmp, src, src_off, src_stride, w, h, 7 - r0, bias),
            _ => prep_copy::<P, 0>(tmp, src, src_off, src_stride, w, h, 7 - r0, bias),
        },
        (true, false) => h_pass_dyn::<P, i16>(
            tmp,
            w,
            src,
            src_off - 3,
            src_stride,
            w,
            h,
            &fh,
            r0 - 1,
            move |v| v - bias,
        ),
        (false, true) => {
            // pred = Round2(sum(f * px), r0) = Round2(256 * A + B, r0 - 1).
            let top = src_off - 3 * src_stride;
            let n = r0 - 1;
            let rnd = 1i16 << (n - 1);
            if P::HBD {
                let up = 8 - n;
                v_pass_dyn::<P, i16, V_SPLIT>(
                    tmp,
                    w,
                    src,
                    &[],
                    top,
                    src_stride,
                    w,
                    h,
                    &fv,
                    |p: P| p.to_i32() as i16,
                    move |a, b| (a << up).wrapping_add((b + rnd) >> n).wrapping_sub(bias),
                );
            } else {
                v_pass_dyn::<P, i16, V_PLAIN>(
                    tmp,
                    w,
                    src,
                    &[],
                    top,
                    src_stride,
                    w,
                    h,
                    &fv,
                    |p: P| p.to_i32() as i16,
                    move |_, b| ((b + rnd) >> n) - bias,
                );
            }
        }
        // pred = Round2(2 * (256 * A + B), 7) = 4 * A + Round2(B, 6).
        (true, true) => hv::<P, i16>(
            tmp,
            w,
            src,
            src_off,
            src_stride,
            w,
            h,
            &fh,
            &fv,
            r0,
            move |a, b| (a << 2).wrapping_add((b + 32) >> 6).wrapping_sub(bias),
        ),
    }
}

/// Copies a `w x h` block, `W` wide (0: any width), without a call to
/// `memcpy` per row for the constant widths.
#[inline(always)]
fn copy_w<P: Pixel, const W: usize>(
    dst: &mut [P],
    dst_stride: usize,
    src: &[P],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
) {
    let w = if W != 0 { W } else { w };
    for r in 0..h {
        let (Some(d), Some(s)) = (
            dst.get_mut(r * dst_stride..r * dst_stride + w),
            src.get(src_off + r * src_stride..src_off + r * src_stride + w),
        ) else {
            return;
        };
        if W != 0 {
            d[..W].copy_from_slice(&s[..W]);
        } else {
            d.copy_from_slice(s);
        }
    }
}

/// Whole-sample [`prep`] for blocks `W` wide (0: any width).
#[inline(always)]
fn prep_copy<P: Pixel, const W: usize>(
    tmp: &mut [i16],
    src: &[P],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    sh: u32,
    bias: i16,
) {
    let w = if W != 0 { W } else { w };
    for r in 0..h {
        let d = &mut tmp[r * w..][..w];
        let s = &src[src_off + r * src_stride..][..w];
        for c in 0..w {
            d[c] = ((s[c].to_i32() as i16) << sh) - bias;
        }
    }
}

/// Width and height of the source region the scaled kernels read, starting
/// 3 samples left of and 3 rows above the integer start position: columns
/// `-3 ..= ((x_frac + (w - 1) * x_step) >> 10) + 4` and rows
/// `-3 ..= ((y_frac + (h - 1) * y_step) >> 10) + 4`.
pub fn scaled_footprint(
    w: usize,
    h: usize,
    x_frac: usize,
    y_frac: usize,
    x_step: usize,
    y_step: usize,
) -> (usize, usize) {
    (
        ((x_frac + (w - 1) * x_step) >> SCALE_SUBPEL_BITS) + 8,
        ((y_frac + (h - 1) * y_step) >> SCALE_SUBPEL_BITS) + 8,
    )
}

#[inline]
fn scaled<P: Pixel, D>(
    dst: &mut [D],
    dst_stride: usize,
    src: &[P],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    x_frac: usize,
    y_frac: usize,
    x_step: usize,
    y_step: usize,
    fx: usize,
    fy: usize,
    r0: u32,
    wr: impl Fn(i32) -> D,
) {
    let (_, ih) = scaled_footprint(w, h, x_frac, y_frac, x_step, y_step);
    let mut cols = [0usize; 128];
    let mut fils = [0usize; 128];
    for c in 0..w {
        let p = x_frac + c * x_step;
        cols[c] = p >> SCALE_SUBPEL_BITS;
        fils[c] = (p >> 6) & SUBPEL_MASK as usize;
    }
    let bank = &SUBPEL_FILTERS[fx];
    let rnd0 = 1i32 << (r0 - 1);
    with_mid(w * ih, |mid| {
        let start = src_off - 3 * src_stride - 3;
        for k in 0..ih {
            let row = &src[start + k * src_stride..];
            let m = &mut mid[k * w..][..w];
            for c in 0..w {
                let f = &bank[fils[c]];
                let px = &row[cols[c]..][..8];
                let mut s = rnd0;
                for t in 0..8 {
                    s += i32::from(f[t]) * px[t].to_i32();
                }
                m[c] = (s >> r0) as i16;
            }
        }
        for r in 0..h {
            let p = y_frac + r * y_step;
            let base = p >> SCALE_SUBPEL_BITS;
            let f = &SUBPEL_FILTERS[fy][(p >> 6) & SUBPEL_MASK as usize];
            let d = &mut dst[r * dst_stride..][..w];
            for c in 0..w {
                let mut s = 0i32;
                for t in 0..8 {
                    s += i32::from(f[t]) * i32::from(mid[(base + t) * w + c]);
                }
                d[c] = wr(s);
            }
        }
    });
}

/// Scaled prediction (spec 7.11.3.4 with any `xStep`, `yStep`) written as
/// final pixels.
///
/// `src_off` indexes the integer sample `(startX >> 10, startY >> 10)`;
/// `x_frac = startX & 1023`, `y_frac = startY & 1023`; `x_step`, `y_step`
/// are the spec's `xStep`, `yStep`. `src` must be readable over the region
/// given by [`scaled_footprint`] (from 3 left / 3 up of `src_off`). The
/// filter rows `fx`, `fy` are as for [`put`].
pub fn put_scaled<P: Pixel>(
    dst: &mut [P],
    dst_stride: usize,
    src: &[P],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    x_frac: usize,
    y_frac: usize,
    x_step: usize,
    y_step: usize,
    fx: usize,
    fy: usize,
    bd: u32,
) {
    let r1 = inter_round1(bd, false);
    let rnd1 = rnd_of(r1);
    let max = ((1i32 << bd) - 1) as i16;
    scaled::<P, P>(
        dst,
        dst_stride,
        src,
        src_off,
        src_stride,
        w,
        h,
        x_frac,
        y_frac,
        x_step,
        y_step,
        fx,
        fy,
        inter_round0(bd),
        move |s| put_px::<P>(s, rnd1, r1, max),
    );
}

/// Like [`put_scaled`], but writes the compound intermediate to `tmp`
/// (stride `w`).
pub fn prep_scaled<P: Pixel>(
    tmp: &mut [i16],
    src: &[P],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    x_frac: usize,
    y_frac: usize,
    x_step: usize,
    y_step: usize,
    fx: usize,
    fy: usize,
    bd: u32,
) {
    let bias = prep_bias(bd);
    scaled::<P, i16>(
        tmp,
        w,
        src,
        src_off,
        src_stride,
        w,
        h,
        x_frac,
        y_frac,
        x_step,
        y_step,
        fx,
        fy,
        inter_round0(bd),
        move |s| (((s + 64) >> 7) - bias) as i16,
    );
}

// ---------------------------------------------------------------------------
// Compound combination
// ---------------------------------------------------------------------------

/// `COMPOUND_AVERAGE`:
/// `Clip1(Round2(preds[0] + preds[1], 1 + InterPostRound))`.
/// `t0`, `t1` are [`prep`] outputs with stride `w`.
pub fn avg<P: Pixel>(
    dst: &mut [P],
    dst_stride: usize,
    t0: &[i16],
    t1: &[i16],
    w: usize,
    h: usize,
    bd: u32,
) {
    match w {
        4 => avg_w::<P, 4>(dst, dst_stride, t0, t1, w, h, bd),
        8 => avg_w::<P, 8>(dst, dst_stride, t0, t1, w, h, bd),
        16 => avg_w::<P, 16>(dst, dst_stride, t0, t1, w, h, bd),
        32 => avg_w::<P, 32>(dst, dst_stride, t0, t1, w, h, bd),
        64 => avg_w::<P, 64>(dst, dst_stride, t0, t1, w, h, bd),
        _ => avg_w::<P, 0>(dst, dst_stride, t0, t1, w, h, bd),
    }
}

/// `COMPOUND_AVERAGE` of two whole-sample predictions, read straight from
/// the references: [`prep`] then [`avg`] reduce to `(a + b + 1) >> 1`.
pub fn avg_copy<P: Pixel>(
    dst: &mut [P],
    dst_stride: usize,
    (a, a_stride): (&[P], usize),
    (b, b_stride): (&[P], usize),
    w: usize,
    h: usize,
) {
    match w {
        4 => avg_copy_w::<P, 4>(dst, dst_stride, a, a_stride, b, b_stride, w, h),
        8 => avg_copy_w::<P, 8>(dst, dst_stride, a, a_stride, b, b_stride, w, h),
        16 => avg_copy_w::<P, 16>(dst, dst_stride, a, a_stride, b, b_stride, w, h),
        32 => avg_copy_w::<P, 32>(dst, dst_stride, a, a_stride, b, b_stride, w, h),
        64 => avg_copy_w::<P, 64>(dst, dst_stride, a, a_stride, b, b_stride, w, h),
        _ => avg_copy_w::<P, 0>(dst, dst_stride, a, a_stride, b, b_stride, w, h),
    }
}

#[inline(always)]
fn avg_copy_w<P: Pixel, const W: usize>(
    dst: &mut [P],
    dst_stride: usize,
    a: &[P],
    a_stride: usize,
    b: &[P],
    b_stride: usize,
    w: usize,
    h: usize,
) {
    let w = if W != 0 { W } else { w };
    for r in 0..h {
        let d = &mut dst[r * dst_stride..][..w];
        let x = &a[r * a_stride..][..w];
        let y = &b[r * b_stride..][..w];
        for c in 0..w {
            if P::HBD {
                d[c] = P::from_i32((x[c].to_i32() + y[c].to_i32() + 1) >> 1);
            } else {
                // A 16-bit sum: the rounding average of two bytes.
                let s = (x[c].to_i32() as u16 + y[c].to_i32() as u16 + 1) >> 1;
                d[c] = P::from_i32(i32::from(s));
            }
        }
    }
}

/// [`avg`] for blocks `W` wide (0: any width).
#[inline(always)]
fn avg_w<P: Pixel, const W: usize>(
    dst: &mut [P],
    dst_stride: usize,
    t0: &[i16],
    t1: &[i16],
    w: usize,
    h: usize,
    bd: u32,
) {
    let w = if W != 0 { W } else { w };
    let sh = 1 + inter_post_round(bd, true);
    let off = (1 << (sh - 1)) + 2 * prep_bias(bd);
    if !P::HBD {
        // 8-bit intermediates lie in -5150..=9245: the sum and the rounding
        // stay inside i16.
        let off = off as i16;
        for r in 0..h {
            let d = &mut dst[r * dst_stride..][..w];
            let a = &t0[r * w..][..w];
            let b = &t1[r * w..][..w];
            for c in 0..w {
                d[c] = clip16::<P>((a[c] + b[c] + off) >> sh, 255);
            }
        }
        return;
    }
    for r in 0..h {
        let d = &mut dst[r * dst_stride..][..w];
        let a = &t0[r * w..][..w];
        let b = &t1[r * w..][..w];
        for c in 0..w {
            d[c] = clip_px::<P>((i32::from(a[c]) + i32::from(b[c]) + off) >> sh, bd);
        }
    }
}

/// `COMPOUND_DISTANCE`: `Clip1(Round2(fwd * preds[0] + bck * preds[1],
/// 4 + InterPostRound))` with `fwd = FwdWeight`, `bck = BckWeight`.
pub fn w_avg<P: Pixel>(
    dst: &mut [P],
    dst_stride: usize,
    t0: &[i16],
    t1: &[i16],
    w: usize,
    h: usize,
    fwd: i32,
    bck: i32,
    bd: u32,
) {
    let sh = 4 + inter_post_round(bd, true);
    let off = (1 << (sh - 1)) + (fwd + bck) * prep_bias(bd);
    for r in 0..h {
        let d = &mut dst[r * dst_stride..][..w];
        let a = &t0[r * w..][..w];
        let b = &t1[r * w..][..w];
        for c in 0..w {
            d[c] = clip_px::<P>(
                (fwd * i32::from(a[c]) + bck * i32::from(b[c]) + off) >> sh,
                bd,
            );
        }
    }
}

/// Reads the blend weight of the mask blend process (7.11.3.14) for sample
/// `(x, y)` of the current plane.
#[inline(always)]
fn mask_at(mask: &[u8], stride: usize, x: usize, y: usize, sub_x: bool, sub_y: bool) -> i32 {
    match (sub_x, sub_y) {
        (false, false) => i32::from(mask[y * stride + x]),
        (true, false) => {
            let m = &mask[y * stride + 2 * x..];
            (i32::from(m[0]) + i32::from(m[1]) + 1) >> 1
        }
        _ => {
            let a = &mask[2 * y * stride + 2 * x..];
            let b = &mask[(2 * y + 1) * stride + 2 * x..];
            (i32::from(a[0]) + i32::from(a[1]) + i32::from(b[0]) + i32::from(b[1]) + 2) >> 2
        }
    }
}

/// Compound mask blend (7.11.3.14, `interintra == 0`):
/// `Clip1(Round2(m * preds[0] + (64 - m) * preds[1], 6 + InterPostRound))`.
///
/// For the wedge and difference-weighted compound types the mask is the
/// luma `Mask` array (luma block size, stride `mask_stride`), also for the
/// chroma planes: pass the chroma plane's `sub_x`/`sub_y` (both `false` for
/// luma) and the kernel averages 2 or 4 luma weights per sample as the spec
/// does. Note the spec's subsampled reads are `(!subX && !subY)` direct,
/// `subX && !subY` horizontal pair, otherwise the 2x2 quad.
pub fn mask_blend<P: Pixel>(
    dst: &mut [P],
    dst_stride: usize,
    t0: &[i16],
    t1: &[i16],
    w: usize,
    h: usize,
    mask: &[u8],
    mask_stride: usize,
    sub_x: bool,
    sub_y: bool,
    bd: u32,
) {
    let sh = 6 + inter_post_round(bd, true);
    let off = (1 << (sh - 1)) + 64 * prep_bias(bd);
    for r in 0..h {
        let d = &mut dst[r * dst_stride..][..w];
        let a = &t0[r * w..][..w];
        let b = &t1[r * w..][..w];
        if !sub_x && !sub_y {
            let m = &mask[r * mask_stride..][..w];
            for c in 0..w {
                let m = i32::from(m[c]);
                d[c] = clip_px::<P>(
                    (m * i32::from(a[c]) + (64 - m) * i32::from(b[c]) + off) >> sh,
                    bd,
                );
            }
        } else {
            for c in 0..w {
                let m = mask_at(mask, mask_stride, c, r, sub_x, sub_y);
                d[c] = clip_px::<P>(
                    (m * i32::from(a[c]) + (64 - m) * i32::from(b[c]) + off) >> sh,
                    bd,
                );
            }
        }
    }
}

/// Difference weight mask process (7.11.3.12) on the luma compound
/// intermediates `t0`, `t1` (stride `w`). Writes `mask` with stride `w`.
/// `inverse` is `mask_type`.
pub fn diffwtd_mask(
    mask: &mut [u8],
    t0: &[i16],
    t1: &[i16],
    w: usize,
    h: usize,
    inverse: bool,
    bd: u32,
) {
    let sh = (bd - 8) + inter_post_round(bd, true);
    let rnd = 1i32 << (sh - 1);
    let n = w * h;
    let (m, a, b) = (&mut mask[..n], &t0[..n], &t1[..n]);
    for i in 0..n {
        let diff = (i32::from(a[i]) - i32::from(b[i])).abs();
        let diff = (diff + rnd) >> sh;
        let v = (38 + diff / 16).min(64);
        m[i] = if inverse { 64 - v } else { v } as u8;
    }
}

// ---------------------------------------------------------------------------
// Inter-intra
// ---------------------------------------------------------------------------

/// Intra mode variant mask process (7.11.3.13) for a `w` x `h` block of the
/// current plane (any plane; the mask is at the plane's own resolution).
/// `mode` is `interintra_mode` (`II_DC_PRED` 0, `II_V_PRED` 1,
/// `II_H_PRED` 2, `II_SMOOTH_PRED` 3).
pub fn ii_smooth_mask(mask: &mut [u8], mask_stride: usize, w: usize, h: usize, mode: u8) {
    let scale = MAX_SB_SIZE as usize / w.max(h);
    let mode = i32::from(mode);
    for i in 0..h {
        let row = &mut mask[i * mask_stride..][..w];
        for j in 0..w {
            row[j] = if mode == II_V_PRED {
                II_WEIGHTS_1D[i * scale]
            } else if mode == II_H_PRED {
                II_WEIGHTS_1D[j * scale]
            } else if mode == II_SMOOTH_PRED {
                II_WEIGHTS_1D[i.min(j) * scale]
            } else {
                32
            };
        }
    }
}

/// Inter-intra mask blend (7.11.3.14, `interintra == 1`):
/// `dst = Round2(m * intra + (64 - m) * dst, 6)`, where `dst` holds the
/// final (clipped, `isCompound = 0`) inter prediction on entry.
///
/// For smooth inter-intra (`wedge_interintra == 0`) pass the
/// [`ii_smooth_mask`] built at the current plane's size with
/// `sub_x = sub_y = false`. For wedge inter-intra pass the luma wedge mask
/// ([`wedge_mask`] of `MiSize`, stride = luma block width) and the plane's
/// subsampling.
pub fn interintra_blend<P: Pixel>(
    dst: &mut [P],
    dst_stride: usize,
    intra: &[P],
    intra_stride: usize,
    w: usize,
    h: usize,
    mask: &[u8],
    mask_stride: usize,
    sub_x: bool,
    sub_y: bool,
) {
    for r in 0..h {
        let d = &mut dst[r * dst_stride..][..w];
        let s = &intra[r * intra_stride..][..w];
        for c in 0..w {
            let m = mask_at(mask, mask_stride, c, r, sub_x, sub_y);
            let v = (m * s[c].to_i32() + (64 - m) * d[c].to_i32() + 32) >> 6;
            d[c] = P::from_i32(v);
        }
    }
}

// ---------------------------------------------------------------------------
// Wedge masks (7.11.3.11)
// ---------------------------------------------------------------------------

/// Every `WedgeMasks[ bsize ][ sign ][ index ]` array of the specification.
pub struct WedgeMasks {
    data: Vec<u8>,
    /// Start of each block size's masks in `data`, `usize::MAX` when the
    /// block size has no wedges.
    offset: [usize; 22],
}

impl Default for WedgeMasks {
    fn default() -> Self {
        Self::new()
    }
}

impl WedgeMasks {
    /// Builds the tables (`initialise_wedge_mask_table` of the spec).
    pub fn new() -> Self {
        const N: usize = MASK_MASTER_SIZE as usize;
        let mut master = vec![[[0u8; N]; N]; 6];
        let (o63, ov) = (WEDGE_OBLIQUE63 as usize, WEDGE_VERTICAL as usize);
        for j in 0..N {
            let mut shift = (N / 4) as i32;
            for i in (0..N).step_by(2) {
                let k = (j as i32 - shift).clamp(0, N as i32 - 1) as usize;
                master[o63][i][j] = WEDGE_MASTER_OBLIQUE_EVEN[k];
                shift -= 1;
                let k = (j as i32 - shift).clamp(0, N as i32 - 1) as usize;
                master[o63][i + 1][j] = WEDGE_MASTER_OBLIQUE_ODD[k];
                master[ov][i][j] = WEDGE_MASTER_VERTICAL[j];
                master[ov][i + 1][j] = WEDGE_MASTER_VERTICAL[j];
            }
        }
        for i in 0..N {
            for j in 0..N {
                let msk = master[o63][i][j];
                master[WEDGE_OBLIQUE27 as usize][j][i] = msk;
                master[WEDGE_OBLIQUE117 as usize][i][N - 1 - j] = 64 - msk;
                master[WEDGE_OBLIQUE153 as usize][N - 1 - j][i] = 64 - msk;
                master[WEDGE_HORIZONTAL as usize][j][i] = master[ov][i][j];
            }
        }
        let mut data = Vec::new();
        let mut offset = [usize::MAX; 22];
        for bsize in 0..22 {
            if WEDGE_BITS[bsize] == 0 {
                continue;
            }
            let w4 = NUM_4X4_BLOCKS_WIDE[bsize];
            let h4 = NUM_4X4_BLOCKS_HIGH[bsize];
            let w = 4 * w4 as usize;
            let h = 4 * h4 as usize;
            let shape = if h4 > w4 {
                0
            } else if h4 < w4 {
                1
            } else {
                2
            };
            let start = data.len();
            offset[bsize] = start;
            data.resize(start + 2 * WEDGE_TYPES as usize * w * h, 0);
            for wedge in 0..WEDGE_TYPES as usize {
                let cb = &WEDGE_CODEBOOK[shape][wedge];
                let dir = cb[0] as usize;
                let xoff = N / 2 - ((cb[1] as usize * w) >> 3);
                let yoff = N / 2 - ((cb[2] as usize * h) >> 3);
                let m = &master[dir];
                let mut sum = 0usize;
                for i in 0..w {
                    sum += m[yoff][xoff + i] as usize;
                }
                for i in 1..h {
                    sum += m[yoff + i][xoff] as usize;
                }
                let avg = (sum + (w + h - 1) / 2) / (w + h - 1);
                let flip = usize::from(avg < 32);
                let at = |sign: usize| start + (sign * WEDGE_TYPES as usize + wedge) * w * h;
                let (pf, pn) = (at(flip), at(1 - flip));
                for i in 0..h {
                    for j in 0..w {
                        let v = m[yoff + i][xoff + j];
                        data[pf + i * w + j] = v;
                        data[pn + i * w + j] = 64 - v;
                    }
                }
            }
        }
        Self { data, offset }
    }

    /// `WedgeMasks[ bsize ][ sign ][ index ]` with stride = block width, or
    /// `None` when `bsize` has no wedges (`Wedge_Bits[ bsize ] == 0`) or the
    /// index is out of range.
    pub fn get(&self, bsize: usize, sign: bool, index: usize) -> Option<&[u8]> {
        let start = *self.offset.get(bsize)?;
        if start == usize::MAX || index >= WEDGE_TYPES as usize {
            return None;
        }
        let n = 16 * NUM_4X4_BLOCKS_WIDE[bsize] as usize * NUM_4X4_BLOCKS_HIGH[bsize] as usize;
        let at = start + (usize::from(sign) * WEDGE_TYPES as usize + index) * n;
        self.data.get(at..at + n)
    }
}

/// The process-wide wedge tables, built on first use.
pub fn wedge_masks() -> &'static WedgeMasks {
    static MASKS: OnceLock<WedgeMasks> = OnceLock::new();
    MASKS.get_or_init(WedgeMasks::new)
}

/// `WedgeMasks[ bsize ][ wedge_sign ][ wedge_index ]` (luma block size
/// `bsize` = `MiSize`), stride = block width. See [`WedgeMasks::get`].
pub fn wedge_mask(bsize: usize, sign: bool, index: usize) -> Option<&'static [u8]> {
    wedge_masks().get(bsize, sign, index)
}

// ---------------------------------------------------------------------------
// Overlapped motion compensation (7.11.3.9 / 7.11.3.10)
// ---------------------------------------------------------------------------

/// `get_obmc_mask( length )`.
pub fn obmc_mask(len: usize) -> &'static [u8] {
    match len {
        2 => &OBMC_MASK_2,
        4 => &OBMC_MASK_4,
        8 => &OBMC_MASK_8,
        16 => &OBMC_MASK_16,
        _ => &OBMC_MASK_32,
    }
}

/// Overlap blending (pass 0): blends the above neighbour's prediction
/// `pred` (clipped pixels, `w` x `overlap_h`, i.e. `predW` x `predH`) into
/// the top rows of `dst` with `get_obmc_mask(overlap_h)`.
pub fn obmc_blend_above<P: Pixel>(
    dst: &mut [P],
    dst_stride: usize,
    pred: &[P],
    pred_stride: usize,
    w: usize,
    overlap_h: usize,
) {
    let mask = obmc_mask(overlap_h);
    for i in 0..overlap_h {
        let m = i32::from(mask[i]);
        let d = &mut dst[i * dst_stride..][..w];
        let p = &pred[i * pred_stride..][..w];
        for j in 0..w {
            d[j] = P::from_i32((m * d[j].to_i32() + (64 - m) * p[j].to_i32() + 32) >> 6);
        }
    }
}

/// Overlap blending (pass 1): blends the left neighbour's prediction `pred`
/// (`overlap_w` x `h`, i.e. `predW` x `predH`) into the left columns of
/// `dst` with `get_obmc_mask(overlap_w)`.
pub fn obmc_blend_left<P: Pixel>(
    dst: &mut [P],
    dst_stride: usize,
    pred: &[P],
    pred_stride: usize,
    overlap_w: usize,
    h: usize,
) {
    let mask = &obmc_mask(overlap_w)[..overlap_w];
    for i in 0..h {
        let d = &mut dst[i * dst_stride..][..overlap_w];
        let p = &pred[i * pred_stride..][..overlap_w];
        for j in 0..overlap_w {
            let m = i32::from(mask[j]);
            d[j] = P::from_i32((m * d[j].to_i32() + (64 - m) * p[j].to_i32() + 32) >> 6);
        }
    }
}

// ---------------------------------------------------------------------------
// Warped motion (7.11.3.5 - 7.11.3.7)
// ---------------------------------------------------------------------------

#[inline]
fn round2_signed_64(x: i64, n: u32) -> i64 {
    if n == 0 {
        x
    } else if x >= 0 {
        (x + (1 << (n - 1))) >> n
    } else {
        -((-x + (1 << (n - 1))) >> n)
    }
}

/// Resolve divisor process (7.11.3.7): returns `(divShift, divFactor)`, or
/// `None` for `d == 0` (where `FloorLog2` is undefined).
pub fn resolve_divisor(d: i64) -> Option<(u32, i32)> {
    if d == 0 {
        return None;
    }
    let a = d.unsigned_abs();
    let n = 63 - a.leading_zeros();
    let e = a - (1u64 << n);
    let lut_bits = DIV_LUT_BITS as u32;
    let f = if n > lut_bits {
        let s = n - lut_bits;
        (e + (1 << (s - 1))) >> s
    } else {
        e << (lut_bits - n)
    };
    let factor = i32::from(DIV_LUT[f as usize]);
    Some((
        n + DIV_LUT_PREC_BITS as u32,
        if d < 0 { -factor } else { factor },
    ))
}

/// Setup shear process (7.11.3.6). Returns `[alpha, beta, gamma, delta]`
/// when `warpValid` is 1 and `None` when it is 0 (also when
/// `warpParams[2] == 0`, where the spec's divisor is undefined).
pub fn setup_shear(mat: &[i32; 6]) -> Option<[i32; 4]> {
    let one = 1i64 << WARPEDMODEL_PREC_BITS;
    let clip = |v: i64| v.clamp(-32768, 32767);
    let alpha0 = clip(i64::from(mat[2]) - one);
    let beta0 = clip(i64::from(mat[3]));
    let (div_shift, div_factor) = resolve_divisor(i64::from(mat[2]))?;
    let div_factor = i64::from(div_factor);
    let v = i64::from(mat[4]) << WARPEDMODEL_PREC_BITS;
    let gamma0 = clip(round2_signed_64(v * div_factor, div_shift));
    let w = i64::from(mat[3]) * i64::from(mat[4]);
    let delta0 = clip(i64::from(mat[5]) - round2_signed_64(w * div_factor, div_shift) - one);
    let rb = WARP_PARAM_REDUCE_BITS as u32;
    let reduce = |v: i64| (round2_signed_64(v, rb) << rb) as i32;
    let (alpha, beta, gamma, delta) = (
        reduce(alpha0),
        reduce(beta0),
        reduce(gamma0),
        reduce(delta0),
    );
    let lim = 1i32 << WARPEDMODEL_PREC_BITS;
    if 4 * alpha.abs() + 7 * beta.abs() >= lim || 4 * gamma.abs() + 4 * delta.abs() >= lim {
        return None;
    }
    Some([alpha, beta, gamma, delta])
}

/// Block warp process for one 8x8 block; `out(row, col, value)` receives
/// `Round2(sum, InterRound1)`.
#[inline(always)]
fn warp_core<P: Pixel>(
    refp: &[P],
    ref_stride: usize,
    plane_w: usize,
    plane_h: usize,
    mat: &[i32; 6],
    shear: &[i32; 4],
    x: usize,
    y: usize,
    sub_x: bool,
    sub_y: bool,
    r0: u32,
    r1: u32,
    mut out: impl FnMut(usize, usize, i32),
) {
    let [alpha, beta, gamma, delta] = *shear;
    let (sx_sh, sy_sh) = (u32::from(sub_x), u32::from(sub_y));
    let src_x = (x as i64 + 4) << sx_sh;
    let src_y = (y as i64 + 4) << sy_sh;
    let dst_x = i64::from(mat[2]) * src_x + i64::from(mat[3]) * src_y + i64::from(mat[0]);
    let dst_y = i64::from(mat[4]) * src_x + i64::from(mat[5]) * src_y + i64::from(mat[1]);
    let x4 = dst_x >> sx_sh;
    let y4 = dst_y >> sy_sh;
    let prec = WARPEDMODEL_PREC_BITS as u32;
    let mask = (1i64 << prec) - 1;
    let ix4 = x4 >> prec;
    let sx4 = (x4 & mask) as i32;
    let iy4 = y4 >> prec;
    let sy4 = (y4 & mask) as i32;
    let last_x = plane_w as i64 - 1;
    let last_y = plane_h as i64 - 1;

    // Column ix4 - 7 + k for k = 0..15 covers ix4 + i2 - 3 + i3.
    let mut cols = [0usize; 15];
    for k in 0..15 {
        cols[k] = (ix4 - 7 + k as i64).clamp(0, last_x) as usize;
    }
    let diff = WARPEDDIFF_PREC_BITS as u32;
    let mut inter = [[0i32; 8]; 15];
    for (k1, irow) in inter.iter_mut().enumerate() {
        let i1 = k1 as i32 - 7;
        let ry = (iy4 + i64::from(i1)).clamp(0, last_y) as usize;
        let row = &refp[ry * ref_stride..];
        let mut px = [0i32; 15];
        for k in 0..15 {
            px[k] = row[cols[k]].to_i32();
        }
        for k2 in 0..8 {
            let i2 = k2 as i32 - 4;
            let sx = sx4 + alpha * i2 + beta * i1;
            let offs = (round2(sx, diff) + WARPEDPIXEL_PREC_SHIFTS) as usize;
            let f = &WARPED_FILTERS[offs];
            let mut s = 0i32;
            for i3 in 0..8 {
                s += i32::from(f[i3]) * px[k2 + i3];
            }
            irow[k2] = round2(s, r0);
        }
    }
    for k1 in 0..8 {
        let i1 = k1 as i32 - 4;
        for k2 in 0..8 {
            let i2 = k2 as i32 - 4;
            let sy = sy4 + gamma * i2 + delta * i1;
            let offs = (round2(sy, diff) + WARPEDPIXEL_PREC_SHIFTS) as usize;
            let f = &WARPED_FILTERS[offs];
            let mut s = 0i32;
            for i3 in 0..8 {
                s += i32::from(f[i3]) * inter[k1 + i3][k2];
            }
            out(k1, k2, round2(s, r1));
        }
    }
}

/// Block warp process (7.11.3.5) for one 8x8 block, written as final pixels
/// (`isCompound = 0`) to the 8x8 area of `dst`.
///
/// `refp` is the whole reference plane (`plane_w` x `plane_h` samples, i.e.
/// `lastX + 1` by `lastY + 1`, with stride `ref_stride`); reads clamp to it
/// as the spec does. `mat` is `warpParams`, `shear` the output of
/// [`setup_shear`] for it. `(x, y)` is the top-left of this 8x8 block in the
/// current plane (block position plus `j8 * 8`, `i8 * 8`); `sub_x`, `sub_y`
/// the plane subsampling. Warped blocks are always multiples of 8 in each
/// plane, so the spec's clipping of the 8x8 area never applies.
pub fn warp_8x8_put<P: Pixel>(
    dst: &mut [P],
    dst_stride: usize,
    refp: &[P],
    ref_stride: usize,
    plane_w: usize,
    plane_h: usize,
    mat: &[i32; 6],
    shear: &[i32; 4],
    x: usize,
    y: usize,
    sub_x: bool,
    sub_y: bool,
    bd: u32,
) {
    warp_core(
        refp,
        ref_stride,
        plane_w,
        plane_h,
        mat,
        shear,
        x,
        y,
        sub_x,
        sub_y,
        inter_round0(bd),
        inter_round1(bd, false),
        |i, j, v| dst[i * dst_stride + j] = clip_px::<P>(v, bd),
    );
}

/// Like [`warp_8x8_put`], but writes the compound intermediate (minus
/// [`prep_bias`]) to the 8x8 area of `tmp` (stride `tmp_stride`, normally
/// the block width).
pub fn warp_8x8_prep<P: Pixel>(
    tmp: &mut [i16],
    tmp_stride: usize,
    refp: &[P],
    ref_stride: usize,
    plane_w: usize,
    plane_h: usize,
    mat: &[i32; 6],
    shear: &[i32; 4],
    x: usize,
    y: usize,
    sub_x: bool,
    sub_y: bool,
    bd: u32,
) {
    let bias = prep_bias(bd);
    warp_core(
        refp,
        ref_stride,
        plane_w,
        plane_h,
        mat,
        shear,
        x,
        y,
        sub_x,
        sub_y,
        inter_round0(bd),
        inter_round1(bd, true),
        |i, j, v| tmp[i * tmp_stride + j] = (v - bias) as i16,
    );
}

// ---------------------------------------------------------------------------
// Edge emulation
// ---------------------------------------------------------------------------

/// Copies the `bw` x `bh` block whose top-left is `(x, y)` in a
/// `plane_w` x `plane_h` plane into `dst`, clamping coordinates to the plane
/// like the spec's `Clip3( 0, lastX, .. )` / `Clip3( 0, lastY, .. )`.
pub fn emu_edge<P: Pixel>(
    dst: &mut [P],
    dst_stride: usize,
    src: &[P],
    src_stride: usize,
    plane_w: usize,
    plane_h: usize,
    x: isize,
    y: isize,
    bw: usize,
    bh: usize,
) {
    emu_edge_rows(
        dst,
        dst_stride,
        |sy| src.get(sy * src_stride..),
        plane_w,
        plane_h,
        x,
        y,
        bw,
        bh,
    );
}

/// [`emu_edge`] with the plane's rows from `row` (the samples from the
/// start of row `y` on). Returns false, leaving the rest of `dst`, when a
/// row is missing or shorter than `plane_w`.
pub fn emu_edge_rows<'a, P: Pixel + 'a>(
    dst: &mut [P],
    dst_stride: usize,
    row: impl Fn(usize) -> Option<&'a [P]>,
    plane_w: usize,
    plane_h: usize,
    x: isize,
    y: isize,
    bw: usize,
    bh: usize,
) -> bool {
    let pw = plane_w as isize;
    // Columns [0, left) clamp to 0, [left, right) copy, [right, bw) clamp to
    // the last column.
    let left = (-x).clamp(0, bw as isize) as usize;
    let right = (pw - x).clamp(left as isize, bw as isize) as usize;
    for r in 0..bh {
        let sy = (y + r as isize).clamp(0, plane_h as isize - 1) as usize;
        let Some(srow) = row(sy).and_then(|s| s.get(..plane_w)) else {
            return false;
        };
        let d = &mut dst[r * dst_stride..][..bw];
        d[..left].fill(srow[0]);
        if right > left {
            let sx = (x + left as isize) as usize;
            d[left..right].copy_from_slice(&srow[sx..sx + right - left]);
        }
        d[right..].fill(srow[plane_w - 1]);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::{
        Pixel, avg, diffwtd_mask, emu_edge, ii_smooth_mask, interintra_blend, mask_blend,
        obmc_blend_above, obmc_blend_left, obmc_mask, prep, prep_bias, prep_scaled, put,
        put_h_rows, put_scaled, put_v_rows, scaled_footprint, setup_shear, w_avg, warp_8x8_prep,
        warp_8x8_put, wedge_mask,
    };
    use crate::spec::tables::{
        DIV_LUT, II_WEIGHTS_1D, NUM_4X4_BLOCKS_HIGH, NUM_4X4_BLOCKS_WIDE, SUBPEL_FILTERS,
        WARPED_FILTERS, WEDGE_BITS, WEDGE_CODEBOOK, WEDGE_MASTER_OBLIQUE_EVEN,
        WEDGE_MASTER_OBLIQUE_ODD, WEDGE_MASTER_VERTICAL,
    };

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
        fn range(&mut self, lo: i64, hi: i64) -> i64 {
            lo + (self.next() % (hi - lo + 1) as u64) as i64
        }
    }

    // ---- literal spec transliterations ----

    fn round2(x: i64, n: u32) -> i64 {
        if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
    }
    fn round2s(x: i64, n: u32) -> i64 {
        if x >= 0 { round2(x, n) } else { -round2(-x, n) }
    }
    fn clip3(lo: i64, hi: i64, v: i64) -> i64 {
        if v < lo {
            lo
        } else if v > hi {
            hi
        } else {
            v
        }
    }
    fn rounding(bd: u32, compound: bool) -> (u32, u32, u32) {
        let mut r0 = 3;
        let mut r1 = if compound { 7 } else { 11 };
        if bd == 12 {
            r0 += 2;
        }
        if bd == 12 && !compound {
            r1 -= 2;
        }
        (r0, r1, 14 - r0 - r1)
    }

    struct Plane {
        px: Vec<i64>,
        w: usize,
        h: usize,
    }
    impl Plane {
        fn at(&self, y: i64, x: i64) -> i64 {
            let y = clip3(0, self.h as i64 - 1, y) as usize;
            let x = clip3(0, self.w as i64 - 1, x) as usize;
            self.px[y * self.w + x]
        }
    }

    /// Block inter prediction process, 7.11.3.4.
    fn ref_block_inter(
        p: &Plane,
        x: i64,
        y: i64,
        xs: i64,
        ys: i64,
        w: usize,
        h: usize,
        fx: usize,
        fy: usize,
        bd: u32,
        compound: bool,
    ) -> Vec<i64> {
        let (r0, r1, _) = rounding(bd, compound);
        let ih = ((((h as i64) - 1) * ys + (1 << 10) - 1) >> 10) as usize + 8;
        let mut inter = vec![0i64; ih * w];
        for r in 0..ih {
            for c in 0..w {
                let mut s = 0;
                let pp = x + xs * c as i64;
                for t in 0..8 {
                    s += i64::from(SUBPEL_FILTERS[fx][((pp >> 6) & 15) as usize][t])
                        * p.at((y >> 10) + r as i64 - 3, (pp >> 10) + t as i64 - 3);
                }
                inter[r * w + c] = round2(s, r0);
            }
        }
        let mut pred = vec![0i64; h * w];
        for r in 0..h {
            for c in 0..w {
                let mut s = 0;
                let pp = (y & 1023) + ys * r as i64;
                for t in 0..8 {
                    s += i64::from(SUBPEL_FILTERS[fy][((pp >> 6) & 15) as usize][t])
                        * inter[((pp >> 10) as usize + t) * w + c];
                }
                pred[r * w + c] = round2(s, r1);
            }
        }
        pred
    }

    fn ref_setup_shear(m: &[i32; 6]) -> Option<[i32; 4]> {
        let m: Vec<i64> = m.iter().map(|&v| i64::from(v)).collect();
        let alpha0 = clip3(-32768, 32767, m[2] - (1 << 16));
        let beta0 = clip3(-32768, 32767, m[3]);
        let d = m[2];
        let n = 63 - d.unsigned_abs().leading_zeros() as i64;
        let e = d.abs() - (1 << n);
        let f = if n > 8 {
            round2(e, (n - 8) as u32)
        } else {
            e << (8 - n)
        };
        let div_shift = (n + 14) as u32;
        let div_factor = if d < 0 {
            -i64::from(DIV_LUT[f as usize])
        } else {
            i64::from(DIV_LUT[f as usize])
        };
        let v = m[4] << 16;
        let gamma0 = clip3(-32768, 32767, round2s(v * div_factor, div_shift));
        let w = m[3] * m[4];
        let delta0 = clip3(
            -32768,
            32767,
            m[5] - round2s(w * div_factor, div_shift) - (1 << 16),
        );
        let a = round2s(alpha0, 6) << 6;
        let b = round2s(beta0, 6) << 6;
        let g = round2s(gamma0, 6) << 6;
        let dl = round2s(delta0, 6) << 6;
        if 4 * a.abs() + 7 * b.abs() >= (1 << 16) {
            return None;
        }
        if 4 * g.abs() + 4 * dl.abs() >= (1 << 16) {
            return None;
        }
        Some([a as i32, b as i32, g as i32, dl as i32])
    }

    /// Block warp process 7.11.3.5 for one full 8x8 block at (x, y).
    fn ref_warp(
        p: &Plane,
        m: &[i32; 6],
        x: i64,
        y: i64,
        sub_x: u32,
        sub_y: u32,
        bd: u32,
        compound: bool,
    ) -> Vec<i64> {
        let (r0, r1, _) = rounding(bd, compound);
        let wp: Vec<i64> = m.iter().map(|&v| i64::from(v)).collect();
        let src_x = (x + 4) << sub_x;
        let src_y = (y + 4) << sub_y;
        let dst_x = wp[2] * src_x + wp[3] * src_y + wp[0];
        let dst_y = wp[4] * src_x + wp[5] * src_y + wp[1];
        let s = ref_setup_shear(m).unwrap_or([0; 4]);
        let (alpha, beta, gamma, delta) = (
            i64::from(s[0]),
            i64::from(s[1]),
            i64::from(s[2]),
            i64::from(s[3]),
        );
        let x4 = dst_x >> sub_x;
        let y4 = dst_y >> sub_y;
        let ix4 = x4 >> 16;
        let sx4 = x4 & ((1 << 16) - 1);
        let iy4 = y4 >> 16;
        let sy4 = y4 & ((1 << 16) - 1);
        let mut inter = [[0i64; 8]; 15];
        for i1 in -7i64..8 {
            for i2 in -4i64..4 {
                let sx = sx4 + alpha * i2 + beta * i1;
                let offs = (round2(sx, 10) + 64) as usize;
                let mut s = 0;
                for i3 in 0..8 {
                    s += i64::from(WARPED_FILTERS[offs][i3 as usize])
                        * p.at(iy4 + i1, ix4 + i2 - 3 + i3);
                }
                inter[(i1 + 7) as usize][(i2 + 4) as usize] = round2(s, r0);
            }
        }
        let mut pred = vec![0i64; 64];
        for i1 in -4i64..4 {
            for i2 in -4i64..4 {
                let sy = sy4 + gamma * i2 + delta * i1;
                let offs = (round2(sy, 10) + 64) as usize;
                let mut s = 0;
                for i3 in 0..8 {
                    s += i64::from(WARPED_FILTERS[offs][i3])
                        * inter[(i1 + i3 as i64 + 4) as usize][(i2 + 4) as usize];
                }
                pred[((i1 + 4) * 8 + i2 + 4) as usize] = round2(s, r1);
            }
        }
        pred
    }

    fn ref_wedge_masks() -> Vec<Option<Vec<Vec<Vec<u8>>>>> {
        // [bsize] -> [sign][wedge] -> w*h
        let n = 64usize;
        let mut mm = vec![vec![vec![0i64; n]; n]; 6];
        for j in 0..n {
            let mut shift = 16i64;
            let mut i = 0;
            while i < n {
                mm[3][i][j] =
                    i64::from(WEDGE_MASTER_OBLIQUE_EVEN[clip3(0, 63, j as i64 - shift) as usize]);
                shift -= 1;
                mm[3][i + 1][j] =
                    i64::from(WEDGE_MASTER_OBLIQUE_ODD[clip3(0, 63, j as i64 - shift) as usize]);
                mm[1][i][j] = i64::from(WEDGE_MASTER_VERTICAL[j]);
                mm[1][i + 1][j] = i64::from(WEDGE_MASTER_VERTICAL[j]);
                i += 2;
            }
        }
        for i in 0..n {
            for j in 0..n {
                let msk = mm[3][i][j];
                mm[2][j][i] = msk;
                mm[4][i][n - 1 - j] = 64 - msk;
                mm[5][n - 1 - j][i] = 64 - msk;
                mm[0][j][i] = mm[1][i][j];
            }
        }
        let mut out = Vec::new();
        for bsize in 0..22 {
            if bsize < 3 || WEDGE_BITS[bsize] == 0 {
                out.push(None);
                continue;
            }
            let w4 = NUM_4X4_BLOCKS_WIDE[bsize];
            let h4 = NUM_4X4_BLOCKS_HIGH[bsize];
            let shape = if h4 > w4 {
                0
            } else if h4 < w4 {
                1
            } else {
                2
            };
            let w = 4 * w4 as usize;
            let h = 4 * h4 as usize;
            let mut res = vec![vec![vec![0u8; w * h]; 16]; 2];
            for wedge in 0..16 {
                let dir = WEDGE_CODEBOOK[shape][wedge][0] as usize;
                let xoff = 32 - ((WEDGE_CODEBOOK[shape][wedge][1] as usize * w) >> 3);
                let yoff = 32 - ((WEDGE_CODEBOOK[shape][wedge][2] as usize * h) >> 3);
                let mut sum = 0;
                for i in 0..w {
                    sum += mm[dir][yoff][xoff + i];
                }
                for i in 1..h {
                    sum += mm[dir][yoff + i][xoff];
                }
                let avg = (sum + (w + h - 1) as i64 / 2) / (w + h - 1) as i64;
                let flip = usize::from(avg < 32);
                for i in 0..h {
                    for j in 0..w {
                        res[flip][wedge][i * w + j] = mm[dir][yoff + i][xoff + j] as u8;
                        res[1 - flip][wedge][i * w + j] = (64 - mm[dir][yoff + i][xoff + j]) as u8;
                    }
                }
            }
            out.push(Some(res));
        }
        out
    }

    // ---- helpers ----

    fn make_plane(rng: &mut Rng, w: usize, h: usize, bd: u32) -> Plane {
        let max = (1i64 << bd) - 1;
        let mode = rng.below(3);
        let px = (0..w * h)
            .map(|_| match mode {
                0 => rng.range(0, max),
                1 => {
                    if rng.below(2) == 0 {
                        0
                    } else {
                        max
                    }
                }
                _ => {
                    let v = rng.range(0, max);
                    if rng.below(4) == 0 {
                        max - v / 8
                    } else {
                        v / 8
                    }
                }
            })
            .collect();
        Plane { px, w, h }
    }

    fn to_px<P: Pixel>(p: &Plane) -> Vec<P> {
        p.px.iter().map(|&v| P::from_i32(v as i32)).collect()
    }

    const SIZES: [usize; 7] = [2, 4, 8, 16, 32, 64, 128];

    /// Runs the unscaled or scaled kernels against the reference; returns
    /// the reference preds and our prep output when compound.
    fn check_mc<P: Pixel>(rng: &mut Rng, bd: u32, scaled: bool, compound: bool) {
        let w = SIZES[rng.below(7)];
        let h = SIZES[rng.below(7)];
        let fx = rng.below(6);
        let fy = rng.below(6);
        let (pw, ph) = (1 + rng.below(200), 1 + rng.below(200));
        let plane = make_plane(rng, pw, ph, bd);
        let pp = to_px::<P>(&plane);
        let (xs, ys) = if scaled {
            (rng.range(64, 2048), rng.range(64, 2048))
        } else {
            (1024, 1024)
        };
        let mut pos = |n: usize| -> i64 {
            let ip = rng.range(-150, n as i64 + 20);
            let frac = if !scaled && rng.below(4) == 0 {
                rng.range(0, 63)
            } else {
                rng.range(0, 1023)
            };
            ip * 1024 + frac
        };
        let x = pos(plane.w);
        let y = pos(plane.h);
        let pred = ref_block_inter(&plane, x, y, xs, ys, w, h, fx, fy, bd, compound);
        let (ix, iy) = (x >> 10, y >> 10);
        let (fw, fh) = if scaled {
            scaled_footprint(
                w,
                h,
                (x & 1023) as usize,
                (y & 1023) as usize,
                xs as usize,
                ys as usize,
            )
        } else {
            (w + 7, h + 7)
        };
        let stride = fw + rng.below(3);
        let mut emu = vec![P::default(); stride * fh];
        emu_edge(
            &mut emu,
            stride,
            &pp,
            plane.w,
            plane.w,
            plane.h,
            (ix - 3) as isize,
            (iy - 3) as isize,
            fw,
            fh,
        );
        let off = 3 * stride + 3;
        let (mx, my) = (((x >> 6) & 15) as usize, ((y >> 6) & 15) as usize);
        let ctx = (bd, w, h, fx, fy, x, y, xs, ys);
        if compound {
            let mut tmp = vec![0i16; w * h];
            if scaled {
                prep_scaled(
                    &mut tmp,
                    &emu,
                    off,
                    stride,
                    w,
                    h,
                    (x & 1023) as usize,
                    (y & 1023) as usize,
                    xs as usize,
                    ys as usize,
                    fx,
                    fy,
                    bd,
                );
            } else {
                prep(&mut tmp, &emu, off, stride, w, h, mx, my, fx, fy, bd);
            }
            let bias = i64::from(prep_bias(bd));
            for i in 0..w * h {
                assert_eq!(i64::from(tmp[i]) + bias, pred[i], "prep {ctx:?} at {i}");
            }
        } else {
            let ds = w + 5;
            let mut dst = vec![P::default(); ds * h];
            if scaled {
                put_scaled(
                    &mut dst,
                    ds,
                    &emu,
                    off,
                    stride,
                    w,
                    h,
                    (x & 1023) as usize,
                    (y & 1023) as usize,
                    xs as usize,
                    ys as usize,
                    fx,
                    fy,
                    bd,
                );
            } else {
                put(&mut dst, ds, &emu, off, stride, w, h, mx, my, fx, fy, bd);
            }
            let max = (1i64 << bd) - 1;
            for r in 0..h {
                for c in 0..w {
                    assert_eq!(
                        i64::from(dst[r * ds + c].to_i32()),
                        clip3(0, max, pred[r * w + c]),
                        "put {ctx:?} at {r},{c}"
                    );
                }
            }
        }
    }

    fn run_mc(scaled: bool, compound: bool, iters: usize) {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ (u64::from(scaled) << 3 | u64::from(compound)));
        for _ in 0..iters {
            check_mc::<u8>(&mut rng, 8, scaled, compound);
            check_mc::<u16>(&mut rng, 10, scaled, compound);
            check_mc::<u16>(&mut rng, 12, scaled, compound);
        }
    }

    #[test]
    fn put_unscaled() {
        run_mc(false, false, 1500);
    }

    /// Shared horizontal rows (`put_h_rows`, `put_v_rows`) give `put`.
    fn check_shared_rows<P: Pixel>(rng: &mut Rng, bd: u32) {
        let sizes = [2usize, 4, 8, 16, 32, 64];
        let w = sizes[rng.below(sizes.len())];
        let h = sizes[rng.below(sizes.len())];
        let stride = w + 16;
        let rows = h + 12;
        let src: Vec<P> = (0..stride * rows)
            .map(|_| P::from_i32(rng.below(1 << bd) as i32))
            .collect();
        let fx = rng.below(6);
        let fy = rng.below(6);
        let mx = 1 + rng.below(15);
        // Rows from 1 above to 3 below the block's own rows.
        let n = h + 11;
        let mut mid = vec![0i16; n * w];
        let mut hi = vec![0i16; n * w];
        put_h_rows(
            &mut mid,
            &mut hi,
            &src,
            stride + 1,
            stride,
            w,
            n,
            mx,
            fx,
            bd,
        );
        for dy in 0..4 {
            let my = 1 + rng.below(15);
            // The block starts at source row 4 + dy, column 4.
            let mut want = vec![P::from_i32(0); w * h];
            put(
                &mut want,
                w,
                &src,
                (4 + dy) * stride + 4,
                stride,
                w,
                h,
                mx,
                my,
                fx,
                fy,
                bd,
            );
            let mut got = vec![P::from_i32(0); w * h];
            put_v_rows(&mut got, w, &mid, &hi, dy * w, w, h, my, fy, bd);
            assert_eq!(got, want, "w {w} h {h} mx {mx} my {my} bd {bd}");
        }
    }

    #[test]
    fn put_shared_rows() {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        for _ in 0..400 {
            check_shared_rows::<u8>(&mut rng, 8);
            check_shared_rows::<u16>(&mut rng, 10);
            check_shared_rows::<u16>(&mut rng, 12);
        }
    }

    #[test]
    fn prep_unscaled() {
        run_mc(false, true, 1500);
    }

    #[test]
    fn put_scaled_matches() {
        run_mc(true, false, 600);
    }

    #[test]
    fn prep_scaled_matches() {
        run_mc(true, true, 600);
    }

    /// Every (filter, fraction) pair, both directions, small block.
    #[test]
    fn all_filters_and_fractions() {
        let mut rng = Rng(77);
        for bd in [8u32, 10, 12] {
            let plane = make_plane(&mut rng, 40, 40, bd);
            for f in 0..6 {
                for mx in 0..16 {
                    for my in 0..16 {
                        let (w, h) = (8, 4);
                        let x = (10 << 10) + (mx << 6);
                        let y = (12 << 10) + (my << 6);
                        for compound in [false, true] {
                            let pred = ref_block_inter(
                                &plane,
                                x,
                                y,
                                1024,
                                1024,
                                w,
                                h,
                                f,
                                5 - f,
                                bd,
                                compound,
                            );
                            let off = 12 * 40 + 10;
                            let bias = i64::from(prep_bias(bd));
                            let max = (1i64 << bd) - 1;
                            let (mxu, myu) = (mx as usize, my as usize);
                            if bd == 8 {
                                let pp = to_px::<u8>(&plane);
                                let mut d = vec![0u8; w * h];
                                let mut t = vec![0i16; w * h];
                                put(&mut d, w, &pp, off, 40, w, h, mxu, myu, f, 5 - f, bd);
                                prep(&mut t, &pp, off, 40, w, h, mxu, myu, f, 5 - f, bd);
                                for i in 0..w * h {
                                    if compound {
                                        assert_eq!(i64::from(t[i]) + bias, pred[i]);
                                    } else {
                                        assert_eq!(i64::from(d[i]), clip3(0, max, pred[i]));
                                    }
                                }
                            } else {
                                let pp = to_px::<u16>(&plane);
                                let mut d = vec![0u16; w * h];
                                let mut t = vec![0i16; w * h];
                                put(&mut d, w, &pp, off, 40, w, h, mxu, myu, f, 5 - f, bd);
                                prep(&mut t, &pp, off, 40, w, h, mxu, myu, f, 5 - f, bd);
                                for i in 0..w * h {
                                    if compound {
                                        assert_eq!(i64::from(t[i]) + bias, pred[i]);
                                    } else {
                                        assert_eq!(i64::from(d[i]), clip3(0, max, pred[i]));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Source built to drive the sharp filter to its largest and smallest
    /// outputs, checking the compound intermediate never overflows `i16`.
    #[test]
    fn extreme_compound_range() {
        for bd in [8u32, 10, 12] {
            let max = (1i64 << bd) - 1;
            for (frac, high) in [(8usize, true), (8, false), (4, true), (12, false)] {
                let f = &SUBPEL_FILTERS[2][frac];
                let (w, h) = (8usize, 8usize);
                let pw = 32;
                let mut px = vec![0i64; pw * pw];
                for r in 0..pw {
                    // Row r feeds vertical tap r for output row 0.
                    let vt = f[r.min(7)];
                    let want_high = (vt >= 0) == high;
                    for c in 0..pw {
                        let ht = f[c.min(7)];
                        px[r * pw + c] = if (ht >= 0) == want_high { max } else { 0 };
                    }
                }
                let plane = Plane { px, w: pw, h: pw };
                let x = (3 << 10) + ((frac as i64) << 6);
                let y = (3 << 10) + ((frac as i64) << 6);
                let pred = ref_block_inter(&plane, x, y, 1024, 1024, w, h, 2, 2, bd, true);
                let off = 3 * pw + 3;
                let mut t = vec![0i16; w * h];
                if bd == 8 {
                    prep(
                        &mut t,
                        &to_px::<u8>(&plane),
                        off,
                        pw,
                        w,
                        h,
                        frac,
                        frac,
                        2,
                        2,
                        bd,
                    );
                } else {
                    prep(
                        &mut t,
                        &to_px::<u16>(&plane),
                        off,
                        pw,
                        w,
                        h,
                        frac,
                        frac,
                        2,
                        2,
                        bd,
                    );
                }
                if bd > 8 && high && frac == 8 {
                    assert!(pred[0] > 32767, "{}", pred[0]);
                }
                let bias = i64::from(prep_bias(bd));
                for i in 0..w * h {
                    assert_eq!(i64::from(t[i]) + bias, pred[i], "bd {bd} frac {frac} {i}");
                }
            }
        }
    }

    fn check_compound<P: Pixel>(rng: &mut Rng, bd: u32) {
        let w = SIZES[rng.below(7)];
        let h = SIZES[rng.below(7)];
        let (_, _, post) = rounding(bd, true);
        let max = (1i64 << bd) - 1;
        // Two predictions from real filtering so the values are realistic.
        let plane = make_plane(rng, 150, 150, bd);
        let pp = to_px::<P>(&plane);
        let mut preds = Vec::new();
        let mut tmps = Vec::new();
        for _ in 0..2 {
            let (ix, iy) = (rng.range(0, 10), rng.range(0, 10));
            let (mx, my) = (rng.below(16), rng.below(16));
            let (fx, fy) = (rng.below(6), rng.below(6));
            let x = (ix + 3) * 1024 + (mx as i64) * 64;
            let y = (iy + 3) * 1024 + (my as i64) * 64;
            preds.push(ref_block_inter(
                &plane, x, y, 1024, 1024, w, h, fx, fy, bd, true,
            ));
            let mut t = vec![0i16; w * h];
            let off = (iy as usize + 3) * 150 + ix as usize + 3;
            prep(&mut t, &pp, off, 150, w, h, mx, my, fx, fy, bd);
            tmps.push(t);
        }
        let (p0, p1) = (&preds[0], &preds[1]);
        let (t0, t1) = (&tmps[0], &tmps[1]);
        let ds = w + 3;
        let mut d = vec![P::default(); ds * h];
        avg(&mut d, ds, t0, t1, w, h, bd);
        for r in 0..h {
            for c in 0..w {
                let i = r * w + c;
                let e = clip3(0, max, round2(p0[i] + p1[i], 1 + post));
                assert_eq!(i64::from(d[r * ds + c].to_i32()), e, "avg");
            }
        }
        let k = rng.below(4);
        let order = rng.below(2);
        let fwd = [9, 11, 12, 13][k];
        let fwd = if order == 1 { fwd } else { 16 - fwd };
        let bck = 16 - fwd;
        w_avg(&mut d, ds, t0, t1, w, h, fwd, bck, bd);
        for r in 0..h {
            for c in 0..w {
                let i = r * w + c;
                let e = clip3(
                    0,
                    max,
                    round2(i64::from(fwd) * p0[i] + i64::from(bck) * p1[i], 4 + post),
                );
                assert_eq!(i64::from(d[r * ds + c].to_i32()), e, "w_avg");
            }
        }
        // diffwtd mask
        let inverse = rng.below(2) == 1;
        let mut m = vec![0u8; w * h];
        diffwtd_mask(&mut m, t0, t1, w, h, inverse, bd);
        for i in 0..w * h {
            let diff = round2((p0[i] - p1[i]).abs(), (bd - 8) + post);
            let mm = clip3(0, 64, 38 + diff / 16);
            let e = if inverse { 64 - mm } else { mm };
            assert_eq!(i64::from(m[i]), e, "diffwtd");
        }
        // mask blend, luma and subsampled (chroma block = luma >> sub)
        for (sx, sy) in [(false, false), (true, false), (true, true)] {
            let (lw, lh) = (w << usize::from(sx), h << usize::from(sy));
            let ms = lw + rng.below(3);
            let mask: Vec<u8> = (0..ms * lh).map(|_| rng.below(65) as u8).collect();
            mask_blend(&mut d, ds, t0, t1, w, h, &mask, ms, sx, sy, bd);
            for y in 0..h {
                for x in 0..w {
                    let mk = |yy: usize, xx: usize| i64::from(mask[yy * ms + xx]);
                    let mv = if !sx && !sy {
                        mk(y, x)
                    } else if sx && !sy {
                        round2(mk(y, 2 * x) + mk(y, 2 * x + 1), 1)
                    } else {
                        round2(
                            mk(2 * y, 2 * x)
                                + mk(2 * y, 2 * x + 1)
                                + mk(2 * y + 1, 2 * x)
                                + mk(2 * y + 1, 2 * x + 1),
                            2,
                        )
                    };
                    let i = y * w + x;
                    let e = clip3(0, max, round2(mv * p0[i] + (64 - mv) * p1[i], 6 + post));
                    assert_eq!(i64::from(d[y * ds + x].to_i32()), e, "mask_blend {sx} {sy}");
                }
            }
        }
    }

    #[test]
    fn compound_combinations() {
        let mut rng = Rng(12345);
        for _ in 0..300 {
            check_compound::<u8>(&mut rng, 8);
            check_compound::<u16>(&mut rng, 10);
            check_compound::<u16>(&mut rng, 12);
        }
    }

    #[test]
    fn interintra_and_obmc() {
        let mut rng = Rng(999);
        for _ in 0..400 {
            let bd = [10u32, 12][rng.below(2)];
            let max = (1i64 << bd) - 1;
            let w = SIZES[rng.below(6)];
            let h = SIZES[rng.below(6)];
            let ds = w + 2;
            let inter: Vec<u16> = (0..ds * h).map(|_| rng.range(0, max) as u16).collect();
            let intra: Vec<u16> = (0..w * h).map(|_| rng.range(0, max) as u16).collect();
            // smooth interintra
            let mode = rng.below(4) as u8;
            let ms = w + 1;
            let mut mask = vec![0u8; ms * h];
            ii_smooth_mask(&mut mask, ms, w, h, mode);
            let scale = 128 / w.max(h);
            for i in 0..h {
                for j in 0..w {
                    let e = match mode {
                        1 => II_WEIGHTS_1D[i * scale],
                        2 => II_WEIGHTS_1D[j * scale],
                        3 => II_WEIGHTS_1D[i.min(j) * scale],
                        _ => 32,
                    };
                    assert_eq!(mask[i * ms + j], e);
                }
            }
            let (sx, sy) = [(false, false), (true, false), (true, true)][rng.below(3)];
            let lm: Vec<u8> = (0..(4 * w * h)).map(|_| rng.below(65) as u8).collect();
            let lms = w << usize::from(sx);
            let mut d = inter.clone();
            interintra_blend(&mut d, ds, &intra, w, w, h, &lm, lms, sx, sy);
            for y in 0..h {
                for x in 0..w {
                    let mk = |yy: usize, xx: usize| i64::from(lm[yy * lms + xx]);
                    let m = if !sx && !sy {
                        mk(y, x)
                    } else if sx && !sy {
                        round2(mk(y, 2 * x) + mk(y, 2 * x + 1), 1)
                    } else {
                        round2(
                            mk(2 * y, 2 * x)
                                + mk(2 * y, 2 * x + 1)
                                + mk(2 * y + 1, 2 * x)
                                + mk(2 * y + 1, 2 * x + 1),
                            2,
                        )
                    };
                    let e = round2(
                        m * i64::from(intra[y * w + x]) + (64 - m) * i64::from(inter[y * ds + x]),
                        6,
                    );
                    assert_eq!(i64::from(d[y * ds + x]), e);
                }
            }
            // OBMC above and left
            let oh = (h / 2).clamp(1, 32);
            let ow = (w / 2).clamp(1, 32);
            let pred: Vec<u16> = (0..w * h).map(|_| rng.range(0, max) as u16).collect();
            let mut d = inter.clone();
            obmc_blend_above(&mut d, ds, &pred, w, w, oh);
            let mask = obmc_mask(oh);
            for i in 0..h {
                for j in 0..w {
                    let e = if i < oh {
                        let m = i64::from(mask[i]);
                        round2(
                            m * i64::from(inter[i * ds + j])
                                + (64 - m) * i64::from(pred[i * w + j]),
                            6,
                        )
                    } else {
                        i64::from(inter[i * ds + j])
                    };
                    assert_eq!(i64::from(d[i * ds + j]), e);
                }
            }
            let mut d = inter.clone();
            obmc_blend_left(&mut d, ds, &pred, w, ow, h);
            let mask = obmc_mask(ow);
            for i in 0..h {
                for j in 0..w {
                    let e = if j < ow {
                        let m = i64::from(mask[j]);
                        round2(
                            m * i64::from(inter[i * ds + j])
                                + (64 - m) * i64::from(pred[i * w + j]),
                            6,
                        )
                    } else {
                        i64::from(inter[i * ds + j])
                    };
                    assert_eq!(i64::from(d[i * ds + j]), e);
                }
            }
        }
    }

    #[test]
    fn wedge_masks_match() {
        let reference = ref_wedge_masks();
        for bsize in 0..22 {
            match &reference[bsize] {
                None => assert!(wedge_mask(bsize, false, 0).is_none()),
                Some(r) => {
                    for sign in 0..2 {
                        for wedge in 0..16 {
                            let got = wedge_mask(bsize, sign == 1, wedge);
                            assert_eq!(got, Some(&r[sign][wedge][..]), "{bsize} {sign} {wedge}");
                        }
                    }
                }
            }
        }
        assert!(wedge_mask(3, false, 16).is_none());
        assert!(wedge_mask(22, false, 0).is_none());
    }

    #[test]
    fn emu_edge_matches_clamp() {
        let mut rng = Rng(4242);
        for _ in 0..500 {
            let (pw, ph) = (1 + rng.below(30), 1 + rng.below(30));
            let plane = make_plane(&mut rng, pw, ph, 8);
            let pp = to_px::<u8>(&plane);
            let (bw, bh) = (1 + rng.below(50), 1 + rng.below(50));
            let x = rng.range(-60, 60);
            let y = rng.range(-60, 60);
            let ds = bw + 1;
            let mut d = vec![0u8; ds * bh];
            emu_edge(&mut d, ds, &pp, pw, pw, ph, x as isize, y as isize, bw, bh);
            for r in 0..bh {
                for c in 0..bw {
                    assert_eq!(
                        i64::from(d[r * ds + c]),
                        plane.at(y + r as i64, x + c as i64)
                    );
                }
            }
        }
    }

    #[test]
    fn setup_shear_matches() {
        let mut rng = Rng(31337);
        for _ in 0..20000 {
            let mut m = [0i32; 6];
            let spread = [1i64 << 10, 1 << 13, 1 << 16][rng.below(3)];
            m[2] = ((1i64 << 16) + rng.range(-spread, spread)) as i32;
            m[3] = rng.range(-spread, spread) as i32;
            m[4] = rng.range(-spread, spread) as i32;
            m[5] = ((1i64 << 16) + rng.range(-spread, spread)) as i32;
            if m[2] == 0 {
                continue;
            }
            assert_eq!(setup_shear(&m), ref_setup_shear(&m), "{m:?}");
        }
    }

    fn check_warp<P: Pixel>(rng: &mut Rng, bd: u32) {
        let (pw, ph) = (8 + rng.below(100), 8 + rng.below(100));
        let plane = make_plane(rng, pw, ph, bd);
        let pp = to_px::<P>(&plane);
        let (sx, sy) = [(0u32, 0u32), (1, 1), (1, 0)][rng.below(3)];
        let mut m;
        let shear = loop {
            m = [0i32; 6];
            m[0] = rng.range(-(40 << 16), 40 << 16) as i32;
            m[1] = rng.range(-(40 << 16), 40 << 16) as i32;
            m[2] = ((1 << 16) + rng.range(-6000, 6000)) as i32;
            m[3] = rng.range(-6000, 6000) as i32;
            m[4] = rng.range(-6000, 6000) as i32;
            m[5] = ((1 << 16) + rng.range(-6000, 6000)) as i32;
            if let Some(s) = setup_shear(&m) {
                break s;
            }
        };
        let x = rng.range(0, plane.w as i64 + 8);
        let y = rng.range(0, plane.h as i64 + 8);
        let max = (1i64 << bd) - 1;
        let pred = ref_warp(&plane, &m, x, y, sx, sy, bd, false);
        let mut d = vec![P::default(); 10 * 8];
        warp_8x8_put(
            &mut d,
            10,
            &pp,
            plane.w,
            plane.w,
            plane.h,
            &m,
            &shear,
            x as usize,
            y as usize,
            sx == 1,
            sy == 1,
            bd,
        );
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(
                    i64::from(d[r * 10 + c].to_i32()),
                    clip3(0, max, pred[r * 8 + c]),
                    "warp put {m:?}"
                );
            }
        }
        let pred = ref_warp(&plane, &m, x, y, sx, sy, bd, true);
        let mut t = vec![0i16; 16 * 8];
        warp_8x8_prep(
            &mut t,
            16,
            &pp,
            plane.w,
            plane.w,
            plane.h,
            &m,
            &shear,
            x as usize,
            y as usize,
            sx == 1,
            sy == 1,
            bd,
        );
        let bias = i64::from(prep_bias(bd));
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(
                    i64::from(t[r * 16 + c]) + bias,
                    pred[r * 8 + c],
                    "warp prep"
                );
            }
        }
    }

    #[test]
    fn warp_matches() {
        let mut rng = Rng(2718);
        for _ in 0..1500 {
            check_warp::<u8>(&mut rng, 8);
            check_warp::<u16>(&mut rng, 10);
            check_warp::<u16>(&mut rng, 12);
        }
    }
}
