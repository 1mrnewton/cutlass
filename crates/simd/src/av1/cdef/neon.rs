//! The CDEF kernels on NEON.

use core::arch::aarch64::{
    int8x16_t, uint8x8_t, uint8x16x4_t, vaddq_s8, vaddq_u8, vdup_n_u8, vqtbl4q_u8,
};
use core::arch::aarch64::{
    int16x8_t, int32x4_t, vaddq_s16, vaddq_s32, vdupq_n_s16, vdupq_n_s32, vextq_s16, vget_low_s16,
    vmovl_u8, vmull_high_s16, vmull_s16, vmulq_s32, vnegq_s16, vpaddq_s16, vpaddq_s32,
    vreinterpretq_s16_u16, vrev64q_s16, vshlq_u16, vsubq_s16,
};
use core::arch::aarch64::{
    uint8x16_t, vabdq_u8, vcltq_u8, vcombine_s8, vcombine_u8, vdupq_n_s8, vdupq_n_u8, veorq_u8,
    vget_low_s8, vmaxq_u8, vminq_u8, vmlal_high_s8, vmlal_s8, vmovn_s16, vmovn_u16, vqsubq_u8,
    vreinterpretq_s8_u8, vshlq_u8, vsqaddq_u8, vsubq_u8,
};
use core::arch::aarch64::{
    uint16x8_t, vabdq_u16, vandq_u16, vdupq_n_u16, vmaxq_s16, vminq_s16, vmulq_s16,
    vnegq_s16 as neg, vqsubq_u16, vreinterpretq_s16_u16 as signed, vreinterpretq_u16_s16,
    vshlq_u16 as shift_u16, vshrq_n_s16,
};

use super::scalar::best;
use super::{CENTRE, Filter, STRIDE, WINDOW};
use crate::neon::{halves, i32x4, i32x4_array, quarters, u8x8, u8x16, u16x8, u16x8_array};

/// Weights of the diagonal sums: a lane summing `n` samples gets
/// `Div_Table[n]`. Lane 15 sums nothing.
const W_DIAG: [i32; 16] = [
    840, 420, 280, 210, 168, 140, 120, 105, 120, 140, 168, 210, 280, 420, 840, 0,
];
/// Weights of the sums along the four in-between directions, 11 lanes.
const W_ALT: [i32; 16] = [
    420, 210, 140, 105, 105, 105, 105, 105, 140, 210, 420, 0, 0, 0, 0, 0,
];

#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(super) fn direction_u8(px: &[[u8; 8]; 8]) -> (u8, i32) {
    let mid = vdupq_n_s16(128);
    direction(
        px.each_ref()
            .map(|r| vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u8x8(r))), mid)),
    )
}

#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
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

/// `lane^2 * weight` over the sixteen lanes, summed into four.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn weighted_squares(v: Sums, w: &[i32; 16]) -> int32x4_t {
    let w = |i: usize| i32x4([w[i], w[i + 1], w[i + 2], w[i + 3]]);
    let a = vmulq_s32(vmull_s16(vget_low_s16(v.0), vget_low_s16(v.0)), w(0));
    let b = vmulq_s32(vmull_high_s16(v.0, v.0), w(4));
    let c = vmulq_s32(vmull_s16(vget_low_s16(v.1), vget_low_s16(v.1)), w(8));
    let d = vmulq_s32(vmull_high_s16(v.1, v.1), w(12));
    vaddq_s32(vaddq_s32(a, b), vaddq_s32(c, d))
}

/// `lane^2 * 105` over eight lanes, summed into four.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn flat_squares(v: int16x8_t) -> int32x4_t {
    let a = vmull_s16(vget_low_s16(v), vget_low_s16(v));
    let b = vmull_high_s16(v, v);
    vmulq_s32(vaddq_s32(a, b), vdupq_n_s32(105))
}

/// The direction process on rows of `sample - 128`.
///
/// A sum along a diagonal is a sum of whole vectors once row `i` is
/// shifted `i` lanes up: the eight lanes of a row land in sixteen.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
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
    macro_rules! row {
        ($i:tt) => {{
            let v = r[$i];
            let rev = vrev64q_s16(v);
            let rev = vextq_s16::<4>(rev, rev);
            cols = vaddq_s16(cols, v);
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
    // The row sums by pairwise additions: lane i is row i.
    let rows = vpaddq_s16(
        vpaddq_s16(vpaddq_s16(r[0], r[1]), vpaddq_s16(r[2], r[3])),
        vpaddq_s16(vpaddq_s16(r[4], r[5]), vpaddq_s16(r[6], r[7])),
    );
    let c = [
        weighted_squares(d0, &W_DIAG),
        weighted_squares(a0, &W_ALT),
        flat_squares(rows),
        weighted_squares(a1, &W_ALT),
        weighted_squares(d1, &W_DIAG),
        weighted_squares(a2, &W_ALT),
        flat_squares(cols),
        weighted_squares(a3, &W_ALT),
    ];
    // Each cost's four lanes summed, the eight costs in two vectors.
    let lo = vpaddq_s32(vpaddq_s32(c[0], c[1]), vpaddq_s32(c[2], c[3]));
    let hi = vpaddq_s32(vpaddq_s32(c[4], c[5]), vpaddq_s32(c[6], c[7]));
    let mut cost = [0i32; 8];
    if let Some((a, b)) = cost.split_first_chunk_mut::<4>() {
        *a = i32x4_array(lo);
        if let Some(b) = b.first_chunk_mut::<4>() {
            *b = i32x4_array(hi);
        }
    }
    best(&cost)
}

/// Eight samples of the window from `at`.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
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
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
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
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
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

#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(super) fn filter(win: &[u16; WINDOW], f: &Filter, rows: usize) -> [[u16; 8]; 8] {
    match (f.pri != 0, f.sec != 0) {
        (true, true) => filter_with::<true, true, true>(win, f, rows),
        (true, false) => filter_with::<true, false, false>(win, f, rows),
        _ => filter_with::<false, true, false>(win, f, rows),
    }
}

/// Eight samples of the window from `at`, which the caller keeps inside
/// it (zeros otherwise).
/// A padded source the byte filter reads: of 16-bit samples, or of bytes.
#[derive(Clone, Copy)]
pub(super) enum WinRef<'a> {
    Wide(&'a [u16; WINDOW]),
    Bytes(&'a [u8; WINDOW]),
}

/// The windows [`filter_inner`] takes. Each type always gives the same
/// [`WinRef`] variant, so the loads below are chosen at compile time.
pub(super) trait Window {
    fn win(&self) -> WinRef<'_>;
}

impl Window for [u16; WINDOW] {
    #[inline(always)]
    fn win(&self) -> WinRef<'_> {
        WinRef::Wide(self)
    }
}

impl Window for [u8; WINDOW] {
    #[inline(always)]
    fn win(&self) -> WinRef<'_> {
        WinRef::Bytes(self)
    }
}

/// The samples of the window from a group's first row on, `N` of them:
/// one bounds check for the group, after which every load is known to
/// be inside.
#[derive(Clone, Copy)]
enum Part<'a, const N: usize> {
    Wide(&'a [u16; N]),
    Bytes(&'a [u8; N]),
}

impl<'a, const N: usize> Part<'a, N> {
    /// The `N` samples of `win` from `base`, if it has them.
    #[inline(always)]
    fn of<W: Window>(win: &'a W, base: usize) -> Option<Self> {
        match win.win() {
            WinRef::Wide(w) => w.get(base..)?.first_chunk::<N>().map(Self::Wide),
            WinRef::Bytes(w) => w.get(base..)?.first_chunk::<N>().map(Self::Bytes),
        }
    }

    /// Eight samples from `at` as bytes; `at` is held to `N - 8`.
    #[inline]
    #[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
    #[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
    fn row8(self, at: usize) -> uint8x8_t {
        let at = at.min(N - 8);
        match self {
            Self::Wide(w) => match w[at..].first_chunk::<8>() {
                Some(a) => vmovn_u16(u16x8(a)),
                None => vdup_n_u8(0),
            },
            Self::Bytes(w) => match w[at..].first_chunk::<8>() {
                Some(a) => u8x8(a),
                None => vdup_n_u8(0),
            },
        }
    }
}

/// The samples an 8-wide group's taps read: its two rows with two rows
/// and two columns of border each side, up to the last tap's eight.
const ROWS_PART: usize = 2 * CENTRE + STRIDE + 8;
/// The samples a 4-wide group's table is built from: eight rows.
const TABLE_PART: usize = 7 * STRIDE + 8;

/// Where the samples of a group of sixteen are found for a tap.
#[derive(Clone, Copy)]
enum Source<'a> {
    /// An 8-wide block, two rows to a group: the eight samples from the
    /// tap's place in each of the two rows, straight from the window.
    Rows(Part<'a, ROWS_PART>),
    /// A 4-wide block, four rows to a group: the 8x8 samples around the
    /// group (two rows and two columns of border each side) as bytes in
    /// four vectors, and for each of the sixteen samples its place among
    /// them. A tap is then one table lookup at those places plus the
    /// tap's distance: its sixteen samples in one instruction.
    Table(uint8x16x4_t, uint8x16_t),
}

/// The places of the sixteen samples of a 4-wide group in its table:
/// row `2 + r`, column `2 + c` of eight.
const PLACES: [u8; 16] = [
    18, 19, 20, 21, 26, 27, 28, 29, 34, 35, 36, 37, 42, 43, 44, 45,
];

impl<'a> Source<'a> {
    /// The source for the group whose first row starts at `base` in the
    /// window, if the window holds all of it.
    #[inline]
    #[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
    #[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
    fn new<const WIDE: bool, W: Window>(win: &'a W, base: usize) -> Option<Self> {
        if WIDE {
            return Part::of(win, base).map(Self::Rows);
        }
        let part = Part::<TABLE_PART>::of(win, base)?;
        let pair = |r: usize| vcombine_u8(part.row8(r * STRIDE), part.row8((r + 1) * STRIDE));
        Some(Self::Table(
            uint8x16x4_t(pair(0), pair(2), pair(4), pair(6)),
            u8x16(&PLACES),
        ))
    }

    /// The group's samples moved by a tap at `at` from the window's
    /// first sample of a row ([`CENTRE`] is no move). `at` is at most
    /// `2 * CENTRE`.
    #[inline]
    #[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
    #[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
    fn at(&self, at: usize) -> uint8x16_t {
        match *self {
            Self::Rows(part) => vcombine_u8(part.row8(at), part.row8(at + STRIDE)),
            Self::Table(table, places) => {
                // Rows and columns of the tap from the window's first
                // sample: 2 and 2 is the sample itself. A place outside
                // the table reads 0.
                let (row, col) = (at / STRIDE, at % STRIDE);
                let step = (8 * row + col).wrapping_sub(18) as u8;
                vqtbl4q_u8(table, vaddq_u8(places, vdupq_n_u8(step)))
            }
        }
    }
}

/// What a group of sixteen samples adds up to, and the range of the
/// samples its taps read.
struct Group {
    x: uint8x16_t,
    sum: (int16x8_t, int16x8_t),
    min: uint8x16_t,
    max: uint8x16_t,
}

/// The strength, shift and weight of a tap, splat once for a block.
#[derive(Clone, Copy)]
struct TapKind {
    strength: uint8x16_t,
    down: int8x16_t,
    weight: int8x16_t,
}

impl TapKind {
    #[inline]
    #[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
    #[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
    fn new(strength: u16, shift: u32, weight: i8) -> Self {
        Self {
            // Two constrained differences must add up within a byte.
            strength: vdupq_n_u8(strength.min(63) as u8),
            // A shift of 8 or more leaves nothing of a byte.
            down: vdupq_n_s8(-(shift.min(8) as i8)),
            weight: vdupq_n_s8(weight),
        }
    }
}

/// Adds `weight * constrain(p - x)` for the tap at `at` and the one
/// opposite it, on bytes: the size of the difference is clamped to
/// `max(0, strength - (size >> shift))` and takes the difference's sign.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn tap_inner<const RANGE: bool>(src: &Source<'_>, at: usize, k: TapKind, g: &mut Group) {
    // The two taps' constrained differences add up within a signed byte
    // (a strength is at most 15), so one widening multiply takes both.
    let mut pair = vdupq_n_s8(0);
    for at in [at, 2 * CENTRE - at] {
        let p = src.at(at);
        let size = vabdq_u8(p, g.x);
        let bound = vqsubq_u8(k.strength, vshlq_u8(size, k.down));
        let size = vminq_u8(size, bound);
        // Minus the size where the tap is below the sample.
        let below = vcltq_u8(p, g.x);
        let v = vreinterpretq_s8_u8(vsubq_u8(veorq_u8(size, below), below));
        pair = vaddq_s8(pair, v);
        if RANGE {
            g.min = vminq_u8(g.min, p);
            g.max = vmaxq_u8(g.max, p);
        }
    }
    g.sum.0 = vmlal_s8(g.sum.0, vget_low_s8(pair), vget_low_s8(k.weight));
    g.sum.1 = vmlal_high_s8(g.sum.1, pair, k.weight);
}

/// Filters the block into `dst`, rows `dst_stride` apart: four rows of a
/// 4-wide block or two of an 8-wide one to a vector.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn inner_with<const WIDE: bool, const PRI: bool, const SEC: bool, const BOTH: bool, W: Window>(
    dst: &mut [u8],
    dst_stride: usize,
    win: &W,
    f: &Filter,
    rows: usize,
) {
    let per = if WIDE { 2 } else { 4 };
    let pri = [
        TapKind::new(f.pri, f.pri_shift, f.pri_taps[0] as i8),
        TapKind::new(f.pri, f.pri_shift, f.pri_taps[1] as i8),
    ];
    let sec = [
        TapKind::new(f.sec, f.sec_shift, 2),
        TapKind::new(f.sec, f.sec_shift, 1),
    ];
    // Every tap within reach, so that no load needs checking.
    let pri_at = f.pri_at.map(|at| at.min(2 * CENTRE));
    let sec_at = f.sec_at.map(|at| at.min(2 * CENTRE));
    for g in 0..rows.div_ceil(per) {
        let Some(src) = Source::new::<WIDE, W>(win, g * per * STRIDE) else {
            break;
        };
        let x = src.at(CENTRE);
        let zero = vdupq_n_s16(0);
        let mut grp = Group {
            x,
            sum: (zero, zero),
            min: x,
            max: x,
        };
        for (k, &first) in pri.iter().enumerate() {
            if PRI {
                tap_inner::<BOTH>(&src, pri_at[k], first, &mut grp);
            }
            if SEC {
                tap_inner::<BOTH>(&src, sec_at[k], sec[0], &mut grp);
                tap_inner::<BOTH>(&src, sec_at[2 + k], sec[1], &mut grp);
            }
        }
        // (8 + sum - (sum < 0)) >> 4, which fits a signed byte.
        let step = |s: int16x8_t| {
            vmovn_s16(vshrq_n_s16::<4>(vaddq_s16(
                vaddq_s16(s, vdupq_n_s16(8)),
                vshrq_n_s16::<15>(s),
            )))
        };
        let mut y = vsqaddq_u8(x, vcombine_s8(step(grp.sum.0), step(grp.sum.1)));
        if BOTH {
            y = vminq_u8(vmaxq_u8(y, grp.min), grp.max);
        }
        // Whole rows at a time: 8 bytes, or 4.
        let row = g * per;
        if WIDE {
            let (lo, hi) = halves(y);
            for (r, b) in [(row, lo), (row + 1, hi)] {
                if r < rows
                    && let Some(d) = dst.get_mut(r * dst_stride..r * dst_stride + 8)
                {
                    d.copy_from_slice(&b);
                }
            }
        } else {
            for (r, b) in quarters(y).iter().enumerate() {
                let r = row + r;
                if r < rows
                    && let Some(d) = dst.get_mut(r * dst_stride..r * dst_stride + 4)
                {
                    d.copy_from_slice(&b.to_le_bytes());
                }
            }
        }
    }
}

#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(super) fn filter_inner<W: Window>(
    dst: &mut [u8],
    dst_stride: usize,
    win: &W,
    f: &Filter,
    wide: bool,
    rows: usize,
) {
    match (wide, f.pri != 0, f.sec != 0) {
        (true, true, true) => {
            inner_with::<true, true, true, true, W>(dst, dst_stride, win, f, rows)
        }
        (true, true, false) => {
            inner_with::<true, true, false, false, W>(dst, dst_stride, win, f, rows);
        }
        (true, false, _) => {
            inner_with::<true, false, true, false, W>(dst, dst_stride, win, f, rows)
        }
        (false, true, true) => {
            inner_with::<false, true, true, true, W>(dst, dst_stride, win, f, rows);
        }
        (false, true, false) => {
            inner_with::<false, true, false, false, W>(dst, dst_stride, win, f, rows);
        }
        (false, false, _) => {
            inner_with::<false, false, true, false, W>(dst, dst_stride, win, f, rows);
        }
    }
}
