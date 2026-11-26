//! The CDEF kernels on AVX2.
//!
//! The direction search adds each row of an 8x8 block into sixteen lanes
//! at its own offset, as the NEON kernel does, on 128-bit vectors.
//!
//! The filter on 16-bit samples takes two rows of eight to a vector. The
//! filter on bytes, for blocks whose taps all lie in the picture, takes
//! thirty-two samples to a vector: four rows of an 8-wide block, or the
//! eight rows of a 4-wide one. There the two taps of a pair are
//! constrained as bytes and added, and `pmaddubsw` multiplies two such
//! sums by their two weights at once.

use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_add_epi32, _mm_cvtepu8_epi16, _mm_cvtsi32_si128,
    _mm_cvtsi128_si64, _mm_extract_epi64, _mm_hadd_epi16, _mm_hadd_epi32, _mm_madd_epi16,
    _mm_mullo_epi32, _mm_set1_epi16, _mm_set1_epi32, _mm_setr_epi8, _mm_setr_epi16,
    _mm_setzero_si128, _mm_shuffle_epi8, _mm_slli_si128, _mm_srl_epi16, _mm_srli_si128,
    _mm_sub_epi16, _mm_unpacklo_epi64, _mm256_add_epi8, _mm256_add_epi16, _mm256_add_epi32,
    _mm256_and_si256, _mm256_castsi256_si128, _mm256_cmpeq_epi8, _mm256_cvtepi16_epi32,
    _mm256_extract_epi64, _mm256_extracti128_si256, _mm256_maddubs_epi16, _mm256_max_epi16,
    _mm256_max_epu8, _mm256_max_epu16, _mm256_min_epi16, _mm256_min_epu8, _mm256_min_epu16,
    _mm256_mullo_epi16, _mm256_mullo_epi32, _mm256_packus_epi16, _mm256_set_epi64x,
    _mm256_set_m128i, _mm256_set1_epi8, _mm256_set1_epi16, _mm256_setr_epi32, _mm256_setzero_si256,
    _mm256_srai_epi16, _mm256_srl_epi16, _mm256_sub_epi8, _mm256_sub_epi16, _mm256_subs_epu8,
    _mm256_subs_epu16, _mm256_unpackhi_epi8, _mm256_unpacklo_epi8, _mm256_xor_si256,
};

use super::scalar::best;
use super::{CENTRE, Filter, STRIDE, WINDOW};
use crate::av1::cpu::x86::ld64;

/// Weights of the diagonal sums: a lane summing `n` samples gets
/// `Div_Table[n]`. Lane 15 sums nothing.
const W_DIAG: [i32; 16] = [
    840, 420, 280, 210, 168, 140, 120, 105, 120, 140, 168, 210, 280, 420, 840, 0,
];
/// Weights of the sums along the four in-between directions, 11 lanes.
const W_ALT: [i32; 16] = [
    420, 210, 140, 105, 105, 105, 105, 105, 140, 210, 420, 0, 0, 0, 0, 0,
];

/// Eight 16-bit values as one vector.
#[inline]
#[target_feature(enable = "avx2")]
fn words(a: &[u16; 8]) -> __m128i {
    _mm_setr_epi16(
        a[0] as i16,
        a[1] as i16,
        a[2] as i16,
        a[3] as i16,
        a[4] as i16,
        a[5] as i16,
        a[6] as i16,
        a[7] as i16,
    )
}

/// The eight 16-bit values of a vector.
#[inline]
#[target_feature(enable = "avx2")]
fn unwords(v: __m128i) -> [u16; 8] {
    let (lo, hi) = (
        _mm_cvtsi128_si64(v) as u64,
        _mm_extract_epi64::<1>(v) as u64,
    );
    [
        lo as u16,
        (lo >> 16) as u16,
        (lo >> 32) as u16,
        (lo >> 48) as u16,
        hi as u16,
        (hi >> 16) as u16,
        (hi >> 32) as u16,
        (hi >> 48) as u16,
    ]
}

#[inline]
#[target_feature(enable = "avx2")]
pub(super) fn direction_u8(px: &[[u8; 8]; 8]) -> (u8, i32) {
    let mid = _mm_set1_epi16(128);
    direction(
        px.each_ref()
            .map(|r| _mm_sub_epi16(_mm_cvtepu8_epi16(ld64(r)), mid)),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
pub(super) fn direction_u16(px: &[[u16; 8]; 8], bd: u32) -> (u8, i32) {
    let mid = _mm_set1_epi16(128);
    let down = _mm_cvtsi32_si128(bd as i32 - 8);
    direction(
        px.each_ref()
            .map(|r| _mm_sub_epi16(_mm_srl_epi16(words(r), down), mid)),
    )
}

/// Sixteen lanes of partial sums.
type Sums = (__m128i, __m128i);

/// `lane^2 * weight` over the sixteen lanes, summed into four.
#[inline]
#[target_feature(enable = "avx2")]
fn weighted_squares(v: Sums, w: &[i32; 16]) -> __m128i {
    let half = |x: __m128i, i: usize| {
        let e = _mm256_cvtepi16_epi32(x);
        let w = _mm256_setr_epi32(
            w[i],
            w[i + 1],
            w[i + 2],
            w[i + 3],
            w[i + 4],
            w[i + 5],
            w[i + 6],
            w[i + 7],
        );
        _mm256_mullo_epi32(_mm256_mullo_epi32(e, e), w)
    };
    let s = _mm256_add_epi32(half(v.0, 0), half(v.1, 8));
    _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256::<1>(s))
}

/// `lane^2 * 105` over eight lanes, summed into four.
#[inline]
#[target_feature(enable = "avx2")]
fn flat_squares(v: __m128i) -> __m128i {
    _mm_mullo_epi32(_mm_madd_epi16(v, v), _mm_set1_epi32(105))
}

/// The direction process on rows of `sample - 128`.
///
/// A sum along a diagonal is a sum of whole vectors once row `i` is
/// shifted `i` lanes up: the eight lanes of a row land in sixteen.
#[inline]
#[target_feature(enable = "avx2")]
fn direction(r: [__m128i; 8]) -> (u8, i32) {
    let z = _mm_setzero_si128();
    // Adds `v` to lanes `i..i + 8` of the sums.
    macro_rules! add_at {
        ($acc:ident, $v:expr, 0) => {
            $acc.0 = _mm_add_epi16($acc.0, $v)
        };
        ($acc:ident, $v:expr, $i:literal) => {{
            let v = $v;
            $acc.0 = _mm_add_epi16($acc.0, _mm_slli_si128::<{ 2 * $i }>(v));
            $acc.1 = _mm_add_epi16($acc.1, _mm_srli_si128::<{ 16 - 2 * $i }>(v));
        }};
    }
    // partial[0], [4]: the diagonals. partial[1], [3], [5], [7]: between.
    let (mut d0, mut d1, mut a0, mut a1, mut a2, mut a3): (Sums, Sums, Sums, Sums, Sums, Sums) =
        ((z, z), (z, z), (z, z), (z, z), (z, z), (z, z));
    // partial[6]: the columns. partial[2]: the rows.
    let mut cols = z;
    // The eight lanes of a row in reverse order.
    let back = _mm_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    macro_rules! row {
        ($i:tt) => {{
            let v = r[$i];
            let rev = _mm_shuffle_epi8(v, back);
            cols = _mm_add_epi16(cols, v);
            add_at!(d0, v, $i);
            add_at!(d1, rev, $i);
            // Sums of column pairs in lanes 0..4.
            add_at!(a0, _mm_hadd_epi16(v, z), $i);
            add_at!(a1, _mm_hadd_epi16(rev, z), $i);
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
    let p0 = _mm_add_epi16(r[0], r[1]);
    let p1 = _mm_add_epi16(r[2], r[3]);
    let p2 = _mm_add_epi16(r[4], r[5]);
    let p3 = _mm_add_epi16(r[6], r[7]);
    add_at!(a2, p0, 3);
    add_at!(a2, p1, 2);
    add_at!(a2, p2, 1);
    add_at!(a2, p3, 0);
    add_at!(a3, p0, 0);
    add_at!(a3, p1, 1);
    add_at!(a3, p2, 2);
    add_at!(a3, p3, 3);
    // The row sums by pairwise additions: lane i is row i.
    let rows = _mm_hadd_epi16(
        _mm_hadd_epi16(_mm_hadd_epi16(r[0], r[1]), _mm_hadd_epi16(r[2], r[3])),
        _mm_hadd_epi16(_mm_hadd_epi16(r[4], r[5]), _mm_hadd_epi16(r[6], r[7])),
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
    let lo = _mm_hadd_epi32(_mm_hadd_epi32(c[0], c[1]), _mm_hadd_epi32(c[2], c[3]));
    let hi = _mm_hadd_epi32(_mm_hadd_epi32(c[4], c[5]), _mm_hadd_epi32(c[6], c[7]));
    let pair = |q: i64| [q as i32, (q >> 32) as i32];
    let [c0, c1] = pair(_mm_cvtsi128_si64(lo));
    let [c2, c3] = pair(_mm_extract_epi64::<1>(lo));
    let [c4, c5] = pair(_mm_cvtsi128_si64(hi));
    let [c6, c7] = pair(_mm_extract_epi64::<1>(hi));
    best(&[c0, c1, c2, c3, c4, c5, c6, c7])
}

/// Eight samples of the window from `at` (zeros past its end).
#[inline]
#[target_feature(enable = "avx2")]
fn window(win: &[u16; WINDOW], at: usize) -> __m128i {
    match win.get(at..).and_then(|s| s.first_chunk::<8>()) {
        Some(a) => words(a),
        None => _mm_setzero_si128(),
    }
}

/// Eight samples from `at` and the eight a row below, one row to a lane.
#[inline]
#[target_feature(enable = "avx2")]
fn rows2(win: &[u16; WINDOW], at: usize) -> __m256i {
    _mm256_set_m128i(window(win, at + STRIDE), window(win, at))
}

/// What two rows' taps add up to, and the range of the samples they read.
struct Rows {
    x: __m256i,
    sum: __m256i,
    min: __m256i,
    max: __m256i,
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
#[target_feature(enable = "avx2")]
fn tap<const RANGE: bool>(
    win: &[u16; WINDOW],
    base: usize,
    at: usize,
    weight: i16,
    strength: u16,
    shift: u32,
    r: &mut Rows,
) {
    let at = at.min(2 * CENTRE);
    let down = _mm_cvtsi32_si128(shift.min(15) as i32);
    let strength = _mm256_set1_epi16(strength as i16);
    let zero = _mm256_setzero_si256();
    for at in [at, 2 * CENTRE - at] {
        let p = rows2(win, base + at);
        let d = _mm256_sub_epi16(p, r.x);
        let size = _mm256_sub_epi16(_mm256_max_epu16(p, r.x), _mm256_min_epu16(p, r.x));
        let bound = _mm256_subs_epu16(strength, _mm256_srl_epi16(size, down));
        let v = _mm256_min_epi16(_mm256_max_epi16(d, _mm256_sub_epi16(zero, bound)), bound);
        r.sum = _mm256_add_epi16(r.sum, _mm256_mullo_epi16(v, _mm256_set1_epi16(weight)));
        if RANGE {
            r.min = _mm256_min_epi16(r.min, p);
            r.max = _mm256_max_epi16(r.max, _mm256_and_si256(p, _mm256_set1_epi16(0xfff)));
        }
    }
}

/// `(8 + sum - (sum < 0)) >> 4` on sixteen lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn step(sum: __m256i) -> __m256i {
    let round = _mm256_add_epi16(
        _mm256_add_epi16(sum, _mm256_set1_epi16(8)),
        _mm256_srai_epi16::<15>(sum),
    );
    _mm256_srai_epi16::<4>(round)
}

/// The filter with the primary taps (`PRI`), the secondary taps (`SEC`),
/// or both. With one of them the result cannot leave the range of the
/// samples read (the weights add up to 12 of 16), so only both clamp.
#[inline]
#[target_feature(enable = "avx2")]
fn filter_with<const PRI: bool, const SEC: bool, const BOTH: bool>(
    win: &[u16; WINDOW],
    f: &Filter,
    rows: usize,
) -> [[u16; 8]; 8] {
    let mut out = [[0u16; 8]; 8];
    for (g, o) in out.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        if 2 * g >= rows {
            break;
        }
        let base = 2 * g * STRIDE;
        let x = rows2(win, base + CENTRE);
        let mut r = Rows {
            x,
            sum: _mm256_setzero_si256(),
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
        let mut y = _mm256_add_epi16(x, step(r.sum));
        if BOTH {
            y = _mm256_min_epi16(_mm256_max_epi16(y, r.min), r.max);
        }
        o[0] = unwords(_mm256_castsi256_si128(y));
        if 2 * g + 1 < rows {
            o[1] = unwords(_mm256_extracti128_si256::<1>(y));
        }
    }
    out
}

#[target_feature(enable = "avx2")]
pub(super) fn filter(win: &[u16; WINDOW], f: &Filter, rows: usize) -> [[u16; 8]; 8] {
    match (f.pri != 0, f.sec != 0) {
        (true, true) => filter_with::<true, true, true>(win, f, rows),
        (true, false) => filter_with::<true, false, false>(win, f, rows),
        _ => filter_with::<false, true, false>(win, f, rows),
    }
}

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

/// The thirty-two samples of a group as bytes: eight from each of four
/// rows (`WIDE`), or four from each of eight, in row order, the first at
/// `base + at` in the window. Of the eight rows only the first four are
/// read when no more than four are `live`. `base` is held to the last
/// group's (`4 * STRIDE`, or 0) and `at` to `2 * CENTRE`, which keeps
/// every load inside the window.
#[inline]
#[target_feature(enable = "avx2")]
fn group<const WIDE: bool>(win: WinRef<'_>, base: usize, at: usize, live: usize) -> __m256i {
    let at = base.min(if WIDE { 4 * STRIDE } else { 0 }) + at.min(2 * CENTRE);
    match win {
        WinRef::Bytes(w) => {
            if WIDE {
                let q = |r: usize| {
                    w.get(at + r * STRIDE..)
                        .and_then(|s| s.first_chunk::<8>())
                        .map_or(0, |a| i64::from_le_bytes(*a))
                };
                _mm256_set_epi64x(q(3), q(2), q(1), q(0))
            } else {
                let d = |r: usize| {
                    w.get(at + r * STRIDE..)
                        .and_then(|s| s.first_chunk::<4>())
                        .map_or(0, |a| i32::from_le_bytes(*a))
                };
                if live <= 4 {
                    _mm256_setr_epi32(d(0), d(1), d(2), d(3), 0, 0, 0, 0)
                } else {
                    _mm256_setr_epi32(d(0), d(1), d(2), d(3), d(4), d(5), d(6), d(7))
                }
            }
        }
        WinRef::Wide(w) => {
            let row = |r: usize| window(w, at + r * STRIDE);
            // The low byte of each sample; the pack then keeps it whole.
            let low = _mm256_set1_epi16(0xff);
            let (a, b) = if WIDE {
                (
                    _mm256_set_m128i(row(2), row(0)),
                    _mm256_set_m128i(row(3), row(1)),
                )
            } else {
                let two = |r: usize| _mm_unpacklo_epi64(row(r), row(r + 1));
                (
                    _mm256_set_m128i(two(4), two(0)),
                    _mm256_set_m128i(two(6), two(2)),
                )
            };
            _mm256_packus_epi16(_mm256_and_si256(a, low), _mm256_and_si256(b, low))
        }
    }
}

/// The strength and shift of a kind of tap, splat once for a block.
#[derive(Clone, Copy)]
struct TapKind {
    strength: __m256i,
    down: __m128i,
    /// What a shift of 16-bit lanes must clear to be a shift of bytes.
    keep: __m256i,
}

impl TapKind {
    #[inline]
    #[target_feature(enable = "avx2")]
    fn new(strength: u16, shift: u32) -> Self {
        // A shift of 8 or more leaves nothing of a byte.
        let shift = shift.min(8);
        Self {
            // Two constrained differences must add up within a byte.
            strength: _mm256_set1_epi8(strength.min(63) as i8),
            down: _mm_cvtsi32_si128(shift as i32),
            keep: _mm256_set1_epi8((0xffu32 >> shift) as u8 as i8),
        }
    }
}

/// Filters the block into `dst`, rows `dst_stride` apart: four rows of an
/// 8-wide block or eight of a 4-wide one to a vector.
#[inline]
#[target_feature(enable = "avx2")]
fn inner_with<const WIDE: bool, const PRI: bool, const SEC: bool, const BOTH: bool, W: Window>(
    dst: &mut [u8],
    dst_stride: usize,
    win: &W,
    f: &Filter,
    rows: usize,
) {
    let per = if WIDE { 4 } else { 8 };
    let pri = TapKind::new(f.pri, f.pri_shift);
    let sec = TapKind::new(f.sec, f.sec_shift);
    // The weights of two taps, side by side as their sums are.
    let two = |a: i16, b: i16| _mm256_set1_epi16(i16::from_le_bytes([a as u8, b as u8]));
    let pri_w = two(f.pri_taps[0], f.pri_taps[1]);
    let sec_w = two(2, 1);
    // Where the taps are, held within reach: the two primary places,
    // then the secondary ones, a weight of 2 beside a weight of 1.
    let places = [
        f.pri_at[0],
        f.pri_at[1],
        f.sec_at[0],
        f.sec_at[2],
        f.sec_at[1],
        f.sec_at[3],
    ]
    .map(|at| at.min(2 * CENTRE));
    let win = win.win();
    let zero = _mm256_setzero_si256();
    // A 4-wide block of four rows fills half a vector.
    let live = if WIDE { 4 } else { rows };
    for n in 0..rows.div_ceil(per) {
        let base = n * per * STRIDE;
        let x = group::<WIDE>(win, base, CENTRE, live);
        // The range of the samples the taps read.
        let (mut min, mut max) = (x, x);
        // For each place, `constrain(p - x)` for the tap there plus the
        // same for the one opposite it, on bytes: the size of the
        // difference is clamped to `max(0, strength - (size >> shift))`
        // and takes the difference's sign. The two add up within a
        // signed byte.
        let mut pairs = [zero; 6];
        for i in (if PRI { 0 } else { 2 })..(if SEC { 6 } else { 2 }) {
            let k = if i < 2 { &pri } else { &sec };
            let mut both = zero;
            for at in [places[i], 2 * CENTRE - places[i]] {
                let p = group::<WIDE>(win, base, at, live);
                let top = _mm256_max_epu8(p, x);
                let size = _mm256_sub_epi8(top, _mm256_min_epu8(p, x));
                let far = _mm256_and_si256(_mm256_srl_epi16(size, k.down), k.keep);
                let size = _mm256_min_epu8(size, _mm256_subs_epu8(k.strength, far));
                // Minus the size where the tap is not above the sample.
                let below = _mm256_cmpeq_epi8(top, x);
                let v = _mm256_sub_epi8(_mm256_xor_si256(size, below), below);
                both = _mm256_add_epi8(both, v);
                if BOTH {
                    min = _mm256_min_epu8(min, p);
                    max = _mm256_max_epu8(max, p);
                }
            }
            pairs[i] = both;
        }
        // The sums of the low and of the high eight bytes of each lane:
        // two places' pairs side by side, times their two weights.
        let (mut lo, mut hi) = (zero, zero);
        for (i, w) in [pri_w, sec_w, sec_w].into_iter().enumerate() {
            if if i == 0 { PRI } else { SEC } {
                let (a, b) = (pairs[2 * i], pairs[2 * i + 1]);
                lo = _mm256_add_epi16(lo, _mm256_maddubs_epi16(w, _mm256_unpacklo_epi8(a, b)));
                hi = _mm256_add_epi16(hi, _mm256_maddubs_epi16(w, _mm256_unpackhi_epi8(a, b)));
            }
        }
        let mut y = _mm256_packus_epi16(
            _mm256_add_epi16(_mm256_unpacklo_epi8(x, zero), step(lo)),
            _mm256_add_epi16(_mm256_unpackhi_epi8(x, zero), step(hi)),
        );
        if BOTH {
            y = _mm256_min_epu8(_mm256_max_epu8(y, min), max);
        }
        let q = [
            _mm256_extract_epi64::<0>(y),
            _mm256_extract_epi64::<1>(y),
            _mm256_extract_epi64::<2>(y),
            _mm256_extract_epi64::<3>(y),
        ];
        // Whole rows at a time: 8 bytes, or 4.
        let row = n * per;
        if WIDE {
            for (r, b) in q.iter().enumerate() {
                let r = row + r;
                if r < rows
                    && let Some(d) = dst
                        .get_mut(r * dst_stride..)
                        .and_then(|d| d.first_chunk_mut::<8>())
                {
                    *d = b.to_le_bytes();
                }
            }
        } else {
            for (r, b) in q
                .iter()
                .flat_map(|q| [*q as u32, (*q >> 32) as u32])
                .enumerate()
            {
                let r = row + r;
                if r < rows
                    && let Some(d) = dst
                        .get_mut(r * dst_stride..)
                        .and_then(|d| d.first_chunk_mut::<4>())
                {
                    *d = b.to_le_bytes();
                }
            }
        }
    }
}

#[target_feature(enable = "avx2")]
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
