//! The block transform on AVX2: an entry of `T` is the sixteen 16-bit
//! lanes of a vector, one lane for each of sixteen rows or sixteen
//! columns.
//!
//! The row transforms take sixteen rows at a time: rows `r` and `r + 8`
//! share a vector, one in each 128-bit half, so that the exchange of rows
//! and columns stays inside the halves. The column transforms take
//! sixteen columns at a time, in the order of the samples.

use core::arch::x86_64::{
    __m128i, __m256i, _mm_adds_epu8, _mm_and_si128, _mm_packus_epi32, _mm_set_epi64x,
    _mm_set1_epi8, _mm_set1_epi32, _mm_setr_epi32, _mm_setzero_si128, _mm_subs_epu8,
    _mm256_add_epi16, _mm256_adds_epu8, _mm256_and_si256, _mm256_castsi256_si128,
    _mm256_cvtepu8_epi16, _mm256_extracti128_si256, _mm256_packus_epi16, _mm256_packus_epi32,
    _mm256_permute4x64_epi64, _mm256_set_m128i, _mm256_set1_epi8, _mm256_set1_epi16,
    _mm256_set1_epi32, _mm256_setr_epi16, _mm256_setr_epi32, _mm256_setzero_si256,
    _mm256_shuffle_epi8, _mm256_shufflelo_epi16, _mm256_sra_epi16, _mm256_subs_epu8,
    _mm256_unpackhi_epi16, _mm256_unpackhi_epi32, _mm256_unpackhi_epi64, _mm256_unpacklo_epi16,
    _mm256_unpacklo_epi32, _mm256_unpacklo_epi64, _mm256_zextsi128_si256,
};

use super::steps::{Prog, Step, kind, plan, table};
use super::{Block, Kind};
use crate::av1::cpu::x86::{ld64, ld128, ld256, st64, st128, st128_i16, st256};

type V = __m256i;

std::thread_local! {
    /// The 64 vectors [`sized`] works in, kept between blocks rather than
    /// cleared for each: it loads or clears every entry a transform
    /// reads.
    static SCRATCH: core::cell::RefCell<Option<[V; 64]>> =
        const { core::cell::RefCell::new(None) };
}

/// The arithmetic of a step on the sixteen values of an entry.
mod ops {
    use core::arch::x86_64::{
        __m128i, __m256i, _mm256_add_epi16, _mm256_add_epi32, _mm256_adds_epi16, _mm256_and_si256,
        _mm256_cmpeq_epi32, _mm256_madd_epi16, _mm256_mulhrs_epi16, _mm256_mullo_epi32,
        _mm256_packs_epi32, _mm256_packus_epi32, _mm256_set1_epi16, _mm256_set1_epi32,
        _mm256_setzero_si256, _mm256_slli_epi32, _mm256_sra_epi32, _mm256_srai_epi32,
        _mm256_sub_epi16, _mm256_subs_epi16, _mm256_unpackhi_epi16, _mm256_unpacklo_epi16,
    };

    pub(super) type V = __m256i;

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) fn zero() -> V {
        _mm256_setzero_si256()
    }

    /// Two 16-bit multipliers as the pair `pmaddwd` takes: `a` for the
    /// first value of each pair and `b` for the second.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn pair(a: i16, b: i16) -> __m256i {
        _mm256_set1_epi32(i32::from(a as u16) | (i32::from(b) << 16))
    }

    /// The low 16 bits of each of two vectors of 32-bit values, the
    /// first's in the low half of each 128 bits.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn low16(lo: __m256i, hi: __m256i) -> V {
        let m = _mm256_set1_epi32(0xffff);
        _mm256_packus_epi32(_mm256_and_si256(lo, m), _mm256_and_si256(hi, m))
    }

    /// `Round2(x, 12)` of two vectors of 32-bit values, their low 16
    /// bits.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn round12(lo: __m256i, hi: __m256i) -> V {
        let r = _mm256_set1_epi32(2048);
        low16(
            _mm256_srai_epi32::<12>(_mm256_add_epi32(lo, r)),
            _mm256_srai_epi32::<12>(_mm256_add_epi32(hi, r)),
        )
    }

    /// `x cos - y sin` and `x sin + y cos`, rounded at 12 bits: in 32
    /// bits, of which the low 16 are kept.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) fn rot(x: V, y: V, cos: i16, sin: i16) -> (V, V) {
        let (lo, hi) = (_mm256_unpacklo_epi16(x, y), _mm256_unpackhi_epi16(x, y));
        let (k, l) = (pair(cos, sin.wrapping_neg()), pair(sin, cos));
        (
            round12(_mm256_madd_epi16(lo, k), _mm256_madd_epi16(hi, k)),
            round12(_mm256_madd_epi16(lo, l), _mm256_madd_epi16(hi, l)),
        )
    }

    /// `x c`, rounded at 12 bits, for `c` within 4095 of zero: the
    /// rounded high half of the product by `8 c`, which rounds the same
    /// way.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) fn scale(x: V, c: i16) -> V {
        _mm256_mulhrs_epi16(x, _mm256_set1_epi16(c.wrapping_mul(8)))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) fn add(x: V, y: V) -> V {
        _mm256_adds_epi16(x, y)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) fn sub(x: V, y: V) -> V {
        _mm256_subs_epi16(x, y)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) fn neg(x: V) -> V {
        _mm256_subs_epi16(_mm256_setzero_si256(), x)
    }

    /// The ADST of four values (7.13.2.6). The sums are exact in 32
    /// bits; an output of 32768 becomes 32767, which the rounding after
    /// it cannot tell apart.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) fn adst4(t: [V; 4]) -> [V; 4] {
        let [t0, t1, t2, t3] = t;
        // `t0 - t2 + t3` in 16 bits, as the specification keeps it.
        let b7 = _mm256_add_epi16(_mm256_sub_epi16(t0, t2), t3);
        let top = _mm256_set1_epi32(32768);
        let round = _mm256_set1_epi32(2048);
        // The four outputs of the low or the high four lanes of each
        // half, from `(t0, t2)`, `(t3, t1)` and `(b7, b7)` in pairs.
        let half = |p02: __m256i, p31: __m256i, p77: __m256i| {
            let two = |a: (i16, i16), b: (i16, i16)| {
                _mm256_add_epi32(
                    _mm256_madd_epi16(p02, pair(a.0, a.1)),
                    _mm256_madd_epi16(p31, pair(b.0, b.1)),
                )
            };
            [
                two((1321, 3803), (2482, 3344)),
                two((2482, -1321), (-3803, 3344)),
                _mm256_madd_epi16(p77, pair(3344, 0)),
                two((3803, 2482), (-1321, -3344)),
            ]
            .map(|x| {
                let r = _mm256_srai_epi32::<12>(_mm256_add_epi32(x, round));
                // All ones, which is minus one, where it is 32768.
                _mm256_add_epi32(r, _mm256_cmpeq_epi32(r, top))
            })
        };
        let lo = half(
            _mm256_unpacklo_epi16(t0, t2),
            _mm256_unpacklo_epi16(t3, t1),
            _mm256_unpacklo_epi16(b7, b7),
        );
        let hi = half(
            _mm256_unpackhi_epi16(t0, t2),
            _mm256_unpackhi_epi16(t3, t1),
            _mm256_unpackhi_epi16(b7, b7),
        );
        [
            low16(lo[0], hi[0]),
            low16(lo[1], hi[1]),
            low16(lo[2], hi[2]),
            low16(lo[3], hi[3]),
        ]
    }

    /// The values of `x` in 32 bits: the low and the high four of each
    /// half.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn widen(x: V) -> (__m256i, __m256i) {
        (
            _mm256_srai_epi32::<16>(_mm256_unpacklo_epi16(x, x)),
            _mm256_srai_epi32::<16>(_mm256_unpackhi_epi16(x, x)),
        )
    }

    /// The identity transform of `n` values on eight of them
    /// (7.13.2.15): a scaling by `sqrt(2)`, 2, `2 sqrt(2)` or 4, in 32
    /// bits.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn identity(x: __m256i, n: usize) -> __m256i {
        let round12 = |x: __m256i, c: i32| {
            _mm256_srai_epi32::<12>(_mm256_add_epi32(
                _mm256_mullo_epi32(x, _mm256_set1_epi32(c)),
                _mm256_set1_epi32(2048),
            ))
        };
        match n {
            4 => _mm256_add_epi32(x, round12(x, 1697)),
            8 => _mm256_slli_epi32::<1>(x),
            16 => _mm256_add_epi32(_mm256_slli_epi32::<1>(x), round12(x, 3394)),
            _ => _mm256_slli_epi32::<2>(x),
        }
    }

    /// The outputs of a row identity transform of `n` values, after the
    /// rounding shift by `shift` (`half` is what it adds first), clamped
    /// to 16 bits.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) fn identity_row(x: V, n: usize, half: __m256i, shift: __m128i) -> V {
        let (lo, hi) = widen(x);
        let lo = _mm256_sra_epi32(_mm256_add_epi32(identity(lo, n), half), shift);
        let hi = _mm256_sra_epi32(_mm256_add_epi32(identity(hi, n), half), shift);
        _mm256_packs_epi32(lo, hi)
    }

    /// The residuals of a column identity transform of `n` values:
    /// `Round2` by 4 of the outputs.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) fn identity_col(x: V, n: usize) -> V {
        let (lo, hi) = widen(x);
        let r = _mm256_set1_epi32(8);
        low16(
            _mm256_srai_epi32::<4>(_mm256_add_epi32(identity(lo, n), r)),
            _mm256_srai_epi32::<4>(_mm256_add_epi32(identity(hi, n), r)),
        )
    }
}

transforms!(
    #[target_feature(enable = "avx2")]
);

/// Eight vectors with rows and columns exchanged in each 128-bit half.
#[inline]
#[target_feature(enable = "avx2")]
fn transpose(r: [V; 8]) -> [V; 8] {
    let a0 = _mm256_unpacklo_epi16(r[0], r[1]);
    let a1 = _mm256_unpackhi_epi16(r[0], r[1]);
    let a2 = _mm256_unpacklo_epi16(r[2], r[3]);
    let a3 = _mm256_unpackhi_epi16(r[2], r[3]);
    let a4 = _mm256_unpacklo_epi16(r[4], r[5]);
    let a5 = _mm256_unpackhi_epi16(r[4], r[5]);
    let a6 = _mm256_unpacklo_epi16(r[6], r[7]);
    let a7 = _mm256_unpackhi_epi16(r[6], r[7]);
    let b0 = _mm256_unpacklo_epi32(a0, a2);
    let b1 = _mm256_unpackhi_epi32(a0, a2);
    let b2 = _mm256_unpacklo_epi32(a1, a3);
    let b3 = _mm256_unpackhi_epi32(a1, a3);
    let b4 = _mm256_unpacklo_epi32(a4, a6);
    let b5 = _mm256_unpackhi_epi32(a4, a6);
    let b6 = _mm256_unpacklo_epi32(a5, a7);
    let b7 = _mm256_unpackhi_epi32(a5, a7);
    [
        _mm256_unpacklo_epi64(b0, b4),
        _mm256_unpackhi_epi64(b0, b4),
        _mm256_unpacklo_epi64(b1, b5),
        _mm256_unpackhi_epi64(b1, b5),
        _mm256_unpacklo_epi64(b2, b6),
        _mm256_unpackhi_epi64(b2, b6),
        _mm256_unpacklo_epi64(b3, b7),
        _mm256_unpackhi_epi64(b3, b7),
    ]
}

/// The low 16 bits of the coefficients from `a` on in the low half of a
/// vector and of those from `b` on in the high half: eight of each, or
/// with `four` the first four and zeros. Zero past the end of `coefs`,
/// and for no `b`.
#[inline]
#[target_feature(enable = "avx2")]
fn coefs16(coefs: &[i32], a: usize, b: Option<usize>, four: bool) -> V {
    if four {
        let load = |at: usize| match coefs.get(at..).and_then(|c| c.first_chunk::<4>()) {
            Some(c) => {
                let v = _mm_setr_epi32(c[0], c[1], c[2], c[3]);
                _mm_packus_epi32(
                    _mm_and_si128(v, _mm_set1_epi32(0xffff)),
                    _mm_setzero_si128(),
                )
            }
            None => _mm_setzero_si128(),
        };
        return match b {
            Some(b) => _mm256_set_m128i(load(b), load(a)),
            None => _mm256_zextsi128_si256(load(a)),
        };
    }
    let load = |at: usize| match coefs.get(at..).and_then(|c| c.first_chunk::<8>()) {
        Some(c) => _mm256_and_si256(
            _mm256_setr_epi32(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]),
            _mm256_set1_epi32(0xffff),
        ),
        None => _mm256_setzero_si256(),
    };
    let hi = match b {
        Some(b) => load(b),
        None => _mm256_setzero_si256(),
    };
    // The pack takes four values of each in turn: put the halves back
    // in order.
    _mm256_permute4x64_epi64::<0b1101_1000>(_mm256_packus_epi32(load(a), hi))
}

/// Sixteen 16-bit values as one vector.
#[inline]
#[target_feature(enable = "avx2")]
fn ld256_i16(a: &[i16; 16]) -> V {
    _mm256_setr_epi16(
        a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13],
        a[14], a[15],
    )
}

/// `Round2(v, n)` for `n` from 1 on, which no 16-bit value overflows:
/// `count` holds `n` and `under` holds `n - 1`.
#[inline]
#[target_feature(enable = "avx2")]
fn round2(v: V, count: __m128i, under: __m128i) -> V {
    _mm256_add_epi16(
        _mm256_sra_epi16(v, count),
        _mm256_and_si256(_mm256_sra_epi16(v, under), _mm256_set1_epi16(1)),
    )
}

/// A shift count for the 16-bit and 32-bit shifts.
#[inline]
#[target_feature(enable = "avx2")]
fn count(n: u32) -> __m128i {
    _mm_set_epi64x(0, i64::from(n))
}

/// The smallest count of inputs a transform is made for that covers
/// `nz` of them: the entries of `T` from `nz` up to it are read too.
fn made_for(nz: usize) -> usize {
    match nz {
        0 | 1 => 1,
        2..=4 => 4,
        _ => nz.next_power_of_two(),
    }
}

#[target_feature(enable = "avx2")]
pub(super) fn add(b: &mut Block<'_>) {
    // By width and by the rows of values between the two passes that
    // the column transforms read: those the transform made for the
    // rows of coefficients takes. They take no more room than that, and
    // only that much is cleared.
    let rows = b.h.min(32).min(made_for(b.nz_rows).max(4));
    let zero = _mm256_setzero_si256();
    SCRATCH.with_borrow_mut(|s| {
        let s = s.get_or_insert([zero; 64]);
        match (b.w, rows) {
            (4, 4) => sized::<4, 16, 4>(b, s),
            (4, 8) => sized::<4, 16, 8>(b, s),
            (4, _) => sized::<4, 16, 16>(b, s),
            (8, 4) => sized::<8, 16, 4>(b, s),
            (8, 8) => sized::<8, 16, 8>(b, s),
            (8, 16) => sized::<8, 16, 16>(b, s),
            (8, _) => sized::<8, 16, 32>(b, s),
            (16, 4) => sized::<16, 16, 4>(b, s),
            (16, 8) => sized::<16, 16, 8>(b, s),
            (16, 16) => sized::<16, 16, 16>(b, s),
            (16, _) => sized::<16, 16, 32>(b, s),
            (32, 4) => sized::<32, 32, 4>(b, s),
            (32, 8) => sized::<32, 32, 8>(b, s),
            (32, 16) => sized::<32, 32, 16>(b, s),
            (32, _) => sized::<32, 32, 32>(b, s),
            (_, 4) => sized::<64, 64, 4>(b, s),
            (_, 8) => sized::<64, 64, 8>(b, s),
            (_, 16) => sized::<64, 64, 16>(b, s),
            _ => sized::<64, 64, 32>(b, s),
        }
    })
}

/// [`add`] for a block `W` wide whose column transforms read at most
/// `TH` rows of values from the row transforms. `MW` is the width in
/// whole vectors: 16 for a block 4 or 8 wide, whose vectors have lanes
/// that nothing reads.
#[target_feature(enable = "avx2")]
fn sized<const W: usize, const MW: usize, const TH: usize>(b: &mut Block<'_>, t: &mut [V; 64]) {
    // Everything of the block in values of its own: a store to a sample
    // then says nothing about them.
    let (h, stride, coefs) = (b.h, b.stride, b.coefs);
    let (row_kind, col_kind) = (b.row, b.col);
    let (flip_ud, flip_lr) = (b.flip_ud, b.flip_lr);
    let dst = &mut *b.dst;
    let tw = W.min(32);
    let nr = b.nz_rows.clamp(1, TH);
    let nc = b.nz_cols.clamp(1, tw);
    let rect = W.max(h) == 2 * W.min(h);
    let zero = _mm256_setzero_si256();
    let shift = b.shift;
    let (shift_count, shift_under) = (count(shift), count(shift.saturating_sub(1)));
    let half32 = _mm256_set1_epi32((1i32 << shift) >> 1);
    let four = count(4);
    let three = count(3);
    let (row_identity, col_identity) = (row_kind == Kind::Identity, col_kind == Kind::Identity);
    if row_identity && col_identity && !flip_ud && !flip_lr {
        // Both transforms scale each value on its own, so only the
        // samples under the coefficients change: no rows to gather into
        // vectors and none to spread back. (No transform type flips
        // these; a caller that does takes the long way.)
        for r in 0..nr {
            for k0 in (0..nc).step_by(16) {
                let at = r * tw + k0;
                let mut v = coefs16(coefs, at, (W > 8).then_some(at + 8), W == 4);
                if rect {
                    v = ops::scale(v, 2896);
                }
                let res = ops::identity_col(ops::identity_row(v, W, half32, shift_count), h);
                add_some::<W>(dst, r * stride + k0, res);
            }
        }
        return;
    }
    // The entries of `T` a transform reads are loaded or cleared here;
    // the rest it never reads.
    // The values after the row transforms, by row.
    let mut mid = [[0i16; MW]; TH];
    let read = made_for(nc).min(tw);
    for r0 in (0..nr).step_by(16) {
        // Sixteen rows of eight coefficients become eight entries of
        // `T`: rows `r0 + l` and `r0 + 8 + l` in the halves of a vector.
        for k0 in (0..read).step_by(8) {
            let mut m = [zero; 8];
            if k0 < nc {
                for (l, row) in m.iter_mut().enumerate() {
                    let (lo, hi) = (r0 + l, r0 + 8 + l);
                    if lo < nr {
                        *row = coefs16(
                            coefs,
                            lo * tw + k0,
                            (hi < nr).then_some(hi * tw + k0),
                            W == 4,
                        );
                    }
                }
                m = transpose(m);
                if rect {
                    for v in &mut m {
                        *v = ops::scale(*v, 2896);
                    }
                }
            }
            if let Some(d) = t.get_mut(k0..k0 + 8) {
                d.copy_from_slice(&m);
            }
        }
        if row_identity {
            // The outputs are the inputs: those past the last one read
            // are zero.
            for v in t.iter_mut().take(W).skip(read.next_multiple_of(8)) {
                *v = zero;
            }
        }
        run_1d(row_kind, W, nc, t);
        // And back: eight outputs become eight values of each row.
        for j0 in (0..W).step_by(8) {
            let Some(m) = t.get(j0..).and_then(|t| t.first_chunk::<8>()) else {
                continue;
            };
            let m = transpose(m.map(|v| {
                if row_identity {
                    ops::identity_row(v, W, half32, shift_count)
                } else if shift == 0 {
                    v
                } else {
                    round2(v, shift_count, shift_under)
                }
            }));
            for (l, v) in m.into_iter().enumerate() {
                let halves = [
                    (r0 + l, _mm256_castsi256_si128(v)),
                    (r0 + 8 + l, _mm256_extracti128_si256::<1>(v)),
                ];
                for (r, x) in halves {
                    if let Some(d) = mid
                        .get_mut(r)
                        .and_then(|row| row.as_chunks_mut::<8>().0.get_mut(j0 / 8))
                    {
                        st128_i16(d, x);
                    }
                }
            }
        }
    }
    // Rows the row transforms did not reach are zero in `mid`. The ADST
    // of four values is not made for fewer inputs: it reads all four.
    let read = if col_kind == Kind::Adst && h == 4 {
        4
    } else {
        made_for(nr).min(TH)
    };
    if nr == 1 && col_kind == Kind::Dct {
        // One row of values: every column is flat (the DCT of its first
        // input alone), so every row of the block gets the same residual.
        let mut res = [zero; 4];
        let groups = MW / 16;
        for (v, m) in res.iter_mut().zip(mid[0].as_chunks::<16>().0) {
            *v = round2(ops::scale(ld256_i16(m), 2896), four, three);
        }
        if flip_lr {
            res[..groups].reverse();
            for v in &mut res {
                *v = back::<W>(*v);
            }
        }
        for y in 0..h {
            let Some(row) = dst.get_mut(y * stride..).and_then(|d| d.get_mut(..W)) else {
                continue;
            };
            if W <= 16 {
                add_some::<W>(row, 0, res[0]);
                continue;
            }
            for (px, &r) in row.as_chunks_mut::<16>().0.iter_mut().zip(&res) {
                add16(px, r);
            }
        }
        return;
    }
    // Sixteen columns at a time: sixteen samples of a row.
    for j0 in (0..W).step_by(16) {
        for (v, row) in t.iter_mut().zip(&mid).take(read) {
            if let Some(m) = row.as_chunks::<16>().0.get(j0 / 16) {
                *v = ld256_i16(m);
            }
        }
        if col_identity {
            for v in t.iter_mut().take(h).skip(read) {
                *v = zero;
            }
        }
        run_1d(col_kind, h, nr, t);
        let x0 = if flip_lr { MW - 16 - j0 } else { j0 };
        for (i, &v) in t.iter().enumerate().take(h) {
            let y = if flip_ud { h - 1 - i } else { i };
            let res = if col_identity {
                ops::identity_col(v, h)
            } else {
                round2(v, four, three)
            };
            let res = if flip_lr { back::<W>(res) } else { res };
            add_some::<W>(dst, y * stride + x0, res);
        }
    }
}

/// Adds the residuals `res` to sixteen samples: the sum is within 16
/// bits, and the narrowing clamps it to a sample.
#[inline]
#[target_feature(enable = "avx2")]
fn add16(px: &mut [u8; 16], res: V) {
    let sum = _mm256_add_epi16(res, _mm256_cvtepu8_epi16(ld128(px)));
    let b = _mm256_permute4x64_epi64::<0b1000>(_mm256_packus_epi16(sum, sum));
    st128(px, _mm256_castsi256_si128(b));
}

/// [`add16`] for the eight samples under the low eight residuals.
#[inline]
#[target_feature(enable = "avx2")]
fn add8(px: &mut [u8; 8], res: V) {
    let sum = _mm256_add_epi16(res, _mm256_cvtepu8_epi16(ld64(px)));
    st64(px, _mm256_castsi256_si128(_mm256_packus_epi16(sum, sum)));
}

/// The lanes of a block `W` wide back to front: all sixteen, or the
/// first eight or four of a narrower block.
#[inline]
#[target_feature(enable = "avx2")]
fn back<const W: usize>(v: V) -> V {
    match W {
        4 => _mm256_shufflelo_epi16::<0b0001_1011>(v),
        8 => _mm256_shuffle_epi8(v, reversed()),
        _ => _mm256_permute4x64_epi64::<0b0100_1110>(_mm256_shuffle_epi8(v, reversed())),
    }
}

/// The byte order that turns the eight 16-bit values of each half back
/// to front.
#[inline]
#[target_feature(enable = "avx2")]
fn reversed() -> __m256i {
    let b: [u8; 32] = core::array::from_fn(|i| (14 - 2 * (i % 16 / 2) + i % 2) as u8);
    ld256(&b)
}

/// [`add16`] on the samples from `at` on of a block `W` wide: sixteen,
/// or the eight or four of a narrower block. Samples past the end are
/// left alone.
#[inline]
#[target_feature(enable = "avx2")]
fn add_some<const W: usize>(dst: &mut [u8], at: usize, res: V) {
    let Some(d) = dst.get_mut(at..) else {
        return;
    };
    match W {
        4 => {
            if let Some(px) = d.first_chunk_mut::<4>() {
                let mut eight = [0u8; 8];
                eight[..4].copy_from_slice(px);
                add8(&mut eight, res);
                px.copy_from_slice(&eight[..4]);
            }
        }
        8 => {
            if let Some(px) = d.first_chunk_mut::<8>() {
                add8(px, res);
            }
        }
        _ => {
            if let Some(px) = d.first_chunk_mut::<16>() {
                add16(px, res);
            }
        }
    }
}

#[target_feature(enable = "avx2")]
pub(super) fn add_flat(dst: &mut [u8], stride: usize, w: usize, h: usize, v: i32) {
    // Past 255 either way every sample ends at the same end. One of the
    // two is zero: a saturating add of the one and subtract of the other.
    let (up, down) = (
        v.clamp(0, 255) as u8 as i8,
        v.saturating_neg().clamp(0, 255) as u8 as i8,
    );
    let (up32, down32) = (_mm256_set1_epi8(up), _mm256_set1_epi8(down));
    let (up16, down16) = (_mm_set1_epi8(up), _mm_set1_epi8(down));
    let narrow = |x: __m128i| _mm_subs_epu8(_mm_adds_epu8(x, up16), down16);
    for y in 0..h {
        let Some(row) = dst.get_mut(y * stride..).and_then(|d| d.get_mut(..w)) else {
            continue;
        };
        let (wide, rest) = row.as_chunks_mut::<32>();
        for px in wide {
            st256(
                px,
                _mm256_subs_epu8(_mm256_adds_epu8(ld256(px), up32), down32),
            );
        }
        // A row of 16, 8 or 4.
        if let Some(px) = rest.first_chunk_mut::<16>() {
            st128(px, narrow(ld128(px)));
        } else if let Some(px) = rest.first_chunk_mut::<8>() {
            st64(px, narrow(ld64(px)));
        } else {
            for px in rest {
                *px = i32::from(*px).saturating_add(v).clamp(0, 255) as u8;
            }
        }
    }
}

/// The transform `kind` of `n` values on sixteen lanes, for tests.
#[cfg(test)]
#[target_feature(enable = "avx2")]
pub(super) fn one_dimension(kind: Kind, n: usize, nz: usize, t: &mut [[i16; 16]; 64]) -> bool {
    let mut v = [_mm256_setzero_si256(); 64];
    for (v, t) in v.iter_mut().zip(t.iter()) {
        *v = ld256_i16(t);
    }
    let ok = run_1d(kind, n, nz, &mut v);
    for (v, t) in v.iter().zip(t.iter_mut()) {
        crate::av1::cpu::x86::st256_i16(t, *v);
    }
    ok
}
