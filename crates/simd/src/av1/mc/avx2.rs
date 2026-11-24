//! The two-pass subpel filter on AVX2.
//!
//! Across, a shuffle puts samples `j + 2k` and `j + 2k + 1` side by side
//! for each output `j`, and `pmaddubsw` multiplies them by taps `2k` and
//! `2k + 1`. The taps go in halved (every AV1 tap is even), so that a sum
//! of products fits 16 bits: [`exact`] says whether a filter's do. A
//! vector holds sixteen outputs, eight from each of two loads eight
//! samples apart.
//!
//! Down, two rows are interleaved value by value and `pmaddwd` (or
//! `pmaddubsw`, when there was no pass across) multiplies each pair by two
//! taps. Output row `r` takes the pairs of rows `r`, `r + 2`, `r + 4` and
//! `r + 6`, and row `r + 2` shares three of them: the pairs of the even
//! and of the odd rows are kept apart, and each output row makes one new
//! pair.
//!
//! A filter whose first and last taps are 0 (the regular and smooth
//! filters, and every filter of a short block) runs with three pairs
//! instead of four, on two rows and two columns fewer.

use core::arch::x86_64::{
    __m256i, _mm256_add_epi16, _mm256_add_epi32, _mm256_castsi256_si128, _mm256_madd_epi16,
    _mm256_maddubs_epi16, _mm256_mulhrs_epi16, _mm256_packs_epi32, _mm256_packus_epi16,
    _mm256_permute2x128_si256, _mm256_permute4x64_epi64, _mm256_set_m128i, _mm256_set1_epi16,
    _mm256_set1_epi32, _mm256_shuffle_epi8, _mm256_srai_epi32, _mm256_unpackhi_epi8,
    _mm256_unpackhi_epi16, _mm256_unpacklo_epi8, _mm256_unpacklo_epi16, _mm256_zextsi128_si256,
};

use super::Block;
use crate::av1::cpu::x86::{ld64, ld128, ld256, st64, st128, st128_i16, st256, st256_i16};

/// Where a block's prediction goes.
pub(super) enum Out<'a> {
    Prep(&'a mut [i16]),
    Put(&'a mut [u8]),
}

/// Whether the kernels filter with `f` exactly: its taps are even, and
/// the halved positive taps and the halved negative ones each sum to at
/// most 128, so that no sum of their products with 8-bit samples leaves
/// 16 bits. Every row of the AV1 subpel filters passes.
#[inline]
pub(super) fn exact(f: &[i8; 8]) -> bool {
    let (mut pos, mut neg, mut odd) = (0i32, 0i32, 0i32);
    for &t in f {
        let t = i32::from(t);
        odd |= t & 1;
        pos += t.max(0);
        neg -= t.min(0);
    }
    odd == 0 && pos <= 256 && neg <= 256
}

/// Whether a filter's first and last taps are 0.
#[inline]
pub(super) fn six(f: &[i8; 8]) -> bool {
    f[0] == 0 && f[7] == 0
}

/// The interleaved halves of two rows.
type Pair = (__m256i, __m256i);

/// The pass across with `N` pairs of taps, the first tap `4 - N`.
#[derive(Clone, Copy)]
struct Across<'a, const N: usize> {
    src: &'a [u8],
    idx: [__m256i; N],
    taps: [__m256i; N],
}

impl<'a, const N: usize> Across<'a, N> {
    #[inline]
    #[target_feature(enable = "avx2")]
    fn new(src: &'a [u8], f: &[i8; 8]) -> Self {
        let t0 = 4 - N;
        Self {
            src,
            idx: core::array::from_fn(|k| {
                let b: [u8; 32] = core::array::from_fn(|i| (t0 + 2 * k + i % 16 / 2 + i % 2) as u8);
                ld256(&b)
            }),
            taps: core::array::from_fn(|k| {
                let (a, b) = (f[t0 + 2 * k] >> 1, f[t0 + 2 * k + 1] >> 1);
                _mm256_set1_epi16(i16::from_le_bytes([a as u8, b as u8]))
            }),
        }
    }

    /// Half the sums of eight outputs in each lane, from the lane's
    /// sixteen samples.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn sums(&self, v: __m256i) -> __m256i {
        let mut acc = _mm256_maddubs_epi16(_mm256_shuffle_epi8(v, self.idx[0]), self.taps[0]);
        for k in 1..N {
            acc = _mm256_add_epi16(
                acc,
                _mm256_maddubs_epi16(_mm256_shuffle_epi8(v, self.idx[k]), self.taps[k]),
            );
        }
        acc
    }

    /// `Round2(sum, 3)` of sixteen outputs whose first tap-0 sample is
    /// `src[at]`, or of eight (in the low lane) when not `WIDE`. `None`
    /// past the end of `src`.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn row<const WIDE: bool>(&self, at: usize) -> Option<__m256i> {
        let s = self.src.get(at..)?;
        let lo = ld128(s.first_chunk::<16>()?);
        let v = if WIDE {
            _mm256_set_m128i(ld128(s.get(8..)?.first_chunk::<16>()?), lo)
        } else {
            _mm256_zextsi128_si256(lo)
        };
        Some(_mm256_mulhrs_epi16(self.sums(v), _mm256_set1_epi16(8192)))
    }
}

/// One strip of a pass down with `N` pairs of taps: `next` gives the rows
/// in order, `pair` interleaves two of them, and `emit` takes each output
/// row and the pairs it is the sum of. `None` when `next` runs out.
#[inline]
#[target_feature(enable = "avx2")]
fn down<const N: usize>(
    h: usize,
    mut next: impl FnMut() -> Option<__m256i>,
    pair: impl Fn(__m256i, __m256i) -> Pair,
    mut emit: impl FnMut(usize, &[Pair; N]),
) -> Option<()> {
    let first = next()?;
    let mut last = first;
    // The pairs of the even output rows and of the odd ones; the last of
    // each is made in the loop.
    let mut even = [(first, first); N];
    let mut odd = even;
    for k in 0..N - 1 {
        let a = next()?;
        let b = next()?;
        even[k] = pair(last, a);
        odd[k] = pair(a, b);
        last = b;
    }
    let mut r = 0;
    while r < h {
        let a = next()?;
        even[N - 1] = pair(last, a);
        emit(r, &even);
        r += 1;
        if r == h {
            break;
        }
        let b = next()?;
        odd[N - 1] = pair(a, b);
        emit(r, &odd);
        r += 1;
        even.copy_within(1.., 0);
        odd.copy_within(1.., 0);
        last = b;
    }
    Some(())
}

/// The `N` pairs of taps of a pass down over 16-bit values, the first
/// tap `4 - N`.
#[inline]
#[target_feature(enable = "avx2")]
fn taps16<const N: usize>(f: &[i8; 8]) -> [__m256i; N] {
    let t0 = 4 - N;
    core::array::from_fn(|k| {
        let (a, b) = (i32::from(f[t0 + 2 * k]), i32::from(f[t0 + 2 * k + 1]));
        _mm256_set1_epi32((a & 0xffff) | (b << 16))
    })
}

/// The sums of one output row of the second pass, in 32 bits: the low
/// and the high four of each lane's eight.
#[inline]
#[target_feature(enable = "avx2")]
fn sums32<const N: usize>(p: &[Pair; N], t: &[__m256i; N]) -> Pair {
    let mut lo = _mm256_madd_epi16(p[0].0, t[0]);
    let mut hi = _mm256_madd_epi16(p[0].1, t[0]);
    for k in 1..N {
        lo = _mm256_add_epi32(lo, _mm256_madd_epi16(p[k].0, t[k]));
        hi = _mm256_add_epi32(hi, _mm256_madd_epi16(p[k].1, t[k]));
    }
    (lo, hi)
}

/// Stores sixteen 16-bit values at `dst[at..]`, or the low eight when
/// not `WIDE`, if they fit.
#[inline]
#[target_feature(enable = "avx2")]
fn store_i16<const WIDE: bool>(dst: &mut [i16], at: usize, v: __m256i) {
    let Some(d) = dst.get_mut(at..) else {
        return;
    };
    if WIDE {
        if let Some(d) = d.first_chunk_mut::<16>() {
            st256_i16(d, v);
        }
    } else if let Some(d) = d.first_chunk_mut::<8>() {
        st128_i16(d, _mm256_castsi256_si128(v));
    }
}

/// Stores sixteen 16-bit values clipped to 8 bits at `dst[at..]`, or the
/// low eight when not `WIDE`, if they fit.
#[inline]
#[target_feature(enable = "avx2")]
fn store_u8<const WIDE: bool>(dst: &mut [u8], at: usize, v: __m256i) {
    let Some(d) = dst.get_mut(at..) else {
        return;
    };
    let b = _mm256_packus_epi16(v, v);
    if WIDE {
        if let Some(d) = d.first_chunk_mut::<16>() {
            let b = _mm256_permute4x64_epi64::<0b1000>(b);
            st128(d, _mm256_castsi256_si128(b));
        }
    } else if let Some(d) = d.first_chunk_mut::<8>() {
        st64(d, _mm256_castsi256_si128(b));
    }
}

/// Both passes over a block in strips of sixteen columns (one of eight
/// when not `WIDE`), `NH` pairs of taps across and `NV` down.
#[inline]
#[target_feature(enable = "avx2")]
fn hv_strips<const NH: usize, const NV: usize, const WIDE: bool>(
    mut out: Out<'_>,
    dst_stride: usize,
    b: &Block<'_>,
) {
    let hz = Across::<NH>::new(b.src, b.fh);
    let taps = taps16::<NV>(b.fv);
    // The first row a pass down with `NV` pairs reads.
    let first = b.start + (4 - NV) * b.src_stride;
    let pair = |x: __m256i, y: __m256i| (_mm256_unpacklo_epi16(x, y), _mm256_unpackhi_epi16(x, y));
    for c in (0..b.w).step_by(16) {
        let mut at = first + c;
        let next = || {
            let v = hz.row::<WIDE>(at);
            at += b.src_stride;
            v
        };
        let done = match &mut out {
            Out::Prep(dst) => {
                let round = _mm256_set1_epi32(64);
                down::<NV>(b.h, next, pair, |r, p| {
                    let (lo, hi) = sums32(p, &taps);
                    let v = _mm256_packs_epi32(
                        _mm256_srai_epi32::<7>(_mm256_add_epi32(lo, round)),
                        _mm256_srai_epi32::<7>(_mm256_add_epi32(hi, round)),
                    );
                    store_i16::<WIDE>(dst, r * dst_stride + c, v);
                })
            }
            Out::Put(dst) => {
                let round = _mm256_set1_epi32(1024);
                down::<NV>(b.h, next, pair, |r, p| {
                    let (lo, hi) = sums32(p, &taps);
                    let v = _mm256_packs_epi32(
                        _mm256_srai_epi32::<11>(_mm256_add_epi32(lo, round)),
                        _mm256_srai_epi32::<11>(_mm256_add_epi32(hi, round)),
                    );
                    store_u8::<WIDE>(dst, r * dst_stride + c, v);
                })
            }
        };
        if done.is_none() {
            return;
        }
    }
}

/// Horizontal then vertical filter of a block. `out` chooses the
/// rounding of the vertical pass.
#[target_feature(enable = "avx2")]
pub(super) fn hv(out: Out<'_>, dst_stride: usize, b: &Block<'_>) {
    match (six(b.fh), six(b.fv), b.w >= 16) {
        (true, true, true) => hv_strips::<3, 3, true>(out, dst_stride, b),
        (true, true, false) => hv_strips::<3, 3, false>(out, dst_stride, b),
        (true, false, true) => hv_strips::<3, 4, true>(out, dst_stride, b),
        (true, false, false) => hv_strips::<3, 4, false>(out, dst_stride, b),
        (false, true, true) => hv_strips::<4, 3, true>(out, dst_stride, b),
        (false, true, false) => hv_strips::<4, 3, false>(out, dst_stride, b),
        (false, false, true) => hv_strips::<4, 4, true>(out, dst_stride, b),
        (false, false, false) => hv_strips::<4, 4, false>(out, dst_stride, b),
    }
}

/// The pass across alone, `N` pairs of taps.
#[inline]
#[target_feature(enable = "avx2")]
fn h_rows<const N: usize, const WIDE: bool>(mut out: Out<'_>, dst_stride: usize, b: &Block<'_>) {
    let hz = Across::<N>::new(b.src, b.fh);
    let first = b.start + 3 * b.src_stride;
    for r in 0..b.h {
        let (from, to) = (first + r * b.src_stride, r * dst_stride);
        for c in (0..b.w).step_by(16) {
            let Some(v) = hz.row::<WIDE>(from + c) else {
                return;
            };
            match &mut out {
                Out::Prep(dst) => store_i16::<WIDE>(dst, to + c, v),
                // Round2 by 4 more.
                Out::Put(dst) => {
                    store_u8::<WIDE>(dst, to + c, _mm256_mulhrs_epi16(v, _mm256_set1_epi16(2048)))
                }
            }
        }
    }
}

/// The horizontal filter alone (the vertical fraction is 0).
#[target_feature(enable = "avx2")]
pub(super) fn h(out: Out<'_>, dst_stride: usize, b: &Block<'_>) {
    match (six(b.fh), b.w >= 16) {
        (true, true) => h_rows::<3, true>(out, dst_stride, b),
        (true, false) => h_rows::<3, false>(out, dst_stride, b),
        (false, true) => h_rows::<4, true>(out, dst_stride, b),
        (false, false) => h_rows::<4, false>(out, dst_stride, b),
    }
}

/// `W` (8, 16 or 32) samples from `at` in the low bytes of a vector, or
/// `None` past the end of `src`.
#[inline]
#[target_feature(enable = "avx2")]
fn samples<const W: usize>(src: &[u8], at: usize) -> Option<__m256i> {
    let s = src.get(at..)?;
    Some(match W {
        8 => _mm256_zextsi128_si256(ld64(s.first_chunk::<8>()?)),
        16 => _mm256_zextsi128_si256(ld128(s.first_chunk::<16>()?)),
        _ => ld256(s.first_chunk::<32>()?),
    })
}

/// The pass down alone over samples, in strips of `W` columns, `N` pairs
/// of taps.
#[inline]
#[target_feature(enable = "avx2")]
fn v_strips<const N: usize, const W: usize>(mut out: Out<'_>, dst_stride: usize, b: &Block<'_>) {
    let t0 = 4 - N;
    let taps: [__m256i; N] = core::array::from_fn(|k| {
        let (x, y) = (b.fv[t0 + 2 * k] >> 1, b.fv[t0 + 2 * k + 1] >> 1);
        _mm256_set1_epi16(i16::from_le_bytes([x as u8, y as u8]))
    });
    let first = b.start + t0 * b.src_stride + 3;
    let pair = |x: __m256i, y: __m256i| (_mm256_unpacklo_epi8(x, y), _mm256_unpackhi_epi8(x, y));
    // Half the sums of an output row: the low and the high eight of each
    // lane's sixteen.
    let sums = |p: &[Pair; N]| {
        let mut lo = _mm256_maddubs_epi16(p[0].0, taps[0]);
        let mut hi = _mm256_maddubs_epi16(p[0].1, taps[0]);
        for k in 1..N {
            lo = _mm256_add_epi16(lo, _mm256_maddubs_epi16(p[k].0, taps[k]));
            hi = _mm256_add_epi16(hi, _mm256_maddubs_epi16(p[k].1, taps[k]));
        }
        (lo, hi)
    };
    for c in (0..b.w).step_by(W) {
        let mut at = first + c;
        let next = || {
            let v = samples::<W>(b.src, at);
            at += b.src_stride;
            v
        };
        let done = match &mut out {
            Out::Prep(dst) => down::<N>(b.h, next, pair, |r, p| {
                // Round2(sum, 3) of the whole taps.
                let (lo, hi) = sums(p);
                let k = _mm256_set1_epi16(8192);
                let (lo, hi) = (_mm256_mulhrs_epi16(lo, k), _mm256_mulhrs_epi16(hi, k));
                let to = r * dst_stride + c;
                match W {
                    8 => store_i16::<false>(dst, to, lo),
                    16 => store_i16::<true>(dst, to, _mm256_permute2x128_si256::<0x20>(lo, hi)),
                    _ => {
                        store_i16::<true>(dst, to, _mm256_permute2x128_si256::<0x20>(lo, hi));
                        store_i16::<true>(dst, to + 16, _mm256_permute2x128_si256::<0x31>(lo, hi));
                    }
                }
            }),
            Out::Put(dst) => down::<N>(b.h, next, pair, |r, p| {
                // Round2(sum, 7) of the whole taps, clipped.
                let (lo, hi) = sums(p);
                let k = _mm256_set1_epi16(512);
                let v = _mm256_packus_epi16(_mm256_mulhrs_epi16(lo, k), _mm256_mulhrs_epi16(hi, k));
                let Some(d) = dst.get_mut(r * dst_stride + c..) else {
                    return;
                };
                match W {
                    8 => {
                        if let Some(d) = d.first_chunk_mut::<8>() {
                            st64(d, _mm256_castsi256_si128(v));
                        }
                    }
                    16 => {
                        if let Some(d) = d.first_chunk_mut::<16>() {
                            st128(d, _mm256_castsi256_si128(v));
                        }
                    }
                    _ => {
                        if let Some(d) = d.first_chunk_mut::<32>() {
                            st256(d, v);
                        }
                    }
                }
            }),
        };
        if done.is_none() {
            return;
        }
    }
}

/// The vertical filter alone (the horizontal fraction is 0).
#[target_feature(enable = "avx2")]
pub(super) fn v(out: Out<'_>, dst_stride: usize, b: &Block<'_>) {
    match (six(b.fv), b.w) {
        (true, 8) => v_strips::<3, 8>(out, dst_stride, b),
        (true, 16) => v_strips::<3, 16>(out, dst_stride, b),
        (true, _) => v_strips::<3, 32>(out, dst_stride, b),
        (false, 8) => v_strips::<4, 8>(out, dst_stride, b),
        (false, 16) => v_strips::<4, 16>(out, dst_stride, b),
        (false, _) => v_strips::<4, 32>(out, dst_stride, b),
    }
}
