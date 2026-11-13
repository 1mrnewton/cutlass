//! The filtered block (see [`super::predict`]) with AVX-512 for blocks 32
//! and 64 wide: both passes of the filter on 512-bit registers, 32 values
//! of a row at a time, and each half of a result stored as the AVX2
//! kernel stores sixteen values ([`super::avx2`]). Narrower blocks and
//! blocks copied without a filter go to the AVX2 kernel.

// The loops index rows, strips and taps by one position at once.
#![allow(clippy::needless_range_loop)]

use core::arch::x86_64::{
    __m128i, __m512i, _mm_set_epi64x, _mm_setzero_si128, _mm256_set_m128i, _mm256_setzero_si256,
    _mm512_add_epi16, _mm512_add_epi32, _mm512_broadcast_i32x4, _mm512_castsi128_si512,
    _mm512_castsi512_si256, _mm512_cvtepu8_epi16, _mm512_extracti64x4_epi64, _mm512_inserti32x4,
    _mm512_madd_epi16, _mm512_maddubs_epi16, _mm512_mullo_epi16, _mm512_packs_epi32,
    _mm512_set1_epi16, _mm512_set1_epi32, _mm512_setzero_si512, _mm512_shuffle_epi8,
    _mm512_srai_epi32, _mm512_unpackhi_epi16, _mm512_unpacklo_epi16,
};

use super::avx2::{self, Out, Row, Strips, pair_mask, src_row, tap_pair, tap_pair32};
use super::{Src, Taps};

/// Rows of horizontally filtered values kept between the passes: up to
/// two strips of 32 for each row of a block's window.
pub(super) type Wide = [__m512i; 2];

/// The filtered block (see [`super::predict`]).
#[target_feature(enable = "avx2,avx512f,avx512bw")]
pub(super) fn predict(
    dst: super::Dst<'_>,
    stride: usize,
    src: &Src<'_>,
    (w, h): (usize, usize),
    taps: &Taps,
    (tmp, wide): (&mut Vec<Strips>, &mut Vec<Wide>),
) {
    if tmp.len() < 64 + 7 {
        tmp.resize(64 + 7, [_mm256_setzero_si256(); 4]);
    }
    if wide.len() < 64 + 7 {
        wide.resize(64 + 7, [_mm512_setzero_si512(); 2]);
    }
    let (tmp, wide) = (tmp.as_mut_slice(), wide.as_mut_slice());
    let (mode, mut out) = avx2::out_of(dst, stride, w);
    let split = |a: usize| Src {
        at: src.at + a,
        ..*src
    };
    macro_rules! go {
        ($w:literal, $r:literal, $t:literal, $x:expr) => {
            match mode {
                0 => avx2::put_w::<$w, $r, $t, 0>(&mut out.at($x), &split($x), h, taps, tmp),
                1 => avx2::put_w::<$w, $r, $t, 1>(&mut out.at($x), &split($x), h, taps, tmp),
                2 => avx2::put_w::<$w, $r, $t, 2>(&mut out.at($x), &split($x), h, taps, tmp),
                3 => avx2::put_w::<$w, $r, $t, 3>(&mut out.at($x), &split($x), h, taps, tmp),
                _ => avx2::put_w::<$w, $r, $t, 4>(&mut out.at($x), &split($x), h, taps, tmp),
            }
        };
    }
    macro_rules! wide {
        ($w:literal, $r:literal, $t:literal) => {
            match mode {
                0 => put_wide::<$w, $r, $t, 0>(&mut out.at(0), src, h, taps, (tmp, wide)),
                1 => put_wide::<$w, $r, $t, 1>(&mut out.at(0), src, h, taps, (tmp, wide)),
                2 => put_wide::<$w, $r, $t, 2>(&mut out.at(0), src, h, taps, (tmp, wide)),
                3 => put_wide::<$w, $r, $t, 3>(&mut out.at(0), src, h, taps, (tmp, wide)),
                _ => put_wide::<$w, $r, $t, 4>(&mut out.at(0), src, h, taps, (tmp, wide)),
            }
        };
    }
    macro_rules! widths {
        ($t:literal) => {
            match w {
                4 => go!(4, 24, $t, 0),
                8 => go!(8, 24, $t, 0),
                12 => {
                    go!(8, 24, $t, 0);
                    go!(4, 24, $t, 8);
                }
                16 => go!(16, 24, $t, 0),
                24 => {
                    go!(16, 24, $t, 0);
                    go!(8, 24, $t, 16);
                }
                32 => wide!(32, 40, $t),
                48 => {
                    wide!(32, 40, $t);
                    go!(16, 24, $t, 32);
                }
                _ => wide!(64, 72, $t),
            }
        };
    }
    if taps.len == 8 {
        widths!(8);
    } else {
        widths!(4);
    }
}

/// The horizontal filter of one row, `T` taps, 32 outputs at a time.
#[derive(Clone, Copy)]
struct Across<const T: usize> {
    masks: [__m512i; 4],
    pairs: [__m512i; 4],
}

impl<const T: usize> Across<T> {
    #[inline]
    #[target_feature(enable = "avx2,avx512f,avx512bw")]
    fn new(f: &[i8; 8]) -> Self {
        Self {
            masks: core::array::from_fn(|k| _mm512_broadcast_i32x4(pair_mask(k as i8))),
            pairs: core::array::from_fn(|k| _mm512_set1_epi16(tap_pair(f, k))),
        }
    }

    /// 32 outputs from `v`, whose four 128-bit lanes hold the sixteen
    /// samples from columns 0, 8, 16 and 24 of the outputs.
    #[inline]
    #[target_feature(enable = "avx2,avx512f,avx512bw")]
    fn thirty_two(&self, v: __m512i) -> __m512i {
        let mut sum = _mm512_maddubs_epi16(_mm512_shuffle_epi8(v, self.masks[0]), self.pairs[0]);
        for k in 1..T / 2 {
            let p = _mm512_maddubs_epi16(_mm512_shuffle_epi8(v, self.masks[k]), self.pairs[k]);
            sum = _mm512_add_epi16(sum, p);
        }
        sum
    }
}

/// The 32 values `v` from column `x` into output row `o`, as two halves.
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw")]
fn emit<const W: usize, const M: u8>(o: &mut Row<'_, W>, x: usize, v: __m512i) {
    o.v16::<M>(x, _mm512_castsi512_si256(v));
    o.v16::<M>(x + 16, _mm512_extracti64x4_epi64::<1>(v));
}

/// [`super::predict`] of a block `W` wide (32 or 64) with `T` taps into
/// output kind `M`; `R` is `W + 8`. As [`avx2::put_w`], on 512-bit
/// registers; `wide` keeps the horizontal pass of each window row.
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw")]
fn put_wide<const W: usize, const R: usize, const T: usize, const M: u8>(
    out: &mut Out<'_>,
    src: &Src<'_>,
    h: usize,
    taps: &Taps,
    (tmp, wide): (&mut [Strips], &mut [Wide]),
) {
    let h0 = T / 2 - 1;
    // At most 64 rows (the entry checks): `wide` holds their window.
    let h = h.min(64);
    let strips = W / 32;
    // Sixteen samples of row `r` from column `x`.
    let load = |r: &[u8; R], x: usize| -> __m128i {
        match r.get(x..x + 16).and_then(|a| a.first_chunk::<16>()) {
            Some(a) => {
                let v = u128::from_le_bytes(*a);
                _mm_set_epi64x((v >> 64) as i64, v as i64)
            }
            None => _mm_setzero_si128(),
        }
    };
    // The samples of columns `x`, `x + 8`, `x + 16` and `x + 24` on, a
    // lane each, for 32 outputs of the horizontal filter.
    let lanes = |r: &[u8; R], x: usize| -> __m512i {
        let v = _mm512_castsi128_si512(load(r, x));
        let v = _mm512_inserti32x4::<1>(v, load(r, x + 8));
        let v = _mm512_inserti32x4::<2>(v, load(r, x + 16));
        _mm512_inserti32x4::<3>(v, load(r, x + 24))
    };
    match (taps.across, taps.down) {
        (None, None) => avx2::put_w::<W, R, T, M>(out, src, h, taps, tmp),
        (Some(fh), None) => {
            let across = Across::<T>::new(&fh);
            for j in 0..h {
                let (Some(r), Some(mut o)) = (src_row::<R>(src, j + h0), out.row::<W, M>(j)) else {
                    return;
                };
                for s in 0..strips {
                    emit::<W, M>(&mut o, 32 * s, across.thirty_two(lanes(r, 32 * s)));
                }
            }
        }
        (None, Some(fv)) => {
            // Rows of samples as 16-bit values, then the taps down.
            let c: [__m512i; 8] = core::array::from_fn(|k| _mm512_set1_epi16(i16::from(fv[k])));
            let Some(rows) = wide.get_mut(..h + T - 1) else {
                return;
            };
            for (j, t) in rows.iter_mut().enumerate() {
                let Some(r) = src_row::<R>(src, j) else {
                    return;
                };
                for s in 0..strips {
                    let x = h0 + 32 * s;
                    let bytes = _mm256_set_m128i(load(r, x + 16), load(r, x));
                    t[s] = _mm512_cvtepu8_epi16(bytes);
                }
            }
            for j in 0..h {
                let (Some(win), Some(mut o)) = (
                    rows.get(j..).and_then(|w| w.first_chunk::<T>()),
                    out.row::<W, M>(j),
                ) else {
                    return;
                };
                for s in 0..strips {
                    let mut sum = _mm512_mullo_epi16(win[0][s], c[0]);
                    for k in 1..T {
                        sum = _mm512_add_epi16(sum, _mm512_mullo_epi16(win[k][s], c[k]));
                    }
                    emit::<W, M>(&mut o, 32 * s, sum);
                }
            }
        }
        (Some(fh), Some(fv)) => {
            // The horizontal pass of each window row once; the vertical
            // pass multiplies pairs of rows by pairs of taps into 32-bit
            // sums.
            let across = Across::<T>::new(&fh);
            let p: [__m512i; 4] = core::array::from_fn(|k| _mm512_set1_epi32(tap_pair32(&fv, k)));
            let Some(rows) = wide.get_mut(..h + T - 1) else {
                return;
            };
            for (j, t) in rows.iter_mut().enumerate() {
                let Some(r) = src_row::<R>(src, j) else {
                    return;
                };
                for s in 0..strips {
                    t[s] = across.thirty_two(lanes(r, 32 * s));
                }
            }
            for j in 0..h {
                let (Some(win), Some(mut o)) = (
                    rows.get(j..).and_then(|w| w.first_chunk::<T>()),
                    out.row::<W, M>(j),
                ) else {
                    return;
                };
                for s in 0..strips {
                    let (mut lo, mut hi) = (_mm512_setzero_si512(), _mm512_setzero_si512());
                    for k in 0..T / 2 {
                        let (a, b) = (win[2 * k][s], win[2 * k + 1][s]);
                        lo = _mm512_add_epi32(
                            lo,
                            _mm512_madd_epi16(_mm512_unpacklo_epi16(a, b), p[k]),
                        );
                        hi = _mm512_add_epi32(
                            hi,
                            _mm512_madd_epi16(_mm512_unpackhi_epi16(a, b), p[k]),
                        );
                    }
                    let v =
                        _mm512_packs_epi32(_mm512_srai_epi32::<6>(lo), _mm512_srai_epi32::<6>(hi));
                    emit::<W, M>(&mut o, 32 * s, v);
                }
            }
        }
    }
}
