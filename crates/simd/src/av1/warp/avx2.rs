//! The warp on AVX2: the eight outputs of a row to a vector.
//!
//! A horizontal row multiplies the eight samples each output reads by its
//! filter, in 16-bit lanes with `pmaddwd`, two outputs to a vector: the
//! samples go in less 128, as the NEON kernel's do, and the rounding adds
//! back what that leaves out. Its sums come out with the even columns in
//! one half of the vector and the odd ones in the other, and stay so.
//!
//! The vertical pass turns the eight filters of an output row around, so
//! that a vector holds taps `2 j` and `2 j + 1` of every column side by
//! side, and multiplies them by two rows of the horizontal pass laid out
//! the same way.

use core::arch::x86_64::{
    __m128i, __m256i, _mm_set_epi64x, _mm_set1_epi8, _mm_shuffle_epi8, _mm_unpacklo_epi8,
    _mm_unpacklo_epi16, _mm_xor_si128, _mm256_add_epi32, _mm256_blend_epi16,
    _mm256_castsi256_si128, _mm256_cvtepi8_epi16, _mm256_extracti128_si256, _mm256_hadd_epi32,
    _mm256_madd_epi16, _mm256_packs_epi32, _mm256_packus_epi16, _mm256_set1_epi32,
    _mm256_setzero_si256, _mm256_slli_epi32, _mm256_srai_epi32, _mm256_unpackhi_epi32,
    _mm256_unpackhi_epi64, _mm256_unpacklo_epi32, _mm256_unpacklo_epi64,
};

use super::Filters;
use crate::av1::cpu::x86::{ld128, st64, st128_i16};

/// Two filters as sixteen 16-bit lanes: `lo` in the low half.
#[inline]
#[target_feature(enable = "avx2")]
fn taps2(lo: &[i8; 8], hi: &[i8; 8]) -> __m256i {
    let q = |f: &[i8; 8]| i64::from_le_bytes(f.map(|t| t as u8));
    _mm256_cvtepi8_epi16(_mm_set_epi64x(q(hi), q(lo)))
}

/// Row `r` of the horizontal pass: `Round2(sum, 3)` of each column in 32
/// bits, columns 0, 2, 4 and 6 in the low half and 1, 3, 5 and 7 in the
/// high one.
#[inline]
#[target_feature(enable = "avx2")]
fn horizontal(row: &[u8; 16], windows: &[__m128i; 4], f: &Filters<'_>, r: i32) -> __m256i {
    let [sx4, alpha, beta] = f.h;
    let at = sx4 + beta * (r - 7);
    let at = |c: i32| f.at(at + alpha * (c - 4));
    // The samples less 128: each filter sums to 128, so the sum is the
    // true one less 128 times 128, which the rounding adds back.
    let p = _mm_xor_si128(ld128(row), _mm_set1_epi8(-128));
    let m: [__m256i; 4] = core::array::from_fn(|k| {
        let c = 2 * k as i32;
        _mm256_madd_epi16(
            _mm256_cvtepi8_epi16(_mm_shuffle_epi8(p, windows[k])),
            taps2(at(c), at(c + 1)),
        )
    });
    let s = _mm256_hadd_epi32(_mm256_hadd_epi32(m[0], m[1]), _mm256_hadd_epi32(m[2], m[3]));
    _mm256_srai_epi32::<3>(_mm256_add_epi32(s, _mm256_set1_epi32(128 * 128 + 4)))
}

/// The vertical sums of output row `r`, columns laid out as
/// [`horizontal`] leaves them, from the four pairs of rows it reads.
#[inline]
#[target_feature(enable = "avx2")]
fn vertical(pairs: [__m256i; 4], f: &Filters<'_>, r: i32) -> __m256i {
    let [sy4, gamma, delta] = f.v;
    let at = sy4 + delta * (r - 4);
    let at = |c: i32| f.at(at + gamma * (c - 4));
    // The filters of columns c and c + 1 in the halves of one vector,
    // each as four pairs of taps; then turned around, so that a vector
    // holds one pair of taps of every column.
    let (x0, x1, x2, x3) = (
        taps2(at(0), at(1)),
        taps2(at(2), at(3)),
        taps2(at(4), at(5)),
        taps2(at(6), at(7)),
    );
    let (a0, a1) = (_mm256_unpacklo_epi32(x0, x1), _mm256_unpackhi_epi32(x0, x1));
    let (a2, a3) = (_mm256_unpacklo_epi32(x2, x3), _mm256_unpackhi_epi32(x2, x3));
    let t = [
        _mm256_unpacklo_epi64(a0, a2),
        _mm256_unpackhi_epi64(a0, a2),
        _mm256_unpacklo_epi64(a1, a3),
        _mm256_unpackhi_epi64(a1, a3),
    ];
    let mut s = _mm256_madd_epi16(pairs[0], t[0]);
    for k in 1..4 {
        s = _mm256_add_epi32(s, _mm256_madd_epi16(pairs[k], t[k]));
    }
    s
}

/// Both passes, `out` taking each output row's sums (columns laid out as
/// [`horizontal`] leaves them).
#[inline]
#[target_feature(enable = "avx2")]
fn warp<T>(rows: [&[u8; 16]; 15], f: &Filters<'_>, mut out: impl FnMut(__m256i) -> T) -> [T; 8] {
    // The eight samples of outputs `2 k` and `2 k + 1`.
    let windows: [__m128i; 4] = core::array::from_fn(|k| {
        let b: [u8; 16] = core::array::from_fn(|i| (2 * k + i / 8 + i % 8) as u8);
        ld128(&b)
    });
    let mut inter = [_mm256_setzero_si256(); 15];
    for (r, (v, row)) in inter.iter_mut().zip(rows).enumerate() {
        *v = horizontal(row, &windows, f, r as i32);
    }
    // Rows `k` and `k + 1` side by side in 16-bit lanes (a row's values
    // fit them).
    let pairs: [__m256i; 14] = core::array::from_fn(|k| {
        _mm256_blend_epi16::<0xAA>(inter[k], _mm256_slli_epi32::<16>(inter[k + 1]))
    });
    core::array::from_fn(|r| {
        let p = [pairs[r], pairs[r + 2], pairs[r + 4], pairs[r + 6]];
        out(vertical(p, f, r as i32))
    })
}

#[target_feature(enable = "avx2")]
pub(super) fn put8x8(rows: [&[u8; 16]; 15], f: &Filters<'_>) -> [[u8; 8]; 8] {
    warp(rows, f, |s| {
        let v = _mm256_srai_epi32::<11>(_mm256_add_epi32(s, _mm256_set1_epi32(1024)));
        let w = _mm256_packs_epi32(v, v);
        let b = _mm256_packus_epi16(w, w);
        let mut px = [0u8; 8];
        st64(
            &mut px,
            _mm_unpacklo_epi8(_mm256_castsi256_si128(b), _mm256_extracti128_si256::<1>(b)),
        );
        px
    })
}

#[target_feature(enable = "avx2")]
pub(super) fn prep8x8(rows: [&[u8; 16]; 15], f: &Filters<'_>) -> [[i16; 8]; 8] {
    warp(rows, f, |s| {
        let v = _mm256_srai_epi32::<7>(_mm256_add_epi32(s, _mm256_set1_epi32(64)));
        let w = _mm256_packs_epi32(v, v);
        let mut px = [0i16; 8];
        st128_i16(
            &mut px,
            _mm_unpacklo_epi16(_mm256_castsi256_si128(w), _mm256_extracti128_si256::<1>(w)),
        );
        px
    })
}
