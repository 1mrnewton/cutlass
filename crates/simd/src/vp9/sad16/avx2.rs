//! 10- and 12-bit block distortion on AVX2.
//!
//! Samples of at most 4095 differ by at most 4095, which `psubw` and
//! `pabsw` give exactly. A row's absolute differences, up to four vectors
//! of them (16380 a lane), are added as 16-bit lanes and then summed in
//! pairs into 32-bit lanes by `pmaddwd` with ones. Squares go through
//! `pmaddwd` of the differences by themselves: two squares a 32-bit lane,
//! at most 33.5 million, four vectors a row at most 134 million; each
//! row's sums are widened to 64 bits. Blocks 8 wide take two rows a
//! vector, 4 wide two rows a 128-bit half.
//!
//! Loads are unaligned vector moves of `W` samples from the start of each
//! of the `h` rows that [`fits`](super::fits) checked lie in the slice.

use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi32, _mm_add_epi64, _mm_cvtsi128_si32, _mm_cvtsi128_si64,
    _mm_loadl_epi64, _mm_loadu_si128, _mm_setzero_si128, _mm_shuffle_epi32, _mm_unpackhi_epi64,
    _mm_unpacklo_epi64, _mm256_abs_epi16, _mm256_add_epi16, _mm256_add_epi32, _mm256_add_epi64,
    _mm256_castsi256_si128, _mm256_cvtepu32_epi64, _mm256_extracti128_si256, _mm256_loadu_si256,
    _mm256_madd_epi16, _mm256_set_m128i, _mm256_set1_epi16, _mm256_setzero_si256, _mm256_sub_epi16,
};

/// # Safety
/// `o + 4 <= s.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn l64(s: &[u16], o: usize) -> __m128i {
    debug_assert!(o + 4 <= s.len());
    // SAFETY: the caller proves the 4 samples lie in `s`.
    unsafe { _mm_loadl_epi64(s.as_ptr().add(o).cast()) }
}

/// # Safety
/// `o + 8 <= s.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn l128(s: &[u16], o: usize) -> __m128i {
    debug_assert!(o + 8 <= s.len());
    // SAFETY: the caller proves the 8 samples lie in `s`.
    unsafe { _mm_loadu_si128(s.as_ptr().add(o).cast()) }
}

/// # Safety
/// `o + 16 <= s.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn l256(s: &[u16], o: usize) -> __m256i {
    debug_assert!(o + 16 <= s.len());
    // SAFETY: the caller proves the 16 samples lie in `s`.
    unsafe { _mm256_loadu_si256(s.as_ptr().add(o).cast()) }
}

/// Rows `r` and `r + 1` of a block `W` (4 or 8) wide in one vector: for
/// 8, one row a lane; for 4, both in the low lane.
///
/// # Safety
/// Rows `r` and `r + 1` (`r` alone when `one`) hold `W` samples.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn two<const W: usize>(s: &[u16], st: usize, r: usize, one: bool) -> __m256i {
    // SAFETY: as the caller proves; with `one` the second row is zero.
    unsafe {
        if W == 8 {
            let lo = l128(s, r * st);
            let hi = if one {
                _mm_setzero_si128()
            } else {
                l128(s, (r + 1) * st)
            };
            _mm256_set_m128i(hi, lo)
        } else {
            let lo = l64(s, r * st);
            let hi = if one {
                _mm_setzero_si128()
            } else {
                l64(s, (r + 1) * st)
            };
            _mm256_set_m128i(_mm_setzero_si128(), _mm_unpacklo_epi64(lo, hi))
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn hsum32(v: __m256i) -> u32 {
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s));
    _mm_cvtsi128_si32(s) as u32
}

#[inline]
#[target_feature(enable = "avx2")]
fn hsum64(v: __m256i) -> u64 {
    let s = _mm_add_epi64(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let s = _mm_add_epi64(s, _mm_unpackhi_epi64(s, s));
    _mm_cvtsi128_si64(s) as u64
}

/// Sums a 32-bit lane vector into 64-bit lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn widen(acc: __m256i, v: __m256i) -> __m256i {
    let lo = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(v));
    let hi = _mm256_cvtepu32_epi64(_mm256_extracti128_si256::<1>(v));
    _mm256_add_epi64(acc, _mm256_add_epi64(lo, hi))
}

#[inline]
#[target_feature(enable = "avx2")]
fn absd(x: __m256i, y: __m256i) -> __m256i {
    _mm256_abs_epi16(_mm256_sub_epi16(x, y))
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
pub(super) fn sad(a: &[u16], sa: usize, b: &[u16], sb: usize, w: usize, h: usize) -> u32 {
    match w {
        4 => sad_w::<4>(a, sa, b, sb, h),
        8 => sad_w::<8>(a, sa, b, sb, h),
        16 => sad_w::<16>(a, sa, b, sb, h),
        32 => sad_w::<32>(a, sa, b, sb, h),
        _ => sad_w::<64>(a, sa, b, sb, h),
    }
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
fn sad_w<const W: usize>(a: &[u16], sa: usize, b: &[u16], sb: usize, h: usize) -> u32 {
    let ones = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_si256();
    // SAFETY: rows `0..h` of both blocks hold `W` samples (`fits`).
    unsafe {
        if W <= 8 {
            let mut r = 0;
            while r < h {
                let one = r + 1 == h;
                let d = absd(two::<W>(a, sa, r, one), two::<W>(b, sb, r, one));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d, ones));
                r += 2;
            }
        } else {
            for r in 0..h {
                let (oa, ob) = (r * sa, r * sb);
                let mut s = absd(l256(a, oa), l256(b, ob));
                let mut c = 16;
                while c < W {
                    s = _mm256_add_epi16(s, absd(l256(a, oa + c), l256(b, ob + c)));
                    c += 16;
                }
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(s, ones));
            }
        }
    }
    hsum32(acc)
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
pub(super) fn sad4(
    a: &[u16],
    sa: usize,
    b: [&[u16]; 4],
    sb: usize,
    w: usize,
    h: usize,
) -> [u32; 4] {
    match w {
        4 => sad4_w::<4>(a, sa, b, sb, h),
        8 => sad4_w::<8>(a, sa, b, sb, h),
        16 => sad4_w::<16>(a, sa, b, sb, h),
        32 => sad4_w::<32>(a, sa, b, sb, h),
        _ => sad4_w::<64>(a, sa, b, sb, h),
    }
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
fn sad4_w<const W: usize>(a: &[u16], sa: usize, b: [&[u16]; 4], sb: usize, h: usize) -> [u32; 4] {
    let ones = _mm256_set1_epi16(1);
    let mut acc = [_mm256_setzero_si256(); 4];
    // SAFETY: rows `0..h` of all five blocks hold `W` samples (`fits`).
    unsafe {
        if W <= 8 {
            let mut r = 0;
            while r < h {
                let one = r + 1 == h;
                let x = two::<W>(a, sa, r, one);
                for k in 0..4 {
                    let d = absd(x, two::<W>(b[k], sb, r, one));
                    acc[k] = _mm256_add_epi32(acc[k], _mm256_madd_epi16(d, ones));
                }
                r += 2;
            }
        } else {
            for r in 0..h {
                let (oa, ob) = (r * sa, r * sb);
                let mut s = [_mm256_setzero_si256(); 4];
                let mut c = 0;
                while c < W {
                    let x = l256(a, oa + c);
                    for k in 0..4 {
                        s[k] = _mm256_add_epi16(s[k], absd(x, l256(b[k], ob + c)));
                    }
                    c += 16;
                }
                for k in 0..4 {
                    acc[k] = _mm256_add_epi32(acc[k], _mm256_madd_epi16(s[k], ones));
                }
            }
        }
    }
    acc.map(|v| hsum32(v))
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
pub(super) fn sse(a: &[u16], sa: usize, b: &[u16], sb: usize, w: usize, h: usize) -> u64 {
    match w {
        4 => sse_w::<4>(a, sa, b, sb, h),
        8 => sse_w::<8>(a, sa, b, sb, h),
        16 => sse_w::<16>(a, sa, b, sb, h),
        32 => sse_w::<32>(a, sa, b, sb, h),
        _ => sse_w::<64>(a, sa, b, sb, h),
    }
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
fn sse_w<const W: usize>(a: &[u16], sa: usize, b: &[u16], sb: usize, h: usize) -> u64 {
    let mut acc = _mm256_setzero_si256();
    // SAFETY: rows `0..h` of both blocks hold `W` samples (`fits`).
    unsafe {
        if W <= 8 {
            // Two rows a vector: each 32-bit lane takes at most 33.5
            // million a step, so eight steps fit before widening.
            let mut part = _mm256_setzero_si256();
            let mut r = 0;
            let mut n = 0;
            while r < h {
                let one = r + 1 == h;
                let d = _mm256_sub_epi16(two::<W>(a, sa, r, one), two::<W>(b, sb, r, one));
                part = _mm256_add_epi32(part, _mm256_madd_epi16(d, d));
                n += 1;
                if n == 8 {
                    acc = widen(acc, part);
                    part = _mm256_setzero_si256();
                    n = 0;
                }
                r += 2;
            }
            acc = widen(acc, part);
        } else {
            for r in 0..h {
                let (oa, ob) = (r * sa, r * sb);
                let mut s = _mm256_setzero_si256();
                let mut c = 0;
                while c < W {
                    let d = _mm256_sub_epi16(l256(a, oa + c), l256(b, ob + c));
                    s = _mm256_add_epi32(s, _mm256_madd_epi16(d, d));
                    c += 16;
                }
                acc = widen(acc, s);
            }
        }
    }
    hsum64(acc)
}
