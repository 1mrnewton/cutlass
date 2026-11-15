//! 8-bit block distortion on AVX2.
//!
//! SADs are `psadbw`: each 64-bit lane takes the sum of eight absolute
//! differences, at most 2040, and a block of 64×64 sums to at most a
//! million, far inside the lanes. Blocks 32 and 64 wide take a vector a
//! row (two for 64), 16 wide two rows a vector, 8 wide four rows. Squares
//! widen the samples to 16 bits and go through `pmaddwd` of the
//! differences by themselves: two squares a 32-bit lane (at most 130050),
//! and at most 128 such steps a lane over a 64×64 block (16.6 million).
//!
//! Loads are unaligned vector moves of `W` samples (or 8, or 16) from the
//! start of each of the `h` rows that [`fits`](super::fits) checked lie in
//! the slice.

use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi32, _mm_add_epi64, _mm_cvtsi128_si32, _mm_cvtsi128_si64,
    _mm_loadl_epi64, _mm_loadu_si128, _mm_sad_epu8, _mm_shuffle_epi32, _mm_unpackhi_epi64,
    _mm_unpacklo_epi64, _mm256_add_epi32, _mm256_add_epi64, _mm256_castsi256_si128,
    _mm256_cvtepu8_epi16, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16,
    _mm256_sad_epu8, _mm256_set_m128i, _mm256_setzero_si256, _mm256_sub_epi16,
};

/// # Safety
/// `o + 8 <= s.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn l64(s: &[u8], o: usize) -> __m128i {
    debug_assert!(o + 8 <= s.len());
    // SAFETY: the caller proves the 8 samples lie in `s`.
    unsafe { _mm_loadl_epi64(s.as_ptr().add(o).cast()) }
}

/// # Safety
/// `o + 16 <= s.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn l128(s: &[u8], o: usize) -> __m128i {
    debug_assert!(o + 16 <= s.len());
    // SAFETY: the caller proves the 16 samples lie in `s`.
    unsafe { _mm_loadu_si128(s.as_ptr().add(o).cast()) }
}

/// # Safety
/// `o + 32 <= s.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn l256(s: &[u8], o: usize) -> __m256i {
    debug_assert!(o + 32 <= s.len());
    // SAFETY: the caller proves the 32 samples lie in `s`.
    unsafe { _mm256_loadu_si256(s.as_ptr().add(o).cast()) }
}

/// Rows `r` to `r + 3` of a block 8 wide in one vector.
///
/// # Safety
/// The four rows hold 8 samples.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn four8(s: &[u8], st: usize, r: usize) -> __m256i {
    // SAFETY: as the caller proves.
    unsafe {
        let lo = _mm_unpacklo_epi64(l64(s, r * st), l64(s, (r + 1) * st));
        let hi = _mm_unpacklo_epi64(l64(s, (r + 2) * st), l64(s, (r + 3) * st));
        _mm256_set_m128i(hi, lo)
    }
}

/// Rows `r` and `r + 1` of a block 16 wide in one vector.
///
/// # Safety
/// The two rows hold 16 samples.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn two16(s: &[u8], st: usize, r: usize) -> __m256i {
    // SAFETY: as the caller proves.
    unsafe { _mm256_set_m128i(l128(s, (r + 1) * st), l128(s, r * st)) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn hsum64(v: __m256i) -> u64 {
    let s = _mm_add_epi64(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let s = _mm_add_epi64(s, _mm_unpackhi_epi64(s, s));
    _mm_cvtsi128_si64(s) as u64
}

#[inline]
#[target_feature(enable = "avx2")]
fn hsum32(v: __m256i) -> u32 {
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s));
    _mm_cvtsi128_si32(s) as u32
}

/// The SAD of rows `r..` of a block `W` wide, from the row group that
/// does not fill a vector (8 wide: under four rows; 16 wide: one row).
///
/// # Safety
/// Rows `r..h` hold `W` samples.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn tail<const W: usize>(
    a: &[u8],
    sa: usize,
    b: &[u8],
    sb: usize,
    r: usize,
    h: usize,
) -> u64 {
    let mut s = 0u64;
    // SAFETY: as the caller proves.
    unsafe {
        for r in r..h {
            let v = if W == 8 {
                _mm_sad_epu8(l64(a, r * sa), l64(b, r * sb))
            } else {
                _mm_sad_epu8(l128(a, r * sa), l128(b, r * sb))
            };
            s += _mm_cvtsi128_si64(_mm_add_epi64(v, _mm_unpackhi_epi64(v, v))) as u64;
        }
    }
    s
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
pub(super) fn sad(a: &[u8], sa: usize, b: &[u8], sb: usize, w: usize, h: usize) -> u32 {
    match w {
        8 => sad_w::<8>(a, sa, b, sb, h),
        16 => sad_w::<16>(a, sa, b, sb, h),
        32 => sad_w::<32>(a, sa, b, sb, h),
        _ => sad_w::<64>(a, sa, b, sb, h),
    }
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
fn sad_w<const W: usize>(a: &[u8], sa: usize, b: &[u8], sb: usize, h: usize) -> u32 {
    let mut acc = _mm256_setzero_si256();
    let mut rest = 0;
    // SAFETY: rows `0..h` of both blocks hold `W` samples (`fits`).
    unsafe {
        match W {
            8 => {
                let mut r = 0;
                while r + 4 <= h {
                    acc = _mm256_add_epi64(acc, _mm256_sad_epu8(four8(a, sa, r), four8(b, sb, r)));
                    r += 4;
                }
                rest = tail::<8>(a, sa, b, sb, r, h);
            }
            16 => {
                let mut r = 0;
                while r + 2 <= h {
                    acc = _mm256_add_epi64(acc, _mm256_sad_epu8(two16(a, sa, r), two16(b, sb, r)));
                    r += 2;
                }
                rest = tail::<16>(a, sa, b, sb, r, h);
            }
            _ => {
                for r in 0..h {
                    let (oa, ob) = (r * sa, r * sb);
                    let mut c = 0;
                    while c < W {
                        acc = _mm256_add_epi64(
                            acc,
                            _mm256_sad_epu8(l256(a, oa + c), l256(b, ob + c)),
                        );
                        c += 32;
                    }
                }
            }
        }
    }
    (hsum64(acc) + rest) as u32
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
pub(super) fn sad4(a: &[u8], sa: usize, b: [&[u8]; 4], sb: usize, w: usize, h: usize) -> [u32; 4] {
    match w {
        8 => sad4_w::<8>(a, sa, b, sb, h),
        16 => sad4_w::<16>(a, sa, b, sb, h),
        32 => sad4_w::<32>(a, sa, b, sb, h),
        _ => sad4_w::<64>(a, sa, b, sb, h),
    }
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
fn sad4_w<const W: usize>(a: &[u8], sa: usize, b: [&[u8]; 4], sb: usize, h: usize) -> [u32; 4] {
    let mut acc = [_mm256_setzero_si256(); 4];
    let mut rest = [0u64; 4];
    // SAFETY: rows `0..h` of all five blocks hold `W` samples (`fits`).
    unsafe {
        match W {
            8 => {
                let mut r = 0;
                while r + 4 <= h {
                    let x = four8(a, sa, r);
                    for k in 0..4 {
                        acc[k] = _mm256_add_epi64(acc[k], _mm256_sad_epu8(x, four8(b[k], sb, r)));
                    }
                    r += 4;
                }
                for k in 0..4 {
                    rest[k] = tail::<8>(a, sa, b[k], sb, r, h);
                }
            }
            16 => {
                let mut r = 0;
                while r + 2 <= h {
                    let x = two16(a, sa, r);
                    for k in 0..4 {
                        acc[k] = _mm256_add_epi64(acc[k], _mm256_sad_epu8(x, two16(b[k], sb, r)));
                    }
                    r += 2;
                }
                for k in 0..4 {
                    rest[k] = tail::<16>(a, sa, b[k], sb, r, h);
                }
            }
            _ => {
                for r in 0..h {
                    let (oa, ob) = (r * sa, r * sb);
                    let mut c = 0;
                    while c < W {
                        let x = l256(a, oa + c);
                        for k in 0..4 {
                            acc[k] =
                                _mm256_add_epi64(acc[k], _mm256_sad_epu8(x, l256(b[k], ob + c)));
                        }
                        c += 32;
                    }
                }
            }
        }
    }
    let mut out = [0u32; 4];
    for k in 0..4 {
        out[k] = (hsum64(acc[k]) + rest[k]) as u32;
    }
    out
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
pub(super) fn sse(a: &[u8], sa: usize, b: &[u8], sb: usize, w: usize, h: usize) -> u64 {
    match w {
        8 => sse_w::<8>(a, sa, b, sb, h),
        16 => sse_w::<16>(a, sa, b, sb, h),
        32 => sse_w::<32>(a, sa, b, sb, h),
        _ => sse_w::<64>(a, sa, b, sb, h),
    }
}

/// The squared differences of 16 samples, two to a 32-bit lane.
///
/// # Safety
/// Both hold 16 samples at their offsets.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn sq16(x: __m128i, y: __m128i) -> __m256i {
    let d = _mm256_sub_epi16(_mm256_cvtepu8_epi16(x), _mm256_cvtepu8_epi16(y));
    _mm256_madd_epi16(d, d)
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
fn sse_w<const W: usize>(a: &[u8], sa: usize, b: &[u8], sb: usize, h: usize) -> u64 {
    let mut acc = _mm256_setzero_si256();
    // SAFETY: rows `0..h` of both blocks hold `W` samples (`fits`).
    unsafe {
        if W == 8 {
            let mut r = 0;
            while r + 2 <= h {
                let x = _mm_unpacklo_epi64(l64(a, r * sa), l64(a, (r + 1) * sa));
                let y = _mm_unpacklo_epi64(l64(b, r * sb), l64(b, (r + 1) * sb));
                acc = _mm256_add_epi32(acc, sq16(x, y));
                r += 2;
            }
            if r < h {
                // The last of an odd count of rows.
                let mut t = 0u64;
                for c in 0..8 {
                    let d = u64::from(a[r * sa + c].abs_diff(b[r * sb + c]));
                    t += d * d;
                }
                return u64::from(hsum32(acc)) + t;
            }
        } else {
            for r in 0..h {
                let (oa, ob) = (r * sa, r * sb);
                let mut c = 0;
                while c < W {
                    acc = _mm256_add_epi32(acc, sq16(l128(a, oa + c), l128(b, ob + c)));
                    c += 16;
                }
            }
        }
    }
    u64::from(hsum32(acc))
}
