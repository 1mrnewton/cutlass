//! The 10- and 12-bit interpolation on AVX2.
//!
//! Samples (below 32768) and taps are both signed 16-bit values, so
//! `pmaddwd` of two neighbouring samples by two neighbouring taps gives
//! an exact 32-bit partial sum. Across, a load from `o + 2k` multiplied by
//! the tap pair `k` gives, in 32-bit lane `m`, the part of output `2m`
//! that taps `2k` and `2k + 1` make; the load from `o + 2k + 1` gives the
//! part of output `2m + 1`. The four pairs are summed for the even and the
//! odd outputs, rounded by `(x + 64) >> 7`, interleaved back into order
//! and packed with unsigned saturation, which holds them at 0, and the
//! minimum with `max` holds them at the top. Down, two rows are
//! interleaved sample by sample and multiplied the same way. The sums are
//! exact, so the result is the scalar twin's.
//!
//! Sixteen outputs go in a 256-bit vector, eight (and four, for blocks 4
//! wide) in a 128-bit one. With a fraction down, each column strip keeps
//! the eight source rows its next output row needs in registers: rows are
//! loaded, or filtered across, once each.
//!
//! Loads and stores are unaligned vector moves on the region
//! [`Block::fits`](super::Block) checked once for the block: every row the
//! kernel touches lies in `src` with `w + 24` samples from the first it
//! reads (3 left of the block with a fraction across, 3 rows above and 4
//! below with one down), and `dst` holds `h` rows of `w`. The farthest
//! load across reads to 23 samples past the first of a strip of 16, which
//! starts at most 16 before the end of the row.

use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi32, _mm_avg_epu16, _mm_loadl_epi64, _mm_loadu_si128,
    _mm_madd_epi16, _mm_min_epu16, _mm_packus_epi32, _mm_set1_epi16, _mm_set1_epi32,
    _mm_setzero_si128, _mm_srai_epi32, _mm_storel_epi64, _mm_storeu_si128, _mm_unpackhi_epi32,
    _mm_unpacklo_epi32, _mm256_add_epi32, _mm256_avg_epu16, _mm256_castsi128_si256,
    _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16,
    _mm256_min_epu16, _mm256_packus_epi32, _mm256_permute2x128_si256, _mm256_set_m128i,
    _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setzero_si256, _mm256_srai_epi32,
    _mm256_storeu_si256, _mm256_unpackhi_epi16, _mm256_unpackhi_epi32, _mm256_unpacklo_epi16,
    _mm256_unpacklo_epi32,
};

use super::Block;

// Loads and stores ---------------------------------------------------------
//
// Offsets count samples. Each states the bound its caller proves;
// `debug_assert!` checks it in tests.

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

/// # Safety
/// `o + 4 <= s.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn s64(s: &mut [u16], o: usize, v: __m128i) {
    debug_assert!(o + 4 <= s.len());
    // SAFETY: the caller proves the 4 samples lie in `s`.
    unsafe { _mm_storel_epi64(s.as_mut_ptr().add(o).cast(), v) }
}

/// # Safety
/// `o + 8 <= s.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn s128(s: &mut [u16], o: usize, v: __m128i) {
    debug_assert!(o + 8 <= s.len());
    // SAFETY: the caller proves the 8 samples lie in `s`.
    unsafe { _mm_storeu_si128(s.as_mut_ptr().add(o).cast(), v) }
}

/// # Safety
/// `o + 16 <= s.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn s256(s: &mut [u16], o: usize, v: __m256i) {
    debug_assert!(o + 16 <= s.len());
    // SAFETY: the caller proves the 16 samples lie in `s`.
    unsafe { _mm256_storeu_si256(s.as_mut_ptr().add(o).cast(), v) }
}

// Filters ------------------------------------------------------------------

/// A filter's tap pairs `2k`, `2k + 1` in each 32-bit lane, and the top
/// of the sample range.
#[derive(Clone, Copy)]
struct Taps {
    p: [__m256i; 4],
    p1: [__m128i; 4],
    max: __m256i,
    max1: __m128i,
}

impl Taps {
    #[inline]
    #[target_feature(enable = "avx2")]
    fn new(f: Option<&super::Filter>, max: u16) -> Self {
        let t = f.map_or([0; 8], |f| *f.taps());
        let pair = |k: usize| i32::from(t[2 * k] as u16) | (i32::from(t[2 * k + 1]) << 16);
        let pr: [i32; 4] = core::array::from_fn(pair);
        Self {
            p: pr.map(|v| _mm256_set1_epi32(v)),
            p1: pr.map(|v| _mm_set1_epi32(v)),
            max: _mm256_set1_epi16(max as i16),
            max1: _mm_set1_epi16(max as i16),
        }
    }

    /// `(x + 64) >> 7` of the even and odd sums, back in order, held to
    /// `0..=max`.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn pack(&self, e: __m256i, o: __m256i) -> __m256i {
        let r = _mm256_set1_epi32(64);
        let e = _mm256_srai_epi32::<7>(_mm256_add_epi32(e, r));
        let o = _mm256_srai_epi32::<7>(_mm256_add_epi32(o, r));
        let v = _mm256_packus_epi32(_mm256_unpacklo_epi32(e, o), _mm256_unpackhi_epi32(e, o));
        _mm256_min_epu16(v, self.max)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn pack1(&self, e: __m128i, o: __m128i) -> __m128i {
        let r = _mm_set1_epi32(64);
        let e = _mm_srai_epi32::<7>(_mm_add_epi32(e, r));
        let o = _mm_srai_epi32::<7>(_mm_add_epi32(o, r));
        let v = _mm_packus_epi32(_mm_unpacklo_epi32(e, o), _mm_unpackhi_epi32(e, o));
        _mm_min_epu16(v, self.max1)
    }

    /// The same for sums of the low and high halves of each lane (the
    /// filter down): no interleave.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn pack_halves(&self, lo: __m256i, hi: __m256i) -> __m256i {
        let r = _mm256_set1_epi32(64);
        let lo = _mm256_srai_epi32::<7>(_mm256_add_epi32(lo, r));
        let hi = _mm256_srai_epi32::<7>(_mm256_add_epi32(hi, r));
        _mm256_min_epu16(_mm256_packus_epi32(lo, hi), self.max)
    }

    /// Sixteen outputs across from `s[o..]` (3 before the first output).
    ///
    /// # Safety
    /// `o + 24 <= s.len()`.
    #[inline]
    #[target_feature(enable = "avx2")]
    #[allow(unsafe_code)]
    unsafe fn h16(&self, s: &[u16], o: usize) -> __m256i {
        // SAFETY: the loads read `s[o..o + 23]`, as the caller proves.
        unsafe {
            let m = |k: usize, j: usize| _mm256_madd_epi16(l256(s, o + 2 * k + j), self.p[k]);
            let e = _mm256_add_epi32(
                _mm256_add_epi32(m(0, 0), m(1, 0)),
                _mm256_add_epi32(m(2, 0), m(3, 0)),
            );
            let d = _mm256_add_epi32(
                _mm256_add_epi32(m(0, 1), m(1, 1)),
                _mm256_add_epi32(m(2, 1), m(3, 1)),
            );
            self.pack(e, d)
        }
    }

    /// Eight outputs across from `s[o..]` (3 before the first output).
    ///
    /// # Safety
    /// `o + 16 <= s.len()`.
    #[inline]
    #[target_feature(enable = "avx2")]
    #[allow(unsafe_code)]
    unsafe fn h8(&self, s: &[u16], o: usize) -> __m128i {
        // SAFETY: the loads read `s[o..o + 15]`, as the caller proves.
        unsafe {
            let m = |k: usize, j: usize| _mm_madd_epi16(l128(s, o + 2 * k + j), self.p1[k]);
            let e = _mm_add_epi32(
                _mm_add_epi32(m(0, 0), m(1, 0)),
                _mm_add_epi32(m(2, 0), m(3, 0)),
            );
            let d = _mm_add_epi32(
                _mm_add_epi32(m(0, 1), m(1, 1)),
                _mm_add_epi32(m(2, 1), m(3, 1)),
            );
            self.pack1(e, d)
        }
    }

    /// Eight outputs across of two rows, `s[o..]` and `s[o + ss..]`, one
    /// in each 128-bit lane.
    ///
    /// # Safety
    /// `o + ss + 16 <= s.len()`.
    #[inline]
    #[target_feature(enable = "avx2")]
    #[allow(unsafe_code)]
    unsafe fn h8x2(&self, s: &[u16], o: usize, ss: usize) -> __m256i {
        // SAFETY: the loads read `s[o..o + 15]` and `s[o + ss..o + ss + 15]`,
        // as the caller proves.
        unsafe {
            let m = |k: usize, j: usize| {
                let x = _mm256_set_m128i(l128(s, o + ss + 2 * k + j), l128(s, o + 2 * k + j));
                _mm256_madd_epi16(x, self.p[k])
            };
            let e = _mm256_add_epi32(
                _mm256_add_epi32(m(0, 0), m(1, 0)),
                _mm256_add_epi32(m(2, 0), m(3, 0)),
            );
            let d = _mm256_add_epi32(
                _mm256_add_epi32(m(0, 1), m(1, 1)),
                _mm256_add_epi32(m(2, 1), m(3, 1)),
            );
            self.pack(e, d)
        }
    }

    /// Sixteen outputs down from eight rows.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn v16(&self, x: &[__m256i; 8]) -> __m256i {
        let lo =
            |k: usize| _mm256_madd_epi16(_mm256_unpacklo_epi16(x[2 * k], x[2 * k + 1]), self.p[k]);
        let hi =
            |k: usize| _mm256_madd_epi16(_mm256_unpackhi_epi16(x[2 * k], x[2 * k + 1]), self.p[k]);
        let l = _mm256_add_epi32(
            _mm256_add_epi32(lo(0), lo(1)),
            _mm256_add_epi32(lo(2), lo(3)),
        );
        let h = _mm256_add_epi32(
            _mm256_add_epi32(hi(0), hi(1)),
            _mm256_add_epi32(hi(2), hi(3)),
        );
        self.pack_halves(l, h)
    }
}

/// The filter down over `h` output rows of a strip of 16, from source
/// rows `row(0)` to `row(h + 6)` (each asked for once, in order), to
/// `put(r, v)`. The eight rows an output row needs stay in registers.
#[inline]
#[target_feature(enable = "avx2")]
fn down16(
    t: &Taps,
    h: usize,
    mut row: impl FnMut(usize) -> __m256i,
    mut put: impl FnMut(usize, __m256i),
) {
    let mut x: [__m256i; 8] = core::array::from_fn(|k| {
        if k < 7 {
            row(k)
        } else {
            _mm256_setzero_si256()
        }
    });
    for r in 0..h {
        x[7] = row(r + 7);
        put(r, t.v16(&x));
        x = [x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[7]];
    }
}

/// Writes eight outputs (`W` of 4 or 8) at `dst[d..]`, averaged with
/// `avg`.
///
/// # Safety
/// `d + W <= dst.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn put8<const W: usize>(dst: &mut [u16], d: usize, avg: bool, v: __m128i) {
    // SAFETY: as the caller proves.
    unsafe {
        if W == 4 {
            let v = if avg {
                _mm_avg_epu16(v, l64(dst, d))
            } else {
                v
            };
            s64(dst, d, v);
        } else {
            let v = if avg {
                _mm_avg_epu16(v, l128(dst, d))
            } else {
                v
            };
            s128(dst, d, v);
        }
    }
}

/// Writes sixteen outputs at `dst[d..]`, averaged with `avg`.
///
/// # Safety
/// `d + 16 <= dst.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn put16(dst: &mut [u16], d: usize, avg: bool, v: __m256i) {
    // SAFETY: as the caller proves.
    unsafe {
        let v = if avg {
            _mm256_avg_epu16(v, l256(dst, d))
        } else {
            v
        };
        s256(dst, d, v);
    }
}

/// Writes two rows of eight outputs (`W` of 4 or 8), the low lane of
/// `v` at `dst[d..]` and, unless `one`, the high lane a row below.
///
/// # Safety
/// `d + W <= dst.len()`, and `d + ds + W <= dst.len()` unless `one`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn put8x2<const W: usize>(
    dst: &mut [u16],
    d: usize,
    ds: usize,
    avg: bool,
    v: __m256i,
    one: bool,
) {
    // SAFETY: as the caller proves.
    unsafe {
        put8::<W>(dst, d, avg, _mm256_castsi256_si128(v));
        if !one {
            put8::<W>(dst, d + ds, avg, _mm256_extracti128_si256::<1>(v));
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
pub(super) fn predict(b: &mut Block<'_>) {
    match b.w {
        4 => sized::<4>(b),
        8 => sized::<8>(b),
        16 => sized::<16>(b),
        32 => sized::<32>(b),
        _ => sized::<64>(b),
    }
}

/// One block of width `W`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
fn sized<const W: usize>(b: &mut Block<'_>) {
    let (ds, src, at, ss, h, fx, fy, avg, max) = (
        b.dst_stride,
        b.src,
        b.at,
        b.src_stride,
        b.h,
        b.fx,
        b.fy,
        b.avg,
        b.max,
    );
    let dst = &mut *b.dst;
    // SAFETY: `fits` checked the region (the module's notes): rows of
    // `src` from the first read hold `W + 24` samples; a strip of 16
    // starts at column `c <= W - 16` and its loads read to `c + 23`
    // (`c + 15` for the 8-wide forms, from `c = 0`); `dst` holds `h` rows
    // of `W`, and strip `c` writes `dst[c..c + 16]` of a row (`W` for the
    // narrow forms).
    unsafe {
        match (fx, fy) {
            (None, None) => {
                for r in 0..h {
                    let (o, d) = (at + r * ss, r * ds);
                    match W {
                        4 => put8::<4>(dst, d, avg, l64(src, o)),
                        8 => put8::<8>(dst, d, avg, l128(src, o)),
                        _ => {
                            let mut c = 0;
                            while c < W {
                                put16(dst, d + c, avg, l256(src, o + c));
                                c += 16;
                            }
                        }
                    }
                }
            }
            (Some(fx), None) => {
                let t = Taps::new(Some(fx), max);
                for r in 0..h {
                    let (o, d) = (at + r * ss - 3, r * ds);
                    if W <= 8 {
                        if r % 2 == 1 {
                            continue;
                        }
                        if r + 1 < h {
                            put8x2::<W>(dst, d, ds, avg, t.h8x2(src, o, ss), false);
                        } else {
                            put8::<W>(dst, d, avg, t.h8(src, o));
                        }
                    } else {
                        let mut c = 0;
                        while c < W {
                            put16(dst, d + c, avg, t.h16(src, o + c));
                            c += 16;
                        }
                    }
                }
            }
            (None, Some(fy)) => {
                let t = Taps::new(Some(fy), max);
                let top = at - 3 * ss;
                if W <= 8 {
                    // Lane 0 of window vector `k` is source row `r + k`,
                    // lane 1 row `r + k + 1`: two output rows a pass.
                    let last = h + 6;
                    let row = |j: usize| {
                        if j <= last {
                            l128(src, top + j * ss)
                        } else {
                            _mm_setzero_si128()
                        }
                    };
                    let pair = |j: usize| _mm256_set_m128i(row(j + 1), row(j));
                    let mut x: [__m256i; 8] = core::array::from_fn(|k| {
                        if k < 6 {
                            pair(k)
                        } else {
                            _mm256_setzero_si256()
                        }
                    });
                    let mut r = 0;
                    while r < h {
                        x[6] = pair(r + 6);
                        x[7] = pair(r + 7);
                        put8x2::<W>(dst, r * ds, ds, avg, t.v16(&x), r + 1 == h);
                        x = [x[2], x[3], x[4], x[5], x[6], x[7], x[7], x[7]];
                        r += 2;
                    }
                } else {
                    let mut c = 0;
                    while c < W {
                        down16(
                            &t,
                            h,
                            |j| l256(src, top + j * ss + c),
                            |r, v| put16(dst, r * ds + c, avg, v),
                        );
                        c += 16;
                    }
                }
            }
            (Some(fx), Some(fy)) => {
                let (tx, ty) = (Taps::new(Some(fx), max), Taps::new(Some(fy), max));
                let top = at - 3 * ss - 3;
                if W <= 8 {
                    // `b(i)` is rows `2i` and `2i + 1` filtered across, one
                    // a lane (a row past the last is zero); the window
                    // vector of rows `j`, `j + 1` is `b(j / 2)` for even
                    // `j`, and the high lane of `b(j / 2)` with the low one
                    // of the next for odd `j`.
                    let last = h + 6;
                    let b = |i: usize| {
                        let j = 2 * i;
                        if j < last {
                            tx.h8x2(src, top + j * ss, ss)
                        } else if j == last {
                            _mm256_castsi128_si256(tx.h8(src, top + j * ss))
                        } else {
                            _mm256_setzero_si256()
                        }
                    };
                    let odd = |a: __m256i, b: __m256i| _mm256_permute2x128_si256::<0x21>(a, b);
                    let (b0, b1, b2, b3) = (b(0), b(1), b(2), b(3));
                    let mut x = [
                        b0,
                        odd(b0, b1),
                        b1,
                        odd(b1, b2),
                        b2,
                        odd(b2, b3),
                        b3,
                        _mm256_setzero_si256(),
                    ];
                    let mut prev = b3;
                    let mut i = 4;
                    let mut r = 0;
                    while r < h {
                        let next = b(i);
                        x[7] = odd(prev, next);
                        put8x2::<W>(dst, r * ds, ds, avg, ty.v16(&x), r + 1 == h);
                        x = [x[2], x[3], x[4], x[5], x[6], x[7], next, next];
                        prev = next;
                        i += 1;
                        r += 2;
                    }
                } else {
                    let mut c = 0;
                    while c < W {
                        down16(
                            &ty,
                            h,
                            |j| tx.h16(src, top + j * ss + c),
                            |r, v| put16(dst, r * ds + c, avg, v),
                        );
                        c += 16;
                    }
                }
            }
        }
    }
}
