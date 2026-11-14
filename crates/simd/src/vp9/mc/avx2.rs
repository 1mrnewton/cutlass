//! The 8-bit interpolation on AVX2.
//!
//! Across, a shuffle puts samples `j + 2k` and `j + 2k + 1` side by side
//! for each output `j`, and `pmaddubsw` multiplies them by taps `2k` and
//! `2k + 1`. No pair sum can saturate (each pair of taps sums to at most
//! 136 in magnitude with signs that differ, or to less than 128). The four
//! sums are added outer pairs first and the larger inner pair last, with
//! saturation, so the total saturates only when the result would be held
//! to 0 or 255 anyway: the same samples as libvpx's C. `pmulhrsw` by 256
//! is `(x + 64) >> 7`, and `packuswb` holds the result to 0..=255. Down,
//! two rows are interleaved byte by byte and multiplied the same way.
//!
//! Thirty-two outputs across come from two loads of each lane eight
//! samples apart: the pack of their sums is already in order. With both
//! fractions the rows filtered across go to `tmp`, 64 samples apart, and
//! the filter down reads them there.
//!
//! Loads and stores are unaligned vector moves on the region
//! [`Block::fits`](super::Block) checked once for the block: every row the
//! kernel touches lies in `src` with `w + 24` samples from the first it
//! reads (3 left of the block with a fraction across, 3 rows above and 4
//! below with one down), `dst` holds `h` rows of `w`, and `tmp` holds
//! [`TMP`](super::TMP) samples. The offsets below stay inside those: the
//! farthest load across reads to `w + 8` of its row.

use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_adds_epi16, _mm_avg_epu8, _mm_cvtsi32_si128,
    _mm_cvtsi128_si32, _mm_loadl_epi64, _mm_loadu_si128, _mm_maddubs_epi16, _mm_mulhrs_epi16,
    _mm_packus_epi16, _mm_set1_epi16, _mm_setr_epi8, _mm_shuffle_epi8, _mm_storel_epi64,
    _mm_storeu_si128, _mm_unpackhi_epi8, _mm_unpacklo_epi8, _mm256_add_epi16, _mm256_adds_epi16,
    _mm256_avg_epu8, _mm256_castsi128_si256, _mm256_castsi256_si128, _mm256_extracti128_si256,
    _mm256_inserti128_si256, _mm256_loadu_si256, _mm256_maddubs_epi16, _mm256_mulhrs_epi16,
    _mm256_packus_epi16, _mm256_set_m128i, _mm256_set1_epi16, _mm256_shuffle_epi8,
    _mm256_storeu_si256, _mm256_unpackhi_epi8, _mm256_unpacklo_epi8,
};

use super::Block;

// Loads and stores ---------------------------------------------------------
//
// Each states the bound its caller proves; `debug_assert!` checks it in
// tests.

/// # Safety
/// `o + 4 <= s.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn l32(s: &[u8], o: usize) -> __m128i {
    debug_assert!(o + 4 <= s.len());
    // SAFETY: the caller proves the 4 bytes lie in `s`.
    _mm_cvtsi32_si128(unsafe { s.as_ptr().add(o).cast::<i32>().read_unaligned() })
}

/// # Safety
/// `o + 8 <= s.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn l64(s: &[u8], o: usize) -> __m128i {
    debug_assert!(o + 8 <= s.len());
    // SAFETY: the caller proves the 8 bytes lie in `s`.
    unsafe { _mm_loadl_epi64(s.as_ptr().add(o).cast()) }
}

/// # Safety
/// `o + 16 <= s.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn l128(s: &[u8], o: usize) -> __m128i {
    debug_assert!(o + 16 <= s.len());
    // SAFETY: the caller proves the 16 bytes lie in `s`.
    unsafe { _mm_loadu_si128(s.as_ptr().add(o).cast()) }
}

/// # Safety
/// `o + 32 <= s.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn l256(s: &[u8], o: usize) -> __m256i {
    debug_assert!(o + 32 <= s.len());
    // SAFETY: the caller proves the 32 bytes lie in `s`.
    unsafe { _mm256_loadu_si256(s.as_ptr().add(o).cast()) }
}

/// # Safety
/// `o + 4 <= s.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn s32(s: &mut [u8], o: usize, v: __m128i) {
    debug_assert!(o + 4 <= s.len());
    // SAFETY: the caller proves the 4 bytes lie in `s`.
    unsafe {
        s.as_mut_ptr()
            .add(o)
            .cast::<i32>()
            .write_unaligned(_mm_cvtsi128_si32(v));
    }
}

/// # Safety
/// `o + 8 <= s.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn s64(s: &mut [u8], o: usize, v: __m128i) {
    debug_assert!(o + 8 <= s.len());
    // SAFETY: the caller proves the 8 bytes lie in `s`.
    unsafe { _mm_storel_epi64(s.as_mut_ptr().add(o).cast(), v) }
}

/// # Safety
/// `o + 16 <= s.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn s128(s: &mut [u8], o: usize, v: __m128i) {
    debug_assert!(o + 16 <= s.len());
    // SAFETY: the caller proves the 16 bytes lie in `s`.
    unsafe { _mm_storeu_si128(s.as_mut_ptr().add(o).cast(), v) }
}

/// # Safety
/// `o + 32 <= s.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn s256(s: &mut [u8], o: usize, v: __m256i) {
    debug_assert!(o + 32 <= s.len());
    // SAFETY: the caller proves the 32 bytes lie in `s`.
    unsafe { _mm256_storeu_si256(s.as_mut_ptr().add(o).cast(), v) }
}

// Filters ------------------------------------------------------------------

/// Taps `2k` and `2k + 1` as the byte pair `pmaddubsw` multiplies by.
#[inline]
fn pair(f: &[i16; 8], k: usize) -> i16 {
    i16::from_le_bytes([f[2 * k] as i8 as u8, f[2 * k + 1] as i8 as u8])
}

/// For output `j` of eight, samples `j + 2k` and `j + 2k + 1`.
#[inline]
#[target_feature(enable = "avx2")]
fn pair_mask(k: i8) -> __m128i {
    let b = 2 * k;
    _mm_setr_epi8(
        b,
        b + 1,
        b + 1,
        b + 2,
        b + 2,
        b + 3,
        b + 3,
        b + 4,
        b + 4,
        b + 5,
        b + 5,
        b + 6,
        b + 6,
        b + 7,
        b + 7,
        b + 8,
    )
}

#[derive(Clone, Copy)]
struct Taps {
    masks: [__m256i; 4],
    pairs: [__m256i; 4],
    masks1: [__m128i; 4],
    pairs1: [__m128i; 4],
}

impl Taps {
    #[inline]
    #[target_feature(enable = "avx2")]
    fn new(f: &[i16; 8]) -> Self {
        let m = [pair_mask(0), pair_mask(1), pair_mask(2), pair_mask(3)];
        let p: [i16; 4] = core::array::from_fn(|k| pair(f, k));
        Self {
            masks: m.map(|m| _mm256_set_m128i(m, m)),
            pairs: p.map(|v| _mm256_set1_epi16(v)),
            masks1: m,
            pairs1: p.map(|v| _mm_set1_epi16(v)),
        }
    }

    /// The rounded sum of four pair products, added in the order that
    /// keeps it exact.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn sum(p: [__m256i; 4]) -> __m256i {
        let s = _mm256_add_epi16(p[0], p[3]);
        let s = _mm256_add_epi16(s, p[1]);
        let s = _mm256_adds_epi16(s, p[2]);
        _mm256_mulhrs_epi16(s, _mm256_set1_epi16(256))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn sum1(p: [__m128i; 4]) -> __m128i {
        let s = _mm_add_epi16(p[0], p[3]);
        let s = _mm_add_epi16(s, p[1]);
        let s = _mm_adds_epi16(s, p[2]);
        _mm_mulhrs_epi16(s, _mm_set1_epi16(256))
    }

    /// Across, for the eight outputs of each lane of `v` (16 samples from
    /// 3 before the lane's first output).
    #[inline]
    #[target_feature(enable = "avx2")]
    fn across(&self, v: __m256i) -> __m256i {
        Self::sum(core::array::from_fn(|k| {
            _mm256_maddubs_epi16(_mm256_shuffle_epi8(v, self.masks[k]), self.pairs[k])
        }))
    }

    /// Eight outputs across (the low 8 bytes) from `s[o..]`.
    ///
    /// # Safety
    /// `o + 16 <= s.len()`.
    #[inline]
    #[target_feature(enable = "avx2")]
    #[allow(unsafe_code)]
    unsafe fn h8(&self, s: &[u8], o: usize) -> __m128i {
        // SAFETY: as the caller proves.
        let v = unsafe { l128(s, o) };
        let r = Self::sum1(core::array::from_fn(|k| {
            _mm_maddubs_epi16(_mm_shuffle_epi8(v, self.masks1[k]), self.pairs1[k])
        }));
        _mm_packus_epi16(r, r)
    }

    /// Sixteen outputs across from `s[o..]`.
    ///
    /// # Safety
    /// `o + 24 <= s.len()`.
    #[inline]
    #[target_feature(enable = "avx2")]
    #[allow(unsafe_code)]
    unsafe fn h16(&self, s: &[u8], o: usize) -> __m128i {
        // SAFETY: as the caller proves.
        let v = unsafe {
            _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(l128(s, o)), l128(s, o + 8))
        };
        let r = self.across(v);
        _mm_packus_epi16(_mm256_castsi256_si128(r), _mm256_extracti128_si256::<1>(r))
    }

    /// Thirty-two outputs across from `s[o..]`.
    ///
    /// # Safety
    /// `o + 40 <= s.len()`.
    #[inline]
    #[target_feature(enable = "avx2")]
    #[allow(unsafe_code)]
    unsafe fn h32(&self, s: &[u8], o: usize) -> __m256i {
        // SAFETY: as the caller proves.
        let (a, b) = unsafe {
            (
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(l128(s, o)), l128(s, o + 16)),
                _mm256_inserti128_si256::<1>(
                    _mm256_castsi128_si256(l128(s, o + 8)),
                    l128(s, o + 24),
                ),
            )
        };
        _mm256_packus_epi16(self.across(a), self.across(b))
    }

    /// Eight outputs down (the low 8 bytes) from rows `s[o..]`, `ss`
    /// apart.
    ///
    /// # Safety
    /// `o + 7 * ss + 8 <= s.len()`.
    #[inline]
    #[target_feature(enable = "avx2")]
    #[allow(unsafe_code)]
    unsafe fn v8(&self, s: &[u8], o: usize, ss: usize) -> __m128i {
        // SAFETY: as the caller proves.
        let x: [__m128i; 8] = core::array::from_fn(|k| unsafe { l64(s, o + k * ss) });
        let r = Self::sum1(core::array::from_fn(|k| {
            _mm_maddubs_epi16(_mm_unpacklo_epi8(x[2 * k], x[2 * k + 1]), self.pairs1[k])
        }));
        _mm_packus_epi16(r, r)
    }

    /// Sixteen outputs down.
    ///
    /// # Safety
    /// `o + 7 * ss + 16 <= s.len()`.
    #[inline]
    #[target_feature(enable = "avx2")]
    #[allow(unsafe_code)]
    unsafe fn v16(&self, s: &[u8], o: usize, ss: usize) -> __m128i {
        // SAFETY: as the caller proves.
        let x: [__m128i; 8] = core::array::from_fn(|k| unsafe { l128(s, o + k * ss) });
        let lo = Self::sum1(core::array::from_fn(|k| {
            _mm_maddubs_epi16(_mm_unpacklo_epi8(x[2 * k], x[2 * k + 1]), self.pairs1[k])
        }));
        let hi = Self::sum1(core::array::from_fn(|k| {
            _mm_maddubs_epi16(_mm_unpackhi_epi8(x[2 * k], x[2 * k + 1]), self.pairs1[k])
        }));
        _mm_packus_epi16(lo, hi)
    }

    /// Thirty-two outputs down.
    ///
    /// # Safety
    /// `o + 7 * ss + 32 <= s.len()`.
    #[inline]
    #[target_feature(enable = "avx2")]
    #[allow(unsafe_code)]
    unsafe fn v32(&self, s: &[u8], o: usize, ss: usize) -> __m256i {
        // SAFETY: as the caller proves.
        let x: [__m256i; 8] = core::array::from_fn(|k| unsafe { l256(s, o + k * ss) });
        let lo = Self::sum(core::array::from_fn(|k| {
            _mm256_maddubs_epi16(_mm256_unpacklo_epi8(x[2 * k], x[2 * k + 1]), self.pairs[k])
        }));
        let hi = Self::sum(core::array::from_fn(|k| {
            _mm256_maddubs_epi16(_mm256_unpackhi_epi8(x[2 * k], x[2 * k + 1]), self.pairs[k])
        }));
        _mm256_packus_epi16(lo, hi)
    }
}

/// Every block of [`Block::fits`](super::Block).
#[target_feature(enable = "avx2")]
pub(super) fn predict(b: &mut Block<'_>) {
    match b.w {
        4 => sized::<4>(b),
        8 => sized::<8>(b),
        16 => sized::<16>(b),
        32 => sized::<32>(b),
        _ => sized::<64>(b),
    }
}

/// Where the filtered rows go between the passes: 64 samples apart.
const TS: usize = 64;

/// Writes one row of `W` outputs at `dst[d..]`, from `g8` (4 and 8
/// wide), `g16` or `g32` (each 32 from column `c`), averaged into what is
/// there with `avg`.
///
/// # Safety
/// `d + W <= dst.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn put<const W: usize>(
    dst: &mut [u8],
    d: usize,
    avg: bool,
    g8: impl Fn() -> __m128i,
    g16: impl Fn() -> __m128i,
    g32: impl Fn(usize) -> __m256i,
) {
    // SAFETY: each store and load of `dst` covers `d..d + W`.
    unsafe {
        match W {
            4 => {
                let v = g8();
                let v = if avg { _mm_avg_epu8(v, l32(dst, d)) } else { v };
                s32(dst, d, v);
            }
            8 => {
                let v = g8();
                let v = if avg { _mm_avg_epu8(v, l64(dst, d)) } else { v };
                s64(dst, d, v);
            }
            16 => {
                let v = g16();
                let v = if avg {
                    _mm_avg_epu8(v, l128(dst, d))
                } else {
                    v
                };
                s128(dst, d, v);
            }
            _ => {
                let mut c = 0;
                while c < W {
                    let v = g32(c);
                    let v = if avg {
                        _mm256_avg_epu8(v, l256(dst, d + c))
                    } else {
                        v
                    };
                    s256(dst, d + c, v);
                    c += 32;
                }
            }
        }
    }
}

/// One block of width `W`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
fn sized<const W: usize>(b: &mut Block<'_>) {
    // Field by field: the block is read where it was written, without a
    // wide copy of the whole struct.
    let (ds, src, at, ss, h, fx, fy, avg) = (
        b.dst_stride,
        b.src,
        b.at,
        b.src_stride,
        b.h,
        b.fx,
        b.fy,
        b.avg,
    );
    let (dst, tmp) = (&mut *b.dst, &mut *b.tmp);
    // SAFETY: `fits` checked the region (the module's notes): rows of
    // `src` from the first read hold `W + 24` samples, the loads across
    // read 16 from `c + 8` for `c < W - 8` at most and 8 or 16 for `W` of
    // 4 or 8; `dst` holds `h` rows of `W`; `tmp` holds `h + 7` rows of
    // `TS`.
    unsafe {
        match (fx, fy) {
            (None, None) => {
                for r in 0..h {
                    let o = at + r * ss;
                    put::<W>(
                        dst,
                        r * ds,
                        avg,
                        || if W == 4 { l32(src, o) } else { l64(src, o) },
                        || l128(src, o),
                        |c| l256(src, o + c),
                    );
                }
            }
            (Some(fx), None) => {
                let t = Taps::new(&fx);
                for r in 0..h {
                    let o = at + r * ss - 3;
                    put::<W>(
                        dst,
                        r * ds,
                        avg,
                        || t.h8(src, o),
                        || t.h16(src, o),
                        |c| t.h32(src, o + c),
                    );
                }
            }
            (None, Some(fy)) => {
                let t = Taps::new(&fy);
                let top = at - 3 * ss;
                for r in 0..h {
                    let o = top + r * ss;
                    put::<W>(
                        dst,
                        r * ds,
                        avg,
                        || t.v8(src, o, ss),
                        || t.v16(src, o, ss),
                        |c| t.v32(src, o + c, ss),
                    );
                }
            }
            (Some(fx), Some(fy)) => {
                let (tx, ty) = (Taps::new(&fx), Taps::new(&fy));
                let top = at - 3 * ss - 3;
                for r in 0..h + 7 {
                    let (o, t) = (top + r * ss, r * TS);
                    match W {
                        4 | 8 => s64(tmp, t, tx.h8(src, o)),
                        16 => s128(tmp, t, tx.h16(src, o)),
                        _ => {
                            let mut c = 0;
                            while c < W {
                                s256(tmp, t + c, tx.h32(src, o + c));
                                c += 32;
                            }
                        }
                    }
                }
                let tmp = &*tmp;
                for r in 0..h {
                    let o = r * TS;
                    put::<W>(
                        dst,
                        r * ds,
                        avg,
                        || ty.v8(tmp, o, TS),
                        || ty.v16(tmp, o, TS),
                        |c| ty.v32(tmp, o + c, TS),
                    );
                }
            }
        }
    }
}
