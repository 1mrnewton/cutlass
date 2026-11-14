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
//! Loads and stores go through fixed-size arrays, which the compiler turns
//! into single vector moves.

use core::arch::x86_64::{
    __m128i, __m256i, _mm_avg_epu8,
    _mm_cvtsi128_si64, _mm_extract_epi64, _mm_maddubs_epi16, _mm_set_epi64x, _mm_setr_epi8,
    _mm_unpackhi_epi8, _mm_unpacklo_epi8, _mm256_adds_epi16, _mm256_add_epi16,
    _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_maddubs_epi16, _mm256_unpackhi_epi8, _mm256_unpacklo_epi8, _mm256_mulhrs_epi16, _mm256_packus_epi16,
    _mm256_permute4x64_epi64, _mm256_set_m128i, _mm256_set1_epi16, _mm256_shuffle_epi8,
    _mm_add_epi16, _mm_adds_epi16, _mm_mulhrs_epi16, _mm_packus_epi16, _mm_set1_epi16,
    _mm_shuffle_epi8,
};

use super::Block;

// Loads and stores ---------------------------------------------------------

/// Taps `2k` and `2k + 1` as the byte pair `pmaddubsw` multiplies by.
#[inline]
fn pair(f: &[i16; 8], k: usize) -> i16 {
    i16::from_le_bytes([f[2 * k] as i8 as u8, f[2 * k + 1] as i8 as u8])
}

// Filters ------------------------------------------------------------------

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

    /// The sum of four pair products before rounding, in the order that
    /// keeps it exact.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn raw(&self, p: [__m256i; 4]) -> __m256i {
        let s = _mm256_add_epi16(p[0], p[3]);
        let s = _mm256_add_epi16(s, p[1]);
        _mm256_adds_epi16(s, p[2])
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn raw1(&self, p: [__m128i; 4]) -> __m128i {
        let s = _mm_add_epi16(p[0], p[3]);
        let s = _mm_add_epi16(s, p[1]);
        _mm_adds_epi16(s, p[2])
    }

    /// Sums of the four pair products, in the order that keeps them exact.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn sum(&self, p: [__m256i; 4]) -> __m256i {
        let s = _mm256_add_epi16(p[0], p[3]);
        let s = _mm256_add_epi16(s, p[1]);
        let s = _mm256_adds_epi16(s, p[2]);
        _mm256_mulhrs_epi16(s, _mm256_set1_epi16(256))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn sum1(&self, p: [__m128i; 4]) -> __m128i {
        let s = _mm_add_epi16(p[0], p[3]);
        let s = _mm_add_epi16(s, p[1]);
        let s = _mm_adds_epi16(s, p[2]);
        _mm_mulhrs_epi16(s, _mm_set1_epi16(256))
    }

}

/// Sixteen bytes of `a` from `at` (a constant after inlining, so the
/// bounds check folds away).
#[inline]
#[target_feature(enable = "avx2")]
fn la16<const R: usize>(a: &[u8; R], at: usize) -> __m128i {
    match a.get(at..).and_then(|t| t.first_chunk::<16>()) {
        Some(t) => {
            let lo = i64::from_le_bytes([t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]]);
            let hi = i64::from_le_bytes([t[8], t[9], t[10], t[11], t[12], t[13], t[14], t[15]]);
            _mm_set_epi64x(hi, lo)
        }
        None => {
            core::arch::x86_64::_mm_setzero_si128()
        }
    }
}

/// The first `n` (4, 8 or 16) bytes of `v` to `d[at..]` (constants).
#[inline]
#[target_feature(enable = "avx2")]
fn sa<const W: usize>(d: &mut [u8; W], at: usize, v: __m128i, n: usize) {
    let (lo, hi) = (_mm_cvtsi128_si64(v), _mm_extract_epi64::<1>(v));
    match n {
        16 => {
            if let Some(t) = d.get_mut(at..).and_then(|t| t.first_chunk_mut::<16>()) {
                t[..8].copy_from_slice(&lo.to_le_bytes());
                t[8..].copy_from_slice(&hi.to_le_bytes());
            }
        }
        8 => {
            if let Some(t) = d.get_mut(at..).and_then(|t| t.first_chunk_mut::<8>()) {
                *t = lo.to_le_bytes();
            }
        }
        _ => {
            if let Some(t) = d.get_mut(at..).and_then(|t| t.first_chunk_mut::<4>()) {
                let b = lo.to_le_bytes();
                *t = [b[0], b[1], b[2], b[3]];
            }
        }
    }
}

/// The first `n` bytes of `d` from `at` as a vector, for averaging.
#[inline]
#[target_feature(enable = "avx2")]
fn la<const W: usize>(d: &[u8; W], at: usize, n: usize) -> __m128i {
    let mut t = [0u8; 16];
    t[..n].copy_from_slice(&d[at..at + n]);
    la16(&t, 0)
}

impl Taps {
    /// Sixteen outputs across from `row[at..]`, whose sample 3 is the
    /// first output's.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn across16a<const R: usize>(&self, row: &[u8; R], at: usize) -> __m128i {
        let v = _mm256_set_m128i(la16(row, at + 8), la16(row, at));
        let p: [__m256i; 4] = core::array::from_fn(|k| {
            _mm256_maddubs_epi16(_mm256_shuffle_epi8(v, self.masks[k]), self.pairs[k])
        });
        let r = self.sum(p);
        let packed = _mm256_packus_epi16(r, r);
        _mm256_castsi256_si128(_mm256_permute4x64_epi64::<0b1000>(packed))
    }

    /// Eight outputs across (the low 8 bytes) from `row[at..]`.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn across8a<const R: usize>(&self, row: &[u8; R], at: usize) -> __m128i {
        let v = la16(row, at);
        let p: [__m128i; 4] = core::array::from_fn(|k| {
            _mm_maddubs_epi16(_mm_shuffle_epi8(v, self.masks1[k]), self.pairs1[k])
        });
        let r = self.sum1(p);
        _mm_packus_epi16(r, r)
    }
}

/// Every block of [`Block::fits`].
#[target_feature(enable = "avx2")]
pub(super) fn predict(b: Block<'_>) {
    match b.w {
        4 => sized::<4, 32>(b),
        8 => sized::<8, 32>(b),
        16 => sized::<16, 48>(b),
        32 => sized::<32, 64>(b),
        _ => sized::<64, 96>(b),
    }
}

/// One block of width `W`; `R` is the source samples read per row,
/// `W + 16` and more.
#[inline]
#[target_feature(enable = "avx2")]
fn sized<const W: usize, const R: usize>(b: Block<'_>) {
    let Block {
        dst,
        dst_stride: ds,
        src,
        at,
        src_stride: ss,
        w: _,
        h,
        fx,
        fy,
        avg,
        tmp,
    } = b;
    let step = W.min(16);
    let put = |dst: &mut [u8], r: usize, v: [__m128i; 4]| {
        if let Some(d) = dst.get_mut(r * ds..).and_then(|t| t.first_chunk_mut::<W>()) {
            let mut c = 0;
            while c < W {
                let x = v[c / 16];
                let x = if avg {
                    _mm_avg_epu8(x, la(d, c, step))
                } else {
                    x
                };
                sa(d, c, x, step);
                c += 16;
            }
        }
    };
    let zero = core::arch::x86_64::_mm_setzero_si128();
    match (fx, fy) {
        (None, None) if !avg => {
            for r in 0..h {
                let (Some(row), Some(d)) = (
                    row_of::<W>(src, at + r * ss),
                    dst.get_mut(r * ds..).and_then(|t| t.first_chunk_mut::<W>()),
                ) else {
                    return;
                };
                *d = *row;
            }
        }
        (None, None) => {
            for r in 0..h {
                let Some(row) = row_of::<R>(src, at + r * ss) else { return };
                let mut v = [zero; 4];
                let mut c = 0;
                while c < W {
                    v[c / 16] = la16(row, c);
                    c += 16;
                }
                put(dst, r, v);
            }
        }
        (Some(fx), None) => {
            let t = Taps::new(&fx);
            for r in 0..h {
                let Some(row) = row_of::<R>(src, at + r * ss - 3) else { return };
                let mut v = [zero; 4];
                let mut c = 0;
                while c < W {
                    v[c / 16] = if step == 16 { t.across16a(row, c) } else { t.across8a(row, c) };
                    c += 16;
                }
                put(dst, r, v);
            }
        }
        (None, Some(fy)) => {
            let t = Taps::new(&fy);
            down_sized::<W, R>(&t, dst, ds, src, at - 3 * ss, ss, h, avg);
        }
        (Some(fx), Some(fy)) => {
            let (tx, ty) = (Taps::new(&fx), Taps::new(&fy));
            for (r, out) in tmp.chunks_exact_mut(R).enumerate().take(h + 7) {
                let Some(out) = out.first_chunk_mut::<R>() else { return };
                let Some(row) = row_of::<R>(src, at + r * ss - 3 * ss - 3) else { return };
                let mut c = 0;
                while c < W {
                    let v = if step == 16 { tx.across16a(row, c) } else { tx.across8a(row, c) };
                    sa(out, c, v, 16);
                    c += 16;
                }
            }
            down_sized::<W, R>(&ty, dst, ds, tmp, 0, R, h, avg);
        }
    }
}

/// The filter down for width `W`, from `src[top..]`, 3 rows above the
/// block's first. Each output row reads its eight source rows afresh, as
/// fixed-size arrays.
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
fn down_sized<const W: usize, const R: usize>(t: &Taps, dst: &mut [u8], ds: usize, src: &[u8], top: usize, ss: usize, h: usize, avg: bool) {
    for r in 0..h {
        let base = top + r * ss;
        let mut rows: [&[u8; R]; 8] = [&[0u8; R]; 8];
        for (k, row) in rows.iter_mut().enumerate() {
            match src.get(base + k * ss..).and_then(|s| s.first_chunk::<R>()) {
                Some(s) => *row = s,
                None => return,
            }
        }
        let Some(d) = dst.get_mut(r * ds..).and_then(|t| t.first_chunk_mut::<W>()) else {
            return;
        };
        if W >= 32 {
            let mut c = 0;
            while c < W {
                let rv: [__m256i; 8] = core::array::from_fn(|k| _mm256_set_m128i(la16(rows[k], c + 16), la16(rows[k], c)));
                let lo: [__m256i; 4] = core::array::from_fn(|k| {
                    _mm256_maddubs_epi16(_mm256_unpacklo_epi8(rv[2 * k], rv[2 * k + 1]), t.pairs[k])
                });
                let hi: [__m256i; 4] = core::array::from_fn(|k| {
                    _mm256_maddubs_epi16(_mm256_unpackhi_epi8(rv[2 * k], rv[2 * k + 1]), t.pairs[k])
                });
                let r256 = _mm256_set1_epi16(256);
                let lo = _mm256_mulhrs_epi16(t.raw(lo), r256);
                let hi = _mm256_mulhrs_epi16(t.raw(hi), r256);
                let v = _mm256_packus_epi16(lo, hi);
                let (a, b) = (_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
                let (a, b) = if avg {
                    (_mm_avg_epu8(a, la(d, c, 16)), _mm_avg_epu8(b, la(d, c + 16, 16)))
                } else {
                    (a, b)
                };
                sa(d, c, a, 16);
                sa(d, c + 16, b, 16);
                c += 32;
            }
        } else {
            let rv: [__m128i; 8] = core::array::from_fn(|k| la16(rows[k], 0));
            let r128 = _mm_set1_epi16(256);
            let lo: [__m128i; 4] = core::array::from_fn(|k| {
                _mm_maddubs_epi16(_mm_unpacklo_epi8(rv[2 * k], rv[2 * k + 1]), t.pairs1[k])
            });
            let lo = _mm_mulhrs_epi16(t.raw1(lo), r128);
            let v = if W == 16 {
                let hi: [__m128i; 4] = core::array::from_fn(|k| {
                    _mm_maddubs_epi16(_mm_unpackhi_epi8(rv[2 * k], rv[2 * k + 1]), t.pairs1[k])
                });
                let hi = _mm_mulhrs_epi16(t.raw1(hi), r128);
                _mm_packus_epi16(lo, hi)
            } else {
                _mm_packus_epi16(lo, lo)
            };
            let v = if avg { _mm_avg_epu8(v, la(d, 0, W)) } else { v };
            sa(d, 0, v, W);
        }
    }
}

#[inline(always)]
fn row_of<const R: usize>(s: &[u8], i: usize) -> Option<&[u8; R]> {
    s.get(i..).and_then(|t| t.first_chunk::<R>())
}
