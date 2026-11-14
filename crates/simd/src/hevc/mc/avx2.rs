//! The 8-bit interpolation and default weighting on AVX2.
//!
//! The horizontal filter multiplies pairs of samples by pairs of taps
//! (`pmaddubsw`): a shuffle puts samples `j + 2k` and `j + 2k + 1` side by
//! side for each output `j`, and four such products (two for 4 taps) add
//! up to the sum. Every pair sum and the total fit 16 bits for 8-bit
//! samples: at most 88 · 255. The vertical filter of samples multiplies
//! 16-bit values by each tap. The vertical filter of intermediates
//! interleaves two rows and multiplies pairs of 16-bit values by pairs of
//! taps into 32-bit sums (`pmaddwd`).
//!
//! Loads and stores go through fixed-size arrays (`ld*`, `st*`), which the
//! compiler turns into single vector moves.

// The loops index rows, strips and taps by one position at once.
#![allow(clippy::needless_range_loop)]

use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_add_epi32, _mm_adds_epi16, _mm_cvtepu8_epi16,
    _mm_cvtsi32_si128, _mm_cvtsi128_si64, _mm_extract_epi64, _mm_madd_epi16, _mm_maddubs_epi16,
    _mm_mullo_epi16, _mm_packs_epi32, _mm_packus_epi16, _mm_set_epi64x, _mm_set1_epi16,
    _mm_setr_epi8, _mm_setzero_si128, _mm_shuffle_epi8, _mm_slli_epi16, _mm_srai_epi16,
    _mm_srai_epi32, _mm_unpackhi_epi16, _mm_unpacklo_epi16, _mm256_add_epi16, _mm256_add_epi32,
    _mm256_adds_epi16, _mm256_castsi128_si256, _mm256_castsi256_si128, _mm256_cvtepu8_epi16,
    _mm256_extract_epi64, _mm256_extracti128_si256, _mm256_madd_epi16, _mm256_maddubs_epi16,
    _mm256_mullo_epi16, _mm256_packs_epi32, _mm256_set_m128i, _mm256_set1_epi16, _mm256_set1_epi32,
    _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_slli_epi16, _mm256_sra_epi32,
    _mm256_srai_epi16, _mm256_srai_epi32, _mm256_unpackhi_epi16, _mm256_unpacklo_epi16,
};

use super::{Src, Taps};

// Loads and stores ---------------------------------------------------------

/// Four 16-bit values as one 64-bit lane.
#[inline]
fn q(a: &[i16]) -> i64 {
    let mut v = 0u64;
    for (k, &x) in a.iter().take(4).enumerate() {
        v |= u64::from(x as u16) << (16 * k);
    }
    v as i64
}

/// Eight 16-bit values from `at`, zeros past the end of `s`.
#[inline]
#[target_feature(enable = "avx2")]
fn ld16x8(s: &[i16], at: usize) -> __m128i {
    match s.get(at..).and_then(|t| t.first_chunk::<8>()) {
        Some(a) => _mm_set_epi64x(q(&a[4..]), q(&a[..4])),
        None => {
            let mut a = [0i16; 8];
            if let Some(t) = s.get(at..) {
                let n = t.len().min(8);
                a[..n].copy_from_slice(&t[..n]);
            }
            _mm_set_epi64x(q(&a[4..]), q(&a[..4]))
        }
    }
}

/// Sixteen 16-bit values from `at`, zeros past the end of `s`.
#[inline]
#[target_feature(enable = "avx2")]
fn ld16x16(s: &[i16], at: usize) -> __m256i {
    _mm256_set_m128i(ld16x8(s, at + 8), ld16x8(s, at))
}

/// The four 16-bit values of a 64-bit lane.
#[inline]
fn unq(v: i64) -> [i16; 4] {
    let v = v as u64;
    [
        v as i16,
        (v >> 16) as i16,
        (v >> 32) as i16,
        (v >> 48) as i16,
    ]
}

/// The first `n` (4 or 8) values of `v` to `d[at..]`.
#[inline]
#[target_feature(enable = "avx2")]
fn st16x8(d: &mut [i16], at: usize, v: __m128i, n: usize) {
    let mut a = [0i16; 8];
    a[..4].copy_from_slice(&unq(_mm_cvtsi128_si64(v)));
    a[4..].copy_from_slice(&unq(_mm_extract_epi64::<1>(v)));
    let d = d.get_mut(at..).unwrap_or_default();
    if n == 8 {
        if let Some(d) = d.first_chunk_mut::<8>() {
            *d = a;
        }
    } else if let (Some(d), Some(a)) = (d.first_chunk_mut::<4>(), a.first_chunk::<4>()) {
        *d = *a;
    }
}

/// Sixteen values of `v` to `d[at..]`.
#[inline]
#[target_feature(enable = "avx2")]
fn st16x16(d: &mut [i16], at: usize, v: __m256i) {
    let mut a = [0i16; 16];
    a[..4].copy_from_slice(&unq(_mm256_extract_epi64::<0>(v)));
    a[4..8].copy_from_slice(&unq(_mm256_extract_epi64::<1>(v)));
    a[8..12].copy_from_slice(&unq(_mm256_extract_epi64::<2>(v)));
    a[12..].copy_from_slice(&unq(_mm256_extract_epi64::<3>(v)));
    if let Some(d) = d.get_mut(at..).and_then(|d| d.first_chunk_mut::<16>()) {
        *d = a;
    }
}

/// The first `n` (4, 8 or 16) bytes of `v` to `d[at..]`.
#[inline]
#[target_feature(enable = "avx2")]
fn st8(d: &mut [u8], at: usize, v: __m128i, n: usize) {
    let lo = (_mm_cvtsi128_si64(v) as u64).to_le_bytes();
    let hi = (_mm_extract_epi64::<1>(v) as u64).to_le_bytes();
    let mut a = [0u8; 16];
    a[..8].copy_from_slice(&lo);
    a[8..].copy_from_slice(&hi);
    let d = d.get_mut(at..).unwrap_or_default();
    match n {
        16 => {
            if let Some(d) = d.first_chunk_mut::<16>() {
                *d = a;
            }
        }
        8 => {
            if let (Some(d), Some(a)) = (d.first_chunk_mut::<8>(), a.first_chunk::<8>()) {
                *d = *a;
            }
        }
        _ => {
            if let (Some(d), Some(a)) = (d.first_chunk_mut::<4>(), a.first_chunk::<4>()) {
                *d = *a;
            }
        }
    }
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

/// Taps `2k` and `2k + 1` as the byte pair `pmaddubsw` multiplies by.
#[inline]
fn tap_pair(f: &[i8; 8], k: usize) -> i16 {
    i16::from_le_bytes([f[2 * k] as u8, f[2 * k + 1] as u8])
}

/// Taps `2k` and `2k + 1` as the 16-bit pair `pmaddwd` multiplies by.
#[inline]
fn tap_pair32(f: &[i8; 8], k: usize) -> i32 {
    i32::from(f[2 * k] as i16 as u16) | i32::from(f[2 * k + 1] as i16) << 16
}

/// The horizontal filter of one row, `T` taps.
#[derive(Clone, Copy)]
struct Across<const T: usize> {
    masks: [__m256i; 4],
    pairs: [__m256i; 4],
}

impl<const T: usize> Across<T> {
    #[inline]
    #[target_feature(enable = "avx2")]
    fn new(f: &[i8; 8]) -> Self {
        let m = [pair_mask(0), pair_mask(1), pair_mask(2), pair_mask(3)];
        Self {
            masks: m.map(|m| _mm256_set_m128i(m, m)),
            pairs: core::array::from_fn(|k| _mm256_set1_epi16(tap_pair(f, k))),
        }
    }

    /// Eight outputs from the sixteen samples `v`.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn eight(&self, v: __m128i) -> __m128i {
        let m = |k: usize| _mm256_castsi256_si128(self.masks[k]);
        let p = |k: usize| _mm256_castsi256_si128(self.pairs[k]);
        let mut sum = _mm_maddubs_epi16(_mm_shuffle_epi8(v, m(0)), p(0));
        for k in 1..T / 2 {
            sum = _mm_add_epi16(sum, _mm_maddubs_epi16(_mm_shuffle_epi8(v, m(k)), p(k)));
        }
        sum
    }

    /// Sixteen outputs from the sixteen samples `a` and the sixteen `b`
    /// eight further on.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn sixteen(&self, a: __m128i, b: __m128i) -> __m256i {
        let v = _mm256_set_m128i(b, a);
        let mut sum = _mm256_maddubs_epi16(_mm256_shuffle_epi8(v, self.masks[0]), self.pairs[0]);
        for k in 1..T / 2 {
            let p = _mm256_maddubs_epi16(_mm256_shuffle_epi8(v, self.masks[k]), self.pairs[k]);
            sum = _mm256_add_epi16(sum, p);
        }
        sum
    }
}

/// Where the filtered values of a block go: `M` 0 keeps them as 14-bit
/// intermediates in `mid`; 1 rounds them to samples in `px`; 2 averages
/// them with the intermediates of the other list in `other` into `px`;
/// 3 and 4 weigh them explicitly, alone or with `other`.
struct Out<'a> {
    mid: &'a mut [i16],
    px: &'a mut [u8],
    other: &'a [i16],
    stride: usize,
    other_stride: usize,
    /// Explicit weighting (modes 3 and 4): the 16-bit pair each pair of
    /// values is multiplied by, the 32-bit term added, and the shift.
    pair: __m256i,
    add: __m256i,
    shift: __m128i,
}

/// One output row of a block `W` wide, as arrays of its width: stores at
/// fixed offsets need no checks. Only the fields of the output kind are
/// filled.
struct Row<'r, const W: usize> {
    mid: Option<&'r mut [i16; W]>,
    px: Option<&'r mut [u8; W]>,
    other: Option<&'r [i16; W]>,
    pair: __m256i,
    add: __m256i,
    shift: __m128i,
}

impl Out<'_> {
    /// A sub-block `x` columns in.
    fn at(&mut self, x: usize) -> Out<'_> {
        Out {
            mid: self.mid.get_mut(x..).unwrap_or_default(),
            px: self.px.get_mut(x..).unwrap_or_default(),
            other: self.other.get(x..).unwrap_or_default(),
            stride: self.stride,
            other_stride: self.other_stride,
            pair: self.pair,
            add: self.add,
            shift: self.shift,
        }
    }

    /// Output row `j`, `None` when a buffer of output kind `M` is too
    /// short for it (the entry checks that it never is).
    #[inline(always)]
    fn row<const W: usize, const M: u8>(&mut self, j: usize) -> Option<Row<'_, W>> {
        let at = j * self.stride;
        let mut row = Row {
            mid: None,
            px: None,
            other: None,
            pair: self.pair,
            add: self.add,
            shift: self.shift,
        };
        if M == 0 {
            row.mid = Some(self.mid.get_mut(at..)?.first_chunk_mut::<W>()?);
        } else {
            row.px = Some(self.px.get_mut(at..)?.first_chunk_mut::<W>()?);
        }
        if M == 2 || M == 4 {
            row.other = Some(
                self.other
                    .get(j * self.other_stride..)?
                    .first_chunk::<W>()?,
            );
        }
        Some(row)
    }
}

impl<const W: usize> Row<'_, W> {
    /// Explicit weighting of sixteen values `v` (mode 3) or of the other
    /// list's `o` with `v` (mode 4), to sixteen 16-bit sums.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn weigh16<const M: u8>(&self, v: __m256i, o: __m256i) -> __m256i {
        let (a, b) = if M == 3 {
            (v, _mm256_set1_epi16(1))
        } else {
            (o, v)
        };
        let lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), self.pair);
        let hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), self.pair);
        let (lo, hi) = if M == 3 {
            (
                _mm256_add_epi32(_mm256_sra_epi32(lo, self.shift), self.add),
                _mm256_add_epi32(_mm256_sra_epi32(hi, self.shift), self.add),
            )
        } else {
            (
                _mm256_sra_epi32(_mm256_add_epi32(lo, self.add), self.shift),
                _mm256_sra_epi32(_mm256_add_epi32(hi, self.add), self.shift),
            )
        };
        _mm256_packs_epi32(lo, hi)
    }

    /// The other list's sixteen values from column `x`, or `v` without
    /// them.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn other16(&self, x: usize, v: __m256i) -> __m256i {
        if let Some(o) = self.other
            && let Some(o) = o.get(x..x + 16)
        {
            return ld16x16(o, 0);
        }
        v
    }

    /// Sixteen values from column `x` (a multiple of 16 below `W`).
    #[inline]
    #[target_feature(enable = "avx2")]
    fn v16<const M: u8>(&mut self, x: usize, v: __m256i) {
        if M == 0 {
            if let Some(d) = &mut self.mid
                && let Some(d) = d.get_mut(x..x + 16)
            {
                st16x16(d, 0, v);
            }
            return;
        }
        let v = match M {
            1 => _mm256_srai_epi16::<6>(_mm256_add_epi16(v, _mm256_set1_epi16(32))),
            2 => {
                let o = self.other16(x, v);
                let s = _mm256_adds_epi16(_mm256_adds_epi16(v, o), _mm256_set1_epi16(64));
                _mm256_srai_epi16::<7>(s)
            }
            _ => {
                let o = if M == 4 { self.other16(x, v) } else { v };
                self.weigh16::<M>(v, o)
            }
        };
        let p = _mm_packus_epi16(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
        if let Some(d) = &mut self.px
            && let Some(d) = d.get_mut(x..x + 16)
        {
            st8(d, 0, p, 16);
        }
    }

    /// The `W` (8 or 4) values of a narrow block.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn v8<const M: u8>(&mut self, v: __m128i) {
        if M == 0 {
            if let Some(d) = &mut self.mid {
                st16x8(&mut d[..], 0, v, W);
            }
            return;
        }
        let o = match self.other {
            Some(o) if M == 2 || M == 4 => ld16x8(&o[..], 0),
            _ => v,
        };
        let p = match M {
            1 => {
                let v = _mm_srai_epi16::<6>(_mm_add_epi16(v, _mm_set1_epi16(32)));
                _mm_packus_epi16(v, v)
            }
            2 => {
                let v =
                    _mm_srai_epi16::<7>(_mm_adds_epi16(_mm_adds_epi16(v, o), _mm_set1_epi16(64)));
                _mm_packus_epi16(v, v)
            }
            _ => {
                let w = self.weigh16::<M>(_mm256_castsi128_si256(v), _mm256_castsi128_si256(o));
                let w = _mm256_castsi256_si128(w);
                _mm_packus_epi16(w, w)
            }
        };
        if let Some(d) = &mut self.px {
            st8(&mut d[..], 0, p, W);
        }
    }
}

/// Rows of horizontally filtered values kept between the passes: up to
/// four strips of sixteen for each row of a block's window.
pub(super) type Strips = [__m256i; 4];

/// The filtered block (see [`super::predict`]).
#[target_feature(enable = "avx2")]
pub(super) fn predict(
    dst: super::Dst<'_>,
    stride: usize,
    src: &Src<'_>,
    (w, h): (usize, usize),
    taps: &Taps,
    tmp: &mut Vec<Strips>,
) {
    prefetch(src, (w, h), taps.len);
    if tmp.len() < 64 + 7 {
        tmp.resize(64 + 7, [_mm256_setzero_si256(); 4]);
    }
    let tmp = tmp.as_mut_slice();
    let zero = _mm256_setzero_si256();
    let plain = |mid, px, other, other_stride| Out {
        mid,
        px,
        other,
        stride,
        other_stride,
        pair: zero,
        add: zero,
        shift: _mm_setzero_si128(),
    };
    let (mode, mut out) = match dst {
        super::Dst::Mid(mid) => (0, plain(mid, &mut [], &[], 0)),
        super::Dst::Uni(px) => (1, plain(&mut [], px, &[], 0)),
        super::Dst::Bi(px, other) => (2, plain(&mut [], px, other, w)),
        super::Dst::UniW(px, wt) => {
            let round = 1i32 << (wt.log2wd - 1);
            let mut out = plain(&mut [], px, &[], 0);
            out.pair = _mm256_set1_epi32((round << 16) | i32::from(wt.w0 as u16));
            out.add = _mm256_set1_epi32(wt.o0);
            out.shift = _mm_cvtsi32_si128(wt.log2wd as i32);
            (3, out)
        }
        super::Dst::BiW(px, other, wt) => {
            let mut out = plain(&mut [], px, other, w);
            out.pair = _mm256_set1_epi32((i32::from(wt.w1) << 16) | i32::from(wt.w0 as u16));
            out.add = _mm256_set1_epi32((wt.o0 + wt.o1 + 1) << wt.log2wd);
            out.shift = _mm_cvtsi32_si128(wt.log2wd as i32 + 1);
            (4, out)
        }
    };
    // A width of three quarters of a power of two is two blocks.
    let split = |a: usize| Src {
        at: src.at + a,
        ..*src
    };
    // `R`: the samples of each source row the loads of a block `W` wide
    // read: `W + 8`, and 24 for narrow blocks, whose one load starts up to
    // three samples in.
    macro_rules! go {
        ($w:literal, $r:literal, $t:literal, $x:expr) => {
            match mode {
                0 => put_w::<$w, $r, $t, 0>(&mut out.at($x), &split($x), h, taps, tmp),
                1 => put_w::<$w, $r, $t, 1>(&mut out.at($x), &split($x), h, taps, tmp),
                2 => put_w::<$w, $r, $t, 2>(&mut out.at($x), &split($x), h, taps, tmp),
                3 => put_w::<$w, $r, $t, 3>(&mut out.at($x), &split($x), h, taps, tmp),
                _ => put_w::<$w, $r, $t, 4>(&mut out.at($x), &split($x), h, taps, tmp),
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
                32 => go!(32, 40, $t, 0),
                48 => {
                    go!(32, 40, $t, 0);
                    go!(16, 24, $t, 32);
                }
                _ => go!(64, 72, $t, 0),
            }
        };
    }
    if taps.len == 8 {
        widths!(8);
    } else {
        widths!(4);
    }
}

/// Asks for every line of the window of a `w`×`h` block with `t` taps
/// to be brought into cache: the kernels then find rows of a reference
/// read from memory already on their way.
#[inline]
#[target_feature(enable = "avx2")]
fn prefetch(src: &Src<'_>, (w, h): (usize, usize), t: usize) {
    use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
    for j in 0..h + t - 1 {
        if let Some(row) = src.src.get(src.row(j)..) {
            let p = row.as_ptr().cast::<i8>();
            _mm_prefetch::<_MM_HINT_T0>(p);
            _mm_prefetch::<_MM_HINT_T0>(p.wrapping_add(w + t - 2));
        }
    }
}

/// Source row `j` of the window: its first `R` samples, which the entry
/// checked are there.
#[inline(always)]
fn src_row<'s, const R: usize>(src: &Src<'s>, j: usize) -> Option<&'s [u8; R]> {
    src.src.get(src.row(j)..)?.first_chunk::<R>()
}

/// [`predict`] of a block `W` wide (4, 8, 16, 32 or 64) with `T` taps
/// into output kind `M`; `R` is `W + 8`, or 24 below 16 wide. Rows of the window are
/// read and rows of the output written as arrays of fixed size, so the
/// loads and stores inside a row need no checks; the horizontal pass of
/// each window row is kept in `tmp`, row after row, for the vertical one.
#[inline]
#[target_feature(enable = "avx2")]
fn put_w<const W: usize, const R: usize, const T: usize, const M: u8>(
    out: &mut Out<'_>,
    src: &Src<'_>,
    h: usize,
    taps: &Taps,
    tmp: &mut [Strips],
) {
    let h0 = T / 2 - 1;
    // At most 64 rows (the entry checks): `tmp` holds their window.
    let h = h.min(64);
    let strips = W / 16;
    // Sixteen (or, narrower, eight) values of row `r` from column `x`.
    let load = |r: &[u8; R], x: usize| -> __m128i {
        match r.get(x..x + 16).and_then(|a| a.first_chunk::<16>()) {
            Some(a) => {
                let v = u128::from_le_bytes(*a);
                _mm_set_epi64x((v >> 64) as i64, v as i64)
            }
            None => _mm_setzero_si128(),
        }
    };
    match (taps.across, taps.down) {
        (None, None) => {
            for j in 0..h {
                let (Some(r), Some(mut o)) = (src_row::<R>(src, j + h0), out.row::<W, M>(j)) else {
                    return;
                };
                if W >= 16 {
                    for s in 0..strips {
                        let v = _mm256_cvtepu8_epi16(load(r, h0 + 16 * s));
                        o.v16::<M>(16 * s, _mm256_slli_epi16::<6>(v));
                    }
                } else {
                    o.v8::<M>(_mm_slli_epi16::<6>(_mm_cvtepu8_epi16(load(r, h0))));
                }
            }
        }
        (Some(fh), None) => {
            let across = Across::<T>::new(&fh);
            for j in 0..h {
                let (Some(r), Some(mut o)) = (src_row::<R>(src, j + h0), out.row::<W, M>(j)) else {
                    return;
                };
                if W >= 16 {
                    for s in 0..strips {
                        let x = 16 * s;
                        o.v16::<M>(x, across.sixteen(load(r, x), load(r, x + 8)));
                    }
                } else {
                    o.v8::<M>(across.eight(load(r, 0)));
                }
            }
        }
        (None, Some(fv)) => {
            // Rows of samples as 16-bit values, then the taps down.
            let c: [__m256i; 8] = core::array::from_fn(|k| _mm256_set1_epi16(i16::from(fv[k])));
            let Some(rows) = tmp.get_mut(..h + T - 1) else {
                return;
            };
            for (j, t) in rows.iter_mut().enumerate() {
                let Some(r) = src_row::<R>(src, j) else {
                    return;
                };
                if W >= 16 {
                    for s in 0..strips {
                        t[s] = _mm256_cvtepu8_epi16(load(r, h0 + 16 * s));
                    }
                } else {
                    t[0] = _mm256_castsi128_si256(_mm_cvtepu8_epi16(load(r, h0)));
                }
            }
            for j in 0..h {
                let (Some(win), Some(mut o)) = (
                    rows.get(j..).and_then(|w| w.first_chunk::<T>()),
                    out.row::<W, M>(j),
                ) else {
                    return;
                };
                if W >= 16 {
                    for s in 0..strips {
                        let mut sum = _mm256_mullo_epi16(win[0][s], c[0]);
                        for k in 1..T {
                            sum = _mm256_add_epi16(sum, _mm256_mullo_epi16(win[k][s], c[k]));
                        }
                        o.v16::<M>(16 * s, sum);
                    }
                } else {
                    let lo = |v: __m256i| _mm256_castsi256_si128(v);
                    let mut sum = _mm_mullo_epi16(lo(win[0][0]), lo(c[0]));
                    for k in 1..T {
                        sum = _mm_add_epi16(sum, _mm_mullo_epi16(lo(win[k][0]), lo(c[k])));
                    }
                    o.v8::<M>(sum);
                }
            }
        }
        (Some(fh), Some(fv)) => {
            // The horizontal pass of each window row once; the vertical
            // pass multiplies pairs of rows by pairs of taps into 32-bit
            // sums.
            let across = Across::<T>::new(&fh);
            let p: [__m256i; 4] = core::array::from_fn(|k| _mm256_set1_epi32(tap_pair32(&fv, k)));
            let Some(rows) = tmp.get_mut(..h + T - 1) else {
                return;
            };
            for (j, t) in rows.iter_mut().enumerate() {
                let Some(r) = src_row::<R>(src, j) else {
                    return;
                };
                if W >= 16 {
                    for s in 0..strips {
                        let x = 16 * s;
                        t[s] = across.sixteen(load(r, x), load(r, x + 8));
                    }
                } else {
                    t[0] = _mm256_castsi128_si256(across.eight(load(r, 0)));
                }
            }
            for j in 0..h {
                let (Some(win), Some(mut o)) = (
                    rows.get(j..).and_then(|w| w.first_chunk::<T>()),
                    out.row::<W, M>(j),
                ) else {
                    return;
                };
                if W >= 16 {
                    for s in 0..strips {
                        let (mut lo, mut hi) = (_mm256_setzero_si256(), _mm256_setzero_si256());
                        for k in 0..T / 2 {
                            let (a, b) = (win[2 * k][s], win[2 * k + 1][s]);
                            lo = _mm256_add_epi32(
                                lo,
                                _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), p[k]),
                            );
                            hi = _mm256_add_epi32(
                                hi,
                                _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), p[k]),
                            );
                        }
                        let v = _mm256_packs_epi32(
                            _mm256_srai_epi32::<6>(lo),
                            _mm256_srai_epi32::<6>(hi),
                        );
                        o.v16::<M>(16 * s, v);
                    }
                } else {
                    let c = |v: __m256i| _mm256_castsi256_si128(v);
                    let (mut lo, mut hi) = (_mm_setzero_si128(), _mm_setzero_si128());
                    for k in 0..T / 2 {
                        let (a, b) = (c(win[2 * k][0]), c(win[2 * k + 1][0]));
                        lo = _mm_add_epi32(lo, _mm_madd_epi16(_mm_unpacklo_epi16(a, b), c(p[k])));
                        hi = _mm_add_epi32(hi, _mm_madd_epi16(_mm_unpackhi_epi16(a, b), c(p[k])));
                    }
                    o.v8::<M>(_mm_packs_epi32(
                        _mm_srai_epi32::<6>(lo),
                        _mm_srai_epi32::<6>(hi),
                    ));
                }
            }
        }
    }
}

/// `Clip1((x + 32) >> 6)` of a block (see [`super::uni`]).
#[target_feature(enable = "avx2")]
pub(super) fn uni(
    dst: &mut [u8],
    dst_stride: usize,
    a: &[i16],
    a_stride: usize,
    (w, h): (usize, usize),
) {
    let add = _mm_set1_epi16(32);
    let one = |at: usize| _mm_srai_epi16::<6>(_mm_add_epi16(ld16x8(a, at), add));
    for j in 0..h {
        let (o, i) = (j * dst_stride, j * a_stride);
        let mut x = 0;
        while x < w {
            let n = match w - x {
                16.. => 16,
                8..16 => 8,
                _ => 4,
            };
            let lo = one(i + x);
            let hi = if n > 8 { one(i + x + 8) } else { lo };
            st8(dst, o + x, _mm_packus_epi16(lo, hi), n);
            x += n;
        }
    }
}

/// `Clip1((a + b + 64) >> 7)` of two blocks (see [`super::bi`]). The sums
/// saturate at 16 bits: a sum past either end clips to the same sample.
#[target_feature(enable = "avx2")]
pub(super) fn bi(
    dst: &mut [u8],
    dst_stride: usize,
    (a, b): (&[i16], &[i16]),
    ab_stride: usize,
    (w, h): (usize, usize),
) {
    let add = _mm_set1_epi16(64);
    let one = |at: usize| {
        let s = _mm_adds_epi16(ld16x8(a, at), ld16x8(b, at));
        _mm_srai_epi16::<7>(_mm_adds_epi16(s, add))
    };
    for j in 0..h {
        let (o, i) = (j * dst_stride, j * ab_stride);
        let mut x = 0;
        while x < w {
            let n = match w - x {
                16.. => 16,
                8..16 => 8,
                _ => 4,
            };
            let lo = one(i + x);
            let hi = if n > 8 { one(i + x + 8) } else { lo };
            st8(dst, o + x, _mm_packus_epi16(lo, hi), n);
            x += n;
        }
    }
}
