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

use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_add_epi32, _mm_adds_epi16, _mm_cvtepu8_epi16,
    _mm_cvtsi128_si64, _mm_extract_epi64, _mm_madd_epi16, _mm_maddubs_epi16, _mm_mullo_epi16,
    _mm_packs_epi32, _mm_packus_epi16, _mm_set_epi64x, _mm_set1_epi16, _mm_setr_epi8,
    _mm_setzero_si128, _mm_shuffle_epi8, _mm_slli_epi16, _mm_srai_epi16, _mm_srai_epi32,
    _mm_unpackhi_epi16, _mm_unpacklo_epi16, _mm256_add_epi16, _mm256_add_epi32, _mm256_adds_epi16,
    _mm256_castsi256_si128, _mm256_cvtepu8_epi16, _mm256_extract_epi64, _mm256_extracti128_si256,
    _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_mullo_epi16, _mm256_packs_epi32,
    _mm256_set_m128i, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setzero_si256,
    _mm256_shuffle_epi8, _mm256_slli_epi16, _mm256_srai_epi16, _mm256_srai_epi32,
    _mm256_unpackhi_epi16, _mm256_unpacklo_epi16,
};

use super::{Src, Taps};

// Loads and stores ---------------------------------------------------------

/// Sixteen bytes from `at`, zeros where `s` ends first.
#[inline]
#[target_feature(enable = "avx2")]
fn ld8x16(s: &[u8], at: usize) -> __m128i {
    match s.get(at..).and_then(|t| t.first_chunk::<16>()) {
        Some(a) => {
            let v = u128::from_le_bytes(*a);
            _mm_set_epi64x((v >> 64) as i64, v as i64)
        }
        None => {
            let mut a = [0u8; 16];
            if let Some(t) = s.get(at..) {
                let n = t.len().min(16);
                a[..n].copy_from_slice(&t[..n]);
            }
            let v = u128::from_le_bytes(a);
            _mm_set_epi64x((v >> 64) as i64, v as i64)
        }
    }
}

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

    /// Eight outputs from the sixteen samples at `at`.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn eight(&self, s: &[u8], at: usize) -> __m128i {
        let v = ld8x16(s, at);
        let m = |k: usize| _mm256_castsi256_si128(self.masks[k]);
        let p = |k: usize| _mm256_castsi256_si128(self.pairs[k]);
        let mut sum = _mm_maddubs_epi16(_mm_shuffle_epi8(v, m(0)), p(0));
        for k in 1..T / 2 {
            sum = _mm_add_epi16(sum, _mm_maddubs_epi16(_mm_shuffle_epi8(v, m(k)), p(k)));
        }
        sum
    }

    /// Sixteen outputs from the 24 samples at `at`.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn sixteen(&self, s: &[u8], at: usize) -> __m256i {
        let v = _mm256_set_m128i(ld8x16(s, at + 8), ld8x16(s, at));
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
/// them with the intermediates of the other list in `other` into `px`.
struct Out<'a> {
    mid: &'a mut [i16],
    px: &'a mut [u8],
    other: &'a [i16],
    stride: usize,
    other_stride: usize,
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
        }
    }

    /// Sixteen values of row `j` from column `x`.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn v16<const M: u8>(&mut self, j: usize, x: usize, v: __m256i) {
        let at = j * self.stride + x;
        match M {
            0 => st16x16(self.mid, at, v),
            1 => {
                let v = _mm256_srai_epi16::<6>(_mm256_add_epi16(v, _mm256_set1_epi16(32)));
                let p =
                    _mm_packus_epi16(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
                st8(self.px, at, p, 16);
            }
            _ => {
                let o = ld16x16(self.other, j * self.other_stride + x);
                let s = _mm256_adds_epi16(_mm256_adds_epi16(v, o), _mm256_set1_epi16(64));
                let v = _mm256_srai_epi16::<7>(s);
                let p =
                    _mm_packus_epi16(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
                st8(self.px, at, p, 16);
            }
        }
    }

    /// The first `n` (8 or 4) values of row `j` from column `x`.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn v8<const M: u8>(&mut self, j: usize, x: usize, v: __m128i, n: usize) {
        let at = j * self.stride + x;
        match M {
            0 => st16x8(self.mid, at, v, n),
            1 => {
                let v = _mm_srai_epi16::<6>(_mm_add_epi16(v, _mm_set1_epi16(32)));
                st8(self.px, at, _mm_packus_epi16(v, v), n);
            }
            _ => {
                let o = ld16x8(self.other, j * self.other_stride + x);
                let v =
                    _mm_srai_epi16::<7>(_mm_adds_epi16(_mm_adds_epi16(v, o), _mm_set1_epi16(64)));
                st8(self.px, at, _mm_packus_epi16(v, v), n);
            }
        }
    }
}

/// The filtered block (see [`super::predict`]).
#[target_feature(enable = "avx2")]
pub(super) fn predict(
    dst: super::Dst<'_>,
    stride: usize,
    src: &Src<'_>,
    (w, h): (usize, usize),
    taps: &Taps,
    tmp: &mut [i16],
) {
    let (mode, mut out) = match dst {
        super::Dst::Mid(mid) => (
            0,
            Out {
                mid,
                px: &mut [],
                other: &[],
                stride,
                other_stride: 0,
            },
        ),
        super::Dst::Uni(px) => (
            1,
            Out {
                mid: &mut [],
                px,
                other: &[],
                stride,
                other_stride: 0,
            },
        ),
        super::Dst::Bi(px, other) => (
            2,
            Out {
                mid: &mut [],
                px,
                other,
                stride,
                other_stride: w,
            },
        ),
    };
    // A width of three quarters of a power of two is two blocks.
    let split = |a: usize| Src {
        src: src.src,
        at: src.at + a,
        stride: src.stride,
    };
    macro_rules! go {
        ($w:literal, $t:literal, $x:expr) => {
            match mode {
                0 => put_w::<$w, $t, 0>(&mut out.at($x), &split($x), h, taps, tmp),
                1 => put_w::<$w, $t, 1>(&mut out.at($x), &split($x), h, taps, tmp),
                _ => put_w::<$w, $t, 2>(&mut out.at($x), &split($x), h, taps, tmp),
            }
        };
    }
    macro_rules! widths {
        ($t:literal) => {
            match w {
                4 => go!(4, $t, 0),
                8 => go!(8, $t, 0),
                12 => {
                    go!(8, $t, 0);
                    go!(4, $t, 8);
                }
                16 => go!(16, $t, 0),
                24 => {
                    go!(16, $t, 0);
                    go!(8, $t, 16);
                }
                32 => go!(32, $t, 0),
                48 => {
                    go!(32, $t, 0);
                    go!(16, $t, 32);
                }
                _ => go!(64, $t, 0),
            }
        };
    }
    if taps.len == 8 {
        widths!(8);
    } else {
        widths!(4);
    }
}

/// [`predict`] of a block `W` wide with `T` taps into output kind `M`.
#[inline]
#[target_feature(enable = "avx2")]
fn put_w<const W: usize, const T: usize, const M: u8>(
    out: &mut Out<'_>,
    src: &Src<'_>,
    h: usize,
    taps: &Taps,
    tmp: &mut [i16],
) {
    let h0 = T / 2 - 1;
    let (s, stride) = (src.src, src.stride);
    let row = |j: usize| src.at + j * stride;
    match (taps.across, taps.down) {
        (None, None) => {
            for j in 0..h {
                let at = row(j + h0) + h0;
                if W >= 16 {
                    for x in (0..W).step_by(16) {
                        let v = _mm256_slli_epi16::<6>(_mm256_cvtepu8_epi16(ld8x16(s, at + x)));
                        out.v16::<M>(j, x, v);
                    }
                } else {
                    let v = _mm_slli_epi16::<6>(_mm_cvtepu8_epi16(ld8x16(s, at)));
                    out.v8::<M>(j, 0, v, W);
                }
            }
        }
        (Some(fh), None) => {
            let across = Across::<T>::new(&fh);
            for j in 0..h {
                let at = row(j + h0);
                if W >= 16 {
                    for x in (0..W).step_by(16) {
                        out.v16::<M>(j, x, across.sixteen(s, at + x));
                    }
                } else {
                    out.v8::<M>(j, 0, across.eight(s, at), W);
                }
            }
        }
        (None, Some(fv)) => {
            // Rows of samples as 16-bit values, a window of `T` sliding
            // down: one new row for each row out.
            let c: [__m256i; 8] = core::array::from_fn(|k| _mm256_set1_epi16(i16::from(fv[k])));
            if W >= 16 {
                for x in (0..W).step_by(16) {
                    let load = |j: usize| _mm256_cvtepu8_epi16(ld8x16(s, row(j) + h0 + x));
                    let mut win = [_mm256_setzero_si256(); 8];
                    for (k, w) in win.iter_mut().enumerate().take(T - 1) {
                        *w = load(k);
                    }
                    for j in 0..h {
                        win[T - 1] = load(j + T - 1);
                        let mut sum = _mm256_mullo_epi16(win[0], c[0]);
                        for k in 1..T {
                            sum = _mm256_add_epi16(sum, _mm256_mullo_epi16(win[k], c[k]));
                        }
                        out.v16::<M>(j, x, sum);
                        for k in 0..T - 1 {
                            win[k] = win[k + 1];
                        }
                    }
                }
            } else {
                let c: [__m128i; 8] = c.map(|c| _mm256_castsi256_si128(c));
                let load = |j: usize| _mm_cvtepu8_epi16(ld8x16(s, row(j) + h0));
                let mut win = [_mm_setzero_si128(); 8];
                for (k, w) in win.iter_mut().enumerate().take(T - 1) {
                    *w = load(k);
                }
                for j in 0..h {
                    win[T - 1] = load(j + T - 1);
                    let mut sum = _mm_mullo_epi16(win[0], c[0]);
                    for k in 1..T {
                        sum = _mm_add_epi16(sum, _mm_mullo_epi16(win[k], c[k]));
                    }
                    out.v8::<M>(j, 0, sum, W);
                    for k in 0..T - 1 {
                        win[k] = win[k + 1];
                    }
                }
            }
        }
        (Some(fh), Some(fv)) => {
            // The horizontal pass of each source row once, into a window of
            // `T` rows sliding down; the vertical pass multiplies pairs of
            // window rows by pairs of taps into 32-bit sums.
            let _ = tmp;
            let across = Across::<T>::new(&fh);
            let p: [__m256i; 4] = core::array::from_fn(|k| _mm256_set1_epi32(tap_pair32(&fv, k)));
            if W >= 16 {
                for x in (0..W).step_by(16) {
                    let load = |j: usize| across.sixteen(s, row(j) + x);
                    let mut win = [_mm256_setzero_si256(); 8];
                    for (k, w) in win.iter_mut().enumerate().take(T - 1) {
                        *w = load(k);
                    }
                    for j in 0..h {
                        win[T - 1] = load(j + T - 1);
                        let (mut lo, mut hi) = (_mm256_setzero_si256(), _mm256_setzero_si256());
                        for k in 0..T / 2 {
                            let (a, b) = (win[2 * k], win[2 * k + 1]);
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
                        out.v16::<M>(j, x, v);
                        for k in 0..T - 1 {
                            win[k] = win[k + 1];
                        }
                    }
                }
            } else {
                let p: [__m128i; 4] = p.map(|p| _mm256_castsi256_si128(p));
                let load = |j: usize| across.eight(s, row(j));
                let mut win = [_mm_setzero_si128(); 8];
                for (k, w) in win.iter_mut().enumerate().take(T - 1) {
                    *w = load(k);
                }
                for j in 0..h {
                    win[T - 1] = load(j + T - 1);
                    let (mut lo, mut hi) = (_mm_setzero_si128(), _mm_setzero_si128());
                    for k in 0..T / 2 {
                        let (a, b) = (win[2 * k], win[2 * k + 1]);
                        lo = _mm_add_epi32(lo, _mm_madd_epi16(_mm_unpacklo_epi16(a, b), p[k]));
                        hi = _mm_add_epi32(hi, _mm_madd_epi16(_mm_unpackhi_epi16(a, b), p[k]));
                    }
                    let v = _mm_packs_epi32(_mm_srai_epi32::<6>(lo), _mm_srai_epi32::<6>(hi));
                    out.v8::<M>(j, 0, v, W);
                    for k in 0..T - 1 {
                        win[k] = win[k + 1];
                    }
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
