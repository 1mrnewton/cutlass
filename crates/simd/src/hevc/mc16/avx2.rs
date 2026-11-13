//! The 16-bit interpolation on AVX2. Samples of 12 bits or fewer fit
//! signed 16 bits, so every pass multiplies pairs of samples by pairs of
//! taps (`pmaddwd`) into 32-bit sums. Across a row, the sums of even and
//! odd outputs come from loads two samples apart and are interleaved back
//! into order; down the rows, pairs of rows are interleaved first.

// The loops index rows, strips and taps by one position at once.
#![allow(clippy::needless_range_loop)]

use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi32, _mm_cvtsi32_si128, _mm_cvtsi128_si64, _mm_extract_epi64,
    _mm_madd_epi16, _mm_packs_epi32, _mm_set_epi64x, _mm_setzero_si128, _mm_sll_epi16,
    _mm_sra_epi32, _mm_unpackhi_epi32, _mm_unpacklo_epi32, _mm256_add_epi32, _mm256_adds_epi16,
    _mm256_castsi128_si256, _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_madd_epi16,
    _mm256_max_epi16, _mm256_min_epi16, _mm256_packs_epi32, _mm256_set_epi64x, _mm256_set1_epi16,
    _mm256_set1_epi32, _mm256_setzero_si256, _mm256_sll_epi16, _mm256_sra_epi16, _mm256_sra_epi32,
    _mm256_unpackhi_epi16, _mm256_unpackhi_epi32, _mm256_unpacklo_epi16, _mm256_unpacklo_epi32,
};

use super::{Src, Taps};

/// Rows of horizontally filtered values kept between the passes: up to
/// four strips of sixteen for each row of a block's window.
pub(super) type Strips = [__m256i; 4];

/// Four samples as one 64-bit lane.
#[inline]
fn q(a: &[u16]) -> i64 {
    let mut v = 0u64;
    for (k, &x) in a.iter().take(4).enumerate() {
        v |= u64::from(x) << (16 * k);
    }
    v as i64
}

/// Sixteen samples of `r` from `x`, zeros past its end.
#[inline]
#[target_feature(enable = "avx2")]
fn ld16<const R: usize>(r: &[u16; R], x: usize) -> __m256i {
    match r.get(x..x + 16).and_then(|a| a.first_chunk::<16>()) {
        Some(a) => _mm256_set_epi64x(q(&a[12..]), q(&a[8..]), q(&a[4..]), q(a)),
        None => _mm256_setzero_si256(),
    }
}

/// Eight samples of `r` from `x`, zeros past its end.
#[inline]
#[target_feature(enable = "avx2")]
fn ld8<const R: usize>(r: &[u16; R], x: usize) -> __m128i {
    match r.get(x..x + 8).and_then(|a| a.first_chunk::<8>()) {
        Some(a) => _mm_set_epi64x(q(&a[4..]), q(a)),
        None => _mm_setzero_si128(),
    }
}

/// The sixteen values of `v` to `d[x..]`.
#[inline]
#[target_feature(enable = "avx2")]
fn st16<const W: usize>(d: &mut [i16; W], x: usize, v: __m256i) {
    let (lo, hi) = (
        lanes(_mm256_castsi256_si128(v)),
        lanes(_mm256_extracti128_si256::<1>(v)),
    );
    if let Some(d) = d.get_mut(x..x + 16) {
        d[..8].copy_from_slice(&lo);
        d[8..].copy_from_slice(&hi);
    }
}

/// The eight 16-bit values of `v`.
#[inline]
#[target_feature(enable = "avx2")]
fn lanes(v: __m128i) -> [i16; 8] {
    let (a, b) = (
        _mm_cvtsi128_si64(v) as u64,
        _mm_extract_epi64::<1>(v) as u64,
    );
    let mut out = [0i16; 8];
    for k in 0..4 {
        out[k] = (a >> (16 * k)) as i16;
        out[k + 4] = (b >> (16 * k)) as i16;
    }
    out
}

/// The first `W` (at most 8) values of `v` to `d`.
#[inline]
#[target_feature(enable = "avx2")]
fn st8<const W: usize>(d: &mut [i16; W], v: __m128i) {
    let l = lanes(v);
    let n = W.min(8);
    d[..n].copy_from_slice(&l[..n]);
}

/// Taps `2k` and `2k + 1` as the 16-bit pair `pmaddwd` multiplies by.
#[inline]
fn pair(f: &[i8; 8], k: usize) -> i32 {
    i32::from(f[2 * k] as i16 as u16) | i32::from(f[2 * k + 1] as i16) << 16
}

/// The horizontal filter of one row, `T` taps, shifted right by `shift`.
#[derive(Clone, Copy)]
struct Across<const T: usize> {
    pairs: [__m256i; 4],
    shift: __m128i,
}

impl<const T: usize> Across<T> {
    #[inline]
    #[target_feature(enable = "avx2")]
    fn new(f: &[i8; 8], shift: u32) -> Self {
        Self {
            pairs: core::array::from_fn(|k| _mm256_set1_epi32(pair(f, k))),
            shift: _mm_cvtsi32_si128(shift as i32),
        }
    }

    /// Sixteen outputs from column `x` of row `r`.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn sixteen<const R: usize>(&self, r: &[u16; R], x: usize) -> __m256i {
        let (mut even, mut odd) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        for k in 0..T / 2 {
            let p = self.pairs[k];
            even = _mm256_add_epi32(even, _mm256_madd_epi16(ld16(r, x + 2 * k), p));
            odd = _mm256_add_epi32(odd, _mm256_madd_epi16(ld16(r, x + 2 * k + 1), p));
        }
        let (even, odd) = (
            _mm256_sra_epi32(even, self.shift),
            _mm256_sra_epi32(odd, self.shift),
        );
        _mm256_packs_epi32(
            _mm256_unpacklo_epi32(even, odd),
            _mm256_unpackhi_epi32(even, odd),
        )
    }

    /// Eight outputs from column `x` of row `r`.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn eight<const R: usize>(&self, r: &[u16; R], x: usize) -> __m128i {
        let (mut even, mut odd) = (_mm_setzero_si128(), _mm_setzero_si128());
        for k in 0..T / 2 {
            let p = _mm256_castsi256_si128(self.pairs[k]);
            even = _mm_add_epi32(even, _mm_madd_epi16(ld8(r, x + 2 * k), p));
            odd = _mm_add_epi32(odd, _mm_madd_epi16(ld8(r, x + 2 * k + 1), p));
        }
        let (even, odd) = (
            _mm_sra_epi32(even, self.shift),
            _mm_sra_epi32(odd, self.shift),
        );
        _mm_packs_epi32(_mm_unpacklo_epi32(even, odd), _mm_unpackhi_epi32(even, odd))
    }
}

/// The vertical filter over window rows `win` of one strip, pairs of rows
/// times pairs of taps, shifted right by `shift`: sixteen outputs.
#[inline]
#[target_feature(enable = "avx2")]
fn down16<const T: usize>(win: &[__m256i; T], p: &[__m256i; 4], shift: __m128i) -> __m256i {
    let (mut lo, mut hi) = (_mm256_setzero_si256(), _mm256_setzero_si256());
    for k in 0..T / 2 {
        let (a, b) = (win[2 * k], win[2 * k + 1]);
        lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), p[k]));
        hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), p[k]));
    }
    _mm256_packs_epi32(_mm256_sra_epi32(lo, shift), _mm256_sra_epi32(hi, shift))
}

/// Where the filtered values of a block go: kept as intermediates
/// (mode 0), or made samples alone (1) or averaged with the intermediates
/// of the other list (2), clipped to the sample range.
pub(super) struct Sink<'a> {
    mid: &'a mut [i16],
    px: &'a mut [u16],
    other: &'a [i16],
    stride: usize,
    other_stride: usize,
    /// Column of the block's first sample, for a block done in parts.
    x0: usize,
    add: __m256i,
    shift: __m128i,
    max: __m256i,
}

impl<'a> Sink<'a> {
    /// The sink of `dst` for `bit_depth`, and its mode.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) fn new(dst: super::Dst<'a>, stride: usize, w: usize, bit_depth: u32) -> (Self, u8) {
        let max = _mm256_set1_epi16(((1i32 << bit_depth) - 1) as i16);
        let (shift, add) = (15 - bit_depth, 1i16 << (14 - bit_depth));
        let (uni_shift, uni_add) = (14 - bit_depth, 1i16 << (13 - bit_depth));
        let sink = |mid, px, other, (shift, add): (u32, i16)| Self {
            mid,
            px,
            other,
            stride,
            other_stride: w,
            x0: 0,
            add: _mm256_set1_epi16(add),
            shift: _mm_cvtsi32_si128(shift as i32),
            max,
        };
        match dst {
            super::Dst::Mid(mid) => (sink(mid, &mut [], &[], (0, 0)), 0),
            super::Dst::Uni(px) => (sink(&mut [], px, &[], (uni_shift, uni_add)), 1),
            super::Dst::Bi(px, other) => (sink(&mut [], px, other, (shift, add)), 2),
        }
    }

    /// The samples of `v` and, for mode 2, `o`.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn samples(&self, v: __m256i, o: __m256i) -> __m256i {
        let v = _mm256_adds_epi16(_mm256_adds_epi16(v, o), self.add);
        _mm256_min_epi16(
            _mm256_max_epi16(_mm256_sra_epi16(v, self.shift), _mm256_setzero_si256()),
            self.max,
        )
    }

    /// Sixteen values of row `j` from column `x`.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn sixteen<const M: u8>(&mut self, j: usize, x: usize, v: __m256i) {
        let at = j * self.stride + self.x0 + x;
        match M {
            0 => {
                if let Some(d) = self
                    .mid
                    .get_mut(at..)
                    .and_then(|d| d.first_chunk_mut::<16>())
                {
                    st16(d, 0, v);
                }
            }
            _ => {
                let o = if M == 2 {
                    ldi16(self.other, j * self.other_stride + self.x0 + x)
                } else {
                    _mm256_setzero_si256()
                };
                let v = self.samples(v, o);
                if let Some(d) = self
                    .px
                    .get_mut(at..)
                    .and_then(|d| d.first_chunk_mut::<16>())
                {
                    let (lo, hi) = (
                        lanes(_mm256_castsi256_si128(v)),
                        lanes(_mm256_extracti128_si256::<1>(v)),
                    );
                    for k in 0..8 {
                        d[k] = lo[k] as u16;
                        d[k + 8] = hi[k] as u16;
                    }
                }
            }
        }
    }

    /// The first `W` (4 or 8) values of `v`, row `j`.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn few<const M: u8, const W: usize>(&mut self, j: usize, v: __m128i) {
        let at = j * self.stride + self.x0;
        match M {
            0 => {
                if let Some(d) = self
                    .mid
                    .get_mut(at..)
                    .and_then(|d| d.first_chunk_mut::<W>())
                {
                    st8(d, v);
                }
            }
            _ => {
                let o = if M == 2 {
                    ldi8(self.other, j * self.other_stride + self.x0, W)
                } else {
                    _mm_setzero_si128()
                };
                let v = self.samples(_mm256_castsi128_si256(v), _mm256_castsi128_si256(o));
                let l = lanes(_mm256_castsi256_si128(v));
                if let Some(d) = self.px.get_mut(at..).and_then(|d| d.first_chunk_mut::<W>()) {
                    for (d, &s) in d.iter_mut().zip(&l) {
                        *d = s as u16;
                    }
                }
            }
        }
    }
}

/// The prediction of a block (see [`super::predict`]) into `out`, in mode
/// `M` (see [`Sink`]).
#[target_feature(enable = "avx2")]
pub(super) fn predict<const M: u8>(
    out: &mut Sink<'_>,
    src: &Src<'_>,
    (w, h): (usize, usize),
    taps: &Taps,
    bit_depth: u32,
    tmp: &mut Vec<Strips>,
) {
    // Ask for the window's lines first, as the 8-bit kernels do.
    for j in 0..h + taps.len - 1 {
        if let Some(row) = src.src.get(src.row(j)..) {
            // Two bytes a sample: up to three lines a row.
            let p = row.as_ptr().cast::<i8>();
            let last = 2 * (w + taps.len - 2);
            for at in (0..last).step_by(64).chain([last]) {
                core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(
                    p.wrapping_add(at),
                );
            }
        }
    }
    if tmp.len() < 64 + 7 {
        tmp.resize(64 + 7, [_mm256_setzero_si256(); 4]);
    }
    let tmp = tmp.as_mut_slice();
    let split = |a: usize| Src {
        at: src.at + a,
        ..*src
    };
    macro_rules! go {
        ($w:literal, $t:literal, $x:expr) => {{
            out.x0 = $x;
            put_w::<$w, { $w + 24 }, $t, M>(out, &split($x), h, taps, bit_depth, tmp)
        }};
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

/// Window row `j` of `src`: its first `R` samples, which the entry checked
/// are there.
#[inline(always)]
fn src_row<'s, const R: usize>(src: &Src<'s>, j: usize) -> Option<&'s [u16; R]> {
    src.src.get(src.row(j)..)?.first_chunk::<R>()
}

/// [`predict`] of a block `W` wide with `T` taps; `R` is `W + 24`, the
/// samples of each window row the loads read.
#[inline]
#[target_feature(enable = "avx2")]
fn put_w<const W: usize, const R: usize, const T: usize, const M: u8>(
    out: &mut Sink<'_>,
    src: &Src<'_>,
    h: usize,
    taps: &Taps,
    bit_depth: u32,
    tmp: &mut [Strips],
) {
    let h0 = T / 2 - 1;
    let h = h.min(64);
    let strips = W / 16;
    let shift1 = bit_depth - 8;
    match (taps.across, taps.down) {
        (None, None) => {
            let shift = _mm_cvtsi32_si128((14 - bit_depth) as i32);
            for j in 0..h {
                let Some(r) = src_row::<R>(src, j + h0) else {
                    return;
                };
                if W >= 16 {
                    for s in 0..strips {
                        let v = _mm256_sll_epi16(ld16(r, h0 + 16 * s), shift);
                        out.sixteen::<M>(j, 16 * s, v);
                    }
                } else {
                    out.few::<M, W>(j, _mm_sll_epi16(ld8(r, h0), shift));
                }
            }
        }
        (Some(fh), None) => {
            let across = Across::<T>::new(&fh, shift1);
            for j in 0..h {
                let Some(r) = src_row::<R>(src, j + h0) else {
                    return;
                };
                if W >= 16 {
                    for s in 0..strips {
                        out.sixteen::<M>(j, 16 * s, across.sixteen(r, 16 * s));
                    }
                } else {
                    out.few::<M, W>(j, across.eight(r, 0));
                }
            }
        }
        (across, Some(fv)) => {
            // Window rows, filtered across or as they are, then the taps
            // down: shifted by `bit_depth - 8` without a filter across, by
            // 6 after one.
            let fh = across.map(|f| Across::<T>::new(&f, shift1));
            let p: [__m256i; 4] = core::array::from_fn(|k| _mm256_set1_epi32(pair(&fv, k)));
            let shift = _mm_cvtsi32_si128(if fh.is_some() { 6 } else { shift1 as i32 });
            let Some(rows) = tmp.get_mut(..h + T - 1) else {
                return;
            };
            for (j, t) in rows.iter_mut().enumerate() {
                let Some(r) = src_row::<R>(src, j) else {
                    return;
                };
                match fh {
                    Some(a) if W >= 16 => {
                        for s in 0..strips {
                            t[s] = a.sixteen(r, 16 * s);
                        }
                    }
                    Some(a) => t[0] = _mm256_castsi128_si256(a.eight(r, 0)),
                    None if W >= 16 => {
                        for s in 0..strips {
                            t[s] = ld16(r, h0 + 16 * s);
                        }
                    }
                    None => t[0] = _mm256_castsi128_si256(ld8(r, h0)),
                }
            }
            for j in 0..h {
                let Some(win) = rows.get(j..).and_then(|w| w.first_chunk::<T>()) else {
                    return;
                };
                if W >= 16 {
                    for s in 0..strips {
                        let strip: [__m256i; T] = core::array::from_fn(|k| win[k][s]);
                        out.sixteen::<M>(j, 16 * s, down16(&strip, &p, shift));
                    }
                } else {
                    let strip: [__m256i; T] = core::array::from_fn(|k| win[k][0]);
                    out.few::<M, W>(j, _mm256_castsi256_si128(down16(&strip, &p, shift)));
                }
            }
        }
    }
}

// The default weighting. The sums are taken with saturation in 16 bits:
// a sum that saturates lies past the sample range either way, since the
// largest sample shifted back up, `((1 << bit_depth) - 1) << shift`, is
// below 32767 at every bit depth taken.

/// Sixteen intermediates from `s[at..]`, zeros past its end.
#[inline]
#[target_feature(enable = "avx2")]
fn ldi16(s: &[i16], at: usize) -> __m256i {
    match s.get(at..).and_then(|t| t.first_chunk::<16>()) {
        Some(a) => {
            let q = |k: usize| {
                let mut v = 0u64;
                for (i, &x) in a[4 * k..4 * k + 4].iter().enumerate() {
                    v |= u64::from(x as u16) << (16 * i);
                }
                v as i64
            };
            _mm256_set_epi64x(q(3), q(2), q(1), q(0))
        }
        None => _mm256_setzero_si256(),
    }
}

/// Eight, or four, intermediates from `s[at..]` (the rest zeros).
#[inline]
#[target_feature(enable = "avx2")]
fn ldi8(s: &[i16], at: usize, n: usize) -> __m128i {
    let t = s.get(at..).unwrap_or_default();
    let q = |k: usize| {
        let mut v = 0u64;
        for (i, &x) in t.iter().skip(4 * k).take(4).enumerate() {
            v |= u64::from(x as u16) << (16 * i);
        }
        v as i64
    };
    if n >= 8 && t.len() >= 8 {
        _mm_set_epi64x(q(1), q(0))
    } else if t.len() >= 4 {
        _mm_set_epi64x(0, q(0))
    } else {
        _mm_setzero_si128()
    }
}

#[target_feature(enable = "avx2")]
pub(super) fn store(
    dst: &mut [u16],
    dst_stride: usize,
    (a, b): (&[i16], Option<&[i16]>),
    ab_stride: usize,
    (w, h): (usize, usize),
    bit_depth: u32,
) {
    let (shift, add) = match b {
        Some(_) => (15 - bit_depth, 1i16 << (14 - bit_depth)),
        None => (14 - bit_depth, 1i16 << (13 - bit_depth)),
    };
    let max = ((1i32 << bit_depth) - 1) as i16;
    let shift = _mm_cvtsi32_si128(shift as i32);
    let (add16, max16) = (_mm256_set1_epi16(add), _mm256_set1_epi16(max));
    let one = |x: __m256i, y: __m256i| -> __m256i {
        let v = _mm256_adds_epi16(_mm256_adds_epi16(x, y), add16);
        _mm256_min_epi16(
            _mm256_max_epi16(_mm256_sra_epi16(v, shift), _mm256_setzero_si256()),
            max16,
        )
    };
    for j in 0..h {
        let (ra, rb) = (j * ab_stride, b.map(|b| (b, j * ab_stride)));
        let Some(o) = dst.get_mut(j * dst_stride..j * dst_stride + w) else {
            return;
        };
        let mut x = 0;
        while x + 16 <= w {
            let y = rb.map_or(_mm256_setzero_si256(), |(b, r)| ldi16(b, r + x));
            let v = one(ldi16(a, ra + x), y);
            if let Some(d) = o.get_mut(x..x + 16).and_then(|d| d.first_chunk_mut::<16>()) {
                let lo = lanes(_mm256_castsi256_si128(v));
                let hi = lanes(_mm256_extracti128_si256::<1>(v));
                for k in 0..8 {
                    d[k] = lo[k] as u16;
                    d[k + 8] = hi[k] as u16;
                }
            }
            x += 16;
        }
        while x < w {
            let n = (w - x).min(8);
            let y = rb.map_or(_mm_setzero_si128(), |(b, r)| ldi8(b, r + x, n));
            let v = one(
                _mm256_castsi128_si256(ldi8(a, ra + x, n)),
                _mm256_castsi128_si256(y),
            );
            let l = lanes(_mm256_castsi256_si128(v));
            for (d, &s) in o[x..x + n].iter_mut().zip(&l) {
                *d = s as u16;
            }
            x += n;
        }
    }
}
