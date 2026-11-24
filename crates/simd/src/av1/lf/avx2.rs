//! The deblocking sample filter on x86-64 with AVX2: sixteen lines in the
//! sixteen 8-bit lanes of a 128-bit vector, one vector per sample of the
//! line, and the sums of the wide filters in the sixteen 16-bit lanes of a
//! 256-bit one.

use core::arch::x86_64::{
    __m128i, __m256i, _mm_adds_epi8, _mm_adds_epu8, _mm_and_si128, _mm_andnot_si128,
    _mm_blendv_epi8, _mm_cmpeq_epi8, _mm_max_epu8, _mm_or_si128, _mm_packus_epi16, _mm_set1_epi8,
    _mm_setzero_si128, _mm_srli_epi16, _mm_sub_epi8, _mm_subs_epi8, _mm_subs_epu8, _mm_testz_si128,
    _mm_unpackhi_epi8, _mm_unpackhi_epi16, _mm_unpackhi_epi32, _mm_unpackhi_epi64,
    _mm_unpacklo_epi8, _mm_unpacklo_epi16, _mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm_xor_si128,
    _mm256_add_epi16, _mm256_castsi256_si128, _mm256_cvtepu8_epi16, _mm256_extracti128_si256,
    _mm256_set1_epi16, _mm256_setzero_si256, _mm256_srli_epi16, _mm256_sub_epi16,
};

use super::{Limits, Size};
use crate::av1::cpu::x86::{ld64, ld128, st64, st128};

type V = __m128i;

/// Sixteen 16-bit sums, one for each line.
type Sums = __m256i;

/// `|a - b|` in each lane.
#[inline]
#[target_feature(enable = "avx2")]
fn abd(a: V, b: V) -> V {
    _mm_or_si128(_mm_subs_epu8(a, b), _mm_subs_epu8(b, a))
}

/// All ones in the lanes where `a <= b`, unsigned.
#[inline]
#[target_feature(enable = "avx2")]
fn le(a: V, b: V) -> V {
    _mm_cmpeq_epi8(_mm_max_epu8(a, b), b)
}

/// Whether any bit of `a` is set.
#[inline]
#[target_feature(enable = "avx2")]
fn any(a: V) -> bool {
    _mm_testz_si128(a, a) == 0
}

#[inline]
#[target_feature(enable = "avx2")]
fn max3(a: V, b: V, c: V) -> V {
    _mm_max_epu8(a, _mm_max_epu8(b, c))
}

#[inline]
#[target_feature(enable = "avx2")]
fn splat(v: u8) -> V {
    _mm_set1_epi8(v as i8)
}

/// Each signed lane shifted right by 3, keeping its sign: the lane with
/// its sign bit turned shifts as an unsigned one, 16 too high.
#[inline]
#[target_feature(enable = "avx2")]
fn sra3(a: V) -> V {
    let u = _mm_xor_si128(a, splat(0x80));
    let s = _mm_and_si128(_mm_srli_epi16::<3>(u), splat(0x1f));
    _mm_sub_epi8(s, splat(0x10))
}

/// Each signed lane shifted right by 1, keeping its sign.
#[inline]
#[target_feature(enable = "avx2")]
fn sra1(a: V) -> V {
    let u = _mm_xor_si128(a, splat(0x80));
    let s = _mm_and_si128(_mm_srli_epi16::<1>(u), splat(0x7f));
    _mm_sub_epi8(s, splat(0x40))
}

/// The lanes widened to 16 bits.
#[inline]
#[target_feature(enable = "avx2")]
fn widen(a: V) -> Sums {
    _mm256_cvtepu8_epi16(a)
}

/// The wide filter (7.14.6.4) for taps `-N .. N` of the lines `o`,
/// written to `out` in the lanes `on` selects.
///
/// Output `i` is the rounded mean of the samples `i - N ..= i + N`, the
/// ones within `N2` of `i` counted twice, and a sample past the filter's
/// reach replaced by the last one inside it. So each output's sum is the
/// one before it with one sample entering and one leaving each of the two
/// windows. A sum of at most 16 samples fits 16 bits.
#[inline]
#[target_feature(enable = "avx2")]
fn wide<const N: i32, const N2: i32, const LOG2: i32>(o: &[V; 14], out: &mut [V; 14], on: V) {
    // The lines the filter reads, widened once.
    let mut w = [_mm256_setzero_si256(); 14];
    for k in (6 - N)..=(7 + N) {
        w[k as usize] = widen(o[k as usize]);
    }
    let at = |p: i32| w[(7 + p.clamp(-(N + 1), N)) as usize];
    // The first output's samples: the window of `2 N + 1` and the
    // `2 N2 + 1` counted twice, with the rounding.
    let mut sum = _mm256_set1_epi16(1 << (LOG2 - 1));
    for j in -N..=N {
        sum = _mm256_add_epi16(sum, at(j - N));
    }
    for j in -N2..=N2 {
        sum = _mm256_add_epi16(sum, at(j - N));
    }
    // Each output spelled out with a constant index, so the lines stay
    // in registers.
    let mut output = |i: i32| {
        if i < -N || i >= N {
            return;
        }
        if i > -N {
            let moved = _mm256_add_epi16(
                _mm256_sub_epi16(at(i + N), at(i - 1 - N)),
                _mm256_sub_epi16(at(i + N2), at(i - 1 - N2)),
            );
            sum = _mm256_add_epi16(sum, moved);
        }
        let s = _mm256_srli_epi16::<LOG2>(sum);
        let r = _mm_packus_epi16(_mm256_castsi256_si128(s), _mm256_extracti128_si256::<1>(s));
        let k = (7 + i) as usize;
        out[k] = _mm_blendv_epi8(out[k], r, on);
    };
    output(-6);
    output(-5);
    output(-4);
    output(-3);
    output(-2);
    output(-1);
    output(0);
    output(1);
    output(2);
    output(3);
    output(4);
    output(5);
}

/// Filters the sixteen lines in `v` (index 7 is `q0`). `SIZE` is 4, 6, 8
/// or 16. Returns whether any line passed the filter mask.
#[inline]
#[target_feature(enable = "avx2")]
fn filter<const SIZE: usize>(v: &mut [V; 14], lim: Limits) -> bool {
    let o = *v;
    let (p1, p0, q0, q1) = (o[5], o[6], o[7], o[8]);
    let step = _mm_max_epu8(abd(p1, p0), abd(q1, q0));
    // The lines without high edge variance.
    let calm = le(step, splat(lim.thresh));
    let mut inner = step;
    if SIZE >= 6 {
        inner = max3(inner, abd(o[4], p1), abd(o[9], q1));
    }
    if SIZE >= 8 {
        inner = max3(inner, abd(o[3], o[4]), abd(o[10], o[9]));
    }
    // |p0 - q0| * 2 + |p1 - q1| / 2. Saturation keeps it above `blimit`,
    // which is at most 193.
    let across = abd(p0, q0);
    let half = _mm_and_si128(_mm_srli_epi16::<1>(abd(p1, q1)), splat(0x7f));
    let outer = _mm_adds_epu8(_mm_adds_epu8(across, across), half);
    let pass = _mm_and_si128(le(inner, splat(lim.limit)), le(outer, splat(lim.blimit)));
    if !any(pass) {
        return false;
    }
    let one = splat(1);
    let mut flat = _mm_setzero_si128();
    if SIZE >= 6 {
        let mut f = max3(step, abd(o[4], p0), abd(o[9], q0));
        if SIZE >= 8 {
            f = max3(f, abd(o[3], p0), abd(o[10], q0));
        }
        flat = _mm_and_si128(le(f, one), pass);
    }
    let mut flat2 = _mm_setzero_si128();
    if SIZE == 16 {
        let f = _mm_max_epu8(
            max3(abd(o[0], p0), abd(o[1], p0), abd(o[2], p0)),
            max3(abd(o[13], q0), abd(o[12], q0), abd(o[11], q0)),
        );
        flat2 = _mm_and_si128(le(f, one), flat);
    }
    // The narrow filter (7.14.6.3), unless every line that passed takes
    // a wide one.
    if SIZE == 4 || any(_mm_andnot_si128(flat, pass)) {
        // On `sample - 128` as signed bytes. A saturating add of the
        // saturated difference three times equals the clamp of
        // `f + 3 * (qs0 - ps0)`: the sum moves one way only, so it can
        // only stick at the bound the exact value passes.
        let sign = splat(0x80);
        let ps1 = _mm_xor_si128(p1, sign);
        let ps0 = _mm_xor_si128(p0, sign);
        let qs0 = _mm_xor_si128(q0, sign);
        let qs1 = _mm_xor_si128(q1, sign);
        let d = _mm_subs_epi8(qs0, ps0);
        let f = _mm_andnot_si128(calm, _mm_subs_epi8(ps1, qs1));
        let f = _mm_adds_epi8(_mm_adds_epi8(_mm_adds_epi8(f, d), d), d);
        // A line that fails the mask moves by 0.
        let f = _mm_and_si128(f, pass);
        let f1 = sra3(_mm_adds_epi8(f, splat(4)));
        let f2 = sra3(_mm_adds_epi8(f, splat(3)));
        // `(f1 + 1) >> 1` (`f1` is -16 to 15), on the calm lines only.
        let f3 = _mm_and_si128(sra1(_mm_sub_epi8(f1, splat(0xff))), calm);
        v[5] = _mm_xor_si128(_mm_adds_epi8(ps1, f3), sign);
        v[6] = _mm_xor_si128(_mm_adds_epi8(ps0, f2), sign);
        v[7] = _mm_xor_si128(_mm_subs_epi8(qs0, f1), sign);
        v[8] = _mm_xor_si128(_mm_subs_epi8(qs1, f3), sign);
    }
    if SIZE >= 6 && any(flat) {
        match SIZE {
            6 => wide::<2, 1, 3>(&o, v, flat),
            8 => wide::<3, 0, 3>(&o, v, flat),
            _ => {
                // The filter over `p2 .. q2`, unless every flat line
                // takes the one over `p5 .. q5`, which writes over it.
                if any(_mm_andnot_si128(flat2, flat)) {
                    wide::<3, 0, 3>(&o, v, flat);
                }
                if any(flat2) {
                    wide::<6, 1, 4>(&o, v, flat2);
                }
            }
        }
    }
    true
}

#[inline]
#[target_feature(enable = "avx2")]
fn filter_size(v: &mut [V; 14], size: Size, lim: Limits) -> bool {
    match size {
        Size::Four => filter::<4>(v, lim),
        Size::Six => filter::<6>(v, lim),
        Size::Eight => filter::<8>(v, lim),
        Size::Sixteen => filter::<16>(v, lim),
    }
}

#[target_feature(enable = "avx2")]
pub(super) fn by_sample(taps: &mut [[u8; 16]; 14], size: Size, lim: Limits) -> bool {
    let (lo, hi) = size.reads();
    let mut v = [_mm_setzero_si128(); 14];
    for k in lo..hi {
        v[k] = ld128(&taps[k]);
    }
    if !filter_size(&mut v, size, lim) {
        return false;
    }
    let (lo, hi) = size.writes();
    for k in lo..hi {
        st128(&mut taps[k], v[k]);
    }
    true
}

/// Sixteen lines of eight samples (each in the low half of a vector) as
/// eight vectors of sixteen: lane `l` of vector `k` is sample `k` of line
/// `l`.
#[inline]
#[target_feature(enable = "avx2")]
fn columns8(l: &[V; 16]) -> [V; 8] {
    // Pairs of lines, then fours, then eights, each interleaved.
    let a: [V; 8] = core::array::from_fn(|i| _mm_unpacklo_epi8(l[2 * i], l[2 * i + 1]));
    // `b[2 j]` holds samples 0 to 3 of lines `4 j ..`, `b[2 j + 1]`
    // samples 4 to 7.
    let b: [V; 8] = core::array::from_fn(|i| {
        let (x, y) = (a[i / 2 * 2], a[i / 2 * 2 + 1]);
        if i % 2 == 0 {
            _mm_unpacklo_epi16(x, y)
        } else {
            _mm_unpackhi_epi16(x, y)
        }
    });
    // `c[4 g + s]`: samples `2 s, 2 s + 1` of lines `8 g ..`.
    let c: [V; 8] = core::array::from_fn(|i| {
        let (g, s) = (i / 4, i % 4);
        let (x, y) = (b[4 * g + s / 2], b[4 * g + 2 + s / 2]);
        if s % 2 == 0 {
            _mm_unpacklo_epi32(x, y)
        } else {
            _mm_unpackhi_epi32(x, y)
        }
    });
    core::array::from_fn(|k| {
        let (x, y) = (c[k / 2], c[4 + k / 2]);
        if k % 2 == 0 {
            _mm_unpacklo_epi64(x, y)
        } else {
            _mm_unpackhi_epi64(x, y)
        }
    })
}

/// The reverse of [`columns8`], two lines to a vector: lines `2 i` and
/// `2 i + 1` in the halves of vector `i`.
#[inline]
#[target_feature(enable = "avx2")]
fn lines8(c: &[V; 8]) -> [V; 8] {
    // `a[2 p + t]`: samples `2 p, 2 p + 1` of lines `8 t ..`.
    let a: [V; 8] = core::array::from_fn(|i| {
        let (x, y) = (c[i / 2 * 2], c[i / 2 * 2 + 1]);
        if i % 2 == 0 {
            _mm_unpacklo_epi8(x, y)
        } else {
            _mm_unpackhi_epi8(x, y)
        }
    });
    // `b[4 t + 2 m + n]`: samples `4 m ..` of lines `8 t + 4 n ..`.
    let b: [V; 8] = core::array::from_fn(|i| {
        let (t, m, n) = (i / 4, i / 2 % 2, i % 2);
        let (x, y) = (a[4 * m + t], a[4 * m + 2 + t]);
        if n == 0 {
            _mm_unpacklo_epi16(x, y)
        } else {
            _mm_unpackhi_epi16(x, y)
        }
    });
    // Vector `4 t + 2 n + u`: lines `8 t + 4 n + 2 u` and the next.
    core::array::from_fn(|i| {
        let (t, n, u) = (i / 4, i / 2 % 2, i % 2);
        let (x, y) = (b[4 * t + n], b[4 * t + 2 + n]);
        if u == 0 {
            _mm_unpacklo_epi32(x, y)
        } else {
            _mm_unpackhi_epi32(x, y)
        }
    })
}

/// Sixteen vectors of sixteen lanes, transposed.
#[inline]
#[target_feature(enable = "avx2")]
fn transpose16(r: &[V; 16]) -> [V; 16] {
    let a: [V; 16] = core::array::from_fn(|i| {
        let (x, y) = (r[i / 2 * 2], r[i / 2 * 2 + 1]);
        if i % 2 == 0 {
            _mm_unpacklo_epi8(x, y)
        } else {
            _mm_unpackhi_epi8(x, y)
        }
    });
    // `b[4 j + s]`: columns `4 s ..` of rows `4 j ..`.
    let b: [V; 16] = core::array::from_fn(|i| {
        let (j, s) = (i / 4, i % 4);
        let (x, y) = (a[4 * j + s / 2], a[4 * j + 2 + s / 2]);
        if s % 2 == 0 {
            _mm_unpacklo_epi16(x, y)
        } else {
            _mm_unpackhi_epi16(x, y)
        }
    });
    // `c[8 m + s]`: columns `2 s, 2 s + 1` of rows `8 m ..`.
    let c: [V; 16] = core::array::from_fn(|i| {
        let (m, s) = (i / 8, i % 8);
        let (x, y) = (b[8 * m + s / 2], b[8 * m + 4 + s / 2]);
        if s % 2 == 0 {
            _mm_unpacklo_epi32(x, y)
        } else {
            _mm_unpackhi_epi32(x, y)
        }
    });
    core::array::from_fn(|k| {
        let (x, y) = (c[k / 2], c[8 + k / 2]);
        if k % 2 == 0 {
            _mm_unpacklo_epi64(x, y)
        } else {
            _mm_unpackhi_epi64(x, y)
        }
    })
}

/// Filters sixteen lines of eight samples, `p3 .. q3`, one in the low
/// half of each vector. Gives the lines back two to a vector (see
/// [`lines8`]), or `None` when no line passed the filter mask.
#[inline]
#[target_feature(enable = "avx2")]
fn filter_lines8(l: &[V; 16], size: Size, lim: Limits) -> Option<[V; 8]> {
    let cols = columns8(l);
    let mut v = [_mm_setzero_si128(); 14];
    v[3..11].copy_from_slice(&cols);
    if !filter_size(&mut v, size, lim) {
        return None;
    }
    let mut cols = [_mm_setzero_si128(); 8];
    cols.copy_from_slice(&v[3..11]);
    Some(lines8(&cols))
}

/// Filters sixteen lines of sixteen samples: one unused, `p6 .. q6`, one
/// unused. `None` when no line passed the filter mask.
#[inline]
#[target_feature(enable = "avx2")]
fn filter_lines16(l: &[V; 16], lim: Limits) -> Option<[V; 16]> {
    let mut cols = transpose16(l);
    let mut v = [_mm_setzero_si128(); 14];
    v.copy_from_slice(&cols[1..15]);
    if !filter::<16>(&mut v, lim) {
        return None;
    }
    cols[1..15].copy_from_slice(&v);
    Some(transpose16(&cols))
}

/// The upper half of a vector in its lower half.
#[inline]
#[target_feature(enable = "avx2")]
fn upper(a: V) -> V {
    _mm_unpackhi_epi64(a, a)
}

#[target_feature(enable = "avx2")]
pub(super) fn by_line(lines: &mut [[u8; 8]; 16], size: Size, lim: Limits) -> bool {
    let l: [V; 16] = core::array::from_fn(|i| ld64(&lines[i]));
    let Some(rows) = filter_lines8(&l, size, lim) else {
        return false;
    };
    for (pair, &r) in lines.as_chunks_mut::<2>().0.iter_mut().zip(&rows) {
        st64(&mut pair[0], r);
        st64(&mut pair[1], upper(r));
    }
    true
}

#[target_feature(enable = "avx2")]
pub(super) fn by_line_16(lines: &mut [[u8; 16]; 16], lim: Limits) -> bool {
    let l: [V; 16] = core::array::from_fn(|i| ld128(&lines[i]));
    let Some(rows) = filter_lines16(&l, lim) else {
        return false;
    };
    for (line, &r) in lines.iter_mut().zip(&rows) {
        st128(line, r);
    }
    true
}

/// The `N` samples of `buf` from `o`.
#[inline(always)]
fn at<const N: usize>(buf: &[u8], o: usize) -> &[u8; N] {
    &buf[o..o + N].as_chunks::<N>().0[0]
}

/// The `N` samples of `buf` from `o`, to write.
#[inline(always)]
fn at_mut<const N: usize>(buf: &mut [u8], o: usize) -> &mut [u8; N] {
    &mut buf[o..o + N].as_chunks_mut::<N>().0[0]
}

#[target_feature(enable = "avx2")]
pub(super) fn rows_in_place(
    buf: &mut [u8],
    stride: usize,
    q0: usize,
    size: Size,
    lim: Limits,
) -> bool {
    let (lo, hi) = size.reads();
    // Line `k` is `k - 7` rows from `q0`; a filter near the top of the
    // picture reads none of the lines above it that are missing.
    let row = |k: usize| q0 + k * stride - 7 * stride;
    let mut v = [_mm_setzero_si128(); 14];
    for (k, vk) in v.iter_mut().enumerate().take(hi).skip(lo) {
        *vk = ld128(at::<16>(buf, row(k)));
    }
    if !filter_size(&mut v, size, lim) {
        return false;
    }
    let (lo, hi) = size.writes();
    for (k, &vk) in v.iter().enumerate().take(hi).skip(lo) {
        st128(at_mut::<16>(buf, row(k)), vk);
    }
    true
}

#[target_feature(enable = "avx2")]
pub(super) fn cols_in_place(
    buf: &mut [u8],
    stride: usize,
    q0: usize,
    size: Size,
    lim: Limits,
) -> bool {
    let left = q0 - 4;
    let l: [V; 16] = core::array::from_fn(|i| ld64(at::<8>(buf, left + i * stride)));
    let Some(rows) = filter_lines8(&l, size, lim) else {
        return false;
    };
    for (i, &r) in rows.iter().enumerate() {
        st64(at_mut::<8>(buf, left + 2 * i * stride), r);
        st64(at_mut::<8>(buf, left + (2 * i + 1) * stride), upper(r));
    }
    true
}

#[target_feature(enable = "avx2")]
pub(super) fn cols_16_in_place(buf: &mut [u8], stride: usize, q0: usize, lim: Limits) -> bool {
    let left = q0 - 8;
    let l: [V; 16] = core::array::from_fn(|i| ld128(at::<16>(buf, left + i * stride)));
    let Some(rows) = filter_lines16(&l, lim) else {
        return false;
    };
    for (i, &r) in rows.iter().enumerate() {
        st128(at_mut::<16>(buf, left + i * stride), r);
    }
    true
}
