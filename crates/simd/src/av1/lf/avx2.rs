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

/// [`Size::reads`] of the filter of `SIZE` samples (4, 6, 8 or 16), known
/// where the kernel is compiled so that its loops over samples unroll.
const fn reads(size: usize) -> (usize, usize) {
    match size {
        4 => Size::Four.reads(),
        6 => Size::Six.reads(),
        8 => Size::Eight.reads(),
        _ => Size::Sixteen.reads(),
    }
}

/// [`Size::writes`] of the filter of `SIZE` samples.
const fn writes(size: usize) -> (usize, usize) {
    match size {
        4 => Size::Four.writes(),
        6 => Size::Six.writes(),
        8 => Size::Eight.writes(),
        _ => Size::Sixteen.writes(),
    }
}

/// Runs `$f::<SIZE>($args)` with the `SIZE` of a [`Size`].
macro_rules! sized {
    ($size:expr, $f:ident($($arg:expr),*)) => {
        match $size {
            Size::Four => $f::<4>($($arg),*),
            Size::Six => $f::<6>($($arg),*),
            Size::Eight => $f::<8>($($arg),*),
            Size::Sixteen => $f::<16>($($arg),*),
        }
    };
}

#[inline]
#[target_feature(enable = "avx2")]
fn samples<const SIZE: usize>(taps: &mut [[u8; 16]; 14], lim: Limits) -> bool {
    let (lo, hi) = reads(SIZE);
    let mut v = [_mm_setzero_si128(); 14];
    for k in lo..hi {
        v[k] = ld128(&taps[k]);
    }
    if !filter::<SIZE>(&mut v, lim) {
        return false;
    }
    let (lo, hi) = writes(SIZE);
    for k in lo..hi {
        st128(&mut taps[k], v[k]);
    }
    true
}

#[target_feature(enable = "avx2")]
pub(super) fn by_sample(taps: &mut [[u8; 16]; 14], size: Size, lim: Limits) -> bool {
    sized!(size, samples(taps, lim))
}

/// The low and the high halves of two vectors interleaved, by lanes of
/// 8, 16, 32 and 64 bits.
#[inline]
#[target_feature(enable = "avx2")]
fn zip8(x: V, y: V) -> (V, V) {
    (_mm_unpacklo_epi8(x, y), _mm_unpackhi_epi8(x, y))
}

#[inline]
#[target_feature(enable = "avx2")]
fn zip16(x: V, y: V) -> (V, V) {
    (_mm_unpacklo_epi16(x, y), _mm_unpackhi_epi16(x, y))
}

#[inline]
#[target_feature(enable = "avx2")]
fn zip32(x: V, y: V) -> (V, V) {
    (_mm_unpacklo_epi32(x, y), _mm_unpackhi_epi32(x, y))
}

#[inline]
#[target_feature(enable = "avx2")]
fn zip64(x: V, y: V) -> (V, V) {
    (_mm_unpacklo_epi64(x, y), _mm_unpackhi_epi64(x, y))
}

// The transposes below name every vector, so that they stay in
// registers: written as loops over arrays they were left rolled, moving
// the vectors through memory (a third of the instructions of a size-16
// vertical edge).

/// Sixteen lines of eight samples (each in the low half of a vector) as
/// eight vectors of sixteen: lane `l` of vector `k` is sample `k` of line
/// `l`.
#[inline]
#[target_feature(enable = "avx2")]
fn columns8(l: &[V; 16]) -> [V; 8] {
    let [
        l0,
        l1,
        l2,
        l3,
        l4,
        l5,
        l6,
        l7,
        l8,
        l9,
        l10,
        l11,
        l12,
        l13,
        l14,
        l15,
    ] = *l;
    // Pairs of lines interleaved, then fours, then eights.
    let a0 = _mm_unpacklo_epi8(l0, l1);
    let a1 = _mm_unpacklo_epi8(l2, l3);
    let a2 = _mm_unpacklo_epi8(l4, l5);
    let a3 = _mm_unpacklo_epi8(l6, l7);
    let a4 = _mm_unpacklo_epi8(l8, l9);
    let a5 = _mm_unpacklo_epi8(l10, l11);
    let a6 = _mm_unpacklo_epi8(l12, l13);
    let a7 = _mm_unpacklo_epi8(l14, l15);
    // `b[2 j]`: samples 0 to 3 of lines `4 j ..`; `b[2 j + 1]`: samples 4 to 7.
    let (b0, b1) = zip16(a0, a1);
    let (b2, b3) = zip16(a2, a3);
    let (b4, b5) = zip16(a4, a5);
    let (b6, b7) = zip16(a6, a7);
    // `c[4 g + s]`: samples `2 s, 2 s + 1` of lines `8 g ..`.
    let (c0, c1) = zip32(b0, b2);
    let (c2, c3) = zip32(b1, b3);
    let (c4, c5) = zip32(b4, b6);
    let (c6, c7) = zip32(b5, b7);
    let (o0, o1) = zip64(c0, c4);
    let (o2, o3) = zip64(c1, c5);
    let (o4, o5) = zip64(c2, c6);
    let (o6, o7) = zip64(c3, c7);
    [o0, o1, o2, o3, o4, o5, o6, o7]
}

/// The reverse of [`columns8`], two lines to a vector: lines `2 i` and
/// `2 i + 1` in the halves of vector `i`.
#[inline]
#[target_feature(enable = "avx2")]
fn lines8(c: &[V; 8]) -> [V; 8] {
    let [c0, c1, c2, c3, c4, c5, c6, c7] = *c;
    // `a[2 p + t]`: samples `2 p, 2 p + 1` of lines `8 t ..`.
    let (a0, a1) = zip8(c0, c1);
    let (a2, a3) = zip8(c2, c3);
    let (a4, a5) = zip8(c4, c5);
    let (a6, a7) = zip8(c6, c7);
    // `b[4 t + 2 m + n]`: samples `4 m ..` of lines `8 t + 4 n ..`.
    let (b0, b1) = zip16(a0, a2);
    let (b2, b3) = zip16(a4, a6);
    let (b4, b5) = zip16(a1, a3);
    let (b6, b7) = zip16(a5, a7);
    // Vector `4 t + 2 n + u`: lines `8 t + 4 n + 2 u` and the next.
    let (o0, o1) = zip32(b0, b2);
    let (o2, o3) = zip32(b1, b3);
    let (o4, o5) = zip32(b4, b6);
    let (o6, o7) = zip32(b5, b7);
    [o0, o1, o2, o3, o4, o5, o6, o7]
}

/// Sixteen vectors of sixteen lanes, transposed.
#[inline]
#[target_feature(enable = "avx2")]
fn transpose16(r: &[V; 16]) -> [V; 16] {
    let [
        r0,
        r1,
        r2,
        r3,
        r4,
        r5,
        r6,
        r7,
        r8,
        r9,
        r10,
        r11,
        r12,
        r13,
        r14,
        r15,
    ] = *r;
    let (a0, a1) = zip8(r0, r1);
    let (a2, a3) = zip8(r2, r3);
    let (a4, a5) = zip8(r4, r5);
    let (a6, a7) = zip8(r6, r7);
    let (a8, a9) = zip8(r8, r9);
    let (a10, a11) = zip8(r10, r11);
    let (a12, a13) = zip8(r12, r13);
    let (a14, a15) = zip8(r14, r15);
    // `b[4 j + s]`: columns `4 s ..` of rows `4 j ..`.
    let (b0, b1) = zip16(a0, a2);
    let (b2, b3) = zip16(a1, a3);
    let (b4, b5) = zip16(a4, a6);
    let (b6, b7) = zip16(a5, a7);
    let (b8, b9) = zip16(a8, a10);
    let (b10, b11) = zip16(a9, a11);
    let (b12, b13) = zip16(a12, a14);
    let (b14, b15) = zip16(a13, a15);
    // `c[8 m + s]`: columns `2 s, 2 s + 1` of rows `8 m ..`.
    let (c0, c1) = zip32(b0, b4);
    let (c2, c3) = zip32(b1, b5);
    let (c4, c5) = zip32(b2, b6);
    let (c6, c7) = zip32(b3, b7);
    let (c8, c9) = zip32(b8, b12);
    let (c10, c11) = zip32(b9, b13);
    let (c12, c13) = zip32(b10, b14);
    let (c14, c15) = zip32(b11, b15);
    let (o0, o1) = zip64(c0, c8);
    let (o2, o3) = zip64(c1, c9);
    let (o4, o5) = zip64(c2, c10);
    let (o6, o7) = zip64(c3, c11);
    let (o8, o9) = zip64(c4, c12);
    let (o10, o11) = zip64(c5, c13);
    let (o12, o13) = zip64(c6, c14);
    let (o14, o15) = zip64(c7, c15);
    [
        o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15,
    ]
}

/// Filters sixteen lines of eight samples, `p3 .. q3`, one in the low
/// half of each vector. Gives the lines back two to a vector (see
/// [`lines8`]), or `None` when no line passed the filter mask.
#[inline]
#[target_feature(enable = "avx2")]
fn filter_lines8<const SIZE: usize>(l: &[V; 16], lim: Limits) -> Option<[V; 8]> {
    let [c0, c1, c2, c3, c4, c5, c6, c7] = columns8(l);
    let z = _mm_setzero_si128();
    let mut v = [z, z, z, c0, c1, c2, c3, c4, c5, c6, c7, z, z, z];
    if !filter::<SIZE>(&mut v, lim) {
        return None;
    }
    Some(lines8(&[v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10]]))
}

/// Filters sixteen lines of sixteen samples: one unused, `p6 .. q6`, one
/// unused. `None` when no line passed the filter mask.
#[inline]
#[target_feature(enable = "avx2")]
fn filter_lines16(l: &[V; 16], lim: Limits) -> Option<[V; 16]> {
    let c = transpose16(l);
    let mut v = [
        c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11], c[12], c[13], c[14],
    ];
    if !filter::<16>(&mut v, lim) {
        return None;
    }
    Some(transpose16(&[
        c[0], v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12],
        v[13], c[15],
    ]))
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
    let Some(rows) = sized!(size, filter_lines8(&l, lim)) else {
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

#[inline]
#[target_feature(enable = "avx2")]
fn rows<const SIZE: usize>(buf: &mut [u8], stride: usize, q0: usize, lim: Limits) -> bool {
    let (lo, hi) = reads(SIZE);
    // Line `k` is `k - 7` rows from `q0`; a filter near the top of the
    // picture reads none of the lines above it that are missing.
    let row = |k: usize| q0 + k * stride - 7 * stride;
    let mut v = [_mm_setzero_si128(); 14];
    for (k, vk) in v.iter_mut().enumerate().take(hi).skip(lo) {
        *vk = ld128(at::<16>(buf, row(k)));
    }
    if !filter::<SIZE>(&mut v, lim) {
        return false;
    }
    let (lo, hi) = writes(SIZE);
    for (k, &vk) in v.iter().enumerate().take(hi).skip(lo) {
        st128(at_mut::<16>(buf, row(k)), vk);
    }
    true
}

#[target_feature(enable = "avx2")]
pub(super) fn rows_in_place(
    buf: &mut [u8],
    stride: usize,
    q0: usize,
    size: Size,
    lim: Limits,
) -> bool {
    sized!(size, rows(buf, stride, q0, lim))
}

#[inline]
#[target_feature(enable = "avx2")]
fn cols<const SIZE: usize>(buf: &mut [u8], stride: usize, q0: usize, lim: Limits) -> bool {
    let left = q0 - 4;
    let l: [V; 16] = core::array::from_fn(|i| ld64(at::<8>(buf, left + i * stride)));
    let Some(rows) = filter_lines8::<SIZE>(&l, lim) else {
        return false;
    };
    for (i, &r) in rows.iter().enumerate() {
        st64(at_mut::<8>(buf, left + 2 * i * stride), r);
        st64(at_mut::<8>(buf, left + (2 * i + 1) * stride), upper(r));
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
    sized!(size, cols(buf, stride, q0, lim))
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
