//! The luma deblocking filter on 128-bit vectors: one 16-bit lane for
//! each of the eight lines, one vector for each sample across the edge.
//! Both filters are worked out for every line and the decisions of each
//! segment pick between them and the input.

use core::arch::x86_64::{
    __m128i, _mm_abs_epi16, _mm_add_epi16, _mm_and_si128, _mm_blendv_epi8, _mm_cmpgt_epi16,
    _mm_cvtepu8_epi16, _mm_cvtsi128_si64, _mm_extract_epi64, _mm_max_epi16, _mm_min_epi16,
    _mm_packus_epi16, _mm_set_epi64x, _mm_set1_epi16, _mm_slli_epi16, _mm_srai_epi16,
    _mm_srli_si128, _mm_sub_epi16, _mm_unpackhi_epi16, _mm_unpackhi_epi32, _mm_unpackhi_epi64,
    _mm_unpacklo_epi8, _mm_unpacklo_epi16, _mm_unpacklo_epi32, _mm_unpacklo_epi64,
};

use super::Segment;

/// Eight bytes as eight 16-bit lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn widen(b: &[u8; 8]) -> __m128i {
    _mm_cvtepu8_epi16(_mm_set_epi64x(0, i64::from_le_bytes(*b)))
}

/// Eight 16-bit lanes as bytes, clipped to `0..=255`.
#[inline]
#[target_feature(enable = "avx2")]
fn narrow(v: __m128i) -> [u8; 8] {
    (_mm_cvtsi128_si64(_mm_packus_epi16(v, v)) as u64).to_le_bytes()
}

/// The lanes of `v`.
#[inline]
#[target_feature(enable = "avx2")]
fn lanes(v: __m128i) -> [i16; 8] {
    let lo = (_mm_cvtsi128_si64(v) as u64).to_le_bytes();
    let hi = (_mm_extract_epi64::<1>(v) as u64).to_le_bytes();
    core::array::from_fn(|i| {
        let b = if i < 4 { &lo } else { &hi };
        let k = (i % 4) * 2;
        i16::from_le_bytes([b[k], b[k + 1]])
    })
}

/// Eight 16-bit values as lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn vector(a: [i16; 8]) -> __m128i {
    let q = |s: &[i16]| {
        s.iter()
            .enumerate()
            .fold(0u64, |v, (k, &x)| v | u64::from(x as u16) << (16 * k)) as i64
    };
    _mm_set_epi64x(q(&a[4..]), q(&a[..4]))
}

/// `x` clamped to `lo..=hi`, lane by lane.
#[inline]
#[target_feature(enable = "avx2")]
fn clamp(x: __m128i, lo: __m128i, hi: __m128i) -> __m128i {
    _mm_min_epi16(_mm_max_epi16(x, lo), hi)
}

/// Eight 8-byte rows transposed: eight 128-bit vectors whose low halves
/// hold the columns.
#[inline]
#[target_feature(enable = "avx2")]
fn transpose(a: [__m128i; 8]) -> [__m128i; 8] {
    let t = [
        _mm_unpacklo_epi8(a[0], a[1]),
        _mm_unpacklo_epi8(a[2], a[3]),
        _mm_unpacklo_epi8(a[4], a[5]),
        _mm_unpacklo_epi8(a[6], a[7]),
    ];
    let u = [
        _mm_unpacklo_epi16(t[0], t[1]),
        _mm_unpackhi_epi16(t[0], t[1]),
        _mm_unpacklo_epi16(t[2], t[3]),
        _mm_unpackhi_epi16(t[2], t[3]),
    ];
    let v = [
        _mm_unpacklo_epi32(u[0], u[2]),
        _mm_unpackhi_epi32(u[0], u[2]),
        _mm_unpacklo_epi32(u[1], u[3]),
        _mm_unpackhi_epi32(u[1], u[3]),
    ];
    [
        v[0],
        _mm_srli_si128::<8>(v[0]),
        v[1],
        _mm_srli_si128::<8>(v[1]),
        v[2],
        _mm_srli_si128::<8>(v[2]),
        v[3],
        _mm_srli_si128::<8>(v[3]),
    ]
}

/// The kernel on eight lines along a vertical edge: `lines[l]` holds the
/// eight samples of line `l` across it.
#[target_feature(enable = "avx2")]
pub(super) fn luma8_lines(lines: &mut [[u8; 8]; 8], seg: [Segment; 2]) {
    let a = lines
        .each_ref()
        .map(|l| _mm_set_epi64x(0, i64::from_le_bytes(*l)));
    let rows = transpose(a).map(|v| (_mm_cvtsi128_si64(v) as u64).to_le_bytes());
    let mut rows = rows;
    if luma8(&mut rows, seg) {
        let b = rows
            .each_ref()
            .map(|r| _mm_set_epi64x(0, i64::from_le_bytes(*r)));
        for (l, v) in lines.iter_mut().zip(transpose(b)) {
            *l = (_mm_cvtsi128_si64(v) as u64).to_le_bytes();
        }
    }
}

/// The kernel; `false` when no line changed.
#[target_feature(enable = "avx2")]
pub(super) fn luma8(rows: &mut [[u8; 8]; 8], seg: [Segment; 2]) -> bool {
    let Some(out) = filter(rows.each_ref().map(|r| widen(r)), seg) else {
        return false;
    };
    for (row, v) in rows[1..7].iter_mut().zip(out) {
        *row = narrow(v);
    }
    true
}

/// Eight 16-bit samples as lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn ld16(r: &[u16; 8]) -> __m128i {
    let q = |s: &[u16]| {
        let mut v = 0u64;
        for (k, &x) in s.iter().enumerate() {
            v |= u64::from(x) << (16 * k);
        }
        v as i64
    };
    _mm_set_epi64x(q(&r[4..]), q(&r[..4]))
}

/// The lanes of `v` as samples.
#[inline]
#[target_feature(enable = "avx2")]
fn st16(d: &mut [u16; 8], v: __m128i) {
    let (lo, hi) = (
        _mm_cvtsi128_si64(v) as u64,
        _mm_extract_epi64::<1>(v) as u64,
    );
    for k in 0..4 {
        d[k] = (lo >> (16 * k)) as u16;
        d[k + 4] = (hi >> (16 * k)) as u16;
    }
}

/// The kernel on samples of 9 to 11 bits, `max` the largest sample.
#[target_feature(enable = "avx2")]
pub(super) fn luma16(rows: &mut [[u16; 8]; 8], seg: [Segment; 2], max: i16) -> bool {
    let Some(out) = filter(rows.each_ref().map(|r| ld16(r)), seg) else {
        return false;
    };
    let (lo, hi) = (_mm_set1_epi16(0), _mm_set1_epi16(max));
    for (row, v) in rows[1..7].iter_mut().zip(out) {
        st16(row, clamp(v, lo, hi));
    }
    true
}

/// [`luma16`] on eight lines along a vertical edge: transposed in
/// registers, filtered, and transposed back.
#[target_feature(enable = "avx2")]
pub(super) fn luma16_lines(lines: &mut [[u16; 8]; 8], seg: [Segment; 2], max: i16) -> bool {
    let mut t = transpose16(lines.each_ref().map(|l| ld16(l)));
    let Some(out) = filter(t, seg) else {
        return false;
    };
    let (lo, hi) = (_mm_set1_epi16(0), _mm_set1_epi16(max));
    for (r, v) in t[1..7].iter_mut().zip(out) {
        *r = clamp(v, lo, hi);
    }
    for (l, v) in lines.iter_mut().zip(transpose16(t)) {
        st16(l, v);
    }
    true
}

/// Eight vectors of eight 16-bit values transposed.
#[inline]
#[target_feature(enable = "avx2")]
fn transpose16(a: [__m128i; 8]) -> [__m128i; 8] {
    let t = [
        _mm_unpacklo_epi16(a[0], a[1]),
        _mm_unpackhi_epi16(a[0], a[1]),
        _mm_unpacklo_epi16(a[2], a[3]),
        _mm_unpackhi_epi16(a[2], a[3]),
        _mm_unpacklo_epi16(a[4], a[5]),
        _mm_unpackhi_epi16(a[4], a[5]),
        _mm_unpacklo_epi16(a[6], a[7]),
        _mm_unpackhi_epi16(a[6], a[7]),
    ];
    let u = [
        _mm_unpacklo_epi32(t[0], t[2]),
        _mm_unpackhi_epi32(t[0], t[2]),
        _mm_unpacklo_epi32(t[1], t[3]),
        _mm_unpackhi_epi32(t[1], t[3]),
        _mm_unpacklo_epi32(t[4], t[6]),
        _mm_unpackhi_epi32(t[4], t[6]),
        _mm_unpacklo_epi32(t[5], t[7]),
        _mm_unpackhi_epi32(t[5], t[7]),
    ];
    [
        _mm_unpacklo_epi64(u[0], u[4]),
        _mm_unpackhi_epi64(u[0], u[4]),
        _mm_unpacklo_epi64(u[1], u[5]),
        _mm_unpackhi_epi64(u[1], u[5]),
        _mm_unpacklo_epi64(u[2], u[6]),
        _mm_unpackhi_epi64(u[2], u[6]),
        _mm_unpacklo_epi64(u[3], u[7]),
        _mm_unpackhi_epi64(u[3], u[7]),
    ]
}

/// Both filters on eight lines `[p3, p2, p1, p0, q0, q1, q2, q3]`, picked
/// by the decisions of each segment: the new `p2` to `q2`, not clipped to
/// the sample range; `None` when no line changes.
#[inline]
#[target_feature(enable = "avx2")]
fn filter(v: [__m128i; 8], seg: [Segment; 2]) -> Option<[__m128i; 6]> {
    let [p3, p2, p1, p0, q0, q1, q2, q3] = v;
    let two = |v: __m128i| _mm_slli_epi16::<1>(v);
    // Decisions of each segment, from its lines 0 and 3.
    let dp = lanes(_mm_abs_epi16(_mm_add_epi16(_mm_sub_epi16(p2, two(p1)), p0)));
    let dq = lanes(_mm_abs_epi16(_mm_add_epi16(_mm_sub_epi16(q2, two(q1)), q0)));
    let ends = lanes(_mm_add_epi16(
        _mm_abs_epi16(_mm_sub_epi16(p3, p0)),
        _mm_abs_epi16(_mm_sub_epi16(q0, q3)),
    ));
    let step = lanes(_mm_abs_epi16(_mm_sub_epi16(p0, q0)));
    let mut strong = [0i16; 8];
    let mut weak = [0i16; 8];
    let mut fp1 = [0i16; 8];
    let mut fq1 = [0i16; 8];
    let mut keep_p = [0i16; 8];
    let mut keep_q = [0i16; 8];
    let mut tc = [0i16; 8];
    for (s, sg) in seg.iter().enumerate() {
        let (a, b) = (4 * s, 4 * s + 3);
        let (beta, t) = (i32::from(sg.beta), i32::from(sg.tc));
        let (d0, d3) = (
            i32::from(dp[a]) + i32::from(dq[a]),
            i32::from(dp[b]) + i32::from(dq[b]),
        );
        if d0 + d3 >= beta {
            continue;
        }
        let strong_line = |l: usize, d: i32| {
            2 * d < (beta >> 2)
                && i32::from(ends[l]) < (beta >> 3)
                && i32::from(step[l]) < ((5 * t + 1) >> 1)
        };
        let is_strong = strong_line(a, d0) && strong_line(b, d3);
        let side = (beta + (beta >> 1)) >> 3;
        let f_p1 = i32::from(dp[a]) + i32::from(dp[b]) < side;
        let f_q1 = i32::from(dq[a]) + i32::from(dq[b]) < side;
        for l in a..=b {
            strong[l] = -i16::from(is_strong);
            weak[l] = -i16::from(!is_strong);
            fp1[l] = -i16::from(f_p1);
            fq1[l] = -i16::from(f_q1);
            keep_p[l] = -i16::from(sg.no_p);
            keep_q[l] = -i16::from(sg.no_q);
            tc[l] = sg.tc;
        }
    }
    if strong.iter().chain(&weak).all(|&m| m == 0) {
        return None;
    }
    let (strong, weak, fp1, fq1) = (vector(strong), vector(weak), vector(fp1), vector(fq1));
    let (keep_p, keep_q, tc) = (vector(keep_p), vector(keep_q), vector(tc));
    // The strong filter.
    let tc2 = two(tc);
    let near = |v: __m128i, x: __m128i| clamp(v, _mm_sub_epi16(x, tc2), _mm_add_epi16(x, tc2));
    let r = |v: __m128i, add: i16, sh: i32| -> __m128i {
        let v = _mm_add_epi16(v, _mm_set1_epi16(add));
        if sh == 3 {
            _mm_srai_epi16::<3>(v)
        } else {
            _mm_srai_epi16::<2>(v)
        }
    };
    let s_p0 = near(
        r(
            _mm_add_epi16(
                _mm_add_epi16(p2, two(_mm_add_epi16(p1, _mm_add_epi16(p0, q0)))),
                q1,
            ),
            4,
            3,
        ),
        p0,
    );
    let s_p1 = near(
        r(
            _mm_add_epi16(_mm_add_epi16(p2, p1), _mm_add_epi16(p0, q0)),
            2,
            2,
        ),
        p1,
    );
    let s_p2 = near(
        r(
            _mm_add_epi16(
                _mm_add_epi16(two(p3), _mm_add_epi16(two(p2), p2)),
                _mm_add_epi16(_mm_add_epi16(p1, p0), q0),
            ),
            4,
            3,
        ),
        p2,
    );
    let s_q0 = near(
        r(
            _mm_add_epi16(
                _mm_add_epi16(q2, two(_mm_add_epi16(q1, _mm_add_epi16(q0, p0)))),
                p1,
            ),
            4,
            3,
        ),
        q0,
    );
    let s_q1 = near(
        r(
            _mm_add_epi16(_mm_add_epi16(q2, q1), _mm_add_epi16(q0, p0)),
            2,
            2,
        ),
        q1,
    );
    let s_q2 = near(
        r(
            _mm_add_epi16(
                _mm_add_epi16(two(q3), _mm_add_epi16(two(q2), q2)),
                _mm_add_epi16(_mm_add_epi16(q1, q0), p0),
            ),
            4,
            3,
        ),
        q2,
    );
    // The weak filter.
    let nine = _mm_add_epi16(
        _mm_slli_epi16::<3>(_mm_sub_epi16(q0, p0)),
        _mm_sub_epi16(q0, p0),
    );
    let three = _mm_add_epi16(two(_mm_sub_epi16(q1, p1)), _mm_sub_epi16(q1, p1));
    let delta = _mm_srai_epi16::<4>(_mm_add_epi16(_mm_sub_epi16(nine, three), _mm_set1_epi16(8)));
    let ten_tc = _mm_add_epi16(_mm_slli_epi16::<3>(tc), two(tc));
    let apply = _mm_and_si128(weak, _mm_cmpgt_epi16(ten_tc, _mm_abs_epi16(delta)));
    let neg_tc = _mm_sub_epi16(_mm_set1_epi16(0), tc);
    let delta = clamp(delta, neg_tc, tc);
    let w_p0 = _mm_add_epi16(p0, delta);
    let w_q0 = _mm_sub_epi16(q0, delta);
    let half = _mm_srai_epi16::<1>(tc);
    let neg_half = _mm_sub_epi16(_mm_set1_epi16(0), half);
    let one = _mm_set1_epi16(1);
    let avg = |a: __m128i, b: __m128i| _mm_srai_epi16::<1>(_mm_add_epi16(_mm_add_epi16(a, b), one));
    let w_p1 = _mm_add_epi16(
        p1,
        clamp(
            _mm_srai_epi16::<1>(_mm_add_epi16(_mm_sub_epi16(avg(p2, p0), p1), delta)),
            neg_half,
            half,
        ),
    );
    let w_q1 = _mm_add_epi16(
        q1,
        clamp(
            _mm_srai_epi16::<1>(_mm_sub_epi16(_mm_sub_epi16(avg(q2, q0), q1), delta)),
            neg_half,
            half,
        ),
    );
    // Pick, line by line: strong, weak where it applies, else the input;
    // a kept side stays the input.
    let pick = |x: __m128i, s: __m128i, w: Option<(__m128i, __m128i)>, keep: __m128i| {
        let mut v = _mm_blendv_epi8(x, s, strong);
        if let Some((w, m)) = w {
            v = _mm_blendv_epi8(v, w, m);
        }
        _mm_blendv_epi8(v, x, keep)
    };
    Some([
        pick(p2, s_p2, None, keep_p),
        pick(p1, s_p1, Some((w_p1, _mm_and_si128(apply, fp1))), keep_p),
        pick(p0, s_p0, Some((w_p0, apply)), keep_p),
        pick(q0, s_q0, Some((w_q0, apply)), keep_q),
        pick(q1, s_q1, Some((w_q1, _mm_and_si128(apply, fq1))), keep_q),
        pick(q2, s_q2, None, keep_q),
    ])
}
