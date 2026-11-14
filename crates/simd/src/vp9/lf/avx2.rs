//! The 8-bit loop filter on AVX2, sixteen lines at a time in 16-bit
//! lanes: tap `k` of every line is one vector. A horizontal edge loads its
//! rows straight into those vectors; a vertical edge loads its lines and
//! transposes them, and back.

use core::arch::x86_64::{
    __m128i, __m256i, _mm_cvtsi64_si128, _mm_cvtsi128_si64, _mm_extract_epi64, _mm_set_epi64x,
    _mm_setzero_si128, _mm_unpackhi_epi8, _mm_unpackhi_epi16, _mm_unpackhi_epi32,
    _mm_unpackhi_epi64, _mm_unpacklo_epi8, _mm_unpacklo_epi16, _mm_unpacklo_epi32,
    _mm_unpacklo_epi64, _mm256_abs_epi16, _mm256_add_epi16, _mm256_and_si256,
    _mm256_andnot_si256, _mm256_blendv_epi8, _mm256_castsi256_si128, _mm256_cmpgt_epi16,
    _mm256_cvtepu8_epi16, _mm256_max_epi16, _mm256_min_epi16, _mm256_or_si256,
    _mm256_packus_epi16, _mm256_permute4x64_epi64, _mm256_set_m128i, _mm256_set1_epi16,
    _mm256_slli_epi16, _mm256_srai_epi16, _mm256_srli_epi16, _mm256_sub_epi16,
};

use super::{Edge, Thresh, Wd};

#[inline]
#[target_feature(enable = "avx2")]
fn ld16(s: &[u8], at: usize) -> __m128i {
    let a = &s[at..at + 16];
    let lo = i64::from_le_bytes([a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]]);
    let hi = i64::from_le_bytes([a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]]);
    _mm_set_epi64x(hi, lo)
}

#[inline]
#[target_feature(enable = "avx2")]
fn ld8(s: &[u8], at: usize) -> __m128i {
    let a = &s[at..at + 8];
    _mm_cvtsi64_si128(i64::from_le_bytes([a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]]))
}

#[inline]
#[target_feature(enable = "avx2")]
fn st16(d: &mut [u8], at: usize, v: __m128i) {
    d[at..at + 8].copy_from_slice(&_mm_cvtsi128_si64(v).to_le_bytes());
    d[at + 8..at + 16].copy_from_slice(&_mm_extract_epi64::<1>(v).to_le_bytes());
}

#[inline]
#[target_feature(enable = "avx2")]
fn st8(d: &mut [u8], at: usize, v: __m128i) {
    d[at..at + 8].copy_from_slice(&_mm_cvtsi128_si64(v).to_le_bytes());
}

/// Sixteen 16-bit lanes to sixteen bytes.
#[inline]
#[target_feature(enable = "avx2")]
fn pack(v: __m256i) -> __m128i {
    let p = _mm256_packus_epi16(v, v);
    _mm256_castsi256_si128(_mm256_permute4x64_epi64::<0b1000>(p))
}

/// The transpose of a 16×16 block of bytes.
#[inline]
#[target_feature(enable = "avx2")]
fn transpose(r: &[__m128i; 16]) -> [__m128i; 16] {
    let z = _mm_setzero_si128();
    let mut a = [z; 16];
    for i in 0..8 {
        a[2 * i] = _mm_unpacklo_epi8(r[2 * i], r[2 * i + 1]);
        a[2 * i + 1] = _mm_unpackhi_epi8(r[2 * i], r[2 * i + 1]);
    }
    let mut b = [z; 16];
    for j in 0..4 {
        b[4 * j] = _mm_unpacklo_epi16(a[4 * j], a[4 * j + 2]);
        b[4 * j + 1] = _mm_unpackhi_epi16(a[4 * j], a[4 * j + 2]);
        b[4 * j + 2] = _mm_unpacklo_epi16(a[4 * j + 1], a[4 * j + 3]);
        b[4 * j + 3] = _mm_unpackhi_epi16(a[4 * j + 1], a[4 * j + 3]);
    }
    let mut c = [z; 16];
    for g in 0..2 {
        for q in 0..4 {
            c[8 * g + 2 * q] = _mm_unpacklo_epi32(b[8 * g + q], b[8 * g + 4 + q]);
            c[8 * g + 2 * q + 1] = _mm_unpackhi_epi32(b[8 * g + q], b[8 * g + 4 + q]);
        }
    }
    let mut o = [z; 16];
    for q in 0..4 {
        o[4 * q] = _mm_unpacklo_epi64(c[2 * q], c[8 + 2 * q]);
        o[4 * q + 1] = _mm_unpackhi_epi64(c[2 * q], c[8 + 2 * q]);
        o[4 * q + 2] = _mm_unpacklo_epi64(c[2 * q + 1], c[8 + 2 * q + 1]);
        o[4 * q + 3] = _mm_unpackhi_epi64(c[2 * q + 1], c[8 + 2 * q + 1]);
    }
    o
}

#[inline]
#[target_feature(enable = "avx2")]
fn thr_vec(thr: &[Thresh; 2], f: impl Fn(&Thresh) -> u8) -> __m256i {
    let lo = i16::from(f(&thr[0]));
    let hi = i16::from(f(&thr[1]));
    _mm256_set_m128i(
        core::arch::x86_64::_mm_set1_epi16(hi),
        core::arch::x86_64::_mm_set1_epi16(lo),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn ad(a: __m256i, b: __m256i) -> __m256i {
    _mm256_abs_epi16(_mm256_sub_epi16(a, b))
}

/// Filters the taps `v` (p7 at 0, q0 at 8) in place.
#[inline]
#[target_feature(enable = "avx2")]
fn lanes(v: &mut [__m256i; 16], wd: Wd, thr: &[Thresh; 2]) {
    let (p3, p2, p1, p0) = (v[4], v[5], v[6], v[7]);
    let (q0, q1, q2, q3) = (v[8], v[9], v[10], v[11]);
    let limit = thr_vec(thr, |t| t.limit);
    let blimit = thr_vec(thr, |t| t.blimit);
    let hevt = thr_vec(thr, |t| t.hev);
    let one = _mm256_set1_epi16(1);
    let m = _mm256_max_epi16(
        _mm256_max_epi16(_mm256_max_epi16(ad(p3, p2), ad(p2, p1)), _mm256_max_epi16(ad(p1, p0), ad(q1, q0))),
        _mm256_max_epi16(ad(q2, q1), ad(q3, q2)),
    );
    let e = _mm256_add_epi16(_mm256_slli_epi16::<1>(ad(p0, q0)), _mm256_srli_epi16::<1>(ad(p1, q1)));
    let fail = _mm256_or_si256(_mm256_cmpgt_epi16(m, limit), _mm256_cmpgt_epi16(e, blimit));
    // mask: all ones where the edge is filtered.
    let all = _mm256_cmpgt_epi16(one, _mm256_set1_epi16(0));
    let mask = _mm256_andnot_si256(fail, all);
    let hev = _mm256_or_si256(_mm256_cmpgt_epi16(ad(p1, p0), hevt), _mm256_cmpgt_epi16(ad(q1, q0), hevt));
    // filter4, in the signed domain.
    let c128 = _mm256_set1_epi16(128);
    let lo = _mm256_set1_epi16(-128);
    let hi = _mm256_set1_epi16(127);
    let cl = |x: __m256i| _mm256_max_epi16(_mm256_min_epi16(x, hi), lo);
    let (ps1, ps0) = (_mm256_sub_epi16(p1, c128), _mm256_sub_epi16(p0, c128));
    let (qs0, qs1) = (_mm256_sub_epi16(q0, c128), _mm256_sub_epi16(q1, c128));
    let f = _mm256_and_si256(cl(_mm256_sub_epi16(ps1, qs1)), hev);
    let d = _mm256_sub_epi16(qs0, ps0);
    let f = cl(_mm256_add_epi16(f, _mm256_add_epi16(d, _mm256_add_epi16(d, d))));
    let f = _mm256_and_si256(f, mask);
    let f1 = _mm256_srai_epi16::<3>(cl(_mm256_add_epi16(f, _mm256_set1_epi16(4))));
    let f2 = _mm256_srai_epi16::<3>(cl(_mm256_add_epi16(f, _mm256_set1_epi16(3))));
    let nq0 = _mm256_add_epi16(cl(_mm256_sub_epi16(qs0, f1)), c128);
    let np0 = _mm256_add_epi16(cl(_mm256_add_epi16(ps0, f2)), c128);
    let fo = _mm256_andnot_si256(hev, _mm256_srai_epi16::<1>(_mm256_add_epi16(f1, one)));
    let nq1 = _mm256_add_epi16(cl(_mm256_sub_epi16(qs1, fo)), c128);
    let np1 = _mm256_add_epi16(cl(_mm256_add_epi16(ps1, fo)), c128);
    v[6] = np1;
    v[7] = np0;
    v[8] = nq0;
    v[9] = nq1;
    if wd == Wd::W4 {
        return;
    }
    let fm = _mm256_max_epi16(
        _mm256_max_epi16(_mm256_max_epi16(ad(p1, p0), ad(q1, q0)), _mm256_max_epi16(ad(p2, p0), ad(q2, q0))),
        _mm256_max_epi16(ad(p3, p0), ad(q3, q0)),
    );
    let flat = _mm256_andnot_si256(_mm256_cmpgt_epi16(fm, one), mask);
    let add = |a: __m256i, b: __m256i| _mm256_add_epi16(a, b);
    let r3 = |x: __m256i| _mm256_srli_epi16::<3>(add(x, _mm256_set1_epi16(4)));
    // 7-tap sums.
    let o2 = r3(add(add(add(p3, p3), add(p3, p2)), add(add(p2, p1), add(p0, q0))));
    let o1 = r3(add(add(add(p3, p3), add(p2, p1)), add(add(p1, p0), add(q0, q1))));
    let o0 = r3(add(add(add(p3, p2), add(p1, p0)), add(add(p0, q0), add(q1, q2))));
    let oq0 = r3(add(add(add(p2, p1), add(p0, q0)), add(add(q0, q1), add(q2, q3))));
    let oq1 = r3(add(add(add(p1, p0), add(q0, q1)), add(add(q1, q2), add(q3, q3))));
    let oq2 = r3(add(add(add(p0, q0), add(q1, q2)), add(add(q2, q3), add(q3, q3))));
    v[5] = _mm256_blendv_epi8(p2, o2, flat);
    v[6] = _mm256_blendv_epi8(np1, o1, flat);
    v[7] = _mm256_blendv_epi8(np0, o0, flat);
    v[8] = _mm256_blendv_epi8(nq0, oq0, flat);
    v[9] = _mm256_blendv_epi8(nq1, oq1, flat);
    v[10] = _mm256_blendv_epi8(q2, oq2, flat);
    if wd != Wd::W16 {
        return;
    }
    let s = [
        v[0], v[1], v[2], v[3], p3, p2, p1, p0, q0, q1, q2, q3, v[12], v[13], v[14], v[15],
    ];
    let f2m = _mm256_max_epi16(
        _mm256_max_epi16(_mm256_max_epi16(ad(s[3], p0), ad(s[2], p0)), _mm256_max_epi16(ad(s[1], p0), ad(s[0], p0))),
        _mm256_max_epi16(_mm256_max_epi16(ad(s[12], q0), ad(s[13], q0)), _mm256_max_epi16(ad(s[14], q0), ad(s[15], q0))),
    );
    let flat2 = _mm256_andnot_si256(_mm256_cmpgt_epi16(f2m, one), flat);
    // The 15-tap window as a running sum, as libvpx writes it out.
    let mut sum = add(
        add(_mm256_slli_epi16::<3>(s[0]), _mm256_sub_epi16(s[1], s[0])),
        add(add(s[1], s[2]), add(add(s[3], s[4]), add(add(s[5], s[6]), add(s[7], s[8])))),
    );
    let r4 = |x: __m256i| _mm256_srli_epi16::<4>(add(x, _mm256_set1_epi16(8)));
    let mut out = [s[0]; 14];
    out[0] = r4(sum);
    for k in 1..14 {
        let drop_a = s[if k <= 7 { 0 } else { k - 7 }];
        let drop_b = s[k];
        let add_a = s[k + 1];
        let add_b = s[(k + 8).min(15)];
        sum = _mm256_sub_epi16(add(sum, add(add_a, add_b)), add(drop_a, drop_b));
        out[k] = r4(sum);
    }
    for k in 0..14 {
        v[k + 1] = _mm256_blendv_epi8(v[k + 1], out[k], flat2);
    }
}

#[target_feature(enable = "avx2")]
pub(super) fn filter(e: Edge<'_>) {
    let Edge {
        buf,
        at,
        stride,
        vertical,
        n,
        wd,
        thr,
    } = e;
    let z = _mm_setzero_si128();
    let zz = _mm256_cvtepu8_epi16(z);
    if vertical {
        let mut rows = [z; 16];
        for (i, r) in rows.iter_mut().enumerate().take(n) {
            *r = ld16(buf, at + i * stride - 8);
        }
        let cols = transpose(&rows);
        let mut v: [__m256i; 16] = core::array::from_fn(|k| _mm256_cvtepu8_epi16(cols[k]));
        lanes(&mut v, wd, &thr);
        let back: [__m128i; 16] = core::array::from_fn(|k| pack(v[k]));
        let rows = transpose(&back);
        for (i, r) in rows.iter().enumerate().take(n) {
            st16(buf, at + i * stride - 8, *r);
        }
    } else {
        let taps = if wd == Wd::W16 { 8 } else { 4 };
        let mut v = [zz; 16];
        for k in 0..2 * taps {
            let row = at + k * stride - taps * stride;
            let r = if n == 16 { ld16(buf, row) } else { ld8(buf, row) };
            v[k + 8 - taps] = _mm256_cvtepu8_epi16(r);
        }
        lanes(&mut v, wd, &thr);
        let (first, last) = match wd {
            Wd::W4 => (6, 10),
            Wd::W8 => (5, 11),
            Wd::W16 => (1, 15),
        };
        for k in first..last {
            let row = at + (k + taps - 8) * stride - taps * stride;
            let p = pack(v[k]);
            if n == 16 {
                st16(buf, row, p);
            } else {
                st8(buf, row, p);
            }
        }
    }
    let _ = zz;
}
