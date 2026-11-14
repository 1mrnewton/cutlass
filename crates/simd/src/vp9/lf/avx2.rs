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
fn st16(d: &mut [u8], at: usize, v: __m128i) {
    d[at..at + 8].copy_from_slice(&_mm_cvtsi128_si64(v).to_le_bytes());
    d[at + 8..at + 16].copy_from_slice(&_mm_extract_epi64::<1>(v).to_le_bytes());
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

/// All ones in the lanes of the halves `on` picks (0 to 7, 8 to 15).
#[inline]
#[target_feature(enable = "avx2")]
fn halves(on: [bool; 2]) -> __m256i {
    let m = |b: bool| core::arch::x86_64::_mm_set1_epi16(if b { -1 } else { 0 });
    _mm256_set_m128i(m(on[1]), m(on[0]))
}

/// Filters the taps `v` (p7 at 0, q0 at 8) in place, each half of the
/// lanes with its own filter.
#[inline]
#[target_feature(enable = "avx2")]
fn lanes(v: &mut [__m256i; 16], wds: [Option<Wd>; 2], wd: Wd, thr: &[Thresh; 2]) {
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
    let on = halves([wds[0].is_some(), wds[1].is_some()]);
    let mask = _mm256_andnot_si256(fail, on);
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
    let w8 = halves(wds.map(|w| matches!(w, Some(Wd::W8 | Wd::W16))));
    let flat = _mm256_and_si256(_mm256_andnot_si256(_mm256_cmpgt_epi16(fm, one), mask), w8);
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
    let w16 = halves(wds.map(|w| w == Some(Wd::W16)));
    let flat2 = _mm256_and_si256(_mm256_andnot_si256(_mm256_cmpgt_epi16(f2m, one), flat), w16);
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

/// The transpose of 16 rows of 8 bytes (the low half of each): 8
/// vectors of 16, one per column.
#[inline]
#[target_feature(enable = "avx2")]
fn transpose16x8(r: &[__m128i; 16]) -> [__m128i; 8] {
    let z = _mm_setzero_si128();
    let a: [__m128i; 8] = core::array::from_fn(|i| _mm_unpacklo_epi8(r[2 * i], r[2 * i + 1]));
    let mut b = [z; 8];
    for j in 0..4 {
        b[2 * j] = _mm_unpacklo_epi16(a[2 * j], a[2 * j + 1]);
        b[2 * j + 1] = _mm_unpackhi_epi16(a[2 * j], a[2 * j + 1]);
    }
    let mut c = [z; 8];
    for g in 0..2 {
        for q in 0..2 {
            c[4 * g + 2 * q] = _mm_unpacklo_epi32(b[4 * g + q], b[4 * g + 2 + q]);
            c[4 * g + 2 * q + 1] = _mm_unpackhi_epi32(b[4 * g + q], b[4 * g + 2 + q]);
        }
    }
    let mut o = [z; 8];
    for q in 0..2 {
        for t in 0..2 {
            o[4 * q + 2 * t] = _mm_unpacklo_epi64(c[2 * q + t], c[4 + 2 * q + t]);
            o[4 * q + 2 * t + 1] = _mm_unpackhi_epi64(c[2 * q + t], c[4 + 2 * q + t]);
        }
    }
    o
}

/// The transpose back: 8 columns of 16 bytes to 16 rows of 8 (each row's
/// 8 bytes in the low half, then the high half, of the 8 vectors: row
/// `2i` in the low half of vector `i`, row `2i + 1` in its high half).
#[inline]
#[target_feature(enable = "avx2")]
fn transpose8x16(c: &[__m128i; 8]) -> [__m128i; 8] {
    let z = _mm_setzero_si128();
    let mut p = [z; 8];
    for k in 0..4 {
        p[2 * k] = _mm_unpacklo_epi8(c[2 * k], c[2 * k + 1]);
        p[2 * k + 1] = _mm_unpackhi_epi8(c[2 * k], c[2 * k + 1]);
    }
    let mut q = [z; 8];
    for h in 0..2 {
        q[4 * h] = _mm_unpacklo_epi16(p[h], p[2 + h]);
        q[4 * h + 1] = _mm_unpackhi_epi16(p[h], p[2 + h]);
        q[4 * h + 2] = _mm_unpacklo_epi16(p[4 + h], p[6 + h]);
        q[4 * h + 3] = _mm_unpackhi_epi16(p[4 + h], p[6 + h]);
    }
    let mut o = [z; 8];
    for h in 0..2 {
        for k in 0..2 {
            o[4 * h + 2 * k] = _mm_unpacklo_epi32(q[4 * h + k], q[4 * h + 2 + k]);
            o[4 * h + 2 * k + 1] = _mm_unpackhi_epi32(q[4 * h + k], q[4 * h + 2 + k]);
        }
    }
    o
}

#[inline]
#[target_feature(enable = "avx2")]
fn ld8a(s: &[u8], at: usize) -> __m128i {
    match s.get(at..).and_then(|t| t.first_chunk::<8>()) {
        Some(a) => _mm_cvtsi64_si128(i64::from_le_bytes(*a)),
        None => _mm_setzero_si128(),
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn ld16a(s: &[u8], at: usize) -> __m128i {
    match s.get(at..).and_then(|t| t.first_chunk::<16>()) {
        Some(a) => {
            let lo = i64::from_le_bytes([a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]]);
            let hi = i64::from_le_bytes([a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]]);
            _mm_set_epi64x(hi, lo)
        }
        None => _mm_setzero_si128(),
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn st8a(d: &mut [u8], at: usize, v: i64) {
    if let Some(t) = d.get_mut(at..).and_then(|t| t.first_chunk_mut::<8>()) {
        *t = v.to_le_bytes();
    }
}

#[target_feature(enable = "avx2")]
pub(super) fn filter(e: Edge<'_>) {
    let Some(wd) = e.widest() else { return };
    let Edge {
        buf,
        at,
        stride,
        vertical,
        wd: wds,
        thr,
    } = e;
    let z = _mm_setzero_si128();
    let zz = _mm256_cvtepu8_epi16(z);
    if vertical && wd != Wd::W16 {
        let rows: [__m128i; 16] = core::array::from_fn(|i| ld8a(buf, at + i * stride - 4));
        let cols = transpose16x8(&rows);
        let mut v = [zz; 16];
        for k in 0..8 {
            v[4 + k] = _mm256_cvtepu8_epi16(cols[k]);
        }
        lanes(&mut v, wds, wd, &thr);
        let back: [__m128i; 8] = core::array::from_fn(|k| pack(v[4 + k]));
        let out = transpose8x16(&back);
        for (i, o) in out.iter().enumerate() {
            st8a(buf, at + 2 * i * stride - 4, _mm_cvtsi128_si64(*o));
            st8a(buf, at + (2 * i + 1) * stride - 4, _mm_extract_epi64::<1>(*o));
        }
    } else if vertical {
        let rows: [__m128i; 16] = core::array::from_fn(|i| ld16a(buf, at + i * stride - 8));
        let cols = transpose(&rows);
        let mut v: [__m256i; 16] = core::array::from_fn(|k| _mm256_cvtepu8_epi16(cols[k]));
        lanes(&mut v, wds, wd, &thr);
        let back: [__m128i; 16] = core::array::from_fn(|k| pack(v[k]));
        let rows = transpose(&back);
        for (i, r) in rows.iter().enumerate() {
            st16(buf, at + i * stride - 8, *r);
        }
    } else {
        let taps = if wd == Wd::W16 { 8 } else { 4 };
        let mut v = [zz; 16];
        for k in 0..2 * taps {
            let row = at + k * stride - taps * stride;
            v[k + 8 - taps] = _mm256_cvtepu8_epi16(ld16a(buf, row));
        }
        lanes(&mut v, wds, wd, &thr);
        let (first, last) = match wd {
            Wd::W4 => (6, 10),
            Wd::W8 => (5, 11),
            Wd::W16 => (1, 15),
        };
        for k in first..last {
            let row = at + (k + taps - 8) * stride - taps * stride;
            st16(buf, row, pack(v[k]));
        }
    }
}
