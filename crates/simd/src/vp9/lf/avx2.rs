//! The 8-bit loop filter on AVX2, sixteen lines at a time in 16-bit
//! lanes: tap `k` of every line is one vector. A horizontal edge loads its
//! rows straight into those vectors; a vertical edge loads its lines and
//! transposes them, and back.

use core::arch::x86_64::{
    _mm_adds_epi8, _mm_adds_epu8, _mm_and_si128, _mm_andnot_si128, _mm_blendv_epi8,
    _mm_cmpeq_epi8, _mm_cmpgt_epi8, _mm_max_epu8, _mm_or_si128, _mm_set1_epi8, _mm_srli_epi16,
    _mm_subs_epi8, _mm_subs_epu8, _mm_testz_si128, _mm_xor_si128,
    __m128i, __m256i, _mm_cvtsi64_si128, _mm_cvtsi128_si64, _mm_extract_epi64, _mm_set_epi64x,
    _mm_setzero_si128, _mm_unpackhi_epi8, _mm_unpackhi_epi16, _mm_unpackhi_epi32,
    _mm_unpackhi_epi64, _mm_unpacklo_epi8, _mm_unpacklo_epi16, _mm_unpacklo_epi32,
    _mm_unpacklo_epi64, _mm256_add_epi16,
    _mm256_castsi256_si128, 
    _mm256_cvtepu8_epi16, 
    _mm256_packus_epi16, _mm256_permute4x64_epi64, _mm256_set1_epi16,
    _mm256_slli_epi16, _mm256_srli_epi16, _mm256_sub_epi16,
};

use super::{Edge, Thresh, Wd};

#[inline]
#[target_feature(enable = "avx2")]
fn st16(d: &mut [u8], at: usize, v: __m128i) {
    if let Some(t) = d.get_mut(at..).and_then(|t| t.first_chunk_mut::<16>()) {
        let (lo, hi) = (_mm_cvtsi128_si64(v).to_le_bytes(), _mm_extract_epi64::<1>(v).to_le_bytes());
        *t = [
            lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7], hi[0], hi[1], hi[2], hi[3], hi[4], hi[5], hi[6], hi[7],
        ];
    }
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

/// Bytes `lo` in lanes 0 to 7 and `hi` in lanes 8 to 15.
#[inline]
#[target_feature(enable = "avx2")]
fn split8(lo: u8, hi: u8) -> __m128i {
    _mm_set_epi64x(i64::from_le_bytes([hi; 8]), i64::from_le_bytes([lo; 8]))
}

/// All ones in the lanes of the halves `on` picks (0 to 7, 8 to 15).
#[inline]
#[target_feature(enable = "avx2")]
fn halves8(on: [bool; 2]) -> __m128i {
    split8(if on[0] { 0xff } else { 0 }, if on[1] { 0xff } else { 0 })
}

#[inline]
#[target_feature(enable = "avx2")]
fn absd(a: __m128i, b: __m128i) -> __m128i {
    _mm_or_si128(_mm_subs_epu8(a, b), _mm_subs_epu8(b, a))
}

/// All ones where `x <= t` (unsigned bytes).
#[inline]
#[target_feature(enable = "avx2")]
fn le(x: __m128i, t: __m128i) -> __m128i {
    _mm_cmpeq_epi8(_mm_subs_epu8(x, t), _mm_setzero_si128())
}

/// Signed bytes shifted right by `N` (3 or 1), as libvpx's SSE2 does it.
#[inline]
#[target_feature(enable = "avx2")]
fn sra8<const N: i32>(x: __m128i) -> __m128i {
    // The high bits the logical 16-bit shift lets in from the next byte
    // are masked off, and the sign bits put back.
    let neg = _mm_cmpgt_epi8(_mm_setzero_si128(), x);
    let keep = (0xffu8 >> N) as i8;
    let sign = !keep;
    let shifted = _mm_and_si128(_mm_srli_epi16::<N>(x), _mm_set1_epi8(keep));
    _mm_or_si128(shifted, _mm_and_si128(neg, _mm_set1_epi8(sign)))
}

/// Sixteen bytes to 16-bit lanes and back.
#[inline]
#[target_feature(enable = "avx2")]
fn wide16(x: __m128i) -> __m256i {
    _mm256_cvtepu8_epi16(x)
}

/// Filters the taps `v` (p7 at 0, q0 at 8; sixteen lines a vector) in
/// place, each half of the lines with its own filter (libvpx's SSE2
/// arithmetic, which equals its C).
#[inline]
#[target_feature(enable = "avx2")]
fn lanes(v: &mut [__m128i; 16], wds: [Option<Wd>; 2], wd: Wd, thr: &[Thresh; 2]) -> bool {
    let (p3, p2, p1, p0) = (v[4], v[5], v[6], v[7]);
    let (q0, q1, q2, q3) = (v[8], v[9], v[10], v[11]);
    let limit = split8(thr[0].limit, thr[1].limit);
    let blimit = split8(thr[0].blimit, thr[1].blimit);
    let hevt = split8(thr[0].hev, thr[1].hev);
    let ap = absd(p1, p0);
    let aq = absd(q1, q0);
    let m = _mm_max_epu8(
        _mm_max_epu8(_mm_max_epu8(absd(p3, p2), absd(p2, p1)), _mm_max_epu8(ap, aq)),
        _mm_max_epu8(absd(q2, q1), absd(q3, q2)),
    );
    let apq0 = absd(p0, q0);
    let apq1 = _mm_and_si128(_mm_srli_epi16::<1>(absd(p1, q1)), _mm_set1_epi8(0x7f));
    let e = _mm_adds_epu8(_mm_adds_epu8(apq0, apq0), apq1);
    let on = halves8([wds[0].is_some(), wds[1].is_some()]);
    let mask = _mm_and_si128(_mm_and_si128(le(m, limit), le(e, blimit)), on);
    if _mm_testz_si128(mask, mask) != 0 {
        return false;
    }
    let hev = _mm_xor_si128(le(_mm_max_epu8(ap, aq), hevt), _mm_set1_epi8(-1));
    // filter4, in signed bytes.
    let x80 = _mm_set1_epi8(-128);
    let (ps1, ps0) = (_mm_xor_si128(p1, x80), _mm_xor_si128(p0, x80));
    let (qs0, qs1) = (_mm_xor_si128(q0, x80), _mm_xor_si128(q1, x80));
    let f = _mm_and_si128(_mm_subs_epi8(ps1, qs1), hev);
    let d = _mm_subs_epi8(qs0, ps0);
    let f = _mm_adds_epi8(_mm_adds_epi8(_mm_adds_epi8(f, d), d), d);
    let f = _mm_and_si128(f, mask);
    let f1 = sra8::<3>(_mm_adds_epi8(f, _mm_set1_epi8(4)));
    let f2 = sra8::<3>(_mm_adds_epi8(f, _mm_set1_epi8(3)));
    let nq0 = _mm_xor_si128(_mm_subs_epi8(qs0, f1), x80);
    let np0 = _mm_xor_si128(_mm_adds_epi8(ps0, f2), x80);
    let fo = _mm_andnot_si128(hev, sra8::<1>(_mm_adds_epi8(f1, _mm_set1_epi8(1))));
    let nq1 = _mm_xor_si128(_mm_subs_epi8(qs1, fo), x80);
    let np1 = _mm_xor_si128(_mm_adds_epi8(ps1, fo), x80);
    v[6] = np1;
    v[7] = np0;
    v[8] = nq0;
    v[9] = nq1;
    if wd == Wd::W4 {
        return true;
    }
    let one = _mm_set1_epi8(1);
    let fm = _mm_max_epu8(
        _mm_max_epu8(_mm_max_epu8(ap, aq), _mm_max_epu8(absd(p2, p0), absd(q2, q0))),
        _mm_max_epu8(absd(p3, p0), absd(q3, q0)),
    );
    let w8 = halves8(wds.map(|w| matches!(w, Some(Wd::W8 | Wd::W16))));
    let flat = _mm_and_si128(_mm_and_si128(le(fm, one), mask), w8);
    if _mm_testz_si128(flat, flat) != 0 {
        return true;
    }
    // Where the 15-tap filter runs it replaces the 7-tap one.
    let flat2 = if wd == Wd::W16 {
        let f2m = _mm_max_epu8(
            _mm_max_epu8(_mm_max_epu8(absd(v[3], p0), absd(v[2], p0)), _mm_max_epu8(absd(v[1], p0), absd(v[0], p0))),
            _mm_max_epu8(_mm_max_epu8(absd(v[12], q0), absd(v[13], q0)), _mm_max_epu8(absd(v[14], q0), absd(v[15], q0))),
        );
        let w16 = halves8(wds.map(|w| w == Some(Wd::W16)));
        _mm_and_si128(_mm_and_si128(le(f2m, one), flat), w16)
    } else {
        _mm_setzero_si128()
    };
    let flat_only = _mm_andnot_si128(flat2, flat);
    if _mm_testz_si128(flat_only, flat_only) == 0 {
        let flat = flat_only;
        // The 7-tap sums in 16-bit lanes.
        let (w3, w2, w1, w0) = (wide16(p3), wide16(p2), wide16(p1), wide16(p0));
        let (x0, x1, x2, x3) = (wide16(q0), wide16(q1), wide16(q2), wide16(q3));
        let add = |a: __m256i, b: __m256i| _mm256_add_epi16(a, b);
        let r3 = |x: __m256i| pack(_mm256_srli_epi16::<3>(add(x, _mm256_set1_epi16(4))));
        let o2 = r3(add(add(add(w3, w3), add(w3, w2)), add(add(w2, w1), add(w0, x0))));
        let o1 = r3(add(add(add(w3, w3), add(w2, w1)), add(add(w1, w0), add(x0, x1))));
        let o0 = r3(add(add(add(w3, w2), add(w1, w0)), add(add(w0, x0), add(x1, x2))));
        let oq0 = r3(add(add(add(w2, w1), add(w0, x0)), add(add(x0, x1), add(x2, x3))));
        let oq1 = r3(add(add(add(w1, w0), add(x0, x1)), add(add(x1, x2), add(x3, x3))));
        let oq2 = r3(add(add(add(w0, x0), add(x1, x2)), add(add(x2, x3), add(x3, x3))));
        v[5] = _mm_blendv_epi8(p2, o2, flat);
        v[6] = _mm_blendv_epi8(np1, o1, flat);
        v[7] = _mm_blendv_epi8(np0, o0, flat);
        v[8] = _mm_blendv_epi8(nq0, oq0, flat);
        v[9] = _mm_blendv_epi8(nq1, oq1, flat);
        v[10] = _mm_blendv_epi8(q2, oq2, flat);
    }
    if _mm_testz_si128(flat2, flat2) != 0 {
        return true;
    }
    let s: [__m256i; 16] = [
        wide16(v[0]),
        wide16(v[1]),
        wide16(v[2]),
        wide16(v[3]),
        wide16(p3),
        wide16(p2),
        wide16(p1),
        wide16(p0),
        wide16(q0),
        wide16(q1),
        wide16(q2),
        wide16(q3),
        wide16(v[12]),
        wide16(v[13]),
        wide16(v[14]),
        wide16(v[15]),
    ];
    let add = |a: __m256i, b: __m256i| _mm256_add_epi16(a, b);
    // The 15-tap window as a running sum, as libvpx writes it out.
    let mut sum = add(
        add(_mm256_slli_epi16::<3>(s[0]), _mm256_sub_epi16(s[1], s[0])),
        add(add(s[1], s[2]), add(add(s[3], s[4]), add(add(s[5], s[6]), add(s[7], s[8])))),
    );
    let r4 = |x: __m256i| pack(_mm256_srli_epi16::<4>(add(x, _mm256_set1_epi16(8))));
    let mut out = [v[0]; 14];
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
        v[k + 1] = _mm_blendv_epi8(v[k + 1], out[k], flat2);
    }
    true
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
    if vertical && wd != Wd::W16 {
        let rows: [__m128i; 16] = core::array::from_fn(|i| ld8a(buf, at + i * stride - 4));
        let cols = transpose16x8(&rows);
        let mut v = [z; 16];
        v[4..12].copy_from_slice(&cols);
        if !lanes(&mut v, wds, wd, &thr) {
            return;
        }
        let back: [__m128i; 8] = core::array::from_fn(|k| v[4 + k]);
        let out = transpose8x16(&back);
        for (i, o) in out.iter().enumerate() {
            st8a(buf, at + 2 * i * stride - 4, _mm_cvtsi128_si64(*o));
            st8a(buf, at + (2 * i + 1) * stride - 4, _mm_extract_epi64::<1>(*o));
        }
    } else if vertical {
        let rows: [__m128i; 16] = core::array::from_fn(|i| ld16a(buf, at + i * stride - 8));
        let mut v = transpose(&rows);
        if !lanes(&mut v, wds, wd, &thr) {
            return;
        }
        let rows = transpose(&v);
        for (i, r) in rows.iter().enumerate() {
            st16(buf, at + i * stride - 8, *r);
        }
    } else {
        let taps = if wd == Wd::W16 { 8 } else { 4 };
        let mut v = [z; 16];
        for k in 0..2 * taps {
            v[k + 8 - taps] = ld16a(buf, at + k * stride - taps * stride);
        }
        if !lanes(&mut v, wds, wd, &thr) {
            return;
        }
        let (first, last) = match wd {
            Wd::W4 => (6, 10),
            Wd::W8 => (5, 11),
            Wd::W16 => (1, 15),
        };
        for k in first..last {
            st16(buf, at + (k + taps - 8) * stride - taps * stride, v[k]);
        }
    }
}
