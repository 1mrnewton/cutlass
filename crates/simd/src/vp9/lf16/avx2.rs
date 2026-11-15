//! The 10- and 12-bit loop filter on AVX2.
//!
//! The 16 lines of an edge are the 16 lanes of a vector of 16-bit
//! samples, one vector for each position across the edge (`p7` to `q7`,
//! or `p3` to `q3`); lines 0 to 7 are the low 128 bits, with the first
//! half's thresholds. A horizontal edge loads its rows as they are; a
//! vertical one loads 16 rows of 8 or 16 samples and transposes them in
//! 8×8 blocks, rows `i` and `i + 8` sharing a vector (one in each
//! 128-bit lane, which the unpacks keep apart), and back again to store.
//!
//! Every filter is worked out for every line and the result chosen by
//! the masks, as libvpx's vector code does. With samples below 4096 the
//! differences, the filter-4 terms (at most 3 × 4095 + 2048 before their
//! clamps) and the 8-tap sums (at most 8 × 4095 + 4) fit signed 16-bit
//! lanes; the 16-tap sums reach 16 × 4095 + 8 = 65528, which fits
//! unsigned 16-bit lanes, so they are added with wrapping adds and
//! shifted logically: the same samples as the scalar twin.
//!
//! Loads and stores are unaligned moves of 8 or 16 samples on the region
//! [`Edge::fits`](super::Edge) checked: for a horizontal edge, rows
//! `at - taps * stride` to `at + (taps - 1) * stride`, 16 samples each;
//! for a vertical one, rows `at + i * stride` (`i < 16`), from `taps`
//! before the edge to `taps` after it.

use core::arch::x86_64::{
    __m128i, __m256i, _mm_loadu_si128, _mm_set1_epi16, _mm_storeu_si128, _mm256_abs_epi16,
    _mm256_add_epi16, _mm256_and_si256, _mm256_andnot_si256, _mm256_blendv_epi8,
    _mm256_castsi256_si128, _mm256_cmpgt_epi16, _mm256_extracti128_si256, _mm256_loadu_si256,
    _mm256_max_epi16, _mm256_min_epi16, _mm256_or_si256, _mm256_set_m128i, _mm256_set1_epi16,
    _mm256_setzero_si256, _mm256_slli_epi16, _mm256_srai_epi16, _mm256_srli_epi16,
    _mm256_storeu_si256, _mm256_sub_epi16, _mm256_unpackhi_epi16, _mm256_unpackhi_epi32,
    _mm256_unpackhi_epi64, _mm256_unpacklo_epi16, _mm256_unpacklo_epi32, _mm256_unpacklo_epi64,
};

use super::{Thresh, Wd};

/// # Safety
/// `o + 8 <= s.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn l128(s: &[u16], o: usize) -> __m128i {
    debug_assert!(o + 8 <= s.len());
    // SAFETY: the caller proves the 8 samples lie in `s`.
    unsafe { _mm_loadu_si128(s.as_ptr().add(o).cast()) }
}

/// # Safety
/// `o + 16 <= s.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn l256(s: &[u16], o: usize) -> __m256i {
    debug_assert!(o + 16 <= s.len());
    // SAFETY: the caller proves the 16 samples lie in `s`.
    unsafe { _mm256_loadu_si256(s.as_ptr().add(o).cast()) }
}

/// # Safety
/// `o + 8 <= s.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn s128(s: &mut [u16], o: usize, v: __m128i) {
    debug_assert!(o + 8 <= s.len());
    // SAFETY: the caller proves the 8 samples lie in `s`.
    unsafe { _mm_storeu_si128(s.as_mut_ptr().add(o).cast(), v) }
}

/// # Safety
/// `o + 16 <= s.len()`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn s256(s: &mut [u16], o: usize, v: __m256i) {
    debug_assert!(o + 16 <= s.len());
    // SAFETY: the caller proves the 16 samples lie in `s`.
    unsafe { _mm256_storeu_si256(s.as_mut_ptr().add(o).cast(), v) }
}

/// The 8×8 transpose of 16-bit samples within each 128-bit lane.
#[inline]
#[target_feature(enable = "avx2")]
fn transpose8(a: [__m256i; 8]) -> [__m256i; 8] {
    let t0 = _mm256_unpacklo_epi16(a[0], a[1]);
    let t1 = _mm256_unpackhi_epi16(a[0], a[1]);
    let t2 = _mm256_unpacklo_epi16(a[2], a[3]);
    let t3 = _mm256_unpackhi_epi16(a[2], a[3]);
    let t4 = _mm256_unpacklo_epi16(a[4], a[5]);
    let t5 = _mm256_unpackhi_epi16(a[4], a[5]);
    let t6 = _mm256_unpacklo_epi16(a[6], a[7]);
    let t7 = _mm256_unpackhi_epi16(a[6], a[7]);
    let u0 = _mm256_unpacklo_epi32(t0, t2);
    let u1 = _mm256_unpackhi_epi32(t0, t2);
    let u2 = _mm256_unpacklo_epi32(t1, t3);
    let u3 = _mm256_unpackhi_epi32(t1, t3);
    let u4 = _mm256_unpacklo_epi32(t4, t6);
    let u5 = _mm256_unpackhi_epi32(t4, t6);
    let u6 = _mm256_unpacklo_epi32(t5, t7);
    let u7 = _mm256_unpackhi_epi32(t5, t7);
    [
        _mm256_unpacklo_epi64(u0, u4),
        _mm256_unpackhi_epi64(u0, u4),
        _mm256_unpacklo_epi64(u1, u5),
        _mm256_unpackhi_epi64(u1, u5),
        _mm256_unpacklo_epi64(u2, u6),
        _mm256_unpackhi_epi64(u2, u6),
        _mm256_unpacklo_epi64(u3, u7),
        _mm256_unpackhi_epi64(u3, u7),
    ]
}

/// Columns `o..o + 8` of rows `at + i * stride` (`i < 16`) as eight
/// vectors of 16 lines.
///
/// # Safety
/// Each of the 16 rows holds 8 samples from `o`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn load_cols(b: &[u16], o: usize, stride: usize) -> [__m256i; 8] {
    // SAFETY: as the caller proves.
    let rows = core::array::from_fn(|i| unsafe {
        _mm256_set_m128i(l128(b, o + (i + 8) * stride), l128(b, o + i * stride))
    });
    transpose8(rows)
}

/// The inverse of [`load_cols`].
///
/// # Safety
/// As for [`load_cols`].
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn store_cols(b: &mut [u16], o: usize, stride: usize, v: [__m256i; 8]) {
    let rows = transpose8(v);
    for (i, r) in rows.iter().enumerate() {
        // SAFETY: as the caller proves.
        unsafe {
            s128(b, o + i * stride, _mm256_castsi256_si128(*r));
            s128(b, o + (i + 8) * stride, _mm256_extracti128_si256::<1>(*r));
        }
    }
}

/// A vector of `a` for lines 0 to 7 and `b` for lines 8 to 15.
#[inline]
#[target_feature(enable = "avx2")]
fn halves(a: i16, b: i16) -> __m256i {
    _mm256_set_m128i(_mm_set1_epi16(b), _mm_set1_epi16(a))
}

#[inline]
#[target_feature(enable = "avx2")]
fn ad(a: __m256i, b: __m256i) -> __m256i {
    _mm256_abs_epi16(_mm256_sub_epi16(a, b))
}

/// Filters the samples `s` across the edge: `s[k]` is position
/// `k - N / 2` (`N` of 8 or 16, `s[N / 2]` is `q0`).
#[inline]
#[target_feature(enable = "avx2")]
fn filt<const N: usize>(s: &mut [__m256i; N], wd: [Option<Wd>; 2], thr: [Thresh; 2], bd: u32) {
    let c = N / 2;
    let (p3, p2, p1, p0) = (s[c - 4], s[c - 3], s[c - 2], s[c - 1]);
    let (q0, q1, q2, q3) = (s[c], s[c + 1], s[c + 2], s[c + 3]);
    let sh = bd - 8;
    let th =
        |f: fn(&Thresh) -> u8| halves((i16::from(f(&thr[0]))) << sh, (i16::from(f(&thr[1]))) << sh);
    let limit = th(|t| t.limit);
    let blimit = th(|t| t.blimit);
    let hev_t = th(|t| t.hev);
    let on = |k: fn(Wd) -> bool| {
        halves(
            -i16::from(wd[0].is_some_and(k)),
            -i16::from(wd[1].is_some_and(k)),
        )
    };
    let active = on(|_| true);
    let flat_ok = on(|w| w != Wd::W4);
    let flat2_ok = on(|w| w == Wd::W16);
    let one = _mm256_set1_epi16(1 << sh);

    let m = _mm256_max_epi16(
        _mm256_max_epi16(_mm256_max_epi16(ad(p3, p2), ad(p2, p1)), ad(p1, p0)),
        _mm256_max_epi16(_mm256_max_epi16(ad(q1, q0), ad(q2, q1)), ad(q3, q2)),
    );
    let edge = _mm256_add_epi16(
        _mm256_slli_epi16::<1>(ad(p0, q0)),
        _mm256_srli_epi16::<1>(ad(p1, q1)),
    );
    let bad = _mm256_or_si256(
        _mm256_cmpgt_epi16(m, limit),
        _mm256_cmpgt_epi16(edge, blimit),
    );
    let mask = _mm256_andnot_si256(bad, active);
    let f = _mm256_max_epi16(
        _mm256_max_epi16(_mm256_max_epi16(ad(p1, p0), ad(q1, q0)), ad(p2, p0)),
        _mm256_max_epi16(_mm256_max_epi16(ad(q2, q0), ad(p3, p0)), ad(q3, q0)),
    );
    let flat = _mm256_andnot_si256(_mm256_cmpgt_epi16(f, one), _mm256_and_si256(mask, flat_ok));

    // Filter 4.
    let half = _mm256_set1_epi16(0x80 << sh);
    let lo = _mm256_set1_epi16(-(128i16 << sh));
    let hi = _mm256_set1_epi16((128i16 << sh) - 1);
    let cl = |v: __m256i| _mm256_min_epi16(_mm256_max_epi16(v, lo), hi);
    let (ps1, ps0) = (_mm256_sub_epi16(p1, half), _mm256_sub_epi16(p0, half));
    let (qs0, qs1) = (_mm256_sub_epi16(q0, half), _mm256_sub_epi16(q1, half));
    let hev = _mm256_cmpgt_epi16(_mm256_max_epi16(ad(p1, p0), ad(q1, q0)), hev_t);
    let fv = _mm256_and_si256(cl(_mm256_sub_epi16(ps1, qs1)), hev);
    let d = _mm256_sub_epi16(qs0, ps0);
    let fv = cl(_mm256_add_epi16(
        fv,
        _mm256_add_epi16(d, _mm256_add_epi16(d, d)),
    ));
    let f1 = _mm256_srai_epi16::<3>(cl(_mm256_add_epi16(fv, _mm256_set1_epi16(4))));
    let f2 = _mm256_srai_epi16::<3>(cl(_mm256_add_epi16(fv, _mm256_set1_epi16(3))));
    let n_q0 = _mm256_add_epi16(cl(_mm256_sub_epi16(qs0, f1)), half);
    let n_p0 = _mm256_add_epi16(cl(_mm256_add_epi16(ps0, f2)), half);
    let ff = _mm256_srai_epi16::<1>(_mm256_add_epi16(f1, _mm256_set1_epi16(1)));
    let n_q1 = _mm256_blendv_epi8(
        _mm256_add_epi16(cl(_mm256_sub_epi16(qs1, ff)), half),
        q1,
        hev,
    );
    let n_p1 = _mm256_blendv_epi8(
        _mm256_add_epi16(cl(_mm256_add_epi16(ps1, ff)), half),
        p1,
        hev,
    );

    // Filter 8: sums of at most 8 × 4095 + 4.
    let a = |x: __m256i, y: __m256i| _mm256_add_epi16(x, y);
    let r8 = |v: __m256i| _mm256_srli_epi16::<3>(_mm256_add_epi16(v, _mm256_set1_epi16(4)));
    let p3x2 = a(p3, p3);
    let q3x2 = a(q3, q3);
    let f8_p2 = r8(a(a(a(p3x2, p3), a(p2, p2)), a(a(p1, p0), q0)));
    let f8_p1 = r8(a(a(a(p3x2, p2), a(p1, p1)), a(a(p0, q0), q1)));
    let f8_p0 = r8(a(a(a(p3, p2), a(p1, a(p0, p0))), a(a(q0, q1), q2)));
    let f8_q0 = r8(a(a(a(p2, p1), a(p0, a(q0, q0))), a(a(q1, q2), q3)));
    let f8_q1 = r8(a(a(a(p1, p0), a(q0, a(q1, q1))), a(q2, q3x2)));
    let f8_q2 = r8(a(a(a(p0, q0), a(q1, a(q2, q2))), a(q3x2, q3)));

    let pick = |orig: __m256i, f4: Option<__m256i>, f8: __m256i| {
        let v = match f4 {
            Some(f4) => _mm256_blendv_epi8(orig, f4, mask),
            None => orig,
        };
        _mm256_blendv_epi8(v, f8, flat)
    };
    let out_p2 = pick(p2, None, f8_p2);
    let out_p1 = pick(p1, Some(n_p1), f8_p1);
    let out_p0 = pick(p0, Some(n_p0), f8_p0);
    let out_q0 = pick(q0, Some(n_q0), f8_q0);
    let out_q1 = pick(q1, Some(n_q1), f8_q1);
    let out_q2 = pick(q2, None, f8_q2);

    if N == 16 {
        // Filter 16: the running sum of the scalar twin, in wrapping
        // 16-bit lanes (its true value is below 65536).
        let mut f2m = _mm256_setzero_si256();
        for k in 4..8 {
            f2m = _mm256_max_epi16(
                f2m,
                _mm256_max_epi16(ad(s[c - 1 - k], p0), ad(s[c + k], q0)),
            );
        }
        let flat2 = _mm256_andnot_si256(
            _mm256_cmpgt_epi16(f2m, one),
            _mm256_and_si256(flat, flat2_ok),
        );
        let mut sum = a(
            a(
                _mm256_sub_epi16(_mm256_slli_epi16::<3>(s[0]), s[0]),
                a(s[1], s[1]),
            ),
            a(a(a(s[2], s[3]), a(s[4], s[5])), a(a(s[6], s[7]), s[8])),
        );
        let r16 = |v: __m256i| _mm256_srli_epi16::<4>(_mm256_add_epi16(v, _mm256_set1_epi16(8)));
        let mut out16 = [_mm256_setzero_si256(); 14];
        out16[0] = r16(sum);
        for k in 1usize..14 {
            let drop_a = s[k.saturating_sub(7)];
            let drop_b = s[k];
            let add_a = s[k + 1];
            let add_b = s[(k + 8).min(15)];
            sum = _mm256_sub_epi16(_mm256_add_epi16(sum, a(add_a, add_b)), a(drop_a, drop_b));
            out16[k] = r16(sum);
        }
        // Positions p6 to q6 are s[1] to s[14].
        s[c - 3] = out_p2;
        s[c - 2] = out_p1;
        s[c - 1] = out_p0;
        s[c] = out_q0;
        s[c + 1] = out_q1;
        s[c + 2] = out_q2;
        for k in 1..15 {
            s[k] = _mm256_blendv_epi8(s[k], out16[k - 1], flat2);
        }
    } else {
        s[c - 3] = out_p2;
        s[c - 2] = out_p1;
        s[c - 1] = out_p0;
        s[c] = out_q0;
        s[c + 1] = out_q1;
        s[c + 2] = out_q2;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
pub(super) fn filter(
    buf: &mut [u16],
    at: usize,
    stride: usize,
    vertical: bool,
    wd: [Option<Wd>; 2],
    thr: [Thresh; 2],
    bd: u32,
) {
    let wide = super::widest(wd) == Some(Wd::W16);
    // SAFETY: `fits` checked the region (the module's notes).
    unsafe {
        match (vertical, wide) {
            (false, false) => {
                let mut s: [__m256i; 8] =
                    core::array::from_fn(|k| l256(buf, at + k * stride - 4 * stride));
                filt::<8>(&mut s, wd, thr, bd);
                // Rows 1 to 6, by index: the iterator adapters stayed
                // out of line.
                let mut k = 1;
                while k < 7 {
                    s256(buf, at + k * stride - 4 * stride, s[k]);
                    k += 1;
                }
            }
            (false, true) => {
                let mut s: [__m256i; 16] =
                    core::array::from_fn(|k| l256(buf, at + k * stride - 8 * stride));
                filt::<16>(&mut s, wd, thr, bd);
                // Rows 1 to 14, by index: the iterator adapters stayed
                // out of line.
                let mut k = 1;
                while k < 15 {
                    s256(buf, at + k * stride - 8 * stride, s[k]);
                    k += 1;
                }
            }
            (true, false) => {
                let mut s = load_cols(buf, at - 4, stride);
                filt::<8>(&mut s, wd, thr, bd);
                store_cols(buf, at - 4, stride, s);
            }
            (true, true) => {
                let l = load_cols(buf, at - 8, stride);
                let r = load_cols(buf, at, stride);
                let mut s: [__m256i; 16] =
                    core::array::from_fn(|k| if k < 8 { l[k] } else { r[k - 8] });
                filt::<16>(&mut s, wd, thr, bd);
                store_cols(buf, at - 8, stride, core::array::from_fn(|k| s[k]));
                store_cols(buf, at, stride, core::array::from_fn(|k| s[k + 8]));
            }
        }
    }
}
