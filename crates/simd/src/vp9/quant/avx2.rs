//! [`dz`](super::dz) on AVX2, eight coefficients a vector. The lanes come
//! in and go out through values (no load or store intrinsics).

use core::arch::x86_64::{
    __m256, __m256i, _CMP_LT_OQ, _mm256_add_epi32, _mm256_add_ps, _mm256_and_ps, _mm256_blendv_ps,
    _mm256_castps_si256, _mm256_castsi256_ps, _mm256_cmp_ps, _mm256_extract_epi32,
    _mm256_min_epi32, _mm256_min_ps, _mm256_mul_ps, _mm256_or_si256, _mm256_set1_epi32,
    _mm256_set1_ps, _mm256_setr_ps, _mm256_setzero_ps, _mm256_setzero_si256, _mm256_slli_epi32,
    _mm256_sub_epi32, _mm256_sub_ps, _mm256_testz_si256,
};

use super::{CAP, MAGIC, Sums, one};

#[inline]
#[target_feature(enable = "avx2")]
fn lanes_f(v: __m256) -> [f32; 8] {
    let i = _mm256_castps_si256(v);
    [
        f32::from_bits(_mm256_extract_epi32::<0>(i) as u32),
        f32::from_bits(_mm256_extract_epi32::<1>(i) as u32),
        f32::from_bits(_mm256_extract_epi32::<2>(i) as u32),
        f32::from_bits(_mm256_extract_epi32::<3>(i) as u32),
        f32::from_bits(_mm256_extract_epi32::<4>(i) as u32),
        f32::from_bits(_mm256_extract_epi32::<5>(i) as u32),
        f32::from_bits(_mm256_extract_epi32::<6>(i) as u32),
        f32::from_bits(_mm256_extract_epi32::<7>(i) as u32),
    ]
}

#[inline]
#[target_feature(enable = "avx2")]
fn lanes_i(i: __m256i) -> [i32; 8] {
    [
        _mm256_extract_epi32::<0>(i),
        _mm256_extract_epi32::<1>(i),
        _mm256_extract_epi32::<2>(i),
        _mm256_extract_epi32::<3>(i),
        _mm256_extract_epi32::<4>(i),
        _mm256_extract_epi32::<5>(i),
        _mm256_extract_epi32::<6>(i),
        _mm256_extract_epi32::<7>(i),
    ]
}

#[target_feature(enable = "avx2")]
pub(super) fn dz(coef: &[f32], levels: &mut [i32], inv: f32, step: f32, round: f32) -> Sums {
    let rr = round - 0.5;
    let (vinv, vstep, vrr) = (
        _mm256_set1_ps(inv),
        _mm256_set1_ps(step),
        _mm256_set1_ps(rr),
    );
    let (vmagic, vcap) = (_mm256_set1_ps(MAGIC), _mm256_set1_ps(CAP));
    let abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    let (bias, one_i, sixteen) = (
        _mm256_set1_epi32(0x4b00_0000),
        _mm256_set1_epi32(1),
        _mm256_set1_epi32(16),
    );
    let (mut err, mut energy) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let (mut bits, mut nz) = (_mm256_setzero_si256(), _mm256_setzero_si256());
    let n = coef.len().min(levels.len());
    let whole = n / 8 * 8;
    for (c, l) in coef[..whole]
        .as_chunks::<8>()
        .0
        .iter()
        .zip(levels[..whole].as_chunks_mut::<8>().0)
    {
        let v = _mm256_setr_ps(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
        let a = _mm256_min_ps(_mm256_mul_ps(_mm256_and_ps(v, abs), vinv), vcap);
        let t = _mm256_add_ps(_mm256_add_ps(a, vrr), vmagic);
        let q = _mm256_sub_epi32(_mm256_castps_si256(t), bias);
        let e = _mm256_mul_ps(_mm256_sub_ps(a, _mm256_sub_ps(t, vmagic)), vstep);
        err = _mm256_add_ps(err, _mm256_mul_ps(e, e));
        energy = _mm256_add_ps(energy, _mm256_mul_ps(v, v));
        let m1 = _mm256_min_epi32(q, one_i);
        let b = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_slli_epi32::<2>(m1), m1),
            _mm256_min_epi32(q, sixteen),
        );
        bits = _mm256_add_epi32(bits, b);
        nz = _mm256_or_si256(nz, q);
        // The sign of the coefficient onto its level.
        let neg = _mm256_cmp_ps::<_CMP_LT_OQ>(v, _mm256_setzero_ps());
        let qn = _mm256_sub_epi32(_mm256_setzero_si256(), q);
        let signed = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(q),
            _mm256_castsi256_ps(qn),
            neg,
        ));
        l.copy_from_slice(&lanes_i(signed));
    }
    let mut s = Sums {
        err: lanes_f(err).iter().sum(),
        energy: lanes_f(energy).iter().sum(),
        bits: lanes_i(bits).iter().sum(),
        nonzero: _mm256_testz_si256(nz, nz) == 0,
    };
    for (l, &v) in levels[whole..n].iter_mut().zip(&coef[whole..n]) {
        let (q, e, b) = one(v, inv, step, rr);
        *l = q;
        s.err += e;
        s.energy += v * v;
        s.bits += b;
        s.nonzero |= q != 0;
    }
    s
}
