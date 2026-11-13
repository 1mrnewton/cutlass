//! SAO of 8-bit rows on AVX2, 32 samples at a time.
//!
//! The sign of a difference of two bytes is `(x > y) - (x < y)`, the
//! comparisons taken on bytes with their top bit flipped so that signed
//! compares order them as unsigned. The offset of each sample comes from
//! a 16-byte table by `pshufb`, and is added as two saturating steps, the
//! positive part then the negative one, which is the clipped sum.

use core::arch::x86_64::{
    __m256i, _mm_set_epi64x, _mm256_add_epi8, _mm256_adds_epu8, _mm256_and_si256,
    _mm256_cmpgt_epi8, _mm256_extract_epi64, _mm256_max_epi8, _mm256_min_epu8, _mm256_set_m128i,
    _mm256_set1_epi8, _mm256_setr_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8,
    _mm256_srli_epi16, _mm256_sub_epi8, _mm256_subs_epu8, _mm256_xor_si256,
};

/// Thirty-two bytes from `at` (zeros past the end of `s`).
#[inline]
#[target_feature(enable = "avx2")]
fn ld32(s: &[u8], at: usize) -> __m256i {
    let half = |at: usize| match s.get(at..).and_then(|t| t.first_chunk::<16>()) {
        Some(a) => {
            let v = u128::from_le_bytes(*a);
            _mm_set_epi64x((v >> 64) as i64, v as i64)
        }
        None => _mm_set_epi64x(0, 0),
    };
    _mm256_set_m128i(half(at + 16), half(at))
}

/// Thirty-two bytes of `v` to `d[at..]`.
#[inline]
#[target_feature(enable = "avx2")]
fn st32(d: &mut [u8], at: usize, v: __m256i) {
    let mut a = [0u8; 32];
    a[..8].copy_from_slice(&(_mm256_extract_epi64::<0>(v) as u64).to_le_bytes());
    a[8..16].copy_from_slice(&(_mm256_extract_epi64::<1>(v) as u64).to_le_bytes());
    a[16..24].copy_from_slice(&(_mm256_extract_epi64::<2>(v) as u64).to_le_bytes());
    a[24..].copy_from_slice(&(_mm256_extract_epi64::<3>(v) as u64).to_le_bytes());
    if let Some(d) = d.get_mut(at..).and_then(|d| d.first_chunk_mut::<32>()) {
        *d = a;
    }
}

/// `v + o`, clipped to bytes, for signed offsets `o`.
#[inline]
#[target_feature(enable = "avx2")]
fn add_clip(v: __m256i, o: __m256i) -> __m256i {
    let zero = _mm256_setzero_si256();
    let pos = _mm256_max_epi8(o, zero);
    let neg = _mm256_max_epi8(_mm256_sub_epi8(zero, o), zero);
    _mm256_subs_epu8(_mm256_adds_epu8(v, pos), neg)
}

/// A 16-byte table in both lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn table(t: [i8; 16]) -> __m256i {
    _mm256_setr_epi8(
        t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11], t[12], t[13],
        t[14], t[15], t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11],
        t[12], t[13], t[14], t[15],
    )
}

#[target_feature(enable = "avx2")]
pub(super) fn edge_row(dst: &mut [u8], cur: &[u8], (a, b): (&[u8], &[u8]), o: [i8; 4]) {
    let n = dst.len();
    let flip = _mm256_set1_epi8(i8::MIN);
    let two = _mm256_set1_epi8(2);
    let offsets = table([o[0], o[1], 0, o[2], o[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    let sign = |x: __m256i, y: __m256i| {
        let (x, y) = (_mm256_xor_si256(x, flip), _mm256_xor_si256(y, flip));
        // -1 where greater: so `lt - gt` is the sign.
        _mm256_sub_epi8(_mm256_cmpgt_epi8(y, x), _mm256_cmpgt_epi8(x, y))
    };
    let mut x = 0;
    while x + 32 <= n {
        let v = ld32(cur, x);
        let idx = _mm256_add_epi8(
            two,
            _mm256_add_epi8(sign(v, ld32(a, x)), sign(v, ld32(b, x))),
        );
        st32(dst, x, add_clip(v, _mm256_shuffle_epi8(offsets, idx)));
        x += 32;
    }
    for x in x..n {
        let v = i32::from(cur[x]);
        let idx = 2 + (v - i32::from(a[x])).signum() + (v - i32::from(b[x])).signum();
        let add = [o[0], o[1], 0, o[2], o[3]][idx as usize];
        dst[x] = (v + i32::from(add)).clamp(0, 255) as u8;
    }
}

#[target_feature(enable = "avx2")]
pub(super) fn band_row(dst: &mut [u8], cur: &[u8], position: u8, o: [i8; 4]) {
    let n = dst.len();
    let offsets = table([o[0], o[1], o[2], o[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    let pos = _mm256_set1_epi8(position as i8);
    let low5 = _mm256_set1_epi8(31);
    let four = _mm256_set1_epi8(4);
    let mut x = 0;
    while x + 32 <= n {
        let v = ld32(cur, x);
        // The band of each byte: its top five bits.
        let band = _mm256_and_si256(_mm256_srli_epi16::<3>(v), low5);
        let k = _mm256_and_si256(_mm256_sub_epi8(band, pos), low5);
        let k = _mm256_min_epu8(k, four);
        st32(dst, x, add_clip(v, _mm256_shuffle_epi8(offsets, k)));
        x += 32;
    }
    for x in x..n {
        let v = i32::from(cur[x]);
        let k = (((v >> 3) - i32::from(position)) & 31) as usize;
        let add = if k < 4 { i32::from(o[k]) } else { 0 };
        dst[x] = (v + add).clamp(0, 255) as u8;
    }
}
