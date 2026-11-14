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

/// Thirty-two bytes.
#[inline]
#[target_feature(enable = "avx2")]
fn ld32(a: &[u8; 32]) -> __m256i {
    let lo = u128::from_le_bytes(*a.first_chunk::<16>().unwrap_or(&[0; 16]));
    let hi = u128::from_le_bytes(*a.last_chunk::<16>().unwrap_or(&[0; 16]));
    _mm256_set_m128i(
        _mm_set_epi64x((hi >> 64) as i64, hi as i64),
        _mm_set_epi64x((lo >> 64) as i64, lo as i64),
    )
}

/// Thirty-two bytes of `v`.
#[inline]
#[target_feature(enable = "avx2")]
fn st32(d: &mut [u8; 32], v: __m256i) {
    d[..8].copy_from_slice(&(_mm256_extract_epi64::<0>(v) as u64).to_le_bytes());
    d[8..16].copy_from_slice(&(_mm256_extract_epi64::<1>(v) as u64).to_le_bytes());
    d[16..24].copy_from_slice(&(_mm256_extract_epi64::<2>(v) as u64).to_le_bytes());
    d[24..].copy_from_slice(&(_mm256_extract_epi64::<3>(v) as u64).to_le_bytes());
}

/// The 32-byte chunks of a row as arrays, and how many bytes they cover.
#[inline(always)]
fn chunks(s: &[u8]) -> (&[[u8; 32]], usize) {
    let (c, _) = s.as_chunks::<32>();
    (c, 32 * c.len())
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
    let (d, x) = {
        let (d, _) = dst.as_chunks_mut::<32>();
        let x = 32 * d.len();
        (d, x)
    };
    let rows = chunks(cur)
        .0
        .iter()
        .zip(chunks(a).0.iter().zip(chunks(b).0));
    for (d, (c, (a, b))) in d.iter_mut().zip(rows) {
        let v = ld32(c);
        let idx = _mm256_add_epi8(two, _mm256_add_epi8(sign(v, ld32(a)), sign(v, ld32(b))));
        st32(d, add_clip(v, _mm256_shuffle_epi8(offsets, idx)));
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
    let (d, x) = {
        let (d, _) = dst.as_chunks_mut::<32>();
        let x = 32 * d.len();
        (d, x)
    };
    for (d, c) in d.iter_mut().zip(chunks(cur).0) {
        let v = ld32(c);
        // The band of each byte: its top five bits.
        let band = _mm256_and_si256(_mm256_srli_epi16::<3>(v), low5);
        let k = _mm256_and_si256(_mm256_sub_epi8(band, pos), low5);
        let k = _mm256_min_epu8(k, four);
        st32(d, add_clip(v, _mm256_shuffle_epi8(offsets, k)));
    }
    for x in x..n {
        let v = i32::from(cur[x]);
        let k = (((v >> 3) - i32::from(position)) & 31) as usize;
        let add = if k < 4 { i32::from(o[k]) } else { 0 };
        dst[x] = (v + add).clamp(0, 255) as u8;
    }
}

// Samples of 9 to 12 bits, sixteen at a time: every sample and sum fits
// signed 16 bits, so the signs come from plain compares, and the offset
// of each sample from an eight-entry table of 16-bit values by `pshufb`
// on byte pairs.

use core::arch::x86_64 as x86;

/// Sixteen samples.
#[inline]
#[target_feature(enable = "avx2")]
fn ld16(a: &[u16; 16]) -> __m256i {
    let word = |k: usize| {
        let mut v = 0u64;
        for (i, &s) in a[4 * k..4 * k + 4].iter().enumerate() {
            v |= u64::from(s) << (16 * i);
        }
        v as i64
    };
    x86::_mm256_set_epi64x(word(3), word(2), word(1), word(0))
}

/// The sixteen samples of `v`.
#[inline]
#[target_feature(enable = "avx2")]
fn st16(d: &mut [u16; 16], v: __m256i) {
    let lanes = [
        _mm256_extract_epi64::<0>(v),
        _mm256_extract_epi64::<1>(v),
        _mm256_extract_epi64::<2>(v),
        _mm256_extract_epi64::<3>(v),
    ];
    for (k, l) in lanes.into_iter().enumerate() {
        for i in 0..4 {
            d[4 * k + i] = (l as u64 >> (16 * i)) as u16;
        }
    }
}

/// An eight-entry table of 16-bit values in both lanes, and how to pick
/// entry `i` of each 16-bit lane: the byte indices `2i` and `2i + 1`.
#[inline]
#[target_feature(enable = "avx2")]
fn table16(t: [i16; 8]) -> __m256i {
    let b = |i: usize| t[i / 2].to_le_bytes()[i % 2] as i8;
    table(core::array::from_fn(b))
}

/// Entry `idx` (0 to 7 in each 16-bit lane) of `t`.
#[inline]
#[target_feature(enable = "avx2")]
fn pick16(t: __m256i, idx: __m256i) -> __m256i {
    let bytes = x86::_mm256_add_epi16(
        x86::_mm256_mullo_epi16(idx, x86::_mm256_set1_epi16(0x0202)),
        x86::_mm256_set1_epi16(0x0100),
    );
    _mm256_shuffle_epi8(t, bytes)
}

/// `v + o` clipped to `0..=max`.
#[inline]
#[target_feature(enable = "avx2")]
fn add_clip16(v: __m256i, o: __m256i, max: __m256i) -> __m256i {
    x86::_mm256_min_epi16(
        x86::_mm256_max_epi16(x86::_mm256_add_epi16(v, o), _mm256_setzero_si256()),
        max,
    )
}

#[target_feature(enable = "avx2")]
pub(super) fn edge_row16(
    dst: &mut [u16],
    cur: &[u16],
    (a, b): (&[u16], &[u16]),
    o: [i16; 4],
    bit_depth: u32,
) {
    let n = dst.len();
    let top = (1i32 << bit_depth) - 1;
    let max = x86::_mm256_set1_epi16(top as i16);
    let two = x86::_mm256_set1_epi16(2);
    let offsets = table16([o[0], o[1], 0, o[2], o[3], 0, 0, 0]);
    let sign = |x: __m256i, y: __m256i| {
        x86::_mm256_sub_epi16(x86::_mm256_cmpgt_epi16(y, x), x86::_mm256_cmpgt_epi16(x, y))
    };
    let (d, _) = dst.as_chunks_mut::<16>();
    let done = 16 * d.len();
    let rows = cur
        .as_chunks::<16>()
        .0
        .iter()
        .zip(a.as_chunks::<16>().0.iter().zip(b.as_chunks::<16>().0));
    for (d, (c, (a, b))) in d.iter_mut().zip(rows) {
        let v = ld16(c);
        let idx = x86::_mm256_add_epi16(
            two,
            x86::_mm256_add_epi16(sign(v, ld16(a)), sign(v, ld16(b))),
        );
        st16(d, add_clip16(v, pick16(offsets, idx), max));
    }
    for x in done..n {
        let v = i32::from(cur[x]);
        let idx = 2 + (v - i32::from(a[x])).signum() + (v - i32::from(b[x])).signum();
        let add = [o[0], o[1], 0, o[2], o[3]][idx as usize];
        dst[x] = (v + i32::from(add)).clamp(0, top) as u16;
    }
}

#[target_feature(enable = "avx2")]
pub(super) fn band_row16(dst: &mut [u16], cur: &[u16], position: u8, o: [i16; 4], bit_depth: u32) {
    let n = dst.len();
    let top = (1i32 << bit_depth) - 1;
    let max = x86::_mm256_set1_epi16(top as i16);
    let offsets = table16([o[0], o[1], o[2], o[3], 0, 0, 0, 0]);
    let pos = x86::_mm256_set1_epi16(i16::from(position));
    let low5 = x86::_mm256_set1_epi16(31);
    let four = x86::_mm256_set1_epi16(4);
    let shift = x86::_mm_cvtsi32_si128(bit_depth as i32 - 5);
    let (d, _) = dst.as_chunks_mut::<16>();
    let done = 16 * d.len();
    for (d, c) in d.iter_mut().zip(cur.as_chunks::<16>().0) {
        let v = ld16(c);
        let band = x86::_mm256_srl_epi16(v, shift);
        let k = _mm256_and_si256(x86::_mm256_sub_epi16(band, pos), low5);
        let k = x86::_mm256_min_epi16(k, four);
        st16(d, add_clip16(v, pick16(offsets, k), max));
    }
    for x in done..n {
        let v = i32::from(cur[x]);
        let k = (((v >> (bit_depth - 5)) - i32::from(position)) & 31) as usize;
        let add = if k < 4 { i32::from(o[k]) } else { 0 };
        dst[x] = (v + add).clamp(0, top) as u16;
    }
}

/// [`edge_row`] over `rows` rows of `n` samples of one buffer: row `j`'s
/// samples start at `first + j * stride` in `src`, its neighbours `off_a`
/// and `off_b` away; row `j` goes to `dst[j * dst_stride..]`. A row not a
/// multiple of 32 ends with a chunk that overlaps the one before it.
#[target_feature(enable = "avx2")]
pub(super) fn edge_strided(
    (dst, dst_stride): (&mut [u8], usize),
    (src, stride, first): (&[u8], usize, usize),
    (rows, n): (usize, usize),
    (off_a, off_b): (isize, isize),
    o: [i8; 4],
) {
    let flip = _mm256_set1_epi8(i8::MIN);
    let two = _mm256_set1_epi8(2);
    let offsets = table([o[0], o[1], 0, o[2], o[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    let sign = |x: __m256i, y: __m256i| {
        let (x, y) = (_mm256_xor_si256(x, flip), _mm256_xor_si256(y, flip));
        _mm256_sub_epi8(_mm256_cmpgt_epi8(y, x), _mm256_cmpgt_epi8(x, y))
    };
    for j in 0..rows {
        let c = first + j * stride;
        let (a, b) = (c.wrapping_add_signed(off_a), c.wrapping_add_signed(off_b));
        let (Some(cur), Some(na), Some(nb), Some(d)) = (
            src.get(c..c + n),
            src.get(a..a.wrapping_add(n)),
            src.get(b..b.wrapping_add(n)),
            dst.get_mut(j * dst_stride..j * dst_stride + n),
        ) else {
            return;
        };
        if n < 32 {
            edge_row(d, cur, (na, nb), o);
            continue;
        }
        let mut x = 0;
        loop {
            let at = x.min(n - 32);
            if let (Some(cv), Some(av), Some(bv), Some(dv)) = (
                cur[at..].first_chunk::<32>(),
                na[at..].first_chunk::<32>(),
                nb[at..].first_chunk::<32>(),
                d[at..].first_chunk_mut::<32>(),
            ) {
                let v = ld32(cv);
                let idx =
                    _mm256_add_epi8(two, _mm256_add_epi8(sign(v, ld32(av)), sign(v, ld32(bv))));
                st32(dv, add_clip(v, _mm256_shuffle_epi8(offsets, idx)));
            }
            if at + 32 >= n {
                break;
            }
            x += 32;
        }
    }
}
