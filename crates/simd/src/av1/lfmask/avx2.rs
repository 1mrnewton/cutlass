//! The edge codes on x86-64 with AVX2: one unit in each 8-bit lane of a
//! 128-bit vector.

use core::arch::x86_64::{
    __m128i, _mm_add_epi8, _mm_alignr_epi8, _mm_and_si128, _mm_andnot_si128, _mm_blendv_epi8,
    _mm_cmpeq_epi8, _mm_cvtsi32_si128, _mm_cvtsi128_si64, _mm_extract_epi64, _mm_max_epu8,
    _mm_min_epu8, _mm_movemask_epi8, _mm_packus_epi16, _mm_packus_epi32, _mm_set1_epi8,
    _mm_set1_epi32, _mm_setzero_si128, _mm_shuffle_epi8, _mm_shuffle_epi32, _mm_srl_epi32,
    _mm_srli_epi16, _mm_unpackhi_epi8, _mm_unpacklo_epi8, _mm_unpacklo_epi64,
};

use super::{EdgeUnits, Edges};
use crate::av1::cpu::x86::{ld128, st128};

type V = __m128i;

const LANES: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
/// Filter sizes by the smaller transform size `m`.
const LUMA_SIZES: [u8; 16] = [4, 8, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16];
const CHROMA_SIZES: [u8; 16] = [4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6];
/// The bits of a column below a transform's width, by the base-2
/// logarithm of the width in units.
const BELOW: [u8; 16] = [0, 1, 3, 7, 15, 31, 63, 127, 0, 0, 0, 0, 0, 0, 0, 0];

#[inline]
#[target_feature(enable = "avx2")]
fn splat(v: u8) -> V {
    _mm_set1_epi8(v as i8)
}

/// The eight 16-bit lanes of a vector.
#[inline]
#[target_feature(enable = "avx2")]
fn words(v: V) -> [u16; 8] {
    let q = [
        _mm_cvtsi128_si64(v) as u64,
        _mm_extract_epi64::<1>(v) as u64,
    ];
    core::array::from_fn(|i| (q[i / 4] >> (16 * (i % 4))) as u16)
}

#[target_feature(enable = "avx2")]
pub(super) fn edge_codes(u: &EdgeUnits, edges: Edges, luma: bool) -> ([u16; 16], u16) {
    let seven = splat(7);
    let zero = _mm_setzero_si128();
    let dims = ld128(&u.dims);
    let across = ld128(&u.across);
    let lvl = ld128(&u.lvl);
    let (d, a, lvl_across, on) = match edges {
        Edges::Left { first } => {
            // The units to the left: these moved up a lane, after the
            // last of the units before them.
            let across = _mm_alignr_epi8::<15>(dims, across);
            let lvl_across = _mm_alignr_epi8::<15>(lvl, ld128(&u.lvl_across));
            let d = _mm_and_si128(dims, seven);
            let col = _mm_add_epi8(splat(first), ld128(&LANES));
            // The column's bits below the transform's width.
            let below = _mm_shuffle_epi8(ld128(&BELOW), d);
            let on = _mm_cmpeq_epi8(_mm_and_si128(col, below), zero);
            (d, _mm_and_si128(across, seven), lvl_across, on)
        }
        Edges::Top { zeros } => {
            // The bits a 16-bit shift brings in from the next lane are
            // masked off.
            let d = _mm_and_si128(_mm_srli_epi16::<3>(dims), seven);
            let a = _mm_and_si128(_mm_srli_epi16::<3>(across), seven);
            let z = splat(zeros);
            let on = _mm_cmpeq_epi8(_mm_max_epu8(d, z), z);
            (d, a, ld128(&u.lvl_across), on)
        }
    };
    let sizes = ld128(if luma { &LUMA_SIZES } else { &CHROMA_SIZES });
    let size = _mm_shuffle_epi8(sizes, _mm_min_epu8(d, a));
    let lvl = _mm_blendv_epi8(lvl, lvl_across, _mm_cmpeq_epi8(lvl, zero));
    let lvl = _mm_and_si128(lvl, on);
    let off = _mm_cmpeq_epi8(lvl, zero);
    let size = _mm_andnot_si128(off, size);
    let lo = words(_mm_unpacklo_epi8(lvl, size));
    let hi = words(_mm_unpackhi_epi8(lvl, size));
    let mask = !(_mm_movemask_epi8(off) as u16);
    (
        core::array::from_fn(|l| if l < 8 { lo[l] } else { hi[l - 8] }),
        mask,
    )
}

/// The low 6 bits of byte `i` of each 32-bit lane of four vectors.
#[inline]
#[target_feature(enable = "avx2")]
fn picks(v: [V; 4], i: usize) -> [u8; 16] {
    let by = _mm_cvtsi32_si128(8 * (i as i32 & 3));
    let m = _mm_set1_epi32(63);
    let p = |x: V| _mm_and_si128(_mm_srl_epi32(x, by), m);
    let t = _mm_packus_epi16(
        _mm_packus_epi32(p(v[0]), p(v[1])),
        _mm_packus_epi32(p(v[2]), p(v[3])),
    );
    let mut out = [0u8; 16];
    st128(&mut out, t);
    out
}

#[target_feature(enable = "avx2")]
pub(super) fn levels(records: &[[u8; 4]; 16], i: usize) -> [u8; 16] {
    let mut v = [_mm_setzero_si128(); 4];
    let (chunks, _) = records.as_flattened().as_chunks::<16>();
    for (o, c) in v.iter_mut().zip(chunks) {
        *o = ld128(c);
    }
    picks(v, i)
}

#[target_feature(enable = "avx2")]
pub(super) fn levels_odd(records: &[[u8; 4]; 32], i: usize) -> [u8; 16] {
    let mut v = [_mm_setzero_si128(); 8];
    let (chunks, _) = records.as_flattened().as_chunks::<16>();
    for (o, c) in v.iter_mut().zip(chunks) {
        *o = ld128(c);
    }
    // The odd records of each vector in its low half, then of two
    // vectors in one.
    let odd = |a: V, b: V| {
        _mm_unpacklo_epi64(
            _mm_shuffle_epi32::<0b1101>(a),
            _mm_shuffle_epi32::<0b1101>(b),
        )
    };
    picks(
        [
            odd(v[0], v[1]),
            odd(v[2], v[3]),
            odd(v[4], v[5]),
            odd(v[6], v[7]),
        ],
        i,
    )
}
