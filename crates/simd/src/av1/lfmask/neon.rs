//! The edge codes on NEON: one unit in each lane.

use core::arch::aarch64::{
    uint8x16_t, uint8x16x4_t, vaddq_u8, vaddv_u8, vandq_u8, vbslq_u8, vceqq_u8, vcleq_u8,
    vdupq_n_u8, vextq_u8, vget_high_u8, vget_low_u8, vminq_u8, vqtbl1q_u8, vqtbl4q_u8, vqtbx4q_u8,
    vreinterpretq_s8_u8, vreinterpretq_u16_u8, vshlq_u8, vshrq_n_u8, vsubq_u8, vtstq_u8, vzip1q_u8,
    vzip2q_u8,
};

use super::{EdgeUnits, Edges};
use crate::neon::{bytes, u8x16, u16x8_array};

type V = uint8x16_t;

/// Each lane's bit in its half of a mask.
const BITS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
const LANES: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
/// Filter sizes by the smaller transform size `m`.
const LUMA_SIZES: [u8; 16] = [4, 8, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16];
const CHROMA_SIZES: [u8; 16] = [4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6];

#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(super) fn edge_codes(u: &EdgeUnits, edges: Edges, luma: bool) -> ([u16; 16], u16) {
    let seven = vdupq_n_u8(7);
    let dims = u8x16(&u.dims);
    let across = u8x16(&u.across);
    let lvl = u8x16(&u.lvl);
    let (d, a, lvl_across, on) = match edges {
        Edges::Left { first } => {
            // The units to the left: these moved up a lane, after the
            // last of the units before them.
            let across = vextq_u8::<15>(across, dims);
            let lvl_across = vextq_u8::<15>(u8x16(&u.lvl_across), lvl);
            let d = vandq_u8(dims, seven);
            let col = vaddq_u8(vdupq_n_u8(first), u8x16(&LANES));
            // The column's bits below the transform's width.
            let one = vdupq_n_u8(1);
            let below = vsubq_u8(vshlq_u8(one, vreinterpretq_s8_u8(d)), one);
            let on = vceqq_u8(vandq_u8(col, below), vdupq_n_u8(0));
            (d, vandq_u8(across, seven), lvl_across, on)
        }
        Edges::Top { zeros } => {
            let d = vandq_u8(vshrq_n_u8::<3>(dims), seven);
            let a = vandq_u8(vshrq_n_u8::<3>(across), seven);
            let on = vcleq_u8(d, vdupq_n_u8(zeros));
            (d, a, u8x16(&u.lvl_across), on)
        }
    };
    let sizes = u8x16(if luma { &LUMA_SIZES } else { &CHROMA_SIZES });
    let size = vqtbl1q_u8(sizes, vminq_u8(d, a));
    let lvl = vbslq_u8(vceqq_u8(lvl, vdupq_n_u8(0)), lvl_across, lvl);
    let lvl = vandq_u8(lvl, on);
    let size = vandq_u8(size, vtstq_u8(lvl, lvl));
    let lo = u16x8_array(vreinterpretq_u16_u8(vzip1q_u8(lvl, size)));
    let hi = u16x8_array(vreinterpretq_u16_u8(vzip2q_u8(lvl, size)));
    let bits = vandq_u8(vtstq_u8(lvl, lvl), u8x16(&BITS));
    let mask =
        u16::from(vaddv_u8(vget_low_u8(bits))) | u16::from(vaddv_u8(vget_high_u8(bits))) << 8;
    (
        core::array::from_fn(|l| if l < 8 { lo[l] } else { hi[l - 8] }),
        mask,
    )
}

/// The records as vectors of four.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn quads<const N: usize>(records: &[[u8; 4]], out: &mut [V; N]) {
    let (chunks, _) = records.as_flattened().as_chunks::<16>();
    for (o, c) in out.iter_mut().zip(chunks) {
        *o = u8x16(c);
    }
}

/// Out of the range of a lookup in four vectors, also with 3 added: the
/// lane is 0, or keeps its value.
const NO: u8 = 0xF0;

/// Where byte 0 of each of sixteen records lies in four vectors.
const ALL: [u8; 16] = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60];

/// Where byte 0 of the odd records among the first and the last sixteen
/// lies, each in four vectors.
const ODD: [[u8; 16]; 2] = [
    [
        4, 12, 20, 28, 36, 44, 52, 60, NO, NO, NO, NO, NO, NO, NO, NO,
    ],
    [
        NO, NO, NO, NO, NO, NO, NO, NO, 4, 12, 20, 28, 36, 44, 52, 60,
    ],
];

/// The indices `at` moved to byte `i` of the records.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn picks(at: &[u8; 16], i: usize) -> V {
    vaddq_u8(u8x16(at), vdupq_n_u8(i as u8))
}

#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(super) fn levels(records: &[[u8; 4]; 16], i: usize) -> [u8; 16] {
    let mut v = [vdupq_n_u8(0); 4];
    quads(records, &mut v);
    let t = vqtbl4q_u8(uint8x16x4_t(v[0], v[1], v[2], v[3]), picks(&ALL, i));
    bytes(vandq_u8(t, vdupq_n_u8(63)))
}

#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(super) fn levels_odd(records: &[[u8; 4]; 32], i: usize) -> [u8; 16] {
    let mut v = [vdupq_n_u8(0); 8];
    quads(records, &mut v);
    let t = vqtbl4q_u8(uint8x16x4_t(v[0], v[1], v[2], v[3]), picks(&ODD[0], i));
    let t = vqtbx4q_u8(t, uint8x16x4_t(v[4], v[5], v[6], v[7]), picks(&ODD[1], i));
    bytes(vandq_u8(t, vdupq_n_u8(63)))
}
