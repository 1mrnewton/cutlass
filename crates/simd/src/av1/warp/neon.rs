//! The warp on NEON: a row of eight samples to a vector.
//!
//! A horizontal row multiplies the eight samples each output reads by its
//! filter, the samples less 128 so that the products and their sums fit
//! 16 bits, and adds the products of each output pairwise. The vertical
//! pass turns the eight filters of a row around so that a vector holds
//! one tap of each, and sums the rows of the horizontal pass times them.

use core::arch::aarch64::{
    int8x8_t, int8x16_t, int16x8_t, int32x4_t, vaddq_s16, vcombine_s8, vcombine_s16, vcombine_u16,
    vcreate_s8, vdupq_n_s16, vdupq_n_u8, veorq_u8, vext_s8, vget_high_s8, vget_lane_u64,
    vget_low_s8, vget_low_s16, vmlal_high_s16, vmlal_s16, vmovl_high_s8, vmovl_s8, vmull_high_s16,
    vmull_s8, vmull_s16, vpaddq_s16, vqmovn_u16, vqrshrun_n_s32, vreinterpret_u64_u8,
    vreinterpretq_s8_s16, vreinterpretq_s8_s32, vreinterpretq_s8_u8, vreinterpretq_s16_s8,
    vreinterpretq_s32_s8, vrshrn_n_s32, vrshrq_n_s16, vtrn1q_s8, vtrn1q_s16, vtrn2q_s8, vtrn2q_s16,
    vuzp1q_s32, vuzp2q_s32,
};

use super::Filters;
use crate::neon::{i16x8_array, u8x16};

/// A filter as eight lanes.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn taps(f: &[i8; 8]) -> int8x8_t {
    vcreate_s8(u64::from_le_bytes(f.map(|t| t as u8)))
}

/// Two filters as the halves of a vector.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn taps2(lo: &[i8; 8], hi: &[i8; 8]) -> int8x16_t {
    vcombine_s8(taps(lo), taps(hi))
}

/// Row `r` of the horizontal pass: `Round2(sum, 3)` of each column.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn horizontal(row: &[u8; 16], f: &Filters<'_>, r: i32) -> int16x8_t {
    let [sx4, alpha, beta] = f.h;
    let at = sx4 + beta * (r - 7);
    // The samples less 128: each filter sums to 128, so the sum is the
    // true one less 128 times 128, which the rounding adds back as 2048.
    let p = vreinterpretq_s8_u8(veorq_u8(u8x16(row), vdupq_n_u8(128)));
    let (lo, hi) = (vget_low_s8(p), vget_high_s8(p));
    let m0 = vmull_s8(lo, taps(f.at(at - 4 * alpha)));
    let m1 = vmull_s8(vext_s8::<1>(lo, hi), taps(f.at(at - 3 * alpha)));
    let m2 = vmull_s8(vext_s8::<2>(lo, hi), taps(f.at(at - 2 * alpha)));
    let m3 = vmull_s8(vext_s8::<3>(lo, hi), taps(f.at(at - alpha)));
    let m4 = vmull_s8(vext_s8::<4>(lo, hi), taps(f.at(at)));
    let m5 = vmull_s8(vext_s8::<5>(lo, hi), taps(f.at(at + alpha)));
    let m6 = vmull_s8(vext_s8::<6>(lo, hi), taps(f.at(at + 2 * alpha)));
    let m7 = vmull_s8(vext_s8::<7>(lo, hi), taps(f.at(at + 3 * alpha)));
    let s = vpaddq_s16(
        vpaddq_s16(vpaddq_s16(m0, m1), vpaddq_s16(m2, m3)),
        vpaddq_s16(vpaddq_s16(m4, m5), vpaddq_s16(m6, m7)),
    );
    vaddq_s16(vrshrq_n_s16::<3>(s), vdupq_n_s16(2048))
}

/// The vertical sums of output row `r`, low and high halves, from the
/// eight rows of the horizontal pass it reads.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn vertical(inter: &[int16x8_t], f: &Filters<'_>, r: i32) -> (int32x4_t, int32x4_t) {
    let [sy4, gamma, delta] = f.v;
    let at = sy4 + delta * (r - 4);
    let at = |c: i32| f.at(at + gamma * (c - 4));
    // The filters of columns c and c + 4 in one vector, then turned
    // around in three steps of transposes: bytes, pairs, then quads.
    let q0 = taps2(at(0), at(4));
    let q1 = taps2(at(1), at(5));
    let q2 = taps2(at(2), at(6));
    let q3 = taps2(at(3), at(7));
    let a0 = vtrn1q_s8(q0, q1);
    let a1 = vtrn2q_s8(q0, q1);
    let a2 = vtrn1q_s8(q2, q3);
    let a3 = vtrn2q_s8(q2, q3);
    let s16 = vreinterpretq_s16_s8;
    let b04 = vtrn1q_s16(s16(a0), s16(a2));
    let b26 = vtrn2q_s16(s16(a0), s16(a2));
    let b15 = vtrn1q_s16(s16(a1), s16(a3));
    let b37 = vtrn2q_s16(s16(a1), s16(a3));
    let s32 = |v: int16x8_t| vreinterpretq_s32_s8(vreinterpretq_s8_s16(v));
    let t02 = vreinterpretq_s8_s32(vuzp1q_s32(s32(b04), s32(b26)));
    let t46 = vreinterpretq_s8_s32(vuzp2q_s32(s32(b04), s32(b26)));
    let t13 = vreinterpretq_s8_s32(vuzp1q_s32(s32(b15), s32(b37)));
    let t57 = vreinterpretq_s8_s32(vuzp2q_s32(s32(b15), s32(b37)));
    let t = [
        vmovl_s8(vget_low_s8(t02)),
        vmovl_s8(vget_low_s8(t13)),
        vmovl_high_s8(t02),
        vmovl_high_s8(t13),
        vmovl_s8(vget_low_s8(t46)),
        vmovl_s8(vget_low_s8(t57)),
        vmovl_high_s8(t46),
        vmovl_high_s8(t57),
    ];
    let mut lo = vmull_s16(vget_low_s16(inter[0]), vget_low_s16(t[0]));
    let mut hi = vmull_high_s16(inter[0], t[0]);
    for k in 1..8 {
        lo = vmlal_s16(lo, vget_low_s16(inter[k]), vget_low_s16(t[k]));
        hi = vmlal_high_s16(hi, inter[k], t[k]);
    }
    (lo, hi)
}

/// Both passes, `out` taking each output row's sums.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn warp<T>(
    rows: [&[u8; 16]; 15],
    f: &Filters<'_>,
    mut out: impl FnMut(int32x4_t, int32x4_t) -> T,
) -> [T; 8] {
    let mut inter = [vdupq_n_s16(0); 15];
    for (r, (v, row)) in inter.iter_mut().zip(rows).enumerate() {
        *v = horizontal(row, f, r as i32);
    }
    core::array::from_fn(|r| {
        let (lo, hi) = vertical(&inter[r..r + 8], f, r as i32);
        out(lo, hi)
    })
}

#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(super) fn put8x8(rows: [&[u8; 16]; 15], f: &Filters<'_>) -> [[u8; 8]; 8] {
    warp(rows, f, |lo, hi| {
        let px = vqmovn_u16(vcombine_u16(
            vqrshrun_n_s32::<11>(lo),
            vqrshrun_n_s32::<11>(hi),
        ));
        vget_lane_u64::<0>(vreinterpret_u64_u8(px)).to_le_bytes()
    })
}

#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(super) fn prep8x8(rows: [&[u8; 16]; 15], f: &Filters<'_>) -> [[i16; 8]; 8] {
    warp(rows, f, |lo, hi| {
        i16x8_array(vcombine_s16(vrshrn_n_s32::<7>(lo), vrshrn_n_s32::<7>(hi)))
    })
}
