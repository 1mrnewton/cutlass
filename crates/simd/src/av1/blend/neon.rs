//! The blend on NEON: eight samples to a vector.

use core::arch::aarch64::{
    int16x8_t, uint8x8_t, vaddq_s16, vaddq_u16, vget_lane_u64, vget_low_s16, vmovl_u8,
    vmull_high_s16, vmull_s16, vpaddlq_u8, vqrshrun_n_s16, vreinterpret_u64_u8,
    vreinterpretq_s16_u16, vrshrn_n_u16, vshrn_high_n_s32, vshrn_n_s32, vsubq_s16,
};

use crate::neon::{i16x8, u8x8, u8x16};

/// `b + ((m (a - b)) >> 6)` on eight lanes, before the rounding to a
/// sample.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn mix(a: int16x8_t, b: int16x8_t, m: uint8x8_t) -> int16x8_t {
    let m = vreinterpretq_s16_u16(vmovl_u8(m));
    let d = vsubq_s16(a, b);
    // m (a - b) in 32 bits, then >> 6 back to 16.
    let lo = vmull_s16(vget_low_s16(d), vget_low_s16(m));
    let hi = vmull_high_s16(d, m);
    vaddq_s16(b, vshrn_high_n_s32::<6>(vshrn_n_s32::<6>(lo), hi))
}

#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(super) fn mask_blend8(a: &[i16; 8], b: &[i16; 8], m: &[u8; 8]) -> [u8; 8] {
    let out = vqrshrun_n_s16::<4>(mix(i16x8(*a), i16x8(*b), u8x8(m)));
    vget_lane_u64::<0>(vreinterpret_u64_u8(out)).to_le_bytes()
}

#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(super) fn weights_pairs8(m: &[u8; 16]) -> [u8; 8] {
    let out = vrshrn_n_u16::<1>(vpaddlq_u8(u8x16(m)));
    vget_lane_u64::<0>(vreinterpret_u64_u8(out)).to_le_bytes()
}

#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(super) fn weights_quads8(a: &[u8; 16], b: &[u8; 16]) -> [u8; 8] {
    let sum = vaddq_u16(vpaddlq_u8(u8x16(a)), vpaddlq_u8(u8x16(b)));
    let out = vrshrn_n_u16::<2>(sum);
    vget_lane_u64::<0>(vreinterpret_u64_u8(out)).to_le_bytes()
}
