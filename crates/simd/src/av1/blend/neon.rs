//! The blend on NEON: eight samples to a vector.

use core::arch::aarch64::{
    vaddq_s16, vget_lane_u64, vget_low_s16, vmovl_u8, vmull_high_s16, vmull_s16, vqrshrun_n_s16,
    vreinterpret_u64_u8, vreinterpretq_s16_u16, vshrn_high_n_s32, vshrn_n_s32, vsubq_s16,
};

use crate::neon::{i16x8, u8x8};

#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(super) fn mask_blend8(a: &[i16; 8], b: &[i16; 8], m: &[u8; 8]) -> [u8; 8] {
    let (a, b) = (i16x8(*a), i16x8(*b));
    let m = vreinterpretq_s16_u16(vmovl_u8(u8x8(m)));
    let d = vsubq_s16(a, b);
    // m (a - b) in 32 bits, then >> 6 back to 16.
    let lo = vmull_s16(vget_low_s16(d), vget_low_s16(m));
    let hi = vmull_high_s16(d, m);
    let t = vshrn_high_n_s32::<6>(vshrn_n_s32::<6>(lo), hi);
    let out = vqrshrun_n_s16::<4>(vaddq_s16(b, t));
    vget_lane_u64::<0>(vreinterpret_u64_u8(out)).to_le_bytes()
}
