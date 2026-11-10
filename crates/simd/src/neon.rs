//! Loads and constants for the NEON kernels, without a pointer: a vector
//! is built from an array's bytes as 64-bit integers, which compiles to
//! one load.

use core::arch::aarch64::{
    int16x8_t, int32x4_t, uint8x8_t, uint16x8_t, vcombine_u64, vcreate_u8, vcreate_u64,
    vreinterpretq_s16_u64, vreinterpretq_s32_u64, vreinterpretq_u16_u64,
};

/// Eight bytes as eight lanes.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn u8x8(a: &[u8; 8]) -> uint8x8_t {
    vcreate_u8(u64::from_le_bytes(*a))
}

fn pack16(a: [u16; 4]) -> u64 {
    u64::from(a[0]) | u64::from(a[1]) << 16 | u64::from(a[2]) << 32 | u64::from(a[3]) << 48
}

/// Eight 16-bit values as eight lanes.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn u16x8(a: &[u16; 8]) -> uint16x8_t {
    let lo = pack16([a[0], a[1], a[2], a[3]]);
    let hi = pack16([a[4], a[5], a[6], a[7]]);
    vreinterpretq_u16_u64(vcombine_u64(vcreate_u64(lo), vcreate_u64(hi)))
}

/// Eight signed 16-bit values as eight lanes.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn i16x8(a: [i16; 8]) -> int16x8_t {
    let a = a.map(|v| v as u16);
    let lo = pack16([a[0], a[1], a[2], a[3]]);
    let hi = pack16([a[4], a[5], a[6], a[7]]);
    vreinterpretq_s16_u64(vcombine_u64(vcreate_u64(lo), vcreate_u64(hi)))
}

/// Four signed 32-bit values as four lanes.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn i32x4(a: [i32; 4]) -> int32x4_t {
    let p = |x: i32, y: i32| u64::from(x as u32) | u64::from(y as u32) << 32;
    vreinterpretq_s32_u64(vcombine_u64(
        vcreate_u64(p(a[0], a[1])),
        vcreate_u64(p(a[2], a[3])),
    ))
}
