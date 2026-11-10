//! Loads, stores and constants for the NEON kernels, without a pointer.
//!
//! Where the build has the AES extension (every Apple target), a vector
//! is the bytes of an array read as one 128-bit integer, which compiles
//! to one load or store. Elsewhere it is built from two 64-bit integers.
//! A kernel names the same features as these helpers, so it may call them.

use core::arch::aarch64::{
    int16x8_t, int32x4_t, uint8x8_t, uint8x16_t, uint16x8_t, vcombine_u64, vcreate_u8, vcreate_u64,
    vgetq_lane_u64, vreinterpretq_s16_u64, vreinterpretq_s32_u64, vreinterpretq_u64_u16,
};
#[cfg(target_feature = "aes")]
use core::arch::aarch64::{vreinterpretq_p128_u8, vreinterpretq_u8_p128, vreinterpretq_u16_p128};
#[cfg(not(target_feature = "aes"))]
use core::arch::aarch64::{vreinterpretq_u8_u64, vreinterpretq_u16_u64, vreinterpretq_u64_u8};

/// Eight bytes as eight lanes.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(crate) fn u8x8(a: &[u8; 8]) -> uint8x8_t {
    vcreate_u8(u64::from_le_bytes(*a))
}

fn pack16(a: [u16; 4]) -> u64 {
    u64::from(a[0]) | u64::from(a[1]) << 16 | u64::from(a[2]) << 32 | u64::from(a[3]) << 48
}

/// Eight 16-bit values as eight lanes.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(crate) fn u16x8(a: &[u16; 8]) -> uint16x8_t {
    #[cfg(target_feature = "aes")]
    {
        let mut b = [0u8; 16];
        for (d, v) in b.as_chunks_mut::<2>().0.iter_mut().zip(a) {
            *d = v.to_le_bytes();
        }
        vreinterpretq_u16_p128(u128::from_le_bytes(b))
    }
    #[cfg(not(target_feature = "aes"))]
    {
        let lo = pack16([a[0], a[1], a[2], a[3]]);
        let hi = pack16([a[4], a[5], a[6], a[7]]);
        vreinterpretq_u16_u64(vcombine_u64(vcreate_u64(lo), vcreate_u64(hi)))
    }
}

/// Eight signed 16-bit values as eight lanes.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(crate) fn i16x8(a: [i16; 8]) -> int16x8_t {
    let a = a.map(|v| v as u16);
    let lo = pack16([a[0], a[1], a[2], a[3]]);
    let hi = pack16([a[4], a[5], a[6], a[7]]);
    vreinterpretq_s16_u64(vcombine_u64(vcreate_u64(lo), vcreate_u64(hi)))
}

/// Four signed 32-bit values as four lanes.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(crate) fn i32x4(a: [i32; 4]) -> int32x4_t {
    let p = |x: i32, y: i32| u64::from(x as u32) | u64::from(y as u32) << 32;
    vreinterpretq_s32_u64(vcombine_u64(
        vcreate_u64(p(a[0], a[1])),
        vcreate_u64(p(a[2], a[3])),
    ))
}

/// Two runs of eight bytes as sixteen lanes: `lo` in lanes 0 to 7.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(crate) fn u8x16_pair(lo: &[u8; 8], hi: &[u8; 8]) -> uint8x16_t {
    let v = u128::from(u64::from_le_bytes(*lo)) | u128::from(u64::from_le_bytes(*hi)) << 64;
    wide(v)
}

/// A 128-bit integer as sixteen lanes, its low byte in lane 0.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn wide(v: u128) -> uint8x16_t {
    #[cfg(target_feature = "aes")]
    {
        vreinterpretq_u8_p128(v)
    }
    #[cfg(not(target_feature = "aes"))]
    {
        vreinterpretq_u8_u64(vcombine_u64(
            vcreate_u64(v as u64),
            vcreate_u64((v >> 64) as u64),
        ))
    }
}

/// Sixteen lanes as a 128-bit integer, lane 0 its low byte.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn narrow(v: uint8x16_t) -> u128 {
    #[cfg(target_feature = "aes")]
    {
        vreinterpretq_p128_u8(v)
    }
    #[cfg(not(target_feature = "aes"))]
    {
        let v = vreinterpretq_u64_u8(v);
        u128::from(vgetq_lane_u64::<0>(v)) | u128::from(vgetq_lane_u64::<1>(v)) << 64
    }
}

/// Sixteen bytes as sixteen lanes.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(crate) fn u8x16(a: &[u8; 16]) -> uint8x16_t {
    wide(u128::from_le_bytes(*a))
}

/// The lanes of `v` as two runs of eight bytes.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(crate) fn halves(v: uint8x16_t) -> ([u8; 8], [u8; 8]) {
    let v = narrow(v);
    ((v as u64).to_le_bytes(), ((v >> 64) as u64).to_le_bytes())
}

/// The lanes of `v` as sixteen bytes.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(crate) fn bytes(v: uint8x16_t) -> [u8; 16] {
    narrow(v).to_le_bytes()
}

/// The lanes of `v` as eight 16-bit values.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(crate) fn u16x8_array(v: uint16x8_t) -> [u16; 8] {
    let v = vreinterpretq_u64_u16(v);
    let (lo, hi) = (vgetq_lane_u64::<0>(v), vgetq_lane_u64::<1>(v));
    core::array::from_fn(|i| {
        let half = if i < 4 { lo } else { hi };
        (half >> (16 * (i % 4))) as u16
    })
}
