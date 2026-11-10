//! The NEON 4×4 inverse transform. Safe Rust on fixed-size arrays, as in
//! [`crate::neon`].

use core::arch::aarch64::{
    int32x4_t, vaddq_s32, vcombine_s16, vcombine_u64, vcreate_u8, vcreate_u64, vdupq_n_s32,
    vget_lane_u64, vget_low_u16, vmovl_u8, vmovl_u16, vqaddq_s32, vqmovn_s32, vqmovun_s16,
    vreinterpret_u64_u8, vreinterpretq_s32_s64, vreinterpretq_s32_u32, vreinterpretq_s32_u64,
    vreinterpretq_s64_s32, vshrq_n_s32, vsubq_s32, vtrn1q_s32, vtrn1q_s64, vtrn2q_s32, vtrn2q_s64,
};

/// Four 32-bit numbers as a vector.
#[inline]
#[target_feature(enable = "neon")]
fn load4(values: [i32; 4]) -> int32x4_t {
    let half = |a: i32, b: i32| u64::from(a as u32) | u64::from(b as u32) << 32;
    vreinterpretq_s32_u64(vcombine_u64(
        vcreate_u64(half(values[0], values[1])),
        vcreate_u64(half(values[2], values[3])),
    ))
}

/// Rows to columns of a 4×4 block held as four vectors.
#[inline]
#[target_feature(enable = "neon")]
fn turn4(v: [int32x4_t; 4]) -> [int32x4_t; 4] {
    let a = vreinterpretq_s64_s32(vtrn1q_s32(v[0], v[1]));
    let b = vreinterpretq_s64_s32(vtrn2q_s32(v[0], v[1]));
    let c = vreinterpretq_s64_s32(vtrn1q_s32(v[2], v[3]));
    let d = vreinterpretq_s64_s32(vtrn2q_s32(v[2], v[3]));
    [
        vreinterpretq_s32_s64(vtrn1q_s64(a, c)),
        vreinterpretq_s32_s64(vtrn1q_s64(b, d)),
        vreinterpretq_s32_s64(vtrn2q_s64(a, c)),
        vreinterpretq_s32_s64(vtrn2q_s64(b, d)),
    ]
}

/// One pass of the transform on four lines at once: `d` are the four
/// inputs of every line, and the result its four outputs.
#[inline]
#[target_feature(enable = "neon")]
fn pass(d: [int32x4_t; 4]) -> [int32x4_t; 4] {
    let e0 = vaddq_s32(d[0], d[2]);
    let e1 = vsubq_s32(d[0], d[2]);
    let e2 = vsubq_s32(vshrq_n_s32::<1>(d[1]), d[3]);
    let e3 = vaddq_s32(d[1], vshrq_n_s32::<1>(d[3]));
    [
        vaddq_s32(e0, e3),
        vaddq_s32(e1, e2),
        vsubq_s32(e1, e2),
        vsubq_s32(e0, e3),
    ]
}

#[inline]
#[target_feature(enable = "neon")]
fn idct4_rows(block: &[i32; 16], rows: [[u8; 4]; 4]) -> [[u8; 4]; 4] {
    let line = |index: usize| {
        load4([
            block[index * 4],
            block[index * 4 + 1],
            block[index * 4 + 2],
            block[index * 4 + 3],
        ])
    };
    // Along each row first: the columns hold one input of every row.
    let mid = pass(turn4([line(0), line(1), line(2), line(3)]));
    // Then down each column: the rows of the first pass's result.
    let out = pass(turn4(mid));
    let round = vdupq_n_s32(32);
    let finish = |residual: int32x4_t, samples: [u8; 4]| -> [u8; 4] {
        let residual = vshrq_n_s32::<6>(vaddq_s32(residual, round));
        let wide = u64::from(u32::from_le_bytes(samples));
        let samples = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vcreate_u8(wide)))));
        // Saturating at each narrowing clips the sum to a sample.
        let sum = vqmovn_s32(vqaddq_s32(samples, residual));
        let bytes = vqmovun_s16(vcombine_s16(sum, sum));
        let packed = vget_lane_u64::<0>(vreinterpret_u64_u8(bytes)).to_le_bytes();
        [packed[0], packed[1], packed[2], packed[3]]
    };
    [
        finish(out[0], rows[0]),
        finish(out[1], rows[1]),
        finish(out[2], rows[2]),
        finish(out[3], rows[3]),
    ]
}

/// `dc` added to each of the eight samples in `samples`, each held to 0 to
/// 255: its positive part added and its negative part taken away, both
/// saturating.
#[inline]
#[target_feature(enable = "neon")]
fn dc_row(samples: u64, dc: i32) -> u64 {
    use core::arch::aarch64::{
        vcreate_u8, vdup_n_u8, vget_lane_u64, vqadd_u8, vqsub_u8, vreinterpret_u64_u8,
    };
    let rise = vdup_n_u8(dc.clamp(0, 255) as u8);
    let fall = vdup_n_u8((-dc.clamp(-255, 0)) as u8);
    let moved = vqsub_u8(vqadd_u8(vcreate_u8(samples), rise), fall);
    vget_lane_u64::<0>(vreinterpret_u64_u8(moved))
}

crate::residual::residual_kernels!(pub(crate), #[target_feature(enable = "neon")]);
