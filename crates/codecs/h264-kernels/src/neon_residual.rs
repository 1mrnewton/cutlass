//! The NEON 4×4 and 8×8 inverse transforms. Safe Rust on fixed-size arrays, as in
//! [`crate::neon`].

use core::arch::aarch64::{
    int32x4_t, vaddq_s32, vcombine_s16, vcombine_u64, vcreate_u8, vcreate_u64, vdupq_n_s32,
    vget_lane_u64, vget_low_u16, vmovl_high_u16, vmovl_u8, vmovl_u16, vqaddq_s32, vqmovn_s32,
    vqmovun_s16, vreinterpret_u64_u8, vreinterpretq_s32_s64, vreinterpretq_s32_u32,
    vreinterpretq_s32_u64, vreinterpretq_s64_s32, vshrq_n_s32, vsubq_s32, vtrn1q_s32, vtrn1q_s64,
    vtrn2q_s32, vtrn2q_s64,
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

/// One pass of the 8×8 transform on four lines at once, as [`pass`] is for
/// the 4×4 one: `d` are the eight inputs of every line.
#[inline]
#[target_feature(enable = "neon")]
fn pass8(d: [int32x4_t; 8]) -> [int32x4_t; 8] {
    let half = |v: int32x4_t| vshrq_n_s32::<1>(v);
    let quarter = |v: int32x4_t| vshrq_n_s32::<2>(v);
    let a0 = vaddq_s32(d[0], d[4]);
    let a2 = vsubq_s32(d[0], d[4]);
    let a4 = vsubq_s32(half(d[2]), d[6]);
    let a6 = vaddq_s32(half(d[6]), d[2]);
    let b0 = vaddq_s32(a0, a6);
    let b2 = vaddq_s32(a2, a4);
    let b4 = vsubq_s32(a2, a4);
    let b6 = vsubq_s32(a0, a6);
    let a1 = vsubq_s32(vsubq_s32(vsubq_s32(d[5], d[3]), d[7]), half(d[7]));
    let a3 = vsubq_s32(vsubq_s32(vaddq_s32(d[1], d[7]), d[3]), half(d[3]));
    let a5 = vaddq_s32(vaddq_s32(vsubq_s32(d[7], d[1]), d[5]), half(d[5]));
    let a7 = vaddq_s32(vaddq_s32(vaddq_s32(d[3], d[5]), d[1]), half(d[1]));
    let b1 = vaddq_s32(quarter(a7), a1);
    let b3 = vaddq_s32(a3, quarter(a5));
    let b5 = vsubq_s32(quarter(a3), a5);
    let b7 = vsubq_s32(a7, quarter(a1));
    [
        vaddq_s32(b0, b7),
        vaddq_s32(b2, b5),
        vaddq_s32(b4, b3),
        vaddq_s32(b6, b1),
        vsubq_s32(b6, b1),
        vsubq_s32(b4, b3),
        vsubq_s32(b2, b5),
        vsubq_s32(b0, b7),
    ]
}

/// The 8×8 transform as four-line passes: each half of the rows goes
/// along its rows with its 4×4 corners turned to columns, and is turned
/// back; then each half of the columns goes down them.
#[inline]
#[target_feature(enable = "neon")]
fn idct8_rows(block: &[i32; 64], rows: [[u8; 8]; 8]) -> [[u8; 8]; 8] {
    let at = |row: usize, col: usize| {
        let start = row * 8 + col;
        load4([
            block[start],
            block[start + 1],
            block[start + 2],
            block[start + 3],
        ])
    };
    // `mid[i]` is row `i` of the first pass, its left and right halves.
    let mut mid = [[vdupq_n_s32(0); 2]; 8];
    for top in [0, 4] {
        let left = turn4([at(top, 0), at(top + 1, 0), at(top + 2, 0), at(top + 3, 0)]);
        let right = turn4([at(top, 4), at(top + 1, 4), at(top + 2, 4), at(top + 3, 4)]);
        let f = pass8([
            left[0], left[1], left[2], left[3], right[0], right[1], right[2], right[3],
        ]);
        let left = turn4([f[0], f[1], f[2], f[3]]);
        let right = turn4([f[4], f[5], f[6], f[7]]);
        for i in 0..4 {
            mid[top + i] = [left[i], right[i]];
        }
    }
    let left = pass8(core::array::from_fn(|i| mid[i][0]));
    let right = pass8(core::array::from_fn(|i| mid[i][1]));
    let round = vdupq_n_s32(32);
    let mut out = rows;
    for (i, row) in out.iter_mut().enumerate() {
        let residual = |v: int32x4_t| vshrq_n_s32::<6>(vaddq_s32(v, round));
        let wide = vmovl_u8(vcreate_u8(u64::from_le_bytes(*row)));
        let low = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(wide)));
        let high = vreinterpretq_s32_u32(vmovl_high_u16(wide));
        // Saturating at each narrowing clips the sum to a sample.
        let sum = vcombine_s16(
            vqmovn_s32(vqaddq_s32(low, residual(left[i]))),
            vqmovn_s32(vqaddq_s32(high, residual(right[i]))),
        );
        *row = vget_lane_u64::<0>(vreinterpret_u64_u8(vqmovun_s16(sum))).to_le_bytes();
    }
    out
}

/// See [`crate::residual::dc_add`]. The block's rows are found in the
/// plane with one check, and `dc` goes onto all of them at once: its
/// positive part added and its negative part taken away, both saturating,
/// which holds each sample to 0 to 255.
#[target_feature(enable = "neon")]
pub(crate) fn dc_add(plane: &mut [u8], at: usize, down: usize, dc: i32, size: usize) -> bool {
    use core::arch::aarch64::{
        uint8x16_t, vcombine_u8, vcreate_u8, vdupq_n_u8, vgetq_lane_u64, vqaddq_u8, vqsubq_u8,
        vreinterpretq_u64_u8,
    };
    let Some(window) = plane.get_mut(at..) else {
        return false;
    };
    let rise = vdupq_n_u8(dc.clamp(0, 255) as u8);
    let fall = vdupq_n_u8(dc.clamp(-255, 0).unsigned_abs() as u8);
    let moved = |low: u64, high: u64| -> (u64, u64) {
        let samples: uint8x16_t = vcombine_u8(vcreate_u8(low), vcreate_u8(high));
        let out = vreinterpretq_u64_u8(vqsubq_u8(vqaddq_u8(samples, rise), fall));
        (vgetq_lane_u64::<0>(out), vgetq_lane_u64::<1>(out))
    };
    match size {
        4 => {
            let Some([r0, r1, r2, r3]) = crate::neon::rows_of::<4, 4>(window, down) else {
                return false;
            };
            // Two rows to each half of one vector.
            let pair = |first: &[u8; 4], second: &[u8; 4]| {
                u64::from(u32::from_le_bytes(*first)) | u64::from(u32::from_le_bytes(*second)) << 32
            };
            let (low, high) = moved(pair(r0, r1), pair(r2, r3));
            *r0 = (low as u32).to_le_bytes();
            *r1 = ((low >> 32) as u32).to_le_bytes();
            *r2 = (high as u32).to_le_bytes();
            *r3 = ((high >> 32) as u32).to_le_bytes();
            true
        }
        8 => {
            let Some(rows) = crate::neon::rows_of::<8, 8>(window, down) else {
                return false;
            };
            let [r0, r1, r2, r3, r4, r5, r6, r7] = rows;
            for (first, second) in [(r0, r1), (r2, r3), (r4, r5), (r6, r7)] {
                let (low, high) = moved(u64::from_le_bytes(*first), u64::from_le_bytes(*second));
                *first = low.to_le_bytes();
                *second = high.to_le_bytes();
            }
            true
        }
        _ => false,
    }
}

/// See [`crate::residual::dc_add_quads`]. As [`dc_add`] does for one DC,
/// each half of a row has its quarter's positive part added and negative
/// part taken away, both saturating; two rows go in one vector.
#[target_feature(enable = "neon")]
pub(crate) fn dc_add_quads(plane: &mut [u8], at: usize, down: usize, dcs: [i32; 4]) -> bool {
    use core::arch::aarch64::{
        vcombine_u8, vcreate_u8, vgetq_lane_u64, vqaddq_u8, vqsubq_u8, vreinterpretq_u64_u8,
    };
    let Some(window) = plane.get_mut(at..) else {
        return false;
    };
    let Some(rows) = crate::neon::rows_of::<8, 8>(window, down) else {
        return false;
    };
    // Each quarter's DC in the four bytes of its half of a row.
    let spread = |dc: i32| u64::from(dc as u8) * 0x0101_0101;
    let half = |left: i32, right: i32| {
        let rise = spread(left.clamp(0, 255)) | spread(right.clamp(0, 255)) << 32;
        let fall = spread(left.clamp(-255, 0).abs()) | spread(right.clamp(-255, 0).abs()) << 32;
        (
            vcombine_u8(vcreate_u8(rise), vcreate_u8(rise)),
            vcombine_u8(vcreate_u8(fall), vcreate_u8(fall)),
        )
    };
    let top = half(dcs[0], dcs[1]);
    let bottom = half(dcs[2], dcs[3]);
    let [r0, r1, r2, r3, r4, r5, r6, r7] = rows;
    for (index, (first, second)) in [(r0, r1), (r2, r3), (r4, r5), (r6, r7)]
        .into_iter()
        .enumerate()
    {
        let (rise, fall) = if index < 2 { top } else { bottom };
        let samples = vcombine_u8(
            vcreate_u8(u64::from_le_bytes(*first)),
            vcreate_u8(u64::from_le_bytes(*second)),
        );
        let out = vreinterpretq_u64_u8(vqsubq_u8(vqaddq_u8(samples, rise), fall));
        *first = vgetq_lane_u64::<0>(out).to_le_bytes();
        *second = vgetq_lane_u64::<1>(out).to_le_bytes();
    }
    true
}

crate::residual::residual_kernels!(pub(crate), #[target_feature(enable = "neon")]);
