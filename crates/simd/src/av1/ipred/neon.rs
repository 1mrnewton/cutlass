//! Intra prediction on NEON.

use core::arch::aarch64::{
    int16x8_t, vget_lane_u64, vmlaq_laneq_s16, vmovl_u8, vmulq_laneq_s16, vqrshrun_n_s16,
    vreinterpret_u64_u8, vreinterpretq_s16_u16,
};

use super::FilterTaps;
use crate::neon::{i16x8, u8x8};

/// The eight outputs of a 4x2 block as one weighted sum of its seven
/// inputs, each input a lane multiplying the column of its weights.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(super) fn filter_intra(
    dst: &mut [u8],
    stride: usize,
    w: usize,
    h: usize,
    taps: &FilterTaps,
    above: &[u8],
    left: &[u8],
) {
    let cols: [int16x8_t; 7] =
        core::array::from_fn(|m| i16x8(core::array::from_fn(|k| i16::from(taps[k][m]))));
    for r in (0..h).step_by(2) {
        let (Some(&l0), Some(&l1)) = (left.get(r), left.get(r + 1)) else {
            return;
        };
        let (mut p5, mut p6) = (l0, l1);
        let mut corner = if r == 0 {
            above.first().copied()
        } else {
            left.get(r - 1).copied()
        };
        for c in (0..w).step_by(4) {
            let top: Option<&[u8; 4]> = if r == 0 {
                above.get(c + 1..).and_then(|a| a.first_chunk())
            } else {
                dst.get((r - 1) * stride + c..)
                    .and_then(|a| a.first_chunk())
            };
            let (Some(&tl), Some(&[t0, t1, t2, t3])) = (corner.as_ref(), top) else {
                return;
            };
            let p = vreinterpretq_s16_u16(vmovl_u8(u8x8(&[tl, t0, t1, t2, t3, p5, p6, 0])));
            let mut acc = vmulq_laneq_s16::<0>(cols[0], p);
            acc = vmlaq_laneq_s16::<1>(acc, cols[1], p);
            acc = vmlaq_laneq_s16::<2>(acc, cols[2], p);
            acc = vmlaq_laneq_s16::<3>(acc, cols[3], p);
            acc = vmlaq_laneq_s16::<4>(acc, cols[4], p);
            acc = vmlaq_laneq_s16::<5>(acc, cols[5], p);
            acc = vmlaq_laneq_s16::<6>(acc, cols[6], p);
            let out =
                vget_lane_u64::<0>(vreinterpret_u64_u8(vqrshrun_n_s16::<4>(acc))).to_le_bytes();
            for (k, half) in [&out[..4], &out[4..]].into_iter().enumerate() {
                let Some(d) = dst
                    .get_mut((r + k) * stride + c..)
                    .and_then(|d| d.get_mut(..4))
                else {
                    return;
                };
                d.copy_from_slice(half);
            }
            corner = Some(t3);
            (p5, p6) = (out[3], out[7]);
        }
    }
}
