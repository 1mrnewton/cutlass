//! The Wiener filter on NEON.
//!
//! The row is symmetric, so the samples under equal taps are added first
//! and each pass multiplies four times for eight outputs' halves instead
//! of seven. The sums take 32 bits: a 12-bit sample pair times a tap does
//! not fit 16, nor does a pair of intermediates.

use core::arch::aarch64::{
    int16x8_t, int32x4_t, vaddl_high_s16, vaddl_s16, vaddq_s16, vcombine_s16, vcombine_u16,
    vdupq_n_s16, vdupq_n_s32, vdupq_n_u16, vextq_s16, vget_lane_u64, vget_low_s16, vmaxq_s16,
    vminq_s16, vminq_u16, vmlal_high_n_s16, vmlal_n_s16, vmlaq_n_s32, vmovn_u16, vmull_high_n_s16,
    vmull_n_s16, vqmovn_s32, vqmovun_s32, vreinterpret_u64_u8, vrshlq_s32,
};

use super::Rounding;
use crate::neon::{i16x8, i16x8_array};

/// Eight 16-bit values from `at`, or `None` past the end of `src`.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn load(src: &[i16], at: usize) -> Option<int16x8_t> {
    Some(i16x8(*src.get(at..)?.first_chunk::<8>()?))
}

/// The columns a pass filters: every eighth, and the last eight again
/// when the width is not a multiple of eight.
#[inline]
fn columns(w: usize) -> impl Iterator<Item = usize> {
    let whole = w / 8 * 8;
    (0..whole)
        .step_by(8)
        .chain((whole != w).then_some(w.saturating_sub(8)))
}

#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(super) fn horizontal(dst: &mut [i16], src: &[i16], taps: &[i16; 4], r: &Rounding) {
    let down = vdupq_n_s32(-(r.round0 as i32));
    let (low, high) = (vdupq_n_s16(r.low), vdupq_n_s16(r.high));
    for c in columns(dst.len()) {
        // Fifteen samples make eight outputs: the first eight, and the
        // eight that end with the last, turned so that they follow on.
        let (Some(a), Some(b)) = (load(src, c), load(src, c + 7)) else {
            return;
        };
        let b = vextq_s16::<1>(b, b);
        // The samples under tap k of the eight outputs.
        let x3 = vextq_s16::<3>(a, b);
        let p0 = vaddq_s16(a, vextq_s16::<6>(a, b));
        let p1 = vaddq_s16(vextq_s16::<1>(a, b), vextq_s16::<5>(a, b));
        let p2 = vaddq_s16(vextq_s16::<2>(a, b), vextq_s16::<4>(a, b));
        let mut lo = vmull_n_s16(vget_low_s16(x3), taps[3]);
        let mut hi = vmull_high_n_s16(x3, taps[3]);
        lo = vmlal_n_s16(lo, vget_low_s16(p0), taps[0]);
        hi = vmlal_high_n_s16(hi, p0, taps[0]);
        lo = vmlal_n_s16(lo, vget_low_s16(p1), taps[1]);
        hi = vmlal_high_n_s16(hi, p1, taps[1]);
        lo = vmlal_n_s16(lo, vget_low_s16(p2), taps[2]);
        hi = vmlal_high_n_s16(hi, p2, taps[2]);
        let v = vcombine_s16(
            vqmovn_s32(vrshlq_s32(lo, down)),
            vqmovn_s32(vrshlq_s32(hi, down)),
        );
        let v = vminq_s16(vmaxq_s16(v, low), high);
        if let Some(d) = dst.get_mut(c..).and_then(|d| d.first_chunk_mut::<8>()) {
            *d = i16x8_array(v);
        }
    }
}

/// The sums of four columns: the pairs of rows under equal taps added in
/// 32 bits, then each times its tap.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn sums(pairs: [int32x4_t; 3], centre: int32x4_t, taps: &[i16; 4]) -> int32x4_t {
    let mut sum = vmlaq_n_s32(centre, pairs[0], i32::from(taps[0]));
    sum = vmlaq_n_s32(sum, pairs[1], i32::from(taps[1]));
    vmlaq_n_s32(sum, pairs[2], i32::from(taps[2]))
}

#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(super) fn vertical(dst: &mut [u8], rows: &[&[i16]; 7], taps: &[i16; 4], r: &Rounding) {
    let down = vdupq_n_s32(-(r.round1 as i32));
    let max = vdupq_n_u16(r.max);
    for c in columns(dst.len()) {
        let mut m = [vdupq_n_s16(0); 7];
        for (v, row) in m.iter_mut().zip(rows) {
            let Some(loaded) = load(row, c) else {
                return;
            };
            *v = loaded;
        }
        let lo = sums(
            [
                vaddl_s16(vget_low_s16(m[0]), vget_low_s16(m[6])),
                vaddl_s16(vget_low_s16(m[1]), vget_low_s16(m[5])),
                vaddl_s16(vget_low_s16(m[2]), vget_low_s16(m[4])),
            ],
            vmull_n_s16(vget_low_s16(m[3]), taps[3]),
            taps,
        );
        let hi = sums(
            [
                vaddl_high_s16(m[0], m[6]),
                vaddl_high_s16(m[1], m[5]),
                vaddl_high_s16(m[2], m[4]),
            ],
            vmull_high_n_s16(m[3], taps[3]),
            taps,
        );
        // Round2, then the clip: below zero by the narrowing, above the
        // largest sample by the minimum.
        let v = vcombine_u16(
            vqmovun_s32(vrshlq_s32(lo, down)),
            vqmovun_s32(vrshlq_s32(hi, down)),
        );
        let v = vminq_u16(v, max);
        if let Some(d) = dst.get_mut(c..).and_then(|d| d.first_chunk_mut::<8>()) {
            *d = vget_lane_u64::<0>(vreinterpret_u64_u8(vmovn_u16(v))).to_le_bytes();
        }
    }
}
