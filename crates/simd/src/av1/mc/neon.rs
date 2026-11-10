//! The two-pass subpel filter on NEON.
//!
//! The horizontal pass takes four samples times four taps in one dot
//! product instruction. That instruction multiplies signed bytes, so the
//! samples go in as `sample - 128` and the sum gets 128 times the sum of
//! the taps back.

use core::arch::aarch64::{
    int8x16_t, int16x8_t, int32x4_t, vcombine_s16, vcombine_u16, vdotq_s32, vdupq_n_s16,
    vdupq_n_s32, vdupq_n_u8, veorq_u8, vget_lane_u64, vget_low_s16, vmlal_high_s16, vmlal_s16,
    vqmovn_u16, vqshrun_n_s32, vqtbl1q_s8, vreinterpret_u64_u8, vreinterpretq_s8_u8, vshrn_n_s32,
};

use super::Block;
use crate::neon::{i16x8, i16x8_array, u8x16};

/// Where a block's prediction goes.
pub(super) enum Out<'a> {
    Prep(&'a mut [i16]),
    Put(&'a mut [u8]),
}

/// For the outputs `4 k .. 4 k + 4` of a group of eight: the four
/// samples under the first four taps of each output, read from the
/// sixteen samples that start at the group's first.
const FIRST: [[u8; 16]; 3] = [
    [0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6],
    [4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10],
    [8, 9, 10, 11, 9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14],
];

/// Four taps, once for each of four outputs.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes,dotprod"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon,dotprod"))]
fn taps4(f: [i8; 4]) -> int8x16_t {
    let b: [u8; 16] = core::array::from_fn(|i| f[i % 4] as u8);
    vreinterpretq_s8_u8(u8x16(&b))
}

/// The vertical pass of eight columns: `rows` holds the horizontal
/// filter's output for the strip, a row of eight for every source row
/// from the first one the taps `T0 .. T0 + N` read. One output row for
/// every window of `N` rows, written to `store`.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes,dotprod"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon,dotprod"))]
fn vertical<const T0: usize, const N: usize>(
    rows: &[[i16; 8]],
    fv: &[i8; 8],
    put: bool,
    mut store: impl FnMut(usize, int32x4_t, int32x4_t),
) {
    let taps: [int16x8_t; N] = core::array::from_fn(|t| vdupq_n_s16(i16::from(fv[T0 + t])));
    let round = vdupq_n_s32(if put { 1 << 10 } else { 1 << 6 });
    for (r, win) in rows.windows(N).enumerate() {
        let (mut lo, mut hi) = (round, round);
        for (m, f) in win.iter().zip(&taps) {
            let m = i16x8(*m);
            lo = vmlal_s16(lo, vget_low_s16(m), vget_low_s16(*f));
            hi = vmlal_high_s16(hi, m, *f);
        }
        store(r, lo, hi);
    }
}

/// Horizontal then vertical filter of a block, eight columns at a time:
/// each strip of eight columns goes through the horizontal filter for
/// all its rows, into `mid`, then through the vertical one. `out`
/// chooses the rounding of the vertical pass.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes,dotprod"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon,dotprod"))]
pub(super) fn hv(mut out: Out<'_>, dst_stride: usize, b: &Block<'_>, mid: &mut [i16]) {
    let Block {
        src,
        start,
        src_stride,
        w,
        h,
        fh,
        fv,
    } = *b;
    let put = matches!(out, Out::Put(_));
    // A filter's outer taps are often 0: six rows then make an output.
    let six = fv[0] == 0 && fv[7] == 0;
    let (t0, n) = if six { (1, 6) } else { (0, 8) };
    let (lo, hi) = (
        taps4([fh[0], fh[1], fh[2], fh[3]]),
        taps4([fh[4], fh[5], fh[6], fh[7]]),
    );
    let idx = FIRST.map(|i| u8x16(&i));
    let sign = vdupq_n_u8(0x80);
    // Round2(sum, 3), with what the signed samples leave out: 128 times
    // the sum of the taps.
    let all: i32 = fh.iter().map(|&f| i32::from(f)).sum();
    let bias = vdupq_n_s32(128 * all + 4);
    let Some(strip) = mid.as_chunks_mut::<8>().0.get_mut(..h + n - 1) else {
        return;
    };
    for c in (0..w).step_by(8) {
        for (r, m) in strip.iter_mut().enumerate() {
            let at = start + (t0 + r) * src_stride + c;
            let Some(px) = src
                .get(at..at + 16)
                .and_then(|s| <&[u8; 16]>::try_from(s).ok())
            else {
                return;
            };
            let s = vreinterpretq_s8_u8(veorq_u8(u8x16(px), sign));
            let (a, b, d) = (
                vqtbl1q_s8(s, idx[0]),
                vqtbl1q_s8(s, idx[1]),
                vqtbl1q_s8(s, idx[2]),
            );
            let first = vdotq_s32(vdotq_s32(bias, a, lo), b, hi);
            let second = vdotq_s32(vdotq_s32(bias, b, lo), d, hi);
            *m = i16x8_array(vcombine_s16(
                vshrn_n_s32::<3>(first),
                vshrn_n_s32::<3>(second),
            ));
        }
        let rows = &*strip;
        match &mut out {
            Out::Prep(dst) => {
                let store = |r: usize, lo: int32x4_t, hi: int32x4_t| {
                    let v = vcombine_s16(vshrn_n_s32::<7>(lo), vshrn_n_s32::<7>(hi));
                    let at = r * dst_stride + c;
                    if let Some(d) = dst.get_mut(at..at + 8) {
                        d.copy_from_slice(&i16x8_array(v));
                    }
                };
                if six {
                    vertical::<1, 6>(rows, fv, put, store);
                } else {
                    vertical::<0, 8>(rows, fv, put, store);
                }
            }
            Out::Put(dst) => {
                let store = |r: usize, lo: int32x4_t, hi: int32x4_t| {
                    let v = vcombine_u16(vqshrun_n_s32::<11>(lo), vqshrun_n_s32::<11>(hi));
                    let bytes = vget_lane_u64::<0>(vreinterpret_u64_u8(vqmovn_u16(v)));
                    let at = r * dst_stride + c;
                    if let Some(d) = dst.get_mut(at..at + 8) {
                        d.copy_from_slice(&bytes.to_le_bytes());
                    }
                };
                if six {
                    vertical::<1, 6>(rows, fv, put, store);
                } else {
                    vertical::<0, 8>(rows, fv, put, store);
                }
            }
        }
    }
}
