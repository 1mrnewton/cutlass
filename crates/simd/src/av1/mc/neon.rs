//! The two-pass subpel filter on NEON.
//!
//! The horizontal pass takes four samples times four taps in one dot
//! product instruction. That instruction multiplies signed bytes, so the
//! samples go in as `sample - 128` and the sum gets 128 times the sum of
//! the taps back.

use core::arch::aarch64::{
    int8x16_t, int16x8_t, int32x4_t, uint8x16_t, vcombine_s16, vcombine_u16, vdotq_s32,
    vdupq_n_s16, vdupq_n_s32, vdupq_n_u8, veorq_u8, vget_lane_u64, vget_low_s16, vmlal_high_s16,
    vmlal_s16, vqmovn_u16, vqshrun_n_s32, vqtbl1q_s8, vreinterpret_u64_u8, vreinterpretq_s8_u8,
    vshrn_n_s32,
};

use super::Block;
use crate::neon::{i16x8_array, u8x16};

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

/// The horizontal filter of one source row of a strip of eight columns:
/// `Round2(sum, 3)` of the sixteen samples from `at`, as 16 bits, or
/// `None` past the end of `src`.
#[derive(Clone, Copy)]
struct Horizontal<'a> {
    src: &'a [u8],
    lo: int8x16_t,
    hi: int8x16_t,
    idx: [uint8x16_t; 3],
    sign: uint8x16_t,
    bias: int32x4_t,
}

impl Horizontal<'_> {
    #[inline]
    #[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes,dotprod"))]
    #[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon,dotprod"))]
    fn row(&self, at: usize) -> Option<int16x8_t> {
        let px = self.src.get(at..)?.first_chunk::<16>()?;
        let s = vreinterpretq_s8_u8(veorq_u8(u8x16(px), self.sign));
        let (a, b, d) = (
            vqtbl1q_s8(s, self.idx[0]),
            vqtbl1q_s8(s, self.idx[1]),
            vqtbl1q_s8(s, self.idx[2]),
        );
        let first = vdotq_s32(vdotq_s32(self.bias, a, self.lo), b, self.hi);
        let second = vdotq_s32(vdotq_s32(self.bias, b, self.lo), d, self.hi);
        Some(vcombine_s16(
            vshrn_n_s32::<3>(first),
            vshrn_n_s32::<3>(second),
        ))
    }
}

/// Both passes over a block, eight columns at a time, the vertical taps
/// `$t0 .. $t0 + N` for the `N` names after `;`: the last `N` rows of the
/// horizontal filter stay in registers, one in each name, and each
/// output row filters one more source row. `$store` takes the column,
/// the row and the rounded vertical sums. Names rather than an array,
/// which the compiler keeps on the stack.
macro_rules! strips {
    ($hz:expr, $b:expr, $round:expr, $store:expr, $t0:literal; $($v:ident),+; $last:ident) => {{
        let (hz, b, round, mut store) = ($hz, $b, $round, $store);
        let taps: [int16x8_t; 8] = core::array::from_fn(|t| {
            vdupq_n_s16(b.fv.get($t0 + t).map_or(0, |&f| i16::from(f)))
        });
        'strips: for c in (0..b.w).step_by(8) {
            let mut at = b.start + $t0 * b.src_stride + c;
            $(
                let Some(mut $v) = hz.row(at) else {
                    break 'strips;
                };
                at += b.src_stride;
            )+
            for r in 0..b.h {
                let Some($last) = hz.row(at) else {
                    break 'strips;
                };
                at += b.src_stride;
                let (mut lo, mut hi) = (round, round);
                let mut t = 0;
                $(
                    lo = vmlal_s16(lo, vget_low_s16($v), vget_low_s16(taps[t]));
                    hi = vmlal_high_s16(hi, $v, taps[t]);
                    t += 1;
                )+
                lo = vmlal_s16(lo, vget_low_s16($last), vget_low_s16(taps[t]));
                hi = vmlal_high_s16(hi, $last, taps[t]);
                store(c, r, lo, hi);
                strips!(@shift $($v),+; $last);
            }
        }
    }};
    (@shift $a:ident, $($rest:ident),+; $last:ident) => {
        $a = strips!(@first $($rest),+);
        strips!(@shift $($rest),+; $last);
    };
    (@shift $a:ident; $last:ident) => {
        $a = $last;
    };
    (@first $a:ident $(, $rest:ident)*) => {
        $a
    };
}

/// Horizontal then vertical filter of a block, eight columns at a time.
/// `out` chooses the rounding of the vertical pass.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes,dotprod"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon,dotprod"))]
pub(super) fn hv(out: Out<'_>, dst_stride: usize, b: &Block<'_>) {
    let fh = b.fh;
    // Round2(sum, 3), with what the signed samples leave out: 128 times
    // the sum of the taps.
    let all: i32 = fh.iter().map(|&f| i32::from(f)).sum();
    let hz = Horizontal {
        src: b.src,
        lo: taps4([fh[0], fh[1], fh[2], fh[3]]),
        hi: taps4([fh[4], fh[5], fh[6], fh[7]]),
        idx: FIRST.map(|i| u8x16(&i)),
        sign: vdupq_n_u8(0x80),
        bias: vdupq_n_s32(128 * all + 4),
    };
    // A filter's outer taps are often 0: six rows then make an output.
    let six = b.fv[0] == 0 && b.fv[7] == 0;
    match out {
        Out::Prep(dst) => {
            let round = vdupq_n_s32(1 << 6);
            let store = |c: usize, r: usize, lo: int32x4_t, hi: int32x4_t| {
                let v = vcombine_s16(vshrn_n_s32::<7>(lo), vshrn_n_s32::<7>(hi));
                let at = r * dst_stride + c;
                if let Some(d) = dst.get_mut(at..at + 8) {
                    d.copy_from_slice(&i16x8_array(v));
                }
            };
            if six {
                strips!(&hz, b, round, store, 1; v0, v1, v2, v3, v4; v5)
            } else {
                strips!(&hz, b, round, store, 0; v0, v1, v2, v3, v4, v5, v6; v7)
            }
        }
        Out::Put(dst) => {
            let round = vdupq_n_s32(1 << 10);
            let store = |c: usize, r: usize, lo: int32x4_t, hi: int32x4_t| {
                let v = vcombine_u16(vqshrun_n_s32::<11>(lo), vqshrun_n_s32::<11>(hi));
                let bytes = vget_lane_u64::<0>(vreinterpret_u64_u8(vqmovn_u16(v)));
                let at = r * dst_stride + c;
                if let Some(d) = dst.get_mut(at..at + 8) {
                    d.copy_from_slice(&bytes.to_le_bytes());
                }
            };
            if six {
                strips!(&hz, b, round, store, 1; v0, v1, v2, v3, v4; v5)
            } else {
                strips!(&hz, b, round, store, 0; v0, v1, v2, v3, v4, v5, v6; v7)
            }
        }
    }
}
