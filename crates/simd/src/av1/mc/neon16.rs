//! The two-pass subpel filter on NEON for samples past 8 bits.
//!
//! A sample of 10 or 12 bits times a tap does not fit 16 bits, so the
//! horizontal pass widens: eight shifted copies of a strip's samples, each
//! times one tap, summed in 32 bits and rounded back to 16. The vertical
//! pass is the one the 8-bit kernel runs on its intermediates.

use core::arch::aarch64::{
    int16x8_t, int32x4_t, uint16x8_t, vcombine_s16, vcombine_u16, vdupq_n_s16, vdupq_n_s32,
    vdupq_n_u16, vextq_s16, vget_low_s16, vmaxq_s16, vminq_s16, vminq_u16, vmlal_high_laneq_s16,
    vmlal_laneq_s16, vmovn_s32, vmull_high_laneq_s16, vmull_laneq_s16, vqmovun_s32,
    vreinterpretq_s16_u16, vreinterpretq_u16_s16, vrshlq_s16, vrshlq_s32, vrshrn_n_s32, vsubq_s16,
};

use super::Block16;
use crate::neon::{i16x8, i16x8_array, u16x8, u16x8_array};

/// Where a block's prediction goes.
pub(super) enum Out<'a> {
    Prep(&'a mut [i16]),
    Put(&'a mut [u16]),
}

/// The value taken off a compound intermediate so that it fits 16 bits.
const BIAS: i32 = 8192;

/// Eight samples from `at` as signed lanes, or `None` past the end of
/// `src`. A sample is at most 4095, so the sign bit is clear.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes,dotprod"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon,dotprod"))]
fn samples8(src: &[u16], at: usize) -> Option<int16x8_t> {
    let px = src.get(at..)?.first_chunk::<8>()?;
    Some(vreinterpretq_s16_u16(u16x8(px)))
}

/// The horizontal filter of one source row of a strip of eight columns:
/// `Round2(sum, InterRound0)` of the sixteen samples from `at`, as 16
/// bits, or `None` past the end of `src`.
#[derive(Clone, Copy)]
struct Horizontal<'a> {
    src: &'a [u16],
    taps: int16x8_t,
    /// Minus `InterRound0`: a rounding shift to the right.
    down: int32x4_t,
    /// The outer taps are 0, as in most filter rows.
    six: bool,
}

impl Horizontal<'_> {
    #[inline]
    #[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes,dotprod"))]
    #[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon,dotprod"))]
    fn row(&self, at: usize) -> Option<int16x8_t> {
        let a = samples8(self.src, at)?;
        let b = samples8(self.src, at + 8)?;
        let t = self.taps;
        // The samples under tap k of the eight outputs.
        let (s1, s2, s3) = (
            vextq_s16::<1>(a, b),
            vextq_s16::<2>(a, b),
            vextq_s16::<3>(a, b),
        );
        let (s4, s5, s6) = (
            vextq_s16::<4>(a, b),
            vextq_s16::<5>(a, b),
            vextq_s16::<6>(a, b),
        );
        let mut lo = vmull_laneq_s16::<1>(vget_low_s16(s1), t);
        let mut hi = vmull_high_laneq_s16::<1>(s1, t);
        lo = vmlal_laneq_s16::<2>(lo, vget_low_s16(s2), t);
        hi = vmlal_high_laneq_s16::<2>(hi, s2, t);
        lo = vmlal_laneq_s16::<3>(lo, vget_low_s16(s3), t);
        hi = vmlal_high_laneq_s16::<3>(hi, s3, t);
        lo = vmlal_laneq_s16::<4>(lo, vget_low_s16(s4), t);
        hi = vmlal_high_laneq_s16::<4>(hi, s4, t);
        lo = vmlal_laneq_s16::<5>(lo, vget_low_s16(s5), t);
        hi = vmlal_high_laneq_s16::<5>(hi, s5, t);
        lo = vmlal_laneq_s16::<6>(lo, vget_low_s16(s6), t);
        hi = vmlal_high_laneq_s16::<6>(hi, s6, t);
        if !self.six {
            let s7 = vextq_s16::<7>(a, b);
            lo = vmlal_laneq_s16::<0>(lo, vget_low_s16(a), t);
            hi = vmlal_high_laneq_s16::<0>(hi, a, t);
            lo = vmlal_laneq_s16::<7>(lo, vget_low_s16(s7), t);
            hi = vmlal_high_laneq_s16::<7>(hi, s7, t);
        }
        Some(vcombine_s16(
            vmovn_s32(vrshlq_s32(lo, self.down)),
            vmovn_s32(vrshlq_s32(hi, self.down)),
        ))
    }
}

/// The samples themselves as the rows of the vertical filter (the
/// horizontal fraction is 0): the eight from three columns past `at`.
#[derive(Clone, Copy)]
struct Plain<'a> {
    src: &'a [u16],
}

impl Plain<'_> {
    #[inline]
    #[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes,dotprod"))]
    #[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon,dotprod"))]
    fn row(&self, at: usize) -> Option<int16x8_t> {
        samples8(self.src, at + 3)
    }
}

/// Stores eight 16-bit values at `dst[at..]`, if they fit.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes,dotprod"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon,dotprod"))]
fn store_i16(dst: &mut [i16], at: usize, v: int16x8_t) {
    if let Some(d) = dst.get_mut(at..).and_then(|d| d.first_chunk_mut::<8>()) {
        *d = i16x8_array(v);
    }
}

/// Stores eight samples at `dst[at..]`, if they fit.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes,dotprod"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon,dotprod"))]
fn store_u16(dst: &mut [u16], at: usize, v: uint16x8_t) {
    if let Some(d) = dst.get_mut(at..).and_then(|d| d.first_chunk_mut::<8>()) {
        *d = u16x8_array(v);
    }
}

/// `Round2` of both halves of the sums by minus `down`, clipped to
/// `0..=max`.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes,dotprod"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon,dotprod"))]
fn clipped(lo: int32x4_t, hi: int32x4_t, down: int32x4_t, max: uint16x8_t) -> uint16x8_t {
    let v = vcombine_u16(
        vqmovun_s32(vrshlq_s32(lo, down)),
        vqmovun_s32(vrshlq_s32(hi, down)),
    );
    vminq_u16(v, max)
}

/// The vertical filter over the rows `$hz` gives, for both tap spans.
macro_rules! vertical {
    ($hz:expr, $b:expr, $round:expr, $store:expr) => {{
        // A filter's outer taps are often 0: six rows then make an output.
        if $b.fv[0] == 0 && $b.fv[7] == 0 {
            strips!($hz, $b, $round, $store, 1; [v0, v1, v2, v3, v4]; v5;
                [v5; v0 0, v1 1, v2 2, v3 3, v4 4, v5 5]
                [v0; v1 0, v2 1, v3 2, v4 3, v5 4, v0 5]
                [v1; v2 0, v3 1, v4 2, v5 3, v0 4, v1 5]
                [v2; v3 0, v4 1, v5 2, v0 3, v1 4, v2 5]
                [v3; v4 0, v5 1, v0 2, v1 3, v2 4, v3 5]
                [v4; v5 0, v0 1, v1 2, v2 3, v3 4, v4 5])
        } else {
            strips!($hz, $b, $round, $store, 0; [v0, v1, v2, v3, v4, v5, v6]; v7;
                [v7; v0 0, v1 1, v2 2, v3 3, v4 4, v5 5, v6 6, v7 7]
                [v0; v1 0, v2 1, v3 2, v4 3, v5 4, v6 5, v7 6, v0 7]
                [v1; v2 0, v3 1, v4 2, v5 3, v6 4, v7 5, v0 6, v1 7]
                [v2; v3 0, v4 1, v5 2, v6 3, v7 4, v0 5, v1 6, v2 7]
                [v3; v4 0, v5 1, v6 2, v7 3, v0 4, v1 5, v2 6, v3 7]
                [v4; v5 0, v6 1, v7 2, v0 3, v1 4, v2 5, v3 6, v4 7]
                [v5; v6 0, v7 1, v0 2, v1 3, v2 4, v3 5, v4 6, v5 7]
                [v6; v7 0, v0 1, v1 2, v2 3, v3 4, v4 5, v5 6, v6 7])
        }
    }};
}

/// The horizontal filter of `b`.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes,dotprod"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon,dotprod"))]
fn horizontal<'a>(b: &Block16<'a>) -> Horizontal<'a> {
    Horizontal {
        src: b.src,
        taps: i16x8(b.fh.map(i16::from)),
        down: vdupq_n_s32(-b.round0()),
        six: b.fh[0] == 0 && b.fh[7] == 0,
    }
}

/// Horizontal then vertical filter of a block, eight columns at a time.
/// `out` chooses the rounding of the vertical pass.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes,dotprod"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon,dotprod"))]
pub(super) fn hv(out: Out<'_>, dst_stride: usize, b: &Block16<'_>) {
    let hz = horizontal(b);
    match out {
        Out::Prep(dst) => {
            // Round2(sum, 7) less the bias: the sums start at minus it.
            let round = vdupq_n_s32(-(BIAS << 7));
            let store = |c: usize, r: usize, lo: int32x4_t, hi: int32x4_t| {
                let v = vcombine_s16(vrshrn_n_s32::<7>(lo), vrshrn_n_s32::<7>(hi));
                store_i16(dst, r * dst_stride + c, v);
            };
            vertical!(&hz, b, round, store)
        }
        Out::Put(dst) => {
            let round = vdupq_n_s32(0);
            let down = vdupq_n_s32(-b.round1());
            let max = vdupq_n_u16(b.max());
            let store = |c: usize, r: usize, lo: int32x4_t, hi: int32x4_t| {
                store_u16(dst, r * dst_stride + c, clipped(lo, hi, down, max));
            };
            vertical!(&hz, b, round, store)
        }
    }
}

/// The horizontal filter alone (the vertical fraction is 0), eight
/// columns at a time.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes,dotprod"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon,dotprod"))]
pub(super) fn h(out: Out<'_>, dst_stride: usize, b: &Block16<'_>) {
    let hz = horizontal(b);
    let first = b.start + 3 * b.src_stride;
    match out {
        Out::Prep(dst) => {
            let bias = vdupq_n_s16(BIAS as i16);
            for r in 0..b.h {
                let (from, to) = (first + r * b.src_stride, r * dst_stride);
                for c in (0..b.w).step_by(8) {
                    let Some(v) = hz.row(from + c) else {
                        return;
                    };
                    store_i16(dst, to + c, vsubq_s16(v, bias));
                }
            }
        }
        Out::Put(dst) => {
            // The vertical identity: Round2(128 * mid, InterRound1).
            let down = vdupq_n_s16((7 - b.round1()) as i16);
            let (zero, max) = (vdupq_n_s16(0), vdupq_n_s16(b.max() as i16));
            for r in 0..b.h {
                let (from, to) = (first + r * b.src_stride, r * dst_stride);
                for c in (0..b.w).step_by(8) {
                    let Some(v) = hz.row(from + c) else {
                        return;
                    };
                    let v = vminq_s16(vmaxq_s16(vrshlq_s16(v, down), zero), max);
                    store_u16(dst, to + c, vreinterpretq_u16_s16(v));
                }
            }
        }
    }
}

/// The vertical filter alone (the horizontal fraction is 0), eight
/// columns at a time.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes,dotprod"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon,dotprod"))]
pub(super) fn v(out: Out<'_>, dst_stride: usize, b: &Block16<'_>) {
    let rows = Plain { src: b.src };
    match out {
        Out::Prep(dst) => {
            // The horizontal identity gives `x << (7 - InterRound0)`, so
            // the output is Round2(sum, InterRound0), less the bias.
            let r0 = b.round0();
            let round = vdupq_n_s32(-(BIAS << r0));
            let down = vdupq_n_s32(-r0);
            let store = |c: usize, r: usize, lo: int32x4_t, hi: int32x4_t| {
                let v = vcombine_s16(
                    vmovn_s32(vrshlq_s32(lo, down)),
                    vmovn_s32(vrshlq_s32(hi, down)),
                );
                store_i16(dst, r * dst_stride + c, v);
            };
            vertical!(&rows, b, round, store)
        }
        Out::Put(dst) => {
            // Round2(sum, InterRound1 + InterRound0 - 7): 7 at every depth.
            let round = vdupq_n_s32(0);
            let down = vdupq_n_s32(-7);
            let max = vdupq_n_u16(b.max());
            let store = |c: usize, r: usize, lo: int32x4_t, hi: int32x4_t| {
                store_u16(dst, r * dst_stride + c, clipped(lo, hi, down, max));
            };
            vertical!(&rows, b, round, store)
        }
    }
}
