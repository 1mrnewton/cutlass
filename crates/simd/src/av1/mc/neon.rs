//! The two-pass subpel filter on NEON.
//!
//! The horizontal pass takes four samples times four taps in one dot
//! product instruction. That instruction multiplies signed bytes, so the
//! samples go in as `sample - 128` and the sum gets 128 times the sum of
//! the taps back.

use core::arch::aarch64::{
    int8x16_t, int16x8_t, int32x4_t, uint8x16_t, vcombine_s16, vcombine_u16, vdotq_s32,
    vdupq_n_s32, vdupq_n_u8, veorq_u8, vget_lane_u64, vget_low_s16, vget_low_u8,
    vmlal_high_laneq_s16, vmlal_laneq_s16, vmlaq_laneq_s16, vmovl_high_u8, vmovl_u8,
    vmulq_laneq_s16, vqmovn_u16, vqrshrun_n_s16, vqrshrun_n_s32, vqtbl1q_s8, vreinterpret_u64_u8,
    vreinterpretq_s8_u8, vreinterpretq_s16_u16, vrshrn_n_s32, vrshrq_n_s16, vshrn_n_s32,
};

use super::Block;
use crate::neon::{i16x8, i16x8_array, u8x8, u8x16};

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

/// The vertical sums of one output row, low and high halves: `$round`
/// plus each named row times the lane of `$taps` after it.
macro_rules! sums {
    ($round:expr, $taps:expr; $($r:ident $l:literal),+) => {{
        let (mut lo, mut hi) = ($round, $round);
        $(
            lo = vmlal_laneq_s16::<$l>(lo, vget_low_s16($r), $taps);
            hi = vmlal_high_laneq_s16::<$l>(hi, $r, $taps);
        )+
        (lo, hi)
    }};
}

/// Both passes over a block, eight columns at a time, the vertical taps
/// `$t0 ..` in the lanes of one vector. The rows of the horizontal filter
/// an output row reads stay in registers, one in each name: `$init` the
/// first ones, and `$spare` the name the next row goes to. Each step in
/// brackets puts a new row in its first name, the one no later output
/// reads, and filters the names after it with the lanes beside them: as
/// many steps as names, so the names go round and no row moves between
/// registers. `$store` takes the column, the row and the
/// vertical sums. Names rather than an array, which the compiler keeps
/// on the stack.
macro_rules! strips {
    (
        $hz:expr, $b:expr, $round:expr, $store:expr, $t0:literal;
        [$($init:ident),+]; $spare:ident;
        $([$new:ident; $($r:ident $l:literal),+])+
    ) => {{
        let (hz, b, round, mut store) = ($hz, $b, $round, $store);
        // Behind `black_box`, so that the multiplies take their lanes
        // rather than taps the compiler works out apart.
        let taps = core::hint::black_box(i16x8(core::array::from_fn(|t| {
            b.fv.get($t0 + t).map_or(0, |&f| i16::from(f))
        })));
        'strips: for c in (0..b.w).step_by(8) {
            let mut at = b.start + $t0 * b.src_stride + c;
            $(
                let Some(mut $init) = hz.row(at) else {
                    break 'strips;
                };
                at += b.src_stride;
            )+
            let mut $spare: int16x8_t;
            let mut r = 0;
            'rows: loop {
                $(
                    if r == b.h {
                        break 'rows;
                    }
                    let Some(next) = hz.row(at) else {
                        break 'strips;
                    };
                    $new = next;
                    at += b.src_stride;
                    let (lo, hi) = sums!(round, taps; $($r $l),+);
                    store(c, r, lo, hi);
                    r += 1;
                )+
            }
        }
    }};
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
            // The stores round: the sums start at 0.
            let round = vdupq_n_s32(0);
            let store = |c: usize, r: usize, lo: int32x4_t, hi: int32x4_t| {
                let v = vcombine_s16(vrshrn_n_s32::<7>(lo), vrshrn_n_s32::<7>(hi));
                store_i16(dst, r * dst_stride + c, v);
            };
            if six {
                strips!(&hz, b, round, store, 1; [v0, v1, v2, v3, v4]; v5;
                    [v5; v0 0, v1 1, v2 2, v3 3, v4 4, v5 5]
                    [v0; v1 0, v2 1, v3 2, v4 3, v5 4, v0 5]
                    [v1; v2 0, v3 1, v4 2, v5 3, v0 4, v1 5]
                    [v2; v3 0, v4 1, v5 2, v0 3, v1 4, v2 5]
                    [v3; v4 0, v5 1, v0 2, v1 3, v2 4, v3 5]
                    [v4; v5 0, v0 1, v1 2, v2 3, v3 4, v4 5])
            } else {
                strips!(&hz, b, round, store, 0; [v0, v1, v2, v3, v4, v5, v6]; v7;
                    [v7; v0 0, v1 1, v2 2, v3 3, v4 4, v5 5, v6 6, v7 7]
                    [v0; v1 0, v2 1, v3 2, v4 3, v5 4, v6 5, v7 6, v0 7]
                    [v1; v2 0, v3 1, v4 2, v5 3, v6 4, v7 5, v0 6, v1 7]
                    [v2; v3 0, v4 1, v5 2, v6 3, v7 4, v0 5, v1 6, v2 7]
                    [v3; v4 0, v5 1, v6 2, v7 3, v0 4, v1 5, v2 6, v3 7]
                    [v4; v5 0, v6 1, v7 2, v0 3, v1 4, v2 5, v3 6, v4 7]
                    [v5; v6 0, v7 1, v0 2, v1 3, v2 4, v3 5, v4 6, v5 7]
                    [v6; v7 0, v0 1, v1 2, v2 3, v3 4, v4 5, v5 6, v6 7])
            }
        }
        Out::Put(dst) => {
            let round = vdupq_n_s32(0);
            let store = |c: usize, r: usize, lo: int32x4_t, hi: int32x4_t| {
                let v = vcombine_u16(vqrshrun_n_s32::<11>(lo), vqrshrun_n_s32::<11>(hi));
                let bytes = vget_lane_u64::<0>(vreinterpret_u64_u8(vqmovn_u16(v)));
                store_bytes(dst, r * dst_stride + c, bytes);
            };
            if six {
                strips!(&hz, b, round, store, 1; [v0, v1, v2, v3, v4]; v5;
                    [v5; v0 0, v1 1, v2 2, v3 3, v4 4, v5 5]
                    [v0; v1 0, v2 1, v3 2, v4 3, v5 4, v0 5]
                    [v1; v2 0, v3 1, v4 2, v5 3, v0 4, v1 5]
                    [v2; v3 0, v4 1, v5 2, v0 3, v1 4, v2 5]
                    [v3; v4 0, v5 1, v0 2, v1 3, v2 4, v3 5]
                    [v4; v5 0, v0 1, v1 2, v2 3, v3 4, v4 5])
            } else {
                strips!(&hz, b, round, store, 0; [v0, v1, v2, v3, v4, v5, v6]; v7;
                    [v7; v0 0, v1 1, v2 2, v3 3, v4 4, v5 5, v6 6, v7 7]
                    [v0; v1 0, v2 1, v3 2, v4 3, v5 4, v6 5, v7 6, v0 7]
                    [v1; v2 0, v3 1, v4 2, v5 3, v6 4, v7 5, v0 6, v1 7]
                    [v2; v3 0, v4 1, v5 2, v6 3, v7 4, v0 5, v1 6, v2 7]
                    [v3; v4 0, v5 1, v6 2, v7 3, v0 4, v1 5, v2 6, v3 7]
                    [v4; v5 0, v6 1, v7 2, v0 3, v1 4, v2 5, v3 6, v4 7]
                    [v5; v6 0, v7 1, v0 2, v1 3, v2 4, v3 5, v4 6, v5 7]
                    [v6; v7 0, v0 1, v1 2, v2 3, v3 4, v4 5, v5 6, v6 7])
            }
        }
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

/// Stores eight bytes (in memory order) at `dst[at..]`, if they fit.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes,dotprod"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon,dotprod"))]
fn store_bytes(dst: &mut [u8], at: usize, bytes: u64) {
    if let Some(d) = dst.get_mut(at..).and_then(|d| d.first_chunk_mut::<8>()) {
        *d = bytes.to_le_bytes();
    }
}

/// [`store_bytes`] of `Round2(v, N)` clipped to 8 bits.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes,dotprod"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon,dotprod"))]
fn store_u8<const N: i32>(dst: &mut [u8], at: usize, v: int16x8_t) {
    store_bytes(
        dst,
        at,
        vget_lane_u64::<0>(vreinterpret_u64_u8(vqrshrun_n_s16::<N>(v))),
    );
}

/// The horizontal filter alone (the vertical fraction is 0), eight
/// columns at a time.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes,dotprod"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon,dotprod"))]
pub(super) fn h(out: Out<'_>, dst_stride: usize, b: &Block<'_>) {
    let fh = b.fh;
    let all: i32 = fh.iter().map(|&f| i32::from(f)).sum();
    let hz = Horizontal {
        src: b.src,
        lo: taps4([fh[0], fh[1], fh[2], fh[3]]),
        hi: taps4([fh[4], fh[5], fh[6], fh[7]]),
        idx: FIRST.map(|i| u8x16(&i)),
        sign: vdupq_n_u8(0x80),
        bias: vdupq_n_s32(128 * all + 4),
    };
    let first = b.start + 3 * b.src_stride;
    match out {
        Out::Prep(dst) => {
            for r in 0..b.h {
                let (from, to) = (first + r * b.src_stride, r * dst_stride);
                for c in (0..b.w).step_by(8) {
                    let Some(v) = hz.row(from + c) else {
                        return;
                    };
                    store_i16(dst, to + c, v);
                }
            }
        }
        Out::Put(dst) => {
            for r in 0..b.h {
                let (from, to) = (first + r * b.src_stride, r * dst_stride);
                for c in (0..b.w).step_by(8) {
                    let Some(v) = hz.row(from + c) else {
                        return;
                    };
                    store_u8::<4>(dst, to + c, v);
                }
            }
        }
    }
}

/// Eight samples from `at` widened to 16 bits, or `None` past the end of
/// `src`.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes,dotprod"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon,dotprod"))]
fn wide8(src: &[u8], at: usize) -> Option<int16x8_t> {
    let px = src.get(at..)?.first_chunk::<8>()?;
    Some(vreinterpretq_s16_u16(vmovl_u8(u8x8(px))))
}

/// Sixteen samples from `at` widened to 16 bits, as two halves, or `None`
/// past the end of `src`.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes,dotprod"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon,dotprod"))]
fn wide16(src: &[u8], at: usize) -> Option<(int16x8_t, int16x8_t)> {
    let px = u8x16(src.get(at..)?.first_chunk::<16>()?);
    Some((
        vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(px))),
        vreinterpretq_s16_u16(vmovl_high_u8(px)),
    ))
}

/// The vertical filter alone over a block, eight columns at a time, with
/// the taps `$t0 ..` halved (they are all even) so the sums fit 16 bits:
/// the rows an output reads stay in registers and go round the names, as
/// in `strips!`. `$store` takes the column, the row and the sum.
macro_rules! vstrips {
    (
        $b:expr, $store:expr, $n:literal, $load:ident, $mul:ident, $mla:ident, $each:ident, $t0:literal;
        [$($init:ident),+]; $spare:ident;
        $([$new:ident; $r0:ident $l0:literal $(, $r:ident $l:literal)*])+
    ) => {{
        let (b, mut store) = ($b, $store);
        // As in `strips!`.
        let taps = core::hint::black_box(i16x8(core::array::from_fn(|t| {
            b.fv.get($t0 + t).map_or(0, |&f| i16::from(f) >> 1)
        })));
        let top = b.start + 3;
        'strips: for c in (0..b.w).step_by($n) {
            let mut at = top + $t0 * b.src_stride + c;
            $(
                let Some(mut $init) = $load(b.src, at) else {
                    break 'strips;
                };
                at += b.src_stride;
            )+
            let mut $spare;
            let mut r = 0;
            'rows: loop {
                $(
                    if r == b.h {
                        break 'rows;
                    }
                    let Some(next) = $load(b.src, at) else {
                        break 'strips;
                    };
                    $new = next;
                    at += b.src_stride;
                    let mut acc = $mul::<$l0>($r0, taps);
                    $(
                        acc = $mla::<$l>(acc, $r, taps);
                    )*
                    $each(acc, |k, v| store(c + 8 * k, r, v));
                    r += 1;
                )+
            }
        }
    }};
}

/// The multiply, multiply-add and store of [`vstrips`] on eight lanes.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes,dotprod"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon,dotprod"))]
fn mul8<const L: i32>(a: int16x8_t, t: int16x8_t) -> int16x8_t {
    vmulq_laneq_s16::<L>(a, t)
}

#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes,dotprod"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon,dotprod"))]
fn mla8<const L: i32>(acc: int16x8_t, a: int16x8_t, t: int16x8_t) -> int16x8_t {
    vmlaq_laneq_s16::<L>(acc, a, t)
}

#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes,dotprod"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon,dotprod"))]
fn each8(acc: int16x8_t, mut f: impl FnMut(usize, int16x8_t)) {
    f(0, acc);
}

/// The same on sixteen lanes, as two halves.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes,dotprod"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon,dotprod"))]
fn mul16<const L: i32>(a: (int16x8_t, int16x8_t), t: int16x8_t) -> (int16x8_t, int16x8_t) {
    (vmulq_laneq_s16::<L>(a.0, t), vmulq_laneq_s16::<L>(a.1, t))
}

#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes,dotprod"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon,dotprod"))]
fn mla16<const L: i32>(
    acc: (int16x8_t, int16x8_t),
    a: (int16x8_t, int16x8_t),
    t: int16x8_t,
) -> (int16x8_t, int16x8_t) {
    (
        vmlaq_laneq_s16::<L>(acc.0, a.0, t),
        vmlaq_laneq_s16::<L>(acc.1, a.1, t),
    )
}

#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes,dotprod"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon,dotprod"))]
fn each16(acc: (int16x8_t, int16x8_t), mut f: impl FnMut(usize, int16x8_t)) {
    f(0, acc.0);
    f(1, acc.1);
}

/// The vertical filter alone (the horizontal fraction is 0), eight
/// columns at a time.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes,dotprod"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon,dotprod"))]
pub(super) fn v(out: Out<'_>, dst_stride: usize, b: &Block<'_>) {
    let six = b.fv[0] == 0 && b.fv[7] == 0;
    macro_rules! taps {
        ($store:expr) => {
            match (six, b.w >= 16) {
                (true, true) => vstrips!(b, $store, 16, wide16, mul16, mla16, each16, 1; [v0, v1, v2, v3, v4]; v5;
                    [v5; v0 0, v1 1, v2 2, v3 3, v4 4, v5 5]
                    [v0; v1 0, v2 1, v3 2, v4 3, v5 4, v0 5]
                    [v1; v2 0, v3 1, v4 2, v5 3, v0 4, v1 5]
                    [v2; v3 0, v4 1, v5 2, v0 3, v1 4, v2 5]
                    [v3; v4 0, v5 1, v0 2, v1 3, v2 4, v3 5]
                    [v4; v5 0, v0 1, v1 2, v2 3, v3 4, v4 5]),
                (true, false) => vstrips!(b, $store, 8, wide8, mul8, mla8, each8, 1; [v0, v1, v2, v3, v4]; v5;
                    [v5; v0 0, v1 1, v2 2, v3 3, v4 4, v5 5]
                    [v0; v1 0, v2 1, v3 2, v4 3, v5 4, v0 5]
                    [v1; v2 0, v3 1, v4 2, v5 3, v0 4, v1 5]
                    [v2; v3 0, v4 1, v5 2, v0 3, v1 4, v2 5]
                    [v3; v4 0, v5 1, v0 2, v1 3, v2 4, v3 5]
                    [v4; v5 0, v0 1, v1 2, v2 3, v3 4, v4 5]),
                (false, true) => {
                    vstrips!(b, $store, 16, wide16, mul16, mla16, each16, 0; [v0, v1, v2, v3, v4, v5, v6]; v7;
                    [v7; v0 0, v1 1, v2 2, v3 3, v4 4, v5 5, v6 6, v7 7]
                    [v0; v1 0, v2 1, v3 2, v4 3, v5 4, v6 5, v7 6, v0 7]
                    [v1; v2 0, v3 1, v4 2, v5 3, v6 4, v7 5, v0 6, v1 7]
                    [v2; v3 0, v4 1, v5 2, v6 3, v7 4, v0 5, v1 6, v2 7]
                    [v3; v4 0, v5 1, v6 2, v7 3, v0 4, v1 5, v2 6, v3 7]
                    [v4; v5 0, v6 1, v7 2, v0 3, v1 4, v2 5, v3 6, v4 7]
                    [v5; v6 0, v7 1, v0 2, v1 3, v2 4, v3 5, v4 6, v5 7]
                    [v6; v7 0, v0 1, v1 2, v2 3, v3 4, v4 5, v5 6, v6 7])
                }
                (false, false) => {
                    vstrips!(b, $store, 8, wide8, mul8, mla8, each8, 0; [v0, v1, v2, v3, v4, v5, v6]; v7;
                    [v7; v0 0, v1 1, v2 2, v3 3, v4 4, v5 5, v6 6, v7 7]
                    [v0; v1 0, v2 1, v3 2, v4 3, v5 4, v6 5, v7 6, v0 7]
                    [v1; v2 0, v3 1, v4 2, v5 3, v6 4, v7 5, v0 6, v1 7]
                    [v2; v3 0, v4 1, v5 2, v6 3, v7 4, v0 5, v1 6, v2 7]
                    [v3; v4 0, v5 1, v6 2, v7 3, v0 4, v1 5, v2 6, v3 7]
                    [v4; v5 0, v6 1, v7 2, v0 3, v1 4, v2 5, v3 6, v4 7]
                    [v5; v6 0, v7 1, v0 2, v1 3, v2 4, v3 5, v4 6, v5 7]
                    [v6; v7 0, v0 1, v1 2, v2 3, v3 4, v4 5, v5 6, v6 7])
                }
            }
        };
    }
    match out {
        Out::Prep(dst) => {
            // Round2(sum, 3) of the whole taps.
            taps!(|c: usize, r: usize, acc: int16x8_t| {
                store_i16(dst, r * dst_stride + c, vrshrq_n_s16::<2>(acc));
            })
        }
        Out::Put(dst) => {
            // Round2(sum, 7) of the whole taps, clipped.
            taps!(|c: usize, r: usize, acc: int16x8_t| {
                store_u8::<6>(dst, r * dst_stride + c, acc);
            })
        }
    }
}
