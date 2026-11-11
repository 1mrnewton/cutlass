//! NEON inter prediction of whole blocks, clause 8.4.2.2, the way FFmpeg's
//! `h264qpel_neon.S` and `h264cmc_neon.S` go about it: the six taps of a row
//! are one or two loads and four byte shifts (`vext`), a column filter keeps
//! the six rows it needs in registers and loads one new row each step, and
//! a quarter sample is its first source written to `dst`, then averaged with
//! its second.
//!
//! # Memory
//!
//! Unlike the rest of this crate, these kernels load and store through
//! pointers, because a block is read as rows of a plane and built from them
//! in registers. Each kernel checks once, before any load, that every byte
//! any of its loads or stores touches lies inside the slice it was given:
//! [`luma_block`] and [`chroma_block`] work out the first and last byte read
//! and written and return `false` when they do not fit. Every pointer below
//! is derived from those slices within the checked span. The plain twin in
//! [`crate::scalar`] is the reference, and the tests in [`crate::mc`] compare
//! the two on every quarter and eighth offset.
#![allow(unsafe_code)]

use core::arch::aarch64::{
    int16x8_t, uint8x8_t, uint8x16_t, vaddl_high_s16, vaddl_high_u8, vaddl_s16, vaddl_u8,
    vcombine_u8, vcombine_u16, vdup_n_u8, vextq_u8, vget_lane_u32, vget_low_s16, vget_low_u8,
    vld1_u8, vld1q_u8, vmlal_u8, vmlaq_n_s32, vmlaq_n_u16, vmlsq_n_s32, vmlsq_n_u16, vmull_u8,
    vqmovn_u16, vqrshrun_high_n_s16, vqrshrun_n_s16, vqrshrun_n_s32, vreinterpret_u32_u8,
    vreinterpretq_s16_u16, vrhadd_u8, vrhaddq_u8, vrshrn_n_u16, vst1_u8, vst1q_u8,
};

use crate::mc_block::{Block, Source, quarter_sources};

/// `(a + f) − 5(b + e) + 20(c + d)` on eight lanes, as wrapping 16 bits: the
/// true sum lies in −2550..=10710, so read as signed it is exact.
#[inline]
#[target_feature(enable = "neon")]
fn taps8(t: [uint8x8_t; 6]) -> int16x8_t {
    let outer = vaddl_u8(t[0], t[5]);
    let near = vaddl_u8(t[1], t[4]);
    let mid = vaddl_u8(t[2], t[3]);
    vreinterpretq_s16_u16(vmlaq_n_u16(vmlsq_n_u16(outer, near, 5), mid, 20))
}

/// [`taps8`] on the high eight lanes of six sixteen-lane vectors.
#[inline]
#[target_feature(enable = "neon")]
fn taps8_high(t: [uint8x16_t; 6]) -> int16x8_t {
    let outer = vaddl_high_u8(t[0], t[5]);
    let near = vaddl_high_u8(t[1], t[4]);
    let mid = vaddl_high_u8(t[2], t[3]);
    vreinterpretq_s16_u16(vmlaq_n_u16(vmlsq_n_u16(outer, near, 5), mid, 20))
}

/// The six horizontal taps of sixteen samples from the 32 bytes at `p`, the
/// first being two samples left of the block.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn row_taps16(p: *const u8) -> [uint8x16_t; 6] {
    // SAFETY: the caller's span covers 32 bytes from `p`.
    let (a, b) = unsafe { (vld1q_u8(p), vld1q_u8(p.add(16))) };
    [
        a,
        vextq_u8::<1>(a, b),
        vextq_u8::<2>(a, b),
        vextq_u8::<3>(a, b),
        vextq_u8::<4>(a, b),
        vextq_u8::<5>(a, b),
    ]
}

/// The six horizontal taps of eight samples from the 16 bytes at `p`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn row_taps8(p: *const u8) -> [uint8x8_t; 6] {
    // SAFETY: the caller's span covers 16 bytes from `p`.
    let a = unsafe { vld1q_u8(p) };
    [
        vget_low_u8(a),
        vget_low_u8(vextq_u8::<1>(a, a)),
        vget_low_u8(vextq_u8::<2>(a, a)),
        vget_low_u8(vextq_u8::<3>(a, a)),
        vget_low_u8(vextq_u8::<4>(a, a)),
        vget_low_u8(vextq_u8::<5>(a, a)),
    ]
}

/// The centre sample from six rows of unrounded horizontal sums, clause
/// 8.4.2.2.1 `j`: the same taps down the column in 32 bits, then
/// `(sum + 512) >> 10`, clipped.
#[inline]
#[target_feature(enable = "neon")]
fn centre8(s: [int16x8_t; 6]) -> uint8x8_t {
    let low = {
        let outer = vaddl_s16(vget_low_s16(s[0]), vget_low_s16(s[5]));
        let near = vaddl_s16(vget_low_s16(s[1]), vget_low_s16(s[4]));
        let mid = vaddl_s16(vget_low_s16(s[2]), vget_low_s16(s[3]));
        vmlaq_n_s32(vmlsq_n_s32(outer, near, 5), mid, 20)
    };
    let high = {
        let outer = vaddl_high_s16(s[0], s[5]);
        let near = vaddl_high_s16(s[1], s[4]);
        let mid = vaddl_high_s16(s[2], s[3]);
        vmlaq_n_s32(vmlsq_n_s32(outer, near, 5), mid, 20)
    };
    vqmovn_u16(vcombine_u16(
        vqrshrun_n_s32::<10>(low),
        vqrshrun_n_s32::<10>(high),
    ))
}

/// Writes `row` to `out`, or its rounded average with what `out` holds.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn put16<const AVG: bool>(out: *mut u8, row: uint8x16_t) {
    // SAFETY: the caller's span covers 16 bytes from `out`.
    unsafe {
        let row = if AVG {
            vrhaddq_u8(vld1q_u8(out), row)
        } else {
            row
        };
        vst1q_u8(out, row);
    }
}

/// [`put16`] for eight samples.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn put8<const AVG: bool>(out: *mut u8, row: uint8x8_t) {
    // SAFETY: the caller's span covers 8 bytes from `out`.
    unsafe {
        let row = if AVG {
            vrhadd_u8(vld1_u8(out), row)
        } else {
            row
        };
        vst1_u8(out, row);
    }
}

/// Where a block's rows start: the source sample at the block's top left,
/// rows `stride` apart, and the destination with rows `dst_stride` apart.
#[derive(Clone, Copy)]
struct Rows {
    src: *const u8,
    stride: usize,
    dst: *mut u8,
    dst_stride: usize,
    h: usize,
}

/// One source of a 16-wide block, written or averaged into `dst`.
///
/// # Safety
///
/// `rows` must come from [`luma_block`]'s check: two rows above and three
/// below the block, and two samples left of it and 30 to its right of its
/// start, are inside the source slice, and `h` rows of 16 inside `dst`.
#[target_feature(enable = "neon")]
unsafe fn source16<const AVG: bool>(rows: Rows, source: Source) {
    let Rows {
        src,
        stride,
        dst,
        dst_stride,
        h,
    } = rows;
    // SAFETY: every offset below stays inside the span `rows` promises.
    unsafe {
        match source {
            Source::Full(dx, dy) => {
                let mut p = src.add(dy * stride + dx);
                for row in 0..h {
                    put16::<AVG>(dst.add(row * dst_stride), vld1q_u8(p));
                    p = p.add(stride);
                }
            }
            Source::HalfH(dy) => {
                let mut p = src.add(dy * stride).sub(2);
                for row in 0..h {
                    let t = row_taps16(p);
                    let low = taps8([
                        vget_low_u8(t[0]),
                        vget_low_u8(t[1]),
                        vget_low_u8(t[2]),
                        vget_low_u8(t[3]),
                        vget_low_u8(t[4]),
                        vget_low_u8(t[5]),
                    ]);
                    let high = taps8_high(t);
                    let half = vqrshrun_high_n_s16::<5>(vqrshrun_n_s16::<5>(low), high);
                    put16::<AVG>(dst.add(row * dst_stride), half);
                    p = p.add(stride);
                }
            }
            Source::HalfV(dx) => {
                let mut p = src.add(dx).sub(2 * stride);
                let mut w = [vld1q_u8(p); 6];
                for slot in w.iter_mut().skip(1) {
                    p = p.add(stride);
                    *slot = vld1q_u8(p);
                }
                for row in 0..h {
                    let low = taps8([
                        vget_low_u8(w[0]),
                        vget_low_u8(w[1]),
                        vget_low_u8(w[2]),
                        vget_low_u8(w[3]),
                        vget_low_u8(w[4]),
                        vget_low_u8(w[5]),
                    ]);
                    let high = taps8_high(w);
                    let half = vqrshrun_high_n_s16::<5>(vqrshrun_n_s16::<5>(low), high);
                    put16::<AVG>(dst.add(row * dst_stride), half);
                    if row + 1 < h {
                        p = p.add(stride);
                        w = [w[1], w[2], w[3], w[4], w[5], vld1q_u8(p)];
                    }
                }
            }
            Source::Centre => {
                // Unrounded horizontal sums of the six rows around each
                // output row, low and high halves.
                let sums = |p: *const u8| {
                    let t = row_taps16(p);
                    let low = taps8([
                        vget_low_u8(t[0]),
                        vget_low_u8(t[1]),
                        vget_low_u8(t[2]),
                        vget_low_u8(t[3]),
                        vget_low_u8(t[4]),
                        vget_low_u8(t[5]),
                    ]);
                    (low, taps8_high(t))
                };
                let mut p = src.sub(2 * stride + 2);
                let first = sums(p);
                let mut low = [first.0; 6];
                let mut high = [first.1; 6];
                for index in 1..6 {
                    p = p.add(stride);
                    (low[index], high[index]) = sums(p);
                }
                for row in 0..h {
                    let out = vcombine_u8(centre8(low), centre8(high));
                    put16::<AVG>(dst.add(row * dst_stride), out);
                    if row + 1 < h {
                        p = p.add(stride);
                        let (l, hi) = sums(p);
                        low = [low[1], low[2], low[3], low[4], low[5], l];
                        high = [high[1], high[2], high[3], high[4], high[5], hi];
                    }
                }
            }
        }
    }
}

/// [`source16`] for a block 8 wide.
///
/// # Safety
///
/// As for [`source16`], with 14 samples right of the block's start.
#[target_feature(enable = "neon")]
unsafe fn source8<const AVG: bool>(rows: Rows, source: Source) {
    let Rows {
        src,
        stride,
        dst,
        dst_stride,
        h,
    } = rows;
    // SAFETY: every offset below stays inside the span `rows` promises.
    unsafe {
        match source {
            Source::Full(dx, dy) => {
                let mut p = src.add(dy * stride + dx);
                for row in 0..h {
                    put8::<AVG>(dst.add(row * dst_stride), vld1_u8(p));
                    p = p.add(stride);
                }
            }
            Source::HalfH(dy) => {
                let mut p = src.add(dy * stride).sub(2);
                for row in 0..h {
                    let half = vqrshrun_n_s16::<5>(taps8(row_taps8(p)));
                    put8::<AVG>(dst.add(row * dst_stride), half);
                    p = p.add(stride);
                }
            }
            Source::HalfV(dx) => {
                let mut p = src.add(dx).sub(2 * stride);
                let mut w = [vld1_u8(p); 6];
                for slot in w.iter_mut().skip(1) {
                    p = p.add(stride);
                    *slot = vld1_u8(p);
                }
                for row in 0..h {
                    let half = vqrshrun_n_s16::<5>(taps8(w));
                    put8::<AVG>(dst.add(row * dst_stride), half);
                    if row + 1 < h {
                        p = p.add(stride);
                        w = [w[1], w[2], w[3], w[4], w[5], vld1_u8(p)];
                    }
                }
            }
            Source::Centre => {
                let mut p = src.sub(2 * stride + 2);
                let mut s = [taps8(row_taps8(p)); 6];
                for slot in s.iter_mut().skip(1) {
                    p = p.add(stride);
                    *slot = taps8(row_taps8(p));
                }
                for row in 0..h {
                    put8::<AVG>(dst.add(row * dst_stride), centre8(s));
                    if row + 1 < h {
                        p = p.add(stride);
                        s = [s[1], s[2], s[3], s[4], s[5], taps8(row_taps8(p))];
                    }
                }
            }
        }
    }
}

/// See [`crate::mc::luma_block`]. Blocks 8 or 16 wide and at most 16 high;
/// `false` means nothing was done: another size, or a block whose samples
/// and the reach of the loads around them are not all in the slices.
#[target_feature(enable = "neon")]
pub(crate) fn luma_block(
    dst: &mut [u8],
    dst_stride: usize,
    block: Block<'_>,
    frac: (u8, u8),
) -> bool {
    let Block {
        src,
        stride,
        at: (x, y),
        size: (w, h),
    } = block;
    if (w != 8 && w != 16) || h == 0 || h > 16 || frac.0 > 3 || frac.1 > 3 {
        return false;
    }
    // The loads reach two rows above the block and three below, two
    // samples left of it, and 30 (16 wide) or 14 (8 wide) right of x.
    if x < 2 || y < 2 || stride < w + 5 {
        return false;
    }
    let reach = if w == 16 { 30 } else { 14 };
    let Some(last) = (y + h + 2)
        .checked_mul(stride)
        .and_then(|rows| rows.checked_add(x + reach))
    else {
        return false;
    };
    if last > src.len() || dst_stride < w || (h - 1) * dst_stride + w > dst.len() {
        return false;
    }
    let (first, second) = match quarter_sources(frac) {
        Some((first, second)) => (first, Some(second)),
        None => match frac {
            (2, 0) => (Source::HalfH(0), None),
            (0, 2) => (Source::HalfV(0), None),
            (2, 2) => (Source::Centre, None),
            _ => (Source::Full(0, 0), None),
        },
    };
    let rows = Rows {
        // In bounds: `y * stride + x` is below `last`.
        src: src[y * stride + x..].as_ptr(),
        stride,
        dst: dst.as_mut_ptr(),
        dst_stride,
        h,
    };
    // SAFETY: the checks above are the span each source needs.
    unsafe {
        if w == 16 {
            source16::<false>(rows, first);
            if let Some(second) = second {
                source16::<true>(rows, second);
            }
        } else {
            source8::<false>(rows, first);
            if let Some(second) = second {
                source8::<true>(rows, second);
            }
        }
    }
    true
}

/// See [`crate::mc::chroma_block`]. Blocks 4 or 8 wide; `false` as for
/// [`luma_block`].
#[target_feature(enable = "neon")]
pub(crate) fn chroma_block(
    dst: &mut [u8],
    dst_stride: usize,
    block: Block<'_>,
    frac: (u8, u8),
) -> bool {
    let Block {
        src,
        stride,
        at: (x, y),
        size: (w, h),
    } = block;
    let (xf, yf) = frac;
    if (w != 4 && w != 8) || h == 0 || xf > 7 || yf > 7 {
        return false;
    }
    // Each row reads the 16 bytes from its start (8 + 8 one later for the
    // second tap), and the row below it.
    let Some(last) = (y + h)
        .checked_mul(stride)
        .and_then(|rows| rows.checked_add(x + 9))
    else {
        return false;
    };
    if last > src.len() || dst_stride < w || (h - 1) * dst_stride + w > dst.len() {
        return false;
    }
    let wa = vdup_n_u8((8 - xf) * (8 - yf));
    let wb = vdup_n_u8(xf * (8 - yf));
    let wc = vdup_n_u8((8 - xf) * yf);
    let wd = vdup_n_u8(xf * yf);
    let src = src[y * stride + x..].as_ptr();
    let out = dst.as_mut_ptr();
    // SAFETY: rows `y..=y + h` from `x` to `x + 9` are inside `src`, and
    // `h` rows of `w` inside `dst`, by the checks above.
    unsafe {
        let mut top = (vld1_u8(src), vld1_u8(src.add(1)));
        for row in 0..h {
            let below = src.add((row + 1) * stride);
            let bottom = (vld1_u8(below), vld1_u8(below.add(1)));
            let sum = vmlal_u8(
                vmlal_u8(vmlal_u8(vmull_u8(top.0, wa), top.1, wb), bottom.0, wc),
                bottom.1,
                wd,
            );
            let made = vrshrn_n_u16::<6>(sum);
            let at = out.add(row * dst_stride);
            if w == 8 {
                vst1_u8(at, made);
            } else {
                let four = vget_lane_u32::<0>(vreinterpret_u32_u8(made));
                at.cast::<u32>().write_unaligned(four);
            }
            top = bottom;
        }
    }
    true
}
