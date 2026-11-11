//! NEON pieces of inter prediction: the six-tap filter on a row, the way
//! FFmpeg's `h264qpel_neon.S` goes about it (one or two loads and four
//! byte shifts, `vext`), which [`crate::neon_pel`] builds its routines from,
//! and the kernels that average or weigh whole predicted blocks, clause
//! 8.4.2.3.
//!
//! # Memory
//!
//! Unlike the rest of this crate, these kernels load and store through
//! pointers, because a block is read and written as rows of a plane. Each
//! block kernel checks once, before any load, that every byte any of its
//! loads or stores touches lies inside the slices it was given, and
//! returns `false` when one does not; the row helpers leave that check to
//! their callers, which say what it covers. The plain twins in
//! [`crate::scalar`] are the reference, and the tests in [`crate::mc`]
//! compare the two.
#![allow(unsafe_code)]

use core::arch::aarch64::{
    int16x4_t, int16x8_t, int32x4_t, uint8x8_t, uint8x16_t, uint16x8_t, vaddl_high_s16,
    vaddl_high_u8, vaddl_s16, vaddl_u8, vaddq_s32, vcombine_s16, vcombine_u8, vcombine_u16,
    vdup_n_s16, vdup_n_u32, vdupq_n_s16, vdupq_n_s32, vdupq_n_u8, vdupq_n_u16, vextq_u8,
    vget_high_u8, vget_lane_u32, vget_low_s16, vget_low_u8, vld1_u8, vld1q_u8, vmlal_high_s16,
    vmlal_high_u8, vmlal_s16, vmlal_u8, vmlaq_n_s32, vmlaq_n_u16, vmlsl_high_u8, vmlsl_u8,
    vmlsq_n_s32, vmlsq_n_u16, vmovl_u8, vmull_high_s16, vmull_s16, vmulq_s16, vqaddq_s16,
    vqmovn_u16, vqmovun_s16, vqmovun_s32, vqrshrun_n_s32, vreinterpret_u8_u32, vreinterpret_u32_u8,
    vreinterpretq_s16_u16, vrhadd_u8, vrhaddq_u8, vrshlq_s16, vrshlq_s32, vshlq_s16, vst1_u8,
    vst1q_u8,
};

/// `(a + f) − 5(b + e) + 20(c + d)` on eight lanes, as wrapping 16 bits: the
/// true sum lies in −2550..=10710, so read as signed it is exact.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn taps8(t: [uint8x8_t; 6]) -> int16x8_t {
    let outer = vaddl_u8(t[0], t[5]);
    let near = vaddl_u8(t[1], t[4]);
    let mid = vaddl_u8(t[2], t[3]);
    vreinterpretq_s16_u16(vmlaq_n_u16(vmlsq_n_u16(outer, near, 5), mid, 20))
}

/// [`taps8`] on the high eight lanes of six sixteen-lane vectors.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn taps8_high(t: [uint8x16_t; 6]) -> int16x8_t {
    let outer = vaddl_high_u8(t[0], t[5]);
    let near = vaddl_high_u8(t[1], t[4]);
    let mid = vaddl_high_u8(t[2], t[3]);
    vreinterpretq_s16_u16(vmlaq_n_u16(vmlsq_n_u16(outer, near, 5), mid, 20))
}

/// The six horizontal taps of sixteen samples from the 32 bytes at `p`, the
/// first being two samples left of the block.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn row_taps16(p: *const u8) -> [uint8x16_t; 6] {
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
pub(crate) unsafe fn row_taps8(p: *const u8) -> [uint8x8_t; 6] {
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
pub(crate) fn centre8(s: [int16x8_t; 6]) -> uint8x8_t {
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
pub(crate) unsafe fn put16<const AVG: bool>(out: *mut u8, row: uint8x16_t) {
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
pub(crate) unsafe fn put8<const AVG: bool>(out: *mut u8, row: uint8x8_t) {
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

/// Two predictions of eight samples mixed by `weight` (see
/// [`crate::mc::weight_block`]) in 32 bits, then clipped to 8.
#[inline]
#[target_feature(enable = "neon")]
fn weigh8(
    a: uint8x8_t,
    b: uint8x8_t,
    w: (int16x4_t, int16x4_t, int32x4_t, int32x4_t),
) -> uint8x8_t {
    let (w0, w1, shift, offset) = w;
    let a = vreinterpretq_s16_u16(vmovl_u8(a));
    let b = vreinterpretq_s16_u16(vmovl_u8(b));
    let low = vmlal_s16(vmull_s16(vget_low_s16(a), w0), vget_low_s16(b), w1);
    let high = vmlal_high_s16(
        vmull_high_s16(a, vcombine_s16(w0, w0)),
        b,
        vcombine_s16(w1, w1),
    );
    // A rounding shift right is the add of half and the shift.
    let low = vaddq_s32(vrshlq_s32(low, shift), offset);
    let high = vaddq_s32(vrshlq_s32(high, shift), offset);
    vqmovn_u16(vcombine_u16(vqmovun_s32(low), vqmovun_s32(high)))
}

/// Checks a mix of two blocks of `size` packed `size.0` to a row into `dst`
/// with rows `dst_stride` apart: widths a multiple of 4 up to 16.
fn mix_fits(dst: &[u8], dst_stride: usize, a: &[u8], b: &[u8], size: (usize, usize)) -> bool {
    let (w, h) = size;
    w > 0
        && w <= 16
        && w % 4 == 0
        && h > 0
        && a.len() >= w * h
        && b.len() >= w * h
        && dst_stride >= w
        && (h - 1) * dst_stride + w <= dst.len()
}

/// Four samples from `p` in the low lanes.
///
/// # Safety
///
/// Four bytes from `p` must be readable.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn load4(p: *const u8) -> uint8x8_t {
    // SAFETY: the caller's promise.
    let four = unsafe { p.cast::<u32>().read_unaligned() };
    vreinterpret_u8_u32(vdup_n_u32(four))
}

/// The low four lanes of `v` to `p`.
///
/// # Safety
///
/// Four bytes from `p` must be writable.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn store4(p: *mut u8, v: uint8x8_t) {
    // SAFETY: the caller's promise.
    unsafe {
        p.cast::<u32>()
            .write_unaligned(vget_lane_u32::<0>(vreinterpret_u32_u8(v)))
    };
}

/// One row of two blocks mixed into `out`, 16 then 8 then 4 samples at a
/// time, by `wide` on sixteen and `narrow` on eight.
///
/// # Safety
///
/// `w` bytes from `a`, `b` and `out` must be in their slices.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn mix_row(
    (a, b, out): (*const u8, *const u8, *mut u8),
    w: usize,
    wide: impl Fn(uint8x16_t, uint8x16_t) -> uint8x16_t,
    narrow: impl Fn(uint8x8_t, uint8x8_t) -> uint8x8_t,
) {
    let mut done = 0;
    // SAFETY: each step reads and writes inside the `w` bytes.
    unsafe {
        if w == 16 {
            vst1q_u8(out, wide(vld1q_u8(a), vld1q_u8(b)));
            return;
        }
        if w >= 8 {
            vst1_u8(out, narrow(vld1_u8(a), vld1_u8(b)));
            done = 8;
        }
        if done < w {
            let (a, b, out) = (a.add(done), b.add(done), out.add(done));
            store4(out, narrow(load4(a), load4(b)));
        }
    }
}

/// See [`crate::mc::avg_block`].
#[target_feature(enable = "neon")]
pub(crate) fn avg_block(
    dst: &mut [u8],
    dst_stride: usize,
    a: &[u8],
    b: &[u8],
    size: (usize, usize),
) -> bool {
    if !mix_fits(dst, dst_stride, a, b, size) {
        return false;
    }
    let (w, h) = size;
    let (a, b, out) = (a.as_ptr(), b.as_ptr(), dst.as_mut_ptr());
    for row in 0..h {
        // SAFETY: `mix_fits` checked `w * h` bytes of `a` and `b` and `h`
        // rows of `w` in `dst`.
        unsafe {
            let at = (a.add(row * w), b.add(row * w), out.add(row * dst_stride));
            mix_row(at, w, |x, y| vrhaddq_u8(x, y), |x, y| vrhadd_u8(x, y));
        }
    }
    true
}

/// See [`crate::mc::weight_block`].
#[target_feature(enable = "neon")]
pub(crate) fn weight_block(
    dst: &mut [u8],
    dst_stride: usize,
    a: &[u8],
    b: &[u8],
    size: (usize, usize),
    weight: crate::mc_block::BiWeight,
) -> bool {
    if !weight.fits() || !mix_fits(dst, dst_stride, a, b, size) {
        return false;
    }
    if let Some(k) = Bi16::new(weight) {
        let (a, b, out) = (a.as_ptr(), b.as_ptr(), dst.as_mut_ptr());
        let w = size.0;
        // SAFETY: `mix_fits` checked `h` rows of `w` in `dst` and `w·h`
        // samples in `a` and `b`.
        unsafe {
            bi16_block(k, size, |row| {
                (a.add(row * w), b.add(row * w), out.add(row * dst_stride))
            });
        }
        return true;
    }
    let (w, h) = size;
    let (Ok(w0), Ok(w1)) = (i16::try_from(weight.w0), i16::try_from(weight.w1)) else {
        return false;
    };
    let mix = (
        vdup_n_s16(w0),
        vdup_n_s16(w1),
        vdupq_n_s32(-weight.shift),
        vdupq_n_s32(weight.offset),
    );
    let (a, b, out) = (a.as_ptr(), b.as_ptr(), dst.as_mut_ptr());
    for row in 0..h {
        // SAFETY: as in `avg_block`.
        unsafe {
            let at = (a.add(row * w), b.add(row * w), out.add(row * dst_stride));
            mix_row(
                at,
                w,
                |x, y| {
                    let low = weigh8(vget_low_u8(x), vget_low_u8(y), mix);
                    vcombine_u8(low, weigh8(vget_high_u8(x), vget_high_u8(y), mix))
                },
                |x, y| weigh8(x, y, mix),
            );
        }
    }
    true
}

/// The constants of FFmpeg's 16-bit two-prediction weighting, its
/// `biweight`: the rounding and the offset folded into one start value,
/// each weight's size, and whether it is negative, which picks a
/// multiply-add or a multiply-subtract. `None` when some sum of two
/// samples times the weights, with the start value, would leave 16 bits.
#[derive(Clone, Copy)]
struct Bi16 {
    start: i16,
    sizes: (u8, u8),
    negative: (bool, bool),
    shift: i16,
}

impl Bi16 {
    fn new(weight: crate::mc_block::BiWeight) -> Option<Self> {
        let (w0, w1) = (i64::from(weight.w0), i64::from(weight.w1));
        let shift = i64::from(weight.shift);
        let start = (1i64 << (shift - 1)) + (i64::from(weight.offset) << shift);
        let most = start + 255 * (w0.max(0) + w1.max(0));
        let least = start + 255 * (w0.min(0) + w1.min(0));
        if most > i64::from(i16::MAX) || least < i64::from(i16::MIN) {
            return None;
        }
        Some(Self {
            start: i16::try_from(start).ok()?,
            sizes: (
                u8::try_from(w0.unsigned_abs()).ok()?,
                u8::try_from(w1.unsigned_abs()).ok()?,
            ),
            negative: (w0 < 0, w1 < 0),
            shift: i16::try_from(-shift).ok()?,
        })
    }
}

/// `start ± a·w0 ± b·w1` on eight lanes, in wrapping 16 bits that the
/// checks in [`Bi16::new`] keep exact, shifted and clipped.
#[inline]
#[target_feature(enable = "neon")]
fn bi16_8<const N0: bool, const N1: bool>(
    a: uint8x8_t,
    b: uint8x8_t,
    (start, w0, w1, shift): (uint16x8_t, uint8x8_t, uint8x8_t, int16x8_t),
) -> uint8x8_t {
    let sum = if N0 {
        vmlsl_u8(start, a, w0)
    } else {
        vmlal_u8(start, a, w0)
    };
    let sum = if N1 {
        vmlsl_u8(sum, b, w1)
    } else {
        vmlal_u8(sum, b, w1)
    };
    vqmovun_s16(vshlq_s16(vreinterpretq_s16_u16(sum), shift))
}

/// [`bi16_8`] on sixteen lanes.
#[inline]
#[target_feature(enable = "neon")]
fn bi16_16<const N0: bool, const N1: bool>(
    a: uint8x16_t,
    b: uint8x16_t,
    (start, w0, w1, shift): (uint16x8_t, uint8x16_t, uint8x16_t, int16x8_t),
) -> uint8x16_t {
    let low = bi16_8::<N0, N1>(
        vget_low_u8(a),
        vget_low_u8(b),
        (start, vget_low_u8(w0), vget_low_u8(w1), shift),
    );
    let high = if N0 {
        vmlsl_high_u8(start, a, w0)
    } else {
        vmlal_high_u8(start, a, w0)
    };
    let high = if N1 {
        vmlsl_high_u8(high, b, w1)
    } else {
        vmlal_high_u8(high, b, w1)
    };
    vcombine_u8(
        low,
        vqmovun_s16(vshlq_s16(vreinterpretq_s16_u16(high), shift)),
    )
}

/// Rows `0..h` of `w` samples mixed by `k`, row `r`'s two predictions and
/// output at `at(r)`.
///
/// # Safety
///
/// `w` bytes from each of `at(r)`'s pointers must be in their slices, for
/// every row.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn bi16_rows<const N0: bool, const N1: bool>(
    k: Bi16,
    (w, h): (usize, usize),
    at: impl Fn(usize) -> (*const u8, *const u8, *mut u8),
) {
    let start = vdupq_n_u16(k.start as u16);
    let shift = vdupq_n_s16(k.shift);
    let (w0, w1) = (vdupq_n_u8(k.sizes.0), vdupq_n_u8(k.sizes.1));
    let wide = (start, w0, w1, shift);
    let narrow = (start, vget_low_u8(w0), vget_low_u8(w1), shift);
    for row in 0..h {
        // SAFETY: the caller's promise.
        unsafe {
            mix_row(
                at(row),
                w,
                |x, y| bi16_16::<N0, N1>(x, y, wide),
                |x, y| bi16_8::<N0, N1>(x, y, narrow),
            );
        }
    }
}

/// [`bi16_rows`] with the weights' signs picked at run time.
///
/// # Safety
///
/// As for [`bi16_rows`].
#[target_feature(enable = "neon")]
unsafe fn bi16_block(
    k: Bi16,
    size: (usize, usize),
    at: impl Fn(usize) -> (*const u8, *const u8, *mut u8),
) {
    // SAFETY: the caller's promise.
    unsafe {
        match k.negative {
            (false, false) => bi16_rows::<false, false>(k, size, at),
            (true, false) => bi16_rows::<true, false>(k, size, at),
            (false, true) => bi16_rows::<false, true>(k, size, at),
            (true, true) => bi16_rows::<true, true>(k, size, at),
        }
    }
}

/// See [`crate::mc::weight_onto`]: [`weight_block`] with the first
/// prediction read from `dst`.
#[target_feature(enable = "neon")]
pub(crate) fn weight_onto(
    dst: &mut [u8],
    dst_stride: usize,
    b: &[u8],
    size: (usize, usize),
    weight: crate::mc_block::BiWeight,
) -> bool {
    if !crate::mc_block::onto_fits(dst, dst_stride, b, size, weight) {
        return false;
    }
    if let Some(k) = Bi16::new(weight) {
        let (b, out) = (b.as_ptr(), dst.as_mut_ptr());
        let w = size.0;
        // SAFETY: `onto_fits` checked `h` rows of `w` in `dst` and `w·h`
        // samples in `b`.
        unsafe {
            bi16_block(k, size, |row| {
                let at = out.add(row * dst_stride);
                (at.cast_const(), b.add(row * w), at)
            });
        }
        return true;
    }
    let (Ok(w0), Ok(w1)) = (i16::try_from(weight.w0), i16::try_from(weight.w1)) else {
        return false;
    };
    let mix = (
        vdup_n_s16(w0),
        vdup_n_s16(w1),
        vdupq_n_s32(-weight.shift),
        vdupq_n_s32(weight.offset),
    );
    let (w, h) = size;
    let (b, out) = (b.as_ptr(), dst.as_mut_ptr());
    for row in 0..h {
        // SAFETY: `onto_fits` checked `h` rows of `w` in `dst` and `w·h`
        // samples in `b`.
        unsafe {
            let at = out.add(row * dst_stride);
            mix_row(
                (at, b.add(row * w), at),
                w,
                |x, y| {
                    let low = weigh8(vget_low_u8(x), vget_low_u8(y), mix);
                    vcombine_u8(low, weigh8(vget_high_u8(x), vget_high_u8(y), mix))
                },
                |x, y| weigh8(x, y, mix),
            );
        }
    }
    true
}

/// See [`crate::mc::scale_block`]. A sample times a weight of at most 128
/// fits 16 bits, and the rounding shift does not overflow on the way.
#[target_feature(enable = "neon")]
pub(crate) fn scale_block(
    dst: &mut [u8],
    dst_stride: usize,
    size: (usize, usize),
    weight: crate::mc_block::UniWeight,
) -> bool {
    if !weight.fits() || !crate::mc_block::scale_fits(dst, dst_stride, size) {
        return false;
    }
    let (Ok(mul), Ok(shift), Ok(offset)) = (
        i16::try_from(weight.weight),
        i16::try_from(-weight.shift),
        i16::try_from(weight.offset),
    ) else {
        return false;
    };
    let (mul, shift, offset) = (vdupq_n_s16(mul), vdupq_n_s16(shift), vdupq_n_s16(offset));
    let scale = |samples: uint8x8_t| {
        let product = vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(samples)), mul);
        vqmovun_s16(vqaddq_s16(vrshlq_s16(product, shift), offset))
    };
    let (w, h) = size;
    let out = dst.as_mut_ptr();
    for row in 0..h {
        // SAFETY: `scale_fits` checked `h` rows of `w` in `dst`.
        unsafe {
            let at = out.add(row * dst_stride);
            mix_row(
                (at, at, at),
                w,
                |x, _| vcombine_u8(scale(vget_low_u8(x)), scale(vget_high_u8(x))),
                |x, _| scale(x),
            );
        }
    }
    true
}
