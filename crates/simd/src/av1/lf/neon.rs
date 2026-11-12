//! The deblocking sample filter on NEON: sixteen lines in the sixteen
//! 8-bit lanes of a vector, one vector per sample of the line.

use core::arch::aarch64::{
    uint8x16_t, uint16x8_t, vabdq_u8, vaddl_high_u8, vaddl_u8, vaddq_u16, vandq_s8, vandq_u8,
    vbicq_s8, vbicq_u8, vbslq_u8, vcgtq_u8, vcleq_u8, vcombine_u8, vdupq_n_s8, vdupq_n_u8,
    vdupq_n_u16, veorq_u8, vget_low_u8, vmaxq_u8, vmaxvq_u8, vqaddq_s8, vqaddq_u8, vqsubq_s8,
    vreinterpretq_s8_u8, vreinterpretq_u8_s8, vreinterpretq_u8_u16, vreinterpretq_u8_u32,
    vreinterpretq_u8_u64, vreinterpretq_u16_u8, vreinterpretq_u32_u8, vreinterpretq_u64_u8,
    vrshrn_n_u16, vrshrq_n_s8, vshrq_n_s8, vshrq_n_u8, vsubl_high_u8, vsubl_u8, vtrn1q_u8,
    vtrn1q_u16, vtrn1q_u32, vtrn1q_u64, vtrn2q_u8, vtrn2q_u16, vtrn2q_u32, vtrn2q_u64,
};

use super::{Limits, Size};
use crate::neon::{bytes, halves, u8x16, u8x16_pair};

type V = uint8x16_t;

/// Sixteen 16-bit sums, one for each line.
type Sums = (uint16x8_t, uint16x8_t);

/// `a + b`, each lane widened to 16 bits.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn sum2(a: V, b: V) -> Sums {
    (
        vaddl_u8(vget_low_u8(a), vget_low_u8(b)),
        vaddl_high_u8(a, b),
    )
}

/// `a - b`, each lane widened to 16 bits (a negative lane wraps, and
/// adds as its negative).
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn diff(a: V, b: V) -> Sums {
    (
        vsubl_u8(vget_low_u8(a), vget_low_u8(b)),
        vsubl_high_u8(a, b),
    )
}

#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn plus(a: Sums, b: Sums) -> Sums {
    (vaddq_u16(a.0, b.0), vaddq_u16(a.1, b.1))
}

/// The wide filter (7.14.6.4) for taps `-N .. N` of the lines `o`,
/// written to `out` in the lanes `on` selects.
///
/// Output `i` is the rounded mean of the samples `i - N ..= i + N`, the
/// ones within `N2` of `i` counted twice, and a sample past the filter's
/// reach replaced by the last one inside it. So each output's sum is the
/// one before it with one sample entering and one leaving each of the two
/// windows. A sum of at most 16 samples fits 16 bits.
///
/// What enters and leaves is worked out apart from the running sum, so
/// only one addition for each output waits for the output before it.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn wide<const N: i32, const N2: i32, const LOG2: i32>(o: &[V; 14], out: &mut [V; 14], on: V) {
    let at = |p: i32| o[(7 + p.clamp(-(N + 1), N)) as usize];
    // The first output's samples, two at a time: an even count, the
    // window of `2 N + 1` and the `2 N2 + 1` counted twice.
    let first = |k: i32| {
        let j = if k <= 2 * N {
            k - N
        } else {
            k - 2 * N - 1 - N2
        };
        at(j - N)
    };
    let zero = (vdupq_n_u16(0), vdupq_n_u16(0));
    let mut parts = [zero; 2];
    for pair in 0..N + N2 + 1 {
        let s = sum2(first(2 * pair), first(2 * pair + 1));
        parts[(pair & 1) as usize] = plus(parts[(pair & 1) as usize], s);
    }
    let mut sum = plus(parts[0], parts[1]);
    // Each output spelled out with a constant index, so the lines stay
    // in registers: a loop over them was left rolled, indexing copies of
    // the lines on the stack (two fifths of the size-16 filter's time).
    let mut output = |i: i32| {
        if i < -N || i >= N {
            return;
        }
        if i > -N {
            let moved = plus(
                diff(at(i + N), at(i - 1 - N)),
                diff(at(i + N2), at(i - 1 - N2)),
            );
            sum = plus(sum, moved);
        }
        let r = vcombine_u8(vrshrn_n_u16::<LOG2>(sum.0), vrshrn_n_u16::<LOG2>(sum.1));
        let k = (7 + i) as usize;
        out[k] = vbslq_u8(on, r, out[k]);
    };
    output(-6);
    output(-5);
    output(-4);
    output(-3);
    output(-2);
    output(-1);
    output(0);
    output(1);
    output(2);
    output(3);
    output(4);
    output(5);
}

/// The wide filters of size 16 for the flat lines, taking and giving back
/// the lines by value. Inlined into [`filter`]: called, it took its
/// twenty-eight vectors through memory (0.5% more cycles on 4K clips).
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn widest(o: [V; 14], mut out: [V; 14], flat: V, flat2: V) -> [V; 14] {
    // The filter over `p2 .. q2`, unless every flat line takes the one
    // over `p5 .. q5`, which writes over it.
    if vmaxvq_u8(vbicq_u8(flat, flat2)) != 0 {
        wide::<3, 0, 3>(&o, &mut out, flat);
    }
    if vmaxvq_u8(flat2) != 0 {
        wide::<6, 1, 4>(&o, &mut out, flat2);
    }
    out
}

#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn max3(a: V, b: V, c: V) -> V {
    vmaxq_u8(a, vmaxq_u8(b, c))
}

/// Filters the sixteen lines in `v` (index 7 is `q0`). `SIZE` is 4, 6, 8
/// or 16. Returns whether any line passed the filter mask.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn filter<const SIZE: usize>(v: &mut [V; 14], lim: Limits) -> bool {
    let o = *v;
    let (p1, p0, q0, q1) = (o[5], o[6], o[7], o[8]);
    let step = vmaxq_u8(vabdq_u8(p1, p0), vabdq_u8(q1, q0));
    let hev = vcgtq_u8(step, vdupq_n_u8(lim.thresh));
    let mut inner = step;
    if SIZE >= 6 {
        inner = max3(inner, vabdq_u8(o[4], p1), vabdq_u8(o[9], q1));
    }
    if SIZE >= 8 {
        inner = max3(inner, vabdq_u8(o[3], o[4]), vabdq_u8(o[10], o[9]));
    }
    // |p0 - q0| * 2 + |p1 - q1| / 2. Saturation keeps it above `blimit`,
    // which is at most 193.
    let across = vabdq_u8(p0, q0);
    let outer = vqaddq_u8(vqaddq_u8(across, across), vshrq_n_u8::<1>(vabdq_u8(p1, q1)));
    let pass = vandq_u8(
        vcleq_u8(inner, vdupq_n_u8(lim.limit)),
        vcleq_u8(outer, vdupq_n_u8(lim.blimit)),
    );
    if vmaxvq_u8(pass) == 0 {
        return false;
    }
    let one = vdupq_n_u8(1);
    let mut flat = vdupq_n_u8(0);
    if SIZE >= 6 {
        let mut f = max3(step, vabdq_u8(o[4], p0), vabdq_u8(o[9], q0));
        if SIZE >= 8 {
            f = max3(f, vabdq_u8(o[3], p0), vabdq_u8(o[10], q0));
        }
        flat = vandq_u8(vcleq_u8(f, one), pass);
    }
    let mut flat2 = vdupq_n_u8(0);
    if SIZE == 16 {
        let f = vmaxq_u8(
            max3(vabdq_u8(o[0], p0), vabdq_u8(o[1], p0), vabdq_u8(o[2], p0)),
            max3(
                vabdq_u8(o[13], q0),
                vabdq_u8(o[12], q0),
                vabdq_u8(o[11], q0),
            ),
        );
        flat2 = vandq_u8(vcleq_u8(f, one), flat);
    }
    // The narrow filter (7.14.6.3), unless every line that passed takes
    // a wide one.
    if SIZE == 4 || vmaxvq_u8(vbicq_u8(pass, flat)) != 0 {
        // On `sample - 128` as signed bytes. A saturating add of the
        // saturated difference three times equals the clamp of
        // `f + 3 * (qs0 - ps0)`: the sum moves one way only, so it can
        // only stick at the bound the exact value passes.
        let sign = vdupq_n_u8(0x80);
        let ps1 = vreinterpretq_s8_u8(veorq_u8(p1, sign));
        let ps0 = vreinterpretq_s8_u8(veorq_u8(p0, sign));
        let qs0 = vreinterpretq_s8_u8(veorq_u8(q0, sign));
        let qs1 = vreinterpretq_s8_u8(veorq_u8(q1, sign));
        let hev_s = vreinterpretq_s8_u8(hev);
        let d = vqsubq_s8(qs0, ps0);
        let f = vandq_s8(vqsubq_s8(ps1, qs1), hev_s);
        let f = vqaddq_s8(vqaddq_s8(vqaddq_s8(f, d), d), d);
        // A line that fails the mask moves by 0.
        let f = vandq_s8(f, vreinterpretq_s8_u8(pass));
        let f1 = vshrq_n_s8::<3>(vqaddq_s8(f, vdupq_n_s8(4)));
        let f2 = vshrq_n_s8::<3>(vqaddq_s8(f, vdupq_n_s8(3)));
        let f3 = vbicq_s8(vrshrq_n_s8::<1>(f1), hev_s);
        v[5] = veorq_u8(vreinterpretq_u8_s8(vqaddq_s8(ps1, f3)), sign);
        v[6] = veorq_u8(vreinterpretq_u8_s8(vqaddq_s8(ps0, f2)), sign);
        v[7] = veorq_u8(vreinterpretq_u8_s8(vqsubq_s8(qs0, f1)), sign);
        v[8] = veorq_u8(vreinterpretq_u8_s8(vqsubq_s8(qs1, f3)), sign);
    }
    if SIZE >= 6 && vmaxvq_u8(flat) != 0 {
        match SIZE {
            6 => wide::<2, 1, 3>(&o, v, flat),
            8 => wide::<3, 0, 3>(&o, v, flat),
            _ => *v = widest(o, *v, flat, flat2),
        }
    }
    true
}

#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn filter_size(v: &mut [V; 14], size: Size, lim: Limits) -> bool {
    match size {
        Size::Four => filter::<4>(v, lim),
        Size::Six => filter::<6>(v, lim),
        Size::Eight => filter::<8>(v, lim),
        Size::Sixteen => filter::<16>(v, lim),
    }
}

#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(super) fn by_sample(taps: &mut [[u8; 16]; 14], size: Size, lim: Limits) -> bool {
    let (lo, hi) = size.reads();
    let mut v = [vdupq_n_u8(0); 14];
    for k in lo..hi {
        v[k] = u8x16(&taps[k]);
    }
    if !filter_size(&mut v, size, lim) {
        return false;
    }
    let (lo, hi) = size.writes();
    for k in lo..hi {
        taps[k] = bytes(v[k]);
    }
    true
}

/// Turns eight vectors of rows into eight vectors of columns, in each
/// half of the vectors apart: lane `l` of row `r` becomes lane `r` of
/// column `l` (and lane `8 + r` of column `l` from lane `8 + l`).
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn transpose(r: [V; 8]) -> [V; 8] {
    let pair8 = |a: V, b: V| (vtrn1q_u8(a, b), vtrn2q_u8(a, b));
    let pair16 = |a: V, b: V| {
        let (a, b) = (vreinterpretq_u16_u8(a), vreinterpretq_u16_u8(b));
        (
            vreinterpretq_u8_u16(vtrn1q_u16(a, b)),
            vreinterpretq_u8_u16(vtrn2q_u16(a, b)),
        )
    };
    let pair32 = |a: V, b: V| {
        let (a, b) = (vreinterpretq_u32_u8(a), vreinterpretq_u32_u8(b));
        (
            vreinterpretq_u8_u32(vtrn1q_u32(a, b)),
            vreinterpretq_u8_u32(vtrn2q_u32(a, b)),
        )
    };
    let (b0, b1) = pair8(r[0], r[1]);
    let (b2, b3) = pair8(r[2], r[3]);
    let (b4, b5) = pair8(r[4], r[5]);
    let (b6, b7) = pair8(r[6], r[7]);
    let (c0, c2) = pair16(b0, b2);
    let (c1, c3) = pair16(b1, b3);
    let (c4, c6) = pair16(b4, b6);
    let (c5, c7) = pair16(b5, b7);
    let (d0, d4) = pair32(c0, c4);
    let (d1, d5) = pair32(c1, c5);
    let (d2, d6) = pair32(c2, c6);
    let (d3, d7) = pair32(c3, c7);
    [d0, d1, d2, d3, d4, d5, d6, d7]
}

#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(super) fn by_line(lines: &mut [[u8; 8]; 16], size: Size, lim: Limits) -> bool {
    // Lines `l` and `8 + l` share a vector, so the columns come out with
    // all sixteen lines in order.
    let rows: [V; 8] = core::array::from_fn(|l| u8x16_pair(&lines[l], &lines[8 + l]));
    let cols = transpose(rows);
    let mut v = [vdupq_n_u8(0); 14];
    v[3..11].copy_from_slice(&cols);
    if !filter_size(&mut v, size, lim) {
        return false;
    }
    let mut cols = [vdupq_n_u8(0); 8];
    cols.copy_from_slice(&v[3..11]);
    let rows = transpose(cols);
    for l in 0..8 {
        (lines[l], lines[8 + l]) = halves(rows[l]);
    }
    true
}

/// Swaps the halves of two sets of half-transposed vectors: sixteen
/// vectors of sixteen lanes, fully transposed.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn transpose16(r: &[V; 16]) -> [V; 16] {
    let top: [V; 8] = core::array::from_fn(|l| r[l]);
    let bottom: [V; 8] = core::array::from_fn(|l| r[8 + l]);
    let (d, e) = (transpose(top), transpose(bottom));
    core::array::from_fn(|c| {
        let (a, b) = (
            vreinterpretq_u64_u8(d[c % 8]),
            vreinterpretq_u64_u8(e[c % 8]),
        );
        if c < 8 {
            vreinterpretq_u8_u64(vtrn1q_u64(a, b))
        } else {
            vreinterpretq_u8_u64(vtrn2q_u64(a, b))
        }
    })
}

#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(super) fn by_line_16(lines: &mut [[u8; 16]; 16], lim: Limits) -> bool {
    let rows: [V; 16] = core::array::from_fn(|l| u8x16(&lines[l]));
    let mut cols = transpose16(&rows);
    let mut v = [vdupq_n_u8(0); 14];
    v.copy_from_slice(&cols[1..15]);
    if !filter::<16>(&mut v, lim) {
        return false;
    }
    cols[1..15].copy_from_slice(&v);
    let rows = transpose16(&cols);
    for l in 0..16 {
        lines[l] = bytes(rows[l]);
    }
    true
}

/// The sixteen samples of `buf` from `o`.
#[inline(always)]
fn at16(buf: &[u8], o: usize) -> &[u8; 16] {
    &buf[o..o + 16].as_chunks::<16>().0[0]
}

#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(super) fn rows_in_place(
    buf: &mut [u8],
    stride: usize,
    at: usize,
    size: Size,
    lim: Limits,
) -> bool {
    let (lo, hi) = size.reads();
    let top = at - 7 * stride;
    let mut v = [vdupq_n_u8(0); 14];
    for (k, vk) in v.iter_mut().enumerate().take(hi).skip(lo) {
        *vk = u8x16(at16(buf, top + k * stride));
    }
    if !filter_size(&mut v, size, lim) {
        return false;
    }
    let (lo, hi) = size.writes();
    for (k, &vk) in v.iter().enumerate().take(hi).skip(lo) {
        let o = top + k * stride;
        buf[o..o + 16].copy_from_slice(&bytes(vk));
    }
    true
}

#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(super) fn cols_in_place(
    buf: &mut [u8],
    stride: usize,
    at: usize,
    size: Size,
    lim: Limits,
) -> bool {
    let left = at - 4;
    let line = |l: usize| -> &[u8; 8] {
        let o = left + l * stride;
        &buf[o..o + 8].as_chunks::<8>().0[0]
    };
    let rows: [V; 8] = core::array::from_fn(|l| u8x16_pair(line(l), line(8 + l)));
    let cols = transpose(rows);
    let mut v = [vdupq_n_u8(0); 14];
    v[3..11].copy_from_slice(&cols);
    if !filter_size(&mut v, size, lim) {
        return false;
    }
    let mut cols = [vdupq_n_u8(0); 8];
    cols.copy_from_slice(&v[3..11]);
    let rows = transpose(cols);
    for (l, &r) in rows.iter().enumerate() {
        let (a, b) = halves(r);
        let o = left + l * stride;
        buf[o..o + 8].copy_from_slice(&a);
        let o = left + (8 + l) * stride;
        buf[o..o + 8].copy_from_slice(&b);
    }
    true
}

#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(super) fn cols_16_in_place(buf: &mut [u8], stride: usize, at: usize, lim: Limits) -> bool {
    let left = at - 8;
    let rows: [V; 16] = core::array::from_fn(|l| u8x16(at16(buf, left + l * stride)));
    let mut cols = transpose16(&rows);
    let mut v = [vdupq_n_u8(0); 14];
    v.copy_from_slice(&cols[1..15]);
    if !filter::<16>(&mut v, lim) {
        return false;
    }
    cols[1..15].copy_from_slice(&v);
    let rows = transpose16(&cols);
    for (l, &r) in rows.iter().enumerate() {
        let o = left + l * stride;
        buf[o..o + 16].copy_from_slice(&bytes(r));
    }
    true
}
