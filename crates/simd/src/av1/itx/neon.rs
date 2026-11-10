//! The block transform on NEON: an entry of `T` is the eight 16-bit lanes
//! of a vector, one lane for each of eight rows or eight columns.

use core::arch::aarch64::{
    int16x8_t, vaddw_high_u8, vaddw_u8, vcombine_s16, vdupq_n_s16, vextq_s16, vget_lane_u64,
    vget_low_u8, vmovn_s32, vqmovun_high_s16, vqmovun_s16, vreinterpret_u64_u8,
    vreinterpretq_s16_s64, vreinterpretq_s16_u16, vreinterpretq_s32_s16, vreinterpretq_s64_s32,
    vreinterpretq_u16_s16, vrev64q_s16, vrshlq_s16, vrshrq_n_s16, vtrn1q_s16, vtrn1q_s32,
    vtrn1q_s64, vtrn2q_s16, vtrn2q_s32, vtrn2q_s64,
};

use super::steps::{Prog, Step, kind, plan, table};
use super::{Block, Kind};
use crate::neon::{bytes, i16x8, i16x8_array, i32x4, u8x8, u8x16};

type V = int16x8_t;

/// The arithmetic of a step on the eight values of an entry.
mod ops {
    use core::arch::aarch64::{
        int16x8_t, vdupq_n_s16, vget_low_s16, vmlal_high_n_s16, vmlal_n_s16, vmlsl_high_n_s16,
        vmlsl_n_s16, vmull_high_n_s16, vmull_n_s16, vqaddq_s16, vqnegq_s16, vqrdmulhq_n_s16,
        vqsubq_s16, vrshrn_high_n_s32, vrshrn_n_s32,
    };

    pub(super) type V = int16x8_t;

    #[inline]
    #[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
    #[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
    pub(super) fn zero() -> V {
        vdupq_n_s16(0)
    }

    /// `x cos - y sin` and `x sin + y cos`, rounded at 12 bits: in 32
    /// bits, and the narrowing shift rounds and keeps the low 16.
    #[inline]
    #[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
    #[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
    pub(super) fn rot(x: V, y: V, cos: i16, sin: i16) -> (V, V) {
        let (xl, yl) = (vget_low_s16(x), vget_low_s16(y));
        let lo = vmlsl_n_s16(vmull_n_s16(xl, cos), yl, sin);
        let hi = vmlsl_high_n_s16(vmull_high_n_s16(x, cos), y, sin);
        let p = vrshrn_high_n_s32::<12>(vrshrn_n_s32::<12>(lo), hi);
        let lo = vmlal_n_s16(vmull_n_s16(xl, sin), yl, cos);
        let hi = vmlal_high_n_s16(vmull_high_n_s16(x, sin), y, cos);
        (p, vrshrn_high_n_s32::<12>(vrshrn_n_s32::<12>(lo), hi))
    }

    /// `x c`, rounded at 12 bits, for `c` within 4095 of zero: the high
    /// half of the doubled product by `8 c`, which rounds the same way.
    #[inline]
    #[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
    #[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
    pub(super) fn scale(x: V, c: i16) -> V {
        vqrdmulhq_n_s16(x, c.wrapping_mul(8))
    }

    #[inline]
    #[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
    #[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
    pub(super) fn add(x: V, y: V) -> V {
        vqaddq_s16(x, y)
    }

    #[inline]
    #[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
    #[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
    pub(super) fn sub(x: V, y: V) -> V {
        vqsubq_s16(x, y)
    }

    #[inline]
    #[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
    #[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
    pub(super) fn neg(x: V) -> V {
        vqnegq_s16(x)
    }
}

transforms!(
    #[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
    #[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
);

/// Eight vectors of eight with rows and columns exchanged.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn transpose(r: [V; 8]) -> [V; 8] {
    let pairs = |a: V, b: V| {
        (
            vreinterpretq_s32_s16(vtrn1q_s16(a, b)),
            vreinterpretq_s32_s16(vtrn2q_s16(a, b)),
        )
    };
    let (a0, a1) = pairs(r[0], r[1]);
    let (a2, a3) = pairs(r[2], r[3]);
    let (a4, a5) = pairs(r[4], r[5]);
    let (a6, a7) = pairs(r[6], r[7]);
    let quads = |a, b| {
        (
            vreinterpretq_s64_s32(vtrn1q_s32(a, b)),
            vreinterpretq_s64_s32(vtrn2q_s32(a, b)),
        )
    };
    let (b0, b2) = quads(a0, a2);
    let (b1, b3) = quads(a1, a3);
    let (b4, b6) = quads(a4, a6);
    let (b5, b7) = quads(a5, a7);
    let low = |a, b| vreinterpretq_s16_s64(vtrn1q_s64(a, b));
    let high = |a, b| vreinterpretq_s16_s64(vtrn2q_s64(a, b));
    [
        low(b0, b4),
        low(b1, b5),
        low(b2, b6),
        low(b3, b7),
        high(b0, b4),
        high(b1, b5),
        high(b2, b6),
        high(b3, b7),
    ]
}

/// Eight coefficients from `at` on, their low 16 bits; zero past the end.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn coefs8(coefs: &[i32], at: usize) -> V {
    let Some(c) = coefs.get(at..).and_then(|c| c.first_chunk::<8>()) else {
        return vdupq_n_s16(0);
    };
    let lo = i32x4([c[0], c[1], c[2], c[3]]);
    let hi = i32x4([c[4], c[5], c[6], c[7]]);
    vcombine_s16(vmovn_s32(lo), vmovn_s32(hi))
}

/// The smallest count of inputs a transform is made for that covers
/// `nz` of them: the entries of `T` from `nz` up to it are read too.
fn made_for(nz: usize) -> usize {
    match nz {
        0 | 1 => 1,
        2..=4 => 4,
        _ => nz.next_power_of_two(),
    }
}

#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(super) fn add(b: &mut Block<'_>) {
    // By width and by rows of coefficients, so the values between the
    // two passes take no more room than the block has.
    match (b.w, b.h.min(32)) {
        (8, 8) => sized::<8, 8>(b),
        (8, 16) => sized::<8, 16>(b),
        (8, _) => sized::<8, 32>(b),
        (16, 8) => sized::<16, 8>(b),
        (16, 16) => sized::<16, 16>(b),
        (16, _) => sized::<16, 32>(b),
        (32, 8) => sized::<32, 8>(b),
        (32, 16) => sized::<32, 16>(b),
        (32, _) => sized::<32, 32>(b),
        (_, 8) => sized::<64, 8>(b),
        (_, 16) => sized::<64, 16>(b),
        _ => sized::<64, 32>(b),
    }
}

/// [`add`] for a block `W` wide with `TH` rows of coefficients.
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn sized<const W: usize, const TH: usize>(b: &mut Block<'_>) {
    // Everything of the block in values of its own: a store to a sample
    // then says nothing about them.
    let (h, stride, coefs) = (b.h, b.stride, b.coefs);
    let (row_kind, col_kind) = (b.row, b.col);
    let (flip_ud, flip_lr) = (b.flip_ud, b.flip_lr);
    let dst = &mut *b.dst;
    let tw = W.min(32);
    let nr = b.nz_rows.clamp(1, TH);
    let nc = b.nz_cols.clamp(1, tw);
    let rect = W.max(h) == 2 * W.min(h);
    let zero = vdupq_n_s16(0);
    let shift = vdupq_n_s16(-(b.shift as i16));
    // The entries of `T` a transform reads are loaded or cleared here;
    // the rest it never reads.
    let mut t = [zero; 64];
    // The values after the row transforms, by row.
    let mut mid = [[0i16; W]; TH];
    let read = made_for(nc).min(tw);
    for r0 in (0..nr).step_by(8) {
        // Eight rows of eight coefficients become eight entries of `T`.
        for k0 in (0..read).step_by(8) {
            let mut m = [zero; 8];
            if k0 < nc {
                for (l, row) in m.iter_mut().enumerate() {
                    if r0 + l < nr {
                        *row = coefs8(coefs, (r0 + l) * tw + k0);
                    }
                }
                m = transpose(m);
                if rect {
                    for v in &mut m {
                        *v = ops::scale(*v, 2896);
                    }
                }
            }
            if let Some(d) = t.get_mut(k0..k0 + 8) {
                d.copy_from_slice(&m);
            }
        }
        run_1d(row_kind, W, nc, &mut t);
        // And back: eight outputs become eight values of each row.
        for j0 in (0..W).step_by(8) {
            let Some(m) = t.get(j0..).and_then(|t| t.first_chunk::<8>()) else {
                continue;
            };
            let m = transpose(m.map(|v| vrshlq_s16(v, shift)));
            for (row, v) in mid.iter_mut().skip(r0).zip(m) {
                if let Some(d) = row.as_chunks_mut::<8>().0.get_mut(j0 / 8) {
                    *d = i16x8_array(v);
                }
            }
        }
    }
    // Rows the row transforms did not reach are zero in `mid`.
    let read = made_for(nr).min(TH);
    if nr == 1 && col_kind == Kind::Dct {
        // One row of values: every column is flat (the DCT of its first
        // input alone), so every row of the block gets the same residual.
        let mut res = [zero; 8];
        for (v, m) in res.iter_mut().zip(mid[0].as_chunks::<8>().0) {
            *v = vrshrq_n_s16::<4>(ops::scale(i16x8(*m), 2896));
        }
        if flip_lr {
            res[..W / 8].reverse();
            for v in &mut res {
                *v = back(*v);
            }
        }
        for y in 0..h {
            let Some(row) = dst.get_mut(y * stride..).and_then(|d| d.get_mut(..W)) else {
                continue;
            };
            if W == 8 {
                if let Some(px) = row.first_chunk_mut::<8>() {
                    add8(px, res[0]);
                }
                continue;
            }
            for (px, r) in row
                .as_chunks_mut::<16>()
                .0
                .iter_mut()
                .zip(res.as_chunks::<2>().0)
            {
                add16(px, r[0], r[1]);
            }
        }
        return;
    }
    if W <= 16 {
        for j0 in (0..W).step_by(8) {
            for (v, row) in t.iter_mut().zip(&mid).take(read) {
                if let Some(m) = row.as_chunks::<8>().0.get(j0 / 8) {
                    *v = i16x8(*m);
                }
            }
            run_1d(col_kind, h, nr, &mut t);
            let x0 = if flip_lr { W - 8 - j0 } else { j0 };
            for (i, &v) in t.iter().enumerate().take(h) {
                let y = if flip_ud { h - 1 - i } else { i };
                let Some(px) = dst
                    .get_mut(y * stride + x0..)
                    .and_then(|d| d.first_chunk_mut::<8>())
                else {
                    continue;
                };
                let res = vrshrq_n_s16::<4>(v);
                add8(px, if flip_lr { back(res) } else { res });
            }
        }
        return;
    }
    // Two groups of eight columns at a time: sixteen samples of a row.
    let mut second = [zero; 64];
    for j0 in (0..W).step_by(16) {
        for (g, t) in [&mut t, &mut second].into_iter().enumerate() {
            for (v, row) in t.iter_mut().zip(&mid).take(read) {
                if let Some(m) = row.as_chunks::<8>().0.get(j0 / 8 + g) {
                    *v = i16x8(*m);
                }
            }
            run_1d(col_kind, h, nr, t);
        }
        let x0 = if flip_lr { W - 16 - j0 } else { j0 };
        for (i, (&lo, &hi)) in t.iter().zip(&second).enumerate().take(h) {
            let y = if flip_ud { h - 1 - i } else { i };
            let Some(px) = dst
                .get_mut(y * stride + x0..)
                .and_then(|d| d.first_chunk_mut::<16>())
            else {
                continue;
            };
            let (mut lo, mut hi) = (vrshrq_n_s16::<4>(lo), vrshrq_n_s16::<4>(hi));
            if flip_lr {
                // The columns from the right: the second group first.
                (lo, hi) = (back(hi), back(lo));
            }
            add16(px, lo, hi);
        }
    }
}

/// Adds the residuals `res` to eight samples: the sum is within 16 bits,
/// and the narrowing clamps it to a sample.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn add8(px: &mut [u8; 8], res: V) {
    let sum = vreinterpretq_s16_u16(vaddw_u8(vreinterpretq_u16_s16(res), u8x8(px)));
    *px = vget_lane_u64::<0>(vreinterpret_u64_u8(vqmovun_s16(sum))).to_le_bytes();
}

/// [`add8`] for sixteen samples: `lo` for the first eight.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn add16(px: &mut [u8; 16], lo: V, hi: V) {
    let old = u8x16(px);
    let lo = vaddw_u8(vreinterpretq_u16_s16(lo), vget_low_u8(old));
    let hi = vaddw_high_u8(vreinterpretq_u16_s16(hi), old);
    *px = bytes(vqmovun_high_s16(
        vqmovun_s16(vreinterpretq_s16_u16(lo)),
        vreinterpretq_s16_u16(hi),
    ));
}

/// The eight lanes back to front.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn back(v: V) -> V {
    let r = vrev64q_s16(v);
    vextq_s16::<4>(r, r)
}
