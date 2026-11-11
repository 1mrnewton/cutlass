//! The block transform on NEON: an entry of `T` is the eight 16-bit lanes
//! of a vector, one lane for each of eight rows or eight columns.

use core::arch::aarch64::{
    int16x8_t, vaddw_high_u8, vaddw_u8, vcombine_s16, vcombine_u8, vdup_n_s16, vdupq_n_s16,
    vdupq_n_s32, vdupq_n_u8, vextq_s16, vget_lane_u64, vget_low_u8, vmovn_s32, vqaddq_u8,
    vqmovun_high_s16, vqmovun_s16, vqsubq_u8, vreinterpret_u64_u8, vreinterpretq_s16_s64,
    vreinterpretq_s16_u16, vreinterpretq_s32_s16, vreinterpretq_s64_s32, vreinterpretq_u16_s16,
    vrev64q_s16, vrshlq_s16, vrshrq_n_s16, vtrn1q_s16, vtrn1q_s32, vtrn1q_s64, vtrn2q_s16,
    vtrn2q_s32, vtrn2q_s64,
};

use super::steps::{Prog, Step, kind, plan, table};
use super::{Block, Kind};
use crate::neon::{bytes, halves, i16x8, i16x8_array, i32x4, u8x8, u8x16};

type V = int16x8_t;

/// The arithmetic of a step on the eight values of an entry.
mod ops {
    use core::arch::aarch64::{
        int16x8_t, int32x4_t, vaddq_s16, vaddq_s32, vceqq_s32, vcombine_s16, vdupq_n_s16,
        vdupq_n_s32, vget_high_s16, vget_low_s16, vmlal_high_n_s16, vmlal_n_s16, vmlsl_high_n_s16,
        vmlsl_n_s16, vmovl_high_s16, vmovl_s16, vmovn_s32, vmull_high_n_s16, vmull_n_s16,
        vmulq_n_s32, vqaddq_s16, vqmovn_high_s32, vqmovn_s32, vqnegq_s16, vqrdmulhq_n_s16,
        vqsubq_s16, vreinterpretq_s32_u32, vrshlq_s32, vrshrn_high_n_s32, vrshrn_n_s32,
        vrshrq_n_s32, vshlq_n_s32, vsubq_s16, vsubq_s32,
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

    /// The ADST of four values (7.13.2.6). The sums are exact in 32
    /// bits; an output of 32768 becomes 32767, which the rounding after
    /// it cannot tell apart.
    #[inline]
    #[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
    #[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
    pub(super) fn adst4(t: [V; 4]) -> [V; 4] {
        let [t0, t1, t2, t3] = t;
        // `t0 - t2 + t3` in 16 bits, as the specification keeps it.
        let b7 = vaddq_s16(vsubq_s16(t0, t2), t3);
        let top = vdupq_n_s32(32768);
        // The four outputs of the low or the high four lanes.
        let half = |t0, t1, t2, t3, b7| {
            let s0 = vmlal_n_s16(vmlal_n_s16(vmull_n_s16(t0, 1321), t2, 3803), t3, 2482);
            let s1 = vmlsl_n_s16(vmlsl_n_s16(vmull_n_s16(t0, 2482), t2, 1321), t3, 3803);
            let s3 = vmull_n_s16(t1, 3344);
            let s2 = vmull_n_s16(b7, 3344);
            [
                vaddq_s32(s0, s3),
                vaddq_s32(s1, s3),
                s2,
                vsubq_s32(vaddq_s32(s0, s1), s3),
            ]
            .map(|x| {
                let r = vrshrq_n_s32::<12>(x);
                // All ones, which is minus one, where it is 32768.
                vmovn_s32(vaddq_s32(r, vreinterpretq_s32_u32(vceqq_s32(r, top))))
            })
        };
        let lo = half(
            vget_low_s16(t0),
            vget_low_s16(t1),
            vget_low_s16(t2),
            vget_low_s16(t3),
            vget_low_s16(b7),
        );
        let hi = half(
            vget_high_s16(t0),
            vget_high_s16(t1),
            vget_high_s16(t2),
            vget_high_s16(t3),
            vget_high_s16(b7),
        );
        [
            vcombine_s16(lo[0], hi[0]),
            vcombine_s16(lo[1], hi[1]),
            vcombine_s16(lo[2], hi[2]),
            vcombine_s16(lo[3], hi[3]),
        ]
    }

    /// The identity transform of `n` values on four of them (7.13.2.15):
    /// a scaling by `sqrt(2)`, 2, `2 sqrt(2)` or 4, in 32 bits.
    #[inline]
    #[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
    #[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
    fn identity(x: int32x4_t, n: usize) -> int32x4_t {
        match n {
            4 => vaddq_s32(x, vrshrq_n_s32::<12>(vmulq_n_s32(x, 1697))),
            8 => vshlq_n_s32::<1>(x),
            16 => vaddq_s32(
                vshlq_n_s32::<1>(x),
                vrshrq_n_s32::<12>(vmulq_n_s32(x, 3394)),
            ),
            _ => vshlq_n_s32::<2>(x),
        }
    }

    /// The outputs of a row identity transform of `n` values, after the
    /// rounding shift (`shift` holds its negative), clamped to 16 bits.
    #[inline]
    #[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
    #[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
    pub(super) fn identity_row(x: V, n: usize, shift: int32x4_t) -> V {
        let lo = vrshlq_s32(identity(vmovl_s16(vget_low_s16(x)), n), shift);
        let hi = vrshlq_s32(identity(vmovl_high_s16(x), n), shift);
        vqmovn_high_s32(vqmovn_s32(lo), hi)
    }

    /// The residuals of a column identity transform of `n` values:
    /// `Round2` by 4 of the outputs.
    #[inline]
    #[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
    #[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
    pub(super) fn identity_col(x: V, n: usize) -> V {
        let lo = identity(vmovl_s16(vget_low_s16(x)), n);
        let hi = identity(vmovl_high_s16(x), n);
        vrshrn_high_n_s32::<4>(vrshrn_n_s32::<4>(lo), hi)
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

/// Eight coefficients from `at` on (`four`: four of them), their low 16
/// bits; zero past the end.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn coefs8(coefs: &[i32], at: usize, four: bool) -> V {
    if four {
        // A row of a block four wide: the other four lanes are zero.
        let Some(c) = coefs.get(at..).and_then(|c| c.first_chunk::<4>()) else {
            return vdupq_n_s16(0);
        };
        return vcombine_s16(vmovn_s32(i32x4(*c)), vdup_n_s16(0));
    }
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
    // By width and by the rows of values between the two passes that
    // the column transforms read: those the transform made for the
    // rows of coefficients takes. They take no more room than that, and
    // only that much is cleared.
    let rows = b.h.min(32).min(made_for(b.nz_rows).max(4));
    match (b.w, rows) {
        (4, 4) => sized::<4, 8, 4>(b),
        (4, 8) => sized::<4, 8, 8>(b),
        (4, _) => sized::<4, 8, 16>(b),
        (8, 4) => sized::<8, 8, 4>(b),
        (8, 8) => sized::<8, 8, 8>(b),
        (8, 16) => sized::<8, 8, 16>(b),
        (8, _) => sized::<8, 8, 32>(b),
        (16, 4) => sized::<16, 16, 4>(b),
        (16, 8) => sized::<16, 16, 8>(b),
        (16, 16) => sized::<16, 16, 16>(b),
        (16, _) => sized::<16, 16, 32>(b),
        (32, 4) => sized::<32, 32, 4>(b),
        (32, 8) => sized::<32, 32, 8>(b),
        (32, 16) => sized::<32, 32, 16>(b),
        (32, _) => sized::<32, 32, 32>(b),
        (_, 4) => sized::<64, 64, 4>(b),
        (_, 8) => sized::<64, 64, 8>(b),
        (_, 16) => sized::<64, 64, 16>(b),
        _ => sized::<64, 64, 32>(b),
    }
}

/// [`add`] for a block `W` wide whose column transforms read at most
/// `TH` rows of values from the row transforms. `MW` is
/// the width in whole vectors: 8 for a block 4 wide, whose vectors have
/// four lanes that nothing reads.
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn sized<const W: usize, const MW: usize, const TH: usize>(b: &mut Block<'_>) {
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
    let shift32 = vdupq_n_s32(-(b.shift as i32));
    let (row_identity, col_identity) = (row_kind == Kind::Identity, col_kind == Kind::Identity);
    if row_identity && col_identity && !flip_ud && !flip_lr {
        // Both transforms scale each value on its own, so only the
        // samples under the coefficients change: no rows to gather into
        // vectors and none to spread back. (No transform type flips
        // these; a caller that does takes the long way.)
        for r in 0..nr {
            for k0 in (0..nc).step_by(8) {
                let mut v = coefs8(coefs, r * tw + k0, W == 4);
                if rect {
                    v = ops::scale(v, 2896);
                }
                let res = ops::identity_col(ops::identity_row(v, W, shift32), h);
                add_some::<W>(dst, r * stride + k0, res);
            }
        }
        return;
    }
    // The entries of `T` a transform reads are loaded or cleared here;
    // the rest it never reads.
    let mut t = [zero; 64];
    // The values after the row transforms, by row.
    let mut mid = [[0i16; MW]; TH];
    let read = made_for(nc).min(tw);
    for r0 in (0..nr).step_by(8) {
        // Eight rows of eight coefficients become eight entries of `T`.
        for k0 in (0..read).step_by(8) {
            let mut m = [zero; 8];
            if k0 < nc {
                for (l, row) in m.iter_mut().enumerate() {
                    if r0 + l < nr {
                        *row = coefs8(coefs, (r0 + l) * tw + k0, W == 4);
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
        if row_identity {
            // The outputs are the inputs: those past the last one read
            // are zero.
            for v in t.iter_mut().take(W).skip(read.next_multiple_of(8)) {
                *v = zero;
            }
        }
        run_1d(row_kind, W, nc, &mut t);
        // And back: eight outputs become eight values of each row.
        for j0 in (0..W).step_by(8) {
            let Some(m) = t.get(j0..).and_then(|t| t.first_chunk::<8>()) else {
                continue;
            };
            let m = transpose(m.map(|v| {
                if row_identity {
                    ops::identity_row(v, W, shift32)
                } else {
                    vrshlq_s16(v, shift)
                }
            }));
            for (row, v) in mid.iter_mut().skip(r0).zip(m) {
                if let Some(d) = row.as_chunks_mut::<8>().0.get_mut(j0 / 8) {
                    *d = i16x8_array(v);
                }
            }
        }
    }
    // Rows the row transforms did not reach are zero in `mid`. The ADST
    // of four values is not made for fewer inputs: it reads all four.
    let read = if col_kind == Kind::Adst && h == 4 {
        4
    } else {
        made_for(nr).min(TH)
    };
    if nr == 1 && col_kind == Kind::Dct {
        // One row of values: every column is flat (the DCT of its first
        // input alone), so every row of the block gets the same residual.
        let mut res = [zero; 8];
        for (v, m) in res.iter_mut().zip(mid[0].as_chunks::<8>().0) {
            *v = vrshrq_n_s16::<4>(ops::scale(i16x8(*m), 2896));
        }
        if flip_lr {
            res[..MW / 8].reverse();
            for v in &mut res {
                *v = back::<W>(*v);
            }
        }
        for y in 0..h {
            let Some(row) = dst.get_mut(y * stride..).and_then(|d| d.get_mut(..W)) else {
                continue;
            };
            if W <= 8 {
                add_some::<W>(row, 0, res[0]);
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
            if col_identity {
                for v in t.iter_mut().take(h).skip(read) {
                    *v = zero;
                }
            }
            run_1d(col_kind, h, nr, &mut t);
            let x0 = if flip_lr { MW - 8 - j0 } else { j0 };
            for (i, &v) in t.iter().enumerate().take(h) {
                let y = if flip_ud { h - 1 - i } else { i };
                let res = residual(v, col_identity, h);
                let res = if flip_lr { back::<W>(res) } else { res };
                add_some::<W>(dst, y * stride + x0, res);
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
            if col_identity {
                for v in t.iter_mut().take(h).skip(read) {
                    *v = zero;
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
            let mut lo = residual(lo, col_identity, h);
            let mut hi = residual(hi, col_identity, h);
            if flip_lr {
                // The columns from the right: the second group first.
                (lo, hi) = (back::<W>(hi), back::<W>(lo));
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

/// The residuals of eight outputs of a column transform of `h` values:
/// `Round2` by 4, after the scaling of an identity transform.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn residual(v: V, identity: bool, h: usize) -> V {
    if identity {
        ops::identity_col(v, h)
    } else {
        vrshrq_n_s16::<4>(v)
    }
}

/// The lanes of a block `W` wide back to front: all eight, or the first
/// four of a block 4 wide.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn back<const W: usize>(v: V) -> V {
    let r = vrev64q_s16(v);
    if W == 4 { r } else { vextq_s16::<4>(r, r) }
}

/// [`add8`] on the samples from `at` on of a block `W` wide: eight, or
/// the four of a block 4 wide. Samples past the end are left alone.
#[inline]
#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
fn add_some<const W: usize>(dst: &mut [u8], at: usize, res: V) {
    let Some(d) = dst.get_mut(at..) else {
        return;
    };
    if W == 4 {
        if let Some(px) = d.first_chunk_mut::<4>() {
            let mut eight = [0u8; 8];
            eight[..4].copy_from_slice(px);
            add8(&mut eight, res);
            px.copy_from_slice(&eight[..4]);
        }
    } else if let Some(px) = d.first_chunk_mut::<8>() {
        add8(px, res);
    }
}

#[cfg_attr(target_feature = "aes", target_feature(enable = "neon,aes"))]
#[cfg_attr(not(target_feature = "aes"), target_feature(enable = "neon"))]
pub(super) fn add_flat(dst: &mut [u8], stride: usize, w: usize, h: usize, v: i32) {
    // Past 255 either way every sample ends at the same end. One of the
    // two is zero: a saturating add of the one and subtract of the other.
    let up = vdupq_n_u8(v.clamp(0, 255) as u8);
    let down = vdupq_n_u8(v.saturating_neg().clamp(0, 255) as u8);
    for y in 0..h {
        let Some(row) = dst.get_mut(y * stride..).and_then(|d| d.get_mut(..w)) else {
            continue;
        };
        let (sixteens, rest) = row.as_chunks_mut::<16>();
        for px in sixteens {
            *px = bytes(vqsubq_u8(vqaddq_u8(u8x16(px), up), down));
        }
        // A row of 8 or 4.
        if let Some(px) = rest.first_chunk_mut::<8>() {
            let old = vcombine_u8(u8x8(px), u8x8(px));
            let new = vqsubq_u8(vqaddq_u8(old, up), down);
            *px = halves(new).0;
        } else {
            for px in rest {
                *px = i32::from(*px).saturating_add(v).clamp(0, 255) as u8;
            }
        }
    }
}
