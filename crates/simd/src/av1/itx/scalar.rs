//! The block transform a value at a time: what the vector kernel is
//! checked against, and what other targets run.
#![cfg_attr(
    all(target_arch = "aarch64", target_feature = "neon"),
    allow(dead_code)
)]

use super::steps::{Prog, Step, kind, plan, table};
use super::{Block, Kind};

/// The arithmetic of a step on the eight values of an entry.
mod ops {
    pub(super) type V = [i16; 8];

    pub(super) fn zero() -> V {
        [0; 8]
    }

    /// `Round2(x, 12)`, its low 16 bits.
    fn round12(x: i32) -> i16 {
        (x.wrapping_add(2048) >> 12) as i16
    }

    /// `x cos - y sin` and `x sin + y cos`, rounded at 12 bits.
    pub(super) fn rot(x: V, y: V, cos: i16, sin: i16) -> (V, V) {
        let (c, s) = (i32::from(cos), i32::from(sin));
        let at = |l: usize| (i32::from(x[l]), i32::from(y[l]));
        (
            core::array::from_fn(|l| round12(at(l).0 * c - at(l).1 * s)),
            core::array::from_fn(|l| round12(at(l).0 * s + at(l).1 * c)),
        )
    }

    /// `x c`, rounded at 12 bits.
    pub(super) fn scale(x: V, c: i16) -> V {
        x.map(|v| round12(i32::from(v) * i32::from(c)))
    }

    pub(super) fn add(x: V, y: V) -> V {
        core::array::from_fn(|l| x[l].saturating_add(y[l]))
    }

    pub(super) fn sub(x: V, y: V) -> V {
        core::array::from_fn(|l| x[l].saturating_sub(y[l]))
    }

    pub(super) fn neg(x: V) -> V {
        x.map(i16::saturating_neg)
    }

    /// The identity transform of `n` values on one of them (7.13.2.15):
    /// a scaling by `sqrt(2)`, 2, `2 sqrt(2)` or 4.
    fn identity(x: i32, n: usize) -> i32 {
        let round12 = |x: i32| (x + 2048) >> 12;
        match n {
            4 => x + round12(x * 1697),
            8 => 2 * x,
            16 => 2 * x + round12(x * 3394),
            _ => 4 * x,
        }
    }

    /// An output of a row identity transform of `n` values, after the
    /// rounding shift, clamped to 16 bits.
    pub(super) fn identity_row(x: V, n: usize, shift: u32) -> V {
        let half = (1i32 << shift) >> 1;
        x.map(|v| ((identity(i32::from(v), n) + half) >> shift).clamp(-32768, 32767) as i16)
    }

    /// The residual of a column identity transform of `n` values:
    /// `Round2` by 4 of the output.
    pub(super) fn identity_col(x: V, n: usize) -> V {
        x.map(|v| ((identity(i32::from(v), n) + 8) >> 4) as i16)
    }
}

transforms!();

/// The transform `kind` of `n` values on eight lanes, for tests.
#[cfg(test)]
pub(super) fn one_dimension(kind: Kind, n: usize, nz: usize, t: &mut [[i16; 8]; 64]) -> bool {
    run_1d(kind, n, nz, t)
}

pub(super) fn add(b: &mut Block<'_>) {
    let (w, h) = (b.w, b.h);
    let (tw, th) = (w.min(32), h.min(32));
    let nr = b.nz_rows.clamp(1, th);
    let nc = b.nz_cols.clamp(1, tw);
    let rect = w.max(h) == 2 * w.min(h);
    let half = (1i32 << b.shift) >> 1;
    // The values after the row transforms, by row.
    let mut mid = [[0i16; 64]; 32];
    for r0 in (0..nr).step_by(8) {
        let mut t = [[0i16; 8]; 64];
        for (k, v) in t.iter_mut().enumerate().take(nc.next_multiple_of(8)) {
            for (l, x) in v.iter_mut().enumerate() {
                if r0 + l < nr {
                    *x = b.coefs[(r0 + l) * tw + k] as i16;
                }
            }
            if rect {
                *v = ops::scale(*v, 2896);
            }
        }
        run_1d(b.row, w, nc, &mut t);
        for (j, v) in t.iter().enumerate().take(w) {
            let out = if b.row == Kind::Identity {
                ops::identity_row(*v, w, b.shift)
            } else {
                v.map(|x| ((i32::from(x) + half) >> b.shift) as i16)
            };
            for (l, &x) in out.iter().enumerate() {
                mid[r0 + l][j] = x;
            }
        }
    }
    for j0 in (0..w).step_by(8) {
        let mut t = [[0i16; 8]; 64];
        for (k, v) in t.iter_mut().enumerate().take(th) {
            v.copy_from_slice(&mid[k][j0..j0 + 8]);
        }
        run_1d(b.col, h, nr, &mut t);
        for (i, v) in t.iter().enumerate().take(h) {
            let y = if b.flip_ud { h - 1 - i } else { i };
            let res = if b.col == Kind::Identity {
                ops::identity_col(*v, h)
            } else {
                // Round2(x, 4), which no 16-bit value overflows.
                v.map(|x| (x >> 4) + ((x >> 3) & 1))
            };
            for (l, &r) in res.iter().enumerate() {
                let col = if b.flip_lr { w - 1 - (j0 + l) } else { j0 + l };
                let px = &mut b.dst[y * b.stride + col];
                *px = (i16::from(*px) + r).clamp(0, 255) as u8;
            }
        }
    }
}

pub(super) fn add_flat(dst: &mut [u8], stride: usize, w: usize, h: usize, v: i32) {
    for y in 0..h {
        for px in &mut dst[y * stride..][..w] {
            *px = i32::from(*px).saturating_add(v).clamp(0, 255) as u8;
        }
    }
}
