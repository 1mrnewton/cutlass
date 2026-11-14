//! Inverse transforms added to the prediction (clause 8.7.2), on lanes.
//!
//! Coefficients are stored transposed: the coefficient of row `r` and
//! column `c` of an `n`×`n` block is at `c * n + r`, so the first pass
//! (over rows) reads each input as one contiguous run of lanes.

use super::itx1d::{iadst4, iadst8, iadst16, idct4, idct8, idct16, idct32};
use super::lanes::{Elem, L, Lanes};
use super::pixel::Pixel;

/// Which 1-D transform runs on the columns and on the rows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Dct,
    Adst,
}

#[inline(always)]
fn tx1d<V: Lanes>(n: usize, adst: bool, i: &[V], o: &mut [V]) {
    match (n, adst) {
        (4, false) => idct4(i, o),
        (4, true) => iadst4(i, o),
        (8, false) => idct8(i, o),
        (8, true) => iadst8(i, o),
        (16, false) => idct16(i, o),
        (16, true) => iadst16(i, o),
        _ => idct32(i, o),
    }
}

/// The residual of one transform block added into `dst` (rows `stride`
/// apart), whose coefficients `coef` (transposed, `n * n` of them) are
/// cleared again. `nz` is the number of leading rows and columns that can
/// hold a coefficient other than 0, from the end of block.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn add<P: Pixel>(
    dst: &mut [P],
    stride: usize,
    coef: &mut [i32],
    log2n: usize,
    col_type: Kind,
    row_type: Kind,
    eob: usize,
    nz: (usize, usize),
    max: i32,
) {
    let n = 4usize << log2n;
    if eob == 1 && col_type == Kind::Dct && row_type == Kind::Dct {
        dc_only(dst, stride, coef, n, max);
        return;
    }
    if P::HIGH {
        run::<P, i64, 4>(dst, stride, coef, n, col_type, row_type, nz, max);
    } else if n == 4 {
        run::<P, i32, 4>(dst, stride, coef, n, col_type, row_type, nz, max);
    } else {
        run::<P, i32, 8>(dst, stride, coef, n, col_type, row_type, nz, max);
    }
}

#[inline(always)]
fn shift_of(n: usize) -> u32 {
    match n {
        4 => 4,
        8 => 5,
        _ => 6,
    }
}

#[inline(always)]
fn dc_only<P: Pixel>(dst: &mut [P], stride: usize, coef: &mut [i32], n: usize, max: i32) {
    let dc = i64::from(coef[0]);
    coef[0] = 0;
    let out = (dc * 11585 + 8192) >> 14;
    let out = (out * 11585 + 8192) >> 14;
    let s = shift_of(n);
    let a = ((out + (1 << (s - 1))) >> s) as i32;
    for r in 0..n {
        let row = &mut dst[r * stride..r * stride + n];
        for p in row {
            *p = P::clip(p.get() + a, max);
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn run<P: Pixel, T: Elem, const LN: usize>(
    dst: &mut [P],
    stride: usize,
    coef: &mut [i32],
    n: usize,
    col_type: Kind,
    row_type: Kind,
    nz: (usize, usize),
    max: i32,
) {
    // tmp is transposed like the input: (row r, column c) at c * n + r.
    let mut tmp = [T::ZERO; 32 * 32];
    let mut vin = [L::<T, LN>::ZERO; 32];
    let mut vout = [L::<T, LN>::ZERO; 32];
    let (nz_rows, nz_cols) = nz;
    let rows = nz_rows.min(n).div_ceil(LN) * LN;
    let cols = nz_cols.min(n);
    // First pass, lanes over rows.
    let mut r0 = 0;
    while r0 < rows {
        for k in 0..n {
            if k < cols {
                let src = &coef[k * n + r0..k * n + r0 + LN];
                vin[k] = L(core::array::from_fn(|l| T::from_i32(src[l])));
            } else {
                vin[k] = L::ZERO;
            }
        }
        tx1d(n, row_type == Kind::Adst, &vin[..n], &mut vout[..n]);
        for k in 0..n {
            tmp[k * n + r0..k * n + r0 + LN].copy_from_slice(&vout[k].0);
        }
        r0 += LN;
    }
    for k in 0..cols {
        coef[k * n..k * n + n].fill(0);
    }
    // Second pass, lanes over columns: input row r of column c is
    // tmp[c * n + r].
    let s = shift_of(n);
    let round = T::k(1 << (s - 1));
    let mut c0 = 0;
    while c0 < n {
        for r in 0..n {
            vin[r] = if r < rows {
                L(core::array::from_fn(|l| tmp[(c0 + l) * n + r]))
            } else {
                L::ZERO
            };
        }
        tx1d(n, col_type == Kind::Adst, &vin[..n], &mut vout[..n]);
        for r in 0..n {
            let row = &mut dst[r * stride + c0..r * stride + c0 + LN];
            for l in 0..LN {
                let v = vout[r].0[l].wa(round).sr(s).to_i32();
                row[l] = P::clip(row[l].get() + v, max);
            }
        }
        c0 += LN;
    }
}

/// The lossless 4×4 Walsh-Hadamard residual added into `dst`.
pub fn wht_add<P: Pixel>(dst: &mut [P], stride: usize, coef: &mut [i32], max: i32) {
    let mut t = [0i32; 16];
    // Rows: coefficient (r, c) is at c * 4 + r.
    for r in 0..4 {
        let row = [coef[r], coef[4 + r], coef[8 + r], coef[12 + r]];
        let o = wht(row, 2);
        t[r * 4..r * 4 + 4].copy_from_slice(&o);
    }
    coef[..16].fill(0);
    for c in 0..4 {
        let o = wht([t[c], t[4 + c], t[8 + c], t[12 + c]], 0);
        for r in 0..4 {
            let p = &mut dst[r * stride + c];
            *p = P::clip(p.get() + o[r], max);
        }
    }
}

#[inline(always)]
fn wht(i: [i32; 4], shift: u32) -> [i32; 4] {
    let a0 = i[0] >> shift;
    let c0 = i[1] >> shift;
    let d0 = i[2] >> shift;
    let b0 = i[3] >> shift;
    let a1 = a0.wrapping_add(c0);
    let d1 = d0.wrapping_sub(b0);
    let e = a1.wrapping_sub(d1) >> 1;
    let b1 = e.wrapping_sub(b0);
    let c1 = e.wrapping_sub(c0);
    [a1.wrapping_sub(b1), b1, c1, d1.wrapping_add(c1)]
}

#[cfg(test)]
mod tests {
    use super::{Kind, add};

    /// The lane transforms against the scalar ones of `txfm.rs`.
    #[test]
    fn matches_the_scalar_transforms() {
        let mut seed = 7u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (seed >> 8) as i32
        };
        for log2n in 0..4 {
            let n = 4usize << log2n;
            for t in 0..4u8 {
                if n == 32 && t != 0 {
                    continue;
                }
                for _ in 0..20 {
                    let mut natural = vec![0i32; n * n];
                    let k = (rnd() as usize) % (n * n) + 1;
                    for _ in 0..k.min(40) {
                        let at = (rnd() as usize) % (n * n);
                        natural[at] = rnd() % 600 - 300;
                    }
                    let mut want = vec![128u8; n * n];
                    crate::txfm::iht_add(&natural, &mut want, n, n, t);
                    let mut transposed = vec![0i32; n * n];
                    for r in 0..n {
                        for c in 0..n {
                            transposed[c * n + r] = natural[r * n + c];
                        }
                    }
                    let mut got = vec![128u8; n * n];
                    let col = if t & 1 == 1 { Kind::Adst } else { Kind::Dct };
                    let row = if t & 2 == 2 { Kind::Adst } else { Kind::Dct };
                    add(&mut got, n, &mut transposed, log2n, col, row, 2, (n, n), 255);
                    assert_eq!(got, want, "n {n} type {t}");
                    assert!(transposed.iter().all(|&v| v == 0));
                }
            }
        }
    }
}
