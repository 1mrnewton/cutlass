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

/// Scratch of the transforms' first pass, kept from block to block: the
/// second pass reads only the rows the first wrote, so it is never
/// cleared.
pub struct Tmp {
    i: Vec<i32>,
    l: Vec<i64>,
}

impl Tmp {
    pub fn new() -> Self {
        Self {
            i: vec![0; 32 * 32],
            l: vec![0; 32 * 32],
        }
    }
}

impl Default for Tmp {
    fn default() -> Self {
        Self::new()
    }
}

/// The scratch of an element type.
trait Pick: Elem {
    fn pick(t: &mut Tmp) -> &mut [Self];
}

impl Pick for i32 {
    #[inline(always)]
    fn pick(t: &mut Tmp) -> &mut [Self] {
        &mut t.i
    }
}

impl Pick for i64 {
    #[inline(always)]
    fn pick(t: &mut Tmp) -> &mut [Self] {
        &mut t.l
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
    scratch: &mut Tmp,
) {
    let n = 4usize << log2n;
    if eob == 1 && col_type == Kind::Dct && row_type == Kind::Dct {
        dc_only(dst, stride, coef, n, max);
        return;
    }
    cutlass_simd::vp9::wide(
        #[inline(always)]
        || {
            macro_rules! go {
                ($t:ty, $ln:literal, $n:literal) => {{
                    let tmp = &mut <$t as Pick>::pick(scratch)[..$n * $n];
                    run::<P, $t, $ln, $n>(dst, stride, coef, tmp, col_type, row_type, nz, max)
                }};
            }
            // Deep pictures take the 32-bit lanes when no sum or product
            // can leave `i32` (see `NARROW`), where they give the 64-bit
            // lanes' values exactly.
            let narrow = P::HIGH && magnitude(coef, n, nz) <= u64::from(NARROW[log2n]);
            match (P::HIGH && !narrow, n) {
                (true, 4) => go!(i64, 4, 4),
                (true, 8) => go!(i64, 4, 8),
                (true, 16) => go!(i64, 4, 16),
                (true, _) => go!(i64, 4, 32),
                (false, 4) => go!(i32, 4, 4),
                (false, 8) => go!(i32, 8, 8),
                (false, 16) => go!(i32, 8, 16),
                (false, _) => go!(i32, 8, 32),
            }
        },
    );
}

/// By `log2n`, the largest sum of the coefficients' magnitudes for which
/// every value the 64-bit lanes make for a block (each sum, difference,
/// product and rounding of both passes, and the final rounding) lies in
/// `i32`: there the 32-bit lanes' wrapping arithmetic never wraps, and
/// they make the same values. `tests::narrow_bounds_hold` derives the
/// bound from the transforms themselves, run on bounds of magnitudes
/// (55 661, 77 464, 40 572 and 39 509), and checks these against it.
const NARROW: [u32; 4] = [55_000, 77_000, 40_000, 39_000];

/// The sum of the magnitudes of every coefficient either lane width reads
/// (columns `0..nz.1`, rows `0..nz.0` rounded up to 8, transposed).
#[inline(always)]
fn magnitude(coef: &[i32], n: usize, nz: (usize, usize)) -> u64 {
    let (rows, cols) = (nz.0.min(n).div_ceil(8) * 8, nz.1.min(n));
    let rows = rows.min(n);
    let mut m = 0u64;
    for k in 0..cols {
        for &v in &coef[k * n..k * n + rows] {
            m += u64::from(v.unsigned_abs());
        }
    }
    m
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
fn run<P: Pixel, T: Elem, const LN: usize, const N: usize>(
    dst: &mut [P],
    stride: usize,
    coef: &mut [i32],
    tmp: &mut [T],
    col_type: Kind,
    row_type: Kind,
    nz: (usize, usize),
    max: i32,
) {
    // tmp is transposed like the input: (row r, column c) at c * n + r.
    let n = N;
    let mut vin = [L::<T, LN>::ZERO; N];
    let mut vout = [L::<T, LN>::ZERO; N];
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
        tx1d(N, row_type == Kind::Adst, &vin, &mut vout);
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
        tx1d(N, col_type == Kind::Adst, &vin, &mut vout);
        for r in 0..n {
            let row = &mut dst[r * stride + c0..r * stride + c0 + LN];
            #[allow(clippy::needless_range_loop)]
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
    use super::{Kind, NARROW, add, shift_of, tx1d};
    use crate::dsp::lanes::{K, Lanes};
    use std::cell::Cell;

    thread_local! {
        /// The 1-D input's bound on the sum of magnitudes, and the
        /// largest bound a step has made from it.
        static SUM: Cell<f64> = const { Cell::new(0.0) };
        static TOP: Cell<f64> = const { Cell::new(0.0) };
    }

    /// A bound on the magnitude of a value of a 1-D transform of inputs
    /// `x`: `sum(w[i] * |x[i]|) + c`. Each operation records the bound its
    /// result has when the inputs' magnitudes sum to at most `SUM`, which
    /// is `max(w) * SUM + c`, in `TOP`. The values are linear in the
    /// inputs but for the roundings, each of which adds at most 1.5.
    #[derive(Clone, Copy)]
    struct Mag {
        w: [f64; 32],
        c: f64,
    }

    fn rec(m: Mag) -> Mag {
        let g = m.w.iter().fold(0f64, |a, &b| a.max(b));
        let v = g * SUM.with(Cell::get) + m.c;
        TOP.with(|t| t.set(t.get().max(v)));
        m
    }

    fn input(i: usize) -> Mag {
        let mut m = Mag::ZERO;
        m.w[i] = 1.0;
        m
    }

    impl Mag {
        /// `k * self + o`, plus `c`.
        fn of(self, o: Self, k: f64, c: f64) -> Self {
            rec(Mag {
                w: core::array::from_fn(|i| k * self.w[i] + o.w[i]),
                c: k * self.c + o.c + c,
            })
        }
    }

    impl Lanes for Mag {
        const ZERO: Self = Mag {
            w: [0.0; 32],
            c: 0.0,
        };
        fn wrapping_add(self, o: Self) -> Self {
            self.of(o, 1.0, 0.0)
        }
        fn wrapping_sub(self, o: Self) -> Self {
            self.of(o, 1.0, 0.0)
        }
        fn wrapping_mul(self, k: K) -> Self {
            self.of(Mag::ZERO, k.0.unsigned_abs() as f64, 0.0)
        }
        fn wrapping_neg(self) -> Self {
            self
        }
        fn round_shift(self) -> Self {
            // `x + 8192`, then `>> 14`: `|(x + 8192) >> 14|` is at most
            // `|x| / 16384 + 1.5`.
            self.of(Mag::ZERO, 1.0, 8192.0);
            self.of(Mag::ZERO, 1.0 / 16384.0, 1.5)
        }
    }

    /// The largest bound any step of a 1-D transform of `n` inputs makes
    /// when their magnitudes sum to at most `sum`, and the bound of its
    /// outputs as `(max(w), c)`.
    fn pass(n: usize, adst: bool, sum: f64) -> (f64, (f64, f64)) {
        SUM.with(|s| s.set(sum));
        TOP.with(|t| t.set(sum));
        let i: Vec<Mag> = (0..n).map(input).collect();
        let mut o = vec![Mag::ZERO; n];
        tx1d(n, adst, &i, &mut o);
        let g = o.iter().flat_map(|m| m.w).fold(0f64, f64::max);
        let c = o.iter().map(|m| m.c).fold(0f64, f64::max);
        (TOP.with(Cell::get), (g, c))
    }

    /// The largest magnitude any step of a 2-D transform of size `n`
    /// makes from coefficients whose magnitudes sum to at most `sum`, over
    /// every pair of 1-D transforms. A first-pass output (row `r`, column
    /// `k`) is at most `g * (row r's sum) + c`, so a column's inputs to
    /// the second pass sum to at most `g * sum + n * c`.
    fn top(n: usize, sum: f64) -> f64 {
        let kinds: &[bool] = if n == 32 { &[false] } else { &[false, true] };
        let mut worst = 0f64;
        for &row in kinds {
            for &col in kinds {
                let (t1, (g, c)) = pass(n, row, sum);
                let sum2 = g * sum + n as f64 * c;
                let (t2, (g2, c2)) = pass(n, col, sum2);
                let last = g2 * sum2 + c2 + f64::from(1 << (shift_of(n) - 1));
                worst = worst.max(t1).max(t2).max(last);
            }
        }
        worst
    }

    /// Whether every value fits in `i32`, with room for the bound's own
    /// floating-point error (far below a part in 10^9).
    fn fits(n: usize, sum: u32) -> bool {
        top(n, f64::from(sum)) <= f64::from(i32::MAX) * (1.0 - 1e-9)
    }

    /// The 32-bit lanes are taken only where they cannot wrap.
    #[test]
    fn narrow_bounds_hold() {
        for (log2n, &bound) in NARROW.iter().enumerate() {
            let n = 4 << log2n;
            let (mut lo, mut hi) = (0u32, u32::MAX);
            while lo + 1 < hi {
                let mid = lo + (hi - lo) / 2;
                if fits(n, mid) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            println!("n {n}: largest bound {lo}, constant {bound}");
            assert!(fits(n, bound), "n {n}");
        }
    }

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
                    add(
                        &mut got,
                        n,
                        &mut transposed,
                        log2n,
                        col,
                        row,
                        2,
                        (n, n),
                        255,
                        &mut super::Tmp::new(),
                    );
                    assert_eq!(got, want, "n {n} type {t}");
                    assert!(transposed.iter().all(|&v| v == 0));
                }
            }
        }
    }

    /// Deep samples, on both lane widths: coefficients whose magnitudes
    /// sum to around the bound (some to just under it, the 32-bit lanes;
    /// some past it, the 64-bit ones), few or many, against the scalar
    /// transforms.
    #[test]
    fn deep_samples_match_the_scalar_transforms_near_the_bound() {
        let mut seed = 11u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            seed >> 8
        };
        for (log2n, &bound) in NARROW.iter().enumerate() {
            let n = 4usize << log2n;
            for t in 0..4u8 {
                if n == 32 && t != 0 {
                    continue;
                }
                for trial in 0..200 {
                    let total = match trial % 3 {
                        0 => bound,
                        1 => bound / 3,
                        _ => bound * 3,
                    } as i32;
                    let count = 1 + (rnd() as usize) % if trial % 2 == 0 { 4 } else { n * n };
                    let mut natural = vec![0i32; n * n];
                    let mut left = total;
                    for k in 0..count {
                        let at = (rnd() as usize) % (n * n);
                        let v = if k + 1 == count {
                            left
                        } else {
                            (rnd() as i32) % (left + 1)
                        };
                        left -= v;
                        natural[at] = if rnd() % 2 == 0 { v } else { -v };
                    }
                    let mid: Vec<u16> = (0..n * n).map(|_| (rnd() % 4096) as u16).collect();
                    let mut want: Vec<crate::pel::Wide<12>> =
                        mid.iter().map(|&v| crate::pel::Wide(v)).collect();
                    let wide: Vec<i64> = natural.iter().map(|&v| i64::from(v)).collect();
                    crate::txfm::iht_add_pix(&wide, &mut want, n, n, t);
                    let mut transposed = vec![0i32; n * n];
                    for r in 0..n {
                        for c in 0..n {
                            transposed[c * n + r] = natural[r * n + c];
                        }
                    }
                    let mut got = mid.clone();
                    let col = if t & 1 == 1 { Kind::Adst } else { Kind::Dct };
                    let row = if t & 2 == 2 { Kind::Adst } else { Kind::Dct };
                    add(
                        &mut got,
                        n,
                        &mut transposed,
                        log2n,
                        col,
                        row,
                        2,
                        (n, n),
                        4095,
                        &mut super::Tmp::new(),
                    );
                    let want: Vec<u16> = want.iter().map(|w| w.0).collect();
                    assert_eq!(got, want, "n {n} type {t} sum {total}");
                }
            }
        }
    }
}
