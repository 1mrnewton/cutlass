//! Scaling and inverse transforms (clauses 8.6.2 to 8.6.4): the 4×4 DST,
//! the 4×4 to 32×32 DCTs, transform skip, and the dequantization of
//! clause 8.6.3.

use std::sync::OnceLock;

/// `transMatrix` of equation 8-317, 32×32, `[k][n]`.
fn dct32() -> &'static [[i32; 32]; 32] {
    static MATRIX: OnceLock<[[i32; 32]; 32]> = OnceLock::new();
    MATRIX.get_or_init(|| {
        // Integer cosines of m·π/64 for m in 0..=32 (the coefficient set of
        // clause 8.6.4.2).
        const COS: [i32; 33] = [
            64, 90, 90, 90, 89, 88, 87, 85, 83, 82, 80, 78, 75, 73, 70, 67, 64, 61, 57, 54, 50, 46,
            43, 38, 36, 31, 25, 22, 18, 13, 9, 4, 0,
        ];
        let mut out = [[0i32; 32]; 32];
        for (k, row) in out.iter_mut().enumerate() {
            for (n, cell) in row.iter_mut().enumerate() {
                if k == 0 {
                    *cell = 64;
                    continue;
                }
                let mut a = ((2 * n + 1) * k) % 128;
                let mut sign = 1;
                if a > 64 {
                    a = 128 - a;
                }
                if a > 32 {
                    a = 64 - a;
                    sign = -1;
                }
                *cell = sign * COS[a];
            }
        }
        out
    })
}

const DST: [[i32; 4]; 4] = [
    [29, 55, 74, 84],
    [74, 74, 0, -74],
    [84, -29, -74, 55],
    [55, -84, 74, -29],
];

const LEVEL_SCALE: [i64; 6] = [40, 45, 51, 57, 64, 72];

/// Dequantize `coeffs` in place (raster order, `n`×`n`). `scale` is the
/// scaling factor `m` of each position, or `None` for a flat 16.
pub fn dequantize(coeffs: &mut [i32], log2: u32, qp: i32, bit_depth: u32, scale: Option<&[u8]>) {
    let shift = bit_depth + log2 - 5;
    let add = 1i64 << (shift - 1);
    let factor = LEVEL_SCALE[(qp % 6) as usize] << (qp / 6);
    match scale {
        None => {
            let f = factor * 16;
            for c in coeffs.iter_mut() {
                if *c != 0 {
                    *c = ((i64::from(*c) * f + add) >> shift).clamp(-32768, 32767) as i32;
                }
            }
        }
        Some(m) => {
            for (c, &m) in coeffs.iter_mut().zip(m) {
                if *c != 0 {
                    let value = (i64::from(*c) * i64::from(m) * factor + add) >> shift;
                    *c = value.clamp(-32768, 32767) as i32;
                }
            }
        }
    }
}

/// Inverse transform of an `n`×`n` block in place. Coefficients beyond
/// `max_x` and `max_y` are zero. `dst` selects the 4×4 DST.
pub fn inverse(
    coeffs: &mut [i32],
    log2: u32,
    dst: bool,
    bit_depth: u32,
    max_x: usize,
    max_y: usize,
) {
    let n = 1usize << log2;
    let shift2 = 20 - bit_depth;
    let add2 = 1i32 << (shift2 - 1);
    if !dst && max_x == 0 && max_y == 0 {
        // A DC-only block is flat.
        let first = ((64 * coeffs[0] + 64) >> 7).clamp(-32768, 32767);
        let value = (64 * first + add2) >> shift2;
        coeffs[..n * n].fill(value);
        return;
    }
    let mut tmp = [0i32; 32 * 32];
    let mut column = [0i32; 32];
    let mut out = [0i32; 32];
    if dst {
        for x in 0..4 {
            for y in 0..4 {
                let sum: i32 = (0..4).map(|k| DST[k][y] * coeffs[k * 4 + x]).sum();
                tmp[y * 4 + x] = ((sum + 64) >> 7).clamp(-32768, 32767);
            }
        }
        for y in 0..4 {
            for x in 0..4 {
                let sum: i32 = (0..4).map(|k| DST[k][x] * tmp[y * 4 + k]).sum();
                coeffs[y * 4 + x] = (sum + add2) >> shift2;
            }
        }
        return;
    }
    // Vertical: each column x, outputs y.
    for x in 0..=max_x {
        for (k, c) in column.iter_mut().enumerate().take(max_y + 1) {
            *c = coeffs[k * n + x];
        }
        inverse_1d(&column, n, max_y + 1, &mut out);
        for (y, &v) in out.iter().enumerate().take(n) {
            tmp[y * n + x] = ((v + 64) >> 7).clamp(-32768, 32767);
        }
    }
    // Horizontal: each row y.
    for y in 0..n {
        column[..=max_x].copy_from_slice(&tmp[y * n..y * n + max_x + 1]);
        inverse_1d(&column, n, max_x + 1, &mut out);
        for (dst, &v) in coeffs[y * n..y * n + n].iter_mut().zip(&out) {
            *dst = (v + add2) >> shift2;
        }
    }
}

/// One `n`-point inverse DCT of `coef`, of which the first `count` may be
/// nonzero, by the even/odd split of the matrix: even rows are the
/// `n/2`-point transform and odd rows are antisymmetric. The sums are the
/// same integers as the plain matrix product.
fn inverse_1d(coef: &[i32; 32], n: usize, count: usize, out: &mut [i32; 32]) {
    let m32 = dct32();
    let step = 32 / n;
    if n == 4 {
        for (i, o) in out.iter_mut().enumerate().take(4) {
            *o = (0..count.min(4)).map(|k| m32[k * 8][i] * coef[k]).sum();
        }
        return;
    }
    let half = n / 2;
    let mut even = [0i32; 32];
    for (k, e) in even.iter_mut().enumerate().take(half) {
        *e = coef[2 * k];
    }
    let mut e_out = [0i32; 32];
    inverse_1d(&even, half, count.div_ceil(2), &mut e_out);
    for i in 0..half {
        let mut odd = 0;
        let mut k = 1;
        while k < count.min(n) {
            odd += m32[k * step][i] * coef[k];
            k += 2;
        }
        out[i] = e_out[i] + odd;
        out[n - 1 - i] = e_out[i] - odd;
    }
}

/// Transform skip residual of a 4×4 block (equation 8-307 with `tsShift` 7).
pub fn skip(coeffs: &mut [i32], bit_depth: u32) {
    let shift = 20 - bit_depth;
    let add = 1i32 << (shift - 1);
    for c in coeffs.iter_mut().take(16) {
        *c = ((*c << 7) + add) >> shift;
    }
}

#[cfg(test)]
mod tests {
    use super::{dct32, inverse};

    #[test]
    fn the_matrix_matches_the_coefficients_of_equation_8_317() {
        let m = dct32();
        assert_eq!(&m[1][..4], &[90, 90, 88, 85]);
        assert_eq!(m[16][0], 64);
        assert_eq!(m[16][1], -64);
        assert_eq!(&m[8][..4], &[83, 36, -36, -83]);
        assert_eq!(m[2][15], -90);
        assert_eq!(m[31][31], -4);
        // The 4-point DCT is rows 0, 8, 16, 24 at columns 0..4.
        assert_eq!([m[24][0], m[24][1], m[24][2], m[24][3]], [36, -83, 83, -36]);
    }

    #[test]
    fn dc_only_and_full_paths_agree() {
        for log2 in 2..=5u32 {
            let n = 1usize << log2;
            let mut fast = vec![0i32; n * n];
            fast[0] = 300;
            let mut slow = fast.clone();
            inverse(&mut fast, log2, false, 8, 0, 0);
            // Force the general path with a wider bound on zero coefficients.
            inverse(&mut slow, log2, false, 8, 1, 1);
            assert_eq!(fast, slow, "log2 {log2}");
        }
    }
}
