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

pub const LEVEL_SCALE: [i64; 6] = [40, 45, 51, 57, 64, 72];

/// Inverse transform of the `n`×`n` scaled coefficients `coeffs` into the
/// residual `out`. Coefficients beyond `max_x` and `max_y` are zero.
/// `dst` selects the 4×4 DST. A block whose only coefficient is its DC is
/// flat: then nothing is written and the value is returned.
pub fn inverse(
    coeffs: &[i32],
    out: &mut [i32],
    log2: u32,
    dst: bool,
    bit_depth: u32,
    max_x: usize,
    max_y: usize,
) -> Option<i32> {
    let n = 1usize << log2;
    let shift2 = 20 - bit_depth;
    let add2 = 1i32 << (shift2 - 1);
    if !dst && max_x == 0 && max_y == 0 {
        let first = ((64 * coeffs[0] + 64) >> 7).clamp(-32768, 32767);
        return Some((64 * first + add2) >> shift2);
    }
    if dst {
        let mut tmp = [0i32; 16];
        for x in 0..4 {
            for y in 0..4 {
                let sum: i32 = (0..4).map(|k| DST[k][y] * coeffs[k * 4 + x]).sum();
                tmp[y * 4 + x] = ((sum + 64) >> 7).clamp(-32768, 32767);
            }
        }
        for y in 0..4 {
            for x in 0..4 {
                let sum: i32 = (0..4).map(|k| DST[k][x] * tmp[y * 4 + k]).sum();
                out[y * 4 + x] = (sum + add2) >> shift2;
            }
        }
        return None;
    }
    let block = (coeffs, out, max_x + 1, max_y + 1, add2, shift2);
    cutlass_simd::wide::wide(
        #[inline(always)]
        || match n {
            4 => dct::<4>(block),
            8 => dct::<8>(block),
            16 => dct::<16>(block),
            _ => dct::<32>(block),
        },
    );
    None
}

/// The two passes of an `N`-point inverse DCT. Each adds one basis row of
/// the matrix per nonzero input, scaled by it: runs of `N` multiply-adds.
/// The sums are the same integers as the matrix products of clause 8.6.4.2.
#[inline(always)]
fn dct<const N: usize>(
    (coeffs, out, cols, rows, add2, shift2): (&[i32], &mut [i32], usize, usize, i32, u32),
) {
    let m32 = dct32();
    let basis = |k: usize| -> &[i32; N] { m32[k * (32 / N)].first_chunk::<N>().unwrap_or(&[0; N]) };
    // Vertical: input column x to intermediate row x.
    let mut columns = [[0i32; N]; N];
    for (x, column) in columns.iter_mut().enumerate().take(cols) {
        let mut acc = [0i32; N];
        for k in 0..rows {
            let c = coeffs[k * N + x];
            if c != 0 {
                let b = basis(k);
                for i in 0..N {
                    acc[i] += c * b[i];
                }
            }
        }
        for i in 0..N {
            column[i] = ((acc[i] + 64) >> 7).clamp(-32768, 32767);
        }
    }
    // Horizontal: output row y from the first `cols` intermediate values.
    for (y, o) in out.as_chunks_mut::<N>().0.iter_mut().take(N).enumerate() {
        let mut acc = [0i32; N];
        for (k, column) in columns.iter().enumerate().take(cols) {
            let t = column[y];
            if t != 0 {
                let b = basis(k);
                for i in 0..N {
                    acc[i] += t * b[i];
                }
            }
        }
        for i in 0..N {
            o[i] = (acc[i] + add2) >> shift2;
        }
    }
}

/// Transform skip residual of a 4×4 block (equation 8-307 with `tsShift` 7).
pub fn skip(coeffs: &[i32], out: &mut [i32], bit_depth: u32) {
    let shift = 20 - bit_depth;
    let add = 1i32 << (shift - 1);
    for (o, &c) in out.iter_mut().zip(coeffs).take(16) {
        *o = ((c << 7) + add) >> shift;
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
            let mut coeffs = vec![0i32; n * n];
            coeffs[0] = 300;
            let mut slow = vec![0i32; n * n];
            let flat = inverse(&coeffs, &mut slow, log2, false, 8, 0, 0);
            // Force the general path with a wider bound on zero coefficients.
            assert_eq!(inverse(&coeffs, &mut slow, log2, false, 8, 1, 1), None);
            assert!(slow.iter().all(|&v| Some(v) == flat), "log2 {log2}");
        }
    }
}
