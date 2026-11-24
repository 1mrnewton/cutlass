//! The 8x8 DCT pair in single precision.
//!
//! The inverse is separable: one pass over rows, one over columns, each
//! split into even and odd halves. RDD 36 Annex A qualifies it; the test
//! below runs that procedure.

use std::sync::LazyLock;

/// `BASIS[x][u] = C(u) / 2 * cos((2x + 1) u pi / 16)`, `C(0) = 1 / sqrt(2)`.
static BASIS: LazyLock<[[f32; 8]; 8]> = LazyLock::new(|| {
    let mut basis = [[0f32; 8]; 8];
    for (x, row) in basis.iter_mut().enumerate() {
        for (u, cell) in row.iter_mut().enumerate() {
            let c = if u == 0 {
                std::f64::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            let angle = ((2 * x + 1) * u) as f64 * std::f64::consts::PI / 16.0;
            *cell = (c / 2.0 * angle.cos()) as f32;
        }
    }
    basis
});

/// The basis table, for a caller that transforms many blocks.
pub(crate) fn basis() -> &'static [[f32; 8]; 8] {
    &BASIS
}

/// Index of raster coefficient `[v * 8 + u]` in the layout
/// [`idct_columns`] reads: `[u * 8 + v]`.
#[inline]
pub(crate) const fn transposed(raster: u8) -> u8 {
    ((raster & 7) << 3) | (raster >> 3)
}

/// Inverse DCT of coefficients stored by column, `t[u * 8 + v]`, to sample
/// rows `[y][x]`.
///
/// Each pass is eight lanes wide: the first over the eight coefficient
/// rows, the second over the eight sample columns. The sums are taken in
/// one fixed order, so every build gives the same samples.
#[inline(always)]
pub(crate) fn idct_columns(t: &[f32; 64], basis: &[[f32; 8]; 8]) -> [[f32; 8]; 8] {
    // Rows: r[v][x] = sum_u F[v][u] * B[x][u], kept as columns c[x][v].
    let mut c = [[0f32; 8]; 8];
    for x in 0..4 {
        let mut even = [0f32; 8];
        let mut odd = [0f32; 8];
        for u in (0..8).step_by(2) {
            let be = basis[x][u];
            let bo = basis[x][u + 1];
            let (te, to) = (lane(t, u), lane(t, u + 1));
            for v in 0..8 {
                even[v] += te[v] * be;
                odd[v] += to[v] * bo;
            }
        }
        for v in 0..8 {
            c[x][v] = even[v] + odd[v];
            c[7 - x][v] = even[v] - odd[v];
        }
    }
    let mut r = [[0f32; 8]; 8];
    for v in 0..8 {
        for x in 0..8 {
            r[v][x] = c[x][v];
        }
    }
    // Columns: f[y][x] = sum_v B[y][v] * r[v][x], eight columns at once.
    let mut out = [[0f32; 8]; 8];
    for y in 0..4 {
        let mut even = [0f32; 8];
        let mut odd = [0f32; 8];
        for v in (0..8).step_by(2) {
            let be = basis[y][v];
            let bo = basis[y][v + 1];
            for x in 0..8 {
                even[x] += r[v][x] * be;
                odd[x] += r[v + 1][x] * bo;
            }
        }
        for x in 0..8 {
            out[y][x] = even[x] + odd[x];
            out[7 - y][x] = even[x] - odd[x];
        }
    }
    out
}

/// Eight values of `t` from `index * 8`.
#[inline(always)]
fn lane(t: &[f32; 64], index: usize) -> [f32; 8] {
    let mut out = [0f32; 8];
    out.copy_from_slice(&t[index * 8..index * 8 + 8]);
    out
}

/// Inverse DCT of `block`, coefficients `[v * 8 + u]`, in place to samples `[y * 8 + x]`.
#[cfg(test)]
pub(crate) fn idct(block: &mut [f32; 64]) {
    let mut t = [0f32; 64];
    for (raster, &value) in block.iter().enumerate() {
        t[usize::from(transposed(raster as u8))] = value;
    }
    let rows = idct_columns(&t, basis());
    for (y, row) in rows.iter().enumerate() {
        block[y * 8..y * 8 + 8].copy_from_slice(row);
    }
}

/// Forward DCT of samples `[y * 8 + x]`, in place to coefficients `[v * 8 + u]`.
pub(crate) fn fdct(block: &mut [f32; 64]) {
    let basis = &*BASIS;
    // Columns: c[v][x] = sum_y B[y][v] * f[y][x].
    let mut c = [0f32; 64];
    for v in 0..8 {
        let mut acc = [0f32; 8];
        for y in 0..8 {
            let b = basis[y][v];
            for x in 0..8 {
                acc[x] += block[y * 8 + x] * b;
            }
        }
        c[v * 8..v * 8 + 8].copy_from_slice(&acc);
    }
    // Rows: F[v][u] = sum_x B[x][u] * c[v][x]. Transpose first: ct[x * 8 + v].
    let mut ct = [0f32; 64];
    for v in 0..8 {
        for x in 0..8 {
            ct[x * 8 + v] = c[v * 8 + x];
        }
    }
    for u in 0..8 {
        let mut acc = [0f32; 8];
        for x in 0..8 {
            let b = basis[x][u];
            for v in 0..8 {
                acc[v] += ct[x * 8 + v] * b;
            }
        }
        for v in 0..8 {
            block[v * 8 + u] = acc[v];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{fdct, idct};

    /// The IEEE 1180-1990 appendix generator.
    struct Ieee1180 {
        state: u32,
    }

    impl Ieee1180 {
        fn next(&mut self, low: i64, high: i64) -> i64 {
            self.state = self.state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let i = i64::from(self.state & 0x7fff_fffe);
            let x = i as f64 / f64::from(0x7fff_ffff_u32) * (low + high + 1) as f64;
            x as i64 - low
        }
    }

    fn cos_table() -> [[f64; 8]; 8] {
        let mut table = [[0f64; 8]; 8];
        for (x, row) in table.iter_mut().enumerate() {
            for (u, cell) in row.iter_mut().enumerate() {
                let c = if u == 0 {
                    std::f64::consts::FRAC_1_SQRT_2
                } else {
                    1.0
                };
                *cell = c / 2.0 * (((2 * x + 1) * u) as f64 * std::f64::consts::PI / 16.0).cos();
            }
        }
        table
    }

    fn reference_fdct(pixels: &[f64; 64], table: &[[f64; 8]; 8]) -> [f64; 64] {
        let mut out = [0f64; 64];
        for v in 0..8 {
            for u in 0..8 {
                let mut sum = 0.0;
                for y in 0..8 {
                    for x in 0..8 {
                        sum += pixels[y * 8 + x] * table[x][u] * table[y][v];
                    }
                }
                out[v * 8 + u] = sum;
            }
        }
        out
    }

    fn reference_idct(coefs: &[f64; 64], table: &[[f64; 8]; 8]) -> [f64; 64] {
        let mut out = [0f64; 64];
        for y in 0..8 {
            for x in 0..8 {
                let mut sum = 0.0;
                for v in 0..8 {
                    for u in 0..8 {
                        sum += coefs[v * 8 + u] * table[x][u] * table[y][v];
                    }
                }
                out[y * 8 + x] = sum;
            }
        }
        out
    }

    /// RDD 36 Annex A: peak, mean and mean-square error against a
    /// double-precision IDCT for the IEEE 1180 ranges, both signs.
    #[test]
    fn idct_meets_annex_a() {
        let table = cos_table();
        for (low, high) in [(256, 255), (5, 5), (300, 300)] {
            for sign in [1.0, -1.0] {
                let mut rng = Ieee1180 { state: 1 };
                let mut sum = [0f64; 64];
                let mut sum_sq = [0f64; 64];
                let mut peak = 0f64;
                let blocks = 10_000;
                for _ in 0..blocks {
                    let mut pixels = [0f64; 64];
                    for p in &mut pixels {
                        *p = sign * rng.next(low, high) as f64 / 8.0;
                    }
                    let mut coefs = reference_fdct(&pixels, &table);
                    for c in &mut coefs {
                        *c = ((*c * 4.0).round() / 4.0).clamp(-2048.0, 2047.75);
                    }
                    let reference = reference_idct(&coefs, &table).map(|v| v.clamp(-256.0, 256.0));
                    let mut test = [0f32; 64];
                    for (t, c) in test.iter_mut().zip(coefs) {
                        *t = c as f32;
                    }
                    idct(&mut test);
                    for k in 0..64 {
                        let err = f64::from(test[k]).clamp(-256.0, 256.0) - reference[k];
                        sum[k] += err;
                        sum_sq[k] += err * err;
                        peak = peak.max(err.abs());
                    }
                }
                let n = f64::from(blocks);
                let pmse = sum_sq.iter().map(|s| s / n).fold(0.0, f64::max);
                let omse = sum_sq.iter().sum::<f64>() / (64.0 * n);
                let pme = sum.iter().map(|s| (s / n).abs()).fold(0.0, f64::max);
                let ome = (sum.iter().sum::<f64>() / (64.0 * n)).abs();
                assert!(peak <= 0.15, "ppe {peak} for {low},{high} sign {sign}");
                assert!(pmse <= 0.002, "pmse {pmse}");
                assert!(omse <= 0.001, "omse {omse}");
                assert!(pme <= 0.0015, "pme {pme}");
                assert!(ome <= 0.000_15, "ome {ome}");
            }
        }
    }

    #[test]
    fn forward_then_inverse_is_the_identity() {
        let mut rng = Ieee1180 { state: 7 };
        for _ in 0..200 {
            let mut block = [0f32; 64];
            for p in &mut block {
                *p = rng.next(256, 255) as f32;
            }
            let original = block;
            fdct(&mut block);
            idct(&mut block);
            for (a, b) in block.iter().zip(original) {
                assert!((a - b).abs() < 1e-3, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn a_dc_coefficient_is_a_flat_block() {
        let mut block = [0f32; 64];
        block[0] = 80.0;
        idct(&mut block);
        for value in block {
            assert!((value - 10.0).abs() < 1e-5);
        }
    }
}
