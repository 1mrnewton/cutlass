//! The 8×8 inverse DCT of Annex A and a forward DCT for the encoder.
//!
//! The inverse transform is separable fixed point. Each 1-D pass multiplies
//! by `W[k] = round(2^14 · √2 · cos(kπ/16))`, the row pass keeps 3 fraction
//! bits, and the column pass removes the rest. It meets IEEE 1180-1990, which
//! H.262 Annex A requires; the test below runs that procedure.

/// `round(2^14 · √2 · cos(kπ/16))`, `W[4]` being `2^14`.
const W1: i32 = 22725;
const W2: i32 = 21407;
const W3: i32 = 19266;
const W4: i32 = 16384;
const W5: i32 = 12873;
const W6: i32 = 8867;
const W7: i32 = 4520;

const ROW_SHIFT: u32 = 11;
const COL_SHIFT: u32 = 20;

/// Largest row output for which the 32-bit column pass cannot overflow:
/// the column sums at most `122_426 · |R|` plus the rounding term.
const COL_I32_LIMIT: i32 = 17_000;

/// One row in place. Returns the largest magnitude written.
#[inline(always)]
fn row(r: &mut [i32]) -> i32 {
    let (x0, x1, x2, x3, x4, x5, x6, x7) = (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
    if x1 | x2 | x3 | x4 | x5 | x6 | x7 == 0 {
        // W4 is 2^14, so a lone DC term is exact.
        let dc = x0 << 3;
        r.fill(dc);
        return dc.abs();
    }
    let round = 1 << (ROW_SHIFT - 1);
    let a = W4 * (x0 + x4) + round;
    let a_ = W4 * (x0 - x4) + round;
    let b0 = W2 * x2 + W6 * x6;
    let b1 = W6 * x2 - W2 * x6;
    let e0 = a + b0;
    let e3 = a - b0;
    let e1 = a_ + b1;
    let e2 = a_ - b1;
    let o0 = W1 * x1 + W3 * x3 + W5 * x5 + W7 * x7;
    let o1 = W3 * x1 - W7 * x3 - W1 * x5 - W5 * x7;
    let o2 = W5 * x1 - W1 * x3 + W7 * x5 + W3 * x7;
    let o3 = W7 * x1 - W5 * x3 + W3 * x5 - W1 * x7;
    r[0] = (e0 + o0) >> ROW_SHIFT;
    r[7] = (e0 - o0) >> ROW_SHIFT;
    r[1] = (e1 + o1) >> ROW_SHIFT;
    r[6] = (e1 - o1) >> ROW_SHIFT;
    r[2] = (e2 + o2) >> ROW_SHIFT;
    r[5] = (e2 - o2) >> ROW_SHIFT;
    r[3] = (e3 + o3) >> ROW_SHIFT;
    r[4] = (e3 - o3) >> ROW_SHIFT;
    r.iter().fold(0, |max, &v| max.max(v.abs()))
}

macro_rules! columns {
    ($name:ident, $t:ty, $widen:expr, $narrow:expr) => {
        #[inline(always)]
        fn $name(block: &mut [i32; 64]) {
            let widen: fn(i32) -> $t = $widen;
            let narrow: fn($t) -> i32 = $narrow;
            let w = |k: i32| -> $t { widen(k) };
            let round: $t = 1 << (COL_SHIFT - 1);
            for c in 0..8 {
                let x = |n: usize| -> $t { widen(block[n * 8 + c]) };
                let a = w(W4) * (x(0) + x(4)) + round;
                let a_ = w(W4) * (x(0) - x(4)) + round;
                let b0 = w(W2) * x(2) + w(W6) * x(6);
                let b1 = w(W6) * x(2) - w(W2) * x(6);
                let e0 = a + b0;
                let e3 = a - b0;
                let e1 = a_ + b1;
                let e2 = a_ - b1;
                let o0 = w(W1) * x(1) + w(W3) * x(3) + w(W5) * x(5) + w(W7) * x(7);
                let o1 = w(W3) * x(1) - w(W7) * x(3) - w(W1) * x(5) - w(W5) * x(7);
                let o2 = w(W5) * x(1) - w(W1) * x(3) + w(W7) * x(5) + w(W3) * x(7);
                let o3 = w(W7) * x(1) - w(W5) * x(3) + w(W3) * x(5) - w(W1) * x(7);
                let out = [
                    e0 + o0,
                    e1 + o1,
                    e2 + o2,
                    e3 + o3,
                    e3 - o3,
                    e2 - o2,
                    e1 - o1,
                    e0 - o0,
                ];
                for (n, v) in out.into_iter().enumerate() {
                    block[n * 8 + c] = narrow(v >> COL_SHIFT);
                }
            }
        }
    };
}

columns!(columns_i32, i32, |v| v, |v| v);
// The shifted sum is at most 2^36 / 2^20, so it fits.
columns!(columns_i64, i64, i64::from, |v| v as i32);

/// Inverse DCT in place. Input is the saturated coefficients `F[v][u]` in
/// raster order; output is `f[y][x]`, not yet saturated to [-256, 255].
pub(crate) fn idct(block: &mut [i32; 64]) {
    let mut max = 0;
    for r in block.chunks_exact_mut(8) {
        max = max.max(row(r));
    }
    if max <= COL_I32_LIMIT {
        columns_i32(block);
    } else {
        columns_i64(block);
    }
}

/// Inverse DCT of a block whose only non-zero coefficient is `F[0][0]`.
pub(crate) fn idct_dc(dc: i32) -> i32 {
    // Row pass gives `dc << 3`, the column pass `(W4 · (dc << 3) + 2^19) >> 20`.
    ((dc << 3) * W4 + (1 << (COL_SHIFT - 1))) >> COL_SHIFT
}

/// `cos((2x + 1)uπ/16) · C(u) / 2`, the 1-D DCT basis.
struct Basis([[f32; 8]; 8]);

fn basis() -> &'static Basis {
    static BASIS: std::sync::OnceLock<Basis> = std::sync::OnceLock::new();
    BASIS.get_or_init(|| {
        let mut m = [[0f32; 8]; 8];
        for (u, row) in m.iter_mut().enumerate() {
            let c = if u == 0 {
                std::f64::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            for (x, value) in row.iter_mut().enumerate() {
                let angle = (2 * x + 1) as f64 * u as f64 * std::f64::consts::PI / 16.0;
                *value = (c / 2.0 * angle.cos()) as f32;
            }
        }
        Basis(m)
    })
}

/// Forward DCT of Annex A, unrounded: `F[v][u]` in raster order.
pub(crate) fn fdct(input: &[i32; 64], out: &mut [f32; 64]) {
    let m = &basis().0;
    let mut tmp = [0f32; 64];
    // Rows: tmp[y][u] = Σx m[u][x] · f[y][x].
    for y in 0..8 {
        let line = &input[y * 8..y * 8 + 8];
        for u in 0..8 {
            let mut sum = 0f32;
            for x in 0..8 {
                sum += m[u][x] * line[x] as f32;
            }
            tmp[y * 8 + u] = sum;
        }
    }
    // Columns: F[v][u] = Σy m[v][y] · tmp[y][u].
    for v in 0..8 {
        for u in 0..8 {
            let mut sum = 0f32;
            for y in 0..8 {
                sum += m[v][y] * tmp[y * 8 + u];
            }
            out[v * 8 + u] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The random block generator of IEEE 1180-1990.
    struct IeeeRand(u32);

    impl IeeeRand {
        fn next(&mut self, low: i64, high: i64) -> i64 {
            self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12345);
            let i = self.0 & 0x7fff_fffe;
            let x = f64::from(i) / f64::from(0x7fff_ffffu32) * (low + high + 1) as f64;
            x as i64 - low
        }
    }

    fn reference_fdct(block: &[f64; 64]) -> [f64; 64] {
        let mut out = [0f64; 64];
        for v in 0..8 {
            for u in 0..8 {
                let mut sum = 0.0;
                for y in 0..8 {
                    for x in 0..8 {
                        sum += block[y * 8 + x]
                            * ((2 * x + 1) as f64 * u as f64 * std::f64::consts::PI / 16.0).cos()
                            * ((2 * y + 1) as f64 * v as f64 * std::f64::consts::PI / 16.0).cos();
                    }
                }
                let cu = if u == 0 {
                    std::f64::consts::FRAC_1_SQRT_2
                } else {
                    1.0
                };
                let cv = if v == 0 {
                    std::f64::consts::FRAC_1_SQRT_2
                } else {
                    1.0
                };
                out[v * 8 + u] = sum * cu * cv / 4.0;
            }
        }
        out
    }

    fn reference_idct(coef: &[f64; 64]) -> [f64; 64] {
        let mut out = [0f64; 64];
        for y in 0..8 {
            for x in 0..8 {
                let mut sum = 0.0;
                for v in 0..8 {
                    for u in 0..8 {
                        let cu = if u == 0 {
                            std::f64::consts::FRAC_1_SQRT_2
                        } else {
                            1.0
                        };
                        let cv = if v == 0 {
                            std::f64::consts::FRAC_1_SQRT_2
                        } else {
                            1.0
                        };
                        sum += cu
                            * cv
                            * coef[v * 8 + u]
                            * ((2 * x + 1) as f64 * u as f64 * std::f64::consts::PI / 16.0).cos()
                            * ((2 * y + 1) as f64 * v as f64 * std::f64::consts::PI / 16.0).cos();
                    }
                }
                out[y * 8 + x] = sum / 4.0;
            }
        }
        out
    }

    /// One IEEE 1180 run: 10 000 blocks in `[-low, high]`, times `sign`.
    fn ieee_1180_run(low: i64, high: i64, sign: i64) {
        let mut rand = IeeeRand(1);
        let mut err_sum = [0f64; 64];
        let mut err_sq = [0f64; 64];
        let mut peak = 0i64;
        let blocks = 10_000;
        for _ in 0..blocks {
            let mut pixels = [0f64; 64];
            for value in &mut pixels {
                *value = (rand.next(low, high) * sign) as f64;
            }
            let coef = reference_fdct(&pixels).map(|c| c.round().clamp(-2048.0, 2047.0));
            let want = reference_idct(&coef).map(|v| v.round().clamp(-256.0, 255.0) as i64);
            let mut ours = coef.map(|c| c as i32);
            idct(&mut ours);
            for i in 0..64 {
                let got = i64::from(ours[i].clamp(-256, 255));
                let err = got - want[i];
                peak = peak.max(err.abs());
                err_sum[i] += err as f64;
                err_sq[i] += (err * err) as f64;
            }
        }
        let n = f64::from(blocks);
        assert!(peak <= 1, "peak error {peak} for [{low}, {high}] × {sign}");
        for i in 0..64 {
            assert!(err_sq[i] / n <= 0.06, "pmse at {i}: {}", err_sq[i] / n);
            assert!(
                (err_sum[i] / n).abs() <= 0.015,
                "pme at {i}: {}",
                err_sum[i] / n
            );
        }
        let omse = err_sq.iter().sum::<f64>() / (n * 64.0);
        let ome = err_sum.iter().sum::<f64>() / (n * 64.0);
        assert!(omse <= 0.02, "omse {omse}");
        assert!(ome.abs() <= 0.0015, "ome {ome}");
    }

    #[test]
    fn ieee_1180_range_256_255() {
        ieee_1180_run(256, 255, 1);
        ieee_1180_run(256, 255, -1);
    }

    #[test]
    fn ieee_1180_range_5() {
        ieee_1180_run(5, 5, 1);
        ieee_1180_run(5, 5, -1);
    }

    #[test]
    fn ieee_1180_range_300() {
        ieee_1180_run(300, 300, 1);
        ieee_1180_run(300, 300, -1);
    }

    #[test]
    fn zero_in_gives_zero_out() {
        let mut block = [0i32; 64];
        idct(&mut block);
        assert_eq!(block, [0; 64]);
    }

    #[test]
    fn the_dc_shortcut_matches_the_full_transform() {
        for dc in [-2048, -1023, -9, -1, 0, 1, 7, 8, 1016, 2047] {
            let mut block = [0i32; 64];
            block[0] = dc;
            idct(&mut block);
            assert!(block.iter().all(|&v| v == idct_dc(dc)), "dc {dc}");
        }
    }

    #[test]
    fn extreme_coefficients_take_the_wide_path_without_overflow() {
        let mut block = [2047i32; 64];
        for (i, value) in block.iter_mut().enumerate() {
            if (i * 7) % 3 == 0 {
                *value = -2048;
            }
        }
        let want = reference_idct(&block.map(f64::from));
        idct(&mut block);
        for i in 0..64 {
            assert!((f64::from(block[i]) - want[i]).abs() <= 1.0, "{i}");
        }
    }

    #[test]
    fn forward_then_inverse_is_close_to_identity() {
        let mut rand = IeeeRand(7);
        let mut pixels = [0i32; 64];
        for value in &mut pixels {
            *value = rand.next(256, 255) as i32;
        }
        let mut coef = [0f32; 64];
        fdct(&pixels, &mut coef);
        let reference = reference_fdct(&pixels.map(f64::from));
        for i in 0..64 {
            assert!((f64::from(coef[i]) - reference[i]).abs() < 0.01);
        }
        let mut block = coef.map(|c| c.round() as i32);
        idct(&mut block);
        for i in 0..64 {
            assert!((block[i] - pixels[i]).abs() <= 1, "{i}");
        }
    }
}
