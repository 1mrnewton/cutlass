//! The 8x8 inverse DCT.
//!
//! A separable fixed-point transform with 14-bit cosine weights: rows are
//! rounded to 16 bits, columns shifted down by 20. It meets the IEEE 1180
//! accuracy the standard requires (Annex A), and matches the integer
//! transform FFmpeg selects with `-idct simple`, so decoded pictures can be
//! compared with it sample for sample.

const W1: i32 = 22725;
const W2: i32 = 21407;
const W3: i32 = 19266;
const W4: i32 = 16383;
const W5: i32 = 12873;
const W6: i32 = 8867;
const W7: i32 = 4520;
const ROW_SHIFT: u32 = 11;
const COL_SHIFT: u32 = 20;

#[inline]
fn row(r: &mut [i16]) {
    let [r0, r1, r2, r3, r4, r5, r6, r7] = match r {
        [a, b, c, d, e, f, g, h] => [*a, *b, *c, *d, *e, *f, *g, *h].map(i32::from),
        _ => return,
    };
    if r1 == 0 && r2 == 0 && r3 == 0 && r4 == 0 && r5 == 0 && r6 == 0 && r7 == 0 {
        // A DC-only row scales by 8, keeping 16 bits.
        let v = r0.wrapping_mul(8) as i16;
        r.fill(v);
        return;
    }
    let round = 1i32 << (ROW_SHIFT - 1);
    let mut a0 = W4.wrapping_mul(r0).wrapping_add(round);
    let mut a1 = a0;
    let mut a2 = a0;
    let mut a3 = a0;
    a0 = a0.wrapping_add(W2.wrapping_mul(r2));
    a1 = a1.wrapping_add(W6.wrapping_mul(r2));
    a2 = a2.wrapping_sub(W6.wrapping_mul(r2));
    a3 = a3.wrapping_sub(W2.wrapping_mul(r2));
    let mut b0 = W1.wrapping_mul(r1).wrapping_add(W3.wrapping_mul(r3));
    let mut b1 = W3.wrapping_mul(r1).wrapping_sub(W7.wrapping_mul(r3));
    let mut b2 = W5.wrapping_mul(r1).wrapping_sub(W1.wrapping_mul(r3));
    let mut b3 = W7.wrapping_mul(r1).wrapping_sub(W5.wrapping_mul(r3));
    if r4 != 0 || r5 != 0 || r6 != 0 || r7 != 0 {
        a0 = a0
            .wrapping_add(W4.wrapping_mul(r4))
            .wrapping_add(W6.wrapping_mul(r6));
        a1 = a1
            .wrapping_sub(W4.wrapping_mul(r4))
            .wrapping_sub(W2.wrapping_mul(r6));
        a2 = a2
            .wrapping_sub(W4.wrapping_mul(r4))
            .wrapping_add(W2.wrapping_mul(r6));
        a3 = a3
            .wrapping_add(W4.wrapping_mul(r4))
            .wrapping_sub(W6.wrapping_mul(r6));
        b0 = b0
            .wrapping_add(W5.wrapping_mul(r5))
            .wrapping_add(W7.wrapping_mul(r7));
        b1 = b1
            .wrapping_sub(W1.wrapping_mul(r5))
            .wrapping_sub(W5.wrapping_mul(r7));
        b2 = b2
            .wrapping_add(W7.wrapping_mul(r5))
            .wrapping_add(W3.wrapping_mul(r7));
        b3 = b3
            .wrapping_add(W3.wrapping_mul(r5))
            .wrapping_sub(W1.wrapping_mul(r7));
    }
    let out = [
        a0.wrapping_add(b0),
        a1.wrapping_add(b1),
        a2.wrapping_add(b2),
        a3.wrapping_add(b3),
        a3.wrapping_sub(b3),
        a2.wrapping_sub(b2),
        a1.wrapping_sub(b1),
        a0.wrapping_sub(b0),
    ];
    for (dst, v) in r.iter_mut().zip(out) {
        *dst = (v >> ROW_SHIFT) as i16;
    }
}

/// One column, returning the eight outputs before clipping.
#[inline]
fn column(block: &[i16; 64], c: usize) -> [i32; 8] {
    let at = |k: usize| i32::from(block[k * 8 + c]);
    let mut a0 = W4.wrapping_mul(at(0) + ((1 << (COL_SHIFT - 1)) / W4));
    let mut a1 = a0;
    let mut a2 = a0;
    let mut a3 = a0;
    let c2 = at(2);
    a0 = a0.wrapping_add(W2.wrapping_mul(c2));
    a1 = a1.wrapping_add(W6.wrapping_mul(c2));
    a2 = a2.wrapping_sub(W6.wrapping_mul(c2));
    a3 = a3.wrapping_sub(W2.wrapping_mul(c2));
    let (c1, c3) = (at(1), at(3));
    let mut b0 = W1.wrapping_mul(c1).wrapping_add(W3.wrapping_mul(c3));
    let mut b1 = W3.wrapping_mul(c1).wrapping_sub(W7.wrapping_mul(c3));
    let mut b2 = W5.wrapping_mul(c1).wrapping_sub(W1.wrapping_mul(c3));
    let mut b3 = W7.wrapping_mul(c1).wrapping_sub(W5.wrapping_mul(c3));
    let c4 = at(4);
    if c4 != 0 {
        a0 = a0.wrapping_add(W4.wrapping_mul(c4));
        a1 = a1.wrapping_sub(W4.wrapping_mul(c4));
        a2 = a2.wrapping_sub(W4.wrapping_mul(c4));
        a3 = a3.wrapping_add(W4.wrapping_mul(c4));
    }
    let c5 = at(5);
    if c5 != 0 {
        b0 = b0.wrapping_add(W5.wrapping_mul(c5));
        b1 = b1.wrapping_sub(W1.wrapping_mul(c5));
        b2 = b2.wrapping_add(W7.wrapping_mul(c5));
        b3 = b3.wrapping_add(W3.wrapping_mul(c5));
    }
    let c6 = at(6);
    if c6 != 0 {
        a0 = a0.wrapping_add(W6.wrapping_mul(c6));
        a1 = a1.wrapping_sub(W2.wrapping_mul(c6));
        a2 = a2.wrapping_add(W2.wrapping_mul(c6));
        a3 = a3.wrapping_sub(W6.wrapping_mul(c6));
    }
    let c7 = at(7);
    if c7 != 0 {
        b0 = b0.wrapping_add(W7.wrapping_mul(c7));
        b1 = b1.wrapping_sub(W5.wrapping_mul(c7));
        b2 = b2.wrapping_add(W3.wrapping_mul(c7));
        b3 = b3.wrapping_sub(W1.wrapping_mul(c7));
    }
    [
        a0.wrapping_add(b0) >> COL_SHIFT,
        a1.wrapping_add(b1) >> COL_SHIFT,
        a2.wrapping_add(b2) >> COL_SHIFT,
        a3.wrapping_add(b3) >> COL_SHIFT,
        a3.wrapping_sub(b3) >> COL_SHIFT,
        a2.wrapping_sub(b2) >> COL_SHIFT,
        a1.wrapping_sub(b1) >> COL_SHIFT,
        a0.wrapping_sub(b0) >> COL_SHIFT,
    ]
}

fn rows(block: &mut [i16; 64]) {
    for r in block.chunks_exact_mut(8) {
        row(r);
    }
}

#[inline]
fn clip(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Replaces `dst` (8 rows `stride` apart) with the transform of `block`.
/// `block` is consumed as scratch.
pub(crate) fn put(block: &mut [i16; 64], dst: &mut [u8], offset: usize, stride: usize) {
    rows(block);
    for c in 0..8 {
        let col = column(block, c);
        for (k, v) in col.into_iter().enumerate() {
            if let Some(p) = dst.get_mut(offset + k * stride + c) {
                *p = clip(v);
            }
        }
    }
}

/// Adds the transform of `block` to `dst` (8 rows `stride` apart),
/// clipping to 0..=255.
pub(crate) fn add(block: &mut [i16; 64], dst: &mut [u8], offset: usize, stride: usize) {
    rows(block);
    for c in 0..8 {
        let col = column(block, c);
        for (k, v) in col.into_iter().enumerate() {
            if let Some(p) = dst.get_mut(offset + k * stride + c) {
                *p = clip(i32::from(*p) + v);
            }
        }
    }
}

/// The transform of `block` as signed values, for the encoder's
/// reconstruction checks.
#[cfg(test)]
pub(crate) fn inverse(block: &[i16; 64]) -> [i32; 64] {
    let mut b = *block;
    rows(&mut b);
    let mut out = [0i32; 64];
    for c in 0..8 {
        for (k, v) in column(&b, c).into_iter().enumerate() {
            out[k * 8 + c] = v;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::inverse;

    /// `basis[k][n]`: `c(k) / 2 * cos((2n + 1) k pi / 16)`.
    fn basis() -> [[f64; 8]; 8] {
        let mut b = [[0f64; 8]; 8];
        for (k, row) in b.iter_mut().enumerate() {
            let c = if k == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
            for (n, v) in row.iter_mut().enumerate() {
                *v = c / 2.0
                    * ((2.0 * n as f64 + 1.0) * k as f64 * std::f64::consts::PI / 16.0).cos();
            }
        }
        b
    }

    /// The reference inverse DCT in double precision, rounded.
    fn reference(block: &[i16; 64], b: &[[f64; 8]; 8]) -> [i32; 64] {
        let mut tmp = [0f64; 64];
        for v in 0..8 {
            for x in 0..8 {
                tmp[v * 8 + x] = (0..8).map(|u| f64::from(block[v * 8 + u]) * b[u][x]).sum();
            }
        }
        let mut out = [0i32; 64];
        for y in 0..8 {
            for x in 0..8 {
                let s: f64 = (0..8).map(|v| tmp[v * 8 + x] * b[v][y]).sum();
                out[y * 8 + x] = s.round() as i32;
            }
        }
        out
    }

    /// The reference forward DCT in double precision, rounded and clipped.
    fn forward(pixels: &[f64; 64], b: &[[f64; 8]; 8]) -> [i16; 64] {
        let mut tmp = [0f64; 64];
        for y in 0..8 {
            for u in 0..8 {
                tmp[y * 8 + u] = (0..8).map(|x| pixels[y * 8 + x] * b[u][x]).sum();
            }
        }
        let mut out = [0i16; 64];
        for v in 0..8 {
            for u in 0..8 {
                let s: f64 = (0..8).map(|y| tmp[y * 8 + u] * b[v][y]).sum();
                out[v * 8 + u] = s.round().clamp(-2048.0, 2047.0) as i16;
            }
        }
        out
    }

    /// A small deterministic generator for the accuracy test.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        fn range(&mut self, low: i32, high: i32) -> i32 {
            low + (self.next() % (high - low + 1) as u32) as i32
        }
    }

    /// IEEE 1180-1990: for random blocks in [-L, H], the peak error is at
    /// most 1, the per-sample mean square error at most 0.06, the overall
    /// at most 0.02, and the per-sample mean error at most 0.015.
    #[test]
    fn meets_ieee_1180_accuracy() {
        let b = basis();
        for (low, high) in [(-256i32, 255i32), (-5, 5), (-300, 300)] {
            for sign in [1i32, -1] {
                let mut rng = Lcg(0x1180 + low.unsigned_abs() as u64);
                let trials = 10_000;
                let mut sq = [0f64; 64];
                let mut mean = [0f64; 64];
                let mut peak = 0;
                for _ in 0..trials {
                    // Random spatial samples, forward transformed in double
                    // precision and rounded, as the standard specifies.
                    let mut pixels = [0f64; 64];
                    for p in pixels.iter_mut() {
                        *p = f64::from(rng.range(low, high) * sign);
                    }
                    let coefs = forward(&pixels, &b);
                    let want = reference(&coefs, &b).map(|v| v.clamp(-256, 255));
                    let got = inverse(&coefs).map(|v| v.clamp(-256, 255));
                    for i in 0..64 {
                        let e = got[i] - want[i];
                        peak = peak.max(e.abs());
                        sq[i] += f64::from(e * e);
                        mean[i] += f64::from(e);
                    }
                }
                let n = f64::from(trials);
                assert!(peak <= 1, "peak error {peak}");
                let worst_sq = sq.iter().fold(0f64, |a, &b| a.max(b / n));
                let total_sq = sq.iter().sum::<f64>() / (n * 64.0);
                let worst_mean = mean.iter().fold(0f64, |a, &b| a.max((b / n).abs()));
                let total_mean = (mean.iter().sum::<f64>() / (n * 64.0)).abs();
                assert!(worst_sq <= 0.06, "per-sample mse {worst_sq}");
                assert!(total_sq <= 0.02, "overall mse {total_sq}");
                assert!(worst_mean <= 0.015, "per-sample mean {worst_mean}");
                assert!(total_mean <= 0.0015, "overall mean {total_mean}");
            }
        }
    }

    #[test]
    fn a_dc_block_is_flat() {
        let mut block = [0i16; 64];
        block[0] = 64;
        let out = inverse(&block);
        assert!(out.iter().all(|&v| v == 8), "{out:?}");
    }
}
