//! The 4×4 transforms of RFC 6386 Section 14, and forward versions for
//! the encoder.
//!
//! The inverse transforms keep their first pass in 16 bits, as the
//! reference decoder does, so overflowing input wraps the same way.

const C1: i32 = 20_091; // cos(pi/8) * sqrt(2) - 1, in 1/65536
const S1: i32 = 35_468; // sin(pi/8) * sqrt(2), in 1/65536

#[inline]
fn wrap(value: i32) -> i16 {
    value as i16
}

/// Inverse Walsh-Hadamard transform of the Y2 block (Section 14.3).
#[must_use]
pub fn iwht(input: &[i16; 16]) -> [i16; 16] {
    let mut mid = [0i16; 16];
    for i in 0..4 {
        let a1 = i32::from(input[i]) + i32::from(input[12 + i]);
        let b1 = i32::from(input[4 + i]) + i32::from(input[8 + i]);
        let c1 = i32::from(input[4 + i]) - i32::from(input[8 + i]);
        let d1 = i32::from(input[i]) - i32::from(input[12 + i]);
        mid[i] = wrap(a1 + b1);
        mid[4 + i] = wrap(c1 + d1);
        mid[8 + i] = wrap(a1 - b1);
        mid[12 + i] = wrap(d1 - c1);
    }
    let mut out = [0i16; 16];
    for i in 0..4 {
        let row = &mid[4 * i..4 * i + 4];
        let a1 = i32::from(row[0]) + i32::from(row[3]);
        let b1 = i32::from(row[1]) + i32::from(row[2]);
        let c1 = i32::from(row[1]) - i32::from(row[2]);
        let d1 = i32::from(row[0]) - i32::from(row[3]);
        out[4 * i] = wrap((a1 + b1 + 3) >> 3);
        out[4 * i + 1] = wrap((c1 + d1 + 3) >> 3);
        out[4 * i + 2] = wrap((a1 - b1 + 3) >> 3);
        out[4 * i + 3] = wrap((d1 - c1 + 3) >> 3);
    }
    out
}

#[inline]
fn mul_c1(x: i32) -> i32 {
    x + ((x * C1) >> 16)
}

#[inline]
fn mul_s1(x: i32) -> i32 {
    (x * S1) >> 16
}

/// Inverse DCT of one block, added to the prediction already in `dst`
/// (Section 14.4).
pub fn idct_add(coeffs: &[i16; 16], dst: &mut [u8], at: usize, stride: usize) {
    let mut mid = [0i16; 16];
    for i in 0..4 {
        let x0 = i32::from(coeffs[i]);
        let x1 = i32::from(coeffs[4 + i]);
        let x2 = i32::from(coeffs[8 + i]);
        let x3 = i32::from(coeffs[12 + i]);
        let a1 = x0 + x2;
        let b1 = x0 - x2;
        let c1 = mul_s1(x1) - mul_c1(x3);
        let d1 = mul_c1(x1) + mul_s1(x3);
        mid[i] = wrap(a1 + d1);
        mid[12 + i] = wrap(a1 - d1);
        mid[4 + i] = wrap(b1 + c1);
        mid[8 + i] = wrap(b1 - c1);
    }
    for i in 0..4 {
        let x0 = i32::from(mid[4 * i]);
        let x1 = i32::from(mid[4 * i + 1]);
        let x2 = i32::from(mid[4 * i + 2]);
        let x3 = i32::from(mid[4 * i + 3]);
        let a1 = x0 + x2;
        let b1 = x0 - x2;
        let c1 = mul_s1(x1) - mul_c1(x3);
        let d1 = mul_c1(x1) + mul_s1(x3);
        let start = at + i * stride;
        let Some(row) = dst.get_mut(start..start + 4) else {
            return;
        };
        let add = |pixel: &mut u8, delta: i32| {
            *pixel = (i32::from(*pixel) + delta).clamp(0, 255) as u8;
        };
        add(&mut row[0], (a1 + d1 + 4) >> 3);
        add(&mut row[3], (a1 - d1 + 4) >> 3);
        add(&mut row[1], (b1 + c1 + 4) >> 3);
        add(&mut row[2], (b1 - c1 + 4) >> 3);
    }
}

/// Adds a block that has only a DC coefficient. Equal to [`idct_add`]
/// for such a block.
pub fn idct_dc_add(dc: i16, dst: &mut [u8], at: usize, stride: usize) {
    let delta = (i32::from(dc) + 4) >> 3;
    for i in 0..4 {
        let start = at + i * stride;
        if let Some(row) = dst.get_mut(start..start + 4) {
            for pixel in row {
                *pixel = (i32::from(*pixel) + delta).clamp(0, 255) as u8;
            }
        }
    }
}

// Forward DCT basis in Q12, scaled so that `idct_add` inverts it:
// row 0 is cos(pi/4), rows 1..3 are cos((2n+1)k pi/8).
const FDCT: [[i64; 4]; 4] = [
    [2896, 2896, 2896, 2896],
    [3784, 1567, -1567, -3784],
    [2896, -2896, -2896, 2896],
    [1567, -3784, 3784, -1567],
];

/// Forward DCT of a residual block, the inverse of [`idct_add`] up to
/// rounding.
#[must_use]
pub fn fdct(residual: &[i16; 16]) -> [i16; 16] {
    let mut rows = [0i64; 16];
    for r in 0..4 {
        for (k, basis) in FDCT.iter().enumerate() {
            rows[4 * r + k] = (0..4)
                .map(|n| basis[n] * i64::from(residual[4 * r + n]))
                .sum();
        }
    }
    let mut out = [0i16; 16];
    for c in 0..4 {
        for (k, basis) in FDCT.iter().enumerate() {
            let sum: i64 = (0..4).map(|n| basis[n] * rows[4 * n + c]).sum();
            // Two Q12 passes. The basis already carries the factor of 2
            // the inverse takes away (each inverse pass gains 2, then /8).
            let value = (sum + (1 << 23)) >> 24;
            out[4 * k + c] = value.clamp(-32_768, 32_767) as i16;
        }
    }
    out
}

/// Forward Walsh-Hadamard transform of the 16 luma DC values, the
/// inverse of [`iwht`] up to rounding.
#[must_use]
pub fn fwht(dc: &[i16; 16]) -> [i16; 16] {
    const H: [[i32; 4]; 4] = [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, -1, 1], [1, -1, 1, -1]];
    let mut mid = [0i32; 16];
    for r in 0..4 {
        for (k, basis) in H.iter().enumerate() {
            mid[4 * r + k] = (0..4).map(|n| basis[n] * i32::from(dc[4 * r + n])).sum();
        }
    }
    let mut out = [0i16; 16];
    for c in 0..4 {
        for (k, basis) in H.iter().enumerate() {
            let sum: i32 = (0..4).map(|n| basis[n] * mid[4 * n + c]).sum();
            let value = if sum >= 0 {
                (sum + 1) >> 1
            } else {
                -((-sum + 1) >> 1)
            };
            out[4 * k + c] = value.clamp(-32_768, 32_767) as i16;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{fdct, fwht, idct_add, idct_dc_add, iwht};

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (self.0 >> 33) as u32
        }
    }

    #[test]
    fn a_dc_only_block_matches_the_shortcut() {
        for dc in [-2048i16, -100, -5, -4, -3, 0, 3, 4, 5, 77, 2047] {
            let mut coeffs = [0i16; 16];
            coeffs[0] = dc;
            let mut full = vec![128u8; 16];
            let mut short = full.clone();
            idct_add(&coeffs, &mut full, 0, 4);
            idct_dc_add(dc, &mut short, 0, 4);
            assert_eq!(full, short, "dc {dc}");
        }
    }

    #[test]
    fn the_forward_transforms_invert() {
        let mut rng = Lcg(7);
        let mut worst = 0;
        for _ in 0..2000 {
            let mut residual = [0i16; 16];
            for value in &mut residual {
                *value = (rng.next() % 511) as i16 - 255;
            }
            let coeffs = fdct(&residual);
            let mut pixels = vec![128u8; 16];
            idct_add(&coeffs, &mut pixels, 0, 4);
            for (pixel, value) in pixels.iter().zip(residual) {
                let want = (128 + i32::from(value)).clamp(0, 255);
                worst = worst.max((i32::from(*pixel) - want).abs());
            }
        }
        assert!(worst <= 1, "round trip error {worst}");

        for _ in 0..2000 {
            let mut dc = [0i16; 16];
            for value in &mut dc {
                *value = (rng.next() % 8001) as i16 - 4000;
            }
            let back = iwht(&fwht(&dc));
            for (a, b) in back.iter().zip(dc) {
                assert!((i32::from(*a) - i32::from(b)).abs() <= 1);
            }
        }
    }

    #[test]
    fn the_inverse_walsh_spreads_a_dc() {
        let mut input = [0i16; 16];
        input[0] = 80;
        assert_eq!(iwht(&input), [10; 16]);
    }
}
