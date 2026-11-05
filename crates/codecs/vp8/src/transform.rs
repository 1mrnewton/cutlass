//! The inverse transforms of RFC 6386 section 14.
//!
//! Both work in place on sixteen coefficients in raster order. The DCT
//! adds its result to a 4x4 prediction and clamps to a byte; the WHT
//! returns the sixteen luma DC values the second-order block carried.

/// `sqrt(2) * cos(pi / 8) - 1` and `sqrt(2) * sin(pi / 8)`, in 16-bit fixed
/// point (section 14.3).
const COS_MINUS_ONE: i32 = 20_091;
const SIN: i32 = 35_468;

/// Inverse Walsh-Hadamard transform of the Y2 block (section 14.3).
///
/// Inputs are 16-bit values, so no stage can overflow.
pub(crate) fn inverse_wht(input: &[i32; 16]) -> [i32; 16] {
    let mut mid = [0_i32; 16];
    for column in 0..4 {
        let a = input[column] + input[12 + column];
        let b = input[4 + column] + input[8 + column];
        let c = input[4 + column] - input[8 + column];
        let d = input[column] - input[12 + column];
        mid[column] = a + b;
        mid[4 + column] = c + d;
        mid[8 + column] = a - b;
        mid[12 + column] = d - c;
    }
    let mut out = [0_i32; 16];
    for row in 0..4 {
        let at = row * 4;
        let a = mid[at] + mid[at + 3];
        let b = mid[at + 1] + mid[at + 2];
        let c = mid[at + 1] - mid[at + 2];
        let d = mid[at] - mid[at + 3];
        out[at] = wrap16((a + b + 3) >> 3);
        out[at + 1] = wrap16((c + d + 3) >> 3);
        out[at + 2] = wrap16((a - b + 3) >> 3);
        out[at + 3] = wrap16((d - c + 3) >> 3);
    }
    out
}

/// One 1-D pass of the inverse DCT over `x0..x3`.
fn idct_1d(x0: i32, x1: i32, x2: i32, x3: i32) -> [i32; 4] {
    let a = x0 + x2;
    let b = x0 - x2;
    let c = ((x1 * SIN) >> 16) - (x3 + ((x3 * COS_MINUS_ONE) >> 16));
    let d = (x1 + ((x1 * COS_MINUS_ONE) >> 16)) + ((x3 * SIN) >> 16);
    [a + d, b + c, b - c, a - d]
}

/// Inverse DCT of `coeffs`, added to the 4x4 block at `at` of `plane`.
///
/// The columns go first, then the rows with `(x + 4) >> 3` rounding.
pub(crate) fn idct_add(coeffs: &[i32; 16], plane: &mut [u8], at: usize, stride: usize) {
    let mut mid = [0_i32; 16];
    for column in 0..4 {
        let out = idct_1d(
            coeffs[column],
            coeffs[4 + column],
            coeffs[8 + column],
            coeffs[12 + column],
        );
        for (row, value) in out.into_iter().enumerate() {
            mid[row * 4 + column] = wrap16(value);
        }
    }
    for row in 0..4 {
        let out = idct_1d(
            mid[row * 4],
            mid[row * 4 + 1],
            mid[row * 4 + 2],
            mid[row * 4 + 3],
        );
        let line = at + row * stride;
        for (column, value) in out.into_iter().enumerate() {
            if let Some(pixel) = plane.get_mut(line + column) {
                *pixel = clamp255(i32::from(*pixel) + ((value + 4) >> 3));
            }
        }
    }
}

/// Keep the low 16 bits, signed: the reference decoder holds coefficients
/// and the transform's middle stage in 16-bit storage, so a hostile stream
/// wraps there instead of overflowing.
pub(crate) fn wrap16(value: i32) -> i32 {
    i32::from(value as i16)
}

pub(crate) fn clamp255(value: i32) -> u8 {
    u8::try_from(value.clamp(0, 255)).unwrap_or(u8::MAX)
}

#[cfg(test)]
mod tests {
    use super::{idct_add, inverse_wht};

    #[test]
    fn a_lone_dc_spreads_evenly() {
        let mut coeffs = [0; 16];
        coeffs[0] = 80;
        let mut block = [100_u8; 16];
        idct_add(&coeffs, &mut block, 0, 4);
        // (80 + 4) >> 3 = 10 on every pixel.
        assert_eq!(block, [110; 16]);
        let mut wht = [0; 16];
        wht[0] = 77;
        // (77 + 3) >> 3 = 10 for each luma block.
        assert_eq!(inverse_wht(&wht), [10; 16]);
    }

    #[test]
    fn the_idct_matches_the_float_transform_within_rounding() {
        // A first-row AC coefficient: a horizontal cosine.
        let mut coeffs = [0; 16];
        coeffs[1] = 64;
        let mut block = [128_u8; 16];
        idct_add(&coeffs, &mut block, 0, 4);
        for row in 0..4 {
            for column in 0..4 {
                let angle = std::f64::consts::PI * (2.0 * column as f64 + 1.0) / 8.0;
                let ideal = 128.0 + 64.0 * std::f64::consts::SQRT_2 * angle.cos() / 8.0;
                let got = f64::from(block[row * 4 + column]);
                assert!((got - ideal).abs() <= 1.0, "{row} {column} {got} {ideal}");
            }
        }
        // Clamping at both ends.
        let mut bright = [250_u8; 16];
        let mut dark = [3_u8; 16];
        let mut big = [0; 16];
        big[0] = 800;
        idct_add(&big, &mut bright, 0, 4);
        big[0] = -800;
        idct_add(&big, &mut dark, 0, 4);
        assert_eq!((bright, dark), ([255; 16], [0; 16]));
    }

    #[test]
    fn the_wht_inverts_the_forward_walsh() {
        // The forward 4x4 Walsh-Hadamard (unnormalized) of a known block,
        // scaled by 8 so the inverse's rounding lands exactly.
        let values: [i32; 16] = [3, -1, 4, 1, -5, 9, -2, 6, 5, -3, 5, 8, -9, 7, 9, -3];
        let h = |v: [i32; 4]| {
            [
                v[0] + v[1] + v[2] + v[3],
                v[0] + v[1] - v[2] - v[3],
                v[0] - v[1] - v[2] + v[3],
                v[0] - v[1] + v[2] - v[3],
            ]
        };
        let mut rows = [0; 16];
        for r in 0..4 {
            let out = h([
                values[r * 4],
                values[r * 4 + 1],
                values[r * 4 + 2],
                values[r * 4 + 3],
            ]);
            rows[r * 4..r * 4 + 4].copy_from_slice(&out);
        }
        let mut forward = [0; 16];
        for c in 0..4 {
            let out = h([rows[c], rows[4 + c], rows[8 + c], rows[12 + c]]);
            for r in 0..4 {
                forward[r * 4 + c] = out[r] / 2;
            }
        }
        assert_eq!(inverse_wht(&forward), values);
    }
}
