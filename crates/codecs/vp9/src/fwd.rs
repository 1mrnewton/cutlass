//! Forward transforms for the profile-0 encoder.
//!
//! The Walsh-Hadamard matches libvpx `vp9_fwht4x4_c`, including the factor
//! of four that the lossless dequantizer and the inverse row shift cancel.
//! The DCT matches libvpx `vpx_fdct4x4_c` and `vpx_fdct8x8_c`, with each
//! butterfly stored in 16 bits the way the 8-bit inverse reads it.

const COSPI_4: i64 = 16069;
const COSPI_8: i64 = 15137;
const COSPI_12: i64 = 13623;
const COSPI_16: i64 = 11585;
const COSPI_20: i64 = 9102;
const COSPI_24: i64 = 6270;
const COSPI_28: i64 = 3196;

fn round_shift(value: i64) -> i32 {
    let shifted = (value + 8192) >> 14;
    i32::from(shifted as i16)
}

/// 4x4 reversible Walsh-Hadamard. Output entries are multiples of four.
pub fn fwht4x4(input: &[i16], stride: usize, output: &mut [i32]) {
    let mut temp = [0i32; 16];
    for col in 0..4 {
        let a = i32::from(input[col]);
        let b = i32::from(input[stride + col]);
        let c = i32::from(input[2 * stride + col]);
        let d = i32::from(input[3 * stride + col]);
        let (a1, c1, d1, b1) = wht_row(a, b, c, d);
        temp[col] = a1;
        temp[4 + col] = c1;
        temp[8 + col] = d1;
        temp[12 + col] = b1;
    }
    for row in 0..4 {
        let base = row * 4;
        let (a1, c1, d1, b1) = wht_row(temp[base], temp[base + 1], temp[base + 2], temp[base + 3]);
        output[base] = a1 * 4;
        output[base + 1] = c1 * 4;
        output[base + 2] = d1 * 4;
        output[base + 3] = b1 * 4;
    }
}

fn wht_row(a: i32, b: i32, c: i32, d: i32) -> (i32, i32, i32, i32) {
    let mut a1 = a + b;
    let mut d1 = d - c;
    let e1 = (a1 - d1) >> 1;
    let b1 = e1 - b;
    let c1 = e1 - c;
    a1 -= c1;
    d1 += b1;
    (a1, c1, d1, b1)
}

/// 4x4 forward DCT. `output` is 16 coefficients in raster order.
pub fn fdct4x4(input: &[i16], stride: usize, output: &mut [i32]) {
    let mut intermediate = [0i32; 16];
    for col in 0..4 {
        let mut a = i64::from(input[col]) * 16;
        let b = i64::from(input[stride + col]) * 16;
        let c = i64::from(input[2 * stride + col]) * 16;
        let d = i64::from(input[3 * stride + col]) * 16;
        if col == 0 && a != 0 {
            a += 1;
        }
        let (o0, o1, o2, o3) = fdct4(a, b, c, d);
        let at = col * 4;
        intermediate[at] = o0;
        intermediate[at + 1] = o1;
        intermediate[at + 2] = o2;
        intermediate[at + 3] = o3;
    }
    for row in 0..4 {
        let a = i64::from(intermediate[row]);
        let b = i64::from(intermediate[4 + row]);
        let c = i64::from(intermediate[8 + row]);
        let d = i64::from(intermediate[12 + row]);
        let (o0, o1, o2, o3) = fdct4(a, b, c, d);
        let at = row * 4;
        output[at] = (o0 + 1) >> 2;
        output[at + 1] = (o1 + 1) >> 2;
        output[at + 2] = (o2 + 1) >> 2;
        output[at + 3] = (o3 + 1) >> 2;
    }
}

fn fdct4(a: i64, b: i64, c: i64, d: i64) -> (i32, i32, i32, i32) {
    let step0 = a + d;
    let step1 = b + c;
    let step2 = b - c;
    let step3 = a - d;
    let o0 = round_shift((step0 + step1) * COSPI_16);
    let o2 = round_shift((step0 - step1) * COSPI_16);
    let o1 = round_shift(step2 * COSPI_24 + step3 * COSPI_8);
    let o3 = round_shift(-step2 * COSPI_8 + step3 * COSPI_24);
    (o0, o1, o2, o3)
}

/// 8x8 forward DCT. `output` is 64 coefficients in raster order.
pub fn fdct8x8(input: &[i16], stride: usize, output: &mut [i32]) {
    let mut intermediate = [0i32; 64];
    for col in 0..8 {
        let mut column = [0i64; 8];
        for row in 0..8 {
            column[row] = i64::from(input[row * stride + col]) * 4;
        }
        let transformed = fdct8(&column);
        let at = col * 8;
        intermediate[at..at + 8].copy_from_slice(&transformed);
    }
    for row in 0..8 {
        let mut column = [0i64; 8];
        for (freq, slot) in column.iter_mut().enumerate() {
            *slot = i64::from(intermediate[freq * 8 + row]);
        }
        let transformed = fdct8(&column);
        let at = row * 8;
        for (coef, value) in transformed.iter().enumerate() {
            output[at + coef] = value / 2;
        }
    }
}

fn fdct8(input: &[i64; 8]) -> [i32; 8] {
    let s0 = input[0] + input[7];
    let s1 = input[1] + input[6];
    let s2 = input[2] + input[5];
    let s3 = input[3] + input[4];
    let s4 = input[3] - input[4];
    let s5 = input[2] - input[5];
    let s6 = input[1] - input[6];
    let s7 = input[0] - input[7];
    let (even0, even2, even4, even6) = fdct4(s0, s1, s2, s3);
    let t0 = round_shift((s6 - s5) * COSPI_16);
    let t1 = round_shift((s6 + s5) * COSPI_16);
    let x0 = s4 + i64::from(t0);
    let x1 = s4 - i64::from(t0);
    let x2 = s7 - i64::from(t1);
    let x3 = s7 + i64::from(t1);
    let odd1 = round_shift(x0 * COSPI_28 + x3 * COSPI_4);
    let odd5 = round_shift(x1 * COSPI_12 + x2 * COSPI_20);
    let odd3 = round_shift(x2 * COSPI_12 + x1 * -COSPI_20);
    let odd7 = round_shift(x3 * COSPI_28 + x0 * -COSPI_4);
    [even0, odd1, even2, odd3, even4, odd5, even6, odd7]
}

/// Nearest token for a forward coefficient. The decoder reconstructs `token * dq`.
pub fn quantize(coef: i32, dq: i32) -> i32 {
    if coef == 0 || dq <= 0 {
        return 0;
    }
    let step = u32::try_from(dq).unwrap_or(1);
    let mut token = (coef.unsigned_abs() + step / 2) / step;
    while token > 0 && token.saturating_mul(step) > 32_767 {
        token -= 1;
    }
    let signed = i32::try_from(token).unwrap_or(0);
    if coef < 0 { -signed } else { signed }
}

#[cfg(test)]
mod tests {
    use super::{fdct4x4, fdct8x8, fwht4x4, quantize};
    use crate::txfm::{iht_add, iwht4x4_add};

    #[test]
    fn fwht_then_iwht_restores_the_residual() {
        let residual: [i16; 16] = [7, -3, 12, 0, 4, 9, -8, 2, 1, -15, 6, 11, -2, 5, 0, -4];
        let mut coef = [0i32; 16];
        fwht4x4(&residual, 4, &mut coef);
        for value in coef {
            assert_eq!(value % 4, 0, "forward scale {value}");
        }
        let tokens: Vec<i32> = coef.iter().map(|value| value / 4).collect();
        let mut recon_coef = [0i32; 16];
        for (slot, token) in recon_coef.iter_mut().zip(tokens) {
            *slot = token * 4;
        }
        let mut dest = [128u8; 16];
        iwht4x4_add(&recon_coef, &mut dest, 4);
        for (got, src) in dest.iter().zip(residual) {
            assert_eq!(i32::from(*got) - 128, i32::from(src));
        }
    }

    #[test]
    fn fdct_round_trip_stays_within_a_few_levels() {
        let mut residual = [0i16; 64];
        for (i, slot) in residual.iter_mut().enumerate() {
            *slot = i16::try_from((i * 17) % 41).unwrap_or(0) - 20;
        }
        check_dct(4, &residual);
        check_dct(8, &residual);
    }

    fn check_dct(n: usize, residual: &[i16]) {
        let mut coef = vec![0i32; n * n];
        if n == 4 {
            fdct4x4(residual, n, &mut coef);
        } else {
            fdct8x8(residual, n, &mut coef);
        }
        let mut tokens = vec![0i32; n * n];
        for (token, value) in tokens.iter_mut().zip(&coef) {
            *token = quantize(*value, 4);
        }
        let mut dequant = vec![0i32; n * n];
        for (slot, token) in dequant.iter_mut().zip(&tokens) {
            *slot = token * 4;
        }
        let mut dest = vec![128u8; n * n];
        iht_add(&dequant, &mut dest, n, n, 0);
        let mut worst = 0i32;
        for (got, src) in dest.iter().zip(residual) {
            let err = (i32::from(*got) - 128 - i32::from(*src)).abs();
            worst = worst.max(err);
        }
        assert!(worst <= 6, "{n}x{n} error {worst}");
    }
}
