//! Separable DCT from ITU-T T.81.
//!
//! One dimension applies `C(u) / 2`. Two dimensions are the factor `1/4`
//! in the specification. Samples are rounded only after the second pass.

use std::sync::OnceLock;

fn cos_table() -> &'static [[f64; 8]; 8] {
    static TABLE: OnceLock<[[f64; 8]; 8]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [[0.0; 8]; 8];
        for (x, row) in table.iter_mut().enumerate() {
            for (u, value) in row.iter_mut().enumerate() {
                let angle = (2.0 * x as f64 + 1.0) * u as f64 * std::f64::consts::PI / 16.0;
                *value = angle.cos();
            }
        }
        table
    })
}

fn scale(u: usize) -> f64 {
    if u == 0 {
        std::f64::consts::FRAC_1_SQRT_2
    } else {
        1.0
    }
}

fn idct1(input: &[f64; 8]) -> [f64; 8] {
    let cos = cos_table();
    let mut out = [0.0; 8];
    for x in 0..8 {
        let mut sum = 0.0;
        for u in 0..8 {
            sum += scale(u) * input[u] * cos[x][u];
        }
        out[x] = sum / 2.0;
    }
    out
}

fn fdct1(input: &[f64; 8]) -> [f64; 8] {
    let cos = cos_table();
    let mut out = [0.0; 8];
    for u in 0..8 {
        let mut sum = 0.0;
        for x in 0..8 {
            sum += input[x] * cos[x][u];
        }
        out[u] = scale(u) * sum / 2.0;
    }
    out
}

pub(crate) fn idct_u8(coeffs: &[i32; 64]) -> [u8; 64] {
    if coeffs[1..].iter().all(|coeff| *coeff == 0) {
        let level = (f64::from(coeffs[0]) / 8.0).round() as i32 + 128;
        return [clamp_u8(level); 64];
    }
    let mut columns = [0.0f64; 64];
    for x in 0..8 {
        let mut col = [0.0; 8];
        for y in 0..8 {
            col[y] = f64::from(coeffs[y * 8 + x]);
        }
        let out = idct1(&col);
        for y in 0..8 {
            columns[y * 8 + x] = out[y];
        }
    }
    let mut samples = [0u8; 64];
    for y in 0..8 {
        let mut row = [0.0; 8];
        for x in 0..8 {
            row[x] = columns[y * 8 + x];
        }
        let out = idct1(&row);
        for x in 0..8 {
            let level = out[x].round() as i32 + 128;
            samples[y * 8 + x] = clamp_u8(level);
        }
    }
    samples
}

pub(crate) fn fdct(samples: &[f64; 64]) -> [i32; 64] {
    let mut rows = [0.0f64; 64];
    for y in 0..8 {
        let mut row = [0.0; 8];
        for x in 0..8 {
            row[x] = samples[y * 8 + x];
        }
        let out = fdct1(&row);
        for x in 0..8 {
            rows[y * 8 + x] = out[x];
        }
    }
    let mut coeffs = [0i32; 64];
    for x in 0..8 {
        let mut col = [0.0; 8];
        for y in 0..8 {
            col[y] = rows[y * 8 + x];
        }
        let out = fdct1(&col);
        for y in 0..8 {
            coeffs[y * 8 + x] = out[y].round() as i32;
        }
    }
    coeffs
}

pub(crate) fn quantize(coeff: i32, step: u16) -> i32 {
    let step = i32::from(step);
    if step <= 0 {
        return coeff;
    }
    (f64::from(coeff) / f64::from(step)).round() as i32
}

pub(crate) fn dequant_block(coeffs: &[i32], quant: &[u16; 64]) -> Result<[i32; 64], ()> {
    let mut block = [0i32; 64];
    for index in 0..64 {
        let coeff = i64::from(*coeffs.get(index).ok_or(())?);
        let product = coeff * i64::from(quant[index]);
        block[index] = i32::try_from(product).map_err(|_| ())?;
    }
    Ok(block)
}

fn clamp_u8(value: i32) -> u8 {
    u8::try_from(value.clamp(0, 255)).unwrap_or(0)
}
