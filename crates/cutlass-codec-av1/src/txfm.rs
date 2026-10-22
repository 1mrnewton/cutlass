//! Inverse transforms and dequantization.
//!
//! Lossless 4×4 uses the Walsh–Hadamard process. Other sizes are not wired
//! yet; callers reject them before reaching this module.

use crate::error::DecodeError;
use crate::math::clip1;
use crate::quant::{
    AC_QLOOKUP_8, AC_QLOOKUP_10, AC_QLOOKUP_12, DC_QLOOKUP_8, DC_QLOOKUP_10, DC_QLOOKUP_12,
};

pub fn dc_quant(bit_depth: u8, qindex: i32) -> Result<i32, DecodeError> {
    lookup(
        bit_depth,
        qindex,
        &DC_QLOOKUP_8,
        &DC_QLOOKUP_10,
        &DC_QLOOKUP_12,
    )
}

pub fn ac_quant(bit_depth: u8, qindex: i32) -> Result<i32, DecodeError> {
    lookup(
        bit_depth,
        qindex,
        &AC_QLOOKUP_8,
        &AC_QLOOKUP_10,
        &AC_QLOOKUP_12,
    )
}

fn lookup(
    bit_depth: u8,
    qindex: i32,
    q8: &[i16; 256],
    q10: &[i16; 256],
    q12: &[i16; 256],
) -> Result<i32, DecodeError> {
    let q = qindex.clamp(0, 255) as usize;
    let table = match bit_depth {
        8 => q8,
        10 => q10,
        12 => q12,
        _ => return Err(DecodeError::malformed("quantizer bit depth")),
    };
    Ok(i32::from(table[q]))
}

/// Dequantize, inverse-WHT, and add a lossless 4×4 into `samples`.
///
/// `quant` is row-major and already signed. The quantizer indexes are the
/// plane's DC and AC quantizers (`get_dc_quant` / `get_ac_quant`).
pub fn add_lossless_wht4x4(
    samples: &mut [u16],
    stride: usize,
    x: usize,
    y: usize,
    quant: &[i32; 16],
    dc_q: i32,
    ac_q: i32,
    bit_depth: u8,
) -> Result<(), DecodeError> {
    if !(8..=12).contains(&bit_depth) {
        return Err(DecodeError::malformed("WHT bit depth"));
    }
    let mut dequant = [0i32; 16];
    let lim = (1i32 << (7 + bit_depth)) - 1;
    let lo = -(1i32 << (7 + bit_depth));
    for row in 0..4 {
        for col in 0..4 {
            let q = if row == 0 && col == 0 { dc_q } else { ac_q };
            let dq = i64::from(quant[row * 4 + col]) * i64::from(q);
            let sign = if dq < 0 { -1i64 } else { 1 };
            let mag = dq.abs() & 0xFF_FFFF;
            let dq2 = sign * mag;
            dequant[row * 4 + col] = dq2.clamp(i64::from(lo), i64::from(lim)) as i32;
        }
    }
    let mut residual = [0i32; 16];
    for row in 0..4 {
        let mut t = [
            dequant[row * 4],
            dequant[row * 4 + 1],
            dequant[row * 4 + 2],
            dequant[row * 4 + 3],
        ];
        wht4(&mut t, 2);
        residual[row * 4..row * 4 + 4].copy_from_slice(&t);
    }
    let col_range = i32::from(bit_depth) + 6;
    let col_range = col_range.max(16);
    let clip_hi = (1i32 << (col_range - 1)) - 1;
    let clip_lo = -(1i32 << (col_range - 1));
    for value in &mut residual {
        *value = (*value).clamp(clip_lo, clip_hi);
    }
    for col in 0..4 {
        let mut t = [
            residual[col],
            residual[4 + col],
            residual[8 + col],
            residual[12 + col],
        ];
        wht4(&mut t, 0);
        for row in 0..4 {
            residual[row * 4 + col] = t[row];
        }
    }
    // Lossless residuals must fit in a signed integer of BitDepth + 1 bits.
    let res_hi = (1i32 << bit_depth) - 1;
    let res_lo = -(1i32 << bit_depth);
    for row in 0..4 {
        for col in 0..4 {
            let r = residual[row * 4 + col];
            if r < res_lo || r > res_hi {
                return Err(DecodeError::malformed(
                    "lossless residual is outside the bit-depth range",
                ));
            }
            let idx = (y + row)
                .checked_mul(stride)
                .and_then(|i| i.checked_add(x + col))
                .ok_or_else(|| DecodeError::malformed("residual address"))?;
            let sample = samples
                .get(idx)
                .copied()
                .ok_or_else(|| DecodeError::malformed("residual write is outside the plane"))?;
            samples[idx] = clip1(bit_depth, i32::from(sample) + r) as u16;
        }
    }
    Ok(())
}

/// In-place inverse Walsh–Hadamard of length 4. `shift` is the pre-scale.
fn wht4(t: &mut [i32; 4], shift: u32) {
    let mut a = t[0] >> shift;
    let mut c = t[1] >> shift;
    let mut d = t[2] >> shift;
    let mut b = t[3] >> shift;
    a += c;
    d -= b;
    let e = (a - d) >> 1;
    b = e - b;
    c = e - c;
    a -= b;
    d += c;
    t[0] = a;
    t[1] = b;
    t[2] = c;
    t[3] = d;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lossless_dc_level_one_adds_one_at_the_origin() {
        // qindex 0 → quantizer 4. Level +1 dequantizes to 4, and the WHT
        // with the row shift of 2 yields residual 1 at (0, 0) only.
        let mut samples = [128u16; 16];
        let mut quant = [0i32; 16];
        quant[0] = 1;
        add_lossless_wht4x4(&mut samples, 4, 0, 0, &quant, 4, 4, 8).unwrap();
        assert_eq!(samples[0], 129);
        assert!(samples[1..].iter().all(|&s| s == 128));
    }

    #[test]
    fn qindex_zero_is_four_at_eight_bits() {
        assert_eq!(dc_quant(8, 0).unwrap(), 4);
        assert_eq!(ac_quant(8, 0).unwrap(), 4);
        assert_eq!(dc_quant(10, 0).unwrap(), 4);
        assert_eq!(dc_quant(12, 0).unwrap(), 4);
    }
}
