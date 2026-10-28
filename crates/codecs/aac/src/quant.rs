//! Inverse quantization and the quarter-step powers of two used for scale factors.
//!
//! ISO/IEC 14496-3 reconstructs a spectral line as `sign(q) · |q|^(4/3) · 2^((sf−100)/4)`.
//! `sf` is the running scale-factor value in `0..=255`, not a PCM bit depth.

use std::sync::OnceLock;

pub const MAX_QUANT: i32 = 8191;

pub fn pow2_quarter(exp: i32) -> f32 {
    if exp == 0 {
        return 1.0;
    }
    let negative = exp < 0;
    let magnitude = exp.unsigned_abs();
    let frac = match magnitude & 3 {
        0 => 1.0,
        1 => 1.189_207_1,
        2 => std::f32::consts::SQRT_2,
        _ => 1.681_792_9,
    };
    let shift = magnitude >> 2;
    if shift >= 128 {
        return if negative { 0.0 } else { f32::INFINITY };
    }
    let base = f32::from_bits((shift + 127) << 23);
    let value = frac * base;
    if negative { 1.0 / value } else { value }
}

fn pow43_table() -> &'static [f32] {
    static TABLE: OnceLock<Vec<f32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut values = Vec::with_capacity(8192);
        values.push(0.0);
        for index in 1..8192 {
            values.push((index as f32).powf(4.0 / 3.0));
        }
        values
    })
}

pub fn pow43(quant: i32) -> f32 {
    let magnitude = quant.unsigned_abs();
    let table = pow43_table();
    let value = if let Some(entry) = table.get(magnitude as usize) {
        *entry
    } else {
        (magnitude as f32).powf(4.0 / 3.0)
    };
    if quant < 0 { -value } else { value }
}

pub fn dequant(quant: i32, scale_factor: i32) -> f32 {
    pow43(quant) * pow2_quarter(scale_factor - 100)
}

pub fn quantize(sample: f32, scale_factor: i32) -> i32 {
    let scale = pow2_quarter(scale_factor - 100);
    if sample == 0.0 || scale == 0.0 || !scale.is_finite() {
        return 0;
    }
    let magnitude = (sample.abs() / scale).powf(0.75);
    let units = if !magnitude.is_finite() || magnitude >= MAX_QUANT as f32 {
        MAX_QUANT
    } else {
        let rounded = magnitude.round();
        if rounded >= MAX_QUANT as f32 {
            MAX_QUANT
        } else {
            rounded as i32
        }
    };
    if sample < 0.0 { -units } else { units }
}

#[cfg(test)]
mod tests {
    use super::{dequant, pow2_quarter, pow43, quantize};

    #[test]
    fn powers_of_two_land_on_quarter_steps() {
        assert!((pow2_quarter(0) - 1.0).abs() < 1.0e-6);
        assert!((pow2_quarter(4) - 2.0).abs() < 1.0e-5);
        assert!((pow2_quarter(-4) - 0.5).abs() < 1.0e-5);
        assert!((pow2_quarter(1) - 2.0f32.powf(0.25)).abs() < 1.0e-5);
    }

    #[test]
    fn inverse_quant_of_one_at_unity_scale_is_one() {
        assert!((pow43(0)).abs() < 1.0e-8);
        assert!((pow43(1) - 1.0).abs() < 1.0e-6);
        assert!((pow43(-2) + pow43(2)).abs() < 1.0e-5);
        assert!((dequant(1, 100) - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn quantize_preserves_silence_sign_and_the_legal_range() {
        assert_eq!(quantize(0.0, 100), 0);
        assert!(quantize(1.0, 100) >= 1);
        assert!(quantize(-1.0, 100) <= -1);
        for sf in [60, 100, 180] {
            for value in [0.0, 0.25, -0.25, 1.0, -1.0] {
                let q = quantize(value, sf);
                assert!(q.abs() <= super::MAX_QUANT);
                if q != 0 {
                    assert_eq!(q.signum(), value.signum() as i32);
                }
                assert!(dequant(q, sf).is_finite());
            }
        }
    }
}
