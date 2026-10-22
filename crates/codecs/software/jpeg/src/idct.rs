//! Inverse DCT. T.81 Annex A, in fixed point.
//!
//! `WEIGHT[x][u]` is `C(u) * cos((2x+1) * u * pi / 16)` scaled by 2^16 and
//! rounded to the nearest integer. The sum is divided by `4 * 2^32`, half
//! away from zero. For ordinary coefficient magnitudes the result is within
//! 1 of the real-valued sum. The level shift of 128 is applied by the caller.

use crate::error::Error;

const SCALE: i128 = 1 << 16;

/// Natural index of each zigzag position. T.81 Figure A.6.
const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

const WEIGHT: [[i64; 8]; 8] = [
    [46341, 64277, 60547, 54491, 46341, 36410, 25080, 12785],
    [46341, 54491, 25080, -12785, -46341, -64277, -60547, -36410],
    [46341, 36410, -25080, -64277, -46341, 12785, 60547, 54491],
    [46341, 12785, -60547, -36410, 46341, 54491, -25080, -64277],
    [46341, -12785, -60547, 36410, 46341, -54491, -25080, 64277],
    [46341, -36410, -25080, 64277, -46341, -12785, 60547, -54491],
    [46341, -54491, 25080, 12785, -46341, 64277, -60547, 36410],
    [46341, -64277, 60547, -54491, 46341, -36410, 25080, -12785],
];

/// Zigzag coefficients times the quant table, in natural order.
pub(crate) fn dequant(zz: &[i32; 64], quant: &[u16; 64]) -> Result<[i32; 64], Error> {
    let mut natural = [0i32; 64];
    for k in 0..64 {
        let product = i64::from(zz[k])
            .checked_mul(i64::from(quant[k]))
            .ok_or(Error::Inconsistent("a coefficient does not fit"))?;
        let at = ZIGZAG[k];
        natural[at] = i32::try_from(product)
            .map_err(|_| Error::Inconsistent("a coefficient does not fit"))?;
    }
    Ok(natural)
}

/// Annex A. The values are centered on zero, before the level shift.
pub(crate) fn idct(coef: &[i32; 64]) -> Result<[i32; 64], Error> {
    let mut out = [0i32; 64];
    let divisor = 4 * SCALE * SCALE;
    for y in 0..8 {
        for x in 0..8 {
            let mut sum = 0i128;
            for v in 0..8 {
                for u in 0..8 {
                    let sample = coef[v * 8 + u];
                    if sample == 0 {
                        continue;
                    }
                    let term = i128::from(sample)
                        .checked_mul(i128::from(WEIGHT[y][v]))
                        .and_then(|value| value.checked_mul(i128::from(WEIGHT[x][u])))
                        .ok_or(Error::Inconsistent("a coefficient does not fit"))?;
                    sum = sum
                        .checked_add(term)
                        .ok_or(Error::Inconsistent("a coefficient does not fit"))?;
                }
            }
            out[y * 8 + x] = narrow(div_round(sum, divisor)?)?;
        }
    }
    Ok(out)
}

/// Add 128 and clamp. T.81 A.3.1.
pub(crate) fn level(centered: i32) -> u8 {
    let shifted = centered.saturating_add(128);
    if shifted <= 0 {
        0
    } else if shifted >= 255 {
        255
    } else {
        u8::try_from(shifted).expect("a sample between 1 and 254 fits in a byte")
    }
}

fn div_round(value: i128, divisor: i128) -> Result<i128, Error> {
    let half = divisor / 2;
    if value >= 0 {
        let shifted = value
            .checked_add(half)
            .ok_or(Error::Inconsistent("a coefficient does not fit"))?;
        return Ok(shifted / divisor);
    }
    let flipped = value
        .checked_neg()
        .ok_or(Error::Inconsistent("a coefficient does not fit"))?;
    let rounded = flipped
        .checked_add(half)
        .ok_or(Error::Inconsistent("a coefficient does not fit"))?
        / divisor;
    rounded
        .checked_neg()
        .ok_or(Error::Inconsistent("a coefficient does not fit"))
}

fn narrow(value: i128) -> Result<i32, Error> {
    if value > i128::from(i32::MAX) {
        return Ok(i32::MAX);
    }
    if value < i128::from(i32::MIN) {
        return Ok(i32::MIN);
    }
    i32::try_from(value).map_err(|_| Error::Inconsistent("a coefficient does not fit"))
}

#[cfg(test)]
mod tests {
    use super::{ZIGZAG, dequant, idct, level};

    fn centered(dc: i32) -> i32 {
        let mut coef = [0i32; 64];
        coef[0] = dc;
        idct(&coef).expect("dc idct")[0]
    }

    #[test]
    fn zigzag_visits_each_frequency_once() {
        let mut seen = [false; 64];
        for index in ZIGZAG {
            seen[index] = true;
        }
        assert!(seen.iter().all(|slot| *slot));
    }

    #[test]
    fn a_dc_coefficient_is_one_eighth() {
        assert_eq!(centered(0), 0);
        assert_eq!(centered(8), 1);
        assert_eq!(centered(16), 2);
        assert_eq!(centered(-16), -2);
        assert_eq!(centered(1016), 127);
        assert_eq!(centered(-416), -52);
        assert_eq!(level(centered(0)), 128);
        assert_eq!(level(centered(16)), 130);
    }

    #[test]
    fn one_horizontal_ac_repeats_down_the_block() {
        let mut coef = [0i32; 64];
        coef[1] = 80;
        let out = idct(&coef).expect("ac idct");
        let row = [14, 12, 8, 3, -3, -8, -12, -14];
        for y in 0..8 {
            assert_eq!(&out[y * 8..y * 8 + 8], &row);
        }
    }

    #[test]
    fn dequant_follows_zigzag_order() {
        let mut zz = [0i32; 64];
        zz[1] = 80;
        let quant = [1u16; 64];
        let natural = dequant(&zz, &quant).expect("dequant");
        assert_eq!(natural[1], 80);
        assert_eq!(natural[0], 0);
    }
}
