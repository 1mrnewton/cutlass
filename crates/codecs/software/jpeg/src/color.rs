//! JFIF full-range YCbCr to RGB.

/// JFIF full range, nearest integer, clamped.
///
/// The products are the spec coefficients scaled by 2^16:
/// `1.402`, `0.344136`, `0.714136`, and `1.772`.
pub(crate) fn ycbcr(y: u8, cb: u8, cr: u8) -> [u8; 3] {
    let y = i32::from(y);
    let cb = i32::from(cb) - 128;
    let cr = i32::from(cr) - 128;
    let red = y + mul(cr, 91_881);
    let green = y - mul(cb, 22_553) - mul(cr, 46_802);
    let blue = y + mul(cb, 116_130);
    [byte(red), byte(green), byte(blue)]
}

fn mul(delta: i32, coef: i32) -> i32 {
    let product = i64::from(delta) * i64::from(coef);
    let rounded = div_round(product, 65_536);
    i32::try_from(rounded).expect("a JFIF product fits in i32")
}

fn div_round(value: i64, divisor: i64) -> i64 {
    if value >= 0 {
        (value + divisor / 2) / divisor
    } else {
        -(((-value) + divisor / 2) / divisor)
    }
}

fn byte(value: i32) -> u8 {
    if value <= 0 {
        0
    } else if value >= 255 {
        255
    } else {
        u8::try_from(value).expect("a clamped color fits in a byte")
    }
}

#[cfg(test)]
mod tests {
    use super::ycbcr;

    #[test]
    fn neutral_chroma_keeps_luma() {
        assert_eq!(ycbcr(128, 128, 128), [128, 128, 128]);
        assert_eq!(ycbcr(160, 128, 128), [160, 160, 160]);
    }

    #[test]
    fn a_chroma_step_rounds_to_the_nearest_byte() {
        assert_eq!(ycbcr(128, 130, 128), [128, 127, 132]);
        assert_eq!(ycbcr(10, 20, 30), [0, 117, 0]);
    }
}
