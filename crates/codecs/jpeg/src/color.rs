//! JFIF YCbCr, ITU-T T.871. The factors match the 16-bit fixed form
//! `round(factor * 65536)`.

fn clamp_u8(value: i32) -> u8 {
    u8::try_from(value.clamp(0, 255)).unwrap_or(0)
}

pub(crate) fn ycbcr_to_rgb(y: i32, cb: i32, cr: i32) -> (u8, u8, u8) {
    let cb = cb - 128;
    let cr = cr - 128;
    let red = y + ((91_881 * cr + 32_768) >> 16);
    let green = y + ((-22_554 * cb - 46_802 * cr + 32_768) >> 16);
    let blue = y + ((116_130 * cb + 32_768) >> 16);
    (clamp_u8(red), clamp_u8(green), clamp_u8(blue))
}

pub(crate) fn rgb_to_ycbcr(red: i32, green: i32, blue: i32) -> (u8, u8, u8) {
    let y = (19_595 * red + 38_470 * green + 7_471 * blue + 32_768) >> 16;
    let cb = ((-11_059 * red - 21_709 * green + 32_768 * blue + 32_768) >> 16) + 128;
    let cr = ((32_768 * red - 27_439 * green - 5_329 * blue + 32_768) >> 16) + 128;
    (clamp_u8(y), clamp_u8(cb), clamp_u8(cr))
}
