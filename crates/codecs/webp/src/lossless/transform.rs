//! The four VP8L transforms (RFC 9649 section 4) and their inverses.
//!
//! The forward directions are here too, for the encoder: each forward
//! function is the exact inverse of the decoder's step.

use cutlass_decode::CodecError;

use super::decode::{add_pixels, subsample};
use crate::limits::pixel_count;

pub(crate) enum Transform {
    /// Prediction modes in the green byte of each `2^bits` tile.
    Predictor {
        bits: u32,
        data: Vec<u32>,
        width: u32,
    },
    /// Color transform multipliers per `2^bits` tile.
    Color {
        bits: u32,
        data: Vec<u32>,
        width: u32,
    },
    SubtractGreen,
    /// Palette, and `bits` > 0 when indices are bundled into one pixel.
    ColorIndexing {
        bits: u32,
        palette: Vec<u32>,
        width: u32,
    },
}

/// Undo one transform. `pixels` is the image as the following transforms
/// left it.
pub(crate) fn inverse(
    transform: &Transform,
    mut pixels: Vec<u32>,
    height: u32,
) -> Result<Vec<u32>, CodecError> {
    match transform {
        Transform::Predictor { bits, data, width } => {
            inverse_predictor(&mut pixels, *width as usize, *bits, data);
            Ok(pixels)
        }
        Transform::Color { bits, data, width } => {
            inverse_color(&mut pixels, *width as usize, *bits, data);
            Ok(pixels)
        }
        Transform::SubtractGreen => {
            for pixel in &mut pixels {
                *pixel = add_green(*pixel);
            }
            Ok(pixels)
        }
        Transform::ColorIndexing {
            bits,
            palette,
            width,
        } => expand_palette(&pixels, *width, height, *bits, palette),
    }
}

#[inline]
fn add_green(pixel: u32) -> u32 {
    let green = (pixel >> 8) & 0xff;
    let red_blue = (pixel & 0x00ff_00ff).wrapping_add((green << 16) | green);
    (pixel & 0xff00_ff00) | (red_blue & 0x00ff_00ff)
}

#[cfg(test)]
fn subtract_green(pixel: u32) -> u32 {
    let green = (pixel >> 8) & 0xff;
    let red = ((pixel >> 16).wrapping_sub(green)) & 0xff;
    let blue = (pixel.wrapping_sub(green)) & 0xff;
    (pixel & 0xff00_ff00) | (red << 16) | blue
}

#[inline]
fn average2(a: u32, b: u32) -> u32 {
    (((a ^ b) & 0xfefe_fefe) >> 1) + (a & b)
}

#[inline]
fn channel(pixel: u32, shift: u32) -> i32 {
    ((pixel >> shift) & 0xff) as i32
}

/// Select (RFC 9649 section 4.1): `top` when the left gradient is no
/// smaller than the top gradient.
#[inline]
fn select(top: u32, left: u32, top_left: u32) -> u32 {
    let mut score = 0i32;
    for shift in [24, 16, 8, 0] {
        let t = channel(top, shift);
        let l = channel(left, shift);
        let c = channel(top_left, shift);
        score += (l - c).abs() - (t - c).abs();
    }
    if score <= 0 { top } else { left }
}

#[inline]
fn clamp_add_subtract_full(left: u32, top: u32, top_left: u32) -> u32 {
    let mut out = 0u32;
    for shift in [24, 16, 8, 0] {
        let value = channel(left, shift) + channel(top, shift) - channel(top_left, shift);
        out |= (value.clamp(0, 255) as u32) << shift;
    }
    out
}

#[inline]
fn clamp_add_subtract_half(left: u32, top: u32, top_left: u32) -> u32 {
    let average = average2(left, top);
    let mut out = 0u32;
    for shift in [24, 16, 8, 0] {
        let a = channel(average, shift);
        let b = channel(top_left, shift);
        // C division: truncation toward zero.
        let value = a + (a - b) / 2;
        out |= (value.clamp(0, 255) as u32) << shift;
    }
    out
}

/// Prediction of mode `mode` from the left, top, top-left and top-right
/// neighbors. Modes 14 and 15 fit in the four bits the format reads and
/// predict opaque black, as mode 0 does (libwebp behaves the same).
#[inline]
pub(crate) fn predict(mode: u32, left: u32, top: u32, top_left: u32, top_right: u32) -> u32 {
    match mode {
        1 => left,
        2 => top,
        3 => top_right,
        4 => top_left,
        5 => average2(average2(left, top_right), top),
        6 => average2(left, top_left),
        7 => average2(left, top),
        8 => average2(top_left, top),
        9 => average2(top, top_right),
        10 => average2(average2(left, top_left), average2(top, top_right)),
        11 => select(top, left, top_left),
        12 => clamp_add_subtract_full(left, top, top_left),
        13 => clamp_add_subtract_half(left, top, top_left),
        _ => 0xff00_0000,
    }
}

fn inverse_predictor(pixels: &mut [u32], width: usize, bits: u32, modes: &[u32]) {
    if width == 0 || pixels.is_empty() {
        return;
    }
    let tiles_wide = subsample(width as u32, bits) as usize;
    let height = pixels.len() / width;
    pixels[0] = add_pixels(pixels[0], 0xff00_0000);
    for x in 1..width {
        pixels[x] = add_pixels(pixels[x], pixels[x - 1]);
    }
    for y in 1..height {
        let row = y * width;
        let up = row - width;
        pixels[row] = add_pixels(pixels[row], pixels[up]);
        let tile_row = (y >> bits) * tiles_wide;
        for x in 1..width {
            let mode = (modes[tile_row + (x >> bits)] >> 8) & 0xf;
            let at = row + x;
            // The top-right of the last column is the first pixel of this row.
            let top_right = pixels[up + x + 1];
            let guess = predict(
                mode,
                pixels[at - 1],
                pixels[up + x],
                pixels[up + x - 1],
                top_right,
            );
            pixels[at] = add_pixels(pixels[at], guess);
        }
    }
}

/// `(color_pred * color) >> 5` with both as signed bytes.
#[inline]
pub(crate) fn color_delta(multiplier: u8, color: u8) -> i32 {
    (i32::from(multiplier as i8) * i32::from(color as i8)) >> 5
}

fn inverse_color(pixels: &mut [u32], width: usize, bits: u32, multipliers: &[u32]) {
    if width == 0 {
        return;
    }
    let tiles_wide = subsample(width as u32, bits) as usize;
    for (index, pixel) in pixels.iter_mut().enumerate() {
        let (y, x) = (index / width, index % width);
        let code = multipliers[(y >> bits) * tiles_wide + (x >> bits)];
        let green_to_red = (code & 0xff) as u8;
        let green_to_blue = ((code >> 8) & 0xff) as u8;
        let red_to_blue = ((code >> 16) & 0xff) as u8;
        let argb = *pixel;
        let green = ((argb >> 8) & 0xff) as u8;
        let mut red = ((argb >> 16) & 0xff) as i32;
        let mut blue = (argb & 0xff) as i32;
        red = (red + color_delta(green_to_red, green)) & 0xff;
        blue += color_delta(green_to_blue, green);
        blue += color_delta(red_to_blue, red as u8);
        blue &= 0xff;
        *pixel = (argb & 0xff00_ff00) | ((red as u32) << 16) | blue as u32;
    }
}

fn expand_palette(
    pixels: &[u32],
    width: u32,
    height: u32,
    bits: u32,
    palette: &[u32],
) -> Result<Vec<u32>, CodecError> {
    let mut table = [0u32; 256];
    for (slot, &color) in table.iter_mut().zip(palette) {
        *slot = color;
    }
    let count = pixel_count(width, height)?;
    if bits == 0 {
        return Ok(pixels
            .iter()
            .map(|&pixel| table[((pixel >> 8) & 0xff) as usize])
            .collect());
    }
    let mut out = Vec::with_capacity(count);
    let coded_width = subsample(width, bits) as usize;
    let per_pixel = 8 >> bits;
    let per_byte = 1usize << bits;
    let mask = (1u32 << per_pixel) - 1;
    for row in pixels.chunks_exact(coded_width).take(height as usize) {
        for x in 0..width as usize {
            let packed = (row[x / per_byte] >> 8) & 0xff;
            let shift = (x % per_byte) as u32 * per_pixel;
            out.push(table[((packed >> shift) & mask) as usize]);
        }
    }
    if out.len() != count {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "vp8l color indexing",
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{add_green, color_delta, predict, subtract_green};

    #[test]
    fn subtract_green_round_trips() {
        for pixel in [0u32, 0xff00_ff00, 0x8012_3456, 0x01ff_00fe, 0xffff_ffff] {
            assert_eq!(add_green(subtract_green(pixel)), pixel);
        }
    }

    #[test]
    fn color_delta_is_a_signed_product_shifted_by_five() {
        assert_eq!(color_delta(0x20, 0x20), 32);
        assert_eq!(color_delta(0xff, 0x40), -2);
        assert_eq!(color_delta(0x80, 0x80), 512);
        assert_eq!(color_delta(0x7f, 0x81), -505);
    }

    #[test]
    fn half_gradient_truncates_toward_zero() {
        // a = (10 + 13) / 2 = 11, b = 14: 11 + (11 - 14) / 2 = 11 - 1 = 10.
        assert_eq!(predict(13, 10, 13, 14, 0) & 0xff, 10);
        // a = 11, b = 8: 11 + 3 / 2 = 12.
        assert_eq!(predict(13, 10, 13, 8, 0) & 0xff, 12);
    }

    #[test]
    fn select_prefers_top_on_a_tie() {
        let top = 0x0000_0010;
        let left = 0x0000_0020;
        // |left - tl| = 8, |top - tl| = 8.
        assert_eq!(predict(11, left, top, 0x0000_0018, 0), top);
        // |left - tl| = 16, |top - tl| = 0: left is further, pick left.
        assert_eq!(predict(11, left, top, 0x0000_0010, 0), left);
    }

    #[test]
    fn unused_modes_predict_black() {
        assert_eq!(predict(0, 1, 2, 3, 4), 0xff00_0000);
        assert_eq!(predict(14, 1, 2, 3, 4), 0xff00_0000);
        assert_eq!(predict(15, 1, 2, 3, 4), 0xff00_0000);
    }
}
