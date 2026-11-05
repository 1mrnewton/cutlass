//! Lossy WebP pictures from 4:2:0 planes to RGBA.
//!
//! Chroma is upsampled with the "fancy" filter: each output pixel takes
//! 9/16 of the nearest chroma sample, 3/16 of each of the two next nearest
//! and 1/16 of the diagonal, computed as two rounded averages. The
//! conversion is BT.601 studio swing with 14-bit fixed-point coefficients.
//! Both follow the reference decoder's arithmetic, so a picture matches
//! `dwebp` and browsers bit for bit.

use cutlass_vp8::Vp8Picture;

/// 1.164 (luma), 1.596 (V to red), 0.391 and 0.813 (U and V to green),
/// 2.018 (U to blue), each times 2^14; the offsets fold in the -16 and
/// -128 biases and the rounding of the final 6-bit shift.
const Y_SCALE: i32 = 19_077;
const V_TO_R: i32 = 26_149;
const U_TO_G: i32 = 6_419;
const V_TO_G: i32 = 13_320;
const U_TO_B: i32 = 33_050;
const R_OFFSET: i32 = -14_234;
const G_OFFSET: i32 = 8_708;
const B_OFFSET: i32 = -17_685;

fn mult_hi(value: u8, coefficient: i32) -> i32 {
    (i32::from(value) * coefficient) >> 8
}

fn clip(value: i32) -> u8 {
    if value & !0x3FFF == 0 {
        u8::try_from(value >> 6).unwrap_or(u8::MAX)
    } else if value < 0 {
        0
    } else {
        255
    }
}

pub(crate) fn yuv_to_rgb(y: u8, u: u8, v: u8) -> [u8; 3] {
    let luma = mult_hi(y, Y_SCALE);
    [
        clip(luma + mult_hi(v, V_TO_R) + R_OFFSET),
        clip(luma - mult_hi(u, U_TO_G) - mult_hi(v, V_TO_G) + G_OFFSET),
        clip(luma + mult_hi(u, U_TO_B) + B_OFFSET),
    ]
}

/// The chroma for each pixel of a row between chroma rows `near` and
/// `far` (`near` is the closer one), for one plane.
fn upsample_row(near: &[u8], far: &[u8], width: usize) -> Vec<u8> {
    let mut out = vec![0_u8; width];
    let edge = |a: u8, b: u8| u8::try_from((3 * u32::from(a) + u32::from(b) + 2) >> 2).unwrap_or(0);
    out[0] = edge(near[0], far[0]);
    let pairs = (width - 1) >> 1;
    for x in 1..=pairs {
        let (tl, t) = (u32::from(near[x - 1]), u32::from(near[x]));
        let (l, c) = (u32::from(far[x - 1]), u32::from(far[x]));
        let sum = tl + t + l + c + 8;
        let diag_far = (sum + 2 * (t + l)) >> 3;
        let diag_near = (sum + 2 * (tl + c)) >> 3;
        out[2 * x - 1] = u8::try_from((diag_far + tl) >> 1).unwrap_or(0);
        out[2 * x] = u8::try_from((diag_near + t) >> 1).unwrap_or(0);
    }
    if width & 1 == 0 {
        out[width - 1] = edge(near[pairs], far[pairs]);
    }
    out
}

/// Convert a decoded frame to straight RGBA, with `alpha` or opaque.
pub(crate) fn to_rgba(picture: &Vp8Picture, alpha: Option<&[u8]>) -> Vec<u8> {
    let width = usize::try_from(picture.width).unwrap_or(0);
    let height = usize::try_from(picture.height).unwrap_or(0);
    let chroma_width = picture.chroma_width();
    let chroma = |plane: &'_ [u8], row: usize| -> Vec<u8> {
        plane
            .get(row * chroma_width..(row + 1) * chroma_width)
            .map_or_else(|| vec![128; chroma_width], <[u8]>::to_vec)
    };
    let mut out = vec![0_u8; width * height * 4];
    for y in 0..height {
        // Row 0 uses chroma row 0 alone; then each odd row is nearest the
        // chroma row above and each even row the one below, except a last
        // odd row, which has no row below.
        let (near, far) = if y == 0 {
            (0, 0)
        } else if y % 2 == 1 {
            let above = (y - 1) / 2;
            let below = if y + 1 < height { above + 1 } else { above };
            (above, below)
        } else {
            (y / 2, y / 2 - 1)
        };
        let u = upsample_row(&chroma(&picture.u, near), &chroma(&picture.u, far), width);
        let v = upsample_row(&chroma(&picture.v, near), &chroma(&picture.v, far), width);
        for x in 0..width {
            let at = y * width + x;
            let luma = picture.y.get(at).copied().unwrap_or(0);
            let [r, g, b] = yuv_to_rgb(luma, u[x], v[x]);
            let a = alpha
                .and_then(|plane| plane.get(at))
                .copied()
                .unwrap_or(255);
            out[at * 4..at * 4 + 4].copy_from_slice(&[r, g, b, a]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn studio_swing_extremes_and_grey() {
        assert_eq!(yuv_to_rgb(16, 128, 128), [0, 0, 0]);
        assert_eq!(yuv_to_rgb(235, 128, 128), [255, 255, 255]);
        let [r, g, b] = yuv_to_rgb(126, 128, 128);
        assert!(r == g && g == b && (127..=129).contains(&r));
        // Saturated red in BT.601 studio swing.
        let [r, g, b] = yuv_to_rgb(81, 90, 240);
        assert!(r >= 250 && g <= 3 && b <= 3, "{r} {g} {b}");
    }

    #[test]
    fn fancy_upsampling_weights_the_nearest_sample() {
        // Uniform chroma stays uniform.
        assert_eq!(upsample_row(&[77, 77], &[77, 77], 4), [77; 4]);
        // One step horizontally: the pixel next to the step leans 3:1.
        let row = upsample_row(&[0, 160], &[0, 160], 4);
        assert_eq!(row, [0, 40, 120, 160]);
        // Vertically, the near row weighs 3, the far row 1.
        assert_eq!(upsample_row(&[160], &[0], 1), [120]);
        // Odd width: the last pixel pairs with the last chroma sample.
        assert_eq!(upsample_row(&[10, 20], &[10, 20], 3).len(), 3);
    }
}
