//! Forward transforms, the encoder's side of [`super::dsp`], in libwebp's
//! integer arithmetic so that its quantizer settings carry over.

/// The forward DCT of `src - pred` over one 4x4 block, raster order.
pub(crate) fn fdct(
    src: &[u8],
    src_at: usize,
    src_stride: usize,
    pred: &[u8],
    pred_at: usize,
    pred_stride: usize,
) -> [i16; 16] {
    let mut tmp = [0i32; 16];
    for row in 0..4 {
        let s = src_at + row * src_stride;
        let p = pred_at + row * pred_stride;
        let d: [i32; 4] = std::array::from_fn(|i| i32::from(src[s + i]) - i32::from(pred[p + i]));
        let a0 = d[0] + d[3];
        let a1 = d[1] + d[2];
        let a2 = d[1] - d[2];
        let a3 = d[0] - d[3];
        tmp[row * 4] = (a0 + a1) * 8;
        tmp[row * 4 + 1] = (a2 * 2217 + a3 * 5352 + 1812) >> 9;
        tmp[row * 4 + 2] = (a0 - a1) * 8;
        tmp[row * 4 + 3] = (a3 * 2217 - a2 * 5352 + 937) >> 9;
    }
    let mut out = [0i16; 16];
    for col in 0..4 {
        let a0 = tmp[col] + tmp[12 + col];
        let a1 = tmp[4 + col] + tmp[8 + col];
        let a2 = tmp[4 + col] - tmp[8 + col];
        let a3 = tmp[col] - tmp[12 + col];
        out[col] = ((a0 + a1 + 7) >> 4) as i16;
        out[4 + col] = (((a2 * 2217 + a3 * 5352 + 12_000) >> 16) + i32::from(a3 != 0)) as i16;
        out[8 + col] = ((a0 - a1 + 7) >> 4) as i16;
        out[12 + col] = ((a3 * 2217 - a2 * 5352 + 51_000) >> 16) as i16;
    }
    out
}

/// The forward Walsh-Hadamard transform of the sixteen luma DCs, in block
/// raster order.
pub(crate) fn fwht(dc: &[i16; 16]) -> [i16; 16] {
    let mut tmp = [0i32; 16];
    for row in 0..4 {
        let d: [i32; 4] = std::array::from_fn(|i| i32::from(dc[row * 4 + i]));
        let a0 = d[0] + d[2];
        let a1 = d[1] + d[3];
        let a2 = d[1] - d[3];
        let a3 = d[0] - d[2];
        tmp[row * 4] = a0 + a1;
        tmp[row * 4 + 1] = a3 + a2;
        tmp[row * 4 + 2] = a3 - a2;
        tmp[row * 4 + 3] = a0 - a1;
    }
    let mut out = [0i16; 16];
    for col in 0..4 {
        let a0 = tmp[col] + tmp[8 + col];
        let a1 = tmp[4 + col] + tmp[12 + col];
        let a2 = tmp[4 + col] - tmp[12 + col];
        let a3 = tmp[col] - tmp[8 + col];
        out[col] = ((a0 + a1) >> 1) as i16;
        out[4 + col] = ((a3 + a2) >> 1) as i16;
        out[8 + col] = ((a3 - a2) >> 1) as i16;
        out[12 + col] = ((a0 - a1) >> 1) as i16;
    }
    out
}

/// Luma weights of the texture distortion, low frequencies first.
const WEIGHT_Y: [i32; 16] = [38, 32, 20, 9, 32, 28, 17, 7, 20, 17, 10, 4, 9, 7, 4, 2];

/// Weighted sum of the Hadamard magnitudes of one 4x4 block.
fn texture(buf: &[u8], at: usize, stride: usize) -> i32 {
    let mut tmp = [0i32; 16];
    for row in 0..4 {
        let s = at + row * stride;
        let v: [i32; 4] = std::array::from_fn(|i| i32::from(buf[s + i]));
        let a0 = v[0] + v[2];
        let a1 = v[1] + v[3];
        let a2 = v[1] - v[3];
        let a3 = v[0] - v[2];
        tmp[row * 4] = a0 + a1;
        tmp[row * 4 + 1] = a3 + a2;
        tmp[row * 4 + 2] = a3 - a2;
        tmp[row * 4 + 3] = a0 - a1;
    }
    let mut sum = 0;
    for col in 0..4 {
        let a0 = tmp[col] + tmp[8 + col];
        let a1 = tmp[4 + col] + tmp[12 + col];
        let a2 = tmp[4 + col] - tmp[12 + col];
        let a3 = tmp[col] - tmp[8 + col];
        sum += WEIGHT_Y[col] * (a0 + a1).abs();
        sum += WEIGHT_Y[4 + col] * (a3 + a2).abs();
        sum += WEIGHT_Y[8 + col] * (a3 - a2).abs();
        sum += WEIGHT_Y[12 + col] * (a0 - a1).abs();
    }
    sum
}

/// How much texture one 4x4 block lost or gained, libwebp's `Disto4x4`.
pub(crate) fn texture_distortion(
    a: &[u8],
    a_at: usize,
    a_stride: usize,
    b: &[u8],
    b_at: usize,
    b_stride: usize,
) -> i32 {
    (texture(b, b_at, b_stride) - texture(a, a_at, a_stride)).abs() >> 5
}

/// Sum of squared differences of a `w x h` area.
pub(crate) fn sse(
    a: &[u8],
    a_at: usize,
    a_stride: usize,
    b: &[u8],
    b_at: usize,
    b_stride: usize,
    (w, h): (usize, usize),
) -> u32 {
    let mut sum = 0u32;
    for row in 0..h {
        let x = &a[a_at + row * a_stride..a_at + row * a_stride + w];
        let y = &b[b_at + row * b_stride..b_at + row * b_stride + w];
        for (&p, &q) in x.iter().zip(y) {
            let d = i32::from(p) - i32::from(q);
            sum += (d * d) as u32;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::{fdct, fwht};
    use crate::lossy::dsp::{add_idct, inverse_wht};

    #[test]
    fn the_inverse_dct_undoes_the_forward_one() {
        let mut seed = 99u32;
        for _ in 0..200 {
            let src: Vec<u8> = (0..16)
                .map(|_| {
                    seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
                    (seed >> 24) as u8
                })
                .collect();
            let pred = vec![128u8; 16];
            let coeffs = fdct(&src, 0, 4, &pred, 0, 4);
            let mut out = pred.clone();
            add_idct(&coeffs, &mut out, 0, 4);
            for (a, b) in out.iter().zip(&src) {
                assert!(a.abs_diff(*b) <= 1, "{out:?} {src:?}");
            }
        }
    }

    #[test]
    fn the_inverse_wht_undoes_the_forward_one() {
        let mut seed = 5u32;
        for _ in 0..200 {
            let dc: [i16; 16] = std::array::from_fn(|_| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
                ((seed >> 20) as i16 & 0x7ff) - 1024
            });
            let back = inverse_wht(&fwht(&dc));
            for (a, b) in back.iter().zip(&dc) {
                assert!((a - b).abs() <= 1, "{back:?} {dc:?}");
            }
        }
    }
}
