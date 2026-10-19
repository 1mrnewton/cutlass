//! CPU RGBA → I420 (planar YUV 4:2:0) conversion.
//!
//! The renderer hands the export loop packed RGBA, but hardware video encoders
//! take YUV. Apple's VideoToolbox accepts BGRA directly (see the Apple
//! backend); MediaCodec does not, so the Android backend converts here first.
//! Kept platform-independent so it's unit-testable on the host.
//!
//! Uses BT.601 limited-range coefficients (the conventional space tagged on
//! H.264 SD/HD streams). Chroma is 2×2 box-averaged.

/// Convert a packed RGBA image to a tightly-packed I420 buffer (`Y` plane, then
/// `U`, then `V`).
///
/// `stride` is the source row length in bytes (≥ `width * 4`). `width` and
/// `height` are rounded down to even for the 4:2:0 chroma grid. Returns a
/// buffer of `w*h + 2*(w/2)*(h/2)` bytes.
// The Android encoder is the only non-test caller; allow dead code on hosts
// where that backend is cfg'd out (the unit tests below still exercise it).
#[cfg_attr(not(target_os = "android"), allow(dead_code))]
pub fn rgba_to_i420(rgba: &[u8], width: u32, height: u32, stride: usize) -> Vec<u8> {
    let w = (width & !1) as usize;
    let h = (height & !1) as usize;
    let cw = w / 2;
    let ch = h / 2;
    let y_size = w * h;
    let c_size = cw * ch;
    let mut out = vec![0u8; y_size + 2 * c_size];

    let (y_plane, uv) = out.split_at_mut(y_size);
    let (u_plane, v_plane) = uv.split_at_mut(c_size);

    let px = |x: usize, y: usize| -> (i32, i32, i32) {
        let base = y * stride + x * 4;
        (
            rgba[base] as i32,
            rgba[base + 1] as i32,
            rgba[base + 2] as i32,
        )
    };

    for y in 0..h {
        for x in 0..w {
            let (r, g, b) = px(x, y);
            // Y = 0.257R + 0.504G + 0.098B + 16 (fixed-point /256).
            let luma = (66 * r + 129 * g + 25 * b + 128) >> 8;
            y_plane[y * w + x] = (luma + 16).clamp(0, 255) as u8;
        }
    }

    for cy in 0..ch {
        for cx in 0..cw {
            // Average the 2×2 RGBA block for one chroma sample.
            let (mut rs, mut gs, mut bs) = (0i32, 0i32, 0i32);
            for dy in 0..2 {
                for dx in 0..2 {
                    let (r, g, b) = px(cx * 2 + dx, cy * 2 + dy);
                    rs += r;
                    gs += g;
                    bs += b;
                }
            }
            let (r, g, b) = (rs / 4, gs / 4, bs / 4);
            // U = -0.148R - 0.291G + 0.439B + 128, V = 0.439R - 0.368G - 0.071B + 128.
            let u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
            let v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
            u_plane[cy * cw + cx] = u.clamp(0, 255) as u8;
            v_plane[cy * cw + cx] = v.clamp(0, 255) as u8;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `width*height` packed RGBA image from a per-pixel closure.
    fn image(width: u32, height: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
        let mut buf = vec![0u8; (width * height * 4) as usize];
        for y in 0..height {
            for x in 0..width {
                let base = ((y * width + x) * 4) as usize;
                buf[base..base + 4].copy_from_slice(&f(x, y));
            }
        }
        buf
    }

    #[test]
    fn plane_sizes_are_correct_for_420() {
        let rgba = image(4, 4, |_, _| [0, 0, 0, 255]);
        let out = rgba_to_i420(&rgba, 4, 4, 16);
        // 4*4 luma + 2 * (2*2) chroma = 16 + 8 = 24.
        assert_eq!(out.len(), 24);
    }

    #[test]
    fn solid_black_and_white_map_to_expected_luma() {
        let black = image(2, 2, |_, _| [0, 0, 0, 255]);
        let white = image(2, 2, |_, _| [255, 255, 255, 255]);
        let yb = rgba_to_i420(&black, 2, 2, 8);
        let yw = rgba_to_i420(&white, 2, 2, 8);
        // Limited-range luma: black ≈ 16, white ≈ 235.
        assert_eq!(yb[0], 16);
        assert!((233..=235).contains(&yw[0]), "white luma was {}", yw[0]);
    }

    #[test]
    fn gray_is_chroma_neutral() {
        let gray = image(2, 2, |_, _| [128, 128, 128, 255]);
        let out = rgba_to_i420(&gray, 2, 2, 8);
        // One U and one V sample, both ≈ 128 (neutral chroma) for a gray frame.
        let u = out[4];
        let v = out[5];
        assert!((127..=129).contains(&u), "U was {u}");
        assert!((127..=129).contains(&v), "V was {v}");
    }

    #[test]
    fn honors_source_stride() {
        // 2×2 image in a buffer padded to a 12-byte stride; padding must be
        // ignored.
        let width = 2;
        let height = 2;
        let stride = 12;
        let mut rgba = vec![7u8; stride * height as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let base = y * stride + x * 4;
                rgba[base..base + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }
        let out = rgba_to_i420(&rgba, width, height, stride);
        assert!((233..=235).contains(&out[0]));
    }
}
