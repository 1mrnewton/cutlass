//! Bilinear sampling of an RGBA picture, as the GPU compositor draws it.
//!
//! Pixel `i` spans `i..i + 1`, so its center is `i + 0.5`. A point is the
//! four nearest pixel centers, weighted by distance, with the edge pixel
//! repeated past the edge. A point on a pixel center is that pixel exactly.

use cutlass_picture::rgba_len;

/// The picture at the continuous point `(u, v)`.
pub(crate) fn bilinear(src: &[u8], width: u32, height: u32, u: f64, v: f64) -> Option<[u8; 4]> {
    if width == 0 || height == 0 || !u.is_finite() || !v.is_finite() {
        return None;
    }
    let (x, y) = (u - 0.5, v - 0.5);
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let last_x = f64::from(width - 1);
    let last_y = f64::from(height - 1);
    let column = |at: f64| at.clamp(0.0, last_x) as usize;
    let row = |at: f64| at.clamp(0.0, last_y) as usize;
    let (left, right) = (column(x0), column(x0 + 1.0));
    let (top, bottom) = (row(y0), row(y0 + 1.0));
    let stride = usize::try_from(width).ok()?.checked_mul(4)?;
    let pixel = |cx: usize, cy: usize| {
        let at = cy.checked_mul(stride)?.checked_add(cx.checked_mul(4)?)?;
        src.get(at..at + 4)
    };
    let (a, b) = (pixel(left, top)?, pixel(right, top)?);
    let (c, d) = (pixel(left, bottom)?, pixel(right, bottom)?);
    let mut out = [0_u8; 4];
    for (channel, slot) in out.iter_mut().enumerate() {
        let upper = f64::from(a[channel]) + (f64::from(b[channel]) - f64::from(a[channel])) * fx;
        let lower = f64::from(c[channel]) + (f64::from(d[channel]) - f64::from(c[channel])) * fx;
        let value = upper + (lower - upper) * fy;
        *slot = value.round().clamp(0.0, 255.0) as u8;
    }
    Some(out)
}

/// `src` scaled to `dst_w` by `dst_h`, each output pixel center mapped to
/// the matching point of the source.
pub(crate) fn scale(src_w: u32, src_h: u32, src: &[u8], dst_w: u32, dst_h: u32) -> Option<Vec<u8>> {
    if src.len() != rgba_len(src_w, src_h)? || dst_w == 0 || dst_h == 0 {
        return None;
    }
    if (src_w, src_h) == (dst_w, dst_h) {
        return Some(src.to_vec());
    }
    let mut out = Vec::with_capacity(rgba_len(dst_w, dst_h)?);
    let kx = f64::from(src_w) / f64::from(dst_w);
    let ky = f64::from(src_h) / f64::from(dst_h);
    for y in 0..dst_h {
        let v = (f64::from(y) + 0.5) * ky;
        for x in 0..dst_w {
            let u = (f64::from(x) + 0.5) * kx;
            out.extend_from_slice(&bilinear(src, src_w, src_h, u, v)?);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pixel_center_is_that_pixel() {
        let src = [10, 0, 0, 255, 30, 0, 0, 255];
        assert_eq!(bilinear(&src, 2, 1, 0.5, 0.5), Some([10, 0, 0, 255]));
        assert_eq!(bilinear(&src, 2, 1, 1.5, 0.5), Some([30, 0, 0, 255]));
        assert_eq!(scale(2, 1, &src, 2, 1).unwrap(), src);
    }

    #[test]
    fn between_centers_blends_and_past_the_edge_repeats() {
        let src = [10, 0, 0, 255, 30, 0, 0, 255];
        assert_eq!(bilinear(&src, 2, 1, 1.0, 0.5), Some([20, 0, 0, 255]));
        assert_eq!(bilinear(&src, 2, 1, 0.0, 0.0), Some([10, 0, 0, 255]));
        assert_eq!(bilinear(&src, 2, 1, 2.0, 1.0), Some([30, 0, 0, 255]));
        // Doubling 2 pixels to 4: centers land a quarter in from each.
        let doubled = scale(2, 1, &src, 4, 1).unwrap();
        let reds: Vec<u8> = doubled.chunks(4).map(|pixel| pixel[0]).collect();
        assert_eq!(reds, [10, 15, 25, 30]);
    }
}
