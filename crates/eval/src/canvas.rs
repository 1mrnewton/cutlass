//! The project canvas and a segment's canvas.
//!
//! `canvas_config.background` is the color under every layer. An absent,
//! empty, or unreadable string leaves the clear frame. A segment canvas is a
//! different object: `canvas_color` fills that segment's letterbox,
//! `canvas_blur` covers the canvas with a blurred copy of the picture, and
//! `canvas_image` covers it with the named still. The fitted picture is then
//! drawn over that backdrop. A blur of zero is a sharp cover. The radius is
//! `blur` times the short side, clamped to that side.

use cutlass_model::MaterialCanvas;
use cutlass_picture::{CpuBytes, rgba_len};

/// A segment canvas resolved for one canvas size.
#[derive(Clone, Debug)]
pub(crate) enum Backdrop {
    /// No segment canvas. The project color shows through.
    None,
    /// A solid letterbox.
    Color([u8; 4]),
    /// Box radius, in pixels, of a cover-scaled copy.
    Blur(u32),
    /// A still already scaled to cover the canvas.
    Image(CpuBytes),
}

/// RGBA from a canvas color string.
///
/// Six hex digits, with an optional `#`, are opaque RGB. Eight digits are
/// the packed `0xAARRGGBB` form, including a `#` or `0x` prefix.
pub(crate) fn canvas_rgba(text: &str) -> Option<[u8; 4]> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    if let Some(packed) = cutlass_generated::parse_packed(text) {
        return Some(cutlass_generated::rgba_from_packed(packed));
    }
    let hex = text.strip_prefix('#').unwrap_or(text);
    if hex.len() != 6 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    Some([
        u8::from_str_radix(&hex[0..2], 16).ok()?,
        u8::from_str_radix(&hex[2..4], 16).ok()?,
        u8::from_str_radix(&hex[4..6], 16).ok()?,
        255,
    ])
}

/// The project canvas, or nothing when it does not name a color.
pub(crate) fn project_canvas(text: Option<&str>, width: u32, height: u32) -> Option<CpuBytes> {
    let pixel = canvas_rgba(text?)?;
    fill(width, height, pixel)
}

/// The backdrop a segment canvas paints before its picture.
pub(crate) fn segment_backdrop(canvas: &MaterialCanvas, width: u32, height: u32) -> Backdrop {
    match canvas.kind.as_str() {
        "canvas_blur" => {
            blur_backdrop(canvas.blur, width, height).unwrap_or_else(|| color_backdrop(canvas))
        }
        _ => color_backdrop(canvas),
    }
}

/// `canvas_image` once the still has been opened. A missing still uses the color.
pub(crate) fn image_backdrop(canvas: &MaterialCanvas, image: Option<CpuBytes>) -> Backdrop {
    if canvas.kind != "canvas_image" {
        return Backdrop::None;
    }
    match image {
        Some(bytes) => Backdrop::Image(bytes),
        None => color_backdrop(canvas),
    }
}

fn color_backdrop(canvas: &MaterialCanvas) -> Backdrop {
    match canvas_rgba(&canvas.color) {
        Some(pixel) => Backdrop::Color(pixel),
        None => Backdrop::None,
    }
}

fn blur_backdrop(blur: f64, width: u32, height: u32) -> Option<Backdrop> {
    if !blur.is_finite() || blur < 0.0 || width == 0 || height == 0 {
        return None;
    }
    let short = width.min(height);
    let fraction = blur.min(1.0);
    let radius = round_u32(fraction * f64::from(short))?.min(short);
    Some(Backdrop::Blur(radius))
}

/// Draw `top` over the backdrop. `sharp` is the picture before it was fitted.
pub(crate) fn over_backdrop(
    top: CpuBytes,
    sharp: Option<&CpuBytes>,
    backdrop: &Backdrop,
) -> CpuBytes {
    let width = top.width();
    let height = top.height();
    let Some(mut under) = backdrop_pixels(width, height, sharp, backdrop) else {
        return top;
    };
    over_into(&mut under, top.rgba());
    CpuBytes::new(width, height, under).unwrap_or(top)
}

fn backdrop_pixels(
    width: u32,
    height: u32,
    sharp: Option<&CpuBytes>,
    backdrop: &Backdrop,
) -> Option<Vec<u8>> {
    match backdrop {
        Backdrop::None => None,
        Backdrop::Color(pixel) => fill(width, height, *pixel).map(CpuBytes::into_rgba),
        Backdrop::Image(image) if image.width() == width && image.height() == height => {
            Some(image.rgba().to_vec())
        }
        Backdrop::Image(_) => None,
        Backdrop::Blur(radius) => {
            let mut covered = cover(sharp?, width, height)?.into_rgba();
            box_blur(&mut covered, width, height, *radius)?;
            Some(covered)
        }
    }
}

/// Scale `picture` until it covers the canvas, then keep the center.
pub(crate) fn cover(picture: &CpuBytes, canvas_w: u32, canvas_h: u32) -> Option<CpuBytes> {
    let src_w = picture.width();
    let src_h = picture.height();
    if src_w == 0 || src_h == 0 || canvas_w == 0 || canvas_h == 0 {
        return None;
    }
    if src_w == canvas_w && src_h == canvas_h {
        return Some(picture.clone());
    }
    let (draw_w, draw_h) = cover_size(src_w, src_h, canvas_w, canvas_h)?;
    let scaled = crate::sample::scale(src_w, src_h, picture.rgba(), draw_w, draw_h)?;
    let x0 = draw_w.checked_sub(canvas_w)? / 2;
    let y0 = draw_h.checked_sub(canvas_h)? / 2;
    let mut rgba = vec![0_u8; rgba_len(canvas_w, canvas_h)?];
    let src_stride = usize::try_from(draw_w).ok()?.checked_mul(4)?;
    let dst_stride = usize::try_from(canvas_w).ok()?.checked_mul(4)?;
    for row in 0..canvas_h {
        let src_y = usize::try_from(y0.checked_add(row)?).ok()?;
        let src_at = src_y
            .checked_mul(src_stride)?
            .checked_add(usize::try_from(x0).ok()?.checked_mul(4)?)?;
        let dst_at = usize::try_from(row).ok()?.checked_mul(dst_stride)?;
        rgba.get_mut(dst_at..dst_at + dst_stride)?
            .copy_from_slice(scaled.get(src_at..src_at + dst_stride)?);
    }
    CpuBytes::new(canvas_w, canvas_h, rgba).ok()
}

pub(crate) fn cover_size(
    src_w: u32,
    src_h: u32,
    canvas_w: u32,
    canvas_h: u32,
) -> Option<(u32, u32)> {
    let by_width = u64::from(canvas_w).checked_mul(u64::from(src_h))?;
    let by_height = u64::from(canvas_h).checked_mul(u64::from(src_w))?;
    if by_width <= by_height {
        let draw_w = u64::from(src_w)
            .checked_mul(u64::from(canvas_h))?
            .checked_div(u64::from(src_h))?;
        let draw_w = u32::try_from(draw_w).ok()?.max(canvas_w);
        Some((draw_w, canvas_h))
    } else {
        let draw_h = u64::from(src_h)
            .checked_mul(u64::from(canvas_w))?
            .checked_div(u64::from(src_w))?;
        let draw_h = u32::try_from(draw_h).ok()?.max(canvas_h);
        Some((canvas_w, draw_h))
    }
}

/// Separable box blur. Samples past an edge repeat that edge.
fn box_blur(pixels: &mut [u8], width: u32, height: u32, radius: u32) -> Option<()> {
    if radius == 0 {
        return Some(());
    }
    let len = rgba_len(width, height)?;
    if pixels.len() != len {
        return None;
    }
    let mut horizontal = vec![0_u8; len];
    let mut prefix = Vec::new();
    blur_horizontal(pixels, &mut horizontal, width, height, radius, &mut prefix)?;
    blur_vertical(&horizontal, pixels, width, height, radius, &mut prefix)
}

fn blur_horizontal(
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    height: u32,
    radius: u32,
    prefix: &mut Vec<[u32; 4]>,
) -> Option<()> {
    let width_us = usize::try_from(width).ok()?;
    let stride = width_us.checked_mul(4)?;
    let radius_us = usize::try_from(radius).ok()?;
    let window = usize::try_from(radius.checked_mul(2)?.checked_add(1)?).ok()?;
    let extended = width_us.checked_add(radius_us)?.checked_add(radius_us)?;
    prefix.clear();
    prefix.try_reserve_exact(extended.checked_add(1)?).ok()?;
    for y in 0..usize::try_from(height).ok()? {
        let row = y.checked_mul(stride)?;
        let samples = src.get(row..row + stride)?;
        fill_prefix(prefix, extended, |index| {
            let sample_x = index
                .saturating_sub(radius_us)
                .min(width_us.saturating_sub(1));
            let pixel = samples.get(sample_x * 4..sample_x * 4 + 4)?;
            Some([pixel[0], pixel[1], pixel[2], pixel[3]])
        })?;
        let out = dst.get_mut(row..row + stride)?;
        for x in 0..width_us {
            let sum = prefix_window(prefix, x, window)?;
            let pixel = out.get_mut(x * 4..x * 4 + 4)?;
            for channel in 0..4 {
                pixel[channel] = average(sum[channel], window)?;
            }
        }
    }
    Some(())
}

fn blur_vertical(
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    height: u32,
    radius: u32,
    prefix: &mut Vec<[u32; 4]>,
) -> Option<()> {
    let width_us = usize::try_from(width).ok()?;
    let height_us = usize::try_from(height).ok()?;
    let stride = width_us.checked_mul(4)?;
    let window = usize::try_from(radius.checked_mul(2)?.checked_add(1)?).ok()?;
    let radius_us = usize::try_from(radius).ok()?;
    let extended = height_us.checked_add(window)?.checked_sub(1)?;
    prefix.clear();
    prefix.try_reserve_exact(extended.checked_add(1)?).ok()?;
    for x in 0..width_us {
        fill_prefix(prefix, extended, |index| {
            let sample_y = index
                .saturating_sub(radius_us)
                .min(height_us.saturating_sub(1));
            let at = sample_y
                .checked_mul(stride)?
                .checked_add(x.checked_mul(4)?)?;
            let pixel = src.get(at..at + 4)?;
            Some([pixel[0], pixel[1], pixel[2], pixel[3]])
        })?;
        for y in 0..height_us {
            let sum = prefix_window(prefix, y, window)?;
            let at = y.checked_mul(stride)?.checked_add(x.checked_mul(4)?)?;
            let pixel = dst.get_mut(at..at + 4)?;
            for channel in 0..4 {
                pixel[channel] = average(sum[channel], window)?;
            }
        }
    }
    Some(())
}

fn fill_prefix(
    prefix: &mut Vec<[u32; 4]>,
    extended: usize,
    mut sample: impl FnMut(usize) -> Option<[u8; 4]>,
) -> Option<()> {
    prefix.clear();
    prefix.push([0_u32; 4]);
    let mut sum = [0_u32; 4];
    for index in 0..extended {
        let pixel = sample(index)?;
        for channel in 0..4 {
            sum[channel] = sum[channel].saturating_add(u32::from(pixel[channel]));
        }
        prefix.push(sum);
    }
    Some(())
}

fn prefix_window(prefix: &[[u32; 4]], index: usize, window: usize) -> Option<[u32; 4]> {
    let end = prefix.get(index.checked_add(window)?)?;
    let start = prefix.get(index)?;
    Some([
        end[0].saturating_sub(start[0]),
        end[1].saturating_sub(start[1]),
        end[2].saturating_sub(start[2]),
        end[3].saturating_sub(start[3]),
    ])
}

fn average(sum: u32, window: usize) -> Option<u8> {
    let window = u32::try_from(window).ok()?;
    if window == 0 {
        return None;
    }
    u8::try_from(sum / window).ok()
}

fn fill(width: u32, height: u32, pixel: [u8; 4]) -> Option<CpuBytes> {
    let len = rgba_len(width, height)?;
    let mut rgba = Vec::new();
    rgba.try_reserve_exact(len).ok()?;
    rgba.resize(len, 0);
    for chunk in rgba.chunks_mut(4) {
        chunk.copy_from_slice(&pixel);
    }
    CpuBytes::new(width, height, rgba).ok()
}

/// `output = source + destination * (1 - source.alpha)`, truncated.
fn over_into(destination: &mut [u8], source: &[u8]) {
    for (dest, src) in destination.chunks_mut(4).zip(source.chunks(4)) {
        let (Some(dest), Some(src)) = (dest.get_mut(..4), src.get(..4)) else {
            continue;
        };
        let inverse = u16::from(255_u8.saturating_sub(src[3]));
        for channel in 0..4 {
            let mixed = u16::from(src[channel]) + u16::from(dest[channel]) * inverse / 255;
            dest[channel] = u8::try_from(mixed.min(255)).unwrap_or(255);
        }
    }
}

fn round_u32(value: f64) -> Option<u32> {
    if !value.is_finite() || value < 0.0 {
        return None;
    }
    let rounded = value.round();
    if rounded > f64::from(u32::MAX) {
        return None;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Some(rounded as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixel(width: u32, height: u32, rgba: &[u8]) -> CpuBytes {
        CpuBytes::new(width, height, rgba.to_vec()).expect("bytes")
    }

    #[test]
    fn a_canvas_color_is_six_hex_digits_or_a_packed_word() {
        assert_eq!(canvas_rgba("#336699"), Some([0x33, 0x66, 0x99, 255]));
        assert_eq!(canvas_rgba("336699"), Some([0x33, 0x66, 0x99, 255]));
        assert_eq!(canvas_rgba("0x80336699"), Some([0x33, 0x66, 0x99, 0x80]));
        assert_eq!(canvas_rgba("white"), None);
        assert_eq!(canvas_rgba(""), None);
        assert_eq!(canvas_rgba("#12345"), None);
    }

    #[test]
    fn a_cover_keeps_the_center_and_a_blur_of_one_pixel_averages_the_edge() {
        let sharp = pixel(2, 1, &[255, 0, 0, 255, 0, 0, 255, 255]);
        // Scaled to 4x2 between the pixel centers, then the middle two columns.
        let covered = cover(&sharp, 2, 2).expect("cover");
        assert_eq!(
            covered.rgba(),
            &[
                191, 0, 64, 255, 64, 0, 191, 255, 191, 0, 64, 255, 64, 0, 191, 255
            ]
        );
        let placed = pixel(
            2,
            2,
            &[255, 0, 0, 255, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0],
        );
        let blurred = over_backdrop(placed, Some(&sharp), &Backdrop::Blur(1));
        assert_eq!(
            blurred.rgba(),
            &[
                255, 0, 0, 255, 0, 0, 255, 255, 148, 0, 106, 255, 106, 0, 148, 255
            ]
        );
    }

    #[test]
    fn a_color_shows_through_a_transparent_pixel() {
        let top = pixel(1, 1, &[0, 0, 0, 0]);
        let painted = over_backdrop(top, None, &Backdrop::Color([1, 2, 3, 255]));
        assert_eq!(painted.rgba(), &[1, 2, 3, 255]);
    }

    #[test]
    fn a_blur_that_is_not_a_radius_uses_the_color() {
        let mut canvas = MaterialCanvas {
            kind: "canvas_blur".to_owned(),
            blur: f64::NAN,
            color: "#010203".to_owned(),
            ..MaterialCanvas::default()
        };
        assert!(matches!(
            segment_backdrop(&canvas, 2, 2),
            Backdrop::Color([1, 2, 3, 255])
        ));
        canvas.blur = -1.0;
        assert!(matches!(
            segment_backdrop(&canvas, 2, 2),
            Backdrop::Color([1, 2, 3, 255])
        ));
    }
}
