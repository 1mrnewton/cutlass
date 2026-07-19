//! CPU paint for text stroke, background card, and drop shadow.
//!
//! Applied only by [`crate::TextRenderer::rasterize`] — [`crate::TextRenderer::shape`]
//! stays ink-tight so character-level animation can still place clusters freely.

use std::collections::VecDeque;

use crate::style::TextStyle;
use crate::{ShapedText, over_straight};
use cutlass_core::RgbaImage;

/// Extra margin (as a fraction of font size) between glyph ink and the
/// background card's edge — CapCut-ish breathing room.
const BG_INSET_FRAC: f32 = 0.18;
const BOX_BLUR_ITERATIONS: u32 = 2;

/// Defensive render-side ceiling for direct `cutlass-text` callers. The app's
/// model validation is tighter in reference space, but this crate also has a
/// public pixel-space API and must not turn a corrupt/hostile float into an
/// overflowing allocation or practically unbounded morphology loop.
const MAX_EFFECT_EXTENT_PX: f32 = 4096.0;

/// Compose stroke / background / shadow around an already-shaped run.
pub(crate) fn paint(shaped: &ShapedText, style: &TextStyle) -> RgbaImage {
    let pad = effect_padding(style);
    let width = shaped.extent.0.saturating_add(pad.saturating_mul(2));
    let height = shaped.extent.1.saturating_add(pad.saturating_mul(2));
    let Some(pixel_len) = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(4))
    else {
        return RgbaImage::transparent(0, 0);
    };
    let mut pixels = vec![0u8; pixel_len];

    // Coverage of the fill glyphs — substrate for stroke dilation and shadow blur.
    let cover = glyph_coverage(shaped, pad, width, height);

    if let Some(bg) = style.background {
        let inset = effect_extent(style.font_size * BG_INSET_FRAC).max(2.0);
        let rect = CardRect {
            x0: pad as f32 - inset,
            y0: pad as f32 - inset,
            x1: pad as f32 + shaped.extent.0 as f32 + inset,
            y1: pad as f32 + shaped.extent.1 as f32 + inset,
        };
        fill_rounded_rect(&mut pixels, width, height, rect, bg.radius, bg.rgba);
    }

    // Stroke is the expensive mask operation. Compute it once: a shadow can
    // bloom from the stroked silhouette and the visible outline uses the same
    // pixels immediately afterwards.
    let stroke_radius = style
        .stroke
        .map_or(0.0, |stroke| effect_extent(stroke.width));
    let stroke_mask =
        (stroke_radius > 0.0).then(|| dilate_alpha(&cover, width, height, stroke_radius));

    if let Some(shadow) = style.shadow {
        // Glow/outline shadows should bloom from the stroked silhouette when
        // a stroke is present (neon / chrome presets).
        let shadow_src = stroke_mask.as_deref().unwrap_or(&cover);
        let blur_px = shadow_blur_px(style, shadow.blur);
        let blurred = (blur_px > 0.5).then(|| box_blur_alpha(shadow_src, width, height, blur_px));
        let shadow_mask = blurred.as_deref().unwrap_or(shadow_src);
        // CapCut-style 45° down-right offset.
        let offset = signed_effect_extent(shadow.distance) / std::f32::consts::SQRT_2;
        let dx = offset.round() as i32;
        let dy = dx;
        blit_tinted(&mut pixels, width, height, shadow_mask, dx, dy, shadow.rgba);
    }

    if let (Some(stroke), Some(dilated)) = (style.stroke, stroke_mask.as_deref()) {
        blit_tinted(&mut pixels, width, height, dilated, 0, 0, stroke.rgba);
    }

    blit_clusters(&mut pixels, width, height, shaped, pad);
    RgbaImage::new(width, height, pixels)
}

/// Bitmap headroom needed so stroke / shadow / background don't clip.
pub(crate) fn effect_padding(style: &TextStyle) -> u32 {
    let stroke_radius = style
        .stroke
        .map_or(0.0, |stroke| effect_extent(stroke.width));
    let mut need = (style.padding as f32).min(MAX_EFFECT_EXTENT_PX);
    if stroke_radius > 0.0 {
        need = need.max(stroke_radius + 1.0);
    }
    if let Some(shadow) = style.shadow {
        let blur_px = shadow_blur_px(style, shadow.blur);
        let blur_support = box_blur_support(blur_px);
        let offset = effect_extent(shadow.distance.abs()) / std::f32::consts::SQRT_2;
        // The shadow source is the stroked silhouette, so these extents are
        // additive rather than alternatives. `max(stroke, blur + offset)`
        // clips combined outline/shadow treatments at the bitmap edge.
        need = need.max(stroke_radius + blur_support + offset + 1.0);
    }
    if style.background.is_some() {
        let inset = effect_extent(style.font_size * BG_INSET_FRAC).max(2.0);
        need = need.max(inset);
    }
    need.min(MAX_EFFECT_EXTENT_PX).ceil() as u32
}

fn effect_extent(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, MAX_EFFECT_EXTENT_PX)
    } else {
        0.0
    }
}

fn signed_effect_extent(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(-MAX_EFFECT_EXTENT_PX, MAX_EFFECT_EXTENT_PX)
    } else {
        0.0
    }
}

fn unit_fraction(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn shadow_blur_px(style: &TextStyle, blur: f32) -> f32 {
    effect_extent(unit_fraction(blur) * effect_extent(style.font_size))
}

fn box_blur_support(radius: f32) -> f32 {
    if radius > 0.5 {
        BOX_BLUR_ITERATIONS as f32 * radius.round().max(1.0)
    } else {
        0.0
    }
}

fn glyph_coverage(shaped: &ShapedText, pad: u32, width: u32, height: u32) -> Vec<u8> {
    let mut cover = vec![0u8; (width as usize) * (height as usize)];
    for cluster in &shaped.clusters {
        let (cw, ch) = (cluster.image.width, cluster.image.height);
        if cw == 0 || ch == 0 {
            continue;
        }
        let ox = cluster.offset[0].round() as i64 + i64::from(pad);
        let oy = cluster.offset[1].round() as i64 + i64::from(pad);
        for row in 0..ch {
            for col in 0..cw {
                let src = cluster.image.pixel(col, row);
                if src[3] == 0 {
                    continue;
                }
                let (px, py) = (ox + i64::from(col), oy + i64::from(row));
                if px < 0 || py < 0 || px >= i64::from(width) || py >= i64::from(height) {
                    continue;
                }
                let idx = (py as u32 * width + px as u32) as usize;
                cover[idx] = cover[idx].max(src[3]);
            }
        }
    }
    cover
}

fn blit_clusters(pixels: &mut [u8], width: u32, height: u32, shaped: &ShapedText, pad: u32) {
    for cluster in &shaped.clusters {
        let (cw, ch) = (cluster.image.width, cluster.image.height);
        if cw == 0 || ch == 0 {
            continue;
        }
        let ox = cluster.offset[0].round() as i64 + i64::from(pad);
        let oy = cluster.offset[1].round() as i64 + i64::from(pad);
        for row in 0..ch {
            for col in 0..cw {
                let src = cluster.image.pixel(col, row);
                if src[3] == 0 {
                    continue;
                }
                let (px, py) = (ox + i64::from(col), oy + i64::from(row));
                if px < 0 || py < 0 || px >= i64::from(width) || py >= i64::from(height) {
                    continue;
                }
                let idx = ((py as u32 * width + px as u32) * 4) as usize;
                over_straight(&mut pixels[idx..idx + 4], src);
            }
        }
    }
}

/// Tint an alpha mask with `color` and composite it (source-over) at `(dx, dy)`.
fn blit_tinted(
    pixels: &mut [u8],
    width: u32,
    height: u32,
    cover: &[u8],
    dx: i32,
    dy: i32,
    color: [u8; 4],
) {
    let color_a = u32::from(color[3]);
    if color_a == 0 {
        return;
    }
    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let sx = x - dx;
            let sy = y - dy;
            if sx < 0 || sy < 0 || sx >= width as i32 || sy >= height as i32 {
                continue;
            }
            let a = cover[(sy as u32 * width + sx as u32) as usize];
            if a == 0 {
                continue;
            }
            let out_a = ((u32::from(a) * color_a + 127) / 255) as u8;
            if out_a == 0 {
                continue;
            }
            let idx = ((y as u32 * width + x as u32) * 4) as usize;
            over_straight(
                &mut pixels[idx..idx + 4],
                [color[0], color[1], color[2], out_a],
            );
        }
    }
}

/// Morphological dilation of an alpha mask by `radius` px (circular kernel).
///
/// For each integer y-offset in the circle, a monotonic-queue horizontal max
/// filter produces that row's exact chord in O(width), then the chord is
/// merged into the two symmetric destination offsets. This preserves the old
/// circular result while reducing O(width * height * radius²) to
/// O(width * height * radius).
fn dilate_alpha(src: &[u8], w: u32, h: u32, radius: f32) -> Vec<u8> {
    let radius = effect_extent(radius);
    let r = radius.ceil() as usize;
    if r == 0 {
        return src.to_vec();
    }
    let r2 = radius * radius;
    let mut dst = vec![0u8; src.len()];
    let mut chord = vec![0u8; src.len()];
    let height = h as usize;
    let width = w as usize;

    for dy in 0..=r {
        let dy2 = (dy * dy) as f32;
        if dy2 > r2 {
            continue;
        }
        let half_chord = (r2 - dy2).sqrt().floor() as usize;
        horizontal_max_filter(src, width, height, half_chord, &mut chord);

        for y in 0..height {
            if let Some(source_y) = y.checked_add(dy).filter(|source_y| *source_y < height) {
                merge_mask_row(&mut dst, &chord, width, y, source_y);
            }
            if dy > 0
                && let Some(source_y) = y.checked_sub(dy)
            {
                merge_mask_row(&mut dst, &chord, width, y, source_y);
            }
        }
    }
    dst
}

/// Row-wise centered maximum filter with clipped edges, using a monotonic
/// queue so every input sample enters and leaves once.
fn horizontal_max_filter(src: &[u8], w: usize, h: usize, radius: usize, dst: &mut [u8]) {
    if w == 0 || h == 0 {
        return;
    }
    let mut queue = VecDeque::<usize>::with_capacity(radius.saturating_mul(2).saturating_add(1));
    for y in 0..h {
        queue.clear();
        let row = y * w;
        let mut next = 0usize;
        for x in 0..w {
            let upper = x.saturating_add(radius).min(w - 1);
            while next <= upper {
                while queue
                    .back()
                    .is_some_and(|index| src[row + *index] <= src[row + next])
                {
                    queue.pop_back();
                }
                queue.push_back(next);
                next += 1;
            }
            let lower = x.saturating_sub(radius);
            while queue.front().is_some_and(|index| *index < lower) {
                queue.pop_front();
            }
            dst[row + x] = src[row + queue[0]];
        }
    }
}

fn merge_mask_row(dst: &mut [u8], src: &[u8], width: usize, dst_y: usize, src_y: usize) {
    let dst_row = dst_y * width;
    let src_row = src_y * width;
    for x in 0..width {
        dst[dst_row + x] = dst[dst_row + x].max(src[src_row + x]);
    }
}

/// Approximate Gaussian blur with two separable box passes.
fn box_blur_alpha(src: &[u8], w: u32, h: u32, radius: f32) -> Vec<u8> {
    let r = radius.round().max(1.0) as i32;
    let mut tmp = vec![0u8; src.len()];
    let mut out = src.to_vec();
    for _ in 0..BOX_BLUR_ITERATIONS {
        box_blur_pass(&out, &mut tmp, w, h, r, true);
        box_blur_pass(&tmp, &mut out, w, h, r, false);
    }
    out
}

fn box_blur_pass(src: &[u8], dst: &mut [u8], w: u32, h: u32, r: i32, horizontal: bool) {
    let (outer, inner) = if horizontal {
        (h as i32, w as i32)
    } else {
        (w as i32, h as i32)
    };
    let window = (2 * r + 1) as u32;
    for o in 0..outer {
        // Running sum for O(n) sliding window.
        let mut sum: u32 = 0;
        for i in -r..=r {
            sum += sample_1d(src, w, h, o, i.clamp(0, inner - 1), horizontal) as u32;
        }
        for i in 0..inner {
            dst[index_1d(w, o, i, horizontal)] = ((sum + window / 2) / window) as u8;
            let leave = i - r;
            let enter = i + r + 1;
            if leave >= 0 {
                sum -= sample_1d(src, w, h, o, leave, horizontal) as u32;
            } else {
                sum -= sample_1d(src, w, h, o, 0, horizontal) as u32;
            }
            if enter < inner {
                sum += sample_1d(src, w, h, o, enter, horizontal) as u32;
            } else {
                sum += sample_1d(src, w, h, o, inner - 1, horizontal) as u32;
            }
        }
    }
}

fn sample_1d(src: &[u8], w: u32, h: u32, outer: i32, inner: i32, horizontal: bool) -> u8 {
    let (x, y) = if horizontal {
        (inner as u32, outer as u32)
    } else {
        (outer as u32, inner as u32)
    };
    debug_assert!(x < w && y < h);
    src[(y * w + x) as usize]
}

fn index_1d(w: u32, outer: i32, inner: i32, horizontal: bool) -> usize {
    let (x, y) = if horizontal {
        (inner as u32, outer as u32)
    } else {
        (outer as u32, inner as u32)
    };
    (y * w + x) as usize
}

/// Axis-aligned card rectangle in bitmap space.
struct CardRect {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
}

/// Fill a rounded rectangle (axis-aligned) with analytic coverage.
fn fill_rounded_rect(
    pixels: &mut [u8],
    width: u32,
    height: u32,
    rect: CardRect,
    radius_frac: f32,
    color: [u8; 4],
) {
    let CardRect { x0, y0, x1, y1 } = rect;
    if color[3] == 0 || x1 <= x0 || y1 <= y0 {
        return;
    }
    let rw = x1 - x0;
    let rh = y1 - y0;
    let max_r = rw.min(rh) * 0.5;
    let radius = unit_fraction(radius_frac) * max_r;

    let min_x = x0.floor().max(0.0) as i32;
    let min_y = y0.floor().max(0.0) as i32;
    let max_x = x1.ceil().min(width as f32) as i32;
    let max_y = y1.ceil().min(height as f32) as i32;

    for y in min_y..max_y {
        for x in min_x..max_x {
            let cx = x as f32 + 0.5;
            let cy = y as f32 + 0.5;
            let d = sd_rounded_box(
                cx - (x0 + x1) * 0.5,
                cy - (y0 + y1) * 0.5,
                rw * 0.5,
                rh * 0.5,
                radius,
            );
            // 1px AA band.
            let cover = (0.5 - d).clamp(0.0, 1.0);
            if cover <= 0.0 {
                continue;
            }
            let a = ((cover * f32::from(color[3])).round() as u32).min(255) as u8;
            if a == 0 {
                continue;
            }
            let idx = ((y as u32 * width + x as u32) * 4) as usize;
            over_straight(&mut pixels[idx..idx + 4], [color[0], color[1], color[2], a]);
        }
    }
}

/// Signed distance to a rounded box centered at the origin with half-extents
/// `(hx, hy)` and corner radius `r` (Inigo Quilez).
fn sd_rounded_box(px: f32, py: f32, hx: f32, hy: f32, r: f32) -> f32 {
    let qx = px.abs() - hx + r;
    let qy = py.abs() - hy + r;
    let outside = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt();
    outside + qx.min(0.0).max(qy.min(0.0)) - r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::style::{TextBackground, TextShadow, TextStroke};
    use crate::{TextRenderer, TextStyle};

    const TEST_FONT: &[u8] = include_bytes!("../assets/Micro5-Regular.ttf");

    fn renderer() -> TextRenderer {
        let mut r = TextRenderer::new();
        assert!(r.load_font(TEST_FONT.to_vec()) > 0);
        r
    }

    fn covered(img: &RgbaImage) -> usize {
        img.pixels.chunks_exact(4).filter(|p| p[3] != 0).count()
    }

    fn count_near(img: &RgbaImage, rgb: [u8; 3], tol: i16) -> usize {
        img.pixels
            .chunks_exact(4)
            .filter(|p| {
                p[3] > 32
                    && (i16::from(p[0]) - i16::from(rgb[0])).abs() <= tol
                    && (i16::from(p[1]) - i16::from(rgb[1])).abs() <= tol
                    && (i16::from(p[2]) - i16::from(rgb[2])).abs() <= tol
            })
            .count()
    }

    fn dilate_alpha_reference(src: &[u8], w: u32, h: u32, radius: f32) -> Vec<u8> {
        let r = radius.ceil().max(0.0) as i32;
        if r == 0 {
            return src.to_vec();
        }
        let r2 = radius * radius;
        let mut dst = vec![0u8; src.len()];
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let mut max_a = 0u8;
                for dy in -r..=r {
                    for dx in -r..=r {
                        if (dx * dx + dy * dy) as f32 > r2 {
                            continue;
                        }
                        let nx = x + dx;
                        let ny = y + dy;
                        if nx >= 0 && ny >= 0 && nx < w as i32 && ny < h as i32 {
                            max_a = max_a.max(src[(ny as u32 * w + nx as u32) as usize]);
                        }
                    }
                }
                dst[(y as u32 * w + x as u32) as usize] = max_a;
            }
        }
        dst
    }

    #[test]
    fn fast_dilation_matches_circular_reference() {
        let (w, h) = (9, 7);
        let mut src = vec![0u8; (w * h) as usize];
        for (index, alpha) in [(0, 17), (7, 93), (22, 255), (41, 128), (62, 211)] {
            src[index] = alpha;
        }
        for radius in [0.25, 1.0, 1.75, 2.0, 3.4, 5.0] {
            assert_eq!(
                dilate_alpha(&src, w, h, radius),
                dilate_alpha_reference(&src, w, h, radius),
                "radius {radius}"
            );
        }
    }

    #[test]
    fn combined_stroke_shadow_padding_adds_every_extent() {
        let style = TextStyle::new(100.0)
            .with_stroke(TextStroke {
                rgba: [0, 0, 0, 255],
                width: 10.0,
            })
            .with_shadow(TextShadow {
                rgba: [0, 0, 0, 255],
                blur: 0.2,
                distance: 10.0 * std::f32::consts::SQRT_2,
            });
        // stroke 10 + two 20px blur iterations + per-axis offset 10 + 1px AA guard.
        assert_eq!(effect_padding(&style), 61);
    }

    #[test]
    fn non_finite_effect_metrics_are_bounded() {
        let style = TextStyle::new(48.0)
            .with_stroke(TextStroke {
                rgba: [0, 0, 0, 255],
                width: f32::INFINITY,
            })
            .with_shadow(TextShadow {
                rgba: [0, 0, 0, 255],
                blur: f32::NAN,
                distance: f32::NEG_INFINITY,
            });
        assert_eq!(effect_padding(&style), 1);
        let img = renderer().rasterize("Safe", &style);
        assert!(img.width > 0 && img.height > 0);
    }

    #[test]
    fn stroke_adds_outline_pixels_beyond_fill() {
        let mut r = renderer();
        let plain = r.rasterize("A", &TextStyle::new(48.0).with_color([255, 255, 255, 255]));
        let stroked = r.rasterize(
            "A",
            &TextStyle::new(48.0)
                .with_color([255, 255, 255, 255])
                .with_stroke(TextStroke {
                    rgba: [255, 0, 0, 255],
                    width: 4.0,
                }),
        );
        assert!(stroked.width > plain.width && stroked.height > plain.height);
        assert!(covered(&stroked) > covered(&plain));
        assert!(
            count_near(&stroked, [255, 0, 0], 40) > 20,
            "expected red stroke pixels"
        );
    }

    #[test]
    fn inspector_max_stroke_rasterizes_a_title() {
        let mut r = renderer();
        let img = r.rasterize(
            "Responsive stroke preview",
            &TextStyle::new(96.0)
                .with_color([255, 255, 255, 255])
                .with_stroke(TextStroke {
                    rgba: [0, 0, 0, 255],
                    width: 40.0,
                }),
        );
        assert!(img.width > 0 && img.height > 0);
    }

    #[test]
    fn background_fills_card_behind_glyphs() {
        let mut r = renderer();
        let img = r.rasterize(
            "Hi",
            &TextStyle::new(40.0)
                .with_color([255, 255, 255, 255])
                .with_background(TextBackground {
                    rgba: [0, 0, 255, 255],
                    radius: 0.0,
                }),
        );
        assert!(
            count_near(&img, [0, 0, 255], 20) > 100,
            "expected blue card"
        );
        // A pixel well inside the card (just outside the glyph ink on the left)
        // should be solid blue — not the AA fringe at the bitmap edge.
        let sample = img.pixel(2, img.height / 2);
        assert_eq!(&sample[..3], &[0, 0, 255]);
        assert!(
            sample[3] > 200,
            "interior card pixel should be opaque: {sample:?}"
        );
    }

    #[test]
    fn shadow_tints_offset_pixels() {
        let mut r = renderer();
        let img = r.rasterize(
            "A",
            &TextStyle::new(48.0)
                .with_color([255, 255, 255, 255])
                .with_shadow(TextShadow {
                    rgba: [0, 255, 0, 255],
                    blur: 0.0,
                    distance: 10.0,
                }),
        );
        assert!(
            count_near(&img, [0, 255, 0], 40) > 10,
            "expected green shadow pixels"
        );
    }
}
