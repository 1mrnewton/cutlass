//! Scale-aware supersampling for CPU-rasterized layers (text + pen paths).
//!
//! Parametric SDF shapes fold transform scale into the box at resolve time and
//! stay analytic — they never need this. Text and pen paths rasterize to
//! bitmaps; applying transform scale only as [`SizeSpec::BitmapScaled`]
//! upsamples with bilinear filtering and looks soft.
//!
//! Instead we raise the raster density by a supersample factor `S` derived
//! from the clip's transform scale, then leave a residual `scale / S` on the
//! quad (≤ 1 on the dominant axis unless the size clamp backs `S` off).
//!
//! ## Quantization
//!
//! `S` is quantized to **quarter steps** (`1.0`, `1.25`, `1.5`, …). Animated
//! scale then re-rasters only when crossing a step, so memo stays warm during
//! drag/playback between steps. Quarters balance visible sharpness pop against
//! memo churn better than integer steps (too soft between 1× and 2×) or finer
//! tenths (re-raster too often).
//!
//! ## Clamp
//!
//! Bitmap long edge is capped at `min(4096, 2 × canvas long edge)`. The 4096
//! floor matches the compositor glyph-atlas / RGBA-upload downscale path
//! (`min(device.max_texture_dimension_2d, 4096)`). When the clamp binds,
//! residual quad scale may exceed 1 (GPU stretch at the max feasible density).

use cutlass_models::Scale2;
use cutlass_shapes::{BezierPath, path_bounds};
use cutlass_text::TextStyle;

use crate::scene::SizeSpec;

/// Quarter-step quantization for transform-scale supersampling.
pub(super) const SUPER_SAMPLE_STEP: f32 = 0.25;

/// Absolute max edge for a CPU-raster bitmap (compositor atlas / upload floor).
pub(super) const MAX_RASTER_EDGE: f32 = 4096.0;

/// Cap relative to the canvas long edge.
pub(super) const MAX_RASTER_EDGE_CANVAS_MULT: f32 = 2.0;

/// Quantize a raw supersample factor to [`SUPER_SAMPLE_STEP`] increments.
///
/// Values below 1 collapse to 1 — we never re-raster smaller than the
/// reference; GPU downscale of the reference bitmap is fine for scale < 1.
pub(super) fn quantize_supersample(raw: f32) -> f32 {
    if !raw.is_finite() || raw <= 1.0 {
        return 1.0;
    }
    (raw / SUPER_SAMPLE_STEP).round() * SUPER_SAMPLE_STEP
}

/// Desired supersample from a clip transform scale (before size clamp).
pub(super) fn supersample_from_scale(scale: Scale2) -> f32 {
    quantize_supersample(scale.x.max(scale.y).max(1.0))
}

/// Back `s` off so `ref_long_edge * s` stays within the raster cap.
pub(super) fn clamp_supersample(s: f32, ref_long_edge: f32, canvas_long: f32) -> f32 {
    let s = if s.is_finite() && s >= 1.0 { s } else { 1.0 };
    let canvas_long = if canvas_long.is_finite() && canvas_long > 0.0 {
        canvas_long
    } else {
        1.0
    };
    let cap = MAX_RASTER_EDGE
        .min(canvas_long * MAX_RASTER_EDGE_CANVAS_MULT)
        .max(1.0);
    if !ref_long_edge.is_finite() || ref_long_edge <= 0.0 {
        return s.min(cap);
    }
    let max_s = (cap / ref_long_edge).max(1.0);
    s.min(max_s)
}

/// Residual [`SizeSpec::BitmapScaled`] multipliers after folding `s` into the raster.
pub(super) fn residual_bitmap_scale(scale: Scale2, s: f32) -> SizeSpec {
    let s = if s.is_finite() && s > 0.0 { s } else { 1.0 };
    SizeSpec::BitmapScaled([scale.x / s, scale.y / s])
}

/// Fold supersample `s` into every raster-pixel field of a mapped [`TextStyle`].
///
/// Scaled (raster px): `font_size`, `line_height`, `letter_spacing`, `max_width`,
/// `padding`, `stroke.width`, `shadow.distance`.
///
/// Left alone (relative / non-px): `shadow.blur` (fraction of font size),
/// `background.radius` (0..=1 corner fraction), colors, flags, alignment, family.
pub(super) fn apply_text_supersample(mut style: TextStyle, s: f32) -> TextStyle {
    style.scale_raster_metrics(s);
    style
}

/// Conservative long-edge estimate of a text bitmap at supersample 1.0.
pub(super) fn estimate_text_ref_long_edge(style: &TextStyle, content: &str) -> f32 {
    let n_lines = content.lines().count().max(1) as f32;
    let max_chars = content
        .lines()
        .map(|l| l.chars().count())
        .max()
        .unwrap_or(1)
        .max(1) as f32;
    let mut w = style.font_size * 0.65 * max_chars
        + style.letter_spacing.abs() * (max_chars - 1.0).max(0.0);
    let mut h = style.line_height.max(style.font_size) * n_lines;
    if let Some(mw) = style.max_width
        && mw.is_finite()
        && mw > 0.0
        && w > mw
    {
        let wrapped = (w / mw).ceil().max(1.0) * n_lines;
        w = mw;
        h = style.line_height.max(style.font_size) * wrapped;
    }
    let pad = (style.padding as f32)
        + style.stroke.map(|s| s.width * 2.0).unwrap_or(0.0)
        + style
            .shadow
            .map(|s| s.distance.abs() + s.blur.max(0.0) * style.font_size * 2.0)
            .unwrap_or(0.0)
        + if style.background.is_some() {
            style.font_size * 0.25
        } else {
            0.0
        }
        + 4.0;
    (w + 2.0 * pad).max(h + 2.0 * pad).max(1.0)
}

/// Long-edge estimate of a path bitmap at `raster_scale == ref_scale` (S = 1).
///
/// `stroke_width` is the unscaled model value — [`cutlass_shapes::PathRaster`]
/// multiplies by `raster_scale` itself.
pub(super) fn estimate_path_ref_long_edge(
    path: &BezierPath,
    stroke_width: f32,
    ref_scale: f32,
) -> f32 {
    let ref_scale = if ref_scale.is_finite() && ref_scale > 0.0 {
        ref_scale
    } else {
        1.0
    };
    let Some((min, max)) = path_bounds(path) else {
        return 1.0;
    };
    let w = (max[0] - min[0]).abs() * ref_scale;
    let h = (max[1] - min[1]).abs() * ref_scale;
    let stroke = if stroke_width.is_finite() {
        stroke_width.max(0.0)
    } else {
        0.0
    };
    let pad = stroke * ref_scale * 0.5 + 2.0;
    (w + 2.0 * pad).max(h + 2.0 * pad).max(1.0)
}

/// Resolve `(S, residual SizeSpec)` for a bitmap layer given transform scale
/// and a reference-resolution long-edge estimate.
pub(super) fn bitmap_supersample(
    scale: Scale2,
    ref_long_edge: f32,
    canvas_long: f32,
) -> (f32, SizeSpec) {
    let desired = supersample_from_scale(scale);
    let s = clamp_supersample(desired, ref_long_edge, canvas_long);
    (s, residual_bitmap_scale(scale, s))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_text::{TextShadow, TextStroke};

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-4, "{a} != {b}");
    }

    #[test]
    fn quantize_quarter_steps() {
        approx(quantize_supersample(0.5), 1.0);
        approx(quantize_supersample(1.0), 1.0);
        approx(quantize_supersample(1.1), 1.0);
        approx(quantize_supersample(1.2), 1.25);
        approx(quantize_supersample(1.9), 2.0);
        approx(quantize_supersample(2.0), 2.0);
        approx(quantize_supersample(2.1), 2.0);
        approx(quantize_supersample(2.2), 2.25);
    }

    #[test]
    fn clamp_backs_off_absurd_scale() {
        // Tiny 10px ref edge on a 1920 canvas → cap 3840 → max S = 384.
        let s = clamp_supersample(50.0, 10.0, 1920.0);
        approx(s, 50.0); // 50 still fits
        let s = clamp_supersample(1000.0, 10.0, 1920.0);
        approx(s, 384.0); // min(4096, 3840) / 10
    }

    #[test]
    fn text_supersample_scales_raster_px_only() {
        let style = TextStyle::new(40.0)
            .with_letter_spacing(4.0)
            .with_line_height(50.0)
            .with_max_width(200.0)
            .with_padding(2)
            .with_stroke(TextStroke {
                rgba: [0, 0, 0, 255],
                width: 3.0,
            })
            .with_shadow(TextShadow {
                rgba: [0, 0, 0, 128],
                blur: 0.2,
                distance: 5.0,
            });
        let out = apply_text_supersample(style, 2.0);
        approx(out.font_size, 80.0);
        approx(out.letter_spacing, 8.0);
        approx(out.line_height, 100.0);
        approx(out.max_width.unwrap(), 400.0);
        assert_eq!(out.padding, 4);
        approx(out.stroke.unwrap().width, 6.0);
        approx(out.shadow.unwrap().distance, 10.0);
        approx(out.shadow.unwrap().blur, 0.2); // fraction — unchanged
    }

    #[test]
    fn residual_divides_each_axis() {
        let SizeSpec::BitmapScaled(r) = residual_bitmap_scale(Scale2 { x: 1.0, y: 3.0 }, 3.0)
        else {
            panic!("expected BitmapScaled");
        };
        approx(r[0], 1.0 / 3.0);
        approx(r[1], 1.0);
    }
}
