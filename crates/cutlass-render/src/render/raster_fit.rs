//! Hard-cap CPU raster density against the compositor texture edge.
//!
//! Resolve-side estimates can understate wide fonts / emoji / effect pads.
//! Before painting, measure (or bound) the actual raster and shrink density
//! when needed — residual grows so on-canvas size stays unchanged. Shrink
//! factors may be `< 1` when the reference size alone exceeds the cap.

use cutlass_shapes::{BezierPath, path_bounds};
use cutlass_text::{TextRenderer, TextStyle, painted_size};

use crate::resolve::RASTER_EDGE_CAP;

/// Shrink factor (`≤ 1`) so `long_edge * shrink ≤ cap`.
pub(super) fn fit_shrink(long_edge: f32, cap: f32) -> f32 {
    if !long_edge.is_finite() || long_edge <= 0.0 || !cap.is_finite() || cap <= 0.0 {
        return 1.0;
    }
    if long_edge <= cap {
        1.0
    } else {
        cap / long_edge
    }
}

/// Grow residual when density shrinks so `raster × residual` is unchanged.
pub(super) fn residual_after_shrink(scale: [f32; 2], shrink: f32) -> [f32; 2] {
    if !shrink.is_finite() || shrink <= 0.0 {
        return scale;
    }
    [scale[0] / shrink, scale[1] / shrink]
}

/// Fit a text style to the texture edge using a shaped probe + painted pad.
///
/// Returns `(style, residual, density, shrink)` ready for rasterize/place.
pub(super) fn fit_text_style(
    text: &mut TextRenderer,
    content: &str,
    style: &TextStyle,
    residual: [f32; 2],
    density: f32,
) -> (TextStyle, [f32; 2], f32, f32) {
    let mut style = style.clone();
    let mut residual = residual;
    let mut density = if density.is_finite() && density > 0.0 {
        density
    } else {
        1.0
    };
    let shaped = text.shape(content, &style);
    if !shaped.has_ink() {
        return (style, residual, density, 1.0);
    }
    let (pw, ph) = painted_size(&shaped, &style);
    let long = pw.max(ph) as f32;
    let shrink = fit_shrink(long, RASTER_EDGE_CAP);
    if shrink < 1.0 {
        style.scale_raster_metrics(shrink);
        residual = residual_after_shrink(residual, shrink);
        density *= shrink;
    }
    (style, residual, density, shrink)
}

/// Fit a pen-path `raster_scale` so the painted bitmap stays ≤ the edge cap.
pub(super) fn fit_path_raster_scale(
    path: &BezierPath,
    stroke_width: f32,
    raster_scale: f32,
    residual: [f32; 2],
) -> (f32, [f32; 2], f32) {
    let scale = if raster_scale.is_finite() && raster_scale > 0.0 {
        raster_scale
    } else {
        1.0
    };
    let Some((min, max)) = path_bounds(path) else {
        return (scale, residual, 1.0);
    };
    let stroke = if stroke_width.is_finite() {
        stroke_width.max(0.0)
    } else {
        0.0
    };
    let pad = stroke * scale * 0.5 + 2.0;
    let w = (max[0] - min[0]).abs() * scale + 2.0 * pad;
    let h = (max[1] - min[1]).abs() * scale + 2.0 * pad;
    let shrink = fit_shrink(w.max(h), RASTER_EDGE_CAP);
    if shrink < 1.0 {
        (
            scale * shrink,
            residual_after_shrink(residual, shrink),
            shrink,
        )
    } else {
        (scale, residual, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_text::TextRenderer;

    const TEST_FONT: &[u8] = include_bytes!("../../../cutlass-text/assets/Micro5-Regular.ttf");

    #[test]
    fn fit_shrink_allows_below_one() {
        assert!((fit_shrink(8192.0, 4096.0) - 0.5).abs() < 1e-5);
        assert!((fit_shrink(100.0, 4096.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn residual_compensates_shrink() {
        let r = residual_after_shrink([1.0, 1.0], 0.5);
        assert!((r[0] - 2.0).abs() < 1e-5);
        assert!((r[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn absurd_font_raster_stays_within_edge_cap_with_compensated_residual() {
        let mut text = TextRenderer::new();
        assert!(text.load_font(TEST_FONT.to_vec()) > 0);
        // Far past the texture edge at reference density.
        let style = TextStyle::new(8000.0);
        let (fitted, residual, _density, shrink) =
            fit_text_style(&mut text, "W", &style, [1.0, 1.0], 1.0);
        assert!(shrink < 1.0, "expected hard-cap shrink, got {shrink}");
        let image = text.rasterize("W", &fitted);
        assert!(
            image.width.max(image.height) as f32 <= RASTER_EDGE_CAP,
            "raster {}×{} exceeds cap",
            image.width,
            image.height
        );
        // On-canvas size ≈ unfitted estimate: fitted_dim × residual ≈ fitted_dim / shrink.
        let on_canvas = image.width as f32 * residual[0];
        let unfitted_probe = text.shape("W", &style);
        let (uw, _) = painted_size(&unfitted_probe, &style);
        // Allow generous tolerance — pad/rounding differ after metric scale.
        assert!(
            (on_canvas - uw as f32).abs() / (uw as f32).max(1.0) < 0.15,
            "on-canvas {on_canvas} vs unfitted {uw}"
        );
    }
}
