//! Per-cluster effect paint for character-level animation.
//!
//! Stroke and shadow are folded into each cluster's bitmap so outlines move
//! with their letter. The background card stays a whole-run quad drawn behind
//! the glyphs (CapCut-style).

use crate::effects::{effect_padding, paint as paint_run};
use crate::style::TextStyle;
use crate::{ClusterBox, ShapedText};
use cutlass_core::RgbaImage;

/// Shaped text with per-cluster stroke/shadow folded in, plus an optional
/// whole-run background card.
#[derive(Debug, Clone, PartialEq)]
pub struct AnimatedText {
    /// Ink-tight extent of the unstyled shaped run (alignment reference).
    pub extent: (u32, u32),
    /// Clusters with stroke/shadow painted into their images. Offsets are
    /// relative to the ink-tight origin (same space as [`ShapedText`]); a
    /// cluster's image may extend into negative offset when padded.
    pub clusters: Vec<ClusterBox>,
    /// Whole-run background card, or `None`.
    pub background: Option<RgbaImage>,
    /// Top-left of `background` relative to the ink-tight origin.
    pub background_offset: [f32; 2],
}

impl AnimatedText {
    pub fn has_ink(&self) -> bool {
        self.extent.0 > 0 && self.extent.1 > 0
    }
}

/// Paint stroke / shadow per cluster and an optional background card for
/// character-level animation. When the style has no per-glyph treatments and
/// no background, this is equivalent to cloning the shaped clusters.
pub fn paint_animated(shaped: &ShapedText, style: &TextStyle) -> AnimatedText {
    if !shaped.has_ink() {
        return AnimatedText {
            extent: (0, 0),
            clusters: Vec::new(),
            background: None,
            background_offset: [0.0, 0.0],
        };
    }

    let needs_cluster_fx = style.stroke.is_some() || style.shadow.is_some() || style.underline;
    let clusters = if needs_cluster_fx {
        shaped
            .clusters
            .iter()
            .map(|c| paint_cluster(c, style))
            .collect()
    } else {
        shaped.clusters.clone()
    };

    let (background, background_offset) = paint_background(shaped, style);

    AnimatedText {
        extent: shaped.extent,
        clusters,
        background,
        background_offset,
    }
}

/// Cluster-local pad for stroke / shadow (no background inset — that lives on
/// the whole-run card).
fn cluster_pad(style: &TextStyle) -> u32 {
    // Reuse the run padding helper but ignore background-only growth by
    // painting against a style without background.
    let mut local = style.clone();
    local.background = None;
    local.padding = 0;
    effect_padding(&local)
}

fn paint_cluster(cluster: &ClusterBox, style: &TextStyle) -> ClusterBox {
    if cluster.image.width == 0 || cluster.image.height == 0 {
        return cluster.clone();
    }
    let pad = cluster_pad(style);
    if pad == 0 && !style.underline {
        return cluster.clone();
    }

    // Build a one-cluster shaped run so we can reuse the whole-run painter's
    // stroke/shadow path (minus background).
    let local = ShapedText {
        extent: (cluster.image.width, cluster.image.height),
        clusters: vec![ClusterBox {
            text_range: cluster.text_range.clone(),
            line: 0,
            offset: [0.0, 0.0],
            baseline: cluster.baseline - cluster.offset[1],
            image: cluster.image.clone(),
        }],
    };
    let mut local_style = style.clone();
    local_style.background = None;
    local_style.padding = 0;
    // Underlines need line context; skip on the isolated cluster and paint
    // them into the background card instead.
    local_style.underline = false;
    let painted = paint_run(&local, &local_style);
    ClusterBox {
        text_range: cluster.text_range.clone(),
        line: cluster.line,
        offset: [
            cluster.offset[0] - pad as f32,
            cluster.offset[1] - pad as f32,
        ],
        baseline: cluster.baseline,
        image: painted,
    }
}

fn paint_background(shaped: &ShapedText, style: &TextStyle) -> (Option<RgbaImage>, [f32; 2]) {
    let has_bg = style.background.is_some();
    let has_underline = style.underline;
    if !has_bg && !has_underline {
        return (None, [0.0, 0.0]);
    }

    // Paint the full run with fill alpha zeroed so only bg + underline remain.
    let mut ghost = style.clone();
    ghost.color = [style.color[0], style.color[1], style.color[2], 0];
    ghost.stroke = None;
    ghost.shadow = None;
    if !has_bg {
        ghost.background = None;
    }
    let painted = paint_run(shaped, &ghost);
    if painted.width == 0 || painted.height == 0 {
        return (None, [0.0, 0.0]);
    }
    // `paint_run` pads symmetrically; origin of the padded bitmap relative to
    // the ink-tight extent is (−pad, −pad).
    let pad = effect_padding(&ghost) as f32;
    // Strip fully-transparent results (e.g. underline-only with no coverage).
    let lit = painted.pixels.chunks_exact(4).any(|p| p[3] > 0);
    if !lit {
        return (None, [0.0, 0.0]);
    }
    (Some(painted), [-pad, -pad])
}

/// Composite animated clusters at identity into a single bitmap — test helper
/// and documentation of the placement contract.
#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn composite_animated(anim: &AnimatedText) -> RgbaImage {
    if !anim.has_ink() {
        return RgbaImage::transparent(0, 0);
    }
    // Bound = extent union every cluster image (which may overhang).
    let mut min_x = 0.0f32;
    let mut min_y = 0.0f32;
    let mut max_x = anim.extent.0 as f32;
    let mut max_y = anim.extent.1 as f32;
    for c in &anim.clusters {
        min_x = min_x.min(c.offset[0]);
        min_y = min_y.min(c.offset[1]);
        max_x = max_x.max(c.offset[0] + c.image.width as f32);
        max_y = max_y.max(c.offset[1] + c.image.height as f32);
    }
    if let Some(bg) = &anim.background {
        min_x = min_x.min(anim.background_offset[0]);
        min_y = min_y.min(anim.background_offset[1]);
        max_x = max_x.max(anim.background_offset[0] + bg.width as f32);
        max_y = max_y.max(anim.background_offset[1] + bg.height as f32);
    }
    let origin = [min_x, min_y];
    let width = (max_x - min_x).ceil().max(0.0) as u32;
    let height = (max_y - min_y).ceil().max(0.0) as u32;
    let mut pixels = vec![0u8; (width as usize) * (height as usize) * 4];

    if let Some(bg) = &anim.background {
        blit(
            &mut pixels,
            width,
            height,
            bg,
            anim.background_offset[0] - origin[0],
            anim.background_offset[1] - origin[1],
        );
    }
    for c in &anim.clusters {
        if c.image.width == 0 {
            continue;
        }
        blit(
            &mut pixels,
            width,
            height,
            &c.image,
            c.offset[0] - origin[0],
            c.offset[1] - origin[1],
        );
    }
    RgbaImage::new(width, height, pixels)
}

#[cfg(test)]
fn blit(dst: &mut [u8], dw: u32, dh: u32, src: &RgbaImage, ox: f32, oy: f32) {
    use crate::over_straight;
    let ox = ox.round() as i64;
    let oy = oy.round() as i64;
    for row in 0..src.height {
        for col in 0..src.width {
            let px = src.pixel(col, row);
            if px[3] == 0 {
                continue;
            }
            let x = ox + i64::from(col);
            let y = oy + i64::from(row);
            if x < 0 || y < 0 || x >= i64::from(dw) || y >= i64::from(dh) {
                continue;
            }
            let idx = ((y as u32 * dw + x as u32) * 4) as usize;
            over_straight(&mut dst[idx..idx + 4], px);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::style::{TextBackground, TextStroke};
    use crate::{TextRenderer, TextStyle};

    const TEST_FONT: &[u8] = include_bytes!("../assets/Micro5-Regular.ttf");

    fn renderer() -> TextRenderer {
        let mut r = TextRenderer::new();
        assert!(r.load_font(TEST_FONT.to_vec()) > 0);
        r
    }

    #[test]
    fn animated_stroke_grows_cluster_images() {
        let mut r = renderer();
        let style = TextStyle::new(48.0).with_stroke(TextStroke {
            rgba: [0, 0, 0, 255],
            width: 4.0,
        });
        let shaped = r.shape("Hi", &style);
        let anim = paint_animated(&shaped, &style);
        assert_eq!(anim.clusters.len(), shaped.clusters.len());
        for (a, b) in anim.clusters.iter().zip(shaped.clusters.iter()) {
            if b.image.width == 0 {
                continue;
            }
            assert!(
                a.image.width > b.image.width && a.image.height > b.image.height,
                "stroke should pad the cluster bitmap"
            );
        }
    }

    #[test]
    fn animated_background_is_separate() {
        let mut r = renderer();
        let style = TextStyle::new(40.0).with_background(TextBackground {
            rgba: [20, 20, 20, 200],
            radius: 0.4,
        });
        let shaped = r.shape("A", &style);
        let anim = paint_animated(&shaped, &style);
        assert!(anim.background.is_some());
        // Fill glyphs stay ink-tight (no stroke/shadow).
        assert_eq!(anim.clusters[0].image.width, shaped.clusters[0].image.width);
    }
}
