//! Text / glyph realize arms — bitmap (static) and per-character GPU glyphs.

use cutlass_compositor::{
    BlendMode, ColorGrade, GlyphInstance, LayerEffects, LayerPlacement, LayerStyles, RgbaImage,
};
use cutlass_text::{TextRenderer, TextStyle};

use crate::error::RenderError;
use crate::scene::{ResolvedPass, SceneLayer, SizeSpec, TextAnimation, TextHighlight};

use super::super::raster_fit::fit_text_style;
use super::super::text_anim::{
    ClusterDelta, atlas_key, cluster_deltas, extent_origin, place_clusters,
};
use super::Realized;

/// Realize a text layer for the main scene walk.
///
/// Returns `None` when the run has no ink (empty / no fonts) — the caller
/// skips the layer, matching the previous inline `continue`.
#[allow(clippy::too_many_arguments)] // mirrors former inline match-arm locals
pub(super) fn realize_text_layer(
    text: &mut TextRenderer,
    layer: &SceneLayer,
    content: &str,
    style: &TextStyle,
    animation: &Option<TextAnimation>,
    highlight: &Option<TextHighlight>,
    raster_density: f32,
    canvas: [f32; 2],
    effects: Vec<ResolvedPass>,
    fx: LayerEffects,
    color_grade: Option<ColorGrade>,
    lut: Option<crate::scene::SceneLut>,
    blend_mode: BlendMode,
    styles: LayerStyles,
) -> Option<Realized> {
    let residual = match layer.size {
        SizeSpec::BitmapScaled(s) => s,
        SizeSpec::Fixed(_) => [1.0, 1.0],
    };
    // Hard-cap against the texture edge using measured painted size; residual
    // grows so on-canvas placement is unchanged.
    let (style, scale, density, _) = fit_text_style(text, content, style, residual, raster_density);

    // Both per-character animation and caption highlighting work on clusters,
    // so either one takes the instanced-glyph path.
    if animation.is_some() || highlight.is_some() {
        let shaped = text.shape(content, &style);
        if !shaped.has_ink() {
            return None;
        }
        // A highlighted run is painted twice (resting and highlight fill); an
        // unhighlighted one keeps `lit` and `covered` empty, which zips away to
        // nothing below.
        let (painted, lit, covered, plate) = match highlight
            .as_ref()
            .map(|highlight| paint_highlight(text, content, &style, highlight))
        {
            Some(painted) => (painted.rest, painted.lit, painted.covered, painted.plate),
            None => (text.animate(content, &style), Vec::new(), Vec::new(), None),
        };
        // Catalog deltas are reference run-pixels; multiply by cumulative
        // raster density so on-canvas motion tracks transform scale (and
        // stays invariant across supersample step crossings).
        let mut deltas: Vec<ClusterDelta> = match animation {
            Some(anim) => cluster_deltas(&shaped, anim)
                .into_iter()
                .map(|mut d| {
                    d.position = [d.position[0] * density, d.position[1] * density];
                    d
                })
                .collect(),
            None => vec![ClusterDelta::IDENTITY; painted.clusters.len()],
        };
        // The active word's pop composes with whatever the look preset is
        // already doing to that cluster.
        if let Some(highlight) = highlight {
            for (delta, covered) in deltas.iter_mut().zip(&covered) {
                if *covered {
                    delta.scale *= highlight.scale;
                }
            }
        }
        let extent_size = [
            painted.extent.0 as f32 * scale[0],
            painted.extent.1 as f32 * scale[1],
        ];
        let aligned = layer.text_quad_center(&style, extent_size, canvas);
        let origin = extent_origin(aligned, painted.extent, scale);
        // place_clusters reads offsets/baselines from ShapedText;
        // rebuild a shaped view over the painted clusters.
        let painted_shaped = cutlass_text::ShapedText {
            extent: painted.extent,
            clusters: painted.clusters.clone(),
        };
        let mut instances = place_clusters(
            &painted_shaped,
            &deltas,
            origin,
            scale,
            layer.rotation,
            layer.opacity,
        );
        // A per-character entrance holds every cluster at zero opacity on the
        // run's first tick (and a clip can be keyframed transparent anywhere),
        // which is a layer with nothing to draw — not a layer the compositor
        // should refuse the whole frame over.
        if !instances.iter().any(visible) {
            return None;
        }
        // Highlighted runs upload both paints as one glyph set and point the
        // covered instances at the second half. The atlas then stays valid for
        // the whole cue — keying it on the active word instead would rebuild
        // and re-upload it every time the word moved.
        let mut glyphs: Vec<RgbaImage> = painted.clusters.iter().map(|c| c.image.clone()).collect();
        if !lit.is_empty() {
            let rest_count = glyphs.len() as u32;
            glyphs.extend(lit);
            for instance in &mut instances {
                if covered.get(instance.glyph as usize) == Some(&true) {
                    instance.glyph += rest_count;
                }
            }
        }
        // One card: its bitmap placed at `offset` in run space.
        let card = |image: RgbaImage, offset: [f32; 2]| {
            let size = [
                image.width as f32 * scale[0],
                image.height as f32 * scale[1],
            ];
            let placement = LayerPlacement {
                center: [
                    origin[0] + offset[0] * scale[0] + size[0] * 0.5,
                    origin[1] + offset[1] * scale[1] + size[1] * 0.5,
                ],
                size,
                rotation: layer.rotation,
                opacity: layer.opacity,
            };
            (image, placement)
        };
        let mut cards = Vec::new();
        if let Some(background) = painted.background {
            cards.push(card(background, painted.background_offset));
        }
        if let Some((plate, offset)) = plate {
            cards.push(card(plate, offset));
        }
        Some(Realized::Glyphs {
            glyphs,
            instances,
            atlas_key: atlas_key(content, &style, highlight.as_ref()),
            cards,
            placement: LayerPlacement {
                center: aligned,
                size: extent_size,
                rotation: 0.0,
                opacity: 1.0,
            },
            effects,
            fx,
            color_grade,
            lut,
            blend_mode,
            styles,
        })
    } else {
        let image = text.rasterize(content, &style);
        if image.width == 0 || image.height == 0 {
            return None; // nothing rasterized (no fonts / empty run)
        }
        debug_assert!(
            image.width.max(image.height) as f32 <= crate::resolve::RASTER_EDGE_CAP + 1.0,
            "text raster {}×{} exceeds edge cap",
            image.width,
            image.height
        );
        let size = [
            image.width as f32 * scale[0],
            image.height as f32 * scale[1],
        ];
        let placement = LayerPlacement {
            center: layer.text_quad_center(&style, size, canvas),
            size,
            rotation: layer.rotation,
            opacity: layer.opacity,
        };
        Some(Realized::Bitmap {
            image,
            placement,
            uv: layer.uv,
            effects,
            fx,
            color_grade,
            lut,
            blend_mode,
            styles,
        })
    }
}

/// Whether one glyph instance would put ink on the canvas — the same test the
/// compositor applies before it uploads an instance.
fn visible(instance: &GlyphInstance) -> bool {
    instance.opacity > 0.0 && instance.size[0] > 0.0 && instance.size[1] > 0.0
}

/// Paint the run for a sampled caption highlight.
///
/// The range is clamped onto character boundaries of `content`: resolve maps it
/// through the casing transform, and a range that lands mid-character (a
/// hand-edited project, or a casing that changed a letter's byte length) must
/// highlight nothing rather than panic on a slice.
fn paint_highlight(
    text: &mut TextRenderer,
    content: &str,
    style: &TextStyle,
    highlight: &TextHighlight,
) -> cutlass_text::HighlightedText {
    let end = highlight.range.end.min(content.len());
    let start = highlight.range.start.min(end);
    let range = if content.is_char_boundary(start) && content.is_char_boundary(end) {
        start..end
    } else {
        0..0
    };
    let plate = highlight.plate.map(|rgba| cutlass_text::HighlightPlate {
        rgba,
        radius: highlight.plate_radius,
    });
    text.paint_highlighted(
        content,
        style,
        &cutlass_text::Highlight {
            range,
            fill: highlight.fill,
            plate,
        },
    )
}

/// Realize text for a transition side — bitmap path only.
///
/// Per-character animation on a transition edge is not a supported surface.
#[allow(clippy::too_many_arguments)] // mirrors former inline match-arm locals
pub(super) fn realize_text_bitmap(
    text: &mut TextRenderer,
    layer: &SceneLayer,
    content: &str,
    style: &TextStyle,
    canvas: [f32; 2],
    effects: Vec<ResolvedPass>,
    fx: LayerEffects,
    color_grade: Option<ColorGrade>,
    blend_mode: BlendMode,
    styles: LayerStyles,
) -> Result<Realized, RenderError> {
    let residual = match layer.size {
        SizeSpec::BitmapScaled(s) => s,
        SizeSpec::Fixed(_) => [1.0, 1.0],
    };
    let (style, scale, _, _) = fit_text_style(text, content, style, residual, 1.0);
    let image = text.rasterize(content, &style);
    if image.width == 0 || image.height == 0 {
        return Err(RenderError::unsupported("empty text layer"));
    }
    let size = [
        image.width as f32 * scale[0],
        image.height as f32 * scale[1],
    ];
    let placement = LayerPlacement {
        center: layer.text_quad_center(&style, size, canvas),
        size,
        rotation: layer.rotation,
        opacity: layer.opacity,
    };
    Ok(Realized::Bitmap {
        image,
        placement,
        uv: layer.uv,
        effects,
        fx,
        color_grade,
        lut: None,
        blend_mode,
        styles,
    })
}
