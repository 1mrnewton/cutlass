//! Text / glyph realize arms — bitmap (static) and per-character GPU glyphs.

use cutlass_compositor::{
    BlendMode, ColorGrade, LayerEffects, LayerPlacement, LayerStyles, RgbaImage,
};
use cutlass_text::{TextRenderer, TextStyle};

use crate::error::RenderError;
use crate::scene::{ResolvedPass, SceneLayer, SizeSpec, TextAnimation};

use super::super::text_anim::{atlas_key, cluster_deltas, extent_origin, place_clusters};
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
    canvas: [f32; 2],
    effects: Vec<ResolvedPass>,
    fx: LayerEffects,
    color_grade: Option<ColorGrade>,
    lut: Option<crate::scene::SceneLut>,
    blend_mode: BlendMode,
    styles: LayerStyles,
) -> Option<Realized> {
    let scale = match layer.size {
        SizeSpec::BitmapScaled(s) => s,
        SizeSpec::Fixed(_) => 1.0,
    };
    if let Some(anim) = animation {
        let shaped = text.shape(content, style);
        if !shaped.has_ink() {
            return None;
        }
        let painted = cutlass_text::paint_animated(&shaped, style);
        // Deltas use the ink-tight shaped clusters (logical
        // order / baselines); placement uses painted images.
        let deltas = cluster_deltas(&shaped, anim);
        let extent_size = [
            painted.extent.0 as f32 * scale,
            painted.extent.1 as f32 * scale,
        ];
        let aligned = layer.text_quad_center(style, extent_size, canvas);
        let origin = extent_origin(aligned, painted.extent, scale);
        let glyphs: Vec<RgbaImage> = painted.clusters.iter().map(|c| c.image.clone()).collect();
        // place_clusters reads offsets/baselines from ShapedText;
        // rebuild a shaped view over the painted clusters.
        let painted_shaped = cutlass_text::ShapedText {
            extent: painted.extent,
            clusters: painted.clusters.clone(),
        };
        let instances = place_clusters(
            &painted_shaped,
            &deltas,
            origin,
            scale,
            layer.rotation,
            layer.opacity,
        );
        if instances.is_empty() {
            return None;
        }
        let background = painted.background.map(|bg| {
            let size = [bg.width as f32 * scale, bg.height as f32 * scale];
            let center = [
                origin[0] + painted.background_offset[0] * scale + size[0] * 0.5,
                origin[1] + painted.background_offset[1] * scale + size[1] * 0.5,
            ];
            (
                bg,
                LayerPlacement {
                    center,
                    size,
                    rotation: layer.rotation,
                    opacity: layer.opacity,
                },
            )
        });
        Some(Realized::Glyphs {
            glyphs,
            instances,
            atlas_key: atlas_key(content, style),
            background,
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
        let image = text.rasterize(content, style);
        if image.width == 0 || image.height == 0 {
            return None; // nothing rasterized (no fonts / empty run)
        }
        let size = [image.width as f32 * scale, image.height as f32 * scale];
        let placement = LayerPlacement {
            center: layer.text_quad_center(style, size, canvas),
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
    let image = text.rasterize(content, style);
    if image.width == 0 || image.height == 0 {
        return Err(RenderError::unsupported("empty text layer"));
    }
    let scale = match layer.size {
        SizeSpec::BitmapScaled(s) => s,
        SizeSpec::Fixed(_) => 1.0,
    };
    let size = [image.width as f32 * scale, image.height as f32 * scale];
    let placement = LayerPlacement {
        center: layer.text_quad_center(style, size, canvas),
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
