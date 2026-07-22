use cutlass_compositor::ColorGrade;
use cutlass_models::{Param, Shape, ShapePath, ShapeStroke};
use cutlass_shapes::{BezierPath, PathPoint, SDF_AA, SdfParams, Stroke};

use crate::scene::{LayerSource, ResolvedPass, SceneLayer, SizeSpec};

/// Resolve one shape generator at `tick` into a placed layer.
///
/// All `Param` curves are sampled here (the resolver is the "animation →
/// values" boundary), and every length is converted to canvas pixels with
/// `px_scale` (reference scale × the clip's animated transform scale), so
/// downstream stages see plain numbers. Parametric shapes become SDF layers
/// whose quad is padded for stroke overhang + anti-aliasing; pen paths become
/// CPU-raster layers that scale like text bitmaps.
#[allow(clippy::too_many_arguments)]
pub(super) fn resolve_shape(
    shape: &Shape,
    rgba: &Param<[u8; 4]>,
    width: &Param<f32>,
    height: &Param<f32>,
    corner_radius: &Param<f32>,
    stroke: Option<&ShapeStroke>,
    tick: i64,
    px_scale: f32,
    center: [f32; 2],
    anchor_point: [f32; 2],
    rotation: f32,
    opacity: f32,
    uv: [f32; 4],
    color_grade: Option<ColorGrade>,
    transform_scale: f32,
    effects: Vec<ResolvedPass>,
) -> Option<SceneLayer> {
    let fill = rgba.sample(tick);
    let stroke_px = stroke.map(|s| Stroke {
        rgba: s.rgba.sample(tick),
        width: (s.width.sample(tick) * px_scale).max(0.0),
    });

    // Pen paths: rasterized on the CPU at the *reference* scale so the memo
    // stays warm under transform-scale animation (the quad magnifies the
    // bitmap, like text). `px_scale / transform_scale` recovers ref_scale.
    if let Shape::Path(path) = shape {
        let bezier = to_bezier(path);
        if !bezier.is_drawable() {
            return None;
        }
        let raster_scale = if transform_scale > 0.0 {
            px_scale / transform_scale
        } else {
            px_scale
        };
        return Some(SceneLayer {
            clip: None,
            source: LayerSource::PathShape {
                path: bezier,
                fill,
                // Raster-space stroke: `PathRaster` folds `raster_scale` into
                // the width itself, so hand it the unscaled model value.
                stroke: stroke.map(|s| Stroke {
                    rgba: s.rgba.sample(tick),
                    width: s.width.sample(tick).max(0.0),
                }),
                raster_scale,
            },
            center,
            anchor_point,
            size: SizeSpec::BitmapScaled(transform_scale),
            rotation,
            opacity,
            uv,
            effects,
            mask: None,
            chroma_key: None,
            color_grade,
            lut: None,
        });
    }

    let w = width.sample(tick) * px_scale;
    let h = height.sample(tick) * px_scale;
    if w <= 0.0 || h <= 0.0 {
        return None;
    }
    let radius = (corner_radius.sample(tick) * px_scale).max(0.0);

    // Plain rectangles keep the no-texture solid fast path.
    if matches!(shape, Shape::Rectangle) && radius == 0.0 && stroke_px.is_none() {
        return Some(SceneLayer {
            clip: None,
            source: LayerSource::Solid(fill),
            center,
            anchor_point,
            size: SizeSpec::Fixed([w, h]),
            rotation,
            opacity,
            uv,
            effects,
            mask: None,
            chroma_key: None,
            color_grade,
            lut: None,
        });
    }

    let params = match shape {
        Shape::Rectangle => SdfParams::RoundedRect { radius },
        Shape::Ellipse => SdfParams::Ellipse,
        Shape::Polygon { sides } => SdfParams::polygon(*sides, radius),
        Shape::Star {
            points,
            inner_ratio,
        } => SdfParams::Star {
            points: *points,
            inner: inner_ratio.sample(tick).clamp(0.0, 1.0),
            round: radius,
        },
        Shape::Line => SdfParams::Line,
        Shape::Arrow => SdfParams::Arrow,
        Shape::Heart => SdfParams::Heart,
        Shape::Path(_) => unreachable!("handled above"),
    };

    // The quad must cover the stroke's outward half plus the AA ramp, or the
    // shader's ink clips at the quad edge (same margin as the CPU raster).
    let pad = stroke_px.map_or(0.0, |s| s.width * 0.5) + 2.0 * SDF_AA;
    Some(SceneLayer {
        clip: None,
        source: LayerSource::Shape {
            params,
            fill,
            stroke: stroke_px,
            pad,
        },
        center,
        anchor_point,
        size: SizeSpec::Fixed([w + 2.0 * pad, h + 2.0 * pad]),
        rotation,
        opacity,
        uv,
        effects,
        mask: None,
        chroma_key: None,
        color_grade,
        lut: None,
    })
}

/// Convert the model's serialized path into the shapes crate's bezier form.
fn to_bezier(path: &ShapePath) -> BezierPath {
    BezierPath {
        points: path
            .points
            .iter()
            .map(|p| PathPoint {
                anchor: p.anchor,
                handle_in: p.handle_in,
                handle_out: p.handle_out,
            })
            .collect(),
        closed: path.closed,
    }
}
