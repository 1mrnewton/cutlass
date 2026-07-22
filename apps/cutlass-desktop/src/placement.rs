//! Clip → canvas placement math for preview hit-testing and gestures.
//!
//! Mirrors `cutlass-render`'s resolver (`resolve.rs`), which is what the
//! compositor actually draws on this branch: content is placed centered at
//! `canvas_center + position · canvas`, media aspect-fits the canvas at
//! scale 1.0, generators raster at canvas pixel scale (1:1), rotation is
//! clockwise about the placed quad. Hit boxes and selection outlines built
//! from these functions therefore agree with rendered pixels.
//!
//! The anchor helpers implement the model's documented `anchor_point`
//! semantics (pivot within the content bounds): the anchor sits at
//! `canvas_center + position · canvas` and the quad center offsets from it
//! by the rotated anchor→center vector — the same math the resolver now
//! applies via `SceneLayer::quad_center`, so gesture boxes and rendered
//! pixels agree for any anchor.
//!
//! Crop: main's engine re-fit *cropped* media to the canvas (kept region cut
//! out and aspect-fit — CapCut semantics). This branch's resolver keeps the
//! full-frame quad and stretches the kept region across it via UV, so
//! placement here deliberately ignores the crop; revisit if the resolver
//! ever adopts kept-region re-fit.

use cutlass_compositor::{CompositorConfig, LayerPlacement};
use cutlass_models::ClipTransform;

/// Placement for media content of `content_w × content_h` native pixels:
/// aspect-fit into the canvas at scale 1.0 (the resolver's `fit_scale`),
/// then the clip transform applies.
pub(crate) fn media_layer_placement(
    transform: &ClipTransform,
    content_w: u32,
    content_h: u32,
    canvas: &CompositorConfig,
) -> LayerPlacement {
    let (cw, ch) = (canvas.width as f32, canvas.height as f32);
    let (w, h) = (content_w as f32, content_h as f32);
    let fit = if w > 0.0 && h > 0.0 {
        (cw / w).min(ch / h)
    } else {
        1.0
    };
    placement_from_size(transform, w, h, fit, canvas)
}

/// Placement for a generator raster authored at canvas pixel scale: its
/// pixels map 1:1 onto the canvas (no aspect-fit), centered, then the clip
/// transform applies. Text rasters can exceed the canvas to hold content
/// that overflows the frame.
pub(crate) fn generator_layer_placement(
    transform: &ClipTransform,
    content_w: u32,
    content_h: u32,
    canvas: &CompositorConfig,
) -> LayerPlacement {
    placement_from_size(transform, content_w as f32, content_h as f32, 1.0, canvas)
}

/// Shared geometry: place content of size `w × h` at `fit · transform.scale`
/// (per-axis), centered on the canvas and offset/rotated by the transform.
/// Non-center anchors offset the quad so the anchor stays at the transform's
/// position — the same derivation as the renderer's `SceneLayer::quad_center`.
fn placement_from_size(
    transform: &ClipTransform,
    w: f32,
    h: f32,
    fit: f32,
    canvas: &CompositorConfig,
) -> LayerPlacement {
    let (cw, ch) = (canvas.width as f32, canvas.height as f32);
    // Per-axis placement: width × x, height × y.
    let size = [w * fit * transform.scale.x, h * fit * transform.scale.y];
    let anchor = [
        cw * 0.5 + transform.position[0] * cw,
        ch * 0.5 + transform.position[1] * ch,
    ];
    let to_center = [
        (0.5 - transform.anchor_point[0]) * size[0],
        (0.5 - transform.anchor_point[1]) * size[1],
    ];
    let (sin, cos) = transform.rotation.to_radians().sin_cos();
    let center = [
        anchor[0] + to_center[0] * cos - to_center[1] * sin,
        anchor[1] + to_center[0] * sin + to_center[1] * cos,
    ];
    LayerPlacement {
        center,
        size,
        rotation: transform.rotation.to_radians(),
        opacity: transform.opacity.clamp(0.0, 1.0),
    }
}

/// The anchor pivot in canvas pixels for a placed layer — the point scale and
/// rotation gestures pivot about, and what `ClipTransform::position` places.
pub(crate) fn anchor_canvas_position(
    transform: &ClipTransform,
    placement: &LayerPlacement,
) -> [f32; 2] {
    let offset = [
        (transform.anchor_point[0] - 0.5) * placement.size[0],
        (transform.anchor_point[1] - 0.5) * placement.size[1],
    ];
    let (sin, cos) = placement.rotation.sin_cos();
    [
        placement.center[0] + offset[0] * cos - offset[1] * sin,
        placement.center[1] + offset[0] * sin + offset[1] * cos,
    ]
}

/// Given a fixed content center and placed size, derive the normalized
/// anchor + position that keep the frame unchanged while moving the pivot to
/// `anchor_canvas`.
pub(crate) fn reposition_anchor(
    anchor_canvas: [f32; 2],
    center: [f32; 2],
    size: [f32; 2],
    rotation_deg: f32,
    canvas: &CompositorConfig,
) -> ([f32; 2], [f32; 2]) {
    let (cw, ch) = (canvas.width as f32, canvas.height as f32);
    let (sin, cos) = rotation_deg.to_radians().sin_cos();
    let delta = [center[0] - anchor_canvas[0], center[1] - anchor_canvas[1]];
    // Invert the clockwise rotation used by placement (same matrix as hit-test).
    let to_center = [
        delta[0] * cos + delta[1] * sin,
        -delta[0] * sin + delta[1] * cos,
    ];
    let anchor_point = [0.5 - to_center[0] / size[0], 0.5 - to_center[1] / size[1]];
    let position = [
        (anchor_canvas[0] - cw * 0.5) / cw,
        (anchor_canvas[1] - ch * 0.5) / ch,
    ];
    (anchor_point, position)
}

/// Canvas `position` that keeps `center` fixed when `anchor_point` changes
/// (inspector anchor sliders and keyframe edits that should not shift pixels).
pub(crate) fn position_preserving_center(
    center: [f32; 2],
    size: [f32; 2],
    anchor_point: [f32; 2],
    rotation_deg: f32,
    canvas: &CompositorConfig,
) -> [f32; 2] {
    let to_center = [
        (0.5 - anchor_point[0]) * size[0],
        (0.5 - anchor_point[1]) * size[1],
    ];
    let (sin, cos) = rotation_deg.to_radians().sin_cos();
    let anchor = [
        center[0] - (to_center[0] * cos - to_center[1] * sin),
        center[1] - (to_center[0] * sin + to_center[1] * cos),
    ];
    let (cw, ch) = (canvas.width as f32, canvas.height as f32);
    [(anchor[0] - cw * 0.5) / cw, (anchor[1] - ch * 0.5) / ch]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canvas() -> CompositorConfig {
        CompositorConfig::new(1920, 1080)
    }

    fn approx2(a: [f32; 2], b: [f32; 2]) {
        assert!(
            (a[0] - b[0]).abs() < 1e-3 && (a[1] - b[1]).abs() < 1e-3,
            "expected ~{b:?}, got {a:?}"
        );
    }

    #[test]
    fn identity_media_placement_matches_the_resolver() {
        // Full-frame media, identity transform: centered, canvas-sized —
        // exactly `resolve.rs`' single_video_clip_is_centered_and_aspect_fit.
        let p = media_layer_placement(&ClipTransform::IDENTITY, 1920, 1080, &canvas());
        approx2(p.center, [960.0, 540.0]);
        approx2(p.size, [1920.0, 1080.0]);
        assert_eq!(p.rotation, 0.0);
    }

    #[test]
    fn position_and_scale_offset_the_center_like_the_resolver() {
        // Mirrors resolve.rs' transform_offsets_center_and_scales_size.
        let t = ClipTransform {
            position: [0.25, -0.1],
            scale: 2.0.into(),
            opacity: 0.5,
            ..ClipTransform::IDENTITY
        };
        let p = media_layer_placement(&t, 1920, 1080, &canvas());
        approx2(p.center, [1920.0 * 0.75, 1080.0 * 0.4]);
        approx2(p.size, [3840.0, 2160.0]);
        assert_eq!(p.opacity, 0.5);
    }

    #[test]
    fn portrait_media_aspect_fits_the_canvas() {
        // 1080×1920 into 1920×1080: fit = min(1920/1080, 1080/1920) = 0.5625.
        let p = media_layer_placement(&ClipTransform::IDENTITY, 1080, 1920, &canvas());
        approx2(p.size, [607.5, 1080.0]);
        approx2(p.center, [960.0, 540.0]);
    }

    #[test]
    fn generator_places_pixels_one_to_one() {
        let p = generator_layer_placement(&ClipTransform::IDENTITY, 1920, 1080, &canvas());
        approx2(p.size, [1920.0, 1080.0]);
        // Oversized text raster keeps its pixel size (no fit-down).
        let p = generator_layer_placement(&ClipTransform::IDENTITY, 2400, 300, &canvas());
        approx2(p.size, [2400.0, 300.0]);
        approx2(p.center, [960.0, 540.0]);
    }

    #[test]
    fn center_anchor_pivot_is_the_center() {
        let t = ClipTransform::IDENTITY;
        let p = media_layer_placement(&t, 1920, 1080, &canvas());
        approx2(anchor_canvas_position(&t, &p), p.center);
    }

    #[test]
    fn anchor_roundtrip_preserves_the_center() {
        // Move the pivot to the content's top-left quadrant; the derived
        // anchor+position must keep the placed center where it was.
        let t = ClipTransform::IDENTITY;
        let c = canvas();
        let p = media_layer_placement(&t, 1920, 1080, &c);
        let (anchor_point, position) = reposition_anchor([480.0, 270.0], p.center, p.size, 0.0, &c);
        let moved = ClipTransform {
            position,
            anchor_point,
            ..t
        };
        let p2 = media_layer_placement(&moved, 1920, 1080, &c);
        approx2(p2.center, p.center);
        // And the pivot now reads back at the requested canvas point.
        approx2(anchor_canvas_position(&moved, &p2), [480.0, 270.0]);
    }

    #[test]
    fn position_preserving_center_undoes_anchor_shift() {
        let c = canvas();
        let t = ClipTransform {
            rotation: 30.0,
            ..ClipTransform::IDENTITY
        };
        let p = media_layer_placement(&t, 1920, 1080, &c);
        let position = position_preserving_center(p.center, p.size, [0.2, 0.8], 30.0, &c);
        let moved = ClipTransform {
            position,
            anchor_point: [0.2, 0.8],
            ..t
        };
        let p2 = media_layer_placement(&moved, 1920, 1080, &c);
        approx2(p2.center, p.center);
    }
}
