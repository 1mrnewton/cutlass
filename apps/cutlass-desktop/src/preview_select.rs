//! Preview-viewport geometry: the canvas → viewport mapping, click
//! hit-testing, and the selection outline (preview roadmap Phase 2).
//!
//! The preview shows the composited frame aspect-fitted (`ImageFit.contain`)
//! inside a zoomable/pannable viewport. Hit-testing inverts that mapping into
//! canvas pixels, then asks [`crate::placement`] — the same geometry the
//! render resolver places layers with — whether a layer's rotated quad
//! contains the point, walking lanes top-first (CapCut: the topmost layer under
//! the cursor wins). The selection box runs the mapping forward to outline the
//! selected clip's placement in viewport coordinates.

use cutlass_compositor::{CompositorConfig, LayerPlacement};
use cutlass_models::{ClipTransform, CropRect};
use slint::Model;

use crate::placement::{anchor_canvas_position, generator_layer_placement, media_layer_placement};
use crate::{
    Clip, PreviewDragResolution, PreviewHit, PreviewSelectionBox, PreviewSpritePlacement, Sequence,
    TrackKind,
};

/// Aspect-fit (`ImageFit.contain`) mapping of the canvas into the viewport:
/// `(scale, offset_x, offset_y)` such that `view = canvas · scale + offset`.
pub(crate) fn contain_mapping(
    canvas_w: f32,
    canvas_h: f32,
    view_w: f32,
    view_h: f32,
) -> (f32, f32, f32) {
    if canvas_w <= 0.0 || canvas_h <= 0.0 || view_w <= 0.0 || view_h <= 0.0 {
        return (1.0, 0.0, 0.0);
    }
    let scale = (view_w / canvas_w).min(view_h / canvas_h);
    (
        scale,
        (view_w - canvas_w * scale) / 2.0,
        (view_h - canvas_h * scale) / 2.0,
    )
}

/// Zoom/pan-aware canvas mapping. `zoom = 1, pan = 0` matches
/// [`contain_mapping`]. Pan is in viewport logical px and moves the canvas
/// center relative to the viewport center.
pub(crate) fn viewport_mapping(
    canvas_w: f32,
    canvas_h: f32,
    view_w: f32,
    view_h: f32,
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
) -> (f32, f32, f32) {
    let (base_scale, _, _) = contain_mapping(canvas_w, canvas_h, view_w, view_h);
    let zoom = if zoom.is_finite() {
        zoom.max(0.01)
    } else {
        1.0
    };
    let scale = base_scale * zoom;
    (
        scale,
        (view_w - canvas_w * scale) / 2.0 + pan_x,
        (view_h - canvas_h * scale) / 2.0 + pan_y,
    )
}

pub(crate) fn canvas_config(sequence: &Sequence) -> CompositorConfig {
    CompositorConfig::new(
        sequence.width.max(1.0).round() as u32,
        sequence.height.max(1.0).round() as u32,
    )
}

/// Whether the composite path draws this clip at all: media, or a generator
/// the raster step supports. Effect/filter/adjustment clips (and stickers
/// with no valid asset) aren't composited, so they can't be picked (mirrors
/// `resolve_layers`).
pub(crate) fn is_composited(clip: &Clip) -> bool {
    !clip.media_id.is_empty()
        || matches!(
            clip.generator_kind.as_str(),
            "text" | "solid" | "rect" | "ellipse" | "sticker"
        )
}

/// Build the clip's current `ClipTransform` from projected fields.
pub(crate) fn clip_transform(clip: &Clip) -> ClipTransform {
    ClipTransform {
        position: [clip.transform_position_x, clip.transform_position_y],
        anchor_point: [clip.transform_anchor_x, clip.transform_anchor_y],
        scale: cutlass_models::Scale2 {
            x: clip.transform_scale,
            y: clip.transform_scale_y,
        },
        rotation: clip.transform_rotation,
        opacity: clip.transform_opacity,
    }
}

/// The clip's canvas placement, sized to its visible content.
///
/// Media clips use the shared placement helper (native size aspect-fit into
/// the canvas) — identical to what the render resolver places. The crop does
/// not shrink the quad on this branch: the compositor stretches the kept
/// region across the full-frame quad via UV (CapCut's kept-region re-fit
/// would land engine-side, if ever). Generators raster at canvas pixel scale
/// (fit 1:1) and hug the drawn-content bounds the projection measured — the
/// selection box and hit-test wrap the shape/text, not its transparent
/// raster (CapCut). Those bounds can exceed the canvas for text that
/// overflows the frame, so the box extends past the frame too. Cropped
/// generators and unknown bounds (0×0, e.g. empty text or a stale
/// projection) keep the full-canvas quad. Content size feeds the placement
/// math itself so non-center anchors pivot exactly like the renderer.
pub(crate) fn clip_placement(clip: &Clip, canvas: &CompositorConfig) -> LayerPlacement {
    let transform = clip_transform(clip);
    let has_size = clip.media_width > 0 && clip.media_height > 0;
    if !clip.media_id.is_empty() {
        let (w, h) = if has_size {
            (clip.media_width as u32, clip.media_height as u32)
        } else {
            // Media that vanished from the pool: degrade to canvas size.
            (canvas.width, canvas.height)
        };
        return media_layer_placement(&transform, w, h, canvas);
    }
    // Generators raster at canvas pixel scale (1:1), so their placement never
    // aspect-fits — same as the resolver's generator path.
    let (w, h) = if has_size && clip_crop(clip).is_full() {
        (clip.media_width as u32, clip.media_height as u32)
    } else {
        (canvas.width, canvas.height)
    };
    let mut placement = generator_layer_placement(&transform, w, h, canvas);
    let offset = text_alignment_offset(clip, &placement, canvas);
    placement.center[0] += offset[0];
    placement.center[1] += offset[1];
    placement
}

/// Canvas-layout offset applied after the ordinary clip transform for text.
/// Gesture code needs this separately: the visible text anchor includes the
/// offset, while `ClipTransform::position` stores only the transform portion.
pub(crate) fn text_alignment_offset(
    clip: &Clip,
    placement: &LayerPlacement,
    canvas: &CompositorConfig,
) -> [f32; 2] {
    if clip.generator_kind.as_str() != "text" {
        return [0.0, 0.0];
    }
    [
        match clip.text_style.align_h {
            0 => (placement.size[0] - canvas.width as f32) * 0.5,
            2 => (canvas.width as f32 - placement.size[0]) * 0.5,
            _ => 0.0,
        },
        match clip.text_style.align_v {
            0 => (placement.size[1] - canvas.height as f32) * 0.5,
            2 => (canvas.height as f32 - placement.size[1]) * 0.5,
            _ => 0.0,
        },
    ]
}

/// The clip's crop window. Projections written before crop existed (and
/// default-constructed test rows) carry an all-zero rect, which means "no
/// crop", not "keep nothing".
pub(crate) fn clip_crop(clip: &Clip) -> CropRect {
    if clip.crop_w > 0.0 && clip.crop_h > 0.0 {
        CropRect {
            x: clip.crop_x,
            y: clip.crop_y,
            w: clip.crop_w,
            h: clip.crop_h,
        }
    } else {
        CropRect::FULL
    }
}

pub(crate) fn covers_tick(clip: &Clip, tick: i32) -> bool {
    let start = clip.timeline_start.value;
    let end = start.saturating_add(clip.source_range.duration.value);
    start <= tick && tick < end
}

/// Point-in-rotated-rect, both in canvas pixels. Inverts the compositor's
/// clockwise rotation `R = [cos, -sin; sin, cos]` (+y down) about the center.
fn placement_contains(p: &LayerPlacement, x: f32, y: f32) -> bool {
    let dx = x - p.center[0];
    let dy = y - p.center[1];
    let (sin, cos) = p.rotation.sin_cos();
    let local_x = dx * cos + dy * sin;
    let local_y = -dx * sin + dy * cos;
    local_x.abs() <= p.size[0] / 2.0 && local_y.abs() <= p.size[1] / 2.0
}

/// Topmost visible, unlocked clip under `(x, y)` (viewport-element logical
/// px) at `tick`. Lanes walk top-first; hidden lanes aren't composited and
/// locked lanes don't hit-test (same rule as timeline selection), both fall
/// through to the layer below. Empty `clip_id` ⇔ miss.
#[allow(dead_code)]
pub fn hit_test(
    sequence: &Sequence,
    tick: i32,
    x: f32,
    y: f32,
    view_w: f32,
    view_h: f32,
) -> PreviewHit {
    hit_test_in_viewport(sequence, tick, x, y, view_w, view_h, 1.0, 0.0, 0.0)
}

#[allow(clippy::too_many_arguments)]
pub fn hit_test_in_viewport(
    sequence: &Sequence,
    tick: i32,
    x: f32,
    y: f32,
    view_w: f32,
    view_h: f32,
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
) -> PreviewHit {
    let canvas = canvas_config(sequence);
    let (cw, ch) = (canvas.width as f32, canvas.height as f32);
    let (scale, ox, oy) = viewport_mapping(cw, ch, view_w, view_h, zoom, pan_x, pan_y);
    if scale <= 0.0 {
        return PreviewHit::default();
    }
    let px = (x - ox) / scale;
    let py = (y - oy) / scale;
    if px < 0.0 || py < 0.0 || px > cw || py > ch {
        return PreviewHit::default(); // letterbox bar
    }

    for row in 0..sequence.tracks.row_count() {
        let Some(track) = sequence.tracks.row_data(row) else {
            continue;
        };
        if track.kind == TrackKind::Audio || !track.enabled || track.locked {
            continue;
        }
        for idx in 0..track.clips.row_count() {
            let Some(mut clip) = track.clips.row_data(idx) else {
                continue;
            };
            if !covers_tick(&clip, tick) || !is_composited(&clip) {
                continue;
            }
            // Animated clips are picked where the playhead renders them.
            crate::params::apply_sampled_transform(&mut clip, tick);
            if placement_contains(&clip_placement(&clip, &canvas), px, py) {
                return PreviewHit {
                    track_id: track.id.clone(),
                    clip_id: clip.id.clone(),
                };
            }
        }
    }
    PreviewHit::default()
}

/// Whether `(x, y)` (viewport-element px) lands on `clip_id`'s placement quad,
/// *ignoring* the letterbox guard that [`hit_test_in_viewport`] applies. The
/// normal hit-test deselects out in the letterbox bars, but a selected clip
/// whose content overflows the frame (e.g. a long title from the text-overflow
/// raster) extends out there — this lets a press on that overflow grab the clip
/// to drag it back into view instead of clearing the selection. Gated to the
/// same visual/enabled/unlocked, composited, under-the-playhead rules as
/// picking, so only a genuinely grabbable selected clip qualifies.
#[allow(clippy::too_many_arguments)]
pub fn selected_clip_contains_in_viewport(
    sequence: &Sequence,
    clip_id: &str,
    tick: i32,
    x: f32,
    y: f32,
    view_w: f32,
    view_h: f32,
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
) -> bool {
    if clip_id.is_empty() {
        return false;
    }
    let canvas = canvas_config(sequence);
    let (cw, ch) = (canvas.width as f32, canvas.height as f32);
    let (scale, ox, oy) = viewport_mapping(cw, ch, view_w, view_h, zoom, pan_x, pan_y);
    if scale <= 0.0 {
        return false;
    }
    // No letterbox guard here — that's the whole point.
    let px = (x - ox) / scale;
    let py = (y - oy) / scale;

    for row in 0..sequence.tracks.row_count() {
        let Some(track) = sequence.tracks.row_data(row) else {
            continue;
        };
        if track.kind == TrackKind::Audio || !track.enabled || track.locked {
            continue;
        }
        for idx in 0..track.clips.row_count() {
            let Some(mut clip) = track.clips.row_data(idx) else {
                continue;
            };
            if clip.id != clip_id {
                continue;
            }
            if !covers_tick(&clip, tick) || !is_composited(&clip) {
                return false;
            }
            crate::params::apply_sampled_transform(&mut clip, tick);
            return placement_contains(&clip_placement(&clip, &canvas), px, py);
        }
    }
    false
}

/// How far below the box's bottom edge the rotate affordance floats, in
/// viewport px (constant UI size regardless of zoom/letterbox — CapCut).
const ROTATE_HANDLE_OFFSET_PX: f32 = 26.0;

/// The placement's quad corners mapped into viewport coordinates, clockwise
/// from the content's top-left (rotation applied about the center).
fn placement_corners(p: &LayerPlacement, scale: f32, ox: f32, oy: f32) -> [[f32; 2]; 4] {
    let (sin, cos) = p.rotation.sin_cos();
    let (hw, hh) = (p.size[0] / 2.0, p.size[1] / 2.0);
    [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)].map(|(lx, ly)| {
        // Clockwise rotation in +y-down screen coords (same matrix as the
        // compositor's placement uniforms), then canvas → viewport.
        let x = p.center[0] + lx * cos - ly * sin;
        let y = p.center[1] + lx * sin + ly * cos;
        [ox + x * scale, oy + y * scale]
    })
}

/// Selection outline for `clip_id` in viewport-element coordinates.
/// Invisible when the id is empty/unknown, the clip isn't under the
/// playhead, or its lane is hidden — the layer has no pixels on screen.
///
/// During a transform gesture the projection still holds the press-time
/// transform (the live value is a worker-side override, by design), so the
/// panel passes the gesture's resolution to keep the box glued to the
/// content — position for moves, scale for corner drags, rotation for the
/// rotate affordance.
#[allow(dead_code)]
pub fn selection_box(
    sequence: &Sequence,
    clip_id: &str,
    tick: i32,
    view_w: f32,
    view_h: f32,
    gesture: Option<&PreviewDragResolution>,
) -> PreviewSelectionBox {
    selection_box_in_viewport(
        sequence, clip_id, tick, view_w, view_h, 1.0, 0.0, 0.0, gesture,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn selection_box_in_viewport(
    sequence: &Sequence,
    clip_id: &str,
    tick: i32,
    view_w: f32,
    view_h: f32,
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
    gesture: Option<&PreviewDragResolution>,
) -> PreviewSelectionBox {
    if clip_id.is_empty() {
        return PreviewSelectionBox::default();
    }
    let canvas = canvas_config(sequence);
    let (cw, ch) = (canvas.width as f32, canvas.height as f32);
    let (scale, ox, oy) = viewport_mapping(cw, ch, view_w, view_h, zoom, pan_x, pan_y);

    for row in 0..sequence.tracks.row_count() {
        let Some(track) = sequence.tracks.row_data(row) else {
            continue;
        };
        if track.kind == TrackKind::Audio || !track.enabled {
            continue;
        }
        for idx in 0..track.clips.row_count() {
            let Some(mut clip) = track.clips.row_data(idx) else {
                continue;
            };
            if clip.id != clip_id {
                continue;
            }
            if !covers_tick(&clip, tick) || !is_composited(&clip) {
                return PreviewSelectionBox::default();
            }
            // Box follows the rendered frame on animated clips; a live
            // gesture's resolution then wins (it previews via override).
            crate::params::apply_sampled_transform(&mut clip, tick);
            if let Some(res) = gesture {
                clip.transform_position_x = res.position_x;
                clip.transform_position_y = res.position_y;
                clip.transform_anchor_x = res.anchor_x;
                clip.transform_anchor_y = res.anchor_y;
                clip.transform_scale = res.scale;
                clip.transform_scale_y = res.scale_y;
                clip.transform_scale_linked = res.scale == res.scale_y;
                clip.transform_rotation = res.rotation;
            }
            let p = clip_placement(&clip, &canvas);
            let [c0, c1, c2, c3] = placement_corners(&p, scale, ox, oy);
            let transform = clip_transform(&clip);
            let anchor = anchor_canvas_position(&transform, &p);
            let ax = ox + anchor[0] * scale;
            let ay = oy + anchor[1] * scale;
            // Rotate affordance: floats a constant viewport distance below
            // the content's bottom edge (between c3 and c2), riding the
            // box's rotation. Outward = the edge direction rotated +90°
            // (y-down), which points away from the content for any angle.
            let mid = [(c2[0] + c3[0]) / 2.0, (c2[1] + c3[1]) / 2.0];
            let edge = [c2[0] - c3[0], c2[1] - c3[1]];
            let len = edge[0].hypot(edge[1]).max(f32::EPSILON);
            let out = [-edge[1] / len, edge[0] / len];
            return PreviewSelectionBox {
                visible: true,
                x0: c0[0],
                y0: c0[1],
                x1: c1[0],
                y1: c1[1],
                x2: c2[0],
                y2: c2[1],
                x3: c3[0],
                y3: c3[1],
                hx: mid[0] + out[0] * ROTATE_HANDLE_OFFSET_PX,
                hy: mid[1] + out[1] * ROTATE_HANDLE_OFFSET_PX,
                ax,
                ay,
            };
        }
    }
    PreviewSelectionBox::default()
}

fn apply_identity_transform(clip: &mut Clip) {
    clip.transform_position_x = 0.0;
    clip.transform_position_y = 0.0;
    clip.transform_anchor_x = 0.5;
    clip.transform_anchor_y = 0.5;
    clip.transform_scale = 1.0;
    clip.transform_scale_y = 1.0;
    clip.transform_scale_linked = true;
    clip.transform_rotation = 0.0;
}

fn apply_gesture_transform(clip: &mut Clip, gesture: &PreviewDragResolution) {
    clip.transform_position_x = gesture.position_x;
    clip.transform_position_y = gesture.position_y;
    clip.transform_anchor_x = gesture.anchor_x;
    clip.transform_anchor_y = gesture.anchor_y;
    // Corner-drag gizmo stays uniform.
    clip.transform_scale = gesture.scale;
    clip.transform_scale_y = gesture.scale_y;
    clip.transform_scale_linked = gesture.scale == gesture.scale_y;
    clip.transform_rotation = gesture.rotation;
}

fn fitted_frame_rect(
    canvas_w: f32,
    canvas_h: f32,
    view_w: f32,
    view_h: f32,
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
) -> (f32, f32, f32, f32) {
    let aspect = if canvas_h > 0.0 {
        canvas_w / canvas_h
    } else {
        1.0
    };
    let fitted_w = view_w.min(view_h * aspect);
    let fitted_h = if aspect > 0.0 {
        fitted_w / aspect
    } else {
        fitted_w
    };
    let zoom = if zoom.is_finite() {
        zoom.max(0.01)
    } else {
        1.0
    };
    let frame_w = fitted_w * zoom;
    let frame_h = fitted_h * zoom;
    let frame_x = (view_w - frame_w) * 0.5 + pan_x;
    let frame_y = (view_h - frame_h) * 0.5 + pan_y;
    (frame_x, frame_y, frame_w, frame_h)
}

/// Sprite image placement during a zero-drift transform gesture.
#[allow(clippy::too_many_arguments)]
pub fn sprite_placement_in_viewport(
    sequence: &Sequence,
    clip_id: &str,
    tick: i32,
    view_w: f32,
    view_h: f32,
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
    gesture: Option<&PreviewDragResolution>,
) -> PreviewSpritePlacement {
    let gesture = gesture.filter(|res| res.valid);
    if clip_id.is_empty() {
        return PreviewSpritePlacement::default();
    }

    let canvas = canvas_config(sequence);
    let (cw, ch) = (canvas.width as f32, canvas.height as f32);
    let (scale, ox, oy) = viewport_mapping(cw, ch, view_w, view_h, zoom, pan_x, pan_y);
    if scale <= 0.0 {
        return PreviewSpritePlacement::default();
    }
    let (frame_x, frame_y, frame_w, frame_h) =
        fitted_frame_rect(cw, ch, view_w, view_h, zoom, pan_x, pan_y);

    for row in 0..sequence.tracks.row_count() {
        let Some(track) = sequence.tracks.row_data(row) else {
            continue;
        };
        if track.kind == TrackKind::Audio || !track.enabled {
            continue;
        }
        for idx in 0..track.clips.row_count() {
            let Some(mut clip) = track.clips.row_data(idx) else {
                continue;
            };
            if clip.id != clip_id {
                continue;
            }
            if !covers_tick(&clip, tick) || !is_composited(&clip) {
                return PreviewSpritePlacement::default();
            }

            crate::params::apply_sampled_transform(&mut clip, tick);
            let mut identity_clip = clip.clone();
            apply_identity_transform(&mut identity_clip);
            let p_id = clip_placement(&identity_clip, &canvas);

            let opacity = if let Some(gesture) = gesture {
                apply_gesture_transform(&mut clip, gesture);
                gesture.opacity
            } else {
                clip.transform_opacity
            };
            let p_g = clip_placement(&clip, &canvas);

            let id_center = [ox + p_id.center[0] * scale, oy + p_id.center[1] * scale];
            let ges_center = [ox + p_g.center[0] * scale, oy + p_g.center[1] * scale];
            let size_scale = if p_id.size[0] > f32::EPSILON {
                p_g.size[0] / p_id.size[0]
            } else {
                1.0
            };
            let origin_x = id_center[0] - frame_x;
            let origin_y = id_center[1] - frame_y;
            let x = frame_x + (ges_center[0] - id_center[0]);
            let y = frame_y + (ges_center[1] - id_center[1]);
            let rotation = (p_g.rotation - p_id.rotation).to_degrees();

            return PreviewSpritePlacement {
                visible: true,
                x,
                y,
                width: frame_w,
                height: frame_h,
                origin_x,
                origin_y,
                rotation_degrees: rotation,
                scale: size_scale,
                opacity,
            };
        }
    }
    PreviewSpritePlacement::default()
}

#[cfg(test)]
mod tests;
