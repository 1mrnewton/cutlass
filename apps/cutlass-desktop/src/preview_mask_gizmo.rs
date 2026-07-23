//! On-canvas mask gizmo geometry for the preview viewport.
//!
//! Mask params live in **layer fractions** (see [`cutlass_models::Mask`]):
//! center is an offset from the layer center, size scales the mask
//! (`[1,1]` = full layer), rotation is degrees about the mask center.
//! This module composes those with the clip's [`LayerPlacement`] and the
//! preview's letterbox/zoom/pan mapping to produce outline + handle
//! positions in **viewport-element logical px**, plus hit-testing.
//!
//! Live ParamOverride values are passed in explicitly (projection does not
//! republish during an override) so the gizmo tracks slider/gizmo drags.

use cutlass_compositor::LayerPlacement;
use cutlass_models::MaskKind;
use slint::SharedString;

use crate::params::{sampled_scalar_param, sampled_vec2_param};
use crate::preview_motion_path::find_projected_clip;
use crate::preview_select::{
    canvas_config, clip_placement, covers_tick, is_composited, viewport_mapping,
};
use crate::{Clip, MaskGizmo, MaskGizmoDragResolution, Sequence};

/// Hit-test / drag handle ids published to Slint (`0` = none).
pub const HANDLE_NONE: i32 = 0;
pub const HANDLE_CENTER: i32 = 1;
pub const HANDLE_BODY: i32 = 2;
pub const HANDLE_SIZE_X: i32 = 3;
pub const HANDLE_SIZE_Y: i32 = 4;
pub const HANDLE_ROTATION: i32 = 5;
pub const HANDLE_FEATHER: i32 = 6;
pub const HANDLE_ROUNDNESS: i32 = 7;

/// Default hit radius in viewport px.
pub const DEFAULT_HIT_TOLERANCE_PX: f32 = 12.0;
/// Rotation handle floats this many viewport px past the size-Y edge.
const ROTATION_HANDLE_GAP_PX: f32 = 28.0;
/// Feather handle sits past the size-X edge by this base + feather×range.
const FEATHER_HANDLE_BASE_PX: f32 = 16.0;
const FEATHER_HANDLE_RANGE_PX: f32 = 36.0;
/// Samples for ellipse / closed outlines.
const OUTLINE_SEGMENTS: usize = 48;

/// Sampled (or live-overridden) mask geometry in model units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaskGizmoParams {
    pub kind: MaskKind,
    pub center: [f32; 2],
    pub size: [f32; 2],
    pub rotation_deg: f32,
    pub feather: f32,
    pub roundness: f32,
}

/// Intermediate geometry used by hit-testing (viewport px).
#[derive(Debug, Clone)]
struct GizmoLayout {
    kind: MaskKind,
    params: MaskGizmoParams,
    center_v: [f32; 2],
    size_x_v: [f32; 2],
    size_y_v: [f32; 2],
    rotation_v: [f32; 2],
    feather_v: [f32; 2],
    roundness_v: [f32; 2],
    has_size_x: bool,
    has_size_y: bool,
    has_roundness: bool,
    /// Half-extents of the mask in layer px (pre-placement).
    half_layer: [f32; 2],
    placement: LayerPlacement,
    scale: f32,
    ox: f32,
    oy: f32,
}

fn parse_kind(id: &str) -> Option<MaskKind> {
    match id {
        "linear" => Some(MaskKind::Linear),
        "mirror" => Some(MaskKind::Mirror),
        "circle" => Some(MaskKind::Circle),
        "rectangle" => Some(MaskKind::Rectangle),
        "heart" => Some(MaskKind::Heart),
        "star" => Some(MaskKind::Star),
        _ => None,
    }
}

fn deg_to_rad(deg: f32) -> f32 {
    deg.to_radians()
}

/// Clockwise rotation in +y-down screen space (matches compositor placement).
fn rotate(v: [f32; 2], rad: f32) -> [f32; 2] {
    let (sin, cos) = rad.sin_cos();
    [v[0] * cos - v[1] * sin, v[0] * sin + v[1] * cos]
}

fn inv_rotate(v: [f32; 2], rad: f32) -> [f32; 2] {
    rotate(v, -rad)
}

/// Layer-local px (origin at layer center) → canvas px.
pub fn layer_to_canvas(local: [f32; 2], placement: &LayerPlacement) -> [f32; 2] {
    let r = rotate(local, placement.rotation);
    [placement.center[0] + r[0], placement.center[1] + r[1]]
}

/// Canvas px → layer-local px.
pub fn canvas_to_layer(canvas: [f32; 2], placement: &LayerPlacement) -> [f32; 2] {
    let d = [
        canvas[0] - placement.center[0],
        canvas[1] - placement.center[1],
    ];
    inv_rotate(d, placement.rotation)
}

/// Canvas px → viewport-element logical px.
pub fn canvas_to_viewport(canvas: [f32; 2], scale: f32, ox: f32, oy: f32) -> [f32; 2] {
    [ox + canvas[0] * scale, oy + canvas[1] * scale]
}

/// Viewport-element logical px → canvas px.
pub fn viewport_to_canvas(view: [f32; 2], scale: f32, ox: f32, oy: f32) -> [f32; 2] {
    [(view[0] - ox) / scale, (view[1] - oy) / scale]
}

/// Mask center in layer fractions → layer-local px.
pub fn center_fractions_to_layer(center: [f32; 2], layer_size: [f32; 2]) -> [f32; 2] {
    [center[0] * layer_size[0], center[1] * layer_size[1]]
}

/// Layer-local px → mask center fractions.
pub fn layer_to_center_fractions(local: [f32; 2], layer_size: [f32; 2]) -> [f32; 2] {
    [
        if layer_size[0].abs() > 1e-6 {
            local[0] / layer_size[0]
        } else {
            0.0
        },
        if layer_size[1].abs() > 1e-6 {
            local[1] / layer_size[1]
        } else {
            0.0
        },
    ]
}

/// Half-extents of the mask shape in layer px ( CapCut / shader semantics).
fn mask_half_layer(size: [f32; 2], layer_size: [f32; 2]) -> [f32; 2] {
    [size[0] * layer_size[0] * 0.5, size[1] * layer_size[1] * 0.5]
}

/// Point in mask-aligned layer px (relative to mask center, pre-rotation)
/// → viewport px.
fn mask_offset_to_viewport(
    offset: [f32; 2],
    params: &MaskGizmoParams,
    placement: &LayerPlacement,
    scale: f32,
    ox: f32,
    oy: f32,
) -> [f32; 2] {
    let layer_size = placement.size;
    let center_local = center_fractions_to_layer(params.center, layer_size);
    let rotated = rotate(offset, deg_to_rad(params.rotation_deg));
    let layer = [center_local[0] + rotated[0], center_local[1] + rotated[1]];
    let canvas = layer_to_canvas(layer, placement);
    canvas_to_viewport(canvas, scale, ox, oy)
}

fn sample_mask_params(clip: &Clip, playhead: i32) -> Option<MaskGizmoParams> {
    let kind = parse_kind(clip.mask_kind.as_str())?;
    let center = sampled_vec2_param(clip, "look_mask_center", playhead)
        .unwrap_or([clip.mask_center_x, clip.mask_center_y]);
    let size = sampled_vec2_param(clip, "look_mask_size", playhead)
        .unwrap_or([clip.mask_size_w, clip.mask_size_h]);
    let rotation =
        sampled_scalar_param(clip, "look_mask_rotation", playhead).unwrap_or(clip.mask_rotation);
    let feather =
        sampled_scalar_param(clip, "look_mask_feather", playhead).unwrap_or(clip.mask_feather);
    let roundness =
        sampled_scalar_param(clip, "look_mask_roundness", playhead).unwrap_or(clip.mask_roundness);
    Some(MaskGizmoParams {
        kind,
        center,
        size,
        rotation_deg: rotation,
        feather,
        roundness,
    })
}

fn apply_live(
    mut params: MaskGizmoParams,
    live_active: bool,
    live_center: [f32; 2],
    live_size: [f32; 2],
    live_rotation: f32,
    live_feather: f32,
    live_roundness: f32,
) -> MaskGizmoParams {
    if !live_active {
        return params;
    }
    params.center = live_center;
    params.size = [live_size[0].max(0.05), live_size[1].max(0.05)];
    params.rotation_deg = live_rotation;
    params.feather = live_feather.clamp(0.0, 1.0);
    params.roundness = live_roundness.clamp(0.0, 1.0);
    params
}

fn kind_handle_flags(kind: MaskKind) -> (bool, bool, bool) {
    match kind {
        MaskKind::Linear | MaskKind::Mirror => (true, false, false),
        MaskKind::Rectangle => (true, true, true),
        MaskKind::Circle | MaskKind::Heart | MaskKind::Star => (true, true, false),
    }
}

fn svg_polyline(points: &[[f32; 2]], close: bool) -> SharedString {
    if points.is_empty() {
        return SharedString::default();
    }
    let mut s = String::with_capacity(points.len() * 18);
    for (i, p) in points.iter().enumerate() {
        if i == 0 {
            s.push_str(&format!("M {:.2} {:.2}", p[0], p[1]));
        } else {
            s.push_str(&format!(" L {:.2} {:.2}", p[0], p[1]));
        }
    }
    if close {
        s.push_str(" Z");
    }
    SharedString::from(s)
}

fn outline_commands(layout: &GizmoLayout) -> (SharedString, SharedString) {
    let hx = layout.half_layer[0];
    let hy = layout.half_layer[1];
    let p = |offset: [f32; 2]| {
        mask_offset_to_viewport(
            offset,
            &layout.params,
            &layout.placement,
            layout.scale,
            layout.ox,
            layout.oy,
        )
    };
    match layout.kind {
        MaskKind::Linear => {
            // Dividing line through center along local Y (shader: x = 0).
            let len = hx.max(hy).max(1.0) * 2.0;
            let a = p([0.0, -len]);
            let b = p([0.0, len]);
            (svg_polyline(&[a, b], false), SharedString::default())
        }
        MaskKind::Mirror => {
            // Band edges at ± half-width in mask space (= size[0] × layer
            // half-width) — matches `mask.wgsl` Mirror SDF.
            let len = hy.max(1.0) * 2.0;
            let left = svg_polyline(&[p([-hx, -len]), p([-hx, len])], false);
            let right = svg_polyline(&[p([hx, -len]), p([hx, len])], false);
            (left, right)
        }
        MaskKind::Circle | MaskKind::Heart | MaskKind::Star => {
            let mut pts = Vec::with_capacity(OUTLINE_SEGMENTS + 1);
            for i in 0..OUTLINE_SEGMENTS {
                let t = std::f32::consts::TAU * (i as f32) / (OUTLINE_SEGMENTS as f32);
                pts.push(p([hx * t.cos(), hy * t.sin()]));
            }
            (svg_polyline(&pts, true), SharedString::default())
        }
        MaskKind::Rectangle => {
            let r = layout.params.roundness.clamp(0.0, 1.0) * 0.5 * hx.min(hy);
            if r <= 1e-3 {
                let pts = [p([-hx, -hy]), p([hx, -hy]), p([hx, hy]), p([-hx, hy])];
                return (svg_polyline(&pts, true), SharedString::default());
            }
            // Approximate rounded rect with arc samples per corner.
            let mut pts = Vec::with_capacity(OUTLINE_SEGMENTS);
            let corners = [
                ([hx - r, -hy + r], 0.0f32),
                ([hx - r, hy - r], std::f32::consts::FRAC_PI_2),
                ([-hx + r, hy - r], std::f32::consts::PI),
                ([-hx + r, -hy + r], -std::f32::consts::FRAC_PI_2),
            ];
            let per = (OUTLINE_SEGMENTS / 4).max(4);
            for &(c, start) in &corners {
                for i in 0..per {
                    let a = start + std::f32::consts::FRAC_PI_2 * (i as f32) / (per as f32);
                    pts.push(p([c[0] + r * a.cos(), c[1] + r * a.sin()]));
                }
            }
            (svg_polyline(&pts, true), SharedString::default())
        }
    }
}

fn build_layout(
    params: MaskGizmoParams,
    placement: LayerPlacement,
    scale: f32,
    ox: f32,
    oy: f32,
) -> GizmoLayout {
    let half = mask_half_layer(params.size, placement.size);
    let (has_size_x, has_size_y, has_roundness) = kind_handle_flags(params.kind);
    let center_v = mask_offset_to_viewport([0.0, 0.0], &params, &placement, scale, ox, oy);
    let size_x_v = mask_offset_to_viewport([half[0], 0.0], &params, &placement, scale, ox, oy);
    let size_y_v = mask_offset_to_viewport([0.0, half[1]], &params, &placement, scale, ox, oy);

    // Unit axes in viewport for constant-px handle offsets.
    let axis_x = {
        let a = mask_offset_to_viewport([1.0, 0.0], &params, &placement, scale, ox, oy);
        let dx = a[0] - center_v[0];
        let dy = a[1] - center_v[1];
        let len = (dx * dx + dy * dy).sqrt().max(1e-6);
        [dx / len, dy / len]
    };
    let axis_y = {
        let a = mask_offset_to_viewport([0.0, 1.0], &params, &placement, scale, ox, oy);
        let dx = a[0] - center_v[0];
        let dy = a[1] - center_v[1];
        let len = (dx * dx + dy * dy).sqrt().max(1e-6);
        [dx / len, dy / len]
    };

    let rotation_v = [
        size_y_v[0] + axis_y[0] * ROTATION_HANDLE_GAP_PX,
        size_y_v[1] + axis_y[1] * ROTATION_HANDLE_GAP_PX,
    ];
    let feather_gap =
        FEATHER_HANDLE_BASE_PX + params.feather.clamp(0.0, 1.0) * FEATHER_HANDLE_RANGE_PX;
    let feather_v = [
        size_x_v[0] + axis_x[0] * feather_gap,
        size_x_v[1] + axis_x[1] * feather_gap,
    ];
    // Roundness handle near the top-right corner, inset by roundness.
    let inset = params.roundness.clamp(0.0, 1.0) * half[0].min(half[1]) * 0.5;
    let roundness_v = mask_offset_to_viewport(
        [half[0] - inset, -half[1] + inset],
        &params,
        &placement,
        scale,
        ox,
        oy,
    );

    GizmoLayout {
        kind: params.kind,
        params,
        center_v,
        size_x_v,
        size_y_v,
        rotation_v,
        feather_v,
        roundness_v,
        has_size_x,
        has_size_y,
        has_roundness,
        half_layer: half,
        placement,
        scale,
        ox,
        oy,
    }
}

fn layout_to_gizmo(layout: &GizmoLayout) -> MaskGizmo {
    let (outline, outline2) = outline_commands(layout);
    MaskGizmo {
        visible: true,
        kind: SharedString::from(layout.params.kind.id()),
        outline_commands: outline,
        outline2_commands: outline2,
        center_x: layout.center_v[0],
        center_y: layout.center_v[1],
        has_size_x: layout.has_size_x,
        size_x_x: layout.size_x_v[0],
        size_x_y: layout.size_x_v[1],
        has_size_y: layout.has_size_y,
        size_y_x: layout.size_y_v[0],
        size_y_y: layout.size_y_v[1],
        rotation_x: layout.rotation_v[0],
        rotation_y: layout.rotation_v[1],
        feather_x: layout.feather_v[0],
        feather_y: layout.feather_v[1],
        has_roundness: layout.has_roundness,
        roundness_x: layout.roundness_v[0],
        roundness_y: layout.roundness_v[1],
        center_fx: layout.params.center[0],
        center_fy: layout.params.center[1],
        size_w: layout.params.size[0],
        size_h: layout.params.size[1],
        rotation_deg: layout.params.rotation_deg,
        feather: layout.params.feather,
        roundness: layout.params.roundness,
    }
}

fn dist2(a: [f32; 2], b: [f32; 2]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

fn near(a: [f32; 2], b: [f32; 2], tol: f32) -> bool {
    dist2(a, b) <= tol * tol
}

/// Point-in-mask body in viewport px (axis-aligned ellipse/rect in mask space).
fn body_contains(layout: &GizmoLayout, view: [f32; 2]) -> bool {
    let canvas = viewport_to_canvas(view, layout.scale, layout.ox, layout.oy);
    let layer = canvas_to_layer(canvas, &layout.placement);
    let center_local = center_fractions_to_layer(layout.params.center, layout.placement.size);
    let d = [layer[0] - center_local[0], layer[1] - center_local[1]];
    let local = inv_rotate(d, deg_to_rad(layout.params.rotation_deg));
    let hx = layout.half_layer[0].max(1e-3);
    let hy = layout.half_layer[1].max(1e-3);
    match layout.kind {
        MaskKind::Linear => {
            // Thin strip around the dividing line.
            let band = DEFAULT_HIT_TOLERANCE_PX / layout.scale.max(1e-6);
            local[0].abs() <= band.max(hx * 0.15) && local[1].abs() <= hy.max(band) * 1.25
        }
        MaskKind::Mirror => {
            // Full band body (matches rendered Mirror thickness).
            local[0].abs() <= hx && local[1].abs() <= hy.max(hx) * 1.25
        }
        MaskKind::Circle | MaskKind::Heart | MaskKind::Star => {
            let nx = local[0] / hx;
            let ny = local[1] / hy;
            nx * nx + ny * ny <= 1.0
        }
        MaskKind::Rectangle => local[0].abs() <= hx && local[1].abs() <= hy,
    }
}

/// Which handle (if any) is under `(x, y)` in viewport px.
fn hit_test_layout(layout: &GizmoLayout, x: f32, y: f32, tolerance: f32) -> i32 {
    let p = [x, y];
    let tol = if tolerance.is_finite() && tolerance > 0.0 {
        tolerance
    } else {
        DEFAULT_HIT_TOLERANCE_PX
    };
    // Priority: small affordances first, then center, then body.
    if near(p, layout.rotation_v, tol) {
        return HANDLE_ROTATION;
    }
    if near(p, layout.feather_v, tol) {
        return HANDLE_FEATHER;
    }
    if layout.has_roundness && near(p, layout.roundness_v, tol) {
        return HANDLE_ROUNDNESS;
    }
    if layout.has_size_x && near(p, layout.size_x_v, tol) {
        return HANDLE_SIZE_X;
    }
    if layout.has_size_y && near(p, layout.size_y_v, tol) {
        return HANDLE_SIZE_Y;
    }
    if near(p, layout.center_v, tol) {
        return HANDLE_CENTER;
    }
    if body_contains(layout, p) {
        return HANDLE_BODY;
    }
    HANDLE_NONE
}

/// Hit-test a published [`MaskGizmo`] (rebuilds layout from its param fields).
#[allow(clippy::too_many_arguments)]
pub fn hit_test_mask_gizmo(
    gizmo: &MaskGizmo,
    x: f32,
    y: f32,
    tolerance: f32,
    placement: &LayerPlacement,
    scale: f32,
    ox: f32,
    oy: f32,
) -> i32 {
    if !gizmo.visible {
        return HANDLE_NONE;
    }
    let Some(kind) = parse_kind(gizmo.kind.as_str()) else {
        return HANDLE_NONE;
    };
    let params = MaskGizmoParams {
        kind,
        center: [gizmo.center_fx, gizmo.center_fy],
        size: [gizmo.size_w.max(0.05), gizmo.size_h.max(0.05)],
        rotation_deg: gizmo.rotation_deg,
        feather: gizmo.feather,
        roundness: gizmo.roundness,
    };
    let layout = build_layout(params, *placement, scale, ox, oy);
    hit_test_layout(&layout, x, y, tolerance)
}

/// Build the mask gizmo for `clip_id` in viewport-element coordinates.
///
/// When `live_active` is true, the `live_*` fields replace playhead samples
/// so the overlay tracks ParamOverride / gizmo drags (projection stays stale
/// until commit).
#[allow(clippy::too_many_arguments)]
pub fn mask_gizmo_in_viewport(
    sequence: &Sequence,
    clip_id: &str,
    playhead: i32,
    view_w: f32,
    view_h: f32,
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
    live_active: bool,
    live_center_x: f32,
    live_center_y: f32,
    live_size_w: f32,
    live_size_h: f32,
    live_rotation: f32,
    live_feather: f32,
    live_roundness: f32,
) -> MaskGizmo {
    if clip_id.is_empty() {
        return MaskGizmo::default();
    }
    let Some(mut clip) = find_projected_clip(sequence, clip_id) else {
        return MaskGizmo::default();
    };
    if !covers_tick(&clip, playhead) || !is_composited(&clip) {
        return MaskGizmo::default();
    }
    // Placement follows the rendered transform at the playhead.
    crate::params::apply_sampled_transform(&mut clip, playhead);
    let Some(base) = sample_mask_params(&clip, playhead) else {
        return MaskGizmo::default();
    };
    let params = apply_live(
        base,
        live_active,
        [live_center_x, live_center_y],
        [live_size_w, live_size_h],
        live_rotation,
        live_feather,
        live_roundness,
    );

    let canvas = canvas_config(sequence);
    let (cw, ch) = (canvas.width as f32, canvas.height as f32);
    let (scale, ox, oy) = viewport_mapping(cw, ch, view_w, view_h, zoom, pan_x, pan_y);
    if scale <= 0.0 {
        return MaskGizmo::default();
    }
    let placement = clip_placement(&clip, &canvas);
    if placement.size[0] <= 0.0 || placement.size[1] <= 0.0 {
        return MaskGizmo::default();
    }
    let layout = build_layout(params, placement, scale, ox, oy);
    layout_to_gizmo(&layout)
}

/// Convenience hit-test that rebuilds layout from sequence + live params.
#[allow(clippy::too_many_arguments)]
pub fn hit_test_mask_gizmo_in_viewport(
    sequence: &Sequence,
    clip_id: &str,
    playhead: i32,
    x: f32,
    y: f32,
    view_w: f32,
    view_h: f32,
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
    live_active: bool,
    live_center_x: f32,
    live_center_y: f32,
    live_size_w: f32,
    live_size_h: f32,
    live_rotation: f32,
    live_feather: f32,
    live_roundness: f32,
    tolerance: f32,
) -> i32 {
    let gizmo = mask_gizmo_in_viewport(
        sequence,
        clip_id,
        playhead,
        view_w,
        view_h,
        zoom,
        pan_x,
        pan_y,
        live_active,
        live_center_x,
        live_center_y,
        live_size_w,
        live_size_h,
        live_rotation,
        live_feather,
        live_roundness,
    );
    if !gizmo.visible {
        return HANDLE_NONE;
    }
    let Some(mut clip) = find_projected_clip(sequence, clip_id) else {
        return HANDLE_NONE;
    };
    crate::params::apply_sampled_transform(&mut clip, playhead);
    let canvas = canvas_config(sequence);
    let (cw, ch) = (canvas.width as f32, canvas.height as f32);
    let (scale, ox, oy) = viewport_mapping(cw, ch, view_w, view_h, zoom, pan_x, pan_y);
    let placement = clip_placement(&clip, &canvas);
    hit_test_mask_gizmo(&gizmo, x, y, tolerance, &placement, scale, ox, oy)
}

/// Resolve a mask-handle drag for the selected clip in viewport coordinates.
#[allow(clippy::too_many_arguments)]
pub fn resolve_mask_gizmo_drag_in_viewport(
    sequence: &Sequence,
    clip_id: &str,
    playhead: i32,
    handle: i32,
    press: [f32; 2],
    cursor: [f32; 2],
    view_w: f32,
    view_h: f32,
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
    start: MaskGizmoParams,
) -> MaskGizmoDragResolution {
    if clip_id.is_empty() || handle == HANDLE_NONE {
        return MaskGizmoDragResolution::default();
    }
    let Some(mut clip) = find_projected_clip(sequence, clip_id) else {
        return MaskGizmoDragResolution::default();
    };
    crate::params::apply_sampled_transform(&mut clip, playhead);
    let canvas = canvas_config(sequence);
    let (cw, ch) = (canvas.width as f32, canvas.height as f32);
    let (scale, ox, oy) = viewport_mapping(cw, ch, view_w, view_h, zoom, pan_x, pan_y);
    if scale <= 0.0 {
        return MaskGizmoDragResolution::default();
    }
    let placement = clip_placement(&clip, &canvas);
    let out = resolve_mask_handle_drag(handle, start, &placement, scale, ox, oy, press, cursor);
    MaskGizmoDragResolution {
        valid: true,
        center_fx: out.center[0],
        center_fy: out.center[1],
        size_w: out.size[0],
        size_h: out.size[1],
        rotation_deg: out.rotation_deg,
        feather: out.feather,
        roundness: out.roundness,
    }
}

/// Resolve a handle drag into updated mask params (layer fractions / degrees).
///
/// `press` / `cursor` are viewport px. Size handles scale the matching axis
/// from the mask center; rotation uses the angle about the center; feather /
/// roundness map radial distance past the size edge into `0…1`.
///
/// Used by on-canvas mask gestures (overlay commit) and unit tests.
#[allow(clippy::too_many_arguments)]
pub fn resolve_mask_handle_drag(
    handle: i32,
    start: MaskGizmoParams,
    placement: &LayerPlacement,
    scale: f32,
    ox: f32,
    oy: f32,
    press: [f32; 2],
    cursor: [f32; 2],
) -> MaskGizmoParams {
    let mut out = start;
    let layer_size = placement.size;
    let center_local = center_fractions_to_layer(start.center, layer_size);
    let rot = deg_to_rad(start.rotation_deg);

    let cursor_layer = {
        let c = viewport_to_canvas(cursor, scale, ox, oy);
        canvas_to_layer(c, placement)
    };
    let press_layer = {
        let c = viewport_to_canvas(press, scale, ox, oy);
        canvas_to_layer(c, placement)
    };

    match handle {
        HANDLE_CENTER | HANDLE_BODY => {
            let delta = [
                cursor_layer[0] - press_layer[0],
                cursor_layer[1] - press_layer[1],
            ];
            let new_center_local = [center_local[0] + delta[0], center_local[1] + delta[1]];
            out.center = layer_to_center_fractions(new_center_local, layer_size);
            out.center[0] = out.center[0].clamp(-10.0, 10.0);
            out.center[1] = out.center[1].clamp(-10.0, 10.0);
        }
        HANDLE_SIZE_X => {
            let d = [
                cursor_layer[0] - center_local[0],
                cursor_layer[1] - center_local[1],
            ];
            let local = inv_rotate(d, rot);
            let half_w = (layer_size[0] * 0.5).max(1e-3);
            out.size[0] = (local[0].abs() / half_w).clamp(0.05, 3.0);
            if matches!(start.kind, MaskKind::Circle) {
                // Keep circle uniform when only the X handle exists in UI;
                // both handles still publish independently for ellipses.
            }
        }
        HANDLE_SIZE_Y => {
            let d = [
                cursor_layer[0] - center_local[0],
                cursor_layer[1] - center_local[1],
            ];
            let local = inv_rotate(d, rot);
            let half_h = (layer_size[1] * 0.5).max(1e-3);
            out.size[1] = (local[1].abs() / half_h).clamp(0.05, 3.0);
        }
        HANDLE_ROTATION => {
            let d = [
                cursor_layer[0] - center_local[0],
                cursor_layer[1] - center_local[1],
            ];
            // Angle of cursor in layer space; +y down → clockwise degrees.
            let angle = d[1].atan2(d[0]).to_degrees();
            // Handle sits on +Y axis at 90° in mask space when rotation=0.
            out.rotation_deg = (angle - 90.0).rem_euclid(360.0);
            if out.rotation_deg > 180.0 {
                out.rotation_deg -= 360.0;
            }
        }
        HANDLE_FEATHER => {
            let half = mask_half_layer(start.size, layer_size);
            let size_x_layer = {
                let off = rotate([half[0], 0.0], rot);
                [center_local[0] + off[0], center_local[1] + off[1]]
            };
            let dx = cursor_layer[0] - size_x_layer[0];
            let dy = cursor_layer[1] - size_x_layer[1];
            let dist_layer = (dx * dx + dy * dy).sqrt();
            let dist_view = dist_layer * scale;
            out.feather =
                ((dist_view - FEATHER_HANDLE_BASE_PX) / FEATHER_HANDLE_RANGE_PX).clamp(0.0, 1.0);
            let _ = press;
        }
        HANDLE_ROUNDNESS => {
            let half = mask_half_layer(start.size, layer_size);
            let d = [
                cursor_layer[0] - center_local[0],
                cursor_layer[1] - center_local[1],
            ];
            let local = inv_rotate(d, rot);
            // Inset from the top-right corner toward the center → roundness.
            let inset_x = (half[0] - local[0]).clamp(0.0, half[0]);
            let inset_y = (local[1] + half[1]).clamp(0.0, half[1]);
            let inset = inset_x.min(inset_y);
            let max_r = 0.5 * half[0].min(half[1]).max(1e-3);
            out.roundness = (inset / max_r).clamp(0.0, 1.0);
        }
        _ => {}
    }
    out
}

#[cfg(test)]
mod tests;
