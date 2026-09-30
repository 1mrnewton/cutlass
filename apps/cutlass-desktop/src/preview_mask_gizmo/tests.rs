use super::*;
use cutlass_compositor::{CompositorConfig, LayerPlacement};
use cutlass_models::MaskKind;

fn identity_placement(w: f32, h: f32) -> LayerPlacement {
    LayerPlacement {
        center: [w * 0.5, h * 0.5],
        size: [w, h],
        rotation: 0.0,
        opacity: 1.0,
    }
}

fn params(kind: MaskKind) -> MaskGizmoParams {
    MaskGizmoParams {
        kind,
        center: [0.0, 0.0],
        size: [1.0, 1.0],
        rotation_deg: 0.0,
        feather: 0.0,
        roundness: 0.0,
    }
}

#[test]
fn layer_canvas_viewport_round_trip_identity() {
    let placement = identity_placement(1920.0, 1080.0);
    let (scale, ox, oy) = viewport_mapping(1920.0, 1080.0, 960.0, 540.0, 1.0, 0.0, 0.0);
    let local = [120.0, -80.0];
    let canvas = layer_to_canvas(local, &placement);
    let view = canvas_to_viewport(canvas, scale, ox, oy);
    let back_canvas = viewport_to_canvas(view, scale, ox, oy);
    let back_local = canvas_to_layer(back_canvas, &placement);
    assert!((back_local[0] - local[0]).abs() < 1e-3);
    assert!((back_local[1] - local[1]).abs() < 1e-3);
}

#[test]
fn layer_canvas_round_trip_rotated_scaled_placement() {
    let placement = LayerPlacement {
        center: [400.0, 300.0],
        size: [640.0, 360.0],
        rotation: 35f32.to_radians(),
        opacity: 1.0,
    };
    let local = [-50.0, 90.0];
    let canvas = layer_to_canvas(local, &placement);
    let back = canvas_to_layer(canvas, &placement);
    assert!((back[0] - local[0]).abs() < 1e-4);
    assert!((back[1] - local[1]).abs() < 1e-4);
}

#[test]
fn center_fractions_round_trip_under_zoom_pan() {
    let placement = identity_placement(1920.0, 1080.0);
    let (scale, ox, oy) = viewport_mapping(1920.0, 1080.0, 800.0, 600.0, 1.5, 40.0, -20.0);
    let center = [0.25, -0.1];
    let local = center_fractions_to_layer(center, placement.size);
    let canvas = layer_to_canvas(local, &placement);
    let view = canvas_to_viewport(canvas, scale, ox, oy);
    let back_local = canvas_to_layer(viewport_to_canvas(view, scale, ox, oy), &placement);
    let back = layer_to_center_fractions(back_local, placement.size);
    assert!((back[0] - center[0]).abs() < 1e-4);
    assert!((back[1] - center[1]).abs() < 1e-4);
}

#[test]
fn circle_handle_layout_size_axes() {
    let placement = identity_placement(1000.0, 1000.0);
    let (scale, ox, oy) = (1.0, 0.0, 0.0);
    let mut p = params(MaskKind::Circle);
    p.size = [0.5, 0.25];
    let layout = build_layout(p, placement, scale, ox, oy);
    assert!(layout.has_size_x && layout.has_size_y && !layout.has_roundness);
    // Size-X at +half_w from center in canvas (= layer for identity placement).
    let expected_x = canvas_to_viewport(
        [
            placement.center[0] + 0.5 * 1000.0 * 0.5,
            placement.center[1],
        ],
        scale,
        ox,
        oy,
    );
    assert!((layout.size_x_v[0] - expected_x[0]).abs() < 1e-3);
    assert!((layout.size_x_v[1] - expected_x[1]).abs() < 1e-3);
    let expected_y = canvas_to_viewport(
        [
            placement.center[0],
            placement.center[1] + 0.25 * 1000.0 * 0.5,
        ],
        scale,
        ox,
        oy,
    );
    assert!((layout.size_y_v[0] - expected_y[0]).abs() < 1e-3);
    assert!((layout.size_y_v[1] - expected_y[1]).abs() < 1e-3);
}

#[test]
fn rectangle_exposes_roundness_linear_does_not() {
    let placement = identity_placement(800.0, 600.0);
    let rect = build_layout(params(MaskKind::Rectangle), placement, 1.0, 0.0, 0.0);
    assert!(rect.has_roundness && rect.has_size_x && rect.has_size_y);
    let linear = build_layout(params(MaskKind::Linear), placement, 1.0, 0.0, 0.0);
    assert!(linear.has_size_x && !linear.has_size_y && !linear.has_roundness);
    let mirror = build_layout(params(MaskKind::Mirror), placement, 1.0, 0.0, 0.0);
    assert!(mirror.has_size_x && !mirror.has_size_y);
}

#[test]
fn rotated_mask_moves_size_handle() {
    let placement = identity_placement(1000.0, 1000.0);
    let mut p = params(MaskKind::Rectangle);
    p.rotation_deg = 90.0;
    p.size = [0.4, 0.2];
    let layout = build_layout(p, placement, 1.0, 0.0, 0.0);
    // 90° clockwise: local +X → canvas +Y.
    let expected = [
        placement.center[0],
        placement.center[1] + 0.4 * 1000.0 * 0.5,
    ];
    assert!((layout.size_x_v[0] - expected[0]).abs() < 1e-2);
    assert!((layout.size_x_v[1] - expected[1]).abs() < 1e-2);
}

#[test]
fn hit_test_prefers_handles_over_body() {
    let placement = identity_placement(1000.0, 1000.0);
    let layout = build_layout(params(MaskKind::Circle), placement, 1.0, 0.0, 0.0);
    assert_eq!(
        hit_test_layout(&layout, layout.rotation_v[0], layout.rotation_v[1], 12.0),
        HANDLE_ROTATION
    );
    assert_eq!(
        hit_test_layout(&layout, layout.feather_v[0], layout.feather_v[1], 12.0),
        HANDLE_FEATHER
    );
    assert_eq!(
        hit_test_layout(&layout, layout.size_x_v[0], layout.size_x_v[1], 12.0),
        HANDLE_SIZE_X
    );
    assert_eq!(
        hit_test_layout(&layout, layout.center_v[0], layout.center_v[1], 12.0),
        HANDLE_CENTER
    );
    // Past center hit radius but still inside the ellipse → body.
    assert_eq!(
        hit_test_layout(&layout, layout.center_v[0] + 40.0, layout.center_v[1], 12.0),
        HANDLE_BODY
    );
    assert_eq!(
        hit_test_layout(&layout, 0.0, 0.0, 12.0),
        HANDLE_NONE,
        "outside the layer should miss"
    );
}

#[test]
fn hit_test_roundness_rectangle_only() {
    let placement = identity_placement(1000.0, 1000.0);
    let rect = build_layout(params(MaskKind::Rectangle), placement, 1.0, 0.0, 0.0);
    assert_eq!(
        hit_test_layout(&rect, rect.roundness_v[0], rect.roundness_v[1], 12.0),
        HANDLE_ROUNDNESS
    );
    let circle = build_layout(params(MaskKind::Circle), placement, 1.0, 0.0, 0.0);
    assert_ne!(
        hit_test_layout(&circle, circle.roundness_v[0], circle.roundness_v[1], 12.0),
        HANDLE_ROUNDNESS
    );
}

#[test]
fn resolve_center_drag_updates_fractions() {
    let placement = identity_placement(1000.0, 1000.0);
    let start = params(MaskKind::Circle);
    // Drag 50 canvas px right (= 50 view px at scale 1).
    let press = canvas_to_viewport(placement.center, 1.0, 0.0, 0.0);
    let cursor = canvas_to_viewport(
        [placement.center[0] + 50.0, placement.center[1]],
        1.0,
        0.0,
        0.0,
    );
    let out = resolve_mask_handle_drag(
        HANDLE_CENTER,
        start,
        &placement,
        1.0,
        0.0,
        0.0,
        press,
        cursor,
        false,
    );
    assert!((out.center[0] - 0.05).abs() < 1e-4);
    assert!(out.center[1].abs() < 1e-4);
}

#[test]
fn resolve_size_x_drag_sets_width_fraction() {
    let placement = identity_placement(1000.0, 1000.0);
    let start = params(MaskKind::Rectangle);
    let press = [0.0, 0.0];
    // Cursor at +200 px from center on X → size_w = 200 / 500 = 0.4
    let cursor = canvas_to_viewport(
        [placement.center[0] + 200.0, placement.center[1]],
        1.0,
        0.0,
        0.0,
    );
    let out = resolve_mask_handle_drag(
        HANDLE_SIZE_X,
        start,
        &placement,
        1.0,
        0.0,
        0.0,
        press,
        cursor,
        false,
    );
    assert!((out.size[0] - 0.4).abs() < 1e-4);
    assert!((out.size[1] - 1.0).abs() < 1e-4);
}

#[test]
fn resolve_size_x_drag_keeps_aspect_when_requested() {
    let placement = identity_placement(1000.0, 1000.0);
    let mut start = params(MaskKind::Rectangle);
    start.size = [0.8, 0.4];
    let press = [0.0, 0.0];
    // Cursor at +200 px from center on X → size_w = 0.4; aspect 0.4/0.8 → h = 0.2
    let cursor = canvas_to_viewport(
        [placement.center[0] + 200.0, placement.center[1]],
        1.0,
        0.0,
        0.0,
    );
    let out = resolve_mask_handle_drag(
        HANDLE_SIZE_X,
        start,
        &placement,
        1.0,
        0.0,
        0.0,
        press,
        cursor,
        true,
    );
    assert!((out.size[0] - 0.4).abs() < 1e-4);
    assert!((out.size[1] - 0.2).abs() < 1e-4);
}

#[test]
fn resolve_rotation_drag_about_center() {
    let placement = identity_placement(1000.0, 1000.0);
    let start = params(MaskKind::Circle);
    let press = canvas_to_viewport(
        [placement.center[0], placement.center[1] + 100.0],
        1.0,
        0.0,
        0.0,
    );
    // Cursor on +X from center → angle 0°, handle reference is +Y (90°) → rot = -90.
    let cursor = canvas_to_viewport(
        [placement.center[0] + 100.0, placement.center[1]],
        1.0,
        0.0,
        0.0,
    );
    let out = resolve_mask_handle_drag(
        HANDLE_ROTATION,
        start,
        &placement,
        1.0,
        0.0,
        0.0,
        press,
        cursor,
        false,
    );
    assert!((out.rotation_deg - (-90.0)).abs() < 1.0);
}

#[test]
fn live_override_moves_center_handle() {
    let placement = identity_placement(1000.0, 1000.0);
    let base = params(MaskKind::Circle);
    let live = apply_live(base, true, [0.2, -0.1], [1.0, 1.0], 0.0, 0.0, 0.0);
    let layout = build_layout(live, placement, 1.0, 0.0, 0.0);
    let expected = canvas_to_viewport(
        [
            placement.center[0] + 0.2 * 1000.0,
            placement.center[1] - 0.1 * 1000.0,
        ],
        1.0,
        0.0,
        0.0,
    );
    assert!((layout.center_v[0] - expected[0]).abs() < 1e-3);
    assert!((layout.center_v[1] - expected[1]).abs() < 1e-3);
}

#[test]
fn heart_and_star_use_bounding_ellipse_handles() {
    let placement = identity_placement(800.0, 600.0);
    for kind in [MaskKind::Heart, MaskKind::Star] {
        let layout = build_layout(params(kind), placement, 1.0, 0.0, 0.0);
        assert!(layout.has_size_x && layout.has_size_y && !layout.has_roundness);
        let (outline, outline2) = outline_commands(&layout);
        assert!(!outline.is_empty());
        assert!(outline2.is_empty());
    }
}

#[test]
fn mirror_outline_has_two_band_edges() {
    let placement = identity_placement(800.0, 600.0);
    let layout = build_layout(params(MaskKind::Mirror), placement, 1.0, 0.0, 0.0);
    let (a, b) = outline_commands(&layout);
    assert!(!a.is_empty() && !b.is_empty());
}

#[test]
fn mirror_band_edge_handle_matches_half_thickness() {
    // Mirror thickness uses size[0] × layer half-width; the size-X handle
    // sits on that edge so gizmo outlines match the shader band.
    let placement = identity_placement(800.0, 600.0);
    let (scale, ox, oy) = viewport_mapping(1920.0, 1080.0, 960.0, 540.0, 1.25, 12.0, -8.0);
    let mut p = params(MaskKind::Mirror);
    p.size = [0.6, 1.0];
    p.rotation_deg = 25.0;
    p.center = [0.1, -0.05];
    let layout = build_layout(p, placement, scale, ox, oy);
    let half_w = placement.size[0] * 0.5;
    let expected_dist = p.size[0] * half_w * scale;
    let dx = layout.size_x_v[0] - layout.center_v[0];
    let dy = layout.size_x_v[1] - layout.center_v[1];
    let dist = (dx * dx + dy * dy).sqrt();
    assert!(
        (dist - expected_dist).abs() < 0.5,
        "band edge handle dist {dist} vs expected {expected_dist}"
    );
}

#[test]
fn size_drag_rebuilds_handle_near_cursor_under_letterbox() {
    // widget → resolve drag → rebuild layout → size handle ≈ cursor
    let placement = LayerPlacement {
        center: [960.0, 540.0],
        size: [1280.0, 720.0],
        rotation: 15f32.to_radians(),
        opacity: 1.0,
    };
    let (scale, ox, oy) = viewport_mapping(1920.0, 1080.0, 800.0, 600.0, 1.0, 0.0, 0.0);
    let start = {
        let mut p = params(MaskKind::Rectangle);
        p.size = [0.7, 0.5];
        p.rotation_deg = 15.0;
        p
    };
    let layout0 = build_layout(start, placement, scale, ox, oy);
    let press = layout0.size_x_v;
    let cursor = [press[0] + 40.0, press[1] - 10.0];
    let out = resolve_mask_handle_drag(
        HANDLE_SIZE_X,
        start,
        &placement,
        scale,
        ox,
        oy,
        press,
        cursor,
        false,
    );
    let layout1 = build_layout(out, placement, scale, ox, oy);
    let dx = layout1.size_x_v[0] - cursor[0];
    let dy = layout1.size_x_v[1] - cursor[1];
    // Radial size: handle lands on the ray from center through cursor at
    // the resolved half-width — distance to cursor should be small along
    // the axis (projection), not necessarily zero if cursor left the axis.
    let center = layout1.center_v;
    let to_handle = [
        layout1.size_x_v[0] - center[0],
        layout1.size_x_v[1] - center[1],
    ];
    let to_cursor = [cursor[0] - center[0], cursor[1] - center[1]];
    let handle_len = (to_handle[0] * to_handle[0] + to_handle[1] * to_handle[1])
        .sqrt()
        .max(1e-3);
    let proj = (to_cursor[0] * to_handle[0] + to_cursor[1] * to_handle[1]) / handle_len;
    assert!(
        (proj - handle_len).abs() < 1.5,
        "projected cursor radius {proj} vs handle {handle_len} (dx={dx}, dy={dy})"
    );
}

#[test]
fn widget_layer_widget_round_trip_letterbox_rotated() {
    let placement = LayerPlacement {
        center: [400.0, 300.0],
        size: [640.0, 360.0],
        rotation: 40f32.to_radians(),
        opacity: 1.0,
    };
    let (scale, ox, oy) = viewport_mapping(1920.0, 1080.0, 640.0, 480.0, 1.8, -30.0, 20.0);
    for local in [[0.0, 0.0], [80.0, -40.0], [-60.0, 90.0], [120.0, 10.0]] {
        let canvas = layer_to_canvas(local, &placement);
        let view = canvas_to_viewport(canvas, scale, ox, oy);
        let back = canvas_to_layer(viewport_to_canvas(view, scale, ox, oy), &placement);
        assert!((back[0] - local[0]).abs() < 1e-2, "x {local:?} → {back:?}");
        assert!((back[1] - local[1]).abs() < 1e-2, "y {local:?} → {back:?}");
    }
}

#[allow(dead_code)]
fn _canvas_config_smoke() {
    let _ = CompositorConfig::new(16, 9);
}
