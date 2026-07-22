//! GPU readback checks for mask geometry (center / size / rotation / roundness).
//!
//! Solid layers skip the fx pipeline, so probes use a uniform RGBA bitmap —
//! the same path as `circle_mask_cuts_rgba_corners` in `render.rs`.

use std::f32::consts::FRAC_PI_4;

use cutlass_compositor::{
    CompositeLayer, Compositor, CompositorConfig, GpuContext, LayerEffects, LayerMask,
    LayerPlacement, RgbaImage, mask_kind,
};

/// Try to bring up a headless GPU; `None` (skip) if the host has no adapter.
fn try_gpu() -> Option<GpuContext> {
    match GpuContext::new_headless_blocking() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("skipping masks test: no GPU adapter ({e})");
            None
        }
    }
}

macro_rules! gpu_or_skip {
    () => {
        match try_gpu() {
            Some(g) => g,
            None => return,
        }
    };
}

#[track_caller]
fn assert_px(img: &RgbaImage, x: u32, y: u32, expect: [u8; 4], tol: i32) {
    let got = img.pixel(x, y);
    for ch in 0..4 {
        let d = i32::from(got[ch]) - i32::from(expect[ch]);
        assert!(
            d.abs() <= tol,
            "pixel({x},{y}) = {got:?}, expected ~{expect:?} (channel {ch} off by {d}, tol {tol})"
        );
    }
}

fn rgba_uniform(w: u32, h: u32, rgba: [u8; 4]) -> RgbaImage {
    let mut pixels = vec![0u8; (w * h * 4) as usize];
    for px in pixels.chunks_exact_mut(4) {
        px.copy_from_slice(&rgba);
    }
    RgbaImage::new(w, h, pixels)
}

fn render_masked(
    gpu: &GpuContext,
    size: u32,
    bg: [u8; 4],
    fg: [u8; 4],
    mask: LayerMask,
) -> RgbaImage {
    let mut comp = Compositor::new(gpu);
    let config = CompositorConfig::new(size, size).with_background(bg);
    let bmp = rgba_uniform(size, size, fg);
    let placement = LayerPlacement::full_canvas(&config);
    let layer = CompositeLayer::rgba(&bmp, placement).with_fx(LayerEffects {
        mask: Some(mask),
        chroma_key: None,
    });
    comp.render(gpu, &config, &[layer]).expect("render")
}

#[test]
fn circle_default_geometry_center_opaque_corner_transparent() {
    let gpu = gpu_or_skip!();
    let img = render_masked(
        &gpu,
        64,
        [0, 0, 255, 255],
        [255, 0, 0, 255],
        LayerMask::new(mask_kind::CIRCLE),
    );
    assert_px(&img, 32, 32, [255, 0, 0, 255], 3);
    assert_px(&img, 2, 2, [0, 0, 255, 255], 3);
}

#[test]
fn circle_half_size_excludes_mid_radius_point() {
    // On-axis probe at 0.7×half-width: inside at size 1, outside at size 0.5
    // (divide-by-size maps 0.7 → 1.4). Spec mentioned 0.4×half; that stays
    // inside under correct math, so the probe is 0.7.
    let gpu = gpu_or_skip!();
    let n = 200u32;
    let cx = n / 2;
    let cy = n / 2;
    let half = n as f32 * 0.5;
    let probe_x = cx + (0.7 * half).round() as u32;

    let full = render_masked(
        &gpu,
        n,
        [0, 0, 255, 255],
        [255, 0, 0, 255],
        LayerMask::new(mask_kind::CIRCLE),
    );
    assert_px(&full, probe_x, cy, [255, 0, 0, 255], 3);

    let mut half_size = LayerMask::new(mask_kind::CIRCLE);
    half_size.size = [0.5, 0.5];
    let shrunk = render_masked(&gpu, n, [0, 0, 255, 255], [255, 0, 0, 255], half_size);
    assert_px(&shrunk, probe_x, cy, [0, 0, 255, 255], 3);
    // Center still covered.
    assert_px(&shrunk, cx, cy, [255, 0, 0, 255], 3);
}

#[test]
fn circle_center_shifted_right() {
    let gpu = gpu_or_skip!();
    let n = 200u32;
    let cx = n / 2;
    let cy = n / 2;
    let mut mask = LayerMask::new(mask_kind::CIRCLE);
    mask.center = [0.5, 0.0];
    let img = render_masked(&gpu, n, [0, 0, 255, 255], [255, 0, 0, 255], mask);

    // Circle center sits on the right edge → that point is opaque.
    assert_px(&img, n - 2, cy, [255, 0, 0, 255], 3);
    // Left of layer center falls outside the shifted circle.
    assert_px(&img, cx - 40, cy, [0, 0, 255, 255], 3);
}

#[test]
fn rectangle_rotation_puts_corner_outside() {
    let gpu = gpu_or_skip!();
    let n = 200u32;
    let cx = n / 2;
    let cy = n / 2;
    let half = n as f32 * 0.5;
    // Near-corner of the unrotated rect (inside when rot=0).
    let px = cx + (0.92 * half).round() as u32;
    let py = cy + (0.92 * half).round() as u32;

    let axis = render_masked(
        &gpu,
        n,
        [0, 0, 255, 255],
        [255, 0, 0, 255],
        LayerMask::new(mask_kind::RECTANGLE),
    );
    assert_px(&axis, px, py, [255, 0, 0, 255], 3);

    let mut rotated = LayerMask::new(mask_kind::RECTANGLE);
    rotated.rotation_rad = FRAC_PI_4;
    let img = render_masked(&gpu, n, [0, 0, 255, 255], [255, 0, 0, 255], rotated);
    assert_px(&img, px, py, [0, 0, 255, 255], 3);
}

#[test]
fn rectangle_roundness_clears_extreme_corner() {
    let gpu = gpu_or_skip!();
    let n = 200u32;
    let cx = n / 2;
    let cy = n / 2;
    let half = n as f32 * 0.5;
    // Deep in the sharp-rect corner; outside a fully rounded box.
    let px = cx + (0.92 * half).round() as u32;
    let py = cy + (0.92 * half).round() as u32;

    let sharp = render_masked(
        &gpu,
        n,
        [0, 0, 255, 255],
        [255, 0, 0, 255],
        LayerMask::new(mask_kind::RECTANGLE),
    );
    assert_px(&sharp, px, py, [255, 0, 0, 255], 3);

    let mut round = LayerMask::new(mask_kind::RECTANGLE);
    round.roundness = 1.0;
    let img = render_masked(&gpu, n, [0, 0, 255, 255], [255, 0, 0, 255], round);
    assert_px(&img, px, py, [0, 0, 255, 255], 3);
}

#[test]
fn inverted_shifted_circle_is_hole() {
    let gpu = gpu_or_skip!();
    let n = 200u32;
    let cy = n / 2;
    let mut mask = LayerMask::new(mask_kind::CIRCLE);
    mask.center = [0.5, 0.0];
    mask.invert = 1;
    let img = render_masked(&gpu, n, [0, 0, 255, 255], [255, 0, 0, 255], mask);

    // Inverted: the shifted-circle interior (right edge) becomes a hole.
    assert_px(&img, n - 2, cy, [0, 0, 255, 255], 3);
    // Far left stays opaque (outside the circle → kept by invert).
    assert_px(&img, 10, cy, [255, 0, 0, 255], 3);
}
