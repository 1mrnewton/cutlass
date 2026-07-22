//! GPU readback checks for per-layer blend modes (dst-sampling composite).

use cutlass_compositor::{
    BlendMode, CompositeLayer, Compositor, CompositorConfig, GpuContext, LayerPlacement, RgbaImage,
};

/// Try to bring up a headless GPU; `None` (skip) if the host has no adapter.
fn try_gpu() -> Option<GpuContext> {
    match GpuContext::new_headless_blocking() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("skipping blend test: no GPU adapter ({e})");
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

fn render_blend(
    gpu: &GpuContext,
    background: [u8; 4],
    layer_rgba: [u8; 4],
    mode: BlendMode,
    opacity: f32,
) -> RgbaImage {
    let mut comp = Compositor::new(gpu);
    let config = CompositorConfig::new(16, 16).with_background(background);
    let mut placement = LayerPlacement::full_canvas(&config);
    placement.opacity = opacity;
    let layer = CompositeLayer::solid(layer_rgba, placement).with_blend_mode(mode);
    comp.render(gpu, &config, &[layer]).expect("render")
}

#[test]
fn multiply_red_over_green_is_black() {
    let gpu = gpu_or_skip!();
    let img = render_blend(
        &gpu,
        [0, 255, 0, 255],
        [255, 0, 0, 255],
        BlendMode::Multiply,
        1.0,
    );
    assert_px(&img, 8, 8, [0, 0, 0, 255], 2);
}

#[test]
fn screen_red_over_green_is_yellow() {
    let gpu = gpu_or_skip!();
    let img = render_blend(
        &gpu,
        [0, 255, 0, 255],
        [255, 0, 0, 255],
        BlendMode::Screen,
        1.0,
    );
    assert_px(&img, 8, 8, [255, 255, 0, 255], 2);
}

#[test]
fn add_mid_gray_over_mid_gray_is_white() {
    let gpu = gpu_or_skip!();
    let img = render_blend(
        &gpu,
        [128, 128, 128, 255],
        [128, 128, 128, 255],
        BlendMode::Add,
        1.0,
    );
    assert_px(&img, 8, 8, [255, 255, 255, 255], 2);
}

#[test]
fn half_opacity_multiply_red_over_white() {
    // B(white, red) = multiply = red; co = red*0.5 + white*0.5 → (255,128,128).
    let gpu = gpu_or_skip!();
    let img = render_blend(
        &gpu,
        [255, 255, 255, 255],
        [255, 0, 0, 255],
        BlendMode::Multiply,
        0.5,
    );
    assert_px(&img, 8, 8, [255, 128, 128, 255], 2);
}

#[test]
fn normal_red_over_green_stays_fast_path_red() {
    let gpu = gpu_or_skip!();
    let img = render_blend(
        &gpu,
        [0, 255, 0, 255],
        [255, 0, 0, 255],
        BlendMode::Normal,
        1.0,
    );
    assert_px(&img, 8, 8, [255, 0, 0, 255], 1);
}
