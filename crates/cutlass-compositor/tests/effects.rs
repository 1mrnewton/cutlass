//! GPU readback checks for catalog effect passes (color_overlay, duotone).

use cutlass_compositor::{
    CompositeLayer, Compositor, CompositorConfig, GpuContext, LayerPlacement, PassInstance,
    RgbaImage,
};

/// Try to bring up a headless GPU; `None` (skip) if the host has no adapter.
fn try_gpu() -> Option<GpuContext> {
    match GpuContext::new_headless_blocking() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("skipping effects test: no GPU adapter ({e})");
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

fn horizontal_gradient(width: u32, height: u32) -> RgbaImage {
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for _y in 0..height {
        for x in 0..width {
            let t = if width <= 1 {
                0.0
            } else {
                x as f32 / (width - 1) as f32
            };
            let v = (t * 255.0).round() as u8;
            pixels.extend_from_slice(&[v, v, v, 255]);
        }
    }
    RgbaImage::new(width, height, pixels)
}

#[test]
fn duotone_maps_black_and_white_to_shadow_and_highlight() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(64, 16).with_background([0, 0, 0, 255]);
    let bmp = horizontal_gradient(64, 16);

    // shadow=[20,16,60,255], highlight=[255,220,160,255], intensity=1
    let params = [
        20.0 / 255.0,
        16.0 / 255.0,
        60.0 / 255.0,
        1.0,
        255.0 / 255.0,
        220.0 / 255.0,
        160.0 / 255.0,
        1.0,
        1.0,
    ];
    let effects = [PassInstance {
        id: "duotone",
        params: &params,
    }];
    let layer =
        CompositeLayer::rgba(&bmp, LayerPlacement::full_canvas(&config)).with_effects(&effects);
    let img = comp.render(&gpu, &config, &[layer]).expect("duotone");

    assert_px(&img, 0, 8, [20, 16, 60, 255], 3);
    assert_px(&img, 63, 8, [255, 220, 160, 255], 3);
}

#[test]
fn color_overlay_amount_one_tints_gray_red() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(16, 16).with_background([0, 0, 0, 255]);

    // color=red opaque, offset=0, amount=1 → full red overlay
    let params = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let effects = [PassInstance {
        id: "color_overlay",
        params: &params,
    }];
    let layer = CompositeLayer::solid([128, 128, 128, 255], LayerPlacement::full_canvas(&config))
        .with_effects(&effects);
    let img = comp.render(&gpu, &config, &[layer]).expect("overlay");

    assert_px(&img, 8, 8, [255, 0, 0, 255], 3);
}

#[test]
fn color_overlay_amount_zero_leaves_gray_unchanged() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(16, 16).with_background([0, 0, 0, 255]);

    // amount=0 → noop (skipped by effect_is_noop)
    let params = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let effects = [PassInstance {
        id: "color_overlay",
        params: &params,
    }];
    let layer = CompositeLayer::solid([128, 128, 128, 255], LayerPlacement::full_canvas(&config))
        .with_effects(&effects);
    let img = comp.render(&gpu, &config, &[layer]).expect("overlay zero");

    assert_px(&img, 8, 8, [128, 128, 128, 255], 2);
}
