//! GPU readback checks for per-layer shadow, glow, outline, and background styles.

use cutlass_compositor::{
    CompositeLayer, Compositor, CompositorConfig, GpuContext, LayerBackground, LayerGlow,
    LayerOutline, LayerPlacement, LayerShadow, LayerStyles, RgbaImage,
};

/// Try to bring up a headless GPU; `None` (skip) if the host has no adapter.
fn try_gpu() -> Option<GpuContext> {
    match GpuContext::new_headless_blocking() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("skipping styles test: no GPU adapter ({e})");
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

#[track_caller]
fn assert_not_px(img: &RgbaImage, x: u32, y: u32, forbidden: [u8; 4], tol: i32) {
    let got = img.pixel(x, y);
    let all_close = (0..4).all(|ch| (i32::from(got[ch]) - i32::from(forbidden[ch])).abs() <= tol);
    assert!(
        !all_close,
        "pixel({x},{y}) = {got:?} unexpectedly matches ~{forbidden:?} (tol {tol})"
    );
}

fn centered_red_layer(config: &CompositorConfig) -> CompositeLayer<'static> {
    let placement = LayerPlacement {
        center: [config.width as f32 / 2.0, config.height as f32 / 2.0],
        size: [100.0, 100.0],
        rotation: 0.0,
        opacity: 1.0,
    };
    CompositeLayer::solid([255, 0, 0, 255], placement)
}

#[test]
fn shadow_offset_no_blur_on_white_canvas() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(400, 400).with_background([255, 255, 255, 255]);
    let layer = centered_red_layer(&config).with_styles(LayerStyles {
        shadow: Some(LayerShadow {
            rgba: [0, 0, 0, 255],
            offset: [50.0, 0.0],
            blur: 0.0,
        }),
        ..Default::default()
    });
    let img = comp.render(&gpu, &config, &[layer]).expect("render");

    let cx = 200u32;
    let cy = 200u32;
    // Content center stays red.
    assert_px(&img, cx, cy, [255, 0, 0, 255], 3);
    // Inside the offset silhouette (center + 75, center), outside the red quad.
    assert_px(&img, cx + 75, cy, [0, 0, 0, 255], 3);
    // Far corner still white.
    assert_px(&img, 10, 10, [255, 255, 255, 255], 3);
}

#[test]
fn shadow_blur_softens_silhouette_edge() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(400, 400).with_background([255, 255, 255, 255]);
    let layer = centered_red_layer(&config).with_styles(LayerStyles {
        shadow: Some(LayerShadow {
            rgba: [0, 0, 0, 255],
            offset: [50.0, 0.0],
            blur: 12.0,
        }),
        ..Default::default()
    });
    let img = comp.render(&gpu, &config, &[layer]).expect("render");

    // Sharp silhouette right edge is at x = 200+50+50 = 300. Probe just outside.
    let probe_x = 302u32;
    let probe_y = 200u32;
    let got = img.pixel(probe_x, probe_y);
    assert_not_px(&img, probe_x, probe_y, [255, 255, 255, 255], 3);
    assert_not_px(&img, probe_x, probe_y, [0, 0, 0, 255], 3);
    // Partial darkening: channels between white and black.
    assert!(
        got[0] > 3 && got[0] < 252,
        "expected partial shadow darkening, got {got:?}"
    );
}

#[test]
fn glow_extends_outside_content_on_black() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(400, 400).with_background([0, 0, 0, 255]);
    let layer = centered_red_layer(&config).with_styles(LayerStyles {
        glow: Some(LayerGlow {
            rgba: [255, 255, 255, 255],
            radius: 10.0,
            intensity: 2.0,
        }),
        ..Default::default()
    });
    let img = comp.render(&gpu, &config, &[layer]).expect("render");

    let cx = 200u32;
    let cy = 200u32;
    // Content remains red-ish (glow under content).
    let center = img.pixel(cx, cy);
    assert!(
        center[0] > 200 && center[1] < 80 && center[2] < 80,
        "content should stay red-ish, got {center:?}"
    );
    // 4px outside the content edge (right edge at x=250).
    let glow_px = img.pixel(254, cy);
    assert!(
        glow_px[0] > 3 || glow_px[1] > 3 || glow_px[2] > 3,
        "expected non-black glow outside content, got {glow_px:?}"
    );
}

#[test]
fn empty_styles_matches_styles_free_render() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(400, 400).with_background([0, 0, 0, 255]);
    let layer = centered_red_layer(&config);
    let control = comp.render(&gpu, &config, &[layer]).expect("control");

    let layer_empty = centered_red_layer(&config).with_styles(LayerStyles::default());
    let with_empty = comp
        .render(&gpu, &config, &[layer_empty])
        .expect("empty styles");

    assert_px(&with_empty, 200, 200, control.pixel(200, 200), 3);
    assert_px(&with_empty, 10, 10, control.pixel(10, 10), 3);
    assert_px(&with_empty, 250, 200, control.pixel(250, 200), 3);
}

#[test]
fn outline_white_ring_outside_red_on_black() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(400, 400).with_background([0, 0, 0, 255]);
    let layer = centered_red_layer(&config).with_styles(LayerStyles {
        outline: Some(LayerOutline {
            rgba: [255, 255, 255, 255],
            width: 6.0,
        }),
        ..Default::default()
    });
    let img = comp.render(&gpu, &config, &[layer]).expect("render");

    let cx = 200u32;
    let cy = 200u32;
    // Content center stays red.
    assert_px(&img, cx, cy, [255, 0, 0, 255], 3);
    // ~2–3px outside the right content edge (edge near x=250). With
    // smoothstep(0.02, 0.35) harden, full white covers the inner ring.
    let ring = img.pixel(252, cy);
    assert!(
        ring[0] > 200 && ring[1] > 200 && ring[2] > 200,
        "expected white-ish outline ring, got {ring:?}"
    );
    // Well outside the dilation.
    assert_px(&img, cx + 80, cy, [0, 0, 0, 255], 3);
}

#[test]
fn background_plate_padded_white_behind_red() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(400, 400).with_background([0, 0, 0, 255]);
    let layer = centered_red_layer(&config).with_styles(LayerStyles {
        background: Some(LayerBackground {
            rgba: [255, 255, 255, 255],
            padding: 20.0,
            radius: 0.0,
        }),
        ..Default::default()
    });
    let img = comp.render(&gpu, &config, &[layer]).expect("render");

    let cx = 200u32;
    let cy = 200u32;
    assert_px(&img, cx, cy, [255, 0, 0, 255], 3);
    // Content right edge at 250; +10px is inside the padding plate.
    assert_px(&img, 260, cy, [255, 255, 255, 255], 3);
    // Far corner stays black.
    assert_px(&img, 10, 10, [0, 0, 0, 255], 3);
}

#[test]
fn background_plate_rotates_with_layer() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(400, 400).with_background([0, 0, 0, 255]);
    let mut layer = centered_red_layer(&config);
    layer.placement.rotation = std::f32::consts::FRAC_PI_4;
    let layer = layer.with_styles(LayerStyles {
        background: Some(LayerBackground {
            rgba: [255, 255, 255, 255],
            padding: 20.0,
            radius: 0.0,
        }),
        ..Default::default()
    });
    let img = comp.render(&gpu, &config, &[layer]).expect("render");

    // Unrotated plate corner (130,130) lies outside the 45°-rotated plate.
    assert_px(&img, 130, 130, [0, 0, 0, 255], 3);
    // Content center still red.
    assert_px(&img, 200, 200, [255, 0, 0, 255], 3);
}

#[test]
fn shadow_outline_background_combined() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(400, 400).with_background([0, 0, 0, 255]);
    let layer = centered_red_layer(&config).with_styles(LayerStyles {
        shadow: Some(LayerShadow {
            rgba: [0, 0, 0, 180],
            offset: [8.0, 8.0],
            blur: 4.0,
        }),
        outline: Some(LayerOutline {
            rgba: [255, 255, 255, 255],
            width: 4.0,
        }),
        background: Some(LayerBackground {
            rgba: [40, 40, 40, 255],
            padding: 12.0,
            radius: 4.0,
        }),
        ..Default::default()
    });
    let img = comp.render(&gpu, &config, &[layer]).expect("render");
    assert_px(&img, 200, 200, [255, 0, 0, 255], 3);
}
