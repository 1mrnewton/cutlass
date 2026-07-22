//! GPU readback check: temporal supersampling smears a moving white bar.

use cutlass_compositor::{
    CompositeLayer, Compositor, CompositorConfig, GpuContext, LayerPlacement, RgbaImage,
};

fn try_gpu() -> Option<GpuContext> {
    match GpuContext::new_headless_blocking() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("skipping motion blur test: no GPU adapter ({e})");
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

fn luma(img: &RgbaImage, x: u32, y: u32) -> u8 {
    let p = img.pixel(x, y);
    // Approximate luminance from premultiplied-ish opaque white smear.
    p[0]
}

#[test]
fn moving_bar_supersamples_to_smear() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(64, 16).with_background([0, 0, 0, 255]);

    // Thin white bar: 8×8, swept horizontally across 8 placements.
    let bar_size = [8.0f32, 8.0];
    let y = 8.0;
    let mut passes = Vec::new();
    for i in 0..8 {
        let x = 8.0 + i as f32 * 6.0; // centers 8,14,20,...,50
        passes.push(LayerPlacement {
            center: [x, y],
            size: bar_size,
            rotation: 0.0,
            opacity: 1.0,
        });
    }
    let mid = passes[3];
    let layer = CompositeLayer::solid([255, 255, 255, 255], mid).with_blur_passes(passes);

    let smeared = comp.render(&gpu, &config, &[layer]).expect("render blur");

    // Static control at the mid placement only.
    let static_layer = CompositeLayer::solid([255, 255, 255, 255], mid);
    let sharp = comp
        .render(&gpu, &config, &[static_layer])
        .expect("render static");

    // Center of the sweep (~x=29) should be brighter than a far end with blur,
    // and the smear should light a pixel that the static mid bar does not cover.
    let center_x = 29u32;
    let end_x = 8u32;
    let between_x = 20u32; // between start and mid — covered by smear, not by static mid

    let smear_center = luma(&smeared, center_x, 8);
    let smear_end = luma(&smeared, end_x, 8);
    let smear_between = luma(&smeared, between_x, 8);
    let sharp_between = luma(&sharp, between_x, 8);

    assert!(
        smear_center > smear_end,
        "center of sweep should be brighter than the end: center={smear_center} end={smear_end}"
    );
    assert!(
        smear_between > 20,
        "smear should cover intermediate x={between_x}: luma={smear_between}"
    );
    assert!(
        sharp_between < 10,
        "static mid bar should not cover x={between_x}: luma={sharp_between}"
    );
    assert!(
        smear_between != sharp_between || smear_center != luma(&sharp, center_x, 8),
        "blurred and static frames must differ"
    );
}
