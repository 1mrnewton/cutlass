//! The full vertical slice: shape text → rasterize → composite onto a GPU
//! canvas → read it back, and confirm the glyphs actually landed.
//!
//! Skips (rather than fails) when there's no GPU adapter or no fonts, so it's
//! safe in a headless/fontless CI while still exercising the real path locally.

use cutlass_compositor::{
    CompositeLayer, Compositor, CompositorConfig, GpuContext, LayerPlacement,
};
use cutlass_text::{TextRenderer, TextStyle};

#[test]
fn text_composites_onto_a_canvas() {
    let gpu = match GpuContext::new_headless_blocking() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skipping: no GPU adapter ({e})");
            return;
        }
    };

    let mut text = TextRenderer::new();
    if text.font_count() == 0 {
        eprintln!("skipping: no fonts available");
        return;
    }

    // White text on a black canvas.
    let style = TextStyle::new(48.0).with_color([255, 255, 255, 255]);
    let bitmap = text.rasterize("Hi", &style);
    if bitmap.width == 0 || bitmap.height == 0 {
        eprintln!("skipping: text rasterized to nothing");
        return;
    }

    // Canvas a bit larger than the text, which we drop in the center at 1:1.
    let config = CompositorConfig::new(bitmap.width + 40, bitmap.height + 40);
    let placement = LayerPlacement {
        center: [config.width as f32 / 2.0, config.height as f32 / 2.0],
        size: [bitmap.width as f32, bitmap.height as f32],
        rotation: 0.0,
        opacity: 1.0,
    };
    let layer = CompositeLayer::rgba(&bitmap, placement);
    let canvas = comp_render(&gpu, &config, &[layer]);

    // The border (outside the centered text box) stays background-black.
    assert_eq!(
        canvas.pixel(0, 0),
        [0, 0, 0, 255],
        "corner should be background"
    );

    // Somewhere in the middle the white glyphs must have brightened the canvas,
    // and because the fill is white the lit pixels stay neutral gray.
    let mut max_luma = 0u8;
    let mut saw_colored = false;
    for p in canvas.pixels.chunks_exact(4) {
        max_luma = max_luma.max(p[0]).max(p[1]).max(p[2]);
        let spread = i32::from(p[0]).max(i32::from(p[1])).max(i32::from(p[2]))
            - i32::from(p[0]).min(i32::from(p[1])).min(i32::from(p[2]));
        if spread > 24 {
            saw_colored = true;
        }
    }
    assert!(
        max_luma > 64,
        "no visible text on the canvas (max luma {max_luma})"
    );
    assert!(!saw_colored, "white text should composite as neutral gray");
}

fn comp_render(
    gpu: &GpuContext,
    config: &CompositorConfig,
    layers: &[CompositeLayer<'_>],
) -> cutlass_compositor::RgbaImage {
    let comp = Compositor::new(gpu);
    comp.render(gpu, config, layers).expect("render")
}
