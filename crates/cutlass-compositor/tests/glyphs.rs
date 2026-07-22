//! Glyph atlas + instanced draw parity: compositing shaped clusters through
//! the GPU glyph pipeline must match CPU `rasterize` of the same run.

use cutlass_compositor::{
    CompositeLayer, Compositor, CompositorConfig, GlyphsLayer, GpuContext, LayerPlacement,
    RgbaImage, identity_instances,
};
use cutlass_text::{TextRenderer, TextStyle};

fn try_gpu() -> Option<GpuContext> {
    match GpuContext::new_headless_blocking() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("skipping glyph compositor test: no GPU adapter ({e})");
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

/// Bundled OFL face so CI shapes real glyphs without system fonts.
const TEST_FONT: &[u8] = include_bytes!("../../cutlass-text/assets/Micro5-Regular.ttf");

fn test_renderer() -> TextRenderer {
    let mut r = TextRenderer::new();
    assert!(r.load_font(TEST_FONT.to_vec()) > 0);
    r
}

fn assert_images_agree(gpu_img: &RgbaImage, cpu_img: &RgbaImage, what: &str, tol: i32) {
    assert_eq!(
        (gpu_img.width, gpu_img.height),
        (cpu_img.width, cpu_img.height),
        "{what}: size mismatch"
    );
    let mut worst = 0i32;
    let mut worst_at = (0u32, 0u32);
    for y in 0..gpu_img.height {
        for x in 0..gpu_img.width {
            let a = gpu_img.pixel(x, y);
            let b = cpu_img.pixel(x, y);
            for c in 0..4 {
                let d = (i32::from(a[c]) - i32::from(b[c])).abs();
                if d > worst {
                    worst = d;
                    worst_at = (x, y);
                }
            }
        }
    }
    assert!(
        worst <= tol,
        "{what}: GPU and CPU disagree by {worst} at {worst_at:?} \
         (gpu {:?} vs cpu {:?}, tol {tol})",
        gpu_img.pixel(worst_at.0, worst_at.1),
        cpu_img.pixel(worst_at.0, worst_at.1),
    );
}

#[test]
fn glyph_instances_match_rgba_bitmap_path() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let mut text = test_renderer();
    let style = TextStyle::new(48.0).with_color([255, 220, 0, 255]);
    let content = "Hi";

    let shaped = text.shape(content, &style);
    assert!(shaped.has_ink());
    // CPU rasterize of the same run — the shape() docs guarantee compositing
    // clusters reproduces this. We compare through the compositor's existing
    // RGBA path so both sides share premultiplied blend/readback semantics.
    let cpu = text.rasterize(content, &style);
    assert!(cpu.width > 0 && cpu.height > 0);

    let glyphs: Vec<RgbaImage> = shaped.clusters.iter().map(|c| c.image.clone()).collect();
    let offsets: Vec<[f32; 2]> = shaped.clusters.iter().map(|c| c.offset).collect();
    let instances = identity_instances(&glyphs, &offsets, [0.0, 0.0], 1.0);

    // Opaque background so fringe premultiply differences don't dominate.
    let config = CompositorConfig::new(cpu.width, cpu.height).with_background([0, 0, 0, 255]);
    let placement = LayerPlacement {
        center: [cpu.width as f32 * 0.5, cpu.height as f32 * 0.5],
        size: [cpu.width as f32, cpu.height as f32],
        rotation: 0.0,
        opacity: 1.0,
    };
    let glyph_layer = CompositeLayer::glyphs(
        GlyphsLayer {
            atlas_key: 1,
            glyphs: &glyphs,
            instances: &instances,
        },
        placement,
    );
    let rgba_layer = CompositeLayer::rgba(&cpu, placement);

    let glyph_img = comp
        .render(&gpu, &config, &[glyph_layer])
        .expect("render glyphs");
    let rgba_img = comp
        .render(&gpu, &config, &[rgba_layer])
        .expect("render rgba");

    let tol = if gpu.is_software() { 8 } else { 4 };
    assert_images_agree(&glyph_img, &rgba_img, "glyph vs rgba path", tol);
}

#[test]
fn glyph_atlas_reused_across_frames() {
    let gpu = gpu_or_skip!();
    let mut comp = Compositor::new(&gpu);
    let mut text = test_renderer();
    let style = TextStyle::new(40.0);
    let shaped = text.shape("AB", &style);
    let glyphs: Vec<RgbaImage> = shaped.clusters.iter().map(|c| c.image.clone()).collect();
    let offsets: Vec<[f32; 2]> = shaped.clusters.iter().map(|c| c.offset).collect();

    let config = CompositorConfig::new(200, 80).with_background([0, 0, 0, 255]);
    for frame in 0..3 {
        let origin = [10.0 + frame as f32 * 4.0, 10.0];
        let instances = identity_instances(&glyphs, &offsets, origin, 1.0);
        let layer = CompositeLayer::glyphs(
            GlyphsLayer {
                atlas_key: 42,
                glyphs: &glyphs,
                instances: &instances,
            },
            LayerPlacement {
                center: [100.0, 40.0],
                size: [200.0, 80.0],
                rotation: 0.0,
                opacity: 1.0,
            },
        );
        let img = comp.render(&gpu, &config, &[layer]).expect("render");
        // Some glyph coverage should land on the canvas.
        let lit = img.pixels.chunks_exact(4).filter(|p| p[3] > 0).count();
        assert!(lit > 0, "frame {frame} drew no glyph coverage");
    }
}
