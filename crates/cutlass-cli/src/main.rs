//! A tiny end-to-end demo of the Cutlass compositor.
//!
//! Builds a scene from a solid fill plus a rasterized text layer, renders it on
//! a headless GPU, and writes the composited canvas out as a PNG so the whole
//! generator → composite → readback path can be eyeballed.
//!
//! ```text
//! cargo run -p cutlass-cli              # writes ./cutlass-demo.png
//! cargo run -p cutlass-cli out.png      # writes ./out.png
//! ```

use std::error::Error;
use std::path::{Path, PathBuf};

use cutlass_compositor::{
    CompositeLayer, Compositor, CompositorConfig, GpuContext, LayerPlacement, RgbaImage,
};
use cutlass_text::{TextRenderer, TextStyle};

const WIDTH: u32 = 1920 * 4;
const HEIGHT: u32 = 1080 * 4;
const BACKGROUND_COLOR: [u8; 4] = [20, 20, 30, 255];
const TEXT_COLOR: [u8; 4] = [255, 255, 255, 255];
const TEXT_SIZE: f32 = 600.0;

fn main() -> Result<(), Box<dyn Error>> {
    let out_path = std::env::args()
        .nth(1)
        .map_or_else(|| PathBuf::from("cutlass-demo.png"), PathBuf::from);

    let gpu = GpuContext::new_headless_blocking()?; // bring up a headless GPU
    let mut comp = Compositor::new(&gpu); // build pipelines once
    let config = CompositorConfig::new(WIDTH, HEIGHT).with_background(BACKGROUND_COLOR);

    // Text is a generator: it rasterizes to a straight-alpha RgbaImage.
    let mut text = TextRenderer::new();
    let title = text.rasterize("Cutlass", &TextStyle::new(TEXT_SIZE).with_color(TEXT_COLOR));

    let mut layers = vec![CompositeLayer::solid(
        BACKGROUND_COLOR,
        LayerPlacement::full_canvas(&config),
    )];
    // A fontless host rasterizes to a 0×0 bitmap, which the compositor rejects;
    // only stack the text layer when there are actually glyphs to place.
    if title.width > 0 && title.height > 0 {
        let (tw, th) = (title.width as f32, title.height as f32);
        // Snap the top-left corner to an integer pixel. This is an unscaled 1:1
        // blit, so a half-pixel corner makes every destination pixel sample
        // halfway between two texels and the linear sampler smears the whole
        // run by half a texel (soft, but not blurry). An odd width/height
        // centered on an integer canvas midpoint is exactly that case.
        let left = (WIDTH as f32 / 2.0 - tw / 2.0).round();
        let top = (HEIGHT as f32 / 2.0 - th / 2.0).round();
        eprintln!("text bitmap {tw}x{th}; top-left snapped to ({left}, {top})");
        layers.push(CompositeLayer::rgba(
            &title,
            LayerPlacement {
                center: [left + tw / 2.0, top + th / 2.0],
                size: [tw, th],
                rotation: 0.0,
                opacity: 1.0,
            },
        ));
    } else {
        eprintln!("note: no fonts available; rendering the background only");
    }

    let out = comp.render(&gpu, &config, &layers)?; // -> RgbaImage { width, height, pixels }
    write_png(&out, &out_path)?;

    println!(
        "wrote {}x{} canvas to {}",
        out.width,
        out.height,
        out_path.display()
    );
    Ok(())
}

/// Write a straight-alpha [`RgbaImage`] to an 8-bit RGBA PNG.
fn write_png(img: &RgbaImage, path: &Path) -> Result<(), Box<dyn Error>> {
    let file = std::fs::File::create(path)?;
    let writer = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, img.width, img.height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&img.pixels)?;
    Ok(())
}
