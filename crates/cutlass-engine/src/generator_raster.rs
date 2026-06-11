//! CPU rasterization of generated clips (text, shapes) into full-canvas RGBA
//! buffers that ride the existing [`CompositeLayer::Rgba`] upload path.
//!
//! Rasters are cached by `(content, canvas size)` so playback — which calls
//! `resolve_layers` once per frame — pays the raster cost once per distinct
//! generator/canvas pair, not every frame. The output is straight-alpha RGBA
//! over a transparent ground, matching the compositor's src-over blend.

use std::collections::VecDeque;
use std::sync::Arc;

use cosmic_text::{Attrs, Buffer, Color as TextColor, FontSystem, Metrics, Shaping, SwashCache};
use cutlass_models::{Generator, Shape};
use tiny_skia::{FillRule, Paint, PathBuilder, Pixmap, Rect, Transform};

/// Most timelines stack only a handful of generated clips; a small cache keeps
/// every visible text/shape raster warm without unbounded growth.
const CACHE_CAP: usize = 24;

/// Cache key for a rasterized generator. Keyed on the visible parameters plus
/// the canvas size (a resize invalidates the bitmap).
#[derive(Clone, PartialEq, Eq, Hash)]
enum RasterKey {
    Text { content: String, w: u32, h: u32 },
    Shape { shape: ShapeKey, rgba: [u8; 4], w: u32, h: u32 },
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum ShapeKey {
    Rectangle,
    Ellipse,
}

fn shape_key(shape: Shape) -> ShapeKey {
    match shape {
        Shape::Rectangle => ShapeKey::Rectangle,
        Shape::Ellipse => ShapeKey::Ellipse,
    }
}

/// A cached raster plus the tight bounding-box size of its non-transparent
/// pixels. Both raster paths center their content in the canvas, so the size
/// alone places the content rect (centered on the layer center).
#[derive(Clone)]
struct CachedRaster {
    bytes: Arc<Vec<u8>>,
    content: (u32, u32),
}

/// Rasterizes text and shape generators, caching results. Owned by the engine
/// alongside the decoder pool.
pub struct GeneratorRaster {
    /// Lazily initialized: scanning system fonts is slow, and most engines
    /// (tests, audio-only sessions) never render text.
    font_system: Option<FontSystem>,
    swash_cache: SwashCache,
    cache: VecDeque<(RasterKey, CachedRaster)>,
}

impl Default for GeneratorRaster {
    fn default() -> Self {
        Self::new()
    }
}

impl GeneratorRaster {
    pub fn new() -> Self {
        Self {
            font_system: None,
            swash_cache: SwashCache::new(),
            cache: VecDeque::new(),
        }
    }

    /// Rasterize a generator to a full-canvas straight-alpha RGBA buffer.
    /// Returns `None` for generators that have no raster representation yet
    /// (sticker/effect/filter/adjustment) or for a zero-size canvas — callers
    /// skip those layers, as before.
    pub fn raster(&mut self, generator: &Generator, width: u32, height: u32) -> Option<Arc<Vec<u8>>> {
        self.entry(generator, width, height).map(|e| e.bytes)
    }

    /// Tight size (canvas px) of the content a generator actually draws on a
    /// `width`×`height` canvas: the whole canvas for solids, the measured
    /// alpha bounding box for text and shapes (`(0, 0)` when nothing is
    /// drawn, e.g. empty text). `None` for generators the compositor doesn't
    /// draw. This is what a selection box should hug — the raster itself is
    /// canvas-sized and mostly transparent.
    pub fn content_size(
        &mut self,
        generator: &Generator,
        width: u32,
        height: u32,
    ) -> Option<(u32, u32)> {
        if width == 0 || height == 0 {
            return None;
        }
        if matches!(generator, Generator::SolidColor { .. }) {
            return Some((width, height));
        }
        self.entry(generator, width, height).map(|e| e.content)
    }

    /// Cache-or-raster: the shared path behind [`raster`](Self::raster) and
    /// [`content_size`](Self::content_size).
    fn entry(&mut self, generator: &Generator, width: u32, height: u32) -> Option<CachedRaster> {
        if width == 0 || height == 0 {
            return None;
        }
        let key = match generator {
            Generator::Text { content } => RasterKey::Text {
                content: content.clone(),
                w: width,
                h: height,
            },
            Generator::Shape { shape, rgba } => RasterKey::Shape {
                shape: shape_key(*shape),
                rgba: *rgba,
                w: width,
                h: height,
            },
            _ => return None,
        };

        if let Some(hit) = self.lookup(&key) {
            return Some(hit);
        }

        let bytes = match generator {
            Generator::Text { content } => self.raster_text(content, width, height),
            Generator::Shape { shape, rgba } => raster_shape(*shape, *rgba, width, height),
            _ => unreachable!("filtered above"),
        };
        let entry = CachedRaster {
            content: alpha_bbox_size(&bytes, width),
            bytes: Arc::new(bytes),
        };
        self.insert(key, entry.clone());
        Some(entry)
    }

    fn lookup(&mut self, key: &RasterKey) -> Option<CachedRaster> {
        let pos = self.cache.iter().position(|(k, _)| k == key)?;
        let (k, entry) = self.cache.remove(pos).expect("position is valid");
        self.cache.push_back((k, entry.clone()));
        Some(entry)
    }

    fn insert(&mut self, key: RasterKey, value: CachedRaster) {
        if self.cache.len() >= CACHE_CAP {
            self.cache.pop_front();
        }
        self.cache.push_back((key, value));
    }

    fn raster_text(&mut self, content: &str, width: u32, height: u32) -> Vec<u8> {
        let mut out = vec![0u8; (width as usize) * (height as usize) * 4];
        if content.trim().is_empty() {
            return out;
        }

        let font_system = self.font_system.get_or_insert_with(FontSystem::new);
        let swash = &mut self.swash_cache;

        // CapCut-like default title styling: white, centered, sized to the
        // canvas, wrapped at 90% width.
        let font_size = (height as f32 / 12.0).max(8.0);
        let line_height = font_size * 1.2;
        let metrics = Metrics::new(font_size, line_height);

        let mut buffer = Buffer::new(font_system, metrics);
        let wrap_w = width as f32 * 0.9;
        buffer.set_size(font_system, Some(wrap_w), Some(height as f32));
        buffer.set_text(font_system, content, &Attrs::new(), Shaping::Advanced);
        buffer.shape_until_scroll(font_system, false);

        // Vertically center the laid-out block within the canvas.
        let text_h = buffer
            .layout_runs()
            .fold(0.0_f32, |m, run| m.max(run.line_top + run.line_height));
        let y_off = (((height as f32) - text_h) / 2.0).round() as i32;

        let text_color = TextColor::rgba(255, 255, 255, 255);
        let canvas_w = width as i32;
        let canvas_h = height as i32;

        for run in buffer.layout_runs() {
            // Horizontally center each line within the full canvas width.
            let line_x_off = (((width as f32) - run.line_w) / 2.0).round() as i32;
            let base_y = run.line_y as i32 + y_off;
            for glyph in run.glyphs.iter() {
                let physical = glyph.physical((0.0, 0.0), 1.0);
                let glyph_color = glyph.color_opt.unwrap_or(text_color);
                swash.with_pixels(font_system, physical.cache_key, glyph_color, |x, y, color| {
                    let px = line_x_off + physical.x + x;
                    let py = base_y + physical.y + y;
                    blend_over(&mut out, canvas_w, canvas_h, px, py, color);
                });
            }
        }

        out
    }
}

/// Tight bounding-box size of pixels with non-zero alpha in an RGBA buffer,
/// `(0, 0)` when fully transparent. One O(w·h) pass per raster build (cold:
/// rasters are cached), so the box is exact for any generator — including
/// text, whose extent only layout knows.
fn alpha_bbox_size(bytes: &[u8], width: u32) -> (u32, u32) {
    let w = width as usize;
    let (mut min_x, mut min_y) = (usize::MAX, usize::MAX);
    let (mut max_x, mut max_y) = (0usize, 0usize);
    let mut any = false;
    for (i, px) in bytes.chunks_exact(4).enumerate() {
        if px[3] == 0 {
            continue;
        }
        let (x, y) = (i % w, i / w);
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
        any = true;
    }
    if !any {
        return (0, 0);
    }
    ((max_x - min_x + 1) as u32, (max_y - min_y + 1) as u32)
}

/// Straight-alpha src-over of a coverage-weighted glyph pixel onto the buffer.
fn blend_over(buf: &mut [u8], w: i32, h: i32, x: i32, y: i32, src: TextColor) {
    if x < 0 || y < 0 || x >= w || y >= h {
        return;
    }
    let sa = src.a() as f32 / 255.0;
    if sa <= 0.0 {
        return;
    }
    let idx = ((y * w + x) * 4) as usize;
    let da = buf[idx + 3] as f32 / 255.0;
    let out_a = sa + da * (1.0 - sa);
    if out_a <= 0.0 {
        return;
    }
    let blend = |s: u8, d: u8| -> u8 {
        let s = s as f32 / 255.0;
        let d = d as f32 / 255.0;
        let v = (s * sa + d * da * (1.0 - sa)) / out_a;
        (v * 255.0).round().clamp(0.0, 255.0) as u8
    };
    buf[idx] = blend(src.r(), buf[idx]);
    buf[idx + 1] = blend(src.g(), buf[idx + 1]);
    buf[idx + 2] = blend(src.b(), buf[idx + 2]);
    buf[idx + 3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
}

/// Rasterize a centered shape covering the middle 50% of the canvas.
fn raster_shape(shape: Shape, rgba: [u8; 4], width: u32, height: u32) -> Vec<u8> {
    let mut out = vec![0u8; (width as usize) * (height as usize) * 4];
    let Some(mut pixmap) = Pixmap::new(width, height) else {
        return out;
    };

    let mut paint = Paint::default();
    paint.set_color_rgba8(rgba[0], rgba[1], rgba[2], rgba[3]);
    paint.anti_alias = true;

    let ext_w = width as f32 * 0.5;
    let ext_h = height as f32 * 0.5;
    let x = (width as f32 - ext_w) / 2.0;
    let y = (height as f32 - ext_h) / 2.0;
    let Some(rect) = Rect::from_xywh(x, y, ext_w, ext_h) else {
        return out;
    };

    let mut pb = PathBuilder::new();
    match shape {
        Shape::Rectangle => pb.push_rect(rect),
        Shape::Ellipse => pb.push_oval(rect),
    }
    let Some(path) = pb.finish() else {
        return out;
    };

    pixmap.fill_path(
        &path,
        &paint,
        FillRule::Winding,
        Transform::identity(),
        None,
    );

    // tiny-skia stores premultiplied alpha; the compositor blends straight
    // alpha, so demultiply each pixel.
    for (i, px) in pixmap.pixels().iter().enumerate() {
        let c = px.demultiply();
        let o = i * 4;
        out[o] = c.red();
        out[o + 1] = c.green();
        out[o + 2] = c.blue();
        out[o + 3] = c.alpha();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alpha_at(buf: &[u8], w: u32, x: u32, y: u32) -> u8 {
        buf[((y * w + x) * 4 + 3) as usize]
    }

    #[test]
    fn shape_rect_fills_center_not_corner() {
        let (w, h) = (64, 64);
        let buf = raster_shape(Shape::Rectangle, [255, 0, 0, 255], w, h);
        assert_eq!(buf.len(), (w * h * 4) as usize);
        // Center is inside the middle 50% box.
        assert_eq!(alpha_at(&buf, w, w / 2, h / 2), 255);
        // Red channel set at the center.
        let center = ((h / 2 * w + w / 2) * 4) as usize;
        assert_eq!(buf[center], 255);
        // Corner is outside the box → transparent.
        assert_eq!(alpha_at(&buf, w, 0, 0), 0);
    }

    #[test]
    fn shape_ellipse_corner_of_box_is_empty() {
        let (w, h) = (64, 64);
        let buf = raster_shape(Shape::Ellipse, [0, 255, 0, 255], w, h);
        // Center filled.
        assert_eq!(alpha_at(&buf, w, w / 2, h / 2), 255);
        // The box spans [16,48); its top-left corner (16,16) sits outside the
        // inscribed ellipse, so it should be (near) transparent.
        assert_eq!(alpha_at(&buf, w, 16, 16), 0);
    }

    #[test]
    fn raster_caches_repeated_lookups() {
        let mut raster = GeneratorRaster::new();
        let generator = Generator::Shape {
            shape: Shape::Rectangle,
            rgba: [10, 20, 30, 255],
        };
        let a = raster.raster(&generator, 32, 32).unwrap();
        let b = raster.raster(&generator, 32, 32).unwrap();
        // Same Arc allocation on the second call ⇒ cache hit.
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn content_size_hugs_the_drawn_content() {
        let mut raster = GeneratorRaster::new();
        // Shapes draw the centered middle 50% of the canvas; the rectangle's
        // box is pixel-exact, the ellipse inscribes the same rect.
        let rect = Generator::Shape {
            shape: Shape::Rectangle,
            rgba: [255, 0, 0, 255],
        };
        assert_eq!(raster.content_size(&rect, 64, 64), Some((32, 32)));
        let ellipse = Generator::Shape {
            shape: Shape::Ellipse,
            rgba: [0, 255, 0, 255],
        };
        let (ew, eh) = raster.content_size(&ellipse, 64, 64).unwrap();
        assert!((30..=32).contains(&ew) && (30..=32).contains(&eh), "{ew}x{eh}");
        // Solids cover the whole canvas (no raster involved).
        let solid = Generator::SolidColor { rgba: [1, 2, 3, 255] };
        assert_eq!(raster.content_size(&solid, 64, 48), Some((64, 48)));
        // Text measures its laid-out block — smaller than the canvas.
        let text = Generator::Text { content: "Hi".into() };
        let (tw, th) = raster.content_size(&text, 256, 128).unwrap();
        assert!(tw > 0 && tw < 256, "text width {tw}");
        assert!(th > 0 && th < 128, "text height {th}");
        // Empty text draws nothing.
        let empty = Generator::Text { content: " ".into() };
        assert_eq!(raster.content_size(&empty, 256, 128), Some((0, 0)));
        // Unsupported generators have no raster, hence no content.
        assert_eq!(raster.content_size(&Generator::Sticker, 64, 64), None);
        assert_eq!(raster.content_size(&rect, 0, 64), None);
    }

    #[test]
    fn unsupported_generators_return_none() {
        let mut raster = GeneratorRaster::new();
        assert!(raster.raster(&Generator::Sticker, 32, 32).is_none());
        assert!(
            raster
                .raster(
                    &Generator::Text {
                        content: "x".into()
                    },
                    0,
                    0
                )
                .is_none()
        );
    }

    #[test]
    fn text_draws_pixels() {
        let mut raster = GeneratorRaster::new();
        let buf = raster
            .raster(
                &Generator::Text {
                    content: "Hi".into(),
                },
                256,
                128,
            )
            .unwrap();
        let any_opaque = buf.chunks_exact(4).any(|p| p[3] > 0);
        assert!(any_opaque, "text raster produced no visible pixels");
    }
}
