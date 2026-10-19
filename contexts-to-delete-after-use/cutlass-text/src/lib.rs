//! cutlass-text: shape and rasterize styled text into a straight-alpha
//! [`RgbaImage`] the compositor can place as a generator layer.
//!
//! Text is a **generator** — pixels are produced, not decoded — so this crate
//! stays pure-Rust and GPU-free: it depends only on [`cosmic_text`] (shaping +
//! layout via `rustybuzz`/`swash`) and [`cutlass_core`]. The engine wires the
//! resulting bitmap to a `cutlass_compositor` RGBA layer; nothing here knows
//! about `wgpu`.
//!
//! ```no_run
//! use cutlass_text::{TextRenderer, TextStyle};
//!
//! let mut text = TextRenderer::new();
//! let style = TextStyle::new(48.0).with_color([255, 220, 0, 255]);
//! let bitmap = text.rasterize("Cutlass", &style); // straight-alpha RgbaImage
//! ```
//!
//! Fonts come from the host by default ([`TextRenderer::new`]); call
//! [`TextRenderer::load_font`] to add bundled/custom faces for deterministic,
//! platform-independent output.

mod style;

use cosmic_text::{Align, Attrs, Buffer, Color, Family, FontSystem, Metrics, Shaping, SwashCache};
use cutlass_core::RgbaImage;

pub use style::{FontFamily, TextAlign, TextStyle};

/// Shapes and rasterizes text. Holds the font set and a glyph cache, so reuse
/// one renderer across many `rasterize` calls (the cache amortizes shaping).
pub struct TextRenderer {
    font_system: FontSystem,
    swash_cache: SwashCache,
}

impl TextRenderer {
    /// A renderer backed by the host's installed fonts.
    pub fn new() -> Self {
        Self {
            font_system: FontSystem::new(),
            swash_cache: SwashCache::new(),
        }
    }

    /// Add a font from raw TTF/OTF bytes (e.g. a bundled face) to the set.
    pub fn load_font(&mut self, data: Vec<u8>) {
        self.font_system.db_mut().load_font_data(data);
    }

    /// Number of font faces available for shaping. Zero means no glyphs can be
    /// produced (no system fonts and none loaded).
    pub fn font_count(&self) -> usize {
        self.font_system.db().len()
    }

    /// Shape `text` with `style` and rasterize it to a straight-alpha
    /// [`RgbaImage`] sized to the laid-out text (plus [`TextStyle::padding`]).
    ///
    /// The bitmap is transparent except where glyphs cover it. Empty or
    /// whitespace-only text yields a zero-area image.
    pub fn rasterize(&mut self, text: &str, style: &TextStyle) -> RgbaImage {
        let metrics = Metrics::new(style.font_size.max(1.0), style.line_height.max(1.0));
        let mut buffer = Buffer::new(&mut self.font_system, metrics);

        // Width constraint drives word wrapping; height is unbounded so every
        // line is laid out (and thus measurable).
        buffer.set_size(&mut self.font_system, style.max_width, None);

        let family = match &style.family {
            FontFamily::SansSerif => Family::SansSerif,
            FontFamily::Serif => Family::Serif,
            FontFamily::Monospace => Family::Monospace,
            FontFamily::Named(name) => Family::Name(name),
        };
        let attrs = Attrs::new().family(family);
        let align = match style.align {
            TextAlign::Left => Align::Left,
            TextAlign::Center => Align::Center,
            TextAlign::Right => Align::Right,
        };
        buffer.set_text(
            &mut self.font_system,
            text,
            &attrs,
            Shaping::Advanced,
            Some(align),
        );

        // Measure the laid-out text box from its visual lines.
        let mut content_w = 0.0f32;
        let mut content_h = 0.0f32;
        for run in buffer.layout_runs() {
            content_w = content_w.max(run.line_w);
            content_h = content_h.max(run.line_top + run.line_height);
        }

        let pad = style.padding;
        let width = content_w.ceil() as u32 + 2 * pad;
        let height = content_h.ceil() as u32 + 2 * pad;
        if width == 0 || height == 0 {
            return RgbaImage::transparent(0, 0);
        }

        let mut pixels = vec![0u8; (width as usize) * (height as usize) * 4];
        let style_alpha = u32::from(style.color[3]);
        // Pass full alpha to the shaper; the per-pixel alpha it returns is pure
        // coverage, and we fold the style's own opacity in below.
        let base = Color::rgba(style.color[0], style.color[1], style.color[2], 255);
        let offset = pad as i32;

        buffer.draw(
            &mut self.font_system,
            &mut self.swash_cache,
            base,
            |gx, gy, gw, gh, color| {
                let coverage = u32::from(color.a());
                if coverage == 0 {
                    return;
                }
                let alpha = ((coverage * style_alpha + 127) / 255) as u8;
                if alpha == 0 {
                    return;
                }
                let src = [color.r(), color.g(), color.b(), alpha];

                for row in 0..gh as i32 {
                    let py = gy + row + offset;
                    if py < 0 || py >= height as i32 {
                        continue;
                    }
                    for col in 0..gw as i32 {
                        let px = gx + col + offset;
                        if px < 0 || px >= width as i32 {
                            continue;
                        }
                        let idx = ((py as u32 * width + px as u32) * 4) as usize;
                        over_straight(&mut pixels[idx..idx + 4], src);
                    }
                }
            },
        );

        RgbaImage::new(width, height, pixels)
    }
}

impl Default for TextRenderer {
    fn default() -> Self {
        Self::new()
    }
}

/// Straight-alpha "source over destination" compositing of one pixel. Both
/// `dst` and `src` are non-premultiplied RGBA; `dst` is updated in place.
fn over_straight(dst: &mut [u8], src: [u8; 4]) {
    let sa = f32::from(src[3]) / 255.0;
    if sa <= 0.0 {
        return;
    }
    // Fast path: an opaque source (the common interior-of-glyph case) replaces.
    if src[3] == 255 {
        dst.copy_from_slice(&src);
        return;
    }
    let da = f32::from(dst[3]) / 255.0;
    let out_a = sa + da * (1.0 - sa);
    if out_a <= 0.0 {
        dst.fill(0);
        return;
    }
    for i in 0..3 {
        let s = f32::from(src[i]) / 255.0;
        let d = f32::from(dst[i]) / 255.0;
        let c = (s * sa + d * da * (1.0 - sa)) / out_a;
        dst[i] = (c * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    dst[3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Count pixels with non-zero alpha (glyph coverage).
    fn covered(img: &RgbaImage) -> usize {
        img.pixels.chunks_exact(4).filter(|p| p[3] != 0).count()
    }

    #[test]
    fn renderer_reports_font_count() {
        let r = TextRenderer::new();
        // Don't assert > 0 (CI may be fontless); just that it's callable and
        // consistent with whether glyphs can be produced.
        let _ = r.font_count();
    }

    #[test]
    fn empty_text_is_zero_area() {
        let mut r = TextRenderer::new();
        let img = r.rasterize("", &TextStyle::new(32.0));
        assert_eq!(img.width, 0);
        assert_eq!(img.height, 0);
        assert!(img.is_well_formed());
    }

    #[test]
    fn rasterizes_glyph_coverage() {
        let mut r = TextRenderer::new();
        if r.font_count() == 0 {
            eprintln!("skipping: no fonts available");
            return;
        }
        let style = TextStyle::new(48.0).with_color([255, 0, 0, 255]);
        let img = r.rasterize("Hi", &style);

        assert!(img.is_well_formed());
        assert!(
            img.width > 0 && img.height > 0,
            "got {}x{}",
            img.width,
            img.height
        );

        // Some pixels must be covered, and covered pixels carry the fill color
        // with no green/blue (we asked for pure red).
        let lit = covered(&img);
        assert!(lit > 0, "no glyph coverage rasterized");
        for p in img.pixels.chunks_exact(4).filter(|p| p[3] != 0) {
            assert!(
                p[0] >= p[1] && p[0] >= p[2],
                "fill color not red-dominant: {p:?}"
            );
            assert_eq!(p[2], 0, "blue leaked into a red fill: {p:?}");
        }
    }

    #[test]
    fn padding_grows_the_bitmap_symmetrically() {
        let mut r = TextRenderer::new();
        if r.font_count() == 0 {
            return;
        }
        let base = r.rasterize("A", &TextStyle::new(40.0));
        let padded = r.rasterize("A", &TextStyle::new(40.0).with_padding(5));
        assert_eq!(padded.width, base.width + 10);
        assert_eq!(padded.height, base.height + 10);
        // The outermost ring of the padded bitmap is fully transparent.
        assert_eq!(padded.pixel(0, 0), [0, 0, 0, 0]);
    }

    #[test]
    fn style_alpha_scales_coverage() {
        let mut r = TextRenderer::new();
        if r.font_count() == 0 {
            return;
        }
        let opaque = r.rasterize("W", &TextStyle::new(48.0).with_color([255, 255, 255, 255]));
        let faded = r.rasterize("W", &TextStyle::new(48.0).with_color([255, 255, 255, 128]));

        let max_a = |img: &RgbaImage| img.pixels.chunks_exact(4).map(|p| p[3]).max().unwrap_or(0);
        assert!(
            max_a(&opaque) > max_a(&faded),
            "style alpha did not reduce coverage"
        );
    }
}
