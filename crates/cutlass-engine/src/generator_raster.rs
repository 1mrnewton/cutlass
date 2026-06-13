//! CPU rasterization of generated clips (text, shapes) into full-canvas RGBA
//! buffers that ride the existing [`CompositeLayer::Rgba`] upload path.
//!
//! Rasters are cached by `(content, canvas size)` so playback — which calls
//! `resolve_layers` once per frame — pays the raster cost once per distinct
//! generator/canvas pair, not every frame. The output is straight-alpha RGBA
//! over a transparent ground, matching the compositor's src-over blend.

use std::collections::VecDeque;
use std::sync::Arc;

use cosmic_text::{
    Attrs, Buffer, Color as TextColor, Family, FontSystem, Metrics, Shaping, Style as FontStyle,
    SwashCache, Weight, Wrap,
};
use cutlass_models::{Generator, Shape, TextAlignH, TextAlignV, TextStyle};
use tiny_skia::{FillRule, Paint, PathBuilder, Pixmap, Rect, Transform};

/// Reference canvas height the style's pixel sizes are authored against; the
/// rasterizer scales every length by `height / REFERENCE_HEIGHT` so a title
/// looks identical at any output resolution. Matches [`cutlass_models`].
const REFERENCE_HEIGHT: f32 = 1080.0;

/// Enumerate installed font family names (deduped, sorted) for the text
/// inspector's font picker. Scanning the system font directories is slow
/// (hundreds of ms), so callers should run this off the UI thread once.
pub fn system_font_families() -> Vec<String> {
    let mut db = cosmic_text::fontdb::Database::new();
    db.load_system_fonts();
    let mut names: Vec<String> = db
        .faces()
        .filter_map(|face| face.families.first().map(|(name, _)| name.clone()))
        .collect();
    names.sort_unstable();
    names.dedup();
    names
}

/// Most timelines stack only a handful of generated clips; a small cache keeps
/// every visible text/shape raster warm without unbounded growth.
const CACHE_CAP: usize = 24;

/// Cache key for a rasterized generator. Keyed on the visible parameters plus
/// the canvas size (a resize invalidates the bitmap).
#[derive(Clone, PartialEq, Eq, Hash)]
enum RasterKey {
    Text {
        content: String,
        style: TextStyleKey,
        w: u32,
        h: u32,
    },
    Shape { shape: ShapeKey, rgba: [u8; 4], w: u32, h: u32, width_bits: u32, height_bits: u32 },
}

/// Hashable mirror of [`TextStyle`]: `f32` fields are stored as their IEEE bit
/// patterns so the whole style participates in the raster cache key (two
/// styles that differ only in, say, stroke width must raster separately).
#[derive(Clone, PartialEq, Eq, Hash)]
struct TextStyleKey {
    font: String,
    size_bits: u32,
    bold: bool,
    italic: bool,
    underline: bool,
    case: cutlass_models::TextCase,
    fill: [u8; 4],
    letter_spacing_bits: u32,
    line_spacing_bits: u32,
    align_h: TextAlignH,
    align_v: TextAlignV,
    wrap: bool,
    stroke: Option<([u8; 4], u32)>,
    background: Option<([u8; 4], u32)>,
    shadow: Option<([u8; 4], u32, u32)>,
}

impl TextStyleKey {
    fn new(style: &TextStyle) -> Self {
        Self {
            font: style.font.clone(),
            size_bits: style.size.to_bits(),
            bold: style.bold,
            italic: style.italic,
            underline: style.underline,
            case: style.case,
            fill: style.fill,
            letter_spacing_bits: style.letter_spacing.to_bits(),
            line_spacing_bits: style.line_spacing.to_bits(),
            align_h: style.align_h,
            align_v: style.align_v,
            wrap: style.wrap,
            stroke: style.stroke.map(|s| (s.rgba, s.width.to_bits())),
            background: style.background.map(|b| (b.rgba, b.radius.to_bits())),
            shadow: style
                .shadow
                .map(|s| (s.rgba, s.blur.to_bits(), s.distance.to_bits())),
        }
    }
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
            Generator::Text { content, style } => RasterKey::Text {
                content: content.clone(),
                style: TextStyleKey::new(style),
                w: width,
                h: height,
            },
            Generator::Shape {
                shape,
                rgba,
                width: shape_w,
                height: shape_h,
            } => RasterKey::Shape {
                shape: shape_key(*shape),
                rgba: *rgba,
                w: width,
                h: height,
                width_bits: shape_w.to_bits(),
                height_bits: shape_h.to_bits(),
            },
            _ => return None,
        };

        if let Some(hit) = self.lookup(&key) {
            return Some(hit);
        }

        let bytes = match generator {
            Generator::Text { content, style } => self.raster_text(content, style, width, height),
            Generator::Shape {
                shape,
                rgba,
                width: shape_w,
                height: shape_h,
            } => raster_shape(*shape, *rgba, *shape_w, *shape_h, width, height),
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

    fn raster_text(&mut self, content: &str, style: &TextStyle, width: u32, height: u32) -> Vec<u8> {
        let mut out = vec![0u8; (width as usize) * (height as usize) * 4];

        // The string actually shaped: the casing transform is applied up front
        // so cosmic-text measures the displayed glyphs.
        let shaped = style.case.apply(content);
        if shaped.trim().is_empty() {
            return out;
        }

        let w = width as usize;
        let h = height as usize;
        let scale = height as f32 / REFERENCE_HEIGHT;
        let font_size = (style.size * scale).max(4.0);
        let line_height = (font_size * style.line_spacing.max(0.1)).max(font_size);
        let letter_spacing = style.letter_spacing * scale;

        // Coverage of the glyph/underline shapes (canvas-sized, 0..=255). Built
        // once, then reused as the source for the fill, stroke (dilated) and
        // shadow (blurred) passes so they stay perfectly registered.
        let coverage = self.text_coverage(
            &shaped,
            style,
            font_size,
            line_height,
            letter_spacing,
            width,
            height,
        );
        let Some(bbox) = coverage_bbox(&coverage, w) else {
            return out;
        };

        // Background card sits behind everything, hugging the text block.
        if let Some(bg) = style.background {
            let pad = (font_size * 0.3).round() as i32;
            let (x0, y0, x1, y1) = bbox;
            let rx = (x0 as i32 - pad).max(0);
            let ry = (y0 as i32 - pad).max(0);
            let rw = ((x1 as i32 + pad).min(w as i32 - 1) - rx + 1).max(0);
            let rh = ((y1 as i32 + pad).min(h as i32 - 1) - ry + 1).max(0);
            if rw > 0 && rh > 0 {
                let radius = (bg.radius.clamp(0.0, 1.0)) * (rh as f32 / 2.0);
                fill_rounded_rect(&mut out, w, h, rx, ry, rw, rh, radius, bg.rgba);
            }
        }

        // Drop shadow: blurred coverage, tinted, offset down-right at 45°.
        if let Some(shadow) = style.shadow {
            let blur_r = (shadow.blur.clamp(0.0, 1.0) * font_size).round() as usize;
            let blurred = box_blur(&coverage, w, h, blur_r);
            let off = (shadow.distance * scale / std::f32::consts::SQRT_2).round() as i32;
            composite_mask(&mut out, w, h, &blurred, shadow.rgba, off, off);
        }

        // Outline: coverage dilated by the stroke width, drawn under the fill
        // so only the ring outside the glyphs shows.
        if let Some(stroke) = style.stroke {
            let r = (stroke.width * scale).round().max(0.0) as usize;
            if r > 0 {
                let dilated = dilate(&coverage, w, h, r);
                composite_mask(&mut out, w, h, &dilated, stroke.rgba, 0, 0);
            }
        }

        // Fill on top.
        composite_mask(&mut out, w, h, &coverage, style.fill, 0, 0);

        out
    }

    /// Lay out `text` and accumulate its glyph (and underline) alpha coverage
    /// into a canvas-sized 0..=255 mask. Centering, alignment, letter spacing
    /// and underline are all resolved here; effect passes only reshape this
    /// mask.
    fn text_coverage(
        &mut self,
        text: &str,
        style: &TextStyle,
        font_size: f32,
        line_height: f32,
        letter_spacing: f32,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let w = width as usize;
        let h = height as usize;
        let mut mask = vec![0u8; w * h];

        let font_system = self.font_system.get_or_insert_with(FontSystem::new);
        let swash = &mut self.swash_cache;

        let mut attrs = Attrs::new();
        if !style.font.is_empty() {
            attrs = attrs.family(Family::Name(&style.font));
        }
        if style.bold {
            attrs = attrs.weight(Weight::BOLD);
        }
        if style.italic {
            attrs = attrs.style(FontStyle::Italic);
        }

        let metrics = Metrics::new(font_size, line_height);
        let mut buffer = Buffer::new(font_system, metrics);
        // Small inset so descenders/ascenders of big titles don't clip against
        // the edge; it also serves as the left/right alignment margin below.
        let margin = width as f32 * 0.05;
        if style.wrap {
            // Wrap the title inside the canvas width.
            let wrap_w = (width as f32 - 2.0 * margin).max(1.0);
            buffer.set_size(font_system, Some(wrap_w), Some(height as f32));
        } else {
            // No wrap: lay the text on a single line (explicit newlines still
            // break) that may overflow the canvas edges, where it's clipped to
            // the frame — `Wrap::None` plus an unbounded layout width so
            // cosmic-text never inserts soft breaks.
            buffer.set_wrap(font_system, Wrap::None);
            buffer.set_size(font_system, None, Some(height as f32));
        }
        buffer.set_text(font_system, text, &attrs, Shaping::Advanced);
        buffer.shape_until_scroll(font_system, false);

        // Block height (for vertical alignment).
        let text_h = buffer
            .layout_runs()
            .fold(0.0_f32, |m, run| m.max(run.line_top + run.line_height));
        let y_off = match style.align_v {
            TextAlignV::Top => margin.round() as i32,
            TextAlignV::Middle => (((height as f32) - text_h) / 2.0).round() as i32,
            TextAlignV::Bottom => ((height as f32) - text_h - margin).round() as i32,
        };

        let white = TextColor::rgba(255, 255, 255, 255);
        let canvas_w = w as i32;
        let canvas_h = h as i32;
        let underline_thickness = (font_size * 0.06).max(1.0);

        for run in buffer.layout_runs() {
            let glyph_count = run.glyphs.len();
            // Total extra width letter spacing adds to this line (between
            // glyphs only), so alignment accounts for it.
            let extra = if glyph_count > 1 {
                letter_spacing * (glyph_count as f32 - 1.0)
            } else {
                0.0
            };
            let line_w = run.line_w + extra.max(0.0);
            let line_x_off = match style.align_h {
                TextAlignH::Left => margin,
                TextAlignH::Center => ((width as f32) - line_w) / 2.0,
                TextAlignH::Right => (width as f32) - margin - line_w,
            }
            .round() as i32;
            let base_y = run.line_y as i32 + y_off;

            let mut min_gx = i32::MAX;
            let mut max_gx = i32::MIN;
            for (i, glyph) in run.glyphs.iter().enumerate() {
                let spacing = (letter_spacing * i as f32).round() as i32;
                let physical = glyph.physical((0.0, 0.0), 1.0);
                let gx = line_x_off + spacing + physical.x;
                let gy = base_y + physical.y;
                swash.with_pixels(font_system, physical.cache_key, white, |x, y, color| {
                    add_coverage(&mut mask, canvas_w, canvas_h, gx + x, gy + y, color.a());
                });
                let left = line_x_off + spacing + glyph.x.round() as i32;
                let right = left + glyph.w.round() as i32;
                min_gx = min_gx.min(left);
                max_gx = max_gx.max(right);
            }

            if style.underline && max_gx > min_gx {
                // Sit the underline just below the baseline.
                let uy = base_y + (font_size * 0.12).round() as i32;
                let ut = underline_thickness.round().max(1.0) as i32;
                for dy in 0..ut {
                    for x in min_gx..max_gx {
                        add_coverage(&mut mask, canvas_w, canvas_h, x, uy + dy, 255);
                    }
                }
            }
        }

        mask
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

/// Accumulate glyph coverage into a single-channel mask, keeping the maximum
/// where shapes overlap (so antialiased edges of adjacent glyphs don't double
/// up into a visible seam).
fn add_coverage(mask: &mut [u8], w: i32, h: i32, x: i32, y: i32, cov: u8) {
    if x < 0 || y < 0 || x >= w || y >= h || cov == 0 {
        return;
    }
    let idx = (y * w + x) as usize;
    mask[idx] = mask[idx].max(cov);
}

/// Straight-alpha src-over of a flat `rgba` color, weighted per-pixel by a
/// coverage `mask`, onto `out`. `(dx, dy)` shifts the mask sample (used by the
/// shadow pass), with off-canvas samples treated as zero coverage.
fn composite_mask(out: &mut [u8], w: usize, h: usize, mask: &[u8], rgba: [u8; 4], dx: i32, dy: i32) {
    let src_a = rgba[3] as f32 / 255.0;
    if src_a <= 0.0 {
        return;
    }
    for y in 0..h {
        let sy = y as i32 - dy;
        if sy < 0 || sy >= h as i32 {
            continue;
        }
        for x in 0..w {
            let sx = x as i32 - dx;
            if sx < 0 || sx >= w as i32 {
                continue;
            }
            let cov = mask[sy as usize * w + sx as usize];
            if cov == 0 {
                continue;
            }
            let sa = src_a * (cov as f32 / 255.0);
            blend_rgba_over(out, (y * w + x) * 4, rgba, sa);
        }
    }
}

/// Straight-alpha src-over of a premultiplied-by-coverage color (`sa` is the
/// effective source alpha) onto pixel `idx` of an RGBA buffer.
fn blend_rgba_over(buf: &mut [u8], idx: usize, src: [u8; 4], sa: f32) {
    if sa <= 0.0 {
        return;
    }
    let da = buf[idx + 3] as f32 / 255.0;
    let out_a = sa + da * (1.0 - sa);
    if out_a <= 0.0 {
        return;
    }
    for c in 0..3 {
        let s = src[c] as f32 / 255.0;
        let d = buf[idx + c] as f32 / 255.0;
        let v = (s * sa + d * da * (1.0 - sa)) / out_a;
        buf[idx + c] = (v * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    buf[idx + 3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
}

/// Tight `(min_x, min_y, max_x, max_y)` of non-zero coverage, or `None` if the
/// mask is empty.
fn coverage_bbox(mask: &[u8], w: usize) -> Option<(u32, u32, u32, u32)> {
    let (mut min_x, mut min_y) = (usize::MAX, usize::MAX);
    let (mut max_x, mut max_y) = (0usize, 0usize);
    let mut any = false;
    for (i, &c) in mask.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let (x, y) = (i % w, i / w);
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
        any = true;
    }
    any.then_some((min_x as u32, min_y as u32, max_x as u32, max_y as u32))
}

/// Morphological dilation (separable max filter) of a coverage mask by a square
/// of radius `r`. Grows the glyph coverage outward — the basis for the stroke
/// outline. O(w·h·r); cold path only (rasters are cached).
fn dilate(mask: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    if r == 0 {
        return mask.to_vec();
    }
    let r = r as i32;
    let (wi, hi) = (w as i32, h as i32);
    let mut tmp = vec![0u8; mask.len()];
    for y in 0..h {
        let row = y * w;
        for x in 0..wi {
            let mut m = 0u8;
            for dx in -r..=r {
                let xx = x + dx;
                if xx >= 0 && xx < wi {
                    m = m.max(mask[row + xx as usize]);
                }
            }
            tmp[row + x as usize] = m;
        }
    }
    let mut out = vec![0u8; mask.len()];
    for y in 0..hi {
        for x in 0..w {
            let mut m = 0u8;
            for dy in -r..=r {
                let yy = y + dy;
                if yy >= 0 && yy < hi {
                    m = m.max(tmp[yy as usize * w + x]);
                }
            }
            out[y as usize * w + x] = m;
        }
    }
    out
}

/// Approximate Gaussian blur (three passes of a separable box blur with edge
/// clamping) of a coverage mask. Each pass is a sliding-window average, so the
/// whole thing is O(w·h) regardless of radius.
fn box_blur(mask: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    if r == 0 {
        return mask.to_vec();
    }
    let mut buf = mask.to_vec();
    for _ in 0..3 {
        buf = box_blur_h(&buf, w, h, r);
        buf = transpose(&buf, w, h);
        buf = box_blur_h(&buf, h, w, r);
        buf = transpose(&buf, h, w);
    }
    buf
}

/// One horizontal sliding-window average pass with edge replication.
fn box_blur_h(src: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    let mut out = vec![0u8; src.len()];
    let win = (2 * r + 1) as u32;
    let last = w as i64 - 1;
    let clamp = |i: i64| -> usize { i.clamp(0, last) as usize };
    for y in 0..h {
        let row = y * w;
        let mut sum: u32 = 0;
        for k in -(r as i64)..=(r as i64) {
            sum += src[row + clamp(k)] as u32;
        }
        for x in 0..w {
            out[row + x] = (sum / win) as u8;
            let add = src[row + clamp(x as i64 + r as i64 + 1)] as u32;
            let sub = src[row + clamp(x as i64 - r as i64)] as u32;
            sum = sum + add - sub;
        }
    }
    out
}

/// Transpose a `w`×`h` single-channel buffer to `h`×`w`, so the vertical blur
/// pass can reuse the horizontal kernel with cache-friendly row access.
fn transpose(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; src.len()];
    for y in 0..h {
        for x in 0..w {
            out[x * h + y] = src[y * w + x];
        }
    }
    out
}

/// Fill an axis-aligned rounded rectangle (straight-alpha src-over) using
/// tiny-skia, then composite onto `out`. Used for the text background card.
fn fill_rounded_rect(
    out: &mut [u8],
    w: usize,
    h: usize,
    x: i32,
    y: i32,
    rw: i32,
    rh: i32,
    radius: f32,
    rgba: [u8; 4],
) {
    let Some(mut pixmap) = Pixmap::new(w as u32, h as u32) else {
        return;
    };
    let mut paint = Paint::default();
    paint.set_color_rgba8(rgba[0], rgba[1], rgba[2], rgba[3]);
    paint.anti_alias = true;

    let Some(rect) = Rect::from_xywh(x as f32, y as f32, rw as f32, rh as f32) else {
        return;
    };
    let radius = radius.clamp(0.0, (rw.min(rh) as f32) / 2.0);
    let mut pb = PathBuilder::new();
    if radius <= 0.5 {
        pb.push_rect(rect);
    } else {
        push_round_rect(&mut pb, rect, radius);
    }
    let Some(path) = pb.finish() else {
        return;
    };
    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);

    for (i, px) in pixmap.pixels().iter().enumerate() {
        let c = px.demultiply();
        if c.alpha() == 0 {
            continue;
        }
        blend_rgba_over(
            out,
            i * 4,
            [c.red(), c.green(), c.blue(), c.alpha()],
            c.alpha() as f32 / 255.0,
        );
    }
}

/// Append a rounded-rectangle subpath (four quadratic corners) to `pb`.
fn push_round_rect(pb: &mut PathBuilder, rect: Rect, r: f32) {
    let (l, t, right, b) = (rect.left(), rect.top(), rect.right(), rect.bottom());
    pb.move_to(l + r, t);
    pb.line_to(right - r, t);
    pb.quad_to(right, t, right, t + r);
    pb.line_to(right, b - r);
    pb.quad_to(right, b, right - r, b);
    pb.line_to(l + r, b);
    pb.quad_to(l, b, l, b - r);
    pb.line_to(l, t + r);
    pb.quad_to(l, t, l + r, t);
    pb.close();
}

/// Rasterize a centered shape at the given reference-pixel size.
fn raster_shape(
    shape: Shape,
    rgba: [u8; 4],
    shape_w: f32,
    shape_h: f32,
    canvas_w: u32,
    canvas_h: u32,
) -> Vec<u8> {
    let mut out = vec![0u8; (canvas_w as usize) * (canvas_h as usize) * 4];
    let Some(mut pixmap) = Pixmap::new(canvas_w, canvas_h) else {
        return out;
    };

    let mut paint = Paint::default();
    paint.set_color_rgba8(rgba[0], rgba[1], rgba[2], rgba[3]);
    paint.anti_alias = true;

    let scale = canvas_h as f32 / REFERENCE_HEIGHT;
    let ext_w = shape_w * scale;
    let ext_h = shape_h * scale;
    let x = (canvas_w as f32 - ext_w) / 2.0;
    let y = (canvas_h as f32 - ext_h) / 2.0;
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
        let buf = raster_shape(Shape::Rectangle, [255, 0, 0, 255], 960.0, 540.0, w, h);
        assert_eq!(buf.len(), (w * h * 4) as usize);
        // Center is inside the legacy-default box.
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
        let buf = raster_shape(Shape::Ellipse, [0, 255, 0, 255], 960.0, 540.0, w, h);
        // Center filled.
        assert_eq!(alpha_at(&buf, w, w / 2, h / 2), 255);
        // Legacy default at 64×64: box ≈ 57×32 centered; corner (16,16) sits
        // outside the inscribed ellipse.
        assert_eq!(alpha_at(&buf, w, 16, 16), 0);
    }

    #[test]
    fn raster_caches_repeated_lookups() {
        let mut raster = GeneratorRaster::new();
        let generator = Generator::shape(Shape::Rectangle, [10, 20, 30, 255]);
        let a = raster.raster(&generator, 32, 32).unwrap();
        let b = raster.raster(&generator, 32, 32).unwrap();
        // Same Arc allocation on the second call ⇒ cache hit.
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn content_size_hugs_the_drawn_content() {
        let mut raster = GeneratorRaster::new();
        // Legacy-default shapes scale with canvas height; at 64×64 the box is
        // 960×540 ref px → ≈57×32 canvas px (rectangle exact, ellipse inscribes).
        let rect = Generator::Shape {
            shape: Shape::Rectangle,
            rgba: [255, 0, 0, 255],
            width: 960.0,
            height: 540.0,
        };
        assert_eq!(raster.content_size(&rect, 64, 64), Some((58, 32)));
        let ellipse = Generator::Shape {
            shape: Shape::Ellipse,
            rgba: [0, 255, 0, 255],
            width: 960.0,
            height: 540.0,
        };
        let (ew, eh) = raster.content_size(&ellipse, 64, 64).unwrap();
        assert!((55..=58).contains(&ew) && (30..=32).contains(&eh), "{ew}x{eh}");
        // Drop-size shapes at 1080p canvas land at 200×200 px.
        let small = Generator::shape(Shape::Rectangle, [255, 255, 255, 255]);
        assert_eq!(raster.content_size(&small, 1920, 1080), Some((200, 200)));
        // Solids cover the whole canvas (no raster involved).
        let solid = Generator::SolidColor { rgba: [1, 2, 3, 255] };
        assert_eq!(raster.content_size(&solid, 64, 48), Some((64, 48)));
        // Text measures its laid-out block — smaller than the canvas.
        let text = Generator::text("Hi");
        let (tw, th) = raster.content_size(&text, 256, 128).unwrap();
        assert!(tw > 0 && tw < 256, "text width {tw}");
        assert!(th > 0 && th < 128, "text height {th}");
        // Empty text draws nothing.
        let empty = Generator::text(" ");
        assert_eq!(raster.content_size(&empty, 256, 128), Some((0, 0)));
        // Unsupported generators have no raster, hence no content.
        assert_eq!(raster.content_size(&Generator::Sticker, 64, 64), None);
        assert_eq!(raster.content_size(&rect, 0, 64), None);
    }

    #[test]
    fn unsupported_generators_return_none() {
        let mut raster = GeneratorRaster::new();
        assert!(raster.raster(&Generator::Sticker, 32, 32).is_none());
        assert!(raster.raster(&Generator::text("x"), 0, 0).is_none());
    }

    #[test]
    fn text_draws_pixels() {
        let mut raster = GeneratorRaster::new();
        let buf = raster.raster(&Generator::text("Hi"), 256, 128).unwrap();
        let any_opaque = buf.chunks_exact(4).any(|p| p[3] > 0);
        assert!(any_opaque, "text raster produced no visible pixels");
    }

    fn styled(content: &str, style: TextStyle) -> Generator {
        Generator::Text {
            content: content.into(),
            style,
        }
    }

    #[test]
    fn text_fill_color_is_honored() {
        let mut raster = GeneratorRaster::new();
        let style = TextStyle {
            fill: [255, 0, 0, 255],
            ..TextStyle::default()
        };
        let buf = raster.raster(&styled("Hi", style), 256, 128).unwrap();
        // A solid glyph interior is the requested red.
        let px = buf
            .chunks_exact(4)
            .find(|p| p[3] > 220)
            .expect("an opaque glyph pixel");
        assert!(px[0] > 180 && px[1] < 80 && px[2] < 80, "fill not red: {px:?}");
    }

    #[test]
    fn distinct_styles_get_distinct_rasters() {
        let mut raster = GeneratorRaster::new();
        let plain = Generator::text("Hi");
        let bold = styled(
            "Hi",
            TextStyle {
                bold: true,
                ..TextStyle::default()
            },
        );
        let a = raster.raster(&plain, 160, 80).unwrap();
        let b = raster.raster(&bold, 160, 80).unwrap();
        // Different style ⇒ separate cache entry (and almost certainly
        // different pixels).
        assert!(!Arc::ptr_eq(&a, &b));
        // Same style ⇒ cache hit.
        let a2 = raster.raster(&plain, 160, 80).unwrap();
        assert!(Arc::ptr_eq(&a, &a2));
    }

    #[test]
    fn case_transform_changes_raster() {
        let mut raster = GeneratorRaster::new();
        let lower = raster.raster(&Generator::text("hi"), 256, 128).unwrap();
        let upper = raster
            .raster(
                &styled(
                    "hi",
                    TextStyle {
                        case: cutlass_models::TextCase::Upper,
                        ..TextStyle::default()
                    },
                ),
                256,
                128,
            )
            .unwrap();
        assert_ne!(lower, upper, "uppercasing should change the raster");
    }

    #[test]
    fn stroke_grows_content_box() {
        let mut raster = GeneratorRaster::new();
        let base = raster.content_size(&Generator::text("Hi"), 512, 256).unwrap();
        let stroked = styled(
            "Hi",
            TextStyle {
                stroke: Some(cutlass_models::TextStroke {
                    rgba: [0, 0, 0, 255],
                    width: 16.0,
                }),
                ..TextStyle::default()
            },
        );
        let grown = raster.content_size(&stroked, 512, 256).unwrap();
        assert!(
            grown.0 > base.0 && grown.1 > base.1,
            "stroke should enlarge the content box: {base:?} -> {grown:?}"
        );
    }

    #[test]
    fn wrap_off_lays_out_fewer_lines_than_wrap_on() {
        let mut raster = GeneratorRaster::new();
        // A long run of words that must wrap to several lines inside the canvas.
        let long = "the quick brown fox jumps over the lazy dog again and again";
        let wrapped = raster
            .content_size(&Generator::text(long), 256, 256)
            .unwrap();
        let no_wrap = styled(
            long,
            TextStyle {
                wrap: false,
                ..TextStyle::default()
            },
        );
        let single = raster.content_size(&no_wrap, 256, 256).unwrap();
        // One overflowing line is far shorter than the multi-line wrapped block.
        assert!(
            single.1 < wrapped.1,
            "wrap-off should be shorter (single line): {single:?} vs wrapped {wrapped:?}"
        );
    }

    #[test]
    fn background_card_adds_its_color() {
        let mut raster = GeneratorRaster::new();
        let style = TextStyle {
            background: Some(cutlass_models::TextBackground {
                rgba: [0, 0, 255, 255],
                radius: 0.0,
            }),
            ..TextStyle::default()
        };
        let buf = raster.raster(&styled("Hi", style), 256, 128).unwrap();
        let has_blue = buf
            .chunks_exact(4)
            .any(|p| p[3] > 220 && p[2] > 180 && p[0] < 80 && p[1] < 80);
        assert!(has_blue, "background card should paint blue pixels behind the text");
    }
}
