//! cutlass-text: shape and rasterize styled text into straight-alpha
//! [`RgbaImage`] bitmaps the compositor can place as generator layers.
//!
//! Text is a **generator** — pixels are produced, not decoded — so this crate
//! stays pure-Rust and GPU-free: it depends only on [`cosmic_text`] (shaping +
//! layout via `rustybuzz`/`swash`) and [`cutlass_core`]. The engine wires the
//! resulting bitmaps to `cutlass_compositor` RGBA layers; nothing here knows
//! about `wgpu`.
//!
//! Two entry points share one shaping/measurement path:
//!
//! - [`TextRenderer::rasterize`]: the whole run as a single bitmap, for static
//!   text.
//! - [`TextRenderer::shape`]: the run as positioned per-cluster bitmaps
//!   ([`ShapedText`]), for character-level animation. The full string is
//!   shaped **once** — kerning, ligatures, BiDi, and complex scripts stay
//!   correct — and the animation system then transforms each cluster's
//!   placement without ever re-shaping.
//!
//! ```no_run
//! use cutlass_text::{TextRenderer, TextStyle};
//!
//! let mut text = TextRenderer::new();
//! let style = TextStyle::new(48.0).with_color([255, 220, 0, 255]);
//! let bitmap = text.rasterize("Cutlass", &style); // straight-alpha RgbaImage
//! let shaped = text.shape("Cutlass", &style); // one ClusterBox per character
//! ```
//!
//! Fonts come from the host by default ([`TextRenderer::new`]); call
//! [`TextRenderer::load_font`] to add bundled/custom faces for deterministic,
//! platform-independent output.

mod effects;
mod style;

use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Range;

use cosmic_text::{
    Align, Attrs, Buffer, CacheKey, Family, FontSystem, LineIter, Metrics, Shaping, SwashCache,
    SwashContent, SwashImage,
};
use cutlass_core::RgbaImage;

pub use style::{FontFamily, TextAlign, TextBackground, TextShadow, TextStroke, TextStyle};

/// Enumerate installed font family names (deduped, sorted) for a font picker.
/// Scanning the system font directories is slow (hundreds of ms), so callers
/// should run this off the UI thread once and reuse the result.
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

/// A text run shaped and rasterized as individually placeable pieces — the
/// substrate for character-level animation (typewriter, per-char fade, wave).
///
/// Produced by [`TextRenderer::shape`]. Compositing every cluster's `image` at
/// its `offset` reproduces [`TextRenderer::rasterize`] exactly (minus padding);
/// an animator instead perturbs each cluster's placement/opacity per frame,
/// which never re-shapes the text.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapedText {
    /// Ink-tight pixel size of the whole run (no padding). `(0, 0)` when
    /// nothing produced coverage (empty/whitespace-only text, or no fonts).
    pub extent: (u32, u32),
    /// Animation units in logical (text) order: sorted by line, then by byte
    /// position — the order a typewriter effect reveals them, also for RTL.
    pub clusters: Vec<ClusterBox>,
}

impl ShapedText {
    /// True when no cluster produced any coverage.
    pub fn has_ink(&self) -> bool {
        self.extent.0 > 0 && self.extent.1 > 0
    }
}

/// One shaping cluster of a [`ShapedText`]: the animation unit.
///
/// A cluster is what the shaper considers indivisible — usually one user-
/// perceived character ("é", emoji + ZWJ sequences), and one unit per ligature
/// ("fi" moves as a whole, like CapCut/AE text animators). Whitespace clusters
/// are kept with a zero-area `image` so stagger timing still counts them as a
/// beat.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterBox {
    /// Byte range of this cluster in the source string.
    pub text_range: Range<usize>,
    /// Zero-based visual line index (wrapped lines count separately) — for
    /// by-line staggers.
    pub line: usize,
    /// Top-left of `image` within the run's [`extent`](ShapedText::extent)
    /// box, in pixels.
    pub offset: [f32; 2],
    /// Baseline y of this cluster's line, relative to the extent box top —
    /// the anchor for rise/drop animations.
    pub baseline: f32,
    /// This cluster's glyphs, ink-tight, straight alpha, style color/opacity
    /// folded in. Zero-area for clusters with no coverage (spaces).
    pub image: RgbaImage,
}

/// Shapes and rasterizes text. Holds the font set, a glyph cache, and memo
/// caches of whole-run results, so reuse one renderer across many calls:
/// repeating a `shape`/`rasterize` call with unchanged input costs a memo
/// lookup plus a copy-out of the cached bitmap(s) — no re-shaping. That keeps
/// per-frame callers (preview scrub, export loops) off the shaping path.
pub struct TextRenderer {
    font_system: FontSystem,
    swash_cache: SwashCache,
    /// Memoized [`shape`](Self::shape) results, keyed by (text, style).
    /// Cleared by [`load_font`](Self::load_font) — a new face can change
    /// shaping/fallback for any string.
    shape_memo: HashMap<ShapeKey, ShapedText>,
    /// Memoized [`rasterize`](Self::rasterize) results (padding folded in).
    raster_memo: HashMap<RasterKey, RgbaImage>,
}

impl TextRenderer {
    /// A renderer backed by the host's installed fonts.
    pub fn new() -> Self {
        Self {
            font_system: FontSystem::new(),
            swash_cache: SwashCache::new(),
            shape_memo: HashMap::new(),
            raster_memo: HashMap::new(),
        }
    }

    /// Add a font from raw TTF/OTF bytes (e.g. a bundled face) to the set.
    ///
    /// Returns the number of font faces actually added — `0` means the bytes
    /// were not a parseable font (`fontdb` drops bad data silently, so this
    /// count is the only load-time failure signal; a face collection can add
    /// more than one). Invalidates the memo caches, since a new face can
    /// change shaping or fallback for any string.
    pub fn load_font(&mut self, data: Vec<u8>) -> usize {
        let before = self.font_system.db().len();
        self.font_system.db_mut().load_font_data(data);
        self.shape_memo.clear();
        self.raster_memo.clear();
        self.font_system.db().len().saturating_sub(before)
    }

    /// Number of font faces available for shaping. Zero means no glyphs can be
    /// produced (no system fonts and none loaded).
    pub fn font_count(&self) -> usize {
        self.font_system.db().len()
    }

    /// Shape `text` with `style` into positioned per-cluster bitmaps.
    ///
    /// The full string is shaped once (correct kerning/ligatures/BiDi), then
    /// each cluster's glyphs are rasterized into their own ink-tight image.
    /// [`TextStyle::padding`] is **not** applied here — it is bitmap headroom
    /// for [`rasterize`](Self::rasterize); animated cluster quads overlap
    /// freely and need none.
    ///
    /// Results are memoized: repeating a call with identical input skips
    /// shaping entirely and costs one copy-out of the cached clusters, so
    /// per-frame animators can call this unconditionally.
    pub fn shape(&mut self, text: &str, style: &TextStyle) -> ShapedText {
        let key = ShapeKey::new(text, style);
        if let Some(hit) = self.shape_memo.get(&key) {
            return hit.clone();
        }
        let shaped = self.shape_uncached(text, style);
        Self::memo_insert(&mut self.shape_memo, key, shaped.clone());
        shaped
    }

    fn shape_uncached(&mut self, text: &str, style: &TextStyle) -> ShapedText {
        let buffer = self.layout(text, style);

        // Source byte offset of each buffer line. `LineIter` is the exact
        // split `Buffer::set_text` uses, so indices line up with `line_i`.
        let line_starts: Vec<usize> = LineIter::new(text).map(|(range, _)| range.start).collect();

        // Pass A: walk the laid-out glyphs, group them into shaping clusters,
        // and record every glyph's draw box (the swash image is cached, so
        // pass B's second lookup is a map hit).
        let mut clusters: Vec<PendingCluster> = Vec::new();
        for (visual_line, run) in buffer.layout_runs().enumerate() {
            let line_start = line_starts.get(run.line_i).copied().unwrap_or(0);
            for glyph in run.glyphs {
                // Mirrors cosmic-text's `Buffer::render`: the physical
                // position bakes in the baseline; the image placement then
                // offsets left/up from it.
                let phys = glyph.physical((0., run.line_y), 1.0);
                let ink = self
                    .swash_cache
                    .get_image(&mut self.font_system, phys.cache_key)
                    .as_ref()
                    .filter(|img| img.placement.width > 0 && img.placement.height > 0)
                    .map(|img| InkBox {
                        left: phys.x + img.placement.left,
                        top: phys.y - img.placement.top,
                        width: img.placement.width,
                        height: img.placement.height,
                    });

                let range = line_start + glyph.start..line_start + glyph.end;
                // Glyphs of one cluster (base + combining marks, or a single
                // ligature glyph) are adjacent within a run, so comparing to
                // the last cluster suffices.
                let same_cluster = clusters
                    .last()
                    .is_some_and(|c| c.line == visual_line && c.range == range);
                if !same_cluster {
                    clusters.push(PendingCluster {
                        line: visual_line,
                        range,
                        pen_x: glyph.x,
                        line_top: run.line_top,
                        baseline_y: run.line_y,
                        glyphs: Vec::new(),
                        ink: None,
                    });
                }
                let cluster = clusters.last_mut().expect("just ensured non-empty");
                if let Some(b) = ink {
                    cluster.ink = Some(match cluster.ink {
                        None => b.bounds(),
                        Some(u) => union(u, b.bounds()),
                    });
                }
                cluster.glyphs.push(PendingGlyph {
                    key: phys.cache_key,
                    ink,
                });
            }
        }

        // Run-wide ink box; every offset/extent is relative to its origin, so
        // alignment shifts baked into glyph positions normalize out and the
        // bitmap stays tight no matter how lines are aligned.
        let global = clusters.iter().filter_map(|c| c.ink).reduce(union);
        let (gx, gy, extent) = match global {
            Some((l, t, r, b)) => (l, t, ((r - l) as u32, (b - t) as u32)),
            None => (0, 0, (0, 0)),
        };

        // Pass B: rasterize each cluster's glyphs into its own tight image.
        let style_alpha = u32::from(style.color[3]);
        let mut out = Vec::with_capacity(clusters.len());
        for c in clusters {
            let (image, offset) = match c.ink {
                Some((l, t, r, b)) => {
                    let (w, h) = ((r - l) as u32, (b - t) as u32);
                    let mut pixels = vec![0u8; (w as usize) * (h as usize) * 4];
                    for g in &c.glyphs {
                        let Some(ink) = g.ink else { continue };
                        if let Some(img) = self
                            .swash_cache
                            .get_image(&mut self.font_system, g.key)
                            .as_ref()
                        {
                            write_glyph(
                                &mut pixels,
                                w,
                                ink.left - l,
                                ink.top - t,
                                img,
                                style.color,
                                style_alpha,
                            );
                        }
                    }
                    (
                        RgbaImage::new(w, h, pixels),
                        [(l - gx) as f32, (t - gy) as f32],
                    )
                }
                // No coverage (whitespace): keep the slot, at the pen position.
                None => (
                    RgbaImage::transparent(0, 0),
                    [c.pen_x - gx as f32, c.line_top - gy as f32],
                ),
            };
            out.push(ClusterBox {
                text_range: c.range,
                line: c.line,
                offset,
                baseline: c.baseline_y - gy as f32,
                image,
            });
        }

        // Visual order (per run) -> logical order: line, then byte position.
        // This is the reveal order a typewriter should follow, also for RTL.
        out.sort_by_key(|c| (c.line, c.text_range.start));

        ShapedText {
            extent,
            clusters: out,
        }
    }

    /// Shape `text` with `style` and rasterize it to a single straight-alpha
    /// [`RgbaImage`], sized to the **ink extents** of the laid-out text plus
    /// [`TextStyle::padding`] on every side.
    ///
    /// Text with no coverage — empty or whitespace-only — yields a zero-area
    /// image regardless of padding.
    ///
    /// Results are memoized: repeating a call with identical input costs one
    /// bitmap copy, not a re-shape — safe to call every frame.
    pub fn rasterize(&mut self, text: &str, style: &TextStyle) -> RgbaImage {
        let key = RasterKey::new(text, style);
        if let Some(hit) = self.raster_memo.get(&key) {
            return hit.clone();
        }
        let image = self.rasterize_uncached(text, style);
        Self::memo_insert(&mut self.raster_memo, key, image.clone());
        image
    }

    fn rasterize_uncached(&mut self, text: &str, style: &TextStyle) -> RgbaImage {
        let shaped = self.shape(text, style);
        if !shaped.has_ink() {
            return RgbaImage::transparent(0, 0);
        }
        // Stroke / background / shadow (and plain padding) are painted here —
        // `shape` stays ink-tight so cluster animation can place glyphs freely.
        effects::paint(&shaped, style)
    }

    /// Insert into a memo map, bounding memory with a wholesale clear at the
    /// cap. Frame loops keep a handful of live keys, so an LRU would buy
    /// nothing; the clear only ever costs one re-shape per entry after it.
    fn memo_insert<K: Eq + Hash, V>(memo: &mut HashMap<K, V>, key: K, value: V) {
        const MEMO_CAP: usize = 64;
        if memo.len() >= MEMO_CAP {
            memo.clear();
        }
        memo.insert(key, value);
    }

    /// Memo entry counts `(shape, rasterize)` — test observability.
    #[cfg(test)]
    fn memo_sizes(&self) -> (usize, usize) {
        (self.shape_memo.len(), self.raster_memo.len())
    }

    /// Build and lay out a buffer for `text` + `style` (the single shaping
    /// path both `shape` and `rasterize` go through).
    fn layout(&mut self, text: &str, style: &TextStyle) -> Buffer {
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

        // Alignment needs a container width: without one, cosmic-text aligns
        // each paragraph against its own width (offset 0 — a no-op). Measure,
        // then relayout at the widest line so Center/Right align lines against
        // each other. The +1 slack keeps the widest line from re-wrapping;
        // ink-extent measurement trims the slack back off the bitmap.
        if style.max_width.is_none() && style.align != TextAlign::Left {
            let widest = buffer
                .layout_runs()
                .map(|run| run.line_w)
                .fold(0.0f32, f32::max);
            if widest > 0.0 && buffer.layout_runs().count() > 1 {
                buffer.set_size(&mut self.font_system, Some(widest.ceil() + 1.0), None);
            }
        }

        buffer
    }
}

impl Default for TextRenderer {
    fn default() -> Self {
        Self::new()
    }
}

/// Hashable identity of the fields that affect shaping and glyph pixels.
/// Bitmap-only treatments deliberately stay out: dragging a stroke/shadow/
/// background control must reuse the already-shaped glyph run.
///
/// `f32`s are keyed by bit pattern, so styles that differ only in float
/// encoding are simply distinct keys (never a wrong hit).
#[derive(Clone, PartialEq, Eq, Hash)]
struct ShapeKey {
    text: String,
    font_size: u32,
    line_height: u32,
    color: [u8; 4],
    family: FontFamily,
    align: TextAlign,
    max_width: Option<u32>,
}

impl ShapeKey {
    fn new(text: &str, style: &TextStyle) -> Self {
        Self {
            text: text.to_owned(),
            font_size: style.font_size.to_bits(),
            line_height: style.line_height.to_bits(),
            color: style.color,
            family: style.family.clone(),
            align: style.align,
            max_width: style.max_width.map(f32::to_bits),
        }
    }
}

/// Full bitmap identity: the shaped run plus bitmap padding/treatments.
#[derive(PartialEq, Eq, Hash)]
struct RasterKey {
    shape: ShapeKey,
    padding: u32,
    stroke: Option<([u8; 4], u32)>,
    background: Option<([u8; 4], u32)>,
    shadow: Option<([u8; 4], u32, u32)>,
}

impl RasterKey {
    fn new(text: &str, style: &TextStyle) -> Self {
        Self {
            shape: ShapeKey::new(text, style),
            padding: style.padding,
            stroke: style.stroke.map(|s| (s.rgba, s.width.to_bits())),
            background: style.background.map(|b| (b.rgba, b.radius.to_bits())),
            shadow: style
                .shadow
                .map(|s| (s.rgba, s.blur.to_bits(), s.distance.to_bits())),
        }
    }
}

/// A glyph's draw rectangle in buffer space.
#[derive(Clone, Copy)]
struct InkBox {
    left: i32,
    top: i32,
    width: u32,
    height: u32,
}

impl InkBox {
    /// `(min_x, min_y, max_x, max_y)`, exclusive maxima.
    fn bounds(self) -> (i32, i32, i32, i32) {
        (
            self.left,
            self.top,
            self.left + self.width as i32,
            self.top + self.height as i32,
        )
    }
}

fn union(a: (i32, i32, i32, i32), b: (i32, i32, i32, i32)) -> (i32, i32, i32, i32) {
    (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3))
}

struct PendingGlyph {
    key: CacheKey,
    ink: Option<InkBox>,
}

struct PendingCluster {
    line: usize,
    range: Range<usize>,
    pen_x: f32,
    line_top: f32,
    baseline_y: f32,
    glyphs: Vec<PendingGlyph>,
    ink: Option<(i32, i32, i32, i32)>,
}

/// Blend one glyph image into a cluster bitmap at `(dx, dy)` (in-bounds by
/// construction: the cluster box is the union of its glyphs' boxes).
///
/// Color semantics mirror cosmic-text's `SwashCache::with_pixels`: `Mask`
/// content is coverage tinted with the style color; `Color` content (emoji)
/// carries its own RGBA. The style's alpha scales both.
fn write_glyph(
    pixels: &mut [u8],
    stride_w: u32,
    dx: i32,
    dy: i32,
    img: &SwashImage,
    color: [u8; 4],
    style_alpha: u32,
) {
    let (gw, gh) = (img.placement.width as usize, img.placement.height as usize);
    let fold = |coverage: u8| ((u32::from(coverage) * style_alpha + 127) / 255) as u8;
    let mut blend = |col: usize, row: usize, src: [u8; 4]| {
        if src[3] == 0 {
            return;
        }
        let (px, py) = (dx as usize + col, dy as usize + row);
        let idx = (py * stride_w as usize + px) * 4;
        over_straight(&mut pixels[idx..idx + 4], src);
    };

    match img.content {
        SwashContent::Mask => {
            for row in 0..gh {
                for col in 0..gw {
                    let coverage = img.data[row * gw + col];
                    if coverage == 0 {
                        continue;
                    }
                    blend(col, row, [color[0], color[1], color[2], fold(coverage)]);
                }
            }
        }
        SwashContent::Color => {
            for row in 0..gh {
                for col in 0..gw {
                    let i = (row * gw + col) * 4;
                    blend(
                        col,
                        row,
                        [
                            img.data[i],
                            img.data[i + 1],
                            img.data[i + 2],
                            fold(img.data[i + 3]),
                        ],
                    );
                }
            }
        }
        // swash never emits subpixel masks for the alpha/color sources
        // cosmic-text requests; if that changes, read the RGB mean as coverage.
        SwashContent::SubpixelMask => {
            for row in 0..gh {
                for col in 0..gw {
                    let i = (row * gw + col) * 4;
                    let mean = (u16::from(img.data[i])
                        + u16::from(img.data[i + 1])
                        + u16::from(img.data[i + 2]))
                        / 3;
                    blend(col, row, [color[0], color[1], color[2], fold(mean as u8)]);
                }
            }
        }
    }
}

/// Straight-alpha "source over destination" compositing of one pixel. Both
/// `dst` and `src` are non-premultiplied RGBA; `dst` is updated in place.
/// Callers skip fully transparent sources (`src[3] > 0` is a precondition).
pub(crate) fn over_straight(dst: &mut [u8], src: [u8; 4]) {
    // Fast path: an opaque source (the common interior-of-glyph case) replaces.
    if src[3] == 255 {
        dst.copy_from_slice(&src);
        return;
    }
    let sa = f32::from(src[3]) / 255.0;
    let da = f32::from(dst[3]) / 255.0;
    let out_a = sa + da * (1.0 - sa);
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

    /// Bundled OFL test face (`assets/`), so every test shapes real glyphs
    /// deterministically — even on a fontless CI box.
    const TEST_FONT: &[u8] = include_bytes!("../assets/Micro5-Regular.ttf");

    /// A renderer guaranteed to have at least the bundled face.
    fn test_renderer() -> TextRenderer {
        let mut r = TextRenderer::new();
        let added = r.load_font(TEST_FONT.to_vec());
        assert!(added > 0, "bundled test font failed to parse");
        assert!(r.font_count() > 0);
        r
    }

    /// Count pixels with non-zero alpha (glyph coverage).
    fn covered(img: &RgbaImage) -> usize {
        img.pixels.chunks_exact(4).filter(|p| p[3] != 0).count()
    }

    #[test]
    fn load_font_reports_added_faces() {
        let mut r = TextRenderer::new();
        assert_eq!(
            r.load_font(vec![0xDE, 0xAD, 0xBE, 0xEF]),
            0,
            "garbage bytes are not a font"
        );
        let before = r.font_count();
        assert!(r.load_font(TEST_FONT.to_vec()) > 0);
        assert!(r.font_count() > before);
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
    fn whitespace_is_zero_area_even_with_padding() {
        let mut r = TextRenderer::new();
        // Spaces have advances but no ink; padding must not manufacture a
        // blank bitmap out of them (holds with or without fonts installed).
        let img = r.rasterize("   ", &TextStyle::new(32.0).with_padding(8));
        assert_eq!((img.width, img.height), (0, 0));
        let empty_padded = r.rasterize("", &TextStyle::new(32.0).with_padding(8));
        assert_eq!((empty_padded.width, empty_padded.height), (0, 0));
    }

    #[test]
    fn rasterizes_glyph_coverage() {
        let mut r = test_renderer();
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
        let mut r = test_renderer();
        let base = r.rasterize("A", &TextStyle::new(40.0));
        let padded = r.rasterize("A", &TextStyle::new(40.0).with_padding(5));
        assert_eq!(padded.width, base.width + 10);
        assert_eq!(padded.height, base.height + 10);
        // The outermost ring of the padded bitmap is fully transparent.
        assert_eq!(padded.pixel(0, 0), [0, 0, 0, 0]);
    }

    #[test]
    fn style_alpha_scales_coverage() {
        let mut r = test_renderer();
        let opaque = r.rasterize("W", &TextStyle::new(48.0).with_color([255, 255, 255, 255]));
        let faded = r.rasterize("W", &TextStyle::new(48.0).with_color([255, 255, 255, 128]));

        let max_a = |img: &RgbaImage| img.pixels.chunks_exact(4).map(|p| p[3]).max().unwrap_or(0);
        assert!(
            max_a(&opaque) > max_a(&faded),
            "style alpha did not reduce coverage"
        );
    }

    #[test]
    fn shape_exposes_clusters_in_text_order() {
        let mut r = test_renderer();
        let shaped = r.shape("Hi", &TextStyle::new(48.0));
        assert!(shaped.has_ink());
        assert_eq!(shaped.clusters.len(), 2);
        assert_eq!(shaped.clusters[0].text_range, 0..1);
        assert_eq!(shaped.clusters[1].text_range, 1..2);
        assert!(shaped.clusters.iter().all(|c| c.image.width > 0));
        // LTR: the second character sits to the right of the first.
        assert!(shaped.clusters[1].offset[0] > shaped.clusters[0].offset[0]);
        assert_eq!(shaped.clusters[0].line, 0);
    }

    #[test]
    fn shape_maps_lines_and_baselines() {
        let mut r = test_renderer();
        let shaped = r.shape("A\nB", &TextStyle::new(48.0));
        assert_eq!(shaped.clusters.len(), 2);
        // Byte ranges are in the source string, skipping the newline.
        assert_eq!(shaped.clusters[0].text_range, 0..1);
        assert_eq!(shaped.clusters[1].text_range, 2..3);
        assert_eq!((shaped.clusters[0].line, shaped.clusters[1].line), (0, 1));
        assert!(
            shaped.clusters[1].baseline > shaped.clusters[0].baseline,
            "second line's baseline must be lower"
        );
    }

    #[test]
    fn shape_extent_matches_rasterize() {
        let mut r = test_renderer();
        let style = TextStyle::new(48.0);
        let shaped = r.shape("AV", &style);
        let img = r.rasterize("AV", &style);
        assert_eq!((img.width, img.height), shaped.extent);
        assert!(covered(&img) > 0);
    }

    #[test]
    fn center_align_with_wrap_width_does_not_clip() {
        let mut r = test_renderer();
        // Regression: centered text in a wide wrap box used to render into a
        // bitmap sized without the alignment offset, clipping half the glyphs.
        let left = r.rasterize("Hi", &TextStyle::new(48.0).with_max_width(400.0));
        let centered = r.rasterize(
            "Hi",
            &TextStyle::new(48.0)
                .with_align(TextAlign::Center)
                .with_max_width(400.0),
        );
        // Subpixel positioning may wiggle the AA fringe by a pixel; anything
        // more means the alignment offset leaked into the bitmap size again.
        assert!(
            left.width.abs_diff(centered.width) <= 1 && left.height.abs_diff(centered.height) <= 1,
            "centered {}x{} vs left {}x{}",
            centered.width,
            centered.height,
            left.width,
            left.height
        );
        // Coverage must match within a small tolerance, not lose half the
        // glyphs (the old bug clipped the right half of the centered run).
        let (a, b) = (covered(&left) as i64, covered(&centered) as i64);
        assert!(
            (a - b).abs() <= (a / 5).max(8),
            "centered run lost coverage: left={a} centered={b}"
        );
    }

    #[test]
    fn center_align_without_wrap_width_centers_short_lines() {
        let mut r = test_renderer();
        // The lone "H" on line 2 must shift right when centering against the
        // wider first line — previously a silent no-op without max_width.
        let style = TextStyle::new(40.0);
        let lone_h = |shaped: &ShapedText| {
            shaped
                .clusters
                .iter()
                .find(|c| c.line == 1)
                .expect("second line cluster")
                .offset[0]
        };
        let flush = r.shape("HHHH\nH", &style);
        let centered = r.shape("HHHH\nH", &style.clone().with_align(TextAlign::Center));
        assert!(lone_h(&flush) < 5.0, "left-aligned H should hug the margin");
        assert!(
            lone_h(&centered) > lone_h(&flush) + 10.0,
            "centered H did not move: flush={} centered={}",
            lone_h(&flush),
            lone_h(&centered)
        );
    }

    #[test]
    fn memo_caches_repeat_calls_and_invalidate_on_font_load() {
        let mut r = test_renderer();
        let style = TextStyle::new(48.0);

        // Repeat rasterize: one entry each (rasterize populates shape too),
        // identical output.
        let first = r.rasterize("Hi", &style);
        let again = r.rasterize("Hi", &style);
        assert_eq!(first, again);
        assert_eq!(r.memo_sizes(), (1, 1));

        // Same key through shape() hits the shared entry. Bitmap-only style
        // changes create raster entries but keep reusing that shaped run.
        let _ = r.shape("Hi", &style);
        assert_eq!(r.memo_sizes(), (1, 1));
        let _ = r.rasterize("Hi", &style.clone().with_padding(3));
        assert_eq!(r.memo_sizes(), (1, 2));
        let _ = r.rasterize(
            "Hi",
            &style.clone().with_stroke(TextStroke {
                rgba: [0, 0, 0, 255],
                width: 6.0,
            }),
        );
        assert_eq!(r.memo_sizes(), (1, 3));

        // Loading a font can change shaping for any string: memos must drop.
        assert!(r.load_font(TEST_FONT.to_vec()) > 0);
        assert_eq!(r.memo_sizes(), (0, 0));
    }
}
