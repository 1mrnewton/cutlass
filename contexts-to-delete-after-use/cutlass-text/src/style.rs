//! How a text run should look: size, color, family, alignment, wrapping.

/// Horizontal alignment of wrapped / multi-line text inside its bitmap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextAlign {
    #[default]
    Left,
    Center,
    Right,
}

/// Which font family to shape with. `Named` looks the family up by name in the
/// loaded font set; the generic families fall back to whatever the platform
/// (or a loaded font) provides.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum FontFamily {
    #[default]
    SansSerif,
    Serif,
    Monospace,
    Named(String),
}

/// The styling for a rasterized text run.
///
/// Construct with [`TextStyle::new`] (a size, white, left-aligned, sans-serif,
/// unwrapped) and adjust with the `with_*` builders.
#[derive(Debug, Clone, PartialEq)]
pub struct TextStyle {
    /// Font size in pixels.
    pub font_size: f32,
    /// Baseline-to-baseline line height in pixels.
    pub line_height: f32,
    /// Straight-alpha RGBA fill (the `a` scales the whole run's opacity).
    pub color: [u8; 4],
    pub family: FontFamily,
    pub align: TextAlign,
    /// Wrap width in pixels. `None` lays each paragraph out on one line; `Some`
    /// word-wraps to that width.
    pub max_width: Option<f32>,
    /// Transparent margin (px) added on every side of the measured text box —
    /// headroom for glyph overhang, soft shadows, or strokes added later.
    pub padding: u32,
}

impl TextStyle {
    /// A white, left-aligned, unwrapped sans-serif run at `font_size` px, with
    /// a 1.25× line height.
    pub fn new(font_size: f32) -> Self {
        Self {
            font_size,
            line_height: font_size * 1.25,
            color: [255, 255, 255, 255],
            family: FontFamily::SansSerif,
            align: TextAlign::Left,
            max_width: None,
            padding: 0,
        }
    }

    pub fn with_color(mut self, color: [u8; 4]) -> Self {
        self.color = color;
        self
    }

    pub fn with_family(mut self, family: FontFamily) -> Self {
        self.family = family;
        self
    }

    pub fn with_align(mut self, align: TextAlign) -> Self {
        self.align = align;
        self
    }

    pub fn with_line_height(mut self, line_height: f32) -> Self {
        self.line_height = line_height;
        self
    }

    /// Word-wrap to `width` pixels.
    pub fn with_max_width(mut self, width: f32) -> Self {
        self.max_width = Some(width);
        self
    }

    pub fn with_padding(mut self, padding: u32) -> Self {
        self.padding = padding;
        self
    }
}
