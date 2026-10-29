//! Text runs: glyph positions, line layout, and a picture sampled from a
//! signed-distance atlas.
//!
//! Rich-text commands, bitmap atlases, stroke, shadow, background, and
//! caption curves are not implemented here.

#![forbid(unsafe_code)]

mod layout;
mod richtext;
mod shape;

pub use layout::{Alignment, Layout, Line, PlacedGlyph, layout, layout_text};
pub use shape::{MAX_CHARS, ShapeItem, Shaped, ShapedGlyph, shape, shape_text};

use std::error::Error;
use std::fmt;

use cutlass_font::FontError;

/// Why a text run could not be shaped or drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextError {
    /// The font file could not be read, or a glyph id is not in the face.
    Font(FontError),
    /// The material is marked as rich text. Commands are not read in this pass.
    RichTextCommands,
    /// Typesetting is not the horizontal run this pass lays out.
    Typesetting,
    /// Alignment is not left, center, or right.
    Alignment,
    /// Size, spacing, or width is not a usable number.
    Measure,
    /// The string, the pen, or the picture is past a limit.
    Limit,
}

impl fmt::Display for TextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Font(error) => write!(formatter, "{error}"),
            Self::RichTextCommands => formatter.write_str("rich-text commands are not read"),
            Self::Typesetting => formatter.write_str("vertical text is not laid out"),
            Self::Alignment => formatter.write_str("text alignment is not left, center, or right"),
            Self::Measure => formatter.write_str("text size, spacing, or width is not usable"),
            Self::Limit => formatter.write_str("text is past the limit"),
        }
    }
}

impl Error for TextError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Font(error) => Some(error),
            Self::RichTextCommands
            | Self::Typesetting
            | Self::Alignment
            | Self::Measure
            | Self::Limit => None,
        }
    }
}
