//! Text runs: glyph positions, line layout, and a picture sampled from a
//! signed-distance atlas.
//!
//! Rich-text commands, bitmap atlases, stroke, shadow, background, and
//! caption curves are not implemented here.

#![forbid(unsafe_code)]

mod shape;

pub use shape::{MAX_CHARS, ShapeItem, Shaped, ShapedGlyph, shape, shape_text};

use std::error::Error;
use std::fmt;

use cutlass_font::FontError;

/// Why a text run could not be shaped or drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextError {
    /// The font file could not be read, or a glyph id is not in the face.
    Font(FontError),
    /// The string, the pen, or the picture is past a limit.
    Limit,
}

impl fmt::Display for TextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Font(error) => write!(formatter, "{error}"),
            Self::Limit => formatter.write_str("text is past the limit"),
        }
    }
}

impl Error for TextError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Font(error) => Some(error),
            Self::Limit => None,
        }
    }
}
