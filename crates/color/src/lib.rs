//! Pixel-format conversion.
//!
//! Callers that need CPU pixels ask [`to_rgba`]. This build converts no
//! layout: every request is [`ConvertError::Unsupported`]. A later change
//! in this crate is the converter. Callers do not invent a second one.

#![forbid(unsafe_code)]

use std::fmt;

/// Coded samples were not turned into RGBA.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvertError {
    /// `layout` has no conversion in this build.
    Unsupported { layout: &'static str },
}

impl fmt::Display for ConvertError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported { layout } => write!(formatter, "no conversion for {layout}"),
        }
    }
}

impl std::error::Error for ConvertError {}

/// Tightly packed 8-bit RGBA, or a miss when this build cannot convert `layout`.
///
/// `width` and `height` are the coded picture. `bytes` are the coded samples.
/// They are not rearranged here.
pub fn to_rgba(
    layout: &'static str,
    _width: u32,
    _height: u32,
    _bytes: &[u8],
) -> Result<Vec<u8>, ConvertError> {
    Err(ConvertError::Unsupported { layout })
}

#[cfg(test)]
mod tests {
    use super::{ConvertError, to_rgba};

    #[test]
    fn no_layout_converts_yet() {
        let err = to_rgba("rgb8", 2, 2, &[1, 2, 3, 4]).expect_err("rgb");
        assert_eq!(err, ConvertError::Unsupported { layout: "rgb8" });
        let err = to_rgba("yuv420p", 2, 2, &[0, 1, 2, 3, 4, 5]).expect_err("yuv");
        assert_eq!(err, ConvertError::Unsupported { layout: "yuv420p" });
    }
}
