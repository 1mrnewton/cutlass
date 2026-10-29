//! Caption sampling: a word clock, pages, selectors, and animation batches.
//!
//! Word times stay in milliseconds. Segment placement stays in microseconds.
//! Sampling reads those records and returns batches the text system can place.
//! It does not write the draft, and it does not run recognition.

#![forbid(unsafe_code)]

mod animate;
mod clock;
mod page;
mod select;

use std::error::Error;
use std::fmt;

use cutlass_text::TextError;

/// Why a caption sample could not be produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CaptionError {
    /// `start_time`, `end_time`, and `text` have different lengths.
    Ragged,
    /// A word ends before it starts.
    Inverted,
    /// A timestamp does not fit in `i64`.
    Overflow,
    /// A word index is not in the columns.
    Missing,
    /// A keyword does not fall inside its word.
    KeywordRange,
    /// A shake, pulse, or stagger period is shorter than two milliseconds.
    Period,
    /// A keyframe or amplitude is not a finite number.
    Measure,
    /// The edited string is past the rematch limit.
    Limit,
    /// Layout refused the text.
    Text(TextError),
}

impl fmt::Display for CaptionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ragged => formatter.write_str("word columns have different lengths"),
            Self::Inverted => formatter.write_str("a word ends before it starts"),
            Self::Overflow => formatter.write_str("a caption time overflowed"),
            Self::Missing => formatter.write_str("a word index is not in the columns"),
            Self::KeywordRange => formatter.write_str("a keyword does not fall inside its word"),
            Self::Period => formatter.write_str("a caption motion period is too short"),
            Self::Measure => formatter.write_str("a caption value is not finite"),
            Self::Limit => formatter.write_str("caption text is past the limit"),
            Self::Text(error) => write!(formatter, "{error}"),
        }
    }
}

impl Error for CaptionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Text(error) => Some(error),
            Self::Ragged
            | Self::Inverted
            | Self::Overflow
            | Self::Missing
            | Self::KeywordRange
            | Self::Period
            | Self::Measure
            | Self::Limit => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CaptionError;
    use cutlass_text::TextError;
    use std::error::Error;

    #[test]
    fn a_layout_failure_keeps_its_source() {
        let error = CaptionError::Text(TextError::Measure);
        assert_eq!(
            error.to_string(),
            "text size, spacing, or width is not usable"
        );
        assert!(error.source().is_some());
        assert!(CaptionError::Ragged.source().is_none());
        assert_eq!(
            CaptionError::KeywordRange.to_string(),
            "a keyword does not fall inside its word"
        );
    }
}
