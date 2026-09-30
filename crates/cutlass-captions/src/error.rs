// --- Caption errors -------------------------------------------------------------------

use cutlass_models::ModelError;

/// Everything that can go wrong turning words or a subtitle file into cues.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CaptionError {
    /// The segmentation rules were unusable (see [`CaptionLayout::validate`]).
    ///
    /// [`CaptionLayout::validate`]: cutlass_models::CaptionLayout::validate
    #[error("{0}")]
    Rules(String),
    /// A subtitle file could not be read. `line` is 1-based, as an editor shows
    /// it, so the message can be pointed straight at the offending line.
    #[error("line {line}: {message}")]
    Parse { line: usize, message: String },
    /// The input would produce more cues than one group may hold.
    #[error("{count} cues exceeds the {max} a caption group holds")]
    TooManyCues { count: usize, max: usize },
}

impl CaptionError {
    pub(crate) fn parse(line: usize, message: impl Into<String>) -> Self {
        Self::Parse {
            line,
            message: message.into(),
        }
    }
}

impl From<ModelError> for CaptionError {
    fn from(error: ModelError) -> Self {
        Self::Rules(error.to_string())
    }
}
