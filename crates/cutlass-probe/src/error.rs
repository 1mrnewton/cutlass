use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProbeError {
    #[cfg(feature = "ffmpeg")]
    #[error("failed to open media")]
    Open(#[source] ffmpeg_next::Error),

    #[error("unsupported: {what}")]
    Unsupported { what: String },

    #[error("path is not valid UTF-8")]
    InvalidPath,
}

impl ProbeError {
    pub fn unsupported(what: impl Into<String>) -> Self {
        ProbeError::Unsupported { what: what.into() }
    }
}
