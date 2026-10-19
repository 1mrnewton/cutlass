use thiserror::Error;

#[derive(Debug, Error)]
pub enum EncodeError {
    #[cfg(feature = "ffmpeg")]
    #[error("failed to open media")]
    Open(#[source] ffmpeg_next::Error),

    #[cfg(feature = "ffmpeg")]
    #[error("muxer I/O failed")]
    Io(#[source] ffmpeg_next::Error),

    #[cfg(feature = "ffmpeg")]
    #[error("encode failed")]
    Encode(#[source] ffmpeg_next::Error),

    #[error("unsupported: {what}")]
    Unsupported { what: String },

    #[error(transparent)]
    Decode(#[from] cutlass_decoder::DecodeError),
}

impl EncodeError {
    pub fn unsupported(what: impl Into<String>) -> Self {
        EncodeError::Unsupported { what: what.into() }
    }
}
