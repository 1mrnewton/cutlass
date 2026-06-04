use cutlass_decode::DecodeError;
use cutlass_models::{MediaId, ModelError};

/// Errors from the engine's frame-resolution and edit-command paths.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// A frame was requested for media not registered in the pool.
    #[error("unknown media: {0}")]
    UnknownMedia(MediaId),

    /// The requested source frame lies past the end of the media.
    #[error("source frame {frame} is past the end of {media}")]
    FrameOutOfRange { media: MediaId, frame: i64 },

    /// The underlying decoder failed.
    #[error(transparent)]
    Decode(#[from] DecodeError),

    /// An edit command violated a model invariant (overlap, bounds, bad ref).
    #[error(transparent)]
    Model(#[from] ModelError),

    /// An export could not be produced (e.g. an empty timeline).
    #[error("cannot export: {0}")]
    Export(String),
}
