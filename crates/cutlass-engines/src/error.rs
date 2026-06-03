use cutlass_decode::DecodeError;
use cutlass_models::MediaId;

/// Errors from the engine's frame-resolution path.
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
}
