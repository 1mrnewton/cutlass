//! Errors raised while resolving or rendering a project.

use cutlass_compositor::CompositorError;
use cutlass_core::{DecodeError, EncodeError};
use cutlass_models::{MediaId, ModelError};

/// A render failure: a bad model query, a decode/compositor error, or content
/// the current renderer doesn't handle yet.
#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    /// A timeline/clip query against the model failed.
    #[error("model error: {0}")]
    Model(#[from] ModelError),

    /// Decoding a media frame failed.
    #[error("decode error: {0}")]
    Decode(#[from] DecodeError),

    /// The compositor failed to render the layer stack.
    #[error("compositor error: {0}")]
    Compositor(#[from] CompositorError),

    /// Encoding or muxing an exported frame failed.
    #[error("encode error: {0}")]
    Encode(#[from] EncodeError),

    /// Reading or writing a file failed (project load, frame output).
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    /// A clip references a media id that isn't in the project's media pool.
    #[error("media {0:?} is not in the project media pool")]
    MissingMedia(MediaId),

    /// The decoder reached end-of-stream before the requested source time.
    #[error("no frame available for media {media:?} at {time:?}")]
    NoFrame {
        media: MediaId,
        time: cutlass_core::RationalTime,
    },

    /// Content or a platform path the current renderer doesn't support yet.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// An observed export was cancelled by its progress callback. The output
    /// file was not finalized; callers should treat it as garbage.
    #[error("export cancelled")]
    Cancelled,
}

impl RenderError {
    pub fn unsupported(msg: impl Into<String>) -> Self {
        RenderError::Unsupported(msg.into())
    }
}
