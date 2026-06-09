use cutlass_cache::DiskCacheError;
use cutlass_compositor::CompositorError;
use cutlass_decoder::DecodeError;
use cutlass_models::ModelError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error(transparent)]
    Compositor(#[from] CompositorError),

    #[error(transparent)]
    Model(#[from] ModelError),

    #[error(transparent)]
    Decode(#[from] DecodeError),

    #[error(transparent)]
    Cache(#[from] DiskCacheError),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("import failed: {0}")]
    Import(String),

    #[error("media file not found: {0}")]
    MissingMedia(String),

    #[error("preview: {0}")]
    Preview(String),
}
