use cutlass_cache::DiskCacheError;
use cutlass_compositor::CompositorError;
use cutlass_decoder::DecodeError;
use cutlass_encoder::EncodeError;
use cutlass_models::ModelError;
use cutlass_probe::ProbeError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error(transparent)]
    Compositor(#[from] CompositorError),

    #[error(transparent)]
    Encode(#[from] EncodeError),

    #[error(transparent)]
    Model(#[from] ModelError),

    #[error(transparent)]
    Decode(#[from] DecodeError),

    #[error(transparent)]
    Probe(#[from] ProbeError),

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

    #[error("export: {0}")]
    Export(String),

    #[error("export cancelled")]
    ExportCancelled,

    #[error("transcription: {0}")]
    Transcribe(String),
}

impl From<cutlass_ml::TranscribeError> for EngineError {
    fn from(err: cutlass_ml::TranscribeError) -> Self {
        EngineError::Transcribe(err.to_string())
    }
}
