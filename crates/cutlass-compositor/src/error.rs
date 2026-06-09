use thiserror::Error;

#[derive(Debug, Error)]
pub enum CompositorError {
    #[error("no GPU adapter available")]
    NoAdapter,

    #[error("failed to request GPU device: {0}")]
    RequestDevice(String),

    #[error("invalid canvas dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },

    #[error("layer RGBA buffer size mismatch: got {got} bytes, expected {expected}")]
    LayerSizeMismatch { got: usize, expected: usize },

    #[error("GPU buffer map failed")]
    MapFailed,

    #[error("wgpu surface error: {0}")]
    Wgpu(String),
}

impl From<wgpu::BufferAsyncError> for CompositorError {
    fn from(_: wgpu::BufferAsyncError) -> Self {
        Self::MapFailed
    }
}
