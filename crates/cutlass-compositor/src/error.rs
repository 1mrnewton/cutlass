//! Compositor failures, kept backend-agnostic.

use thiserror::Error;

/// Something went wrong setting up the GPU or compositing a frame.
#[derive(Debug, Error)]
pub enum CompositorError {
    /// No GPU adapter satisfied the request (no hardware, no driver).
    #[error("no GPU adapter available: {0}")]
    NoAdapter(String),

    /// The adapter was found but a logical device couldn't be created.
    #[error("failed to request GPU device: {0}")]
    RequestDevice(String),

    /// Canvas dimensions were zero (nothing to render into).
    #[error("invalid canvas dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },

    /// A frame's pixel format isn't handled by the compositor yet (e.g. 10-bit
    /// P010, which needs a 16-bit sampling path).
    #[error("unsupported frame format: {0}")]
    UnsupportedFormat(String),

    /// A frame's planes don't match its declared geometry (a plane buffer is
    /// too small for its `stride × rows`, or the wrong plane count).
    #[error("malformed frame: {0}")]
    MalformedFrame(String),

    /// Reading the rendered canvas back to the CPU failed.
    #[error("GPU buffer map failed")]
    MapFailed,
}

impl From<wgpu::BufferAsyncError> for CompositorError {
    fn from(_: wgpu::BufferAsyncError) -> Self {
        Self::MapFailed
    }
}
