//! Failures while opening a GPU or uploading a frame.

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Color(#[from] cutlass_color::Error),
    #[error("no GPU adapter")]
    NoAdapter,
    /// P010 needs 16-bit unorm textures, and this device does not have them.
    #[error("this GPU cannot sample P010")]
    NoP010,
    #[error("the GPU rejected the device: {0}")]
    Device(String),
    #[error("waiting for the GPU failed: {0}")]
    Wait(String),
    #[error("reading the frame back failed: {0}")]
    Readback(String),
}
