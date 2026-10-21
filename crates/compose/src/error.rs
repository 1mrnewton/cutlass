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
    /// The device is not the Metal device that can wrap an IOSurface.
    #[cfg(target_vendor = "apple")]
    #[error("this GPU is not Metal")]
    NotMetal,
    /// The pixel buffer has no IOSurface to sample.
    #[cfg(target_vendor = "apple")]
    #[error("the frame has no IOSurface")]
    NoSurface,
    /// Core Video returned a format this compositor does not sample.
    #[cfg(target_vendor = "apple")]
    #[error("the frame format is not NV12 or P010")]
    UnsupportedFormat,
    /// Metal refused to wrap a plane.
    #[cfg(target_vendor = "apple")]
    #[error("importing the frame failed: {0}")]
    Import(String),
}
