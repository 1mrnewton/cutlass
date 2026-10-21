//! Errors from a YUV frame that cannot be converted.

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("width and height must be nonzero")]
    ZeroDimension,
    /// NV12 and P010 store chroma on a 2×2 grid.
    #[error("width and height must be even")]
    OddDimension,
    #[error("the frame is too large to address")]
    TooLarge,
    #[error("the {0} plane is shorter than its stride and size")]
    ShortPlane(&'static str),
    /// An H.273 matrix code this conversion does not implement.
    #[error("unsupported matrix code {0}")]
    UnsupportedMatrix(u16),
}
