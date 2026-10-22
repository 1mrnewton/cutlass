//! Failures from reading a JPEG file.

/// Why a file could not be decoded.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// The file does not start with the JPEG start marker.
    #[error("the file is not JPEG")]
    Signature,
    /// The file ended before a marker, a segment, or a coefficient did.
    #[error("file ended at byte {at} while reading {need} bytes")]
    Truncated { at: u64, need: u64 },
    /// Width, height, or a decoded buffer is past the decoder limit.
    #[error("the picture is larger than the decoder allows")]
    TooLarge,
    /// A field uses a value this decoder does not read.
    #[error("{0}")]
    Unsupported(&'static str),
    /// The file breaks a rule this decoder enforces.
    #[error("{0}")]
    Inconsistent(&'static str),
}
