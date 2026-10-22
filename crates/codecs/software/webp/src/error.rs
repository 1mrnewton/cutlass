//! Failures from reading a WebP file.

/// Why a file could not be decoded.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// The file does not start with a WebP RIFF header.
    #[error("the file is not WebP")]
    Signature,
    /// The file ended before a field or a code did.
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
