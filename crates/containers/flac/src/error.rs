//! Failures from reading a FLAC file.

/// Why a file could not be read.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// The file ended before a metadata block did.
    #[error("file ended at byte {at} while reading {need} bytes")]
    Truncated { at: u64, need: u64 },
    /// The marker is not `fLaC`.
    #[error("the file is not FLAC")]
    Form,
    /// A number does not fit the field it is stored in.
    #[error("a number in the file does not fit")]
    TooLarge,
    /// The file breaks a rule this reader enforces.
    #[error("{0}")]
    Inconsistent(&'static str),
}
