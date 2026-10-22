//! Failures from reading an AVI file.

/// Why a file could not be read.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// The file ended before the chunk did.
    #[error("file ended at byte {at} while reading {need} bytes")]
    Truncated { at: u64, need: u64 },
    /// The form is not `AVI `.
    #[error("the file is not AVI")]
    Form,
    /// A number does not fit the field it is stored in.
    #[error("a number in the file does not fit")]
    TooLarge,
    /// The file breaks a rule this reader enforces.
    #[error("{0}")]
    Inconsistent(&'static str),
}
