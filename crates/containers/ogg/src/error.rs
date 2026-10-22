//! Failures from reading an Ogg file.

/// Why a file could not be read.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// The file ended before a page or a segment did.
    #[error("file ended at byte {at} while reading {need} bytes")]
    Truncated { at: u64, need: u64 },
    /// The file does not start with `OggS`.
    #[error("the file is not Ogg")]
    Form,
    /// A number does not fit the field it is stored in.
    #[error("a number in the file does not fit")]
    TooLarge,
    /// The file breaks a rule this reader enforces.
    #[error("{0}")]
    Inconsistent(&'static str),
}
