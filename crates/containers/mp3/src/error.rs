//! Failures from reading an MP3 file.

/// Why a file could not be read.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// The file ended before a frame or a tag did.
    #[error("file ended at byte {at} while reading {need} bytes")]
    Truncated { at: u64, need: u64 },
    /// The file does not start with a frame or an ID3v2 tag.
    #[error("the file is not MP3")]
    Form,
    /// A number does not fit the field it is stored in.
    #[error("a number in the file does not fit")]
    TooLarge,
    /// The file breaks a rule this reader enforces.
    #[error("{0}")]
    Inconsistent(&'static str),
}
