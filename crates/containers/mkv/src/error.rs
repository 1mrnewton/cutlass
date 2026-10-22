//! Failures from reading an MKV or WebM file.

/// Why a file could not be read.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// The file ended before the element did.
    #[error("file ended at byte {at} while reading {need} bytes")]
    Truncated { at: u64, need: u64 },
    /// An element id is longer than this file allows, or the width marker is empty.
    #[error("the element id is invalid")]
    BadId,
    /// An element size is longer than this file allows, or the width marker is empty.
    #[error("the element size is invalid")]
    BadSize,
    /// The doctype is not `matroska` or `webm`.
    #[error("the file is not Matroska or WebM")]
    DocType,
    /// `EBMLReadVersion` is above 1.
    #[error("the EBML version is not supported")]
    Version,
    /// A number does not fit the field it is stored in.
    #[error("a number in the file does not fit")]
    TooLarge,
    /// The file breaks a rule this reader enforces.
    #[error("{0}")]
    Inconsistent(&'static str),
}
