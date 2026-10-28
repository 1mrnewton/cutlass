//! The project folder on disk.
//!
//! Open is parse, then upgrade, then a report. Upgrade and repair return a
//! draft. They do not push history. The undo stack stays in the session.

#![forbid(unsafe_code)]

mod json;

use std::fmt;
use std::path::PathBuf;

pub use json::{parse_draft, write_draft};

/// Why a folder could not be read or written.
#[derive(Debug)]
pub enum StoreError {
    /// The bytes are not the JSON this loader accepts.
    Json { path: PathBuf, message: String },
}

impl fmt::Display for StoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json { path, message } => {
                write!(formatter, "could not read {}: {message}", path.display())
            }
        }
    }
}

impl std::error::Error for StoreError {}
