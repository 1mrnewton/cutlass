//! The project folder on disk.
//!
//! Open is parse, then upgrade, then a report. Upgrade and repair return a
//! draft. They do not push history. The undo stack stays in the session.

#![forbid(unsafe_code)]

mod json;
mod upgrade;

use std::fmt;
use std::path::PathBuf;

pub use json::{parse_draft, write_draft};
pub use upgrade::{BUILD, Protocol, upgrade};

/// Why a folder could not be read or written.
#[derive(Debug)]
pub enum StoreError {
    /// The bytes are not the JSON this loader accepts.
    Json { path: PathBuf, message: String },
    /// `new_version` is empty, so no upgrade task can be ordered.
    EmptyProtocol { draft_id: String },
    /// `new_version` is not a numeric protocol version.
    InvalidProtocol { draft_id: String, found: String },
    /// The file is newer than this build. It is not downgraded into something this build can edit.
    NewerProtocol { draft_id: String, found: String },
}

impl fmt::Display for StoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json { path, message } => {
                write!(formatter, "could not read {}: {message}", path.display())
            }
            Self::EmptyProtocol { draft_id } => {
                write!(formatter, "draft {draft_id} has an empty protocol version")
            }
            Self::InvalidProtocol { draft_id, found } => {
                write!(
                    formatter,
                    "draft {draft_id} has protocol version {found}, which is not a version"
                )
            }
            Self::NewerProtocol { draft_id, found } => {
                write!(
                    formatter,
                    "draft {draft_id} has protocol {found}, which is newer than this build ({BUILD})"
                )
            }
        }
    }
}

impl std::error::Error for StoreError {}
