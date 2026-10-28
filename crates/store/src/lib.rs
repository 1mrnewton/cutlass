//! The project folder on disk.
//!
//! Open is parse, then upgrade, then a report. Upgrade returns a draft.
//! Repair names problems and does not change that draft. Neither pushes
//! history. The undo stack stays in the session.

#![forbid(unsafe_code)]

mod backup;
mod json;
mod repair;
mod upgrade;

use std::fmt;
use std::path::PathBuf;

pub use backup::copy_backup;
pub use json::{parse_draft, write_draft};
pub use repair::{RepairNote, repair};
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
    /// A backup is a copy of `draft_info.json` or `project.json`, and of no other file.
    BackupRefused { path: PathBuf },
    /// The filesystem refused the read or the write.
    Io { path: PathBuf, message: String },
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
            Self::BackupRefused { path } => {
                write!(
                    formatter,
                    "a backup copies draft_info.json or project.json, not {}",
                    path.display()
                )
            }
            Self::Io { path, message } => {
                write!(formatter, "could not access {}: {message}", path.display())
            }
        }
    }
}

impl std::error::Error for StoreError {}
