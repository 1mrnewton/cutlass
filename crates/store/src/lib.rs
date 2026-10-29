//! The project folder on disk.
//!
//! Open is parse, then upgrade, then a report. A folder with no project
//! file becomes one timeline. The top draft replaces the main timeline
//! only when it is newer and the platform is outside the desktop. Repair
//! names problems and does not change the loaded draft. Neither pushes
//! history. The undo stack stays in the session.

#![forbid(unsafe_code)]

mod backup;
mod folder;
mod json;
mod repair;
mod upgrade;

use std::fmt;
use std::path::PathBuf;

pub use backup::copy_backup;
pub use cutlass_model::Draft;
pub use folder::{
    DockItem, DraftMeta, Folder, IdMap, IdPair, MaterialBin, Opened, ProjectIndex, TimelineEntry,
    TimelineLayout, open,
};
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
    /// `draft_info.json` is not in the folder.
    MissingDraft { path: PathBuf },
    /// A timeline id or a symlink would leave the project folder.
    PathEscapes { name: String },
    /// The main timeline is marked deleted. Its folder is left in place.
    MainTimelineMarkedDeleted { id: String },
    /// The main timeline folder's draft id is not the directory name.
    TimelineIdMismatch { directory: String, draft_id: String },
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
            Self::MissingDraft { path } => {
                write!(formatter, "missing draft file {}", path.display())
            }
            Self::PathEscapes { name } => {
                write!(formatter, "path {name} leaves the project folder")
            }
            Self::MainTimelineMarkedDeleted { id } => {
                write!(formatter, "the main timeline {id} cannot be marked deleted")
            }
            Self::TimelineIdMismatch {
                directory,
                draft_id,
            } => {
                write!(
                    formatter,
                    "timeline folder {directory} holds draft {draft_id}"
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
