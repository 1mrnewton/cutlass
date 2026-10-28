//! Pause, cancel, and discard.
//!
//! Pause and cancel stop the transfer and keep the partial. Discard deletes
//! it. Resume continues a paused partial. `storage` publishes a finished
//! file only after the checksum matches.

use std::fmt;

use crate::{Job, JobId};

/// How far one transfer has gotten. The partial file is still on disk
/// unless the state is discarded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Progress {
    /// Bytes are still arriving.
    Partial { bytes_written: u64 },
    /// The sockets are stopped. The partial is kept.
    Paused { bytes_written: u64 },
    /// The transfer was cancelled. The partial is kept.
    Cancelled { bytes_written: u64 },
    /// Every byte is in the destination. `storage` has not published it yet.
    Finished { bytes_written: u64 },
}

/// Why a transfer call failed.
#[derive(Debug)]
pub enum DownloadError {
    /// The transfer could not continue.
    Failed { message: String },
    /// No transfer has this id.
    UnknownJob { id: JobId },
}

impl fmt::Display for DownloadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Failed { message } => formatter.write_str(message),
            Self::UnknownJob { id } => write!(formatter, "no transfer has id {id}"),
        }
    }
}

impl std::error::Error for DownloadError {}

/// The calls `storage` makes. A test double stands in for the range fetcher.
pub trait Downloader {
    /// Start writing `job` into its destination.
    fn start(&mut self, job: &Job) -> Result<JobId, DownloadError>;

    /// The latest state of one transfer.
    fn progress(&mut self, id: JobId) -> Result<Progress, DownloadError>;

    /// Stop the sockets and keep the partial.
    fn pause(&mut self, id: JobId) -> Result<(), DownloadError>;

    /// Continue a paused partial.
    fn resume(&mut self, id: JobId) -> Result<(), DownloadError>;

    /// Stop the sockets and keep the partial.
    fn cancel(&mut self, id: JobId) -> Result<(), DownloadError>;

    /// Stop the transfer and delete the partial.
    fn discard(&mut self, id: JobId) -> Result<(), DownloadError>;
}
