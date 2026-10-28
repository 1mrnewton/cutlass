//! One URL into a partial file.
//!
//! `storage` calls this crate. The bytes stay partial until `storage`
//! publishes them. Pause and cancel keep the partial. Discard deletes it.
//! A finished file is published only after the checksum matches, and that
//! publish is `storage`'s step, not this one.

#![forbid(unsafe_code)]

mod checksum;
mod control;
mod job;

pub use checksum::{Checksum, ChecksumError, hash_file};
pub use control::{DownloadError, Downloader, Progress};
pub use job::{Job, JobId, Priority};
