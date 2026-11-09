//! Depth videos for Relight, kept in the project's `Resources/video_relight`
//! folder.
//!
//! A depth video holds how near each pixel of one stretch of one source file
//! is, as a grey picture a frame, made once by a slow model and kept with the
//! project so export and other machines find it. Its name works as an
//! enhanced copy's does: a hash of the file as it is on disk, the model and
//! the rate, then the stretch, so trimming a relit clip keeps its depth and
//! a changed source finds none.

use std::path::{Path, PathBuf};

use cutlass_model::TimeRange;

use crate::StoreError;
use crate::enhance::{copy_file, find_copy, source_stem};

const RELIGHT_DIR: &str = "video_relight";

/// What a depth video was made from and how.
#[derive(Clone, Debug, PartialEq)]
pub struct RelightKey {
    /// The source file, as the shell opened it.
    pub source: PathBuf,
    /// The model that guessed the depth, so a better model makes new ones.
    pub model: String,
    /// Pictures per second in the depth video.
    pub rate: f64,
}

impl RelightKey {
    fn stem(&self) -> Result<String, StoreError> {
        let mut bytes = b"relight\0".to_vec();
        bytes.extend_from_slice(self.model.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(&self.rate.to_bits().to_le_bytes());
        source_stem(&self.source, &bytes)
    }
}

/// Where the depth video of `range` made as `key` says goes, in the project
/// at `project`. The folder is made; the file is not.
pub fn relight_file(
    project: &Path,
    key: &RelightKey,
    range: &TimeRange,
) -> Result<PathBuf, StoreError> {
    copy_file(project, &key.stem()?, RELIGHT_DIR, range)
}

/// The shortest depth video made as `key` says whose stretch covers
/// `range`, and that stretch, or `None` when there is none or the source
/// cannot be read.
#[must_use]
pub fn find_relight(
    project: &Path,
    key: &RelightKey,
    range: &TimeRange,
) -> Option<(PathBuf, TimeRange)> {
    find_copy(project, &key.stem().ok()?, RELIGHT_DIR, range)
}

#[cfg(test)]
mod tests;
