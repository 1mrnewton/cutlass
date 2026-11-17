//! Matte videos for Remove BG, kept in the project's `Resources/video_matte`
//! folder.
//!
//! A matte video holds how much of each pixel of one stretch of one source
//! file is the subject, as a grey picture a frame (white kept, black
//! removed), made once by a slow model and kept with the project so export
//! and other machines find it. Its name works as a depth video's does: a
//! hash of the file as it is on disk, the model, the rate and, for Custom
//! removal, the picks, then the stretch. So trimming a clip keeps its matte,
//! and a changed source or a new pick finds none.

use std::path::{Path, PathBuf};

use cutlass_model::TimeRange;

use crate::StoreError;
use crate::enhance::{copy_file, find_copy, source_stem};
use crate::folder::{RESOURCES_DIR, io_error, stay_inside};

const MATTE_DIR: &str = "video_matte";

/// What a matte video was made from and how.
#[derive(Clone, Debug, PartialEq)]
pub struct MatteKey {
    /// The source file, as the shell opened it.
    pub source: PathBuf,
    /// The model that found the subject, so a better model makes new ones.
    pub model: String,
    /// Pictures per second in the matte video.
    pub rate: f64,
    /// What Custom removal's picks are worth, empty for Auto removal.
    pub picks: String,
}

impl MatteKey {
    fn stem(&self) -> Result<String, StoreError> {
        let mut bytes = b"matte\0".to_vec();
        bytes.extend_from_slice(self.model.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(&self.rate.to_bits().to_le_bytes());
        bytes.push(0);
        bytes.extend_from_slice(self.picks.as_bytes());
        source_stem(&self.source, &bytes)
    }
}

/// Where the matte video of `range` made as `key` says goes, in the project
/// at `project`. The folder is made; the file is not.
pub fn matte_file(
    project: &Path,
    key: &MatteKey,
    range: &TimeRange,
) -> Result<PathBuf, StoreError> {
    copy_file(project, &key.stem()?, MATTE_DIR, range)
}

/// The shortest matte video made as `key` says whose stretch covers
/// `range`, and that stretch, or `None` when there is none or the source
/// cannot be read.
#[must_use]
pub fn find_matte(
    project: &Path,
    key: &MatteKey,
    range: &TimeRange,
) -> Option<(PathBuf, TimeRange)> {
    find_copy(project, &key.stem().ok()?, MATTE_DIR, range)
}

/// Remove every matte video in the project at `project` but those in
/// `keep`, and any a job left half written; how many went. Mattes are made
/// again when needed, so a project does not keep one for every trim and
/// pick it has had.
pub fn remove_mattes_but(project: &Path, keep: &[PathBuf]) -> Result<usize, StoreError> {
    let project = project
        .canonicalize()
        .map_err(|error| io_error(project, error))?;
    let folder = project.join(RESOURCES_DIR).join(MATTE_DIR);
    let entries = match std::fs::read_dir(&folder) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(io_error(&folder, error)),
    };
    stay_inside(&project, &folder)?;
    let keep: Vec<PathBuf> = keep
        .iter()
        .filter_map(|path| path.canonicalize().ok())
        .collect();
    let mut removed = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        let made = matches!(
            path.extension().and_then(|ext| ext.to_str()),
            Some("mp4" | "part")
        );
        if !made || !path.is_file() || stay_inside(&project, &path).is_err() {
            continue;
        }
        if keep.contains(&path) {
            continue;
        }
        std::fs::remove_file(&path).map_err(|error| io_error(&path, error))?;
        removed += 1;
    }
    Ok(removed)
}

#[cfg(test)]
mod tests;
