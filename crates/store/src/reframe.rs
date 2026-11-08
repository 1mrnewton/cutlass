//! Where Auto reframe found each clip's subject, kept in the project's
//! `Resources/video_reframe` folder.
//!
//! Finding the subject reads the clip's whole file with a model, so it is
//! done once and kept with the project; the window is planned from it at
//! draw time, so another shape or setting needs no new search. A file is
//! named by the source as it is on disk (path, size, last change), the
//! model and the rate, so a changed source finds none.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use cutlass_model::TimeRange;
use cutlass_motion::Subject;
use serde::{Deserialize, Serialize};

use crate::StoreError;
use crate::enhance::source_stem;
use crate::folder::{RESOURCES_DIR, io_error, stay_inside, write_bytes};

const REFRAME_DIR: &str = "video_reframe";

/// The version of the file this build writes and reads.
const VERSION: u32 = 1;

/// The largest file read back.
const MAX_BYTES: u64 = 64 * 1024 * 1024;

/// The most samples a kept file may hold: over two days at 5 a second.
const MAX_SUBJECTS: usize = 1_000_000;

/// What was searched: a stretch of one file at a rate, with a model.
#[derive(Clone, Debug, PartialEq)]
pub struct ReframeKey {
    /// The source file, as the shell opened it.
    pub source: PathBuf,
    /// The stretch searched, in the file's own time.
    pub range: TimeRange,
    /// The model that found the subject, so a better one searches again.
    pub model: String,
    /// Samples a second.
    pub rate: f64,
}

/// Where the subject was in each sample of a clip, and the size of the
/// upright frames it was found in.
#[derive(Clone, Debug, PartialEq)]
pub struct SubjectPath {
    pub width: u32,
    pub height: u32,
    pub subjects: Vec<Subject>,
}

#[derive(Serialize, Deserialize)]
struct Stored {
    version: u32,
    width: u32,
    height: u32,
    /// Each sample as `[x, y, weight]`.
    subjects: Vec<[f64; 3]>,
}

impl ReframeKey {
    fn file(&self, project: &Path) -> Result<PathBuf, StoreError> {
        let mut bytes = b"reframe\0".to_vec();
        bytes.extend_from_slice(&self.range.start.0.to_le_bytes());
        bytes.extend_from_slice(&self.range.duration.0.to_le_bytes());
        bytes.extend_from_slice(self.model.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(&self.rate.to_bits().to_le_bytes());
        let stem = source_stem(&self.source, &bytes)?;
        Ok(project
            .join(RESOURCES_DIR)
            .join(REFRAME_DIR)
            .join(format!("{stem}.json")))
    }
}

/// Keep `path`, found as `key` says, in the project at `project`. A sample
/// that is not a number is kept as nothing found.
pub fn save_subject_path(
    project: &Path,
    key: &ReframeKey,
    path: &SubjectPath,
) -> Result<(), StoreError> {
    let project = project
        .canonicalize()
        .map_err(|error| io_error(project, error))?;
    let file = key.file(&project)?;
    let subjects = path
        .subjects
        .iter()
        .map(|s| {
            let row = [s.x, s.y, s.weight];
            if row.iter().all(|v| v.is_finite()) {
                row
            } else {
                [0.5, 0.5, 0.0]
            }
        })
        .collect();
    let stored = Stored {
        version: VERSION,
        width: path.width,
        height: path.height,
        subjects,
    };
    let bytes = serde_json::to_vec(&stored).map_err(|error| StoreError::Json {
        path: file.clone(),
        message: error.to_string(),
    })?;
    write_bytes(&project, &file, &bytes)
}

/// The path kept for `key` in the project at `project`, or `None` when
/// there is none, the source changed since, or the file is not one this
/// build reads.
#[must_use]
pub fn load_subject_path(project: &Path, key: &ReframeKey) -> Option<SubjectPath> {
    let project = project.canonicalize().ok()?;
    let file = key.file(&project).ok()?;
    stay_inside(&project, &file).ok()?;
    let mut bytes = Vec::new();
    File::open(&file)
        .ok()?
        .take(MAX_BYTES + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() as u64 > MAX_BYTES {
        return None;
    }
    let stored: Stored = serde_json::from_slice(&bytes).ok()?;
    if stored.version != VERSION
        || stored.width == 0
        || stored.height == 0
        || stored.subjects.is_empty()
        || stored.subjects.len() > MAX_SUBJECTS
    {
        return None;
    }
    Some(SubjectPath {
        width: stored.width,
        height: stored.height,
        subjects: stored
            .subjects
            .iter()
            .map(|&[x, y, weight]| Subject { x, y, weight })
            .collect(),
    })
}

#[cfg(test)]
mod tests;
