//! Enhanced, denoised, deflickered and interpolated copies of clips, kept
//! in the project's `Resources/video_enhance` folder,
//! `Resources/video_denoise` for a copy that is denoised but not enhanced,
//! `Resources/video_deflicker` for one that is deflickered but neither,
//! `Resources/video_interpolate` for one that is only interpolated, or
//! `Resources/video_motion_blur` for one that is only blurred.
//!
//! Making a copy is slow, so it is kept with the project, where export and
//! other machines find it and no cache eviction takes it away. A copy
//! holds one stretch of one source file. Its name is a hash of the file as
//! it was on disk (path, size, last change), the levels, the model and the
//! copy's frame rate, then the stretch: `<hash>-<start>-<duration>.mp4`.
//! A copy whose stretch covers a clip's serves it, so trimming or splitting
//! an enhanced clip keeps its copy, and a changed source finds none.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use cutlass_model::TimeRange;

use crate::StoreError;
use crate::folder::{RESOURCES_DIR, io_error, stay_inside};

const ENHANCE_DIR: &str = "video_enhance";
const DENOISE_DIR: &str = "video_denoise";
const DEFLICKER_DIR: &str = "video_deflicker";
const INTERPOLATE_DIR: &str = "video_interpolate";
const MOTION_BLUR_DIR: &str = "video_motion_blur";

/// The most copies looked through for one clip.
const MAX_LISTED: usize = 4096;

/// What a copy was made from and how.
#[derive(Clone, Debug, PartialEq)]
pub struct EnhanceKey {
    /// The source file, as the shell opened it.
    pub source: PathBuf,
    /// The Enhance quality level, 0 for a copy that is only denoised.
    pub level: i64,
    /// The Reduce image noise level, 0 for a copy that is not denoised.
    pub denoise: i64,
    /// The Remove flicker level, 0 for a copy whose brightness is as the
    /// clip's.
    pub deflicker: i64,
    /// The Optical flow frame rate, 0 for a copy with no frames made
    /// between the clip's own.
    pub interpolate: i64,
    /// The Motion blur it was made with, written out, or empty for a copy
    /// that is not blurred.
    pub motion_blur: String,
    /// The model that made it, so a better model makes new copies.
    pub model: String,
    /// Pictures per second in the copy.
    pub rate: f64,
}

impl EnhanceKey {
    /// The name every copy of this key starts with, or an error when the
    /// source cannot be read.
    fn stem(&self) -> Result<String, StoreError> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.level.to_le_bytes());
        // Only when on, so a copy that is only enhanced keeps the name it
        // had before copies could be denoised.
        if self.denoise > 0 {
            bytes.extend_from_slice(&self.denoise.to_le_bytes());
        }
        if self.deflicker > 0 {
            bytes.push(b'f');
            bytes.extend_from_slice(&self.deflicker.to_le_bytes());
        }
        if self.interpolate > 0 {
            bytes.push(b'i');
            bytes.extend_from_slice(&self.interpolate.to_le_bytes());
        }
        if !self.motion_blur.is_empty() {
            bytes.push(b'm');
            bytes.extend_from_slice(self.motion_blur.as_bytes());
            bytes.push(0);
        }
        bytes.extend_from_slice(self.model.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(&self.rate.to_bits().to_le_bytes());
        source_stem(&self.source, &bytes)
    }

    /// The folder its copies go in.
    fn folder(&self) -> &'static str {
        if self.level > 0 {
            ENHANCE_DIR
        } else if self.denoise > 0 {
            DENOISE_DIR
        } else if self.deflicker > 0 {
            DEFLICKER_DIR
        } else if self.interpolate > 0 {
            INTERPOLATE_DIR
        } else {
            MOTION_BLUR_DIR
        }
    }
}

/// Where the copy of `range` made as `key` says goes, in the project at
/// `project`. The folder is made; the file is not.
pub fn enhanced_file(
    project: &Path,
    key: &EnhanceKey,
    range: &TimeRange,
) -> Result<PathBuf, StoreError> {
    copy_file(project, &key.stem()?, key.folder(), range)
}

/// The shortest copy made as `key` says whose stretch covers `range`, and
/// that stretch, or `None` when there is none or the source cannot be
/// read.
#[must_use]
pub fn find_enhanced(
    project: &Path,
    key: &EnhanceKey,
    range: &TimeRange,
) -> Option<(PathBuf, TimeRange)> {
    find_copy(project, &key.stem().ok()?, key.folder(), range)
}

/// Where a file named `stem` holding `range` goes in `folder` of the
/// project's resources. The folder is made; the file is not.
pub(crate) fn copy_file(
    project: &Path,
    stem: &str,
    folder: &str,
    range: &TimeRange,
) -> Result<PathBuf, StoreError> {
    if range.start.0 < 0 || range.duration.0 <= 0 {
        return Err(StoreError::PathEscapes {
            name: format!("{}+{}", range.start.0, range.duration.0),
        });
    }
    let project = project
        .canonicalize()
        .map_err(|error| io_error(project, error))?;
    let folder = project.join(RESOURCES_DIR).join(folder);
    stay_inside(&project, &folder)?;
    fs::create_dir_all(&folder).map_err(|error| io_error(&folder, error))?;
    stay_inside(&project, &folder)?;
    let file = folder.join(format!("{stem}-{}-{}.mp4", range.start.0, range.duration.0));
    stay_inside(&project, &file)?;
    Ok(file)
}

/// The shortest file named `stem` in `folder` whose stretch covers
/// `range`, and that stretch.
pub(crate) fn find_copy(
    project: &Path,
    stem: &str,
    folder: &str,
    range: &TimeRange,
) -> Option<(PathBuf, TimeRange)> {
    let project = project.canonicalize().ok()?;
    let folder = project.join(RESOURCES_DIR).join(folder);
    stay_inside(&project, &folder).ok()?;
    let end = range.start.0.checked_add(range.duration.0)?;
    let mut best: Option<(PathBuf, TimeRange)> = None;
    for entry in fs::read_dir(&folder).ok()?.take(MAX_LISTED).flatten() {
        let name = entry.file_name();
        let Some(held) = name.to_str().and_then(|name| stretch(name, stem)) else {
            continue;
        };
        let held_end = held.start.0.saturating_add(held.duration.0);
        if held.start.0 > range.start.0 || held_end < end {
            continue;
        }
        let path = entry.path();
        if !path.is_file() || stay_inside(&project, &path).is_err() {
            continue;
        }
        if best
            .as_ref()
            .is_none_or(|(_, kept)| held.duration.0 < kept.duration.0)
        {
            best = Some((path, held));
        }
    }
    best
}

/// A name for `source` as it is on disk (path, size, last change) and
/// `extra`, or an error when the source cannot be read.
pub(crate) fn source_stem(source: &Path, extra: &[u8]) -> Result<String, StoreError> {
    let meta = fs::metadata(source).map_err(|error| io_error(source, error))?;
    let modified = meta
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .and_then(|since| i64::try_from(since.as_millis()).ok())
        .unwrap_or(i64::MIN);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(source.to_string_lossy().as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(&meta.len().to_le_bytes());
    bytes.extend_from_slice(&modified.to_le_bytes());
    bytes.extend_from_slice(extra);
    // FNV-1a, 64 bits: a file name, not a lock.
    let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    });
    Ok(format!("{hash:016x}"))
}

/// The stretch a copy named `name` holds, when it starts with `stem`.
fn stretch(name: &str, stem: &str) -> Option<TimeRange> {
    let rest = name.strip_prefix(stem)?.strip_prefix('-')?;
    let rest = rest.strip_suffix(".mp4")?;
    let (start, duration) = rest.split_once('-')?;
    let start = start.parse::<i64>().ok()?;
    let duration = duration.parse::<i64>().ok()?;
    (start >= 0 && duration > 0).then(|| TimeRange::new(start, duration))
}

#[cfg(test)]
mod tests;
