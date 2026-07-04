//! App-owned project drafts (CapCut-style storage).
//!
//! Cutlass owns every project: there is no user-chosen `.cutlass` path. Each
//! project lives in its own directory under `projects/` in the per-user OS
//! data dir (see [`crate::paths`]) and auto-saves continuously, so the user
//! never saves by hand and no edit is lost on a clean exit. A directory holds
//! the project itself (`project.cutlass` — a plain project file the engine
//! reads and writes through the normal `Save`/`Load` path) and a small
//! `meta.json` sidecar caching the display name so the launch gallery can
//! list projects without parsing every project file.
//!
//! `.cutlass` files are no longer a user-facing concept; they enter via
//! [`import_external`] (Open file…) and leave via Export. The draft directory
//! is the identity — addressed everywhere by its `project.cutlass` path, so
//! the existing path-keyed engine plumbing (`Save`/`Load`) is reused as-is.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tracing::warn;

/// File name of the project inside a draft directory.
const PROJECT_FILE: &str = "project.cutlass";
/// File name of the display-name sidecar inside a draft directory.
const META_FILE: &str = "meta.json";
/// Shown for a draft whose meta is missing or empty.
const FALLBACK_NAME: &str = "Untitled project";

/// `projects/` in the per-user OS data dir (see [`crate::paths`]):
/// `%APPDATA%\Cutlass` on Windows, `~/Library/Application Support/Cutlass` on
/// macOS, `~/.local/share/Cutlass` on Linux.
pub fn root_dir() -> PathBuf {
    crate::paths::data_dir().join("projects")
}

/// The project file inside a draft directory.
pub fn project_file(dir: &Path) -> PathBuf {
    dir.join(PROJECT_FILE)
}

fn meta_file(dir: &Path) -> PathBuf {
    dir.join(META_FILE)
}

/// Display name + last-modified time for one draft, for the launch gallery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftSummary {
    /// Display name (from the meta sidecar, or a fallback).
    pub name: String,
    /// The draft's `project.cutlass` path — the identity used to open it.
    pub project: PathBuf,
    /// When the project file was last written (newest first in the gallery).
    pub modified: SystemTime,
}

/// Display name persisted alongside a draft so the gallery needn't parse the
/// whole project file.
#[derive(Debug, Default, Serialize, Deserialize)]
struct DraftMeta {
    name: String,
}

/// Create a fresh, empty draft directory and return its `project.cutlass`
/// path (the file itself is written by the first save once the engine binds
/// to it). Ids are time-based with a per-process counter so two creates in
/// the same instant never collide.
pub fn create() -> std::io::Result<PathBuf> {
    let root = root_dir();
    std::fs::create_dir_all(&root)?;
    loop {
        let dir = root.join(new_id());
        match std::fs::create_dir(&dir) {
            Ok(()) => return Ok(project_file(&dir)),
            // Lost the race on this id; spin for the next one.
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
}

/// Copy an external `.cutlass` file into a new draft directory (Open file…),
/// naming the draft after the source file's stem. Returns the new draft's
/// `project.cutlass` path. The original file on disk is left untouched — it
/// was only an import source.
pub fn import_external(source: &Path) -> std::io::Result<PathBuf> {
    let root = root_dir();
    std::fs::create_dir_all(&root)?;
    let dir = loop {
        let dir = root.join(new_id());
        match std::fs::create_dir(&dir) {
            Ok(()) => break dir,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    };
    let project = project_file(&dir);
    std::fs::copy(source, &project)?;
    let name = source
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    write_meta(&project, &name);
    Ok(project)
}

/// Cache `name` in the draft's meta sidecar. Called on every auto-save so the
/// gallery name tracks the project name. Failures only log — the project file
/// is the source of truth; a stale gallery name never loses work.
pub fn write_meta(project: &Path, name: &str) {
    let Some(dir) = project.parent() else {
        return;
    };
    let meta = DraftMeta {
        name: name.to_owned(),
    };
    match serde_json::to_string(&meta) {
        Ok(json) => {
            if let Err(e) = std::fs::write(meta_file(dir), json) {
                warn!(dir = %dir.display(), "draft meta write failed: {e}");
            }
        }
        Err(e) => warn!("draft meta serialize failed: {e}"),
    }
}

/// Delete a draft (its whole directory), addressed by its `project.cutlass`
/// path. Missing files are fine.
pub fn delete(project: &Path) {
    if let Some(dir) = project.parent() {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// Every draft, newest-modified first — the launch gallery's source. A draft
/// directory without a readable `project.cutlass` is skipped (a half-created
/// or corrupt slot never crashes the gallery).
pub fn list() -> Vec<DraftSummary> {
    let Ok(entries) = std::fs::read_dir(root_dir()) else {
        return Vec::new();
    };
    let mut drafts: Vec<DraftSummary> = entries
        .filter_map(|e| e.ok())
        .filter_map(|entry| {
            let dir = entry.path();
            if !dir.is_dir() {
                return None;
            }
            let project = project_file(&dir);
            let modified = std::fs::metadata(&project)
                .and_then(|m| m.modified())
                .ok()?;
            Some(DraftSummary {
                name: read_name(&dir),
                project,
                modified,
            })
        })
        .collect();
    drafts.sort_by_key(|d| std::cmp::Reverse(d.modified));
    drafts
}

fn read_name(dir: &Path) -> String {
    std::fs::read_to_string(meta_file(dir))
        .ok()
        .and_then(|json| serde_json::from_str::<DraftMeta>(&json).ok())
        .map(|m| m.name)
        .filter(|n| !n.trim().is_empty())
        .unwrap_or_else(|| FALLBACK_NAME.to_owned())
}

/// A coarse "5 minutes ago" label for `modified`, dependency-free, for the
/// gallery card subtitle. Future or unreadable times read as "just now".
pub fn relative_time(modified: SystemTime) -> String {
    let secs = SystemTime::now()
        .duration_since(modified)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    match secs {
        0..=59 => "just now".to_owned(),
        60..=3599 => plural(secs / 60, "minute"),
        3600..=86_399 => plural(secs / 3600, "hour"),
        _ => plural(secs / 86_400, "day"),
    }
}

fn plural(n: u64, unit: &str) -> String {
    if n == 1 {
        format!("1 {unit} ago")
    } else {
        format!("{n} {unit}s ago")
    }
}

/// A unique-per-process draft id: millis since the epoch plus a monotonic
/// counter, so rapid successive creates never share a directory name.
fn new_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{millis:x}-{seq:x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_id_is_unique_across_rapid_calls() {
        let ids: std::collections::HashSet<String> = (0..1000).map(|_| new_id()).collect();
        assert_eq!(ids.len(), 1000, "ids collided");
    }

    #[test]
    fn meta_roundtrips_through_a_draft_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let project = project_file(dir.path());
        std::fs::write(&project, b"{}").expect("write project");
        write_meta(&project, "My Movie");
        assert_eq!(read_name(dir.path()), "My Movie");
    }

    #[test]
    fn missing_or_blank_meta_falls_back() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert_eq!(read_name(dir.path()), FALLBACK_NAME);
        write_meta(&project_file(dir.path()), "   ");
        assert_eq!(read_name(dir.path()), FALLBACK_NAME);
    }

    #[test]
    fn relative_time_reads_in_words() {
        let now = SystemTime::now();
        assert_eq!(relative_time(now), "just now");
        assert_eq!(
            relative_time(now - std::time::Duration::from_secs(60)),
            "1 minute ago"
        );
        assert_eq!(
            relative_time(now - std::time::Duration::from_secs(7200)),
            "2 hours ago"
        );
    }
}
