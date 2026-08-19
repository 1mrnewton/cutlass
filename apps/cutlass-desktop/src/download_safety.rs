//! Keeps project-referenced download-cache media out of cache eviction.
//!
//! Draft discovery is intentionally shallow: only a real directory directly
//! beneath the supplied root can contribute its `project.cutlass`. Reports
//! contain bounded counters and flags, never paths or filesystem error text.

use std::path::{Component, Path};

use cutlass_cloud::cache::DownloadCache;
use cutlass_models::Project;

const PROJECT_FILE: &str = "project.cutlass";

/// Maximum number of draft-root entries inspected by one scan.
///
/// Capping entries, rather than only valid directories, also bounds work when
/// an unexpected file-filled directory is supplied as the draft root.
pub(crate) const MAX_DRAFTS_SCANNED: usize = 10_000;

/// Maximum combined logical size of project files handed to the model loader.
pub(crate) const MAX_TOTAL_PROJECT_BYTES: u64 = 256 * 1024 * 1024;

/// Accounting for media paths examined in one or more projects.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub(crate) struct ProtectionCounts {
    /// All media-pool entries examined.
    pub(crate) examined: usize,
    /// Entries whose paths were lexically beneath the download-cache root.
    pub(crate) referenced: usize,
    /// Cache references accepted by [`DownloadCache::protect_path`].
    pub(crate) protected: usize,
    /// Cache references rejected by [`DownloadCache::protect_path`].
    pub(crate) rejected: usize,
}

impl ProtectionCounts {
    fn merge(&mut self, other: Self) {
        self.examined = self.examined.saturating_add(other.examined);
        self.referenced = self.referenced.saturating_add(other.referenced);
        self.protected = self.protected.saturating_add(other.protected);
        self.rejected = self.rejected.saturating_add(other.rejected);
    }
}

/// Bounded accounting for a saved-draft scan.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub(crate) struct DraftScanCounts {
    /// Child entries inspected as possible drafts.
    pub(crate) draft_entries_examined: usize,
    /// Valid `project.cutlass` files loaded.
    pub(crate) projects_loaded: usize,
    /// Logical project-file bytes admitted to the loader, including malformed
    /// files that the loader rejected.
    pub(crate) project_bytes_read: u64,
    /// Known entries or project files skipped or rejected during discovery.
    ///
    /// If `draft_limit_reached` is true, further uncounted entries may remain.
    pub(crate) skipped_or_errored: usize,
    /// The draft-entry cap stopped discovery.
    pub(crate) draft_limit_reached: bool,
    /// At least one project file was skipped to preserve the byte cap.
    pub(crate) byte_limit_reached: bool,
    /// Aggregate media-path accounting from valid projects.
    pub(crate) media: ProtectionCounts,
}

impl DraftScanCounts {
    /// Whether every discoverable draft and cache-contained media path was
    /// classified, making destructive cache operations safe to enable.
    pub(crate) fn is_complete(self) -> bool {
        self.skipped_or_errored == 0
            && !self.draft_limit_reached
            && !self.byte_limit_reached
            && self.media.rejected == 0
    }
}

#[derive(Debug, Clone, Copy)]
struct ScanLimits {
    draft_entries: usize,
    project_bytes: u64,
}

/// Protect every media-pool path that is lexically contained beneath
/// `cache.root()`.
///
/// Outside paths, relative paths, the cache root itself, and paths containing
/// parent traversal after the root are examined but never passed to the cache.
/// A missing cache-owned file is still eligible because `protect_path`
/// supports protecting a future re-download.
pub(crate) fn protect_project_downloads(
    cache: &DownloadCache,
    project: &Project,
) -> ProtectionCounts {
    let mut counts = ProtectionCounts::default();

    for media in project.media_iter() {
        counts.examined = counts.examined.saturating_add(1);
        if !is_lexically_beneath(&cache.root(), media.path()) {
            continue;
        }

        counts.referenced = counts.referenced.saturating_add(1);
        if cache.protect_path(media.path()).is_ok() {
            counts.protected = counts.protected.saturating_add(1);
        } else {
            counts.rejected = counts.rejected.saturating_add(1);
        }
    }

    counts
}

/// Discover saved drafts directly beneath `drafts_root` and protect their
/// download-cache media.
///
/// Discovery never follows a symlink in place of a draft directory or project
/// file. Missing roots, unreadable entries, absent/malformed project files,
/// and individual protection failures are reduced to bounded accounting and
/// do not abort the rest of the scan.
pub(crate) fn protect_saved_draft_downloads(
    cache: &DownloadCache,
    drafts_root: &Path,
) -> DraftScanCounts {
    protect_saved_draft_downloads_with_limits(
        cache,
        drafts_root,
        ScanLimits {
            draft_entries: MAX_DRAFTS_SCANNED,
            project_bytes: MAX_TOTAL_PROJECT_BYTES,
        },
    )
}

fn protect_saved_draft_downloads_with_limits(
    cache: &DownloadCache,
    drafts_root: &Path,
    limits: ScanLimits,
) -> DraftScanCounts {
    let mut counts = DraftScanCounts::default();

    match std::fs::symlink_metadata(drafts_root) {
        Ok(metadata) if metadata.file_type().is_dir() => {}
        Ok(_) => {
            counts.skipped_or_errored = 1;
            return counts;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return counts,
        Err(_) => {
            counts.skipped_or_errored = 1;
            return counts;
        }
    }

    let Ok(entries) = std::fs::read_dir(drafts_root) else {
        counts.skipped_or_errored = 1;
        return counts;
    };

    for (index, entry) in entries.enumerate() {
        if index >= limits.draft_entries {
            counts.draft_limit_reached = true;
            counts.skipped_or_errored = counts.skipped_or_errored.saturating_add(1);
            break;
        }
        counts.draft_entries_examined = counts.draft_entries_examined.saturating_add(1);

        let Ok(entry) = entry else {
            counts.skipped_or_errored = counts.skipped_or_errored.saturating_add(1);
            continue;
        };
        let Ok(file_type) = entry.file_type() else {
            counts.skipped_or_errored = counts.skipped_or_errored.saturating_add(1);
            continue;
        };
        // DirEntry::file_type does not follow a symlink. In particular, do not
        // replace this with Path::is_dir or metadata().
        if !file_type.is_dir() {
            counts.skipped_or_errored = counts.skipped_or_errored.saturating_add(1);
            continue;
        }

        let project_path = entry.path().join(PROJECT_FILE);
        let Ok(metadata) = std::fs::symlink_metadata(&project_path) else {
            counts.skipped_or_errored = counts.skipped_or_errored.saturating_add(1);
            continue;
        };
        if !metadata.file_type().is_file() {
            counts.skipped_or_errored = counts.skipped_or_errored.saturating_add(1);
            continue;
        }

        let Some(next_bytes) = counts.project_bytes_read.checked_add(metadata.len()) else {
            counts.byte_limit_reached = true;
            counts.skipped_or_errored = counts.skipped_or_errored.saturating_add(1);
            continue;
        };
        if next_bytes > limits.project_bytes {
            counts.byte_limit_reached = true;
            counts.skipped_or_errored = counts.skipped_or_errored.saturating_add(1);
            continue;
        }
        counts.project_bytes_read = next_bytes;

        let Ok(project) = Project::load_from_file(&project_path) else {
            counts.skipped_or_errored = counts.skipped_or_errored.saturating_add(1);
            continue;
        };
        counts.projects_loaded = counts.projects_loaded.saturating_add(1);
        counts
            .media
            .merge(protect_project_downloads(cache, &project));
    }

    counts
}

fn is_lexically_beneath(root: &Path, path: &Path) -> bool {
    if !root.is_absolute() || !path.is_absolute() {
        return false;
    }

    let Ok(relative) = path.strip_prefix(root) else {
        return false;
    };
    !relative.as_os_str().is_empty()
        && relative
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use cutlass_models::{MediaSource, Rational};
    use tempfile::TempDir;

    use super::*;

    fn cache_in(temp: &TempDir) -> (DownloadCache, PathBuf) {
        let root = temp.path().join("download-cache");
        fs::create_dir(&root).unwrap();
        (DownloadCache::new(root.clone(), u64::MAX), root)
    }

    fn project_with_media(paths: impl IntoIterator<Item = PathBuf>) -> Project {
        let mut project = Project::new("test", Rational::FPS_30);
        for path in paths {
            project.add_media(MediaSource::new(
                path,
                1_920,
                1_080,
                Rational::FPS_30,
                300,
                true,
            ));
        }
        project
    }

    fn save_draft(root: &Path, name: &str, project: &Project) -> PathBuf {
        let directory = root.join(name);
        fs::create_dir(&directory).unwrap();
        let path = directory.join(PROJECT_FILE);
        project.save_to_file(&path).unwrap();
        path
    }

    #[test]
    fn project_protection_only_submits_lexically_contained_paths() {
        let temp = tempfile::tempdir().unwrap();
        let (cache, cache_root) = cache_in(&temp);

        let inside = cache_root.join("sfx/inside.mp4");
        fs::create_dir(inside.parent().unwrap()).unwrap();
        fs::write(&inside, b"media").unwrap();
        let future = cache_root.join("sfx/future.mp4");
        let directory = cache_root.join("not-media");
        fs::create_dir(&directory).unwrap();
        let outside = temp.path().join("outside.mp4");
        fs::write(&outside, b"outside").unwrap();
        let traversal = cache_root.join("../outside.mp4");

        let project = project_with_media([inside, future, directory, outside, traversal]);
        let counts = protect_project_downloads(&cache, &project);

        assert_eq!(
            counts,
            ProtectionCounts {
                examined: 5,
                referenced: 3,
                protected: 2,
                rejected: 1,
            }
        );
        assert_eq!(cache.protected_path_count(), 2);
    }

    #[test]
    fn scan_continues_past_malformed_and_missing_drafts() {
        let temp = tempfile::tempdir().unwrap();
        let (cache, cache_root) = cache_in(&temp);
        let drafts = temp.path().join("projects");
        fs::create_dir(&drafts).unwrap();

        let inside = cache_root.join("valid.mp4");
        fs::write(&inside, b"media").unwrap();
        let outside = temp.path().join("outside.mp4");
        fs::write(&outside, b"outside").unwrap();
        let valid = project_with_media([inside, outside]);
        let valid_path = save_draft(&drafts, "valid", &valid);
        let valid_bytes = fs::metadata(valid_path).unwrap().len();

        let malformed_dir = drafts.join("malformed");
        fs::create_dir(&malformed_dir).unwrap();
        let malformed = b"{broken";
        fs::write(malformed_dir.join(PROJECT_FILE), malformed).unwrap();
        fs::create_dir(drafts.join("missing-project")).unwrap();
        fs::write(drafts.join("not-a-draft"), b"ignored").unwrap();

        let counts = protect_saved_draft_downloads(&cache, &drafts);

        assert_eq!(counts.draft_entries_examined, 4);
        assert_eq!(counts.projects_loaded, 1);
        assert_eq!(
            counts.project_bytes_read,
            valid_bytes + malformed.len() as u64
        );
        assert_eq!(counts.skipped_or_errored, 3);
        assert!(!counts.draft_limit_reached);
        assert!(!counts.byte_limit_reached);
        assert_eq!(
            counts.media,
            ProtectionCounts {
                examined: 2,
                referenced: 1,
                protected: 1,
                rejected: 0,
            }
        );
        assert_eq!(cache.protected_path_count(), 1);
        assert!(!counts.is_complete());
    }

    #[test]
    fn missing_drafts_root_is_a_complete_empty_inventory() {
        let temp = tempfile::tempdir().unwrap();
        let (cache, _) = cache_in(&temp);

        let counts =
            protect_saved_draft_downloads(&cache, &temp.path().join("missing-projects-root"));

        assert_eq!(counts, DraftScanCounts::default());
        assert!(counts.is_complete());
    }

    #[test]
    fn valid_drafts_produce_a_complete_inventory() {
        let temp = tempfile::tempdir().unwrap();
        let (cache, cache_root) = cache_in(&temp);
        let drafts = temp.path().join("projects");
        fs::create_dir(&drafts).unwrap();
        let media = cache_root.join("sfx/kept.mp4");
        fs::create_dir(media.parent().unwrap()).unwrap();
        fs::write(&media, b"media").unwrap();
        save_draft(&drafts, "valid", &project_with_media([media]));

        let counts = protect_saved_draft_downloads(&cache, &drafts);

        assert_eq!(counts.projects_loaded, 1);
        assert_eq!(counts.media.protected, 1);
        assert!(counts.is_complete());
    }

    #[test]
    fn draft_count_cap_stops_discovery() {
        let temp = tempfile::tempdir().unwrap();
        let (cache, cache_root) = cache_in(&temp);
        let drafts = temp.path().join("projects");
        fs::create_dir(&drafts).unwrap();

        for index in 0..3 {
            let media = cache_root.join(format!("media-{index}.mp4"));
            fs::write(&media, b"media").unwrap();
            save_draft(
                &drafts,
                &format!("draft-{index}"),
                &project_with_media([media]),
            );
        }

        let counts = protect_saved_draft_downloads_with_limits(
            &cache,
            &drafts,
            ScanLimits {
                draft_entries: 2,
                project_bytes: u64::MAX,
            },
        );

        assert_eq!(counts.draft_entries_examined, 2);
        assert_eq!(counts.projects_loaded, 2);
        assert_eq!(counts.skipped_or_errored, 1);
        assert!(counts.draft_limit_reached);
        assert!(!counts.byte_limit_reached);
        assert_eq!(
            counts.media,
            ProtectionCounts {
                examined: 2,
                referenced: 2,
                protected: 2,
                rejected: 0,
            }
        );
        assert_eq!(cache.protected_path_count(), 2);
        assert!(!counts.is_complete());
    }

    #[test]
    fn total_project_byte_cap_skips_files_that_do_not_fit() {
        let temp = tempfile::tempdir().unwrap();
        let (cache, cache_root) = cache_in(&temp);
        let drafts = temp.path().join("projects");
        fs::create_dir(&drafts).unwrap();

        let media = cache_root.join("shared.mp4");
        fs::write(&media, b"media").unwrap();
        let source = temp.path().join("source.cutlass");
        project_with_media([media]).save_to_file(&source).unwrap();
        let project_bytes = fs::read(&source).unwrap();
        for name in ["first", "second"] {
            let directory = drafts.join(name);
            fs::create_dir(&directory).unwrap();
            fs::write(directory.join(PROJECT_FILE), &project_bytes).unwrap();
        }

        let counts = protect_saved_draft_downloads_with_limits(
            &cache,
            &drafts,
            ScanLimits {
                draft_entries: 10,
                project_bytes: project_bytes.len() as u64,
            },
        );

        assert_eq!(counts.draft_entries_examined, 2);
        assert_eq!(counts.projects_loaded, 1);
        assert_eq!(counts.project_bytes_read, project_bytes.len() as u64);
        assert_eq!(counts.skipped_or_errored, 1);
        assert!(!counts.draft_limit_reached);
        assert!(counts.byte_limit_reached);
        assert_eq!(
            counts.media,
            ProtectionCounts {
                examined: 1,
                referenced: 1,
                protected: 1,
                rejected: 0,
            }
        );
        assert_eq!(cache.protected_path_count(), 1);
        assert!(!counts.is_complete());
    }

    #[cfg(unix)]
    #[test]
    fn symlink_draft_directories_are_not_traversed() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let (cache, cache_root) = cache_in(&temp);
        let drafts = temp.path().join("projects");
        fs::create_dir(&drafts).unwrap();

        let media = cache_root.join("linked.mp4");
        fs::write(&media, b"media").unwrap();
        project_with_media([media])
            .save_to_file(&outside.path().join(PROJECT_FILE))
            .unwrap();
        symlink(outside.path(), drafts.join("linked-draft")).unwrap();

        let counts = protect_saved_draft_downloads(&cache, &drafts);

        assert_eq!(counts.draft_entries_examined, 1);
        assert_eq!(counts.projects_loaded, 0);
        assert_eq!(counts.skipped_or_errored, 1);
        assert_eq!(counts.media, ProtectionCounts::default());
        assert_eq!(cache.protected_path_count(), 0);
        assert!(!counts.is_complete());
    }
}
