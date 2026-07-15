//! The quota-managed download cache.
//!
//! Stock files, template bundles, and packs land here before import.
//! Unbounded data-dir growth is a real complaint generator, so the cache
//! enforces a byte quota with LRU eviction (by modification time — files
//! get re-touched on use). Files *imported into a project* are copied out
//! by the import path and become pool media; nothing in a project ever
//! references a cache path, so eviction is always safe.
//!
//! Settings surfaces a "clear download cache" action backed by
//! [`DownloadCache::clear`].

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use crate::error::CloudError;

/// Default quota: 2 GiB of downloaded stock/bundles before eviction.
pub const DEFAULT_QUOTA_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// A directory with a byte budget.
pub struct DownloadCache {
    root: PathBuf,
    quota_bytes: AtomicU64,
}

impl DownloadCache {
    pub fn new(root: PathBuf, quota_bytes: u64) -> Self {
        Self {
            root,
            quota_bytes: AtomicU64::new(quota_bytes),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns the current byte quota.
    ///
    /// Quota enforcement snapshots this value once per invocation, so a
    /// concurrent update takes effect on a subsequent invocation.
    #[must_use]
    pub fn quota_bytes(&self) -> u64 {
        self.quota_bytes.load(Ordering::Relaxed)
    }

    /// Replaces the byte quota used by subsequent quota enforcement.
    ///
    /// An enforcement already in progress continues using the quota it
    /// snapshotted when it started.
    pub fn set_quota_bytes(&self, new: u64) {
        self.quota_bytes.store(new, Ordering::Relaxed);
    }

    /// Where a download for `key` (e.g. `stock/pexels-857191-hd.mp4`)
    /// should land. Creates parent directories.
    pub fn path_for(&self, key: &str) -> Result<PathBuf, CloudError> {
        let path = self.root.join(key);
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        Ok(path)
    }

    /// Whether `key` is already downloaded (touches it for LRU).
    pub fn hit(&self, key: &str) -> Option<PathBuf> {
        let path = self.root.join(key);
        if path.is_file() {
            // Re-touch so recently-used files survive eviction.
            let _ = filetime_touch(&path);
            Some(path)
        } else {
            None
        }
    }

    /// Total bytes currently cached.
    pub fn size_bytes(&self) -> u64 {
        saturating_total_bytes(
            walk_files(&self.root)
                .into_iter()
                .filter_map(|p| std::fs::metadata(p).ok())
                .map(|m| m.len()),
        )
    }

    /// Evict least-recently-modified files until under quota. Called after
    /// each completed download; cheap when under budget.
    pub fn enforce_quota(&self) {
        let quota_bytes = self.quota_bytes();
        let mut files: Vec<(PathBuf, u64, SystemTime)> = walk_files(&self.root)
            .into_iter()
            .filter_map(|p| {
                let meta = std::fs::metadata(&p).ok()?;
                let mtime = meta.modified().ok()?;
                Some((p, meta.len(), mtime))
            })
            .collect();
        let mut total = saturating_total_bytes(files.iter().map(|(_, len, _)| *len));
        if total <= quota_bytes {
            return;
        }
        // Oldest first.
        files.sort_by_key(|(_, _, mtime)| *mtime);
        for (path, len, _) in files {
            if total <= quota_bytes {
                break;
            }
            if std::fs::remove_file(&path).is_ok() {
                tracing::debug!("cache evicted {}", path.display());
                total = total.saturating_sub(len);
            }
        }
    }

    /// Delete everything (the settings "clear download cache" action).
    /// The root itself stays so in-flight paths remain valid.
    pub fn clear(&self) -> Result<(), CloudError> {
        for entry in std::fs::read_dir(&self.root)
            .into_iter()
            .flatten()
            .flatten()
        {
            let path = entry.path();
            let _ = if path.is_dir() {
                std::fs::remove_dir_all(&path)
            } else {
                std::fs::remove_file(&path)
            };
        }
        Ok(())
    }
}

fn saturating_total_bytes(lengths: impl IntoIterator<Item = u64>) -> u64 {
    lengths
        .into_iter()
        .fold(0, |total, len| total.saturating_add(len))
}

fn walk_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                out.push(path);
            }
        }
    }
    out
}

/// Bump a file's mtime to now (LRU touch). `File::set_times` needs no
/// extra dependency.
fn filetime_touch(path: &Path) -> std::io::Result<()> {
    let file = std::fs::OpenOptions::new().append(true).open(path)?;
    let now = std::fs::FileTimes::new().set_modified(SystemTime::now());
    file.set_times(now)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;

    fn write(cache: &DownloadCache, key: &str, len: usize) -> PathBuf {
        let path = cache.path_for(key).unwrap();
        std::fs::write(&path, vec![0u8; len]).unwrap();
        path
    }

    fn backdate(path: &Path) {
        let earlier = SystemTime::now() - std::time::Duration::from_secs(120);
        let file = std::fs::OpenOptions::new().append(true).open(path).unwrap();
        file.set_times(std::fs::FileTimes::new().set_modified(earlier))
            .unwrap();
    }

    #[test]
    fn quota_evicts_oldest_first() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DownloadCache::new(dir.path().to_path_buf(), 250);
        let old = write(&cache, "stock/old.bin", 100);
        // Ensure distinct mtimes even on coarse filesystems.
        backdate(&old);
        let new = write(&cache, "stock/new.bin", 200);

        cache.enforce_quota();
        assert!(!old.exists(), "oldest file evicted");
        assert!(new.exists(), "newest file kept");
        assert_eq!(cache.size_bytes(), 200);
    }

    #[test]
    fn quota_updates_apply_to_subsequent_enforcement() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DownloadCache::new(dir.path().to_path_buf(), 300);
        let old = write(&cache, "stock/old.bin", 100);
        backdate(&old);
        let new = write(&cache, "stock/new.bin", 200);

        assert_eq!(cache.quota_bytes(), 300);
        cache.enforce_quota();
        assert!(old.exists());
        assert!(new.exists());

        cache.set_quota_bytes(200);
        assert_eq!(cache.quota_bytes(), 200);
        cache.enforce_quota();
        assert!(!old.exists(), "updated quota evicts the oldest file");
        assert!(new.exists(), "updated quota does not over-evict");
        assert_eq!(cache.size_bytes(), 200);
    }

    #[test]
    fn quota_is_safe_to_read_and_update_from_concurrent_threads() {
        const WORKERS: usize = 4;
        const ROUNDS: u64 = 64;

        let dir = tempfile::tempdir().unwrap();
        let cache = Arc::new(DownloadCache::new(dir.path().to_path_buf(), 0));
        let phase = Arc::new(Barrier::new(WORKERS));
        let handles: Vec<_> = (0..WORKERS)
            .map(|worker| {
                let cache = Arc::clone(&cache);
                let phase = Arc::clone(&phase);
                thread::spawn(move || {
                    let mut observed_quotas = Vec::with_capacity(ROUNDS as usize);
                    for round in 0..ROUNDS {
                        phase.wait();
                        let first_quota = round * WORKERS as u64;
                        cache.set_quota_bytes(first_quota + worker as u64);
                        phase.wait();
                        observed_quotas.push(cache.quota_bytes());
                        phase.wait();
                    }
                    observed_quotas
                })
            })
            .collect();

        for handle in handles {
            for (round, observed) in handle.join().unwrap().into_iter().enumerate() {
                let first_quota = round as u64 * WORKERS as u64;
                let quotas_this_round = first_quota..first_quota + WORKERS as u64;
                assert!(
                    quotas_this_round.contains(&observed),
                    "observed quota {observed} outside {quotas_this_round:?}"
                );
            }
        }
    }

    #[test]
    fn total_size_saturates_instead_of_overflowing() {
        assert_eq!(saturating_total_bytes([u64::MAX - 1, 1, 1]), u64::MAX);
    }

    #[test]
    fn under_quota_evicts_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DownloadCache::new(dir.path().to_path_buf(), 1000);
        let a = write(&cache, "a.bin", 100);
        cache.enforce_quota();
        assert!(a.exists());
    }

    #[test]
    fn hit_returns_existing_and_misses_absent() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DownloadCache::new(dir.path().to_path_buf(), 1000);
        assert!(cache.hit("nope.bin").is_none());
        write(&cache, "yes.bin", 10);
        assert!(cache.hit("yes.bin").is_some());
    }

    #[test]
    fn clear_empties_but_keeps_root() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DownloadCache::new(dir.path().to_path_buf(), 1000);
        write(&cache, "stock/a.bin", 10);
        write(&cache, "bundles/b.bin", 10);
        cache.clear().unwrap();
        assert_eq!(cache.size_bytes(), 0);
        assert!(cache.root().exists());
    }
}
