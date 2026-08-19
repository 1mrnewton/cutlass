//! The quota-managed download cache.
//!
//! Template bundles, SFX, and packs land here before import.
//! Unbounded data-dir growth is a real complaint generator, so the cache
//! enforces a byte quota with LRU eviction (by modification time — files
//! get re-touched on use). Legacy projects may still reference downloaded
//! source files directly; those paths are explicitly protected until import
//! promotion migrates every project to durable media storage.
//!
//! Settings surfaces a "clear download cache" action backed by
//! [`DownloadCache::clear`].

use std::collections::HashSet;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

use crate::error::CloudError;

/// Default quota: 2 GiB of downloaded assets/bundles before eviction.
pub const DEFAULT_QUOTA_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// A directory with a byte budget.
pub struct DownloadCache {
    root: RwLock<PathBuf>,
    quota_bytes: AtomicU64,
    protected_paths: RwLock<HashSet<PathBuf>>,
    destructive_operations_blocked: AtomicBool,
    activity: Mutex<CacheActivity>,
    activity_changed: Condvar,
}

/// Exact accounting for an explicit download-cache clear.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DownloadCacheClear {
    /// Logical bytes removed from non-directory entries.
    pub removed_bytes: u64,
    /// Number of non-directory entries removed.
    pub removed_files: u64,
    /// Existing protected files retained in the cache.
    pub retained_protected_files: u64,
}

/// Why an exclusive download-cache root switch failed.
#[derive(Debug, thiserror::Error)]
pub enum DownloadCacheSwitchError<E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    /// Project media inventory currently blocks destructive cache operations.
    #[error("download cache root switching is blocked until project media inventory completes")]
    DestructiveOperationsBlocked,
    /// Another maintenance operation or an active lease outlived the wait.
    #[error("download cache root switching timed out while cache files were in use")]
    MaintenanceBusy,
    /// The destination was not absolute.
    #[error("download cache root destination must be absolute")]
    RelativeDestination,
    /// The destination was a filesystem root such as `/`.
    #[error("download cache root destination cannot be a filesystem root")]
    FilesystemRootDestination,
    /// The destination was the current cache root.
    #[error("download cache root destination is already active")]
    SameDestination,
    /// The destination itself was a symbolic link.
    #[error("download cache root destination cannot be a symbolic link")]
    SymlinkDestination,
    /// A non-symlink filesystem entry already occupied the destination.
    #[error("download cache root destination already exists")]
    ExistingDestination,
    /// The destination could not be inspected safely.
    #[error("could not inspect download cache root destination: {0}")]
    DestinationIo(#[source] io::Error),
    /// A protected path did not belong to the current root.
    #[error("a protected download cache path is outside the current root")]
    ProtectedPathOutsideRoot,
    /// The caller's filesystem switch operation failed.
    #[error("download cache root migration failed: {0}")]
    Migration(#[source] E),
}

/// A cache path protected from clear or eviction for the lease's lifetime.
///
/// Keep the lease alive while reading, downloading, verifying, or promoting
/// the path. Drop it before calling [`DownloadCache::enforce_quota`] or
/// [`DownloadCache::clear`].
pub struct DownloadCacheLease<'a> {
    cache: &'a DownloadCache,
    path: PathBuf,
}

impl DownloadCacheLease<'_> {
    /// Absolute path represented by this lease.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for DownloadCacheLease<'_> {
    fn drop(&mut self) {
        self.cache.finish_access();
    }
}

#[derive(Debug, Default)]
struct CacheActivity {
    active_leases: usize,
    maintenance: bool,
}

struct MaintenanceGuard<'a> {
    cache: &'a DownloadCache,
}

impl Drop for MaintenanceGuard<'_> {
    fn drop(&mut self) {
        let mut activity = lock_activity(&self.cache.activity);
        activity.maintenance = false;
        self.cache.activity_changed.notify_all();
    }
}

impl DownloadCache {
    pub fn new(root: PathBuf, quota_bytes: u64) -> Self {
        Self {
            root: RwLock::new(root),
            quota_bytes: AtomicU64::new(quota_bytes),
            protected_paths: RwLock::new(HashSet::new()),
            destructive_operations_blocked: AtomicBool::new(false),
            activity: Mutex::new(CacheActivity::default()),
            activity_changed: Condvar::new(),
        }
    }

    /// Snapshots the currently active cache root.
    ///
    /// A switch that follows this call does not alter the returned path.
    #[must_use]
    pub fn root(&self) -> PathBuf {
        read_root(&self.root).clone()
    }

    /// Exclusively migrate the cache to a new root and publish it on success.
    ///
    /// `destination` must be absolute, absent, different from the current
    /// root, and not a filesystem root or symbolic link. Existing leases are
    /// allowed to finish before `migrate` runs, and new leases wait until this
    /// method returns. The closure receives the old and new roots while cache
    /// maintenance and root access are exclusive. It must use those arguments
    /// instead of calling root-dependent methods on this cache.
    ///
    /// The in-memory root and every protected path are changed together only
    /// after `migrate` succeeds. A closure error leaves both unchanged.
    pub fn switch_root<D, F, E>(
        &self,
        destination: D,
        migrate: F,
    ) -> Result<(), DownloadCacheSwitchError<E>>
    where
        D: AsRef<Path>,
        F: FnOnce(&Path, &Path) -> Result<(), E>,
        E: std::error::Error + Send + Sync + 'static,
    {
        if self.destructive_operations_blocked() {
            return Err(DownloadCacheSwitchError::DestructiveOperationsBlocked);
        }
        let _maintenance = self
            .begin_maintenance(MAINTENANCE_WAIT_TIMEOUT)
            .map_err(|_| DownloadCacheSwitchError::MaintenanceBusy)?;
        if self.destructive_operations_blocked() {
            return Err(DownloadCacheSwitchError::DestructiveOperationsBlocked);
        }

        let destination = destination.as_ref();
        let mut root = write_root(&self.root);
        if !destination.is_absolute() {
            return Err(DownloadCacheSwitchError::RelativeDestination);
        }
        if is_filesystem_root(destination) {
            return Err(DownloadCacheSwitchError::FilesystemRootDestination);
        }
        if destination == root.as_path() {
            return Err(DownloadCacheSwitchError::SameDestination);
        }
        match std::fs::symlink_metadata(destination) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(DownloadCacheSwitchError::SymlinkDestination);
            }
            Ok(_) => return Err(DownloadCacheSwitchError::ExistingDestination),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(DownloadCacheSwitchError::DestinationIo(error)),
        }

        // Root readers are excluded before this lock is acquired, and every
        // protected-path writer takes the root lock first. Holding both locks
        // makes the remap and root publication one observable transition.
        let mut protected = write_protected(&self.protected_paths);
        let remapped = protected
            .iter()
            .map(|path| {
                let relative = path
                    .strip_prefix(root.as_path())
                    .map_err(|_| DownloadCacheSwitchError::ProtectedPathOutsideRoot)?;
                if relative.as_os_str().is_empty()
                    || !relative
                        .components()
                        .all(|component| matches!(component, Component::Normal(_)))
                {
                    return Err(DownloadCacheSwitchError::ProtectedPathOutsideRoot);
                }
                Ok(destination.join(relative))
            })
            .collect::<Result<HashSet<_>, DownloadCacheSwitchError<E>>>()?;

        migrate(root.as_path(), destination).map_err(DownloadCacheSwitchError::Migration)?;

        *protected = remapped;
        *root = destination.to_path_buf();
        // Publish the new root before protected-path readers resume. Any
        // operation that needs both always acquires root first.
        drop(root);
        drop(protected);
        Ok(())
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

    /// Keep a cache-owned source file available while a project references it.
    ///
    /// Missing paths may be protected so a later re-download cannot be
    /// evicted before the project is migrated. Paths outside this cache, the
    /// cache root itself, and paths containing traversal components fail
    /// closed.
    pub fn protect_path(&self, path: &Path) -> Result<(), CloudError> {
        let root = read_root(&self.root);
        let path = contained_cache_path(root.as_path(), path)?;
        if let Ok(metadata) = std::fs::symlink_metadata(&path)
            && !metadata.file_type().is_file()
        {
            return Err(invalid_cache_path());
        }
        write_protected(&self.protected_paths).insert(path);
        Ok(())
    }

    /// Number of project-referenced cache paths currently protected.
    #[must_use]
    pub fn protected_path_count(&self) -> usize {
        read_protected(&self.protected_paths).len()
    }

    /// Fail closed while project references have not been fully inventoried.
    ///
    /// Reads, writes, and explicit protection continue to work. Quota
    /// enforcement becomes a no-op and explicit clearing returns an error.
    pub fn block_destructive_operations(&self) {
        self.destructive_operations_blocked
            .store(true, Ordering::Release);
    }

    /// Re-enable clear and eviction after every project reference was scanned.
    pub fn allow_destructive_operations(&self) {
        self.destructive_operations_blocked
            .store(false, Ordering::Release);
    }

    /// Whether clear and quota eviction currently fail closed.
    #[must_use]
    pub fn destructive_operations_blocked(&self) -> bool {
        self.destructive_operations_blocked.load(Ordering::Acquire)
    }

    /// Reserve one safe cache path for a complete read or write operation.
    ///
    /// Concurrent leases are allowed. Clear and eviction wait for every lease
    /// to drop, while new leases wait for maintenance to finish.
    pub fn lease(&self, key: &str) -> Result<DownloadCacheLease<'_>, CloudError> {
        self.begin_access()?;
        match self.path_for(key) {
            Ok(path) => Ok(DownloadCacheLease { cache: self, path }),
            Err(error) => {
                self.finish_access();
                Err(error)
            }
        }
    }

    /// Where a download for `key` (e.g. `sfx/whoosh-01.mp3`)
    /// should land. Creates parent directories.
    pub fn path_for(&self, key: &str) -> Result<PathBuf, CloudError> {
        let key = validated_key(key)?;
        let root = read_root(&self.root);
        ensure_cache_root(root.as_path())?;
        let parent = key.parent().unwrap_or_else(|| Path::new(""));
        ensure_safe_subdirectories(root.as_path(), parent)?;
        let path = root.join(key);
        reject_symlink_or_directory(&path)?;
        Ok(path)
    }

    /// Whether `key` is already downloaded (touches it for LRU).
    pub fn hit(&self, key: &str) -> Option<PathBuf> {
        let key = validated_key(key).ok()?;
        let root = read_root(&self.root);
        if !safe_existing_root(root.as_path()) {
            return None;
        }
        ensure_existing_safe_subdirectories(
            root.as_path(),
            key.parent().unwrap_or_else(|| Path::new("")),
        )
        .ok()?;
        let path = root.join(key);
        let metadata = std::fs::symlink_metadata(&path).ok()?;
        if metadata.file_type().is_file() {
            // Re-touch so recently-used files survive eviction.
            let _ = filetime_touch(&path);
            Some(path)
        } else {
            None
        }
    }

    /// Total bytes currently cached.
    pub fn size_bytes(&self) -> u64 {
        let root = read_root(&self.root);
        saturating_total_bytes(
            walk_files(root.as_path())
                .into_iter()
                .filter_map(|p| std::fs::symlink_metadata(p).ok())
                .map(|m| m.len()),
        )
    }

    /// Evict least-recently-modified files until under quota. Called after
    /// each completed download; cheap when under budget.
    pub fn enforce_quota(&self) {
        if self.destructive_operations_blocked() {
            tracing::debug!("download cache eviction blocked pending project inventory");
            return;
        }
        let Ok(_maintenance) = self.begin_maintenance(MAINTENANCE_WAIT_TIMEOUT) else {
            tracing::debug!("download cache eviction skipped while files remain in use");
            return;
        };
        if self.destructive_operations_blocked() {
            return;
        }
        let quota_bytes = self.quota_bytes();
        let root = read_root(&self.root);
        let protected = read_protected(&self.protected_paths).clone();
        let mut files: Vec<(PathBuf, u64, SystemTime)> = walk_files(root.as_path())
            .into_iter()
            .filter(|path| !protected.contains(path))
            .filter_map(|p| {
                let meta = std::fs::symlink_metadata(&p).ok()?;
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
    pub fn clear(&self) -> Result<DownloadCacheClear, CloudError> {
        if self.destructive_operations_blocked() {
            return Err(cache_inventory_incomplete());
        }
        let _maintenance = self.begin_maintenance(MAINTENANCE_WAIT_TIMEOUT)?;
        if self.destructive_operations_blocked() {
            return Err(cache_inventory_incomplete());
        }
        let root = read_root(&self.root);
        if safe_root_entries(root.as_path())?.is_none() {
            return Ok(DownloadCacheClear {
                removed_bytes: 0,
                removed_files: 0,
                retained_protected_files: 0,
            });
        }
        let protected = read_protected(&self.protected_paths).clone();
        let (files, mut directories) = walk_tree(root.as_path());
        let mut removed_bytes = 0_u64;
        let mut removed_files = 0_u64;
        let mut retained_protected_files = 0_u64;
        for path in files {
            if protected.contains(&path) {
                retained_protected_files = retained_protected_files.saturating_add(1);
                continue;
            }
            let bytes = std::fs::symlink_metadata(&path)?.len();
            std::fs::remove_file(path)?;
            removed_bytes = removed_bytes.saturating_add(bytes);
            removed_files = removed_files.saturating_add(1);
        }
        directories.sort_by_key(|path| std::cmp::Reverse(path.components().count()));
        for path in directories {
            match std::fs::remove_dir(path) {
                Ok(()) => {}
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::DirectoryNotEmpty | io::ErrorKind::NotFound
                    ) => {}
                Err(error) => return Err(error.into()),
            }
        }
        Ok(DownloadCacheClear {
            removed_bytes,
            removed_files,
            retained_protected_files,
        })
    }

    fn begin_access(&self) -> Result<(), CloudError> {
        let mut activity = lock_activity(&self.activity);
        while activity.maintenance {
            activity = self
                .activity_changed
                .wait(activity)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        activity.active_leases = activity
            .active_leases
            .checked_add(1)
            .ok_or_else(cache_activity_exhausted)?;
        Ok(())
    }

    fn finish_access(&self) {
        let mut activity = lock_activity(&self.activity);
        activity.active_leases = activity.active_leases.saturating_sub(1);
        if activity.active_leases == 0 {
            self.activity_changed.notify_all();
        }
    }

    fn begin_maintenance(&self, timeout: Duration) -> Result<MaintenanceGuard<'_>, CloudError> {
        let deadline = Instant::now() + timeout;
        let mut activity = lock_activity(&self.activity);
        while activity.maintenance || activity.active_leases != 0 {
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                return Err(cache_busy());
            };
            let (next, result) = self
                .activity_changed
                .wait_timeout(activity, remaining)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            activity = next;
            if result.timed_out() && (activity.maintenance || activity.active_leases != 0) {
                return Err(cache_busy());
            }
        }
        activity.maintenance = true;
        Ok(MaintenanceGuard { cache: self })
    }
}

const MAX_CACHE_KEY_BYTES: usize = 1_024;
const MAINTENANCE_WAIT_TIMEOUT: Duration = Duration::from_secs(30);

fn validated_key(key: &str) -> Result<&Path, CloudError> {
    let path = Path::new(key);
    let valid = !key.is_empty()
        && key.len() <= MAX_CACHE_KEY_BYTES
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)));
    if valid {
        Ok(path)
    } else {
        Err(CloudError::Protocol(
            "invalid download cache key".to_owned(),
        ))
    }
}

fn contained_cache_path(root: &Path, path: &Path) -> Result<PathBuf, CloudError> {
    if !root.is_absolute() || !path.is_absolute() {
        return Err(invalid_cache_path());
    }
    let relative = path.strip_prefix(root).map_err(|_| invalid_cache_path())?;
    if relative.as_os_str().is_empty()
        || !relative
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
    {
        return Err(invalid_cache_path());
    }
    Ok(root.join(relative))
}

fn read_root(root: &RwLock<PathBuf>) -> std::sync::RwLockReadGuard<'_, PathBuf> {
    root.read().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn write_root(root: &RwLock<PathBuf>) -> std::sync::RwLockWriteGuard<'_, PathBuf> {
    root.write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn read_protected(
    protected: &RwLock<HashSet<PathBuf>>,
) -> std::sync::RwLockReadGuard<'_, HashSet<PathBuf>> {
    protected
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn write_protected(
    protected: &RwLock<HashSet<PathBuf>>,
) -> std::sync::RwLockWriteGuard<'_, HashSet<PathBuf>> {
    protected
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn lock_activity(activity: &Mutex<CacheActivity>) -> std::sync::MutexGuard<'_, CacheActivity> {
    activity
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn cache_activity_exhausted() -> CloudError {
    io::Error::other("download cache has too many active operations").into()
}

fn cache_busy() -> CloudError {
    io::Error::new(
        io::ErrorKind::TimedOut,
        "download cache files are still in use",
    )
    .into()
}

fn cache_inventory_incomplete() -> CloudError {
    io::Error::new(
        io::ErrorKind::PermissionDenied,
        "download cache is protected until project media inventory completes",
    )
    .into()
}

fn is_filesystem_root(path: &Path) -> bool {
    path.is_absolute() && path.parent().is_none()
}

fn ensure_cache_root(root: &Path) -> Result<(), CloudError> {
    if !root.is_absolute() {
        return Err(invalid_cache_root());
    }
    match std::fs::symlink_metadata(root) {
        Ok(metadata) if metadata.file_type().is_dir() => Ok(()),
        Ok(_) => Err(invalid_cache_root()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            std::fs::create_dir_all(root)?;
            let metadata = std::fs::symlink_metadata(root)?;
            if metadata.file_type().is_dir() {
                Ok(())
            } else {
                Err(invalid_cache_root())
            }
        }
        Err(error) => Err(error.into()),
    }
}

fn safe_existing_root(root: &Path) -> bool {
    root.is_absolute()
        && std::fs::symlink_metadata(root).is_ok_and(|metadata| metadata.file_type().is_dir())
}

fn ensure_safe_subdirectories(root: &Path, relative: &Path) -> Result<(), CloudError> {
    let mut current = root.to_path_buf();
    for component in relative.components() {
        let Component::Normal(name) = component else {
            return Err(invalid_cache_path());
        };
        current.push(name);
        match std::fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_dir() => {}
            Ok(_) => return Err(invalid_cache_path()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                std::fs::create_dir(&current)?;
                let metadata = std::fs::symlink_metadata(&current)?;
                if !metadata.file_type().is_dir() {
                    return Err(invalid_cache_path());
                }
            }
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn ensure_existing_safe_subdirectories(root: &Path, relative: &Path) -> Result<(), ()> {
    let mut current = root.to_path_buf();
    for component in relative.components() {
        let Component::Normal(name) = component else {
            return Err(());
        };
        current.push(name);
        let metadata = std::fs::symlink_metadata(&current).map_err(|_| ())?;
        if !metadata.file_type().is_dir() {
            return Err(());
        }
    }
    Ok(())
}

fn reject_symlink_or_directory(path: &Path) -> Result<(), CloudError> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_file() => Ok(()),
        Ok(_) => Err(invalid_cache_path()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn safe_root_entries(root: &Path) -> Result<Option<std::fs::ReadDir>, CloudError> {
    if !root.is_absolute() {
        return Err(invalid_cache_root());
    }
    match std::fs::symlink_metadata(root) {
        Ok(metadata) if metadata.file_type().is_dir() => {
            std::fs::read_dir(root).map(Some).map_err(Into::into)
        }
        Ok(_) => Err(invalid_cache_root()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn invalid_cache_root() -> CloudError {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "download cache root must be an absolute, real directory",
    )
    .into()
}

fn invalid_cache_path() -> CloudError {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "download cache path crosses an unsafe filesystem entry",
    )
    .into()
}

fn saturating_total_bytes(lengths: impl IntoIterator<Item = u64>) -> u64 {
    lengths
        .into_iter()
        .fold(0, |total, len| total.saturating_add(len))
}

fn walk_files(root: &Path) -> Vec<PathBuf> {
    walk_tree(root).0
}

fn walk_tree(root: &Path) -> (Vec<PathBuf>, Vec<PathBuf>) {
    if !safe_existing_root(root) {
        return (Vec::new(), Vec::new());
    }
    let mut files = Vec::new();
    let mut directories = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                directories.push(path.clone());
                stack.push(path);
            } else {
                files.push(path);
            }
        }
    }
    (files, directories)
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
    use std::sync::atomic::{AtomicU8, AtomicUsize};
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
        let old = write(&cache, "sfx/old.bin", 100);
        // Ensure distinct mtimes even on coarse filesystems.
        backdate(&old);
        let new = write(&cache, "sfx/new.bin", 200);

        cache.enforce_quota();
        assert!(!old.exists(), "oldest file evicted");
        assert!(new.exists(), "newest file kept");
        assert_eq!(cache.size_bytes(), 200);
    }

    #[test]
    fn quota_updates_apply_to_subsequent_enforcement() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DownloadCache::new(dir.path().to_path_buf(), 300);
        let old = write(&cache, "sfx/old.bin", 100);
        backdate(&old);
        let new = write(&cache, "sfx/new.bin", 200);

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
    fn protected_project_media_survives_eviction_and_explicit_clear() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DownloadCache::new(dir.path().to_path_buf(), 1);
        let protected = write(&cache, "sfx/project.mp4", 100);
        backdate(&protected);
        let disposable = write(&cache, "sfx/thumbnail.img", 20);
        cache.protect_path(&protected).unwrap();

        assert_eq!(cache.protected_path_count(), 1);
        cache.enforce_quota();
        assert!(protected.exists());
        assert!(!disposable.exists());

        let another = write(&cache, "ai/unused.png", 12);
        let report = cache.clear().unwrap();
        assert_eq!(report.removed_bytes, 12);
        assert_eq!(report.removed_files, 1);
        assert_eq!(report.retained_protected_files, 1);
        assert!(protected.exists());
        assert!(!another.exists());
    }

    #[test]
    fn missing_project_media_can_be_protected_before_redownload() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DownloadCache::new(dir.path().to_path_buf(), 0);
        let future = dir.path().join("sfx/future.mp4");
        cache.protect_path(&future).unwrap();
        assert_eq!(cache.protected_path_count(), 1);

        write(&cache, "sfx/future.mp4", 8);
        cache.enforce_quota();
        assert!(future.exists());
        let report = cache.clear().unwrap();
        assert_eq!(report.retained_protected_files, 1);
        assert!(future.exists());
    }

    #[test]
    fn protected_paths_must_be_contained_cache_files() {
        let dir = tempfile::tempdir().unwrap();
        let cache_root = dir.path().join("cache");
        let cache = DownloadCache::new(cache_root.clone(), 100);
        std::fs::create_dir_all(&cache_root).unwrap();

        for path in [
            dir.path().join("outside.mp4"),
            cache_root.clone(),
            cache_root.join("../outside.mp4"),
            PathBuf::from("relative.mp4"),
        ] {
            assert!(cache.protect_path(&path).is_err(), "{path:?}");
        }

        let directory = cache_root.join("directory");
        std::fs::create_dir(&directory).unwrap();
        assert!(cache.protect_path(&directory).is_err());
        assert_eq!(cache.protected_path_count(), 0);
    }

    #[test]
    fn switch_root_moves_cache_and_atomically_remaps_protected_paths() {
        let dir = tempfile::tempdir().unwrap();
        let old_root = dir.path().join("old-cache");
        let new_root = dir.path().join("new-cache");
        let cache = DownloadCache::new(old_root.clone(), 1_000);
        let protected = write(&cache, "sfx/project.mp4", 100);
        let disposable = write(&cache, "sfx/disposable.bin", 20);
        cache.protect_path(&protected).unwrap();

        cache
            .switch_root(&new_root, |old, new| {
                assert_eq!(old, old_root);
                assert_eq!(new, new_root);
                std::fs::rename(old, new)
            })
            .unwrap();

        let moved_protected = new_root.join("sfx/project.mp4");
        let moved_disposable = new_root.join("sfx/disposable.bin");
        assert_eq!(cache.root(), new_root);
        assert!(!old_root.exists());
        assert!(moved_protected.exists());
        assert!(moved_disposable.exists());
        assert_eq!(
            &*read_protected(&cache.protected_paths),
            &HashSet::from([moved_protected.clone()])
        );

        cache.set_quota_bytes(0);
        cache.enforce_quota();
        assert!(
            moved_protected.exists(),
            "the remapped protection must survive eviction"
        );
        assert!(!moved_disposable.exists());
        assert!(!disposable.exists());
    }

    #[test]
    fn switch_root_closure_failure_rolls_back_all_memory_state() {
        let dir = tempfile::tempdir().unwrap();
        let old_root = dir.path().join("old-cache");
        let new_root = dir.path().join("new-cache");
        let cache = DownloadCache::new(old_root.clone(), 1_000);
        let protected = write(&cache, "sfx/project.mp4", 8);
        cache.protect_path(&protected).unwrap();
        let protected_before = read_protected(&cache.protected_paths).clone();

        let result = cache.switch_root(&new_root, |old, new| {
            assert_eq!(old, old_root);
            assert_eq!(new, new_root);
            std::fs::create_dir(new)?;
            Err(io::Error::other("injected migration failure"))
        });

        assert!(matches!(
            result,
            Err(DownloadCacheSwitchError::Migration(error))
                if error.kind() == io::ErrorKind::Other
        ));
        assert_eq!(cache.root(), old_root);
        assert_eq!(&*read_protected(&cache.protected_paths), &protected_before);
        assert_eq!(
            cache.path_for("sfx/after-failure.bin").unwrap(),
            old_root.join("sfx/after-failure.bin")
        );
    }

    #[test]
    fn active_lease_delays_switch_root_and_keeps_its_old_path_stable() {
        let dir = tempfile::tempdir().unwrap();
        let old_root = dir.path().join("old-cache");
        let new_root = dir.path().join("new-cache");
        let cache = Arc::new(DownloadCache::new(old_root.clone(), 1_000));
        let lease = cache.lease("sfx/in-flight.mp4").unwrap();
        let lease_path = lease.path().to_path_buf();
        let started = Arc::new(Barrier::new(2));
        let closure_called = Arc::new(AtomicBool::new(false));

        let worker = {
            let cache = Arc::clone(&cache);
            let started = Arc::clone(&started);
            let closure_called = Arc::clone(&closure_called);
            let new_root = new_root.clone();
            thread::spawn(move || {
                started.wait();
                cache.switch_root(&new_root, |old, new| {
                    closure_called.store(true, Ordering::Release);
                    std::fs::rename(old, new)
                })
            })
        };

        started.wait();
        thread::sleep(Duration::from_millis(20));
        assert!(!closure_called.load(Ordering::Acquire));
        assert_eq!(cache.root(), old_root);
        assert_eq!(lease.path(), lease_path);

        drop(lease);
        worker.join().unwrap().unwrap();
        assert!(closure_called.load(Ordering::Acquire));
        assert_eq!(cache.root(), new_root);
        assert_eq!(lease_path, old_root.join("sfx/in-flight.mp4"));
    }

    #[test]
    fn switch_root_rejects_invalid_destinations_without_calling_closure() {
        let dir = tempfile::tempdir().unwrap();
        let old_root = dir.path().join("old-cache");
        std::fs::create_dir(&old_root).unwrap();
        let cache = DownloadCache::new(old_root.clone(), 1_000);
        let calls = AtomicUsize::new(0);
        let migrate = |_: &Path, _: &Path| {
            calls.fetch_add(1, Ordering::Relaxed);
            Ok::<(), io::Error>(())
        };

        assert!(matches!(
            cache.switch_root("relative-cache", migrate),
            Err(DownloadCacheSwitchError::RelativeDestination)
        ));
        let filesystem_root = old_root.ancestors().last().unwrap().to_path_buf();
        assert!(matches!(
            cache.switch_root(&filesystem_root, migrate),
            Err(DownloadCacheSwitchError::FilesystemRootDestination)
        ));
        assert!(matches!(
            cache.switch_root(&old_root, migrate),
            Err(DownloadCacheSwitchError::SameDestination)
        ));

        let existing_directory = dir.path().join("existing-directory");
        std::fs::create_dir(&existing_directory).unwrap();
        assert!(matches!(
            cache.switch_root(&existing_directory, migrate),
            Err(DownloadCacheSwitchError::ExistingDestination)
        ));
        let existing_file = dir.path().join("existing-file");
        std::fs::write(&existing_file, b"occupied").unwrap();
        assert!(matches!(
            cache.switch_root(&existing_file, migrate),
            Err(DownloadCacheSwitchError::ExistingDestination)
        ));

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;

            let symlink_destination = dir.path().join("destination-link");
            symlink(dir.path().join("missing-target"), &symlink_destination).unwrap();
            assert!(matches!(
                cache.switch_root(&symlink_destination, migrate),
                Err(DownloadCacheSwitchError::SymlinkDestination)
            ));
        }

        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_eq!(cache.root(), old_root);
    }

    #[test]
    fn blocked_destructive_operations_prevent_switch_root() {
        let dir = tempfile::tempdir().unwrap();
        let old_root = dir.path().join("old-cache");
        std::fs::create_dir(&old_root).unwrap();
        let cache = DownloadCache::new(old_root.clone(), 1_000);
        let closure_called = AtomicBool::new(false);
        cache.block_destructive_operations();

        let result = cache.switch_root(dir.path().join("new-cache"), |_, _| {
            closure_called.store(true, Ordering::Release);
            Ok::<(), io::Error>(())
        });

        assert!(matches!(
            result,
            Err(DownloadCacheSwitchError::DestructiveOperationsBlocked)
        ));
        assert!(!closure_called.load(Ordering::Acquire));
        assert_eq!(cache.root(), old_root);
    }

    #[test]
    fn concurrent_root_readers_observe_only_complete_old_or_new_paths() {
        const READERS: usize = 8;

        let dir = tempfile::tempdir().unwrap();
        let old_root = dir.path().join("old-cache-with-a-distinct-name");
        let new_root = dir.path().join("new-cache-with-another-distinct-name");
        std::fs::create_dir(&old_root).unwrap();
        let cache = Arc::new(DownloadCache::new(old_root.clone(), 1_000));
        let start = Arc::new(Barrier::new(READERS + 1));
        let stop = Arc::new(AtomicBool::new(false));
        let observed = Arc::new(AtomicU8::new(0));

        let readers: Vec<_> = (0..READERS)
            .map(|_| {
                let cache = Arc::clone(&cache);
                let start = Arc::clone(&start);
                let stop = Arc::clone(&stop);
                let observed = Arc::clone(&observed);
                let old_root = old_root.clone();
                let new_root = new_root.clone();
                thread::spawn(move || {
                    start.wait();
                    while !stop.load(Ordering::Acquire) {
                        let root = cache.root();
                        if root == old_root {
                            observed.fetch_or(0b01, Ordering::Release);
                        } else if root == new_root {
                            observed.fetch_or(0b10, Ordering::Release);
                        } else {
                            panic!("observed a partial or unknown root: {root:?}");
                        }
                        thread::yield_now();
                    }
                })
            })
            .collect();

        start.wait();
        let deadline = Instant::now() + Duration::from_secs(1);
        while observed.load(Ordering::Acquire) & 0b01 == 0 {
            assert!(Instant::now() < deadline, "readers never observed old root");
            thread::yield_now();
        }

        cache
            .switch_root(&new_root, |old, new| {
                std::fs::rename(old, new)?;
                thread::sleep(Duration::from_millis(10));
                Ok::<(), io::Error>(())
            })
            .unwrap();

        let deadline = Instant::now() + Duration::from_secs(1);
        while observed.load(Ordering::Acquire) & 0b10 == 0 {
            assert!(Instant::now() < deadline, "readers never observed new root");
            thread::yield_now();
        }
        stop.store(true, Ordering::Release);
        for reader in readers {
            reader.join().unwrap();
        }
        assert_eq!(observed.load(Ordering::Acquire), 0b11);
    }

    #[test]
    fn root_snapshots_and_switches_recover_from_lock_poisoning() {
        let dir = tempfile::tempdir().unwrap();
        let old_root = dir.path().join("old-cache");
        let new_root = dir.path().join("new-cache");
        std::fs::create_dir(&old_root).unwrap();
        let cache = DownloadCache::new(old_root.clone(), 1_000);

        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _root = cache.root.write().unwrap();
            panic!("poison root lock");
        }));
        assert!(poisoned.is_err());
        assert_eq!(cache.root(), old_root);

        cache
            .switch_root(&new_root, |old, new| std::fs::rename(old, new))
            .unwrap();
        assert_eq!(cache.root(), new_root);
    }

    #[test]
    fn leases_exclude_maintenance_until_they_drop() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DownloadCache::new(dir.path().to_path_buf(), 100);
        let lease = cache.lease("sfx/in-flight.mp4").unwrap();
        assert_eq!(lease.path(), dir.path().join("sfx/in-flight.mp4"));
        assert!(cache.begin_maintenance(Duration::ZERO).is_err());

        drop(lease);
        assert!(cache.begin_maintenance(Duration::ZERO).is_ok());
    }

    #[test]
    fn maintenance_excludes_new_leases_until_it_drops() {
        use std::sync::atomic::AtomicBool;

        let dir = tempfile::tempdir().unwrap();
        let cache = DownloadCache::new(dir.path().to_path_buf(), 100);
        let maintenance = cache.begin_maintenance(Duration::ZERO).unwrap();
        let acquired = AtomicBool::new(false);

        thread::scope(|scope| {
            let worker = scope.spawn(|| {
                let _lease = cache.lease("sfx/after-clear.mp4").unwrap();
                acquired.store(true, Ordering::Release);
            });
            thread::sleep(Duration::from_millis(20));
            assert!(!acquired.load(Ordering::Acquire));
            drop(maintenance);
            worker.join().unwrap();
        });
        assert!(acquired.load(Ordering::Acquire));
    }

    #[test]
    fn a_rejected_lease_does_not_leak_activity() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DownloadCache::new(dir.path().to_path_buf(), 100);
        assert!(cache.lease("../outside.mp4").is_err());
        assert!(cache.begin_maintenance(Duration::ZERO).is_ok());
    }

    #[test]
    fn destructive_operations_fail_closed_during_project_inventory() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DownloadCache::new(dir.path().to_path_buf(), 0);
        let path = write(&cache, "sfx/unclassified.mp4", 8);

        cache.block_destructive_operations();
        assert!(cache.destructive_operations_blocked());
        cache.enforce_quota();
        assert!(path.exists());
        assert!(cache.clear().is_err());
        assert!(path.exists());

        cache.allow_destructive_operations();
        assert!(!cache.destructive_operations_blocked());
        cache.enforce_quota();
        assert!(!path.exists());
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
    fn cache_keys_cannot_escape_or_replace_directories() {
        let dir = tempfile::tempdir().unwrap();
        let cache_root = dir.path().join("cache");
        let cache = DownloadCache::new(cache_root.clone(), 1_000);

        for key in [
            "",
            ".",
            "..",
            "../escaped.bin",
            "sfx/../../escaped.bin",
            "/absolute.bin",
        ] {
            assert!(cache.path_for(key).is_err(), "{key:?} must be rejected");
            assert!(cache.hit(key).is_none(), "{key:?} cannot be a cache hit");
        }
        assert!(!dir.path().join("escaped.bin").exists());

        std::fs::create_dir_all(cache_root.join("occupied")).unwrap();
        assert!(cache.path_for("occupied").is_err());
    }

    #[test]
    fn relative_cache_roots_fail_closed() {
        let cache = DownloadCache::new(PathBuf::from("relative-cache"), 1_000);
        assert!(cache.path_for("asset.bin").is_err());
        assert!(cache.hit("asset.bin").is_none());
        assert_eq!(cache.size_bytes(), 0);
        assert!(cache.clear().is_err());
    }

    #[cfg(unix)]
    #[test]
    fn nested_symlinks_are_never_followed_or_cleared_through() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let cache_root = dir.path().join("cache");
        std::fs::create_dir(&cache_root).unwrap();
        let protected = outside.path().join("protected.bin");
        std::fs::write(&protected, b"keep").unwrap();
        symlink(outside.path(), cache_root.join("linked")).unwrap();

        let cache = DownloadCache::new(cache_root.clone(), 0);
        assert!(cache.path_for("linked/new.bin").is_err());
        assert!(cache.hit("linked/protected.bin").is_none());

        cache.enforce_quota();
        assert!(protected.exists(), "eviction must not traverse the symlink");

        if !cache_root.join("linked").exists() {
            symlink(outside.path(), cache_root.join("linked")).unwrap();
        }
        cache.clear().unwrap();
        assert!(protected.exists(), "clearing must not traverse the symlink");
        assert!(!cache_root.join("linked").exists());
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_cache_root_fails_closed() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let protected = outside.path().join("protected.bin");
        std::fs::write(&protected, b"keep").unwrap();
        let cache_root = dir.path().join("cache-link");
        symlink(outside.path(), &cache_root).unwrap();
        let cache = DownloadCache::new(cache_root, 0);

        assert!(cache.path_for("new.bin").is_err());
        assert!(cache.hit("protected.bin").is_none());
        assert_eq!(cache.size_bytes(), 0);
        cache.enforce_quota();
        assert!(cache.clear().is_err());
        assert!(protected.exists());
    }

    #[test]
    fn clear_empties_but_keeps_root() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DownloadCache::new(dir.path().to_path_buf(), 1000);
        write(&cache, "sfx/a.bin", 10);
        write(&cache, "bundles/b.bin", 10);
        cache.clear().unwrap();
        assert_eq!(cache.size_bytes(), 0);
        assert!(cache.root().exists());
    }
}
