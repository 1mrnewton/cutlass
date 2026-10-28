//! Bundled resources and the app cache.
//!
//! A bundled file is read from the resources directory. A downloaded file is
//! read from the cache directory. The index records the id, the location, the
//! size, the checksum, and whether the bytes are still partial. Eviction drops
//! cached previews, oldest first, and leaves bundled packages in place. A file
//! the panel or playback is showing is pinned and is not evicted.

#![forbid(unsafe_code)]

mod bundle;
mod cache;
mod index;
mod quota;

use std::fmt;
use std::path::{Path, PathBuf};

use cutlass_download::Checksum;

pub use index::{Place, Record, ResourceId};
pub use quota::{Quota, Shortfall};

use index::Index;

/// Why a cache or bundle operation was refused.
#[derive(Debug)]
pub enum StorageError {
    /// The id is empty.
    EmptyId,
    /// The id is not a single file name.
    InvalidId { id: String },
    /// The path is not inside the bundle or the cache.
    LeavesRoot { path: PathBuf },
    /// A symlink is not followed.
    Symlink { path: PathBuf },
    /// The path is not a regular file.
    NotAFile { path: PathBuf },
    /// The bundle or the cache is not a directory.
    NotADirectory { path: PathBuf },
    /// The bundle and the cache are the same directory.
    SameRoot { path: PathBuf },
    /// No complete file has this id.
    Missing { id: ResourceId },
    /// The bytes are still arriving, so there is no published path.
    StillPartial { id: ResourceId },
    /// The bytes are not the checksum.
    ChecksumMismatch { id: ResourceId },
    /// This id is already indexed as a different file.
    AlreadyStored { id: ResourceId },
    /// The index is not this build's index.
    IndexVersion { found: u32 },
    /// The index could not be read.
    Index { path: PathBuf, message: String },
    /// The filesystem refused the read or the write.
    Io { path: PathBuf, message: String },
}

impl fmt::Display for StorageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyId => formatter.write_str("a resource id is empty"),
            Self::InvalidId { id } => write!(formatter, "{id} is not a resource id"),
            Self::LeavesRoot { path } => {
                write!(formatter, "{} leaves its directory", path.display())
            }
            Self::Symlink { path } => {
                write!(formatter, "{} is a symlink", path.display())
            }
            Self::NotAFile { path } => {
                write!(formatter, "{} is not a file", path.display())
            }
            Self::NotADirectory { path } => {
                write!(formatter, "{} is not a directory", path.display())
            }
            Self::SameRoot { path } => {
                write!(
                    formatter,
                    "{} cannot be both the bundle and the cache",
                    path.display()
                )
            }
            Self::Missing { id } => write!(formatter, "{id} is not in the cache"),
            Self::StillPartial { id } => write!(formatter, "{id} is still partial"),
            Self::ChecksumMismatch { id } => {
                write!(formatter, "{id} does not match its checksum")
            }
            Self::AlreadyStored { id } => write!(formatter, "{id} is already indexed"),
            Self::IndexVersion { found } => {
                write!(
                    formatter,
                    "index version {found} is not version {}",
                    index::VERSION
                )
            }
            Self::Index { path, message } => {
                write!(formatter, "could not read {}: {message}", path.display())
            }
            Self::Io { path, message } => {
                write!(formatter, "could not access {}: {message}", path.display())
            }
        }
    }
}

impl std::error::Error for StorageError {}

/// The resources directory and the cache directory.
#[derive(Debug)]
pub struct Storage {
    bundle_root: PathBuf,
    cache_root: PathBuf,
    quota: Quota,
    index: Index,
}

impl Storage {
    /// Open the bundle read-only and the cache. A missing index starts empty.
    pub fn open(bundle: &Path, cache: &Path, quota: Quota) -> Result<Self, StorageError> {
        let bundle_root = cache::directory(bundle)?;
        if let Ok(existing) = cache::directory(cache)
            && existing == bundle_root
        {
            return Err(StorageError::SameRoot { path: existing });
        }
        let cache_root = cache::ensure_cache(cache)?;
        if cache_root == bundle_root {
            return Err(StorageError::SameRoot { path: cache_root });
        }
        let index = cache::load(&bundle_root, &cache_root)?;
        let storage = Self {
            bundle_root,
            cache_root,
            quota,
            index,
        };
        if !storage.cache_root.join("index.json").exists() {
            cache::persist(&storage.cache_root, &storage.index)?;
        }
        Ok(storage)
    }

    /// Index one file that already lives in the bundle. The package is not written.
    pub fn mount_bundle(
        &mut self,
        id: &str,
        relative: &Path,
        checksum: Checksum,
    ) -> Result<(), StorageError> {
        let id = ResourceId::new(id)?;
        let saved = self.index.clone();
        let inserted = bundle::mount(&mut self.index, &self.bundle_root, &id, relative, checksum)?;
        if inserted && let Err(error) = cache::persist(&self.cache_root, &self.index) {
            self.index = saved;
            return Err(error);
        }
        Ok(())
    }

    /// The published file for `id`. A partial file has no published path.
    /// This does not read the other ids and it does not hash `id` again.
    pub fn path(&self, id: &ResourceId) -> Result<PathBuf, StorageError> {
        let Some(record) = self.index.get(id) else {
            return Err(StorageError::Missing { id: id.clone() });
        };
        if record.is_partial() {
            return Err(StorageError::StillPartial { id: id.clone() });
        }
        let root = match record.place() {
            Place::Bundle => &self.bundle_root,
            Place::Cache => &self.cache_root,
        };
        match cache::resolve_file(root, record.relative(), true) {
            Ok(path) => Ok(path),
            Err(StorageError::Io { path, .. }) if !path.exists() => {
                Err(StorageError::Missing { id: id.clone() })
            }
            Err(error) => Err(error),
        }
    }

    #[must_use]
    pub fn record(&self, id: &ResourceId) -> Option<&Record> {
        self.index.get(id)
    }

    /// Bytes in the cache. Bundled files are not counted.
    #[must_use]
    pub fn used_bytes(&self) -> u64 {
        self.index.used_bytes()
    }

    #[must_use]
    pub fn limit(&self) -> u64 {
        self.quota.limit()
    }

    /// Cached previews that would be dropped so `incoming` bytes fit.
    /// Nothing is deleted.
    pub fn plan(&self, incoming: u64) -> Result<Vec<ResourceId>, Shortfall> {
        quota::select(
            &self.entries(),
            self.used_bytes(),
            incoming,
            self.quota.limit(),
        )
    }

    fn entries(&self) -> Vec<quota::Entry> {
        self.index
            .records()
            .map(|record| quota::Entry {
                id: record.id().clone(),
                size: record.size(),
                age: record.age(),
                evictable: record.place() == Place::Cache && record.is_preview(),
            })
            .collect()
    }
}

#[cfg(test)]
pub(crate) struct Temp {
    path: PathBuf,
}

#[cfg(test)]
impl Temp {
    pub(crate) fn new(label: &str) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "cutlass-storage-{label}-{}-{n}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use cutlass_download::Checksum;

    use super::{Place, Quota, ResourceId, Storage, StorageError, Temp};

    struct Roots {
        _temp: Temp,
        bundle: PathBuf,
        cache: PathBuf,
        project: PathBuf,
    }

    impl Roots {
        fn new(label: &str) -> Self {
            let temp = Temp::new(label);
            let bundle = temp.path().join("bundle");
            let cache = temp.path().join("cache");
            let project = temp.path().join("project");
            fs::create_dir(&bundle).unwrap();
            fs::create_dir(&cache).unwrap();
            fs::create_dir(&project).unwrap();
            fs::write(project.join("secret"), b"do-not-touch").unwrap();
            Self {
                _temp: temp,
                bundle,
                cache,
                project,
            }
        }

        fn secret(&self) -> Vec<u8> {
            fs::read(self.project.join("secret")).unwrap()
        }

        fn index_bytes(&self) -> Vec<u8> {
            fs::read(self.cache.join("index.json")).unwrap()
        }
    }

    fn open(roots: &Roots, limit: u64) -> Storage {
        Storage::open(&roots.bundle, &roots.cache, Quota::bytes(limit)).unwrap()
    }

    #[test]
    fn an_empty_cache_has_no_path() {
        let roots = Roots::new("empty");
        let storage = open(&roots, 0);
        let missing = storage.path(&ResourceId::new("glow").unwrap()).unwrap_err();
        assert!(matches!(missing, StorageError::Missing { .. }));
        assert_eq!(storage.used_bytes(), 0);
        assert_eq!(storage.limit(), 0);
        assert!(storage.plan(0).unwrap().is_empty());
        assert_eq!(roots.secret(), b"do-not-touch");
    }

    #[test]
    fn a_bundled_file_is_indexed_and_not_counted_or_copied() {
        let roots = Roots::new("bundle");
        fs::create_dir(roots.bundle.join("effects")).unwrap();
        let file = roots.bundle.join("effects").join("glow.bin");
        fs::write(&file, b"glow").unwrap();
        let file = fs::canonicalize(&file).unwrap();
        let before = fs::read(&file).unwrap();
        let mut storage = open(&roots, 0);
        storage
            .mount_bundle("glow", Path::new("effects/glow.bin"), Checksum::of(b"glow"))
            .unwrap();
        assert_eq!(
            storage.path(&ResourceId::new("glow").unwrap()).unwrap(),
            file
        );
        let record = storage.record(&ResourceId::new("glow").unwrap()).unwrap();
        assert_eq!(record.place(), Place::Bundle);
        assert_eq!(record.size(), 4);
        assert_eq!(record.checksum(), Some(Checksum::of(b"glow")));
        assert!(!record.is_preview());
        assert_eq!(storage.used_bytes(), 0);
        assert_eq!(fs::read(&file).unwrap(), before);
        assert!(storage.plan(1).is_err());
        assert_eq!(fs::read(&file).unwrap(), before);
        storage
            .mount_bundle("glow", Path::new("effects/glow.bin"), Checksum::of(b"glow"))
            .unwrap();
        let again = Storage::open(&roots.bundle, &roots.cache, Quota::bytes(0)).unwrap();
        assert_eq!(again.path(&ResourceId::new("glow").unwrap()).unwrap(), file);
    }

    #[test]
    fn a_checksum_mismatch_does_not_index_the_bundle() {
        let roots = Roots::new("bad-sum");
        let file = roots.bundle.join("glow.bin");
        fs::write(&file, b"glow").unwrap();
        let mut storage = open(&roots, 10);
        let before = roots.index_bytes();
        let error = storage
            .mount_bundle("glow", Path::new("glow.bin"), Checksum::of(b"nope"))
            .unwrap_err();
        assert!(matches!(error, StorageError::ChecksumMismatch { .. }));
        assert!(storage.record(&ResourceId::new("glow").unwrap()).is_none());
        assert_eq!(roots.index_bytes(), before);
        assert_eq!(fs::read(&file).unwrap(), b"glow");
    }

    #[test]
    fn a_second_bundle_file_does_not_replace_the_first() {
        let roots = Roots::new("replace");
        fs::write(roots.bundle.join("a.bin"), b"aaa").unwrap();
        fs::write(roots.bundle.join("b.bin"), b"bbbb").unwrap();
        let mut storage = open(&roots, 10);
        storage
            .mount_bundle("glow", Path::new("a.bin"), Checksum::of(b"aaa"))
            .unwrap();
        let error = storage
            .mount_bundle("glow", Path::new("b.bin"), Checksum::of(b"bbbb"))
            .unwrap_err();
        assert!(matches!(error, StorageError::AlreadyStored { .. }));
        assert_eq!(
            storage
                .record(&ResourceId::new("glow").unwrap())
                .unwrap()
                .size(),
            3
        );
    }

    #[test]
    fn a_bundle_path_that_leaves_the_bundle_is_refused() {
        let roots = Roots::new("bundle-escape");
        let mut storage = open(&roots, 10);
        let before = roots.index_bytes();
        let error = storage
            .mount_bundle(
                "glow",
                Path::new("../project/secret"),
                Checksum::of(b"do-not-touch"),
            )
            .unwrap_err();
        assert!(matches!(error, StorageError::LeavesRoot { .. }));
        assert_eq!(roots.secret(), b"do-not-touch");
        assert_eq!(roots.index_bytes(), before);
        assert!(storage.record(&ResourceId::new("glow").unwrap()).is_none());
    }

    #[test]
    fn the_bundle_and_the_cache_cannot_be_the_same_directory() {
        let temp = Temp::new("same");
        let root = temp.path().join("both");
        fs::create_dir(&root).unwrap();
        let error = Storage::open(&root, &root, Quota::bytes(1)).unwrap_err();
        assert!(matches!(error, StorageError::SameRoot { .. }));
        assert!(!root.join("files").exists());
        assert!(!root.join("index.json").exists());
    }

    #[test]
    fn a_truncated_index_is_refused_and_left_in_place() {
        let roots = Roots::new("truncated");
        fs::write(roots.cache.join("index.json"), b"{").unwrap();
        let error = Storage::open(&roots.bundle, &roots.cache, Quota::bytes(1)).unwrap_err();
        assert!(matches!(error, StorageError::Index { .. }));
        assert_eq!(fs::read(roots.cache.join("index.json")).unwrap(), b"{");
        assert_eq!(roots.secret(), b"do-not-touch");
    }

    #[test]
    fn an_unknown_field_or_another_version_is_refused() {
        let roots = Roots::new("unknown");
        let extra = br#"{"version":1,"next_age":1,"records":[],"extra":true}"#;
        fs::write(roots.cache.join("index.json"), extra).unwrap();
        let error = Storage::open(&roots.bundle, &roots.cache, Quota::bytes(1)).unwrap_err();
        assert!(matches!(error, StorageError::Index { .. }));
        assert_eq!(fs::read(roots.cache.join("index.json")).unwrap(), extra);

        let nested = br#"{"version":1,"next_age":1,"records":[{"id":"glow","place":"bundle","path":"glow.bin","size":0,"checksum":null,"partial":false,"preview":false,"age":0,"extra":1}]}"#;
        fs::write(roots.cache.join("index.json"), nested).unwrap();
        let error = Storage::open(&roots.bundle, &roots.cache, Quota::bytes(1)).unwrap_err();
        assert!(matches!(error, StorageError::Index { .. }));
        assert_eq!(fs::read(roots.cache.join("index.json")).unwrap(), nested);

        fs::write(
            roots.cache.join("index.json"),
            br#"{"version":2,"next_age":1,"records":[]}"#,
        )
        .unwrap();
        let error = Storage::open(&roots.bundle, &roots.cache, Quota::bytes(1)).unwrap_err();
        assert!(matches!(error, StorageError::IndexVersion { found: 2 }));
    }

    #[test]
    fn an_index_path_that_leaves_the_cache_is_refused() {
        let roots = Roots::new("index-escape");
        let checksum = Checksum::of(b"do-not-touch").to_hex();
        let body = format!(
            r#"{{"version":1,"next_age":1,"records":[{{"id":"glow","place":"cache","path":"../project/secret","size":12,"checksum":"{checksum}","partial":false,"preview":false,"age":0}}]}}"#
        );
        fs::write(roots.cache.join("index.json"), &body).unwrap();
        let error = Storage::open(&roots.bundle, &roots.cache, Quota::bytes(1)).unwrap_err();
        assert!(matches!(
            error,
            StorageError::LeavesRoot { .. } | StorageError::Index { .. }
        ));
        assert_eq!(roots.secret(), b"do-not-touch");
        assert_eq!(
            fs::read(roots.cache.join("index.json")).unwrap(),
            body.as_bytes()
        );
    }

    #[test]
    fn a_missing_indexed_file_refuses_open_and_leaves_the_index() {
        let roots = Roots::new("missing-file");
        fs::write(roots.bundle.join("glow.bin"), b"glow").unwrap();
        let mut storage = open(&roots, 10);
        storage
            .mount_bundle("glow", Path::new("glow.bin"), Checksum::of(b"glow"))
            .unwrap();
        fs::write(roots.bundle.join("other.bin"), b"other").unwrap();
        storage
            .mount_bundle("other", Path::new("other.bin"), Checksum::of(b"other"))
            .unwrap();
        let bytes = roots.index_bytes();
        fs::remove_file(roots.bundle.join("glow.bin")).unwrap();
        let error = Storage::open(&roots.bundle, &roots.cache, Quota::bytes(10)).unwrap_err();
        assert!(matches!(
            error,
            StorageError::Io { .. } | StorageError::NotAFile { .. }
        ));
        assert_eq!(roots.index_bytes(), bytes);
        assert_eq!(fs::read(roots.bundle.join("other.bin")).unwrap(), b"other");
    }

    #[test]
    fn a_changed_bundle_file_is_refused_on_open_and_not_on_path() {
        let roots = Roots::new("changed");
        fs::write(roots.bundle.join("glow.bin"), b"glow").unwrap();
        let mut storage = open(&roots, 10);
        storage
            .mount_bundle("glow", Path::new("glow.bin"), Checksum::of(b"glow"))
            .unwrap();
        fs::write(roots.bundle.join("glow.bin"), b"nope").unwrap();
        let id = ResourceId::new("glow").unwrap();
        assert!(storage.path(&id).is_ok());
        assert_eq!(
            storage.record(&id).unwrap().checksum(),
            Some(Checksum::of(b"glow"))
        );
        let error = Storage::open(&roots.bundle, &roots.cache, Quota::bytes(10)).unwrap_err();
        assert!(matches!(error, StorageError::ChecksumMismatch { .. }));
    }

    #[test]
    fn a_missing_file_does_not_hide_another_path() {
        let roots = Roots::new("one-missing");
        fs::write(roots.bundle.join("a.bin"), b"aaa").unwrap();
        fs::write(roots.bundle.join("b.bin"), b"bbbb").unwrap();
        let mut storage = open(&roots, 10);
        storage
            .mount_bundle("a", Path::new("a.bin"), Checksum::of(b"aaa"))
            .unwrap();
        storage
            .mount_bundle("b", Path::new("b.bin"), Checksum::of(b"bbbb"))
            .unwrap();
        let before = roots.index_bytes();
        fs::remove_file(roots.bundle.join("a.bin")).unwrap();
        assert!(storage.path(&ResourceId::new("b").unwrap()).is_ok());
        assert!(matches!(
            storage.path(&ResourceId::new("a").unwrap()).unwrap_err(),
            StorageError::Missing { .. }
        ));
        assert_eq!(roots.index_bytes(), before);
        assert!(storage.record(&ResourceId::new("a").unwrap()).is_some());
    }

    #[test]
    fn a_preview_without_a_checksum_can_be_indexed() {
        let roots = Roots::new("preview-open");
        fs::create_dir(roots.cache.join("files")).unwrap();
        fs::write(roots.cache.join("files").join("poster"), b"mov").unwrap();
        let body = br#"{"version":1,"next_age":2,"records":[{"id":"poster","place":"cache","path":"files/poster","size":3,"checksum":null,"partial":false,"preview":true,"age":1}]}"#;
        fs::write(roots.cache.join("index.json"), body).unwrap();
        let storage = open(&roots, 10);
        let id = ResourceId::new("poster").unwrap();
        assert!(storage.path(&id).is_ok());
        assert!(storage.record(&id).unwrap().checksum().is_none());
        assert!(storage.record(&id).unwrap().is_preview());
        assert_eq!(storage.used_bytes(), 3);
        assert_eq!(storage.plan(8).unwrap(), vec![id]);
        assert_eq!(
            fs::read(roots.cache.join("files").join("poster")).unwrap(),
            b"mov"
        );
    }

    #[test]
    fn a_partial_preview_has_no_published_path() {
        let roots = Roots::new("partial-open");
        fs::create_dir(roots.cache.join("incoming")).unwrap();
        fs::write(roots.cache.join("incoming").join("poster"), b"part").unwrap();
        let body = br#"{"version":1,"next_age":2,"records":[{"id":"poster","place":"cache","path":"incoming/poster","size":4,"checksum":null,"partial":true,"preview":true,"age":1}]}"#;
        fs::write(roots.cache.join("index.json"), body).unwrap();
        let storage = open(&roots, 10);
        let id = ResourceId::new("poster").unwrap();
        assert!(matches!(
            storage.path(&id).unwrap_err(),
            StorageError::StillPartial { .. }
        ));
        assert!(storage.record(&id).unwrap().is_partial());
        assert_eq!(storage.used_bytes(), 4);
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_index_or_bundle_file_is_refused() {
        let roots = Roots::new("symlink");
        let outside = roots.project.join("secret");
        std::os::unix::fs::symlink(&outside, roots.bundle.join("link.bin")).unwrap();
        let mut storage = open(&roots, 10);
        let error = storage
            .mount_bundle("glow", Path::new("link.bin"), Checksum::of(b"do-not-touch"))
            .unwrap_err();
        assert!(matches!(error, StorageError::Symlink { .. }));
        assert_eq!(roots.secret(), b"do-not-touch");

        let real = roots.cache.join("real.json");
        fs::write(&real, b"{").unwrap();
        fs::remove_file(roots.cache.join("index.json")).unwrap();
        std::os::unix::fs::symlink(&real, roots.cache.join("index.json")).unwrap();
        let error = Storage::open(&roots.bundle, &roots.cache, Quota::bytes(1)).unwrap_err();
        assert!(matches!(error, StorageError::Symlink { .. }));
        assert_eq!(fs::read(&real).unwrap(), b"{");
        assert_eq!(roots.secret(), b"do-not-touch");
    }
}
