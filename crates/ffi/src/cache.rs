//! Resolve a bundled or cached id.
//!
//! Opening the index does not start a transfer.

use std::path::{Path, PathBuf};

use cutlass_storage::{
    Checksum, DownloadError, Downloader, Job, JobId, Progress, Quota, ResourceId, Storage,
};

use crate::FfiError;

/// Never starts a transfer. The index can open without one.
struct Idle;

impl Downloader for Idle {
    fn start(&mut self, _job: &Job) -> Result<JobId, DownloadError> {
        Err(DownloadError::Failed {
            message: "a phone lookup does not start a transfer".to_owned(),
        })
    }

    fn progress(&mut self, id: JobId) -> Result<Progress, DownloadError> {
        Err(DownloadError::UnknownJob { id })
    }

    fn pause(&mut self, id: JobId) -> Result<(), DownloadError> {
        Err(DownloadError::UnknownJob { id })
    }

    fn resume(&mut self, id: JobId) -> Result<(), DownloadError> {
        Err(DownloadError::UnknownJob { id })
    }

    fn cancel(&mut self, id: JobId) -> Result<(), DownloadError> {
        Err(DownloadError::UnknownJob { id })
    }

    fn discard(&mut self, id: JobId) -> Result<(), DownloadError> {
        Err(DownloadError::UnknownJob { id })
    }
}

/// The bundle and the cache. Lookup only.
pub struct ResourceCache {
    storage: Storage<Idle>,
}

impl ResourceCache {
    /// Open the bundle and the cache. This does not start a transfer.
    pub fn open(bundle: &Path, cache: &Path, limit: u64) -> Result<Self, FfiError> {
        Ok(Self {
            storage: Storage::open(bundle, cache, Quota::bytes(limit), Idle)?,
        })
    }

    /// Published path for `id`. A missing id is `FfiError::Storage(StorageError::Missing)`.
    pub fn resolve(&self, id: &str) -> Result<PathBuf, FfiError> {
        let id = ResourceId::new(id)?;
        Ok(self.storage.path(&id)?)
    }

    /// Index a file that already lives in the bundle. This does not start a transfer.
    pub fn mount(&mut self, id: &str, relative: &Path, checksum: Checksum) -> Result<(), FfiError> {
        self.storage.mount_bundle(id, relative, checksum)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use cutlass_storage::{Checksum, StorageError};

    use super::ResourceCache;
    use crate::FfiError;

    struct Temp {
        path: PathBuf,
    }

    impl Temp {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("cutlass-ffi-cache-{}-{n}", std::process::id()));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn dirs() -> (Temp, PathBuf, PathBuf) {
        let temp = Temp::new();
        let bundle = temp.path.join("bundle");
        let cache = temp.path.join("cache");
        fs::create_dir(&bundle).unwrap();
        fs::create_dir(&cache).unwrap();
        (temp, bundle, cache)
    }

    /// `Storage::open` creates an empty `incoming` directory. A transfer would put a file in it.
    fn assert_no_transfer(cache: &Path) {
        let incoming = cache.join("incoming");
        assert!(!incoming.join("glow").exists());
        if incoming.exists() {
            assert!(
                fs::read_dir(&incoming).unwrap().next().is_none(),
                "a lookup must not start a transfer"
            );
        }
    }

    #[test]
    fn an_unknown_id_is_missing_and_starts_no_transfer() {
        let (_temp, bundle, cache) = dirs();
        let opened = ResourceCache::open(&bundle, &cache, 0).unwrap();
        let error = opened.resolve("glow").unwrap_err();
        assert!(matches!(
            &error,
            FfiError::Storage(StorageError::Missing { .. })
        ));
        assert!(error.to_string().contains("glow"), "{error}");
        assert_no_transfer(&cache);
    }

    #[test]
    fn an_empty_id_is_refused() {
        let (_temp, bundle, cache) = dirs();
        let opened = ResourceCache::open(&bundle, &cache, 0).unwrap();
        let error = opened.resolve("").unwrap_err();
        assert!(matches!(error, FfiError::Storage(StorageError::EmptyId)));
    }

    #[test]
    fn a_path_like_id_is_refused_and_the_sibling_bytes_stay() {
        let (temp, bundle, cache) = dirs();
        let secret = temp.path.join("secret");
        fs::write(&secret, b"known-secret").unwrap();
        let opened = ResourceCache::open(&bundle, &cache, 0).unwrap();
        let error = opened.resolve("../secret").unwrap_err();
        assert!(matches!(
            error,
            FfiError::Storage(StorageError::InvalidId { .. })
        ));
        assert_eq!(fs::read(&secret).unwrap(), b"known-secret");
    }

    #[test]
    fn a_bundled_file_resolves_inside_the_bundle() {
        let (_temp, bundle, cache) = dirs();
        fs::create_dir(bundle.join("effects")).unwrap();
        let file = bundle.join("effects").join("glow.bin");
        fs::write(&file, b"glow").unwrap();
        let file = fs::canonicalize(&file).unwrap();
        let mut opened = ResourceCache::open(&bundle, &cache, 0).unwrap();
        opened
            .mount("glow", Path::new("effects/glow.bin"), Checksum::of(b"glow"))
            .unwrap();
        let path = opened.resolve("glow").unwrap();
        assert_eq!(path, file);
        assert!(path.starts_with(fs::canonicalize(&bundle).unwrap()));
        assert_eq!(fs::read(&path).unwrap(), b"glow");
        assert_no_transfer(&cache);
    }

    #[test]
    fn the_bundle_and_the_cache_cannot_be_the_same_directory() {
        let temp = Temp::new();
        let both = temp.path.join("both");
        let secret = temp.path.join("secret");
        fs::create_dir(&both).unwrap();
        fs::create_dir(&secret).unwrap();
        fs::write(secret.join("kept"), b"do-not-touch").unwrap();
        let Err(error) = ResourceCache::open(&both, &both, 1) else {
            panic!("the same directory was accepted");
        };
        assert!(matches!(
            error,
            FfiError::Storage(StorageError::SameRoot { .. })
        ));
        assert!(!both.join("index.json").exists());
        assert!(!both.join("files").exists());
        assert!(!both.join("incoming").exists());
        assert!(!secret.join("index.json").exists());
        assert_eq!(fs::read(secret.join("kept")).unwrap(), b"do-not-touch");
    }
}
