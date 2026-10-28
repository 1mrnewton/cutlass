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

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};

use cutlass_download::{
    Checksum, DownloadError, Downloader, Job, JobId, Priority, Progress, hash_file,
};

pub use index::{Place, Record, ResourceId};
pub use quota::{Quota, Shortfall};

use index::Index;

/// Why a cache or bundle operation was refused.
#[derive(Debug)]
pub enum StorageError {
    /// The id is empty.
    EmptyId,
    /// A fetch needs a URL.
    EmptyUrl,
    /// A package is published only when a checksum is named.
    ChecksumRequired { id: ResourceId },
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
    /// The cache cannot hold this file without dropping a pinned preview or a package.
    Quota {
        id: ResourceId,
        limit: u64,
        needed: u64,
    },
    /// Pause, cancel, and discard apply to a partial file.
    NotPartial { id: ResourceId },
    /// That holder is not showing the file.
    NotShown { id: ResourceId },
    /// The transfer did not finish.
    Download { message: String },
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
            Self::EmptyUrl => formatter.write_str("a fetch has no URL"),
            Self::ChecksumRequired { id } => {
                write!(
                    formatter,
                    "{id} needs a checksum before it can be published"
                )
            }
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
            Self::Quota { id, limit, needed } => {
                write!(
                    formatter,
                    "{id} needs {needed} bytes and the cache limit is {limit}"
                )
            }
            Self::NotPartial { id } => write!(formatter, "{id} is already published"),
            Self::NotShown { id } => write!(formatter, "{id} is not showing"),
            Self::Download { message } => formatter.write_str(message),
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

/// Who is showing a file. Either one keeps it out of eviction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Showing {
    Panel,
    Playback,
}

/// A file to fetch into the cache. A preview may omit `checksum`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    pub id: ResourceId,
    pub url: String,
    pub checksum: Option<Checksum>,
    pub priority: Priority,
    pub preview: bool,
}

/// The file is published, or the bytes are still partial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fetched {
    Ready(PathBuf),
    Partial(PathBuf),
}

/// The resources directory and the cache directory.
#[derive(Debug)]
pub struct Storage<D> {
    bundle_root: PathBuf,
    cache_root: PathBuf,
    quota: Quota,
    index: Index,
    download: D,
    jobs: BTreeMap<ResourceId, JobId>,
    panel: BTreeSet<ResourceId>,
    playback: BTreeSet<ResourceId>,
}

impl<D: Downloader> Storage<D> {
    /// Open the bundle read-only and the cache. A missing index starts empty.
    pub fn open(
        bundle: &Path,
        cache: &Path,
        quota: Quota,
        download: D,
    ) -> Result<Self, StorageError> {
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
            download,
            jobs: BTreeMap::new(),
            panel: BTreeSet::new(),
            playback: BTreeSet::new(),
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

    #[cfg(test)]
    pub(crate) fn downloader_mut(&mut self) -> &mut D {
        &mut self.download
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
        let panel = &self.panel;
        let playback = &self.playback;
        self.index
            .records()
            .map(|record| quota::Entry {
                id: record.id().clone(),
                size: record.size(),
                age: record.age(),
                evictable: record.place() == Place::Cache
                    && record.is_preview()
                    && !panel.contains(record.id())
                    && !playback.contains(record.id()),
            })
            .collect()
    }

    /// Ask the downloader for `request` and publish it when the bytes are ready.
    ///
    /// A finished package is indexed only when its checksum matches. A preview
    /// with no checksum is indexed without one. Nothing is evicted unless the
    /// new file fits after the oldest unpinned previews are dropped.
    pub fn fetch(&mut self, request: &Request) -> Result<Fetched, StorageError> {
        if request.url.is_empty() {
            return Err(StorageError::EmptyUrl);
        }
        if !request.preview && request.checksum.is_none() {
            return Err(StorageError::ChecksumRequired {
                id: request.id.clone(),
            });
        }
        if self
            .index
            .get(&request.id)
            .is_some_and(|record| !record.is_partial())
        {
            return self.ready(request);
        }
        let relative = PathBuf::from("incoming").join(request.id.as_str());
        let destination = cache::resolve_file(&self.cache_root, &relative, false)?;
        let job_id = if let Some(id) = self.jobs.get(&request.id).copied() {
            id
        } else {
            let id = self
                .download
                .start(&Job {
                    url: request.url.clone(),
                    destination: destination.clone(),
                    checksum: request.checksum,
                    priority: request.priority,
                })
                .map_err(download_error)?;
            self.jobs.insert(request.id.clone(), id);
            id
        };
        let progress = match self.download.progress(job_id) {
            Ok(progress) => progress,
            Err(error) => {
                self.abandon_fresh(&request.id, &destination);
                return Err(download_error(error));
            }
        };
        match progress {
            Progress::Finished { .. } => self.finish(request, &destination),
            Progress::Partial { .. } | Progress::Paused { .. } | Progress::Cancelled { .. } => {
                self.keep_partial(request, &relative, &destination)
            }
        }
    }

    /// The panel or playback is showing `id`. A missing id is refused.
    pub fn show(&mut self, id: &ResourceId, who: Showing) -> Result<(), StorageError> {
        if self.index.get(id).is_none() {
            return Err(StorageError::Missing { id: id.clone() });
        }
        match who {
            Showing::Panel => {
                self.panel.insert(id.clone());
            }
            Showing::Playback => {
                self.playback.insert(id.clone());
            }
        }
        Ok(())
    }

    /// That holder is no longer showing `id`.
    pub fn hide(&mut self, id: &ResourceId, who: Showing) -> Result<(), StorageError> {
        let removed = match who {
            Showing::Panel => self.panel.remove(id),
            Showing::Playback => self.playback.remove(id),
        };
        if !removed {
            return Err(StorageError::NotShown { id: id.clone() });
        }
        Ok(())
    }

    #[must_use]
    pub fn is_shown(&self, id: &ResourceId, who: Showing) -> bool {
        match who {
            Showing::Panel => self.panel.contains(id),
            Showing::Playback => self.playback.contains(id),
        }
    }

    /// Stop the transfer and keep the partial.
    pub fn pause(&mut self, id: &ResourceId) -> Result<(), StorageError> {
        let job = self.partial_job(id)?;
        self.download.pause(job).map_err(download_error)
    }

    /// Stop the transfer and keep the partial.
    pub fn cancel(&mut self, id: &ResourceId) -> Result<(), StorageError> {
        let job = self.partial_job(id)?;
        self.download.cancel(job).map_err(download_error)
    }

    /// Continue a paused partial. A finished file is published.
    pub fn resume(&mut self, id: &ResourceId) -> Result<Fetched, StorageError> {
        let job = self.partial_job(id)?;
        self.download.resume(job).map_err(download_error)?;
        let Some(record) = self.index.get(id).cloned() else {
            return Err(StorageError::Missing { id: id.clone() });
        };
        let request = Request {
            id: id.clone(),
            url: "resume".to_owned(),
            checksum: record.checksum(),
            priority: Priority::Background,
            preview: record.is_preview(),
        };
        let destination = cache::resolve_file(&self.cache_root, record.relative(), true)?;
        match self.download.progress(job).map_err(download_error)? {
            Progress::Finished { .. } => self.finish(&request, &destination),
            Progress::Partial { .. } | Progress::Paused { .. } | Progress::Cancelled { .. } => {
                Ok(Fetched::Partial(destination))
            }
        }
    }

    /// Delete a partial. A published file is left in place.
    pub fn discard(&mut self, id: &ResourceId) -> Result<(), StorageError> {
        let Some(record) = self.index.get(id).cloned() else {
            return Err(StorageError::Missing { id: id.clone() });
        };
        if !record.is_partial() {
            return Err(StorageError::NotPartial { id: id.clone() });
        }
        let path = cache::resolve_file(&self.cache_root, record.relative(), false)?;
        if let Some(job) = self.jobs.get(id).copied() {
            self.download.discard(job).map_err(download_error)?;
        }
        let saved = self.index.clone();
        self.index.remove(id);
        self.jobs.remove(id);
        self.panel.remove(id);
        self.playback.remove(id);
        if let Err(error) = cache::persist(&self.cache_root, &self.index) {
            self.index = saved;
            return Err(error);
        }
        if path.exists() {
            std::fs::remove_file(&path).map_err(|error| StorageError::Io {
                path: path.clone(),
                message: error.to_string(),
            })?;
        }
        Ok(())
    }

    fn ready(&mut self, request: &Request) -> Result<Fetched, StorageError> {
        let path = self.path(&request.id)?;
        let stored = self.index.get(&request.id).and_then(Record::checksum);
        match (stored, request.checksum) {
            (_, None) => Ok(Fetched::Ready(path)),
            (Some(have), Some(want)) if have == want => Ok(Fetched::Ready(path)),
            (Some(_), Some(_)) => Err(StorageError::ChecksumMismatch {
                id: request.id.clone(),
            }),
            (None, Some(want)) => self.attach_checksum(&request.id, &path, want),
        }
    }

    fn attach_checksum(
        &mut self,
        id: &ResourceId,
        path: &Path,
        want: Checksum,
    ) -> Result<Fetched, StorageError> {
        let (found, size) = hash_file(path).map_err(|error| StorageError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
        let stored_size = self.index.get(id).map(Record::size);
        if found != want || stored_size != Some(size) {
            return Err(StorageError::ChecksumMismatch { id: id.clone() });
        }
        let saved = self.index.clone();
        if !self.index.set_checksum(id, want) {
            return Err(StorageError::Missing { id: id.clone() });
        }
        if let Err(error) = cache::persist(&self.cache_root, &self.index) {
            self.index = saved;
            return Err(error);
        }
        Ok(Fetched::Ready(path.to_path_buf()))
    }

    fn finish(&mut self, request: &Request, source: &Path) -> Result<Fetched, StorageError> {
        let (found, size) = match hash_file(source) {
            Ok(hashed) => hashed,
            Err(error) => {
                self.abandon_fresh(&request.id, source);
                return Err(StorageError::Io {
                    path: source.to_path_buf(),
                    message: error.to_string(),
                });
            }
        };
        if let Some(expected) = request.checksum
            && found != expected
        {
            self.abandon_fresh(&request.id, source);
            return Err(StorageError::ChecksumMismatch {
                id: request.id.clone(),
            });
        }
        let published = PathBuf::from("files").join(request.id.as_str());
        self.accept(request, &published, source, size, request.checksum, false)
    }

    fn keep_partial(
        &mut self,
        request: &Request,
        relative: &Path,
        source: &Path,
    ) -> Result<Fetched, StorageError> {
        let size = match std::fs::metadata(source) {
            Ok(metadata) if metadata.is_file() => metadata.len(),
            Ok(_) => {
                self.abandon_fresh(&request.id, source);
                return Err(StorageError::NotAFile {
                    path: source.to_path_buf(),
                });
            }
            Err(error) => {
                self.abandon_fresh(&request.id, source);
                return Err(StorageError::Io {
                    path: source.to_path_buf(),
                    message: error.to_string(),
                });
            }
        };
        self.accept(request, relative, source, size, request.checksum, true)
    }

    fn accept(
        &mut self,
        request: &Request,
        relative: &Path,
        source: &Path,
        size: u64,
        checksum: Option<Checksum>,
        partial: bool,
    ) -> Result<Fetched, StorageError> {
        let used = self.index.used_except(&request.id);
        let entries = self.entries_except(&request.id);
        let victims = match quota::select(&entries, used, size, self.quota.limit()) {
            Ok(victims) => victims,
            Err(_) => {
                self.abandon_fresh(&request.id, source);
                return Err(StorageError::Quota {
                    id: request.id.clone(),
                    limit: self.quota.limit(),
                    needed: size,
                });
            }
        };
        let destination = if partial {
            source.to_path_buf()
        } else {
            let destination = cache::resolve_file(&self.cache_root, relative, false)?;
            if source != destination {
                if let Some(parent) = destination.parent() {
                    std::fs::create_dir_all(parent).map_err(|error| StorageError::Io {
                        path: parent.to_path_buf(),
                        message: error.to_string(),
                    })?;
                }
                cache::replace_file(source, &destination)?;
            }
            destination
        };
        let age = if request.preview {
            match self.index.get(&request.id) {
                Some(record) if record.is_preview() && record.age() > 0 => record.age(),
                _ => self.index.allocate_age()?,
            }
        } else {
            0
        };
        let saved_index = self.index.clone();
        let saved_jobs = self.jobs.clone();
        let victim_paths = self.victim_paths(&victims)?;
        for id in &victims {
            self.index.remove(id);
            self.jobs.remove(id);
            self.panel.remove(id);
            self.playback.remove(id);
        }
        self.index.insert(Record::cached(
            request.id.clone(),
            relative.to_path_buf(),
            size,
            checksum,
            partial,
            request.preview,
            age,
        ));
        if !partial {
            self.jobs.remove(&request.id);
        }
        if let Err(error) = cache::persist(&self.cache_root, &self.index) {
            self.index = saved_index;
            self.jobs = saved_jobs;
            return Err(error);
        }
        for (id, path, job) in victim_paths {
            if let Some(job) = job {
                let _ = self.download.discard(job);
            }
            self.jobs.remove(&id);
            if path.exists() {
                std::fs::remove_file(&path).map_err(|error| StorageError::Io {
                    path,
                    message: error.to_string(),
                })?;
            }
        }
        if partial {
            Ok(Fetched::Partial(destination))
        } else {
            Ok(Fetched::Ready(destination))
        }
    }

    fn entries_except(&self, id: &ResourceId) -> Vec<quota::Entry> {
        self.entries()
            .into_iter()
            .filter(|entry| &entry.id != id)
            .collect()
    }

    fn victim_paths(
        &self,
        victims: &[ResourceId],
    ) -> Result<Vec<(ResourceId, PathBuf, Option<JobId>)>, StorageError> {
        let mut paths = Vec::new();
        for id in victims {
            let Some(record) = self.index.get(id) else {
                continue;
            };
            let path = cache::resolve_file(&self.cache_root, record.relative(), false)?;
            paths.push((id.clone(), path, self.jobs.get(id).copied()));
        }
        Ok(paths)
    }

    fn partial_job(&mut self, id: &ResourceId) -> Result<JobId, StorageError> {
        let Some(record) = self.index.get(id) else {
            return Err(StorageError::Missing { id: id.clone() });
        };
        if !record.is_partial() {
            return Err(StorageError::NotPartial { id: id.clone() });
        }
        self.jobs
            .get(id)
            .copied()
            .ok_or_else(|| StorageError::Download {
                message: format!("{id} has no transfer"),
            })
    }

    /// Drop a file this fetch created. An indexed partial is left alone.
    fn abandon_fresh(&mut self, id: &ResourceId, file: &Path) {
        if self.index.get(id).is_some() {
            return;
        }
        if let Some(job) = self.jobs.remove(id) {
            let _ = self.download.discard(job);
        }
        if file.exists() {
            let _ = std::fs::remove_file(file);
        }
    }
}

fn download_error(error: DownloadError) -> StorageError {
    StorageError::Download {
        message: error.to_string(),
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

    use cutlass_download::{Checksum, DownloadError, Downloader, Job, JobId, Priority, Progress};

    use super::{Fetched, Place, Quota, Request, ResourceId, Showing, Storage, StorageError, Temp};

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

    fn open(roots: &Roots, limit: u64) -> Storage<Idle> {
        Storage::open(&roots.bundle, &roots.cache, Quota::bytes(limit), Idle).unwrap()
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
        let again = Storage::open(&roots.bundle, &roots.cache, Quota::bytes(0), Idle).unwrap();
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
        let error = Storage::open(&root, &root, Quota::bytes(1), Idle).unwrap_err();
        assert!(matches!(error, StorageError::SameRoot { .. }));
        assert!(!root.join("files").exists());
        assert!(!root.join("index.json").exists());
    }

    #[test]
    fn a_truncated_index_is_refused_and_left_in_place() {
        let roots = Roots::new("truncated");
        fs::write(roots.cache.join("index.json"), b"{").unwrap();
        let error = Storage::open(&roots.bundle, &roots.cache, Quota::bytes(1), Idle).unwrap_err();
        assert!(matches!(error, StorageError::Index { .. }));
        assert_eq!(fs::read(roots.cache.join("index.json")).unwrap(), b"{");
        assert_eq!(roots.secret(), b"do-not-touch");
    }

    #[test]
    fn an_unknown_field_or_another_version_is_refused() {
        let roots = Roots::new("unknown");
        let extra = br#"{"version":1,"next_age":1,"records":[],"extra":true}"#;
        fs::write(roots.cache.join("index.json"), extra).unwrap();
        let error = Storage::open(&roots.bundle, &roots.cache, Quota::bytes(1), Idle).unwrap_err();
        assert!(matches!(error, StorageError::Index { .. }));
        assert_eq!(fs::read(roots.cache.join("index.json")).unwrap(), extra);

        let nested = br#"{"version":1,"next_age":1,"records":[{"id":"glow","place":"bundle","path":"glow.bin","size":0,"checksum":null,"partial":false,"preview":false,"age":0,"extra":1}]}"#;
        fs::write(roots.cache.join("index.json"), nested).unwrap();
        let error = Storage::open(&roots.bundle, &roots.cache, Quota::bytes(1), Idle).unwrap_err();
        assert!(matches!(error, StorageError::Index { .. }));
        assert_eq!(fs::read(roots.cache.join("index.json")).unwrap(), nested);

        fs::write(
            roots.cache.join("index.json"),
            br#"{"version":2,"next_age":1,"records":[]}"#,
        )
        .unwrap();
        let error = Storage::open(&roots.bundle, &roots.cache, Quota::bytes(1), Idle).unwrap_err();
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
        let error = Storage::open(&roots.bundle, &roots.cache, Quota::bytes(1), Idle).unwrap_err();
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
        let error = Storage::open(&roots.bundle, &roots.cache, Quota::bytes(10), Idle).unwrap_err();
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
        let error = Storage::open(&roots.bundle, &roots.cache, Quota::bytes(10), Idle).unwrap_err();
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
        let error = Storage::open(&roots.bundle, &roots.cache, Quota::bytes(1), Idle).unwrap_err();
        assert!(matches!(error, StorageError::Symlink { .. }));
        assert_eq!(fs::read(&real).unwrap(), b"{");
        assert_eq!(roots.secret(), b"do-not-touch");
    }

    #[derive(Debug)]
    struct Idle;

    impl Downloader for Idle {
        fn start(&mut self, job: &Job) -> Result<JobId, DownloadError> {
            Err(DownloadError::Failed {
                message: format!("no transfer for {}", job.url),
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

    #[derive(Clone, Copy)]
    enum Body {
        Full,
        Prefix(usize),
    }

    struct Script {
        body: Vec<u8>,
        stop: Body,
    }

    struct Live {
        destination: PathBuf,
        body: Vec<u8>,
        written: usize,
        phase: Progress,
    }

    struct Double {
        scripts: std::collections::BTreeMap<String, Script>,
        jobs: std::collections::BTreeMap<u64, Live>,
        next: u64,
        starts: Vec<(String, PathBuf, Priority, Option<Checksum>)>,
        calls: Vec<&'static str>,
        fail_discard: bool,
    }

    impl Double {
        fn new() -> Self {
            Self {
                scripts: std::collections::BTreeMap::new(),
                jobs: std::collections::BTreeMap::new(),
                next: 1,
                starts: Vec::new(),
                calls: Vec::new(),
                fail_discard: false,
            }
        }

        fn file(&mut self, url: &str, body: &[u8]) {
            self.scripts.insert(
                url.to_owned(),
                Script {
                    body: body.to_vec(),
                    stop: Body::Full,
                },
            );
        }

        fn prefix(&mut self, url: &str, body: &[u8], bytes: usize) {
            self.scripts.insert(
                url.to_owned(),
                Script {
                    body: body.to_vec(),
                    stop: Body::Prefix(bytes),
                },
            );
        }
    }

    impl Downloader for Double {
        fn start(&mut self, job: &Job) -> Result<JobId, DownloadError> {
            let Some(script) = self.scripts.get(&job.url) else {
                return Err(DownloadError::Failed {
                    message: format!("no bytes for {}", job.url),
                });
            };
            let body = script.body.clone();
            let stop = script.stop;
            self.calls.push("start");
            self.starts.push((
                job.url.clone(),
                job.destination.clone(),
                job.priority,
                job.checksum,
            ));
            let written = match stop {
                Body::Full => body.len(),
                Body::Prefix(bytes) => bytes.min(body.len()),
            };
            if let Some(parent) = job.destination.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(&job.destination, &body[..written]).unwrap();
            let bytes_written = u64::try_from(written).unwrap();
            let phase = if written == body.len() && matches!(stop, Body::Full) {
                Progress::Finished { bytes_written }
            } else {
                Progress::Partial { bytes_written }
            };
            let id = JobId(self.next);
            self.next += 1;
            self.jobs.insert(
                id.0,
                Live {
                    destination: job.destination.clone(),
                    body,
                    written,
                    phase,
                },
            );
            Ok(id)
        }

        fn progress(&mut self, id: JobId) -> Result<Progress, DownloadError> {
            self.jobs
                .get(&id.0)
                .map(|live| live.phase)
                .ok_or(DownloadError::UnknownJob { id })
        }

        fn pause(&mut self, id: JobId) -> Result<(), DownloadError> {
            self.calls.push("pause");
            let live = self
                .jobs
                .get_mut(&id.0)
                .ok_or(DownloadError::UnknownJob { id })?;
            live.phase = Progress::Paused {
                bytes_written: u64::try_from(live.written).unwrap_or(u64::MAX),
            };
            Ok(())
        }

        fn resume(&mut self, id: JobId) -> Result<(), DownloadError> {
            self.calls.push("resume");
            let live = self
                .jobs
                .get_mut(&id.0)
                .ok_or(DownloadError::UnknownJob { id })?;
            fs::write(&live.destination, &live.body).unwrap();
            live.written = live.body.len();
            live.phase = Progress::Finished {
                bytes_written: u64::try_from(live.written).unwrap_or(u64::MAX),
            };
            Ok(())
        }

        fn cancel(&mut self, id: JobId) -> Result<(), DownloadError> {
            self.calls.push("cancel");
            let live = self
                .jobs
                .get_mut(&id.0)
                .ok_or(DownloadError::UnknownJob { id })?;
            live.phase = Progress::Cancelled {
                bytes_written: u64::try_from(live.written).unwrap_or(u64::MAX),
            };
            Ok(())
        }

        fn discard(&mut self, id: JobId) -> Result<(), DownloadError> {
            self.calls.push("discard");
            if self.fail_discard {
                return Err(DownloadError::Failed {
                    message: "discard refused".to_owned(),
                });
            }
            let Some(live) = self.jobs.remove(&id.0) else {
                return Err(DownloadError::UnknownJob { id });
            };
            if live.destination.exists() {
                fs::remove_file(&live.destination).unwrap();
            }
            Ok(())
        }
    }

    fn request(id: &str, url: &str, checksum: Option<Checksum>, preview: bool) -> Request {
        Request {
            id: ResourceId::new(id).unwrap(),
            url: url.to_owned(),
            checksum,
            priority: Priority::Background,
            preview,
        }
    }

    fn ready(fetched: Fetched) -> PathBuf {
        match fetched {
            Fetched::Ready(path) => path,
            Fetched::Partial(path) => panic!("partial {}", path.display()),
        }
    }

    #[test]
    fn a_preview_without_a_checksum_is_published_inside_the_cache() {
        let roots = Roots::new("publish");
        let mut double = Double::new();
        double.file("https://example.test/poster.mp4", b"mov");
        let mut storage =
            Storage::open(&roots.bundle, &roots.cache, Quota::bytes(10), double).unwrap();
        let mut fetch = request("poster", "https://example.test/poster.mp4", None, true);
        fetch.priority = Priority::Visible;
        let path = ready(storage.fetch(&fetch).unwrap());
        assert_eq!(fs::read(&path).unwrap(), b"mov");
        assert!(path.ends_with("files/poster"));
        assert!(!path.starts_with(&roots.project));
        assert!(!roots.cache.join("incoming").join("poster").exists());
        let record = storage.record(&fetch.id).unwrap();
        assert!(record.checksum().is_none());
        assert!(!record.is_partial());
        assert_eq!(storage.used_bytes(), 3);
        assert_eq!(roots.secret(), b"do-not-touch");
        let starts = &storage.downloader_mut().starts;
        assert_eq!(starts.len(), 1);
        assert_eq!(starts[0].2, Priority::Visible);
        assert!(starts[0].1.ends_with("incoming/poster"));
        storage.fetch(&fetch).unwrap();
        assert_eq!(storage.downloader_mut().starts.len(), 1);
    }

    #[test]
    fn a_checksum_mismatch_leaves_the_published_file() {
        let roots = Roots::new("mismatch");
        let mut double = Double::new();
        double.file("https://example.test/a", b"good");
        double.file("https://example.test/b", b"bad!");
        let mut storage =
            Storage::open(&roots.bundle, &roots.cache, Quota::bytes(20), double).unwrap();
        let first = request("clip", "https://example.test/a", None, true);
        let path = ready(storage.fetch(&first).unwrap());
        let before = fs::read(&path).unwrap();
        let index = roots.index_bytes();
        let second = request(
            "other",
            "https://example.test/b",
            Some(Checksum::of(b"good")),
            true,
        );
        let error = storage.fetch(&second).unwrap_err();
        assert!(matches!(error, StorageError::ChecksumMismatch { .. }));
        assert_eq!(fs::read(&path).unwrap(), before);
        assert!(storage.record(&second.id).is_none());
        assert!(!roots.cache.join("incoming").join("other").exists());
        assert_eq!(roots.index_bytes(), index);
        assert_eq!(fs::read(&path).unwrap(), before);
        assert_eq!(roots.secret(), b"do-not-touch");
        let again = Storage::open(&roots.bundle, &roots.cache, Quota::bytes(20), Idle).unwrap();
        assert_eq!(fs::read(again.path(&first.id).unwrap()).unwrap(), b"good");
    }

    #[test]
    fn a_later_checksum_is_stored_only_when_the_bytes_match() {
        let roots = Roots::new("later");
        let mut double = Double::new();
        double.file("https://example.test/poster.mp4", b"mov");
        let mut storage =
            Storage::open(&roots.bundle, &roots.cache, Quota::bytes(10), double).unwrap();
        let mut fetch = request("poster", "https://example.test/poster.mp4", None, true);
        let path = ready(storage.fetch(&fetch).unwrap());
        fetch.checksum = Some(Checksum::of(b"nope"));
        let error = storage.fetch(&fetch).unwrap_err();
        assert!(matches!(error, StorageError::ChecksumMismatch { .. }));
        assert!(storage.record(&fetch.id).unwrap().checksum().is_none());
        assert_eq!(fs::read(&path).unwrap(), b"mov");
        fetch.checksum = Some(Checksum::of(b"mov"));
        ready(storage.fetch(&fetch).unwrap());
        assert_eq!(
            storage.record(&fetch.id).unwrap().checksum(),
            Some(Checksum::of(b"mov"))
        );
        assert_eq!(storage.downloader_mut().starts.len(), 1);
        assert_eq!(fs::read(&path).unwrap(), b"mov");
    }

    #[test]
    fn the_oldest_unpinned_preview_is_evicted_and_a_pin_blocks_it() {
        let roots = Roots::new("evict");
        let mut double = Double::new();
        double.file("https://example.test/a", b"aaaa");
        double.file("https://example.test/b", b"bbbb");
        double.file("https://example.test/c", b"cccc");
        let mut storage =
            Storage::open(&roots.bundle, &roots.cache, Quota::bytes(8), double).unwrap();
        let first = request("a", "https://example.test/a", None, true);
        let second = request("b", "https://example.test/b", None, true);
        let third = request("c", "https://example.test/c", None, true);
        ready(storage.fetch(&first).unwrap());
        ready(storage.fetch(&second).unwrap());
        storage.show(&first.id, Showing::Panel).unwrap();
        storage.show(&second.id, Showing::Playback).unwrap();
        let error = storage.fetch(&third).unwrap_err();
        assert!(matches!(error, StorageError::Quota { .. }));
        assert_eq!(fs::read(storage.path(&first.id).unwrap()).unwrap(), b"aaaa");
        assert_eq!(
            fs::read(storage.path(&second.id).unwrap()).unwrap(),
            b"bbbb"
        );
        assert!(storage.record(&third.id).is_none());
        storage.hide(&first.id, Showing::Panel).unwrap();
        ready(storage.fetch(&third).unwrap());
        assert!(storage.path(&first.id).is_err());
        assert!(!roots.cache.join("files").join("a").exists());
        assert_eq!(
            fs::read(storage.path(&second.id).unwrap()).unwrap(),
            b"bbbb"
        );
        assert_eq!(fs::read(storage.path(&third.id).unwrap()).unwrap(), b"cccc");
        assert_eq!(storage.used_bytes(), 8);
        assert_eq!(roots.secret(), b"do-not-touch");
    }

    #[test]
    fn playback_and_a_bundle_and_a_package_are_not_evicted() {
        let roots = Roots::new("keep");
        fs::write(roots.bundle.join("glow.bin"), b"glow").unwrap();
        let mut double = Double::new();
        double.file("https://example.test/pack", b"pkg!");
        double.file("https://example.test/old", b"old");
        double.file("https://example.test/new", b"new!");
        let mut storage =
            Storage::open(&roots.bundle, &roots.cache, Quota::bytes(8), double).unwrap();
        storage
            .mount_bundle("glow", Path::new("glow.bin"), Checksum::of(b"glow"))
            .unwrap();
        let package = request(
            "pack",
            "https://example.test/pack",
            Some(Checksum::of(b"pkg!")),
            false,
        );
        ready(storage.fetch(&package).unwrap());
        let old = request("old", "https://example.test/old", None, true);
        let new = request("new", "https://example.test/new", None, true);
        ready(storage.fetch(&old).unwrap());
        storage.show(&old.id, Showing::Playback).unwrap();
        let blocked = storage.fetch(&new).unwrap_err();
        assert!(matches!(blocked, StorageError::Quota { .. }));
        assert_eq!(fs::read(roots.bundle.join("glow.bin")).unwrap(), b"glow");
        assert_eq!(
            fs::read(storage.path(&package.id).unwrap()).unwrap(),
            b"pkg!"
        );
        assert_eq!(fs::read(storage.path(&old.id).unwrap()).unwrap(), b"old");
        storage.hide(&old.id, Showing::Playback).unwrap();
        ready(storage.fetch(&new).unwrap());
        assert!(storage.path(&old.id).is_err());
        assert_eq!(fs::read(roots.bundle.join("glow.bin")).unwrap(), b"glow");
        assert_eq!(
            fs::read(storage.path(&package.id).unwrap()).unwrap(),
            b"pkg!"
        );
        assert_eq!(fs::read(storage.path(&new.id).unwrap()).unwrap(), b"new!");
    }

    #[test]
    fn an_exact_file_fits_and_one_byte_more_is_refused() {
        let roots = Roots::new("exact");
        let mut double = Double::new();
        double.file("https://example.test/fit", b"1234");
        double.file("https://example.test/over", b"12345");
        let mut storage =
            Storage::open(&roots.bundle, &roots.cache, Quota::bytes(4), double).unwrap();
        let fit = request("fit", "https://example.test/fit", None, true);
        ready(storage.fetch(&fit).unwrap());
        assert_eq!(storage.used_bytes(), 4);
        let over = request("over", "https://example.test/over", None, true);
        let error = storage.fetch(&over).unwrap_err();
        assert!(matches!(
            error,
            StorageError::Quota {
                needed: 5,
                limit: 4,
                ..
            }
        ));
        assert!(storage.record(&over.id).is_none());
        assert!(!roots.cache.join("incoming").join("over").exists());
        assert_eq!(fs::read(storage.path(&fit.id).unwrap()).unwrap(), b"1234");
        storage.hide(&fit.id, Showing::Panel).unwrap_err();
        storage.show(&fit.id, Showing::Panel).unwrap();
        assert!(storage.is_shown(&fit.id, Showing::Panel));
        let mut double = Double::new();
        double.file("https://example.test/next", b"zzzz");
        let mut again =
            Storage::open(&roots.bundle, &roots.cache, Quota::bytes(4), double).unwrap();
        assert!(!again.is_shown(&fit.id, Showing::Panel));
        let next = request("next", "https://example.test/next", None, true);
        ready(again.fetch(&next).unwrap());
        assert!(again.path(&fit.id).is_err());
        assert_eq!(fs::read(again.path(&next.id).unwrap()).unwrap(), b"zzzz");
    }

    #[test]
    fn pause_and_cancel_keep_a_partial_and_discard_deletes_it() {
        let roots = Roots::new("partial");
        let mut double = Double::new();
        double.prefix("https://example.test/poster.mp4", b"abcdefgh", 4);
        let mut storage =
            Storage::open(&roots.bundle, &roots.cache, Quota::bytes(10), double).unwrap();
        let fetch = request("poster", "https://example.test/poster.mp4", None, true);
        let Fetched::Partial(path) = storage.fetch(&fetch).unwrap() else {
            panic!("expected a partial");
        };
        assert_eq!(fs::read(&path).unwrap(), b"abcd");
        assert!(matches!(
            storage.path(&fetch.id).unwrap_err(),
            StorageError::StillPartial { .. }
        ));
        storage.pause(&fetch.id).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"abcd");
        storage.cancel(&fetch.id).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"abcd");
        assert!(storage.record(&fetch.id).unwrap().is_partial());
        storage.downloader_mut().fail_discard = true;
        let error = storage.discard(&fetch.id).unwrap_err();
        assert!(matches!(error, StorageError::Download { .. }));
        assert_eq!(fs::read(&path).unwrap(), b"abcd");
        storage.downloader_mut().fail_discard = false;
        storage.discard(&fetch.id).unwrap();
        assert!(storage.record(&fetch.id).is_none());
        assert!(!path.exists());
        storage
            .downloader_mut()
            .file("https://example.test/poster.mp4", b"abcdefgh");
        let path = ready(storage.fetch(&fetch).unwrap());
        let published = path.clone();
        let error = storage.discard(&fetch.id).unwrap_err();
        assert!(matches!(error, StorageError::NotPartial { .. }));
        assert_eq!(fs::read(&published).unwrap(), b"abcdefgh");
        storage.resume(&fetch.id).unwrap_err();
        assert_eq!(roots.secret(), b"do-not-touch");
    }

    #[test]
    fn resume_publishes_the_partial() {
        let roots = Roots::new("resume");
        let mut double = Double::new();
        double.prefix("https://example.test/poster.mp4", b"abcdefgh", 3);
        let mut storage =
            Storage::open(&roots.bundle, &roots.cache, Quota::bytes(10), double).unwrap();
        let fetch = request(
            "poster",
            "https://example.test/poster.mp4",
            Some(Checksum::of(b"abcdefgh")),
            true,
        );
        storage.fetch(&fetch).unwrap();
        storage.pause(&fetch.id).unwrap();
        let path = ready(storage.resume(&fetch.id).unwrap());
        assert_eq!(fs::read(&path).unwrap(), b"abcdefgh");
        assert!(!storage.record(&fetch.id).unwrap().is_partial());
        assert_eq!(
            storage.record(&fetch.id).unwrap().checksum(),
            Some(Checksum::of(b"abcdefgh"))
        );
    }

    #[test]
    fn a_download_error_or_an_empty_url_leaves_the_index() {
        let roots = Roots::new("download-error");
        let double = Double::new();
        let mut storage =
            Storage::open(&roots.bundle, &roots.cache, Quota::bytes(10), double).unwrap();
        let before = roots.index_bytes();
        let missing = request("poster", "https://example.test/missing", None, true);
        let error = storage.fetch(&missing).unwrap_err();
        assert!(matches!(error, StorageError::Download { .. }));
        assert!(storage.record(&missing.id).is_none());
        assert_eq!(roots.index_bytes(), before);
        assert!(!roots.cache.join("incoming").join("poster").exists());
        let mut empty = missing.clone();
        empty.url.clear();
        assert!(matches!(
            storage.fetch(&empty).unwrap_err(),
            StorageError::EmptyUrl
        ));
        assert!(storage.downloader_mut().starts.is_empty());
        let package = request("pack", "https://example.test/pack", None, false);
        assert!(matches!(
            storage.fetch(&package).unwrap_err(),
            StorageError::ChecksumRequired { .. }
        ));
        assert!(storage.downloader_mut().calls.is_empty());
        assert_eq!(roots.secret(), b"do-not-touch");
    }

    #[test]
    fn a_missing_cached_file_does_not_hide_another_path() {
        let roots = Roots::new("two-previews");
        let mut double = Double::new();
        double.file("https://example.test/a", b"aaa");
        double.file("https://example.test/b", b"bbbb");
        let mut storage =
            Storage::open(&roots.bundle, &roots.cache, Quota::bytes(10), double).unwrap();
        let first = request("a", "https://example.test/a", None, true);
        let second = request("b", "https://example.test/b", None, true);
        let first_path = ready(storage.fetch(&first).unwrap());
        let second_path = ready(storage.fetch(&second).unwrap());
        let index = roots.index_bytes();
        fs::remove_file(&first_path).unwrap();
        assert_eq!(
            fs::read(storage.path(&second.id).unwrap()).unwrap(),
            b"bbbb"
        );
        assert!(matches!(
            storage.path(&first.id).unwrap_err(),
            StorageError::Missing { .. }
        ));
        assert_eq!(roots.index_bytes(), index);
        assert_eq!(fs::read(&second_path).unwrap(), b"bbbb");
        assert_eq!(roots.secret(), b"do-not-touch");
    }
}
