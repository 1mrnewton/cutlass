//! Stable cache metadata and safety-focused filesystem operations.
//!
//! This crate deliberately stays `std`-only. It owns the identifiers that
//! cross settings, UI, and provider boundaries, plus the low-level operations
//! those layers use to inspect, clear, and relocate cache directories.
//!
//! Directory walks are iterative and inspect entries with
//! [`std::fs::symlink_metadata`], so directory symlinks are never traversed.
//! Callers should avoid concurrently mutating a tree while an operation is in
//! progress; the standard library does not expose race-free `openat`-style
//! traversal on every supported platform.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::collections::BTreeMap;
use std::error::Error;
use std::ffi::OsString;
use std::fmt;
use std::fs::{self, FileType, Metadata};
use std::io;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Component, Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Maximum number of bytes retained from an underlying error message.
const MAX_ERROR_MESSAGE_BYTES: usize = 256;

/// Stable identifiers for every cache currently owned by Cutlass.
///
/// Variant order is the registry order. The string forms are persisted API:
/// changing one requires a migration rather than a rename.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum CacheId {
    /// Decoded frames retained in process memory.
    PreviewFrames,
    /// Library thumbnail images retained in process memory.
    LibraryThumbnails,
    /// Timeline filmstrip images retained in process memory.
    TimelineFilmstrips,
    /// Timeline waveform samples retained in process memory.
    TimelineWaveforms,
    /// Generated lower-resolution media proxies.
    Proxies,
    /// Remotely downloaded source assets.
    Download,
    /// Downloaded asset-catalog data.
    Catalog,
    /// Downloaded lookup-table assets.
    Luts,
    /// Downloaded Lottie animation assets.
    Lottie,
    /// Downloaded template assets.
    Templates,
}

impl CacheId {
    /// Every cache identifier in deterministic registry order.
    pub const ALL: [Self; 10] = [
        Self::PreviewFrames,
        Self::LibraryThumbnails,
        Self::TimelineFilmstrips,
        Self::TimelineWaveforms,
        Self::Proxies,
        Self::Download,
        Self::Catalog,
        Self::Luts,
        Self::Lottie,
        Self::Templates,
    ];

    /// Return the exact stable key used in settings and tool payloads.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PreviewFrames => "preview_frames",
            Self::LibraryThumbnails => "library_thumbnails",
            Self::TimelineFilmstrips => "timeline_filmstrips",
            Self::TimelineWaveforms => "timeline_waveforms",
            Self::Proxies => "proxies",
            Self::Download => "download",
            Self::Catalog => "catalog",
            Self::Luts => "luts",
            Self::Lottie => "lottie",
            Self::Templates => "templates",
        }
    }

    /// Parse an exact stable cache key.
    pub fn parse(key: &str) -> Result<Self, ParseCacheIdError> {
        key.parse()
    }

    /// Return this cache's static descriptor.
    pub const fn descriptor(self) -> &'static CacheDescriptor {
        &CACHE_REGISTRY[self as usize]
    }
}

impl fmt::Display for CacheId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for CacheId {
    type Err = ParseCacheIdError;

    fn from_str(key: &str) -> Result<Self, Self::Err> {
        match key {
            "preview_frames" => Ok(Self::PreviewFrames),
            "library_thumbnails" => Ok(Self::LibraryThumbnails),
            "timeline_filmstrips" => Ok(Self::TimelineFilmstrips),
            "timeline_waveforms" => Ok(Self::TimelineWaveforms),
            "proxies" => Ok(Self::Proxies),
            "download" => Ok(Self::Download),
            "catalog" => Ok(Self::Catalog),
            "luts" => Ok(Self::Luts),
            "lottie" => Ok(Self::Lottie),
            "templates" => Ok(Self::Templates),
            _ => Err(ParseCacheIdError),
        }
    }
}

/// Error returned when a string is not an exact stable [`CacheId`] key.
///
/// The input is intentionally not retained, keeping provider-facing errors
/// bounded even when a caller supplies a hostile string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseCacheIdError;

impl fmt::Display for ParseCacheIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("unknown cache id")
    }
}

impl Error for ParseCacheIdError {}

/// Where a cache's bytes live.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheKind {
    /// Process memory; there is no filesystem path.
    Memory,
    /// A directory beneath the storage root or an absolute override.
    Disk,
}

/// Recovery semantics for cached data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheTier {
    /// Derived data that can be regenerated locally.
    Disposable,
    /// Data that can be downloaded again.
    Redownloadable,
    /// Irreplaceable user-owned data.
    ///
    /// No active clearable cache currently uses this tier. The variant exists
    /// so future registries cannot silently collapse user data into a cache.
    UserData,
}

/// Static metadata for one cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheDescriptor {
    /// Stable identifier.
    pub id: CacheId,
    /// Human-readable label.
    pub label: &'static str,
    /// Memory or disk storage.
    pub kind: CacheKind,
    /// Recovery semantics.
    pub tier: CacheTier,
    /// Relative directory beneath [`StorageLayout::root`] for disk caches.
    ///
    /// Memory descriptors always use `None`.
    pub default_relative: Option<&'static str>,
}

/// Complete cache registry in deterministic display order.
///
/// Projects, configuration, and agent sessions are intentionally absent:
/// they are not clearable caches.
pub static CACHE_REGISTRY: [CacheDescriptor; 10] = [
    CacheDescriptor {
        id: CacheId::PreviewFrames,
        label: "Preview frames",
        kind: CacheKind::Memory,
        tier: CacheTier::Disposable,
        default_relative: None,
    },
    CacheDescriptor {
        id: CacheId::LibraryThumbnails,
        label: "Library thumbnails",
        kind: CacheKind::Memory,
        tier: CacheTier::Disposable,
        default_relative: None,
    },
    CacheDescriptor {
        id: CacheId::TimelineFilmstrips,
        label: "Timeline filmstrips",
        kind: CacheKind::Memory,
        tier: CacheTier::Disposable,
        default_relative: None,
    },
    CacheDescriptor {
        id: CacheId::TimelineWaveforms,
        label: "Timeline waveforms",
        kind: CacheKind::Memory,
        tier: CacheTier::Disposable,
        default_relative: None,
    },
    CacheDescriptor {
        id: CacheId::Proxies,
        label: "Proxies",
        kind: CacheKind::Disk,
        tier: CacheTier::Disposable,
        default_relative: Some("proxies"),
    },
    CacheDescriptor {
        id: CacheId::Download,
        label: "Downloads",
        kind: CacheKind::Disk,
        tier: CacheTier::Redownloadable,
        default_relative: Some("download-cache"),
    },
    CacheDescriptor {
        id: CacheId::Catalog,
        label: "Catalog",
        kind: CacheKind::Disk,
        tier: CacheTier::Redownloadable,
        default_relative: Some("catalog-cache"),
    },
    CacheDescriptor {
        id: CacheId::Luts,
        label: "LUTs",
        kind: CacheKind::Disk,
        tier: CacheTier::Redownloadable,
        default_relative: Some("luts"),
    },
    CacheDescriptor {
        id: CacheId::Lottie,
        label: "Lottie assets",
        kind: CacheKind::Disk,
        tier: CacheTier::Redownloadable,
        default_relative: Some("lottie"),
    },
    CacheDescriptor {
        id: CacheId::Templates,
        label: "Templates",
        kind: CacheKind::Disk,
        tier: CacheTier::Redownloadable,
        default_relative: Some("templates"),
    },
];

/// Return the complete cache registry in deterministic display order.
pub const fn cache_descriptors() -> &'static [CacheDescriptor] {
    &CACHE_REGISTRY
}

/// Look up a descriptor by typed identifier.
pub const fn cache_descriptor(id: CacheId) -> &'static CacheDescriptor {
    id.descriptor()
}

/// Look up a descriptor by exact stable key.
pub fn cache_descriptor_by_key(key: &str) -> Option<&'static CacheDescriptor> {
    CacheId::parse(key).ok().map(CacheId::descriptor)
}

/// Absolute storage root plus optional per-disk-cache path overrides.
///
/// Overrides are held in a [`BTreeMap`] so snapshots and diagnostics remain
/// deterministic regardless of insertion order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageLayout {
    root: PathBuf,
    overrides: BTreeMap<CacheId, PathBuf>,
}

impl StorageLayout {
    /// Create a layout with no overrides.
    pub fn new(root: impl Into<PathBuf>) -> Result<Self, StorageError> {
        let root = normalize_absolute(&root.into(), "storage root", false)?;
        Ok(Self {
            root,
            overrides: BTreeMap::new(),
        })
    }

    /// Create a layout from stable string keys and absolute paths.
    ///
    /// Unknown keys, memory-cache keys, relative paths, and duplicate keys
    /// are rejected rather than ignored.
    pub fn with_overrides<I, K, P>(
        root: impl Into<PathBuf>,
        overrides: I,
    ) -> Result<Self, StorageError>
    where
        I: IntoIterator<Item = (K, P)>,
        K: AsRef<str>,
        P: Into<PathBuf>,
    {
        let mut layout = Self::new(root)?;
        for (key, path) in overrides {
            let id = CacheId::parse(key.as_ref()).map_err(|_| StorageError::UnknownCacheId)?;
            if layout.overrides.contains_key(&id) {
                return Err(StorageError::DuplicateOverride(id));
            }
            layout.set_override(id, path)?;
        }
        Ok(layout)
    }

    /// Absolute default root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Deterministically ordered cache overrides.
    pub fn overrides(&self) -> &BTreeMap<CacheId, PathBuf> {
        &self.overrides
    }

    /// Return the override for one cache, if set.
    pub fn override_for(&self, id: CacheId) -> Option<&Path> {
        self.overrides.get(&id).map(PathBuf::as_path)
    }

    /// Set an absolute override for a disk cache.
    pub fn set_override(
        &mut self,
        id: CacheId,
        path: impl Into<PathBuf>,
    ) -> Result<Option<PathBuf>, StorageError> {
        if id.descriptor().kind != CacheKind::Disk {
            return Err(StorageError::CacheIsNotDisk(id));
        }
        let path = normalize_absolute(&path.into(), "cache override", false)?;
        self.ensure_override_is_disjoint(id, &path)?;
        Ok(self.overrides.insert(id, path))
    }

    /// Set an override using an exact stable cache key.
    pub fn set_override_key(
        &mut self,
        key: &str,
        path: impl Into<PathBuf>,
    ) -> Result<Option<PathBuf>, StorageError> {
        let id = CacheId::parse(key).map_err(|_| StorageError::UnknownCacheId)?;
        self.set_override(id, path)
    }

    /// Remove and return an override.
    pub fn remove_override(&mut self, id: CacheId) -> Option<PathBuf> {
        self.overrides.remove(&id)
    }

    /// Resolve a cache path.
    ///
    /// Memory caches return `None`. Disk caches use an override when present,
    /// otherwise `root/default_relative`.
    pub fn resolve(&self, id: CacheId) -> Option<PathBuf> {
        let descriptor = id.descriptor();
        if descriptor.kind == CacheKind::Memory {
            return None;
        }
        self.overrides.get(&id).cloned().or_else(|| {
            descriptor
                .default_relative
                .map(|relative| self.root.join(relative))
        })
    }

    /// Resolve every disk cache in deterministic registry order.
    pub fn resolved_disk_paths(&self) -> Vec<(CacheId, PathBuf)> {
        cache_descriptors()
            .iter()
            .filter_map(|descriptor| {
                self.resolve(descriptor.id)
                    .map(|path| (descriptor.id, path))
            })
            .collect()
    }

    fn ensure_override_is_disjoint(
        &self,
        id: CacheId,
        candidate: &Path,
    ) -> Result<(), StorageError> {
        for descriptor in cache_descriptors() {
            if descriptor.id == id || descriptor.kind != CacheKind::Disk {
                continue;
            }
            let default = self
                .root
                .join(descriptor.default_relative.expect("disk cache has a path"));
            if paths_overlap(candidate, &default)
                || self
                    .overrides
                    .get(&descriptor.id)
                    .is_some_and(|other| paths_overlap(candidate, other))
            {
                return Err(StorageError::CachePathsOverlap {
                    cache: id,
                    other: descriptor.id,
                });
            }
        }
        Ok(())
    }
}

/// An immutable, versioned copy of a [`SharedStorageLayout`].
///
/// The layout and generation are captured while holding one read lock, so
/// every path resolved through this snapshot belongs to the reported
/// generation. Mutating a layout returned by [`Self::into_parts`] cannot
/// affect the shared value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageLayoutSnapshot {
    layout: StorageLayout,
    generation: u64,
}

impl StorageLayoutSnapshot {
    /// Return the captured layout.
    pub fn layout(&self) -> &StorageLayout {
        &self.layout
    }

    /// Return the generation captured with the layout.
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Resolve a cache against the captured layout.
    pub fn resolve(&self, id: CacheId) -> Option<PathBuf> {
        self.layout.resolve(id)
    }

    /// Consume the snapshot into its detached layout and generation.
    pub fn into_parts(self) -> (StorageLayout, u64) {
        (self.layout, self.generation)
    }
}

/// A borrowed layout generation held stable for one complete cache operation.
///
/// While this lease exists, a writer cannot publish a replacement layout.
/// Cache workers should acquire one at the start of an operation and retain it
/// until every filesystem access using its resolved paths has finished.
pub struct StorageLayoutLease<'a> {
    state: RwLockReadGuard<'a, VersionedStorageLayout>,
}

impl StorageLayoutLease<'_> {
    /// Return the coherently borrowed layout.
    pub fn layout(&self) -> &StorageLayout {
        &self.state.layout
    }

    /// Return the generation held by this lease.
    pub fn generation(&self) -> u64 {
        self.state.generation
    }

    /// Resolve a cache against the held generation.
    pub fn resolve(&self, id: CacheId) -> Option<PathBuf> {
        self.state.layout.resolve(id)
    }
}

/// A cheap-to-clone, thread-safe, versioned [`StorageLayout`].
///
/// Clones share one layout through an [`Arc`]. Readers observe the layout and
/// generation coherently, while writers use an expected generation to prevent
/// stale worker results from replacing newer configuration. The initial
/// generation is zero and each successful write increments it exactly once.
///
/// Lock poisoning is recovered internally. Write operations prepare a
/// complete replacement before committing it, so an unwinding update leaves
/// the prior layout and generation valid.
#[derive(Debug, Clone)]
pub struct SharedStorageLayout {
    inner: Arc<RwLock<VersionedStorageLayout>>,
}

#[derive(Debug)]
struct VersionedStorageLayout {
    layout: StorageLayout,
    generation: u64,
}

impl SharedStorageLayout {
    /// Create shared storage state at generation zero.
    ///
    /// The supplied layout has already been validated by [`StorageLayout`]'s
    /// constructors and mutation methods.
    pub fn new(layout: StorageLayout) -> Self {
        Self::from_generation(layout, 0)
    }

    /// Clone the current layout and generation under one coherent read.
    pub fn snapshot(&self) -> StorageLayoutSnapshot {
        let state = self.read_state();
        StorageLayoutSnapshot {
            layout: state.layout.clone(),
            generation: state.generation,
        }
    }

    /// Hold one generation stable for a complete cache operation.
    ///
    /// Keep the returned lease alive until all work using its paths finishes.
    /// Long-lived leases delay relocation, so callers should scope them to one
    /// refresh, download, install, render, or maintenance operation.
    pub fn lease(&self) -> StorageLayoutLease<'_> {
        StorageLayoutLease {
            state: self.read_state(),
        }
    }

    /// Resolve one cache from a single coherent read of the current layout.
    ///
    /// Memory caches return `None`.
    pub fn resolve(&self, id: CacheId) -> Option<PathBuf> {
        self.resolve_versioned(id).0
    }

    /// Resolve one cache together with the generation used for that path.
    ///
    /// The returned path and generation are captured under the same read lock.
    /// Memory caches return `None` for the path.
    pub fn resolve_versioned(&self, id: CacheId) -> (Option<PathBuf>, u64) {
        let state = self.read_state();
        (state.layout.resolve(id), state.generation)
    }

    /// Replace the layout when `expected_generation` is still current.
    ///
    /// The replacement must be a previously validated [`StorageLayout`].
    /// Success returns the newly installed generation. A stale generation or
    /// exhausted generation leaves the current layout untouched.
    pub fn replace(
        &self,
        expected_generation: u64,
        replacement: StorageLayout,
    ) -> Result<u64, SharedStorageLayoutError> {
        let mut state = self.write_state();
        let next_generation = checked_next_generation(&state, expected_generation)?;
        *state = VersionedStorageLayout {
            layout: replacement,
            generation: next_generation,
        };
        Ok(next_generation)
    }

    /// Run an exclusive transition and publish `replacement` on success.
    ///
    /// Existing [`StorageLayoutLease`] values finish before `transition`
    /// starts, and new leases wait until this method returns. The callback
    /// receives the old and replacement layouts while publication is
    /// exclusive. A callback error or panic leaves the shared layout and
    /// generation unchanged.
    pub fn transition<F, E>(
        &self,
        expected_generation: u64,
        replacement: StorageLayout,
        transition: F,
    ) -> Result<u64, SharedStorageLayoutTransitionError<E>>
    where
        F: FnOnce(&StorageLayout, &StorageLayout) -> Result<(), E>,
    {
        let mut state = self.write_state();
        let next_generation = checked_next_generation(&state, expected_generation)
            .map_err(SharedStorageLayoutTransitionError::Layout)?;
        transition(&state.layout, &replacement)
            .map_err(SharedStorageLayoutTransitionError::Transition)?;
        *state = VersionedStorageLayout {
            layout: replacement,
            generation: next_generation,
        };
        Ok(next_generation)
    }

    /// Update a cloned layout when `expected_generation` is still current.
    ///
    /// The callback runs with the write lock held and should be brief. It must
    /// not call methods on this shared value, which would deadlock. The clone
    /// begins valid, and [`StorageLayout`] mutation methods preserve that
    /// validity. The callback is responsible for handling their errors.
    ///
    /// A stale or exhausted generation refuses the update without invoking the
    /// callback. If the callback panics, the panic propagates, no state is
    /// committed, and subsequent operations recover the poisoned lock.
    pub fn update<F>(
        &self,
        expected_generation: u64,
        update: F,
    ) -> Result<u64, SharedStorageLayoutError>
    where
        F: FnOnce(&mut StorageLayout),
    {
        let mut state = self.write_state();
        let next_generation = checked_next_generation(&state, expected_generation)?;
        let mut replacement = state.layout.clone();
        update(&mut replacement);
        *state = VersionedStorageLayout {
            layout: replacement,
            generation: next_generation,
        };
        Ok(next_generation)
    }

    fn from_generation(layout: StorageLayout, generation: u64) -> Self {
        Self {
            inner: Arc::new(RwLock::new(VersionedStorageLayout { layout, generation })),
        }
    }

    #[cfg(test)]
    fn with_generation_for_test(layout: StorageLayout, generation: u64) -> Self {
        Self::from_generation(layout, generation)
    }

    fn read_state(&self) -> RwLockReadGuard<'_, VersionedStorageLayout> {
        match self.inner.read() {
            Ok(state) => state,
            Err(poisoned) => {
                let state = poisoned.into_inner();
                self.inner.clear_poison();
                state
            }
        }
    }

    fn write_state(&self) -> RwLockWriteGuard<'_, VersionedStorageLayout> {
        match self.inner.write() {
            Ok(state) => state,
            Err(poisoned) => {
                let state = poisoned.into_inner();
                self.inner.clear_poison();
                state
            }
        }
    }
}

impl From<StorageLayout> for SharedStorageLayout {
    fn from(layout: StorageLayout) -> Self {
        Self::new(layout)
    }
}

fn checked_next_generation(
    state: &VersionedStorageLayout,
    expected_generation: u64,
) -> Result<u64, SharedStorageLayoutError> {
    if state.generation != expected_generation {
        return Err(SharedStorageLayoutError::StaleGeneration {
            expected: expected_generation,
            current: state.generation,
        });
    }
    state
        .generation
        .checked_add(1)
        .ok_or(SharedStorageLayoutError::GenerationExhausted)
}

/// A rejected [`SharedStorageLayout`] write.
///
/// This error contains only generation numbers and never retains or displays
/// storage paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedStorageLayoutError {
    /// Another writer replaced the layout after the caller took its snapshot.
    StaleGeneration {
        /// Generation supplied by the caller.
        expected: u64,
        /// Generation currently installed.
        current: u64,
    },
    /// The current generation is `u64::MAX` and cannot advance safely.
    GenerationExhausted,
}

impl fmt::Display for SharedStorageLayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StaleGeneration { expected, current } => write!(
                f,
                "stale storage layout generation: expected {expected}, current {current}"
            ),
            Self::GenerationExhausted => f.write_str("storage layout generation cannot advance"),
        }
    }
}

impl Error for SharedStorageLayoutError {}

/// A rejected exclusive [`SharedStorageLayout::transition`].
#[derive(Debug)]
pub enum SharedStorageLayoutTransitionError<E> {
    /// The expected generation was stale or exhausted.
    Layout(SharedStorageLayoutError),
    /// The caller's transition failed before publication.
    Transition(E),
}

impl<E: fmt::Display> fmt::Display for SharedStorageLayoutTransitionError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Layout(error) => error.fmt(f),
            Self::Transition(error) => write!(f, "storage layout transition failed: {error}"),
        }
    }
}

impl<E> Error for SharedStorageLayoutTransitionError<E>
where
    E: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Layout(error) => Some(error),
            Self::Transition(error) => Some(error),
        }
    }
}

/// Cooperative cancellation check used by filesystem operations.
pub trait Cancellation {
    /// Return `true` when the current operation should stop.
    fn is_cancelled(&self) -> bool;
}

impl<F> Cancellation for F
where
    F: Fn() -> bool,
{
    fn is_cancelled(&self) -> bool {
        self()
    }
}

/// A cancellation check that never cancels.
#[derive(Debug, Default, Clone, Copy)]
pub struct NeverCancelled;

impl Cancellation for NeverCancelled {
    fn is_cancelled(&self) -> bool {
        false
    }
}

/// Logical usage of a directory tree.
///
/// `bytes` sums `symlink_metadata().len()` for every non-directory entry.
/// `files` counts those entries, including symlinks, without following them.
/// Directory inode sizes are not included.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DiskUsage {
    /// Logical bytes across non-directory entries.
    pub bytes: u64,
    /// Number of non-directory entries.
    pub files: u64,
}

/// Result of clearing a cache directory.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ClearReport {
    /// Logical bytes represented by removed non-directory entries.
    pub removed_bytes: u64,
    /// Number of removed non-directory entries.
    pub removed_files: u64,
}

/// Result of a successful cache relocation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RelocationReport {
    /// Logical bytes present when relocation began.
    pub bytes: u64,
    /// Number of non-directory entries present when relocation began.
    pub files: u64,
    /// Whether rename crossed filesystems and the copy fallback was used.
    pub used_copy_fallback: bool,
}

/// Failures from cache layout and filesystem operations.
#[derive(Debug)]
pub enum StorageError {
    /// Cooperative cancellation was requested.
    Cancelled,
    /// A path that must be absolute was relative.
    PathNotAbsolute(&'static str),
    /// An empty path or filesystem root was supplied to a destructive API.
    DangerousPath(&'static str),
    /// An override key was not a registered cache.
    UnknownCacheId,
    /// An override was supplied for a memory cache.
    CacheIsNotDisk(CacheId),
    /// The same override key appeared more than once.
    DuplicateOverride(CacheId),
    /// Two cache roots overlap, so clearing either could remove the other.
    CachePathsOverlap {
        /// Cache whose override was rejected.
        cache: CacheId,
        /// Existing/default cache path it overlaps.
        other: CacheId,
    },
    /// The operation root itself is a symbolic link.
    SymlinkRoot,
    /// A required operation root is not a directory.
    NotDirectory,
    /// The relocation source does not exist.
    SourceMissing,
    /// The relocation destination already exists.
    DestinationExists,
    /// Relocation source and destination are equal or nested.
    PathsOverlap,
    /// Copy fallback encountered a socket, device, FIFO, or other special file.
    UnsupportedFileType,
    /// Byte or file accounting exceeded `u64`.
    AccountingOverflow,
    /// Unique temporary sibling allocation exhausted its bounded attempts.
    TemporaryNameExhausted,
    /// An underlying filesystem operation failed.
    Io {
        /// Short static operation name.
        operation: &'static str,
        /// Operating-system error.
        source: io::Error,
    },
    /// Persistence rejected a complete destination and rollback succeeded.
    PersistenceFailed {
        /// Bounded callback error.
        message: String,
    },
    /// Persistence failed and the best-effort rollback also failed.
    RollbackFailed {
        /// Bounded callback error.
        persistence: String,
        /// Bounded rollback error.
        rollback: String,
    },
    /// Copy failed or was cancelled and its temporary tree could not be removed.
    TemporaryCleanupFailed {
        /// Bounded original error.
        original: String,
        /// Bounded cleanup error.
        cleanup: String,
    },
    /// Persistence committed the destination, but deleting the old copy failed.
    CommittedCleanupFailed {
        /// Bounded cleanup error.
        message: String,
    },
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => f.write_str("storage operation cancelled"),
            Self::PathNotAbsolute(role) => write!(f, "{role} must be absolute"),
            Self::DangerousPath(reason) => write!(f, "dangerous storage path: {reason}"),
            Self::UnknownCacheId => f.write_str("unknown cache override"),
            Self::CacheIsNotDisk(id) => write!(f, "cache {id} has no disk path"),
            Self::DuplicateOverride(id) => write!(f, "duplicate override for cache {id}"),
            Self::CachePathsOverlap { cache, other } => {
                write!(f, "cache paths for {cache} and {other} must not overlap")
            }
            Self::SymlinkRoot => f.write_str("operation root must not be a symbolic link"),
            Self::NotDirectory => f.write_str("operation root must be a directory"),
            Self::SourceMissing => f.write_str("relocation source does not exist"),
            Self::DestinationExists => f.write_str("relocation destination already exists"),
            Self::PathsOverlap => f.write_str("relocation source and destination must not overlap"),
            Self::UnsupportedFileType => {
                f.write_str("copy fallback encountered an unsupported file type")
            }
            Self::AccountingOverflow => f.write_str("storage usage accounting overflowed"),
            Self::TemporaryNameExhausted => {
                f.write_str("could not allocate a temporary relocation directory")
            }
            Self::Io { operation, source } => {
                write!(
                    f,
                    "{operation} failed: {}",
                    bounded_text(&source.to_string())
                )
            }
            Self::PersistenceFailed { message } => {
                write!(f, "persisting relocated cache failed: {message}")
            }
            Self::RollbackFailed {
                persistence,
                rollback,
            } => write!(
                f,
                "persisting relocated cache failed: {persistence}; rollback failed: {rollback}"
            ),
            Self::TemporaryCleanupFailed { original, cleanup } => write!(
                f,
                "{original}; temporary relocation cleanup failed: {cleanup}"
            ),
            Self::CommittedCleanupFailed { message } => write!(
                f,
                "relocation committed, but old cache cleanup failed: {message}"
            ),
        }
    }
}

impl Error for StorageError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Measure a cache tree without following directory symlinks.
///
/// A missing root has zero usage. A symlink root is rejected. Cancellation is
/// checked before each directory and entry; a single large-file metadata call
/// is not interruptible.
pub fn measure_disk_usage<C>(
    root: impl AsRef<Path>,
    cancellation: &C,
) -> Result<DiskUsage, StorageError>
where
    C: Cancellation + ?Sized,
{
    let root = normalize_absolute(root.as_ref(), "measurement root", false)?;
    check_cancelled(cancellation)?;

    let Some(metadata) = optional_symlink_metadata(&root, "inspect measurement root")? else {
        return Ok(DiskUsage::default());
    };
    reject_root_type(&metadata)?;

    let mut usage = DiskUsage::default();
    let mut stack = vec![root];
    while let Some(directory) = stack.pop() {
        check_cancelled(cancellation)?;
        let entries =
            fs::read_dir(&directory).map_err(|error| io_error("read cache directory", error))?;
        for entry in entries {
            check_cancelled(cancellation)?;
            let entry = entry.map_err(|error| io_error("read cache entry", error))?;
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)
                .map_err(|error| io_error("inspect cache entry", error))?;
            if metadata.file_type().is_dir() {
                stack.push(path);
            } else {
                add_usage(&mut usage, &metadata)?;
            }
        }
    }
    Ok(usage)
}

/// Remove a cache's contents while preserving its root directory.
///
/// A missing root is created as an empty directory. Empty paths, filesystem
/// roots, symlink roots, and non-directories are rejected. Nested symlinks are
/// removed as links and never followed. Cancellation can leave a partially
/// cleared tree, but the root itself is never intentionally removed.
pub fn clear_cache<C>(root: impl AsRef<Path>, cancellation: &C) -> Result<ClearReport, StorageError>
where
    C: Cancellation + ?Sized,
{
    let root = normalize_absolute(root.as_ref(), "clear root", false)?;
    check_cancelled(cancellation)?;

    let Some(metadata) = optional_symlink_metadata(&root, "inspect clear root")? else {
        fs::create_dir_all(&root).map_err(|error| io_error("create cache root", error))?;
        let created = fs::symlink_metadata(&root)
            .map_err(|error| io_error("inspect created cache root", error))?;
        reject_root_type(&created)?;
        return Ok(ClearReport::default());
    };
    reject_root_type(&metadata)?;

    let usage = remove_tree(&root, true, cancellation)?;
    fs::create_dir_all(&root).map_err(|error| io_error("recreate cache root", error))?;
    let recreated = fs::symlink_metadata(&root)
        .map_err(|error| io_error("inspect recreated cache root", error))?;
    reject_root_type(&recreated)?;
    Ok(ClearReport {
        removed_bytes: usage.bytes,
        removed_files: usage.files,
    })
}

/// Relocate a cache directory and persist the completed destination.
///
/// The destination must be absolute, absent, non-root, and disjoint from the
/// source. The source must be an absolute, non-symlink directory. The function
/// measures first, then prefers an atomic rename. If rename reports a
/// cross-device move, it copies into a unique temporary sibling and atomically
/// renames that sibling into place.
///
/// `persist` runs only after the destination is complete. On callback failure
/// or panic, rollback is attempted. Copy mode leaves the source untouched
/// until persistence succeeds; after that commit point cleanup runs to
/// completion even if cancellation is requested. Callers receive an explicit
/// error if rollback or post-commit source cleanup fails.
pub fn relocate_cache<C, F>(
    old_path: impl AsRef<Path>,
    new_path: impl AsRef<Path>,
    cancellation: &C,
    persist: F,
) -> Result<RelocationReport, StorageError>
where
    C: Cancellation + ?Sized,
    F: FnOnce(&Path) -> Result<(), String>,
{
    relocate_with_strategy(
        old_path.as_ref(),
        new_path.as_ref(),
        cancellation,
        persist,
        RelocationStrategy::Auto,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RelocationStrategy {
    Auto,
    #[cfg(test)]
    ForceCopy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransferKind {
    Renamed,
    Copied,
}

fn relocate_with_strategy<C, F>(
    old_path: &Path,
    new_path: &Path,
    cancellation: &C,
    persist: F,
    strategy: RelocationStrategy,
) -> Result<RelocationReport, StorageError>
where
    C: Cancellation + ?Sized,
    F: FnOnce(&Path) -> Result<(), String>,
{
    check_cancelled(cancellation)?;
    let (old_path, new_path) = prepare_relocation(old_path, new_path)?;
    check_cancelled(cancellation)?;
    let usage = measure_disk_usage(&old_path, cancellation)?;
    check_cancelled(cancellation)?;

    let transfer = match strategy {
        RelocationStrategy::Auto => {
            ensure_destination_absent(&new_path)?;
            match fs::rename(&old_path, &new_path) {
                Ok(()) => TransferKind::Renamed,
                Err(error) if is_cross_device(&error) => {
                    copy_into_destination(&old_path, &new_path, cancellation)?;
                    TransferKind::Copied
                }
                Err(error) => return Err(io_error("rename cache directory", error)),
            }
        }
        #[cfg(test)]
        RelocationStrategy::ForceCopy => {
            copy_into_destination(&old_path, &new_path, cancellation)?;
            TransferKind::Copied
        }
    };

    let persistence_failure = match catch_unwind(AssertUnwindSafe(|| persist(&new_path))) {
        Ok(Ok(())) => None,
        Ok(Err(message)) => Some(bounded_text(&message)),
        Err(_) => Some(String::from("persistence callback panicked")),
    };

    if let Some(persistence) = persistence_failure {
        let rollback = match transfer {
            TransferKind::Renamed => rollback_rename(&new_path, &old_path),
            TransferKind::Copied => remove_path_if_exists(&new_path),
        };
        return match rollback {
            Ok(()) => Err(StorageError::PersistenceFailed {
                message: persistence,
            }),
            Err(error) => Err(StorageError::RollbackFailed {
                persistence,
                rollback: bounded_text(&error.to_string()),
            }),
        };
    }

    if transfer == TransferKind::Copied {
        remove_path_if_exists(&old_path).map_err(|error| StorageError::CommittedCleanupFailed {
            message: bounded_text(&error.to_string()),
        })?;
    }

    Ok(RelocationReport {
        bytes: usage.bytes,
        files: usage.files,
        used_copy_fallback: transfer == TransferKind::Copied,
    })
}

fn prepare_relocation(
    old_path: &Path,
    new_path: &Path,
) -> Result<(PathBuf, PathBuf), StorageError> {
    let old_path = normalize_absolute(old_path, "relocation source", false)?;
    let new_path = normalize_absolute(new_path, "relocation destination", false)?;
    if paths_overlap(&old_path, &new_path) {
        return Err(StorageError::PathsOverlap);
    }

    let source = optional_symlink_metadata(&old_path, "inspect relocation source")?
        .ok_or(StorageError::SourceMissing)?;
    reject_root_type(&source)?;
    ensure_destination_absent(&new_path)?;

    let canonical_old = fs::canonicalize(&old_path)
        .map_err(|error| io_error("canonicalize relocation source", error))?;
    let canonical_new = canonicalize_candidate(&new_path)?;
    if paths_overlap(&canonical_old, &canonical_new) {
        return Err(StorageError::PathsOverlap);
    }

    let parent = new_path
        .parent()
        .ok_or(StorageError::DangerousPath("destination has no parent"))?;
    fs::create_dir_all(parent)
        .map_err(|error| io_error("create relocation parent directory", error))?;
    let parent_metadata = fs::symlink_metadata(parent)
        .map_err(|error| io_error("inspect relocation parent directory", error))?;
    if !parent_metadata.file_type().is_dir() {
        return Err(StorageError::NotDirectory);
    }

    let canonical_new = canonicalize_candidate(&new_path)?;
    if paths_overlap(&canonical_old, &canonical_new) {
        return Err(StorageError::PathsOverlap);
    }
    ensure_destination_absent(&new_path)?;
    Ok((old_path, new_path))
}

fn copy_into_destination<C>(
    old_path: &Path,
    new_path: &Path,
    cancellation: &C,
) -> Result<(), StorageError>
where
    C: Cancellation + ?Sized,
{
    check_cancelled(cancellation)?;
    ensure_destination_absent(new_path)?;
    let parent = new_path
        .parent()
        .ok_or(StorageError::DangerousPath("destination has no parent"))?;
    let temporary = create_temporary_sibling(parent)?;

    let copy_result = copy_tree_contents(old_path, &temporary, cancellation)
        .and_then(|()| check_cancelled(cancellation))
        .and_then(|()| ensure_destination_absent(new_path))
        .and_then(|()| {
            fs::rename(&temporary, new_path)
                .map_err(|error| io_error("install copied cache directory", error))
        });

    match copy_result {
        Ok(()) => Ok(()),
        Err(original) => match remove_path_if_exists(&temporary) {
            Ok(()) => Err(original),
            Err(cleanup) => Err(StorageError::TemporaryCleanupFailed {
                original: bounded_text(&original.to_string()),
                cleanup: bounded_text(&cleanup.to_string()),
            }),
        },
    }
}

fn copy_tree_contents<C>(
    source_root: &Path,
    destination_root: &Path,
    cancellation: &C,
) -> Result<(), StorageError>
where
    C: Cancellation + ?Sized,
{
    let mut stack = vec![(source_root.to_path_buf(), destination_root.to_path_buf())];
    while let Some((source, destination)) = stack.pop() {
        check_cancelled(cancellation)?;
        let source_metadata = fs::symlink_metadata(&source)
            .map_err(|error| io_error("inspect copy source directory", error))?;
        if !source_metadata.file_type().is_dir() {
            return Err(StorageError::NotDirectory);
        }

        let entries = fs::read_dir(&source).map_err(|error| io_error("read copy source", error))?;
        for entry in entries {
            check_cancelled(cancellation)?;
            let entry = entry.map_err(|error| io_error("read copy entry", error))?;
            let source_path = entry.path();
            let destination_path = destination.join(entry.file_name());
            let metadata = fs::symlink_metadata(&source_path)
                .map_err(|error| io_error("inspect copy entry", error))?;
            let file_type = metadata.file_type();
            if file_type.is_dir() {
                fs::create_dir(&destination_path)
                    .map_err(|error| io_error("create copied directory", error))?;
                stack.push((source_path, destination_path));
            } else if file_type.is_file() {
                fs::copy(&source_path, &destination_path)
                    .map_err(|error| io_error("copy cache file", error))?;
            } else if file_type.is_symlink() {
                copy_symlink(&source_path, &destination_path, &file_type)?;
            } else {
                return Err(StorageError::UnsupportedFileType);
            }
        }
    }
    Ok(())
}

fn rollback_rename(destination: &Path, source: &Path) -> Result<(), StorageError> {
    if optional_symlink_metadata(source, "inspect rollback source")?.is_some() {
        return Err(StorageError::DestinationExists);
    }
    let destination_metadata =
        optional_symlink_metadata(destination, "inspect rollback destination")?
            .ok_or(StorageError::SourceMissing)?;
    reject_root_type(&destination_metadata)?;
    fs::rename(destination, source).map_err(|error| io_error("roll back cache rename", error))
}

fn remove_path_if_exists(path: &Path) -> Result<(), StorageError> {
    let Some(metadata) = optional_symlink_metadata(path, "inspect cleanup root")? else {
        return Ok(());
    };
    if metadata.file_type().is_dir() {
        remove_tree(path, false, &NeverCancelled)?;
    } else {
        remove_non_directory(path, &metadata.file_type())
            .map_err(|error| io_error("remove cleanup entry", error))?;
    }
    Ok(())
}

fn remove_tree<C>(
    root: &Path,
    preserve_root: bool,
    cancellation: &C,
) -> Result<DiskUsage, StorageError>
where
    C: Cancellation + ?Sized,
{
    enum Step {
        Visit(PathBuf),
        RemoveDirectory(PathBuf),
    }

    let mut usage = DiskUsage::default();
    let mut stack = vec![Step::Visit(root.to_path_buf())];
    while let Some(step) = stack.pop() {
        check_cancelled(cancellation)?;
        match step {
            Step::Visit(path) => {
                let metadata = fs::symlink_metadata(&path)
                    .map_err(|error| io_error("inspect removal entry", error))?;
                if metadata.file_type().is_dir() {
                    stack.push(Step::RemoveDirectory(path.clone()));
                    let entries = fs::read_dir(&path)
                        .map_err(|error| io_error("read removal directory", error))?;
                    for entry in entries {
                        check_cancelled(cancellation)?;
                        let entry = entry.map_err(|error| io_error("read removal entry", error))?;
                        stack.push(Step::Visit(entry.path()));
                    }
                } else {
                    add_usage(&mut usage, &metadata)?;
                    remove_non_directory(&path, &metadata.file_type())
                        .map_err(|error| io_error("remove cache entry", error))?;
                }
            }
            Step::RemoveDirectory(path) => {
                if preserve_root && path == root {
                    continue;
                }
                let metadata = fs::symlink_metadata(&path)
                    .map_err(|error| io_error("inspect removal directory", error))?;
                if metadata.file_type().is_dir() {
                    fs::remove_dir(&path)
                        .map_err(|error| io_error("remove cache directory", error))?;
                } else {
                    add_usage(&mut usage, &metadata)?;
                    remove_non_directory(&path, &metadata.file_type())
                        .map_err(|error| io_error("remove replaced cache entry", error))?;
                }
            }
        }
    }
    Ok(usage)
}

fn add_usage(usage: &mut DiskUsage, metadata: &Metadata) -> Result<(), StorageError> {
    usage.bytes = usage
        .bytes
        .checked_add(metadata.len())
        .ok_or(StorageError::AccountingOverflow)?;
    usage.files = usage
        .files
        .checked_add(1)
        .ok_or(StorageError::AccountingOverflow)?;
    Ok(())
}

fn check_cancelled<C>(cancellation: &C) -> Result<(), StorageError>
where
    C: Cancellation + ?Sized,
{
    if cancellation.is_cancelled() {
        Err(StorageError::Cancelled)
    } else {
        Ok(())
    }
}

fn reject_root_type(metadata: &Metadata) -> Result<(), StorageError> {
    let file_type = metadata.file_type();
    if file_type.is_symlink() {
        Err(StorageError::SymlinkRoot)
    } else if !file_type.is_dir() {
        Err(StorageError::NotDirectory)
    } else {
        Ok(())
    }
}

fn optional_symlink_metadata(
    path: &Path,
    operation: &'static str,
) -> Result<Option<Metadata>, StorageError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => Ok(Some(metadata)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(io_error(operation, error)),
    }
}

fn ensure_destination_absent(path: &Path) -> Result<(), StorageError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(StorageError::SymlinkRoot),
        Ok(_) => Err(StorageError::DestinationExists),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(io_error("inspect relocation destination", error)),
    }
}

fn normalize_absolute(
    path: &Path,
    role: &'static str,
    allow_filesystem_root: bool,
) -> Result<PathBuf, StorageError> {
    if path.as_os_str().is_empty() {
        return Err(StorageError::DangerousPath("path is empty"));
    }
    if !path.is_absolute() {
        return Err(StorageError::PathNotAbsolute(role));
    }

    let mut prefix: Option<OsString> = None;
    let mut has_root = false;
    let mut parts: Vec<OsString> = Vec::new();
    for component in path.components() {
        match component {
            Component::Prefix(value) => prefix = Some(value.as_os_str().to_os_string()),
            Component::RootDir => has_root = true,
            Component::CurDir => {}
            Component::ParentDir => {
                parts.pop();
            }
            Component::Normal(value) => parts.push(value.to_os_string()),
        }
    }

    if !allow_filesystem_root && parts.is_empty() {
        return Err(StorageError::DangerousPath(
            "filesystem root is not allowed",
        ));
    }

    let mut normalized = PathBuf::new();
    if let Some(prefix) = prefix {
        normalized.push(prefix);
    }
    if has_root {
        normalized.push(Path::new(std::path::MAIN_SEPARATOR_STR));
    }
    for part in parts {
        normalized.push(part);
    }
    Ok(normalized)
}

fn paths_overlap(left: &Path, right: &Path) -> bool {
    left == right || left.starts_with(right) || right.starts_with(left)
}

fn canonicalize_candidate(path: &Path) -> Result<PathBuf, StorageError> {
    let mut cursor = path.to_path_buf();
    let mut suffix = Vec::new();
    loop {
        match fs::canonicalize(&cursor) {
            Ok(mut canonical) => {
                for part in suffix.iter().rev() {
                    canonical.push(part);
                }
                return normalize_absolute(&canonical, "canonical path", true);
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                let file_name = cursor
                    .file_name()
                    .ok_or(StorageError::DangerousPath("path has no existing ancestor"))?;
                suffix.push(file_name.to_os_string());
                cursor = cursor
                    .parent()
                    .ok_or(StorageError::DangerousPath("path has no existing ancestor"))?
                    .to_path_buf();
            }
            Err(error) => return Err(io_error("canonicalize relocation destination", error)),
        }
    }
}

fn create_temporary_sibling(parent: &Path) -> Result<PathBuf, StorageError> {
    static NEXT_TEMPORARY_ID: AtomicU64 = AtomicU64::new(1);
    const MAX_ATTEMPTS: usize = 128;

    for _ in 0..MAX_ATTEMPTS {
        let id = NEXT_TEMPORARY_ID.fetch_add(1, Ordering::Relaxed);
        let name = format!(
            ".cutlass-storage-relocation-{}-{id}.tmp",
            std::process::id()
        );
        let candidate = parent.join(name);
        match fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(io_error("create temporary relocation directory", error)),
        }
    }
    Err(StorageError::TemporaryNameExhausted)
}

fn is_cross_device(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::CrossesDevices
}

fn io_error(operation: &'static str, source: io::Error) -> StorageError {
    StorageError::Io { operation, source }
}

fn bounded_text(message: &str) -> String {
    if message.len() <= MAX_ERROR_MESSAGE_BYTES {
        return message.to_owned();
    }
    let mut end = MAX_ERROR_MESSAGE_BYTES.saturating_sub(3);
    while !message.is_char_boundary(end) {
        end = end.saturating_sub(1);
    }
    let mut bounded = message[..end].to_owned();
    bounded.push_str("...");
    bounded
}

#[cfg(unix)]
fn copy_symlink(source: &Path, destination: &Path, _: &FileType) -> Result<(), StorageError> {
    let target =
        fs::read_link(source).map_err(|error| io_error("read cache symbolic link", error))?;
    std::os::unix::fs::symlink(target, destination)
        .map_err(|error| io_error("copy cache symbolic link", error))
}

#[cfg(windows)]
fn copy_symlink(
    source: &Path,
    destination: &Path,
    file_type: &FileType,
) -> Result<(), StorageError> {
    use std::os::windows::fs::FileTypeExt;

    let target =
        fs::read_link(source).map_err(|error| io_error("read cache symbolic link", error))?;
    if file_type.is_symlink_dir() {
        std::os::windows::fs::symlink_dir(target, destination)
            .map_err(|error| io_error("copy cache directory symbolic link", error))
    } else {
        std::os::windows::fs::symlink_file(target, destination)
            .map_err(|error| io_error("copy cache file symbolic link", error))
    }
}

#[cfg(not(any(unix, windows)))]
fn copy_symlink(_: &Path, _: &Path, _: &FileType) -> Result<(), StorageError> {
    Err(StorageError::UnsupportedFileType)
}

#[cfg(windows)]
fn remove_non_directory(path: &Path, file_type: &FileType) -> io::Result<()> {
    use std::os::windows::fs::FileTypeExt;

    if file_type.is_symlink_dir() {
        fs::remove_dir(path)
    } else {
        fs::remove_file(path)
    }
}

#[cfg(not(windows))]
fn remove_non_directory(path: &Path, _: &FileType) -> io::Result<()> {
    fs::remove_file(path)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    struct TestDirectory {
        path: PathBuf,
    }

    impl TestDirectory {
        fn new() -> Self {
            static NEXT_ID: AtomicU64 = AtomicU64::new(1);
            for _ in 0..128 {
                let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
                let path = std::env::temp_dir()
                    .join(format!("cutlass-storage-unit-{}-{id}", std::process::id()));
                match fs::create_dir(&path) {
                    Ok(()) => return Self { path },
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                    Err(error) => panic!("create test directory: {error}"),
                }
            }
            panic!("could not allocate test directory");
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn shared_layout_generation_exhaustion_fails_closed() {
        let temporary = TestDirectory::new();
        let original = StorageLayout::new(temporary.path.join("original")).unwrap();
        let replacement = StorageLayout::new(temporary.path.join("replacement")).unwrap();
        let shared = SharedStorageLayout::with_generation_for_test(original.clone(), u64::MAX);

        assert_eq!(
            shared.replace(u64::MAX, replacement),
            Err(SharedStorageLayoutError::GenerationExhausted)
        );

        let update_was_called = std::cell::Cell::new(false);
        assert_eq!(
            shared.update(u64::MAX, |_| update_was_called.set(true)),
            Err(SharedStorageLayoutError::GenerationExhausted)
        );
        assert!(!update_was_called.get());

        let snapshot = shared.snapshot();
        assert_eq!(snapshot.generation(), u64::MAX);
        assert_eq!(snapshot.layout(), &original);
    }

    #[test]
    fn forced_copy_strategy_exercises_fallback_without_second_volume() {
        let temporary = TestDirectory::new();
        let old_path = temporary.path.join("old");
        let new_path = temporary.path.join("new");
        fs::create_dir(&old_path).unwrap();
        fs::create_dir(old_path.join("nested")).unwrap();
        fs::write(old_path.join("nested").join("data"), b"fallback").unwrap();

        let report = relocate_with_strategy(
            &old_path,
            &new_path,
            &NeverCancelled,
            |completed| {
                assert_eq!(
                    fs::read(completed.join("nested").join("data")).unwrap(),
                    b"fallback"
                );
                assert!(old_path.exists(), "copy source must survive until persist");
                Ok(())
            },
            RelocationStrategy::ForceCopy,
        )
        .unwrap();

        assert!(report.used_copy_fallback);
        assert_eq!(report.bytes, 8);
        assert_eq!(report.files, 1);
        assert!(!old_path.exists());
        assert_eq!(
            fs::read(new_path.join("nested").join("data")).unwrap(),
            b"fallback"
        );
    }

    #[test]
    fn forced_copy_persistence_failure_removes_destination_not_source() {
        let temporary = TestDirectory::new();
        let old_path = temporary.path.join("old");
        let new_path = temporary.path.join("new");
        fs::create_dir(&old_path).unwrap();
        fs::write(old_path.join("data"), b"authoritative").unwrap();

        let error = relocate_with_strategy(
            &old_path,
            &new_path,
            &NeverCancelled,
            |_| Err("settings unavailable".into()),
            RelocationStrategy::ForceCopy,
        )
        .unwrap_err();

        assert!(matches!(error, StorageError::PersistenceFailed { .. }));
        assert_eq!(fs::read(old_path.join("data")).unwrap(), b"authoritative");
        assert!(!new_path.exists());
    }

    #[test]
    fn persistence_errors_are_bounded() {
        let temporary = TestDirectory::new();
        let old_path = temporary.path.join("old");
        let new_path = temporary.path.join("new");
        fs::create_dir(&old_path).unwrap();

        let error = relocate_cache(&old_path, &new_path, &NeverCancelled, |_| {
            Err("x".repeat(16_384))
        })
        .unwrap_err();
        assert!(error.to_string().len() < 400);
        assert!(old_path.exists());
        assert!(!new_path.exists());
    }

    #[test]
    fn rollback_failure_is_reported_explicitly() {
        let temporary = TestDirectory::new();
        let old_path = temporary.path.join("old");
        let new_path = temporary.path.join("new");
        fs::create_dir(&old_path).unwrap();
        fs::write(old_path.join("original"), b"data").unwrap();

        let error = relocate_cache(&old_path, &new_path, &NeverCancelled, |_| {
            fs::create_dir(&old_path).unwrap();
            Err("settings unavailable".into())
        })
        .unwrap_err();

        assert!(matches!(error, StorageError::RollbackFailed { .. }));
        assert_eq!(fs::read(new_path.join("original")).unwrap(), b"data");
    }
}
