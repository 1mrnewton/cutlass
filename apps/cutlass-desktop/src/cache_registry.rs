//! Shared desktop cache inventory and exact per-cache clearing.
//!
//! The registry owns the startup storage layout. It deliberately does not
//! reload settings or relocate live writers: those require a later
//! coordination slice. All public operations are synchronous and intended for
//! an off-UI worker thread.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError, bounded};
use cutlass_cloud::cache::DownloadCache;
use cutlass_storage::{
    CacheDescriptor, CacheId, CacheKind, CacheTier, StorageLayout, cache_descriptors, clear_cache,
    measure_disk_usage,
};
use serde::ser::SerializeStruct;
use serde::{Serialize, Serializer};

use crate::AppWindow;
use crate::preview_worker::{PreviewCacheStats, WorkerHandle};

const GATE_WAIT_SLICE: Duration = Duration::from_millis(25);
const GATE_WAIT_TIMEOUT: Duration = Duration::from_secs(10);
const UI_WAIT_SLICE: Duration = Duration::from_millis(25);
const UI_WAIT_TIMEOUT: Duration = Duration::from_secs(2);
const MAX_ERROR_CHARS: usize = 240;
const UI_OPERATION_PENDING: u8 = 0;
const UI_OPERATION_RUNNING: u8 = 1;
const UI_OPERATION_ABANDONED: u8 = 2;

/// One cache's current usage and immutable descriptor metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CacheSnapshot {
    pub(crate) id: CacheId,
    pub(crate) label: &'static str,
    pub(crate) kind: CacheKind,
    pub(crate) tier: CacheTier,
    pub(crate) path: Option<PathBuf>,
    pub(crate) bytes: u64,
    pub(crate) entries: u64,
    pub(crate) files: u64,
}

impl Serialize for CacheSnapshot {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut fields = serializer
            .serialize_struct("CacheSnapshot", if self.path.is_some() { 8 } else { 7 })?;
        fields.serialize_field("cache_id", self.id.as_str())?;
        fields.serialize_field("label", self.label)?;
        fields.serialize_field("kind", cache_kind_key(self.kind))?;
        fields.serialize_field("tier", cache_tier_key(self.tier))?;
        if let Some(path) = &self.path {
            fields.serialize_field("path", &path.to_string_lossy())?;
        }
        fields.serialize_field("bytes", &self.bytes)?;
        fields.serialize_field("entries", &self.entries)?;
        fields.serialize_field("files", &self.files)?;
        fields.end()
    }
}

/// Exact pre-clear accounting plus a current snapshot when it was practical
/// to collect one after the operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CacheClearReport {
    pub(crate) id: CacheId,
    pub(crate) removed_bytes: u64,
    pub(crate) removed_entries: u64,
    pub(crate) removed_files: u64,
    pub(crate) current: Option<CacheSnapshot>,
}

impl Serialize for CacheClearReport {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut fields = serializer.serialize_struct(
            "CacheClearReport",
            if self.current.is_some() { 5 } else { 4 },
        )?;
        fields.serialize_field("cache_id", self.id.as_str())?;
        fields.serialize_field("removed_bytes", &self.removed_bytes)?;
        fields.serialize_field("removed_entries", &self.removed_entries)?;
        fields.serialize_field("removed_files", &self.removed_files)?;
        if let Some(current) = &self.current {
            fields.serialize_field("cache", current)?;
        }
        fields.end()
    }
}

/// The single cloneable desktop cache service shared by agent and Settings.
#[derive(Clone)]
pub(crate) struct CacheRegistry {
    layout: Arc<StorageLayout>,
    app: slint::Weak<AppWindow>,
    preview: WorkerHandle,
    download_cache: Arc<DownloadCache>,
    operation_gate: Arc<Mutex<()>>,
}

impl CacheRegistry {
    pub(crate) fn new(
        layout: StorageLayout,
        app: slint::Weak<AppWindow>,
        preview: WorkerHandle,
        download_cache: Arc<DownloadCache>,
    ) -> Result<Self, String> {
        validate_layout(&layout)?;
        validate_download_root(&layout, download_cache.root())?;
        Ok(Self {
            layout: Arc::new(layout),
            app,
            preview,
            download_cache,
            operation_gate: Arc::new(Mutex::new(())),
        })
    }

    /// The immutable storage root selected when this registry was created.
    pub(crate) fn startup_storage_root(&self) -> &Path {
        self.layout.root()
    }

    /// Resolve one cache's immutable startup path.
    ///
    /// Memory caches deliberately return a short error instead of inventing a
    /// filesystem location.
    pub(crate) fn cache_path(&self, id: CacheId) -> Result<PathBuf, String> {
        disk_path(&self.layout, id)
    }

    /// Snapshot every cache in stable descriptor order.
    ///
    /// This method dispatches Slint-owned cache reads to the UI event loop and
    /// must therefore be called from a worker thread, never from the UI thread.
    pub(crate) fn snapshot_all(&self, cancel: &AtomicBool) -> Result<Vec<CacheSnapshot>, String> {
        let _operation = acquire_operation_gate(
            &self.operation_gate,
            cancel,
            GATE_WAIT_TIMEOUT,
            GATE_WAIT_SLICE,
        )?;
        self.validate_runtime_wiring()?;
        self.snapshot_all_locked(cancel)
    }

    /// Clear exactly one cache. There is intentionally no aggregate clear.
    pub(crate) fn clear(
        &self,
        id: CacheId,
        cancel: &AtomicBool,
    ) -> Result<CacheClearReport, String> {
        let _operation = acquire_operation_gate(
            &self.operation_gate,
            cancel,
            GATE_WAIT_TIMEOUT,
            GATE_WAIT_SLICE,
        )?;
        self.validate_runtime_wiring()?;
        ensure_not_cancelled(cancel, "cancelled before clearing the cache")?;

        let descriptor = id.descriptor();
        ensure_cache_can_be_cleared(id)?;

        let removed = match descriptor.kind {
            CacheKind::Memory => {
                ensure_memory_has_no_path(&self.layout, id)?;
                self.clear_memory_cache(id, cancel)?
            }
            CacheKind::Disk if id == CacheId::Download => self.clear_download_cache(cancel)?,
            CacheKind::Disk => clear_disk_contents(&self.layout, id, cancel)?,
        };

        // A cancellation that arrives after an exact clear should not erase
        // the useful accounting. Skip the optional post-clear read instead.
        let current = if cancel.load(Ordering::Acquire) {
            None
        } else {
            match self.snapshot_one_locked(id, cancel) {
                Ok(snapshot) => Some(snapshot),
                Err(error) => {
                    tracing::debug!(
                        cache = id.as_str(),
                        %error,
                        "cache cleared but its post-clear snapshot was unavailable"
                    );
                    None
                }
            }
        };
        Ok(CacheClearReport {
            id,
            removed_bytes: removed.bytes,
            removed_entries: removed.entries,
            removed_files: removed.files,
            current,
        })
    }

    /// Human-readable target shown before an irreversible cache clear.
    pub(crate) fn clear_approval_detail(&self, id: CacheId) -> String {
        let descriptor = id.descriptor();
        if id == CacheId::Download {
            return format!(
                "Cache: {}\nLocation: {}\n\nUnreferenced downloads are removed. Source files still used by a project are retained.",
                descriptor.label,
                self.download_cache.root().display()
            );
        }
        if cache_requires_reference_migration(id) {
            return format!(
                "Cache: {}\n\nClearing is blocked because existing projects may reference files in this cache.",
                descriptor.label
            );
        }
        match self.layout.resolve(id) {
            Some(path) => format!(
                "Cache: {}\nLocation: {}\n\nClearing runs immediately and cannot be undone from Cutlass.",
                descriptor.label,
                path.display()
            ),
            None => format!(
                "Cache: {}\nLocation: In-memory only\n\nClearing runs immediately and cannot be undone from Cutlass.",
                descriptor.label
            ),
        }
    }

    fn validate_runtime_wiring(&self) -> Result<(), String> {
        validate_download_root(&self.layout, self.download_cache.root())
    }

    fn clear_download_cache(&self, cancel: &AtomicBool) -> Result<CacheUsage, String> {
        ensure_not_cancelled(cancel, "cancelled before protecting project downloads")?;
        let saved = crate::download_safety::protect_saved_draft_downloads(
            self.download_cache.as_ref(),
            &crate::drafts::root_dir(),
        );
        if !saved.is_complete() {
            self.download_cache.block_destructive_operations();
            return Err(
                "saved project media inventory is incomplete; download clear was refused".into(),
            );
        }
        let project = self.preview.snapshot_project().ok_or_else(|| {
            "current project is unavailable; download clear was refused".to_string()
        })?;
        let protected = crate::download_safety::protect_project_downloads(
            self.download_cache.as_ref(),
            &project,
        );
        if protected.rejected != 0 {
            return Err(
                "current project media could not be protected; download clear was refused".into(),
            );
        }
        ensure_not_cancelled(cancel, "cancelled before clearing downloads")?;
        let report = self.download_cache.clear().map_err(|error| {
            bounded_error("download cache could not be cleared", &error.to_string())
        })?;
        Ok(CacheUsage {
            bytes: report.removed_bytes,
            entries: 0,
            files: report.removed_files,
        })
    }

    fn snapshot_all_locked(&self, cancel: &AtomicBool) -> Result<Vec<CacheSnapshot>, String> {
        let memory = self.memory_snapshots(cancel)?;
        let disk = snapshot_disk_caches(&self.layout, cancel)?;
        let mut disk = disk.into_iter();
        let mut snapshots = Vec::with_capacity(cache_descriptors().len());

        for descriptor in cache_descriptors() {
            ensure_not_cancelled(cancel, "cancelled while listing caches")?;
            let snapshot = match descriptor.kind {
                CacheKind::Memory => memory
                    .iter()
                    .find(|snapshot| snapshot.id == descriptor.id)
                    .cloned()
                    .ok_or_else(|| {
                        "cache registry is missing memory-cache statistics".to_string()
                    })?,
                CacheKind::Disk => disk
                    .next()
                    .ok_or_else(|| "cache registry is missing disk-cache statistics".to_string())?,
            };
            snapshots.push(snapshot);
        }
        if disk.next().is_some() {
            return Err("cache registry returned unexpected disk-cache statistics".into());
        }
        Ok(snapshots)
    }

    fn snapshot_one_locked(
        &self,
        id: CacheId,
        cancel: &AtomicBool,
    ) -> Result<CacheSnapshot, String> {
        let descriptor = id.descriptor();
        match descriptor.kind {
            CacheKind::Memory => {
                ensure_memory_has_no_path(&self.layout, id)?;
                let usage = match id {
                    CacheId::PreviewFrames => {
                        ensure_not_cancelled(cancel, "cancelled while reading preview cache")?;
                        let stats = self
                            .preview
                            .preview_cache_stats_with_cancel(cancel)
                            .map_err(|error| {
                                bounded_error("preview cache is unavailable", &error)
                            })?;
                        ensure_not_cancelled(cancel, "cancelled while reading preview cache")?;
                        preview_usage(stats)?
                    }
                    CacheId::LibraryThumbnails => self.dispatch_ui(cancel, |_app| {
                        thumbnail_usage(crate::thumbnails::cache_stats())
                    })?,
                    CacheId::TimelineFilmstrips => self.dispatch_ui(cancel, |_app| {
                        strip_usage(crate::strips::cache_stats().filmstrips)
                    })?,
                    CacheId::TimelineWaveforms => self.dispatch_ui(cancel, |_app| {
                        strip_usage(crate::strips::cache_stats().waves)
                    })?,
                    _ => return Err("disk cache was classified as memory".into()),
                };
                Ok(memory_snapshot(descriptor, usage))
            }
            CacheKind::Disk => snapshot_disk_cache(&self.layout, id, cancel),
        }
    }

    fn memory_snapshots(&self, cancel: &AtomicBool) -> Result<Vec<CacheSnapshot>, String> {
        ensure_not_cancelled(cancel, "cancelled while listing caches")?;
        let preview = self
            .preview
            .preview_cache_stats_with_cancel(cancel)
            .map_err(|error| bounded_error("preview cache is unavailable", &error))?;
        ensure_not_cancelled(cancel, "cancelled while listing caches")?;
        let ui = self.dispatch_ui(cancel, |_app| {
            let strips = crate::strips::cache_stats();
            Ok(UiCacheUsage {
                thumbnails: thumbnail_usage(crate::thumbnails::cache_stats())?,
                filmstrips: strip_usage(strips.filmstrips)?,
                waveforms: strip_usage(strips.waves)?,
            })
        })?;

        Ok(vec![
            memory_snapshot(CacheId::PreviewFrames.descriptor(), preview_usage(preview)?),
            memory_snapshot(CacheId::LibraryThumbnails.descriptor(), ui.thumbnails),
            memory_snapshot(CacheId::TimelineFilmstrips.descriptor(), ui.filmstrips),
            memory_snapshot(CacheId::TimelineWaveforms.descriptor(), ui.waveforms),
        ])
    }

    fn clear_memory_cache(&self, id: CacheId, cancel: &AtomicBool) -> Result<CacheUsage, String> {
        match id {
            CacheId::PreviewFrames => {
                ensure_not_cancelled(cancel, "cancelled before clearing preview cache")?;
                let stats = self
                    .preview
                    .clear_preview_cache_with_cancel(cancel)
                    .map_err(|error| bounded_error("preview cache could not be cleared", &error))?;
                preview_usage(stats)
            }
            CacheId::LibraryThumbnails => self.dispatch_ui(cancel, |_app| {
                thumbnail_usage(crate::thumbnails::clear_all())
            }),
            CacheId::TimelineFilmstrips => {
                self.dispatch_ui(
                    cancel,
                    |_app| strip_usage(crate::strips::clear_filmstrips()),
                )
            }
            CacheId::TimelineWaveforms => {
                self.dispatch_ui(cancel, |_app| strip_usage(crate::strips::clear_waveforms()))
            }
            _ => Err("disk cache was classified as memory".into()),
        }
    }

    fn dispatch_ui<T>(
        &self,
        cancel: &AtomicBool,
        operation: impl FnOnce(&AppWindow) -> Result<T, String> + Send + 'static,
    ) -> Result<T, String>
    where
        T: Send + 'static,
    {
        ensure_not_cancelled(cancel, "cancelled before reading the UI cache")?;

        let (response_tx, response_rx) = bounded::<Result<T, String>>(1);
        let state = Arc::new(AtomicU8::new(UI_OPERATION_PENDING));
        let event_loop_state = Arc::clone(&state);
        let app = self.app.clone();
        if let Err(error) = slint::invoke_from_event_loop(move || {
            // A clear that timed out while still queued must remain a no-op
            // when the event loop eventually reaches this closure.
            if event_loop_state
                .compare_exchange(
                    UI_OPERATION_PENDING,
                    UI_OPERATION_RUNNING,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_err()
            {
                return;
            }
            let response = match app.upgrade() {
                Some(app) => operation(&app),
                None => Err("the Cutlass window is closed".into()),
            };
            let _ = response_tx.try_send(response);
        }) {
            state.store(UI_OPERATION_ABANDONED, Ordering::Release);
            tracing::debug!(%error, "cache registry event-loop dispatch failed");
            return Err("the Cutlass event loop is unavailable".into());
        }

        wait_for_ui_response(&response_rx, cancel, &state, UI_WAIT_TIMEOUT, UI_WAIT_SLICE)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct CacheUsage {
    bytes: u64,
    entries: u64,
    files: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct UiCacheUsage {
    thumbnails: CacheUsage,
    filmstrips: CacheUsage,
    waveforms: CacheUsage,
}

fn memory_snapshot(descriptor: &CacheDescriptor, usage: CacheUsage) -> CacheSnapshot {
    CacheSnapshot {
        id: descriptor.id,
        label: descriptor.label,
        kind: descriptor.kind,
        tier: descriptor.tier,
        path: None,
        bytes: usage.bytes,
        entries: usage.entries,
        files: 0,
    }
}

fn snapshot_disk_caches(
    layout: &StorageLayout,
    cancel: &AtomicBool,
) -> Result<Vec<CacheSnapshot>, String> {
    cache_descriptors()
        .iter()
        .filter(|descriptor| descriptor.kind == CacheKind::Disk)
        .map(|descriptor| snapshot_disk_cache(layout, descriptor.id, cancel))
        .collect()
}

fn snapshot_disk_cache(
    layout: &StorageLayout,
    id: CacheId,
    cancel: &AtomicBool,
) -> Result<CacheSnapshot, String> {
    ensure_not_cancelled(cancel, "cancelled while measuring cache usage")?;
    let descriptor = id.descriptor();
    let path = disk_path(layout, id)?;
    let cancellation = || cancel.load(Ordering::Acquire);
    let usage = measure_disk_usage(&path, &cancellation)
        .map_err(|error| bounded_error("cache usage could not be measured", &error.to_string()))?;
    Ok(CacheSnapshot {
        id,
        label: descriptor.label,
        kind: descriptor.kind,
        tier: descriptor.tier,
        path: Some(path),
        bytes: usage.bytes,
        entries: 0,
        files: usage.files,
    })
}

fn clear_disk_contents(
    layout: &StorageLayout,
    id: CacheId,
    cancel: &AtomicBool,
) -> Result<CacheUsage, String> {
    let descriptor = id.descriptor();
    if descriptor.kind != CacheKind::Disk {
        return Err("memory cache cannot be cleared as a disk cache".into());
    }
    if descriptor.tier == CacheTier::UserData {
        return Err("user data cannot be cleared through the cache registry".into());
    }
    let path = disk_path(layout, id)?;
    let cancellation = || cancel.load(Ordering::Acquire);
    let report = clear_cache(&path, &cancellation)
        .map_err(|error| bounded_error("disk cache could not be cleared", &error.to_string()))?;
    Ok(CacheUsage {
        bytes: report.removed_bytes,
        entries: 0,
        files: report.removed_files,
    })
}

fn ensure_cache_can_be_cleared(id: CacheId) -> Result<(), String> {
    if id.descriptor().tier == CacheTier::UserData {
        return Err("user data cannot be cleared through the cache registry".into());
    }
    if cache_requires_reference_migration(id) {
        return Err(
            "cache clearing is unavailable because projects may reference its files".into(),
        );
    }
    Ok(())
}

pub(crate) fn cache_can_be_cleared(id: CacheId) -> bool {
    ensure_cache_can_be_cleared(id).is_ok()
}

fn cache_requires_reference_migration(id: CacheId) -> bool {
    matches!(id, CacheId::Luts | CacheId::Lottie | CacheId::Templates)
}

fn validate_layout(layout: &StorageLayout) -> Result<(), String> {
    for descriptor in cache_descriptors() {
        match (descriptor.kind, layout.resolve(descriptor.id)) {
            (CacheKind::Memory, None) => {}
            (CacheKind::Memory, Some(_)) => {
                return Err("memory cache unexpectedly resolved to a disk path".into());
            }
            (CacheKind::Disk, Some(path)) if path.is_absolute() => {}
            (CacheKind::Disk, Some(_)) => return Err("cache path must be absolute".into()),
            (CacheKind::Disk, None) => return Err("disk cache has no storage path".into()),
        }
    }
    Ok(())
}

fn validate_download_root(layout: &StorageLayout, download_root: &Path) -> Result<(), String> {
    let resolved = disk_path(layout, CacheId::Download)?;
    if resolved != download_root {
        return Err("download cache root does not match the startup storage layout".into());
    }
    Ok(())
}

fn ensure_memory_has_no_path(layout: &StorageLayout, id: CacheId) -> Result<(), String> {
    if id.descriptor().kind != CacheKind::Memory {
        return Err("disk cache was classified as memory".into());
    }
    if layout.resolve(id).is_some() {
        return Err("memory cache unexpectedly resolved to a disk path".into());
    }
    Ok(())
}

fn disk_path(layout: &StorageLayout, id: CacheId) -> Result<PathBuf, String> {
    if id.descriptor().kind != CacheKind::Disk {
        return Err("memory cache has no disk path".into());
    }
    let path = layout
        .resolve(id)
        .ok_or_else(|| "disk cache has no storage path".to_string())?;
    if !path.is_absolute() {
        return Err("cache path must be absolute".into());
    }
    Ok(path)
}

fn preview_usage(stats: PreviewCacheStats) -> Result<CacheUsage, String> {
    Ok(CacheUsage {
        bytes: stats.bytes,
        entries: exact_count(stats.entries)?,
        files: 0,
    })
}

fn thumbnail_usage(stats: crate::thumbnails::ThumbnailCacheStats) -> Result<CacheUsage, String> {
    Ok(CacheUsage {
        bytes: stats.estimated_rgba_bytes,
        entries: exact_count(stats.entry_count)?,
        files: 0,
    })
}

fn strip_usage(stats: crate::strips::StripImageCacheStats) -> Result<CacheUsage, String> {
    Ok(CacheUsage {
        bytes: stats.estimated_rgba_bytes,
        entries: exact_count(stats.entry_count)?,
        files: 0,
    })
}

fn exact_count(count: usize) -> Result<u64, String> {
    u64::try_from(count).map_err(|_| "cache entry count exceeds the reporting range".into())
}

fn acquire_operation_gate<'a>(
    gate: &'a Mutex<()>,
    cancel: &AtomicBool,
    timeout: Duration,
    wait_slice: Duration,
) -> Result<MutexGuard<'a, ()>, String> {
    let started = Instant::now();
    loop {
        ensure_not_cancelled(
            cancel,
            "cancelled while waiting for another cache operation",
        )?;
        match gate.try_lock() {
            Ok(guard) => return Ok(guard),
            Err(TryLockError::Poisoned(_)) => {
                return Err("cache operation gate is unavailable".into());
            }
            Err(TryLockError::WouldBlock) => {}
        }
        let remaining = timeout.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            return Err("another cache operation did not finish in time".into());
        }
        std::thread::sleep(wait_slice.min(remaining));
    }
}

fn wait_for_ui_response<T>(
    receiver: &Receiver<Result<T, String>>,
    cancel: &AtomicBool,
    state: &AtomicU8,
    timeout: Duration,
    wait_slice: Duration,
) -> Result<T, String> {
    let started = Instant::now();
    loop {
        if cancel.load(Ordering::Acquire)
            && state
                .compare_exchange(
                    UI_OPERATION_PENDING,
                    UI_OPERATION_ABANDONED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
        {
            return Err("cancelled while waiting for cache statistics".into());
        }
        let remaining = timeout.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            let abandoned_while_queued = state
                .compare_exchange(
                    UI_OPERATION_PENDING,
                    UI_OPERATION_ABANDONED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok();
            return Err(if abandoned_while_queued {
                "the Cutlass UI did not respond to the cache operation in time".into()
            } else {
                "the cache UI operation started but did not finish in time".into()
            });
        }
        match receiver.recv_timeout(wait_slice.min(remaining)) {
            // Once the UI has completed an operation, return its result even
            // if cancellation raced with delivery. In particular, a clear is
            // a commit point and must not be reported as though it never ran.
            Ok(response) => return response,
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                let _ = state.compare_exchange(
                    UI_OPERATION_PENDING,
                    UI_OPERATION_ABANDONED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
                return Err("the cache UI response channel closed".into());
            }
        }
    }
}

fn ensure_not_cancelled(cancel: &AtomicBool, message: &'static str) -> Result<(), String> {
    if cancel.load(Ordering::Acquire) {
        Err(message.into())
    } else {
        Ok(())
    }
}

fn bounded_error(prefix: &str, error: &str) -> String {
    let mut bounded: String = error.chars().take(MAX_ERROR_CHARS).collect();
    if error.chars().count() > MAX_ERROR_CHARS {
        bounded.push('…');
    }
    format!("{prefix}: {bounded}")
}

const fn cache_kind_key(kind: CacheKind) -> &'static str {
    match kind {
        CacheKind::Memory => "memory",
        CacheKind::Disk => "disk",
    }
}

const fn cache_tier_key(tier: CacheTier) -> &'static str {
    match tier {
        CacheTier::Disposable => "disposable",
        CacheTier::Redownloadable => "redownloadable",
        CacheTier::UserData => "user_data",
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use serde_json::json;

    use super::*;

    #[test]
    fn disk_snapshots_are_deterministic_and_missing_directories_are_zero() {
        let temporary = tempfile::tempdir().unwrap();
        let layout = StorageLayout::new(temporary.path()).unwrap();
        let download = layout.resolve(CacheId::Download).unwrap();
        let nested = download.join("stock");
        fs::create_dir_all(&nested).unwrap();
        fs::write(nested.join("clip.mp4"), b"12345").unwrap();
        fs::write(download.join("metadata.json"), b"123").unwrap();

        let snapshots = snapshot_disk_caches(&layout, &AtomicBool::new(false)).unwrap();
        assert_eq!(
            snapshots
                .iter()
                .map(|snapshot| snapshot.id)
                .collect::<Vec<_>>(),
            vec![
                CacheId::Proxies,
                CacheId::Download,
                CacheId::Catalog,
                CacheId::Luts,
                CacheId::Lottie,
                CacheId::Templates,
            ]
        );
        let download = snapshots
            .iter()
            .find(|snapshot| snapshot.id == CacheId::Download)
            .unwrap();
        assert_eq!(
            (download.bytes, download.files, download.entries),
            (8, 2, 0)
        );
        assert_eq!(
            snapshots
                .iter()
                .find(|snapshot| snapshot.id == CacheId::Catalog)
                .map(|snapshot| (snapshot.bytes, snapshot.files)),
            Some((0, 0))
        );
    }

    #[test]
    fn disk_clear_removes_only_the_requested_cache_and_reports_exact_usage() {
        let temporary = tempfile::tempdir().unwrap();
        let layout = StorageLayout::new(temporary.path()).unwrap();
        let proxies = layout.resolve(CacheId::Proxies).unwrap();
        let catalog = layout.resolve(CacheId::Catalog).unwrap();
        fs::create_dir_all(&proxies).unwrap();
        fs::create_dir_all(&catalog).unwrap();
        fs::write(proxies.join("proxy.mp4"), b"1234567").unwrap();
        fs::write(catalog.join("catalog.json"), b"keep").unwrap();

        let removed =
            clear_disk_contents(&layout, CacheId::Proxies, &AtomicBool::new(false)).unwrap();
        assert_eq!(
            removed,
            CacheUsage {
                bytes: 7,
                entries: 0,
                files: 1,
            }
        );
        assert!(proxies.is_dir());
        assert_eq!(fs::read_dir(&proxies).unwrap().count(), 0);
        assert_eq!(fs::read(catalog.join("catalog.json")).unwrap(), b"keep");
        assert_eq!(
            snapshot_disk_cache(&layout, CacheId::Proxies, &AtomicBool::new(false))
                .unwrap()
                .bytes,
            0
        );
    }

    #[test]
    fn disk_helpers_honor_cancellation_and_reject_memory_paths() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().to_path_buf();
        let mut layout = StorageLayout::new(&root).unwrap();
        let download_override = root.join("download-override");
        layout
            .set_override(CacheId::Download, &download_override)
            .unwrap();
        let cancelled = AtomicBool::new(true);
        assert!(
            snapshot_disk_caches(&layout, &cancelled)
                .unwrap_err()
                .contains("cancelled")
        );
        assert!(
            clear_disk_contents(&layout, CacheId::Download, &cancelled)
                .unwrap_err()
                .contains("cancelled")
        );
        assert!(
            disk_path(&layout, CacheId::PreviewFrames)
                .unwrap_err()
                .contains("memory")
        );
        assert_eq!(layout.root(), root);
        assert_eq!(
            disk_path(&layout, CacheId::Download).unwrap(),
            download_override
        );
    }

    #[test]
    fn cache_clear_policy_fails_closed_for_project_referenced_catalog_assets() {
        assert!(ensure_cache_can_be_cleared(CacheId::Proxies).is_ok());
        assert!(ensure_cache_can_be_cleared(CacheId::Download).is_ok());
        for id in [CacheId::Luts, CacheId::Lottie, CacheId::Templates] {
            assert!(!cache_can_be_cleared(id), "{id} must fail closed");
        }
    }

    #[test]
    fn operation_gate_wait_is_cancellable_and_bounded() {
        let gate = Mutex::new(());
        let held = gate.lock().unwrap();
        let cancelled = AtomicBool::new(true);
        assert!(
            acquire_operation_gate(
                &gate,
                &cancelled,
                Duration::from_millis(50),
                Duration::from_millis(1),
            )
            .unwrap_err()
            .contains("cancelled")
        );

        let active = AtomicBool::new(false);
        assert!(
            acquire_operation_gate(
                &gate,
                &active,
                Duration::from_millis(5),
                Duration::from_millis(1),
            )
            .unwrap_err()
            .contains("did not finish")
        );
        drop(held);
        assert!(
            acquire_operation_gate(
                &gate,
                &active,
                Duration::from_millis(5),
                Duration::from_millis(1),
            )
            .is_ok()
        );
    }

    #[test]
    fn ui_wait_marks_timed_out_or_cancelled_closures_abandoned() {
        let (_tx, rx) = bounded::<Result<(), String>>(1);
        let state = AtomicU8::new(UI_OPERATION_PENDING);
        let active = AtomicBool::new(false);
        assert!(
            wait_for_ui_response(
                &rx,
                &active,
                &state,
                Duration::from_millis(5),
                Duration::from_millis(1),
            )
            .unwrap_err()
            .contains("did not respond")
        );
        assert_eq!(state.load(Ordering::Acquire), UI_OPERATION_ABANDONED);

        let (_tx, rx) = bounded::<Result<(), String>>(1);
        let state = AtomicU8::new(UI_OPERATION_PENDING);
        let cancelled = AtomicBool::new(true);
        assert!(
            wait_for_ui_response(
                &rx,
                &cancelled,
                &state,
                Duration::from_secs(1),
                Duration::from_millis(1),
            )
            .unwrap_err()
            .contains("cancelled")
        );
        assert_eq!(state.load(Ordering::Acquire), UI_OPERATION_ABANDONED);
    }

    #[test]
    fn ui_wait_returns_a_claimed_clear_result_when_cancellation_races_delivery() {
        let (tx, rx) = bounded(1);
        tx.send(Ok(17_u64)).unwrap();
        let state = AtomicU8::new(UI_OPERATION_RUNNING);
        let cancelled = AtomicBool::new(true);

        assert_eq!(
            wait_for_ui_response(
                &rx,
                &cancelled,
                &state,
                Duration::from_millis(1),
                Duration::from_millis(1),
            )
            .unwrap(),
            17
        );
        assert_eq!(state.load(Ordering::Acquire), UI_OPERATION_RUNNING);
    }

    #[test]
    fn ui_wait_remains_finite_after_the_ui_claims_an_operation() {
        let (_tx, rx) = bounded::<Result<(), String>>(1);
        let state = AtomicU8::new(UI_OPERATION_RUNNING);
        let active = AtomicBool::new(false);

        assert!(
            wait_for_ui_response(
                &rx,
                &active,
                &state,
                Duration::from_millis(5),
                Duration::from_millis(1),
            )
            .unwrap_err()
            .contains("started but did not finish")
        );
        assert_eq!(state.load(Ordering::Acquire), UI_OPERATION_RUNNING);
    }

    #[test]
    fn dto_json_uses_stable_keys_and_omits_memory_paths() {
        let memory = CacheSnapshot {
            id: CacheId::PreviewFrames,
            label: "Preview frames",
            kind: CacheKind::Memory,
            tier: CacheTier::Disposable,
            path: None,
            bytes: 40,
            entries: 2,
            files: 0,
        };
        assert_eq!(
            serde_json::to_value(&memory).unwrap(),
            json!({
                "cache_id": "preview_frames",
                "label": "Preview frames",
                "kind": "memory",
                "tier": "disposable",
                "bytes": 40,
                "entries": 2,
                "files": 0
            })
        );

        let disk = CacheSnapshot {
            id: CacheId::Download,
            label: "Downloads",
            kind: CacheKind::Disk,
            tier: CacheTier::Redownloadable,
            path: Some(PathBuf::from("/tmp/cutlass-downloads")),
            bytes: 11,
            entries: 0,
            files: 1,
        };
        let report = CacheClearReport {
            id: CacheId::Download,
            removed_bytes: 99,
            removed_entries: 0,
            removed_files: 3,
            current: Some(disk),
        };
        assert_eq!(
            serde_json::to_value(report).unwrap(),
            json!({
                "cache_id": "download",
                "removed_bytes": 99,
                "removed_entries": 0,
                "removed_files": 3,
                "cache": {
                    "cache_id": "download",
                    "label": "Downloads",
                    "kind": "disk",
                    "tier": "redownloadable",
                    "path": "/tmp/cutlass-downloads",
                    "bytes": 11,
                    "entries": 0,
                    "files": 1
                }
            })
        );
    }

    #[test]
    fn download_cache_root_must_exactly_match_the_layout() {
        let temporary = tempfile::tempdir().unwrap();
        let layout = StorageLayout::new(temporary.path()).unwrap();
        let exact = layout.resolve(CacheId::Download).unwrap();
        assert_eq!(validate_download_root(&layout, &exact), Ok(()));
        assert_eq!(
            validate_download_root(&layout, &temporary.path().join("other")).unwrap_err(),
            "download cache root does not match the startup storage layout"
        );
    }
}
