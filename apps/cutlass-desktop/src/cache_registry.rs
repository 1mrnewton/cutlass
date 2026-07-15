//! Shared desktop cache inventory, clearing, and coordinated relocation.
//!
//! The registry owns the live storage layout and serializes every destructive
//! cache operation. All public operations are synchronous and intended for an
//! off-UI worker thread.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError, bounded};
use cutlass_cloud::cache::DownloadCache;
use cutlass_storage::{
    CacheDescriptor, CacheId, CacheKind, CacheTier, SharedStorageLayout, StorageError,
    StorageLayout, cache_descriptors, clear_cache, measure_disk_usage, relocate_cache,
};
use serde::ser::SerializeStruct;
use serde::{Serialize, Serializer};

use crate::AppWindow;
use crate::cache_references::{CacheReferenceReport, DraftReferenceReport};
use crate::preview_worker::{PreviewCacheStats, ProjectMaintenanceGuard, WorkerHandle};

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

/// Committed relocation accounting and the newly published cache generation.
#[allow(dead_code)] // Public(crate) DTO for the following UI/agent wiring slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CacheRelocationReport {
    pub(crate) id: CacheId,
    pub(crate) old_path: PathBuf,
    pub(crate) new_path: PathBuf,
    pub(crate) bytes: u64,
    pub(crate) files: u64,
    pub(crate) used_copy_fallback: bool,
    pub(crate) cleanup_warning: Option<String>,
    pub(crate) generation: u64,
    pub(crate) current: Option<CacheSnapshot>,
}

impl Serialize for CacheRelocationReport {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut fields = serializer.serialize_struct(
            "CacheRelocationReport",
            7 + usize::from(self.cleanup_warning.is_some()) + usize::from(self.current.is_some()),
        )?;
        fields.serialize_field("cache_id", self.id.as_str())?;
        fields.serialize_field("old_path", &self.old_path.to_string_lossy())?;
        fields.serialize_field("new_path", &self.new_path.to_string_lossy())?;
        fields.serialize_field("bytes", &self.bytes)?;
        fields.serialize_field("files", &self.files)?;
        fields.serialize_field("used_copy_fallback", &self.used_copy_fallback)?;
        if let Some(warning) = &self.cleanup_warning {
            fields.serialize_field("cleanup_warning", warning)?;
        }
        fields.serialize_field("generation", &self.generation)?;
        if let Some(current) = &self.current {
            fields.serialize_field("cache", current)?;
        }
        fields.end()
    }
}

/// The single cloneable desktop cache service shared by agent and Settings.
#[derive(Clone)]
pub(crate) struct CacheRegistry {
    layout: SharedStorageLayout,
    app: slint::Weak<AppWindow>,
    preview: WorkerHandle,
    download_cache: Arc<DownloadCache>,
    operation_gate: Arc<Mutex<()>>,
}

impl CacheRegistry {
    pub(crate) fn new(
        layout: SharedStorageLayout,
        app: slint::Weak<AppWindow>,
        preview: WorkerHandle,
        download_cache: Arc<DownloadCache>,
    ) -> Result<Self, String> {
        let snapshot = layout.snapshot();
        validate_layout(snapshot.layout())?;
        snapshot.layout().validate_filesystem().map_err(|error| {
            bounded_error(
                "cache layout failed filesystem validation",
                &error.to_string(),
            )
        })?;
        validate_download_root(snapshot.layout(), &download_cache.root())?;
        Ok(Self {
            layout,
            app,
            preview,
            download_cache,
            operation_gate: Arc::new(Mutex::new(())),
        })
    }

    /// The current default storage root.
    pub(crate) fn storage_root(&self) -> PathBuf {
        self.layout.snapshot().layout().root().to_path_buf()
    }

    /// Resolve one cache's current path.
    ///
    /// Memory caches deliberately return a short error instead of inventing a
    /// filesystem location.
    pub(crate) fn cache_path(&self, id: CacheId) -> Result<PathBuf, String> {
        let snapshot = self.layout.snapshot();
        disk_path(snapshot.layout(), id)
    }

    /// Run a short settings read-modify-write while excluding cache
    /// maintenance that persists storage overrides through the same file.
    ///
    /// This is deliberately non-blocking: Settings invokes it on the UI
    /// thread, so an agent-initiated relocation reports busy instead of
    /// freezing the window while a potentially long filesystem move finishes.
    pub(crate) fn try_with_settings_persistence<T>(
        &self,
        operation: impl FnOnce() -> T,
    ) -> Result<T, String> {
        try_with_operation_gate(&self.operation_gate, operation)
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
        let layout = self.layout.lease();
        validate_download_root(layout.layout(), &self.download_cache.root())?;
        self.snapshot_all_locked(layout.layout(), cancel)
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
        let layout = self.layout.lease();
        layout.layout().validate_filesystem().map_err(|error| {
            bounded_error(
                "cache layout failed filesystem validation",
                &error.to_string(),
            )
        })?;
        validate_download_root(layout.layout(), &self.download_cache.root())?;
        ensure_not_cancelled(cancel, "cancelled before clearing the cache")?;

        let descriptor = id.descriptor();
        ensure_cache_can_be_cleared(id)?;

        let removed = match descriptor.kind {
            CacheKind::Memory => {
                ensure_memory_has_no_path(layout.layout(), id)?;
                self.clear_memory_cache(id, cancel)?
            }
            CacheKind::Disk if id == CacheId::Download => {
                self.clear_download_cache(layout.layout(), cancel)?
            }
            CacheKind::Disk => clear_disk_contents(layout.layout(), id, cancel)?,
        };

        // A cancellation that arrives after an exact clear should not erase
        // the useful accounting. Skip the optional post-clear read instead.
        let current = if cancel.load(Ordering::Acquire) {
            None
        } else {
            match self.snapshot_one_locked(layout.layout(), id, cancel) {
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

    /// Relocate exactly one disk cache and publish its new layout generation.
    ///
    /// This method waits for storage users, freezes the preview worker in queue
    /// order, inventories every saved and live project reference, completes
    /// the filesystem move, persists settings, and only then publishes the new
    /// shared layout. It must run on a background maintenance thread.
    #[allow(dead_code)] // Public(crate) seam for the following UI/agent wiring slice.
    pub(crate) fn relocate(
        &self,
        id: CacheId,
        destination: &Path,
        config_path: &Path,
        cancel: &AtomicBool,
    ) -> Result<CacheRelocationReport, String> {
        let _operation = acquire_operation_gate(
            &self.operation_gate,
            cancel,
            GATE_WAIT_TIMEOUT,
            GATE_WAIT_SLICE,
        )?;
        ensure_not_cancelled(cancel, "cancelled before relocating the cache")?;
        ensure_cache_can_be_relocated(id)?;

        // This clone is deliberately detached. It must not hold a read lease
        // while transition waits for exclusive storage access.
        let snapshot = self.layout.snapshot();
        validate_download_root(snapshot.layout(), &self.download_cache.root())?;
        let (mut replacement, expected_generation) = snapshot.into_parts();
        let old_path = disk_path(&replacement, id)?;
        replacement
            .set_override(id, destination.to_path_buf())
            .map_err(|error| {
                bounded_error("cache relocation destination is unsafe", &error.to_string())
            })?;
        replacement.validate_filesystem().map_err(|error| {
            bounded_error(
                "cache relocation layout failed filesystem validation",
                &error.to_string(),
            )
        })?;
        let new_path = disk_path(&replacement, id)?;
        validate_relocation_paths(&old_path, &new_path)?;
        validate_relocation_destination(&new_path)?;

        // A malformed or unreadable config is never replaced with defaults.
        // Prepare the complete settings value before touching either cache.
        let mut settings = cutlass_settings::load(config_path)
            .map_err(|_| "cache relocation could not load the current settings".to_string())?;
        set_storage_path_override(&mut settings, id, new_path.clone())?;
        ensure_not_cancelled(cancel, "cancelled before relocating the cache")?;

        let mut project_guard: Option<ProjectMaintenanceGuard> = None;
        let mut committed = None;
        let transition = self.layout.transition(
            expected_generation,
            replacement,
            |old_layout, new_layout| -> Result<(), CacheRelocationFailure> {
                let transition_old =
                    disk_path(old_layout, id).map_err(CacheRelocationFailure::from_message)?;
                let transition_new =
                    disk_path(new_layout, id).map_err(CacheRelocationFailure::from_message)?;
                if transition_old != old_path || transition_new != new_path {
                    return Err(CacheRelocationFailure::from_message(
                        "storage layout changed before cache relocation",
                    ));
                }
                new_layout.validate_filesystem().map_err(|error| {
                    CacheRelocationFailure::with_detail(
                        "cache relocation layout failed filesystem validation",
                        &error.to_string(),
                    )
                })?;

                // Lock order is SharedStorageLayout -> project maintenance ->
                // DownloadCache (for the download target only).
                project_guard = Some(
                    self.preview
                        .begin_project_maintenance_with_cancel(cancel)
                        .map_err(|error| {
                            CacheRelocationFailure::with_detail(
                                "project maintenance could not begin",
                                &error,
                            )
                        })?,
                );
                ensure_not_cancelled(cancel, "cancelled before inventorying project references")
                    .map_err(CacheRelocationFailure::from_message)?;

                let saved = crate::cache_references::scan_saved_draft_references(
                    &crate::drafts::root_dir(),
                    old_layout,
                );
                ensure_not_cancelled(cancel, "cancelled while inventorying project references")
                    .map_err(CacheRelocationFailure::from_message)?;
                let live = crate::cache_references::analyze_project_references(
                    project_guard
                        .as_ref()
                        .ok_or_else(|| {
                            CacheRelocationFailure::from_message(
                                "project maintenance snapshot is unavailable",
                            )
                        })?
                        .project(),
                    old_layout,
                );
                validate_relocation_references(id, &saved, &live)?;
                ensure_not_cancelled(cancel, "cancelled before moving the cache")
                    .map_err(CacheRelocationFailure::from_message)?;

                let outcome = if id == CacheId::Download {
                    relocate_download_root(
                        self.download_cache.as_ref(),
                        &transition_old,
                        &transition_new,
                        cancel,
                        |_| persist_relocation_settings(config_path, &settings),
                    )?
                } else {
                    relocate_disk_root(&transition_old, &transition_new, cancel, |_| {
                        persist_relocation_settings(config_path, &settings)
                    })?
                };
                committed = Some(outcome);
                Ok(())
            },
        );

        // The guard lives outside the transition closure so preview cannot
        // resume in the gap between a completed move and layout publication.
        let generation = match transition {
            Ok(generation) => generation,
            Err(error) => {
                drop(project_guard);
                return Err(bounded_message(&format!(
                    "cache relocation could not be completed: {error}"
                )));
            }
        };
        let Some(committed) = committed else {
            drop(project_guard);
            return Err("cache relocation completed without accounting".into());
        };
        if id == CacheId::Proxies {
            if let Some(guard) = project_guard.as_mut() {
                guard.refresh_proxies_on_resume();
            }
        }
        drop(project_guard);

        let layout = self.layout.lease();
        if layout.generation() != generation {
            return Err("storage layout changed before relocation could be verified".into());
        }
        validate_download_root(layout.layout(), &self.download_cache.root())?;
        let current = if cancel.load(Ordering::Acquire) {
            None
        } else {
            match snapshot_disk_cache(layout.layout(), id, cancel) {
                Ok(snapshot) => Some(snapshot),
                Err(error) => {
                    tracing::debug!(
                        cache = id.as_str(),
                        %error,
                        "cache relocated but its post-move snapshot was unavailable"
                    );
                    None
                }
            }
        };

        Ok(CacheRelocationReport {
            id,
            old_path,
            new_path,
            bytes: committed.report.bytes,
            files: committed.report.files,
            used_copy_fallback: committed.report.used_copy_fallback,
            cleanup_warning: committed.cleanup_warning,
            generation,
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

    fn clear_download_cache(
        &self,
        layout: &StorageLayout,
        cancel: &AtomicBool,
    ) -> Result<CacheUsage, String> {
        // `clear` already holds the registry operation gate and storage-layout
        // lease. This atomic marker acquires no DownloadCache maintenance lock:
        // the remaining lock order is project maintenance, then the exact
        // DownloadCache maintenance acquired by `DownloadCache::clear`.
        self.download_cache.block_destructive_operations();
        let project_guard = self
            .preview
            .begin_project_maintenance_with_cancel(cancel)
            .map_err(|error| {
                bounded_error(
                    "project maintenance could not begin; download clear was refused",
                    &error,
                )
            })?;

        let result = (|| {
            ensure_not_cancelled(cancel, "cancelled before inventorying project downloads")?;
            let saved = crate::cache_references::scan_saved_draft_references(
                &crate::drafts::root_dir(),
                layout,
            );
            ensure_not_cancelled(cancel, "cancelled while inventorying project downloads")?;
            let live = crate::cache_references::analyze_project_references(
                project_guard.project(),
                layout,
            );
            clear_download_cache_from_inventories(
                self.download_cache.as_ref(),
                &saved,
                &live,
                cancel,
            )
        })();

        // Resume preview only after DownloadCache::clear has released its
        // maintenance guard, including every error and cancellation path.
        drop(project_guard);
        result
    }

    fn snapshot_all_locked(
        &self,
        layout: &StorageLayout,
        cancel: &AtomicBool,
    ) -> Result<Vec<CacheSnapshot>, String> {
        let memory = self.memory_snapshots(cancel)?;
        let disk = snapshot_disk_caches(layout, cancel)?;
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
        layout: &StorageLayout,
        id: CacheId,
        cancel: &AtomicBool,
    ) -> Result<CacheSnapshot, String> {
        let descriptor = id.descriptor();
        match descriptor.kind {
            CacheKind::Memory => {
                ensure_memory_has_no_path(layout, id)?;
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
            CacheKind::Disk => snapshot_disk_cache(layout, id, cancel),
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct CommittedRelocation {
    report: cutlass_storage::RelocationReport,
    cleanup_warning: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CacheRelocationFailure(String);

impl CacheRelocationFailure {
    fn from_message(message: impl Into<String>) -> Self {
        Self(bounded_message(&message.into()))
    }

    fn with_detail(prefix: &str, detail: &str) -> Self {
        Self(bounded_message(&format!("{prefix}: {detail}")))
    }
}

impl fmt::Display for CacheRelocationFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for CacheRelocationFailure {}

fn ensure_cache_can_be_relocated(id: CacheId) -> Result<(), String> {
    if !cache_descriptors()
        .iter()
        .any(|descriptor| descriptor.id == id)
    {
        return Err("unknown cache cannot be relocated".into());
    }
    let descriptor = id.descriptor();
    if descriptor.kind == CacheKind::Memory {
        return Err("memory cache cannot be relocated".into());
    }
    if descriptor.tier == CacheTier::UserData {
        return Err("user data cannot be relocated through the cache registry".into());
    }
    if !matches!(
        id,
        CacheId::Proxies
            | CacheId::Download
            | CacheId::Catalog
            | CacheId::Luts
            | CacheId::Lottie
            | CacheId::Templates
    ) {
        return Err("cache target is not relocatable".into());
    }
    Ok(())
}

fn validate_relocation_paths(old_path: &Path, new_path: &Path) -> Result<(), String> {
    if old_path == new_path || old_path.starts_with(new_path) || new_path.starts_with(old_path) {
        return Err("cache relocation source and destination must not overlap".into());
    }
    Ok(())
}

fn validate_relocation_destination(destination: &Path) -> Result<(), String> {
    match std::fs::symlink_metadata(destination) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            Err("cache relocation destination cannot be a symbolic link".into())
        }
        Ok(_) => Err("cache relocation destination already exists".into()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(_) => Err("cache relocation destination could not be inspected".into()),
    }
}

fn set_storage_path_override(
    settings: &mut cutlass_settings::Settings,
    id: CacheId,
    path: PathBuf,
) -> Result<(), String> {
    ensure_cache_can_be_relocated(id)?;
    let field = match id {
        CacheId::Proxies => &mut settings.storage.paths.proxies,
        CacheId::Download => &mut settings.storage.paths.download,
        CacheId::Catalog => &mut settings.storage.paths.catalog,
        CacheId::Luts => &mut settings.storage.paths.luts,
        CacheId::Lottie => &mut settings.storage.paths.lottie,
        CacheId::Templates => &mut settings.storage.paths.templates,
        _ => return Err("cache target has no storage settings field".into()),
    };
    *field = Some(path);
    Ok(())
}

fn persist_relocation_settings(
    config_path: &Path,
    settings: &cutlass_settings::Settings,
) -> Result<(), String> {
    cutlass_settings::save(config_path, settings)
        .map_err(|_| "storage settings could not be saved".to_string())
}

fn validate_relocation_references(
    id: CacheId,
    saved: &DraftReferenceReport,
    live: &CacheReferenceReport,
) -> Result<(), CacheRelocationFailure> {
    if !saved.is_complete() {
        return Err(CacheRelocationFailure::from_message(
            "saved project reference inventory is incomplete; cache relocation was refused",
        ));
    }
    if !live.is_complete() {
        return Err(CacheRelocationFailure::from_message(
            "current project reference inventory is incomplete; cache relocation was refused",
        ));
    }
    if !matches!(
        id,
        CacheId::Download | CacheId::Luts | CacheId::Lottie | CacheId::Templates
    ) {
        return Ok(());
    }

    let saved_references = saved.references.by_cache.get(&id).ok_or_else(|| {
        CacheRelocationFailure::from_message(
            "saved project reference inventory omitted the target cache",
        )
    })?;
    let live_references = live.by_cache.get(&id).ok_or_else(|| {
        CacheRelocationFailure::from_message(
            "current project reference inventory omitted the target cache",
        )
    })?;
    if !saved_references.is_empty() || !live_references.is_empty() {
        return Err(CacheRelocationFailure::from_message(
            "project files reference this cache; cache relocation was refused",
        ));
    }
    Ok(())
}

fn relocate_disk_root<F>(
    old_path: &Path,
    new_path: &Path,
    cancel: &AtomicBool,
    persist: F,
) -> Result<CommittedRelocation, CacheRelocationFailure>
where
    F: FnOnce(&Path) -> Result<(), String>,
{
    ensure_not_cancelled(cancel, "cancelled before creating the cache source")
        .map_err(CacheRelocationFailure::from_message)?;
    ensure_relocation_source_exists(old_path)?;
    let cancellation = || cancel.load(Ordering::Acquire);
    classify_relocation_result(relocate_cache(old_path, new_path, &cancellation, persist)).map_err(
        |error| {
            CacheRelocationFailure::with_detail(
                "cache filesystem relocation failed",
                &error.to_string(),
            )
        },
    )
}

fn relocate_download_root<F>(
    cache: &DownloadCache,
    expected_old_path: &Path,
    new_path: &Path,
    cancel: &AtomicBool,
    persist: F,
) -> Result<CommittedRelocation, CacheRelocationFailure>
where
    F: FnOnce(&Path) -> Result<(), String>,
{
    let mut committed = None;
    cache
        .switch_root(new_path, |old_path, destination| {
            if old_path != expected_old_path || destination != new_path {
                return Err(CacheRelocationFailure::from_message(
                    "download cache root changed before relocation",
                ));
            }
            committed = Some(relocate_disk_root(old_path, destination, cancel, persist)?);
            Ok(())
        })
        .map_err(|error| {
            CacheRelocationFailure::with_detail(
                "download cache root could not be switched",
                &error.to_string(),
            )
        })?;
    committed.ok_or_else(|| {
        CacheRelocationFailure::from_message(
            "download cache root switched without relocation accounting",
        )
    })
}

fn ensure_relocation_source_exists(path: &Path) -> Result<(), CacheRelocationFailure> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => std::fs::create_dir_all(path)
            .map_err(|error| {
                CacheRelocationFailure::with_detail(
                    "missing cache source could not be created",
                    &error.to_string(),
                )
            }),
        Err(error) => Err(CacheRelocationFailure::with_detail(
            "cache source could not be inspected",
            &error.to_string(),
        )),
    }
}

fn classify_relocation_result(
    result: Result<cutlass_storage::RelocationReport, StorageError>,
) -> Result<CommittedRelocation, StorageError> {
    match result {
        Ok(report) => Ok(CommittedRelocation {
            report,
            cleanup_warning: None,
        }),
        Err(error) => {
            let Some(report) = error.committed_relocation() else {
                return Err(error);
            };
            Ok(CommittedRelocation {
                report,
                cleanup_warning: Some(bounded_message(&format!(
                    "cache relocation committed, but old-copy cleanup did not finish: {error}"
                ))),
            })
        }
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

fn clear_download_cache_from_inventories(
    cache: &DownloadCache,
    saved: &DraftReferenceReport,
    live: &CacheReferenceReport,
    cancel: &AtomicBool,
) -> Result<CacheUsage, String> {
    prepare_download_cache_clear(cache, saved, live, cancel)?;
    if cancel.load(Ordering::Acquire) {
        cache.block_destructive_operations();
        return Err("cancelled before clearing downloads".into());
    }
    let report = cache.clear().map_err(|error| {
        bounded_error("download cache could not be cleared", &error.to_string())
    })?;
    Ok(CacheUsage {
        bytes: report.removed_bytes,
        entries: 0,
        files: report.removed_files,
    })
}

fn prepare_download_cache_clear(
    cache: &DownloadCache,
    saved: &DraftReferenceReport,
    live: &CacheReferenceReport,
    cancel: &AtomicBool,
) -> Result<(), String> {
    // Every attempt starts fail closed. A complete inventory is the only path
    // that deliberately heals a block left by startup or an earlier failure.
    cache.block_destructive_operations();
    ensure_not_cancelled(cancel, "cancelled before protecting project downloads")?;
    if !saved.is_complete() {
        return Err(
            "saved project media inventory is incomplete; download clear was refused".into(),
        );
    }
    if !live.is_complete() {
        return Err(
            "current project media inventory is incomplete; download clear was refused".into(),
        );
    }
    let saved_paths = saved
        .references
        .by_cache
        .get(&CacheId::Download)
        .ok_or_else(|| {
            "saved project media inventory omitted the download cache; download clear was refused"
                .to_string()
        })?;
    let live_paths = live.by_cache.get(&CacheId::Download).ok_or_else(|| {
        "current project media inventory omitted the download cache; download clear was refused"
            .to_string()
    })?;

    protect_download_references(cache, saved_paths, cancel)?;
    protect_download_references(cache, live_paths, cancel)?;
    ensure_not_cancelled(cancel, "cancelled before clearing downloads")?;
    cache.allow_destructive_operations();
    Ok(())
}

fn protect_download_references(
    cache: &DownloadCache,
    paths: &BTreeSet<PathBuf>,
    cancel: &AtomicBool,
) -> Result<(), String> {
    for path in paths {
        ensure_not_cancelled(cancel, "cancelled while protecting project downloads")?;
        cache.protect_path(path).map_err(|error| {
            bounded_error(
                "project download could not be protected",
                &error.to_string(),
            )
        })?;
    }
    Ok(())
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
        return Err("download cache root does not match the active storage layout".into());
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

fn try_with_operation_gate<T>(
    gate: &Mutex<()>,
    operation: impl FnOnce() -> T,
) -> Result<T, String> {
    let _guard = gate.try_lock().map_err(|error| match error {
        TryLockError::WouldBlock => {
            "a cache operation is in progress; settings were not saved".to_string()
        }
        TryLockError::Poisoned(_) => {
            "cache operation coordination is unavailable; settings were not saved".to_string()
        }
    })?;
    Ok(operation())
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

fn bounded_message(message: &str) -> String {
    let mut bounded: String = message.chars().take(MAX_ERROR_CHARS).collect();
    if message.chars().count() > MAX_ERROR_CHARS {
        bounded.push('…');
    }
    bounded
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

    fn saved_download_report(paths: &[PathBuf]) -> DraftReferenceReport {
        let mut report = DraftReferenceReport::default();
        report
            .references
            .by_cache
            .get_mut(&CacheId::Download)
            .unwrap()
            .extend(paths.iter().cloned());
        report
    }

    fn live_download_report(paths: &[PathBuf]) -> CacheReferenceReport {
        let mut report = CacheReferenceReport::default();
        report
            .by_cache
            .get_mut(&CacheId::Download)
            .unwrap()
            .extend(paths.iter().cloned());
        report
    }

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
    fn incomplete_missing_or_cancelled_download_inventory_stays_blocked() {
        let temporary = tempfile::tempdir().unwrap();
        let cache = DownloadCache::new(temporary.path().join("downloads"), 1_000);
        let complete_saved = saved_download_report(&[]);
        let complete_live = live_download_report(&[]);

        let incomplete_saved = DraftReferenceReport {
            skipped_or_errored: 1,
            ..complete_saved.clone()
        };
        let error = prepare_download_cache_clear(
            &cache,
            &incomplete_saved,
            &complete_live,
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert!(error.contains("saved"));
        assert!(error.contains("incomplete"));
        assert!(cache.destructive_operations_blocked());

        cache.allow_destructive_operations();
        let incomplete_live = CacheReferenceReport {
            counts: crate::cache_references::ReferenceCounts {
                rejected: 1,
                ..crate::cache_references::ReferenceCounts::default()
            },
            ..complete_live.clone()
        };
        let error = prepare_download_cache_clear(
            &cache,
            &complete_saved,
            &incomplete_live,
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert!(error.contains("current"));
        assert!(error.contains("incomplete"));
        assert!(cache.destructive_operations_blocked());

        cache.allow_destructive_operations();
        let mut missing_saved = complete_saved.clone();
        missing_saved.references.by_cache.remove(&CacheId::Download);
        let error = prepare_download_cache_clear(
            &cache,
            &missing_saved,
            &complete_live,
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert!(error.contains("omitted"));
        assert!(cache.destructive_operations_blocked());

        cache.allow_destructive_operations();
        let mut missing_live = complete_live.clone();
        missing_live.by_cache.remove(&CacheId::Download);
        let error = prepare_download_cache_clear(
            &cache,
            &complete_saved,
            &missing_live,
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert!(error.contains("omitted"));
        assert!(cache.destructive_operations_blocked());

        cache.allow_destructive_operations();
        let error = prepare_download_cache_clear(
            &cache,
            &complete_saved,
            &complete_live,
            &AtomicBool::new(true),
        )
        .unwrap_err();
        assert!(error.contains("cancelled"));
        assert!(cache.destructive_operations_blocked());
    }

    #[test]
    fn complete_download_inventory_heals_block_after_all_paths_are_protected() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("downloads");
        fs::create_dir_all(root.join("stock")).unwrap();
        let saved_path = root.join("stock/saved.mp4");
        let live_path = root.join("stock/live.mp4");
        fs::write(&saved_path, b"saved").unwrap();
        fs::write(&live_path, b"live").unwrap();
        let cache = DownloadCache::new(root, 1_000);
        cache.block_destructive_operations();

        prepare_download_cache_clear(
            &cache,
            &saved_download_report(std::slice::from_ref(&saved_path)),
            &live_download_report(std::slice::from_ref(&live_path)),
            &AtomicBool::new(false),
        )
        .unwrap();

        assert_eq!(cache.protected_path_count(), 2);
        assert!(!cache.destructive_operations_blocked());
    }

    #[test]
    fn download_protection_failure_leaves_cache_blocked() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("downloads");
        fs::create_dir_all(root.join("directory")).unwrap();
        let valid = root.join("valid.mp4");
        fs::write(&valid, b"valid").unwrap();
        let invalid = root.join("directory");
        let cache = DownloadCache::new(root, 1_000);

        let error = prepare_download_cache_clear(
            &cache,
            &saved_download_report(std::slice::from_ref(&valid)),
            &live_download_report(std::slice::from_ref(&invalid)),
            &AtomicBool::new(false),
        )
        .unwrap_err();

        assert!(error.contains("could not be protected"));
        assert_eq!(cache.protected_path_count(), 1);
        assert!(cache.destructive_operations_blocked());
    }

    #[test]
    fn protected_downloads_survive_clear_with_exact_accounting() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("downloads");
        fs::create_dir_all(root.join("stock")).unwrap();
        let saved_path = root.join("stock/saved.mp4");
        let live_path = root.join("stock/live.mp4");
        let disposable = root.join("stock/disposable.bin");
        fs::write(&saved_path, b"saved").unwrap();
        fs::write(&live_path, b"live").unwrap();
        fs::write(&disposable, b"discard").unwrap();
        let cache = DownloadCache::new(root, 1_000);

        let removed = clear_download_cache_from_inventories(
            &cache,
            &saved_download_report(std::slice::from_ref(&saved_path)),
            &live_download_report(std::slice::from_ref(&live_path)),
            &AtomicBool::new(false),
        )
        .unwrap();

        assert_eq!(
            removed,
            CacheUsage {
                bytes: 7,
                entries: 0,
                files: 1,
            }
        );
        assert_eq!(fs::read(saved_path).unwrap(), b"saved");
        assert_eq!(fs::read(live_path).unwrap(), b"live");
        assert!(!disposable.exists());
        assert!(!cache.destructive_operations_blocked());
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
    fn relocation_field_mapping_covers_every_disk_cache_and_preserves_other_settings() {
        let temporary = tempfile::tempdir().unwrap();
        let disk_ids = [
            CacheId::Proxies,
            CacheId::Download,
            CacheId::Catalog,
            CacheId::Luts,
            CacheId::Lottie,
            CacheId::Templates,
        ];
        let mut original = cutlass_settings::Settings::default();
        original.appearance.theme = cutlass_settings::ThemeChoice::Ember;
        original.ai.base_url = "http://localhost:11434/v1".into();
        original.ai.model = "qwen-test".into();
        original.account.base_url = "https://api.example.test".into();
        original.storage.root = Some(temporary.path().join("configured-root"));
        original.storage.download_quota_mib = 777;
        for id in disk_ids {
            set_storage_path_override(
                &mut original,
                id,
                temporary.path().join(format!("original-{}", id.as_str())),
            )
            .unwrap();
        }

        for id in disk_ids {
            let destination = temporary.path().join(format!("moved-{}", id.as_str()));
            let mut updated = original.clone();
            set_storage_path_override(&mut updated, id, destination.clone()).unwrap();

            assert_eq!(updated.appearance, original.appearance);
            assert_eq!(updated.ai, original.ai);
            assert_eq!(updated.providers, original.providers);
            assert_eq!(updated.account, original.account);
            assert_eq!(updated.storage.root, original.storage.root);
            assert_eq!(
                updated.storage.download_quota_mib,
                original.storage.download_quota_mib
            );
            for candidate in disk_ids {
                let expected = if candidate == id {
                    destination.as_path()
                } else {
                    original
                        .storage
                        .paths
                        .get(candidate.as_str())
                        .expect("seeded override")
                };
                assert_eq!(
                    updated.storage.paths.get(candidate.as_str()),
                    Some(expected),
                    "wrong field mapping for {id}"
                );
            }
        }
    }

    #[test]
    fn relocation_policy_refuses_memory_references_and_incomplete_inventory() {
        assert!(
            ensure_cache_can_be_relocated(CacheId::PreviewFrames)
                .unwrap_err()
                .contains("memory")
        );

        let mut saved = DraftReferenceReport::default();
        saved
            .references
            .by_cache
            .get_mut(&CacheId::Download)
            .unwrap()
            .insert(PathBuf::from("/managed/download.mp4"));
        let live = CacheReferenceReport::default();
        assert!(
            validate_relocation_references(CacheId::Download, &saved, &live)
                .unwrap_err()
                .to_string()
                .contains("reference")
        );
        assert!(
            validate_relocation_references(CacheId::Proxies, &saved, &live).is_ok(),
            "proxies have no persisted reference fields"
        );

        let incomplete_saved = DraftReferenceReport {
            skipped_or_errored: 1,
            ..DraftReferenceReport::default()
        };
        assert!(
            validate_relocation_references(CacheId::Catalog, &incomplete_saved, &live)
                .unwrap_err()
                .to_string()
                .contains("incomplete")
        );

        let saved = DraftReferenceReport::default();
        let incomplete_live = CacheReferenceReport {
            counts: crate::cache_references::ReferenceCounts {
                rejected: 1,
                ..crate::cache_references::ReferenceCounts::default()
            },
            ..CacheReferenceReport::default()
        };
        assert!(
            validate_relocation_references(CacheId::Catalog, &saved, &incomplete_live)
                .unwrap_err()
                .to_string()
                .contains("incomplete")
        );
    }

    #[test]
    fn same_filesystem_relocation_persists_and_publishes_exact_accounting() {
        let temporary = tempfile::tempdir().unwrap();
        let layout = StorageLayout::new(temporary.path().join("storage")).unwrap();
        let old_path = layout.resolve(CacheId::Proxies).unwrap();
        fs::create_dir_all(old_path.join("nested")).unwrap();
        fs::write(old_path.join("nested/proxy.mp4"), b"1234567").unwrap();
        let shared = SharedStorageLayout::new(layout);
        let destination = temporary.path().join("moved-proxies");
        let config_path = temporary.path().join("config.toml");
        fs::write(
            &config_path,
            "# preserved comment\n[appearance]\ntheme = \"ember\"\n\
             [ai]\nbase_url = \"http://localhost:11434/v1\"\nmodel = \"qwen-test\"\n\
             [storage]\ndownload_quota_mib = 321\n\
             [future]\nflag = true\n",
        )
        .unwrap();

        let snapshot = shared.snapshot();
        let (mut replacement, expected_generation) = snapshot.into_parts();
        replacement
            .set_override(CacheId::Proxies, &destination)
            .unwrap();
        let mut settings = cutlass_settings::load(&config_path).unwrap();
        set_storage_path_override(&mut settings, CacheId::Proxies, destination.clone()).unwrap();
        let cancel = AtomicBool::new(false);
        let mut committed = None;

        let generation = shared
            .transition(
                expected_generation,
                replacement,
                |old_layout, new_layout| -> Result<(), CacheRelocationFailure> {
                    committed = Some(relocate_disk_root(
                        &disk_path(old_layout, CacheId::Proxies)
                            .map_err(CacheRelocationFailure::from_message)?,
                        &disk_path(new_layout, CacheId::Proxies)
                            .map_err(CacheRelocationFailure::from_message)?,
                        &cancel,
                        |_| persist_relocation_settings(&config_path, &settings),
                    )?);
                    Ok(())
                },
            )
            .unwrap();
        let committed = committed.unwrap();

        assert_eq!(generation, 1);
        assert_eq!(
            committed.report,
            cutlass_storage::RelocationReport {
                bytes: 7,
                files: 1,
                used_copy_fallback: false,
            }
        );
        assert_eq!(committed.cleanup_warning, None);
        assert!(!old_path.exists());
        assert_eq!(
            fs::read(destination.join("nested/proxy.mp4")).unwrap(),
            b"1234567"
        );
        let published = shared.snapshot();
        assert_eq!(published.generation(), generation);
        assert_eq!(
            published.resolve(CacheId::Proxies),
            Some(destination.clone())
        );
        let persisted = cutlass_settings::load(&config_path).unwrap();
        assert_eq!(persisted.storage.paths.proxies, Some(destination.clone()));
        assert_eq!(persisted.storage.download_quota_mib, 321);
        assert_eq!(persisted.ai.model, "qwen-test");
        let raw = fs::read_to_string(&config_path).unwrap();
        assert!(raw.contains("# preserved comment"));
        assert!(raw.contains("[future]"));
        assert!(raw.contains("flag = true"));

        let current = snapshot_disk_cache(published.layout(), CacheId::Proxies, &cancel).unwrap();
        assert_eq!((current.bytes, current.files), (7, 1));
    }

    #[test]
    fn missing_source_becomes_an_empty_relocated_cache_after_cancellation_check() {
        let temporary = tempfile::tempdir().unwrap();
        let old_path = temporary.path().join("missing-old");
        let destination = temporary.path().join("empty-new");
        let cancelled = AtomicBool::new(true);
        assert!(relocate_disk_root(&old_path, &destination, &cancelled, |_| Ok(())).is_err());
        assert!(!old_path.exists());
        assert!(!destination.exists());

        let outcome =
            relocate_disk_root(&old_path, &destination, &AtomicBool::new(false), |_| Ok(()))
                .unwrap();
        assert_eq!(
            outcome.report,
            cutlass_storage::RelocationReport {
                bytes: 0,
                files: 0,
                used_copy_fallback: false,
            }
        );
        assert!(!old_path.exists());
        assert!(destination.is_dir());
        assert_eq!(fs::read_dir(destination).unwrap().count(), 0);
    }

    #[test]
    fn relocation_persist_failure_rolls_filesystem_and_layout_back() {
        let temporary = tempfile::tempdir().unwrap();
        let layout = StorageLayout::new(temporary.path().join("storage")).unwrap();
        let old_path = layout.resolve(CacheId::Catalog).unwrap();
        fs::create_dir_all(&old_path).unwrap();
        fs::write(old_path.join("catalog.json"), b"unchanged").unwrap();
        let shared = SharedStorageLayout::new(layout);
        let destination = temporary.path().join("moved-catalog");
        let snapshot = shared.snapshot();
        let (mut replacement, expected_generation) = snapshot.into_parts();
        replacement
            .set_override(CacheId::Catalog, &destination)
            .unwrap();
        let cancel = AtomicBool::new(false);
        let persist_saw_complete_destination = AtomicBool::new(false);

        let result = shared.transition(
            expected_generation,
            replacement,
            |old_layout, new_layout| -> Result<(), CacheRelocationFailure> {
                relocate_disk_root(
                    &disk_path(old_layout, CacheId::Catalog)
                        .map_err(CacheRelocationFailure::from_message)?,
                    &disk_path(new_layout, CacheId::Catalog)
                        .map_err(CacheRelocationFailure::from_message)?,
                    &cancel,
                    |completed| {
                        assert_eq!(completed, destination);
                        assert_eq!(
                            fs::read(completed.join("catalog.json")).unwrap(),
                            b"unchanged"
                        );
                        persist_saw_complete_destination.store(true, Ordering::Release);
                        Err("injected settings persistence failure".into())
                    },
                )?;
                Ok(())
            },
        );

        assert!(result.is_err());
        assert!(persist_saw_complete_destination.load(Ordering::Acquire));
        let unchanged = shared.snapshot();
        assert_eq!(unchanged.generation(), 0);
        assert_eq!(unchanged.resolve(CacheId::Catalog), Some(old_path.clone()));
        assert_eq!(
            fs::read(old_path.join("catalog.json")).unwrap(),
            b"unchanged"
        );
        assert!(!destination.exists());
    }

    #[test]
    fn committed_cleanup_failure_is_success_with_a_bounded_warning() {
        let temporary = tempfile::tempdir().unwrap();
        let layout = StorageLayout::new(temporary.path().join("storage")).unwrap();
        let shared = SharedStorageLayout::new(layout);
        let snapshot = shared.snapshot();
        let (mut replacement, expected_generation) = snapshot.into_parts();
        let destination = temporary.path().join("catalog-new");
        replacement
            .set_override(CacheId::Catalog, &destination)
            .unwrap();
        let report = cutlass_storage::RelocationReport {
            bytes: 19,
            files: 2,
            used_copy_fallback: true,
        };
        let mut outcome = None;
        let generation = shared
            .transition(
                expected_generation,
                replacement,
                |_, _| -> Result<(), StorageError> {
                    outcome = Some(classify_relocation_result(Err(
                        StorageError::CommittedCleanupFailed {
                            message: "x".repeat(MAX_ERROR_CHARS * 4),
                            report,
                        },
                    ))?);
                    Ok(())
                },
            )
            .unwrap();
        let outcome = outcome.unwrap();

        assert_eq!(generation, 1);
        assert_eq!(
            shared.snapshot().resolve(CacheId::Catalog),
            Some(destination)
        );
        assert_eq!(outcome.report, report);
        let warning = outcome.cleanup_warning.unwrap();
        assert!(warning.contains("committed"));
        assert!(warning.chars().count() <= MAX_ERROR_CHARS + 1);
        assert!(classify_relocation_result(Err(StorageError::Cancelled)).is_err());
    }

    #[test]
    fn download_relocation_switches_root_and_remaps_protected_paths() {
        let temporary = tempfile::tempdir().unwrap();
        let old_path = temporary.path().join("download-old");
        let destination = temporary.path().join("download-new");
        fs::create_dir_all(old_path.join("stock")).unwrap();
        let protected = old_path.join("stock/project.mp4");
        let disposable = old_path.join("stock/disposable.bin");
        fs::write(&protected, b"project").unwrap();
        fs::write(&disposable, b"temporary").unwrap();
        let cache = DownloadCache::new(old_path.clone(), 1_000);
        cache.protect_path(&protected).unwrap();

        let outcome = relocate_download_root(
            &cache,
            &old_path,
            &destination,
            &AtomicBool::new(false),
            |_| Ok(()),
        )
        .unwrap();

        assert_eq!(outcome.report.bytes, 16);
        assert_eq!(outcome.report.files, 2);
        assert_eq!(cache.root(), destination);
        assert!(!old_path.exists());
        let moved_protected = destination.join("stock/project.mp4");
        let moved_disposable = destination.join("stock/disposable.bin");
        assert!(moved_protected.exists());
        assert!(moved_disposable.exists());

        cache.set_quota_bytes(0);
        cache.enforce_quota();
        assert!(
            moved_protected.exists(),
            "protected path must be remapped with the root"
        );
        assert!(!moved_disposable.exists());
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
    fn settings_persistence_never_runs_while_cache_operation_gate_is_held() {
        let gate = Mutex::new(());
        let held = gate.lock().unwrap();
        let ran = AtomicBool::new(false);

        let error = try_with_operation_gate(&gate, || {
            ran.store(true, Ordering::Release);
        })
        .unwrap_err();
        assert!(error.contains("cache operation is in progress"));
        assert!(!ran.load(Ordering::Acquire));

        drop(held);
        assert_eq!(try_with_operation_gate(&gate, || 42), Ok(42));
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

        let relocation = CacheRelocationReport {
            id: CacheId::Catalog,
            old_path: PathBuf::from("/tmp/catalog-old"),
            new_path: PathBuf::from("/tmp/catalog-new"),
            bytes: 12,
            files: 2,
            used_copy_fallback: true,
            cleanup_warning: Some("old copy remains".into()),
            generation: 4,
            current: None,
        };
        assert_eq!(
            serde_json::to_value(relocation).unwrap(),
            json!({
                "cache_id": "catalog",
                "old_path": "/tmp/catalog-old",
                "new_path": "/tmp/catalog-new",
                "bytes": 12,
                "files": 2,
                "used_copy_fallback": true,
                "cleanup_warning": "old copy remains",
                "generation": 4
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
            "download cache root does not match the active storage layout"
        );
    }
}
