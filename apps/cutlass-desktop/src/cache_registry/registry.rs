use super::*;

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

    /// Run one synchronous worker operation against a resolved disk-cache root.
    ///
    /// The cache operation gate and one [`SharedStorageLayout`] generation
    /// lease remain held until `operation` returns. The leased layout is
    /// validated against the filesystem before the root is resolved and the
    /// callback begins. This method must run on a job or worker thread, never
    /// on the UI or preview thread.
    ///
    /// `cancelled` is borrowed and may directly inspect a future job context.
    /// It is checked before and during the bounded gate wait, after the gate is
    /// acquired, after layout validation, and immediately before the callback.
    /// A panic from it fails closed as cancellation. Cancellation is
    /// deliberately not checked after the callback because it may have
    /// committed a mutation.
    pub(crate) fn with_disk_cache_root<T, E>(
        &self,
        id: CacheId,
        cancelled: &dyn Fn() -> bool,
        operation: impl FnOnce(&Path) -> Result<T, E>,
    ) -> Result<T, CoordinatedCacheError<E>> {
        with_coordinated_disk_cache_root(
            &self.layout,
            &self.operation_gate,
            id,
            cancelled,
            GATE_WAIT_TIMEOUT,
            GATE_WAIT_SLICE,
            operation,
        )
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

pub(super) fn snapshot_disk_caches(
    layout: &StorageLayout,
    cancel: &AtomicBool,
) -> Result<Vec<CacheSnapshot>, String> {
    cache_descriptors()
        .iter()
        .filter(|descriptor| descriptor.kind == CacheKind::Disk)
        .map(|descriptor| snapshot_disk_cache(layout, descriptor.id, cancel))
        .collect()
}

pub(super) fn snapshot_disk_cache(
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

pub(super) fn clear_disk_contents(
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

pub(super) fn ensure_cache_can_be_cleared(id: CacheId) -> Result<(), String> {
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
    match id {
        CacheId::Luts | CacheId::Lottie | CacheId::Templates => true,
        CacheId::PreviewFrames
        | CacheId::LibraryThumbnails
        | CacheId::TimelineFilmstrips
        | CacheId::TimelineWaveforms
        | CacheId::Proxies
        | CacheId::Analysis
        | CacheId::AiModels
        | CacheId::Download
        | CacheId::Catalog => false,
    }
}

pub(super) fn clear_download_cache_from_inventories(
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

pub(super) fn prepare_download_cache_clear(
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

pub(super) fn validate_download_root(
    layout: &StorageLayout,
    download_root: &Path,
) -> Result<(), String> {
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

pub(super) fn disk_path(layout: &StorageLayout, id: CacheId) -> Result<PathBuf, String> {
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
