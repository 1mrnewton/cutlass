use std::convert::Infallible;
use std::fs;
use std::sync::mpsc;
use std::thread;

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
    let analysis = layout.resolve(CacheId::Analysis).unwrap();
    let ai_models = layout.resolve(CacheId::AiModels).unwrap();
    let download = layout.resolve(CacheId::Download).unwrap();
    fs::create_dir_all(&analysis).unwrap();
    fs::create_dir_all(&ai_models).unwrap();
    let nested = download.join("stock");
    fs::create_dir_all(&nested).unwrap();
    fs::write(analysis.join("moments.sqlite3"), b"analysis!").unwrap();
    fs::write(ai_models.join("ggml-base.en.bin"), b"model-weights").unwrap();
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
            CacheId::Analysis,
            CacheId::AiModels,
            CacheId::Download,
            CacheId::Catalog,
            CacheId::Luts,
            CacheId::Lottie,
            CacheId::Templates,
        ]
    );
    assert_eq!(snapshots.len(), 8);
    let download = snapshots
        .iter()
        .find(|snapshot| snapshot.id == CacheId::Download)
        .unwrap();
    assert_eq!(
        (download.bytes, download.files, download.entries),
        (8, 2, 0)
    );
    let analysis = snapshots
        .iter()
        .find(|snapshot| snapshot.id == CacheId::Analysis)
        .unwrap();
    assert_eq!(analysis.label, "Media analysis");
    assert_eq!(analysis.kind, CacheKind::Disk);
    assert_eq!(analysis.tier, CacheTier::Disposable);
    assert_eq!(
        (analysis.bytes, analysis.files, analysis.entries),
        (9, 1, 0)
    );
    let ai_models = snapshots
        .iter()
        .find(|snapshot| snapshot.id == CacheId::AiModels)
        .unwrap();
    assert_eq!(ai_models.label, "AI models");
    assert_eq!(ai_models.kind, CacheKind::Disk);
    assert_eq!(ai_models.tier, CacheTier::Redownloadable);
    assert_eq!(
        (ai_models.bytes, ai_models.files, ai_models.entries),
        (13, 1, 0)
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
    let analysis = layout.resolve(CacheId::Analysis).unwrap();
    let catalog = layout.resolve(CacheId::Catalog).unwrap();
    fs::create_dir_all(&analysis).unwrap();
    fs::create_dir_all(&catalog).unwrap();
    fs::write(analysis.join("moments.sqlite"), b"1234567").unwrap();
    fs::write(catalog.join("catalog.json"), b"keep").unwrap();

    let removed = clear_disk_contents(&layout, CacheId::Analysis, &AtomicBool::new(false)).unwrap();
    assert_eq!(
        removed,
        CacheUsage {
            bytes: 7,
            entries: 0,
            files: 1,
        }
    );
    assert!(analysis.is_dir());
    assert_eq!(fs::read_dir(&analysis).unwrap().count(), 0);
    assert_eq!(fs::read(catalog.join("catalog.json")).unwrap(), b"keep");
    assert_eq!(
        snapshot_disk_cache(&layout, CacheId::Analysis, &AtomicBool::new(false))
            .unwrap()
            .bytes,
        0
    );
}

#[test]
fn ai_models_are_outside_download_quota_and_reference_protection() {
    let temporary = tempfile::tempdir().unwrap();
    let layout = StorageLayout::new(temporary.path()).unwrap();
    let ai_models = layout.resolve(CacheId::AiModels).unwrap();
    let downloads = layout.resolve(CacheId::Download).unwrap();
    fs::create_dir_all(&ai_models).unwrap();
    fs::create_dir_all(&downloads).unwrap();
    let model = ai_models.join("ggml-base.en.bin");
    let protected_source = downloads.join("project-source.mp4");
    fs::write(&model, b"model-weights").unwrap();
    fs::write(&protected_source, b"source").unwrap();

    let download_cache = DownloadCache::new(downloads, 1_000);
    download_cache.protect_path(&protected_source).unwrap();
    download_cache.set_quota_bytes(0);
    download_cache.enforce_quota();
    assert!(model.is_file(), "download quota must not inspect AI models");
    assert!(protected_source.is_file());

    let removed = clear_disk_contents(&layout, CacheId::AiModels, &AtomicBool::new(false)).unwrap();
    assert_eq!(
        removed,
        CacheUsage {
            bytes: 13,
            entries: 0,
            files: 1,
        }
    );
    assert!(ai_models.is_dir());
    assert!(!model.exists());
    assert!(protected_source.is_file());
    assert_eq!(download_cache.protected_path_count(), 1);
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
fn cache_clear_policy_explicitly_allows_unreferenced_ai_models() {
    assert!(ensure_cache_can_be_cleared(CacheId::Proxies).is_ok());
    assert!(ensure_cache_can_be_cleared(CacheId::Analysis).is_ok());
    assert!(ensure_cache_can_be_cleared(CacheId::AiModels).is_ok());
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
        CacheId::Analysis,
        CacheId::AiModels,
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
    original.cloud.base_url = "https://api.example.test".into();
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
        assert_eq!(updated.cloud, original.cloud);
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
    assert!(
        validate_relocation_references(CacheId::Analysis, &saved, &live).is_ok(),
        "media analysis has no persisted project references"
    );
    assert!(
        validate_relocation_references(CacheId::AiModels, &saved, &live).is_ok(),
        "AI model weights have no persisted project references"
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
        relocate_disk_root(&old_path, &destination, &AtomicBool::new(false), |_| Ok(())).unwrap();
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
fn coordinated_disk_root_refuses_memory_caches_and_pre_cancelled_work() {
    let temporary = tempfile::tempdir().unwrap();
    let layout =
        SharedStorageLayout::new(StorageLayout::new(temporary.path().join("storage")).unwrap());
    let gate = Mutex::new(());
    let callback_ran = AtomicBool::new(false);

    let memory_error = with_coordinated_disk_cache_root(
        &layout,
        &gate,
        CacheId::PreviewFrames,
        &|| false,
        Duration::from_millis(50),
        Duration::from_millis(1),
        |_| {
            callback_ran.store(true, Ordering::Release);
            Ok::<_, Infallible>(())
        },
    )
    .unwrap_err();
    assert!(matches!(
        memory_error,
        CoordinatedCacheError::Coordination(CacheCoordinationError::MemoryCache)
    ));

    let cancelled_error = with_coordinated_disk_cache_root(
        &layout,
        &gate,
        CacheId::Analysis,
        &|| true,
        Duration::from_millis(50),
        Duration::from_millis(1),
        |_| {
            callback_ran.store(true, Ordering::Release);
            Ok::<_, Infallible>(())
        },
    )
    .unwrap_err();
    assert!(matches!(
        cancelled_error,
        CoordinatedCacheError::Coordination(CacheCoordinationError::Cancelled)
    ));
    assert!(!callback_ran.load(Ordering::Acquire));
}

#[test]
fn coordinated_disk_root_resolves_only_the_requested_analysis_root() {
    let temporary = tempfile::tempdir().unwrap();
    let expected = temporary.path().join("analysis-override");
    let mut layout = StorageLayout::new(temporary.path().join("storage")).unwrap();
    layout.set_override(CacheId::Analysis, &expected).unwrap();
    let layout = SharedStorageLayout::new(layout);
    let gate = Mutex::new(());

    let resolved = with_coordinated_disk_cache_root(
        &layout,
        &gate,
        CacheId::Analysis,
        &|| false,
        Duration::from_millis(50),
        Duration::from_millis(1),
        |root| Ok::<_, Infallible>(root.to_path_buf()),
    )
    .unwrap();

    assert_eq!(resolved, expected);
    assert_ne!(resolved, layout.resolve(CacheId::Proxies).unwrap());
}

#[test]
fn coordinated_disk_root_validates_layout_before_callback() {
    let temporary = tempfile::tempdir().unwrap();
    let storage = temporary.path().join("storage");
    fs::create_dir_all(&storage).unwrap();
    let layout = StorageLayout::new(&storage).unwrap();
    fs::write(
        layout.resolve(CacheId::Analysis).unwrap(),
        b"not a directory",
    )
    .unwrap();
    let layout = SharedStorageLayout::new(layout);
    let gate = Mutex::new(());
    let callback_ran = AtomicBool::new(false);

    let error = with_coordinated_disk_cache_root(
        &layout,
        &gate,
        CacheId::Analysis,
        &|| false,
        Duration::from_millis(50),
        Duration::from_millis(1),
        |_| {
            callback_ran.store(true, Ordering::Release);
            Ok::<_, Infallible>(())
        },
    )
    .unwrap_err();

    assert!(matches!(
        error,
        CoordinatedCacheError::Coordination(CacheCoordinationError::InvalidLayout { .. })
    ));
    assert!(!callback_ran.load(Ordering::Acquire));
}

#[test]
fn coordinated_disk_root_serializes_against_operation_gate() {
    let temporary = tempfile::tempdir().unwrap();
    let layout =
        SharedStorageLayout::new(StorageLayout::new(temporary.path().join("storage")).unwrap());
    let gate = Arc::new(Mutex::new(()));
    let held = gate.lock().unwrap();
    let (waiting_tx, waiting_rx) = mpsc::channel();
    let (callback_tx, callback_rx) = mpsc::channel();

    let worker_gate = Arc::clone(&gate);
    let worker = thread::spawn(move || {
        with_coordinated_disk_cache_root(
            &layout,
            worker_gate.as_ref(),
            CacheId::Analysis,
            &|| {
                let _ = waiting_tx.send(());
                false
            },
            Duration::from_secs(1),
            Duration::from_millis(1),
            |_| {
                callback_tx.send(()).unwrap();
                Ok::<_, Infallible>(())
            },
        )
    });

    waiting_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(callback_rx.recv_timeout(Duration::from_millis(25)).is_err());
    drop(held);
    callback_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    worker.join().unwrap().unwrap();
}

#[test]
fn coordinated_disk_root_treats_cancellation_panic_as_cancelled() {
    let temporary = tempfile::tempdir().unwrap();
    let layout =
        SharedStorageLayout::new(StorageLayout::new(temporary.path().join("storage")).unwrap());
    let gate = Mutex::new(());
    let callback_ran = AtomicBool::new(false);

    let error = with_coordinated_disk_cache_root(
        &layout,
        &gate,
        CacheId::Analysis,
        &|| panic!("injected cancellation panic"),
        Duration::from_millis(50),
        Duration::from_millis(1),
        |_| {
            callback_ran.store(true, Ordering::Release);
            Ok::<_, Infallible>(())
        },
    )
    .unwrap_err();

    assert!(matches!(
        error,
        CoordinatedCacheError::Coordination(CacheCoordinationError::Cancelled)
    ));
    assert!(!callback_ran.load(Ordering::Acquire));
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
