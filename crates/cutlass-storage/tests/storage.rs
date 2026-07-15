use std::collections::HashSet;
use std::fs;
use std::io;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier, mpsc};
use std::thread;
use std::time::Duration;

use cutlass_storage::{
    CACHE_REGISTRY, CacheId, CacheKind, CacheTier, NeverCancelled, SharedStorageLayout,
    SharedStorageLayoutError, SharedStorageLayoutTransitionError, StorageError, StorageLayout,
    cache_descriptor_by_key, clear_cache, measure_disk_usage, relocate_cache,
};

struct TestDirectory {
    path: PathBuf,
}

impl TestDirectory {
    fn new() -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        for _ in 0..128 {
            let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "cutlass-storage-integration-{}-{id}",
                std::process::id()
            ));
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
fn registry_is_exact_unique_and_round_trips() {
    let expected = [
        (
            "preview_frames",
            "Preview frames",
            CacheKind::Memory,
            CacheTier::Disposable,
            None,
        ),
        (
            "library_thumbnails",
            "Library thumbnails",
            CacheKind::Memory,
            CacheTier::Disposable,
            None,
        ),
        (
            "timeline_filmstrips",
            "Timeline filmstrips",
            CacheKind::Memory,
            CacheTier::Disposable,
            None,
        ),
        (
            "timeline_waveforms",
            "Timeline waveforms",
            CacheKind::Memory,
            CacheTier::Disposable,
            None,
        ),
        (
            "proxies",
            "Proxies",
            CacheKind::Disk,
            CacheTier::Disposable,
            Some("proxies"),
        ),
        (
            "download",
            "Downloads",
            CacheKind::Disk,
            CacheTier::Redownloadable,
            Some("download-cache"),
        ),
        (
            "catalog",
            "Catalog",
            CacheKind::Disk,
            CacheTier::Redownloadable,
            Some("catalog-cache"),
        ),
        (
            "luts",
            "LUTs",
            CacheKind::Disk,
            CacheTier::Redownloadable,
            Some("luts"),
        ),
        (
            "lottie",
            "Lottie assets",
            CacheKind::Disk,
            CacheTier::Redownloadable,
            Some("lottie"),
        ),
        (
            "templates",
            "Templates",
            CacheKind::Disk,
            CacheTier::Redownloadable,
            Some("templates"),
        ),
    ];

    let actual: Vec<_> = CACHE_REGISTRY
        .iter()
        .map(|descriptor| {
            (
                descriptor.id.as_str(),
                descriptor.label,
                descriptor.kind,
                descriptor.tier,
                descriptor.default_relative,
            )
        })
        .collect();
    assert_eq!(actual, expected);

    let unique: HashSet<_> = CACHE_REGISTRY
        .iter()
        .map(|descriptor| descriptor.id.as_str())
        .collect();
    assert_eq!(unique.len(), CACHE_REGISTRY.len());

    for descriptor in CACHE_REGISTRY {
        assert_eq!(CacheId::parse(descriptor.id.as_str()), Ok(descriptor.id));
        assert_eq!(descriptor.id.as_str().parse::<CacheId>(), Ok(descriptor.id));
        assert_eq!(
            cache_descriptor_by_key(descriptor.id.as_str()),
            Some(&descriptor)
        );
        assert_ne!(descriptor.tier, CacheTier::UserData);
    }

    for forbidden in ["projects", "config", "agent_sessions"] {
        assert!(CacheId::parse(forbidden).is_err());
        assert!(cache_descriptor_by_key(forbidden).is_none());
    }
    assert!(CacheId::parse("Proxies").is_err());
    assert!(CacheId::parse("proxies ").is_err());
}

#[test]
fn layout_resolves_roots_and_overrides_deterministically() {
    let temporary = TestDirectory::new();
    let override_root = temporary.path.join("custom-download");
    let luts_root = temporary.path.join("custom-luts");
    let layout = StorageLayout::with_overrides(
        &temporary.path,
        [
            ("luts", luts_root.clone()),
            ("download", override_root.clone()),
        ],
    )
    .unwrap();

    assert_eq!(layout.root(), temporary.path);
    assert_eq!(layout.resolve(CacheId::PreviewFrames), None);
    assert_eq!(
        layout.resolve(CacheId::Proxies),
        Some(temporary.path.join("proxies"))
    );
    assert_eq!(
        layout.resolve(CacheId::Download),
        Some(override_root.clone())
    );
    assert_eq!(layout.resolve(CacheId::Luts), Some(luts_root.clone()));
    assert_eq!(
        layout.resolve(CacheId::Catalog),
        Some(temporary.path.join("catalog-cache"))
    );

    let override_ids: Vec<_> = layout.overrides().keys().copied().collect();
    assert_eq!(override_ids, [CacheId::Download, CacheId::Luts]);
    let resolved_ids: Vec<_> = layout
        .resolved_disk_paths()
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(
        resolved_ids,
        [
            CacheId::Proxies,
            CacheId::Download,
            CacheId::Catalog,
            CacheId::Luts,
            CacheId::Lottie,
            CacheId::Templates,
        ]
    );
}

#[test]
fn layout_rejects_invalid_overrides() {
    let temporary = TestDirectory::new();
    let mut layout = StorageLayout::new(&temporary.path).unwrap();

    assert!(matches!(
        StorageLayout::new("relative"),
        Err(StorageError::PathNotAbsolute(_))
    ));
    assert!(matches!(
        layout.set_override(CacheId::PreviewFrames, temporary.path.join("memory")),
        Err(StorageError::CacheIsNotDisk(CacheId::PreviewFrames))
    ));
    assert!(matches!(
        layout.set_override(CacheId::Proxies, "relative"),
        Err(StorageError::PathNotAbsolute(_))
    ));
    assert!(matches!(
        layout.set_override_key("unknown", temporary.path.join("unknown")),
        Err(StorageError::UnknownCacheId)
    ));
    assert!(matches!(
        layout.set_override(CacheId::Download, temporary.path.join("proxies")),
        Err(StorageError::CachePathsOverlap {
            cache: CacheId::Download,
            other: CacheId::Proxies
        })
    ));
    layout
        .set_override(CacheId::Download, temporary.path.join("custom-download"))
        .unwrap();
    assert!(matches!(
        layout.set_override(
            CacheId::Proxies,
            temporary.path.join("custom-download").join("nested")
        ),
        Err(StorageError::CachePathsOverlap {
            cache: CacheId::Proxies,
            other: CacheId::Download
        })
    ));
    assert!(matches!(
        StorageLayout::with_overrides(
            &temporary.path,
            [
                ("download", temporary.path.join("one")),
                ("download", temporary.path.join("two")),
            ],
        ),
        Err(StorageError::DuplicateOverride(CacheId::Download))
    ));
}

#[test]
fn shared_layout_snapshot_and_path_are_version_coherent() {
    let temporary = TestDirectory::new();
    let first_layout = StorageLayout::new(temporary.path.join("first")).unwrap();
    let second_layout = StorageLayout::new(temporary.path.join("second")).unwrap();
    let shared = SharedStorageLayout::new(first_layout.clone());

    let first_snapshot = shared.snapshot();
    assert_eq!(first_snapshot.generation(), 0);
    assert_eq!(first_snapshot.layout(), &first_layout);
    assert_eq!(
        first_snapshot.resolve(CacheId::Proxies),
        first_layout.resolve(CacheId::Proxies)
    );

    let (first_path, first_path_generation) = shared.resolve_versioned(CacheId::Proxies);
    assert_eq!(first_path_generation, 0);
    assert_eq!(first_path, first_layout.resolve(CacheId::Proxies));

    assert_eq!(shared.replace(0, second_layout.clone()), Ok(1));

    assert_eq!(first_snapshot.generation(), 0);
    assert_eq!(first_snapshot.layout(), &first_layout);
    assert_eq!(
        first_snapshot.resolve(CacheId::Proxies),
        first_layout.resolve(CacheId::Proxies)
    );

    let second_snapshot = shared.snapshot();
    assert_eq!(second_snapshot.generation(), 1);
    assert_eq!(second_snapshot.layout(), &second_layout);
    let (layout, generation) = second_snapshot.into_parts();
    assert_eq!(layout, second_layout);
    assert_eq!(generation, 1);

    let (second_path, second_path_generation) = shared.resolve_versioned(CacheId::Proxies);
    assert_eq!(second_path_generation, 1);
    assert_eq!(second_path, second_layout.resolve(CacheId::Proxies));

    let updated_download = temporary.path.join("updated-download");
    assert_eq!(
        shared.update(1, |layout| {
            layout
                .set_override(CacheId::Download, &updated_download)
                .unwrap();
        }),
        Ok(2)
    );
    let updated_snapshot = shared.snapshot();
    assert_eq!(updated_snapshot.generation(), 2);
    assert_eq!(
        updated_snapshot.resolve(CacheId::Download),
        Some(updated_download)
    );
}

#[test]
fn shared_layout_memory_ids_resolve_without_paths() {
    let temporary = TestDirectory::new();
    let shared = SharedStorageLayout::new(StorageLayout::new(&temporary.path).unwrap());

    for id in [
        CacheId::PreviewFrames,
        CacheId::LibraryThumbnails,
        CacheId::TimelineFilmstrips,
        CacheId::TimelineWaveforms,
    ] {
        assert_eq!(shared.resolve(id), None);
        assert_eq!(shared.resolve_versioned(id), (None, 0));
        assert_eq!(shared.snapshot().resolve(id), None);
    }
}

#[test]
fn shared_layout_refuses_stale_writers_without_exposing_paths() {
    let temporary = TestDirectory::new();
    let initial = StorageLayout::new(temporary.path.join("initial")).unwrap();
    let winner = StorageLayout::new(temporary.path.join("winner")).unwrap();
    let loser_marker = "never-log-this-storage-path";
    let loser = StorageLayout::new(temporary.path.join(loser_marker)).unwrap();
    let shared = SharedStorageLayout::new(initial);

    assert_eq!(shared.replace(0, winner.clone()), Ok(1));

    let error = shared.replace(0, loser).unwrap_err();
    assert_eq!(
        error,
        SharedStorageLayoutError::StaleGeneration {
            expected: 0,
            current: 1,
        }
    );
    assert!(!error.to_string().contains(loser_marker));
    assert!(!format!("{error:?}").contains(loser_marker));

    let update_was_called = std::cell::Cell::new(false);
    assert_eq!(
        shared.update(0, |_| update_was_called.set(true)),
        Err(SharedStorageLayoutError::StaleGeneration {
            expected: 0,
            current: 1,
        })
    );
    assert!(!update_was_called.get());

    let snapshot = shared.snapshot();
    assert_eq!(snapshot.generation(), 1);
    assert_eq!(snapshot.layout(), &winner);
}

#[test]
fn shared_layout_concurrent_readers_observe_only_coherent_versions() {
    const READER_COUNT: usize = 6;
    const LAST_GENERATION: u64 = 64;
    const READS_PER_READER: usize = 512;

    let temporary = TestDirectory::new();
    let layouts: Arc<Vec<_>> = Arc::new(
        (0..=LAST_GENERATION)
            .map(|generation| {
                StorageLayout::new(temporary.path.join(format!("layout-{generation}"))).unwrap()
            })
            .collect(),
    );
    let shared = SharedStorageLayout::new(layouts[0].clone());
    let start = Arc::new(Barrier::new(READER_COUNT + 1));

    let readers: Vec<_> = (0..READER_COUNT)
        .map(|_| {
            let layouts = Arc::clone(&layouts);
            let shared = shared.clone();
            let start = Arc::clone(&start);
            thread::spawn(move || {
                start.wait();
                for read_index in 0..READS_PER_READER {
                    let snapshot = shared.snapshot();
                    let generation = snapshot.generation() as usize;
                    assert_eq!(snapshot.layout(), &layouts[generation]);
                    assert_eq!(
                        snapshot.resolve(CacheId::Proxies),
                        layouts[generation].resolve(CacheId::Proxies)
                    );

                    let (path, path_generation) = shared.resolve_versioned(CacheId::Download);
                    assert_eq!(
                        path,
                        layouts[path_generation as usize].resolve(CacheId::Download)
                    );

                    if read_index % 8 == 0 {
                        thread::yield_now();
                    }
                }
            })
        })
        .collect();

    start.wait();
    for generation in 1..=LAST_GENERATION {
        assert_eq!(
            shared.replace(generation - 1, layouts[generation as usize].clone()),
            Ok(generation)
        );
        thread::yield_now();
    }

    for reader in readers {
        reader.join().unwrap();
    }

    let snapshot = shared.snapshot();
    assert_eq!(snapshot.generation(), LAST_GENERATION);
    assert_eq!(snapshot.layout(), &layouts[LAST_GENERATION as usize]);
}

#[test]
fn shared_layout_recovers_after_a_panicking_update() {
    let temporary = TestDirectory::new();
    let initial = StorageLayout::new(temporary.path.join("initial")).unwrap();
    let replacement = StorageLayout::new(temporary.path.join("replacement")).unwrap();
    let shared = SharedStorageLayout::new(initial.clone());
    let panicking_writer = shared.clone();

    let panic = catch_unwind(AssertUnwindSafe(|| {
        panicking_writer
            .update(0, |candidate| {
                candidate
                    .set_override(
                        CacheId::Download,
                        temporary.path.join("uncommitted-override"),
                    )
                    .unwrap();
                panic!("intentional update panic");
            })
            .unwrap();
    }));
    assert!(panic.is_err());

    assert_eq!(shared.replace(0, replacement.clone()), Ok(1));
    let snapshot = shared.snapshot();
    assert_eq!(snapshot.generation(), 1);
    assert_eq!(snapshot.layout(), &replacement);
    assert_ne!(snapshot.layout(), &initial);
}

#[test]
fn shared_layout_lease_blocks_transition_until_operation_finishes() {
    let temporary = TestDirectory::new();
    let initial = StorageLayout::new(temporary.path.join("initial")).unwrap();
    let replacement = StorageLayout::new(temporary.path.join("replacement")).unwrap();
    let shared = SharedStorageLayout::new(initial.clone());
    let lease = shared.lease();
    assert_eq!(lease.generation(), 0);
    assert_eq!(
        lease.resolve(CacheId::Proxies),
        initial.resolve(CacheId::Proxies)
    );

    let transition_started = Arc::new(AtomicBool::new(false));
    let callback_started = Arc::new(AtomicBool::new(false));
    let (attempt_tx, attempt_rx) = mpsc::channel();
    let worker = {
        let shared = shared.clone();
        let transition_started = Arc::clone(&transition_started);
        let callback_started = Arc::clone(&callback_started);
        thread::spawn(move || {
            transition_started.store(true, Ordering::Release);
            attempt_tx.send(()).unwrap();
            shared.transition(0, replacement, |old, new| {
                callback_started.store(true, Ordering::Release);
                assert_ne!(old, new);
                Ok::<(), io::Error>(())
            })
        })
    };

    attempt_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(transition_started.load(Ordering::Acquire));
    thread::sleep(Duration::from_millis(20));
    assert!(
        !callback_started.load(Ordering::Acquire),
        "exclusive transition must wait for the active lease"
    );

    drop(lease);
    assert_eq!(worker.join().unwrap().unwrap(), 1);
    assert!(callback_started.load(Ordering::Acquire));
    assert_eq!(shared.snapshot().generation(), 1);
}

#[test]
fn shared_layout_failed_transition_does_not_publish_replacement() {
    let temporary = TestDirectory::new();
    let initial = StorageLayout::new(temporary.path.join("initial")).unwrap();
    let replacement = StorageLayout::new(temporary.path.join("replacement")).unwrap();
    let shared = SharedStorageLayout::new(initial.clone());

    let result = shared.transition(0, replacement.clone(), |old, new| {
        assert_eq!(old, &initial);
        assert_eq!(new, &replacement);
        Err(io::Error::other("injected transition failure"))
    });
    assert!(matches!(
        result,
        Err(SharedStorageLayoutTransitionError::Transition(error))
            if error.kind() == io::ErrorKind::Other
    ));

    let snapshot = shared.snapshot();
    assert_eq!(snapshot.generation(), 0);
    assert_eq!(snapshot.layout(), &initial);

    let callback_called = AtomicBool::new(false);
    let stale = shared.transition(7, replacement, |_, _| {
        callback_called.store(true, Ordering::Release);
        Ok::<(), io::Error>(())
    });
    assert!(matches!(
        stale,
        Err(SharedStorageLayoutTransitionError::Layout(
            SharedStorageLayoutError::StaleGeneration {
                expected: 7,
                current: 0,
            }
        ))
    ));
    assert!(!callback_called.load(Ordering::Acquire));
}

#[test]
fn missing_usage_is_zero_and_missing_clear_creates_root() {
    let temporary = TestDirectory::new();
    let missing = temporary.path.join("missing");

    assert_eq!(
        measure_disk_usage(&missing, &NeverCancelled).unwrap(),
        Default::default()
    );
    assert!(!missing.exists());

    assert_eq!(
        clear_cache(&missing, &NeverCancelled).unwrap(),
        Default::default()
    );
    assert!(missing.is_dir());
}

#[test]
fn nested_usage_counts_logical_bytes_and_files() {
    let temporary = TestDirectory::new();
    let root = temporary.path.join("cache");
    fs::create_dir(&root).unwrap();
    fs::create_dir(root.join("one")).unwrap();
    fs::create_dir(root.join("one").join("two")).unwrap();
    fs::write(root.join("top"), b"abc").unwrap();
    fs::write(root.join("one").join("two").join("nested"), b"12345").unwrap();
    fs::write(root.join("empty"), b"").unwrap();

    let usage = measure_disk_usage(&root, &NeverCancelled).unwrap();
    assert_eq!(usage.bytes, 8);
    assert_eq!(usage.files, 3);
}

#[test]
fn clear_removes_nested_contents_and_preserves_root() {
    let temporary = TestDirectory::new();
    let root = temporary.path.join("cache");
    fs::create_dir(&root).unwrap();
    fs::create_dir(root.join("nested")).unwrap();
    fs::write(root.join("top"), b"abc").unwrap();
    fs::write(root.join("nested").join("data"), b"12345").unwrap();

    let report = clear_cache(&root, &NeverCancelled).unwrap();
    assert_eq!(report.removed_bytes, 8);
    assert_eq!(report.removed_files, 2);
    assert!(root.is_dir());
    assert_eq!(fs::read_dir(&root).unwrap().count(), 0);
}

#[test]
fn cancellation_stops_measure_clear_and_relocation() {
    let temporary = TestDirectory::new();
    let root = temporary.path.join("cache");
    fs::create_dir(&root).unwrap();
    fs::write(root.join("data"), b"keep").unwrap();
    let cancelled = || true;

    assert!(matches!(
        measure_disk_usage(&root, &cancelled),
        Err(StorageError::Cancelled)
    ));
    assert!(matches!(
        clear_cache(&root, &cancelled),
        Err(StorageError::Cancelled)
    ));
    assert_eq!(fs::read(root.join("data")).unwrap(), b"keep");

    let destination_parent = temporary.path.join("new-parent");
    let destination = destination_parent.join("moved");
    assert!(matches!(
        relocate_cache(&root, &destination, &cancelled, |_| Ok(())),
        Err(StorageError::Cancelled)
    ));
    assert!(root.is_dir());
    assert!(!destination.exists());
    assert!(!destination_parent.exists());
}

#[test]
fn dangerous_and_overlapping_paths_are_rejected() {
    let temporary = TestDirectory::new();
    let source = temporary.path.join("source");
    fs::create_dir(&source).unwrap();

    assert!(matches!(
        clear_cache(Path::new(""), &NeverCancelled),
        Err(StorageError::DangerousPath(_))
    ));
    assert!(matches!(
        relocate_cache(&source, Path::new("relative"), &NeverCancelled, |_| Ok(())),
        Err(StorageError::PathNotAbsolute(_))
    ));
    assert!(matches!(
        relocate_cache(&source, source.join("nested"), &NeverCancelled, |_| Ok(())),
        Err(StorageError::PathsOverlap)
    ));

    let destination = temporary.path.join("destination");
    fs::create_dir(&destination).unwrap();
    assert!(matches!(
        relocate_cache(&source, &destination, &NeverCancelled, |_| Ok(())),
        Err(StorageError::DestinationExists)
    ));
}

#[cfg(unix)]
#[test]
fn filesystem_root_is_rejected_by_destructive_operations() {
    assert!(matches!(
        StorageLayout::new("/"),
        Err(StorageError::DangerousPath(_))
    ));
    assert!(matches!(
        clear_cache(Path::new("/"), &NeverCancelled),
        Err(StorageError::DangerousPath(_))
    ));

    let temporary = TestDirectory::new();
    let source = temporary.path.join("source");
    fs::create_dir(&source).unwrap();
    assert!(matches!(
        relocate_cache(&source, Path::new("/"), &NeverCancelled, |_| Ok(())),
        Err(StorageError::DangerousPath(_))
    ));
}

#[cfg(unix)]
#[test]
fn symlink_roots_are_rejected_and_nested_links_are_not_followed() {
    use std::os::unix::fs::symlink;

    let temporary = TestDirectory::new();
    let outside = temporary.path.join("outside");
    let root = temporary.path.join("cache");
    let root_link = temporary.path.join("cache-link");
    fs::create_dir(&outside).unwrap();
    fs::create_dir(&root).unwrap();
    fs::write(outside.join("large"), vec![9_u8; 32 * 1024]).unwrap();
    symlink(&root, &root_link).unwrap();

    assert!(matches!(
        measure_disk_usage(&root_link, &NeverCancelled),
        Err(StorageError::SymlinkRoot)
    ));
    assert!(matches!(
        clear_cache(&root_link, &NeverCancelled),
        Err(StorageError::SymlinkRoot)
    ));
    assert!(matches!(
        relocate_cache(
            &root_link,
            temporary.path.join("moved"),
            &NeverCancelled,
            |_| Ok(())
        ),
        Err(StorageError::SymlinkRoot)
    ));

    let destination_link = temporary.path.join("destination-link");
    symlink(&outside, &destination_link).unwrap();
    assert!(matches!(
        relocate_cache(&root, &destination_link, &NeverCancelled, |_| Ok(())),
        Err(StorageError::SymlinkRoot)
    ));

    let parent_alias = temporary.path.join("parent-alias");
    symlink(&temporary.path, &parent_alias).unwrap();
    assert!(matches!(
        relocate_cache(
            &root,
            parent_alias.join("cache").join("nested"),
            &NeverCancelled,
            |_| Ok(())
        ),
        Err(StorageError::PathsOverlap)
    ));

    let directory_link = root.join("outside-link");
    let file_link = root.join("file-link");
    symlink(&outside, &directory_link).unwrap();
    symlink(outside.join("large"), &file_link).unwrap();
    let expected_link_bytes = fs::symlink_metadata(&directory_link).unwrap().len()
        + fs::symlink_metadata(&file_link).unwrap().len();

    let usage = measure_disk_usage(&root, &NeverCancelled).unwrap();
    assert_eq!(usage.bytes, expected_link_bytes);
    assert_eq!(usage.files, 2);

    let report = clear_cache(&root, &NeverCancelled).unwrap();
    assert_eq!(report.removed_bytes, expected_link_bytes);
    assert_eq!(report.removed_files, 2);
    assert!(root.is_dir());
    assert!(outside.join("large").is_file());
    assert_eq!(fs::read(outside.join("large")).unwrap().len(), 32 * 1024);
}

#[test]
fn same_filesystem_relocation_persists_complete_destination() {
    let temporary = TestDirectory::new();
    let source = temporary.path.join("old");
    let destination = temporary.path.join("new");
    fs::create_dir(&source).unwrap();
    fs::create_dir(source.join("nested")).unwrap();
    fs::write(source.join("nested").join("data"), b"complete").unwrap();

    let report = relocate_cache(&source, &destination, &NeverCancelled, |completed| {
        assert_eq!(completed, destination);
        assert_eq!(
            fs::read(completed.join("nested").join("data")).unwrap(),
            b"complete"
        );
        Ok(())
    })
    .unwrap();

    assert_eq!(report.bytes, 8);
    assert_eq!(report.files, 1);
    assert!(!report.used_copy_fallback);
    assert!(!source.exists());
    assert_eq!(
        fs::read(destination.join("nested").join("data")).unwrap(),
        b"complete"
    );
}

#[test]
fn persistence_failure_rolls_atomic_rename_back() {
    let temporary = TestDirectory::new();
    let source = temporary.path.join("old");
    let destination = temporary.path.join("new");
    fs::create_dir(&source).unwrap();
    fs::write(source.join("data"), b"authoritative").unwrap();

    let error = relocate_cache(&source, &destination, &NeverCancelled, |completed| {
        assert_eq!(fs::read(completed.join("data")).unwrap(), b"authoritative");
        Err("could not save settings".into())
    })
    .unwrap_err();

    assert!(matches!(error, StorageError::PersistenceFailed { .. }));
    assert_eq!(fs::read(source.join("data")).unwrap(), b"authoritative");
    assert!(!destination.exists());
}
