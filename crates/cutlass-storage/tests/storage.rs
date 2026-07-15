use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use cutlass_storage::{
    CACHE_REGISTRY, CacheId, CacheKind, CacheTier, NeverCancelled, StorageError, StorageLayout,
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
