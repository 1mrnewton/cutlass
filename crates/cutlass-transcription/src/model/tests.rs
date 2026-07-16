use std::io::Cursor;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Barrier, mpsc};
use std::thread;
use std::time::Instant;

use tempfile::TempDir;

use super::*;

const HELLO_SHA256: &str = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
const TINY_SPEC: ModelSpec = ModelSpec::new(
    "test.tiny",
    "tiny-model.bin",
    "https://example.invalid/tiny-model.bin",
    5,
    HELLO_SHA256,
    true,
    "five test bytes",
);

#[test]
fn catalog_has_verified_base_en_constants() {
    assert_eq!(WhisperModel::catalog(), &[WhisperModel::BaseEn]);
    let spec = WhisperModel::BaseEn.spec();
    assert_eq!(spec.id(), "base.en");
    assert_eq!(spec.filename(), "ggml-base.en.bin");
    assert_eq!(
        spec.url(),
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
    );
    assert_eq!(spec.exact_bytes(), 147_964_211);
    assert_eq!(
        spec.sha256(),
        "a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002"
    );
    assert!(spec.is_english_only());
    assert!(spec.resource_description().contains("148 MB"));
    assert!(validate_spec(spec).is_ok());
}

#[test]
fn validates_root_and_catalog_paths_without_traversal() {
    assert!(matches!(
        ModelManager::new("relative/models"),
        Err(ModelManagerError::RootNotAbsolute { .. })
    ));
    assert!(matches!(
        ModelManager::new(Path::new("/")),
        Err(ModelManagerError::FilesystemRoot { .. })
    ));
    assert!(matches!(
        ModelManager::new(Path::new("/tmp/../escape")),
        Err(ModelManagerError::RootTraversal { .. })
    ));

    let temp = TempDir::new().expect("temporary directory");
    let manager = ModelManager::new(temp.path()).expect("valid absolute root");
    let traversal = ModelSpec::new(
        "test.traversal",
        "../escape.bin",
        "https://example.invalid/escape.bin",
        5,
        HELLO_SHA256,
        true,
        "invalid test entry",
    );
    assert!(matches!(
        manager.path_for_spec(&traversal),
        Err(ModelManagerError::InvalidCatalogEntry { .. })
    ));
    assert!(!temp.path().join("../escape.bin").exists());
}

#[test]
fn pre_cancelled_install_creates_no_root_or_download() {
    let temp = TempDir::new().expect("temporary directory");
    let root = temp.path().join("models");
    let manager = ModelManager::new(&root).expect("valid manager");
    let downloader = CountingDownloader::new(b"unused");

    let error = manager
        .ensure_with_cancellation(WhisperModel::BaseEn, &downloader, &|| true)
        .expect_err("pre-cancelled installation must stop");

    assert!(matches!(&error, ModelManagerError::Cancelled));
    assert_eq!(error.to_string(), "model installation cancelled");
    assert!(!root.exists());
    assert_eq!(downloader.calls(), 0);
}

#[test]
fn panicking_cancellation_callback_fails_closed() {
    let temp = TempDir::new().expect("temporary directory");
    let root = temp.path().join("models");
    let manager = ModelManager::new(&root).expect("valid manager");
    let downloader = CountingDownloader::new(b"unused");

    let error = manager
        .ensure_with_cancellation(WhisperModel::BaseEn, &downloader, &|| {
            panic!("injected cancellation callback panic")
        })
        .expect_err("callback panic must be cancellation");

    assert!(matches!(error, ModelManagerError::Cancelled));
    assert!(!root.exists());
    assert_eq!(downloader.calls(), 0);
}

#[test]
fn reuses_a_verified_existing_model_without_downloading() {
    let temp = TempDir::new().expect("temporary directory");
    let manager = ModelManager::new(temp.path()).expect("valid manager");
    let path = manager.path_for_spec(&TINY_SPEC).expect("known path");
    fs::write(&path, b"hello").expect("write fixture");
    let downloader = |_spec: &ModelSpec| -> Result<DownloadReader, DownloadError> {
        panic!("verified model must not be downloaded")
    };

    assert_eq!(
        manager
            .ensure_spec(&TINY_SPEC, &downloader)
            .expect("reuse succeeds"),
        path
    );
    assert!(matches!(
        manager.status_spec(&TINY_SPEC).expect("status"),
        ModelStatus::Ready { .. }
    ));
}

#[test]
fn redownloads_wrong_size_and_corrupt_existing_models() {
    for existing in [&b"bad"[..], &b"jello"[..]] {
        let temp = TempDir::new().expect("temporary directory");
        let manager = ModelManager::new(temp.path()).expect("valid manager");
        let path = manager.path_for_spec(&TINY_SPEC).expect("known path");
        fs::write(&path, existing).expect("write corrupt fixture");
        let downloader = CountingDownloader::new(b"hello");

        manager
            .ensure_spec(&TINY_SPEC, &downloader)
            .expect("redownload succeeds");

        assert_eq!(downloader.calls(), 1);
        assert_eq!(fs::read(path).expect("read installed model"), b"hello");
        assert_no_temporary_files(temp.path());
    }
}

#[test]
fn rejects_an_overlong_stream_and_cleans_temporary_file() {
    let temp = TempDir::new().expect("temporary directory");
    let manager = ModelManager::new(temp.path()).expect("valid manager");
    let downloader = CountingDownloader::new(b"hello!");

    let error = manager
        .ensure_spec(&TINY_SPEC, &downloader)
        .expect_err("overlong download must fail");
    assert!(matches!(
        error,
        ModelManagerError::DownloadTooLarge {
            expected: 5,
            observed: 6
        }
    ));
    assert!(!manager.path_for_spec(&TINY_SPEC).expect("path").exists());
    assert_directory_empty(temp.path());
}

#[test]
fn rejects_checksum_mismatch_and_cleans_temporary_file() {
    let temp = TempDir::new().expect("temporary directory");
    let manager = ModelManager::new(temp.path()).expect("valid manager");
    let downloader = CountingDownloader::new(b"jello");

    let error = manager
        .ensure_spec(&TINY_SPEC, &downloader)
        .expect_err("checksum mismatch must fail");
    assert!(matches!(
        error,
        ModelManagerError::Integrity {
            reason: ModelIntegrityError::ChecksumMismatch { .. },
            ..
        }
    ));
    assert_directory_empty(temp.path());
}

#[test]
fn interrupted_reader_cleans_temporary_file() {
    let temp = TempDir::new().expect("temporary directory");
    let manager = ModelManager::new(temp.path()).expect("valid manager");
    let downloader = |_spec: &ModelSpec| -> Result<DownloadReader, DownloadError> {
        Ok(Box::new(InterruptedReader { emitted: false }))
    };

    let error = manager
        .ensure_spec(&TINY_SPEC, &downloader)
        .expect_err("interrupted download must fail");
    assert!(matches!(error, ModelManagerError::Io { .. }));
    assert_directory_empty(temp.path());
}

#[test]
fn mid_stream_cancellation_cleans_temp_and_preserves_invalid_target() {
    let temp = TempDir::new().expect("temporary directory");
    let manager = ModelManager::new(temp.path()).expect("valid manager");
    let path = manager.path_for_spec(&TINY_SPEC).expect("known path");
    fs::write(&path, b"bad").expect("write invalid existing model");

    let cancelled = Arc::new(AtomicBool::new(false));
    let downloader_calls = Arc::new(AtomicUsize::new(0));
    let cancelled_for_reader = Arc::clone(&cancelled);
    let calls_for_downloader = Arc::clone(&downloader_calls);
    let downloader = move |_spec: &ModelSpec| -> Result<DownloadReader, DownloadError> {
        calls_for_downloader.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(CancellingReader {
            read_count: 0,
            cancelled: Arc::clone(&cancelled_for_reader),
        }))
    };

    let error = manager
        .ensure_spec_with_cancellation(&TINY_SPEC, &downloader, &|| {
            cancelled.load(Ordering::SeqCst)
        })
        .expect_err("mid-stream cancellation must stop");

    assert!(matches!(error, ModelManagerError::Cancelled));
    assert_eq!(downloader_calls.load(Ordering::SeqCst), 1);
    assert_eq!(fs::read(&path).expect("read invalid target"), b"bad");
    assert_no_temporary_files(temp.path());
}

#[test]
fn begin_commit_is_called_once_for_fresh_install_and_replacement() {
    for existing in [None, Some(&b"jello"[..])] {
        let temp = TempDir::new().expect("temporary directory");
        let manager = ModelManager::new(temp.path()).expect("valid manager");
        let path = manager.path_for_spec(&TINY_SPEC).expect("known path");
        if let Some(bytes) = existing {
            fs::write(&path, bytes).expect("write invalid existing model");
        }
        let downloader = CountingDownloader::new(b"hello");
        let begin_calls = AtomicUsize::new(0);

        let installed = manager
            .ensure_spec_with_cancellation_and_commit(
                &TINY_SPEC,
                &downloader,
                &never_cancelled,
                &|| {
                    begin_calls.fetch_add(1, Ordering::SeqCst);
                    true
                },
            )
            .expect("installation succeeds");

        assert_eq!(installed, path);
        assert_eq!(begin_calls.load(Ordering::SeqCst), 1);
        assert_eq!(downloader.calls(), 1);
        assert_eq!(fs::read(&path).expect("read installed model"), b"hello");
        assert_no_temporary_files(temp.path());
    }
}

#[test]
fn rejected_begin_commit_preserves_target_and_cleans_temp() {
    for existing in [None, Some(&b"bad"[..])] {
        let temp = TempDir::new().expect("temporary directory");
        let manager = ModelManager::new(temp.path()).expect("valid manager");
        let path = manager.path_for_spec(&TINY_SPEC).expect("known path");
        if let Some(bytes) = existing {
            fs::write(&path, bytes).expect("write invalid existing model");
        }
        let downloader = CountingDownloader::new(b"hello");
        let begin_calls = AtomicUsize::new(0);

        let error = manager
            .ensure_spec_with_cancellation_and_commit(
                &TINY_SPEC,
                &downloader,
                &never_cancelled,
                &|| {
                    begin_calls.fetch_add(1, Ordering::SeqCst);
                    false
                },
            )
            .expect_err("rejected commit must cancel");

        assert!(matches!(error, ModelManagerError::Cancelled));
        assert_eq!(begin_calls.load(Ordering::SeqCst), 1);
        match existing {
            Some(bytes) => {
                assert_eq!(fs::read(&path).expect("read preserved target"), bytes);
            }
            None => assert!(!path.exists()),
        }
        assert_no_temporary_files(temp.path());
    }
}

#[test]
fn panicking_begin_commit_preserves_target_and_cleans_temp() {
    for existing in [None, Some(&b"bad"[..])] {
        let temp = TempDir::new().expect("temporary directory");
        let manager = ModelManager::new(temp.path()).expect("valid manager");
        let path = manager.path_for_spec(&TINY_SPEC).expect("known path");
        if let Some(bytes) = existing {
            fs::write(&path, bytes).expect("write invalid existing model");
        }
        let downloader = CountingDownloader::new(b"hello");
        let begin_calls = AtomicUsize::new(0);

        let error = manager
            .ensure_spec_with_cancellation_and_commit(
                &TINY_SPEC,
                &downloader,
                &never_cancelled,
                &|| {
                    begin_calls.fetch_add(1, Ordering::SeqCst);
                    panic!("injected begin-commit panic")
                },
            )
            .expect_err("panicking commit callback must cancel");

        assert!(matches!(error, ModelManagerError::Cancelled));
        assert_eq!(begin_calls.load(Ordering::SeqCst), 1);
        match existing {
            Some(bytes) => {
                assert_eq!(fs::read(&path).expect("read preserved target"), bytes);
            }
            None => assert!(!path.exists()),
        }
        assert_no_temporary_files(temp.path());
    }
}

#[test]
fn begin_commit_observes_complete_verified_and_synced_temporary_file() {
    let temp = TempDir::new().expect("temporary directory");
    let manager = ModelManager::new(temp.path()).expect("valid manager");
    let path = manager.path_for_spec(&TINY_SPEC).expect("known path");
    let read_calls = Arc::new(AtomicUsize::new(0));
    let saw_eof = Arc::new(AtomicBool::new(false));
    let read_calls_for_downloader = Arc::clone(&read_calls);
    let saw_eof_for_downloader = Arc::clone(&saw_eof);
    let downloader = move |_spec: &ModelSpec| -> Result<DownloadReader, DownloadError> {
        Ok(Box::new(CompletionTrackingReader {
            cursor: Cursor::new(b"hello"),
            read_calls: Arc::clone(&read_calls_for_downloader),
            saw_eof: Arc::clone(&saw_eof_for_downloader),
        }))
    };

    manager
        .ensure_spec_with_cancellation_and_commit(
            &TINY_SPEC,
            &downloader,
            &never_cancelled,
            &|| {
                assert!(saw_eof.load(Ordering::SeqCst));
                assert!(read_calls.load(Ordering::SeqCst) >= 2);
                assert!(!path.exists());

                let temporary_paths: Vec<_> = fs::read_dir(temp.path())
                    .expect("read model directory")
                    .map(|entry| entry.expect("directory entry").path())
                    .filter(|entry| {
                        entry
                            .extension()
                            .is_some_and(|extension| extension == "tmp")
                    })
                    .collect();
                assert_eq!(temporary_paths.len(), 1);
                assert_eq!(
                    fs::read(&temporary_paths[0]).expect("read synced temporary model"),
                    b"hello"
                );
                true
            },
        )
        .expect("installation succeeds");

    assert_eq!(fs::read(&path).expect("read installed model"), b"hello");
    assert_no_temporary_files(temp.path());
}

#[test]
fn cancellation_before_commit_skips_begin_commit() {
    let temp = TempDir::new().expect("temporary directory");
    let manager = ModelManager::new(temp.path()).expect("valid manager");
    let path = manager.path_for_spec(&TINY_SPEC).expect("known path");
    fs::write(&path, b"bad").expect("write invalid existing model");
    let cancelled = Arc::new(AtomicBool::new(false));
    let cancelled_for_downloader = Arc::clone(&cancelled);
    let downloader = move |_spec: &ModelSpec| -> Result<DownloadReader, DownloadError> {
        Ok(Box::new(CancelAtEofReader {
            cursor: Cursor::new(b"hello"),
            cancelled: Arc::clone(&cancelled_for_downloader),
        }))
    };
    let begin_calls = AtomicUsize::new(0);

    let error = manager
        .ensure_spec_with_cancellation_and_commit(
            &TINY_SPEC,
            &downloader,
            &|| cancelled.load(Ordering::SeqCst),
            &|| {
                begin_calls.fetch_add(1, Ordering::SeqCst);
                true
            },
        )
        .expect_err("pre-commit cancellation must win");

    assert!(matches!(error, ModelManagerError::Cancelled));
    assert_eq!(begin_calls.load(Ordering::SeqCst), 0);
    assert_eq!(fs::read(&path).expect("read preserved target"), b"bad");
    assert_no_temporary_files(temp.path());
}

#[test]
fn cancellation_after_begin_commit_cannot_reclassify_success() {
    let temp = TempDir::new().expect("temporary directory");
    let manager = ModelManager::new(temp.path()).expect("valid manager");
    let path = manager.path_for_spec(&TINY_SPEC).expect("known path");
    let downloader = CountingDownloader::new(b"hello");
    let cancelled = AtomicBool::new(false);
    let cancellation_checks = AtomicUsize::new(0);
    let checks_at_commit = AtomicUsize::new(0);
    let begin_calls = AtomicUsize::new(0);

    let installed = manager
        .ensure_spec_with_cancellation_and_commit(
            &TINY_SPEC,
            &downloader,
            &|| {
                cancellation_checks.fetch_add(1, Ordering::SeqCst);
                cancelled.load(Ordering::SeqCst)
            },
            &|| {
                begin_calls.fetch_add(1, Ordering::SeqCst);
                checks_at_commit
                    .store(cancellation_checks.load(Ordering::SeqCst), Ordering::SeqCst);
                cancelled.store(true, Ordering::SeqCst);
                true
            },
        )
        .expect("accepted commit must finish despite later cancellation");

    assert_eq!(installed, path);
    assert_eq!(begin_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        cancellation_checks.load(Ordering::SeqCst),
        checks_at_commit.load(Ordering::SeqCst)
    );
    assert_eq!(fs::read(&path).expect("read installed model"), b"hello");
    assert_no_temporary_files(temp.path());
}

#[test]
fn ready_reuse_skips_begin_commit_and_remains_cancellation_aware() {
    let temp = TempDir::new().expect("temporary directory");
    let manager = ModelManager::new(temp.path()).expect("valid manager");
    let path = manager.path_for_spec(&TINY_SPEC).expect("known path");
    fs::write(&path, b"hello").expect("write verified model");
    let downloader = |_spec: &ModelSpec| -> Result<DownloadReader, DownloadError> {
        panic!("verified model must not be downloaded")
    };
    let begin_calls = AtomicUsize::new(0);

    let reused = manager
        .ensure_spec_with_cancellation_and_commit(
            &TINY_SPEC,
            &downloader,
            &never_cancelled,
            &|| {
                begin_calls.fetch_add(1, Ordering::SeqCst);
                true
            },
        )
        .expect("ready model reuse succeeds");
    assert_eq!(reused, path);
    assert_eq!(begin_calls.load(Ordering::SeqCst), 0);

    let cancellation_checks = AtomicUsize::new(0);
    let error = manager
        .ensure_spec_with_cancellation_and_commit(
            &TINY_SPEC,
            &downloader,
            &|| cancellation_checks.fetch_add(1, Ordering::SeqCst) + 1 >= 5,
            &|| {
                begin_calls.fetch_add(1, Ordering::SeqCst);
                true
            },
        )
        .expect_err("cancellation after ready verification must be observed");
    assert!(matches!(error, ModelManagerError::Cancelled));
    assert!(cancellation_checks.load(Ordering::SeqCst) >= 5);
    assert_eq!(begin_calls.load(Ordering::SeqCst), 0);
    assert_eq!(fs::read(&path).expect("read reused model"), b"hello");
    assert_no_temporary_files(temp.path());
}

#[cfg(unix)]
#[test]
fn post_commit_mapping_error_is_not_reclassified_as_cancellation() {
    use std::os::unix::fs::symlink;

    let temp = TempDir::new().expect("temporary directory");
    let root = temp.path().join("models");
    let pinned_root = temp.path().join("models-pinned");
    let outside = temp.path().join("outside");
    fs::create_dir(&root).expect("create model root");
    fs::create_dir(&outside).expect("create outside directory");
    let manager = ModelManager::new(&root).expect("valid manager");
    let downloader = CountingDownloader::new(b"hello");
    let cancelled = AtomicBool::new(false);
    let begin_calls = AtomicUsize::new(0);

    let error = manager
        .ensure_spec_with_cancellation_and_commit(
            &TINY_SPEC,
            &downloader,
            &|| cancelled.load(Ordering::SeqCst),
            &|| {
                begin_calls.fetch_add(1, Ordering::SeqCst);
                cancelled.store(true, Ordering::SeqCst);
                fs::rename(&root, &pinned_root).expect("rename pinned root");
                symlink(&outside, &root).expect("replace root with symlink");
                true
            },
        )
        .expect_err("post-commit mapping change must remain an actual error");

    assert!(matches!(
        error,
        ModelManagerError::RootMappingChanged { .. }
    ));
    assert_eq!(begin_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        fs::read(pinned_root.join(TINY_SPEC.filename())).expect("read committed model"),
        b"hello"
    );
    assert!(!outside.join(TINY_SPEC.filename()).exists());
    assert_no_temporary_files(&pinned_root);
}

#[test]
fn concurrent_ensure_commits_one_complete_download() {
    const WORKERS: usize = 8;

    let temp = TempDir::new().expect("temporary directory");
    let manager = Arc::new(ModelManager::new(temp.path()).expect("valid manager"));
    let downloader = Arc::new(CountingDownloader::new(b"hello"));
    let begin_calls = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(WORKERS));
    let mut workers = Vec::new();

    for _ in 0..WORKERS {
        let manager = Arc::clone(&manager);
        let downloader = Arc::clone(&downloader);
        let begin_calls = Arc::clone(&begin_calls);
        let barrier = Arc::clone(&barrier);
        workers.push(thread::spawn(move || {
            barrier.wait();
            manager.ensure_spec_with_cancellation_and_commit(
                &TINY_SPEC,
                downloader.as_ref(),
                &never_cancelled,
                &|| {
                    begin_calls.fetch_add(1, Ordering::SeqCst);
                    true
                },
            )
        }));
    }

    for worker in workers {
        worker
            .join()
            .expect("worker did not panic")
            .expect("concurrent ensure succeeds");
    }

    assert_eq!(downloader.calls(), 1);
    assert_eq!(begin_calls.load(Ordering::SeqCst), 1);
    let path = manager.path_for_spec(&TINY_SPEC).expect("known path");
    assert_eq!(fs::read(path).expect("read committed model"), b"hello");
    assert_no_temporary_files(temp.path());
}

#[test]
fn cancellation_while_waiting_for_model_lock_returns_promptly() {
    let temp = TempDir::new().expect("temporary directory");
    let manager = Arc::new(ModelManager::new(temp.path()).expect("valid manager"));
    let entered_download = Arc::new(Barrier::new(2));
    let release_download = Arc::new(Barrier::new(2));
    let first_calls = Arc::new(AtomicUsize::new(0));

    let first_downloader = Arc::new({
        let entered_download = Arc::clone(&entered_download);
        let release_download = Arc::clone(&release_download);
        let first_calls = Arc::clone(&first_calls);
        move |_spec: &ModelSpec| -> Result<DownloadReader, DownloadError> {
            first_calls.fetch_add(1, Ordering::SeqCst);
            entered_download.wait();
            release_download.wait();
            Ok(Box::new(Cursor::new(b"hello")))
        }
    });
    let first_worker = thread::spawn({
        let manager = Arc::clone(&manager);
        let first_downloader = Arc::clone(&first_downloader);
        move || manager.ensure_spec(&TINY_SPEC, first_downloader.as_ref())
    });
    entered_download.wait();

    let second_downloader = Arc::new(CountingDownloader::new(b"hello"));
    let cancelled = Arc::new(AtomicBool::new(false));
    let (check_sender, check_receiver) = mpsc::channel();
    let (result_sender, result_receiver) = mpsc::channel();
    let second_worker = thread::spawn({
        let manager = Arc::clone(&manager);
        let second_downloader = Arc::clone(&second_downloader);
        let cancelled = Arc::clone(&cancelled);
        move || {
            let result = manager.ensure_spec_with_cancellation(
                &TINY_SPEC,
                second_downloader.as_ref(),
                &|| {
                    let _ = check_sender.send(());
                    cancelled.load(Ordering::SeqCst)
                },
            );
            result_sender.send(result).expect("send second result");
        }
    });

    let mut observed_lock_retry = true;
    for _ in 0..3 {
        if check_receiver.recv_timeout(Duration::from_secs(1)).is_err() {
            observed_lock_retry = false;
            break;
        }
    }
    let cancellation_started = Instant::now();
    cancelled.store(true, Ordering::SeqCst);
    let second_result = result_receiver.recv_timeout(Duration::from_secs(1));
    let cancellation_elapsed = cancellation_started.elapsed();

    release_download.wait();
    let first_result = first_worker.join().expect("first worker did not panic");
    second_worker.join().expect("second worker did not panic");

    assert!(
        observed_lock_retry,
        "second installation did not reach a contended lock retry"
    );
    assert!(
        cancellation_elapsed < Duration::from_secs(1),
        "lock-wait cancellation took {cancellation_elapsed:?}"
    );
    assert!(matches!(
        second_result.expect("lock waiter did not cancel promptly"),
        Err(ModelManagerError::Cancelled)
    ));
    first_result.expect("first installation succeeds");
    assert_eq!(first_calls.load(Ordering::SeqCst), 1);
    assert_eq!(second_downloader.calls(), 0);
    assert_no_temporary_files(temp.path());
}

#[cfg(unix)]
#[test]
fn canonical_aliases_share_one_model_lock() {
    use std::os::unix::fs::symlink;

    let temp = TempDir::new().expect("temporary directory");
    let physical_root = temp.path().join("physical");
    let alias_parent = temp.path().join("alias");
    fs::create_dir(&physical_root).expect("create physical root");
    symlink(temp.path(), &alias_parent).expect("create intermediate alias");

    let direct =
        ModelManager::new(&physical_root).expect("direct manager has a valid final root");
    let aliased = ModelManager::new(alias_parent.join("physical"))
        .expect("aliased manager has a valid final root");
    let direct_snapshot = direct
        .capture_existing_root()
        .expect("capture direct root")
        .expect("direct root exists");
    let aliased_snapshot = aliased
        .capture_existing_root()
        .expect("capture aliased root")
        .expect("aliased root exists");
    let direct_target = direct_snapshot.physical_model_path(&TINY_SPEC);
    let aliased_target = aliased_snapshot.physical_model_path(&TINY_SPEC);

    assert_eq!(direct_target, aliased_target);
    let direct_lock = model_lock(&direct_target);
    let aliased_lock = model_lock(&aliased_target);
    assert!(Arc::ptr_eq(&direct_lock, &aliased_lock));
}

#[cfg(unix)]
#[test]
fn root_swap_during_download_fails_closed_and_cleans_pinned_temp() {
    use std::os::unix::fs::symlink;

    let temp = TempDir::new().expect("temporary directory");
    let root = temp.path().join("models");
    let pinned_root = temp.path().join("models-pinned");
    let outside = temp.path().join("outside");
    fs::create_dir(&root).expect("create model root");
    fs::create_dir(&outside).expect("create outside directory");
    let manager = ModelManager::new(&root).expect("valid manager");

    let root_for_swap = root.clone();
    let pinned_for_swap = pinned_root.clone();
    let outside_for_swap = outside.clone();
    let downloader = move |_spec: &ModelSpec| -> Result<DownloadReader, DownloadError> {
        fs::rename(&root_for_swap, &pinned_for_swap).expect("rename pinned root");
        symlink(&outside_for_swap, &root_for_swap).expect("replace root with symlink");
        Ok(Box::new(Cursor::new(b"hello")))
    };

    let error = manager
        .ensure_spec(&TINY_SPEC, &downloader)
        .expect_err("changed root mapping must fail");

    assert!(matches!(
        error,
        ModelManagerError::RootMappingChanged { .. }
    ));
    assert!(root.is_symlink());
    assert!(!outside.join(TINY_SPEC.filename()).exists());
    assert!(!pinned_root.join(TINY_SPEC.filename()).exists());
    assert_directory_empty(&pinned_root);
}

#[cfg(unix)]
#[test]
fn remove_refuses_a_changed_root_mapping() {
    use std::os::unix::fs::symlink;

    let temp = TempDir::new().expect("temporary directory");
    let root = temp.path().join("models");
    let pinned_root = temp.path().join("models-pinned");
    let outside = temp.path().join("outside");
    fs::create_dir(&root).expect("create model root");
    fs::create_dir(&outside).expect("create outside directory");
    fs::write(root.join(TINY_SPEC.filename()), b"hello").expect("write model");
    let manager = ModelManager::new(&root).expect("valid manager");
    let snapshot = manager
        .capture_existing_root()
        .expect("capture root")
        .expect("root exists");

    fs::rename(&root, &pinned_root).expect("rename pinned root");
    symlink(&outside, &root).expect("replace root with symlink");
    let error = manager
        .remove_from_snapshot(&TINY_SPEC, &snapshot)
        .expect_err("changed root mapping must fail");

    assert!(matches!(
        error,
        ModelManagerError::RootMappingChanged { .. }
    ));
    assert_eq!(
        fs::read(pinned_root.join(TINY_SPEC.filename())).expect("pinned model survives"),
        b"hello"
    );
    assert!(!outside.join(TINY_SPEC.filename()).exists());
}

#[cfg(unix)]
#[test]
fn verification_never_reports_ready_after_root_mapping_change() {
    use std::os::unix::fs::symlink;

    let temp = TempDir::new().expect("temporary directory");
    let root = temp.path().join("models");
    let pinned_root = temp.path().join("models-pinned");
    let outside = temp.path().join("outside");
    fs::create_dir(&root).expect("create model root");
    fs::create_dir(&outside).expect("create outside directory");
    fs::write(root.join(TINY_SPEC.filename()), b"hello").expect("write model");
    let manager = ModelManager::new(&root).expect("valid manager");
    let snapshot = manager
        .capture_existing_root()
        .expect("capture root")
        .expect("root exists");

    fs::rename(&root, &pinned_root).expect("rename pinned root");
    symlink(&outside, &root).expect("replace root with symlink");
    let error = manager
        .inspect_path(&TINY_SPEC, &snapshot)
        .expect_err("changed mapping cannot produce ready status");

    assert!(matches!(
        error,
        ModelManagerError::RootMappingChanged { .. }
    ));
    assert!(!outside.join(TINY_SPEC.filename()).exists());
}

#[cfg(unix)]
#[test]
fn rejects_symlink_roots_for_every_operation() {
    use std::os::unix::fs::symlink;

    let temp = TempDir::new().expect("temporary directory");
    let outside = temp.path().join("outside");
    let root = temp.path().join("models");
    fs::create_dir(&outside).expect("create outside directory");
    symlink(&outside, &root).expect("create root symlink");
    let manager = ModelManager::new(&root).expect("lexically valid manager");
    let downloader = CountingDownloader::new(b"hello");

    assert!(matches!(
        manager.status_spec(&TINY_SPEC),
        Err(ModelManagerError::UnsafeRoot { .. })
    ));
    assert!(matches!(
        manager.ensure_spec(&TINY_SPEC, &downloader),
        Err(ModelManagerError::UnsafeRoot { .. })
    ));
    assert!(matches!(
        manager.remove_spec(&TINY_SPEC),
        Err(ModelManagerError::UnsafeRoot { .. })
    ));
    assert_eq!(downloader.calls(), 0);
    assert_directory_empty(&outside);
}

#[cfg(unix)]
#[test]
fn rejects_symlink_model_files_without_touching_the_target() {
    use std::os::unix::fs::symlink;

    let temp = TempDir::new().expect("temporary directory");
    let external = TempDir::new().expect("external directory");
    let external_file = external.path().join("external.bin");
    fs::write(&external_file, b"hello").expect("write external file");
    let manager = ModelManager::new(temp.path()).expect("valid manager");
    let model_path = manager.path_for_spec(&TINY_SPEC).expect("known path");
    symlink(&external_file, &model_path).expect("create symlink");
    let downloader = CountingDownloader::new(b"hello");

    assert!(matches!(
        manager.ensure_spec(&TINY_SPEC, &downloader),
        Err(ModelManagerError::UnsafeModelFile { .. })
    ));
    assert!(matches!(
        manager.remove_spec(&TINY_SPEC),
        Err(ModelManagerError::UnsafeModelFile { .. })
    ));
    assert_eq!(downloader.calls(), 0);
    assert_eq!(
        fs::read(&external_file).expect("external target survives"),
        b"hello"
    );
    assert!(model_path.is_symlink());
}

#[test]
fn remove_deletes_only_the_known_regular_file() {
    let temp = TempDir::new().expect("temporary directory");
    let manager = ModelManager::new(temp.path()).expect("valid manager");
    let model_path = manager.path_for_spec(&TINY_SPEC).expect("known path");
    let sibling = temp.path().join("keep.txt");
    fs::write(&model_path, b"hello").expect("write model");
    fs::write(&sibling, b"keep").expect("write sibling");

    assert!(manager.remove_spec(&TINY_SPEC).expect("remove model"));
    assert!(!model_path.exists());
    assert_eq!(fs::read(&sibling).expect("sibling remains"), b"keep");
    assert!(!manager.remove_spec(&TINY_SPEC).expect("already absent"));
}

#[cfg(windows)]
#[test]
fn windows_reparse_attribute_helper_rejects_all_reparse_points() {
    assert!(windows_attributes_are_reparse_point(
        WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT
    ));
    assert!(windows_attributes_are_reparse_point(
        WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT | 0x10
    ));
    assert!(!windows_attributes_are_reparse_point(0x10));
}

struct CountingDownloader {
    bytes: &'static [u8],
    calls: AtomicUsize,
}

impl CountingDownloader {
    fn new(bytes: &'static [u8]) -> Self {
        Self {
            bytes,
            calls: AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl ModelDownloader for CountingDownloader {
    fn download(&self, _spec: &ModelSpec) -> Result<DownloadReader, DownloadError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Cursor::new(self.bytes)))
    }
}

struct InterruptedReader {
    emitted: bool,
}

impl Read for InterruptedReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if self.emitted {
            return Err(io::Error::new(
                io::ErrorKind::ConnectionReset,
                "injected interruption",
            ));
        }
        self.emitted = true;
        buffer[..2].copy_from_slice(b"he");
        Ok(2)
    }
}

struct CancellingReader {
    read_count: usize,
    cancelled: Arc<AtomicBool>,
}

impl Read for CancellingReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let bytes = if self.read_count == 0 {
            &b"he"[..]
        } else {
            self.cancelled.store(true, Ordering::SeqCst);
            &b"ll"[..]
        };
        self.read_count += 1;
        buffer[..bytes.len()].copy_from_slice(bytes);
        Ok(bytes.len())
    }
}

struct CompletionTrackingReader {
    cursor: Cursor<&'static [u8]>,
    read_calls: Arc<AtomicUsize>,
    saw_eof: Arc<AtomicBool>,
}

impl Read for CompletionTrackingReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.read_calls.fetch_add(1, Ordering::SeqCst);
        let count = self.cursor.read(buffer)?;
        if count == 0 {
            self.saw_eof.store(true, Ordering::SeqCst);
        }
        Ok(count)
    }
}

struct CancelAtEofReader {
    cursor: Cursor<&'static [u8]>,
    cancelled: Arc<AtomicBool>,
}

impl Read for CancelAtEofReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let count = self.cursor.read(buffer)?;
        if count == 0 {
            self.cancelled.store(true, Ordering::SeqCst);
        }
        Ok(count)
    }
}

fn assert_directory_empty(path: &Path) {
    assert_eq!(fs::read_dir(path).expect("read model directory").count(), 0);
}

fn assert_no_temporary_files(path: &Path) {
    for entry in fs::read_dir(path).expect("read model directory") {
        let entry = entry.expect("directory entry");
        assert!(
            !entry.file_name().to_string_lossy().ends_with(".tmp"),
            "temporary model file leaked: {}",
            entry.path().display()
        );
    }
}
