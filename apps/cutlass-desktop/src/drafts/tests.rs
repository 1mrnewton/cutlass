use super::*;
use std::cell::Cell;
use std::collections::HashSet;
use std::io::Cursor;

struct RenameFaultFs {
    failed_renames: Vec<(usize, io::ErrorKind)>,
    rename_calls: Cell<usize>,
}

impl MetaReplaceFs for RenameFaultFs {
    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        let call = self.rename_calls.get() + 1;
        self.rename_calls.set(call);
        if let Some((_, kind)) = self
            .failed_renames
            .iter()
            .find(|(failed_call, _)| *failed_call == call)
        {
            return Err(io::Error::new(*kind, "injected metadata rename failure"));
        }
        fs::rename(from, to)
    }

    fn remove_file(&self, path: &Path) -> io::Result<()> {
        fs::remove_file(path)
    }

    fn symlink_metadata(&self, path: &Path) -> io::Result<fs::Metadata> {
        fs::symlink_metadata(path)
    }
}

#[test]
fn new_id_is_unique_and_canonical_across_rapid_calls() {
    let ids: HashSet<String> = (0..1000).map(|_| new_id()).collect();
    assert_eq!(ids.len(), 1000, "ids collided");
    assert!(ids.iter().all(|id| is_valid_draft_id(id)));
}

#[test]
fn project_identity_extracts_from_the_exact_owned_shape_without_io() {
    let sandbox = tempfile::tempdir().expect("tempdir");
    let root = sandbox.path().join("projects");
    let project = root.join("abcdef-12").join(PROJECT_FILE);

    let id =
        draft_id_from_project_in_root(&root, &project).expect("extract valid draft identity");
    assert_eq!(id, "abcdef-12");
    assert!(
        !root.exists(),
        "identity extraction unexpectedly created the root"
    );
}

#[test]
fn project_identity_extraction_rejects_outside_and_malformed_paths() {
    let sandbox = tempfile::tempdir().expect("tempdir");
    let root = sandbox.path().join("projects");
    let absolute_injection = root
        .join(sandbox.path().join("elsewhere").join("abc-1"))
        .join(PROJECT_FILE);
    let invalid = [
        root.clone(),
        root.join("abc-1"),
        root.join("abc-1").join("wrong.cutlass"),
        root.join("abc-1").join("project.cutlass."),
        root.join("abc-1").join("extra").join(PROJECT_FILE),
        root.join("abc-1")
            .join("..")
            .join("def-2")
            .join(PROJECT_FILE),
        root.join(".").join("abc-1").join(PROJECT_FILE),
        root.join("arbitrary-name").join(PROJECT_FILE),
        root.join("ABC-1").join(PROJECT_FILE),
        root.join("0abc-1").join(PROJECT_FILE),
        root.join("abc-01").join(PROJECT_FILE),
        root.join("abc-1-2").join(PROJECT_FILE),
        absolute_injection,
    ];

    for path in invalid {
        assert!(
            draft_id_from_project_in_root(&root, &path).is_err(),
            "accepted invalid path: {}",
            path.display()
        );
    }
}

#[test]
fn draft_identity_helpers_roundtrip_a_real_draft() {
    let sandbox = tempfile::tempdir().expect("tempdir");
    let root = sandbox.path().join("projects");
    let project = write_test_draft(&root, "abcdef-12", b"project");

    let id = draft_id_from_project_in_root(&root, &project).expect("extract draft id");
    assert_eq!(id, "abcdef-12");
    assert_eq!(
        resolve_draft_id_in_root(&root, &id).expect("resolve draft id"),
        project
    );
}

#[test]
fn draft_id_resolution_rejects_noncanonical_ids_and_full_paths() {
    let sandbox = tempfile::tempdir().expect("tempdir");
    let root = sandbox.path().join("projects");
    let project = write_test_draft(&root, "abc-1", b"project");
    let invalid = [
        "",
        "abc",
        "abc-",
        "-1",
        "../abc-1",
        "abc/def-1",
        "ABC-1",
        "abc-A",
        "0abc-1",
        "abc-01",
        "abc-1-2",
    ];

    for id in invalid {
        let error = resolve_draft_id_in_root(&root, id).expect_err("accepted invalid draft id");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput, "id: {id}");
    }

    let full_path = project.to_string_lossy();
    let error = resolve_draft_id_in_root(&root, &full_path)
        .expect_err("accepted a project path as a draft id");
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
}

#[test]
fn draft_id_resolution_requires_an_existing_regular_project() {
    let sandbox = tempfile::tempdir().expect("tempdir");
    let root = sandbox.path().join("projects");
    fs::create_dir(&root).expect("create root");

    let missing_draft =
        resolve_draft_id_in_root(&root, "abc-1").expect_err("resolved missing draft");
    assert_eq!(missing_draft.kind(), io::ErrorKind::NotFound);

    fs::create_dir(root.join("abc-1")).expect("create empty draft");
    let missing_project =
        resolve_draft_id_in_root(&root, "abc-1").expect_err("resolved missing project");
    assert_eq!(missing_project.kind(), io::ErrorKind::NotFound);

    let non_file_project = root.join("abc-1").join(PROJECT_FILE);
    fs::create_dir(&non_file_project).expect("create non-file project entry");
    let error =
        resolve_draft_id_in_root(&root, "abc-1").expect_err("resolved non-file project");
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
}

#[test]
fn arbitrary_parent_delete_is_refused() {
    let sandbox = tempfile::tempdir().expect("tempdir");
    let root = sandbox.path().join("projects");
    fs::create_dir(&root).expect("create root");
    let outside_dir = sandbox.path().join("outside").join("abc-1");
    fs::create_dir_all(&outside_dir).expect("create outside draft");
    let outside_project = outside_dir.join(PROJECT_FILE);
    fs::write(&outside_project, b"outside").expect("write outside project");

    assert!(delete_checked_in_root(&root, &outside_project).is_err());
    assert_eq!(
        fs::read(&outside_project).expect("outside project remains"),
        b"outside"
    );
}

#[test]
fn missing_valid_draft_delete_is_idempotent() {
    let sandbox = tempfile::tempdir().expect("tempdir");
    let root = sandbox.path().join("projects");
    let project = root.join("abc-1").join(PROJECT_FILE);

    assert!(!delete_checked_in_root(&root, &project).expect("missing delete"));
    assert!(!root.exists(), "delete unexpectedly created the root");
    fs::create_dir(&root).expect("create root");
    assert!(!delete_checked_in_root(&root, &project).expect("missing delete"));
}

#[test]
fn valid_delete_removes_only_its_target() {
    let sandbox = tempfile::tempdir().expect("tempdir");
    let root = sandbox.path().join("projects");
    let target = write_test_draft(&root, "abc-1", b"target");
    let survivor = write_test_draft(&root, "abc-2", b"survivor");
    fs::write(root.join("unrelated"), b"keep").expect("write unrelated file");

    assert!(delete_checked_in_root(&root, &target).expect("delete target"));
    assert!(!target.parent().expect("target parent").exists());
    assert_eq!(fs::read(&survivor).expect("survivor remains"), b"survivor");
    assert_eq!(
        fs::read(root.join("unrelated")).expect("unrelated remains"),
        b"keep"
    );
    assert!(!delete_checked_in_root(&root, &target).expect("repeat delete"));
    assert!(
        fs::read_dir(&root)
            .expect("read root")
            .flatten()
            .all(|entry| !entry
                .file_name()
                .to_string_lossy()
                .starts_with(TOMBSTONE_PREFIX))
    );
}

#[test]
fn list_ignores_tombstones_and_uses_a_stable_path_tie_break() {
    let sandbox = tempfile::tempdir().expect("tempdir");
    let root = sandbox.path().join("projects");
    let first = write_test_draft(&root, "abc-1", b"shared");
    let second_dir = root.join("abc-2");
    fs::create_dir(&second_dir).expect("create second draft");
    let second = second_dir.join(PROJECT_FILE);
    fs::hard_link(&first, &second).expect("hard link equal-mtime project");
    write_meta_in_root(&root, &first, "First").expect("first metadata");
    write_meta_in_root(&root, &second, "Second").expect("second metadata");

    let tombstone = root.join(format!("{TOMBSTONE_PREFIX}stale"));
    fs::create_dir(&tombstone).expect("create tombstone");
    fs::write(tombstone.join(PROJECT_FILE), b"deleted").expect("write tombstone project");
    write_test_draft(&root, "not-a-draft", b"invalid id");

    let drafts = list_in_root(&root).expect("list drafts");
    assert_eq!(drafts.len(), 2);
    assert_eq!(drafts[0].project, first);
    assert_eq!(drafts[1].project, second);
    assert_eq!(drafts[0].modified, drafts[1].modified);
    assert_eq!(drafts[0].name, "First");
    assert_eq!(drafts[1].name, "Second");
    assert!(
        drafts
            .windows(2)
            .all(|pair| pair[0].modified >= pair[1].modified)
    );
}

#[test]
fn import_rejects_non_file_sources_before_creating_a_draft() {
    let sandbox = tempfile::tempdir().expect("tempdir");
    let root = sandbox.path().join("projects");
    let source_dir = sandbox.path().join("source");
    fs::create_dir(&source_dir).expect("create source directory");

    assert!(import_external_in_root(&root, &source_dir).is_err());
    assert!(!root.exists());
}

#[test]
fn import_copies_transactionally_and_preserves_the_source() {
    let sandbox = tempfile::tempdir().expect("tempdir");
    let root = sandbox.path().join("projects");
    let source = sandbox.path().join("My Movie.cutlass");
    fs::write(&source, b"external project").expect("write source");

    let project = import_external_in_root(&root, &source).expect("import source");
    assert_eq!(
        fs::read(&source).expect("source remains"),
        b"external project"
    );
    assert_eq!(
        fs::read(&project).expect("imported project"),
        b"external project"
    );
    assert_eq!(
        read_name(project.parent().expect("draft directory")),
        "My Movie"
    );
}

#[test]
fn failed_copy_cleans_the_new_draft_directory() {
    struct FailingReader {
        sent_prefix: bool,
    }

    impl Read for FailingReader {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            if self.sent_prefix {
                return Err(io::Error::other("injected copy failure"));
            }
            self.sent_prefix = true;
            let prefix = b"partial";
            let length = prefix.len().min(buffer.len());
            buffer[..length].copy_from_slice(&prefix[..length]);
            Ok(length)
        }
    }

    let sandbox = tempfile::tempdir().expect("tempdir");
    let root = sandbox.path().join("projects");
    let mut reader = FailingReader { sent_prefix: false };
    let result = import_reader_in_root(&root, &mut reader, "Broken", write_meta_in_root);

    assert!(result.is_err());
    assert_root_is_empty(&root);
}

#[test]
fn failed_metadata_setup_cleans_the_new_draft_directory() {
    let sandbox = tempfile::tempdir().expect("tempdir");
    let root = sandbox.path().join("projects");
    let mut reader = Cursor::new(b"valid project");
    let result =
        import_reader_in_root(&root, &mut reader, "Broken", |_root, _project, _name| {
            Err(io::Error::other("injected metadata failure"))
        });

    assert!(result.is_err());
    assert_root_is_empty(&root);
}

#[test]
fn metadata_refuses_outside_paths_and_roundtrips_inside_root() {
    let sandbox = tempfile::tempdir().expect("tempdir");
    let root = sandbox.path().join("projects");
    let project = create_in_root(&root).expect("create draft");
    fs::write(&project, b"{}").expect("write project");

    write_meta_in_root(&root, &project, "My Movie").expect("write metadata");
    assert_eq!(
        read_name(project.parent().expect("draft directory")),
        "My Movie"
    );
    write_meta_in_root(&root, &project, "My Updated Movie").expect("replace metadata");
    assert_eq!(
        read_name(project.parent().expect("draft directory")),
        "My Updated Movie"
    );
    assert_no_meta_transaction_artifacts(project.parent().expect("draft directory"));

    let outside_dir = sandbox.path().join("outside").join("abc-1");
    fs::create_dir_all(&outside_dir).expect("create outside");
    let outside_project = outside_dir.join(PROJECT_FILE);
    fs::write(&outside_project, b"{}").expect("write outside project");
    assert!(write_meta_in_root(&root, &outside_project, "Outside").is_err());
    assert!(!meta_file(&outside_dir).exists());
}

#[test]
fn metadata_fallback_handles_windows_destination_error_kinds() {
    for error_kind in [
        io::ErrorKind::AlreadyExists,
        io::ErrorKind::PermissionDenied,
    ] {
        let sandbox = tempfile::tempdir().expect("tempdir");
        let root = sandbox.path().join("projects");
        let project = create_in_root(&root).expect("create draft");
        fs::write(&project, b"{}").expect("write project");
        write_meta_in_root(&root, &project, "Original").expect("write original metadata");

        let replace_fs = RenameFaultFs {
            failed_renames: vec![(1, error_kind)],
            rename_calls: Cell::new(0),
        };
        write_meta_in_root_with_ops(&root, &project, "Replacement", &replace_fs)
            .expect("fallback metadata replacement");

        assert_eq!(
            read_name(project.parent().expect("draft directory")),
            "Replacement"
        );
        assert_eq!(replace_fs.rename_calls.get(), 3);
        assert_no_meta_transaction_artifacts(project.parent().expect("draft directory"));
    }
}

#[test]
fn failed_metadata_publication_restores_the_valid_original() {
    let sandbox = tempfile::tempdir().expect("tempdir");
    let root = sandbox.path().join("projects");
    let project = create_in_root(&root).expect("create draft");
    fs::write(&project, b"{}").expect("write project");
    write_meta_in_root(&root, &project, "Known Good").expect("write original metadata");
    let replace_fs = RenameFaultFs {
        failed_renames: vec![
            (1, io::ErrorKind::AlreadyExists),
            (3, io::ErrorKind::PermissionDenied),
        ],
        rename_calls: Cell::new(0),
    };

    let error = write_meta_in_root_with_ops(&root, &project, "Replacement", &replace_fs)
        .expect_err("injected publication failure");

    assert!(
        error.to_string().contains("original metadata was restored"),
        "{error}"
    );
    assert_eq!(replace_fs.rename_calls.get(), 4);
    let dir = project.parent().expect("draft directory");
    assert_eq!(read_name(dir), "Known Good");
    let stored: DraftMeta =
        serde_json::from_slice(&fs::read(meta_file(dir)).expect("read restored metadata"))
            .expect("restored metadata remains valid JSON");
    assert_eq!(stored.name, "Known Good");
    assert_no_meta_transaction_artifacts(dir);
}

#[test]
fn metadata_names_are_utf8_safely_bounded_and_blank_names_fall_back() {
    let sandbox = tempfile::tempdir().expect("tempdir");
    let root = sandbox.path().join("projects");
    let project = create_in_root(&root).expect("create draft");
    fs::write(&project, b"{}").expect("write project");
    let dir = project.parent().expect("draft directory");

    let long_name = "é".repeat(MAX_PROJECT_NAME_BYTES);
    write_meta_in_root(&root, &project, &long_name).expect("write bounded metadata");
    let stored = read_name(dir);
    assert!(stored.len() <= MAX_PROJECT_NAME_BYTES);
    assert!(stored.is_char_boundary(stored.len()));

    write_meta_in_root(&root, &project, "   ").expect("write blank metadata");
    assert_eq!(read_name(dir), FALLBACK_NAME);
}

#[cfg(unix)]
#[test]
fn symlink_draft_directory_is_refused_and_not_listed() {
    use std::os::unix::fs::symlink;

    let sandbox = tempfile::tempdir().expect("tempdir");
    let root = sandbox.path().join("projects");
    fs::create_dir(&root).expect("create root");
    let outside = sandbox.path().join("outside");
    fs::create_dir(&outside).expect("create outside");
    let outside_project = outside.join(PROJECT_FILE);
    fs::write(&outside_project, b"outside").expect("write outside project");
    symlink(&outside, root.join("abc-1")).expect("symlink draft");
    let project = root.join("abc-1").join(PROJECT_FILE);

    assert!(resolve_draft_id_in_root(&root, "abc-1").is_err());
    assert!(delete_checked_in_root(&root, &project).is_err());
    assert!(write_meta_in_root(&root, &project, "Outside").is_err());
    assert_eq!(
        fs::read(&outside_project).expect("outside remains"),
        b"outside"
    );
    assert!(list_in_root(&root).expect("list").is_empty());
}

#[cfg(unix)]
#[test]
fn symlink_project_file_is_refused_and_not_listed() {
    use std::os::unix::fs::symlink;

    let sandbox = tempfile::tempdir().expect("tempdir");
    let root = sandbox.path().join("projects");
    let dir = root.join("abc-1");
    fs::create_dir_all(&dir).expect("create draft");
    let outside = sandbox.path().join("outside.cutlass");
    fs::write(&outside, b"outside").expect("write outside");
    let project = dir.join(PROJECT_FILE);
    symlink(&outside, &project).expect("symlink project");

    assert!(resolve_draft_id_in_root(&root, "abc-1").is_err());
    assert!(delete_checked_in_root(&root, &project).is_err());
    assert!(write_meta_in_root(&root, &project, "Outside").is_err());
    assert_eq!(fs::read(&outside).expect("outside remains"), b"outside");
    assert!(dir.exists());
    assert!(list_in_root(&root).expect("list").is_empty());
}

#[cfg(unix)]
#[test]
fn symlink_metadata_destination_is_never_rewritten() {
    use std::os::unix::fs::symlink;

    let sandbox = tempfile::tempdir().expect("tempdir");
    let root = sandbox.path().join("projects");
    let project = write_test_draft(&root, "abc-1", b"{}");
    let dir = project.parent().expect("draft directory");
    let outside = sandbox.path().join("outside-meta.json");
    let original = br#"{"name":"Outside"}"#;
    fs::write(&outside, original).expect("write outside metadata");
    symlink(&outside, meta_file(dir)).expect("symlink metadata");

    assert!(write_meta_in_root(&root, &project, "Replacement").is_err());
    assert_eq!(
        fs::read(&outside).expect("outside metadata remains"),
        original
    );
    assert!(
        fs::symlink_metadata(meta_file(dir))
            .expect("metadata symlink remains")
            .file_type()
            .is_symlink()
    );
}

#[cfg(unix)]
#[test]
fn symlink_root_is_refused_without_following_it() {
    use std::os::unix::fs::symlink;

    let sandbox = tempfile::tempdir().expect("tempdir");
    let actual_root = sandbox.path().join("actual");
    fs::create_dir(&actual_root).expect("create actual root");
    let root_link = sandbox.path().join("projects");
    symlink(&actual_root, &root_link).expect("symlink root");
    let project = root_link.join("abc-1").join(PROJECT_FILE);

    assert!(create_in_root(&root_link).is_err());
    assert!(delete_checked_in_root(&root_link, &project).is_err());
    assert!(write_meta_in_root(&root_link, &project, "Outside").is_err());
    assert!(list_in_root(&root_link).is_err());
    assert!(
        fs::read_dir(&actual_root)
            .expect("read actual root")
            .next()
            .is_none()
    );
}

#[test]
fn relative_time_reads_in_words() {
    let now = SystemTime::now();
    assert_eq!(relative_time(now), "just now");
    assert_eq!(
        relative_time(now - std::time::Duration::from_secs(60)),
        "1 minute ago"
    );
    assert_eq!(
        relative_time(now - std::time::Duration::from_secs(7200)),
        "2 hours ago"
    );
}

fn write_test_draft(root: &Path, id: &str, contents: &[u8]) -> PathBuf {
    let dir = root.join(id);
    fs::create_dir_all(&dir).expect("create test draft");
    let project = dir.join(PROJECT_FILE);
    fs::write(&project, contents).expect("write test project");
    project
}

fn assert_root_is_empty(root: &Path) {
    assert!(root.is_dir(), "import did not create its root");
    assert!(
        fs::read_dir(root).expect("read root").next().is_none(),
        "failed import left files behind"
    );
}

fn assert_no_meta_transaction_artifacts(dir: &Path) {
    let artifacts: Vec<_> = fs::read_dir(dir)
        .expect("read draft directory")
        .flatten()
        .filter(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            name.starts_with(META_BACKUP_PREFIX) || name.starts_with(".meta-write-")
        })
        .collect();
    assert!(artifacts.is_empty(), "{artifacts:?}");
}
