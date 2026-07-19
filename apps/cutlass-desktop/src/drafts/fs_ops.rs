use super::*;

#[derive(Debug)]
pub(super) struct DraftSlot {
    pub(super) project: PathBuf,
    pub(super) canonical_root: PathBuf,
    pub(super) canonical_dir: PathBuf,
}

pub(super) fn create_in_root(root: &Path) -> io::Result<PathBuf> {
    create_slot_in_root(root).map(|slot| slot.project)
}

pub(super) fn create_slot_in_root(root: &Path) -> io::Result<DraftSlot> {
    let canonical_root = ensure_root(root)?;
    loop {
        let id = new_id();
        let canonical_dir = canonical_root.join(&id);
        match fs::create_dir(&canonical_dir) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(contextual_error(
                    error,
                    "create draft directory",
                    &canonical_dir,
                ));
            }
        }

        let project = project_file(&root.join(&id));
        let setup = (|| -> io::Result<PathBuf> {
            validate_project_identity(root, &project)?;
            verify_root_mapping(root, &canonical_root)?;
            let Some(verified_dir) = canonical_draft_dir(&canonical_root, &id)? else {
                return Err(path_error(
                    io::ErrorKind::NotFound,
                    "new draft directory disappeared",
                    &canonical_dir,
                ));
            };
            Ok(verified_dir)
        })();

        match setup {
            Ok(verified_dir) => {
                return Ok(DraftSlot {
                    project,
                    canonical_root,
                    canonical_dir: verified_dir,
                });
            }
            Err(error) => {
                if let Err(cleanup_error) = fs::remove_dir(&canonical_dir) {
                    warn!(
                        path = %bounded_path(&canonical_dir),
                        error = %cleanup_error,
                        "failed to clean an uninitialized draft directory"
                    );
                }
                return Err(error);
            }
        }
    }
}

pub(super) fn import_external_in_root(root: &Path, source: &Path) -> io::Result<PathBuf> {
    // Opening first intentionally follows a user-selected source symlink. The
    // resulting handle is read-only and is accepted only when it is a file.
    let mut input = File::open(source)
        .map_err(|error| contextual_error(error, "open import source", source))?;
    let source_metadata = input
        .metadata()
        .map_err(|error| contextual_error(error, "inspect import source", source))?;
    if !source_metadata.is_file() {
        return Err(path_error(
            io::ErrorKind::InvalidInput,
            "import source is not a regular file",
            source,
        ));
    }

    let name = source
        .file_stem()
        .map(|stem| bounded_project_name(&stem.to_string_lossy()))
        .unwrap_or_default();
    import_reader_in_root(root, &mut input, &name, write_meta_in_root)
}

pub(super) fn import_reader_in_root<R, F>(
    root: &Path,
    reader: &mut R,
    name: &str,
    metadata_writer: F,
) -> io::Result<PathBuf>
where
    R: Read,
    F: FnOnce(&Path, &Path, &str) -> io::Result<()>,
{
    let slot = create_slot_in_root(root)?;
    let mut staged_path = None;
    let operation = (|| -> io::Result<PathBuf> {
        let (path, mut output) = open_unique_file(&slot.canonical_dir, "project-import")?;
        staged_path = Some(path.clone());
        io::copy(reader, &mut output)
            .map_err(|error| contextual_error(error, "copy imported project", &path))?;
        output
            .sync_all()
            .map_err(|error| contextual_error(error, "sync imported project", &path))?;
        drop(output);

        metadata_writer(root, &slot.project, name)?;
        verify_root_mapping(root, &slot.canonical_root)?;
        let identity = validate_project_identity(root, &slot.project)?;
        let Some(current_dir) = canonical_draft_dir(&slot.canonical_root, &identity.id)? else {
            return Err(path_error(
                io::ErrorKind::NotFound,
                "new draft directory disappeared before publication",
                &slot.canonical_dir,
            ));
        };
        if current_dir != slot.canonical_dir {
            return Err(path_error(
                io::ErrorKind::PermissionDenied,
                "new draft directory changed before publication",
                &slot.canonical_dir,
            ));
        }
        if safe_project_entry(&current_dir)? {
            return Err(path_error(
                io::ErrorKind::AlreadyExists,
                "project destination unexpectedly exists",
                &slot.project,
            ));
        }

        let target = current_dir.join(PROJECT_FILE);
        fs::rename(&path, &target)
            .map_err(|error| contextual_error(error, "publish imported project", &target))?;
        staged_path = None;
        if let Err(error) = sync_directory(&current_dir) {
            warn!(
                path = %bounded_path(&current_dir),
                error = %error,
                "import committed but draft directory sync failed"
            );
        }
        Ok(slot.project.clone())
    })();

    if let Some(path) = staged_path
        && let Err(error) = fs::remove_file(&path)
        && error.kind() != io::ErrorKind::NotFound
    {
        warn!(
            path = %bounded_path(&path),
            error = %error,
            "failed to remove a staged import file"
        );
    }

    match operation {
        Ok(project) => Ok(project),
        Err(error) => {
            match cleanup_created_slot(&slot) {
                Ok(true) => {}
                Ok(false) => warn!(
                    path = %bounded_path(&slot.canonical_dir),
                    "failed import draft disappeared before cleanup"
                ),
                Err(cleanup_error) => warn!(
                    path = %bounded_path(&slot.canonical_dir),
                    error = %cleanup_error,
                    "failed to clean draft after import failure"
                ),
            }
            Err(error)
        }
    }
}

pub(super) fn delete_checked_in_root(root: &Path, project: &Path) -> io::Result<bool> {
    let identity = validate_project_identity(root, project)?;
    let Some(canonical_root) = canonical_existing_root(root)? else {
        return Ok(false);
    };
    let Some(canonical_dir) = canonical_draft_dir(&canonical_root, &identity.id)? else {
        return Ok(false);
    };
    safe_project_entry(&canonical_dir)?;
    verify_root_mapping(root, &canonical_root)?;

    let Some(current_dir) = canonical_draft_dir(&canonical_root, &identity.id)? else {
        return Ok(false);
    };
    if current_dir != canonical_dir {
        return Err(path_error(
            io::ErrorKind::PermissionDenied,
            "draft directory changed before deletion",
            project,
        ));
    }
    safe_project_entry(&current_dir)?;
    tombstone_and_remove(&canonical_root, &current_dir)
}

pub(super) fn list_in_root(root: &Path) -> io::Result<Vec<DraftSummary>> {
    let Some(canonical_root) = canonical_existing_root(root)? else {
        return Ok(Vec::new());
    };
    let entries = fs::read_dir(&canonical_root)
        .map_err(|error| contextual_error(error, "read draft root", &canonical_root))?;
    let mut drafts = Vec::new();

    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let Some(id) = file_name.to_str() else {
            continue;
        };
        if id.starts_with(TOMBSTONE_PREFIX) || !is_valid_draft_id(id) {
            continue;
        }

        let Ok(entry_type) = entry.file_type() else {
            continue;
        };
        if entry_type.is_symlink() || !entry_type.is_dir() {
            continue;
        }

        let dir = entry.path();
        let Ok(dir_metadata) = fs::symlink_metadata(&dir) else {
            continue;
        };
        if dir_metadata.file_type().is_symlink() || !dir_metadata.is_dir() {
            continue;
        }

        let actual_project = dir.join(PROJECT_FILE);
        let Ok(project_metadata) = fs::symlink_metadata(&actual_project) else {
            continue;
        };
        if project_metadata.file_type().is_symlink() || !project_metadata.is_file() {
            continue;
        }
        let Ok(modified) = project_metadata.modified() else {
            continue;
        };

        drafts.push(DraftSummary {
            name: read_name(&dir),
            project: root.join(id).join(PROJECT_FILE),
            modified,
        });
    }

    verify_root_mapping(root, &canonical_root)?;
    drafts.sort_by(|left, right| {
        right
            .modified
            .cmp(&left.modified)
            .then_with(|| left.project.cmp(&right.project))
    });
    Ok(drafts)
}
