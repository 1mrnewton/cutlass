use super::*;

/// Shown for a draft whose meta is missing or empty.
pub(super) const FALLBACK_NAME: &str = "Untitled project";
/// Prefix for rollback copies used by cross-platform metadata replacement.
pub(super) const META_BACKUP_PREFIX: &str = ".meta-backup-";
/// Project names are UI labels, not an unbounded storage channel.
pub(super) const MAX_PROJECT_NAME_BYTES: usize = 512;
pub(super) const MAX_META_BYTES: u64 = 8 * 1024;

/// Display name persisted alongside a draft so the gallery needn't parse the
/// whole project file.
#[derive(Debug, Default, Serialize, Deserialize)]
pub(super) struct DraftMeta {
    pub(super) name: String,
}

pub(super) trait MetaReplaceFs {
    fn rename(&self, from: &Path, to: &Path) -> io::Result<()>;
    fn remove_file(&self, path: &Path) -> io::Result<()>;
    fn symlink_metadata(&self, path: &Path) -> io::Result<fs::Metadata>;
}

pub(super) struct StdMetaReplaceFs;

impl MetaReplaceFs for StdMetaReplaceFs {
    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        fs::rename(from, to)
    }

    fn remove_file(&self, path: &Path) -> io::Result<()> {
        fs::remove_file(path)
    }

    fn symlink_metadata(&self, path: &Path) -> io::Result<fs::Metadata> {
        fs::symlink_metadata(path)
    }
}

pub(super) fn read_name(dir: &Path) -> String {
    read_name_checked(dir).unwrap_or_else(|| FALLBACK_NAME.to_owned())
}

pub(super) fn write_meta_in_root(root: &Path, project: &Path, name: &str) -> io::Result<()> {
    write_meta_in_root_with_ops(root, project, name, &StdMetaReplaceFs)
}

pub(super) fn write_meta_in_root_with_ops(
    root: &Path,
    project: &Path,
    name: &str,
    replace_fs: &impl MetaReplaceFs,
) -> io::Result<()> {
    let identity = validate_project_identity(root, project)?;
    let canonical_root = canonical_existing_root(root)?
        .ok_or_else(|| path_error(io::ErrorKind::NotFound, "draft root does not exist", root))?;
    let Some(canonical_dir) = canonical_draft_dir(&canonical_root, &identity.id)? else {
        return Err(path_error(
            io::ErrorKind::NotFound,
            "draft directory does not exist",
            project,
        ));
    };
    safe_project_entry(&canonical_dir)?;
    ensure_safe_meta_entry(&canonical_dir)?;

    let meta = DraftMeta {
        name: bounded_project_name(name),
    };
    let json = serde_json::to_vec(&meta)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if u64::try_from(json.len()).unwrap_or(u64::MAX) > MAX_META_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bounded draft metadata exceeded its size limit",
        ));
    }

    let (temp_path, mut temp_file) = open_unique_file(&canonical_dir, "meta-write")?;
    let mut committed = false;
    let write_result = (|| -> io::Result<()> {
        temp_file
            .write_all(&json)
            .map_err(|error| contextual_error(error, "write draft metadata", &temp_path))?;
        temp_file
            .sync_all()
            .map_err(|error| contextual_error(error, "sync draft metadata", &temp_path))?;
        drop(temp_file);

        verify_root_mapping(root, &canonical_root)?;
        let Some(current_dir) = canonical_draft_dir(&canonical_root, &identity.id)? else {
            return Err(path_error(
                io::ErrorKind::NotFound,
                "draft directory disappeared before metadata publication",
                project,
            ));
        };
        if current_dir != canonical_dir {
            return Err(path_error(
                io::ErrorKind::PermissionDenied,
                "draft directory changed before metadata publication",
                project,
            ));
        }
        safe_project_entry(&current_dir)?;
        ensure_safe_meta_entry(&current_dir)?;

        let destination = current_dir.join(META_FILE);
        replace_meta_file(&temp_path, &destination, replace_fs)?;
        committed = true;
        if let Err(error) = sync_directory(&current_dir) {
            warn!(
                path = %bounded_path(&current_dir),
                error = %error,
                "metadata committed but draft directory sync failed"
            );
        }
        Ok(())
    })();

    if !committed
        && let Err(error) = fs::remove_file(&temp_path)
        && error.kind() != io::ErrorKind::NotFound
    {
        warn!(
            path = %bounded_path(&temp_path),
            error = %error,
            "failed to remove a staged metadata file"
        );
    }
    write_result
}

pub(super) fn replace_meta_file(
    temporary: &Path,
    destination: &Path,
    replace_fs: &impl MetaReplaceFs,
) -> io::Result<()> {
    if temporary.parent() != destination.parent() {
        return Err(path_error(
            io::ErrorKind::InvalidInput,
            "metadata temporary file is not beside its destination",
            temporary,
        ));
    }
    if !regular_meta_file_exists(temporary, "metadata temporary file", replace_fs)? {
        return Err(path_error(
            io::ErrorKind::NotFound,
            "metadata temporary file does not exist",
            temporary,
        ));
    }

    let atomic_error = match replace_fs.rename(temporary, destination) {
        Ok(()) => return Ok(()),
        Err(error) => error,
    };

    // Windows destination-exists failures have surfaced as both
    // `AlreadyExists` and `PermissionDenied`. Inspecting the destination
    // without following links is more robust than keying recovery to one
    // error kind, and keeps the Unix one-rename success path unchanged.
    match regular_meta_file_exists(destination, "metadata destination", replace_fs) {
        Ok(true) => {}
        Ok(false) => {
            return Err(contextual_error(
                atomic_error,
                "publish draft metadata with atomic rename",
                destination,
            ));
        }
        Err(error) => return Err(error),
    }

    let backup = unique_meta_backup_path(destination, replace_fs).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "atomic metadata replacement failed ({atomic_error}); \
                 could not allocate a rollback backup: {error}"
            ),
        )
    })?;

    if !regular_meta_file_exists(temporary, "metadata temporary file", replace_fs)? {
        return Err(path_error(
            io::ErrorKind::NotFound,
            "metadata temporary file disappeared before fallback publication",
            temporary,
        ));
    }
    if !regular_meta_file_exists(destination, "metadata destination", replace_fs)? {
        return Err(path_error(
            io::ErrorKind::NotFound,
            "metadata destination disappeared before fallback publication",
            destination,
        ));
    }
    ensure_meta_path_absent(&backup, "metadata rollback backup", replace_fs)?;

    if let Err(backup_error) = replace_fs.rename(destination, &backup) {
        return Err(io::Error::new(
            backup_error.kind(),
            format!(
                "atomic metadata replacement failed ({atomic_error}); moving the existing \
                 metadata to a rollback backup also failed: {backup_error}; the existing \
                 metadata was left in place"
            ),
        ));
    }

    let install_result = (|| -> io::Result<()> {
        if !regular_meta_file_exists(&backup, "metadata rollback backup", replace_fs)? {
            return Err(path_error(
                io::ErrorKind::NotFound,
                "metadata rollback backup disappeared before publication",
                &backup,
            ));
        }
        ensure_meta_path_absent(destination, "metadata destination", replace_fs)?;
        replace_fs
            .rename(temporary, destination)
            .map_err(|error| contextual_error(error, "publish replacement metadata", destination))
    })();

    match install_result {
        Ok(()) => {
            if let Err(error) = cleanup_meta_backup_after_commit(destination, &backup, replace_fs) {
                warn!(
                    path = %bounded_path(&backup),
                    error = %error,
                    "metadata committed but rollback backup cleanup failed"
                );
            }
            Ok(())
        }
        Err(install_error) => match restore_meta_backup(&backup, destination, replace_fs) {
            Ok(()) => Err(io::Error::new(
                install_error.kind(),
                format!(
                    "could not publish replacement metadata after backing up the existing file: \
                     {install_error}; the original metadata was restored"
                ),
            )),
            Err(rollback_error) => Err(io::Error::new(
                install_error.kind(),
                format!(
                    "could not publish replacement metadata after backing up the existing file: \
                     {install_error}; restoring the original metadata failed: {rollback_error}; \
                     the original metadata was retained at '{}'",
                    bounded_path(&backup)
                ),
            )),
        },
    }
}

pub(super) fn unique_meta_backup_path(
    destination: &Path,
    replace_fs: &impl MetaReplaceFs,
) -> io::Result<PathBuf> {
    let parent = destination.parent().ok_or_else(|| {
        path_error(
            io::ErrorKind::InvalidInput,
            "metadata destination has no parent directory",
            destination,
        )
    })?;
    for _ in 0..UNIQUE_PATH_ATTEMPTS {
        let candidate = parent.join(format!(
            "{META_BACKUP_PREFIX}{:x}-{}",
            std::process::id(),
            new_id()
        ));
        match replace_fs.symlink_metadata(&candidate) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(candidate),
            Ok(_) => {}
            Err(error) => {
                return Err(contextual_error(
                    error,
                    "inspect prospective metadata rollback backup",
                    &candidate,
                ));
            }
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not allocate a unique metadata rollback backup",
    ))
}

pub(super) fn regular_meta_file_exists(
    path: &Path,
    role: &str,
    replace_fs: &impl MetaReplaceFs,
) -> io::Result<bool> {
    match replace_fs.symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(path_error(
            io::ErrorKind::PermissionDenied,
            &format!("{role} must not be a symlink"),
            path,
        )),
        Ok(metadata) if !metadata.is_file() => Err(path_error(
            io::ErrorKind::InvalidInput,
            &format!("{role} is not a regular file"),
            path,
        )),
        Ok(_) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(contextual_error(error, "inspect metadata file", path)),
    }
}

pub(super) fn ensure_meta_path_absent(
    path: &Path,
    role: &str,
    replace_fs: &impl MetaReplaceFs,
) -> io::Result<()> {
    match replace_fs.symlink_metadata(path) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Ok(_) => Err(path_error(
            io::ErrorKind::AlreadyExists,
            &format!("{role} unexpectedly exists"),
            path,
        )),
        Err(error) => Err(contextual_error(error, "inspect metadata path", path)),
    }
}

pub(super) fn restore_meta_backup(
    backup: &Path,
    destination: &Path,
    replace_fs: &impl MetaReplaceFs,
) -> io::Result<()> {
    if !regular_meta_file_exists(backup, "metadata rollback backup", replace_fs)? {
        return Err(path_error(
            io::ErrorKind::NotFound,
            "metadata rollback backup is missing",
            backup,
        ));
    }
    ensure_meta_path_absent(destination, "metadata rollback destination", replace_fs)?;
    replace_fs
        .rename(backup, destination)
        .map_err(|error| contextual_error(error, "restore metadata rollback backup", destination))
}

pub(super) fn cleanup_meta_backup_after_commit(
    destination: &Path,
    backup: &Path,
    replace_fs: &impl MetaReplaceFs,
) -> io::Result<()> {
    if !regular_meta_file_exists(destination, "committed metadata destination", replace_fs)? {
        return Err(path_error(
            io::ErrorKind::NotFound,
            "committed metadata destination disappeared",
            destination,
        ));
    }
    if !regular_meta_file_exists(backup, "metadata rollback backup", replace_fs)? {
        return Ok(());
    }
    replace_fs
        .remove_file(backup)
        .map_err(|error| contextual_error(error, "remove metadata rollback backup", backup))
}

pub(super) fn open_unique_file(dir: &Path, purpose: &str) -> io::Result<(PathBuf, File)> {
    for _ in 0..UNIQUE_PATH_ATTEMPTS {
        let path = dir.join(format!(
            ".{purpose}-{:x}-{}.tmp",
            std::process::id(),
            new_id()
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(contextual_error(
                    error,
                    "create transactional draft file",
                    &path,
                ));
            }
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not allocate a unique transactional draft file",
    ))
}

pub(super) fn read_name_checked(dir: &Path) -> Option<String> {
    let path = meta_file(dir);
    let metadata = fs::symlink_metadata(&path).ok()?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > MAX_META_BYTES {
        return None;
    }
    let file = File::open(&path).ok()?;
    let opened_metadata = file.metadata().ok()?;
    if !opened_metadata.is_file() || opened_metadata.len() > MAX_META_BYTES {
        return None;
    }
    let mut bytes = Vec::new();
    file.take(MAX_META_BYTES + 1).read_to_end(&mut bytes).ok()?;
    if u64::try_from(bytes.len()).ok()? > MAX_META_BYTES {
        return None;
    }
    let meta: DraftMeta = serde_json::from_slice(&bytes).ok()?;
    let name = bounded_project_name(&meta.name);
    (!name.trim().is_empty()).then_some(name)
}

pub(super) fn bounded_project_name(name: &str) -> String {
    if name.len() <= MAX_PROJECT_NAME_BYTES {
        return name.to_owned();
    }
    let mut end = MAX_PROJECT_NAME_BYTES;
    while name.get(..end).is_none() {
        end = end.saturating_sub(1);
    }
    name.get(..end).unwrap_or_default().to_owned()
}
