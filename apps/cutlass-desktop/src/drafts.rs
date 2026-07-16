//! App-owned project drafts (CapCut-style storage).
//!
//! Cutlass owns every project: there is no user-chosen `.cutlass` path. Each
//! project lives in its own directory under `projects/` in the per-user OS
//! data dir (see [`crate::paths`]) and auto-saves continuously, so the user
//! never saves by hand and no edit is lost on a clean exit. A directory holds
//! the project itself (`project.cutlass` — a plain project file the engine
//! reads and writes through the normal `Save`/`Load` path) and a small
//! `meta.json` sidecar caching the display name so the launch gallery can
//! list projects without parsing every project file.
//!
//! `.cutlass` files are no longer a user-facing concept; they enter via
//! [`import_external`] (Open file…) and leave via Export. The draft directory
//! is the identity — addressed everywhere by its `project.cutlass` path, so
//! the existing path-keyed engine plumbing (`Save`/`Load`) is reused as-is.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tracing::warn;

/// File name of the project inside a draft directory.
const PROJECT_FILE: &str = "project.cutlass";
/// File name of the display-name sidecar inside a draft directory.
const META_FILE: &str = "meta.json";
/// Shown for a draft whose meta is missing or empty.
const FALLBACK_NAME: &str = "Untitled project";
/// Prefix for a draft directory after deletion has been committed.
const TOMBSTONE_PREFIX: &str = ".cutlass-delete-";
/// Prefix for rollback copies used by cross-platform metadata replacement.
const META_BACKUP_PREFIX: &str = ".meta-backup-";
/// The two `new_id` groups are a `u128` timestamp and a `u64` counter.
const ID_TIME_HEX_MAX: usize = 32;
const ID_SEQUENCE_HEX_MAX: usize = 16;
/// Project names are UI labels, not an unbounded storage channel.
const MAX_PROJECT_NAME_BYTES: usize = 512;
const MAX_META_BYTES: u64 = 8 * 1024;
const MAX_ERROR_PATH_CHARS: usize = 240;
const UNIQUE_PATH_ATTEMPTS: usize = 128;

/// `projects/` in the per-user OS data dir (see [`crate::paths`]):
/// `%APPDATA%\Cutlass` on Windows, `~/Library/Application Support/Cutlass` on
/// macOS, `~/.local/share/Cutlass` on Linux.
pub fn root_dir() -> PathBuf {
    crate::paths::data_dir().join("projects")
}

/// The project file inside a draft directory.
pub fn project_file(dir: &Path) -> PathBuf {
    dir.join(PROJECT_FILE)
}

/// Resolve a canonical draft id to its app-owned lexical project path.
///
/// The draft directory and project file must already exist as real directory
/// and regular-file entries. Filesystem paths are never accepted as ids.
pub(crate) fn resolve_draft_id(draft_id: &str) -> io::Result<PathBuf> {
    resolve_draft_id_in_root(&root_dir(), draft_id)
}

/// Extract the canonical id from an exact app-owned lexical project path.
///
/// This validates identity only: neither the draft directory nor project file
/// needs to exist. That keeps newly created, not-yet-saved sessions addressable.
pub(crate) fn draft_id_from_project(project: &Path) -> io::Result<String> {
    draft_id_from_project_in_root(&root_dir(), project)
}

fn meta_file(dir: &Path) -> PathBuf {
    dir.join(META_FILE)
}

/// Display name + last-modified time for one draft, for the launch gallery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftSummary {
    /// Display name (from the meta sidecar, or a fallback).
    pub name: String,
    /// The draft's `project.cutlass` path — the identity used to open it.
    pub project: PathBuf,
    /// When the project file was last written (newest first in the gallery).
    pub modified: SystemTime,
}

/// Display name persisted alongside a draft so the gallery needn't parse the
/// whole project file.
#[derive(Debug, Default, Serialize, Deserialize)]
struct DraftMeta {
    name: String,
}

#[derive(Debug)]
struct DraftIdentity {
    id: String,
}

#[derive(Debug)]
struct DraftSlot {
    project: PathBuf,
    canonical_root: PathBuf,
    canonical_dir: PathBuf,
}

trait MetaReplaceFs {
    fn rename(&self, from: &Path, to: &Path) -> io::Result<()>;
    fn remove_file(&self, path: &Path) -> io::Result<()>;
    fn symlink_metadata(&self, path: &Path) -> io::Result<fs::Metadata>;
}

struct StdMetaReplaceFs;

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

/// Create a fresh, empty draft directory and return its `project.cutlass`
/// path (the file itself is written by the first save once the engine binds
/// to it). Ids are time-based with a per-process counter so two creates in
/// the same instant never collide.
pub fn create() -> std::io::Result<PathBuf> {
    create_in_root(&root_dir())
}

/// Copy an external `.cutlass` file into a new draft directory (Open file…),
/// naming the draft after the source file's stem. Returns the new draft's
/// `project.cutlass` path. The original file on disk is left untouched — it
/// was only an import source. A user-selected source symlink is accepted, but
/// only as a read-only boundary: its opened target must be a regular file.
pub fn import_external(source: &Path) -> std::io::Result<PathBuf> {
    import_external_in_root(&root_dir(), source)
}

/// Cache `name` in the draft's meta sidecar. Called on every auto-save so the
/// gallery name tracks the project name. Failures only log — the project file
/// is the source of truth; a stale gallery name never loses work.
pub fn write_meta(project: &Path, name: &str) {
    if let Err(error) = write_meta_in_root(&root_dir(), project, name) {
        warn!(
            project = %bounded_path(project),
            error = %error,
            "draft meta write failed"
        );
    }
}

/// Delete a draft (its whole directory), addressed by its `project.cutlass`
/// path. Missing files are fine.
pub fn delete(project: &Path) {
    if let Err(error) = delete_checked(project) {
        warn!(
            project = %bounded_path(project),
            error = %error,
            "draft deletion refused or failed"
        );
    }
}

/// Safely delete an app-owned draft.
///
/// Returns `true` once the draft has been atomically removed from the live
/// namespace. Tombstone cleanup failures are logged, but still return `true`
/// because the draft is no longer live.
pub fn delete_checked(project: &Path) -> io::Result<bool> {
    delete_checked_in_root(&root_dir(), project)
}

/// Every draft, newest-modified first — the launch gallery's source. A draft
/// directory without a readable `project.cutlass` is skipped (a half-created
/// or corrupt slot never crashes the gallery).
pub fn list() -> Vec<DraftSummary> {
    match list_in_root(&root_dir()) {
        Ok(drafts) => drafts,
        Err(error) => {
            warn!(error = %error, "draft listing failed");
            Vec::new()
        }
    }
}

/// Checked counterpart to [`list`] for callers that must distinguish an empty
/// draft store from a store that could not be read.
pub(crate) fn list_checked() -> io::Result<Vec<DraftSummary>> {
    list_in_root(&root_dir())
}

fn read_name(dir: &Path) -> String {
    read_name_checked(dir).unwrap_or_else(|| FALLBACK_NAME.to_owned())
}

fn resolve_draft_id_in_root(root: &Path, draft_id: &str) -> io::Result<PathBuf> {
    if !is_valid_draft_id(draft_id) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "draft id must be canonical lowercase hexadecimal time-sequence",
        ));
    }

    let project = root.join(draft_id).join(PROJECT_FILE);
    let identity = validate_project_identity(root, &project)?;
    let canonical_root = canonical_existing_root(root)?
        .ok_or_else(|| path_error(io::ErrorKind::NotFound, "draft root does not exist", root))?;
    let Some(canonical_dir) = canonical_draft_dir(&canonical_root, &identity.id)? else {
        return Err(path_error(
            io::ErrorKind::NotFound,
            "draft directory does not exist",
            &project,
        ));
    };
    if !safe_project_entry(&canonical_dir)? {
        return Err(path_error(
            io::ErrorKind::NotFound,
            "draft project file does not exist",
            &project,
        ));
    }

    verify_root_mapping(root, &canonical_root)?;
    let Some(current_dir) = canonical_draft_dir(&canonical_root, &identity.id)? else {
        return Err(path_error(
            io::ErrorKind::NotFound,
            "draft directory disappeared during resolution",
            &project,
        ));
    };
    if current_dir != canonical_dir {
        return Err(path_error(
            io::ErrorKind::PermissionDenied,
            "draft directory changed during resolution",
            &project,
        ));
    }
    if !safe_project_entry(&current_dir)? {
        return Err(path_error(
            io::ErrorKind::NotFound,
            "draft project file disappeared during resolution",
            &project,
        ));
    }
    verify_root_mapping(root, &canonical_root)?;
    Ok(project)
}

fn draft_id_from_project_in_root(root: &Path, project: &Path) -> io::Result<String> {
    validate_project_identity(root, project).map(|identity| identity.id)
}

fn create_in_root(root: &Path) -> io::Result<PathBuf> {
    create_slot_in_root(root).map(|slot| slot.project)
}

fn create_slot_in_root(root: &Path) -> io::Result<DraftSlot> {
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

fn import_external_in_root(root: &Path, source: &Path) -> io::Result<PathBuf> {
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

fn import_reader_in_root<R, F>(
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

    if let Some(path) = staged_path {
        if let Err(error) = fs::remove_file(&path) {
            if error.kind() != io::ErrorKind::NotFound {
                warn!(
                    path = %bounded_path(&path),
                    error = %error,
                    "failed to remove a staged import file"
                );
            }
        }
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

fn write_meta_in_root(root: &Path, project: &Path, name: &str) -> io::Result<()> {
    write_meta_in_root_with_ops(root, project, name, &StdMetaReplaceFs)
}

fn write_meta_in_root_with_ops(
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

    if !committed {
        if let Err(error) = fs::remove_file(&temp_path) {
            if error.kind() != io::ErrorKind::NotFound {
                warn!(
                    path = %bounded_path(&temp_path),
                    error = %error,
                    "failed to remove a staged metadata file"
                );
            }
        }
    }
    write_result
}

fn replace_meta_file(
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

fn unique_meta_backup_path(
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

fn regular_meta_file_exists(
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

fn ensure_meta_path_absent(
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

fn restore_meta_backup(
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

fn cleanup_meta_backup_after_commit(
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

fn delete_checked_in_root(root: &Path, project: &Path) -> io::Result<bool> {
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

fn list_in_root(root: &Path) -> io::Result<Vec<DraftSummary>> {
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

fn validate_project_identity(root: &Path, project: &Path) -> io::Result<DraftIdentity> {
    if project.file_name() != Some(std::ffi::OsStr::new(PROJECT_FILE)) {
        return Err(invalid_project_path(
            project,
            "filename must be exactly project.cutlass",
        ));
    }
    let Some(dir) = project.parent() else {
        return Err(invalid_project_path(
            project,
            "project path has no draft directory",
        ));
    };
    let Some(id_os) = dir.file_name() else {
        return Err(invalid_project_path(
            project,
            "project path has no draft id",
        ));
    };
    let Some(id) = id_os.to_str() else {
        return Err(invalid_project_path(project, "draft id is not valid UTF-8"));
    };
    if !is_valid_draft_id(id) {
        return Err(invalid_project_path(
            project,
            "draft id must be canonical lowercase hexadecimal time-sequence",
        ));
    }

    let expected = root.join(id).join(PROJECT_FILE);
    if project.as_os_str() != expected.as_os_str() {
        return Err(invalid_project_path(
            project,
            "path must be exactly root/draft-id/project.cutlass",
        ));
    }
    Ok(DraftIdentity { id: id.to_owned() })
}

fn is_valid_draft_id(id: &str) -> bool {
    if id.len() < 3 || id.len() > ID_TIME_HEX_MAX + ID_SEQUENCE_HEX_MAX + 1 {
        return false;
    }
    let Some((time, sequence)) = id.split_once('-') else {
        return false;
    };
    !sequence.contains('-')
        && is_canonical_hex_group(time, ID_TIME_HEX_MAX)
        && is_canonical_hex_group(sequence, ID_SEQUENCE_HEX_MAX)
}

fn is_canonical_hex_group(group: &str, max_len: usize) -> bool {
    !group.is_empty()
        && group.len() <= max_len
        && (group.len() == 1 || !group.starts_with('0'))
        && group
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn ensure_root(root: &Path) -> io::Result<PathBuf> {
    if let Some(canonical_root) = canonical_existing_root(root)? {
        return Ok(canonical_root);
    }
    fs::create_dir_all(root).map_err(|error| contextual_error(error, "create draft root", root))?;
    canonical_existing_root(root)?.ok_or_else(|| {
        path_error(
            io::ErrorKind::NotFound,
            "draft root disappeared after creation",
            root,
        )
    })
}

fn canonical_existing_root(root: &Path) -> io::Result<Option<PathBuf>> {
    let metadata = match fs::symlink_metadata(root) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(contextual_error(error, "inspect draft root", root)),
    };
    if metadata.file_type().is_symlink() {
        return Err(path_error(
            io::ErrorKind::PermissionDenied,
            "draft root must not be a symlink",
            root,
        ));
    }
    if !metadata.is_dir() {
        return Err(path_error(
            io::ErrorKind::InvalidInput,
            "draft root is not a directory",
            root,
        ));
    }

    let canonical_root = fs::canonicalize(root)
        .map_err(|error| contextual_error(error, "canonicalize draft root", root))?;
    let canonical_metadata = fs::symlink_metadata(&canonical_root).map_err(|error| {
        contextual_error(error, "inspect canonical draft root", &canonical_root)
    })?;
    if canonical_metadata.file_type().is_symlink() || !canonical_metadata.is_dir() {
        return Err(path_error(
            io::ErrorKind::PermissionDenied,
            "canonical draft root is not a real directory",
            &canonical_root,
        ));
    }
    Ok(Some(canonical_root))
}

fn verify_root_mapping(root: &Path, expected: &Path) -> io::Result<()> {
    match canonical_existing_root(root)? {
        Some(current) if current == expected => Ok(()),
        Some(_) => Err(path_error(
            io::ErrorKind::PermissionDenied,
            "draft root changed during the operation",
            root,
        )),
        None => Err(path_error(
            io::ErrorKind::NotFound,
            "draft root disappeared during the operation",
            root,
        )),
    }
}

fn canonical_draft_dir(canonical_root: &Path, id: &str) -> io::Result<Option<PathBuf>> {
    let dir = canonical_root.join(id);
    let metadata = match fs::symlink_metadata(&dir) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(contextual_error(error, "inspect draft directory", &dir)),
    };
    if metadata.file_type().is_symlink() {
        return Err(path_error(
            io::ErrorKind::PermissionDenied,
            "draft directory must not be a symlink",
            &dir,
        ));
    }
    if !metadata.is_dir() {
        return Err(path_error(
            io::ErrorKind::InvalidInput,
            "draft path is not a directory",
            &dir,
        ));
    }

    let canonical_dir = fs::canonicalize(&dir)
        .map_err(|error| contextual_error(error, "canonicalize draft directory", &dir))?;
    if canonical_dir.parent() != Some(canonical_root) {
        return Err(path_error(
            io::ErrorKind::PermissionDenied,
            "draft directory is not directly under the draft root",
            &dir,
        ));
    }
    let verified_metadata = fs::symlink_metadata(&canonical_dir).map_err(|error| {
        contextual_error(error, "reinspect canonical draft directory", &canonical_dir)
    })?;
    if verified_metadata.file_type().is_symlink() || !verified_metadata.is_dir() {
        return Err(path_error(
            io::ErrorKind::PermissionDenied,
            "canonical draft directory is not a real directory",
            &canonical_dir,
        ));
    }
    Ok(Some(canonical_dir))
}

fn safe_project_entry(canonical_dir: &Path) -> io::Result<bool> {
    let project = canonical_dir.join(PROJECT_FILE);
    match fs::symlink_metadata(&project) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(path_error(
            io::ErrorKind::PermissionDenied,
            "draft project file must not be a symlink",
            &project,
        )),
        Ok(metadata) if !metadata.is_file() => Err(path_error(
            io::ErrorKind::InvalidInput,
            "draft project path is not a regular file",
            &project,
        )),
        Ok(_) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(contextual_error(
            error,
            "inspect draft project file",
            &project,
        )),
    }
}

fn ensure_safe_meta_entry(canonical_dir: &Path) -> io::Result<()> {
    let meta = canonical_dir.join(META_FILE);
    match fs::symlink_metadata(&meta) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(path_error(
            io::ErrorKind::PermissionDenied,
            "draft metadata file must not be a symlink",
            &meta,
        )),
        Ok(metadata) if !metadata.is_file() => Err(path_error(
            io::ErrorKind::InvalidInput,
            "draft metadata path is not a regular file",
            &meta,
        )),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(contextual_error(
            error,
            "inspect draft metadata file",
            &meta,
        )),
    }
}

fn cleanup_created_slot(slot: &DraftSlot) -> io::Result<bool> {
    if slot.canonical_dir.parent() != Some(slot.canonical_root.as_path()) {
        return Err(path_error(
            io::ErrorKind::PermissionDenied,
            "refusing cleanup outside the canonical draft root",
            &slot.canonical_dir,
        ));
    }
    let metadata = match fs::symlink_metadata(&slot.canonical_dir) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(contextual_error(
                error,
                "inspect failed import draft",
                &slot.canonical_dir,
            ));
        }
    };
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(path_error(
            io::ErrorKind::PermissionDenied,
            "failed import draft is no longer a real directory",
            &slot.canonical_dir,
        ));
    }
    safe_project_entry(&slot.canonical_dir)?;
    tombstone_and_remove(&slot.canonical_root, &slot.canonical_dir)
}

fn tombstone_and_remove(canonical_root: &Path, canonical_dir: &Path) -> io::Result<bool> {
    if canonical_dir.parent() != Some(canonical_root) {
        return Err(path_error(
            io::ErrorKind::PermissionDenied,
            "refusing to tombstone a directory outside the draft root",
            canonical_dir,
        ));
    }
    let metadata = match fs::symlink_metadata(canonical_dir) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(contextual_error(
                error,
                "inspect draft before tombstoning",
                canonical_dir,
            ));
        }
    };
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(path_error(
            io::ErrorKind::PermissionDenied,
            "draft changed before tombstoning",
            canonical_dir,
        ));
    }

    let tombstone = {
        let mut selected = None;
        for _ in 0..UNIQUE_PATH_ATTEMPTS {
            let candidate = canonical_root.join(format!(
                "{TOMBSTONE_PREFIX}{:x}-{}",
                std::process::id(),
                new_id()
            ));
            match fs::symlink_metadata(&candidate) {
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    selected = Some(candidate);
                    break;
                }
                Ok(_) => continue,
                Err(error) => {
                    return Err(contextual_error(
                        error,
                        "inspect prospective draft tombstone",
                        &candidate,
                    ));
                }
            }
        }
        selected.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::AlreadyExists,
                "could not allocate a unique draft tombstone",
            )
        })?
    };

    match fs::rename(canonical_dir, &tombstone) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return match fs::symlink_metadata(canonical_dir) {
                Err(check_error) if check_error.kind() == io::ErrorKind::NotFound => Ok(false),
                _ => Err(contextual_error(
                    error,
                    "rename draft to tombstone",
                    canonical_dir,
                )),
            };
        }
        Err(error) => {
            return Err(contextual_error(
                error,
                "rename draft to tombstone",
                canonical_dir,
            ));
        }
    }

    if let Err(error) = cleanup_tombstone(canonical_root, &tombstone) {
        warn!(
            path = %bounded_path(&tombstone),
            error = %error,
            "draft deletion committed but tombstone cleanup failed"
        );
    } else if let Err(error) = sync_directory(canonical_root) {
        warn!(
            path = %bounded_path(canonical_root),
            error = %error,
            "draft deletion committed but root directory sync failed"
        );
    }
    Ok(true)
}

fn cleanup_tombstone(canonical_root: &Path, tombstone: &Path) -> io::Result<()> {
    if tombstone.parent() != Some(canonical_root) {
        return Err(path_error(
            io::ErrorKind::PermissionDenied,
            "refusing to clean a tombstone outside the draft root",
            tombstone,
        ));
    }
    let metadata = fs::symlink_metadata(tombstone)
        .map_err(|error| contextual_error(error, "inspect draft tombstone", tombstone))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(path_error(
            io::ErrorKind::PermissionDenied,
            "draft tombstone is not a real directory",
            tombstone,
        ));
    }
    // Keep the exact generated path: `remove_dir_all` does not follow
    // symlinks, while canonicalizing again would reintroduce a race that
    // could redirect cleanup to a different directory.
    fs::remove_dir_all(tombstone)
        .map_err(|error| contextual_error(error, "recursively remove draft tombstone", tombstone))
}

fn open_unique_file(dir: &Path, purpose: &str) -> io::Result<(PathBuf, File)> {
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

fn read_name_checked(dir: &Path) -> Option<String> {
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

fn bounded_project_name(name: &str) -> String {
    if name.len() <= MAX_PROJECT_NAME_BYTES {
        return name.to_owned();
    }
    let mut end = MAX_PROJECT_NAME_BYTES;
    while name.get(..end).is_none() {
        end = end.saturating_sub(1);
    }
    name.get(..end).unwrap_or_default().to_owned()
}

fn invalid_project_path(project: &Path, reason: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!(
            "invalid draft project path '{}': {reason}",
            bounded_path(project)
        ),
    )
}

fn path_error(kind: io::ErrorKind, action: &str, path: &Path) -> io::Error {
    io::Error::new(kind, format!("{action}: '{}'", bounded_path(path)))
}

fn contextual_error(error: io::Error, action: &str, path: &Path) -> io::Error {
    io::Error::new(
        error.kind(),
        format!("{action} '{}': {error}", bounded_path(path)),
    )
}

fn bounded_path(path: &Path) -> String {
    let lossy = path.as_os_str().to_string_lossy();
    let mut characters = lossy.chars();
    let mut bounded: String = characters.by_ref().take(MAX_ERROR_PATH_CHARS).collect();
    if characters.next().is_some() {
        bounded.push('…');
    }
    bounded
}

#[cfg(unix)]
fn sync_directory(dir: &Path) -> io::Result<()> {
    File::open(dir)?.sync_all()
}

#[cfg(not(unix))]
fn sync_directory(_dir: &Path) -> io::Result<()> {
    Ok(())
}

/// A coarse "5 minutes ago" label for `modified`, dependency-free, for the
/// gallery card subtitle. Future or unreadable times read as "just now".
pub fn relative_time(modified: SystemTime) -> String {
    let secs = SystemTime::now()
        .duration_since(modified)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    match secs {
        0..=59 => "just now".to_owned(),
        60..=3599 => plural(secs / 60, "minute"),
        3600..=86_399 => plural(secs / 3600, "hour"),
        _ => plural(secs / 86_400, "day"),
    }
}

fn plural(n: u64, unit: &str) -> String {
    if n == 1 {
        format!("1 {unit} ago")
    } else {
        format!("{n} {unit}s ago")
    }
}

/// A unique-per-process draft id: millis since the epoch plus a monotonic
/// counter, so rapid successive creates never share a directory name.
fn new_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{millis:x}-{seq:x}")
}

#[cfg(test)]
mod tests;
