//! On-disk project files (`.cutlass` JSON).

use std::fs::{File, OpenOptions, Permissions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::Deserialize;

use crate::error::ModelError;
use crate::project::Project;
use crate::schema::{PROJECT_SCHEMA_VERSION, ProjectSchema};

/// Recommended extension for Cutlass project files.
pub const PROJECT_FILE_EXTENSION: &str = "cutlass";

/// Numeric version from [`ProjectSchema::current`] for simple callers.
pub const PROJECT_FILE_VERSION: u32 = PROJECT_SCHEMA_VERSION;

const UNIQUE_PATH_ATTEMPTS: usize = 128;
static UNIQUE_PATH_COUNTER: AtomicU64 = AtomicU64::new(0);

impl Project {
    /// Serialize this project to a `.cutlass` JSON file.
    ///
    /// The document is stamped with this build's schema version regardless
    /// of what was loaded: the writer defines the format. (A project opened
    /// from a v1 file may now hold v2-only data like keyframes; persisting
    /// it as "v1" would lie to older readers.)
    pub fn save_to_file(&self, path: &Path) -> io::Result<()> {
        let mut doc = self.clone();
        doc.schema.version = PROJECT_SCHEMA_VERSION;
        doc.schema.kind = crate::schema::PROJECT_SCHEMA_KIND.into();
        let mut json = serde_json::to_string_pretty(&doc).map_err(io::Error::other)?;
        json.push('\n');

        // Keep serialization entirely ahead of filesystem inspection so a
        // serialization failure cannot disturb an existing project.
        persist_serialized(path, json.as_bytes(), &StdPersistenceFs)
    }

    /// Deserialize a project from a `.cutlass` JSON file.
    ///
    /// The document's schema version is read and validated *before* the
    /// typed parse, and [`migrate_document`] rewrites older shapes up to the
    /// current one — so the strict parse below only ever sees the current
    /// format. Files newer than this build are refused, never half-parsed.
    pub fn load_from_file(path: &Path) -> Result<Project, ModelError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| ModelError::InvalidProjectFile(e.to_string()))?;
        let mut doc: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| ModelError::InvalidProjectFile(e.to_string()))?;

        unwrap_legacy_envelope(&mut doc);
        let mut schema = read_schema(&doc)?;
        normalize_legacy_schema(&mut schema);
        validate_schema(&schema)?;
        migrate_document(&mut doc, schema.version);

        let mut project: Project = serde_json::from_value(doc)
            .map_err(|e| ModelError::InvalidProjectFile(e.to_string()))?;
        // Keep the file's original (normalized) schema as provenance; the
        // writer re-stamps the current version on save.
        project.schema = schema;
        // Files written before the main-track flag (or edited externally)
        // re-derive the lane-zone / main-track invariants on entry.
        project.timeline_mut().normalize_lanes();
        Ok(project)
    }
}

trait PersistenceFs {
    fn create_new(&self, path: &Path) -> io::Result<File>;
    fn write_all(&self, file: &mut File, contents: &[u8]) -> io::Result<()>;
    fn flush(&self, file: &mut File) -> io::Result<()>;
    fn set_permissions(&self, file: &File, permissions: Permissions) -> io::Result<()>;
    fn sync_all(&self, file: &File) -> io::Result<()>;
    fn symlink_metadata(&self, path: &Path) -> io::Result<std::fs::Metadata>;
    fn rename(&self, from: &Path, to: &Path) -> io::Result<()>;
    fn create_dir(&self, path: &Path) -> io::Result<()>;
    fn remove_file(&self, path: &Path) -> io::Result<()>;
    fn remove_dir(&self, path: &Path) -> io::Result<()>;
}

struct StdPersistenceFs;

impl PersistenceFs for StdPersistenceFs {
    fn create_new(&self, path: &Path) -> io::Result<File> {
        open_new_temp(path)
    }

    fn write_all(&self, file: &mut File, contents: &[u8]) -> io::Result<()> {
        file.write_all(contents)
    }

    fn flush(&self, file: &mut File) -> io::Result<()> {
        file.flush()
    }

    fn set_permissions(&self, file: &File, permissions: Permissions) -> io::Result<()> {
        file.set_permissions(permissions)
    }

    fn sync_all(&self, file: &File) -> io::Result<()> {
        file.sync_all()
    }

    fn symlink_metadata(&self, path: &Path) -> io::Result<std::fs::Metadata> {
        std::fs::symlink_metadata(path)
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        std::fs::rename(from, to)
    }

    fn create_dir(&self, path: &Path) -> io::Result<()> {
        create_private_dir(path)
    }

    fn remove_file(&self, path: &Path) -> io::Result<()> {
        std::fs::remove_file(path)
    }

    fn remove_dir(&self, path: &Path) -> io::Result<()> {
        std::fs::remove_dir(path)
    }
}

fn persist_serialized(
    destination: &Path,
    contents: &[u8],
    fs: &impl PersistenceFs,
) -> io::Result<()> {
    let permissions = destination_permissions(destination, fs)?;
    let temporary = write_synced_temp(destination, contents, permissions, fs)?;
    install_temp_with_ops(destination, &temporary, fs)
}

fn destination_permissions(
    destination: &Path,
    fs: &impl PersistenceFs,
) -> io::Result<Option<Permissions>> {
    match fs.symlink_metadata(destination) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "refusing to save through a symbolic-link project destination",
        )),
        Ok(metadata) if !metadata.file_type().is_file() => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "project destination is not a regular file",
        )),
        Ok(metadata) => Ok(Some(metadata.permissions())),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(contextual_error(
            "could not inspect the project destination",
            error,
        )),
    }
}

fn write_synced_temp(
    destination: &Path,
    contents: &[u8],
    permissions: Option<Permissions>,
    fs: &impl PersistenceFs,
) -> io::Result<PathBuf> {
    let (temporary, mut file) = create_unique_temp(destination, fs)?;
    let result = (|| {
        fs.write_all(&mut file, contents)
            .map_err(|error| contextual_error("could not write the temporary project", error))?;
        fs.flush(&mut file)
            .map_err(|error| contextual_error("could not flush the temporary project", error))?;
        if let Some(permissions) = permissions {
            fs.set_permissions(&file, permissions).map_err(|error| {
                contextual_error("could not preserve project file permissions", error)
            })?;
        }
        fs.sync_all(&file)
            .map_err(|error| contextual_error("could not sync the temporary project", error))
    })();
    drop(file);

    match result {
        Ok(()) => Ok(temporary),
        Err(error) => Err(cleanup_temp_after_error(fs, &temporary, error)),
    }
}

fn create_unique_temp(destination: &Path, fs: &impl PersistenceFs) -> io::Result<(PathBuf, File)> {
    for _ in 0..UNIQUE_PATH_ATTEMPTS {
        let candidate = unique_sibling_path(destination, "tmp")?;
        match fs.create_new(&candidate) {
            Ok(file) => return Ok((candidate, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(contextual_error(
                    "could not create a temporary project file",
                    error,
                ));
            }
        }
    }

    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not allocate a unique temporary project file",
    ))
}

fn open_new_temp(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)
}

#[derive(Debug)]
struct BackupPaths {
    container: PathBuf,
    project: PathBuf,
}

fn create_unique_backup(destination: &Path, fs: &impl PersistenceFs) -> io::Result<BackupPaths> {
    for _ in 0..UNIQUE_PATH_ATTEMPTS {
        let container = unique_sibling_path(destination, "backup")?;
        match fs.create_dir(&container) {
            Ok(()) => {
                let project = container.join("original");
                return Ok(BackupPaths { container, project });
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(contextual_error(
                    "could not create a rollback backup for the project",
                    error,
                ));
            }
        }
    }

    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not allocate a unique rollback backup for the project",
    ))
}

fn create_private_dir(path: &Path) -> io::Result<()> {
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(path)
}

fn install_temp_with_ops(
    destination: &Path,
    temporary: &Path,
    fs: &impl PersistenceFs,
) -> io::Result<()> {
    // Recheck after the potentially long write/sync phase. This is not a
    // platform-specific openat lock, but it fails closed for every symlink
    // destination observed by the portable persistence path.
    if let Err(error) = destination_permissions(destination, fs) {
        return Err(cleanup_temp_after_error(fs, temporary, error));
    }

    let atomic_error = match fs.rename(temporary, destination) {
        Ok(()) => return Ok(()),
        Err(error) => error,
    };

    match destination_permissions(destination, fs) {
        Ok(Some(_)) => {}
        Ok(None) => {
            let error =
                contextual_error("could not atomically install the new project", atomic_error);
            return Err(cleanup_temp_after_error(fs, temporary, error));
        }
        Err(error) => return Err(cleanup_temp_after_error(fs, temporary, error)),
    }

    // Some platforms cannot rename over an existing file. Reserve a private,
    // unique sibling directory first, then move the old file into it. The
    // original is always either at the destination or in this backup; it is
    // never deleted before the new file reaches the commit point.
    let backup = match create_unique_backup(destination, fs) {
        Ok(backup) => backup,
        Err(error) => {
            let error = io::Error::new(
                error.kind(),
                format!(
                    "atomic project replacement failed ({atomic_error}); \
                     rollback backup creation also failed: {error}"
                ),
            );
            return Err(cleanup_temp_after_error(fs, temporary, error));
        }
    };

    if let Err(backup_error) = fs.rename(destination, &backup.project) {
        let error = io::Error::new(
            backup_error.kind(),
            format!(
                "atomic project replacement failed ({atomic_error}); moving the existing project \
                 to a rollback backup also failed: {backup_error}; the existing project was left \
                 in place"
            ),
        );
        let error = cleanup_empty_backup_after_error(fs, &backup.container, error);
        return Err(cleanup_temp_after_error(fs, temporary, error));
    }

    match fs.rename(temporary, destination) {
        Ok(()) => {
            // The new destination is committed. Cleanup is deliberately
            // best-effort: a stale backup must not turn a successful save into
            // an error after callers can already observe the new project.
            let _ = fs.remove_file(&backup.project);
            let _ = fs.remove_dir(&backup.container);
            Ok(())
        }
        Err(install_error) => match fs.rename(&backup.project, destination) {
            Ok(()) => {
                let error = io::Error::new(
                    install_error.kind(),
                    format!(
                        "could not install the new project after creating a rollback backup: \
                         {install_error}; the original project was restored"
                    ),
                );
                let error = cleanup_empty_backup_after_error(fs, &backup.container, error);
                Err(cleanup_temp_after_error(fs, temporary, error))
            }
            Err(rollback_error) => {
                let error = io::Error::new(
                    install_error.kind(),
                    format!(
                        "could not install the new project after creating a rollback backup: \
                         {install_error}; restoring the original project failed: {rollback_error}; \
                         the original project was retained in a sibling backup"
                    ),
                );
                Err(cleanup_temp_after_error(fs, temporary, error))
            }
        },
    }
}

fn unique_sibling_path(destination: &Path, role: &str) -> io::Result<PathBuf> {
    let parent = destination_parent(destination)?;
    if destination.file_name().is_none() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "project destination has no file name",
        ));
    }

    let nonce = UNIQUE_PATH_COUNTER.fetch_add(1, Ordering::Relaxed);
    Ok(parent.join(format!(
        ".cutlass-project-{role}-{}-{nonce}",
        std::process::id()
    )))
}

fn destination_parent(destination: &Path) -> io::Result<&Path> {
    let parent = destination.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "project destination has no parent directory",
        )
    })?;
    if parent.as_os_str().is_empty() {
        Ok(Path::new("."))
    } else {
        Ok(parent)
    }
}

fn cleanup_temp_after_error(
    fs: &impl PersistenceFs,
    temporary: &Path,
    primary_error: io::Error,
) -> io::Error {
    match fs.remove_file(temporary) {
        Ok(()) => primary_error,
        Err(error) if error.kind() == io::ErrorKind::NotFound => primary_error,
        Err(error) => io::Error::new(
            primary_error.kind(),
            format!(
                "{primary_error}; temporary project cleanup also failed: {error}; \
                 a temporary sibling may remain"
            ),
        ),
    }
}

fn cleanup_empty_backup_after_error(
    fs: &impl PersistenceFs,
    backup_container: &Path,
    primary_error: io::Error,
) -> io::Error {
    match fs.remove_dir(backup_container) {
        Ok(()) => primary_error,
        Err(error) if error.kind() == io::ErrorKind::NotFound => primary_error,
        Err(error) => io::Error::new(
            primary_error.kind(),
            format!(
                "{primary_error}; empty rollback-backup cleanup also failed: {error}; \
                 an empty sibling directory may remain"
            ),
        ),
    }
}

fn contextual_error(context: &'static str, error: io::Error) -> io::Error {
    io::Error::new(error.kind(), format!("{context}: {error}"))
}

/// Unwrap the legacy `{ "version", "project" }` envelope (early v1 saves)
/// to the bare project document, pushing the envelope version into the
/// inner schema when the project carries none (or a placeholder version 0).
fn unwrap_legacy_envelope(doc: &mut serde_json::Value) {
    let Some(obj) = doc.as_object_mut() else {
        return;
    };
    // A real project document has a schema (or its legacy alias); only the
    // envelope nests the project under "project". Unknown root fields named
    // "project"/"version" on a current file must not trip this.
    let looks_like_envelope = obj.contains_key("project")
        && obj.contains_key("version")
        && !obj.contains_key("schema")
        && !obj.contains_key("schema_version");
    if !looks_like_envelope {
        return;
    }
    let version = obj.get("version").and_then(serde_json::Value::as_u64);
    let Some(mut project) = obj.remove("project") else {
        return;
    };
    if let (Some(version), Some(inner)) = (version, project.as_object_mut()) {
        let placeholder = inner
            .get("schema")
            .and_then(|s| s.get("version"))
            .and_then(serde_json::Value::as_u64)
            == Some(0);
        let missing = !inner.contains_key("schema") && !inner.contains_key("schema_version");
        if missing || placeholder {
            inner.insert("schema".into(), serde_json::json!(version));
        }
    }
    *doc = project;
}

/// Read the document's schema (object or legacy bare-integer form) without
/// deserializing the whole project. Shared with the template loader, which
/// wraps a project document and versions in lockstep with it.
pub(crate) fn read_schema(doc: &serde_json::Value) -> Result<ProjectSchema, ModelError> {
    #[derive(Deserialize)]
    struct Holder(#[serde(deserialize_with = "crate::schema::deserialize")] ProjectSchema);

    let raw = doc
        .get("schema")
        .or_else(|| doc.get("schema_version"))
        .ok_or_else(|| ModelError::InvalidProjectFile("missing schema".into()))?;
    serde_json::from_value::<Holder>(raw.clone())
        .map(|holder| holder.0)
        .map_err(|e| ModelError::InvalidProjectFile(format!("invalid schema: {e}")))
}

/// Rewrite a raw project document from `from`'s shape to the current
/// schema's, one version step at a time, before the typed parse.
///
/// This is the format versioning policy (v1 roadmap M0):
///
/// - **Additive optional fields don't bump the version.** New fields ship
///   with `#[serde(default)]` (+ skip-if-default on save); readers of the
///   same version ignore fields they don't know — and drop them on resave.
///   That tolerance is the compatibility contract *within* a version.
/// - **Shape changes bump [`PROJECT_SCHEMA_VERSION`]** and add a
///   `migrate_vN_to_vN1` step here rewriting the previous shape into the
///   new one.
/// - **Newer documents are refused** before this runs
///   ([`ModelError::UnsupportedProjectSchema`] from `validate_schema`) —
///   guessing at a future format risks silent data loss.
///
/// The template loader reuses this on its embedded project document
/// (`.cutlasst` files version in lockstep with [`PROJECT_SCHEMA_VERSION`]).
pub(crate) fn migrate_document(doc: &mut serde_json::Value, from: u32) {
    for step in from..PROJECT_SCHEMA_VERSION {
        match step {
            1 => migrate_v1_to_v2(doc),
            2 => migrate_v2_to_v3(doc),
            // `validate_schema` bounds `from` to supported versions, so a
            // missing arm is a bug: a version bump landed without its step.
            _ => unreachable!("no migration step for schema v{step} -> v{}", step + 1),
        }
    }
}

/// v1 → v2 (M2 animatable params): transform properties gained the
/// `{"kf": [...]}` curve shape, but constants kept the bare v1 value shape,
/// so every v1 document is already a valid v2 document — nothing to
/// rewrite. The step exists to anchor the chain.
fn migrate_v1_to_v2(_doc: &mut serde_json::Value) {}

/// v2 → v3 (Mirror band): legacy Mirror was a half-plane that ignored
/// `size`. CapCut-parity Mirror is a parallel band of thickness `size[0]`;
/// the historical default `[1,1]` (often omitted on the wire via
/// skip-if-default) now covers the full layer width and looks like a
/// no-op. Rewrite omitted / constant `size[0] == 1` (and matching
/// keyframe values) to `0.5` so old projects keep a visible band.
fn migrate_v2_to_v3(doc: &mut serde_json::Value) {
    let Some(tracks) = doc
        .pointer_mut("/timeline/tracks")
        .and_then(|v| v.as_array_mut())
    else {
        return;
    };
    for track_entry in tracks {
        let Some(track) = track_entry.get_mut(1) else {
            continue;
        };
        let Some(clips) = track.get_mut("clips").and_then(|v| v.as_array_mut()) else {
            continue;
        };
        for clip_entry in clips {
            let Some(clip) = clip_entry.get_mut(1) else {
                continue;
            };
            let Some(mask) = clip.get_mut("mask").and_then(|v| v.as_object_mut()) else {
                continue;
            };
            if mask.get("kind").and_then(|k| k.as_str()) != Some("mirror") {
                continue;
            }
            migrate_mirror_band_size(mask);
        }
    }
}

fn migrate_mirror_band_size(mask: &mut serde_json::Map<String, serde_json::Value>) {
    match mask.get_mut("size") {
        None => {
            mask.insert("size".into(), serde_json::json!([0.5, 1.0]));
        }
        Some(serde_json::Value::Array(arr)) if arr.len() == 2 => {
            if size_axis_is_one(&arr[0]) {
                arr[0] = serde_json::json!(0.5);
            }
        }
        Some(serde_json::Value::Object(obj)) => {
            if let Some(serde_json::Value::Array(kfs)) = obj.get_mut("kf") {
                for kf in kfs {
                    let Some(v) = kf.get_mut("v").and_then(|v| v.as_array_mut()) else {
                        continue;
                    };
                    if v.len() == 2 && size_axis_is_one(&v[0]) {
                        v[0] = serde_json::json!(0.5);
                    }
                }
            }
        }
        Some(_) => {}
    }
}

fn size_axis_is_one(v: &serde_json::Value) -> bool {
    match v {
        serde_json::Value::Number(n) => n.as_f64().is_some_and(|x| (x - 1.0).abs() < 1e-6),
        _ => false,
    }
}

fn normalize_legacy_schema(schema: &mut ProjectSchema) {
    if schema.kind.is_empty() {
        schema.kind = crate::schema::PROJECT_SCHEMA_KIND.into();
    }
}

fn validate_schema(found: &ProjectSchema) -> Result<(), ModelError> {
    let expected = ProjectSchema::current();
    if !found.is_supported() {
        return Err(ModelError::UnsupportedProjectSchema {
            found: found.clone(),
            expected,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
