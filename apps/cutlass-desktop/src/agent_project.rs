//! Strict project tools for the desktop agent.
//!
//! Draft paths remain an internal engine/storage detail. The one external-path
//! tool canonicalizes an existing regular media file, never returns that path
//! to the model, and dispatches only through the acknowledged preview-worker
//! RPC.

use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use cutlass_ai::{HostToolSpec, ToolOutput, ToolTier};
use serde_json::{Map, Value, json};
use tracing::error;

use crate::preview_worker::{ImportMediaRpcResult, OpenProjectRpcResult, WorkerHandle};

const PROJECT_LIST_DRAFTS: &str = "project_list_drafts";
const PROJECT_SAVE: &str = "project_save";
const PROJECT_OPEN: &str = "project_open";
pub(crate) const PROJECT_IMPORT_MEDIA: &str = "project_import_media";
const TOOL_NAMES: [&str; 4] = [
    PROJECT_LIST_DRAFTS,
    PROJECT_SAVE,
    PROJECT_OPEN,
    PROJECT_IMPORT_MEDIA,
];

const DEFAULT_DRAFT_LIMIT: usize = 50;
const MAX_DRAFT_LIMIT: usize = 100;
const MIN_DRAFT_ID_CHARS: usize = 3;
/// A canonical id has at most 32 time hex digits, one dash, and 16 sequence
/// hex digits. Canonical ids are ASCII, so this is also their byte bound.
pub(crate) const MAX_DRAFT_ID_CHARS: usize = 32 + 1 + 16;
const MAX_DISPLAY_NAME_CHARS: usize = 128;
/// Deliberately below platform path maxima so approval copy and diagnostics
/// stay bounded even after canonicalization.
pub(crate) const MAX_IMPORT_PATH_CHARS: usize = 1_024;
const MAX_OUTPUT_BYTES: usize = 128 * 1024;

/// Opaque proof that one import path passed the full regular-file preflight.
///
/// The private identity binds Ask-mode approval to the exact file inspected,
/// while callers may expose only the canonical path in approval copy.
pub(crate) struct ValidatedImportMedia {
    canonical_path: PathBuf,
    identity: ImportFileIdentity,
}

impl ValidatedImportMedia {
    pub(crate) fn canonical_path(&self) -> &Path {
        &self.canonical_path
    }

    fn identifies_same_file(&self, other: &Self) -> bool {
        self.canonical_path == other.canonical_path && self.identity == other.identity
    }
}

pub(crate) fn specs() -> Vec<HostToolSpec> {
    vec![
        spec(
            PROJECT_LIST_DRAFTS,
            "List app-owned project drafts newest first using safe draft identities, display names, and modification times.",
            list_schema(),
            ToolTier::ReadOnly,
        ),
        spec(
            PROJECT_SAVE,
            "Save the current app-owned project and return its safe draft identity.",
            empty_object_schema(),
            ToolTier::Workspace,
        ),
        spec(
            PROJECT_OPEN,
            "Open an existing app-owned project by canonical draft identity. This replaces the current editor session.",
            open_schema(),
            ToolTier::System,
        ),
        spec(
            PROJECT_IMPORT_MEDIA,
            "Import one existing external media file by absolute path. Cutlass references the source file rather than copying it.",
            import_media_schema(),
            ToolTier::System,
        ),
    ]
}

/// Whether a registered project tool can change live project/session state.
///
/// The read-only allowlist is explicit; unknown future project tools fail
/// closed as mutations until they are deliberately classified.
pub(crate) fn mutates_live_project(name: &str) -> bool {
    cutlass_ai::namespace(name) == "project" && name != PROJECT_LIST_DRAFTS
}

/// Strictly validate a project-tool request before any System approval UI is
/// published. Path-bearing requests perform their read-only filesystem
/// preflight here; dispatch repeats it after approval.
pub(crate) fn validate_request(name: &str, arguments: &Value) -> Result<(), String> {
    match parse_request(name, arguments)? {
        Request::Open { draft_id } => resolve_open_draft(&draft_id).map(|_| ()),
        Request::ImportMedia { path } => preflight_import_media_path(&path).map(|_| ()),
        Request::ListDrafts { .. } | Request::Save => Ok(()),
    }
}

/// Return an opaque token for the exact regular file approved for import.
///
/// This intentionally performs the same full preflight as
/// [`validate_request`]. The host retains this value across an Ask-mode
/// approval and requires the post-approval preflight to match its path and
/// platform file identity.
pub(crate) fn validated_import_media(arguments: &Value) -> Result<ValidatedImportMedia, String> {
    match parse_request(PROJECT_IMPORT_MEDIA, arguments)? {
        Request::ImportMedia { path } => preflight_import_media_path(&path),
        _ => Err("project_import_media request could not be validated".into()),
    }
}

pub(crate) fn call(
    worker: Option<&WorkerHandle>,
    name: &str,
    arguments: &Value,
    cancel: &AtomicBool,
) -> Result<ToolOutput, String> {
    call_with_approved_import(worker, name, arguments, None, cancel)
}

pub(crate) fn call_with_approved_import(
    worker: Option<&WorkerHandle>,
    name: &str,
    arguments: &Value,
    approved_import: Option<&ValidatedImportMedia>,
    cancel: &AtomicBool,
) -> Result<ToolOutput, String> {
    match parse_request(name, arguments)? {
        Request::ListDrafts { limit } => {
            if cancel.load(Ordering::Acquire) {
                return Err("cancelled before project_list_drafts could run".into());
            }
            let drafts = crate::drafts::list_checked().map_err(|internal| {
                error!(error = %internal, "agent draft listing failed");
                "project_list_drafts failed: the draft store could not be read".to_string()
            })?;
            if cancel.load(Ordering::Acquire) {
                return Err("cancelled while project_list_drafts was running".into());
            }
            list_drafts_output(drafts, limit)
        }
        Request::Save => save_project(worker, cancel),
        Request::Open { draft_id } => open_project(worker, &draft_id, cancel),
        Request::ImportMedia { path } => import_media(worker, &path, approved_import, cancel),
    }
}

fn import_media(
    worker: Option<&WorkerHandle>,
    requested_path: &str,
    approved_import: Option<&ValidatedImportMedia>,
    cancel: &AtomicBool,
) -> Result<ToolOutput, String> {
    // Parsing is memory-only. Cancellation must win before the first
    // filesystem operation.
    if cancel.load(Ordering::Acquire) {
        return Err("project_import_media was cancelled before it started".into());
    }

    // Existing files are canonicalized and that canonical absolute regular
    // file is what the worker receives. Intermediate links may therefore be
    // resolved, but the requested final component and resolved endpoint must
    // not be a symlink/reparse point. Repeating this after approval narrows,
    // but cannot eliminate, the residual path TOCTOU before the path-based RPC
    // opens the source.
    let current = preflight_import_media_path(requested_path)?;
    if let Some(approved) = approved_import {
        ensure_approved_import_unchanged(approved, &current)?;
    }
    if cancel.load(Ordering::Acquire) {
        return Err("project_import_media was cancelled before it started".into());
    }

    let worker = worker.ok_or_else(|| {
        "project_import_media failed: the editor worker is unavailable; not started".to_string()
    })?;
    let canonical = current.canonical_path().to_path_buf();
    let imported = worker
        .import_media_rpc_with_cancel(canonical, cancel)
        .map_err(|internal| {
            error!(error = %internal, "agent media import RPC failed");
            public_import_error(&internal)
        })?;
    import_media_output(&current, &imported)
}

fn ensure_approved_import_unchanged(
    approved: &ValidatedImportMedia,
    current: &ValidatedImportMedia,
) -> Result<(), String> {
    if current.identifies_same_file(approved) {
        return Ok(());
    }
    error!(
        approved = %approved.canonical_path().display(),
        resolved = %current.canonical_path().display(),
        "agent media import path or file identity changed after approval"
    );
    Err("project_import_media failed: the media file changed after approval; not started".into())
}

fn public_import_error(internal: &str) -> String {
    if internal.contains("outcome unknown")
        || internal.contains("outcome uncertain/partially committed")
        || internal.contains("import succeeded")
    {
        "project_import_media was dispatched, but the editor did not confirm its result; outcome unknown: media may have been imported".into()
    } else if internal.contains("cancelled before worker claim; not started") {
        "project_import_media was cancelled before it started".into()
    } else if internal.contains("preview worker is not running; not started")
        || internal.contains("preview worker stopped before replying; not started")
        || internal.contains("timed out before worker claim; not started")
    {
        "project_import_media failed before the editor started it".into()
    } else {
        "project_import_media failed: the media file could not be imported".into()
    }
}

fn import_media_output(
    expected: &ValidatedImportMedia,
    imported: &ImportMediaRpcResult,
) -> Result<ToolOutput, String> {
    let actual = inspect_import_file(&imported.path).map_err(|internal| {
        error!(
            acknowledged = %imported.path.display(),
            error = %internal,
            "agent media import could not verify the acknowledged source path"
        );
        "project_import_media could not verify the acknowledged source file; media may have been imported"
            .to_string()
    })?;
    if !actual.identifies_same_file(expected) {
        error!(
            approved = %expected.canonical_path().display(),
            acknowledged = %actual.canonical_path().display(),
            media_id = imported.media_id,
            "agent media import acknowledged a different source path or file identity"
        );
        return Err(
            "project_import_media acknowledged a different source file; media may have been imported"
                .into(),
        );
    }
    let actual_path = actual.canonical_path();
    if imported.media_id == 0 {
        error!("agent media import acknowledged an invalid zero media id");
        return Err(
            "project_import_media returned an invalid media identity; media may have been imported"
                .into(),
        );
    }
    let filename = actual_path
        .file_name()
        .and_then(|name| name.to_str())
        .map(bounded_safe_display_name)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| {
            error!(
                acknowledged = %actual_path.display(),
                "agent media import source has no safe display filename"
            );
            "project_import_media could not produce a safe filename; media may have been imported"
                .to_string()
        })?;

    encode_output(
        PROJECT_IMPORT_MEDIA,
        json!({
            "status": "ok",
            "media_id": imported.media_id,
            "filename": filename
        }),
    )
}

fn open_project(
    worker: Option<&WorkerHandle>,
    draft_id: &str,
    cancel: &AtomicBool,
) -> Result<ToolOutput, String> {
    if cancel.load(Ordering::Acquire) {
        return Err("project_open was cancelled before it started".into());
    }
    // This is intentionally repeated after approval and immediately before
    // queueing the RPC. The preflight result is never trusted across the
    // approval wait.
    let path = resolve_open_draft(draft_id)?;
    let worker = worker.ok_or_else(|| {
        "project_open failed: the editor worker is unavailable; not started".to_string()
    })?;
    let opened = worker
        .open_project_rpc_with_cancel(path, cancel)
        .map_err(|internal| {
            error!(error = %internal, "agent project open failed");
            public_open_error(&internal)
        })?;
    open_output(draft_id, &opened)
}

fn resolve_open_draft(draft_id: &str) -> Result<std::path::PathBuf, String> {
    crate::drafts::resolve_draft_id(draft_id).map_err(|internal| {
        error!(error = %internal, "agent project-open draft resolution failed");
        public_draft_resolution_error(internal.kind())
    })
}

fn public_draft_resolution_error(kind: io::ErrorKind) -> String {
    match kind {
        io::ErrorKind::InvalidInput => {
            "project_open argument 'draft_id' must be a canonical app-owned draft ID".into()
        }
        io::ErrorKind::NotFound => {
            "project_open failed: the requested app-owned draft does not exist".into()
        }
        io::ErrorKind::PermissionDenied => {
            "project_open failed: the requested app-owned draft could not be safely resolved".into()
        }
        _ => "project_open failed: the draft store could not be read".into(),
    }
}

fn public_open_error(internal: &str) -> String {
    if internal.contains("outcome unknown")
        || internal.contains("outcome uncertain/partially committed")
    {
        "project_open was dispatched, but the editor did not confirm its result; outcome unknown: the project may have opened".into()
    } else if internal.contains("cancelled before worker claim; not started") {
        "project_open was cancelled before it started".into()
    } else if internal.contains("not started") {
        "project_open failed before the editor started it".into()
    } else {
        "project_open failed: the requested draft could not be opened".into()
    }
}

fn open_output(
    requested_draft_id: &str,
    opened: &OpenProjectRpcResult,
) -> Result<ToolOutput, String> {
    let actual_draft_id =
        crate::drafts::draft_id_from_project(&opened.path).map_err(|internal| {
            error!(
                project = %opened.path.display(),
                error = %internal,
                "agent project open returned a non-draft path"
            );
            "project_open opened a project, but could not confirm its app-owned draft identity; the current session was replaced".to_string()
        })?;
    if actual_draft_id != requested_draft_id {
        error!(
            project = %opened.path.display(),
            requested_draft_id,
            actual_draft_id,
            "agent project open acknowledged a different draft identity"
        );
        return Err(
            "project_open opened a project, but its draft identity did not match the request; the current session was replaced".into(),
        );
    }

    encode_output(
        PROJECT_OPEN,
        json!({
            "status": "ok",
            "draft_id": actual_draft_id,
            "project_name": bounded_display_name(&opened.project_name),
            "missing_media_count": opened.missing_media_count
        }),
    )
}

fn save_project(worker: Option<&WorkerHandle>, cancel: &AtomicBool) -> Result<ToolOutput, String> {
    let worker = worker
        .ok_or_else(|| "project_save failed: the editor worker is unavailable".to_string())?;
    let saved = worker
        .save_project_rpc_with_cancel(None, cancel)
        .map_err(|internal| {
            error!(error = %internal, "agent project save failed");
            public_save_error(&internal)
        })?;
    save_output(&saved.path, saved.dirty)
}

fn public_save_error(internal: &str) -> String {
    if internal.contains("cancelled before worker claim; not started") {
        "project_save was cancelled before it started".into()
    } else if internal.contains("no current project path is bound") {
        "project_save failed: no app-owned draft is bound".into()
    } else if internal.contains("outcome unknown") {
        "project_save was dispatched, but the editor did not confirm its result; outcome unknown"
            .into()
    } else if internal.contains("not started") {
        "project_save failed before the editor started it".into()
    } else {
        "project_save failed: the current draft could not be saved".into()
    }
}

fn save_output(path: &Path, dirty: bool) -> Result<ToolOutput, String> {
    if dirty {
        error!(
            project = %path.display(),
            "agent project save acknowledged an unexpectedly dirty project"
        );
        return Err("project_save could not confirm a clean saved state".into());
    }
    let draft_id = crate::drafts::draft_id_from_project(path).map_err(|internal| {
        error!(
            project = %path.display(),
            error = %internal,
            "agent project save returned a non-draft path"
        );
        "project_save could not confirm an app-owned draft identity after saving".to_string()
    })?;
    encode_output(
        PROJECT_SAVE,
        json!({
            "status": "ok",
            "draft_id": draft_id,
            "dirty": false
        }),
    )
}

fn list_drafts_output(
    mut drafts: Vec<crate::drafts::DraftSummary>,
    limit: usize,
) -> Result<ToolOutput, String> {
    drafts.sort_by(|left, right| {
        right
            .modified
            .cmp(&left.modified)
            .then_with(|| left.project.cmp(&right.project))
    });
    let total = drafts.len();
    let mut rows = Vec::with_capacity(total.min(limit));
    for draft in drafts.into_iter().take(limit) {
        let draft_id =
            crate::drafts::draft_id_from_project(&draft.project).map_err(|internal| {
                error!(
                    project = %draft.project.display(),
                    error = %internal,
                    "draft listing returned a non-draft path"
                );
                "project_list_drafts failed: the draft store returned an invalid identity"
                    .to_string()
            })?;
        rows.push(json!({
            "draft_id": draft_id,
            "name": bounded_display_name(&draft.name),
            "modified_unix_ms": modified_unix_ms(draft.modified)
        }));
    }

    encode_output(
        PROJECT_LIST_DRAFTS,
        json!({
            "status": "ok",
            "drafts": rows,
            "total": total,
            "truncated": total > limit
        }),
    )
}

fn modified_unix_ms(modified: SystemTime) -> Option<u64> {
    let milliseconds = modified.duration_since(UNIX_EPOCH).ok()?.as_millis();
    u64::try_from(milliseconds).ok()
}

fn bounded_display_name(name: &str) -> String {
    name.chars().take(MAX_DISPLAY_NAME_CHARS).collect()
}

fn bounded_safe_display_name(name: &str) -> String {
    sanitize_display_text(name, MAX_DISPLAY_NAME_CHARS)
}

pub(crate) fn import_path_approval_display(path: &Path) -> Option<String> {
    let text = path.to_str()?;
    let chars = text.chars().count();
    if !path.is_absolute()
        || chars == 0
        || chars > MAX_IMPORT_PATH_CHARS
        || text.contains('\0')
        || contains_lexical_dot_component(text)
        || path
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
    {
        return None;
    }
    Some(sanitize_display_text(text, MAX_IMPORT_PATH_CHARS))
}

fn sanitize_display_text(text: &str, max_chars: usize) -> String {
    text.chars()
        .take(max_chars)
        .map(|character| {
            if character.is_control() || is_bidi_formatting(character) {
                '\u{fffd}'
            } else {
                character
            }
        })
        .collect()
}

fn is_bidi_formatting(character: char) -> bool {
    matches!(
        character,
        '\u{061c}'
            | '\u{200e}'
            | '\u{200f}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202a}'..='\u{202e}'
            | '\u{2066}'..='\u{2069}'
    )
}

fn preflight_import_media_path(path: &str) -> Result<ValidatedImportMedia, String> {
    inspect_import_file(Path::new(path))
}

fn inspect_import_file(path: &Path) -> Result<ValidatedImportMedia, String> {
    let initial_metadata = fs::symlink_metadata(path).map_err(|internal| {
        error!(
            path = %path.display(),
            error = %internal,
            "agent media import could not inspect requested path"
        );
        public_import_path_io_error(internal.kind())
    })?;
    ensure_import_regular_file(path, &initial_metadata)?;
    let initial_identity = import_file_identity(&initial_metadata);

    let canonical = fs::canonicalize(path).map_err(|internal| {
        error!(
            path = %path.display(),
            error = %internal,
            "agent media import could not canonicalize requested path"
        );
        public_import_path_io_error(internal.kind())
    })?;
    let Some(canonical_text) = canonical.to_str() else {
        error!(
            path = %canonical.display(),
            "agent media import canonical path is not valid Unicode"
        );
        return Err(
            "project_import_media failed: the resolved media path cannot be safely represented"
                .into(),
        );
    };
    if canonical_text.chars().count() > MAX_IMPORT_PATH_CHARS {
        error!(
            path = %canonical.display(),
            "agent media import canonical path exceeds its character bound"
        );
        return Err(
            "project_import_media failed: the resolved media path exceeds the safe length limit"
                .into(),
        );
    }

    let canonical_metadata = fs::symlink_metadata(&canonical).map_err(|internal| {
        error!(
            path = %canonical.display(),
            error = %internal,
            "agent media import could not inspect canonical path"
        );
        public_import_path_io_error(internal.kind())
    })?;
    ensure_import_regular_file(&canonical, &canonical_metadata)?;
    if initial_identity != import_file_identity(&canonical_metadata) {
        error!(
            requested = %path.display(),
            canonical = %canonical.display(),
            "agent media import file identity changed during canonicalization"
        );
        return Err(
            "project_import_media failed: the requested media file changed during safety checks; not started"
                .into(),
        );
    }

    // Reinspect both the final component and its resolution. This catches
    // ordinary replacement races inside the preflight window; the worker API
    // remains path-based, so a residual race still exists after return.
    let final_metadata = fs::symlink_metadata(path).map_err(|internal| {
        error!(
            path = %path.display(),
            error = %internal,
            "agent media import could not reinspect requested path"
        );
        public_import_path_io_error(internal.kind())
    })?;
    ensure_import_regular_file(path, &final_metadata)?;
    if initial_identity != import_file_identity(&final_metadata) {
        error!(
            path = %path.display(),
            "agent media import file identity changed during preflight"
        );
        return Err(
            "project_import_media failed: the requested media file changed during safety checks; not started"
                .into(),
        );
    }
    let verified_canonical = fs::canonicalize(path).map_err(|internal| {
        error!(
            path = %path.display(),
            error = %internal,
            "agent media import could not recanonicalize requested path"
        );
        public_import_path_io_error(internal.kind())
    })?;
    if verified_canonical != canonical {
        error!(
            first = %canonical.display(),
            second = %verified_canonical.display(),
            "agent media import canonical path changed during preflight"
        );
        return Err(
            "project_import_media failed: the requested media file changed during safety checks; not started"
                .into(),
        );
    }

    Ok(ValidatedImportMedia {
        canonical_path: canonical,
        identity: initial_identity,
    })
}

fn ensure_import_regular_file(path: &Path, metadata: &fs::Metadata) -> Result<(), String> {
    if metadata.file_type().is_symlink() || metadata_is_reparse_point(metadata) {
        error!(
            path = %path.display(),
            "agent media import rejected a final symlink or reparse point"
        );
        return Err(
            "project_import_media argument 'path' must not name a symbolic link or reparse point"
                .into(),
        );
    }
    if !metadata.file_type().is_file() {
        error!(
            path = %path.display(),
            "agent media import rejected a non-regular file"
        );
        return Err("project_import_media argument 'path' must name a regular file".into());
    }
    Ok(())
}

fn public_import_path_io_error(kind: io::ErrorKind) -> String {
    if kind == io::ErrorKind::NotFound {
        "project_import_media failed: the requested media file does not exist".into()
    } else {
        "project_import_media failed: the requested media file could not be safely inspected".into()
    }
}

#[cfg(unix)]
#[derive(Debug, PartialEq, Eq)]
struct ImportFileIdentity {
    device: u64,
    inode: u64,
}

#[cfg(unix)]
fn import_file_identity(metadata: &fs::Metadata) -> ImportFileIdentity {
    use std::os::unix::fs::MetadataExt as _;

    ImportFileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
    }
}

// `volume_serial_number()`/`file_index()` are still unstable
// (`windows_by_handle`), so approximate identity with stable metadata.
#[cfg(windows)]
#[derive(Debug, PartialEq, Eq)]
struct ImportFileIdentity {
    creation_time: u64,
    last_write_time: u64,
    file_size: u64,
    file_attributes: u32,
}

#[cfg(windows)]
fn import_file_identity(metadata: &fs::Metadata) -> ImportFileIdentity {
    use std::os::windows::fs::MetadataExt as _;

    ImportFileIdentity {
        creation_time: metadata.creation_time(),
        last_write_time: metadata.last_write_time(),
        file_size: metadata.file_size(),
        file_attributes: metadata.file_attributes(),
    }
}

#[cfg(not(any(unix, windows)))]
#[derive(Debug, PartialEq, Eq)]
struct ImportFileIdentity {
    len: u64,
    modified: Option<SystemTime>,
}

#[cfg(not(any(unix, windows)))]
fn import_file_identity(metadata: &fs::Metadata) -> ImportFileIdentity {
    ImportFileIdentity {
        len: metadata.len(),
        modified: metadata.modified().ok(),
    }
}

#[cfg(windows)]
const WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;

#[cfg(windows)]
fn metadata_is_reparse_point(metadata: &fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt as _;

    metadata.file_attributes() & WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT != 0
}

#[cfg(not(windows))]
fn metadata_is_reparse_point(_metadata: &fs::Metadata) -> bool {
    false
}

fn encode_output(tool: &'static str, value: Value) -> Result<ToolOutput, String> {
    let encoded = serde_json::to_string(&value).map_err(|internal| {
        error!(tool, error = %internal, "could not encode project-tool response");
        format!("{tool} failed: its response could not be encoded")
    })?;
    if encoded.len() > MAX_OUTPUT_BYTES {
        error!(
            tool,
            bytes = encoded.len(),
            "project-tool response exceeded its output bound"
        );
        return Err(format!(
            "{tool} failed: its response exceeded the output limit"
        ));
    }
    Ok(ToolOutput::text(encoded))
}

fn spec(name: &str, description: &str, parameters: Value, tier: ToolTier) -> HostToolSpec {
    HostToolSpec {
        name: name.to_string(),
        description: description.to_string(),
        parameters,
        tier,
    }
}

fn list_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_DRAFT_LIMIT,
                "default": DEFAULT_DRAFT_LIMIT,
                "description": "Maximum number of newest drafts to return."
            }
        },
        "required": []
    })
}

fn empty_object_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {},
        "required": []
    })
}

fn open_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "draft_id": {
                "type": "string",
                "minLength": MIN_DRAFT_ID_CHARS,
                "maxLength": MAX_DRAFT_ID_CHARS,
                "description": "Canonical app-owned draft identity returned by project_list_drafts."
            }
        },
        "required": ["draft_id"]
    })
}

fn import_media_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "path": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_IMPORT_PATH_CHARS,
                "description": "Absolute path to one existing external regular media file."
            }
        },
        "required": ["path"]
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Request {
    ListDrafts { limit: usize },
    Save,
    Open { draft_id: String },
    ImportMedia { path: String },
}

fn parse_request(name: &str, arguments: &Value) -> Result<Request, String> {
    if !TOOL_NAMES.contains(&name) {
        return Err("unknown project tool".into());
    }
    let object = arguments
        .as_object()
        .ok_or_else(|| format!("{name} arguments must be an object"))?;

    match name {
        PROJECT_LIST_DRAFTS => parse_list_request(object),
        PROJECT_SAVE => {
            if !object.is_empty() {
                return Err("project_save does not accept arguments".into());
            }
            Ok(Request::Save)
        }
        PROJECT_OPEN => parse_open_request(object),
        PROJECT_IMPORT_MEDIA => parse_import_media_request(object),
        _ => Err("unknown project tool".into()),
    }
}

fn parse_list_request(object: &Map<String, Value>) -> Result<Request, String> {
    if object.len() > 1 || object.keys().any(|key| key != "limit") {
        return Err("project_list_drafts has an unknown argument".into());
    }
    let Some(value) = object.get("limit") else {
        return Ok(Request::ListDrafts {
            limit: DEFAULT_DRAFT_LIMIT,
        });
    };
    let limit = value.as_u64().ok_or_else(|| {
        "project_list_drafts argument 'limit' must be an integer from 1 through 100".to_string()
    })?;
    if !(1..=MAX_DRAFT_LIMIT as u64).contains(&limit) {
        return Err(
            "project_list_drafts argument 'limit' must be an integer from 1 through 100".into(),
        );
    }
    Ok(Request::ListDrafts {
        limit: limit as usize,
    })
}

fn parse_open_request(object: &Map<String, Value>) -> Result<Request, String> {
    if object.len() != 1 || object.keys().any(|key| key != "draft_id") {
        return Err("project_open requires only the 'draft_id' argument".into());
    }
    let draft_id = object
        .get("draft_id")
        .and_then(Value::as_str)
        .ok_or_else(|| "project_open argument 'draft_id' must be a string".to_string())?;
    let chars = draft_id.chars().count();
    if !(MIN_DRAFT_ID_CHARS..=MAX_DRAFT_ID_CHARS).contains(&chars) {
        return Err(format!(
            "project_open argument 'draft_id' must contain {MIN_DRAFT_ID_CHARS} through {MAX_DRAFT_ID_CHARS} characters"
        ));
    }
    Ok(Request::Open {
        draft_id: draft_id.to_owned(),
    })
}

fn parse_import_media_request(object: &Map<String, Value>) -> Result<Request, String> {
    if object.len() != 1 || object.keys().any(|key| key != "path") {
        return Err("project_import_media requires only the 'path' argument".into());
    }
    let path = object
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| "project_import_media argument 'path' must be a string".to_string())?;
    let chars = path.chars().take(MAX_IMPORT_PATH_CHARS + 1).count();
    if !(1..=MAX_IMPORT_PATH_CHARS).contains(&chars) {
        return Err(format!(
            "project_import_media argument 'path' must contain 1 through {MAX_IMPORT_PATH_CHARS} characters"
        ));
    }
    if path.contains('\0') {
        return Err("project_import_media argument 'path' must not contain NUL characters".into());
    }
    let parsed = Path::new(path);
    if !parsed.is_absolute() {
        return Err("project_import_media argument 'path' must be absolute".into());
    }
    if contains_lexical_dot_component(path)
        || parsed
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
    {
        return Err(
            "project_import_media argument 'path' must not contain '.' or '..' components".into(),
        );
    }
    Ok(Request::ImportMedia {
        path: path.to_owned(),
    })
}

fn contains_lexical_dot_component(path: &str) -> bool {
    path.split(|character| character == '/' || (cfg!(windows) && character == '\\'))
        .any(|component| matches!(component, "." | ".."))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::path::PathBuf;
    use std::time::Duration;

    use super::*;

    fn draft(
        id: &str,
        name: impl Into<String>,
        modified: SystemTime,
    ) -> crate::drafts::DraftSummary {
        crate::drafts::DraftSummary {
            name: name.into(),
            project: crate::drafts::project_file(&crate::drafts::root_dir().join(id)),
            modified,
        }
    }

    fn output_json(output: &ToolOutput) -> Value {
        assert!(output.images.is_empty());
        serde_json::from_str(&output.text).expect("compact JSON project-tool output")
    }

    fn path_arguments(path: &Path) -> Value {
        json!({"path": path.to_str().expect("UTF-8 test path")})
    }

    fn validated_media(path: &Path) -> ValidatedImportMedia {
        validated_import_media(&path_arguments(path)).expect("validated media fixture")
    }

    fn missing_draft_id() -> String {
        let mut time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("current time after epoch")
            .as_nanos();
        loop {
            // Production ids use epoch milliseconds, so an epoch-nanosecond
            // group is also well outside ids created during this test.
            let id = format!("{time:x}-ffffffffffffffff");
            let project = crate::drafts::project_file(&crate::drafts::root_dir().join(&id));
            if !project.exists() {
                return id;
            }
            time = time.checked_add(1).expect("draft-id search overflow");
        }
    }

    #[test]
    fn specs_have_exact_names_tiers_and_strict_schemas() {
        let registry = specs();
        assert_eq!(
            registry
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>(),
            TOOL_NAMES
        );
        assert_eq!(
            registry.iter().map(|entry| entry.tier).collect::<Vec<_>>(),
            [
                ToolTier::ReadOnly,
                ToolTier::Workspace,
                ToolTier::System,
                ToolTier::System,
            ]
        );
        assert_eq!(registry[0].parameters, list_schema());
        assert_eq!(registry[1].parameters, empty_object_schema());
        assert_eq!(registry[2].parameters, open_schema());
        assert_eq!(registry[3].parameters, import_media_schema());
        assert_eq!(registry[0].parameters["properties"]["limit"]["minimum"], 1);
        assert_eq!(
            registry[0].parameters["properties"]["limit"]["maximum"],
            100
        );
        assert_eq!(registry[0].parameters["properties"]["limit"]["default"], 50);
        assert_eq!(
            registry[2].parameters["properties"]["draft_id"]["minLength"],
            MIN_DRAFT_ID_CHARS
        );
        assert_eq!(
            registry[2].parameters["properties"]["draft_id"]["maxLength"],
            MAX_DRAFT_ID_CHARS
        );
        assert_eq!(registry[2].parameters["required"], json!(["draft_id"]));
        assert_eq!(registry[3].parameters["properties"]["path"]["minLength"], 1);
        assert_eq!(
            registry[3].parameters["properties"]["path"]["maxLength"],
            MAX_IMPORT_PATH_CHARS
        );
        assert_eq!(registry[3].parameters["required"], json!(["path"]));
        for entry in registry {
            assert_eq!(entry.parameters["type"], "object");
            assert_eq!(entry.parameters["additionalProperties"], false);
            assert!(!entry.description.is_empty());
        }
    }

    #[test]
    fn parser_rejects_non_objects_unknown_fields_and_non_integer_limits() {
        for arguments in [Value::Null, json!([]), json!(""), json!(false), json!(1)] {
            assert!(parse_request(PROJECT_LIST_DRAFTS, &arguments).is_err());
            assert!(parse_request(PROJECT_SAVE, &arguments).is_err());
            assert!(parse_request(PROJECT_OPEN, &arguments).is_err());
            assert!(parse_request(PROJECT_IMPORT_MEDIA, &arguments).is_err());
        }
        for arguments in [
            json!({"extra": 1}),
            json!({"limit": 1, "extra": 2}),
            json!({"limit": null}),
            json!({"limit": false}),
            json!({"limit": 1.0}),
            json!({"limit": 0}),
            json!({"limit": -1}),
            json!({"limit": 101}),
        ] {
            assert!(
                parse_request(PROJECT_LIST_DRAFTS, &arguments).is_err(),
                "{arguments}"
            );
        }
        assert!(parse_request(PROJECT_SAVE, &json!({"limit": 1})).is_err());
        for arguments in [
            json!({}),
            json!({"extra": "abc-1"}),
            json!({"draft_id": "abc-1", "extra": true}),
            json!({"draft_id": null}),
            json!({"draft_id": false}),
            json!({"draft_id": 1}),
            json!({"draft_id": ""}),
            json!({"draft_id": "a"}),
            json!({"draft_id": "x".repeat(MAX_DRAFT_ID_CHARS + 1)}),
        ] {
            assert!(
                parse_request(PROJECT_OPEN, &arguments).is_err(),
                "{arguments}"
            );
        }
        assert!(parse_request("project_future", &json!({})).is_err());
        assert_eq!(
            parse_request(PROJECT_LIST_DRAFTS, &json!({})),
            Ok(Request::ListDrafts {
                limit: DEFAULT_DRAFT_LIMIT
            })
        );
        assert_eq!(
            parse_request(PROJECT_LIST_DRAFTS, &json!({"limit": 100})),
            Ok(Request::ListDrafts { limit: 100 })
        );
        assert_eq!(parse_request(PROJECT_SAVE, &json!({})), Ok(Request::Save));
        assert_eq!(
            parse_request(PROJECT_OPEN, &json!({"draft_id": "abc-1"})),
            Ok(Request::Open {
                draft_id: "abc-1".into()
            })
        );

        let temp = tempfile::tempdir().expect("tempdir");
        let absolute = temp.path().join("media.mp4");
        let absolute_text = absolute.to_str().expect("UTF-8 temp path");
        for arguments in [
            json!({}),
            json!({"extra": absolute_text}),
            json!({"path": absolute_text, "extra": true}),
            json!({"path": null}),
            json!({"path": false}),
            json!({"path": 1}),
            json!({"path": ""}),
            json!({"path": "relative/media.mp4"}),
            json!({"path": format!("{absolute_text}\0")}),
            json!({"path": "x".repeat(MAX_IMPORT_PATH_CHARS + 1)}),
        ] {
            assert!(
                parse_request(PROJECT_IMPORT_MEDIA, &arguments).is_err(),
                "{arguments}"
            );
        }
        assert_eq!(
            parse_request(PROJECT_IMPORT_MEDIA, &json!({"path": absolute_text})),
            Ok(Request::ImportMedia {
                path: absolute_text.into()
            })
        );
    }

    #[test]
    fn import_preflight_accepts_only_existing_absolute_regular_files() {
        let temp = tempfile::tempdir().expect("tempdir");
        let media = temp.path().join("clip.mp4");
        fs::write(&media, b"media").expect("write media fixture");
        let expected = media.canonicalize().expect("canonical media");
        let arguments = path_arguments(&media);

        let validated = validated_import_media(&arguments).expect("valid media path");
        assert_eq!(validated.canonical_path(), expected);
        validate_request(PROJECT_IMPORT_MEDIA, &arguments).expect("preapproval validation");

        let missing = temp.path().join("agent-secret-missing.mp4");
        let missing_error = validate_request(PROJECT_IMPORT_MEDIA, &path_arguments(&missing))
            .expect_err("missing source must fail");
        assert_eq!(
            missing_error,
            "project_import_media failed: the requested media file does not exist"
        );
        assert!(!missing_error.contains("agent-secret"));
        assert!(!missing_error.contains(temp.path().to_string_lossy().as_ref()));

        let directory_error = validate_request(PROJECT_IMPORT_MEDIA, &path_arguments(temp.path()))
            .expect_err("directory must fail");
        assert_eq!(
            directory_error,
            "project_import_media argument 'path' must name a regular file"
        );
        assert!(!directory_error.contains(temp.path().to_string_lossy().as_ref()));
    }

    #[test]
    fn import_parser_rejects_lexical_dot_components_and_unsafe_text() {
        let temp = tempfile::tempdir().expect("tempdir");
        let media = temp.path().join("clip.mp4");
        fs::write(&media, b"media").expect("write media fixture");
        let child = temp.path().join("child");
        fs::create_dir(&child).expect("create child");

        let dot = temp.path().join(".").join("clip.mp4");
        let parent = child.join("..").join("clip.mp4");
        for path in [&dot, &parent] {
            let error = parse_request(PROJECT_IMPORT_MEDIA, &path_arguments(path))
                .expect_err("lexical traversal component must fail");
            assert!(error.contains("must not contain '.' or '..'"), "{error}");
            assert!(!error.contains(temp.path().to_string_lossy().as_ref()));
        }

        let nul_path = format!("{}\0clip.mp4", temp.path().display());
        let nul = parse_request(PROJECT_IMPORT_MEDIA, &json!({"path": nul_path}))
            .expect_err("NUL must fail");
        assert!(nul.contains("must not contain NUL"), "{nul}");

        let relative = parse_request(PROJECT_IMPORT_MEDIA, &json!({"path": "relative/clip.mp4"}))
            .expect_err("relative path must fail");
        assert_eq!(
            relative,
            "project_import_media argument 'path' must be absolute"
        );

        let oversized = format!(
            "{}{}",
            temp.path().display(),
            "x".repeat(MAX_IMPORT_PATH_CHARS + 1)
        );
        let oversized_error = parse_request(PROJECT_IMPORT_MEDIA, &json!({"path": oversized}))
            .expect_err("oversized path must fail");
        assert!(oversized_error.contains("must contain 1 through"));
        assert!(!oversized_error.contains(temp.path().to_string_lossy().as_ref()));
    }

    #[cfg(unix)]
    #[test]
    fn import_preflight_rejects_final_symlinks_but_canonicalizes_intermediate_links() {
        use std::os::unix::fs::symlink;
        use std::os::unix::net::UnixListener;

        let temp = tempfile::tempdir().expect("tempdir");
        let real_dir = temp.path().join("real");
        fs::create_dir(&real_dir).expect("real directory");
        let media = real_dir.join("clip.mp4");
        fs::write(&media, b"media").expect("write media fixture");

        let final_link = temp.path().join("final-link.mp4");
        symlink(&media, &final_link).expect("final symlink");
        let final_error = validate_request(PROJECT_IMPORT_MEDIA, &path_arguments(&final_link))
            .expect_err("final symlink must fail");
        assert!(final_error.contains("symbolic link or reparse point"));
        assert!(!final_error.contains(temp.path().to_string_lossy().as_ref()));

        let socket = temp.path().join("not-media.sock");
        let _listener = UnixListener::bind(&socket).expect("Unix socket");
        let socket_error = validate_request(PROJECT_IMPORT_MEDIA, &path_arguments(&socket))
            .expect_err("non-regular file must fail");
        assert_eq!(
            socket_error,
            "project_import_media argument 'path' must name a regular file"
        );
        assert!(!socket_error.contains(temp.path().to_string_lossy().as_ref()));

        let directory_link = temp.path().join("linked-directory");
        symlink(&real_dir, &directory_link).expect("intermediate directory symlink");
        let through_intermediate = directory_link.join("clip.mp4");
        let validated = validated_import_media(&path_arguments(&through_intermediate))
            .expect("intermediate link is canonicalized");
        assert_eq!(
            validated.canonical_path(),
            media.canonicalize().expect("canonical media")
        );
    }

    #[test]
    fn open_validation_rejects_malformed_and_missing_ids_without_path_leaks() {
        let private_path = "/private/agent-secret/project.cutlass";
        let malformed = validate_request(
            PROJECT_OPEN,
            &json!({
                "draft_id": private_path
            }),
        )
        .expect_err("filesystem paths are not draft ids");
        assert_eq!(
            malformed,
            "project_open argument 'draft_id' must be a canonical app-owned draft ID"
        );
        assert!(!malformed.contains("/private"));
        assert!(!malformed.contains("agent-secret"));

        let missing_id = missing_draft_id();
        let missing = validate_request(PROJECT_OPEN, &json!({"draft_id": missing_id}))
            .expect_err("missing draft must fail read-only preflight");
        assert!(missing.starts_with("project_open failed:"));
        assert!(
            !missing.contains(crate::drafts::root_dir().to_string_lossy().as_ref()),
            "{missing}"
        );
        assert!(!missing.contains("project.cutlass"), "{missing}");
        assert!(missing.len() < 160);
    }

    #[test]
    fn draft_serialization_is_deterministic_bounded_sorted_and_path_free() {
        let before_epoch = UNIX_EPOCH
            .checked_sub(Duration::from_millis(1))
            .expect("pre-epoch time");
        let oversized_name = "é".repeat(MAX_DISPLAY_NAME_CHARS + 20);
        let fixtures = vec![
            draft("abc-1", "old", UNIX_EPOCH + Duration::from_millis(1_000)),
            draft(
                "abc-3",
                oversized_name,
                UNIX_EPOCH + Duration::from_millis(3_000),
            ),
            draft("abc-0", "pre-epoch", before_epoch),
            draft("abc-2", "middle", UNIX_EPOCH + Duration::from_millis(2_000)),
        ];

        let first = list_drafts_output(fixtures.clone(), 2).expect("serialize drafts");
        let second = list_drafts_output(fixtures, 2).expect("serialize drafts again");
        assert_eq!(first, second);
        assert!(!first.text.contains('\n'), "response must be compact JSON");
        assert!(
            !first
                .text
                .contains(crate::drafts::root_dir().to_string_lossy().as_ref())
        );
        assert!(!first.text.contains("project.cutlass"));

        let value = output_json(&first);
        assert_eq!(
            value
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["drafts", "status", "total", "truncated"])
        );
        assert_eq!(value["status"], "ok");
        assert_eq!(value["total"], 4);
        assert_eq!(value["truncated"], true);
        let rows = value["drafts"].as_array().expect("draft rows");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["draft_id"], "abc-3");
        assert_eq!(rows[0]["modified_unix_ms"], 3_000);
        assert_eq!(
            rows[0]["name"].as_str().unwrap().chars().count(),
            MAX_DISPLAY_NAME_CHARS
        );
        assert_eq!(rows[1]["draft_id"], "abc-2");
        assert_eq!(
            rows[0]
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["draft_id", "modified_unix_ms", "name"])
        );

        let all = output_json(
            &list_drafts_output(
                vec![draft("abc-0", "pre-epoch", before_epoch)],
                MAX_DRAFT_LIMIT,
            )
            .expect("serialize pre-epoch draft"),
        );
        assert!(all["drafts"][0]["modified_unix_ms"].is_null());
        assert_eq!(all["truncated"], false);
    }

    #[test]
    fn open_output_is_bounded_path_free_and_bound_to_the_requested_identity() {
        let draft_id = "abcdef-12";
        let project = crate::drafts::project_file(&crate::drafts::root_dir().join(draft_id));
        let opened = OpenProjectRpcResult {
            path: project.clone(),
            project_name: "é".repeat(MAX_DISPLAY_NAME_CHARS + 20),
            missing_media_count: 7,
        };
        let output = open_output(draft_id, &opened).expect("safe open output");
        assert!(!output.text.contains('\n'), "response must be compact JSON");
        assert!(!output.text.contains(project.to_string_lossy().as_ref()));
        assert!(!output.text.contains("project.cutlass"));

        let value = output_json(&output);
        assert_eq!(
            value
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["draft_id", "missing_media_count", "project_name", "status"])
        );
        assert_eq!(value["status"], "ok");
        assert_eq!(value["draft_id"], draft_id);
        assert_eq!(value["missing_media_count"], 7);
        assert_eq!(
            value["project_name"].as_str().unwrap().chars().count(),
            MAX_DISPLAY_NAME_CHARS
        );

        let mismatch =
            open_output("abcdef-13", &opened).expect_err("identity mismatch must fail closed");
        assert!(mismatch.contains("current session was replaced"));
        assert!(!mismatch.contains(project.to_string_lossy().as_ref()));
        assert!(!mismatch.contains("project.cutlass"));

        let outside = OpenProjectRpcResult {
            path: PathBuf::from("/private/agent-secret/project.cutlass"),
            project_name: "outside".into(),
            missing_media_count: 0,
        };
        let outside_error =
            open_output(draft_id, &outside).expect_err("outside path must fail closed");
        assert!(outside_error.contains("current session was replaced"));
        assert!(!outside_error.contains("/private"));
        assert!(!outside_error.contains("agent-secret"));
    }

    #[test]
    fn open_rpc_errors_preserve_not_started_and_unknown_outcomes_without_paths() {
        let cancelled = public_open_error(
            "open project request cancelled before worker claim; not started for \
             /private/agent-secret/project.cutlass",
        );
        assert_eq!(cancelled, "project_open was cancelled before it started");
        assert!(!cancelled.contains("/private"));

        let not_started = public_open_error(
            "open project request failed: preview worker is not running; not started for \
             /private/agent-secret/project.cutlass",
        );
        assert_eq!(
            not_started,
            "project_open failed before the editor started it"
        );
        assert!(!not_started.contains("/private"));

        for internal in [
            "open project request timed out after worker claim; outcome unknown after 30000 ms \
             for /private/agent-secret/project.cutlass",
            "open project outcome uncertain/partially committed: \
             /private/agent-secret/project.cutlass replaced the session",
        ] {
            let unknown = public_open_error(internal);
            assert!(unknown.contains("outcome unknown"), "{unknown}");
            assert!(unknown.contains("project may have opened"), "{unknown}");
            assert!(!unknown.contains("/private"), "{unknown}");
            assert!(!unknown.contains("agent-secret"), "{unknown}");
            assert!(unknown.len() < 160);
        }

        let failed = public_open_error(
            "open project failed for /private/agent-secret/project.cutlass: corrupt data",
        );
        assert_eq!(
            failed,
            "project_open failed: the requested draft could not be opened"
        );
        assert!(!failed.contains("/private"));
        assert!(!failed.contains("agent-secret"));
        assert!(!failed.contains("may have opened"));
    }

    #[test]
    fn import_output_is_bounded_path_free_and_bound_to_the_acknowledged_file() {
        let temp = tempfile::tempdir().expect("tempdir");
        let filename = format!("{}.mp4", "a".repeat(MAX_DISPLAY_NAME_CHARS + 16));
        let media = temp.path().join(&filename);
        fs::write(&media, b"media").expect("write media fixture");
        let expected = validated_media(&media);
        let imported = ImportMediaRpcResult {
            media_id: 41,
            path: expected.canonical_path().to_path_buf(),
        };
        let output = import_media_output(&expected, &imported).expect("safe import output");
        assert!(!output.text.contains('\n'), "response must be compact JSON");
        assert!(
            !output.text.contains(temp.path().to_string_lossy().as_ref()),
            "{}",
            output.text
        );

        let value = output_json(&output);
        assert_eq!(
            value
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["filename", "media_id", "status"])
        );
        assert_eq!(value["status"], "ok");
        assert_eq!(value["media_id"], 41);
        assert_eq!(
            value["filename"].as_str().unwrap().chars().count(),
            MAX_DISPLAY_NAME_CHARS
        );
        assert!(output.text.len() < 1_024);

        assert_eq!(
            bounded_safe_display_name("clip\n\u{2028}\u{202e}name.mp4"),
            "clip\u{fffd}\u{fffd}\u{fffd}name.mp4"
        );
    }

    #[test]
    fn import_output_fails_closed_on_identity_mismatch_and_invalid_acknowledgement() {
        let temp = tempfile::tempdir().expect("tempdir");
        let approved = temp.path().join("approved-secret.mp4");
        let different = temp.path().join("different-secret.mp4");
        fs::write(&approved, b"approved").expect("approved fixture");
        fs::write(&different, b"different").expect("different fixture");
        let approved = validated_media(&approved);
        let different = different.canonicalize().expect("canonical different");

        let mismatch = import_media_output(
            &approved,
            &ImportMediaRpcResult {
                media_id: 7,
                path: different,
            },
        )
        .expect_err("different acknowledged file must fail");
        assert!(mismatch.contains("different source file"), "{mismatch}");
        assert!(
            mismatch.contains("media may have been imported"),
            "{mismatch}"
        );
        assert!(!mismatch.contains("approved-secret"), "{mismatch}");
        assert!(!mismatch.contains("different-secret"), "{mismatch}");
        assert!(!mismatch.contains(temp.path().to_string_lossy().as_ref()));

        let zero_id = import_media_output(
            &approved,
            &ImportMediaRpcResult {
                media_id: 0,
                path: approved.canonical_path().to_path_buf(),
            },
        )
        .expect_err("zero media id must fail");
        assert!(zero_id.contains("invalid media identity"), "{zero_id}");
        assert!(
            zero_id.contains("media may have been imported"),
            "{zero_id}"
        );
        assert!(!zero_id.contains("approved-secret"), "{zero_id}");

        let missing_acknowledgement = import_media_output(
            &approved,
            &ImportMediaRpcResult {
                media_id: 8,
                path: temp.path().join("missing-agent-secret.mp4"),
            },
        )
        .expect_err("missing acknowledged path must fail");
        assert!(
            missing_acknowledgement.contains("could not verify"),
            "{missing_acknowledgement}"
        );
        assert!(
            missing_acknowledgement.contains("media may have been imported"),
            "{missing_acknowledgement}"
        );
        assert!(
            !missing_acknowledgement.contains("missing-agent-secret"),
            "{missing_acknowledgement}"
        );
        assert!(
            !missing_acknowledgement.contains(temp.path().to_string_lossy().as_ref()),
            "{missing_acknowledgement}"
        );
    }

    #[test]
    fn import_token_rejects_same_path_replacement_before_dispatch_and_after_ack() {
        let temp = tempfile::tempdir().expect("tempdir");
        let media = temp.path().join("approved-secret.mp4");
        let replacement = temp.path().join("replacement-secret.mp4");
        fs::write(&media, b"approved").expect("approved fixture");
        fs::write(&replacement, b"replacement").expect("replacement fixture");

        let approved = validated_media(&media);
        let replacement_before_move = validated_media(&replacement);
        assert_ne!(
            approved.identity, replacement_before_move.identity,
            "simultaneously existing files must have distinct identities"
        );

        fs::remove_file(&media).expect("remove approved fixture");
        fs::rename(&replacement, &media).expect("move replacement onto approved path");
        let current = validated_media(&media);
        assert_eq!(
            approved.canonical_path(),
            current.canonical_path(),
            "replacement deliberately keeps the approved canonical path"
        );
        assert_eq!(current.identity, replacement_before_move.identity);
        assert!(!approved.identifies_same_file(&current));

        let pre_dispatch = call_with_approved_import(
            None,
            PROJECT_IMPORT_MEDIA,
            &path_arguments(&media),
            Some(&approved),
            &AtomicBool::new(false),
        )
        .expect_err("same-path replacement must stop before worker dispatch");
        assert_eq!(
            pre_dispatch,
            "project_import_media failed: the media file changed after approval; not started"
        );
        assert!(!pre_dispatch.contains("approved-secret"));
        assert!(!pre_dispatch.contains(temp.path().to_string_lossy().as_ref()));

        let acknowledged = import_media_output(
            &approved,
            &ImportMediaRpcResult {
                media_id: 9,
                path: media,
            },
        )
        .expect_err("acknowledged same-path replacement must fail closed");
        assert!(
            acknowledged.contains("different source file"),
            "{acknowledged}"
        );
        assert!(
            acknowledged.contains("media may have been imported"),
            "{acknowledged}"
        );
        assert!(!acknowledged.contains("approved-secret"));
        assert!(!acknowledged.contains("replacement-secret"));
        assert!(
            !acknowledged.contains(temp.path().to_string_lossy().as_ref()),
            "{acknowledged}"
        );
    }

    #[test]
    fn import_rpc_errors_preserve_not_started_and_unknown_outcomes_without_paths() {
        let private = "/private/agent-secret/clip.mp4";
        let cancelled = public_import_error(&format!(
            "import media request cancelled before worker claim; not started for {private}"
        ));
        assert_eq!(
            cancelled,
            "project_import_media was cancelled before it started"
        );
        assert!(!cancelled.contains(private));

        let not_started = public_import_error(&format!(
            "import media request failed: preview worker is not running; not started for {private}"
        ));
        assert_eq!(
            not_started,
            "project_import_media failed before the editor started it"
        );
        assert!(!not_started.contains(private));

        for internal in [
            format!(
                "import media request timed out after worker claim; outcome unknown for {private}"
            ),
            format!("import media outcome uncertain/partially committed while decoding {private}"),
            format!("import succeeded for {private} but its pool record could not be read back"),
        ] {
            let unknown = public_import_error(&internal);
            assert!(unknown.contains("outcome unknown"), "{unknown}");
            assert!(
                unknown.contains("media may have been imported"),
                "{unknown}"
            );
            assert!(!unknown.contains(private), "{unknown}");
            assert!(unknown.len() < 180);
        }

        let decoder = public_import_error(&format!(
            "decoder failed for {private}: stream not started; arbitrary details"
        ));
        assert_eq!(
            decoder,
            "project_import_media failed: the media file could not be imported"
        );
        assert!(!decoder.contains(private));
        assert!(!decoder.contains("decoder"));
        assert!(!decoder.contains("arbitrary"));
    }

    #[test]
    fn save_output_contains_only_safe_identity_and_clean_state() {
        let project = crate::drafts::project_file(&crate::drafts::root_dir().join("abcdef-12"));
        let output = save_output(&project, false).expect("safe save output");
        assert!(!output.text.contains(project.to_string_lossy().as_ref()));
        assert!(!output.text.contains("project.cutlass"));
        let value = output_json(&output);
        assert_eq!(
            value
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["dirty", "draft_id", "status"])
        );
        assert_eq!(
            value,
            json!({"status":"ok","draft_id":"abcdef-12","dirty":false})
        );

        let outside = PathBuf::from("/private/agent-secret/project.cutlass");
        let error = save_output(&outside, false).expect_err("outside path must fail");
        assert!(!error.contains("/private"));
        assert!(!error.contains("agent-secret"));

        let dirty_error = save_output(&project, true).expect_err("dirty save must fail");
        assert!(!dirty_error.contains(project.to_string_lossy().as_ref()));

        let internal = "save failed for /private/agent-secret/project.cutlass: disk error";
        let public = public_save_error(internal);
        assert_eq!(
            public,
            "project_save failed: the current draft could not be saved"
        );
        assert!(!public.contains("/private"));
        assert!(!public.contains("agent-secret"));
    }

    #[test]
    fn unavailable_worker_is_an_honest_bounded_error() {
        let error = call(None, PROJECT_SAVE, &json!({}), &AtomicBool::new(false))
            .expect_err("missing worker");
        assert_eq!(
            error,
            "project_save failed: the editor worker is unavailable"
        );
        assert!(error.len() < 128);

        let cancelled = call(
            None,
            PROJECT_OPEN,
            &json!({"draft_id": "abc-1"}),
            &AtomicBool::new(true),
        )
        .expect_err("pre-cancelled open must stop before draft resolution");
        assert_eq!(cancelled, "project_open was cancelled before it started");

        let temp = tempfile::tempdir().expect("tempdir");
        let missing = temp.path().join("missing-before-cancellation.mp4");
        let import_cancelled = call(
            None,
            PROJECT_IMPORT_MEDIA,
            &path_arguments(&missing),
            &AtomicBool::new(true),
        )
        .expect_err("pre-cancelled import must stop before filesystem inspection");
        assert_eq!(
            import_cancelled,
            "project_import_media was cancelled before it started"
        );

        let media = temp.path().join("clip.mp4");
        let other = temp.path().join("other.mp4");
        fs::write(&media, b"media").expect("media fixture");
        fs::write(&other, b"other").expect("other fixture");
        let approved_other = validated_media(&other);
        let changed = call_with_approved_import(
            None,
            PROJECT_IMPORT_MEDIA,
            &path_arguments(&media),
            Some(&approved_other),
            &AtomicBool::new(false),
        )
        .expect_err("post-approval canonical mismatch must stop before queueing");
        assert_eq!(
            changed,
            "project_import_media failed: the media file changed after approval; not started"
        );
        assert!(!changed.contains(temp.path().to_string_lossy().as_ref()));

        let unavailable = call(
            None,
            PROJECT_IMPORT_MEDIA,
            &path_arguments(&media),
            &AtomicBool::new(false),
        )
        .expect_err("valid import without worker");
        assert_eq!(
            unavailable,
            "project_import_media failed: the editor worker is unavailable; not started"
        );
        assert!(!unavailable.contains(temp.path().to_string_lossy().as_ref()));
    }

    #[test]
    fn live_project_mutation_classification_is_explicit() {
        assert!(!mutates_live_project(PROJECT_LIST_DRAFTS));
        assert!(mutates_live_project(PROJECT_SAVE));
        assert!(mutates_live_project(PROJECT_OPEN));
        assert!(mutates_live_project(PROJECT_IMPORT_MEDIA));
        assert!(
            mutates_live_project("project_future"),
            "future project tools must fail closed until explicitly classified as read-only"
        );
        assert!(!mutates_live_project("app_state"));
    }
}
