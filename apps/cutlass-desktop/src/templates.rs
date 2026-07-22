//! Launch-screen templates gallery: the Rust half of `TemplatesBackend`.
//!
//! One worker thread owns all template HTTP (catalog fetches, preview
//! image fetches, bundle downloads) — network never touches the UI or
//! engine threads. The catalog is anonymous and comes from the Cutlass
//! backend; bundles download **directly from the asset CDN** into the
//! quota-managed download cache, install into a per-template folder in the
//! configured templates cache (`template_bundle::install` rewrites media
//! paths to absolute — install once, open forever), and fill through the
//! engine's `ApplyTemplate` after the user picks their media.
//!
//! Threading mirrors `cloud.rs`: commands in over a channel, results
//! hopped to the UI thread with `invoke_from_event_loop`, model rows
//! patched in place. The pick dialog is the one step that must run on
//! the UI thread (`rfd` + `slint::spawn_local`); the worker hands off to
//! it after install and the continuation sends `apply_template` to the
//! engine worker.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::thread::JoinHandle;

use crossbeam_channel::{Sender, unbounded};
use cutlass_cloud::cache::{DownloadCache, DownloadCacheLease};
use cutlass_cloud::dto::CatalogEntry;
use cutlass_cloud::{CloudClient, download};
use cutlass_commands::TemplatePick;
use cutlass_models::{PROJECT_SCHEMA_VERSION, template_bundle};
use cutlass_storage::{CacheId, SharedStorageLayout, StorageLayoutLease};
use slint::{ComponentHandle, Image, Model, ModelRc, Rgba8Pixel, SharedPixelBuffer, VecModel};
use tracing::{info, warn};

use crate::preview_worker::WorkerHandle;
use crate::{TemplateTile, TemplatesBackend};

enum Command {
    Refresh { category: String },
    Use { index: usize },
}

/// Cheap, cloneable sender to the templates thread.
#[derive(Clone)]
pub struct TemplatesHandle {
    tx: Sender<Command>,
}

impl TemplatesHandle {
    pub fn refresh(&self, category: String) {
        let _ = self.tx.send(Command::Refresh { category });
    }

    pub fn use_template(&self, index: usize) {
        let _ = self.tx.send(Command::Use { index });
    }
}

pub struct TemplatesWorker {
    handle: TemplatesHandle,
    _join: JoinHandle<()>,
}

impl TemplatesWorker {
    pub fn spawn(
        backend_weak: slint::Weak<crate::AppWindow>,
        preview_handle: WorkerHandle,
        cache: Arc<DownloadCache>,
        storage_layout: SharedStorageLayout,
    ) -> Result<Self, String> {
        let (tx, rx) = unbounded::<Command>();
        let join = std::thread::Builder::new()
            .name("cutlass-templates".into())
            .spawn(move || {
                let mut worker = Worker::new(backend_weak, preview_handle, cache, storage_layout);
                // Preview fetches interleave with commands: commands take
                // priority (a queued "use" shouldn't wait for previews),
                // previews drain in the gaps.
                let mut previews: VecDeque<(usize, CatalogEntry)> = VecDeque::new();
                loop {
                    let command = if previews.is_empty() {
                        match rx.recv() {
                            Ok(c) => Some(c),
                            Err(_) => return,
                        }
                    } else {
                        rx.try_recv().ok()
                    };
                    match command {
                        Some(c) => worker.run(c, &mut previews),
                        None => {
                            if let Some((row, entry)) = previews.pop_front() {
                                worker.fetch_preview(row, &entry);
                            }
                        }
                    }
                }
            })
            .map_err(|e| e.to_string())?;

        Ok(Self {
            handle: TemplatesHandle { tx },
            _join: join,
        })
    }

    pub fn handle(&self) -> TemplatesHandle {
        self.handle.clone()
    }
}

struct Worker {
    backend_weak: slint::Weak<crate::AppWindow>,
    preview_handle: WorkerHandle,
    cache: Arc<DownloadCache>,
    storage_layout: SharedStorageLayout,
    base_url: String,
    /// The worker-side mirror of the published (filtered) tile list.
    entries: Vec<CatalogEntry>,
}

impl Worker {
    fn new(
        backend_weak: slint::Weak<crate::AppWindow>,
        preview_handle: WorkerHandle,
        cache: Arc<DownloadCache>,
        storage_layout: SharedStorageLayout,
    ) -> Self {
        Self {
            backend_weak,
            preview_handle,
            cache,
            storage_layout,
            base_url: crate::cloud::base_url(),
            entries: Vec::new(),
        }
    }

    fn run(&mut self, command: Command, previews: &mut VecDeque<(usize, CatalogEntry)>) {
        match command {
            Command::Refresh { category } => {
                previews.clear();
                let layout_lease = self.storage_layout.lease();
                let catalog_root = match catalog_root(&layout_lease) {
                    Ok(root) => root,
                    Err(message) => {
                        self.entries.clear();
                        self.set_status("error", message);
                        return;
                    }
                };
                self.set_status("loading", "");
                let client = CloudClient::new(&self.base_url, Some(catalog_root));
                match client.templates() {
                    Ok(catalog) => {
                        self.entries = catalog
                            .entries
                            .into_iter()
                            .filter(|e| category == "all" || e.category == category)
                            .collect();
                        let status = if self.entries.is_empty() {
                            "empty"
                        } else {
                            "results"
                        };
                        self.publish_all(status);
                        for (row, entry) in self.entries.iter().enumerate() {
                            if entry.preview_url.is_some() {
                                previews.push_back((row, entry.clone()));
                            }
                        }
                    }
                    Err(e) => {
                        self.entries.clear();
                        self.set_status("error", &user_message(&e));
                    }
                }
                drop(layout_lease);
            }
            Command::Use { index } => self.use_template(index),
        }
    }

    // --- UI publishing ----------------------------------------------------

    fn set_status(&self, status: &str, error: &str) {
        let status = status.to_string();
        let error = error.to_string();
        self.on_ui(move |backend| {
            backend.set_status(status.as_str().into());
            backend.set_error(error.as_str().into());
            if status == "loading" {
                backend.set_items(ModelRc::default());
            }
        });
    }

    /// Replace the whole tile list, preserving already-fetched previews for
    /// rows that survive (category flips re-list the same entries).
    fn publish_all(&self, status: &str) {
        let seeds: Vec<TileSeed> = self.entries.iter().map(TileSeed::from).collect();
        let status = status.to_string();
        self.on_ui(move |backend| {
            let existing = backend.get_items();
            let rows: Vec<TemplateTile> = seeds
                .iter()
                .map(|seed| {
                    let kept = existing
                        .iter()
                        .find(|t| t.key == seed.key.as_str())
                        .filter(|t| t.has_preview);
                    TemplateTile {
                        key: seed.key.as_str().into(),
                        name: seed.name.as_str().into(),
                        category: seed.category.as_str().into(),
                        preview: kept.as_ref().map(|t| t.preview.clone()).unwrap_or_default(),
                        has_preview: kept.is_some(),
                        label: seed.label.as_str().into(),
                        author: seed.author.as_str().into(),
                        state: if seed.unsupported {
                            "unsupported".into()
                        } else {
                            Default::default()
                        },
                        progress: 0.0,
                    }
                })
                .collect();
            backend.set_items(ModelRc::new(VecModel::from(rows)));
            backend.set_status(status.as_str().into());
        });
    }

    /// Patch one row's fields in place, guarded by key so a stale patch
    /// (list replaced mid-flight) can't hit the wrong tile.
    fn patch_row(
        &self,
        row: usize,
        key: String,
        patch: impl Fn(&mut TemplateTile) + Send + 'static,
    ) {
        self.on_ui(move |backend| {
            let model = backend.get_items();
            if let Some(mut tile) = model.row_data(row)
                && tile.key == key.as_str()
            {
                patch(&mut tile);
                model.set_row_data(row, tile);
            }
        });
    }

    fn on_ui(&self, f: impl FnOnce(TemplatesBackend<'_>) + Send + 'static) {
        let weak = self.backend_weak.clone();
        if let Err(e) = slint::invoke_from_event_loop(move || {
            if let Some(app) = weak.upgrade() {
                f(app.global::<TemplatesBackend>());
            }
        }) {
            warn!("templates UI update failed: {e}");
        }
    }

    // --- previews -----------------------------------------------------------

    fn fetch_preview(&self, row: usize, entry: &CatalogEntry) {
        let Some(url) = entry.preview_url.as_deref().filter(|u| !u.is_empty()) else {
            return;
        };
        let key = entry.id.clone();
        let lease = match self
            .cache
            .lease(&format!("template-previews/{}.img", safe_id(&entry.id)))
        {
            Ok(lease) => lease,
            Err(e) => {
                warn!("template preview cache path failed: {e}");
                return;
            }
        };
        let path = lease.path();
        if !path.is_file() {
            let cancel = Arc::new(AtomicBool::new(false));
            if let Err(e) = download::download_to(url, path, &cancel, |_| {}) {
                warn!("template preview download failed: {e}");
                return;
            }
        }
        let Ok(bytes) = std::fs::read(path) else {
            return;
        };
        drop(lease);
        let Ok(decoded) = cutlass_decoder::decode_image_bytes(&bytes) else {
            warn!("template preview decode failed for {key}");
            return;
        };
        let (width, height, pixels) = (decoded.width, decoded.height, decoded.pixels);
        self.patch_row(row, key, move |tile| {
            let buffer = SharedPixelBuffer::<Rgba8Pixel>::clone_from_slice(&pixels, width, height);
            tile.preview = Image::from_rgba8(buffer);
            tile.has_preview = true;
        });
    }

    // --- use: download → install → pick → apply ------------------------------

    fn use_template(&self, index: usize) {
        let Some(entry) = self.entries.get(index).cloned() else {
            return;
        };
        let key = entry.id.clone();

        // Older app, newer template: refuse gracefully (also pre-marked at
        // publish time; this guards a race where the catalog re-published).
        if entry.min_schema_version.unwrap_or(0) > PROJECT_SCHEMA_VERSION {
            self.patch_row(index, key, |tile| tile.state = "unsupported".into());
            return;
        }

        // Acquire before any DownloadCache lease and keep it through all
        // pre-picker install/load work. The picker handoff carries no path.
        let layout_lease = self.storage_layout.lease();
        let template_id = InstalledTemplateId::from_catalog_id(&entry.id);
        let (install_dir, template_path) =
            match installed_template_paths(&layout_lease, &template_id) {
                Ok(paths) => paths,
                Err(message) => {
                    warn!("{message}");
                    self.patch_row(index, key, |tile| tile.state = "failed".into());
                    return;
                }
            };

        // Install once, open forever: skip download + install when present.
        let template = if template_path.is_file() {
            cutlass_models::Template::load_from_file(&template_path)
        } else {
            match self.download_bundle(index, &entry) {
                Ok((bundle, downloaded)) => {
                    self.patch_row(index, key.clone(), |tile| tile.state = "installing".into());
                    let template = template_bundle::install(bundle.path(), &install_dir);
                    drop(bundle);
                    if downloaded {
                        self.cache.enforce_quota();
                    }
                    template
                }
                Err(message) => {
                    warn!("template bundle download failed: {message}");
                    self.patch_row(index, key, |tile| tile.state = "failed".into());
                    return;
                }
            }
        };
        let template = match template {
            Ok(template) => template,
            Err(e) => {
                warn!("template install failed for {}: {e}", entry.id);
                self.patch_row(index, key, |tile| tile.state = "failed".into());
                return;
            }
        };

        info!(
            id = %entry.id,
            slots = template.slot_count(),
            "template installed; asking for picks"
        );
        self.patch_row(index, key.clone(), |tile| tile.state = "picking".into());
        self.run_pick_flow(index, key, template_id, template.slot_count());
        drop(layout_lease);
    }

    /// Download the bundle into the download cache, with tile progress and
    /// a checksum check when the catalog carries one.
    fn download_bundle(
        &self,
        index: usize,
        entry: &CatalogEntry,
    ) -> Result<(DownloadCacheLease<'_>, bool), String> {
        let cache_key = format!("templates/{}.cutlassb", safe_id(&entry.id));
        let bundle = self.cache.lease(&cache_key).map_err(|e| e.to_string())?;
        if self.cache.hit(&cache_key).is_some() {
            return Ok((bundle, false));
        }
        let dest = bundle.path();
        let key = entry.id.clone();
        self.patch_row(index, key.clone(), |tile| {
            tile.state = "downloading".into();
            tile.progress = 0.0;
        });

        let cancel = Arc::new(AtomicBool::new(false));
        let mut last_published = 0.0_f32;
        download::download_to(&entry.file_url, dest, &cancel, |p| {
            if p.total_bytes == 0 {
                return;
            }
            let fraction = (p.bytes_downloaded as f64 / p.total_bytes as f64) as f32;
            if fraction - last_published >= 0.05 || fraction >= 1.0 {
                last_published = fraction;
                self.patch_row(index, key.clone(), move |tile| tile.progress = fraction);
            }
        })
        .map_err(|e| e.to_string())?;

        if !entry.checksum_sha256.is_empty() {
            let actual = download::sha256_hex(dest).map_err(|e| e.to_string())?;
            if !actual.eq_ignore_ascii_case(&entry.checksum_sha256) {
                let _ = std::fs::remove_file(dest);
                return Err(format!(
                    "checksum mismatch for {} (expected {}, got {actual})",
                    entry.id, entry.checksum_sha256
                ));
            }
        }
        Ok((bundle, true))
    }

    /// Hop to the UI thread for the media-pick dialog, then hand the filled
    /// template to the engine worker. Fewer picks than slots is fine (the
    /// remaining slots keep their sample media, exactly like CapCut); extra
    /// picks are truncated. Cancelling the dialog just resets the tile.
    fn run_pick_flow(
        &self,
        row: usize,
        key: String,
        template_id: InstalledTemplateId,
        slots: usize,
    ) {
        let weak = self.backend_weak.clone();
        let preview_handle = self.preview_handle.clone();
        let storage_layout = self.storage_layout.clone();
        if let Err(e) = slint::invoke_from_event_loop(move || {
            let task = slint::spawn_local(async move {
                let title = if slots == 1 {
                    "Choose a clip for the template's slot".to_string()
                } else {
                    format!("Choose up to {slots} clips, in slot order")
                };
                let files = rfd::AsyncFileDialog::new()
                    .set_title(&title)
                    .add_filter(
                        "Media",
                        &[
                            "mp4", "mov", "mkv", "webm", "m4v", "png", "jpg", "jpeg", "webp",
                        ],
                    )
                    .pick_files()
                    .await;

                // Reset the tile either way: on success the session epoch
                // bump swaps to the editor; on cancel the card re-arms.
                if let Some(app) = weak.upgrade() {
                    let model = app.global::<TemplatesBackend>().get_items();
                    if let Some(mut tile) = model.row_data(row)
                        && tile.key == key.as_str()
                    {
                        tile.state = Default::default();
                        model.set_row_data(row, tile);
                    }
                }

                let Some(files) = files else {
                    return;
                };
                let picks: Vec<TemplatePick> = files
                    .iter()
                    .take(slots)
                    .map(|f| TemplatePick {
                        path: f.path().to_path_buf(),
                        source_in: None,
                    })
                    .collect();
                // Resolving can wait behind a long relocation, so do it off
                // the UI thread. The lease pins the post-relocation path
                // through both ordered preview-worker queue sends.
                if let Err(message) =
                    spawn_template_apply(storage_layout, preview_handle, template_id, picks)
                {
                    warn!("{message}");
                }
            });
            if let Err(e) = task {
                tracing::error!("failed to open template pick dialog: {e}");
            }
        }) {
            warn!("template pick flow failed to reach the UI thread: {e}");
        }
    }
}

const INSTALLED_TEMPLATE_FILE: &str = "template.cutlasst";
const MAX_UNTRUSTED_TEMPLATE_ID_BYTES: usize = 128;

/// Stable, traversal-safe directory name carried across the asynchronous
/// picker. Absolute cache paths are always regenerated from an operation
/// lease after the picker returns.
#[derive(Clone, Debug, PartialEq, Eq)]
struct InstalledTemplateId(String);

impl InstalledTemplateId {
    fn from_catalog_id(id: &str) -> Self {
        Self(safe_id(id))
    }

    fn from_untrusted(id: &str) -> Result<Self, &'static str> {
        // Check the byte bound before sanitizing allocates or any caller joins
        // the value into a filesystem path.
        if id.is_empty() {
            return Err("template id must not be empty");
        }
        if id.len() > MAX_UNTRUSTED_TEMPLATE_ID_BYTES {
            return Err("template id is too long");
        }
        if id == "unnamed" {
            return Err("template id must not be the sanitizer fallback");
        }
        if id.bytes().any(|byte| byte.is_ascii_uppercase()) {
            return Err("template id must be lowercase");
        }

        let sanitized = safe_id(id);
        if sanitized != id {
            return Err("template id must be an exact canonical safe id");
        }
        Ok(Self(sanitized))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

fn installed_template_paths(
    lease: &StorageLayoutLease<'_>,
    template_id: &InstalledTemplateId,
) -> Result<(PathBuf, PathBuf), &'static str> {
    let install_dir = templates_root(lease)?.join(template_id.as_str());
    let template_path = install_dir.join(INSTALLED_TEMPLATE_FILE);
    Ok((install_dir, template_path))
}

/// Resolve an already-installed canonical template id in the leased layout.
///
/// This is deliberately resolution-only: it never creates, downloads, or
/// installs anything. The caller must retain `lease` through the acknowledged
/// worker RPC that consumes the returned path so cache relocation cannot start
/// between resolution and use. Filesystem entries are inspected and
/// re-inspected here, but a path-only handoff cannot pin them against hostile
/// replacement after this function returns.
#[allow(dead_code)] // Phase 2c foundation; consumed by agent project RPCs next.
pub(crate) fn resolve_installed_template_path(
    lease: &StorageLayoutLease<'_>,
    template_id: &str,
) -> Result<PathBuf, String> {
    let template_id = InstalledTemplateId::from_untrusted(template_id).map_err(str::to_owned)?;
    let root = templates_root(lease).map_err(str::to_owned)?;
    if !root.is_absolute() {
        return Err("template cache path is not absolute".into());
    }

    let (install_dir, template_path) =
        installed_template_paths(lease, &template_id).map_err(str::to_owned)?;
    if install_dir.parent() != Some(root.as_path())
        || template_path.parent() != Some(install_dir.as_path())
    {
        return Err("template path escaped the template cache".into());
    }

    let root_entry =
        inspect_installed_entry(&root, "template cache", InstalledEntryKind::Directory)?;
    let install_entry = inspect_installed_entry(
        &install_dir,
        "template install directory",
        InstalledEntryKind::Directory,
    )?;
    if install_entry.canonical != root_entry.canonical.join(template_id.as_str()) {
        return Err("template install directory is not the exact cache child".into());
    }

    let template_entry = inspect_installed_entry(
        &template_path,
        "installed template file",
        InstalledEntryKind::File,
    )?;
    if template_entry.canonical != install_entry.canonical.join(INSTALLED_TEMPLATE_FILE) {
        return Err("installed template file is outside its install directory".into());
    }

    // Repeat the point-in-time inspection before publishing the path. Identity
    // comparison catches ordinary remove/replace races, while canonical path
    // comparison catches parent remapping and filesystem aliases.
    ensure_installed_entry_unchanged(
        &root,
        "template cache",
        InstalledEntryKind::Directory,
        &root_entry,
    )?;
    ensure_installed_entry_unchanged(
        &install_dir,
        "template install directory",
        InstalledEntryKind::Directory,
        &install_entry,
    )?;
    ensure_installed_entry_unchanged(
        &template_path,
        "installed template file",
        InstalledEntryKind::File,
        &template_entry,
    )?;
    ensure_installed_entry_unchanged(
        &root,
        "template cache",
        InstalledEntryKind::Directory,
        &root_entry,
    )?;

    Ok(template_path)
}

#[derive(Clone, Copy)]
enum InstalledEntryKind {
    Directory,
    File,
}

impl InstalledEntryKind {
    fn matches(self, metadata: &std::fs::Metadata) -> bool {
        match self {
            Self::Directory => metadata.file_type().is_dir(),
            Self::File => metadata.file_type().is_file(),
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::Directory => "directory",
            Self::File => "regular file",
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct InspectedInstalledEntry {
    canonical: PathBuf,
    identity: InstalledEntryIdentity,
}

fn inspect_installed_entry(
    path: &Path,
    role: &str,
    kind: InstalledEntryKind,
) -> Result<InspectedInstalledEntry, String> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| format!("could not inspect {role} at {}: {error}", path.display()))?;
    ensure_installed_entry_type(path, role, kind, &metadata)?;

    let canonical = std::fs::canonicalize(path)
        .map_err(|error| format!("could not resolve {role} at {}: {error}", path.display()))?;
    let canonical_metadata = std::fs::symlink_metadata(&canonical).map_err(|error| {
        format!(
            "could not reinspect resolved {role} at {}: {error}",
            canonical.display()
        )
    })?;
    ensure_installed_entry_type(&canonical, role, kind, &canonical_metadata)?;

    let identity = installed_entry_identity(&metadata);
    if identity != installed_entry_identity(&canonical_metadata) {
        return Err(format!(
            "{role} changed while it was inspected at {}",
            path.display()
        ));
    }

    Ok(InspectedInstalledEntry {
        canonical,
        identity,
    })
}

fn ensure_installed_entry_type(
    path: &Path,
    role: &str,
    kind: InstalledEntryKind,
    metadata: &std::fs::Metadata,
) -> Result<(), String> {
    if metadata.file_type().is_symlink() || metadata_is_reparse_point(metadata) {
        return Err(format!(
            "{role} must not be a symlink or reparse point: {}",
            path.display()
        ));
    }
    if !kind.matches(metadata) {
        return Err(format!(
            "{role} is not a real {}: {}",
            kind.description(),
            path.display()
        ));
    }
    Ok(())
}

fn ensure_installed_entry_unchanged(
    path: &Path,
    role: &str,
    kind: InstalledEntryKind,
    expected: &InspectedInstalledEntry,
) -> Result<(), String> {
    let current = inspect_installed_entry(path, role, kind)?;
    if current != *expected {
        return Err(format!(
            "{role} changed during template resolution at {}",
            path.display()
        ));
    }
    Ok(())
}

#[cfg(unix)]
#[derive(Debug, PartialEq, Eq)]
struct InstalledEntryIdentity {
    device: u64,
    inode: u64,
}

#[cfg(unix)]
fn installed_entry_identity(metadata: &std::fs::Metadata) -> InstalledEntryIdentity {
    use std::os::unix::fs::MetadataExt as _;

    InstalledEntryIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
    }
}

// `volume_serial_number()`/`file_index()` are still unstable
// (`windows_by_handle`), so approximate identity with stable metadata.
#[cfg(windows)]
#[derive(Debug, PartialEq, Eq)]
struct InstalledEntryIdentity {
    creation_time: u64,
    last_write_time: u64,
    file_size: u64,
    file_attributes: u32,
}

#[cfg(windows)]
fn installed_entry_identity(metadata: &std::fs::Metadata) -> InstalledEntryIdentity {
    use std::os::windows::fs::MetadataExt as _;

    InstalledEntryIdentity {
        creation_time: metadata.creation_time(),
        last_write_time: metadata.last_write_time(),
        file_size: metadata.file_size(),
        file_attributes: metadata.file_attributes(),
    }
}

#[cfg(not(any(unix, windows)))]
#[derive(Debug, PartialEq, Eq)]
struct InstalledEntryIdentity {
    len: u64,
    modified: Option<std::time::SystemTime>,
}

#[cfg(not(any(unix, windows)))]
fn installed_entry_identity(metadata: &std::fs::Metadata) -> InstalledEntryIdentity {
    InstalledEntryIdentity {
        len: metadata.len(),
        modified: metadata.modified().ok(),
    }
}

#[cfg(windows)]
const WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;

#[cfg(windows)]
fn windows_attributes_are_reparse_point(attributes: u32) -> bool {
    attributes & WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT != 0
}

#[cfg(windows)]
fn metadata_is_reparse_point(metadata: &std::fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt as _;

    windows_attributes_are_reparse_point(metadata.file_attributes())
}

#[cfg(not(windows))]
fn metadata_is_reparse_point(_metadata: &std::fs::Metadata) -> bool {
    false
}

/// Resolve and enqueue after the picker without ever waiting on the Slint UI
/// thread. Queue order is save → apply; a relocation that acquires exclusivity
/// after this lease can only enqueue project maintenance behind both.
fn spawn_template_apply(
    storage_layout: SharedStorageLayout,
    preview_handle: WorkerHandle,
    template_id: InstalledTemplateId,
    picks: Vec<TemplatePick>,
) -> Result<(), &'static str> {
    std::thread::Builder::new()
        .name("cutlass-template-apply".into())
        .spawn(move || {
            let layout_lease = storage_layout.lease();
            let (_, template_path) = match installed_template_paths(&layout_lease, &template_id) {
                Ok(paths) => paths,
                Err(message) => {
                    warn!("{message}");
                    return;
                }
            };

            // Flush the outgoing draft (a no-op from the launch screen), then
            // swap the session. Keep the lease through both channel sends.
            preview_handle.save_project(None);
            preview_handle.apply_template(template_path, picks);
            drop(layout_lease);
        })
        .map(|_| ())
        .map_err(|_| "template apply worker could not start")
}

fn catalog_root(lease: &StorageLayoutLease<'_>) -> Result<PathBuf, &'static str> {
    lease
        .resolve(CacheId::Catalog)
        .ok_or("catalog cache has no disk path")
}

fn templates_root(lease: &StorageLayoutLease<'_>) -> Result<PathBuf, &'static str> {
    lease
        .resolve(CacheId::Templates)
        .ok_or("template cache has no disk path")
}

/// Send-safe snapshot of a `CatalogEntry`'s tile fields (built worker-side;
/// `TemplateTile` itself holds an `Image` and is UI-thread only).
struct TileSeed {
    key: String,
    name: String,
    category: String,
    label: String,
    author: String,
    unsupported: bool,
}

impl From<&CatalogEntry> for TileSeed {
    fn from(entry: &CatalogEntry) -> Self {
        Self {
            key: entry.id.clone(),
            name: entry.name.clone(),
            category: entry.category.clone(),
            label: tile_label(entry),
            author: entry.author.clone(),
            unsupported: entry.min_schema_version.unwrap_or(0) > PROJECT_SCHEMA_VERSION,
        }
    }
}

/// Corner chip: "0:12 · 3 clips".
fn tile_label(entry: &CatalogEntry) -> String {
    let mut parts = Vec::new();
    if let Some(seconds) = entry.duration_seconds {
        let total = seconds.round() as i64;
        parts.push(format!("{}:{:02}", total / 60, total % 60));
    }
    match entry.slot_count {
        Some(1) => parts.push("1 clip".into()),
        Some(n) if n > 1 => parts.push(format!("{n} clips")),
        _ => {}
    }
    parts.join(" · ")
}

/// Catalog ids come from a remote catalog and end up in filesystem paths;
/// keep only characters that can't traverse or surprise.
fn safe_id(id: &str) -> String {
    let cleaned: String = id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        .collect();
    let cleaned = cleaned.trim_matches('.').to_string();
    if cleaned.is_empty() {
        "unnamed".into()
    } else {
        cleaned
    }
}

fn user_message(e: &cutlass_cloud::CloudError) -> String {
    use cutlass_cloud::CloudError;
    match e {
        CloudError::Network(_) => {
            "Couldn't reach the template catalog — check your connection.".into()
        }
        CloudError::Status {
            status, retryable, ..
        } => {
            if *retryable {
                "The template catalog is busy — try again in a moment.".into()
            } else {
                format!("The template catalog rejected the request ({status}).")
            }
        }
        CloudError::Protocol(_) => "The template catalog sent an unexpected response.".into(),
        CloudError::Io(_) | CloudError::Cancelled => "The catalog fetch was interrupted.".into(),
    }
}

#[cfg(test)]
mod tests;
