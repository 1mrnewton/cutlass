//! Sound effects: the Rust half of `SfxBackend`.
//!
//! The worker fetches the anonymous SFX catalog (`/v1/assets/sfx`) and
//! publishes metadata tiles immediately — audio files are bigger than
//! Lottie JSON, so they download lazily on click-to-import (the stock
//! pattern in `cloud.rs`): download into the quota-managed cache, verify
//! the catalog checksum, then ride the normal import path. An imported SFX
//! is ordinary pool media under Audio > Local, draggable to audio lanes.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::thread::JoinHandle;

use crossbeam_channel::{Sender, unbounded};
use cutlass_cloud::cache::DownloadCache;
use cutlass_cloud::dto::CatalogEntry;
use cutlass_cloud::{CloudClient, download};
use cutlass_storage::{CacheId, SharedStorageLayout, StorageLayoutLease};
use slint::{ComponentHandle, Model, ModelRc, VecModel};
use tracing::{info, warn};

use crate::preview_worker::WorkerHandle;
use crate::{SfxBackend, SfxTile};

enum Command {
    Refresh,
    Import { index: usize },
}

/// Cheap, cloneable sender to the SFX thread.
#[derive(Clone)]
pub struct SfxHandle {
    tx: Sender<Command>,
}

impl SfxHandle {
    pub fn refresh(&self) {
        let _ = self.tx.send(Command::Refresh);
    }

    pub fn import(&self, index: usize) {
        let _ = self.tx.send(Command::Import { index });
    }
}

pub struct SfxWorker {
    handle: SfxHandle,
    _join: JoinHandle<()>,
}

impl SfxWorker {
    pub fn spawn(
        backend_weak: slint::Weak<crate::AppWindow>,
        import_handle: WorkerHandle,
        cache: Arc<DownloadCache>,
        storage_layout: SharedStorageLayout,
    ) -> Result<Self, String> {
        let (tx, rx) = unbounded::<Command>();
        let join = std::thread::Builder::new()
            .name("cutlass-sfx".into())
            .spawn(move || {
                let mut worker = Worker::new(backend_weak, import_handle, cache, storage_layout);
                while let Ok(command) = rx.recv() {
                    match command {
                        Command::Refresh => worker.refresh(),
                        Command::Import { index } => worker.import(index),
                    }
                }
            })
            .map_err(|e| e.to_string())?;

        Ok(Self {
            handle: SfxHandle { tx },
            _join: join,
        })
    }

    pub fn handle(&self) -> SfxHandle {
        self.handle.clone()
    }
}

struct Worker {
    backend_weak: slint::Weak<crate::AppWindow>,
    import_handle: WorkerHandle,
    cache: Arc<DownloadCache>,
    storage_layout: SharedStorageLayout,
    base_url: String,
    /// The worker-side mirror of the published tile list.
    entries: Vec<CatalogEntry>,
}

impl Worker {
    fn new(
        backend_weak: slint::Weak<crate::AppWindow>,
        import_handle: WorkerHandle,
        cache: Arc<DownloadCache>,
        storage_layout: SharedStorageLayout,
    ) -> Self {
        Self {
            backend_weak,
            import_handle,
            cache,
            storage_layout,
            base_url: crate::account::base_url(),
            entries: Vec::new(),
        }
    }

    fn refresh(&mut self) {
        let layout_lease = self.storage_layout.lease();
        let catalog_root = match catalog_root(&layout_lease) {
            Ok(root) => root,
            Err(message) => {
                self.set_status("error", message);
                return;
            }
        };
        self.set_status("loading", "");
        let client = CloudClient::new(&self.base_url, Some(catalog_root));
        let entries = match client.sfx() {
            Ok(catalog) => catalog.entries,
            Err(e) => {
                self.set_status("error", &user_message(&e));
                return;
            }
        };
        self.entries = entries;

        let seeds: Vec<TileSeed> = self.entries.iter().map(TileSeed::from).collect();
        let status = if seeds.is_empty() { "empty" } else { "results" };
        let status = status.to_string();
        self.on_ui(move |backend| {
            let rows: Vec<SfxTile> = seeds
                .iter()
                .map(|seed| SfxTile {
                    key: seed.key.as_str().into(),
                    name: seed.name.as_str().into(),
                    duration_label: seed.duration_label.as_str().into(),
                    attribution: seed.attribution.as_str().into(),
                    state: "".into(),
                    progress: 0.0,
                })
                .collect();
            backend.set_items(ModelRc::new(VecModel::from(rows)));
            backend.set_status(status.as_str().into());
            backend.set_error("".into());
        });
        drop(layout_lease);
    }

    fn import(&self, index: usize) {
        let Some(entry) = self.entries.get(index).cloned() else {
            return;
        };
        let key = entry.id.clone();
        let cache_key = format!("sfx/{}.{}", entry.id, extension(&entry.file_url));

        let lease = match self.cache.lease(&cache_key) {
            Ok(lease) => lease,
            Err(e) => {
                warn!("sfx cache path failed: {e}");
                return;
            }
        };

        // Cache hit with a valid checksum: straight to import, no download UI.
        if std::fs::symlink_metadata(lease.path())
            .is_ok_and(|metadata| metadata.file_type().is_file())
        {
            if checksum_ok(lease.path(), &entry.checksum_sha256) {
                if let Err(error) = self.cache.protect_path(lease.path()) {
                    warn!("sfx import could not protect its source: {error}");
                    self.patch_row(index, key, |tile| tile.state = "failed".into());
                    return;
                }
                self.import_handle.import(lease.path().to_path_buf());
                self.patch_row(index, key, |tile| tile.state = "imported".into());
                return;
            }
            let _ = std::fs::remove_file(lease.path());
        }
        self.patch_row(index, key.clone(), |tile| {
            tile.state = "downloading".into();
            tile.progress = 0.0;
        });

        let cancel = Arc::new(AtomicBool::new(false));
        let mut last_published = 0.0_f32;
        let progress_key = key.clone();
        let result = download::download_to(&entry.file_url, lease.path(), &cancel, |p| {
            if p.total_bytes == 0 {
                return;
            }
            let fraction = (p.bytes_downloaded as f64 / p.total_bytes as f64) as f32;
            // Patch at 5% steps, not per chunk — event-loop hops aren't free.
            if fraction - last_published >= 0.05 || fraction >= 1.0 {
                last_published = fraction;
                self.patch_row(index, progress_key.clone(), move |tile| {
                    tile.progress = fraction;
                });
            }
        });

        match result {
            Ok(()) if checksum_ok(lease.path(), &entry.checksum_sha256) => {
                info!(
                    "sfx import: {} -> {}",
                    entry.file_url,
                    lease.path().display()
                );
                if let Err(error) = self.cache.protect_path(lease.path()) {
                    warn!("sfx import could not protect its source: {error}");
                    self.patch_row(index, key, |tile| tile.state = "failed".into());
                    return;
                }
                self.import_handle.import(lease.path().to_path_buf());
                self.patch_row(index, key, |tile| tile.state = "imported".into());
                drop(lease);
                self.cache.enforce_quota();
            }
            Ok(()) => {
                warn!(asset = %entry.id, "sfx checksum mismatch");
                let _ = std::fs::remove_file(lease.path());
                self.patch_row(index, key, |tile| tile.state = "failed".into());
            }
            Err(e) => {
                warn!("sfx download failed: {e}");
                self.patch_row(index, key, |tile| tile.state = "failed".into());
            }
        }
    }

    // --- UI publishing ----------------------------------------------------

    fn set_status(&self, status: &str, error: &str) {
        let status = status.to_string();
        let error = error.to_string();
        self.on_ui(move |backend| {
            backend.set_status(status.as_str().into());
            backend.set_error(error.as_str().into());
        });
    }

    /// Patch one row's field in place, guarded by key so a stale patch
    /// (list replaced mid-flight) can't hit the wrong tile.
    fn patch_row(&self, row: usize, key: String, patch: impl Fn(&mut SfxTile) + Send + 'static) {
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

    fn on_ui(&self, f: impl FnOnce(SfxBackend<'_>) + Send + 'static) {
        let weak = self.backend_weak.clone();
        if let Err(e) = slint::invoke_from_event_loop(move || {
            if let Some(app) = weak.upgrade() {
                f(app.global::<SfxBackend>());
            }
        }) {
            warn!("sfx UI update failed: {e}");
        }
    }
}

fn catalog_root(lease: &StorageLayoutLease<'_>) -> Result<std::path::PathBuf, &'static str> {
    lease
        .resolve(CacheId::Catalog)
        .ok_or("catalog cache has no disk path")
}

/// Send-safe snapshot of a tile (built worker-side).
struct TileSeed {
    key: String,
    name: String,
    duration_label: String,
    attribution: String,
}

impl From<&CatalogEntry> for TileSeed {
    fn from(entry: &CatalogEntry) -> Self {
        Self {
            key: entry.id.clone(),
            name: entry.name.clone(),
            duration_label: duration_label(entry.duration_seconds),
            attribution: attribution(entry),
        }
    }
}

/// Corner chip: "0:03".
fn duration_label(seconds: Option<f64>) -> String {
    let Some(seconds) = seconds else {
        return String::new();
    };
    let total = seconds.round() as i64;
    format!("{}:{:02}", total / 60, total % 60)
}

/// Provenance credit ("CC0 · author"), the pack manifests' license-first
/// promise surfaced in the tile hover strip.
fn attribution(entry: &CatalogEntry) -> String {
    match (entry.license.is_empty(), entry.author.is_empty()) {
        (false, false) => format!("{} · {}", entry.license, entry.author),
        (false, true) => entry.license.clone(),
        (true, false) => entry.author.clone(),
        (true, true) => String::new(),
    }
}

/// Best-effort extension from the CDN URL (the decoder sniffs content
/// anyway; this only keeps cache filenames and import probing sensible).
fn extension(url: &str) -> String {
    let path_part = url.split(['?', '#']).next().unwrap_or("");
    if let Some((_, ext)) = path_part.rsplit_once('.')
        && !ext.is_empty()
        && ext.len() <= 4
        && ext.chars().all(|c| c.is_ascii_alphanumeric())
    {
        return ext.to_lowercase();
    }
    "wav".into()
}

/// True when the file matches the catalog checksum (or the catalog doesn't
/// carry one — first-party entries always do).
fn checksum_ok(path: &std::path::Path, expected: &str) -> bool {
    if expected.is_empty() {
        return true;
    }
    match download::sha256_hex(path) {
        Ok(actual) => actual.eq_ignore_ascii_case(expected),
        Err(_) => false,
    }
}

fn user_message(e: &cutlass_cloud::CloudError) -> String {
    use cutlass_cloud::CloudError;
    match e {
        CloudError::Network(_) => "Couldn't reach the SFX catalog — check your connection.".into(),
        CloudError::Status {
            status, retryable, ..
        } => {
            if *retryable {
                "The SFX catalog is busy — try again in a moment.".into()
            } else {
                format!("The SFX catalog rejected the request ({status}).")
            }
        }
        CloudError::Protocol(_) => "The SFX catalog sent an unexpected response.".into(),
        CloudError::Io(_) | CloudError::Cancelled => "The catalog fetch was interrupted.".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_storage::StorageLayout;

    #[test]
    fn duration_chip_formats_minutes_seconds() {
        assert_eq!(duration_label(Some(3.4)), "0:03");
        assert_eq!(duration_label(Some(72.6)), "1:13");
        assert_eq!(duration_label(None), "");
    }

    #[test]
    fn attribution_joins_license_and_author() {
        let mut entry = CatalogEntry {
            id: "sfx-1".into(),
            kind: cutlass_cloud::dto::AssetKind::Sfx,
            name: "Whoosh".into(),
            category: String::new(),
            tags: vec![],
            file_url: "https://cdn/sfx-1.wav".into(),
            preview_url: None,
            size_bytes: 0,
            checksum_sha256: String::new(),
            min_schema_version: None,
            author: "freesound.org/nick".into(),
            license: "CC0".into(),
            duration_seconds: None,
            slot_count: None,
        };
        assert_eq!(attribution(&entry), "CC0 · freesound.org/nick");
        entry.author.clear();
        assert_eq!(attribution(&entry), "CC0");
    }

    #[test]
    fn extension_from_url_then_default() {
        assert_eq!(extension("https://c/x.WAV?dl=1"), "wav");
        assert_eq!(extension("https://c/x.mp3"), "mp3");
        assert_eq!(extension("https://c/x"), "wav");
    }

    #[test]
    fn refresh_uses_catalog_override_then_picks_up_the_next_generation() {
        let dir = tempfile::tempdir().unwrap();
        let first_root = dir.path().join("catalog-a");
        let second_root = dir.path().join("catalog-b");

        let mut first = StorageLayout::new(dir.path().join("default-a")).unwrap();
        first.set_override(CacheId::Catalog, &first_root).unwrap();
        let layout = SharedStorageLayout::new(first);

        let first_lease = layout.lease();
        let first_generation = first_lease.generation();
        let first_refresh = catalog_root(&first_lease).unwrap();
        assert_eq!(first_refresh, first_root);

        let mut second = StorageLayout::new(dir.path().join("default-b")).unwrap();
        second.set_override(CacheId::Catalog, &second_root).unwrap();
        drop(first_lease);
        layout.replace(first_generation, second).unwrap();

        assert_eq!(first_refresh, first_root);
        let second_lease = layout.lease();
        assert_eq!(second_lease.generation(), first_generation + 1);
        assert_eq!(catalog_root(&second_lease).unwrap(), second_root);
    }
}
