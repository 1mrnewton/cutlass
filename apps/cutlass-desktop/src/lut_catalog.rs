//! Cloud LUTs: the Rust half of the look inspector's LUT section.
//!
//! The worker fetches the anonymous LUT catalog (`/v1/assets/luts`),
//! downloads the `.cube` files into the configured LUT cache (eager — every
//! published entry is immediately applicable offline, and a `.cube` is
//! ~1 MB), verifies checksums, and fills a shared registry the
//! `set-clip-lut` callback reads to resolve a catalog id to a local file
//! path for `EditCommand::SetClipLut`. Entries that fail to download or
//! verify are skipped with a warning — a served asset must never brick the
//! section.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crossbeam_channel::{Sender, unbounded};
use cutlass_cloud::CloudClient;
use cutlass_storage::{CacheId, SharedStorageLayout, StorageLayoutLease};
use slint::{ComponentHandle, ModelRc, VecModel};
use tracing::warn;

use crate::{CatalogEntry, InspectorBackend};

/// Catalog id → downloaded `.cube` path, shared between the fetch worker
/// (writer) and the `set-clip-lut` callback in `main.rs` (reader).
pub type LutRegistry = Arc<Mutex<HashMap<String, PathBuf>>>;

enum Command {
    Refresh,
}

/// Cheap, cloneable sender to the LUT thread.
#[derive(Clone)]
pub struct LutHandle {
    tx: Sender<Command>,
}

impl LutHandle {
    pub fn refresh(&self) {
        let _ = self.tx.send(Command::Refresh);
    }
}

pub struct LutWorker {
    handle: LutHandle,
    _join: JoinHandle<()>,
}

impl LutWorker {
    pub fn spawn(
        backend_weak: slint::Weak<crate::AppWindow>,
        registry: LutRegistry,
        storage_layout: SharedStorageLayout,
    ) -> Result<Self, String> {
        let (tx, rx) = unbounded::<Command>();
        let join = std::thread::Builder::new()
            .name("cutlass-luts".into())
            .spawn(move || {
                let worker = Worker::new(backend_weak, registry, storage_layout);
                while let Ok(Command::Refresh) = rx.recv() {
                    worker.refresh();
                }
            })
            .map_err(|e| e.to_string())?;

        Ok(Self {
            handle: LutHandle { tx },
            _join: join,
        })
    }

    pub fn handle(&self) -> LutHandle {
        self.handle.clone()
    }
}

struct Worker {
    backend_weak: slint::Weak<crate::AppWindow>,
    registry: LutRegistry,
    storage_layout: SharedStorageLayout,
    base_url: String,
}

impl Worker {
    fn new(
        backend_weak: slint::Weak<crate::AppWindow>,
        registry: LutRegistry,
        storage_layout: SharedStorageLayout,
    ) -> Self {
        Self {
            backend_weak,
            registry,
            storage_layout,
            base_url: crate::cloud::base_url(),
        }
    }

    fn refresh(&self) {
        // Pin catalog metadata and downloaded LUTs to one generation until
        // verification and registry/UI publication both finish.
        let layout_lease = self.storage_layout.lease();
        let roots = match refresh_roots(&layout_lease) {
            Ok(roots) => roots,
            Err(message) => {
                self.set_status("error", message);
                return;
            }
        };
        self.set_status("loading", "");
        let client = CloudClient::new(&self.base_url, Some(roots.catalog));
        let entries = match client.luts() {
            Ok(catalog) => catalog.entries,
            Err(e) => {
                self.set_status("error", &user_message(&e));
                return;
            }
        };

        // Download missing (or corrupted) files; failures skip the entry,
        // never the section.
        let mut rows: Vec<(String, String)> = Vec::new();
        let mut assets: HashMap<String, PathBuf> = HashMap::new();
        for entry in &entries {
            let dest = roots.luts.join(format!("{}.cube", entry.id));
            let cached_ok = dest.is_file() && checksum_ok(&dest, &entry.checksum_sha256);
            if !cached_ok {
                let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
                if let Err(e) =
                    cutlass_cloud::download::download_to(&entry.file_url, &dest, &cancel, |_| {})
                {
                    warn!(asset = %entry.id, "LUT download skipped: {e}");
                    continue;
                }
                if !checksum_ok(&dest, &entry.checksum_sha256) {
                    warn!(asset = %entry.id, "LUT checksum mismatch, skipped");
                    let _ = std::fs::remove_file(&dest);
                    continue;
                }
            }
            assets.insert(entry.id.clone(), dest);
            rows.push((entry.id.clone(), entry.name.clone()));
        }

        {
            let mut registry = self.registry.lock().expect("LUT registry poisoned");
            *registry = assets;
        }

        let status = if rows.is_empty() { "empty" } else { "results" };
        let status = status.to_string();
        self.on_ui(move |backend| {
            let entries: Vec<CatalogEntry> = rows
                .iter()
                .map(|(id, name)| CatalogEntry {
                    id: id.as_str().into(),
                    label: name.as_str().into(),
                    has_speed: false,
                    has_intensity: false,
                    has_stagger: false,
                })
                .collect();
            backend.set_lut_catalog(ModelRc::new(VecModel::from(entries)));
            backend.set_lut_status(status.as_str().into());
            backend.set_lut_error("".into());
        });
        drop(layout_lease);
    }

    fn set_status(&self, status: &str, error: &str) {
        let status = status.to_string();
        let error = error.to_string();
        self.on_ui(move |backend| {
            backend.set_lut_status(status.as_str().into());
            backend.set_lut_error(error.as_str().into());
        });
    }

    fn on_ui(&self, f: impl FnOnce(InspectorBackend<'_>) + Send + 'static) {
        let weak = self.backend_weak.clone();
        if let Err(e) = slint::invoke_from_event_loop(move || {
            if let Some(app) = weak.upgrade() {
                f(app.global::<InspectorBackend>());
            }
        }) {
            warn!("LUT UI update failed: {e}");
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct RefreshRoots {
    catalog: PathBuf,
    luts: PathBuf,
}

fn refresh_roots(lease: &StorageLayoutLease<'_>) -> Result<RefreshRoots, &'static str> {
    Ok(RefreshRoots {
        catalog: lease
            .resolve(CacheId::Catalog)
            .ok_or("catalog cache has no disk path")?,
        luts: lease
            .resolve(CacheId::Luts)
            .ok_or("LUT cache has no disk path")?,
    })
}

/// True when the file matches the catalog checksum (or the catalog doesn't
/// carry one — first-party entries always do).
fn checksum_ok(path: &std::path::Path, expected: &str) -> bool {
    if expected.is_empty() {
        return true;
    }
    match cutlass_cloud::download::sha256_hex(path) {
        Ok(actual) => actual.eq_ignore_ascii_case(expected),
        Err(_) => false,
    }
}

fn user_message(e: &cutlass_cloud::CloudError) -> String {
    use cutlass_cloud::CloudError;
    match e {
        CloudError::Network(_) => "Couldn't reach the LUT catalog — check your connection.".into(),
        CloudError::Status {
            status, retryable, ..
        } => {
            if *retryable {
                "The LUT catalog is busy — try again in a moment.".into()
            } else {
                format!("The LUT catalog rejected the request ({status}).")
            }
        }
        CloudError::Protocol(_) => "The LUT catalog sent an unexpected response.".into(),
        CloudError::Io(_) | CloudError::Cancelled => "The catalog fetch was interrupted.".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_storage::StorageLayout;

    #[test]
    fn refresh_uses_coherent_overrides_then_picks_up_the_next_generation() {
        let dir = tempfile::tempdir().unwrap();
        let first_catalog = dir.path().join("catalog-a");
        let first_luts = dir.path().join("luts-a");
        let second_catalog = dir.path().join("catalog-b");
        let second_luts = dir.path().join("luts-b");

        let mut first = StorageLayout::new(dir.path().join("default-a")).unwrap();
        first
            .set_override(CacheId::Catalog, &first_catalog)
            .unwrap();
        first.set_override(CacheId::Luts, &first_luts).unwrap();
        let layout = SharedStorageLayout::new(first);

        let first_lease = layout.lease();
        let first_generation = first_lease.generation();
        let first_refresh = refresh_roots(&first_lease).unwrap();
        assert_eq!(
            first_refresh,
            RefreshRoots {
                catalog: first_catalog.clone(),
                luts: first_luts.clone(),
            }
        );

        let mut second = StorageLayout::new(dir.path().join("default-b")).unwrap();
        second
            .set_override(CacheId::Catalog, &second_catalog)
            .unwrap();
        second.set_override(CacheId::Luts, &second_luts).unwrap();
        drop(first_lease);
        layout.replace(first_generation, second).unwrap();

        assert_eq!(
            first_refresh,
            RefreshRoots {
                catalog: first_catalog,
                luts: first_luts,
            }
        );
        let second_lease = layout.lease();
        assert_eq!(second_lease.generation(), first_generation + 1);
        assert_eq!(
            refresh_roots(&second_lease).unwrap(),
            RefreshRoots {
                catalog: second_catalog,
                luts: second_luts,
            }
        );
    }
}
