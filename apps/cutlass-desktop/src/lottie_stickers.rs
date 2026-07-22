//! Lottie stickers: the Rust half of `LottieBackend`.
//!
//! The worker fetches the anonymous Lottie catalog (`/v1/assets/lottie`),
//! downloads the small `.json` files into the configured Lottie cache
//! (eager — every published tile is immediately droppable offline), probes each
//! composition with the real decoder, rasterizes frame 0 as the tile
//! thumbnail, and fills a shared registry the timeline drop resolver reads
//! to place file-backed `Generator::Lottie` clips. Files that fail to
//! download or parse are skipped with a warning — a served asset must
//! never brick the section (`docs/lottie-design.md`).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crossbeam_channel::{Sender, unbounded};
use cutlass_cloud::CloudClient;
use cutlass_storage::{CacheId, SharedStorageLayout, StorageLayoutLease};
use slint::{ComponentHandle, ModelRc, VecModel};
use tracing::warn;

use crate::{LottieBackend, LottieTile};

/// A downloaded, droppable Lottie asset: what a `lottie:<id>` drop places.
#[derive(Clone)]
pub struct LottieAsset {
    pub path: PathBuf,
    /// Intrinsic composition size (reference pixels for placement).
    pub width: u32,
    pub height: u32,
}

/// Catalog id → asset, shared between the fetch worker (writer) and the
/// timeline drop resolver in `main.rs` (reader).
pub type LottieRegistry = Arc<Mutex<HashMap<String, LottieAsset>>>;

enum Command {
    Refresh,
}

/// Cheap, cloneable sender to the Lottie thread.
#[derive(Clone)]
pub struct LottieHandle {
    tx: Sender<Command>,
}

impl LottieHandle {
    pub fn refresh(&self) {
        let _ = self.tx.send(Command::Refresh);
    }
}

pub struct LottieWorker {
    handle: LottieHandle,
    _join: JoinHandle<()>,
}

impl LottieWorker {
    pub fn spawn(
        backend_weak: slint::Weak<crate::AppWindow>,
        registry: LottieRegistry,
        storage_layout: SharedStorageLayout,
    ) -> Result<Self, String> {
        let (tx, rx) = unbounded::<Command>();
        let join = std::thread::Builder::new()
            .name("cutlass-lottie".into())
            .spawn(move || {
                let worker = Worker::new(backend_weak, registry, storage_layout);
                while let Ok(Command::Refresh) = rx.recv() {
                    worker.refresh();
                }
            })
            .map_err(|e| e.to_string())?;

        Ok(Self {
            handle: LottieHandle { tx },
            _join: join,
        })
    }

    pub fn handle(&self) -> LottieHandle {
        self.handle.clone()
    }
}

struct Worker {
    backend_weak: slint::Weak<crate::AppWindow>,
    registry: LottieRegistry,
    storage_layout: SharedStorageLayout,
    base_url: String,
}

impl Worker {
    fn new(
        backend_weak: slint::Weak<crate::AppWindow>,
        registry: LottieRegistry,
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
        // Pin catalog metadata and downloaded animations until parsing,
        // thumbnail rendering, and registry/UI publication all finish.
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
        let entries = match client.lottie() {
            Ok(catalog) => catalog.entries,
            Err(e) => {
                self.set_status("error", &user_message(&e));
                return;
            }
        };

        // Download missing files and probe each with the real decoder;
        // failures skip the entry, never the section.
        let mut seeds: Vec<TileSeed> = Vec::new();
        let mut assets: HashMap<String, LottieAsset> = HashMap::new();
        for entry in &entries {
            let dest = roots.lottie.join(format!("{}.json", entry.id));
            if !dest.is_file() {
                let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
                if let Err(e) =
                    cutlass_cloud::download::download_to(&entry.file_url, &dest, &cancel, |_| {})
                {
                    warn!(asset = %entry.id, "lottie download skipped: {e}");
                    continue;
                }
            }
            let mut animation = match cutlass_decoder::LottieAnimation::load(&dest) {
                Ok(animation) => animation,
                Err(e) => {
                    warn!(asset = %entry.id, "lottie asset skipped: {e}");
                    // A cached file that no longer parses is junk; drop it
                    // so the next refresh re-downloads.
                    let _ = std::fs::remove_file(&dest);
                    continue;
                }
            };
            let (width, height) = animation.intrinsic_size();
            let thumbnail = match animation.render_frame(0) {
                Ok(frame) => Some(frame),
                Err(e) => {
                    warn!(asset = %entry.id, "lottie thumbnail failed: {e}");
                    None
                }
            };
            assets.insert(
                entry.id.clone(),
                LottieAsset {
                    path: dest,
                    width,
                    height,
                },
            );
            seeds.push(TileSeed {
                key: entry.id.clone(),
                name: entry.name.clone(),
                thumbnail,
            });
        }

        {
            let mut registry = self.registry.lock().expect("lottie registry poisoned");
            *registry = assets;
        }

        let status = if seeds.is_empty() { "empty" } else { "results" };
        let status = status.to_string();
        self.on_ui(move |backend| {
            let rows: Vec<LottieTile> =
                seeds
                    .iter()
                    .map(|seed| LottieTile {
                        key: seed.key.as_str().into(),
                        name: seed.name.as_str().into(),
                        thumbnail: seed
                            .thumbnail
                            .as_ref()
                            .map(|frame| {
                                slint::Image::from_rgba8(slint::SharedPixelBuffer::<
                                    slint::Rgba8Pixel,
                                >::clone_from_slice(
                                    &frame.pixels, frame.width, frame.height
                                ))
                            })
                            .unwrap_or_default(),
                    })
                    .collect();
            backend.set_items(ModelRc::new(VecModel::from(rows)));
            backend.set_status(status.as_str().into());
            backend.set_error("".into());
        });
        drop(layout_lease);
    }

    fn set_status(&self, status: &str, error: &str) {
        let status = status.to_string();
        let error = error.to_string();
        self.on_ui(move |backend| {
            backend.set_status(status.as_str().into());
            backend.set_error(error.as_str().into());
        });
    }

    fn on_ui(&self, f: impl FnOnce(LottieBackend<'_>) + Send + 'static) {
        let weak = self.backend_weak.clone();
        if let Err(e) = slint::invoke_from_event_loop(move || {
            if let Some(app) = weak.upgrade() {
                f(app.global::<LottieBackend>());
            }
        }) {
            warn!("lottie UI update failed: {e}");
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct RefreshRoots {
    catalog: PathBuf,
    lottie: PathBuf,
}

fn refresh_roots(lease: &StorageLayoutLease<'_>) -> Result<RefreshRoots, &'static str> {
    Ok(RefreshRoots {
        catalog: lease
            .resolve(CacheId::Catalog)
            .ok_or("catalog cache has no disk path")?,
        lottie: lease
            .resolve(CacheId::Lottie)
            .ok_or("Lottie cache has no disk path")?,
    })
}

/// Send-safe snapshot of a tile (Slint images must be built on the UI
/// thread; raw RGBA travels).
struct TileSeed {
    key: String,
    name: String,
    thumbnail: Option<cutlass_compositor::RgbaImage>,
}

fn user_message(e: &cutlass_cloud::CloudError) -> String {
    use cutlass_cloud::CloudError;
    match e {
        CloudError::Network(_) => {
            "Couldn't reach the animation catalog — check your connection.".into()
        }
        CloudError::Status {
            status, retryable, ..
        } => {
            if *retryable {
                "The animation catalog is busy — try again in a moment.".into()
            } else {
                format!("The animation catalog rejected the request ({status}).")
            }
        }
        CloudError::Protocol(_) => "The animation catalog sent an unexpected response.".into(),
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
        let first_lottie = dir.path().join("lottie-a");
        let second_catalog = dir.path().join("catalog-b");
        let second_lottie = dir.path().join("lottie-b");

        let mut first = StorageLayout::new(dir.path().join("default-a")).unwrap();
        first
            .set_override(CacheId::Catalog, &first_catalog)
            .unwrap();
        first.set_override(CacheId::Lottie, &first_lottie).unwrap();
        let layout = SharedStorageLayout::new(first);

        let first_lease = layout.lease();
        let first_generation = first_lease.generation();
        let first_refresh = refresh_roots(&first_lease).unwrap();
        assert_eq!(
            first_refresh,
            RefreshRoots {
                catalog: first_catalog.clone(),
                lottie: first_lottie.clone(),
            }
        );

        let mut second = StorageLayout::new(dir.path().join("default-b")).unwrap();
        second
            .set_override(CacheId::Catalog, &second_catalog)
            .unwrap();
        second
            .set_override(CacheId::Lottie, &second_lottie)
            .unwrap();
        drop(first_lease);
        layout.replace(first_generation, second).unwrap();

        assert_eq!(
            first_refresh,
            RefreshRoots {
                catalog: first_catalog,
                lottie: first_lottie,
            }
        );
        let second_lease = layout.lease();
        assert_eq!(second_lease.generation(), first_generation + 1);
        assert_eq!(
            refresh_roots(&second_lease).unwrap(),
            RefreshRoots {
                catalog: second_catalog,
                lottie: second_lottie,
            }
        );
    }
}
