//! Launch-screen templates gallery: the Rust half of `TemplatesBackend`.
//!
//! One worker thread owns all template HTTP (catalog fetches, preview
//! image fetches, bundle downloads) — network never touches the UI or
//! engine threads. The catalog is anonymous and comes from the Cutlass
//! backend; bundles download **directly from the asset CDN** into the
//! quota-managed download cache, install into a per-template folder in
//! the app data dir (`template_bundle::install` rewrites media paths to
//! absolute — install once, open forever), and fill through the engine's
//! `ApplyTemplate` after the user picks their media.
//!
//! Threading mirrors `cloud.rs`: commands in over a channel, results
//! hopped to the UI thread with `invoke_from_event_loop`, model rows
//! patched in place. The pick dialog is the one step that must run on
//! the UI thread (`rfd` + `slint::spawn_local`); the worker hands off to
//! it after install and the continuation sends `apply_template` to the
//! engine worker.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::thread::JoinHandle;

use crossbeam_channel::{Sender, unbounded};
use cutlass_cloud::cache::{DEFAULT_QUOTA_BYTES, DownloadCache};
use cutlass_cloud::dto::CatalogEntry;
use cutlass_cloud::{CloudClient, download};
use cutlass_commands::TemplatePick;
use cutlass_models::{PROJECT_SCHEMA_VERSION, template_bundle};
use slint::{ComponentHandle, Image, Model, ModelRc, Rgba8Pixel, SharedPixelBuffer, VecModel};
use tracing::{info, warn};

use crate::paths;
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
    ) -> Result<Self, String> {
        let (tx, rx) = unbounded::<Command>();
        let join = std::thread::Builder::new()
            .name("cutlass-templates".into())
            .spawn(move || {
                let mut worker = Worker::new(backend_weak, preview_handle);
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
    client: CloudClient,
    cache: DownloadCache,
    /// The worker-side mirror of the published (filtered) tile list.
    entries: Vec<CatalogEntry>,
}

impl Worker {
    fn new(backend_weak: slint::Weak<crate::AppWindow>, preview_handle: WorkerHandle) -> Self {
        Self {
            backend_weak,
            preview_handle,
            client: CloudClient::new(
                &crate::account::base_url(),
                Some(paths::data_dir().join("catalog-cache")),
            ),
            cache: DownloadCache::new(
                paths::data_dir().join("download-cache"),
                DEFAULT_QUOTA_BYTES,
            ),
            entries: Vec::new(),
        }
    }

    fn run(&mut self, command: Command, previews: &mut VecDeque<(usize, CatalogEntry)>) {
        match command {
            Command::Refresh { category } => {
                previews.clear();
                self.set_status("loading", "");
                match self.client.templates() {
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
            if let Some(mut tile) = model.row_data(row) {
                if tile.key == key.as_str() {
                    patch(&mut tile);
                    model.set_row_data(row, tile);
                }
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
        let path = match self
            .cache
            .path_for(&format!("template-previews/{}.img", safe_id(&entry.id)))
        {
            Ok(path) => path,
            Err(e) => {
                warn!("template preview cache path failed: {e}");
                return;
            }
        };
        if !path.is_file() {
            let cancel = Arc::new(AtomicBool::new(false));
            if let Err(e) = download::download_to(url, &path, &cancel, |_| {}) {
                warn!("template preview download failed: {e}");
                return;
            }
        }
        let Ok(bytes) = std::fs::read(&path) else {
            return;
        };
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

        let install_dir = paths::data_dir().join("templates").join(safe_id(&entry.id));
        let template_path = install_dir.join("template.cutlasst");

        // Install once, open forever: skip download + install when present.
        let template = if template_path.is_file() {
            cutlass_models::Template::load_from_file(&template_path)
        } else {
            match self.download_bundle(index, &entry) {
                Ok(bundle) => {
                    self.patch_row(index, key.clone(), |tile| tile.state = "installing".into());
                    template_bundle::install(&bundle, &install_dir)
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
        self.run_pick_flow(index, key, template_path, template.slot_count());
    }

    /// Download the bundle into the download cache, with tile progress and
    /// a checksum check when the catalog carries one.
    fn download_bundle(&self, index: usize, entry: &CatalogEntry) -> Result<PathBuf, String> {
        let cache_key = format!("templates/{}.cutlassb", safe_id(&entry.id));
        if let Some(path) = self.cache.hit(&cache_key) {
            return Ok(path);
        }
        let dest = self.cache.path_for(&cache_key).map_err(|e| e.to_string())?;
        let key = entry.id.clone();
        self.patch_row(index, key.clone(), |tile| {
            tile.state = "downloading".into();
            tile.progress = 0.0;
        });

        let cancel = Arc::new(AtomicBool::new(false));
        let mut last_published = 0.0_f32;
        download::download_to(&entry.file_url, &dest, &cancel, |p| {
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
            let actual = download::sha256_hex(&dest).map_err(|e| e.to_string())?;
            if !actual.eq_ignore_ascii_case(&entry.checksum_sha256) {
                let _ = std::fs::remove_file(&dest);
                return Err(format!(
                    "checksum mismatch for {} (expected {}, got {actual})",
                    entry.id, entry.checksum_sha256
                ));
            }
        }
        self.cache.enforce_quota();
        Ok(dest)
    }

    /// Hop to the UI thread for the media-pick dialog, then hand the filled
    /// template to the engine worker. Fewer picks than slots is fine (the
    /// remaining slots keep their sample media, exactly like CapCut); extra
    /// picks are truncated. Cancelling the dialog just resets the tile.
    fn run_pick_flow(&self, row: usize, key: String, template_path: PathBuf, slots: usize) {
        let weak = self.backend_weak.clone();
        let preview_handle = self.preview_handle.clone();
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
                    if let Some(mut tile) = model.row_data(row) {
                        if tile.key == key.as_str() {
                            tile.state = Default::default();
                            model.set_row_data(row, tile);
                        }
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
                // Flush the outgoing draft (a no-op from the launch screen),
                // then swap the session — ordered on the worker's queue.
                preview_handle.save_project(None);
                preview_handle.apply_template(template_path, picks);
            });
            if let Err(e) = task {
                tracing::error!("failed to open template pick dialog: {e}");
            }
        }) {
            warn!("template pick flow failed to reach the UI thread: {e}");
        }
    }
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
mod tests {
    use super::*;
    use cutlass_cloud::dto::AssetKind;

    fn entry(duration: Option<f64>, slots: Option<u32>) -> CatalogEntry {
        CatalogEntry {
            id: "tpl-1".into(),
            kind: AssetKind::Template,
            name: "T".into(),
            category: "vlog".into(),
            tags: vec![],
            file_url: String::new(),
            preview_url: None,
            size_bytes: 0,
            checksum_sha256: String::new(),
            min_schema_version: None,
            author: "Cutlass".into(),
            license: "CC0".into(),
            duration_seconds: duration,
            slot_count: slots,
        }
    }

    #[test]
    fn tile_labels() {
        assert_eq!(tile_label(&entry(Some(72.4), Some(3))), "1:12 · 3 clips");
        assert_eq!(tile_label(&entry(Some(9.0), Some(1))), "0:09 · 1 clip");
        assert_eq!(tile_label(&entry(None, None)), "");
    }

    #[test]
    fn safe_id_strips_traversal_material() {
        assert_eq!(safe_id("tpl-vlog-1"), "tpl-vlog-1");
        assert_eq!(safe_id("../../etc/passwd"), "etcpasswd");
        assert_eq!(safe_id("..."), "unnamed");
    }
}
