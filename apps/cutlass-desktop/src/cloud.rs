//! Library stock browsing: the Rust half of `CloudBackend`.
//!
//! One worker thread owns all cloud HTTP (searches, thumbnail fetches,
//! media downloads) — network never touches the UI or engine threads.
//! Search metadata comes from the Cutlass backend anonymously, or directly
//! from Pexels/Pixabay when the user brought their own stock keys (the
//! BYOK-first routing rule). Media files always download **directly from
//! the provider CDNs** into a quota-managed cache, then ride the normal
//! import path — an imported stock clip is ordinary pool media.
//!
//! Threading mirrors `thumbnails.rs`: commands in over a channel, results
//! hopped to the UI thread with `invoke_from_event_loop`, model rows
//! patched in place.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::thread::JoinHandle;

use crossbeam_channel::{Sender, unbounded};
use cutlass_cloud::cache::{DEFAULT_QUOTA_BYTES, DownloadCache};
use cutlass_cloud::dto::{StockItem, StockKind};
use cutlass_cloud::stock::{BackendStockProvider, DirectStockProvider, StockProvider};
use cutlass_cloud::{CloudClient, download};
use slint::{ComponentHandle, Image, Model, ModelRc, Rgba8Pixel, SharedPixelBuffer, VecModel};
use tracing::{info, warn};

use crate::paths;
use crate::preview_worker::WorkerHandle;
use crate::{CloudBackend, StockTile};

enum Command {
    Search { query: String, kind: StockKind },
    LoadMore,
    Import { index: usize },
}

/// Cheap, cloneable sender to the cloud thread.
#[derive(Clone)]
pub struct CloudHandle {
    tx: Sender<Command>,
}

impl CloudHandle {
    pub fn search(&self, query: String, kind: &str) {
        let kind = parse_kind(kind);
        let _ = self.tx.send(Command::Search { query, kind });
    }

    pub fn load_more(&self) {
        let _ = self.tx.send(Command::LoadMore);
    }

    pub fn import(&self, index: usize) {
        let _ = self.tx.send(Command::Import { index });
    }
}

fn parse_kind(kind: &str) -> StockKind {
    match kind {
        "photo" => StockKind::Photo,
        "audio" => StockKind::Audio,
        _ => StockKind::Video,
    }
}

pub struct CloudWorker {
    handle: CloudHandle,
    _join: JoinHandle<()>,
}

impl CloudWorker {
    pub fn spawn(
        backend_weak: slint::Weak<crate::AppWindow>,
        import_handle: WorkerHandle,
    ) -> Result<Self, String> {
        let (tx, rx) = unbounded::<Command>();
        let join = std::thread::Builder::new()
            .name("cutlass-cloud".into())
            .spawn(move || {
                let mut worker = Worker::new(backend_weak, import_handle);
                // Thumbnail fetches interleave with commands: commands take
                // priority (a queued import shouldn't wait for a page of
                // thumbnails), thumbs drain in the gaps.
                let mut thumbs: VecDeque<(usize, StockItem)> = VecDeque::new();
                loop {
                    let command = if thumbs.is_empty() {
                        match rx.recv() {
                            Ok(c) => Some(c),
                            Err(_) => return,
                        }
                    } else {
                        rx.try_recv().ok()
                    };
                    match command {
                        Some(c) => worker.run(c, &mut thumbs),
                        None => {
                            if let Some((row, item)) = thumbs.pop_front() {
                                worker.fetch_thumbnail(row, &item);
                            }
                        }
                    }
                }
            })
            .map_err(|e| e.to_string())?;

        Ok(Self {
            handle: CloudHandle { tx },
            _join: join,
        })
    }

    pub fn handle(&self) -> CloudHandle {
        self.handle.clone()
    }
}

struct Worker {
    backend_weak: slint::Weak<crate::AppWindow>,
    import_handle: WorkerHandle,
    provider: Box<dyn StockProvider>,
    cache: DownloadCache,
    /// The worker-side mirror of the published tile list.
    items: Vec<StockItem>,
    query: String,
    kind: StockKind,
    page: u32,
}

impl Worker {
    fn new(backend_weak: slint::Weak<crate::AppWindow>, import_handle: WorkerHandle) -> Self {
        // The BYOK-first routing rule: stock keys from the `[providers.*]`
        // settings registry (env vars still work as a fallback) route search
        // straight to the providers; otherwise anonymous search goes through
        // the Cutlass backend.
        let settings =
            cutlass_settings::load(&cutlass_settings::default_config_path()).unwrap_or_default();
        let key_for = |name: &str, env: &str| {
            settings
                .provider(name)
                .resolve_key()
                .or_else(|| std::env::var(env).ok())
                .filter(|k| !k.is_empty())
        };
        let pexels = key_for("pexels", "PEXELS_API_KEY");
        let pixabay = key_for("pixabay", "PIXABAY_API_KEY");
        let provider: Box<dyn StockProvider> = if pexels.is_some() || pixabay.is_some() {
            info!("stock search: using BYOK provider keys");
            Box::new(DirectStockProvider::new(pexels, pixabay))
        } else {
            let cache_dir = paths::data_dir().join("catalog-cache");
            Box::new(BackendStockProvider::new(CloudClient::new(
                &crate::account::base_url(),
                Some(cache_dir),
            )))
        };
        Self {
            backend_weak,
            import_handle,
            provider,
            cache: DownloadCache::new(
                paths::data_dir().join("download-cache"),
                DEFAULT_QUOTA_BYTES,
            ),
            items: Vec::new(),
            query: String::new(),
            kind: StockKind::Video,
            page: 1,
        }
    }

    fn run(&mut self, command: Command, thumbs: &mut VecDeque<(usize, StockItem)>) {
        match command {
            Command::Search { query, kind } => {
                thumbs.clear();
                self.query = query;
                self.kind = kind;
                self.page = 1;
                self.set_status("searching", "");
                match self.provider.search(&self.query, self.kind, 1) {
                    Ok(response) => {
                        self.items = response.items;
                        let status = if self.items.is_empty() {
                            "empty"
                        } else {
                            "results"
                        };
                        self.publish_all(status, response.has_more);
                        for (row, item) in self.items.iter().enumerate() {
                            thumbs.push_back((row, item.clone()));
                        }
                    }
                    Err(e) => {
                        self.items.clear();
                        self.set_status("error", &user_message(&e));
                    }
                }
            }
            Command::LoadMore => {
                let next = self.page + 1;
                self.set_loading_more(true);
                match self.provider.search(&self.query, self.kind, next) {
                    Ok(response) => {
                        self.page = next;
                        let first_new = self.items.len();
                        self.items.extend(response.items);
                        self.publish_all("results", response.has_more);
                        for (row, item) in self.items.iter().enumerate().skip(first_new) {
                            thumbs.push_back((row, item.clone()));
                        }
                    }
                    Err(e) => {
                        // Keep the page we have; just stop the spinner.
                        warn!("stock load-more failed: {e}");
                        self.set_loading_more(false);
                    }
                }
            }
            Command::Import { index } => self.import(index),
        }
    }

    // --- UI publishing ----------------------------------------------------

    fn set_status(&self, status: &str, error: &str) {
        let status = status.to_string();
        let error = error.to_string();
        self.on_ui(move |backend| {
            backend.set_stock_status(status.as_str().into());
            backend.set_stock_error(error.as_str().into());
            if status == "searching" {
                backend.set_stock_items(ModelRc::default());
                backend.set_stock_has_more(false);
                backend.set_stock_loading_more(false);
            }
        });
    }

    fn set_loading_more(&self, loading: bool) {
        self.on_ui(move |backend| backend.set_stock_loading_more(loading));
    }

    /// Replace the whole tile list (fresh search / appended page). Existing
    /// thumbnails are preserved by re-publishing whatever the current model
    /// rows already carry for matching keys.
    fn publish_all(&self, status: &str, has_more: bool) {
        let tiles: Vec<TileSeed> = self.items.iter().map(TileSeed::from).collect();
        let status = status.to_string();
        self.on_ui(move |backend| {
            let existing = backend.get_stock_items();
            let rows: Vec<StockTile> = tiles
                .iter()
                .map(|seed| {
                    // Keep an already-fetched thumbnail when the row survives
                    // (load-more re-publish).
                    let kept = existing
                        .iter()
                        .find(|t| t.key == seed.key.as_str())
                        .filter(|t| t.has_thumbnail);
                    StockTile {
                        key: seed.key.as_str().into(),
                        thumbnail: kept
                            .as_ref()
                            .map(|t| t.thumbnail.clone())
                            .unwrap_or_default(),
                        has_thumbnail: kept.is_some(),
                        label: seed.label.as_str().into(),
                        attribution: seed.attribution.as_str().into(),
                        state: kept.map(|t| t.state.clone()).unwrap_or_default(),
                        progress: 0.0,
                    }
                })
                .collect();
            backend.set_stock_items(ModelRc::new(VecModel::from(rows)));
            backend.set_stock_status(status.as_str().into());
            backend.set_stock_has_more(has_more);
            backend.set_stock_loading_more(false);
        });
    }

    /// Patch one row's field in place, guarded by key so a stale patch
    /// (list replaced mid-flight) can't hit the wrong tile.
    fn patch_row(&self, row: usize, key: String, patch: impl Fn(&mut StockTile) + Send + 'static) {
        self.on_ui(move |backend| {
            let model = backend.get_stock_items();
            if let Some(mut tile) = model.row_data(row) {
                if tile.key == key.as_str() {
                    patch(&mut tile);
                    model.set_row_data(row, tile);
                }
            }
        });
    }

    fn on_ui(&self, f: impl FnOnce(CloudBackend<'_>) + Send + 'static) {
        let weak = self.backend_weak.clone();
        if let Err(e) = slint::invoke_from_event_loop(move || {
            if let Some(app) = weak.upgrade() {
                f(app.global::<CloudBackend>());
            }
        }) {
            warn!("cloud UI update failed: {e}");
        }
    }

    // --- thumbnails ---------------------------------------------------------

    fn fetch_thumbnail(&self, row: usize, item: &StockItem) {
        if item.thumbnail_url.is_empty() {
            return;
        }
        let key = tile_key(item);
        let path = match self.cache.path_for(&format!("stock-thumbs/{key}.img")) {
            Ok(path) => path,
            Err(e) => {
                warn!("thumb cache path failed: {e}");
                return;
            }
        };
        if !path.is_file() {
            let cancel = Arc::new(AtomicBool::new(false));
            if let Err(e) = download::download_to(&item.thumbnail_url, &path, &cancel, |_| {}) {
                warn!("stock thumbnail download failed: {e}");
                return;
            }
        }
        let Ok(bytes) = std::fs::read(&path) else {
            return;
        };
        let Ok(decoded) = cutlass_decoder::decode_image_bytes(&bytes) else {
            warn!("stock thumbnail decode failed for {key}");
            return;
        };
        let (width, height, pixels) = (decoded.width, decoded.height, decoded.pixels);
        self.patch_row(row, key, move |tile| {
            let buffer = SharedPixelBuffer::<Rgba8Pixel>::clone_from_slice(&pixels, width, height);
            tile.thumbnail = Image::from_rgba8(buffer);
            tile.has_thumbnail = true;
        });
    }

    // --- import -------------------------------------------------------------

    fn import(&self, index: usize) {
        let Some(item) = self.items.get(index).cloned() else {
            return;
        };
        let Some(file) = item.files.first() else {
            warn!("stock item {} has no files", item.id);
            return;
        };
        let key = tile_key(&item);
        let cache_key = format!(
            "stock/{key}.{}",
            extension(&item, &file.content_type, &file.url)
        );

        // Cache hit: straight to import, no download UI.
        if let Some(path) = self.cache.hit(&cache_key) {
            self.import_handle.import(path);
            self.patch_row(index, key, |tile| tile.state = "imported".into());
            return;
        }

        let dest = match self.cache.path_for(&cache_key) {
            Ok(dest) => dest,
            Err(e) => {
                warn!("stock cache path failed: {e}");
                return;
            }
        };
        self.patch_row(index, key.clone(), |tile| {
            tile.state = "downloading".into();
            tile.progress = 0.0;
        });

        let cancel = Arc::new(AtomicBool::new(false));
        let mut last_published = 0.0_f32;
        let progress_key = key.clone();
        let url = file.url.clone();
        let result = download::download_to(&url, &dest, &cancel, |p| {
            if p.total_bytes == 0 {
                return;
            }
            let fraction = (p.bytes_downloaded as f64 / p.total_bytes as f64) as f32;
            // Patch at 5% steps, not per 64 KiB chunk — event-loop hops are
            // not free and the bar doesn't need them.
            if fraction - last_published >= 0.05 || fraction >= 1.0 {
                last_published = fraction;
                self.patch_row(index, progress_key.clone(), move |tile| {
                    tile.progress = fraction;
                });
            }
        });

        match result {
            Ok(()) => {
                info!("stock import: {} -> {}", url, dest.display());
                self.import_handle.import(dest);
                self.patch_row(index, key, |tile| tile.state = "imported".into());
                self.cache.enforce_quota();
            }
            Err(e) => {
                warn!("stock download failed: {e}");
                self.patch_row(index, key, |tile| tile.state = "failed".into());
            }
        }
    }
}

/// Send-safe snapshot of a `StockItem`'s tile fields (built worker-side;
/// `StockTile` itself holds an `Image` and is UI-thread only).
struct TileSeed {
    key: String,
    label: String,
    attribution: String,
}

impl From<&StockItem> for TileSeed {
    fn from(item: &StockItem) -> Self {
        Self {
            key: tile_key(item),
            label: tile_label(item),
            attribution: item.attribution.clone(),
        }
    }
}

fn tile_key(item: &StockItem) -> String {
    format!("{:?}:{}", item.provider, item.id).to_lowercase()
}

/// Corner chip: "0:12" for video, "4000×2250" for photos.
fn tile_label(item: &StockItem) -> String {
    if let Some(seconds) = item.duration_seconds {
        let total = seconds.round() as i64;
        return format!("{}:{:02}", total / 60, total % 60);
    }
    if item.width > 0 && item.height > 0 {
        return format!("{}×{}", item.width, item.height);
    }
    String::new()
}

/// Best-effort file extension: MIME first, then the URL path, then a
/// kind-appropriate default (the decoder sniffs content anyway; this only
/// keeps cache filenames and import probing sensible).
fn extension(item: &StockItem, content_type: &str, url: &str) -> String {
    match content_type {
        "video/mp4" => return "mp4".into(),
        "video/quicktime" => return "mov".into(),
        "image/jpeg" => return "jpg".into(),
        "image/png" => return "png".into(),
        _ => {}
    }
    let path_part = url.split(['?', '#']).next().unwrap_or("");
    if let Some((_, ext)) = path_part.rsplit_once('.') {
        if !ext.is_empty() && ext.len() <= 4 && ext.chars().all(|c| c.is_ascii_alphanumeric()) {
            return ext.to_lowercase();
        }
    }
    match item.kind {
        StockKind::Photo => "jpg".into(),
        StockKind::Audio => "mp3".into(),
        StockKind::Video => "mp4".into(),
    }
}

fn user_message(e: &cutlass_cloud::CloudError) -> String {
    use cutlass_cloud::CloudError;
    match e {
        CloudError::Network(_) => {
            "Couldn't reach the stock service — check your connection.".into()
        }
        CloudError::Status {
            status, retryable, ..
        } => {
            if *retryable {
                "The stock service is busy — try again in a moment.".into()
            } else {
                format!("The stock service rejected the request ({status}).")
            }
        }
        CloudError::Protocol(_) => "The stock service sent an unexpected response.".into(),
        CloudError::Io(_) | CloudError::Cancelled => "The search was interrupted.".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_cloud::dto::StockProviderId;

    fn item(kind: StockKind, duration: Option<f64>) -> StockItem {
        StockItem {
            id: "42".into(),
            provider: StockProviderId::Pexels,
            kind,
            width: 4000,
            height: 2250,
            duration_seconds: duration,
            thumbnail_url: String::new(),
            files: vec![],
            author: "A".into(),
            attribution: "A on Pexels".into(),
            license: "L".into(),
            source_url: String::new(),
        }
    }

    #[test]
    fn tile_labels() {
        assert_eq!(tile_label(&item(StockKind::Video, Some(72.4))), "1:12");
        assert_eq!(tile_label(&item(StockKind::Photo, None)), "4000×2250");
    }

    #[test]
    fn tile_key_is_provider_scoped() {
        assert_eq!(tile_key(&item(StockKind::Video, None)), "pexels:42");
    }

    #[test]
    fn extension_prefers_mime_then_url_then_kind() {
        let i = item(StockKind::Video, None);
        assert_eq!(extension(&i, "video/mp4", "https://c/x"), "mp4");
        assert_eq!(extension(&i, "", "https://c/x.MOV?dl=1"), "mov");
        assert_eq!(extension(&i, "", "https://c/x"), "mp4");
        let p = item(StockKind::Photo, None);
        assert_eq!(extension(&p, "", "https://c/x"), "jpg");
    }
}
