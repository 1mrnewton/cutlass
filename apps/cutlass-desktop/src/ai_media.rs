//! AI generation (Library AI sections): the Rust half of `AiBackend`.
//!
//! One worker thread owns the whole pipeline: route decision → start job →
//! poll to terminal → thumbnail. Generation requires a `[providers.fal]`
//! key (direct fal.ai; backend uninvolved). Without a key, generation is
//! unavailable and the UI says how to enable it.
//!
//! Results are provider-CDN URLs; import downloads into the quota-managed
//! cache and rides the normal import path, so generated media is ordinary
//! pool media (proxies, thumbnails, relink all work).

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{Sender, unbounded};
use cutlass_cloud::cache::DownloadCache;
use cutlass_cloud::dto::{GenerateRequest, JobStatus};
use cutlass_cloud::generate::{FalGenerationProvider, GenerationKind, GenerationProvider};
use cutlass_cloud::{CloudError, download};
use slint::{ComponentHandle, Image, Model, ModelRc, Rgba8Pixel, SharedPixelBuffer, VecModel};
use tracing::{info, warn};

use crate::preview_worker::WorkerHandle;
use crate::{AiBackend, AiItemTile};

/// Generation can take minutes (video); poll at a human pace.
const POLL_INTERVAL: Duration = Duration::from_secs(2);
const JOB_TIMEOUT: Duration = Duration::from_secs(600);

enum Command {
    Generate { kind: String, prompt: String },
    Import { kind: String, index: usize },
    RefreshRoute,
}

#[derive(Clone)]
pub struct AiMediaHandle {
    tx: Sender<Command>,
}

impl AiMediaHandle {
    pub fn generate(&self, kind: String, prompt: String) {
        let _ = self.tx.send(Command::Generate { kind, prompt });
    }

    pub fn import(&self, kind: String, index: usize) {
        let _ = self.tx.send(Command::Import { kind, index });
    }

    pub fn refresh_route(&self) {
        let _ = self.tx.send(Command::RefreshRoute);
    }
}

pub struct AiMediaWorker {
    handle: AiMediaHandle,
    _join: JoinHandle<()>,
}

impl AiMediaWorker {
    pub fn spawn(
        backend_weak: slint::Weak<crate::AppWindow>,
        import_handle: WorkerHandle,
        cache: Arc<DownloadCache>,
    ) -> Result<Self, String> {
        let (tx, rx) = unbounded::<Command>();
        let join = std::thread::Builder::new()
            .name("cutlass-ai-media".into())
            .spawn(move || {
                let mut worker = Worker::new(backend_weak, import_handle, cache);
                worker.publish_route();
                while let Ok(command) = rx.recv() {
                    worker.run(command);
                }
            })
            .map_err(|e| e.to_string())?;
        Ok(Self {
            handle: AiMediaHandle { tx },
            _join: join,
        })
    }

    pub fn handle(&self) -> AiMediaHandle {
        self.handle.clone()
    }
}

/// One finished-or-in-flight generation, worker-side.
struct Entry {
    prompt: String,
    kind: GenerationKind,
    /// CDN URL once the job succeeded.
    result_url: Option<String>,
}

/// Which route a generation takes, decided fresh per request (settings
/// can change under a running app).
enum Route {
    Byok(String),
    None,
}

struct Worker {
    backend_weak: slint::Weak<crate::AppWindow>,
    import_handle: WorkerHandle,
    cache: Arc<DownloadCache>,
    media_entries: Vec<Entry>,
    tts_entries: Vec<Entry>,
}

impl Worker {
    fn new(
        backend_weak: slint::Weak<crate::AppWindow>,
        import_handle: WorkerHandle,
        cache: Arc<DownloadCache>,
    ) -> Self {
        Self {
            backend_weak,
            import_handle,
            cache,
            media_entries: Vec::new(),
            tts_entries: Vec::new(),
        }
    }

    fn run(&mut self, command: Command) {
        match command {
            Command::Generate { kind, prompt } => self.generate(&kind, prompt),
            Command::Import { kind, index } => self.import(&kind, index),
            Command::RefreshRoute => self.publish_route(),
        }
    }

    // --- routing ------------------------------------------------------------

    fn route(&self) -> Route {
        let fal_key = cutlass_settings::load(&cutlass_settings::default_config_path())
            .map(|s| s.provider("fal").resolve_key())
            .unwrap_or_default()
            .filter(|k| !k.is_empty());
        match fal_key {
            Some(key) => Route::Byok(key),
            None => Route::None,
        }
    }

    fn publish_route(&self) {
        let (label, available) = match self.route() {
            Route::Byok(_) => (
                "Runs on your fal.ai key — the Cutlass backend is not involved.",
                true,
            ),
            Route::None => (
                "Generation is off: add a fal.ai key under [providers.fal].",
                false,
            ),
        };
        self.on_ui(move |b| {
            b.set_route_label(label.into());
            b.set_available(available);
        });
    }

    /// A ready-to-call provider for the current route.
    fn provider(&self) -> Result<Box<dyn GenerationProvider>, String> {
        match self.route() {
            Route::Byok(key) => Ok(Box::new(FalGenerationProvider::new(&key))),
            Route::None => Err("Add a fal.ai key under [providers.fal] to generate.".into()),
        }
    }

    // --- generation ---------------------------------------------------------

    fn generate(&mut self, kind_str: &str, prompt: String) {
        let kind = match kind_str {
            "video" => GenerationKind::Video,
            "tts" => GenerationKind::Tts,
            _ => GenerationKind::Image,
        };
        self.publish_route();
        let provider = match self.provider() {
            Ok(provider) => provider,
            Err(message) => {
                self.on_ui(move |b| b.set_error(message.as_str().into()));
                return;
            }
        };

        // Append the in-flight tile.
        let row = self.entries_mut(kind).len();
        self.entries_mut(kind).push(Entry {
            prompt: prompt.clone(),
            kind,
            result_url: None,
        });
        let tile_prompt = prompt.clone();
        let kind_name = kind.as_str().to_string();
        let list_is_tts = kind == GenerationKind::Tts;
        self.on_ui(move |b| {
            b.set_busy(true);
            b.set_error("".into());
            // Rebuild list = existing rows + the new in-flight tile
            // (cloning on the UI thread keeps fetched thumbnails).
            let existing = if list_is_tts {
                b.get_tts_items()
            } else {
                b.get_media_items()
            };
            let mut rows: Vec<AiItemTile> = existing.iter().collect();
            rows.push(AiItemTile {
                key: format!("{kind_name}:{row}").into(),
                label: tile_prompt.as_str().into(),
                kind: kind_name.as_str().into(),
                state: "generating".into(),
                thumbnail: Image::default(),
                has_thumbnail: false,
            });
            let model = ModelRc::new(VecModel::from(rows));
            if list_is_tts {
                b.set_tts_items(model);
            } else {
                b.set_media_items(model);
            }
        });

        let request = GenerateRequest {
            prompt: prompt.clone(),
            model: String::new(),
            duration_seconds: (kind == GenerationKind::Video).then_some(5.0),
            idempotency_key: format!(
                "desktop-{}-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0),
                row
            ),
        };

        let outcome = self.run_job(provider.as_ref(), kind, &request);
        match outcome {
            Ok(url) => {
                info!("generation succeeded: {url}");
                self.entries_mut(kind)[row].result_url = Some(url.clone());
                self.patch(kind, row, |tile| tile.state = "ready".into());
                self.on_ui(|b| b.set_busy(false));
                if kind != GenerationKind::Tts {
                    self.fetch_thumbnail(kind, row, &url);
                }
            }
            Err(message) => {
                warn!("generation failed: {message}");
                self.patch(kind, row, |tile| tile.state = "failed".into());
                self.on_ui(move |b| {
                    b.set_busy(false);
                    b.set_error(message.as_str().into());
                });
            }
        }
    }

    /// Start + poll to a terminal state (blocking; this is the worker).
    fn run_job(
        &self,
        provider: &dyn GenerationProvider,
        kind: GenerationKind,
        request: &GenerateRequest,
    ) -> Result<String, String> {
        let job = provider.start(kind, request).map_err(user_message)?;
        let deadline = Instant::now() + JOB_TIMEOUT;
        let mut current = job;
        loop {
            match current.status {
                JobStatus::Succeeded => {
                    return current
                        .result_url
                        .filter(|u| !u.is_empty())
                        .ok_or_else(|| "The provider returned no media.".into());
                }
                JobStatus::Failed => {
                    return Err(current.error.unwrap_or_else(|| "Generation failed.".into()));
                }
                _ => {}
            }
            if Instant::now() >= deadline {
                return Err("Generation timed out — try again.".into());
            }
            std::thread::sleep(POLL_INTERVAL);
            current = provider.poll(&current.id).map_err(user_message)?;
        }
    }

    // --- import -------------------------------------------------------------

    fn import(&mut self, kind_str: &str, index: usize) {
        let kind = if kind_str == "tts" {
            GenerationKind::Tts
        } else {
            GenerationKind::Image // list selector only; kind field is per-entry
        };
        let Some(entry) = self.entries_mut(kind).get(index) else {
            return;
        };
        let (kind, url, prompt) = (entry.kind, entry.result_url.clone(), entry.prompt.clone());
        let Some(url) = url else { return };

        let cache_key = format!("ai/{}.{}", stable_key(&url), extension_of(kind, &url));
        let lease = match self.cache.lease(&cache_key) {
            Ok(lease) => lease,
            Err(e) => {
                warn!("ai cache path failed: {e}");
                return;
            }
        };
        let dest = lease.path().to_path_buf();
        let downloaded = if self.cache.hit(&cache_key).is_some() {
            false
        } else {
            let cancel = Arc::new(AtomicBool::new(false));
            if let Err(e) = download::download_to(&url, &dest, &cancel, |_| {}) {
                warn!("ai result download failed: {e}");
                let message = "Download failed — the result may have expired; regenerate.";
                self.on_ui(move |b| b.set_error(message.into()));
                return;
            }
            true
        };
        info!("importing AI result \"{prompt}\" from {}", dest.display());
        if let Err(error) = self.cache.protect_path(&dest) {
            warn!("AI import could not protect its source: {error}");
            self.on_ui(|backend| {
                backend.set_error("The downloaded result could not be imported safely.".into());
            });
            return;
        }
        self.import_handle.import(dest);
        drop(lease);
        if downloaded {
            self.cache.enforce_quota();
        }
        self.patch(kind, index, |tile| tile.state = "imported".into());
    }

    // --- thumbnails ---------------------------------------------------------

    /// Image results decode directly; video thumbnails wait for the normal
    /// pool pipeline post-import (no frame extraction here).
    fn fetch_thumbnail(&self, kind: GenerationKind, row: usize, url: &str) {
        if kind != GenerationKind::Image {
            return;
        }
        let cache_key = format!("ai-thumbs/{}.img", stable_key(url));
        let lease = match self.cache.lease(&cache_key) {
            Ok(lease) => lease,
            Err(_) => return,
        };
        let path = lease.path();
        if !path.is_file() {
            let cancel = Arc::new(AtomicBool::new(false));
            if download::download_to(url, path, &cancel, |_| {}).is_err() {
                return;
            }
        }
        let Ok(bytes) = std::fs::read(path) else {
            return;
        };
        drop(lease);
        let Ok(decoded) = cutlass_decoder::decode_image_bytes(&bytes) else {
            return;
        };
        let (width, height, pixels) = (decoded.width, decoded.height, decoded.pixels);
        self.patch(kind, row, move |tile| {
            let buffer = SharedPixelBuffer::<Rgba8Pixel>::clone_from_slice(&pixels, width, height);
            tile.thumbnail = Image::from_rgba8(buffer);
            tile.has_thumbnail = true;
        });
    }

    // --- UI publishing ------------------------------------------------------

    fn entries_mut(&mut self, kind: GenerationKind) -> &mut Vec<Entry> {
        if kind == GenerationKind::Tts {
            &mut self.tts_entries
        } else {
            &mut self.media_entries
        }
    }

    fn patch(
        &self,
        kind: GenerationKind,
        row: usize,
        patch: impl Fn(&mut AiItemTile) + Send + 'static,
    ) {
        let is_tts = kind == GenerationKind::Tts;
        self.on_ui(move |b| {
            let model = if is_tts {
                b.get_tts_items()
            } else {
                b.get_media_items()
            };
            if let Some(mut tile) = model.row_data(row) {
                patch(&mut tile);
                model.set_row_data(row, tile);
            }
        });
    }

    fn on_ui(&self, f: impl FnOnce(AiBackend<'_>) + Send + 'static) {
        let weak = self.backend_weak.clone();
        if let Err(e) = slint::invoke_from_event_loop(move || {
            if let Some(app) = weak.upgrade() {
                f(app.global::<AiBackend>());
            }
        }) {
            warn!("ai UI update failed: {e}");
        }
    }
}

/// A filesystem-safe cache key from a result URL.
fn stable_key(url: &str) -> String {
    url.bytes()
        .fold(0xcbf29ce484222325u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
        })
        .to_string()
}

fn extension_of(kind: GenerationKind, url: &str) -> String {
    let path_part = url.split(['?', '#']).next().unwrap_or("");
    if let Some((_, ext)) = path_part.rsplit_once('.')
        && !ext.is_empty()
        && ext.len() <= 4
        && ext.chars().all(|c| c.is_ascii_alphanumeric())
    {
        return ext.to_lowercase();
    }
    match kind {
        GenerationKind::Image => "png".into(),
        GenerationKind::Video => "mp4".into(),
        GenerationKind::Tts => "wav".into(),
    }
}

fn user_message(e: CloudError) -> String {
    match &e {
        CloudError::Network(_) => "Couldn't reach the generation service.".into(),
        CloudError::Status { status, .. } => format!("The generation service said no ({status})."),
        _ => "Generation failed — try again.".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extensions_fall_back_per_kind() {
        assert_eq!(
            extension_of(GenerationKind::Image, "https://c/x.PNG?sig=1"),
            "png"
        );
        assert_eq!(extension_of(GenerationKind::Video, "https://c/x"), "mp4");
        assert_eq!(extension_of(GenerationKind::Tts, "https://c/x"), "wav");
    }

    #[test]
    fn stable_keys_differ_per_url() {
        assert_ne!(stable_key("https://a"), stable_key("https://b"));
        assert_eq!(stable_key("https://a"), stable_key("https://a"));
    }
}
