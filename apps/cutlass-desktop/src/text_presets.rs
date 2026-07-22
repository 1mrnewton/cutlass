//! Animated text presets: the Rust half of `TextPresetsBackend`.
//!
//! A preset is a served recipe — a `TextStyle` subset plus look-animation
//! catalog ids — rendered by the existing text pipeline (no new render
//! tech). The worker fetches the preset catalog from the Cutlass backend
//! (`/v1/assets/text-presets` lists pack entries; each entry's file is a
//! `TextPresetCatalog` JSON on the CDN), fills a shared registry the drop
//! resolver reads, and publishes Library tiles.
//!
//! **Bundled-OFL-fonts-only**: the text renderer resolves named fonts
//! against host fonts with a *silent* generic fallback, so a preset that
//! referenced a machine-local font would look right on the author's
//! machine and wrong everywhere else. [`BUNDLED_FONTS`] is the allowlist
//! of families every install resolves identically (OFL faces shipped with
//! Cutlass); any other family is dropped to the default sans **with a
//! visible warning** here — documented fallback, never a silent surprise.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crossbeam_channel::{Sender, unbounded};
use cutlass_cloud::CloudClient;
use cutlass_cloud::dto::{TextPreset, TextPresetCatalog};
use cutlass_storage::{CacheId, SharedStorageLayout, StorageLayoutLease};
use slint::{ComponentHandle, ModelRc, VecModel};
use tracing::warn;

use crate::{TextPresetTile, TextPresetsBackend};

/// Font families every Cutlass install resolves identically (OFL faces
/// bundled with the app). Empty until the first font pack ships — presets
/// therefore use the default sans (`font_family: ""`) for now.
const BUNDLED_FONTS: &[&str] = &[];

/// Preset id → preset, shared between the fetch worker (writer) and the
/// timeline drop resolver in `main.rs` (reader).
pub type PresetRegistry = Arc<Mutex<HashMap<String, TextPreset>>>;

enum Command {
    Refresh,
}

/// Cheap, cloneable sender to the presets thread.
#[derive(Clone)]
pub struct TextPresetsHandle {
    tx: Sender<Command>,
}

impl TextPresetsHandle {
    pub fn refresh(&self) {
        let _ = self.tx.send(Command::Refresh);
    }
}

pub struct TextPresetsWorker {
    handle: TextPresetsHandle,
    _join: JoinHandle<()>,
}

impl TextPresetsWorker {
    pub fn spawn(
        backend_weak: slint::Weak<crate::AppWindow>,
        registry: PresetRegistry,
        storage_layout: SharedStorageLayout,
    ) -> Result<Self, String> {
        let (tx, rx) = unbounded::<Command>();
        let join = std::thread::Builder::new()
            .name("cutlass-text-presets".into())
            .spawn(move || {
                let worker = Worker::new(backend_weak, registry, storage_layout);
                while let Ok(Command::Refresh) = rx.recv() {
                    worker.refresh();
                }
            })
            .map_err(|e| e.to_string())?;

        Ok(Self {
            handle: TextPresetsHandle { tx },
            _join: join,
        })
    }

    pub fn handle(&self) -> TextPresetsHandle {
        self.handle.clone()
    }
}

struct Worker {
    backend_weak: slint::Weak<crate::AppWindow>,
    registry: PresetRegistry,
    storage_layout: SharedStorageLayout,
    base_url: String,
}

impl Worker {
    fn new(
        backend_weak: slint::Weak<crate::AppWindow>,
        registry: PresetRegistry,
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
        // Keep the catalog response and every downloaded/parsed pack pinned
        // through registry and UI publication.
        let layout_lease = self.storage_layout.lease();
        let catalog_root = match catalog_root(&layout_lease) {
            Ok(root) => root,
            Err(message) => {
                self.set_status("error", message);
                return;
            }
        };
        self.set_status("loading", "");
        let client = CloudClient::new(&self.base_url, Some(catalog_root.clone()));
        let entries = match client.text_presets() {
            Ok(catalog) => catalog.entries,
            Err(e) => {
                self.set_status("error", &user_message(&e));
                return;
            }
        };

        // Each catalog entry is a pack whose file is a TextPresetCatalog
        // JSON; a bad pack skips (a served file must not brick the section).
        let mut presets: Vec<TextPreset> = Vec::new();
        for entry in &entries {
            match self.fetch_pack(&entry.file_url, &catalog_root) {
                Ok(pack) => presets.extend(pack.presets),
                Err(e) => warn!(pack = %entry.id, "text preset pack skipped: {e}"),
            }
        }
        for preset in &mut presets {
            enforce_bundled_fonts(preset);
        }

        {
            let mut registry = self.registry.lock().expect("preset registry poisoned");
            registry.clear();
            for preset in &presets {
                registry.insert(preset.id.clone(), preset.clone());
            }
        }

        let seeds: Vec<TileSeed> = presets.iter().map(TileSeed::from).collect();
        let status = if seeds.is_empty() { "empty" } else { "results" };
        let status = status.to_string();
        self.on_ui(move |backend| {
            let rows: Vec<TextPresetTile> = seeds
                .iter()
                .map(|seed| TextPresetTile {
                    key: seed.key.as_str().into(),
                    name: seed.name.as_str().into(),
                    sample: seed.sample.as_str().into(),
                    fill: slint::Color::from_argb_u8(
                        seed.fill[3],
                        seed.fill[0],
                        seed.fill[1],
                        seed.fill[2],
                    ),
                    animated: seed.animated,
                })
                .collect();
            backend.set_items(ModelRc::new(VecModel::from(rows)));
            backend.set_status(status.as_str().into());
            backend.set_error("".into());
        });
        drop(layout_lease);
    }

    /// Download a pack file (small JSON, cold path) and parse it. Fetched
    /// fresh on every refresh — packs are tiny and the catalog decides
    /// freshness, not the client.
    fn fetch_pack(&self, url: &str, catalog_root: &Path) -> Result<TextPresetCatalog, String> {
        let dest = catalog_root.join("text-preset-pack.json");
        let _ = std::fs::remove_file(&dest);
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        cutlass_cloud::download::download_to(url, &dest, &cancel, |_| {})
            .map_err(|e| e.to_string())?;
        let bytes = std::fs::read(&dest).map_err(|e| e.to_string())?;
        serde_json::from_slice(&bytes).map_err(|e| e.to_string())
    }

    fn set_status(&self, status: &str, error: &str) {
        let status = status.to_string();
        let error = error.to_string();
        self.on_ui(move |backend| {
            backend.set_status(status.as_str().into());
            backend.set_error(error.as_str().into());
        });
    }

    fn on_ui(&self, f: impl FnOnce(TextPresetsBackend<'_>) + Send + 'static) {
        let weak = self.backend_weak.clone();
        if let Err(e) = slint::invoke_from_event_loop(move || {
            if let Some(app) = weak.upgrade() {
                f(app.global::<TextPresetsBackend>());
            }
        }) {
            warn!("text presets UI update failed: {e}");
        }
    }
}

fn catalog_root(lease: &StorageLayoutLease<'_>) -> Result<PathBuf, &'static str> {
    lease
        .resolve(CacheId::Catalog)
        .ok_or("catalog cache has no disk path")
}

/// The bundled-OFL-fonts-only policy: a family outside [`BUNDLED_FONTS`]
/// falls back to the default sans, loudly (the renderer's own fallback is
/// silent — this is where the surprise gets documented).
fn enforce_bundled_fonts(preset: &mut TextPreset) {
    if preset.font_family.is_empty() {
        return;
    }
    if !BUNDLED_FONTS.contains(&preset.font_family.as_str()) {
        warn!(
            preset = %preset.id,
            font = %preset.font_family,
            "preset font is not bundled with Cutlass; falling back to the default sans"
        );
        preset.font_family.clear();
    }
}

/// Build the styled text generator a preset drop places. Falls back to
/// "Title" when the preset carries no sample text.
pub fn generator_for(preset: &TextPreset) -> cutlass_models::Generator {
    let content = if preset.sample_text.is_empty() {
        "Title".to_string()
    } else {
        preset.sample_text.clone()
    };
    let style = cutlass_models::TextStyle {
        font: preset.font_family.clone(),
        size: preset.font_size,
        fill: preset.fill,
        ..Default::default()
    };
    cutlass_models::Generator::Text { content, style }
}

/// The `(slot, catalog id)` pairs a preset drop attaches to its fresh clip.
pub fn animations_for(preset: &TextPreset) -> Vec<(String, String)> {
    [
        ("in", &preset.animation_in),
        ("out", &preset.animation_out),
        ("combo", &preset.animation_combo),
    ]
    .into_iter()
    .filter_map(|(slot, id)| {
        id.as_deref()
            .filter(|id| !id.is_empty())
            .map(|id| (slot.to_string(), id.to_string()))
    })
    .collect()
}

/// Send-safe snapshot of a preset's tile fields.
struct TileSeed {
    key: String,
    name: String,
    sample: String,
    fill: [u8; 4],
    animated: bool,
}

impl From<&TextPreset> for TileSeed {
    fn from(preset: &TextPreset) -> Self {
        Self {
            key: preset.id.clone(),
            name: preset.name.clone(),
            sample: if preset.sample_text.is_empty() {
                preset.name.clone()
            } else {
                preset.sample_text.clone()
            },
            fill: preset.fill,
            animated: preset.animation_in.is_some()
                || preset.animation_out.is_some()
                || preset.animation_combo.is_some(),
        }
    }
}

fn user_message(e: &cutlass_cloud::CloudError) -> String {
    use cutlass_cloud::CloudError;
    match e {
        CloudError::Network(_) => {
            "Couldn't reach the preset catalog — check your connection.".into()
        }
        CloudError::Status {
            status, retryable, ..
        } => {
            if *retryable {
                "The preset catalog is busy — try again in a moment.".into()
            } else {
                format!("The preset catalog rejected the request ({status}).")
            }
        }
        CloudError::Protocol(_) => "The preset catalog sent an unexpected response.".into(),
        CloudError::Io(_) | CloudError::Cancelled => "The catalog fetch was interrupted.".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_storage::StorageLayout;

    fn preset() -> TextPreset {
        TextPreset {
            id: "tp-1".into(),
            name: "Neon pop".into(),
            category: "titles".into(),
            font_family: String::new(),
            font_size: 72.0,
            fill: [255, 64, 128, 255],
            animation_in: Some("fade_in".into()),
            animation_out: None,
            animation_combo: None,
            sample_text: "NEON".into(),
        }
    }

    #[test]
    fn generator_carries_style_and_sample() {
        let g = generator_for(&preset());
        match g {
            cutlass_models::Generator::Text { content, style } => {
                assert_eq!(content, "NEON");
                assert_eq!(style.size, 72.0);
                assert_eq!(style.fill, [255, 64, 128, 255]);
            }
            other => panic!("expected text generator, got {other:?}"),
        }
    }

    #[test]
    fn animations_collect_only_present_slots() {
        assert_eq!(
            animations_for(&preset()),
            vec![("in".to_string(), "fade_in".to_string())]
        );
    }

    #[test]
    fn unbundled_fonts_fall_back_loudly() {
        let mut p = preset();
        p.font_family = "Comic Sans MS".into();
        enforce_bundled_fonts(&mut p);
        assert!(p.font_family.is_empty());
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
