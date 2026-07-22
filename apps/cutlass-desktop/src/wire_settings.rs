//! Callback wiring extracted from `main` — structural split only.
#![allow(unused_imports)]

use std::cell::Cell;
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use cutlass_engine::EngineConfig;
use slint::ComponentHandle;
use slint::Global;
use slint::Model;
use slint::ModelRc;
use slint::SharedString;
use slint::VecModel;
use slint::winit_030::EventResult;
use slint::winit_030::WinitWindowAccessor;
use slint::winit_030::winit::event::WindowEvent;

use crate::bootstrap::*;
use crate::cache_ui::*;
use crate::library_helpers::*;
use crate::session::*;
use crate::*;

pub(crate) fn wire_settings(
    app: &AppWindow,
    config_path: PathBuf,
    app_settings: &cutlass_settings::Settings,
    download_quota_mib: u64,
    cache_registry: &crate::cache_registry::CacheRegistry,
    download_cache: &Arc<cutlass_cloud::cache::DownloadCache>,
    preview_worker: &crate::preview_worker::PreviewWorker,
) {
    // --- app settings (gear / Cutlass menu → dialog → config.toml) -------

    let settings_backend = app.global::<SettingsBackend>();

    // Seed the dialog from the loaded config. The theme rides AppStore so it
    // drives the live theme binding the whole shell reads.
    settings_backend.set_config_path(config_path.display().to_string().into());
    settings_backend.set_ai_source(app_settings.ai.source.key().into());
    settings_backend.set_ai_base_url(app_settings.ai.base_url.clone().into());
    settings_backend.set_ai_model(app_settings.ai.model.clone().into());
    settings_backend.set_ai_api_protocol(app_settings.ai.api_protocol.key().into());
    settings_backend.set_ai_reasoning_summary(app_settings.ai.reasoning_summary.key().into());
    settings_backend.set_ai_api_key(app_settings.ai.api_key.clone().unwrap_or_default().into());
    settings_backend.set_ai_api_key_env(
        app_settings
            .ai
            .api_key_env
            .clone()
            .unwrap_or_default()
            .into(),
    );
    publish_ai_model_rows(&settings_backend, &app_settings.ai, &[]);
    let storage_root = cache_registry.storage_root();
    settings_backend.set_storage_root(storage_root.to_string_lossy().into_owned().into());
    settings_backend.set_download_quota_mib(download_quota_mib.to_string().into());
    settings_backend.set_cache_relocation_enabled(true);
    settings_backend.set_storage_root_relocation_enabled(false);
    app.global::<AppStore>()
        .set_theme_id(app_settings.appearance.theme.index());

    let cache_ui_generation = Arc::new(AtomicU64::new(0));

    // Cache snapshots can touch the filesystem, preview worker, and Slint-owned
    // image caches. Always collect them on one short-lived named worker.
    {
        let app_weak = app.as_weak();
        let registry = cache_registry.clone();
        let generation = Arc::clone(&cache_ui_generation);
        settings_backend.on_refresh_caches(move || {
            let Some(app) = app_weak.upgrade() else {
                return;
            };
            let sb = app.global::<SettingsBackend>();
            if sb.get_cache_loading() || !sb.get_cache_busy_id().is_empty() {
                return;
            }
            let operation_generation = match next_cache_generation(&generation) {
                Ok(value) => value,
                Err(error) => {
                    sb.set_cache_status(SharedString::new());
                    sb.set_cache_error(error.into());
                    return;
                }
            };

            sb.set_cache_loading(true);
            sb.set_cache_status(SharedString::new());
            sb.set_cache_error(SharedString::new());

            let worker_app = app_weak.clone();
            let worker_registry = registry.clone();
            let worker_generation = Arc::clone(&generation);
            if let Err(error) = spawn_short_lived_worker("cutlass-cache-refresh", move || {
                let cancel = AtomicBool::new(false);
                let result = worker_registry
                    .snapshot_all(&cancel)
                    .and_then(cache_rows_from_snapshots);
                let apply_generation = Arc::clone(&worker_generation);
                if let Err(error) = slint::invoke_from_event_loop(move || {
                    if apply_generation.load(Ordering::Acquire) != operation_generation {
                        return;
                    }
                    let Some(app) = worker_app.upgrade() else {
                        return;
                    };
                    let sb = app.global::<SettingsBackend>();
                    sb.set_cache_loading(false);
                    match result {
                        Ok(rows) => {
                            sb.set_cache_rows(ModelRc::new(VecModel::from(rows)));
                            sb.set_cache_status("Cache usage refreshed.".into());
                            sb.set_cache_error(SharedString::new());
                        }
                        Err(error) => {
                            tracing::warn!(%error, "cache inventory refresh failed");
                            sb.set_cache_status(SharedString::new());
                            sb.set_cache_error(
                                format!(
                                    "Cache usage could not be refreshed: {}",
                                    bounded_cache_ui_error(&error)
                                )
                                .into(),
                            );
                        }
                    }
                }) {
                    tracing::debug!(%error, "cache refresh event-loop publish failed");
                }
            }) {
                tracing::error!(%error, "cache refresh worker could not start");
                sb.set_cache_loading(false);
                sb.set_cache_error("Cache refresh could not start.".into());
            }
        });
    }

    // Clear one exact registry id, then collect the complete inventory on the
    // same worker. A successful clear remains successful even if that
    // follow-up snapshot is unavailable.
    {
        let app_weak = app.as_weak();
        let registry = cache_registry.clone();
        let generation = Arc::clone(&cache_ui_generation);
        settings_backend.on_clear_cache(move |cache_id| {
            let Some(app) = app_weak.upgrade() else {
                return;
            };
            let sb = app.global::<SettingsBackend>();
            let id = match cutlass_storage::CacheId::parse(cache_id.as_str()) {
                Ok(id) => id,
                Err(_) => {
                    sb.set_cache_status(SharedString::new());
                    sb.set_cache_error("Unknown cache.".into());
                    return;
                }
            };
            if id.descriptor().tier == cutlass_storage::CacheTier::UserData {
                sb.set_cache_status(SharedString::new());
                sb.set_cache_error("User data cannot be cleared.".into());
                return;
            }
            if sb.get_cache_loading() || !sb.get_cache_busy_id().is_empty() {
                return;
            }
            let operation_generation = match next_cache_generation(&generation) {
                Ok(value) => value,
                Err(error) => {
                    sb.set_cache_status(SharedString::new());
                    sb.set_cache_error(error.into());
                    return;
                }
            };

            sb.set_cache_busy_id(id.as_str().into());
            sb.set_cache_status(SharedString::new());
            sb.set_cache_error(SharedString::new());

            let worker_app = app_weak.clone();
            let worker_registry = registry.clone();
            let worker_generation = Arc::clone(&generation);
            if let Err(error) = spawn_short_lived_worker("cutlass-cache-clear", move || {
                let cancel = AtomicBool::new(false);
                let clear_result = worker_registry.clear(id, &cancel);
                let rows_result = worker_registry
                    .snapshot_all(&cancel)
                    .and_then(cache_rows_from_snapshots);

                if let Err(error) = &clear_result {
                    tracing::warn!(cache = id.as_str(), %error, "cache clear did not complete");
                }
                if let Err(error) = &rows_result {
                    tracing::warn!(
                        cache = id.as_str(),
                        %error,
                        "cache inventory refresh after clear failed"
                    );
                }

                let (clear_succeeded, status, mut feedback_error) = match clear_result {
                    Ok(report) => (true, cache_clear_success(&report), String::new()),
                    Err(error) => (
                        false,
                        String::new(),
                        format!(
                            "Could not fully clear {}: {}",
                            id.descriptor().label,
                            bounded_cache_ui_error(&error)
                        ),
                    ),
                };
                if rows_result.is_err() {
                    if clear_succeeded {
                        feedback_error =
                            "Cache cleared, but its usage could not be refreshed.".into();
                    } else {
                        feedback_error.push_str(" Cache usage also could not be refreshed.");
                    }
                }

                let apply_generation = Arc::clone(&worker_generation);
                if let Err(error) = slint::invoke_from_event_loop(move || {
                    if apply_generation.load(Ordering::Acquire) != operation_generation {
                        return;
                    }
                    let Some(app) = worker_app.upgrade() else {
                        return;
                    };
                    let sb = app.global::<SettingsBackend>();
                    sb.set_cache_busy_id(SharedString::new());
                    if let Ok(rows) = rows_result {
                        sb.set_cache_rows(ModelRc::new(VecModel::from(rows)));
                    }
                    sb.set_cache_status(status.into());
                    sb.set_cache_error(feedback_error.into());
                }) {
                    tracing::debug!(%error, "cache clear event-loop publish failed");
                }
            }) {
                tracing::error!(%error, "cache clear worker could not start");
                sb.set_cache_busy_id(SharedString::new());
                sb.set_cache_error("Cache clear could not start.".into());
            }
        });
    }

    // Revealing is disk-only. Create a missing cache root and invoke the
    // platform file browser from a worker; no shell is involved.
    {
        let app_weak = app.as_weak();
        let registry = cache_registry.clone();
        let generation = Arc::clone(&cache_ui_generation);
        settings_backend.on_reveal_cache(move |cache_id| {
            let Some(app) = app_weak.upgrade() else {
                return;
            };
            let sb = app.global::<SettingsBackend>();
            let id = match cutlass_storage::CacheId::parse(cache_id.as_str()) {
                Ok(id) => id,
                Err(_) => {
                    sb.set_cache_status(SharedString::new());
                    sb.set_cache_error("Unknown cache.".into());
                    return;
                }
            };
            let path = match registry.cache_path(id) {
                Ok(path) => path,
                Err(error) => {
                    sb.set_cache_status(SharedString::new());
                    sb.set_cache_error(
                        format!(
                            "{} cannot be revealed: {}",
                            id.descriptor().label,
                            bounded_cache_ui_error(&error)
                        )
                        .into(),
                    );
                    return;
                }
            };
            if sb.get_cache_loading() || !sb.get_cache_busy_id().is_empty() {
                return;
            }
            let operation_generation = match next_cache_generation(&generation) {
                Ok(value) => value,
                Err(error) => {
                    sb.set_cache_status(SharedString::new());
                    sb.set_cache_error(error.into());
                    return;
                }
            };

            sb.set_cache_busy_id(id.as_str().into());
            sb.set_cache_status(SharedString::new());
            sb.set_cache_error(SharedString::new());

            let worker_app = app_weak.clone();
            let worker_generation = Arc::clone(&generation);
            if let Err(error) = spawn_short_lived_worker("cutlass-cache-reveal", move || {
                let result = std::fs::create_dir_all(&path)
                    .map_err(|error| format!("could not create the cache directory: {error}"))
                    .and_then(|()| external::reveal_path(&path));
                if let Err(error) = &result {
                    tracing::warn!(
                        cache = id.as_str(),
                        %error,
                        "cache path could not be revealed"
                    );
                }

                let apply_generation = Arc::clone(&worker_generation);
                if let Err(error) = slint::invoke_from_event_loop(move || {
                    if apply_generation.load(Ordering::Acquire) != operation_generation {
                        return;
                    }
                    let Some(app) = worker_app.upgrade() else {
                        return;
                    };
                    let sb = app.global::<SettingsBackend>();
                    sb.set_cache_busy_id(SharedString::new());
                    match result {
                        Ok(()) => {
                            sb.set_cache_status(
                                format!("Revealed {} in the file browser.", id.descriptor().label)
                                    .into(),
                            );
                            sb.set_cache_error(SharedString::new());
                        }
                        Err(error) => {
                            sb.set_cache_status(SharedString::new());
                            sb.set_cache_error(
                                format!(
                                    "Could not reveal {}: {}",
                                    id.descriptor().label,
                                    bounded_cache_ui_error(&error)
                                )
                                .into(),
                            );
                        }
                    }
                }) {
                    tracing::debug!(%error, "cache reveal event-loop publish failed");
                }
            }) {
                tracing::error!(%error, "cache reveal worker could not start");
                sb.set_cache_busy_id(SharedString::new());
                sb.set_cache_error("Cache reveal could not start.".into());
            }
        });
    }

    // Relocate one exact disk cache. The picker chooses an existing parent;
    // the registry receives only the derived, absent cache-specific child.
    // Busy state covers the asynchronous picker and the background move so no
    // other cache or settings-persistence operation can overlap it.
    {
        let app_weak = app.as_weak();
        let registry = cache_registry.clone();
        let generation = Arc::clone(&cache_ui_generation);
        let config_path = config_path.clone();
        let picker_directory = storage_root
            .parent()
            .filter(|directory| directory.is_dir())
            .map(std::path::Path::to_path_buf)
            .or_else(dirs::home_dir);
        settings_backend.on_relocate_cache(move |cache_id| {
            let Some(app) = app_weak.upgrade() else {
                return;
            };
            let sb = app.global::<SettingsBackend>();
            let id = match cutlass_storage::CacheId::parse(cache_id.as_str()) {
                Ok(id) => id,
                Err(_) => {
                    sb.set_cache_status(SharedString::new());
                    sb.set_cache_error("Unknown cache.".into());
                    return;
                }
            };
            if !cache_relocation_supported(id) {
                sb.set_cache_status(SharedString::new());
                sb.set_cache_error("This cache cannot be moved.".into());
                return;
            }
            if sb.get_cache_loading() || !sb.get_cache_busy_id().is_empty() {
                return;
            }
            let operation_generation = match next_cache_generation(&generation) {
                Ok(value) => value,
                Err(error) => {
                    sb.set_cache_status(SharedString::new());
                    sb.set_cache_error(error.into());
                    return;
                }
            };

            sb.set_cache_busy_id(id.as_str().into());
            sb.set_cache_status(SharedString::new());
            sb.set_cache_error(SharedString::new());

            let dialog_app = app_weak.clone();
            let dialog_registry = registry.clone();
            let dialog_generation = Arc::clone(&generation);
            let dialog_config_path = config_path.clone();
            let dialog_directory = picker_directory.clone();
            let task = slint::spawn_local(async move {
                let selected_parent =
                    pick_cache_relocation_parent(id.descriptor().label, dialog_directory).await;
                if dialog_generation.load(Ordering::Acquire) != operation_generation {
                    return;
                }
                let Some(app) = dialog_app.upgrade() else {
                    return;
                };
                let sb = app.global::<SettingsBackend>();
                let Some(selected_parent) = selected_parent else {
                    sb.set_cache_busy_id(SharedString::new());
                    sb.set_cache_error(SharedString::new());
                    return;
                };
                let destination = match cache_relocation_destination(&selected_parent, id) {
                    Ok(destination) => destination,
                    Err(error) => {
                        sb.set_cache_busy_id(SharedString::new());
                        sb.set_cache_status(SharedString::new());
                        sb.set_cache_error(bounded_cache_ui_error(error).into());
                        return;
                    }
                };

                let worker_app = dialog_app.clone();
                let worker_registry = dialog_registry.clone();
                let worker_generation = Arc::clone(&dialog_generation);
                if let Err(error) = spawn_short_lived_worker("cutlass-cache-relocate", move || {
                    let cancel = AtomicBool::new(false);
                    let relocation_result =
                        worker_registry.relocate(id, &destination, &dialog_config_path, &cancel);
                    let rows_result = worker_registry
                        .snapshot_all(&cancel)
                        .and_then(cache_rows_from_snapshots);

                    if let Err(error) = &relocation_result {
                        tracing::warn!(
                            cache = id.as_str(),
                            %error,
                            "cache relocation did not complete"
                        );
                    }
                    if let Err(error) = &rows_result {
                        tracing::warn!(
                            cache = id.as_str(),
                            %error,
                            "cache inventory refresh after relocation failed"
                        );
                    }

                    let relocation_succeeded = relocation_result.is_ok();
                    let refresh_failed = rows_result.is_err();
                    let (status, mut feedback_error) = match relocation_result {
                        Ok(report) => (cache_relocation_success(&report), String::new()),
                        Err(error) => (
                            String::new(),
                            format!("Could not move {}: {error}", id.descriptor().label),
                        ),
                    };
                    if refresh_failed {
                        if relocation_succeeded {
                            feedback_error =
                                "Cache moved, but cache usage could not be refreshed.".into();
                        } else {
                            feedback_error.push_str(" Cache usage also could not be refreshed.");
                        }
                    }
                    if !feedback_error.is_empty() {
                        feedback_error = bounded_cache_ui_error(&feedback_error);
                    }

                    let apply_generation = Arc::clone(&worker_generation);
                    if let Err(error) = slint::invoke_from_event_loop(move || {
                        if apply_generation.load(Ordering::Acquire) != operation_generation {
                            return;
                        }
                        let Some(app) = worker_app.upgrade() else {
                            return;
                        };
                        let sb = app.global::<SettingsBackend>();
                        sb.set_cache_busy_id(SharedString::new());
                        if let Ok(rows) = rows_result {
                            sb.set_cache_rows(ModelRc::new(VecModel::from(rows)));
                        }
                        sb.set_cache_status(status.into());
                        sb.set_cache_error(feedback_error.into());
                    }) {
                        tracing::debug!(%error, "cache relocation event-loop publish failed");
                    }
                }) {
                    tracing::error!(%error, "cache relocation worker could not start");
                    if dialog_generation.load(Ordering::Acquire) == operation_generation {
                        sb.set_cache_busy_id(SharedString::new());
                        sb.set_cache_status(SharedString::new());
                        sb.set_cache_error("Cache move could not start.".into());
                    }
                }
            });
            if let Err(error) = task {
                tracing::error!(%error, "cache relocation dialog could not open");
                if generation.load(Ordering::Acquire) == operation_generation {
                    sb.set_cache_busy_id(SharedString::new());
                    sb.set_cache_status(SharedString::new());
                    sb.set_cache_error("Cache move dialog could not open.".into());
                }
            }
        });
    }

    // Save returns whether dismissal is safe. Load-then-patch preserves
    // unknown TOML, and a malformed existing file is never replaced.
    {
        let app_weak = app.as_weak();
        let config_path = config_path.clone();
        let download_cache = Arc::clone(download_cache);
        let preview = preview_worker.handle();
        let registry = cache_registry.clone();
        settings_backend.on_save(move || {
            let Some(app) = app_weak.upgrade() else {
                return false;
            };
            let sb = app.global::<SettingsBackend>();
            if !sb.get_cache_busy_id().is_empty() {
                sb.set_save_error(
                    "Wait for the cache operation to finish before saving Settings.".into(),
                );
                return false;
            }
            sb.set_save_error(SharedString::new());
            let quota = match parse_download_quota_mib(&sb.get_download_quota_mib()) {
                Ok(quota) => quota,
                Err(error) => {
                    sb.set_save_error(error.into());
                    return false;
                }
            };
            let ai_source =
                cutlass_settings::AiSource::from_key(&sb.get_ai_source()).unwrap_or_default();
            let ai_base_url = sb.get_ai_base_url().trim().to_string();
            let ai_model = sb.get_ai_model().trim().to_string();
            let ai_api_protocol =
                cutlass_settings::AiApiProtocol::from_key(&sb.get_ai_api_protocol())
                    .unwrap_or_default();
            let ai_reasoning_summary =
                cutlass_settings::ReasoningSummary::from_key(&sb.get_ai_reasoning_summary())
                    .unwrap_or_default();
            let ai_api_key = non_empty(&sb.get_ai_api_key());
            let ai_api_key_env = non_empty(&sb.get_ai_api_key_env());
            let draft = cutlass_settings::AiSettings {
                source: ai_source,
                base_url: ai_base_url.clone(),
                model: ai_model.clone(),
                api_protocol: ai_api_protocol,
                reasoning_summary: ai_reasoning_summary,
                api_key: ai_api_key.clone(),
                api_key_env: ai_api_key_env.clone(),
                autonomy: cutlass_settings::Autonomy::default(),
            };
            // Allow saving a partial draft so users can switch modes; reject
            // only when fields look ready but fail allowlist / key checks.
            if draft.is_configured()
                && let Err(error) = cutlass_ai::config::validate_ai_settings(&draft)
            {
                sb.set_save_error(error.into());
                return false;
            }
            // Preserve autonomy from disk — Settings UI does not edit it.
            let theme =
                cutlass_settings::ThemeChoice::from_index(app.global::<AppStore>().get_theme_id());
            let persisted = registry.try_with_settings_persistence(|| {
                let mut settings = cutlass_settings::load(&config_path)
                    .map_err(|error| ("load", error.to_string()))?;
                let autonomy = settings.ai.autonomy;
                settings.ai = draft;
                settings.ai.autonomy = autonomy;
                settings.appearance.theme = theme;
                settings.storage.download_quota_mib = quota.mib;
                cutlass_settings::save(&config_path, &settings)
                    .map_err(|error| ("save", error.to_string()))?;
                Ok::<_, (&'static str, String)>(settings.ai.is_configured())
            });
            let configured = match persisted {
                Err(error) => {
                    tracing::warn!(%error, "settings save deferred by cache maintenance");
                    sb.set_save_error(
                        "Wait for the active cache operation to finish before saving Settings."
                            .into(),
                    );
                    return false;
                }
                Ok(Err(("load", error))) => {
                    tracing::error!(%error, "refusing to overwrite unreadable settings");
                    sb.set_save_error(
                        "Settings could not be saved because the configuration file is invalid."
                            .into(),
                    );
                    return false;
                }
                Ok(Err((_, error))) => {
                    tracing::error!(%error, "failed to save settings");
                    sb.set_save_error(
                        "Settings could not be saved. Check the configuration file.".into(),
                    );
                    return false;
                }
                Ok(Ok(configured)) => configured,
            };

            download_cache.set_quota_bytes(quota.bytes);
            let quota_cache = Arc::clone(&download_cache);
            let quota_preview = preview.clone();
            if let Err(error) = spawn_short_lived_worker("cutlass-cache-quota", move || {
                let Some(project) = quota_preview.snapshot_project() else {
                    tracing::warn!("download quota enforcement skipped: project unavailable");
                    return;
                };
                let protected = download_safety::protect_project_downloads(&quota_cache, &project);
                if protected.rejected != 0 {
                    tracing::warn!(
                        rejected = protected.rejected,
                        "download quota enforcement skipped: project media protection incomplete"
                    );
                    return;
                }
                quota_cache.enforce_quota();
            }) {
                // Persistence and the live quota update are already committed;
                // do not turn that success into a false save failure.
                tracing::error!(%error, "download quota enforcement worker could not start");
            }
            sb.set_download_quota_mib(quota.mib.to_string().into());
            app.global::<AgentStore>().set_configured(configured);
            true
        });
    }

    {
        let app_weak = app.as_weak();
        settings_backend.on_test_connection(move || {
            let Some(app) = app_weak.upgrade() else {
                return;
            };
            let sb = app.global::<SettingsBackend>();
            let draft = ai_draft_from_backend(&sb);
            sb.set_ai_testing(true);
            sb.set_ai_test_ok(false);
            sb.set_ai_test_status(SharedString::new());

            let app_weak = app.as_weak();
            std::thread::spawn(move || {
                let source = draft.source;
                let result = cutlass_ai::config::provider_from_ai(&draft).and_then(|provider| {
                    let message = provider.test_connection().map_err(|e| e.to_string())?;
                    let installed = if source == cutlass_settings::AiSource::Local {
                        provider.list_models().unwrap_or_default()
                    } else {
                        Vec::new()
                    };
                    Ok((message, installed))
                });
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(app) = app_weak.upgrade() {
                        let sb = app.global::<SettingsBackend>();
                        sb.set_ai_testing(false);
                        match result {
                            Ok((msg, installed)) => {
                                sb.set_ai_test_ok(true);
                                sb.set_ai_test_status(msg.into());
                                if source == cutlass_settings::AiSource::Local {
                                    publish_ai_model_rows(&sb, &draft, &installed);
                                }
                            }
                            Err(e) => {
                                sb.set_ai_test_ok(false);
                                sb.set_ai_test_status(e.into());
                            }
                        }
                    }
                });
            });
        });
    }

    {
        let app_weak = app.as_weak();
        settings_backend.on_refresh_local_models(move || {
            let Some(app) = app_weak.upgrade() else {
                return;
            };
            let sb = app.global::<SettingsBackend>();
            let draft = ai_draft_from_backend(&sb);
            if draft.source != cutlass_settings::AiSource::Local {
                publish_ai_model_rows(&sb, &draft, &[]);
                return;
            }
            let base_url = draft.base_url.trim().to_string();
            if base_url.is_empty() {
                sb.set_ai_test_ok(false);
                sb.set_ai_test_status("Enter a local endpoint URL first.".into());
                publish_ai_model_rows(&sb, &draft, &[]);
                return;
            }
            sb.set_ai_testing(true);
            sb.set_ai_test_status(SharedString::new());
            let app_weak = app.as_weak();
            std::thread::spawn(move || {
                let result = cutlass_ai::providers::OpenAiProvider::new(
                    &base_url,
                    "probe",
                    None,
                    cutlass_ai::providers::OpenAiProtocol::ChatCompletions,
                    cutlass_ai::providers::ReasoningSummary::Off,
                )
                .list_models()
                .map_err(|e| e.to_string());
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(app) = app_weak.upgrade() {
                        let sb = app.global::<SettingsBackend>();
                        sb.set_ai_testing(false);
                        match result {
                            Ok(installed) => {
                                let available = cutlass_ai::local_models_availability(&installed)
                                    .iter()
                                    .filter(|(_, ok, _)| *ok)
                                    .count();
                                sb.set_ai_test_ok(true);
                                sb.set_ai_test_status(
                                    format!(
                                        "Found {available} supported model(s) of {} installed.",
                                        installed.len()
                                    )
                                    .into(),
                                );
                                publish_ai_model_rows(&sb, &draft, &installed);
                            }
                            Err(e) => {
                                sb.set_ai_test_ok(false);
                                sb.set_ai_test_status(e.into());
                                publish_ai_model_rows(&sb, &draft, &[]);
                            }
                        }
                    }
                });
            });
        });
    }

    // Reveal the config file in the OS file browser.
    {
        let config_path = config_path.clone();
        settings_backend.on_reveal_config(move || {
            let target = if config_path.exists() {
                config_path.clone()
            } else {
                config_path
                    .parent()
                    .map(std::path::Path::to_path_buf)
                    .unwrap_or_else(|| config_path.clone())
            };
            if let Err(error) = external::reveal_path(&target) {
                tracing::error!(%error, "failed to reveal settings file");
            }
        });
    }
}

fn ai_draft_from_backend(sb: &SettingsBackend<'_>) -> cutlass_settings::AiSettings {
    cutlass_settings::AiSettings {
        source: cutlass_settings::AiSource::from_key(&sb.get_ai_source()).unwrap_or_default(),
        base_url: sb.get_ai_base_url().trim().to_string(),
        model: sb.get_ai_model().trim().to_string(),
        api_protocol: cutlass_settings::AiApiProtocol::from_key(&sb.get_ai_api_protocol())
            .unwrap_or_default(),
        reasoning_summary: cutlass_settings::ReasoningSummary::from_key(
            &sb.get_ai_reasoning_summary(),
        )
        .unwrap_or_default(),
        api_key: non_empty(&sb.get_ai_api_key()),
        api_key_env: non_empty(&sb.get_ai_api_key_env()),
        autonomy: cutlass_settings::Autonomy::default(),
    }
}

fn publish_ai_model_rows(
    sb: &SettingsBackend<'_>,
    ai: &cutlass_settings::AiSettings,
    installed: &[String],
) {
    let selected = ai.model.trim();
    // After a Local probe, rewrite the selection to the installed id so the
    // picker highlight (`ai-model == row.id`) stays correct across aliases.
    if ai.source == cutlass_settings::AiSource::Local
        && !installed.is_empty()
        && let Some(entry) = cutlass_ai::local_model(selected)
        && let Some(resolved) = cutlass_ai::resolve_local_installed_id(entry.id, installed)
        && selected != resolved
    {
        sb.set_ai_model(resolved.into());
    }
    let selected = sb.get_ai_model();
    let selected = selected.as_str();
    let rows: Vec<AiModelRow> = match ai.source {
        cutlass_settings::AiSource::Local => cutlass_ai::LOCAL_MODELS
            .iter()
            .map(|entry| {
                let resolved = if installed.is_empty() {
                    None
                } else {
                    cutlass_ai::resolve_local_installed_id(entry.id, installed)
                };
                // Before a probe, treat curated models as selectable.
                let available = installed.is_empty() || resolved.is_some();
                let id = resolved.unwrap_or_else(|| entry.id.to_string());
                AiModelRow {
                    id: id.clone().into(),
                    display: entry.display.into(),
                    vendor: SharedString::new(),
                    available,
                    selected: selected == id || selected == entry.id,
                }
            })
            .collect(),
        cutlass_settings::AiSource::OpenRouter => cutlass_ai::OPENROUTER_MODELS
            .iter()
            .map(|entry| AiModelRow {
                id: entry.id.into(),
                display: entry.display.into(),
                vendor: entry.vendor.into(),
                available: true,
                selected: selected == entry.id,
            })
            .collect(),
        cutlass_settings::AiSource::Custom => Vec::new(),
    };
    sb.set_ai_model_rows(ModelRc::new(VecModel::from(rows)));
}
