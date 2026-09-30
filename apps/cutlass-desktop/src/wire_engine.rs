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

/// Workers and shared handles created during engine bring-up. Kept alive for
/// the process lifetime so background threads are not torn down early.
pub(crate) struct EngineHandles {
    pub preview_worker: crate::preview_worker::PreviewWorker,
    pub download_cache: Arc<cutlass_cloud::cache::DownloadCache>,
    pub cache_registry: crate::cache_registry::CacheRegistry,
    pub audio_system: crate::audio::AudioSystem,
    pub interaction_gate: Arc<crate::interaction::InteractionGate>,
    pub strip_worker: crate::strips::StripWorker,
    _thumbnail_worker: crate::thumbnails::ThumbnailWorker,
    _cloud_worker: crate::cloud::CloudWorker,
    _templates_worker: crate::templates::TemplatesWorker,
    _ai_media_worker: crate::ai_media::AiMediaWorker,
    _updates_worker: crate::updates::UpdatesWorker,
    _text_presets_worker: crate::text_presets::TextPresetsWorker,
    _lottie_worker: crate::lottie_stickers::LottieWorker,
    _sfx_worker: crate::sfx::SfxWorker,
    _lut_worker: crate::lut_catalog::LutWorker,
    _agent_worker: crate::agent::AgentWorker,
}

pub(crate) fn wire_engine(
    app: &AppWindow,
    storage_layout: cutlass_storage::SharedStorageLayout,
    download_quota_bytes: u64,
    job_manager: cutlass_jobs::JobManager,
) -> Result<EngineHandles, slint::PlatformError> {
    let editor = app.global::<EditorStore>();
    let agent_store = app.global::<AgentStore>();

    // --- engine service (preview worker thread) ---------------------------

    // Audio output + master clock (Phase 3). Starts even before a project
    // opens; a machine without a usable output device degrades to the
    // wall-clock transport (`handle.active() == false`).
    let audio_system = audio::AudioSystem::start();

    // Scrub/playback gate: background tile decode + GPU work defers while
    // the user interacts with the preview, so the preview worker never
    // shares the decode engine or iGPU mid-gesture.
    let interaction_gate = interaction::InteractionGate::new();

    // Library tile thumbnails decode on their own thread so imports never
    // stall preview scrubbing. Keep the worker alive for the app's lifetime.
    let thumbnail_worker = thumbnails::ThumbnailWorker::spawn(
        app.global::<EditorStore>().as_weak(),
        interaction_gate.clone(),
    )
    .map_err(slint::PlatformError::Other)?;

    // Timeline clip content (filmstrip frames, waveform tiles) decodes on a
    // third thread: a long strip batch must not delay library tiles, and
    // neither may ever touch the UI or engine threads.
    let strip_worker = strips::StripWorker::spawn(
        app.global::<StripBackend>().as_weak(),
        interaction_gate.clone(),
    )
    .map_err(slint::PlatformError::Other)?;

    // Preview proxies: a fourth thread re-encodes large video imports to
    // small short-GOP files, one job at a time, deferring to the gate.
    // Results route to the preview worker through the slot installed right
    // after it spawns — requests can only originate from that worker, so
    // the slot is always filled before the first result can fire.
    let proxy_ready_slot: std::sync::Arc<std::sync::Mutex<Option<preview_worker::WorkerHandle>>> =
        std::sync::Arc::new(std::sync::Mutex::new(None));
    let ready_slot = proxy_ready_slot.clone();
    let proxy_handle = proxy::spawn(
        interaction_gate.clone(),
        storage_layout.clone(),
        move |media_id, source, proxy| match ready_slot
            .lock()
            .expect("proxy slot poisoned")
            .as_ref()
        {
            Some(handle) => handle.proxy_ready(media_id, source, proxy),
            None => tracing::error!(
                media_id,
                "proxy finished before the preview worker wired up"
            ),
        },
    )
    .map_err(slint::PlatformError::Other)?;

    // The worker thread owns the Engine (decoders aren't Send); the UI talks
    // to it through a message queue and it answers with projection publishes
    // and preview frames via invoke_from_event_loop.
    let preview_store_weak = app.global::<PreviewStore>().as_weak();
    let editor_store_weak = app.global::<EditorStore>().as_weak();

    let (preview_worker, session) = preview_worker::PreviewWorker::spawn(
        EngineConfig::default(),
        preview_store_weak,
        editor_store_weak,
        app.global::<ExportBackend>().as_weak(),
        audio_system.handle(),
        thumbnail_worker.handle(),
        strip_worker.handle(),
        proxy_handle,
    )
    .map_err(slint::PlatformError::Other)?;
    *proxy_ready_slot.lock().expect("proxy slot poisoned") = Some(preview_worker.handle());
    tracing::debug!(
        duration_ticks = session.duration_ticks,
        "engine session ready"
    );

    let download_layout_lease = storage_layout.lease();
    let download_root = download_layout_lease
        .resolve(cutlass_storage::CacheId::Download)
        .ok_or_else(|| slint::PlatformError::from("download cache has no disk path"))?;
    let download_cache = std::sync::Arc::new(cutlass_cloud::cache::DownloadCache::new(
        download_root,
        download_quota_bytes,
    ));
    // Legacy drafts may still reference cache-owned source files directly.
    // Inventory them before any worker can evict or clear downloaded media.
    download_cache.block_destructive_operations();
    let download_inventory =
        download_safety::protect_saved_draft_downloads(&download_cache, &drafts::root_dir());
    if download_inventory.is_complete() {
        download_cache.allow_destructive_operations();
        tracing::info!(
            drafts = download_inventory.projects_loaded,
            protected_media = download_inventory.media.protected,
            "download cache project inventory complete"
        );
    } else {
        tracing::warn!(
            entries = download_inventory.draft_entries_examined,
            loaded = download_inventory.projects_loaded,
            skipped = download_inventory.skipped_or_errored,
            rejected_media = download_inventory.media.rejected,
            draft_limit = download_inventory.draft_limit_reached,
            byte_limit = download_inventory.byte_limit_reached,
            "download cache maintenance remains blocked because project inventory was incomplete"
        );
    }
    drop(download_layout_lease);
    // Keep this owner in main for the upcoming Settings wiring; the agent
    // receives a clone of the same registry and operation gate.
    let cache_registry = cache_registry::CacheRegistry::new(
        storage_layout.clone(),
        app.as_weak(),
        preview_worker.handle(),
        std::sync::Arc::clone(&download_cache),
    )
    .map_err(slint::PlatformError::from)?;

    // Library stock browsing: search + direct-CDN downloads on their own
    // thread (src/cloud.rs); imports route through the preview worker like
    // any local file.
    let cloud_worker = cloud::CloudWorker::spawn(
        app.as_weak(),
        preview_worker.handle(),
        std::sync::Arc::clone(&download_cache),
        storage_layout.clone(),
    )
    .map_err(slint::PlatformError::Other)?;
    {
        let cloud_backend = app.global::<CloudBackend>();
        let search_handle = cloud_worker.handle();
        let search_app = app.as_weak();
        cloud_backend.on_stock_search(move || {
            let Some(app) = search_app.upgrade() else {
                return;
            };
            let backend = app.global::<CloudBackend>();
            let query = backend.get_stock_query().trim().to_string();
            if query.is_empty() {
                return;
            }
            search_handle.search(query, backend.get_stock_kind().as_str());
        });
        let more_handle = cloud_worker.handle();
        cloud_backend.on_stock_load_more(move || more_handle.load_more());
        let import_handle = cloud_worker.handle();
        cloud_backend.on_stock_import(move |index| {
            if index >= 0 {
                import_handle.import(index as usize);
            }
        });
    }

    // Launch-screen templates gallery: catalog fetches, bundle installs, and
    // the pick flow on their own thread (src/templates.rs); the filled
    // template swaps the session through the preview worker.
    let templates_worker = templates::TemplatesWorker::spawn(
        app.as_weak(),
        preview_worker.handle(),
        std::sync::Arc::clone(&download_cache),
        storage_layout.clone(),
    )
    .map_err(slint::PlatformError::Other)?;
    {
        let templates_backend = app.global::<TemplatesBackend>();
        let refresh_handle = templates_worker.handle();
        templates_backend.on_refresh(move |category| refresh_handle.refresh(category.to_string()));
        let use_handle = templates_worker.handle();
        templates_backend.on_use_template(move |index| {
            if index >= 0 {
                use_handle.use_template(index as usize);
            }
        });
    }

    // AI generation (Library AI sections): prompt → job → poll → import on
    // its own thread (src/ai_media.rs), routed BYOK-or-managed.
    let ai_media_worker = ai_media::AiMediaWorker::spawn(
        app.as_weak(),
        preview_worker.handle(),
        std::sync::Arc::clone(&download_cache),
    )
    .map_err(slint::PlatformError::Other)?;
    {
        let ai_backend = app.global::<AiBackend>();
        let generate_handle = ai_media_worker.handle();
        ai_backend.on_generate(move |kind, prompt| {
            generate_handle.generate(kind.to_string(), prompt.trim().to_string());
        });
        let import_handle = ai_media_worker.handle();
        ai_backend.on_import(move |kind, index| {
            if index >= 0 {
                import_handle.import(kind.to_string(), index as usize);
            }
        });
        let route_handle = ai_media_worker.handle();
        ai_backend.on_refresh_route(move || route_handle.refresh_route());
    }

    // Launch-screen update nudge: anonymous latest-version check on its
    // own thread (src/updates.rs).
    let updates_worker =
        updates::UpdatesWorker::spawn(app.as_weak()).map_err(slint::PlatformError::Other)?;
    {
        let update_backend = app.global::<UpdateBackend>();
        let open_handle = updates_worker.handle();
        update_backend.on_open_update(move || open_handle.open_update());
        updates_worker.handle().check();
    }

    // Animated text presets (Library > Text > Presets): catalog fetches on
    // their own thread (src/text_presets.rs); the registry feeds the
    // generated-drop resolver below.
    let text_preset_registry: text_presets::PresetRegistry =
        std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
    let text_presets_worker = text_presets::TextPresetsWorker::spawn(
        app.as_weak(),
        text_preset_registry.clone(),
        storage_layout.clone(),
    )
    .map_err(slint::PlatformError::Other)?;
    {
        let refresh_handle = text_presets_worker.handle();
        app.global::<TextPresetsBackend>()
            .on_refresh(move || refresh_handle.refresh());
    }

    // Lottie stickers (Library > Stickers > Lottie): catalog fetch, file
    // downloads, and frame-0 thumbnails on their own thread
    // (src/lottie_stickers.rs); the registry feeds the drop resolver below.
    let lottie_registry: lottie_stickers::LottieRegistry =
        std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
    let lottie_worker = lottie_stickers::LottieWorker::spawn(
        app.as_weak(),
        lottie_registry.clone(),
        storage_layout.clone(),
    )
    .map_err(slint::PlatformError::Other)?;
    {
        let refresh_handle = lottie_worker.handle();
        app.global::<LottieBackend>()
            .on_refresh(move || refresh_handle.refresh());
    }

    // Sound effects (Library > Audio > Sound effects): catalog fetch on its
    // own thread, lazy download + normal media import on click (src/sfx.rs).
    let sfx_worker = sfx::SfxWorker::spawn(
        app.as_weak(),
        preview_worker.handle(),
        std::sync::Arc::clone(&download_cache),
        storage_layout.clone(),
    )
    .map_err(slint::PlatformError::Other)?;
    {
        let refresh_handle = sfx_worker.handle();
        app.global::<SfxBackend>()
            .on_refresh(move || refresh_handle.refresh());
    }
    {
        let import_handle = sfx_worker.handle();
        app.global::<SfxBackend>()
            .on_import(move |index| import_handle.import(index.max(0) as usize));
    }

    // Cloud LUTs (look inspector > LUT): catalog fetch + `.cube` downloads
    // on their own thread (src/lut_catalog.rs); the registry resolves
    // catalog ids to downloaded files for `set-clip-lut`.
    let lut_registry: lut_catalog::LutRegistry =
        std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
    let lut_worker =
        lut_catalog::LutWorker::spawn(app.as_weak(), lut_registry.clone(), storage_layout.clone())
            .map_err(slint::PlatformError::Other)?;
    {
        let refresh_handle = lut_worker.handle();
        app.global::<InspectorBackend>()
            .on_refresh_luts(move || refresh_handle.refresh());
    }
    {
        let set_lut_handle = preview_worker.handle();
        let registry = lut_registry.clone();
        app.global::<InspectorBackend>()
            .on_set_clip_lut(move |clip_id, lut_id, intensity| {
                let path = if lut_id.is_empty() {
                    String::new()
                } else if is_absolute_cube_path(lut_id.as_str()) {
                    lut_id.to_string()
                } else {
                    let Some(path) = registry
                        .lock()
                        .expect("LUT registry poisoned")
                        .get(lut_id.as_str())
                        .map(|p| p.to_string_lossy().into_owned())
                    else {
                        tracing::warn!(lut = %lut_id, "set-clip-lut ignored: unknown catalog id");
                        return;
                    };
                    path
                };
                set_lut_handle.set_clip_lut(clip_id.to_string(), path, intensity);
            });
    }
    {
        let pick_lut_handle = preview_worker.handle();
        app.global::<InspectorBackend>()
            .on_pick_clip_lut(move |clip_id| {
                let pick_lut_handle = pick_lut_handle.clone();
                let clip_id = clip_id.to_string();
                // Defer past popup teardown so the macOS sheet can present.
                defer_main_thread(move || {
                    let task = slint::spawn_local(async move {
                        if let Some(path) = pick_lut_path().await {
                            pick_lut_handle.set_clip_lut(
                                clip_id,
                                path.to_string_lossy().into_owned(),
                                0.8,
                            );
                        }
                    });
                    if let Err(e) = task {
                        tracing::error!("failed to open LUT file dialog: {e}");
                    }
                });
            });
    }

    let agent_worker = agent::AgentWorker::spawn(
        preview_worker.handle(),
        agent_store.as_weak(),
        app.as_weak(),
        cache_registry.clone(),
        job_manager.clone(),
    )
    .map_err(slint::PlatformError::from)?;

    let agent_send = agent_worker.handle();
    let agent_app = app.as_weak();
    agent_store.on_send(move |prompt| {
        let Some(app) = agent_app.upgrade() else {
            return;
        };
        let timeline = app.global::<TimelineStore>();
        let fps = app.global::<EditorStore>().get_project().sequence.fps;
        let spf = if fps.num > 0 {
            f64::from(fps.den) / f64::from(fps.num)
        } else {
            0.0
        };
        let to_seconds = |tick: i32| f64::from(tick) * spf;
        let context = cutlass_ai::EditorContext {
            selected_clips: timeline
                .get_selected_ids()
                .iter()
                .filter_map(|id| id.parse().ok())
                .collect(),
            playhead_seconds: to_seconds(timeline.get_playhead_tick()),
            in_point_seconds: (timeline.get_range_in_tick() >= 0)
                .then(|| to_seconds(timeline.get_range_in_tick())),
            out_point_seconds: (timeline.get_range_out_tick() >= 0)
                .then(|| to_seconds(timeline.get_range_out_tick())),
        };
        let dry_run = app.global::<AgentStore>().get_dry_run();
        agent_send.prompt(prompt.to_string(), context, dry_run);
    });

    let agent_cancel = agent_worker.handle();
    agent_store.on_cancel(move || agent_cancel.cancel());

    agent_store.on_copy_text(move |text| {
        match arboard::Clipboard::new().and_then(|mut clipboard| clipboard.set_text(text.as_str()))
        {
            Ok(()) => {}
            Err(err) => tracing::warn!(error = %err, "failed to copy agent message"),
        }
    });

    let agent_approve = agent_worker.handle();
    agent_store.on_approve_system_tool(move || agent_approve.approve_system_tool());

    let agent_deny = agent_worker.handle();
    agent_store.on_deny_system_tool(move || agent_deny.deny_system_tool());

    let agent_apply = agent_worker.handle();
    agent_store.on_apply_plan(move || agent_apply.apply_plan());

    let agent_discard = agent_worker.handle();
    agent_store.on_discard_plan(move || agent_discard.discard_plan());

    let agent_new_chat = agent_worker.handle();
    agent_store.on_new_chat(move || agent_new_chat.new_chat());

    let agent_select_chat = agent_worker.handle();
    let agent_select_app = app.as_weak();
    agent_store.on_select_chat(move |label| {
        let Some(app) = agent_select_app.upgrade() else {
            return;
        };
        let store = app.global::<AgentStore>();
        let labels_model = store.get_chat_labels();
        let ids_model = store.get_chat_ids();
        let labels = (0..labels_model.row_count())
            .filter_map(|index| labels_model.row_data(index))
            .collect::<Vec<_>>();
        let ids = (0..ids_model.row_count())
            .filter_map(|index| ids_model.row_data(index))
            .collect::<Vec<_>>();
        if let Some(id) = resolve_chat_id(&labels, &ids, label.as_str()) {
            agent_select_chat.select_chat(id);
        }
    });

    let agent_session = agent_worker.handle();
    let agent_session_app = app.as_weak();
    agent_store.on_session_changed(move || {
        agent_session.cancel();
        let path = agent_session_app.upgrade().and_then(|app| {
            let path = app
                .global::<EditorStore>()
                .get_project_file_path()
                .to_string();
            (!path.is_empty()).then(|| std::path::PathBuf::from(path))
        });
        agent_session.switch_project(path);
    });

    // The initial project can be restored before callback wiring, so seed the
    // worker explicitly instead of waiting for the next session-epoch change.
    let initial_agent_path = {
        let path = app
            .global::<EditorStore>()
            .get_project_file_path()
            .to_string();
        (!path.is_empty()).then(|| std::path::PathBuf::from(path))
    };
    agent_worker.handle().switch_project(initial_agent_path);

    // Per-project agent rules editor (agent panel) → ProjectMetadata via
    // the engine worker; the projection publishes the saved value back to
    // EditorStore.project.agent-rules.
    let rules_handle = preview_worker.handle();
    agent_store.on_set_project_rules(move |rules| {
        rules_handle.set_agent_rules(rules.to_string());
    });

    // Playhead moves (ruler scrub, frame-step keys, Home/End) become preview
    // frame requests; the worker coalesces a burst to the newest tick.
    let frame_handle = preview_worker.handle();
    let scrub_audio = audio_system.handle();
    let scrub_weak = app.as_weak();
    let scrub_gate = interaction_gate.clone();
    editor.on_on_playhead_changed(move |tick| {
        scrub_gate.touch();
        frame_handle.request_frame(i64::from(tick));
        // Scrub audio: a manual playhead move while paused plays a short
        // burst of the sound under the playhead. During playback the master
        // clock drives the playhead, so suppress scrub there — the mixer is
        // already producing that audio.
        let playing = scrub_weak
            .upgrade()
            .is_some_and(|app| app.global::<TimelineStore>().get_playing());
        if !playing {
            scrub_audio.scrub(i64::from(tick));
        }
    });

    // Preview axis / scrub-style drag previews (speed-ramp handle, transition
    // pill, timeline hover): request frames without moving the playhead (no
    // audio). Pause transport for the gesture so playback Frame msgs can't
    // race the scrub request in the worker queue (timeline hover already
    // gates on `!playing` in Slint; inspector/pill paths call this while
    // playing).
    let hover_frame_handle = preview_worker.handle();
    let hover_begin_weak = app.as_weak();
    let hover_paused = Rc::new(Cell::new(false));
    let hover_paused_begin = hover_paused.clone();
    editor.on_on_hover_preview(move |tick| {
        if let Some(app) = hover_begin_weak.upgrade() {
            let timeline = app.global::<TimelineStore>();
            if timeline.get_playing() && !hover_paused_begin.get() {
                app.global::<TimelineActions>().invoke_pause();
                hover_paused_begin.set(true);
            }
        }
        hover_frame_handle.request_frame(i64::from(tick));
    });
    let hover_restore_handle = preview_worker.handle();
    let hover_end_weak = app.as_weak();
    editor.on_on_hover_preview_ended(move || {
        if let Some(app) = hover_end_weak.upgrade() {
            let tick = app.global::<TimelineStore>().get_playhead_tick();
            hover_restore_handle.request_frame(i64::from(tick));
            if hover_paused.get() {
                hover_paused.set(false);
                app.global::<TimelineActions>().invoke_play();
            }
        }
    });

    // Preview surface size (docked panel or fullscreen) → render fit bound.
    // Slint reports logical px; the worker wants physical so a Retina preview
    // doesn't render at half resolution.
    let viewport_handle = preview_worker.handle();
    let viewport_weak = app.as_weak();
    app.global::<PreviewStore>()
        .on_viewport_changed(move |width, height| {
            let scale = viewport_weak
                .upgrade()
                .map(|app| app.window().scale_factor())
                .unwrap_or(1.0);
            let w = (width * scale).round().max(0.0) as u32;
            let h = (height * scale).round().max(0.0) as u32;
            viewport_handle.set_viewport(w, h);
        });

    let drop_handle = preview_worker.handle();
    editor.on_on_clip_dropped(move |media_id, track_id, start_tick, drop_row, insert| {
        drop_handle.add_clip(
            media_id.to_string(),
            track_id.to_string(),
            i64::from(start_tick),
            i64::from(drop_row),
            insert,
        );
    });

    let generated_drop_handle = preview_worker.handle();
    let drop_preset_registry = text_preset_registry.clone();
    let drop_lottie_registry = lottie_registry.clone();
    editor.on_on_generated_dropped(
        move |generator, track_id, start_tick, duration_ticks, drop_row| {
            // "effect:<id>" drops a standalone effect-lane segment: a bare
            // Effect generator whose chain is seeded with the catalog effect.
            let effect = generator
                .strip_prefix("effect:")
                .map(std::string::ToString::to_string);
            // "text-preset:<id>" drops a styled, animated title from the
            // served preset catalog (src/text_presets.rs fills the registry).
            let (generator, animations) =
                if let Some(preset_id) = generator.as_str().strip_prefix("text-preset:") {
                    let registry = drop_preset_registry
                        .lock()
                        .expect("preset registry poisoned");
                    let Some(preset) = registry.get(preset_id) else {
                        tracing::warn!(preset_id, "ignoring drop of unknown text preset");
                        return;
                    };
                    (
                        text_presets::generator_for(preset),
                        text_presets::animations_for(preset),
                    )
                // "lottie:<id>" drops a file-backed Lottie animation from the
                // asset catalog (src/lottie_stickers.rs fills the registry).
                } else if let Some(lottie_id) = generator.as_str().strip_prefix("lottie:") {
                    let registry = drop_lottie_registry
                        .lock()
                        .expect("lottie registry poisoned");
                    let Some(asset) = registry.get(lottie_id) else {
                        tracing::warn!(lottie_id, "ignoring drop of unknown lottie asset");
                        return;
                    };
                    (
                        cutlass_models::Generator::lottie(
                            asset.path.to_string_lossy(),
                            asset.width,
                            asset.height,
                        ),
                        Vec::new(),
                    )
                } else {
                    let Some(generator) = generator_from_key(generator.as_str()) else {
                        tracing::warn!(%generator, "ignoring drop of unknown generator key");
                        return;
                    };
                    (generator, Vec::new())
                };
            generated_drop_handle.add_generated(
                generator,
                track_id.to_string(),
                i64::from(start_tick),
                i64::from(duration_ticks),
                i64::from(drop_row),
                effect,
                animations,
            );
        },
    );

    let magnet_handle = preview_worker.handle();
    editor.on_on_main_magnet_changed(move |enabled| {
        magnet_handle.set_main_magnet(enabled);
    });

    let import_handle = preview_worker.handle();
    editor.on_on_import_clicked(move || {
        let import_handle = import_handle.clone();
        let task = slint::spawn_local(async move {
            for path in pick_import_paths().await {
                import_handle.import(path);
            }
        });
        if let Err(e) = task {
            tracing::error!("failed to open import dialog: {e}");
        }
    });

    // OS file drag-and-drop (Finder / Explorer): winit 0.30 emits one event
    // per file, with no pointer position (the position-carrying drag API is
    // winit 0.31). A gesture's paths are batched and flushed on a short
    // timer; the flush queries the OS cursor (src/os_drop.rs) snapshotted at
    // the first DroppedFile and hit-tests it against the timeline panel: a
    // drop over the timeline imports *and places* the files end-to-end at
    // the cursor's lane row + tick (one undo group), anywhere else keeps the
    // pool-only import. While files hover, a poll timer feeds the cursor to
    // the timeline's landing preview — there are no hover-move events to
    // react to.
    let drop_import_handle = preview_worker.handle();
    let drop_app_weak = app.as_weak();
    // (visual, audio) counts of the hovered files, by extension — the
    // preview targets a video lane unless the whole set is audio.
    let hover_counts: Rc<Cell<(i32, i32)>> = Rc::new(Cell::new((0, 0)));
    let hover_poll: Rc<slint::Timer> = Rc::new(slint::Timer::default());
    let pending_drop: Rc<RefCell<Vec<PathBuf>>> = Rc::new(RefCell::new(Vec::new()));
    let drop_cursor: Rc<Cell<Option<(f32, f32)>>> = Rc::new(Cell::new(None));
    // Clear every hover-preview state the drag set (drop and cancel paths).
    let end_hover = {
        let hover_counts = hover_counts.clone();
        let hover_poll = hover_poll.clone();
        let app_weak = drop_app_weak.clone();
        move || {
            hover_poll.stop();
            hover_counts.set((0, 0));
            if let Some(app) = app_weak.upgrade() {
                let state = app.global::<AppState>();
                state.set_os_drop_hover(false);
                state.set_os_drop_over_timeline(false);
                state.set_os_drop_file_count(0);
                state.set_os_drop_cursor_x(-1.0);
                state.set_os_drop_cursor_y(-1.0);
            }
        }
    };
    app.window().on_winit_window_event(move |window, event| {
        match event {
            WindowEvent::HoveredFile(path) if media_extension_supported(path) => {
                let (visual, audio) = hover_counts.get();
                let counts = if audio_extension(path) {
                    (visual, audio + 1)
                } else {
                    (visual + 1, audio)
                };
                hover_counts.set(counts);
                if let Some(app) = drop_app_weak.upgrade() {
                    let state = app.global::<AppState>();
                    state.set_os_drop_hover(true);
                    state.set_os_drop_file_count(counts.0 + counts.1);
                    state.set_os_drop_lane_kind(if counts.0 == 0 {
                        TrackKind::Audio
                    } else {
                        TrackKind::Video
                    });
                }
                // ~30 Hz cursor poll for the landing preview, running for
                // the hover's lifetime (restarting per file is harmless —
                // HoveredFile only fires once per file, at drag entry).
                let poll_weak = drop_app_weak.clone();
                hover_poll.start(
                    slint::TimerMode::Repeated,
                    Duration::from_millis(33),
                    move || {
                        let Some(app) = poll_weak.upgrade() else {
                            return;
                        };
                        if let Some((x, y)) = app
                            .window()
                            .with_winit_window(os_drop::cursor_in_window)
                            .flatten()
                        {
                            let state = app.global::<AppState>();
                            state.set_os_drop_cursor_x(x);
                            state.set_os_drop_cursor_y(y);
                        }
                    },
                );
            }
            WindowEvent::DroppedFile(path) => {
                end_hover();
                if !media_extension_supported(path) {
                    tracing::warn!(path = %path.display(), "ignored unsupported dropped file");
                } else {
                    let mut pending = pending_drop.borrow_mut();
                    if pending.is_empty() {
                        // First file of the gesture: snapshot the cursor now
                        // (the flush runs a beat later, when the pointer may
                        // have moved on) and schedule one flush for the whole
                        // batch — the backends deliver a multi-file drop
                        // back-to-back from a single OS callback.
                        drop_cursor.set(
                            window
                                .with_winit_window(os_drop::cursor_in_window)
                                .flatten(),
                        );
                        let pending_drop = pending_drop.clone();
                        let drop_cursor = drop_cursor.clone();
                        let app_weak = drop_app_weak.clone();
                        let handle = drop_import_handle.clone();
                        slint::Timer::single_shot(Duration::from_millis(30), move || {
                            let paths = std::mem::take(&mut *pending_drop.borrow_mut());
                            if paths.is_empty() {
                                return;
                            }
                            let target = app_weak.upgrade().and_then(|app| {
                                let cursor = drop_cursor.take()?;
                                os_drop_timeline_target(&app, cursor)
                            });
                            handle.drop_files(paths, target);
                        });
                    }
                    pending.push(path.clone());
                }
            }
            WindowEvent::HoveredFileCancelled => end_hover(),
            _ => {}
        }
        EventResult::Propagate
    });

    // Library asset delete: right-click a media tile → Remove from project.
    // `force` is decided UI-side (unused tile deletes straight away; a used
    // one confirms first, then sends force=true to cascade the clip removals).
    let delete_media_handle = preview_worker.handle();
    editor.on_on_media_deleted(move |media_id, force| {
        delete_media_handle.remove_media(media_id.to_string(), force);
    });

    // Missing-media relink: "Locate…" in the relink dialog or on a tile's
    // missing badge. Same media picker as import; the worker re-probes the
    // chosen file and swaps the entry's path in place.
    let relink_handle = preview_worker.handle();
    editor.on_on_relink_media_requested(move |media_id| {
        let relink_handle = relink_handle.clone();
        let media_id = media_id.to_string();
        let task = slint::spawn_local(async move {
            if let Some(path) = pick_relink_path().await {
                relink_handle.relink_media(media_id, path);
            }
        });
        if let Err(e) = task {
            tracing::error!("failed to open relink dialog: {e}");
        }
    });

    let relink_folder_handle = preview_worker.handle();
    editor.on_on_relink_folder_requested(move || {
        let handle = relink_folder_handle.clone();
        let task = slint::spawn_local(async move {
            if let Some(folder) = pick_relink_folder().await {
                handle.relink_folder(folder);
            }
        });
        if let Err(e) = task {
            tracing::error!("failed to open relink folder dialog: {e}");
        }
    });

    // Undo/redo (toolbar buttons, Cmd/Ctrl+Z / Shift+Z): the worker replays
    // history and republishes the projection.
    let undo_handle = preview_worker.handle();
    editor.on_on_undo(move || {
        undo_handle.undo();
    });

    let redo_handle = preview_worker.handle();
    editor.on_on_redo(move || {
        redo_handle.redo();
    });

    Ok(EngineHandles {
        preview_worker,
        download_cache,
        cache_registry,
        audio_system,
        interaction_gate,
        strip_worker,
        _thumbnail_worker: thumbnail_worker,
        _cloud_worker: cloud_worker,
        _templates_worker: templates_worker,
        _ai_media_worker: ai_media_worker,
        _updates_worker: updates_worker,
        _text_presets_worker: text_presets_worker,
        _lottie_worker: lottie_worker,
        _sfx_worker: sfx_worker,
        _lut_worker: lut_worker,
        _agent_worker: agent_worker,
    })
}

/// True when `value` is an absolute filesystem path to a `.cube` LUT file.
/// Used by `set-clip-lut` so intensity re-apply and Browse… can pass a path
/// without going through the cloud catalog registry.
fn is_absolute_cube_path(value: &str) -> bool {
    let path = std::path::Path::new(value);
    path.is_absolute()
        && path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("cube"))
}
