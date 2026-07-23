mod agent;
mod agent_app_control;
mod agent_jobs;
mod agent_project;
mod agent_senses;
mod agent_session;
mod agent_system;
mod agent_vision;
mod ai_media;
mod analysis_index;
mod audio;
mod cache_references;
mod cache_registry;
mod cloud;
mod download_safety;
mod drafts;
mod drag_preview;
mod external;
mod graph_editor;
mod inspector;
mod interaction;
mod lottie_stickers;
mod lut_catalog;
mod os_drop;
mod params;
mod paths;
mod placement;
mod preview_gesture;
mod preview_motion_path;
mod preview_select;
mod preview_view;
mod preview_worker;
mod projection;
mod proxy;
mod ruler;
mod selection;
mod sfx;
mod snap;
mod strips;
mod templates;
mod text_presets;
mod thumbnails;
mod timecode;
mod timeline;
mod timeline_map;
mod transform_gesture_session;
mod transport;
mod updates;
mod window;

use slint::BackendSelector;
use slint::ModelRc;
use slint::VecModel;
use slint::wgpu_29::WGPUConfiguration;
use slint::winit_030::WinitWindowAccessor;

slint::include_modules!();

use bootstrap::*;
use cache_ui::*;

// PORT IN PROGRESS (from main's crates/cutlass-ui): Phases 0–1 of the
// desktop-editor port are live — window chrome, launch gallery, drafts,
// settings, the pure timeline/selection/snap backends, and the engine worker
// (project sessions, imports, clip drops, fit-sized preview frames,
// autosave). The rest of the edit surface, audio, thumbnails, export, and
// the agent arrive in later phases; their callbacks are stubbed and noted
// inline.

mod bootstrap;
mod cache_ui;
mod library_helpers;
mod session;
mod wire_engine;
mod wire_inspector;
mod wire_preview;
mod wire_settings;
mod wire_timeline;
mod wire_ui;

use wire_engine::wire_engine;
use wire_inspector::wire_inspector;
use wire_preview::wire_preview;
use wire_settings::wire_settings;
use wire_timeline::wire_timeline;
use wire_ui::wire_ui;

fn main() -> Result<(), slint::PlatformError> {
    setup_tracing();
    BackendSelector::new()
        .require_wgpu_29(WGPUConfiguration::default())
        .select()?;

    let app = AppWindow::new()?;

    // The window (and NSApp) exist now; safe to brand the Dock tile.
    #[cfg(target_os = "macos")]
    set_dock_icon();

    // macOS and Windows both keep the OS-drawn frame (rounded corners, drop
    // shadow, native resize/snap) and only suppress the native caption, so the
    // custom title bar shows through (window::apply_native_chrome). `is-macos`
    // drives the macOS-only bits — caption buttons drop out (the traffic lights
    // handle min/max/close) and the brand insets past them. Only Linux/BSD,
    // which have no "frame minus titlebar" mode, go fully frameless (`no-frame`
    // ← `frameless`) and draw the whole chrome. Set before the window is shown
    // so `no-frame` resolves correctly at creation.
    let app_state = app.global::<AppState>();
    app_state.set_is_macos(cfg!(target_os = "macos"));
    app_state.set_frameless(cfg!(not(any(target_os = "macos", target_os = "windows"))));

    let app_weak = app.as_weak();
    slint::invoke_from_event_loop(move || {
        if let Some(app) = app_weak.upgrade() {
            // Hide the native titlebar once the winit window is realized
            // (no-op off macOS); must run on the event loop, not before show.
            // The window opens at its natural size on the launch screen — the
            // editor maximizes via WindowBackend.set-maximized (app.slint
            // watches launch-visible), not here.
            app.window().with_winit_window(window::apply_native_chrome);
        }
    })
    .map_err(|e| slint::PlatformError::from(format!("failed to apply window chrome: {e}")))?;

    // Frameless shell (`no-frame` in app.slint): the custom title bar
    // replaces the OS decorations, so window management is wired here.
    let window_backend = app.global::<WindowBackend>();

    let weak = app.as_weak();
    window_backend.on_minimize(move || {
        if let Some(app) = weak.upgrade() {
            app.window().set_minimized(true);
        }
    });

    let weak = app.as_weak();
    window_backend.on_toggle_maximize(move || {
        if let Some(app) = weak.upgrade() {
            let maximized = !app.window().is_maximized();
            app.window().set_maximized(maximized);
            app.global::<WindowBackend>().set_maximized(maximized);
        }
    });

    // Surface-driven sizing (app.slint watches launch-visible): the launch
    // screen stays at the window's natural size, the editor maximizes. Goes
    // through window::set_maximized, which on macOS skips the native zoom
    // animation so the editor appears already maximized rather than visibly
    // growing into it.
    let weak = app.as_weak();
    window_backend.on_set_maximized(move |maximized| {
        if let Some(app) = weak.upgrade() {
            app.window()
                .with_winit_window(|w| window::set_maximized(w, maximized));
            app.global::<WindowBackend>().set_maximized(maximized);
        }
    });

    // Native window move: only valid while a pointer button is down (the
    // title bar's drag TouchArea guarantees that); the OS owns the rest of
    // the gesture, so no further pointer events arrive until release.
    let weak = app.as_weak();
    window_backend.on_begin_move(move || {
        if let Some(app) = weak.upgrade() {
            app.window().with_winit_window(|winit_window| {
                if let Err(e) = winit_window.drag_window() {
                    tracing::warn!("window drag rejected by backend: {e}");
                }
            });
        }
    });

    // User settings (~/.cutlass/config.toml). A missing/broken file falls
    // back to defaults so launch never depends on it; the theme applies
    // immediately, the AI fields seed the Settings dialog for later phases.
    let config_path = cutlass_settings::default_config_path();
    let app_settings = match cutlass_settings::load(&config_path) {
        Ok(settings) => settings,
        Err(_) => {
            tracing::warn!(
                path = %config_path.display(),
                "configuration could not be loaded; using in-memory defaults without overwriting it"
            );
            cutlass_settings::Settings::default()
        }
    };
    let download_quota_mib = app_settings.storage.download_quota_mib;
    let storage_layout = match paths::storage_layout(&app_settings.storage) {
        Ok(layout) => layout,
        Err(error) => {
            tracing::error!(
                %error,
                "invalid cache storage layout; using default cache locations"
            );
            // Cache fallback never changes the authoritative project/draft
            // root and performs no relocation.
            let defaults = cutlass_settings::StorageSettings::default();
            paths::storage_layout(&defaults).map_err(|fallback_error| {
                slint::PlatformError::from(format!(
                    "default cache storage layout is invalid: {fallback_error}"
                ))
            })?
        }
    };
    let storage_layout = cutlass_storage::SharedStorageLayout::new(storage_layout);
    let download_quota_bytes = download_quota_mib
        .checked_mul(MIB_BYTES)
        .ok_or_else(|| slint::PlatformError::from("download cache quota exceeds byte range"))?;

    // AI assistant: a dedicated worker rehearses each prompt on a sandbox
    // engine, then replays the validated plan through the preview worker as
    // one undoable group.
    let job_manager = cutlass_jobs::JobManager::new();
    let agent_store = app.global::<AgentStore>();
    agent_store.set_transcript(ModelRc::new(VecModel::<AgentEntry>::default()));
    agent_store.set_configured(app_settings.ai.is_configured());
    agent_store.set_config_path(config_path.display().to_string().into());

    // ENGINE WIRING (Phases 1–5): session flows, imports, the full edit
    // surface (drag/trim/split, inspector commits, clipboard, tracks,
    // keyframes, effects), audio playback, library/timeline tiles, and the
    // export job bind to the workers below. Still to come: live overrides (6).
    let engine = wire_engine(&app, storage_layout, download_quota_bytes, job_manager)?;

    wire_timeline(&app, &engine.preview_worker, &engine.download_cache);

    wire_settings(
        &app,
        config_path,
        &app_settings,
        download_quota_mib,
        &engine.cache_registry,
        &engine.download_cache,
        &engine.preview_worker,
    );

    wire_ui(
        &app,
        &engine.audio_system,
        &engine.interaction_gate,
        &engine.strip_worker,
    );

    wire_preview(&app, &engine.preview_worker);

    wire_inspector(&app, &engine.preview_worker);

    // Hold engine handles for the app lifetime (background worker threads).
    let _engine = engine;

    app.run()
}

#[cfg(test)]
mod settings_cache_tests;
