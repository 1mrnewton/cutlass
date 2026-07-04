mod drafts;
// Sampling/easing helpers go quiet until their consumers (projection,
// inspector) land in Phases 1–2.
#[allow(dead_code)]
mod params;
mod paths;
mod ruler;
mod selection;
mod snap;
mod timecode;
mod timeline;
mod transport;
mod window;

use slint::BackendSelector;
use slint::Model;
use slint::ModelRc;
use slint::SharedString;
use slint::VecModel;
use slint::wgpu_28::WGPUConfiguration;
use slint::winit_030::WinitWindowAccessor;
use tracing_subscriber::EnvFilter;

slint::include_modules!();

// PORT IN PROGRESS (from main's crates/cutlass-ui): this shell is Phase 0 of
// the desktop-editor port — window chrome, launch gallery, drafts, settings,
// and the pure timeline/selection/snap backends are live; everything that
// needs the engine (preview worker, edits, audio, thumbnails, export, agent)
// arrives in later phases. Engine-facing callbacks are stubbed here and noted
// inline.

/// Run `f` on the next event-loop turn, outside whatever callback is
/// currently executing. Used to flip Timer-bound state (see `request-stop`)
/// without re-entering Slint's timer machinery. Must never run anything that
/// blocks on a nested run loop (e.g. a modal `rfd::FileDialog`): the closure
/// executes inside Slint's timer activation, and the display link re-entering
/// it aborts with "Recursion in timer code".
fn defer_main_thread(f: impl FnOnce() + Send + 'static) {
    slint::Timer::single_shot(std::time::Duration::ZERO, f);
}

// File dialogs use `rfd::AsyncFileDialog`: on macOS it presents a sheet via
// `beginSheetModalForWindow:completionHandler:` and never blocks the main
// thread. The blocking `rfd::FileDialog` spins a nested `runModal` run loop,
// during which Slint's display-link tick re-enters timer processing and
// aborts with "Recursion in timer code".

/// File picker for Open file… — choose an external `.cutlass` to import into
/// a new draft (the app-owned store; see [`drafts`]).
async fn pick_open_path() -> Option<std::path::PathBuf> {
    rfd::AsyncFileDialog::new()
        .add_filter("Cutlass project", &["cutlass"])
        .pick_file()
        .await
        .map(|file| file.path().to_path_buf())
}

// --- session lifecycle: app-owned drafts, auto-saved (CapCut-style) -------
//
// Cutlass owns every project as a draft under the per-user data dir (see the
// `drafts` module). Once the engine worker lands (Phase 1) it auto-saves the
// live draft after every edit; for now the stubs below only create/import
// the draft directories and flip the shell into the (empty) editor by
// bumping `session-epoch` — the same signal the worker will send when a real
// session loads.

/// Enter the editor: bump `EditorStore.session-epoch`, which app.slint's
/// watcher answers by hiding the launch screen. The worker will own this
/// signal once sessions really load (Phase 1).
fn enter_editor(app: &AppWindow) {
    let editor = app.global::<EditorStore>();
    editor.set_session_epoch(editor.get_session_epoch() + 1);
}

/// Republish the launch gallery from the draft store, newest first.
fn refresh_projects(app: &AppWindow) {
    let rows: Vec<ProjectSummary> = drafts::list()
        .into_iter()
        .map(|draft| ProjectSummary {
            name: draft.name.into(),
            path: draft.project.to_string_lossy().into_owned().into(),
            modified: drafts::relative_time(draft.modified).into(),
        })
        .collect();
    app.global::<EditorStore>()
        .set_projects(ModelRc::new(VecModel::from(rows)));
}

/// The window close button, context-aware (CapCut-style). In the editor it
/// returns to the launch gallery (refreshed) — once auto-save lands the work
/// is already flushed, so there's no prompt and the app stays open; on the
/// gallery there's nothing left to return to, so it quits. Wired to both the
/// custom caption ✕ and the OS close request (the macOS traffic light).
fn request_close(app_weak: &slint::Weak<AppWindow>) {
    let Some(app) = app_weak.upgrade() else {
        return;
    };
    if app.global::<AppState>().get_launch_visible() {
        let _ = slint::quit_event_loop();
    } else {
        refresh_projects(&app);
        app.global::<AppState>().set_launch_visible(true);
    }
}

/// Reveal a file in the OS file browser, selecting it where the platform
/// supports that (Finder on macOS, Explorer on Windows). On other platforms
/// we fall back to opening the containing directory.
fn reveal_in_file_browser(path: &std::path::Path) {
    let spawn = |program: &str, args: &[&std::ffi::OsStr]| {
        if let Err(e) = std::process::Command::new(program).args(args).spawn() {
            tracing::error!("failed to reveal path in file browser: {e}");
        }
    };

    #[cfg(target_os = "macos")]
    spawn("open", &[std::ffi::OsStr::new("-R"), path.as_os_str()]);

    #[cfg(target_os = "windows")]
    {
        let mut select = std::ffi::OsString::from("/select,");
        select.push(path.as_os_str());
        spawn("explorer", &[select.as_os_str()]);
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let dir = path.parent().unwrap_or(path);
        spawn("xdg-open", &[dir.as_os_str()]);
    }
}

/// A trimmed, non-empty string, else `None` — the shape `cutlass_settings`'
/// optional fields want (an empty text box clears the key rather than writing
/// `""`).
fn non_empty(s: &str) -> Option<String> {
    let t = s.trim();
    (!t.is_empty()).then(|| t.to_string())
}

// The Dock icon of a bare (non-bundled) binary is the generic executable
// glyph: AppKit takes it from the .app bundle, which `cargo run` doesn't
// have, and winit has no window-icon concept on macOS — so `Window.icon`
// in app.slint only covers Windows/Linux. Set it on NSApplication instead.
#[cfg(target_os = "macos")]
fn set_dock_icon() {
    use objc2::{AnyThread, MainThreadMarker};
    use objc2_app_kit::{NSApplication, NSImage};
    use objc2_foundation::NSData;

    static ICON_PNG: &[u8] = include_bytes!("../../../assets/icon/cutlass-in-app.png");

    let Some(mtm) = MainThreadMarker::new() else {
        tracing::warn!("skipping dock icon: not on the main thread");
        return;
    };
    let data = NSData::with_bytes(ICON_PNG);
    match NSImage::initWithData(NSImage::alloc(), &data) {
        Some(image) => {
            // SAFETY: `image` is a valid NSImage and we are on the main
            // thread (proven by `mtm`), which is all AppKit requires here.
            unsafe {
                NSApplication::sharedApplication(mtm).setApplicationIconImage(Some(&image));
            }
        }
        None => tracing::warn!("skipping dock icon: embedded PNG failed to decode"),
    }
}

fn setup_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
}

fn main() -> Result<(), slint::PlatformError> {
    setup_tracing();
    BackendSelector::new()
        .require_wgpu_28(WGPUConfiguration::default())
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
    let app_settings =
        cutlass_settings::load(&cutlass_settings::default_config_path()).unwrap_or_default();

    // AI assistant: not ported yet (the `cutlass-ai` crate arrives after the
    // engine phases). The panel stays in its "connect a provider" state and
    // its callbacks are inert.
    let agent_store = app.global::<AgentStore>();
    agent_store.set_transcript(ModelRc::new(VecModel::<AgentEntry>::default()));
    agent_store.set_configured(false);

    let editor = app.global::<EditorStore>();

    // ENGINE WIRING (Phase 1+): playhead → frame requests, clip drops,
    // import, media delete/relink, magnet, and every edit callback bind to
    // the preview worker when it lands; until then those callbacks are
    // Slint-side no-ops.

    // --- project lifecycle: app-owned drafts ------------------------------

    // Open file… (Open card / Cmd+O / File ▸ Open file…): import an external
    // `.cutlass` into a new draft. Loading it into an engine session is
    // Phase 1; for now the draft is created and the empty editor opens.
    let open_weak = app.as_weak();
    editor.on_on_open_requested(move || {
        let open_weak = open_weak.clone();
        let task = slint::spawn_local(async move {
            if let Some(source) = pick_open_path().await {
                match drafts::import_external(&source) {
                    Ok(path) => {
                        tracing::info!(draft = %path.display(), "imported external project");
                        if let Some(app) = open_weak.upgrade() {
                            enter_editor(&app);
                        }
                    }
                    Err(e) => {
                        tracing::error!("couldn't import {}: {e}", source.display())
                    }
                }
            }
        });
        if let Err(e) = task {
            tracing::error!("failed to open import dialog: {e}");
        }
    });

    // New (New card / Cmd+N / File ▸ New): a fresh draft.
    let new_weak = app.as_weak();
    editor.on_on_new_requested(move || {
        match drafts::create() {
            Ok(path) => {
                tracing::info!(draft = %path.display(), "created draft");
                if let Some(app) = new_weak.upgrade() {
                    enter_editor(&app);
                }
            }
            Err(e) => tracing::error!("couldn't create a new project: {e}"),
        }
    });

    // Launch gallery card → open that draft by its project path (the load
    // itself is Phase 1).
    let open_draft_weak = app.as_weak();
    editor.on_on_open_project_requested(move |path| {
        tracing::info!(draft = path.as_str(), "opening draft");
        if let Some(app) = open_draft_weak.upgrade() {
            enter_editor(&app);
        }
    });

    // Launch gallery → delete a draft (its whole directory), then refresh.
    let delete_app = app.as_weak();
    editor.on_on_delete_project_requested(move |path| {
        drafts::delete(std::path::Path::new(path.as_str()));
        if let Some(app) = delete_app.upgrade() {
            refresh_projects(&app);
        }
    });

    // Seed the launch gallery from the draft store.
    refresh_projects(&app);

    // Window close — the title-bar ✕ and the OS close request both go through
    // the context-aware close: from the editor it returns to the launch
    // gallery, from the gallery it quits.
    let app_weak = app.as_weak();
    app.global::<WindowBackend>().on_close(move || {
        request_close(&app_weak);
    });

    let app_weak = app.as_weak();
    app.window().on_close_requested(move || {
        request_close(&app_weak);
        slint::CloseRequestResponse::KeepWindowShown
    });

    // --- app settings (gear / Cutlass menu → dialog → config.toml) -------

    let settings_backend = app.global::<SettingsBackend>();
    let config_path = cutlass_settings::default_config_path();

    // Seed the dialog from the loaded config. The theme rides AppStore so it
    // drives the live theme binding the whole shell reads.
    settings_backend.set_config_path(config_path.display().to_string().into());
    settings_backend.set_ai_base_url(app_settings.ai.base_url.clone().into());
    settings_backend.set_ai_model(app_settings.ai.model.clone().into());
    settings_backend.set_ai_api_key(app_settings.ai.api_key.clone().unwrap_or_default().into());
    settings_backend.set_ai_api_key_env(
        app_settings
            .ai
            .api_key_env
            .clone()
            .unwrap_or_default()
            .into(),
    );
    app.global::<AppStore>()
        .set_theme_id(app_settings.appearance.theme.index());

    // Persist on dismiss (Done / ✕ / Esc). Load-then-patch so any hand-set
    // keys the UI doesn't surface survive, then apply the live-settable bits
    // immediately.
    {
        let app_weak = app.as_weak();
        let config_path = config_path.clone();
        settings_backend.on_save(move || {
            let Some(app) = app_weak.upgrade() else {
                return;
            };
            let sb = app.global::<SettingsBackend>();
            let mut s = cutlass_settings::load(&config_path).unwrap_or_default();
            s.ai.base_url = sb.get_ai_base_url().trim().to_string();
            s.ai.model = sb.get_ai_model().trim().to_string();
            s.ai.api_key = non_empty(&sb.get_ai_api_key());
            s.ai.api_key_env = non_empty(&sb.get_ai_api_key_env());
            s.appearance.theme =
                cutlass_settings::ThemeChoice::from_index(app.global::<AppStore>().get_theme_id());

            if let Err(e) = cutlass_settings::save(&config_path, &s) {
                tracing::error!("failed to save settings: {e}");
            }
            // The agent isn't ported yet, so `AgentStore.configured` stays
            // false regardless of the provider fields (Phase 1+ flips this).
        });
    }

    // Endpoint test requires the AI crate; report that instead of hanging
    // the button.
    {
        let app_weak = app.as_weak();
        settings_backend.on_test_connection(move || {
            if let Some(app) = app_weak.upgrade() {
                let sb = app.global::<SettingsBackend>();
                sb.set_ai_testing(false);
                sb.set_ai_test_ok(false);
                sb.set_ai_test_status("The AI assistant isn't wired up in this build yet.".into());
            }
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
            reveal_in_file_browser(&target);
        });
    }

    // --- pure UI backends (no engine involved) ----------------------------

    let timeline_lib = app.global::<TimelineLib>();
    timeline_lib.on_sequence_duration(timeline::sequence_duration);
    timeline_lib.on_format_timecode(|frame, fps_num, fps_den, drop_frame| {
        SharedString::from(crate::timecode::format_timecode(
            i64::from(frame),
            i64::from(fps_num),
            i64::from(fps_den),
            drop_frame,
        ))
    });

    app.global::<RulerBackend>()
        .on_ticks(|scroll_x, viewport_w, zoom, fps_num, fps_den| {
            ruler::ticks_model(scroll_x, viewport_w, zoom, fps_num, fps_den)
        });

    // Playback clock: no audio device path yet (Phase 3), so every speed
    // uses the scaled wall clock.
    app.global::<TransportBackend>().on_playback_tick(
        move |anchor_tick, anchor_ms, now_ms, fps_num, fps_den, speed_num, speed_den| {
            transport::playback_tick_scaled(
                anchor_tick,
                anchor_ms,
                now_ms,
                fps_num,
                fps_den,
                speed_num,
                speed_den,
            )
        },
    );

    // End-of-playback auto-stop, deferred off the playback Timer's own
    // callback. `playback-step` calls this instead of flipping
    // `TimelineStore.playing` (the Timer's `running` binding) inline, which
    // re-enters Slint's timer machinery and panics with "Recursion in timer
    // code" (slint-ui/slint#6332). The Slint `playing = false` write — which
    // is what actually stops the Timer — runs on the next event-loop turn,
    // outside the callback.
    let stop_weak = app.as_weak();
    app.global::<TransportBackend>().on_request_stop(move || {
        let stop_weak = stop_weak.clone();
        defer_main_thread(move || {
            if let Some(app) = stop_weak.upgrade() {
                app.global::<TimelineStore>().set_playing(false);
            }
        });
    });

    app.global::<DragBackend>().on_snap_clip_start(
        |sequence,
         dragging_source_track_id,
         dragging_clip_id,
         cursor_start_value,
         clip_duration_ticks,
         snap_threshold_ticks,
         playhead_tick| {
            snap::compute_drag_snap(
                &sequence,
                dragging_source_track_id.as_str(),
                dragging_clip_id.as_str(),
                cursor_start_value,
                clip_duration_ticks,
                snap_threshold_ticks,
                playhead_tick,
            )
        },
    );

    app.global::<DragBackend>().on_resolve_clip_drag(
        |sequence,
         source_track_id,
         dragging_clip_id,
         dx_ticks,
         hover_row,
         playhead_tick,
         snap_threshold_ticks,
         main_magnet| {
            snap::resolve_clip_drag(
                &sequence,
                source_track_id.as_str(),
                dragging_clip_id.as_str(),
                dx_ticks,
                hover_row,
                playhead_tick,
                snap_threshold_ticks,
                main_magnet,
            )
        },
    );

    app.global::<DragBackend>().on_resolve_library_drop(
        |sequence,
         lane_kind,
         duration_ticks,
         cursor_tick,
         drop_row,
         playhead_tick,
         snap_threshold_ticks,
         main_magnet| {
            snap::resolve_library_drop(
                &sequence,
                lane_kind,
                duration_ticks,
                cursor_tick,
                drop_row,
                playhead_tick,
                snap_threshold_ticks,
                main_magnet,
            )
        },
    );

    app.global::<DragBackend>().on_resolve_clip_trim(
        |sequence,
         track_id,
         clip_id,
         trim_head,
         dx_ticks,
         playhead_tick,
         snap_threshold_ticks,
         link_enabled,
         main_magnet| {
            snap::resolve_clip_trim(
                &sequence,
                track_id.as_str(),
                clip_id.as_str(),
                trim_head,
                dx_ticks,
                playhead_tick,
                snap_threshold_ticks,
                link_enabled,
                main_magnet,
            )
        },
    );

    app.global::<DragBackend>()
        .on_group_floaters(|sequence, ids| selection::group_floaters(&sequence, &ids));

    app.global::<DragBackend>().on_resolve_group_drag(
        |sequence,
         ids,
         anchor_track_id,
         anchor_clip_id,
         dx_ticks,
         hover_row,
         playhead_tick,
         snap_threshold_ticks| {
            selection::resolve_group_drag(
                &sequence,
                &ids,
                anchor_track_id.as_str(),
                anchor_clip_id.as_str(),
                dx_ticks,
                hover_row,
                playhead_tick,
                snap_threshold_ticks,
            )
        },
    );

    app.global::<SelectionBackend>()
        .on_contains(|ids, clip_id| selection::selection_contains(&ids, clip_id.as_str()));

    app.global::<SelectionBackend>()
        .on_select_clip(|sequence, track_id, clip_id, link_enabled| {
            selection::select_clip(&sequence, track_id.as_str(), clip_id.as_str(), link_enabled)
        });

    app.global::<SelectionBackend>().on_toggle_clip(
        |sequence, current, track_id, clip_id, link_enabled| {
            selection::toggle_clip(
                &sequence,
                &current,
                track_id.as_str(),
                clip_id.as_str(),
                link_enabled,
            )
        },
    );

    app.global::<SelectionBackend>().on_resolve_marquee(
        |sequence, tick0, tick1, row0, row1, link_enabled| {
            selection::resolve_marquee(&sequence, tick0, tick1, row0, row1, link_enabled)
        },
    );

    // Selection survives undo/redo: every projection republish reconciles
    // the selection against the new clip set.
    app.global::<SelectionBackend>()
        .on_prune(|sequence, current, primary_clip_id| {
            selection::prune_selection(&sequence, &current, primary_clip_id.as_str())
        });

    app.global::<SelectionBackend>()
        .on_has_link(|sequence, ids| selection::selection_has_link(&sequence, &ids));

    // Timeline keyframe diamonds: merged tick model for the selected clip
    // (drag-retime and delete need the engine — Phase 2).
    app.global::<KeyframeBackend>()
        .on_ticks(|clip| params::merged_keyframe_ticks(&clip));

    app.global::<InspectorBackend>()
        .on_filter_fonts(|query, items| {
            let needle = query.to_lowercase();
            let filtered: Vec<SharedString> = items
                .iter()
                .filter(|family| {
                    needle.is_empty() || family.as_str().to_lowercase().contains(&needle)
                })
                .collect();
            ModelRc::new(VecModel::from(filtered))
        });

    // Effects & transitions: fill the Library catalogs once; the add/remove
    // edits route through the engine when it lands (Phase 2).
    {
        let effects = app.global::<EffectsBackend>();
        let effect_rows: Vec<CatalogEntry> = cutlass_models::effect_catalog()
            .iter()
            .map(|s| CatalogEntry {
                id: s.id.into(),
                label: s.label.into(),
            })
            .collect();
        effects.set_effect_catalog(ModelRc::new(VecModel::from(effect_rows)));
        let transition_rows: Vec<CatalogEntry> = cutlass_models::transition_catalog()
            .iter()
            .map(|s| CatalogEntry {
                id: s.id.into(),
                label: s.label.into(),
            })
            .collect();
        effects.set_transition_catalog(ModelRc::new(VecModel::from(transition_rows)));
    }

    app.run()
}
