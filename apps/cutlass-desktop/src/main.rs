mod audio;
mod drafts;
mod inspector;
mod params;
mod paths;
mod placement;
mod preview_gesture;
mod preview_select;
mod preview_view;
mod preview_worker;
mod projection;
mod ruler;
mod selection;
mod snap;
mod strips;
mod thumbnails;
mod timecode;
mod timeline;
mod transport;
mod window;

use cutlass_engine::EngineConfig;
use slint::BackendSelector;
use slint::Global;
use slint::Model;
use slint::ModelRc;
use slint::SharedString;
use slint::VecModel;
use slint::wgpu_28::WGPUConfiguration;
use slint::winit_030::WinitWindowAccessor;
use tracing_subscriber::EnvFilter;

slint::include_modules!();

// PORT IN PROGRESS (from main's crates/cutlass-ui): Phases 0–1 of the
// desktop-editor port are live — window chrome, launch gallery, drafts,
// settings, the pure timeline/selection/snap backends, and the engine worker
// (project sessions, imports, clip drops, fit-sized preview frames,
// autosave). The rest of the edit surface, audio, thumbnails, export, and
// the agent arrive in later phases; their callbacks are stubbed and noted
// inline.

/// Library palette key → engine generator, for drag-drops from the Library's
/// Titles/Shapes tabs. Content and size are placeholders the user edits next
/// (via the inspector); `None` for unknown keys.
fn generator_from_key(key: &str) -> Option<cutlass_models::Generator> {
    use cutlass_models::{Generator, Shape};
    Some(match key {
        "text" => Generator::text("Title"),
        "solid" => Generator::SolidColor {
            rgba: [30, 30, 30, 255],
        },
        "rect" => Generator::shape(Shape::Rectangle, [255, 255, 255, 255]),
        "ellipse" => Generator::shape(Shape::Ellipse, [255, 255, 255, 255]),
        _ => return None,
    })
}

/// Map an inspector param key to the engine's `ClipParam` plus the matching
/// `ParamValue` shape (position is the one vec2; scalars ride `value_x`).
/// `None` for an unknown key.
fn clip_param_value(
    param: &str,
    value_x: f32,
    value_y: f32,
) -> Option<(cutlass_models::ClipParam, cutlass_models::ParamValue)> {
    use cutlass_models::{ClipParam, ParamValue};
    Some(match param {
        "position" => (ClipParam::Position, ParamValue::Vec2([value_x, value_y])),
        "anchor" => (ClipParam::AnchorPoint, ParamValue::Vec2([value_x, value_y])),
        "scale" => (ClipParam::Scale, ParamValue::Scalar(value_x)),
        "rotation" => (ClipParam::Rotation, ParamValue::Scalar(value_x)),
        "opacity" => (ClipParam::Opacity, ParamValue::Scalar(value_x)),
        "volume" => (ClipParam::Volume, ParamValue::Scalar(value_x)),
        _ => return None,
    })
}

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

async fn pick_import_path() -> Option<std::path::PathBuf> {
    rfd::AsyncFileDialog::new()
        .add_filter(
            "Media",
            &[
                "mp4", "mov", "mkv", "webm", "m4v", "mp3", "wav", "m4a", "aac", "flac", "ogg",
                "png", "jpg", "jpeg", "webp",
            ],
        )
        .add_filter("Video", &["mp4", "mov", "mkv", "webm", "m4v"])
        .add_filter("Audio", &["mp3", "wav", "m4a", "aac", "flac", "ogg"])
        .add_filter("Images", &["png", "jpg", "jpeg", "webp"])
        .pick_file()
        .await
        .map(|file| file.path().to_path_buf())
}

/// File picker for Open file… — choose an external `.cutlass` to import into
/// a new draft (the app-owned store; see [`drafts`]).
async fn pick_open_path() -> Option<std::path::PathBuf> {
    rfd::AsyncFileDialog::new()
        .add_filter("Cutlass project", &["cutlass"])
        .pick_file()
        .await
        .map(|file| file.path().to_path_buf())
}

async fn pick_relink_path() -> Option<std::path::PathBuf> {
    rfd::AsyncFileDialog::new()
        .add_filter(
            "Media",
            &[
                "mp4", "mov", "mkv", "webm", "m4v", "mp3", "wav", "m4a", "aac", "flac", "ogg",
                "png", "jpg", "jpeg", "webp",
            ],
        )
        .add_filter("Video", &["mp4", "mov", "mkv", "webm", "m4v"])
        .add_filter("Audio", &["mp3", "wav", "m4a", "aac", "flac", "ogg"])
        .add_filter("Images", &["png", "jpg", "jpeg", "webp"])
        .pick_file()
        .await
        .map(|file| file.path().to_path_buf())
}

async fn pick_relink_folder() -> Option<std::path::PathBuf> {
    rfd::AsyncFileDialog::new()
        .pick_folder()
        .await
        .map(|file| file.path().to_path_buf())
}

// --- session lifecycle: app-owned drafts, auto-saved (CapCut-style) -------
//
// Cutlass owns every project as a draft under the per-user data dir (see the
// `drafts` module); the worker auto-saves the live draft after every edit, so
// the user never saves by hand and a clean exit loses nothing. Switching the
// live session — New, opening another draft, importing a file — flushes the
// outgoing draft first (an ordered `SaveProject` on the worker's queue), then
// swaps. Closing is handled separately (`request_close`): the editor returns
// to the launch gallery, the gallery quits the app.

enum SessionChange {
    /// Create a fresh, empty draft and switch to it.
    New,
    /// Pick an external `.cutlass` file and import it into a new draft.
    Import,
    /// Open an existing draft by its `project.cutlass` path (gallery card).
    OpenDraft(std::path::PathBuf),
}

/// Flush the outgoing draft (a no-op when nothing is bound yet), then replace
/// the session. The flush and the swap are ordered on the worker's single
/// message queue, so the flush always captures the draft we're leaving.
fn change_session(handle: &preview_worker::WorkerHandle, change: SessionChange) {
    match change {
        SessionChange::New => match drafts::create() {
            Ok(path) => {
                handle.save_project(None);
                handle.new_project();
                handle.save_project(Some(path));
            }
            Err(e) => tracing::error!("couldn't create a new project: {e}"),
        },
        SessionChange::OpenDraft(path) => {
            handle.save_project(None);
            handle.open_project(path);
        }
        SessionChange::Import => {
            let handle = handle.clone();
            let task = slint::spawn_local(async move {
                if let Some(source) = pick_open_path().await {
                    match drafts::import_external(&source) {
                        Ok(path) => {
                            handle.save_project(None);
                            handle.open_project(path);
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
        }
    }
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
/// flushes the draft and returns to the launch gallery (refreshed) — the work
/// is already auto-saved, so there's no prompt and the app stays open; on the
/// gallery there's nothing left to return to, so it quits. Wired to both the
/// custom caption ✕ and the OS close request (the macOS traffic light).
fn request_close(app_weak: &slint::Weak<AppWindow>, handle: &preview_worker::WorkerHandle) {
    let Some(app) = app_weak.upgrade() else {
        return;
    };
    if app.global::<AppState>().get_launch_visible() {
        let _ = slint::quit_event_loop();
    } else {
        handle.save_project(None);
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

/// Native save dialog for the export destination, seeded with the current
/// path's folder and file name. `None` when the user cancels.
async fn pick_export_path(current: std::path::PathBuf) -> Option<std::path::PathBuf> {
    let mut dialog = rfd::AsyncFileDialog::new().add_filter("MP4 video", &["mp4"]);
    if let Some(dir) = current.parent().filter(|d| d.is_dir()) {
        dialog = dialog.set_directory(dir);
    }
    dialog = dialog.set_file_name(
        current
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "untitled.mp4".into()),
    );
    dialog
        .save_file()
        .await
        .map(|file| file.path().to_path_buf())
}

/// Prefilled export destination: the OS videos folder (Movies on macOS,
/// Videos on Windows, XDG videos on Linux), else the home directory, else the
/// temp dir — never the working directory, which is the read-only install
/// folder on Windows. Only seeds the save panel; the user picks the real spot.
fn default_export_path() -> SharedString {
    let dir = dirs::video_dir()
        .or_else(dirs::home_dir)
        .unwrap_or_else(std::env::temp_dir);
    dir.join("untitled.mp4")
        .to_string_lossy()
        .into_owned()
        .into()
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

    // ENGINE WIRING (Phases 1–5): session flows, imports, the full edit
    // surface (drag/trim/split, inspector commits, clipboard, tracks,
    // keyframes, effects), audio playback, library/timeline tiles, and the
    // export job bind to the workers below. Still to come: live overrides (6).

    // --- engine service (preview worker thread) ---------------------------

    // Audio output + master clock (Phase 3). Starts even before a project
    // opens; a machine without a usable output device degrades to the
    // wall-clock transport (`handle.active() == false`).
    let audio_system = audio::AudioSystem::start();

    // Library tile thumbnails decode on their own thread so imports never
    // stall preview scrubbing. Keep the worker alive for the app's lifetime.
    let thumbnail_worker =
        thumbnails::ThumbnailWorker::spawn(app.global::<EditorStore>().as_weak())
            .map_err(slint::PlatformError::Other)?;

    // Timeline clip content (filmstrip frames, waveform tiles) decodes on a
    // third thread: a long strip batch must not delay library tiles, and
    // neither may ever touch the UI or engine threads.
    let strip_worker = strips::StripWorker::spawn(app.global::<StripBackend>().as_weak())
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
    )
    .map_err(slint::PlatformError::Other)?;
    tracing::debug!(
        duration_ticks = session.duration_ticks,
        "engine session ready"
    );

    // Playhead moves (ruler scrub, frame-step keys, Home/End) become preview
    // frame requests; the worker coalesces a burst to the newest tick.
    let frame_handle = preview_worker.handle();
    let scrub_audio = audio_system.handle();
    let scrub_weak = app.as_weak();
    editor.on_on_playhead_changed(move |tick| {
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
    editor.on_on_generated_dropped(
        move |generator, track_id, start_tick, duration_ticks, drop_row| {
            let Some(generator) = generator_from_key(generator.as_str()) else {
                tracing::warn!(%generator, "ignoring drop of unknown generator key");
                return;
            };
            generated_drop_handle.add_generated(
                generator,
                track_id.to_string(),
                i64::from(start_tick),
                i64::from(duration_ticks),
                i64::from(drop_row),
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
            if let Some(path) = pick_import_path().await {
                import_handle.import(path);
            }
        });
        if let Err(e) = task {
            tracing::error!("failed to open import dialog: {e}");
        }
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

    // --- timeline edit surface (drag/trim/split land as engine edits) -----

    let move_handle = preview_worker.handle();
    editor.on_on_clip_moved(move |clip_id, track_id, insert_row, start_tick, insert| {
        move_handle.move_clip(
            clip_id.to_string(),
            track_id.to_string(),
            i64::from(insert_row),
            i64::from(start_tick),
            insert,
        );
    });

    let group_move_handle = preview_worker.handle();
    editor.on_on_group_moved(move |moves| {
        let moves: Vec<preview_worker::GroupMove> = moves
            .iter()
            .map(|m| preview_worker::GroupMove {
                clip: m.clip_id.to_string(),
                track: m.track_id.to_string(),
                start_tick: i64::from(m.start_tick),
            })
            .collect();
        group_move_handle.move_group(moves);
    });

    let linkage_handle = preview_worker.handle();
    editor.on_on_linkage_changed(move |enabled| {
        linkage_handle.set_linkage(enabled);
    });

    let trim_handle = preview_worker.handle();
    editor.on_on_clip_trimmed(move |clip_id, start_tick, duration_ticks| {
        trim_handle.trim_clip(
            clip_id.to_string(),
            i64::from(start_tick),
            i64::from(duration_ticks),
        );
    });

    let delete_handle = preview_worker.handle();
    editor.on_on_clips_deleted(move |clip_ids| {
        let clips: Vec<String> = clip_ids.iter().map(|id| id.to_string()).collect();
        delete_handle.remove_clips(clips);
    });

    let split_handle = preview_worker.handle();
    editor.on_on_clip_split(move |clip_id, at_tick| {
        split_handle.split_clip(clip_id.to_string(), i64::from(at_tick));
    });

    let marker_handle = preview_worker.handle();
    let timeline_store = app.global::<TimelineStore>();
    timeline_store.on_on_marker_added(move |at_tick, name, color| {
        marker_handle.add_marker(i64::from(at_tick), name.to_string(), color.to_string());
    });
    let marker_remove_handle = preview_worker.handle();
    timeline_store.on_on_marker_removed(move |marker_id| {
        marker_remove_handle.remove_marker(marker_id.to_string());
    });

    // Clipboard ops (Cmd/Ctrl+C / V / D, context menu): the worker owns the
    // clipboard as project-independent snapshots.
    let copy_handle = preview_worker.handle();
    editor.on_on_clips_copied(move |clip_ids| {
        let clips: Vec<String> = clip_ids.iter().map(|id| id.to_string()).collect();
        copy_handle.copy_clips(clips);
    });

    let paste_handle = preview_worker.handle();
    editor.on_on_paste_at(move |tick| {
        paste_handle.paste_at(i64::from(tick));
    });

    let duplicate_handle = preview_worker.handle();
    editor.on_on_clips_duplicated(move |clip_ids| {
        let clips: Vec<String> = clip_ids.iter().map(|id| id.to_string()).collect();
        duplicate_handle.duplicate_clips(clips);
    });

    let unlink_handle = preview_worker.handle();
    editor.on_on_clips_unlinked(move |clip_ids| {
        let clips: Vec<String> = clip_ids.iter().map(|id| id.to_string()).collect();
        unlink_handle.unlink_clips(clips);
    });

    // Track header toggles (eye/speaker/lock/duck) → undoable track flags.
    let track_flag_handle = preview_worker.handle();
    editor.on_on_track_flag_toggled(move |track_id, flag, value| {
        let flag = match flag.as_str() {
            "enabled" => preview_worker::TrackFlag::Enabled,
            "muted" => preview_worker::TrackFlag::Muted,
            "locked" => preview_worker::TrackFlag::Locked,
            "duck-source" => preview_worker::TrackFlag::DuckSource,
            other => {
                tracing::error!(flag = other, "ignoring unknown track flag");
                return;
            }
        };
        track_flag_handle.set_track_flag(track_id.to_string(), flag, value);
    });

    // Canvas settings (title bar → dialog → engine thread).
    let set_canvas_handle = preview_worker.handle();
    app.global::<CanvasBackend>()
        .on_set_canvas(move |aspect_index, background| {
            set_canvas_handle.set_canvas(
                aspect_index,
                [background.red(), background.green(), background.blue()],
            );
        });

    // --- project lifecycle: app-owned drafts, auto-saved -----------------

    // Cmd/Ctrl+S has no separate "save" in the draft model — every edit is
    // already auto-saved. Keep the shortcut as an explicit "flush now" so the
    // habit still works and a draft about to close is written immediately;
    // the `save-as` argument is ignored (there are no user files to save as).
    let save_handle = preview_worker.handle();
    editor.on_on_save_requested(move |_save_as| {
        save_handle.save_project(None);
    });

    // Open file… (Open card / Cmd+O / File ▸ Open file…): import an external
    // `.cutlass` into a new draft. New (New card / Cmd+N / File ▸ New): a
    // fresh draft. Both flush the outgoing draft before swapping.
    let open_handle = preview_worker.handle();
    editor.on_on_open_requested(move || {
        change_session(&open_handle, SessionChange::Import);
    });

    let new_handle = preview_worker.handle();
    editor.on_on_new_requested(move || {
        change_session(&new_handle, SessionChange::New);
    });

    // Launch gallery card → open that draft by its project path.
    let open_draft_handle = preview_worker.handle();
    editor.on_on_open_project_requested(move |path| {
        change_session(
            &open_draft_handle,
            SessionChange::OpenDraft(std::path::PathBuf::from(path.as_str())),
        );
    });

    // Launch gallery → delete a draft (its whole directory), then refresh.
    let delete_app = app.as_weak();
    editor.on_on_delete_project_requested(move |path| {
        drafts::delete(std::path::Path::new(path.as_str()));
        if let Some(app) = delete_app.upgrade() {
            refresh_projects(&app);
        }
    });

    // Title-bar rename → one undoable edit on the worker; the next auto-save
    // writes the new name into the draft's project file and meta sidecar.
    let rename_handle = preview_worker.handle();
    editor.on_on_rename_project(move |name| {
        let name = name.trim().to_string();
        if !name.is_empty() {
            rename_handle.rename_project(name);
        }
    });

    // Seed the launch gallery from the draft store.
    refresh_projects(&app);

    // Window close — the title-bar ✕ and the OS close request both go through
    // the context-aware close: from the editor it flushes the draft and
    // returns to the launch gallery, from the gallery it quits.
    let close_handle = preview_worker.handle();
    let app_weak = app.as_weak();
    app.global::<WindowBackend>().on_close(move || {
        request_close(&app_weak, &close_handle);
    });

    let close_handle = preview_worker.handle();
    let app_weak = app.as_weak();
    app.window().on_close_requested(move || {
        request_close(&app_weak, &close_handle);
        slint::CloseRequestResponse::KeepWindowShown
    });

    // --- export (title bar → dialog → engine thread → export thread) -----

    let export_backend = app.global::<ExportBackend>();
    export_backend.set_output_path(default_export_path());

    let export_backend_weak = export_backend.as_weak();
    export_backend.on_browse_output_clicked(move || {
        let backend_weak = export_backend_weak.clone();
        let current = backend_weak
            .upgrade()
            .map(|b| b.get_output_path().to_string())
            .unwrap_or_default();
        let task = slint::spawn_local(async move {
            let current = std::path::PathBuf::from(current);
            if let Some(path) = pick_export_path(current).await
                && let Some(backend) = backend_weak.upgrade()
            {
                backend.set_output_path(path.to_string_lossy().into_owned().into());
            }
        });
        if let Err(e) = task {
            tracing::error!("failed to open export dialog: {e}");
        }
    });

    let export_handle = preview_worker.handle();
    export_backend.on_start(move |path, target_height, fps_num| {
        export_handle.export(preview_worker::ExportRequest {
            path: std::path::PathBuf::from(path.as_str()),
            target_height: u32::try_from(target_height).ok().filter(|&h| h > 0),
            fps_num: (fps_num > 0).then_some(fps_num),
        });
    });

    let export_cancel_handle = preview_worker.handle();
    export_backend.on_cancel(move || {
        export_cancel_handle.cancel_export();
    });

    let export_reveal_weak = export_backend.as_weak();
    export_backend.on_reveal_output_clicked(move || {
        if let Some(backend) = export_reveal_weak.upgrade() {
            let path = std::path::PathBuf::from(backend.get_output_path().to_string());
            reveal_in_file_browser(&path);
        }
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

    // Playback clock (Phases 1 + 3): at speed 1/1 with a live output device,
    // *consumed audio frames* are the clock — video follows the sound card,
    // which is what keeps A/V locked. Shuttle speeds and deviceless machines
    // use the scaled wall clock instead.
    let clock_audio = audio_system.handle();
    app.global::<TransportBackend>().on_playback_tick(
        move |anchor_tick, anchor_ms, now_ms, fps_num, fps_den, speed_num, speed_den| {
            if clock_audio.active() && speed_num == 1 && speed_den == 1 {
                clock_audio
                    .current_tick(fps_num, fps_den)
                    .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
            } else {
                transport::playback_tick_scaled(
                    anchor_tick,
                    anchor_ms,
                    now_ms,
                    fps_num,
                    fps_den,
                    speed_num,
                    speed_den,
                )
            }
        },
    );

    // Transport intent → audio engine. Play doubles as the mid-playback
    // seek; non-1x speeds play muted (varispeed audio is a later phase).
    let play_audio = audio_system.handle();
    app.global::<TransportBackend>()
        .on_transport_play(move |tick, speed_num, speed_den| {
            if speed_num == 1 && speed_den == 1 {
                play_audio.play(i64::from(tick));
            } else {
                play_audio.pause();
            }
        });

    let pause_audio = audio_system.handle();
    app.global::<TransportBackend>()
        .on_transport_pause(move || {
            pause_audio.pause();
        });

    // End-of-playback auto-stop, deferred off the playback Timer's own
    // callback. `playback-step` calls this instead of flipping
    // `TimelineStore.playing` (the Timer's `running` binding) inline, which
    // re-enters Slint's timer machinery and panics with "Recursion in timer
    // code" (slint-ui/slint#6332). Audio stops now (lock-free); the Slint
    // `playing = false` write — which is what actually stops the Timer — runs
    // on the next event-loop turn, outside the callback.
    let stop_audio = audio_system.handle();
    let stop_weak = app.as_weak();
    app.global::<TransportBackend>().on_request_stop(move || {
        stop_audio.pause();
        let stop_weak = stop_weak.clone();
        defer_main_thread(move || {
            if let Some(app) = stop_weak.upgrade() {
                app.global::<TimelineStore>().set_playing(false);
            }
        });
    });

    // Timeline clip content tiles (Phase 4). Cache lookups on the UI thread;
    // misses queue decode work on the strip thread and come back through a
    // `StripBackend.generation` bump (the trailing argument both callbacks
    // take exists only to re-trigger evaluation on delivery).
    let filmstrip_handle = strip_worker.handle();
    app.global::<StripBackend>().on_filmstrip_tiles(
        move |media_id,
              source_in_s,
              duration,
              fps_num,
              fps_den,
              speed,
              zoom,
              from_bucket,
              to_bucket,
              _generation| {
            strips::filmstrip_tiles(
                &filmstrip_handle,
                media_id.as_str(),
                source_in_s,
                duration,
                fps_num,
                fps_den,
                speed,
                zoom,
                from_bucket,
                to_bucket,
            )
        },
    );

    let waveform_handle = strip_worker.handle();
    app.global::<StripBackend>().on_waveform_tiles(
        move |media_id,
              source_in_s,
              duration,
              fps_num,
              fps_den,
              speed,
              zoom,
              from_bucket,
              to_bucket,
              _generation| {
            strips::waveform_tiles(
                &waveform_handle,
                media_id.as_str(),
                source_in_s,
                duration,
                fps_num,
                fps_den,
                speed,
                zoom,
                from_bucket,
                to_bucket,
            )
        },
    );

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

    // --- preview viewport: click-to-select, gestures, zoom/pan ------------

    app.global::<PreviewBackend>().on_hit_test(
        |sequence, tick, x, y, view_w, view_h, zoom, pan_x, pan_y| {
            preview_select::hit_test_in_viewport(
                &sequence, tick, x, y, view_w, view_h, zoom, pan_x, pan_y,
            )
        },
    );

    app.global::<PreviewBackend>().on_selected_contains(
        |sequence, clip_id, tick, x, y, view_w, view_h, zoom, pan_x, pan_y| {
            preview_select::selected_clip_contains_in_viewport(
                &sequence,
                clip_id.as_str(),
                tick,
                x,
                y,
                view_w,
                view_h,
                zoom,
                pan_x,
                pan_y,
            )
        },
    );

    app.global::<PreviewBackend>().on_selection_box(
        |sequence, clip_id, tick, view_w, view_h, zoom, pan_x, pan_y, gesture_active, gesture| {
            preview_select::selection_box_in_viewport(
                &sequence,
                clip_id.as_str(),
                tick,
                view_w,
                view_h,
                zoom,
                pan_x,
                pan_y,
                gesture_active.then_some(&gesture),
            )
        },
    );

    app.global::<PreviewBackend>().on_resolve_drag(
        |sequence,
         clip_id,
         tick,
         press_x,
         press_y,
         cursor_x,
         cursor_y,
         view_w,
         view_h,
         zoom,
         pan_x,
         pan_y,
         snap_tol| {
            preview_gesture::resolve_drag_in_viewport(
                &sequence,
                clip_id.as_str(),
                tick,
                press_x,
                press_y,
                cursor_x,
                cursor_y,
                view_w,
                view_h,
                zoom,
                pan_x,
                pan_y,
                snap_tol,
            )
        },
    );

    app.global::<PreviewBackend>()
        .on_nudge(|sequence, clip_id, tick, dx, dy| {
            preview_gesture::nudge(&sequence, clip_id.as_str(), tick, dx, dy)
        });

    app.global::<PreviewBackend>().on_resolve_scale(
        |sequence,
         clip_id,
         tick,
         press_x,
         press_y,
         cursor_x,
         cursor_y,
         view_w,
         view_h,
         zoom,
         pan_x,
         pan_y| {
            preview_gesture::resolve_scale_in_viewport(
                &sequence,
                clip_id.as_str(),
                tick,
                press_x,
                press_y,
                cursor_x,
                cursor_y,
                view_w,
                view_h,
                zoom,
                pan_x,
                pan_y,
            )
        },
    );

    app.global::<PreviewBackend>().on_resolve_rotate(
        |sequence,
         clip_id,
         tick,
         press_x,
         press_y,
         cursor_x,
         cursor_y,
         view_w,
         view_h,
         zoom,
         pan_x,
         pan_y,
         snap_deg| {
            preview_gesture::resolve_rotate_in_viewport(
                &sequence,
                clip_id.as_str(),
                tick,
                press_x,
                press_y,
                cursor_x,
                cursor_y,
                view_w,
                view_h,
                zoom,
                pan_x,
                pan_y,
                snap_deg,
            )
        },
    );

    app.global::<PreviewBackend>().on_resolve_anchor(
        |sequence,
         clip_id,
         tick,
         press_x,
         press_y,
         cursor_x,
         cursor_y,
         view_w,
         view_h,
         zoom,
         pan_x,
         pan_y| {
            preview_gesture::resolve_anchor_in_viewport(
                &sequence,
                clip_id.as_str(),
                tick,
                press_x,
                press_y,
                cursor_x,
                cursor_y,
                view_w,
                view_h,
                zoom,
                pan_x,
                pan_y,
            )
        },
    );

    app.global::<PreviewBackend>().on_clamp_view(
        |canvas_w, canvas_h, view_w, view_h, zoom, pan_x, pan_y| {
            preview_view::clamp_view(canvas_w, canvas_h, view_w, view_h, zoom, pan_x, pan_y)
        },
    );

    app.global::<PreviewBackend>().on_zoom_to(
        |canvas_w,
         canvas_h,
         view_w,
         view_h,
         zoom,
         pan_x,
         pan_y,
         cursor_x,
         cursor_y,
         target_zoom| {
            preview_view::zoom_to(
                canvas_w,
                canvas_h,
                view_w,
                view_h,
                zoom,
                pan_x,
                pan_y,
                cursor_x,
                cursor_y,
                target_zoom,
            )
        },
    );

    app.global::<PreviewBackend>().on_pan_view(
        |canvas_w, canvas_h, view_w, view_h, zoom, pan_x, pan_y, dx, dy| {
            preview_view::pan_by(
                canvas_w, canvas_h, view_w, view_h, zoom, pan_x, pan_y, dx, dy,
            )
        },
    );

    let override_handle = preview_worker.handle();
    editor.on_on_preview_transform_overridden(
        move |clip_id, pos_x, pos_y, anchor_x, anchor_y, scale, rotation, opacity, tick| {
            override_handle.transform_override(
                clip_id.to_string(),
                cutlass_models::ClipTransform {
                    position: [pos_x, pos_y],
                    anchor_point: [anchor_x, anchor_y],
                    scale,
                    rotation,
                    opacity,
                },
                i64::from(tick),
            );
        },
    );

    let override_clear_handle = preview_worker.handle();
    editor.on_on_preview_override_cleared(move |tick| {
        override_clear_handle.clear_transform_override(i64::from(tick));
    });

    let transform_commit_handle = preview_worker.handle();
    editor.on_on_clip_transform_committed(
        move |clip_id, pos_x, pos_y, anchor_x, anchor_y, scale, rotation, opacity, tick| {
            transform_commit_handle.set_transform(
                clip_id.to_string(),
                cutlass_models::ClipTransform {
                    position: [pos_x, pos_y],
                    anchor_point: [anchor_x, anchor_y],
                    scale,
                    rotation,
                    opacity,
                },
                i64::from(tick),
            );
        },
    );

    // --- inspector: selection resolve, playhead sampling, commits ---------

    app.global::<InspectorBackend>()
        .on_resolve_selection(|sequence, track_id, clip_id| {
            inspector::resolve_selection(sequence, track_id.as_str(), clip_id.as_str())
        });

    app.global::<InspectorBackend>()
        .on_sample_transform(|clip, playhead| inspector::sample_transform(&clip, playhead));
    app.global::<InspectorBackend>()
        .on_compensate_anchor_position(
            |clip, sequence, playhead, anchor_x, anchor_y, scale, rotation| {
                inspector::compensate_anchor_position(
                    &clip, sequence, playhead, anchor_x, anchor_y, scale, rotation,
                )
            },
        );

    app.global::<InspectorBackend>()
        .on_sample_audio(|clip, playhead| inspector::sample_audio(&clip, playhead));

    let kf_set_handle = preview_worker.handle();
    app.global::<InspectorBackend>().on_set_param_keyframe(
        move |clip_id, param, tick, value_x, value_y, easing| {
            let Some((param, value)) = clip_param_value(param.as_str(), value_x, value_y) else {
                tracing::error!(param = param.as_str(), "ignoring keyframe on unknown param");
                return;
            };
            kf_set_handle.set_param_keyframe(
                clip_id.to_string(),
                param,
                i64::from(tick),
                value,
                params::easing_from_ui(easing, [0.0; 4]),
            );
        },
    );

    let kf_remove_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_remove_param_keyframe(move |clip_id, param, tick| {
            let Some((param, _)) = clip_param_value(param.as_str(), 0.0, 0.0) else {
                tracing::error!(
                    param = param.as_str(),
                    "ignoring keyframe removal on unknown param"
                );
                return;
            };
            kf_remove_handle.remove_param_keyframe(clip_id.to_string(), param, i64::from(tick));
        });

    // Timeline keyframe diamonds: merged tick model for the selected clip,
    // drag-retime, right-click delete.
    app.global::<KeyframeBackend>()
        .on_ticks(|clip| params::merged_keyframe_ticks(&clip));
    let kf_retime_handle = preview_worker.handle();
    app.global::<KeyframeBackend>()
        .on_retime(move |clip_id, from_tick, to_tick| {
            kf_retime_handle.retime_keyframes(
                clip_id.to_string(),
                i64::from(from_tick),
                i64::from(to_tick),
            );
        });
    let kf_remove_at_handle = preview_worker.handle();
    app.global::<KeyframeBackend>()
        .on_remove_at(move |clip_id, tick| {
            kf_remove_at_handle.remove_keyframes_at(clip_id.to_string(), i64::from(tick));
        });
    let set_speed_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_set_clip_speed(move |clip_id, num, den, reversed| {
            set_speed_handle.set_clip_speed(clip_id.to_string(), num, den, reversed);
        });
    let set_pitch_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_set_clip_pitch(move |clip_id, preserve| {
            set_pitch_handle.set_clip_pitch(clip_id.to_string(), preserve);
        });
    let set_denoise_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_set_denoise(move |clip_id, denoise| {
            set_denoise_handle.set_denoise(clip_id.to_string(), denoise);
        });
    let set_curve_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_set_speed_curve(move |clip_id, preset| {
            set_curve_handle.set_speed_curve(clip_id.to_string(), preset.to_string());
        });
    let set_curve_point_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_set_speed_curve_point(move |clip_id, index, value| {
            set_curve_point_handle.set_speed_curve_point(clip_id.to_string(), index, value);
        });
    let set_audio_handle = preview_worker.handle();
    app.global::<InspectorBackend>().on_set_clip_audio(
        move |clip_id, volume, fade_in_s, fade_out_s| {
            set_audio_handle.set_clip_audio(clip_id.to_string(), volume, fade_in_s, fade_out_s);
        },
    );
    let set_fades_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_set_clip_fades(move |clip_id, fade_in_s, fade_out_s| {
            set_fades_handle.set_clip_fades(clip_id.to_string(), fade_in_s, fade_out_s);
        });
    app.global::<InspectorBackend>()
        .on_can_duck_under_voice(|sequence, track_id| {
            inspector::can_duck_under_voice(sequence, track_id.as_str())
        });
    let duck_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_duck_under_voice(move |clip_id| {
            duck_handle.duck_under_voice(clip_id.to_string());
        });
    let detect_beats_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_detect_beats(move |clip_id| {
            detect_beats_handle.detect_beats(clip_id.to_string());
        });
    let clear_beats_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_clear_beats(move |clip_id| {
            clear_beats_handle.clear_beats(clip_id.to_string());
        });
    let set_crop_handle = preview_worker.handle();
    app.global::<InspectorBackend>().on_set_clip_crop(
        move |clip_id, left, top, right, bottom, flip_h, flip_v| {
            // Insets (UI/agent shape) → kept-region rect (model shape). The
            // sliders cap each inset at 49%, so the window stays valid; the
            // floor only guards float dust against the engine's minimum.
            let crop = cutlass_models::CropRect {
                x: left,
                y: top,
                w: (1.0 - left - right).max(cutlass_models::MIN_CROP_FRACTION),
                h: (1.0 - top - bottom).max(cutlass_models::MIN_CROP_FRACTION),
            };
            set_crop_handle.set_clip_crop(clip_id.to_string(), crop, flip_h, flip_v);
        },
    );

    let fit_clip_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_fit_clip(move |clip_id, fill, tick| {
            fit_clip_handle.fit_clip(clip_id.to_string(), fill, i64::from(tick));
        });

    let set_text_handle = preview_worker.handle();
    app.global::<InspectorBackend>().on_set_text_generator(
        move |_track_id, clip_id, content, style| {
            // Route the edit through the engine (undoable) rather than mutating
            // the Slint model, which the next projection republish would revert.
            // The inspector sends the full style each time, so one committed
            // edit == one coherent `Generator::Text`.
            set_text_handle.set_generator(
                clip_id.to_string(),
                cutlass_models::Generator::Text {
                    content: content.to_string(),
                    style: inspector::text_style_from_ui(&style),
                },
            );
        },
    );

    let preview_text_handle = preview_worker.handle();
    app.global::<InspectorBackend>().on_preview_text_generator(
        move |clip_id, content, style, tick| {
            // Live, uncommitted preview (e.g. font-size drag): render the clip
            // from this generator without touching history. Release commits.
            preview_text_handle.generator_override(
                clip_id.to_string(),
                cutlass_models::Generator::Text {
                    content: content.to_string(),
                    style: inspector::text_style_from_ui(&style),
                },
                i64::from(tick),
            );
        },
    );

    let clear_text_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_clear_text_generator(move |tick| {
            clear_text_handle.clear_generator_override(i64::from(tick));
        });

    let set_shape_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_set_shape_generator(move |clip_id, width, height| {
            set_shape_handle.set_shape_size(clip_id.to_string(), width, height);
        });

    let preview_shape_handle = preview_worker.handle();
    app.global::<InspectorBackend>().on_preview_shape_generator(
        move |clip_id, width, height, tick| {
            preview_shape_handle.preview_shape_size(
                clip_id.to_string(),
                width,
                height,
                i64::from(tick),
            );
        },
    );

    let clear_shape_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_clear_shape_generator(move |tick| {
            clear_shape_handle.clear_generator_override(i64::from(tick));
        });

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

    // Effects & transitions: fill the Library catalogs once, then route the
    // inspector/timeline edits through the engine's undoable commands.
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
    let add_effect_handle = preview_worker.handle();
    app.global::<EffectsBackend>()
        .on_add_effect(move |clip_id, effect_id| {
            add_effect_handle.add_effect(clip_id.to_string(), effect_id.to_string());
        });
    let remove_effect_handle = preview_worker.handle();
    app.global::<EffectsBackend>()
        .on_remove_effect(move |clip_id, index| {
            remove_effect_handle.remove_effect(clip_id.to_string(), index.max(0) as u32);
        });
    let set_effect_param_handle = preview_worker.handle();
    app.global::<EffectsBackend>()
        .on_set_effect_param(move |clip_id, index, param, value| {
            set_effect_param_handle.set_effect_param(
                clip_id.to_string(),
                index.max(0) as u32,
                param.to_string(),
                value,
            );
        });
    let add_transition_handle = preview_worker.handle();
    app.global::<EffectsBackend>()
        .on_add_transition(move |clip_id, transition_id| {
            add_transition_handle.add_transition(clip_id.to_string(), transition_id.to_string());
        });
    let remove_transition_handle = preview_worker.handle();
    app.global::<EffectsBackend>()
        .on_remove_transition(move |clip_id| {
            remove_transition_handle.remove_transition(clip_id.to_string());
        });
    let set_transition_handle = preview_worker.handle();
    app.global::<EffectsBackend>()
        .on_set_transition(move |clip_id, duration| {
            set_transition_handle.set_transition(clip_id.to_string(), i64::from(duration));
        });

    // Enumerate system fonts off the UI thread (the scan is slow) and feed
    // the font picker once ready.
    let font_app = app.as_weak();
    std::thread::spawn(move || {
        let families = cutlass_text::system_font_families();
        let _ = slint::invoke_from_event_loop(move || {
            if let Some(app) = font_app.upgrade() {
                let model: Vec<SharedString> = families.into_iter().map(Into::into).collect();
                app.global::<InspectorBackend>()
                    .set_font_families(ModelRc::new(VecModel::from(model)));
            }
        });
    });

    app.run()
}
