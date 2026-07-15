mod account;
mod agent;
mod agent_senses;
mod agent_session;
mod agent_vision;
mod ai_media;
mod audio;
mod cloud;
mod drafts;
mod inspector;
mod interaction;
mod lottie_stickers;
mod lut_catalog;
mod os_drop;
mod params;
mod paths;
mod placement;
mod preview_gesture;
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
mod transport;
mod window;

use std::cell::Cell;
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;

use cutlass_engine::EngineConfig;
use slint::BackendSelector;
use slint::Global;
use slint::Model;
use slint::ModelRc;
use slint::SharedString;
use slint::VecModel;
use slint::wgpu_28::WGPUConfiguration;
use slint::winit_030::EventResult;
use slint::winit_030::WinitWindowAccessor;
use slint::winit_030::winit::event::WindowEvent;
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
    if let Some(asset) = key.strip_prefix("sticker:") {
        // Only catalog ids drop; a stale key would place an invisible clip.
        cutlass_models::sticker_spec(asset)?;
        return Some(Generator::sticker(asset));
    }
    // A standalone effect-lane segment; the id after the prefix seeds the
    // clip's effect chain (validated by the engine's AddEffect).
    if key.strip_prefix("effect:").is_some() {
        return Some(Generator::Effect);
    }
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

/// A bundled sticker's first frame as a Slint image — the Library tile
/// thumbnail. Blank when decode fails (the tile shows its label only).
fn sticker_thumbnail(spec: &cutlass_models::StickerSpec) -> slint::Image {
    let Ok(frames) = cutlass_decoder::decode_animation(spec.bytes) else {
        return slint::Image::default();
    };
    let first = &frames[0].image;
    let buffer = slint::SharedPixelBuffer::<slint::Rgba8Pixel>::clone_from_slice(
        &first.pixels,
        first.width,
        first.height,
    );
    slint::Image::from_rgba8(buffer)
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

// Extensions accepted by the import dialog and OS file-drop handler.
const MEDIA_IMPORT_EXTENSIONS: &[&str] = &[
    "mp4", "mov", "mkv", "webm", "m4v", "mp3", "wav", "m4a", "aac", "flac", "ogg", "png", "jpg",
    "jpeg", "webp",
];
const VIDEO_IMPORT_EXTENSIONS: &[&str] = &["mp4", "mov", "mkv", "webm", "m4v"];
const AUDIO_IMPORT_EXTENSIONS: &[&str] = &["mp3", "wav", "m4a", "aac", "flac", "ogg"];
const IMAGE_IMPORT_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp"];

fn media_extension_supported(path: &std::path::Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    let ext = ext.to_ascii_lowercase();
    MEDIA_IMPORT_EXTENSIONS
        .iter()
        .any(|&supported| supported == ext)
}

/// Extension-only audio check for files still owned by the OS drag (the
/// probe runs after the drop): drives the hover preview's lane kind.
fn audio_extension(path: &std::path::Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    let ext = ext.to_ascii_lowercase();
    AUDIO_IMPORT_EXTENSIONS
        .iter()
        .any(|&supported| supported == ext)
}

/// Where an OS file drop at window-space `cursor` (logical points) lands on
/// the timeline: `Some((lane row, sequence tick))` when the cursor is over
/// the timeline panel, `None` anywhere else — the caller falls back to the
/// pool-only import. Geometry comes from the AppState mirror TimelinePanel
/// maintains (`sync-os-drop-geometry`); the launch-screen and maximized-
/// preview guards cover the states where that mirror is stale because the
/// panel is unmounted.
fn os_drop_timeline_target(app: &AppWindow, cursor: (f32, f32)) -> Option<(i64, i64)> {
    let state = app.global::<AppState>();
    if state.get_launch_visible() || state.get_preview_maximized() {
        return None;
    }
    let (cx, cy) = cursor;
    let (px, py) = (state.get_timeline_panel_x(), state.get_timeline_panel_y());
    let (pw, ph) = (state.get_timeline_panel_w(), state.get_timeline_panel_h());
    if pw <= 0.0 || ph <= 0.0 || cx < px || cx >= px + pw || cy < py || cy >= py + ph {
        return None;
    }
    let pitch = state.get_timeline_row_pitch();
    let zoom = app.global::<TimelineStore>().get_zoom();
    if pitch <= 0.0 || zoom <= 0.0 {
        return None;
    }
    let view = app.global::<TimelineViewState>();
    // Same math as the library drag targeting in timeline.slint: cursor →
    // lane-content space → tick / row (row 0 is the head spacer, lanes
    // start at row 1).
    let tick = ((cx - state.get_timeline_lanes_x() - view.get_scroll_x()) / zoom)
        .round()
        .max(0.0) as i64;
    let row =
        ((cy - state.get_timeline_lanes_y() - view.get_scroll_y()) / pitch).floor() as i64 - 1;
    Some((row, tick))
}

async fn pick_import_paths() -> Vec<std::path::PathBuf> {
    rfd::AsyncFileDialog::new()
        .add_filter("Media", MEDIA_IMPORT_EXTENSIONS)
        .add_filter("Video", VIDEO_IMPORT_EXTENSIONS)
        .add_filter("Audio", AUDIO_IMPORT_EXTENSIONS)
        .add_filter("Images", IMAGE_IMPORT_EXTENSIONS)
        .pick_files()
        .await
        .map(|files| files.into_iter().map(|f| f.path().to_path_buf()).collect())
        .unwrap_or_default()
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
    let config_path = cutlass_settings::default_config_path();
    let app_settings = cutlass_settings::load(&config_path).unwrap_or_default();

    // AI assistant: a dedicated worker rehearses each prompt on a sandbox
    // engine, then replays the validated plan through the preview worker as
    // one undoable group.
    let agent_store = app.global::<AgentStore>();
    agent_store.set_transcript(ModelRc::new(VecModel::<AgentEntry>::default()));
    agent_store.set_configured(app_settings.ai.is_configured());
    agent_store.set_config_path(config_path.display().to_string().into());

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
    let proxy_handle =
        proxy::spawn(
            interaction_gate.clone(),
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

    // Library stock browsing: search + direct-CDN downloads on their own
    // thread (src/cloud.rs); imports route through the preview worker like
    // any local file.
    let cloud_worker = cloud::CloudWorker::spawn(app.as_weak(), preview_worker.handle())
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
    let templates_worker =
        templates::TemplatesWorker::spawn(app.as_weak(), preview_worker.handle())
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
    let ai_media_worker = ai_media::AiMediaWorker::spawn(app.as_weak(), preview_worker.handle())
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

    // Cutlass account (Settings > Account + launch update nudge): device-
    // flow sign-in, balance, the website hand-off for credits, and the
    // update check on their own thread (src/account.rs); tokens live in
    // the OS keychain.
    let account_worker =
        account::AccountWorker::spawn(app.as_weak()).map_err(slint::PlatformError::Other)?;
    {
        let account_backend = app.global::<AccountBackend>();
        let sign_in_handle = account_worker.handle();
        account_backend.on_sign_in(move || sign_in_handle.sign_in());
        let sign_out_handle = account_worker.handle();
        account_backend.on_sign_out(move || sign_out_handle.sign_out());
        let balance_handle = account_worker.handle();
        account_backend.on_refresh_balance(move || balance_handle.refresh_balance());
        let buy_handle = account_worker.handle();
        account_backend.on_buy_credits(move || buy_handle.buy_credits());
        let update_handle = account_worker.handle();
        account_backend.on_open_update(move || update_handle.open_update());
        // Restore any keychain session and run the update check now, in the
        // background — the launch screen renders regardless.
        account_worker.handle().init();
    }

    // Animated text presets (Library > Text > Presets): catalog fetches on
    // their own thread (src/text_presets.rs); the registry feeds the
    // generated-drop resolver below.
    let text_preset_registry: text_presets::PresetRegistry =
        std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
    let text_presets_worker =
        text_presets::TextPresetsWorker::spawn(app.as_weak(), text_preset_registry.clone())
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
    let lottie_worker =
        lottie_stickers::LottieWorker::spawn(app.as_weak(), lottie_registry.clone())
            .map_err(slint::PlatformError::Other)?;
    {
        let refresh_handle = lottie_worker.handle();
        app.global::<LottieBackend>()
            .on_refresh(move || refresh_handle.refresh());
    }

    // Sound effects (Library > Audio > Sound effects): catalog fetch on its
    // own thread, lazy download + normal media import on click (src/sfx.rs).
    let sfx_worker = sfx::SfxWorker::spawn(app.as_weak(), preview_worker.handle())
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
    let lut_worker = lut_catalog::LutWorker::spawn(app.as_weak(), lut_registry.clone())
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

    let agent_worker = agent::AgentWorker::spawn(preview_worker.handle(), agent_store.as_weak())
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

    let agent_approve = agent_worker.handle();
    agent_store.on_approve_system_tool(move || agent_approve.approve_system_tool());

    let agent_deny = agent_worker.handle();
    agent_store.on_deny_system_tool(move || agent_deny.deny_system_tool());

    let agent_apply = agent_worker.handle();
    agent_store.on_apply_plan(move || agent_apply.apply_plan());

    let agent_discard = agent_worker.handle();
    agent_store.on_discard_plan(move || agent_discard.discard_plan());

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

    // Preview axis: hover-scrub frames without moving the playhead (no audio).
    let hover_frame_handle = preview_worker.handle();
    let hover_playhead_weak = app.as_weak();
    editor.on_on_hover_preview(move |tick| {
        hover_frame_handle.request_frame(i64::from(tick));
    });
    let hover_restore_handle = preview_worker.handle();
    editor.on_on_hover_preview_ended(move || {
        if let Some(app) = hover_playhead_weak.upgrade() {
            let tick = app.global::<TimelineStore>().get_playhead_tick();
            hover_restore_handle.request_frame(i64::from(tick));
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

    let ripple_delete_handle = preview_worker.handle();
    editor.on_on_clips_ripple_deleted(move |clip_ids| {
        let clips: Vec<String> = clip_ids.iter().map(|id| id.to_string()).collect();
        ripple_delete_handle.ripple_delete_clips(clips);
    });

    let reverse_handle = preview_worker.handle();
    editor.on_on_clip_reversed(move |clip_id| {
        reverse_handle.reverse_clip(clip_id.to_string());
    });

    let extract_audio_handle = preview_worker.handle();
    editor.on_on_clip_audio_extracted(move |clip_id| {
        extract_audio_handle.extract_audio(clip_id.to_string());
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
    let marker_set_handle = preview_worker.handle();
    timeline_store.on_on_marker_set(move |marker_id, at_tick, name, color| {
        marker_set_handle.set_marker(
            marker_id.to_string(),
            i64::from(at_tick),
            name.to_string(),
            color.to_string(),
        );
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

    let remove_track_handle = preview_worker.handle();
    editor.on_on_track_removed(move |track_id| {
        remove_track_handle.remove_track_manual(track_id.to_string());
    });
    let move_track_handle = preview_worker.handle();
    editor.on_on_track_moved(move |track_id, index| {
        move_track_handle.move_track_manual(track_id.to_string(), index as usize);
    });
    let rename_track_handle = preview_worker.handle();
    editor.on_on_track_renamed(move |track_id, name| {
        rename_track_handle.set_track_name(track_id.to_string(), name.to_string());
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
    settings_backend.set_ai_use_account(app_settings.ai.use_account);
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
            s.ai.use_account = sb.get_ai_use_account();
            s.appearance.theme =
                cutlass_settings::ThemeChoice::from_index(app.global::<AppStore>().get_theme_id());

            if let Err(e) = cutlass_settings::save(&config_path, &s) {
                tracing::error!("failed to save settings: {e}");
                return;
            }
            app.global::<AgentStore>()
                .set_configured(s.ai.is_configured());
        });
    }

    {
        let app_weak = app.as_weak();
        settings_backend.on_test_connection(move || {
            let Some(app) = app_weak.upgrade() else {
                return;
            };
            let sb = app.global::<SettingsBackend>();
            let base_url = sb.get_ai_base_url().trim().to_string();
            let model = sb.get_ai_model().trim().to_string();
            let api_key = non_empty(&sb.get_ai_api_key());
            let api_key_env = non_empty(&sb.get_ai_api_key_env());
            sb.set_ai_testing(true);
            sb.set_ai_test_ok(false);
            sb.set_ai_test_status(SharedString::new());

            let app_weak = app.as_weak();
            std::thread::spawn(move || {
                let result =
                    cutlass_ai::config::resolve_api_key(api_key.as_deref(), api_key_env.as_deref())
                        .and_then(|key| {
                            cutlass_ai::providers::OpenAiCompatProvider::new(&base_url, &model, key)
                                .test_connection()
                                .map_err(|e| e.to_string())
                        });
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(app) = app_weak.upgrade() {
                        let sb = app.global::<SettingsBackend>();
                        sb.set_ai_testing(false);
                        match result {
                            Ok(msg) => {
                                sb.set_ai_test_ok(true);
                                sb.set_ai_test_status(msg.into());
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
    let play_gate = interaction_gate.clone();
    app.global::<TransportBackend>()
        .on_transport_play(move |tick, speed_num, speed_den| {
            play_gate.set_playing(true);
            if speed_num == 1 && speed_den == 1 {
                play_audio.play(i64::from(tick));
            } else {
                play_audio.pause();
            }
        });

    let pause_audio = audio_system.handle();
    let pause_gate = interaction_gate.clone();
    app.global::<TransportBackend>()
        .on_transport_pause(move || {
            pause_gate.set_playing(false);
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
    let stop_gate = interaction_gate.clone();
    app.global::<TransportBackend>().on_request_stop(move || {
        stop_gate.set_playing(false);
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

    app.global::<DragBackend>().on_resolve_transition_junction(
        |sequence, cursor_tick, hover_row, snap_threshold_ticks| {
            snap::resolve_transition_junction(
                &sequence,
                cursor_tick,
                hover_row,
                snap_threshold_ticks,
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

    app.global::<PreviewBackend>().on_sprite_placement(
        |sequence, clip_id, tick, view_w, view_h, zoom, pan_x, pan_y, gesture_active, gesture| {
            preview_select::sprite_placement_in_viewport(
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

    let gesture_start_handle = preview_worker.handle();
    editor.on_on_preview_gesture_started(move |clip_id, tick| {
        gesture_start_handle.begin_transform_gesture(clip_id.to_string(), i64::from(tick));
    });

    let gesture_abandon_handle = preview_worker.handle();
    editor.on_on_preview_gesture_abandoned(move || {
        gesture_abandon_handle.end_transform_gesture();
    });

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

    let set_filter_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_set_clip_filter(move |clip_id, filter_id, intensity| {
            set_filter_handle.set_clip_filter(
                clip_id.to_string(),
                filter_id.to_string(),
                intensity,
            );
        });

    let set_animation_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_set_clip_animation(move |clip_id, slot, animation_id| {
            set_animation_handle.set_clip_animation(
                clip_id.to_string(),
                slot.to_string(),
                animation_id.to_string(),
            );
        });

    let set_adjust_handle = preview_worker.handle();
    app.global::<InspectorBackend>().on_set_clip_adjust(
        move |clip_id, brightness, contrast, saturation, exposure, temperature| {
            set_adjust_handle.set_clip_adjust(
                clip_id.to_string(),
                cutlass_models::ColorAdjustments {
                    brightness,
                    contrast,
                    saturation,
                    exposure,
                    temperature,
                },
            );
        },
    );

    let preview_look_handle = preview_worker.handle();
    app.global::<InspectorBackend>().on_preview_clip_look(
        move |clip_id,
              filter_id,
              intensity,
              brightness,
              contrast,
              saturation,
              exposure,
              temperature,
              tick| {
            preview_look_handle.preview_clip_look(
                clip_id.to_string(),
                filter_id.to_string(),
                intensity,
                cutlass_models::ColorAdjustments {
                    brightness,
                    contrast,
                    saturation,
                    exposure,
                    temperature,
                },
                i64::from(tick),
            );
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

    // Filter, effect & transition catalogs are filled once from the model
    // catalogs, then inspector/timeline edits route to undoable commands.
    {
        let inspector = app.global::<InspectorBackend>();
        let filter_rows: Vec<CatalogEntry> = cutlass_models::filter_catalog()
            .iter()
            .map(|s| CatalogEntry {
                id: s.id.into(),
                label: s.label.into(),
            })
            .collect();
        inspector.set_filter_catalog(ModelRc::new(VecModel::from(filter_rows)));

        let animation_in: Vec<CatalogEntry> = cutlass_models::animation_catalog()
            .iter()
            .filter(|s| s.slot == cutlass_models::AnimationSlot::In)
            .map(|s| CatalogEntry {
                id: s.id.into(),
                label: s.label.into(),
            })
            .collect();
        inspector.set_animation_in_catalog(ModelRc::new(VecModel::from(animation_in)));

        let animation_out: Vec<CatalogEntry> = cutlass_models::animation_catalog()
            .iter()
            .filter(|s| s.slot == cutlass_models::AnimationSlot::Out)
            .map(|s| CatalogEntry {
                id: s.id.into(),
                label: s.label.into(),
            })
            .collect();
        inspector.set_animation_out_catalog(ModelRc::new(VecModel::from(animation_out)));

        let animation_combo: Vec<CatalogEntry> = cutlass_models::animation_catalog()
            .iter()
            .filter(|s| s.slot == cutlass_models::AnimationSlot::Combo && !s.text_only)
            .map(|s| CatalogEntry {
                id: s.id.into(),
                label: s.label.into(),
            })
            .collect();
        inspector.set_animation_combo_catalog(ModelRc::new(VecModel::from(animation_combo)));

        let animation_text_combo: Vec<CatalogEntry> = cutlass_models::animation_catalog()
            .iter()
            .filter(|s| s.slot == cutlass_models::AnimationSlot::Combo)
            .map(|s| CatalogEntry {
                id: s.id.into(),
                label: s.label.into(),
            })
            .collect();
        inspector
            .set_animation_text_combo_catalog(ModelRc::new(VecModel::from(animation_text_combo)));
    }
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
        let sticker_rows: Vec<StickerTile> = cutlass_models::sticker_catalog()
            .iter()
            .map(|s| StickerTile {
                id: s.id.into(),
                label: s.label.into(),
                icon: sticker_thumbnail(s),
            })
            .collect();
        effects.set_sticker_catalog(ModelRc::new(VecModel::from(sticker_rows)));
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
