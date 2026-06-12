mod agent;
mod audio;
mod autosave;
mod inspector;
mod params;
mod preview;
mod preview_gesture;
mod preview_select;
mod preview_worker;
mod projection;
mod recent;
mod ruler;
mod selection;
mod snap;
mod strips;
mod thumbnails;
mod timecode;
mod timeline;
mod transport;

use slint::BackendSelector;
use slint::Model;
use slint::ModelRc;
use slint::Global;
use slint::SharedString;
use slint::VecModel;
use slint::wgpu_28::WGPUConfiguration;
use slint::winit_030::WinitWindowAccessor;
use tracing::info;
use tracing_subscriber::EnvFilter;

use cutlass_engine::EngineConfig;

slint::include_modules!();

/// Run `f` on the next event-loop turn, outside whatever callback is
/// currently executing. Used to flip Timer-bound state (see `request-stop`)
/// without re-entering Slint's timer machinery. Must never run anything that
/// blocks on a nested run loop (e.g. a modal `rfd::FileDialog`): the closure
/// executes inside Slint's timer activation, and the display link re-entering
/// it aborts with "Recursion in timer code".
fn defer_main_thread(f: impl FnOnce() + Send + 'static) {
    slint::Timer::single_shot(std::time::Duration::ZERO, f);
}

/// Map a "Titles & shapes" tile key to the engine generator it creates, with
/// the default styling for a freshly dropped clip. `None` for an unknown key.
fn generator_from_key(key: &str) -> Option<cutlass_models::Generator> {
    use cutlass_models::{Generator, Shape};
    Some(match key {
        "text" => Generator::text("Title"),
        "solid" => Generator::SolidColor {
            rgba: [30, 30, 30, 255],
        },
        "rect" => Generator::Shape {
            shape: Shape::Rectangle,
            rgba: [255, 255, 255, 255],
        },
        "ellipse" => Generator::Shape {
            shape: Shape::Ellipse,
            rgba: [255, 255, 255, 255],
        },
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
        "scale" => (ClipParam::Scale, ParamValue::Scalar(value_x)),
        "rotation" => (ClipParam::Rotation, ParamValue::Scalar(value_x)),
        "opacity" => (ClipParam::Opacity, ParamValue::Scalar(value_x)),
        _ => return None,
    })
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
            &["mp4", "mov", "mkv", "webm", "m4v", "mp3", "wav", "m4a", "aac", "flac", "ogg"],
        )
        .add_filter("Video", &["mp4", "mov", "mkv", "webm", "m4v"])
        .add_filter("Audio", &["mp3", "wav", "m4a", "aac", "flac", "ogg"])
        .pick_file()
        .await
        .map(|file| file.path().to_path_buf())
}

/// Save panel for the first save / Save As (lifecycle roadmap Phase 1).
/// `default_stem` pre-fills the field (the current file stem on Save As,
/// "Untitled" before the first save); the `.cutlass` extension is enforced
/// on whatever the user types.
async fn pick_save_path(default_stem: String) -> Option<std::path::PathBuf> {
    let stem = if default_stem.is_empty() { "Untitled".to_owned() } else { default_stem };
    let mut path = rfd::AsyncFileDialog::new()
        .add_filter("Cutlass project", &["cutlass"])
        .set_file_name(format!("{stem}.cutlass"))
        .save_file()
        .await
        .map(|file| file.path().to_path_buf())?;
    if path.extension().is_none_or(|ext| ext != "cutlass") {
        // Append rather than `set_extension`: a typed "v1.2" must become
        // "v1.2.cutlass", not "v1.cutlass".
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "Untitled".into());
        path.set_file_name(format!("{name}.cutlass"));
    }
    Some(path)
}

async fn pick_open_path() -> Option<std::path::PathBuf> {
    rfd::AsyncFileDialog::new()
        .add_filter("Cutlass project", &["cutlass"])
        .pick_file()
        .await
        .map(|file| file.path().to_path_buf())
}

// --- session transitions & the unsaved-changes guard (Phase 2) -----------
//
// Open, New, and window close all destroy the current session, so they all
// funnel through `request_transition`: clean sessions proceed immediately,
// dirty ones get one native Save / Don't Save / Cancel dialog. Choosing
// "Save" parks the transition in `pending` and triggers the ordinary save
// path (including the first-save picker); the worker's `save-finished`
// signal then continues or aborts it. A cancelled save picker aborts too
// (the save handler clears `pending`).

#[derive(Clone, PartialEq)]
enum Transition {
    NewProject,
    OpenProject,
    /// Open a known `.cutlass` path (Open Recent, welcome panel) — same
    /// guard as `OpenProject`, no picker.
    OpenPath(std::path::PathBuf),
    CloseWindow,
}

type PendingTransition = std::rc::Rc<std::cell::RefCell<Option<Transition>>>;
type GuardOpen = std::rc::Rc<std::cell::Cell<bool>>;

fn perform_transition(handle: &preview_worker::WorkerHandle, transition: Transition) {
    match transition {
        Transition::NewProject => handle.new_project(),
        Transition::OpenProject => {
            let handle = handle.clone();
            let task = slint::spawn_local(async move {
                if let Some(path) = pick_open_path().await {
                    handle.open_project(path);
                }
            });
            if let Err(e) = task {
                tracing::error!("failed to open project dialog: {e}");
            }
        }
        Transition::OpenPath(path) => handle.open_project(path),
        Transition::CloseWindow => {
            let _ = slint::quit_event_loop();
        }
    }
}

fn request_transition(
    app_weak: &slint::Weak<AppWindow>,
    handle: &preview_worker::WorkerHandle,
    pending: &PendingTransition,
    guard_open: &GuardOpen,
    transition: Transition,
) {
    let Some(app) = app_weak.upgrade() else {
        return;
    };
    // One lifecycle transition at a time: ignore requests while the guard
    // dialog is up or a save-then-continue is in flight.
    if guard_open.get() || pending.borrow().is_some() {
        return;
    }
    if !app.global::<EditorStore>().get_project_dirty() {
        perform_transition(handle, transition);
        return;
    }
    guard_open.set(true);
    let app_weak = app_weak.clone();
    let handle = handle.clone();
    let pending = pending.clone();
    let guard = guard_open.clone();
    let task = slint::spawn_local(async move {
        let choice = rfd::AsyncMessageDialog::new()
            .set_level(rfd::MessageLevel::Warning)
            .set_title("Unsaved changes")
            .set_description("This project has unsaved changes. Save them before continuing?")
            .set_buttons(rfd::MessageButtons::YesNoCancelCustom(
                "Save".to_owned(),
                "Don't Save".to_owned(),
                "Cancel".to_owned(),
            ))
            .show()
            .await;
        guard.set(false);
        match choice {
            rfd::MessageDialogResult::Custom(label) if label == "Save" => {
                *pending.borrow_mut() = Some(transition);
                if let Some(app) = app_weak.upgrade() {
                    app.global::<EditorStore>().invoke_on_save_requested(false);
                }
            }
            rfd::MessageDialogResult::Custom(label) if label == "Don't Save" => {
                // Closing discards for good: drop the autosave slot too, or
                // the next launch would offer back work the user just threw
                // away. Open/New don't need this — the worker's next sweep
                // sees the replaced session and cleans the slot itself.
                if transition == Transition::CloseWindow
                    && let Some(app) = app_weak.upgrade()
                {
                    discard_session_autosave(&app);
                }
                perform_transition(&handle, transition);
            }
            _ => {} // Cancel / dismissed: stay in the session.
        }
    });
    if let Err(e) = task {
        guard_open.set(false);
        tracing::error!("failed to open unsaved-changes dialog: {e}");
    }
}

/// Remove the session's autosave slot (best effort, benign if absent).
/// Addressed the same way the worker writes it: by the project's file
/// path, or the process-keyed unsaved slot before the first save.
fn discard_session_autosave(app: &AppWindow) {
    let editor = app.global::<EditorStore>();
    let source = editor
        .get_project_has_path()
        .then(|| std::path::PathBuf::from(editor.get_project_file_path().to_string()));
    autosave::discard(&autosave::slot_for(&autosave::default_dir(), source.as_deref()));
}

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
    dialog.save_file().await.map(|file| file.path().to_path_buf())
}

/// Prefilled export destination: ~/Movies when present, else the home
/// directory, else the working directory. Only seeds the save panel — the
/// user picks the real spot from the dialog.
fn default_export_path() -> SharedString {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(std::path::PathBuf::from);
    let dir = match home {
        Some(home) => {
            let movies = home.join("Movies");
            if movies.is_dir() { movies } else { home }
        }
        None => std::path::PathBuf::from("."),
    };
    dir.join("untitled.mp4").to_string_lossy().into_owned().into()
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

    let app_weak = app.as_weak();
    slint::invoke_from_event_loop(move || {
        if let Some(app) = app_weak.upgrade() {
            app.window().set_maximized(true);
            app.global::<WindowBackend>().set_maximized(true);
        }
    })
    .map_err(|e| slint::PlatformError::from(format!("failed to schedule maximize: {e}")))?;

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

    // `WindowBackend.close` is wired after the engine worker spawns: the
    // unsaved-changes guard needs the worker handle (see Phase 2 wiring
    // below).

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
    let preview_store_weak = app.global::<PreviewStore>().as_weak();
    let editor_store_weak = app.global::<EditorStore>().as_weak();

    // Library tile thumbnails decode on their own thread so imports never
    // stall preview scrubbing. Keep the worker alive for the app's lifetime.
    let thumbnail_worker =
        thumbnails::ThumbnailWorker::spawn(app.global::<EditorStore>().as_weak())
            .map_err(slint::PlatformError::from)?;

    // Timeline clip content (filmstrip frames, waveform tiles) decodes on a
    // third thread: a long strip batch must not delay library tiles, and
    // neither may ever touch the UI or engine threads.
    let strip_worker = strips::StripWorker::spawn(app.global::<StripBackend>().as_weak())
        .map_err(slint::PlatformError::from)?;

    // Audio playback (playback roadmap Phase 3): device output + mixer
    // thread. The `!Send` cpal stream lives here on the main thread for the
    // app's lifetime; handles go to the worker (snapshots) and the
    // transport callbacks (clock + play/pause). A machine without an
    // output device degrades to the wall-clock transport, silent.
    let audio_system = audio::AudioSystem::start();

    let (preview_worker, session) = preview_worker::PreviewWorker::spawn(
        EngineConfig::default(),
        preview_store_weak,
        editor_store_weak,
        app.global::<ExportBackend>().as_weak(),
        thumbnail_worker.handle(),
        strip_worker.handle(),
        audio_system.handle(),
    )
    .map_err(slint::PlatformError::from)?;

    info!(
        duration_ticks = session.duration_ticks,
        tl_rate = ?session.tl_rate,
        "preview worker ready; import media to populate the timeline"
    );

    // AI assistant (ai-agent roadmap Phase 4): a dedicated worker thread
    // rehearses each prompt on a sandbox engine, then replays the validated
    // plan through the preview worker as one undoable group. The transcript
    // model is created here so the worker can mutate rows while streaming.
    let agent_store = app.global::<AgentStore>();
    agent_store.set_transcript(ModelRc::new(VecModel::<AgentEntry>::default()));
    let agent_worker = agent::AgentWorker::spawn(preview_worker.handle(), agent_store.as_weak())
        .map_err(slint::PlatformError::from)?;

    // The send-time editor snapshot: this is how "the selected clip" and
    // "at the playhead" resolve to ids and seconds for the model.
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

    let agent_apply = agent_worker.handle();
    agent_store.on_apply_plan(move || agent_apply.apply_plan());

    let agent_discard = agent_worker.handle();
    agent_store.on_discard_plan(move || agent_discard.discard_plan());

    // Open / New / Restore replaced the project: a running prompt and any
    // parked plan rehearsed against the old one. Cancel and discard.
    let agent_session = agent_worker.handle();
    agent_store.on_session_changed(move || {
        agent_session.cancel();
        agent_session.discard_plan();
    });

    let editor = app.global::<EditorStore>();

    // Playhead moves (ruler scrub, frame-step keys, Home/End) become preview
    // frame requests; the worker coalesces a burst to the newest tick.
    let frame_handle = preview_worker.handle();
    editor.on_on_playhead_changed(move |tick| {
        frame_handle.request_frame(i64::from(tick));
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

    // --- project lifecycle (Phase 1 + 2) ----------------------------------

    // A guarded transition (open/new/close) waiting on a "Save" choice, and
    // the lock that keeps a second guard dialog from stacking on the first.
    let pending_transition: PendingTransition = std::rc::Rc::new(std::cell::RefCell::new(None));
    let guard_open: GuardOpen = std::rc::Rc::new(std::cell::Cell::new(false));

    // Save / Save As (Cmd/Ctrl+S / +Shift+S, title bar button). A plain
    // save on a session that already has a file goes straight to the
    // worker; Save As — and the first save — pick a path first. The
    // worker republishes the projection on success, which clears the
    // title bar's dirty dot. A cancelled picker aborts any transition
    // waiting on this save (Save-before-open/new/close).
    let save_handle = preview_worker.handle();
    let app_weak = app.as_weak();
    let save_pending = pending_transition.clone();
    editor.on_on_save_requested(move |save_as| {
        let Some(app) = app_weak.upgrade() else {
            return;
        };
        let editor = app.global::<EditorStore>();
        if !save_as && editor.get_project_has_path() {
            save_handle.save_project(None);
            return;
        }
        let save_handle = save_handle.clone();
        let pending = save_pending.clone();
        let default_stem = editor.get_project_file_name().to_string();
        let task = slint::spawn_local(async move {
            if let Some(path) = pick_save_path(default_stem).await {
                save_handle.save_project(Some(path));
            } else {
                pending.borrow_mut().take();
            }
        });
        if let Err(e) = task {
            tracing::error!("failed to open save dialog: {e}");
        }
    });

    // The worker reports every save attempt; a parked transition continues
    // on success and aborts on failure (the worker already surfaced the
    // error dialog).
    let finish_handle = preview_worker.handle();
    let finish_pending = pending_transition.clone();
    editor.on_save_finished(move |ok| {
        let Some(transition) = finish_pending.borrow_mut().take() else {
            return;
        };
        if ok {
            perform_transition(&finish_handle, transition);
        }
    });

    // Open / New (Cmd/Ctrl+O / +N) — guarded session replacements.
    let open_handle = preview_worker.handle();
    let app_weak = app.as_weak();
    let open_pending = pending_transition.clone();
    let open_guard = guard_open.clone();
    editor.on_on_open_requested(move || {
        request_transition(
            &app_weak,
            &open_handle,
            &open_pending,
            &open_guard,
            Transition::OpenProject,
        );
    });

    let new_handle = preview_worker.handle();
    let app_weak = app.as_weak();
    let new_pending = pending_transition.clone();
    let new_guard = guard_open.clone();
    editor.on_on_new_requested(move || {
        request_transition(
            &app_weak,
            &new_handle,
            &new_pending,
            &new_guard,
            Transition::NewProject,
        );
    });

    // Open Recent (lifecycle roadmap Phase 3): a known path skips the
    // picker but runs the same unsaved-changes guard. A file deleted since
    // the list was read fails like any open (session-error dialog).
    let recent_handle = preview_worker.handle();
    let app_weak = app.as_weak();
    let recent_pending = pending_transition.clone();
    let recent_guard = guard_open.clone();
    editor.on_on_open_recent_requested(move |path| {
        request_transition(
            &app_weak,
            &recent_handle,
            &recent_pending,
            &recent_guard,
            Transition::OpenPath(std::path::PathBuf::from(path.as_str())),
        );
    });

    // Seed the MRU list (File menu, welcome panel) from disk; the worker
    // republishes it after every successful save/open.
    editor.set_recent_projects(slint::ModelRc::new(slint::VecModel::from(recent::to_rows(
        &recent::read(&recent::default_path()),
    ))));

    // Window close — the title-bar ✕ and the OS close request both consult
    // the same guard before the event loop quits.
    let close_handle = preview_worker.handle();
    let app_weak = app.as_weak();
    let close_pending = pending_transition.clone();
    let close_guard = guard_open.clone();
    app.global::<WindowBackend>().on_close(move || {
        request_transition(
            &app_weak,
            &close_handle,
            &close_pending,
            &close_guard,
            Transition::CloseWindow,
        );
    });

    let close_handle = preview_worker.handle();
    let app_weak = app.as_weak();
    let close_pending = pending_transition.clone();
    let close_guard = guard_open.clone();
    app.window().on_close_requested(move || {
        request_transition(
            &app_weak,
            &close_handle,
            &close_pending,
            &close_guard,
            Transition::CloseWindow,
        );
        slint::CloseRequestResponse::KeepWindowShown
    });

    // --- autosave & crash recovery (Phase 4) ------------------------------

    // Periodic sweep: the worker snapshots dirty sessions to the sidecar
    // slot (never the user's file) and cleans stale slots up. The timer
    // lives until `run()` returns.
    let autosave_timer = slint::Timer::default();
    let autosave_handle = preview_worker.handle();
    autosave_timer.start(
        slint::TimerMode::Repeated,
        autosave::SWEEP_INTERVAL,
        move || autosave_handle.autosave(),
    );

    // Launch offer: a leftover slot means the previous session never got to
    // clean up (a crash — clean exits remove their slots or date them older
    // than the saved file). Delayed a beat so the window is up before the
    // dialog sheets over it; "Restore" loads the snapshot bound to the real
    // file, "Discard" deletes it, dismissing keeps it for next launch.
    let restore_handle = preview_worker.handle();
    slint::Timer::single_shot(std::time::Duration::from_millis(300), move || {
        let Some(candidate) = autosave::newest_candidate(&autosave::default_dir()) else {
            return;
        };
        let task = slint::spawn_local(async move {
            let name = candidate
                .source
                .as_deref()
                .and_then(|p| p.file_stem())
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "an unsaved project".to_owned());
            let choice = rfd::AsyncMessageDialog::new()
                .set_level(rfd::MessageLevel::Warning)
                .set_title("Restore unsaved work?")
                .set_description(format!(
                    "Cutlass didn't shut down cleanly, and unsaved work for \
                     \u{201c}{name}\u{201d} was recovered. Restore it?"
                ))
                .set_buttons(rfd::MessageButtons::OkCancelCustom(
                    "Restore".to_owned(),
                    "Discard".to_owned(),
                ))
                .show()
                .await;
            match choice {
                rfd::MessageDialogResult::Custom(label) if label == "Restore" => {
                    restore_handle.restore_autosave(candidate.autosave, candidate.source);
                }
                rfd::MessageDialogResult::Custom(label) if label == "Discard" => {
                    autosave::discard(&candidate.autosave);
                }
                _ => {} // dismissed: leave the slot; offer again next launch
            }
        });
        if let Err(e) = task {
            tracing::error!("failed to offer autosave recovery: {e}");
        }
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
            if let Some(path) = pick_export_path(current).await {
                if let Some(backend) = backend_weak.upgrade() {
                    backend.set_output_path(path.to_string_lossy().into_owned().into());
                }
            }
        });
        if let Err(e) = task {
            tracing::error!("failed to open export dialog: {e}");
        }
    });

    let export_handle = preview_worker.handle();
    export_backend.on_start(move |path, target_height, fps_num, crf| {
        export_handle.export(preview_worker::ExportRequest {
            path: std::path::PathBuf::from(path.as_str()),
            target_height: u32::try_from(target_height).ok().filter(|&h| h > 0),
            fps_num: (fps_num > 0).then_some(fps_num),
            crf: crf.clamp(0, 51) as u8,
        });
    });

    let export_cancel_handle = preview_worker.handle();
    export_backend.on_cancel(move || {
        export_cancel_handle.cancel_export();
    });

    let timeline = app.global::<TimelineLib>();
    timeline.on_sequence_duration(timeline::sequence_duration);
    timeline.on_format_timecode(|frame, fps_num, fps_den, drop_frame| {
        SharedString::from(crate::timecode::format_timecode(
            i64::from(frame),
            i64::from(fps_num),
            i64::from(fps_den),
            drop_frame,
        ))
    });

    app.global::<RulerBackend>().on_ticks(|scroll_x, viewport_w, zoom, fps_num, fps_den| {
        ruler::ticks_model(scroll_x, viewport_w, zoom, fps_num, fps_den)
    });

    // Playback clock (playback roadmap Phases 1 + 3): at speed 1/1 with a
    // live output device, *consumed audio frames* are the clock — video
    // follows the sound card, which is what keeps A/V locked. Shuttle
    // speeds and deviceless machines use the scaled wall clock instead.
    let clock_audio = audio_system.handle();
    app.global::<TransportBackend>().on_playback_tick(
        move |anchor_tick, anchor_ms, now_ms, fps_num, fps_den, speed_num, speed_den| {
            if clock_audio.active() && speed_num == 1 && speed_den == 1 {
                clock_audio
                    .current_tick(fps_num, fps_den)
                    .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
            } else {
                transport::playback_tick_scaled(
                    anchor_tick, anchor_ms, now_ms, fps_num, fps_den, speed_num, speed_den,
                )
            }
        },
    );

    // Transport intent → audio engine. Play doubles as the mid-playback
    // seek; non-1x speeds play muted (varispeed audio is a later phase).
    let play_audio = audio_system.handle();
    app.global::<TransportBackend>().on_transport_play(move |tick, speed_num, speed_den| {
        if speed_num == 1 && speed_den == 1 {
            play_audio.play(i64::from(tick));
        } else {
            play_audio.pause();
        }
    });

    let pause_audio = audio_system.handle();
    app.global::<TransportBackend>().on_transport_pause(move || {
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

    // Timeline clip content tiles (Phase 8). Cache lookups on the UI thread;
    // misses queue decode work on the strip thread and come back through a
    // `StripBackend.generation` bump (the trailing argument both callbacks
    // take exists only to re-trigger evaluation on delivery).
    let filmstrip_handle = strip_worker.handle();
    app.global::<StripBackend>().on_filmstrip_tiles(
        move |media_id, source_in_s, duration, fps_num, fps_den, zoom, from_bucket, to_bucket, _generation| {
            strips::filmstrip_tiles(
                &filmstrip_handle,
                media_id.as_str(),
                source_in_s,
                duration,
                fps_num,
                fps_den,
                zoom,
                from_bucket,
                to_bucket,
            )
        },
    );

    let waveform_handle = strip_worker.handle();
    app.global::<StripBackend>().on_waveform_tiles(
        move |media_id, source_in_s, duration, fps_num, fps_den, zoom, from_bucket, to_bucket, _generation| {
            strips::waveform_tiles(
                &waveform_handle,
                media_id.as_str(),
                source_in_s,
                duration,
                fps_num,
                fps_den,
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
        |sequence, source_track_id, dragging_clip_id, dx_ticks, hover_row, playhead_tick, snap_threshold_ticks, main_magnet| {
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
        |sequence, lane_kind, duration_ticks, cursor_tick, drop_row, playhead_tick, snap_threshold_ticks, main_magnet| {
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
        |sequence, track_id, clip_id, trim_head, dx_ticks, playhead_tick, snap_threshold_ticks, link_enabled| {
            snap::resolve_clip_trim(
                &sequence,
                track_id.as_str(),
                clip_id.as_str(),
                trim_head,
                dx_ticks,
                playhead_tick,
                snap_threshold_ticks,
                link_enabled,
            )
        },
    );

    // --- Phase 10: multi-selection, group drag, linkage -------------------

    app.global::<SelectionBackend>()
        .on_contains(|ids, clip_id| selection::selection_contains(&ids, clip_id.as_str()));

    app.global::<SelectionBackend>()
        .on_select_clip(|sequence, track_id, clip_id, link_enabled| {
            selection::select_clip(&sequence, track_id.as_str(), clip_id.as_str(), link_enabled)
        });

    app.global::<SelectionBackend>()
        .on_toggle_clip(|sequence, current, track_id, clip_id, link_enabled| {
            selection::toggle_clip(
                &sequence,
                &current,
                track_id.as_str(),
                clip_id.as_str(),
                link_enabled,
            )
        });

    app.global::<SelectionBackend>()
        .on_resolve_marquee(|sequence, tick0, tick1, row0, row1, link_enabled| {
            selection::resolve_marquee(&sequence, tick0, tick1, row0, row1, link_enabled)
        });

    // --- preview roadmap Phase 2: click-to-select in the viewport ---------

    app.global::<PreviewBackend>().on_hit_test(|sequence, tick, x, y, view_w, view_h| {
        preview_select::hit_test(&sequence, tick, x, y, view_w, view_h)
    });

    app.global::<PreviewBackend>().on_selection_box(
        |sequence, clip_id, tick, view_w, view_h, gesture_active, gesture| {
            preview_select::selection_box(
                &sequence,
                clip_id.as_str(),
                tick,
                view_w,
                view_h,
                gesture_active.then_some(&gesture),
            )
        },
    );

    // --- preview roadmap Phase 3: move gesture, guides, nudges ------------

    app.global::<PreviewBackend>().on_resolve_drag(
        |sequence, clip_id, tick, press_x, press_y, cursor_x, cursor_y, view_w, view_h, snap_tol| {
            preview_gesture::resolve_drag(
                &sequence,
                clip_id.as_str(),
                tick,
                press_x,
                press_y,
                cursor_x,
                cursor_y,
                view_w,
                view_h,
                snap_tol,
            )
        },
    );

    app.global::<PreviewBackend>().on_nudge(|sequence, clip_id, tick, dx, dy| {
        preview_gesture::nudge(&sequence, clip_id.as_str(), tick, dx, dy)
    });

    // --- preview roadmap Phase 4: scale & rotate handles -------------------

    app.global::<PreviewBackend>().on_resolve_scale(
        |sequence, clip_id, tick, press_x, press_y, cursor_x, cursor_y, view_w, view_h| {
            preview_gesture::resolve_scale(
                &sequence,
                clip_id.as_str(),
                tick,
                press_x,
                press_y,
                cursor_x,
                cursor_y,
                view_w,
                view_h,
            )
        },
    );

    app.global::<PreviewBackend>().on_resolve_rotate(
        |sequence, clip_id, tick, press_x, press_y, cursor_x, cursor_y, view_w, view_h, snap_deg| {
            preview_gesture::resolve_rotate(
                &sequence,
                clip_id.as_str(),
                tick,
                press_x,
                press_y,
                cursor_x,
                cursor_y,
                view_w,
                view_h,
                snap_deg,
            )
        },
    );

    let override_handle = preview_worker.handle();
    editor.on_on_preview_transform_overridden(
        move |clip_id, pos_x, pos_y, scale, rotation, opacity, tick| {
            override_handle.transform_override(
                clip_id.to_string(),
                cutlass_models::ClipTransform {
                    position: [pos_x, pos_y],
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
        move |clip_id, pos_x, pos_y, scale, rotation, opacity, tick| {
            transform_commit_handle.set_transform(
                clip_id.to_string(),
                cutlass_models::ClipTransform {
                    position: [pos_x, pos_y],
                    scale,
                    rotation,
                    opacity,
                },
                i64::from(tick),
            );
        },
    );

    app.global::<DragBackend>()
        .on_group_floaters(|sequence, ids| selection::group_floaters(&sequence, &ids));

    app.global::<DragBackend>().on_resolve_group_drag(
        |sequence, ids, anchor_track_id, anchor_clip_id, dx_ticks, hover_row, playhead_tick, snap_threshold_ticks| {
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

    let trim_handle = preview_worker.handle();
    editor.on_on_clip_trimmed(move |clip_id, start_tick, duration_ticks| {
        trim_handle.trim_clip(
            clip_id.to_string(),
            i64::from(start_tick),
            i64::from(duration_ticks),
        );
    });

    // --- Phase 5: selection ops & history (UI gates, engine validates) ---

    let delete_handle = preview_worker.handle();
    editor.on_on_clips_deleted(move |clip_ids| {
        let clips: Vec<String> = clip_ids.iter().map(|id| id.to_string()).collect();
        delete_handle.remove_clips(clips);
    });

    let split_handle = preview_worker.handle();
    editor.on_on_clip_split(move |clip_id, at_tick| {
        split_handle.split_clip(clip_id.to_string(), i64::from(at_tick));
    });

    let undo_handle = preview_worker.handle();
    editor.on_on_undo(move || {
        undo_handle.undo();
    });

    let redo_handle = preview_worker.handle();
    editor.on_on_redo(move || {
        redo_handle.redo();
    });

    let copy_handle = preview_worker.handle();
    editor.on_on_clip_copied(move |clip_id| {
        copy_handle.copy_clip(clip_id.to_string());
    });

    let paste_handle = preview_worker.handle();
    editor.on_on_paste_at(move |tick| {
        paste_handle.paste_at(i64::from(tick));
    });

    let duplicate_handle = preview_worker.handle();
    editor.on_on_clip_duplicated(move |clip_id| {
        duplicate_handle.duplicate_clip(clip_id.to_string());
    });

    let track_flag_handle = preview_worker.handle();
    editor.on_on_track_flag_toggled(move |track_id, flag, value| {
        let flag = match flag.as_str() {
            "enabled" => preview_worker::TrackFlag::Enabled,
            "muted" => preview_worker::TrackFlag::Muted,
            "locked" => preview_worker::TrackFlag::Locked,
            other => {
                tracing::error!(flag = other, "ignoring unknown track flag");
                return;
            }
        };
        track_flag_handle.set_track_flag(track_id.to_string(), flag, value);
    });

    app.global::<InspectorBackend>()
        .on_resolve_selection(|sequence, track_id, clip_id| {
            inspector::resolve_selection(sequence, track_id.as_str(), clip_id.as_str())
        });

    app.global::<InspectorBackend>()
        .on_sample_transform(|clip, playhead| inspector::sample_transform(&clip, playhead));

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
                tracing::error!(param = param.as_str(), "ignoring keyframe removal on unknown param");
                return;
            };
            kf_remove_handle.remove_param_keyframe(clip_id.to_string(), param, i64::from(tick));
        });
    let set_text_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_set_text_generator(move |_track_id, clip_id, content, style| {
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
        });

    let preview_text_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_preview_text_generator(move |clip_id, content, style, tick| {
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
        });

    let clear_text_handle = preview_worker.handle();
    app.global::<InspectorBackend>()
        .on_clear_text_generator(move |tick| {
            clear_text_handle.clear_generator_override(i64::from(tick));
        });

    app.global::<InspectorBackend>()
        .on_filter_fonts(|query, items| {
            let needle = query.to_lowercase();
            let filtered: Vec<SharedString> = items
                .iter()
                .filter(|family| needle.is_empty() || family.as_str().to_lowercase().contains(&needle))
                .collect();
            ModelRc::new(VecModel::from(filtered))
        });

    // Enumerate system fonts off the UI thread (the scan is slow) and feed the
    // Font picker once ready.
    let font_app = app.as_weak();
    std::thread::spawn(move || {
        let families = cutlass_engine::system_font_families();
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
