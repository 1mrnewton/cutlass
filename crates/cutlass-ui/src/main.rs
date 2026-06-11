mod audio;
mod inspector;
mod preview;
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

use slint::BackendSelector;
use slint::Model;
use slint::Global;
use slint::SharedString;
use slint::wgpu_28::WGPUConfiguration;
use slint::winit_030::WinitWindowAccessor;
use tracing::info;
use tracing_subscriber::EnvFilter;

use cutlass_engine::EngineConfig;

slint::include_modules!();

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

    window_backend.on_close(|| {
        let _ = slint::quit_event_loop();
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

    let magnet_handle = preview_worker.handle();
    editor.on_on_main_magnet_changed(move |enabled| {
        magnet_handle.set_main_magnet(enabled);
    });

    let import_handle = preview_worker.handle();
    editor.on_on_import_clicked(move || {
        // Native picker is modal and must run on the main thread — which is
        // exactly where this Slint callback fires. The engine work happens off
        // this thread once we hand the path to the worker.
        if let Some(path) = rfd::FileDialog::new()
            .add_filter(
                "Media",
                &["mp4", "mov", "mkv", "webm", "m4v", "mp3", "wav", "m4a", "aac", "flac", "ogg"],
            )
            .add_filter("Video", &["mp4", "mov", "mkv", "webm", "m4v"])
            .add_filter("Audio", &["mp3", "wav", "m4a", "aac", "flac", "ogg"])
            .pick_file()
        {
            import_handle.import(path);
        }
    });

    // --- export (title bar → dialog → engine thread → export thread) -----

    let export_backend = app.global::<ExportBackend>();
    export_backend.set_output_path(default_export_path());

    // Native save panel, seeded from the previous choice. Modal on the main
    // thread (same policy as the import picker); returns the current path
    // unchanged when dismissed so the dialog binding is a plain assignment.
    export_backend.on_pick_output(|current| {
        let current = std::path::Path::new(current.as_str());
        let mut dialog = rfd::FileDialog::new().add_filter("MP4 video", &["mp4"]);
        if let Some(dir) = current.parent().filter(|d| d.is_dir()) {
            dialog = dialog.set_directory(dir);
        }
        dialog = dialog.set_file_name(
            current
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "untitled.mp4".into()),
        );
        match dialog.save_file() {
            Some(path) => path.to_string_lossy().into_owned().into(),
            None => current.to_string_lossy().into_owned().into(),
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
        |sequence, duration_ticks, cursor_tick, drop_row, playhead_tick, snap_threshold_ticks, main_magnet| {
            snap::resolve_library_drop(
                &sequence,
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

    let editor_weak = app.global::<EditorStore>().as_weak();
    app.global::<InspectorBackend>()
        .on_resolve_selection(|sequence, track_id, clip_id| {
            inspector::resolve_selection(sequence, track_id.as_str(), clip_id.as_str())
        });
    app.global::<InspectorBackend>()
        .on_set_text_content(move |track_id, clip_id, content| {
            let Some(editor) = editor_weak.upgrade() else {
                return;
            };
            let mut project = editor.get_project();
            inspector::set_text_content(
                &mut project,
                track_id.as_str(),
                clip_id.as_str(),
                content.as_str(),
            );
            editor.set_project(project);
        });

    app.run()
}
