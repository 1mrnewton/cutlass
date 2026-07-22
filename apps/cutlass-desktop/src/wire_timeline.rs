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

pub(crate) fn wire_timeline(
    app: &AppWindow,
    preview_worker: &crate::preview_worker::PreviewWorker,
    download_cache: &Arc<cutlass_cloud::cache::DownloadCache>,
) {
    let editor = app.global::<EditorStore>();

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
    let open_download_cache = Arc::clone(download_cache);
    editor.on_on_open_requested(move || {
        change_session(&open_handle, &open_download_cache, SessionChange::Import);
    });

    let new_handle = preview_worker.handle();
    let new_download_cache = Arc::clone(download_cache);
    editor.on_on_new_requested(move || {
        change_session(&new_handle, &new_download_cache, SessionChange::New);
    });

    // Launch gallery card → open that draft by its project path.
    let open_draft_handle = preview_worker.handle();
    let open_draft_download_cache = Arc::clone(download_cache);
    editor.on_on_open_project_requested(move |path| {
        change_session(
            &open_draft_handle,
            &open_draft_download_cache,
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
    refresh_projects(app);

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
            if let Err(error) = external::reveal_path(&path) {
                tracing::error!(%error, "failed to reveal export output");
            }
        }
    });

    // --- keyframe graph editor drawer ------------------------------------
    wire_graph_editor(app, preview_worker);
}

#[derive(Clone)]
struct GraphDragSession {
    clip_id: String,
    key: String,
    channel: i32,
    from_tick: i32,
    tick: i32,
    value: f32,
    moved: bool,
    mapping: crate::graph_editor::PlotMapping,
    playhead: i32,
}

#[derive(Clone)]
struct GraphHandleSession {
    clip_id: String,
    key: String,
    channel: i32,
    from_tick: i32,
    which: crate::graph_editor::HandleId,
    points: [f32; 4],
    moved: bool,
    handles: crate::graph_editor::SegmentHandles,
    mapping: crate::graph_editor::PlotMapping,
    playhead: i32,
}

fn apply_graph_geometry(g: &GraphBackend, geo: crate::graph_editor::GraphGeometry) {
    g.set_path_commands(geo.path_commands);
    g.set_dots(ModelRc::from(Rc::new(VecModel::from(geo.dots))));
    g.set_y_min_label(geo.y_min_label);
    g.set_y_mid_label(geo.y_mid_label);
    g.set_y_max_label(geo.y_max_label);
    g.set_grid_min_y(geo.grid_min_y);
    g.set_grid_mid_y(geo.grid_mid_y);
    g.set_grid_max_y(geo.grid_max_y);
    g.set_playhead_x(geo.playhead_x);
    g.set_playhead_visible(geo.playhead_visible);
    g.set_plot_w(geo.plot_w);
    g.set_plot_h(geo.plot_h);
    if let Some(h) = geo.handles {
        g.set_handles_visible(true);
        g.set_handle_a_x(h.a_px);
        g.set_handle_a_y(h.a_py);
        g.set_handle_b_x(h.b_px);
        g.set_handle_b_y(h.b_py);
        g.set_handle_stem_a(SharedString::from(format!(
            "M {:.2} {:.2} L {:.2} {:.2}",
            h.start_px, h.start_py, h.a_px, h.a_py
        )));
        g.set_handle_stem_b(SharedString::from(format!(
            "M {:.2} {:.2} L {:.2} {:.2}",
            h.end_px, h.end_py, h.b_px, h.b_py
        )));
    } else {
        g.set_handles_visible(false);
        g.set_handle_stem_a(SharedString::default());
        g.set_handle_stem_b(SharedString::default());
    }
    g.set_preset_available(geo.preset_available);
}

fn commit_graph_edit(
    handle: &crate::preview_worker::WorkerHandle,
    clip_id: &str,
    commit: crate::graph_editor::GraphCommit,
) {
    // Value on the commit is authoritative; this lookup only resolves ClipParam.
    let Some((param, _)) = clip_param_value(&commit.param_key, 0.0, 0.0) else {
        tracing::error!(key = %commit.param_key, "graph commit: unknown param");
        return;
    };
    if commit.tick_moved {
        handle.move_param_keyframe(crate::preview_worker::MoveParamKeyframeRequest {
            clip: clip_id.to_string(),
            param,
            from_tick: commit.from_tick,
            to_tick: commit.to_tick,
            value: commit.value,
            easing: commit.easing,
            tangents: commit.tangents,
        });
    } else {
        handle.set_param_keyframe(
            clip_id.to_string(),
            param,
            commit.to_tick,
            commit.value,
            commit.easing,
            commit.tangents,
        );
    }
}

fn wire_graph_editor(app: &AppWindow, preview_worker: &crate::preview_worker::PreviewWorker) {
    let graph = app.global::<GraphBackend>();
    let mapping = Rc::new(RefCell::new(None::<crate::graph_editor::PlotMapping>));
    let drag = Rc::new(RefCell::new(None::<GraphDragSession>));
    let handle_drag = Rc::new(RefCell::new(None::<GraphHandleSession>));

    let toggle_app = app.as_weak();
    graph.on_toggle_visible(move || {
        if let Some(app) = toggle_app.upgrade() {
            let g = app.global::<GraphBackend>();
            g.set_visible(!g.get_visible());
        }
    });

    let set_ch_app = app.as_weak();
    graph.on_set_channel(move |key, channel| {
        if let Some(app) = set_ch_app.upgrade() {
            let g = app.global::<GraphBackend>();
            g.set_selected_key(key);
            g.set_selected_channel(channel);
        }
    });

    let refresh_app = app.as_weak();
    let refresh_mapping = mapping.clone();
    let refresh_drag = drag.clone();
    let refresh_handle = handle_drag.clone();
    graph.on_refresh(move |sequence, clip_id, playhead, width, height| {
        let Some(app) = refresh_app.upgrade() else {
            return;
        };
        if refresh_drag.borrow().is_some() || refresh_handle.borrow().is_some() {
            return;
        }
        let g = app.global::<GraphBackend>();
        let selected_key = g.get_selected_key();
        let selected_channel = g.get_selected_channel();
        let selected_tick = g.get_selected_tick();
        let result = crate::graph_editor::refresh_graph(crate::graph_editor::GraphRefreshInput {
            sequence: &sequence,
            clip_id: clip_id.as_str(),
            playhead,
            width,
            height,
            selected_key: selected_key.as_str(),
            selected_channel,
            selected_tick,
        });
        g.set_channels(result.channels);
        g.set_channel_labels(result.channel_labels);
        g.set_channel_index(result.channel_index);
        g.set_selected_key(result.selected_key);
        g.set_selected_channel(result.selected_channel);
        *refresh_mapping.borrow_mut() = result.geometry.mapping;
        apply_graph_geometry(&g, result.geometry);
    });

    let select_app = app.as_weak();
    let select_mapping = mapping.clone();
    graph.on_select_dot(move |tick| {
        let Some(app) = select_app.upgrade() else {
            return;
        };
        let g = app.global::<GraphBackend>();
        g.set_selected_tick(tick);
        // Rebuild so outgoing-segment handles appear for the new selection.
        let Some(map) = *select_mapping.borrow() else {
            return;
        };
        let clip_id = app.global::<TimelineStore>().get_selected_clip_id();
        let seq = app.global::<EditorStore>().get_project().sequence;
        let playhead = app.global::<TimelineStore>().get_playhead_tick();
        let result = crate::graph_editor::refresh_graph(crate::graph_editor::GraphRefreshInput {
            sequence: &seq,
            clip_id: clip_id.as_str(),
            playhead,
            width: map.width,
            height: map.height,
            selected_key: g.get_selected_key().as_str(),
            selected_channel: g.get_selected_channel(),
            selected_tick: tick,
        });
        *select_mapping.borrow_mut() = result.geometry.mapping;
        apply_graph_geometry(&g, result.geometry);
    });

    let begin_app = app.as_weak();
    let begin_drag = drag.clone();
    let begin_mapping = mapping.clone();
    graph.on_drag_begin(move |tick| {
        let Some(app) = begin_app.upgrade() else {
            return;
        };
        let g = app.global::<GraphBackend>();
        let Some(map) = *begin_mapping.borrow() else {
            return;
        };
        let clip_id = app.global::<TimelineStore>().get_selected_clip_id();
        let key = g.get_selected_key();
        let channel = g.get_selected_channel();
        let playhead = app.global::<TimelineStore>().get_playhead_tick();
        let seq = app.global::<EditorStore>().get_project().sequence;
        let Some(clip) = crate::preview_motion_path::find_projected_clip(&seq, clip_id.as_str())
        else {
            return;
        };
        let Some(param) = crate::graph_editor::channel_param(&clip, key.as_str(), channel) else {
            return;
        };
        let value = param
            .keyframes()
            .iter()
            .find(|kf| kf.tick == i64::from(tick))
            .map(|kf| kf.value)
            .unwrap_or(0.0);
        g.set_selected_tick(tick);
        *begin_drag.borrow_mut() = Some(GraphDragSession {
            clip_id: clip_id.to_string(),
            key: key.to_string(),
            channel,
            from_tick: tick,
            tick,
            value,
            moved: false,
            mapping: map,
            playhead,
        });
    });

    let move_app = app.as_weak();
    let move_drag = drag.clone();
    graph.on_drag_move(move |x, y| {
        let Some(app) = move_app.upgrade() else {
            return;
        };
        let mut slot = move_drag.borrow_mut();
        let Some(session) = slot.as_mut() else {
            return;
        };
        let seq = app.global::<EditorStore>().get_project().sequence;
        let Some(clip) =
            crate::preview_motion_path::find_projected_clip(&seq, session.clip_id.as_str())
        else {
            return;
        };
        let Some((tick, value)) = crate::graph_editor::resolve_drag(
            &clip,
            &session.key,
            session.channel,
            session.from_tick,
            x,
            y,
            session.mapping,
        ) else {
            return;
        };
        if tick != session.tick || (value - session.value).abs() > 1e-6 {
            session.moved = true;
        }
        session.tick = tick;
        session.value = value;
        let Some(param) = crate::graph_editor::live_param(
            &clip,
            &session.key,
            session.channel,
            session.from_tick,
            tick,
            value,
        ) else {
            return;
        };
        let geo = crate::graph_editor::build_geometry(
            &param,
            session.playhead,
            session.mapping.width,
            session.mapping.height,
            tick,
        );
        apply_graph_geometry(&app.global::<GraphBackend>(), geo);
    });

    let end_handle = preview_worker.handle();
    let end_app = app.as_weak();
    let end_drag = drag.clone();
    graph.on_drag_end(move |commit| {
        let Some(app) = end_app.upgrade() else {
            return;
        };
        let Some(session) = end_drag.borrow_mut().take() else {
            return;
        };
        if !commit || !session.moved {
            // Restore committed geometry.
            let g = app.global::<GraphBackend>();
            let seq = app.global::<EditorStore>().get_project().sequence;
            let result =
                crate::graph_editor::refresh_graph(crate::graph_editor::GraphRefreshInput {
                    sequence: &seq,
                    clip_id: session.clip_id.as_str(),
                    playhead: session.playhead,
                    width: session.mapping.width,
                    height: session.mapping.height,
                    selected_key: session.key.as_str(),
                    selected_channel: session.channel,
                    selected_tick: g.get_selected_tick(),
                });
            apply_graph_geometry(&g, result.geometry);
            return;
        }
        let seq = app.global::<EditorStore>().get_project().sequence;
        let Some(clip) =
            crate::preview_motion_path::find_projected_clip(&seq, session.clip_id.as_str())
        else {
            return;
        };
        let Some(plan) = crate::graph_editor::plan_drag_commit(
            &clip,
            &session.key,
            session.channel,
            session.from_tick,
            session.tick,
            session.value,
        ) else {
            return;
        };
        commit_graph_edit(&end_handle, &session.clip_id, plan);
        app.global::<GraphBackend>().set_selected_tick(session.tick);
    });

    let insert_handle = preview_worker.handle();
    let insert_app = app.as_weak();
    let insert_mapping = mapping.clone();
    graph.on_insert_at(move |x, _y| {
        let Some(app) = insert_app.upgrade() else {
            return;
        };
        let Some(map) = *insert_mapping.borrow() else {
            return;
        };
        let g = app.global::<GraphBackend>();
        let clip_id = app.global::<TimelineStore>().get_selected_clip_id();
        let key = g.get_selected_key();
        let channel = g.get_selected_channel();
        let seq = app.global::<EditorStore>().get_project().sequence;
        let Some(clip) = crate::preview_motion_path::find_projected_clip(&seq, clip_id.as_str())
        else {
            return;
        };
        let Some((tick, value)) =
            crate::graph_editor::plan_insert(&clip, key.as_str(), channel, x, map)
        else {
            return;
        };
        let Some(plan) =
            crate::graph_editor::plan_insert_commit(&clip, key.as_str(), channel, tick, value)
        else {
            return;
        };
        commit_graph_edit(&insert_handle, clip_id.as_str(), plan);
        g.set_selected_tick(tick as i32);
    });

    let delete_handle = preview_worker.handle();
    let delete_app = app.as_weak();
    let delete_drag = drag.clone();
    let delete_handle_drag = handle_drag.clone();
    graph.on_delete_dot(move |tick| {
        let Some(app) = delete_app.upgrade() else {
            return;
        };
        *delete_drag.borrow_mut() = None;
        *delete_handle_drag.borrow_mut() = None;
        let g = app.global::<GraphBackend>();
        let clip_id = app.global::<TimelineStore>().get_selected_clip_id();
        let key = g.get_selected_key();
        let Some((param, _)) = clip_param_value(key.as_str(), 0.0, 0.0) else {
            return;
        };
        delete_handle.remove_param_keyframe(clip_id.to_string(), param, i64::from(tick));
        if g.get_selected_tick() == tick {
            g.set_selected_tick(-1);
        }
    });

    // --- bezier easing handle drag ---------------------------------------
    let h_begin_app = app.as_weak();
    let h_begin_drag = handle_drag.clone();
    let h_begin_mapping = mapping.clone();
    graph.on_handle_drag_begin(move |which| {
        let Some(app) = h_begin_app.upgrade() else {
            return;
        };
        let g = app.global::<GraphBackend>();
        let Some(map) = *h_begin_mapping.borrow() else {
            return;
        };
        let selected_tick = g.get_selected_tick();
        if selected_tick < 0 {
            return;
        }
        let clip_id = app.global::<TimelineStore>().get_selected_clip_id();
        let key = g.get_selected_key();
        let channel = g.get_selected_channel();
        let playhead = app.global::<TimelineStore>().get_playhead_tick();
        let seq = app.global::<EditorStore>().get_project().sequence;
        let Some(clip) = crate::preview_motion_path::find_projected_clip(&seq, clip_id.as_str())
        else {
            return;
        };
        let Some(param) = crate::graph_editor::channel_param(&clip, key.as_str(), channel) else {
            return;
        };
        let Some(handles) =
            crate::graph_editor::segment_handles(&param, i64::from(selected_tick), map)
        else {
            return;
        };
        let which = if which <= 0 {
            crate::graph_editor::HandleId::A
        } else {
            crate::graph_editor::HandleId::B
        };
        *h_begin_drag.borrow_mut() = Some(GraphHandleSession {
            clip_id: clip_id.to_string(),
            key: key.to_string(),
            channel,
            from_tick: selected_tick,
            which,
            points: handles.points,
            moved: false,
            handles,
            mapping: map,
            playhead,
        });
    });

    let h_move_app = app.as_weak();
    let h_move_drag = handle_drag.clone();
    graph.on_handle_drag_move(move |x, y| {
        let Some(app) = h_move_app.upgrade() else {
            return;
        };
        let mut slot = h_move_drag.borrow_mut();
        let Some(session) = slot.as_mut() else {
            return;
        };
        let points = crate::graph_editor::resolve_handle_drag(
            &session.handles,
            session.which,
            x,
            y,
            session.mapping,
        );
        if points != session.points {
            session.moved = true;
        }
        session.points = points;
        let seq = app.global::<EditorStore>().get_project().sequence;
        let Some(clip) =
            crate::preview_motion_path::find_projected_clip(&seq, session.clip_id.as_str())
        else {
            return;
        };
        let Some(base) =
            crate::graph_editor::channel_param(&clip, session.key.as_str(), session.channel)
        else {
            return;
        };
        let Some(param) =
            crate::graph_editor::live_handle_param(&base, i64::from(session.from_tick), points)
        else {
            return;
        };
        let geo = crate::graph_editor::build_geometry(
            &param,
            session.playhead,
            session.mapping.width,
            session.mapping.height,
            session.from_tick,
        );
        apply_graph_geometry(&app.global::<GraphBackend>(), geo);
    });

    let preset_handle = preview_worker.handle();
    let preset_app = app.as_weak();
    graph.on_apply_preset(move |preset| {
        let Some(app) = preset_app.upgrade() else {
            return;
        };
        let g = app.global::<GraphBackend>();
        let selected = g.get_selected_tick();
        if selected < 0 {
            return;
        }
        let clip_id = app.global::<TimelineStore>().get_selected_clip_id();
        let key = g.get_selected_key();
        let Some((param, _)) = clip_param_value(key.as_str(), 0.0, 0.0) else {
            return;
        };
        let preset = match preset {
            0 => cutlass_models::PiecewiseEasingPreset::BounceOut,
            1 => cutlass_models::PiecewiseEasingPreset::ElasticOut,
            _ => cutlass_models::PiecewiseEasingPreset::BackOut,
        };
        preset_handle.apply_easing_preset(clip_id.to_string(), param, i64::from(selected), preset);
    });

    let h_end_handle = preview_worker.handle();
    let h_end_app = app.as_weak();
    let h_end_drag = handle_drag.clone();
    graph.on_handle_drag_end(move |commit| {
        let Some(app) = h_end_app.upgrade() else {
            return;
        };
        let Some(session) = h_end_drag.borrow_mut().take() else {
            return;
        };
        if !commit || !session.moved {
            let g = app.global::<GraphBackend>();
            let seq = app.global::<EditorStore>().get_project().sequence;
            let result =
                crate::graph_editor::refresh_graph(crate::graph_editor::GraphRefreshInput {
                    sequence: &seq,
                    clip_id: session.clip_id.as_str(),
                    playhead: session.playhead,
                    width: session.mapping.width,
                    height: session.mapping.height,
                    selected_key: session.key.as_str(),
                    selected_channel: session.channel,
                    selected_tick: session.from_tick,
                });
            apply_graph_geometry(&g, result.geometry);
            return;
        }
        let seq = app.global::<EditorStore>().get_project().sequence;
        let Some(clip) =
            crate::preview_motion_path::find_projected_clip(&seq, session.clip_id.as_str())
        else {
            return;
        };
        let Some(plan) = crate::graph_editor::plan_handle_commit(
            &clip,
            &session.key,
            session.channel,
            session.from_tick,
            session.points,
        ) else {
            return;
        };
        commit_graph_edit(&h_end_handle, &session.clip_id, plan);
    });
}
