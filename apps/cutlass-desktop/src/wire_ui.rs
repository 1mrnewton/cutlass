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

pub(crate) fn wire_ui(
    app: &AppWindow,
    audio_system: &crate::audio::AudioSystem,
    interaction_gate: &Arc<crate::interaction::InteractionGate>,
    strip_worker: &crate::strips::StripWorker,
) {
    // --- pure UI backends (no engine involved) ----------------------------

    crate::wire_color::wire_color(app);

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
}
