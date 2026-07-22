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

pub(crate) fn wire_preview(app: &AppWindow, preview_worker: &crate::preview_worker::PreviewWorker) {
    let editor = app.global::<EditorStore>();

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
                    scale: cutlass_models::Scale2::uniform(scale),
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
                    scale: cutlass_models::Scale2::uniform(scale),
                    rotation,
                    opacity,
                },
                i64::from(tick),
            );
        },
    );
}
