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

    app.global::<PreviewBackend>().on_motion_path(
        |sequence,
         clip_id,
         view_w,
         view_h,
         zoom,
         pan_x,
         pan_y,
         selected_tick,
         edit_mode,
         edit_tick,
         edit_x,
         edit_y,
         mirror| {
            preview_motion_path::motion_path_in_viewport(
                &sequence,
                clip_id.as_str(),
                view_w,
                view_h,
                zoom,
                pan_x,
                pan_y,
                selected_tick,
                edit_mode,
                edit_tick,
                edit_x,
                edit_y,
                mirror,
            )
        },
    );

    app.global::<PreviewBackend>().on_resolve_motion_path_drag(
        |sequence,
         clip_id,
         playhead,
         cursor_x,
         cursor_y,
         view_w,
         view_h,
         zoom,
         pan_x,
         pan_y,
         edit_mode,
         edit_tick| {
            preview_motion_path::resolve_motion_path_drag(
                &sequence,
                clip_id.as_str(),
                playhead,
                cursor_x,
                cursor_y,
                view_w,
                view_h,
                zoom,
                pan_x,
                pan_y,
                edit_mode,
                edit_tick,
            )
        },
    );

    {
        let preview = app.as_weak();
        app.global::<PreviewBackend>()
            .on_motion_path_select_keyframe(move |tick| {
                if let Some(app) = preview.upgrade() {
                    app.global::<PreviewStore>()
                        .set_motion_path_selected_tick(tick);
                }
            });
    }

    let kf_commit_handle = preview_worker.handle();
    let kf_commit_app = app.as_weak();
    app.global::<PreviewBackend>()
        .on_motion_path_commit_keyframe(move |clip_id, tick, x, y| {
            let Some(app) = kf_commit_app.upgrade() else {
                return;
            };
            let seq = app.global::<EditorStore>().get_project().sequence;
            let Some(clip) = preview_motion_path::find_projected_clip(&seq, clip_id.as_str())
            else {
                return;
            };
            let Some(kf) = preview_motion_path::position_keyframe_at(&clip, tick) else {
                return;
            };
            let (easing, tangents) = preview_motion_path::position_keyframe_commit_bits(&kf);
            kf_commit_handle.set_param_keyframe(
                clip_id.to_string(),
                cutlass_models::ClipParam::Position,
                i64::from(tick),
                cutlass_models::ParamValue::Vec2([x, y]),
                easing,
                tangents,
            );
        });

    let tan_commit_handle = preview_worker.handle();
    app.global::<PreviewBackend>()
        .on_motion_path_commit_tangents(move |clip_id, tick, out_x, out_y, in_x, in_y| {
            tan_commit_handle.set_param_keyframe_tangents(
                clip_id.to_string(),
                i64::from(tick),
                Some(cutlass_models::SpatialTangents {
                    out_t: [out_x, out_y],
                    in_t: [in_x, in_y],
                }),
            );
        });

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

    // Shared across begin / override / commit / clear / abandon so
    // override-first callers (inspector sliders) still get a paired
    // BeginTransformGesture, and sequential drags re-begin after end.
    let gesture_session = Rc::new(RefCell::new(
        crate::transform_gesture_session::TransformGestureSession::new(),
    ));

    let override_handle = preview_worker.handle();
    let override_session = gesture_session.clone();
    editor.on_on_preview_transform_overridden(
        move |clip_id,
              pos_x,
              pos_y,
              anchor_x,
              anchor_y,
              scale_x,
              scale_y,
              rotation,
              opacity,
              tick| {
            crate::transform_gesture_session::preview_transform(
                &mut override_session.borrow_mut(),
                &override_handle,
                clip_id.to_string(),
                cutlass_models::ClipTransform {
                    position: [pos_x, pos_y],
                    anchor_point: [anchor_x, anchor_y],
                    scale: cutlass_models::Scale2 {
                        x: scale_x,
                        y: scale_y,
                    },
                    rotation,
                    opacity,
                },
                i64::from(tick),
            );
        },
    );

    let gesture_start_handle = preview_worker.handle();
    let begin_session = gesture_session.clone();
    editor.on_on_preview_gesture_started(move |clip_id, tick| {
        crate::transform_gesture_session::begin_transform_gesture(
            &mut begin_session.borrow_mut(),
            &gesture_start_handle,
            clip_id.to_string(),
            i64::from(tick),
        );
    });

    let gesture_abandon_handle = preview_worker.handle();
    let abandon_session = gesture_session.clone();
    editor.on_on_preview_gesture_abandoned(move || {
        crate::transform_gesture_session::abandon_transform_gesture(
            &mut abandon_session.borrow_mut(),
            &gesture_abandon_handle,
        );
    });

    let override_clear_handle = preview_worker.handle();
    let clear_session = gesture_session.clone();
    editor.on_on_preview_override_cleared(move |tick| {
        crate::transform_gesture_session::clear_transform_override(
            &mut clear_session.borrow_mut(),
            &override_clear_handle,
            i64::from(tick),
        );
    });

    let transform_commit_handle = preview_worker.handle();
    let commit_session = gesture_session;
    editor.on_on_clip_transform_committed(
        move |clip_id,
              pos_x,
              pos_y,
              anchor_x,
              anchor_y,
              scale_x,
              scale_y,
              rotation,
              opacity,
              tick| {
            crate::transform_gesture_session::commit_transform(
                &mut commit_session.borrow_mut(),
                &transform_commit_handle,
                clip_id.to_string(),
                cutlass_models::ClipTransform {
                    position: [pos_x, pos_y],
                    anchor_point: [anchor_x, anchor_y],
                    scale: cutlass_models::Scale2 {
                        x: scale_x,
                        y: scale_y,
                    },
                    rotation,
                    opacity,
                },
                i64::from(tick),
            );
        },
    );
}
