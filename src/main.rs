mod command;
mod models;
mod projector;
mod ruler;
mod snap;
mod state;
mod timecode;
mod timeline;

use std::cell::RefCell;
use std::rc::Rc;

use slint::SharedString;

use slint::BackendSelector;
use slint::wgpu_28::WGPUConfiguration;

use crate::command::Command;
use crate::state::Editor;

slint::include_modules!();

fn main() -> Result<(), slint::PlatformError> {
    BackendSelector::new()
        .require_wgpu_28(WGPUConfiguration::default())
        .select()?;

    let app = AppWindow::new()?;

    // The Editor is the only thing in the program allowed to mutate
    // the project. Slint never writes to `EditorStore.project`; UI
    // gestures dispatch commands through callbacks like `move-clip`
    // below, the Editor applies them to the domain model, and the
    // projector patches the matching row in the Slint projection.
    let editor = Rc::new(RefCell::new(Editor::new(models::sample_project())));

    // One-shot hand-off of the projection to Slint. After this, the
    // projector keeps the underlying `VecModel`s alive and writes
    // into them in place; we never set `project` again.
    app.global::<EditorStore>()
        .set_project(editor.borrow().slint_project().clone());

    {
        let editor = editor.clone();
        app.global::<EditorStore>().on_move_clip(
            move |source_track_id: SharedString,
                  clip_id: SharedString,
                  target_track_id: SharedString,
                  new_start_value: i32| {
                let cmd = Command::MoveClip {
                    source_track_id: source_track_id.into(),
                    clip_id: clip_id.into(),
                    target_track_id: target_track_id.into(),
                    new_start_value,
                };
                if let Err(err) = editor.borrow_mut().apply(&cmd) {
                    // The gesture layer can produce ids that no
                    // longer exist (e.g. the clip was deleted by
                    // another command mid-drag) — log and drop, never
                    // panic on UI input.
                    eprintln!("move-clip rejected: {err}");
                }
            },
        );
    }

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

    {
        // Snap-clip-start handler — re-invoked by Slint every drag
        // frame. Borrows the domain project read-only; the gesture
        // layer never calls this while a command is in-flight, but
        // a stray re-entry would surface as a `RefCell` panic, which
        // is loud enough to be noticed in debug.
        let editor = editor.clone();
        app.global::<DragBackend>().on_snap_clip_start(
            move |dragging_source_track_id: SharedString,
                  dragging_clip_id: SharedString,
                  cursor_start_value: i32,
                  clip_duration_ticks: i32,
                  snap_threshold_ticks: i32| {
                let editor = editor.borrow();
                let r = snap::compute_drag_snap(
                    editor.project(),
                    dragging_source_track_id.as_str(),
                    dragging_clip_id.as_str(),
                    cursor_start_value,
                    clip_duration_ticks,
                    snap_threshold_ticks,
                );
                SnapResult {
                    has_snap: r.has_snap,
                    snapped_start_value: r.snapped_start_value,
                    snap_line_tick: r.snap_line_tick,
                }
            },
        );
    }

    {
        // Resolve-target-lane handler — also a hot per-frame call
        // during drag. Same read-only borrow rules as snap above.
        let editor = editor.clone();
        app.global::<DragBackend>().on_resolve_target_lane(
            move |source_track_id: SharedString, lane_offset: i32| {
                let editor = editor.borrow();
                let r = snap::resolve_drag_target(
                    editor.project(),
                    source_track_id.as_str(),
                    lane_offset,
                );
                ResolvedTarget {
                    track_id: SharedString::from(r.track_id),
                    clamped_offset: r.clamped_offset,
                }
            },
        );
    }

    app.run()
}
