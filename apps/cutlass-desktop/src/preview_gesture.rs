//! Preview transform gestures (preview roadmap Phases 3–4): cursor motion →
//! new clip position ([`resolve_drag`], with CapCut's canvas-center guides),
//! uniform scale about the center ([`resolve_scale`], corner handles), and
//! rotation ([`resolve_rotate`], the affordance below the box, magneting at
//! the cardinal angles).
//!
//! Pure resolution, the `snap.rs` / `selection.rs` pattern: the panel calls
//! a resolver on every pointer move and feeds the result to both the live
//! preview (worker transform override) and the release commit — a gesture
//! can never land somewhere other than where it previewed. All cursor math
//! runs through the same letterbox mapping hit-testing uses
//! (`preview_select::contain_mapping`), so motion tracks the cursor exactly
//! at any panel size.

use slint::Model;

use crate::placement::{anchor_canvas_position, reposition_anchor};
use crate::preview_select::{
    canvas_config, clip_placement, clip_transform, is_composited, text_alignment_offset,
    viewport_mapping,
};
use crate::{Clip, PreviewDragResolution, Sequence, TrackKind};

/// Find a draggable clip by id: on a visual, enabled, unlocked lane, and
/// actually composited. Mirrors the hit-test gates — anything pickable is
/// draggable, nothing else.
///
/// The returned clip's `transform-*` fields hold the playhead sample (`tick`),
/// not the clip-start values: a gesture on an animated clip starts from what
/// the user sees, and its commit keyframes at the playhead (M2 compose
/// semantics), so the sampled value is also the right "unchanged" baseline.
fn draggable_clip(sequence: &Sequence, clip_id: &str, tick: i32) -> Option<Clip> {
    if clip_id.is_empty() {
        return None;
    }
    for row in 0..sequence.tracks.row_count() {
        let Some(track) = sequence.tracks.row_data(row) else {
            continue;
        };
        if track.kind == TrackKind::Audio || !track.enabled || track.locked {
            continue;
        }
        for idx in 0..track.clips.row_count() {
            let Some(mut clip) = track.clips.row_data(idx) else {
                continue;
            };
            if clip.id == clip_id {
                if !is_composited(&clip) {
                    return None;
                }
                crate::params::apply_sampled_transform(&mut clip, tick);
                return Some(clip);
            }
        }
    }
    None
}

fn invalid() -> PreviewDragResolution {
    PreviewDragResolution::default()
}

/// Resolve a move gesture: the cursor's viewport-px displacement since the
/// press, applied to the clip's press-time position (the projection is
/// frozen during the drag — live motion is a worker-side override), with
/// the content center magneting onto the canvas center lines when within
/// `snap_tolerance_px` (viewport px, CapCut-style).
///
/// `moved` is false when the resolved position equals the committed one —
/// a click without displacement (or a drag that snapped back home) commits
/// nothing on release.
#[allow(dead_code, clippy::too_many_arguments)]
pub fn resolve_drag(
    sequence: &Sequence,
    clip_id: &str,
    tick: i32,
    press_x: f32,
    press_y: f32,
    cursor_x: f32,
    cursor_y: f32,
    view_w: f32,
    view_h: f32,
    snap_tolerance_px: f32,
) -> PreviewDragResolution {
    resolve_drag_in_viewport(
        sequence,
        clip_id,
        tick,
        press_x,
        press_y,
        cursor_x,
        cursor_y,
        view_w,
        view_h,
        1.0,
        0.0,
        0.0,
        snap_tolerance_px,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn resolve_drag_in_viewport(
    sequence: &Sequence,
    clip_id: &str,
    tick: i32,
    press_x: f32,
    press_y: f32,
    cursor_x: f32,
    cursor_y: f32,
    view_w: f32,
    view_h: f32,
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
    snap_tolerance_px: f32,
) -> PreviewDragResolution {
    let Some(clip) = draggable_clip(sequence, clip_id, tick) else {
        return invalid();
    };
    let canvas = canvas_config(sequence);
    let (cw, ch) = (canvas.width as f32, canvas.height as f32);
    let (scale, ox, oy) = viewport_mapping(cw, ch, view_w, view_h, zoom, pan_x, pan_y);
    if scale <= 0.0 {
        return invalid();
    }

    // Viewport displacement → canvas px → the anchor's new canvas position.
    let placement = clip_placement(&clip, &canvas);
    let layout_offset = text_alignment_offset(&clip, &placement, &canvas);
    let start = anchor_canvas_position(&clip_transform(&clip), &placement);
    let mut anchor_x = start[0] + (cursor_x - press_x) / scale;
    let mut anchor_y = start[1] + (cursor_y - press_y) / scale;

    // Center guides: magnet the anchor onto the canvas center lines.
    let tolerance = snap_tolerance_px / scale;
    let snap_v = (anchor_x - cw / 2.0).abs() <= tolerance;
    let snap_h = (anchor_y - ch / 2.0).abs() <= tolerance;
    if snap_v {
        anchor_x = cw / 2.0;
    }
    if snap_h {
        anchor_y = ch / 2.0;
    }

    // Text alignment is a canvas-layout offset, not part of the persisted
    // transform position. Remove it when mapping the dragged visible anchor
    // back into ClipTransform; otherwise the first pointer move applies the
    // alignment twice and can throw the text completely out of the preview.
    let position_x = (anchor_x - layout_offset[0] - cw / 2.0) / cw;
    let position_y = (anchor_y - layout_offset[1] - ch / 2.0) / ch;

    PreviewDragResolution {
        valid: true,
        moved: position_x != clip.transform_position_x || position_y != clip.transform_position_y,
        position_x,
        position_y,
        anchor_x: clip.transform_anchor_x,
        anchor_y: clip.transform_anchor_y,
        scale: clip.transform_scale,
        rotation: clip.transform_rotation,
        opacity: clip.transform_opacity,
        snap_h,
        snap_v,
        guide_x: ox + (cw / 2.0) * scale,
        guide_y: oy + (ch / 2.0) * scale,
    }
}

/// Smallest committable uniform scale. Keeps the content (and its corner
/// handles) grabbable — CapCut clamps similarly rather than letting a scale
/// gesture collapse the box to nothing.
const MIN_SCALE: f32 = 0.05;

/// The clip's anchor pivot in viewport-element coordinates — the fixed point
/// scale and rotate gestures pivot about. `None` when the mapping is degenerate.
fn pivot_in_viewport(
    sequence: &Sequence,
    clip: &Clip,
    view_w: f32,
    view_h: f32,
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
) -> Option<(f32, f32)> {
    let canvas = canvas_config(sequence);
    let (cw, ch) = (canvas.width as f32, canvas.height as f32);
    let (scale, ox, oy) = viewport_mapping(cw, ch, view_w, view_h, zoom, pan_x, pan_y);
    if scale <= 0.0 {
        return None;
    }
    let placement = clip_placement(clip, &canvas);
    let anchor = anchor_canvas_position(&clip_transform(clip), &placement);
    Some((ox + anchor[0] * scale, oy + anchor[1] * scale))
}

/// Resolve a corner-handle scale gesture: uniform about the anchor, factor =
/// cursor distance from anchor ÷ press distance from anchor (the grabbed
/// corner stays under the cursor along its anchor ray), clamped at
/// [`MIN_SCALE`]. Position, anchor, rotation, and opacity pass through.
#[allow(dead_code, clippy::too_many_arguments)]
pub fn resolve_scale(
    sequence: &Sequence,
    clip_id: &str,
    tick: i32,
    press_x: f32,
    press_y: f32,
    cursor_x: f32,
    cursor_y: f32,
    view_w: f32,
    view_h: f32,
) -> PreviewDragResolution {
    resolve_scale_in_viewport(
        sequence, clip_id, tick, press_x, press_y, cursor_x, cursor_y, view_w, view_h, 1.0, 0.0,
        0.0,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn resolve_scale_in_viewport(
    sequence: &Sequence,
    clip_id: &str,
    tick: i32,
    press_x: f32,
    press_y: f32,
    cursor_x: f32,
    cursor_y: f32,
    view_w: f32,
    view_h: f32,
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
) -> PreviewDragResolution {
    let Some(clip) = draggable_clip(sequence, clip_id, tick) else {
        return invalid();
    };
    let Some((center_x, center_y)) =
        pivot_in_viewport(sequence, &clip, view_w, view_h, zoom, pan_x, pan_y)
    else {
        return invalid();
    };

    let press_dist = (press_x - center_x).hypot(press_y - center_y);
    let cursor_dist = (cursor_x - center_x).hypot(cursor_y - center_y);
    if press_dist <= f32::EPSILON {
        // Press at the pivot: the ratio is undefined. Can't happen from a
        // corner handle; reject rather than divide by zero.
        return invalid();
    }
    let scale = (clip.transform_scale * cursor_dist / press_dist).max(MIN_SCALE);

    PreviewDragResolution {
        valid: true,
        moved: scale != clip.transform_scale,
        position_x: clip.transform_position_x,
        position_y: clip.transform_position_y,
        anchor_x: clip.transform_anchor_x,
        anchor_y: clip.transform_anchor_y,
        scale,
        rotation: clip.transform_rotation,
        opacity: clip.transform_opacity,
        snap_h: false,
        snap_v: false,
        guide_x: 0.0,
        guide_y: 0.0,
    }
}

/// Normalize an angle in degrees to (-180, 180] — same visual rotation,
/// tidy committed values.
fn normalize_degrees(angle: f32) -> f32 {
    let wrapped = angle - 360.0 * (angle / 360.0).round();
    if wrapped == -180.0 { 180.0 } else { wrapped }
}

/// Resolve a rotate-affordance gesture: the cursor's angle around the anchor,
/// relative to where the press grabbed the handle, added to the clip's
/// committed rotation — so the handle tracks the cursor. Magnets to the
/// nearest cardinal angle (0/90/180/270) within `snap_tolerance_deg`.
#[allow(dead_code, clippy::too_many_arguments)]
pub fn resolve_rotate(
    sequence: &Sequence,
    clip_id: &str,
    tick: i32,
    press_x: f32,
    press_y: f32,
    cursor_x: f32,
    cursor_y: f32,
    view_w: f32,
    view_h: f32,
    snap_tolerance_deg: f32,
) -> PreviewDragResolution {
    resolve_rotate_in_viewport(
        sequence,
        clip_id,
        tick,
        press_x,
        press_y,
        cursor_x,
        cursor_y,
        view_w,
        view_h,
        1.0,
        0.0,
        0.0,
        snap_tolerance_deg,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn resolve_rotate_in_viewport(
    sequence: &Sequence,
    clip_id: &str,
    tick: i32,
    press_x: f32,
    press_y: f32,
    cursor_x: f32,
    cursor_y: f32,
    view_w: f32,
    view_h: f32,
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
    snap_tolerance_deg: f32,
) -> PreviewDragResolution {
    let Some(clip) = draggable_clip(sequence, clip_id, tick) else {
        return invalid();
    };
    let Some((center_x, center_y)) =
        pivot_in_viewport(sequence, &clip, view_w, view_h, zoom, pan_x, pan_y)
    else {
        return invalid();
    };

    // atan2 in y-down screen coords increases clockwise — the same sense as
    // `ClipTransform::rotation`, so the delta adds directly.
    let press_angle = (press_y - center_y).atan2(press_x - center_x).to_degrees();
    let cursor_angle = (cursor_y - center_y)
        .atan2(cursor_x - center_x)
        .to_degrees();
    let mut rotation = normalize_degrees(clip.transform_rotation + (cursor_angle - press_angle));

    let cardinal = normalize_degrees((rotation / 90.0).round() * 90.0);
    if (normalize_degrees(rotation - cardinal)).abs() <= snap_tolerance_deg {
        rotation = cardinal;
    }

    PreviewDragResolution {
        valid: true,
        moved: rotation != clip.transform_rotation,
        position_x: clip.transform_position_x,
        position_y: clip.transform_position_y,
        anchor_x: clip.transform_anchor_x,
        anchor_y: clip.transform_anchor_y,
        scale: clip.transform_scale,
        rotation,
        opacity: clip.transform_opacity,
        snap_h: false,
        snap_v: false,
        guide_x: 0.0,
        guide_y: 0.0,
    }
}

/// Arrow-key nudge: displace the clip by whole canvas pixels (CapCut: 1 px,
/// Shift = 10) through the same commit path as a drag release. No guides —
/// deliberate keyboard motion shouldn't magnet away. Invalid when the clip
/// isn't draggable, so the caller can fall through to frame-stepping.
pub fn nudge(
    sequence: &Sequence,
    clip_id: &str,
    tick: i32,
    dx_canvas_px: f32,
    dy_canvas_px: f32,
) -> PreviewDragResolution {
    let Some(clip) = draggable_clip(sequence, clip_id, tick) else {
        return invalid();
    };
    let canvas = canvas_config(sequence);
    let (cw, ch) = (canvas.width as f32, canvas.height as f32);

    PreviewDragResolution {
        valid: true,
        moved: dx_canvas_px != 0.0 || dy_canvas_px != 0.0,
        position_x: clip.transform_position_x + dx_canvas_px / cw,
        position_y: clip.transform_position_y + dy_canvas_px / ch,
        anchor_x: clip.transform_anchor_x,
        anchor_y: clip.transform_anchor_y,
        scale: clip.transform_scale,
        rotation: clip.transform_rotation,
        opacity: clip.transform_opacity,
        snap_h: false,
        snap_v: false,
        guide_x: 0.0,
        guide_y: 0.0,
    }
}

/// Resolve an anchor-handle drag: the pivot follows the cursor while the
/// rendered frame stays fixed — both `anchor_point` and `position` update.
#[allow(dead_code, clippy::too_many_arguments)]
pub fn resolve_anchor(
    sequence: &Sequence,
    clip_id: &str,
    tick: i32,
    _press_x: f32,
    _press_y: f32,
    cursor_x: f32,
    cursor_y: f32,
    view_w: f32,
    view_h: f32,
) -> PreviewDragResolution {
    resolve_anchor_in_viewport(
        sequence, clip_id, tick, _press_x, _press_y, cursor_x, cursor_y, view_w, view_h, 1.0, 0.0,
        0.0,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn resolve_anchor_in_viewport(
    sequence: &Sequence,
    clip_id: &str,
    tick: i32,
    _press_x: f32,
    _press_y: f32,
    cursor_x: f32,
    cursor_y: f32,
    view_w: f32,
    view_h: f32,
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
) -> PreviewDragResolution {
    let Some(clip) = draggable_clip(sequence, clip_id, tick) else {
        return invalid();
    };
    let canvas = canvas_config(sequence);
    let (cw, ch) = (canvas.width as f32, canvas.height as f32);
    let (scale, ox, oy) = viewport_mapping(cw, ch, view_w, view_h, zoom, pan_x, pan_y);
    if scale <= 0.0 {
        return invalid();
    }

    let placement = clip_placement(&clip, &canvas);
    let anchor_canvas = [(cursor_x - ox) / scale, (cursor_y - oy) / scale];
    let (anchor_point, position) = reposition_anchor(
        anchor_canvas,
        placement.center,
        placement.size,
        clip.transform_rotation,
        &canvas,
    );

    PreviewDragResolution {
        valid: true,
        moved: anchor_point[0] != clip.transform_anchor_x
            || anchor_point[1] != clip.transform_anchor_y
            || position[0] != clip.transform_position_x
            || position[1] != clip.transform_position_y,
        position_x: position[0],
        position_y: position[1],
        anchor_x: anchor_point[0],
        anchor_y: anchor_point[1],
        scale: clip.transform_scale,
        rotation: clip.transform_rotation,
        opacity: clip.transform_opacity,
        snap_h: false,
        snap_v: false,
        guide_x: 0.0,
        guide_y: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preview_select::{canvas_config, clip_placement, clip_transform};
    use crate::{Rational, RationalTime, TimeRange, Track};
    use slint::{ModelRc, SharedString, VecModel};
    use std::rc::Rc;

    fn rt(value: i32) -> RationalTime {
        RationalTime {
            value,
            rate: Rational { num: 24, den: 1 },
        }
    }

    fn media_clip(id: &str, w: i32, h: i32) -> Clip {
        Clip {
            id: SharedString::from(id),
            name: SharedString::from(id),
            timeline_start: rt(0),
            source_range: TimeRange {
                start: rt(0),
                duration: rt(100),
            },
            media_id: SharedString::from("m1"),
            media_width: w,
            media_height: h,
            transform_scale: 1.0,
            transform_opacity: 1.0,
            transform_anchor_x: 0.5,
            transform_anchor_y: 0.5,
            ..Default::default()
        }
    }

    fn track(id: &str, clips: Vec<Clip>) -> Track {
        Track {
            id: SharedString::from(id),
            name: SharedString::from(id),
            kind: TrackKind::Video,
            color: slint::Color::from_rgb_u8(0x4A, 0x6F, 0xA5),
            clips: ModelRc::from(Rc::new(VecModel::from(clips))),
            enabled: true,
            muted: false,
            locked: false,
            duck_source: false,
            pinned: false,
            is_main: false,
            transitions: ModelRc::default(),
        }
    }

    /// 1920×1080 canvas, single video lane.
    fn sequence(tracks: Vec<Track>) -> Sequence {
        Sequence {
            id: SharedString::from("1"),
            name: SharedString::from("Sequence 1"),
            fps: Rational { num: 24, den: 1 },
            drop_frame: false,
            tracks: ModelRc::from(Rc::new(VecModel::from(tracks))),
            markers: Default::default(),
            width: 1920.0,
            height: 1080.0,
            aspect_index: 0,
            background: Default::default(),
        }
    }

    // Viewport at exactly half the canvas: scale 0.5, no letterbox.
    const VW: f32 = 960.0;
    const VH: f32 = 540.0;

    #[test]
    fn drag_maps_viewport_delta_to_normalized_position() {
        let seq = sequence(vec![track("1", vec![media_clip("A", 1920, 1080)])]);
        // 96 px right, 27 px down in the viewport = 192 / 54 canvas px =
        // +0.1 / +0.05 normalized.
        let r = resolve_drag(&seq, "A", 10, 100.0, 100.0, 196.0, 127.0, VW, VH, 0.0);
        assert!(r.valid && r.moved);
        assert!((r.position_x - 0.1).abs() < 1e-6);
        assert!((r.position_y - 0.05).abs() < 1e-6);
        assert!(!r.snap_h && !r.snap_v);
        // Untouched transform components pass through for the commit.
        assert_eq!((r.scale, r.rotation, r.opacity), (1.0, 0.0, 1.0));
    }

    #[test]
    fn drag_respects_preview_zoom() {
        let seq = sequence(vec![track("1", vec![media_clip("A", 1920, 1080)])]);
        // At 2× preview zoom the viewport scale is 1:1 canvas px, so the
        // same 96 px drag moves half as far in normalized canvas space.
        let r = resolve_drag_in_viewport(
            &seq, "A", 10, 100.0, 100.0, 196.0, 100.0, VW, VH, 2.0, 0.0, 0.0, 0.0,
        );
        assert!(r.valid && r.moved);
        assert!((r.position_x - 96.0 / 1920.0).abs() < 1e-6);
    }

    #[test]
    fn drag_without_displacement_is_not_moved() {
        let seq = sequence(vec![track("1", vec![media_clip("A", 1920, 1080)])]);
        let r = resolve_drag(&seq, "A", 10, 100.0, 100.0, 100.0, 100.0, VW, VH, 6.0);
        assert!(r.valid && !r.moved);
        assert_eq!((r.position_x, r.position_y), (0.0, 0.0));
    }

    #[test]
    fn drag_snaps_each_center_axis_independently() {
        let seq = sequence(vec![track("1", vec![media_clip("A", 1920, 1080)])]);
        // 4 viewport px right of dead center with 6 px tolerance: x snaps
        // back to centered (and reads as unmoved); y is 100 px off, free.
        let r = resolve_drag(&seq, "A", 10, 100.0, 100.0, 104.0, 200.0, VW, VH, 6.0);
        assert!(r.valid && r.moved);
        assert!(r.snap_v && !r.snap_h);
        assert_eq!(r.position_x, 0.0);
        assert!((r.position_y - 100.0 / 0.5 / 1080.0).abs() < 1e-6);
        // Guides sit at the canvas center in viewport coordinates.
        assert_eq!((r.guide_x, r.guide_y), (480.0, 270.0));
    }

    #[test]
    fn drag_snapped_back_home_commits_nothing() {
        // Centered clip wiggled 2 px: the magnet returns it to exactly its
        // committed position ⇒ not moved ⇒ release is a no-op.
        let seq = sequence(vec![track("1", vec![media_clip("A", 1920, 1080)])]);
        let r = resolve_drag(&seq, "A", 10, 100.0, 100.0, 102.0, 101.0, VW, VH, 6.0);
        assert!(r.valid && !r.moved);
        assert!(r.snap_v && r.snap_h);
    }

    #[test]
    fn drag_starts_from_the_clips_current_position() {
        let mut clip = media_clip("A", 1920, 1080);
        clip.transform_position_x = 0.25;
        let seq = sequence(vec![track("1", vec![clip])]);
        // 48 viewport px left = 96 canvas px = -0.05 normalized.
        let r = resolve_drag(&seq, "A", 10, 100.0, 100.0, 52.0, 100.0, VW, VH, 0.0);
        assert!(r.valid && r.moved);
        assert!((r.position_x - 0.2).abs() < 1e-6);
    }

    #[test]
    fn drag_does_not_fold_text_alignment_into_transform_position() {
        let mut clip = media_clip("T", 400, 200);
        clip.media_id = SharedString::default();
        clip.generator_kind = SharedString::from("text");
        clip.text_style.align_h = 0;
        clip.text_style.align_v = 2;
        let seq = sequence(vec![track("1", vec![clip])]);

        let still = resolve_drag(&seq, "T", 10, 50.0, 500.0, 50.0, 500.0, VW, VH, 0.0);
        assert!(still.valid && !still.moved);
        assert_eq!((still.position_x, still.position_y), (0.0, 0.0));

        // 48 viewport pixels are 96 canvas pixels at this fit. Alignment
        // stays a layout concern; the persisted transform moves by exactly
        // that cursor delta instead of receiving the left/bottom offsets.
        let moved = resolve_drag(&seq, "T", 10, 50.0, 500.0, 98.0, 500.0, VW, VH, 0.0);
        assert!(moved.valid && moved.moved);
        assert!((moved.position_x - 96.0 / 1920.0).abs() < 1e-6);
        assert_eq!(moved.position_y, 0.0);
    }

    #[test]
    fn drag_rejects_locked_hidden_and_unknown_clips() {
        let mut locked = track("2", vec![media_clip("L", 1920, 1080)]);
        locked.locked = true;
        let mut hidden = track("3", vec![media_clip("H", 1920, 1080)]);
        hidden.enabled = false;
        let seq = sequence(vec![
            locked,
            hidden,
            track("1", vec![media_clip("A", 1920, 1080)]),
        ]);
        assert!(!resolve_drag(&seq, "L", 10, 0.0, 0.0, 50.0, 0.0, VW, VH, 0.0).valid);
        assert!(!resolve_drag(&seq, "H", 10, 0.0, 0.0, 50.0, 0.0, VW, VH, 0.0).valid);
        assert!(!resolve_drag(&seq, "404", 10, 0.0, 0.0, 50.0, 0.0, VW, VH, 0.0).valid);
        assert!(resolve_drag(&seq, "A", 10, 0.0, 0.0, 50.0, 0.0, VW, VH, 0.0).valid);
    }

    // Scale/rotate fixtures: centered 1920×1080 clip in the 960×540
    // viewport ⇒ content center at view (480, 270).

    #[test]
    fn scale_follows_cursor_distance_ratio() {
        let seq = sequence(vec![track("1", vec![media_clip("A", 1920, 1080)])]);
        // Press 200 px from the center, drag to 100 px: scale halves.
        let r = resolve_scale(&seq, "A", 10, 680.0, 270.0, 580.0, 270.0, VW, VH);
        assert!(r.valid && r.moved);
        assert!((r.scale - 0.5).abs() < 1e-6);
        // Everything else passes through for the commit.
        assert_eq!((r.position_x, r.position_y), (0.0, 0.0));
        assert_eq!((r.rotation, r.opacity), (0.0, 1.0));
        assert!(!r.snap_h && !r.snap_v);
    }

    #[test]
    fn scale_compounds_the_committed_scale() {
        let mut clip = media_clip("A", 1920, 1080);
        clip.transform_scale = 0.5;
        let seq = sequence(vec![track("1", vec![clip])]);
        // 200 → 300 px from center: ×1.5 on top of the committed 0.5.
        let r = resolve_scale(&seq, "A", 10, 680.0, 270.0, 780.0, 270.0, VW, VH);
        assert!(r.valid && r.moved);
        assert!((r.scale - 0.75).abs() < 1e-6);
    }

    #[test]
    fn scale_clamps_at_the_minimum() {
        let seq = sequence(vec![track("1", vec![media_clip("A", 1920, 1080)])]);
        // Dragged almost onto the pivot: raw ratio would be 1/200.
        let r = resolve_scale(&seq, "A", 10, 680.0, 270.0, 481.0, 270.0, VW, VH);
        assert!(r.valid && r.moved);
        assert_eq!(r.scale, 0.05);
    }

    #[test]
    fn scale_back_at_the_press_point_is_not_moved() {
        let seq = sequence(vec![track("1", vec![media_clip("A", 1920, 1080)])]);
        let r = resolve_scale(&seq, "A", 10, 680.0, 270.0, 680.0, 270.0, VW, VH);
        assert!(r.valid && !r.moved);
        assert_eq!(r.scale, 1.0);
    }

    #[test]
    fn scale_rejects_press_at_the_pivot_and_unknown_clips() {
        let seq = sequence(vec![track("1", vec![media_clip("A", 1920, 1080)])]);
        assert!(!resolve_scale(&seq, "A", 10, 480.0, 270.0, 580.0, 270.0, VW, VH).valid);
        assert!(!resolve_scale(&seq, "404", 10, 680.0, 270.0, 580.0, 270.0, VW, VH).valid);
    }

    #[test]
    fn rotate_follows_the_cursor_angle_about_the_center() {
        let seq = sequence(vec![track("1", vec![media_clip("A", 1920, 1080)])]);
        // Press at angle 0°, drag to 45° (down-right in y-down coords):
        // far from any cardinal, no magnet.
        let r = resolve_rotate(&seq, "A", 10, 680.0, 270.0, 680.0, 470.0, VW, VH, 3.0);
        assert!(r.valid && r.moved);
        assert!((r.rotation - 45.0).abs() < 1e-3);
        // Position and scale pass through.
        assert_eq!((r.position_x, r.position_y, r.scale), (0.0, 0.0, 1.0));
    }

    #[test]
    fn rotate_magnets_to_cardinal_angles() {
        let seq = sequence(vec![track("1", vec![media_clip("A", 1920, 1080)])]);
        // ~90.6°: inside the 3° magnet ⇒ exactly 90.
        let r = resolve_rotate(&seq, "A", 10, 680.0, 270.0, 478.0, 470.0, VW, VH, 3.0);
        assert!(r.valid && r.moved);
        assert_eq!(r.rotation, 90.0);
    }

    #[test]
    fn rotate_starts_from_the_committed_rotation() {
        let mut clip = media_clip("A", 1920, 1080);
        clip.transform_rotation = 30.0;
        let seq = sequence(vec![track("1", vec![clip])]);
        // +45° of cursor travel on top of the committed 30°.
        let r = resolve_rotate(&seq, "A", 10, 680.0, 270.0, 680.0, 470.0, VW, VH, 3.0);
        assert!(r.valid && r.moved);
        assert!((r.rotation - 75.0).abs() < 1e-3);
    }

    #[test]
    fn rotate_normalizes_past_half_turn() {
        let mut clip = media_clip("A", 1920, 1080);
        clip.transform_rotation = 170.0;
        let seq = sequence(vec![track("1", vec![clip])]);
        // +90° lands at 260 ⇒ normalized to -100 (same visual rotation).
        let r = resolve_rotate(&seq, "A", 10, 680.0, 270.0, 480.0, 470.0, VW, VH, 3.0);
        assert!(r.valid && r.moved);
        assert!((r.rotation + 100.0).abs() < 1e-3);
    }

    #[test]
    fn rotate_back_at_the_press_point_is_not_moved() {
        let seq = sequence(vec![track("1", vec![media_clip("A", 1920, 1080)])]);
        let r = resolve_rotate(&seq, "A", 10, 680.0, 270.0, 680.0, 270.0, VW, VH, 3.0);
        assert!(r.valid && !r.moved);
        assert_eq!(r.rotation, 0.0);

        assert!(!resolve_rotate(&seq, "404", 10, 680.0, 270.0, 680.0, 470.0, VW, VH, 3.0).valid);
    }

    #[test]
    fn anchor_drag_preserves_composited_center() {
        let seq = sequence(vec![track("1", vec![media_clip("A", 1920, 1080)])]);
        let canvas = canvas_config(&seq);
        let clip = media_clip("A", 1920, 1080);
        let center_before = clip_placement(&clip, &canvas).center;

        let r = resolve_anchor(&seq, "A", 10, 480.0, 270.0, 580.0, 320.0, VW, VH);
        assert!(r.valid && r.moved);

        let mut moved = clip.clone();
        moved.transform_position_x = r.position_x;
        moved.transform_position_y = r.position_y;
        moved.transform_anchor_x = r.anchor_x;
        moved.transform_anchor_y = r.anchor_y;
        let center_after = clip_placement(&moved, &canvas).center;
        assert!((center_after[0] - center_before[0]).abs() < 1e-2);
        assert!((center_after[1] - center_before[1]).abs() < 1e-2);
    }

    #[test]
    fn scale_about_off_center_anchor_keeps_pivot_in_view() {
        let mut clip = media_clip("A", 1920, 1080);
        clip.transform_anchor_x = 0.25;
        clip.transform_anchor_y = 0.5;
        let seq = sequence(vec![track("1", vec![clip])]);
        let canvas = canvas_config(&seq);
        let base = seq.tracks.row_data(0).unwrap().clips.row_data(0).unwrap();
        let t = clip_transform(&base);
        let a0 = anchor_canvas_position(&t, &clip_placement(&base, &canvas));

        // Double distance from pivot ⇒ scale doubles.
        let r = resolve_scale(&seq, "A", 10, 680.0, 270.0, 880.0, 270.0, VW, VH);
        assert!(r.valid && r.moved);
        assert!((r.scale - 2.0).abs() < 1e-6);

        let mut scaled = base.clone();
        scaled.transform_scale = r.scale;
        let a1 =
            anchor_canvas_position(&clip_transform(&scaled), &clip_placement(&scaled, &canvas));
        assert!((a0[0] - a1[0]).abs() < 1e-2);
        assert!((a0[1] - a1[1]).abs() < 1e-2);
    }

    #[test]
    fn gestures_start_from_the_playhead_sample_on_animated_clips() {
        use crate::ParamKeyframe;
        let mut clip = media_clip("A", 1920, 1080);
        // Scale animates 1.0 → 2.0 over ticks 0..40: at tick 20 the frame
        // renders 1.5, and that's what a scale gesture must compound.
        clip.kf_scale = ModelRc::from(Rc::new(VecModel::from(vec![
            ParamKeyframe {
                tick: 0,
                value_x: 1.0,
                ..Default::default()
            },
            ParamKeyframe {
                tick: 40,
                value_x: 2.0,
                ..Default::default()
            },
        ])));
        let seq = sequence(vec![track("1", vec![clip])]);
        // Press 200 px from the center, drag to 100 px: halves the sample.
        let r = resolve_scale(&seq, "A", 20, 680.0, 270.0, 580.0, 270.0, VW, VH);
        assert!(r.valid && r.moved);
        assert!((r.scale - 0.75).abs() < 1e-6, "got {}", r.scale);
        // Same gesture at the first keyframe starts from 1.0.
        let r = resolve_scale(&seq, "A", 0, 680.0, 270.0, 580.0, 270.0, VW, VH);
        assert!((r.scale - 0.5).abs() < 1e-6, "got {}", r.scale);
    }

    #[test]
    fn nudge_moves_whole_canvas_pixels() {
        let seq = sequence(vec![track("1", vec![media_clip("A", 1920, 1080)])]);
        let r = nudge(&seq, "A", 10, 10.0, -1.0);
        assert!(r.valid && r.moved);
        assert!((r.position_x - 10.0 / 1920.0).abs() < 1e-7);
        assert!((r.position_y + 1.0 / 1080.0).abs() < 1e-7);
        assert!(!r.snap_h && !r.snap_v);

        let locked_miss = nudge(&seq, "404", 10, 1.0, 0.0);
        assert!(
            !locked_miss.valid,
            "unknown clip falls through to frame-step"
        );
    }
}
