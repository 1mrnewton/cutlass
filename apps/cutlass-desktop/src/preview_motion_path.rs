//! Motion-path overlay geometry for the preview viewport.
//!
//! When the selected clip's position is keyframed, samples the projected
//! position [`Param`] across its keyframe range and maps canvas-fraction
//! positions into viewport px with the same letterbox/zoom/pan mapping the
//! selection box uses. Output is an SVG path-commands string (cheap to
//! diff) plus a model of keyframe dots (and tangent handles for the
//! selected keyframe).

use cutlass_models::{Easing, Keyframe, Param, SpatialTangents};
use slint::{Model, ModelRc, SharedString, VecModel};
use std::rc::Rc;

use crate::params::{easing_from_ui, position_param, sampled_transform};
use crate::preview_select::{canvas_config, viewport_mapping};
use crate::{
    Clip, MotionPath, MotionPathDragResolution, MotionPathKeyframe, ParamKeyframe, Sequence,
    TrackKind,
};

/// Dense samples per keyframe segment (before the global cap).
const POINTS_PER_SEGMENT: usize = 24;
/// Soft ceiling on the polyline point count.
const MAX_PATH_POINTS: usize = 480;
/// Default handle length (canvas fractions) when a keyframe has no tangents
/// yet — gives something to grab without committing until the user drags.
const DEFAULT_HANDLE: f32 = 0.08;

/// Live-edit mode passed from the panel: 0 = none, 1 = move keyframe,
/// 2 = drag out-tangent, 3 = drag in-tangent.
pub const EDIT_NONE: i32 = 0;
pub const EDIT_KEYFRAME: i32 = 1;
pub const EDIT_OUT: i32 = 2;
pub const EDIT_IN: i32 = 3;

/// Canvas-fraction position → viewport-element logical px.
fn position_to_viewport(
    pos: [f32; 2],
    canvas_w: f32,
    canvas_h: f32,
    scale: f32,
    ox: f32,
    oy: f32,
) -> [f32; 2] {
    let cx = canvas_w * 0.5 + pos[0] * canvas_w;
    let cy = canvas_h * 0.5 + pos[1] * canvas_h;
    [ox + cx * scale, oy + cy * scale]
}

/// Viewport-element px → canvas-fraction position.
fn viewport_to_position(
    x: f32,
    y: f32,
    canvas_w: f32,
    canvas_h: f32,
    scale: f32,
    ox: f32,
    oy: f32,
) -> [f32; 2] {
    let cx = (x - ox) / scale;
    let cy = (y - oy) / scale;
    [
        (cx - canvas_w * 0.5) / canvas_w,
        (cy - canvas_h * 0.5) / canvas_h,
    ]
}

/// Sample `param` across its keyframe tick range: `POINTS_PER_SEGMENT` per
/// segment, capped at ~[`MAX_PATH_POINTS`]. Empty when fewer than two
/// keyframes (nothing to stroke).
fn sample_path_points(param: &Param<[f32; 2]>) -> Vec<[f32; 2]> {
    let kfs = param.keyframes();
    if kfs.len() < 2 {
        return Vec::new();
    }
    let n_seg = kfs.len() - 1;
    let mut per_seg = POINTS_PER_SEGMENT;
    let uncapped = n_seg * per_seg + 1;
    if uncapped > MAX_PATH_POINTS {
        per_seg = ((MAX_PATH_POINTS - 1) / n_seg).max(2);
    }
    // Each segment contributes `per_seg` samples at i/per_seg for
    // i ∈ [0, per_seg) (includes the start, excludes the end). The next
    // segment's i=0 (or the final push) supplies the junction / terminus.
    let mut points = Vec::with_capacity(n_seg * per_seg + 1);
    for seg in 0..n_seg {
        let t0 = kfs[seg].tick as f64;
        let t1 = kfs[seg + 1].tick as f64;
        for i in 0..per_seg {
            let t = t0 + (t1 - t0) * (i as f64 / per_seg as f64);
            points.push(param.sample_at(t));
        }
    }
    points.push(param.sample(kfs[kfs.len() - 1].tick));
    points
}

fn svg_path_commands(points: &[[f32; 2]]) -> SharedString {
    if points.is_empty() {
        return SharedString::default();
    }
    let mut s = String::with_capacity(points.len() * 16);
    for (i, p) in points.iter().enumerate() {
        if i == 0 {
            s.push_str(&format!("M {:.2} {:.2}", p[0], p[1]));
        } else {
            s.push_str(&format!(" L {:.2} {:.2}", p[0], p[1]));
        }
    }
    SharedString::from(s)
}

/// Find a projected clip by id (visual lanes). Used by the overlay and by
/// commit wiring that needs the projected keyframe's easing/tangents.
pub fn find_projected_clip(sequence: &Sequence, clip_id: &str) -> Option<Clip> {
    if clip_id.is_empty() {
        return None;
    }
    for row in 0..sequence.tracks.row_count() {
        let Some(track) = sequence.tracks.row_data(row) else {
            continue;
        };
        if track.kind == TrackKind::Audio {
            continue;
        }
        for idx in 0..track.clips.row_count() {
            let Some(clip) = track.clips.row_data(idx) else {
                continue;
            };
            if clip.id == clip_id {
                return Some(clip);
            }
        }
    }
    None
}

fn find_clip(sequence: &Sequence, clip_id: &str) -> Option<Clip> {
    find_projected_clip(sequence, clip_id)
}

fn tangents_or_default(kf: &Keyframe<[f32; 2]>) -> SpatialTangents {
    kf.tangents.unwrap_or(SpatialTangents {
        out_t: [DEFAULT_HANDLE, 0.0],
        in_t: [-DEFAULT_HANDLE, 0.0],
    })
}

/// Apply a live edit to a cloned position param (path preview during drag).
fn apply_live_edit(
    param: &mut Param<[f32; 2]>,
    edit_mode: i32,
    edit_tick: i32,
    edit_x: f32,
    edit_y: f32,
    mirror: bool,
) {
    if edit_mode == EDIT_NONE || edit_tick < 0 {
        return;
    }
    let tick = i64::from(edit_tick);
    let Some(idx) = param.keyframes().iter().position(|kf| kf.tick == tick) else {
        return;
    };
    match edit_mode {
        EDIT_KEYFRAME => {
            let easing = param.keyframes()[idx].easing;
            param.set_keyframe(tick, [edit_x, edit_y], easing);
        }
        EDIT_OUT => {
            let mut t = tangents_or_default(&param.keyframes()[idx]);
            t.out_t = [edit_x, edit_y];
            if mirror {
                t.in_t = [-edit_x, -edit_y];
            }
            let _ = param.set_keyframe_tangents(tick, Some(t));
        }
        EDIT_IN => {
            let mut t = tangents_or_default(&param.keyframes()[idx]);
            t.in_t = [edit_x, edit_y];
            if mirror {
                t.out_t = [-edit_x, -edit_y];
            }
            let _ = param.set_keyframe_tangents(tick, Some(t));
        }
        _ => {}
    }
}

/// Motion-path overlay for `clip_id` in viewport-element coordinates.
///
/// `selected_tick` marks the matching diamond (`-1` ⇔ none). When
/// `edit_mode` ≠ 0 the path/handles reflect a tentative drag
/// (`edit_x`/`edit_y` are canvas-fraction position or tangent offsets).
#[allow(clippy::too_many_arguments)]
pub fn motion_path_in_viewport(
    sequence: &Sequence,
    clip_id: &str,
    view_w: f32,
    view_h: f32,
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
    selected_tick: i32,
    edit_mode: i32,
    edit_tick: i32,
    edit_x: f32,
    edit_y: f32,
    mirror: bool,
) -> MotionPath {
    let Some(clip) = find_clip(sequence, clip_id) else {
        return MotionPath::default();
    };
    let mut param = position_param(&clip);
    if param.keyframes().is_empty() {
        return MotionPath::default();
    }
    apply_live_edit(&mut param, edit_mode, edit_tick, edit_x, edit_y, mirror);

    let canvas = canvas_config(sequence);
    let (cw, ch) = (canvas.width as f32, canvas.height as f32);
    let (scale, ox, oy) = viewport_mapping(cw, ch, view_w, view_h, zoom, pan_x, pan_y);
    if scale <= 0.0 {
        return MotionPath::default();
    }

    let samples = sample_path_points(&param);
    let view_samples: Vec<[f32; 2]> = samples
        .iter()
        .map(|p| position_to_viewport(*p, cw, ch, scale, ox, oy))
        .collect();

    let keyframes: Vec<MotionPathKeyframe> = param
        .keyframes()
        .iter()
        .map(|kf| {
            let [x, y] = position_to_viewport(kf.value, cw, ch, scale, ox, oy);
            let selected = selected_tick >= 0 && kf.tick as i32 == selected_tick;
            let (has_handles, out_x, out_y, in_x, in_y) = if selected {
                let t = tangents_or_default(kf);
                let out = position_to_viewport(
                    [kf.value[0] + t.out_t[0], kf.value[1] + t.out_t[1]],
                    cw,
                    ch,
                    scale,
                    ox,
                    oy,
                );
                let inn = position_to_viewport(
                    [kf.value[0] + t.in_t[0], kf.value[1] + t.in_t[1]],
                    cw,
                    ch,
                    scale,
                    ox,
                    oy,
                );
                (true, out[0], out[1], inn[0], inn[1])
            } else {
                (false, 0.0, 0.0, 0.0, 0.0)
            };
            MotionPathKeyframe {
                x,
                y,
                tick: kf.tick as i32,
                selected,
                has_handles,
                out_x,
                out_y,
                in_x,
                in_y,
            }
        })
        .collect();

    MotionPath {
        visible: true,
        path_commands: svg_path_commands(&view_samples),
        keyframes: ModelRc::from(Rc::new(VecModel::from(keyframes))),
    }
}

/// Resolve a motion-path pointer into canvas-fraction edit values.
///
/// Mode 1 → new keyframe position; mode 2/3 → out/in tangent offset from
/// the keyframe value (mirroring applied by the caller / path sampler).
/// Non-position transform components are the playhead sample so a keyframe
/// drag can feed the existing transform-override path.
#[allow(clippy::too_many_arguments)]
pub fn resolve_motion_path_drag(
    sequence: &Sequence,
    clip_id: &str,
    playhead: i32,
    cursor_x: f32,
    cursor_y: f32,
    view_w: f32,
    view_h: f32,
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
    edit_mode: i32,
    edit_tick: i32,
) -> MotionPathDragResolution {
    let invalid = MotionPathDragResolution::default();
    if edit_mode == EDIT_NONE || edit_tick < 0 {
        return invalid;
    }
    let Some(clip) = find_clip(sequence, clip_id) else {
        return invalid;
    };
    let param = position_param(&clip);
    let tick = i64::from(edit_tick);
    let Some(kf) = param.keyframes().iter().find(|k| k.tick == tick) else {
        return invalid;
    };

    let canvas = canvas_config(sequence);
    let (cw, ch) = (canvas.width as f32, canvas.height as f32);
    let (scale, ox, oy) = viewport_mapping(cw, ch, view_w, view_h, zoom, pan_x, pan_y);
    if scale <= 0.0 {
        return invalid;
    }
    let tip = viewport_to_position(cursor_x, cursor_y, cw, ch, scale, ox, oy);
    let sample = sampled_transform(&clip, playhead);

    match edit_mode {
        EDIT_KEYFRAME => MotionPathDragResolution {
            valid: true,
            x: tip[0],
            y: tip[1],
            anchor_x: sample.anchor_point[0],
            anchor_y: sample.anchor_point[1],
            scale: sample.scale.x,
            scale_y: sample.scale.y,
            rotation: sample.rotation,
            opacity: sample.opacity,
        },
        EDIT_OUT | EDIT_IN => MotionPathDragResolution {
            valid: true,
            x: tip[0] - kf.value[0],
            y: tip[1] - kf.value[1],
            anchor_x: sample.anchor_point[0],
            anchor_y: sample.anchor_point[1],
            scale: sample.scale.x,
            scale_y: sample.scale.y,
            rotation: sample.rotation,
            opacity: sample.opacity,
        },
        _ => invalid,
    }
}

/// Look up a projected position keyframe's value / easing / tangents for a
/// commit that must preserve whatever the drag didn't touch.
pub fn position_keyframe_at(clip: &Clip, tick: i32) -> Option<ParamKeyframe> {
    clip.kf_position.iter().find(|kf| kf.tick == tick)
}

/// Easing + tangents for a projected position keyframe (commit helpers).
pub fn position_keyframe_commit_bits(kf: &ParamKeyframe) -> (Easing, Option<SpatialTangents>) {
    let easing = easing_from_ui(kf.easing, [kf.bez_x1, kf.bez_y1, kf.bez_x2, kf.bez_y2]);
    let tangents = if kf.has_tangents {
        Some(SpatialTangents {
            out_t: [kf.out_tx, kf.out_ty],
            in_t: [kf.in_tx, kf.in_ty],
        })
    } else {
        None
    };
    (easing, tangents)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Rational, RationalTime, TimeRange, Track};

    fn rt(value: i32) -> RationalTime {
        RationalTime {
            value,
            rate: Rational { num: 24, den: 1 },
        }
    }

    fn media_clip(id: &str) -> Clip {
        Clip {
            id: SharedString::from(id),
            name: SharedString::from(id),
            timeline_start: rt(0),
            source_range: TimeRange {
                start: rt(0),
                duration: rt(100),
            },
            media_id: SharedString::from("m1"),
            media_width: 1920,
            media_height: 1080,
            transform_scale: 1.0,
            transform_scale_y: 1.0,
            transform_scale_linked: true,
            transform_opacity: 1.0,
            transform_anchor_x: 0.5,
            transform_anchor_y: 0.5,
            ..Default::default()
        }
    }

    fn track(clips: Vec<Clip>) -> Track {
        Track {
            id: SharedString::from("1"),
            name: SharedString::from("V1"),
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

    fn sequence(clips: Vec<Clip>) -> Sequence {
        Sequence {
            id: SharedString::from("1"),
            name: SharedString::from("Sequence 1"),
            fps: Rational { num: 24, den: 1 },
            drop_frame: false,
            tracks: ModelRc::from(Rc::new(VecModel::from(vec![track(clips)]))),
            markers: Default::default(),
            width: 1920.0,
            height: 1080.0,
            aspect_index: 0,
            background: Default::default(),
        }
    }

    fn pos_kf(tick: i32, x: f32, y: f32, tangents: Option<SpatialTangents>) -> ParamKeyframe {
        let (has_tangents, out_tx, out_ty, in_tx, in_ty) = match tangents {
            Some(t) => (true, t.out_t[0], t.out_t[1], t.in_t[0], t.in_t[1]),
            None => (false, 0.0, 0.0, 0.0, 0.0),
        };
        ParamKeyframe {
            tick,
            value_x: x,
            value_y: y,
            easing: 0,
            bez_x1: 0.0,
            bez_y1: 0.0,
            bez_x2: 0.0,
            bez_y2: 0.0,
            has_tangents,
            out_tx,
            out_ty,
            in_tx,
            in_ty,
        }
    }

    const VW: f32 = 960.0;
    const VH: f32 = 540.0;

    fn path(
        seq: &Sequence,
        selected: i32,
        edit_mode: i32,
        edit_tick: i32,
        edit_x: f32,
        edit_y: f32,
        mirror: bool,
    ) -> MotionPath {
        motion_path_in_viewport(
            seq, "A", VW, VH, 1.0, 0.0, 0.0, selected, edit_mode, edit_tick, edit_x, edit_y, mirror,
        )
    }

    #[test]
    fn straight_two_keyframe_path_is_collinear_with_matching_endpoints() {
        let mut clip = media_clip("A");
        clip.kf_position = ModelRc::from(Rc::new(VecModel::from(vec![
            pos_kf(0, -0.25, -0.25, None),
            pos_kf(40, 0.25, 0.25, None),
        ])));
        let seq = sequence(vec![clip]);
        let path = path(&seq, -1, EDIT_NONE, -1, 0.0, 0.0, true);
        assert!(path.visible);
        assert!(!path.path_commands.is_empty());
        assert_eq!(path.keyframes.row_count(), 2);

        let a = path.keyframes.row_data(0).unwrap();
        let b = path.keyframes.row_data(1).unwrap();
        assert!((a.x - 240.0).abs() < 1e-2 && (a.y - 135.0).abs() < 1e-2);
        assert!((b.x - 720.0).abs() < 1e-2 && (b.y - 405.0).abs() < 1e-2);

        let param = position_param(&seq.tracks.row_data(0).unwrap().clips.row_data(0).unwrap());
        let samples = sample_path_points(&param);
        let canvas = canvas_config(&seq);
        let (cw, ch) = (canvas.width as f32, canvas.height as f32);
        let (scale, ox, oy) = viewport_mapping(cw, ch, VW, VH, 1.0, 0.0, 0.0);
        for p in &samples {
            let [x, y] = position_to_viewport(*p, cw, ch, scale, ox, oy);
            let t = if (b.x - a.x).abs() > 1e-3 {
                (x - a.x) / (b.x - a.x)
            } else {
                (y - a.y) / (b.y - a.y)
            };
            let exp_y = a.y + t * (b.y - a.y);
            assert!(
                (y - exp_y).abs() < 0.5,
                "sample ({x}, {y}) off the line (expected y≈{exp_y})"
            );
        }
    }

    #[test]
    fn curved_path_midpoint_deviates_from_the_chord() {
        let mut clip = media_clip("A");
        clip.kf_position = ModelRc::from(Rc::new(VecModel::from(vec![
            pos_kf(
                0,
                0.0,
                0.0,
                Some(SpatialTangents {
                    out_t: [0.0, 0.55],
                    in_t: [0.0, 0.0],
                }),
            ),
            pos_kf(
                40,
                0.5,
                0.5,
                Some(SpatialTangents {
                    out_t: [0.0, 0.0],
                    in_t: [-0.55, 0.0],
                }),
            ),
        ])));
        let seq = sequence(vec![clip]);
        let path = path(&seq, -1, EDIT_NONE, -1, 0.0, 0.0, true);
        assert!(path.visible);

        let a = path.keyframes.row_data(0).unwrap();
        let b = path.keyframes.row_data(1).unwrap();
        let param = position_param(&seq.tracks.row_data(0).unwrap().clips.row_data(0).unwrap());
        let mid = param.sample_at(20.0);
        let canvas = canvas_config(&seq);
        let (cw, ch) = (canvas.width as f32, canvas.height as f32);
        let (scale, ox, oy) = viewport_mapping(cw, ch, VW, VH, 1.0, 0.0, 0.0);
        let [mx, my] = position_to_viewport(mid, cw, ch, scale, ox, oy);
        let chord_x = (a.x + b.x) * 0.5;
        let chord_y = (a.y + b.y) * 0.5;
        let dist = (mx - chord_x).hypot(my - chord_y);
        assert!(
            dist > 5.0,
            "curved mid ({mx}, {my}) too close to chord ({chord_x}, {chord_y}), dist={dist}"
        );
    }

    #[test]
    fn constant_position_yields_empty_overlay() {
        let seq = sequence(vec![media_clip("A")]);
        let path = path(&seq, -1, EDIT_NONE, -1, 0.0, 0.0, true);
        assert!(!path.visible);
        assert!(path.path_commands.is_empty());
        assert_eq!(path.keyframes.row_count(), 0);
    }

    #[test]
    fn selected_keyframe_exposes_tangent_handles() {
        let mut clip = media_clip("A");
        clip.kf_position = ModelRc::from(Rc::new(VecModel::from(vec![
            pos_kf(
                0,
                0.0,
                0.0,
                Some(SpatialTangents {
                    out_t: [0.1, 0.0],
                    in_t: [-0.1, 0.0],
                }),
            ),
            pos_kf(40, 0.5, 0.0, None),
        ])));
        let seq = sequence(vec![clip]);
        let path = path(&seq, 0, EDIT_NONE, -1, 0.0, 0.0, true);
        let kf = path.keyframes.row_data(0).unwrap();
        assert!(kf.selected && kf.has_handles);
        // out tip at position 0.1 → canvas 1152 → view 576; kf at view 480.
        assert!((kf.out_x - 576.0).abs() < 1e-2);
        assert!((kf.in_x - 384.0).abs() < 1e-2);
        assert!(!path.keyframes.row_data(1).unwrap().has_handles);
    }

    #[test]
    fn live_keyframe_edit_moves_the_dot_without_touching_committed_model() {
        let mut clip = media_clip("A");
        clip.kf_position = ModelRc::from(Rc::new(VecModel::from(vec![
            pos_kf(0, 0.0, 0.0, None),
            pos_kf(40, 0.5, 0.0, None),
        ])));
        let seq = sequence(vec![clip]);
        let path = path(&seq, 0, EDIT_KEYFRAME, 0, 0.2, 0.1, true);
        let kf = path.keyframes.row_data(0).unwrap();
        // 0.2 → canvas 1344 → view 672; 0.1 → canvas 648 → view 324.
        assert!((kf.x - 672.0).abs() < 1e-2);
        assert!((kf.y - 324.0).abs() < 1e-2);
        // Committed projection unchanged.
        let committed = seq.tracks.row_data(0).unwrap().clips.row_data(0).unwrap();
        assert_eq!(committed.kf_position.row_data(0).unwrap().value_x, 0.0);
    }
}
