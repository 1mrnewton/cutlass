//! Drag / insert / delete planning for the keyframe graph editor.
//!
//! Pure helpers over projected clips + scalar params. The wire layer commits
//! via the preview worker (`set_param_keyframe` / grouped tick move).
//!
//! ## Bezier easing handles
//!
//! For the selected keyframe's *outgoing* segment (`kf → next`), CSS-style
//! cubic-bezier handles sit at control points `(x1,y1)` / `(x2,y2)` in
//! **segment space**:
//! - `x ∈ [0,1]` maps to ticks `[kf.tick, next.tick]`
//! - `y` maps to values `[kf.value, next.value]` via `lerp` (y may leave
//!   `[0,1]` for overshoot — handles are allowed outside the segment box
//!   vertically). Graph px use the same [`PlotMapping`] as dots/path.
//!
//! Named easings expose display control points via [`Easing::control_points`];
//! [`Easing::Hold`] has no handles.

use cutlass_models::{Easing, Param, ParamValue, SpatialTangents};

use super::channels::{channel_param, is_vec2_key, keyframe_at};
use super::{PAD_B, PAD_L, PAD_R, PAD_T};
use crate::Clip;
use crate::library_helpers::clip_param_value;
use crate::params::easing_from_ui;

/// Sanity clamp for bezier handle y (CSS allows overshoot; keep UI finite).
const HANDLE_Y_MIN: f32 = -2.0;
const HANDLE_Y_MAX: f32 = 3.0;

/// Which outgoing-segment bezier handle is being dragged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandleId {
    A,
    B,
}

/// Plot-space bezier handles for the selected keyframe's outgoing segment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SegmentHandles {
    pub a_px: f32,
    pub a_py: f32,
    pub b_px: f32,
    pub b_py: f32,
    pub start_px: f32,
    pub start_py: f32,
    pub end_px: f32,
    pub end_py: f32,
    /// Current control points `(x1,y1,x2,y2)` in segment space.
    pub points: [f32; 4],
    pub from_tick: i64,
    pub from_value: f32,
    pub to_tick: i64,
    pub to_value: f32,
}

/// Map segment-space `(u, v)` into graph pixels for an outgoing segment.
fn segment_to_plot(
    mapping: PlotMapping,
    from_tick: i64,
    to_tick: i64,
    from_value: f32,
    to_value: f32,
    u: f32,
    v: f32,
) -> (f32, f32) {
    let tick = from_tick as f64 + f64::from(u) * (to_tick - from_tick) as f64;
    let value = from_value + v * (to_value - from_value);
    let plot_w = (mapping.width - PAD_L - PAD_R).max(1.0);
    let plot_h = (mapping.height - PAD_T - PAD_B).max(1.0);
    let t_span = (mapping.t_max - mapping.t_min).max(1) as f64;
    let y_span = (mapping.y1 - mapping.y0).max(1e-6);
    let px = PAD_L + ((tick - mapping.t_min as f64) / t_span) as f32 * plot_w;
    let py = PAD_T + (1.0 - (value - mapping.y0) / y_span) * plot_h;
    (px, py)
}

/// Inverse of [`segment_to_plot`] for handle drag: cursor → segment `(u,v)`.
fn plot_to_segment(
    mapping: PlotMapping,
    from_tick: i64,
    to_tick: i64,
    from_value: f32,
    to_value: f32,
    x: f32,
    y: f32,
) -> (f32, f32) {
    let plot_w = (mapping.width - PAD_L - PAD_R).max(1.0);
    let plot_h = (mapping.height - PAD_T - PAD_B).max(1.0);
    let t_span = (mapping.t_max - mapping.t_min).max(1) as f64;
    let y_span = (mapping.y1 - mapping.y0).max(1e-6);
    let tick = mapping.t_min as f64 + f64::from(((x - PAD_L) / plot_w).clamp(0.0, 1.0)) * t_span;
    let value = mapping.y0 + (1.0 - ((y - PAD_T) / plot_h)) * y_span;
    let seg_t = (to_tick - from_tick).max(1) as f64;
    let u = ((tick - from_tick as f64) / seg_t) as f32;
    let value_span = to_value - from_value;
    let v = if value_span.abs() < 1e-6 {
        0.0
    } else {
        (value - from_value) / value_span
    };
    (u, v)
}

/// Clamp bezier control points: x ∈ `[0,1]`, y ∈ `[−2, 3]`.
pub fn clamp_bezier_points(points: [f32; 4]) -> [f32; 4] {
    [
        points[0].clamp(0.0, 1.0),
        points[1].clamp(HANDLE_Y_MIN, HANDLE_Y_MAX),
        points[2].clamp(0.0, 1.0),
        points[3].clamp(HANDLE_Y_MIN, HANDLE_Y_MAX),
    ]
}

/// Outgoing-segment handles for the keyframe at `selected_tick`, or `None`
/// when there is no successor / the easing is [`Easing::Hold`].
pub fn segment_handles(
    param: &Param<f32>,
    selected_tick: i64,
    mapping: PlotMapping,
) -> Option<SegmentHandles> {
    let kfs = param.keyframes();
    let idx = kfs.iter().position(|kf| kf.tick == selected_tick)?;
    let next = kfs.get(idx + 1)?;
    let kf = &kfs[idx];
    let points = kf.easing.control_points()?;
    let (start_px, start_py) =
        segment_to_plot(mapping, kf.tick, next.tick, kf.value, next.value, 0.0, 0.0);
    let (end_px, end_py) =
        segment_to_plot(mapping, kf.tick, next.tick, kf.value, next.value, 1.0, 1.0);
    let (a_px, a_py) = segment_to_plot(
        mapping, kf.tick, next.tick, kf.value, next.value, points[0], points[1],
    );
    let (b_px, b_py) = segment_to_plot(
        mapping, kf.tick, next.tick, kf.value, next.value, points[2], points[3],
    );
    Some(SegmentHandles {
        a_px,
        a_py,
        b_px,
        b_py,
        start_px,
        start_py,
        end_px,
        end_py,
        points,
        from_tick: kf.tick,
        from_value: kf.value,
        to_tick: next.tick,
        to_value: next.value,
    })
}

/// Cursor drag → updated clamped bezier points for one handle.
pub fn resolve_handle_drag(
    handles: &SegmentHandles,
    which: HandleId,
    cursor_x: f32,
    cursor_y: f32,
    mapping: PlotMapping,
) -> [f32; 4] {
    let (u, v) = plot_to_segment(
        mapping,
        handles.from_tick,
        handles.to_tick,
        handles.from_value,
        handles.to_value,
        cursor_x,
        cursor_y,
    );
    let mut points = handles.points;
    match which {
        HandleId::A => {
            points[0] = u;
            points[1] = v;
        }
        HandleId::B => {
            points[2] = u;
            points[3] = v;
        }
    }
    clamp_bezier_points(points)
}

/// Live-preview: clone `param` and set the outgoing easing at `from_tick`
/// to `Easing::Bezier { points }`.
pub fn live_handle_param(
    param: &Param<f32>,
    from_tick: i64,
    points: [f32; 4],
) -> Option<Param<f32>> {
    let mut out = param.clone();
    let value = out
        .keyframes()
        .iter()
        .find(|kf| kf.tick == from_tick)
        .map(|kf| kf.value)?;
    out.set_keyframe(
        from_tick,
        value,
        Easing::Bezier {
            points: clamp_bezier_points(points),
        },
    );
    Some(out)
}

/// True when `selected_tick` has a following keyframe (preset can expand).
pub fn can_apply_preset(param: &Param<f32>, selected_tick: i64) -> bool {
    let kfs = param.keyframes();
    let Some(idx) = kfs.iter().position(|kf| kf.tick == selected_tick) else {
        return false;
    };
    kfs.get(idx + 1).is_some()
}

/// Commit that writes a new [`Easing::Bezier`] at the selected keyframe
/// (same tick/value; preserves the other vec2 axis / spatial tangents).
pub fn plan_handle_commit(
    clip: &Clip,
    key: &str,
    _channel: i32,
    from_tick: i32,
    points: [f32; 4],
) -> Option<GraphCommit> {
    let kf = keyframe_at(clip, key, from_tick)?;
    let points = clamp_bezier_points(points);
    let (value_x, value_y) = if is_vec2_key(key) {
        (kf.value_x, kf.value_y)
    } else {
        (kf.value_x, 0.0)
    };
    let (_, param_value) = clip_param_value(key, value_x, value_y)?;
    let tangents = if key == "position" && kf.has_tangents {
        Some(SpatialTangents {
            out_t: [kf.out_tx, kf.out_ty],
            in_t: [kf.in_tx, kf.in_ty],
        })
    } else {
        None
    };
    let tick = i64::from(from_tick);
    Some(GraphCommit {
        param_key: key.to_string(),
        from_tick: tick,
        to_tick: tick,
        value: param_value,
        easing: Easing::Bezier { points },
        tangents,
        tick_moved: false,
    })
}

/// Plot ↔ data mapping captured with the last geometry build.
#[derive(Clone, Copy, Debug)]
pub struct PlotMapping {
    pub t_min: i64,
    pub t_max: i64,
    pub y0: f32,
    pub y1: f32,
    pub width: f32,
    pub height: f32,
}

impl PlotMapping {
    pub fn tick_at(&self, x: f32) -> i64 {
        let plot_w = (self.width - PAD_L - PAD_R).max(1.0);
        let frac = ((x - PAD_L) / plot_w).clamp(0.0, 1.0);
        let t = self.t_min as f64 + frac as f64 * (self.t_max - self.t_min).max(0) as f64;
        t.round() as i64
    }

    pub fn value_at(&self, y: f32) -> f32 {
        let plot_h = (self.height - PAD_T - PAD_B).max(1.0);
        let frac = (1.0 - ((y - PAD_T) / plot_h)).clamp(0.0, 1.0);
        self.y0 + frac * (self.y1 - self.y0)
    }
}

/// Clip absolute tick extent `[start, last]`.
pub fn clip_tick_bounds(clip: &Clip) -> (i64, i64) {
    let start = i64::from(clip.timeline_start.value);
    let dur = i64::from(clip.source_range.duration.value.max(1));
    (start, start + dur - 1)
}

/// Clamp `tick` into the clip and keep ±1 away from neighboring keyframes
/// (neighbors of `from_tick`, the keyframe being moved).
pub fn clamp_keyframe_tick(param: &Param<f32>, clip: &Clip, from_tick: i64, tick: i64) -> i64 {
    let (lo, hi) = clip_tick_bounds(clip);
    let kfs = param.keyframes();
    let prev = kfs
        .iter()
        .map(|kf| kf.tick)
        .filter(|&t| t < from_tick)
        .max();
    let next = kfs
        .iter()
        .map(|kf| kf.tick)
        .filter(|&t| t > from_tick)
        .min();
    let mut lo_b = lo;
    let mut hi_b = hi;
    if let Some(p) = prev {
        lo_b = lo_b.max(p + 1);
    }
    if let Some(n) = next {
        hi_b = hi_b.min(n - 1);
    }
    if lo_b > hi_b {
        return from_tick;
    }
    tick.clamp(lo_b, hi_b)
}

/// Apply a live drag onto a cloned scalar param (path preview).
pub fn apply_live_drag(param: &mut Param<f32>, from_tick: i64, to_tick: i64, value: f32) {
    let easing = param
        .keyframes()
        .iter()
        .find(|kf| kf.tick == from_tick)
        .map(|kf| kf.easing)
        .unwrap_or(Easing::Linear);
    if from_tick != to_tick {
        let _ = param.remove_keyframe(from_tick);
    }
    param.set_keyframe(to_tick, value, easing);
}

/// Cursor → clamped tick/value for an in-progress drag.
pub fn resolve_drag(
    clip: &Clip,
    key: &str,
    channel: i32,
    from_tick: i32,
    cursor_x: f32,
    cursor_y: f32,
    mapping: PlotMapping,
) -> Option<(i32, f32)> {
    let param = channel_param(clip, key, channel)?;
    let raw_tick = mapping.tick_at(cursor_x);
    let value = mapping.value_at(cursor_y);
    let tick = clamp_keyframe_tick(&param, clip, i64::from(from_tick), raw_tick);
    Some((tick as i32, value))
}

/// Planned engine commit for a graph drag release.
#[derive(Clone, Debug)]
pub struct GraphCommit {
    pub param_key: String,
    pub from_tick: i64,
    pub to_tick: i64,
    pub value: ParamValue,
    pub easing: Easing,
    pub tangents: Option<SpatialTangents>,
    /// True when the tick moved — worker should remove+set in one history group.
    pub tick_moved: bool,
}

/// Build a commit that preserves easing (and the other vec2 axis / tangents).
pub fn plan_drag_commit(
    clip: &Clip,
    key: &str,
    channel: i32,
    from_tick: i32,
    to_tick: i32,
    value: f32,
) -> Option<GraphCommit> {
    let kf = keyframe_at(clip, key, from_tick)?;
    let easing = easing_from_ui(kf.easing, [kf.bez_x1, kf.bez_y1, kf.bez_x2, kf.bez_y2]);
    let (value_x, value_y) = if is_vec2_key(key) {
        if channel <= 0 {
            (value, kf.value_y)
        } else {
            (kf.value_x, value)
        }
    } else {
        (value, 0.0)
    };
    let (_, param_value) = clip_param_value(key, value_x, value_y)?;
    let tangents = if key == "position" && kf.has_tangents {
        Some(SpatialTangents {
            out_t: [kf.out_tx, kf.out_ty],
            in_t: [kf.in_tx, kf.in_ty],
        })
    } else {
        None
    };
    let from = i64::from(from_tick);
    let to = i64::from(to_tick);
    Some(GraphCommit {
        param_key: key.to_string(),
        from_tick: from,
        to_tick: to,
        value: param_value,
        easing,
        tangents,
        tick_moved: from != to,
    })
}

/// Tick + sampled curve value for a double-click insert (Linear easing).
pub fn plan_insert(
    clip: &Clip,
    key: &str,
    channel: i32,
    cursor_x: f32,
    mapping: PlotMapping,
) -> Option<(i64, f32)> {
    let param = channel_param(clip, key, channel)?;
    let (lo, hi) = clip_tick_bounds(clip);
    let mut tick = mapping.tick_at(cursor_x).clamp(lo, hi);
    // Don't land on an existing keyframe — nudge toward a gap.
    if param.keyframes().iter().any(|kf| kf.tick == tick) {
        if tick < hi {
            tick += 1;
        } else if tick > lo {
            tick -= 1;
        } else {
            return None;
        }
        if param.keyframes().iter().any(|kf| kf.tick == tick) {
            return None;
        }
    }
    let value = param.sample(tick);
    Some((tick, value))
}

/// Live-edited scalar param for path preview during a drag.
pub fn live_param(
    clip: &Clip,
    key: &str,
    channel: i32,
    from_tick: i32,
    to_tick: i32,
    value: f32,
) -> Option<Param<f32>> {
    let mut param = channel_param(clip, key, channel)?;
    apply_live_drag(&mut param, i64::from(from_tick), i64::from(to_tick), value);
    Some(param)
}

/// Sample a live-edited scalar curve at `playhead` into a
/// `(ClipParam, ParamValue)` for [`crate::preview_worker::WorkerMsg::ParamOverride`].
///
/// The graph edits one channel; the preview frame needs the full property
/// value at the playhead. For vec2 keys the undragged axis is sampled from
/// the committed curve at the same tick (easing/time edits on one axis do
/// not rewrite the other).
pub fn live_playhead_override(
    clip: &Clip,
    key: &str,
    channel: i32,
    live: &Param<f32>,
    playhead: i32,
) -> Option<(cutlass_models::ClipParam, ParamValue)> {
    let ph = i64::from(playhead);
    let edited = live.sample(ph);
    let (value_x, value_y) = if is_vec2_key(key) {
        let other_ch = if channel <= 0 { 1 } else { 0 };
        let other = channel_param(clip, key, other_ch)
            .map(|p| p.sample(ph))
            .unwrap_or(0.0);
        if channel <= 0 {
            (edited, other)
        } else {
            (other, edited)
        }
    } else {
        (edited, 0.0)
    };
    clip_param_value(key, value_x, value_y)
}

/// Commit bits for an insert at `tick` with `value` (Linear).
pub fn plan_insert_commit(
    clip: &Clip,
    key: &str,
    channel: i32,
    tick: i64,
    value: f32,
) -> Option<GraphCommit> {
    // Seed the other axis from the curve sample so vec2 inserts don't zero it.
    let other = if is_vec2_key(key) {
        let other_ch = if channel <= 0 { 1 } else { 0 };
        channel_param(clip, key, other_ch)
            .map(|p| p.sample(tick))
            .unwrap_or(0.0)
    } else {
        0.0
    };
    let (value_x, value_y) = if is_vec2_key(key) {
        if channel <= 0 {
            (value, other)
        } else {
            (other, value)
        }
    } else {
        (value, 0.0)
    };
    let (_, param_value) = clip_param_value(key, value_x, value_y)?;
    Some(GraphCommit {
        param_key: key.to_string(),
        from_tick: tick,
        to_tick: tick,
        value: param_value,
        easing: Easing::Linear,
        tangents: None,
        tick_moved: false,
    })
}
