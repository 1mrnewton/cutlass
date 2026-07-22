//! Drag / insert / delete planning for the keyframe graph editor.
//!
//! Pure helpers over projected clips + scalar params. The wire layer commits
//! via the preview worker (`set_param_keyframe` / grouped tick move).

use cutlass_models::{Easing, Param, ParamValue, SpatialTangents};

use super::channels::{channel_param, is_vec2_key, keyframe_at};
use super::{PAD_B, PAD_L, PAD_R, PAD_T};
use crate::Clip;
use crate::library_helpers::clip_param_value;
use crate::params::easing_from_ui;

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
