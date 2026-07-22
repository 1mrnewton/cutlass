//! Keyframe graph editor: animated scalar-channel enumeration + curve display.
//!
//! The timeline drawer shows one scalar channel at a time. Vec2 params expose
//! X/Y as two channels; color params are never graphed. Geometry is an SVG
//! path-commands string (same cheap-diff pattern as the motion-path overlay)
//! plus a model of keyframe dots in plot coordinates.

mod channels;
mod edit;

#[cfg(test)]
mod tests;

pub use channels::{animated_channels, channel_param};
pub use edit::{
    GraphCommit, HandleId, PlotMapping, SegmentHandles, live_handle_param, live_param,
    plan_drag_commit, plan_handle_commit, plan_insert, plan_insert_commit, resolve_drag,
    resolve_handle_drag, segment_handles,
};

use cutlass_models::{Easing, Param};
use slint::{ModelRc, SharedString, VecModel};
use std::rc::Rc;

use crate::params::{easing_from_ui, easing_to_ui};
use crate::preview_motion_path::find_projected_clip;
use crate::{GraphChannel, GraphDot, ParamKeyframe, Sequence};

fn easing_of(kf: &ParamKeyframe) -> Easing {
    easing_from_ui(kf.easing, [kf.bez_x1, kf.bez_y1, kf.bez_x2, kf.bez_y2])
}

/// Dense samples across the keyframe tick range.
const PATH_SAMPLES: usize = 256;
/// Padding fraction on each side of the value range.
const Y_PAD_FRAC: f32 = 0.10;
/// Left inset for y-axis labels (logical px).
pub(crate) const PAD_L: f32 = 44.0;
/// Other plot insets.
pub(crate) const PAD_R: f32 = 8.0;
pub(crate) const PAD_T: f32 = 10.0;
pub(crate) const PAD_B: f32 = 10.0;

/// Expand `[min, max]` by [`Y_PAD_FRAC`] on each side. Degenerate ranges get
/// a unit-sized pad so a flat curve still has vertical room.
pub fn padded_y_range(min_v: f32, max_v: f32) -> (f32, f32) {
    let (lo, hi) = if min_v <= max_v {
        (min_v, max_v)
    } else {
        (max_v, min_v)
    };
    let span = hi - lo;
    if span < 1e-6 {
        let pad = lo.abs().max(1.0) * Y_PAD_FRAC;
        return (lo - pad, hi + pad);
    }
    let pad = span * Y_PAD_FRAC;
    (lo - pad, hi + pad)
}

fn format_value(v: f32) -> SharedString {
    if v.abs() >= 100.0 || (v.fract().abs() < 1e-3 && v.abs() >= 1.0) {
        SharedString::from(format!("{v:.0}"))
    } else {
        SharedString::from(format!("{v:.2}"))
    }
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

/// Sample `param` at 256 points across `[t_min, t_max]` (inclusive ends).
pub fn sample_curve(param: &Param<f32>, t_min: i64, t_max: i64) -> Vec<(f64, f32)> {
    if t_max < t_min {
        return Vec::new();
    }
    if t_max == t_min {
        return vec![(t_min as f64, param.sample(t_min))];
    }
    let mut out = Vec::with_capacity(PATH_SAMPLES);
    let span = (t_max - t_min) as f64;
    for i in 0..PATH_SAMPLES {
        let t = t_min as f64 + span * (i as f64 / (PATH_SAMPLES - 1) as f64);
        out.push((t, param.sample_at(t)));
    }
    out
}

/// Plot geometry for one channel. Empty path when the channel isn't animated.
pub struct GraphGeometry {
    pub path_commands: SharedString,
    pub dots: Vec<GraphDot>,
    pub y_min_label: SharedString,
    pub y_mid_label: SharedString,
    pub y_max_label: SharedString,
    pub grid_min_y: f32,
    pub grid_mid_y: f32,
    pub grid_max_y: f32,
    pub playhead_x: f32,
    pub playhead_visible: bool,
    pub plot_w: f32,
    pub plot_h: f32,
    pub mapping: Option<edit::PlotMapping>,
    /// Outgoing-segment bezier handles for the selected keyframe (Hold → none).
    pub handles: Option<edit::SegmentHandles>,
}

impl Default for GraphGeometry {
    fn default() -> Self {
        Self {
            path_commands: SharedString::default(),
            dots: Vec::new(),
            y_min_label: SharedString::default(),
            y_mid_label: SharedString::default(),
            y_max_label: SharedString::default(),
            grid_min_y: 0.0,
            grid_mid_y: 0.0,
            grid_max_y: 0.0,
            playhead_x: 0.0,
            playhead_visible: false,
            plot_w: 0.0,
            plot_h: 0.0,
            mapping: None,
            handles: None,
        }
    }
}

/// Build path + dots for `param` into a `width`×`height` plot.
pub fn build_geometry(
    param: &Param<f32>,
    playhead: i32,
    width: f32,
    height: f32,
    selected_tick: i32,
) -> GraphGeometry {
    let kfs = param.keyframes();
    if kfs.is_empty() || width <= PAD_L + PAD_R + 1.0 || height <= PAD_T + PAD_B + 1.0 {
        return GraphGeometry::default();
    }
    let t_min = kfs.first().unwrap().tick;
    let t_max = kfs.last().unwrap().tick;
    let samples = sample_curve(param, t_min, t_max);
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for (_, v) in &samples {
        min_v = min_v.min(*v);
        max_v = max_v.max(*v);
    }
    for kf in kfs {
        min_v = min_v.min(kf.value);
        max_v = max_v.max(kf.value);
    }
    let (y0, y1) = padded_y_range(min_v, max_v);
    let plot_w = width - PAD_L - PAD_R;
    let plot_h = height - PAD_T - PAD_B;
    let t_span = (t_max - t_min).max(1) as f64;
    let y_span = (y1 - y0).max(1e-6);

    let map_x = |t: f64| -> f32 { PAD_L + ((t - t_min as f64) / t_span) as f32 * plot_w };
    let map_y = |v: f32| -> f32 { PAD_T + (1.0 - (v - y0) / y_span) * plot_h };

    let points: Vec<[f32; 2]> = samples
        .iter()
        .map(|(t, v)| [map_x(*t), map_y(*v)])
        .collect();

    let dots: Vec<GraphDot> = kfs
        .iter()
        .map(|kf| GraphDot {
            tick: kf.tick as i32,
            value: kf.value,
            px: map_x(kf.tick as f64),
            py: map_y(kf.value),
            selected: selected_tick >= 0 && kf.tick as i32 == selected_tick,
            easing_tag: easing_to_ui(kf.easing).0,
        })
        .collect();

    let mid_v = (y0 + y1) * 0.5;
    let ph = i64::from(playhead);
    let playhead_visible = ph >= t_min && ph <= t_max;
    let mapping = edit::PlotMapping {
        t_min,
        t_max,
        y0,
        y1,
        width,
        height,
    };
    let handles = if selected_tick >= 0 {
        edit::segment_handles(param, i64::from(selected_tick), mapping)
    } else {
        None
    };
    GraphGeometry {
        path_commands: svg_path_commands(&points),
        dots,
        y_min_label: format_value(y0),
        y_mid_label: format_value(mid_v),
        y_max_label: format_value(y1),
        grid_min_y: map_y(y0),
        grid_mid_y: map_y(mid_v),
        grid_max_y: map_y(y1),
        playhead_x: map_x(ph as f64),
        playhead_visible,
        plot_w: width,
        plot_h: height,
        mapping: Some(mapping),
        handles,
    }
}

/// Resolve the selected channel (falling back to the first animated one).
pub fn resolve_selection(
    channels: &[GraphChannel],
    selected_key: &str,
    selected_channel: i32,
) -> (String, i32, i32) {
    if channels.is_empty() {
        return (String::new(), 0, 0);
    }
    if let Some((i, ch)) = channels
        .iter()
        .enumerate()
        .find(|(_, c)| c.key.as_str() == selected_key && c.channel == selected_channel)
    {
        return (ch.key.to_string(), ch.channel, i as i32);
    }
    let ch = &channels[0];
    (ch.key.to_string(), ch.channel, 0)
}

/// Inputs for a graph refresh (selection + plot size + channel choice).
pub struct GraphRefreshInput<'a> {
    pub sequence: &'a Sequence,
    pub clip_id: &'a str,
    pub playhead: i32,
    pub width: f32,
    pub height: f32,
    pub selected_key: &'a str,
    pub selected_channel: i32,
    pub selected_tick: i32,
}

/// Full refresh payload applied onto `GraphBackend` by the wire layer.
pub struct GraphRefresh {
    pub channels: ModelRc<GraphChannel>,
    pub channel_labels: ModelRc<SharedString>,
    pub channel_index: i32,
    pub selected_key: SharedString,
    pub selected_channel: i32,
    pub geometry: GraphGeometry,
}

pub fn refresh_graph(input: GraphRefreshInput<'_>) -> GraphRefresh {
    let empty = || GraphRefresh {
        channels: ModelRc::from(Rc::new(VecModel::from(Vec::<GraphChannel>::new()))),
        channel_labels: ModelRc::from(Rc::new(VecModel::from(Vec::<SharedString>::new()))),
        channel_index: 0,
        selected_key: SharedString::default(),
        selected_channel: 0,
        geometry: GraphGeometry::default(),
    };
    let Some(clip) = find_projected_clip(input.sequence, input.clip_id) else {
        return empty();
    };
    let channels = animated_channels(&clip);
    if channels.is_empty() {
        return empty();
    }
    let (key, channel, index) =
        resolve_selection(&channels, input.selected_key, input.selected_channel);
    let labels: Vec<SharedString> = channels.iter().map(|c| c.label.clone()).collect();
    let geometry = match channel_param(&clip, &key, channel) {
        Some(param) => build_geometry(
            &param,
            input.playhead,
            input.width,
            input.height,
            input.selected_tick,
        ),
        None => GraphGeometry::default(),
    };
    GraphRefresh {
        channels: ModelRc::from(Rc::new(VecModel::from(channels))),
        channel_labels: ModelRc::from(Rc::new(VecModel::from(labels))),
        channel_index: index,
        selected_key: SharedString::from(key),
        selected_channel: channel,
        geometry,
    }
}
