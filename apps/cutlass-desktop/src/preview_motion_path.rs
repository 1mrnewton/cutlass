//! Motion-path overlay geometry for the preview viewport.
//!
//! When the selected clip's position is keyframed, samples the projected
//! position [`Param`] across its keyframe range and maps canvas-fraction
//! positions into viewport px with the same letterbox/zoom/pan mapping the
//! selection box uses. Output is an SVG path-commands string (cheap to
//! diff) plus a model of keyframe dots.

use cutlass_models::Param;
use slint::{Model, ModelRc, SharedString, VecModel};
use std::rc::Rc;

use crate::params::position_param;
use crate::preview_select::{canvas_config, viewport_mapping};
use crate::{Clip, MotionPath, MotionPathKeyframe, Sequence, TrackKind};

/// Dense samples per keyframe segment (before the global cap).
const POINTS_PER_SEGMENT: usize = 24;
/// Soft ceiling on the polyline point count.
const MAX_PATH_POINTS: usize = 480;

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
    // Total ≈ n_seg * per_seg + 1 (shared endpoints counted once).
    let uncapped = n_seg * per_seg + 1;
    if uncapped > MAX_PATH_POINTS {
        per_seg = ((MAX_PATH_POINTS - 1) / n_seg).max(2);
    }
    // Each segment contributes `per_seg` samples at i/per_seg for
    // i ∈ [0, per_seg) (includes the start, excludes the end). The next
    // segment's i=0 (or the final push) supplies the junction / terminus
    // — no duplicates.
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

fn find_clip(sequence: &Sequence, clip_id: &str) -> Option<Clip> {
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

/// Motion-path overlay for `clip_id` in viewport-element coordinates.
/// Invisible (empty path / empty dots) when the clip is unknown or its
/// position isn't keyframed. `selected_tick` marks the matching diamond
/// (`-1` ⇔ none); commit 1 always passes `-1`.
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
) -> MotionPath {
    let Some(clip) = find_clip(sequence, clip_id) else {
        return MotionPath::default();
    };
    let param = position_param(&clip);
    if param.keyframes().is_empty() {
        return MotionPath::default();
    }

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
            MotionPathKeyframe {
                x,
                y,
                tick: kf.tick as i32,
                selected: selected_tick >= 0 && kf.tick as i32 == selected_tick,
            }
        })
        .collect();

    MotionPath {
        visible: true,
        path_commands: svg_path_commands(&view_samples),
        keyframes: ModelRc::from(Rc::new(VecModel::from(keyframes))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ParamKeyframe, Rational, RationalTime, TimeRange, Track};
    use cutlass_models::SpatialTangents;

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

    #[test]
    fn straight_two_keyframe_path_is_collinear_with_matching_endpoints() {
        let mut clip = media_clip("A");
        clip.kf_position = ModelRc::from(Rc::new(VecModel::from(vec![
            pos_kf(0, -0.25, -0.25, None),
            pos_kf(40, 0.25, 0.25, None),
        ])));
        let seq = sequence(vec![clip]);
        let path = motion_path_in_viewport(&seq, "A", VW, VH, 1.0, 0.0, 0.0, -1);
        assert!(path.visible);
        assert!(!path.path_commands.is_empty());
        assert_eq!(path.keyframes.row_count(), 2);

        let a = path.keyframes.row_data(0).unwrap();
        let b = path.keyframes.row_data(1).unwrap();
        // Viewport at half canvas: position ±0.25 → canvas 480/1440 → view 240/720.
        assert!((a.x - 240.0).abs() < 1e-2 && (a.y - 135.0).abs() < 1e-2);
        assert!((b.x - 720.0).abs() < 1e-2 && (b.y - 405.0).abs() < 1e-2);

        // Every sample lies on the line from a → b.
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
        let path = motion_path_in_viewport(&seq, "A", VW, VH, 1.0, 0.0, 0.0, -1);
        assert!(path.visible);

        let a = path.keyframes.row_data(0).unwrap();
        let b = path.keyframes.row_data(1).unwrap();
        let param = position_param(&seq.tracks.row_data(0).unwrap().clips.row_data(0).unwrap());
        let mid = param.sample_at(20.0);
        let canvas = canvas_config(&seq);
        let (cw, ch) = (canvas.width as f32, canvas.height as f32);
        let (scale, ox, oy) = viewport_mapping(cw, ch, VW, VH, 1.0, 0.0, 0.0);
        let [mx, my] = position_to_viewport(mid, cw, ch, scale, ox, oy);
        // Chord midpoint in viewport.
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
        let path = motion_path_in_viewport(&seq, "A", VW, VH, 1.0, 0.0, 0.0, -1);
        assert!(!path.visible);
        assert!(path.path_commands.is_empty());
        assert_eq!(path.keyframes.row_count(), 0);
    }
}
