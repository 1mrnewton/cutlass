//! Clips that follow a tracked target (`Request::Track`).
//!
//! The tracking job hands over the target's corners on the canvas frame by
//! frame. Each follower keeps its offset from the target as it is at the
//! anchor time: its place turns and scales about the target's middle as the
//! target turns and grows, and its own turn and scale change by the same
//! amounts. The poses are thinned to a few keys, and the path is kept as a
//! `video_tracking` record that each follower lists.

use cutlass_model::{
    Draft, MaterialId, MaterialRef, Micros, POSE_KEYS, RawMaterial, Segment, SegmentId, TrackPath,
    TrackType, VIDEO_TRACKING, clip_at, put_key,
};
use serde_json::{Map, Value, json};

use super::{EditError, ensure_unlocked, locate};

/// The most frames a path may hold: an hour at 60 frames a second.
pub const MAX_TRACK_SAMPLES: usize = 216_000;

/// The farthest a corner may be from the canvas middle, in canvas halves.
const MAX_REACH: f64 = 1_000.0;

/// How far a thinned key path may stray from the frame-by-frame one: a
/// thousandth of the canvas for place, a fifth of a degree, a thousandth
/// of scale.
const PLACE_TOLERANCE: f64 = 0.002;
const TURN_TOLERANCE: f64 = 0.2;
const SCALE_TOLERANCE: f64 = 0.001;

/// Make `followers` follow `target` along `path`, each keeping its offset
/// from the target at `anchor`, and keep the path as the `video_tracking`
/// record `tracking_id`.
pub fn track(
    draft: &mut Draft,
    target: &SegmentId,
    followers: &[SegmentId],
    tracking_id: &MaterialId,
    anchor: Micros,
    path: &TrackPath,
) -> Result<(), EditError> {
    check_path(path)?;
    let (track_index, segment_index) = locate(draft, target)?;
    let target_segment = &draft.tracks[track_index].segments[segment_index];
    if draft.tracks[track_index].kind != TrackType::Video
        || !matches!(
            draft.materials.get(target_segment.material_id.as_str()),
            Some(MaterialRef::Video(_))
        )
    {
        return Err(EditError::NotVideo);
    }
    let (first, last) = span(path);
    let target_range = target_segment.target.clone();
    if first.0 < target_range.start.0 || last.0 > target_range.end().0 {
        return Err(EditError::Track);
    }
    if !(first.0..=last.0).contains(&anchor.0) {
        return Err(EditError::Track);
    }
    if tracking_id.is_empty() || draft.materials.get(tracking_id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    let canvas = (
        f64::from(draft.canvas_config.width),
        f64::from(draft.canvas_config.height),
    );
    if !(canvas.0 > 0.0 && canvas.1 > 0.0) {
        return Err(EditError::CanvasSize);
    }
    let poses = target_poses(path, canvas).ok_or(EditError::Track)?;
    let anchor_index = nearest(path, anchor);

    let mut placed = Vec::with_capacity(followers.len());
    for (i, id) in followers.iter().enumerate() {
        if id == target || followers[..i].contains(id) {
            return Err(EditError::Track);
        }
        let (t, s) = locate(draft, id)?;
        let track = &draft.tracks[t];
        ensure_unlocked(track.attribute)?;
        if !matches!(
            track.kind,
            TrackType::Video | TrackType::Sticker | TrackType::Text
        ) {
            return Err(EditError::Track);
        }
        let range = track.segments[s].target.clone();
        if range.end().0 <= first.0 || range.start.0 > last.0 {
            return Err(EditError::Track);
        }
        placed.push((t, s));
    }
    if placed.is_empty() {
        return Err(EditError::Track);
    }

    let graphs = draft.keyframe_graph_list.clone();
    for &(t, s) in &placed {
        let segment = &mut draft.tracks[t].segments[s];
        follow(segment, &graphs, path, &poses, anchor_index, canvas)?;
        let stale: Vec<MaterialId> = segment
            .extra_material_refs
            .iter()
            .filter(|id| {
                draft
                    .materials
                    .video_trackings
                    .iter()
                    .any(|raw| &raw.id == *id)
            })
            .cloned()
            .collect();
        segment.extra_material_refs.retain(|id| !stale.contains(id));
        segment.extra_material_refs.push(tracking_id.clone());
    }
    draft
        .materials
        .video_trackings
        .push(record(tracking_id, target, anchor, path));
    // A record no clip lists any more goes.
    let listed: Vec<MaterialId> = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .flat_map(|segment| segment.extra_material_refs.iter().cloned())
        .collect();
    draft
        .materials
        .video_trackings
        .retain(|raw| listed.contains(&raw.id));
    Ok(())
}

fn check_path(path: &TrackPath) -> Result<(), EditError> {
    let samples = &path.samples;
    if samples.len() < 2 || samples.len() > MAX_TRACK_SAMPLES {
        return Err(EditError::Track);
    }
    let in_order = samples
        .windows(2)
        .all(|pair| pair[0].time.0 < pair[1].time.0);
    let finite = samples.iter().all(|sample| {
        sample.corners.iter().all(|&(x, y)| {
            x.is_finite() && y.is_finite() && x.abs() <= MAX_REACH && y.abs() <= MAX_REACH
        })
    });
    if !in_order || !finite {
        return Err(EditError::Track);
    }
    Ok(())
}

fn span(path: &TrackPath) -> (Micros, Micros) {
    let first = path.samples.first().map_or(Micros(0), |s| s.time);
    let last = path.samples.last().map_or(Micros(0), |s| s.time);
    (first, last)
}

/// The sample whose time is nearest `time`.
fn nearest(path: &TrackPath, time: Micros) -> usize {
    path.samples
        .iter()
        .enumerate()
        .min_by_key(|(_, sample)| sample.time.0.abs_diff(time.0))
        .map_or(0, |(i, _)| i)
}

/// The target's pose in one frame, in canvas pixels with y up.
#[derive(Clone, Copy, Debug)]
struct Pose {
    middle: (f64, f64),
    /// Radians, counterclockwise, unwrapped so neighbours never jump a turn.
    turn: f64,
    size: f64,
}

/// The target's pose in each frame, or `None` when a target has no size.
fn target_poses(path: &TrackPath, canvas: (f64, f64)) -> Option<Vec<Pose>> {
    let (hw, hh) = (canvas.0 / 2.0, canvas.1 / 2.0);
    let mut poses: Vec<Pose> = Vec::with_capacity(path.samples.len());
    for sample in &path.samples {
        let c = sample.corners.map(|(x, y)| (x * hw, y * hh));
        let middle = (
            c.iter().map(|p| p.0).sum::<f64>() / 4.0,
            c.iter().map(|p| p.1).sum::<f64>() / 4.0,
        );
        // The direction of the top and bottom edges together, so a tilt
        // that narrows one edge does not read as a turn.
        let across = (
            (c[1].0 - c[0].0) + (c[2].0 - c[3].0),
            (c[1].1 - c[0].1) + (c[2].1 - c[3].1),
        );
        let mut twice = 0.0;
        for i in 0..4 {
            let (a, b) = (c[i], c[(i + 1) % 4]);
            twice += a.0 * b.1 - b.0 * a.1;
        }
        let size = (twice / 2.0).abs().sqrt();
        if !(size > 0.0 && size.is_finite()) || across == (0.0, 0.0) {
            return None;
        }
        let mut turn = across.1.atan2(across.0);
        if let Some(before) = poses.last() {
            let tau = std::f64::consts::TAU;
            turn = before.turn + (turn - before.turn + std::f64::consts::PI).rem_euclid(tau)
                - std::f64::consts::PI;
        }
        poses.push(Pose { middle, turn, size });
    }
    Some(poses)
}

/// Key `segment`'s pose along the path, keeping its offset from the target
/// at `anchor`. Pose keys inside the tracked span are replaced.
fn follow(
    segment: &mut Segment,
    graphs: &[cutlass_model::KeyframeGraph],
    path: &TrackPath,
    poses: &[Pose],
    anchor: usize,
    canvas: (f64, f64),
) -> Result<(), EditError> {
    let (hw, hh) = (canvas.0 / 2.0, canvas.1 / 2.0);
    let start = segment.target.start;
    let end = segment.target.end();
    let anchor_time = path.samples[anchor]
        .time
        .0
        .clamp(start.0, end.0.saturating_sub(1).max(start.0));
    let (clip, uniform) =
        clip_at(segment, graphs, Micros(anchor_time)).ok_or(EditError::KeyTime)?;
    let (scale_x, scale_y) = if uniform.on {
        (uniform.value, uniform.value)
    } else {
        (clip.scale.x, clip.scale.y)
    };
    let at = poses[anchor];
    let place = (clip.transform.x * hw, clip.transform.y * hh);
    let offset = (place.0 - at.middle.0, place.1 - at.middle.1);

    // One row of values a frame: time, x, y, turn, scale x, scale y.
    let mut rows: Vec<(i64, [f64; 5])> = Vec::new();
    for (sample, pose) in path.samples.iter().zip(poses) {
        if sample.time.0 < start.0 || sample.time.0 >= end.0 {
            continue;
        }
        let k = pose.size / at.size;
        let turn = pose.turn - at.turn;
        let (sin, cos) = turn.sin_cos();
        let moved = (
            pose.middle.0 + k * (cos * offset.0 - sin * offset.1),
            pose.middle.1 + k * (sin * offset.0 + cos * offset.1),
        );
        rows.push((
            sample.time.0 - start.0,
            [
                moved.0 / hw,
                moved.1 / hh,
                clip.rotation + turn.to_degrees(),
                scale_x * k,
                scale_y * k,
            ],
        ));
    }
    if rows.is_empty() {
        return Err(EditError::Track);
    }
    if rows
        .iter()
        .any(|(_, values)| values.iter().any(|v| !v.is_finite()))
    {
        return Err(EditError::Track);
    }
    let tolerances = [
        PLACE_TOLERANCE,
        PLACE_TOLERANCE,
        TURN_TOLERANCE,
        SCALE_TOLERANCE,
        SCALE_TOLERANCE,
    ];
    let mut keep = vec![false; rows.len()];
    for (channel, tolerance) in tolerances.iter().enumerate() {
        let points: Vec<(f64, f64)> = rows
            .iter()
            .map(|(time, values)| (*time as f64, values[channel]))
            .collect();
        for i in thin(&points, *tolerance) {
            keep[i] = true;
        }
    }
    let (from, to) = (
        rows.first().map_or(0, |row| row.0),
        rows.last().map_or(0, |row| row.0),
    );
    for group in &mut segment.common_keyframes {
        if POSE_KEYS.contains(&group.property_type.as_str()) || group.property_type == "KFTypeScale"
        {
            group
                .keyframe_list
                .retain(|key| key.time_offset.0 < from || key.time_offset.0 > to);
        }
    }
    segment
        .common_keyframes
        .retain(|group| !group.keyframe_list.is_empty());
    for ((time, values), _) in rows.iter().zip(&keep).filter(|(_, keep)| **keep) {
        for (property, value) in POSE_KEYS.iter().zip(values) {
            put_key(segment, property, Micros(*time), *value);
        }
    }
    Ok(())
}

/// The indices Douglas-Peucker keeps of `points` (time, value), so the
/// straight lines between them stay within `tolerance` of every point.
/// The first and last are always kept.
fn thin(points: &[(f64, f64)], tolerance: f64) -> Vec<usize> {
    let Some(last) = points.len().checked_sub(1) else {
        return Vec::new();
    };
    let mut keep = vec![false; points.len()];
    keep[0] = true;
    keep[last] = true;
    let mut spans = vec![(0, last)];
    while let Some((a, b)) = spans.pop() {
        if b <= a + 1 {
            continue;
        }
        let ((ta, va), (tb, vb)) = (points[a], points[b]);
        let mut worst = (0.0, a);
        for (i, &(t, v)) in points.iter().enumerate().take(b).skip(a + 1) {
            let line = if tb > ta {
                va + (vb - va) * (t - ta) / (tb - ta)
            } else {
                va
            };
            let off = (v - line).abs();
            if off > worst.0 {
                worst = (off, i);
            }
        }
        if worst.0 > tolerance {
            keep[worst.1] = true;
            spans.push((a, worst.1));
            spans.push((worst.1, b));
        }
    }
    keep.iter()
        .enumerate()
        .filter_map(|(i, &k)| k.then_some(i))
        .collect()
}

/// The `video_tracking` record: the reference editor's fields, the target,
/// and the path inline.
fn record(id: &MaterialId, target: &SegmentId, anchor: Micros, path: &TrackPath) -> RawMaterial {
    let (first, last) = span(path);
    let points: Vec<Value> = path
        .samples
        .iter()
        .map(|sample| {
            let mut row = vec![json!(sample.time.0)];
            for (x, y) in sample.corners {
                row.push(json!(x));
                row.push(json!(y));
            }
            Value::Array(row)
        })
        .collect();
    let mut fields = Map::new();
    fields.insert("tracker_type".into(), json!(path.mode.tracker_type()));
    fields.insert("enable_video_tracking".into(), json!(true));
    fields.insert("enable_scale".into(), json!(true));
    fields.insert("enable_relative_distance".into(), json!(true));
    fields.insert(
        "tracking_time_range".into(),
        json!({ "start": first.0, "duration": last.0 - first.0 }),
    );
    fields.insert("segment_id".into(), json!(target.as_str()));
    fields.insert("anchor".into(), json!(anchor.0));
    fields.insert("path".into(), Value::Array(points));
    RawMaterial {
        id: id.clone(),
        kind: VIDEO_TRACKING.to_owned(),
        fields,
    }
}

#[cfg(test)]
mod tests;
