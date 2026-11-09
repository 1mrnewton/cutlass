//! Clips that follow a tracked target (`Request::Track`).
//!
//! The tracking job hands over the target's corners on the canvas frame by
//! frame. Each follower keeps its offset from the target as it is at the
//! anchor time: its place turns and scales about the target's middle as the
//! target turns and grows, and its own turn and scale change by the same
//! amounts. The poses are thinned to a few keys, and the path is kept as a
//! `video_tracking` record that each follower lists. A clip's mask can
//! follow instead, from the path in that clip's own frame.

use cutlass_effects::{Mask, MaskShape};
use cutlass_model::{
    Draft, MASK_KEYS, MaterialId, MaterialRef, Micros, POSE_KEYS, RawMaterial, Segment, SegmentId,
    TrackPath, TrackType, VIDEO_TRACKING, clip_at, put_key,
};
use serde_json::{Map, Value, json};

use super::{EditError, ensure_unlocked, locate};

/// The most frames a path may hold: an hour at 60 frames a second.
pub const MAX_TRACK_SAMPLES: usize = 216_000;

/// The farthest a corner may be from the canvas middle, in canvas halves.
const MAX_REACH: f64 = 1_000.0;

/// How far a thinned key path may stray from the frame-by-frame one:
/// 0.004 canvas halves for place (about two pixels on a 1080p canvas),
/// half a degree, half a percent of scale. A real track jitters by about
/// that much from frame to frame, so a tighter path keys every frame.
const PLACE_TOLERANCE: f64 = 0.004;
const TURN_TOLERANCE: f64 = 0.5;
const SCALE_TOLERANCE: f64 = 0.005;

/// Make `followers` follow `target` along `path`, each keeping its offset
/// from the target at `anchor`, and keep the path as the `video_tracking`
/// record `tracking_id`. Each of `masks` is a clip whose mask follows
/// instead, and the path in that clip's own frame.
pub fn track(
    draft: &mut Draft,
    target: &SegmentId,
    followers: &[SegmentId],
    masks: &[(SegmentId, TrackPath)],
    tracking_id: &MaterialId,
    anchor: Micros,
    path: &TrackPath,
) -> Result<(), EditError> {
    check_path(path)?;
    for (_, clip_path) in masks {
        check_path(clip_path)?;
    }
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
    for (i, (id, _)) in masks.iter().enumerate() {
        if followers.contains(id) || masks[..i].iter().any(|(other, _)| other == id) {
            return Err(EditError::Track);
        }
        let (t, s) = locate(draft, id)?;
        let track = &draft.tracks[t];
        ensure_unlocked(track.attribute)?;
        if !matches!(track.kind, TrackType::Video | TrackType::Sticker) {
            return Err(EditError::Track);
        }
        let range = track.segments[s].target.clone();
        if range.end().0 <= first.0 || range.start.0 > last.0 {
            return Err(EditError::Track);
        }
    }
    if placed.is_empty() && masks.is_empty() {
        return Err(EditError::Track);
    }

    let graphs = draft.keyframe_graph_list.clone();
    for &(t, s) in &placed {
        let segment = &mut draft.tracks[t].segments[s];
        follow(segment, &graphs, path, &poses, anchor_index, canvas)?;
        list_record(segment, &draft.materials.video_trackings, tracking_id);
    }
    for (index, (id, clip_path)) in masks.iter().enumerate() {
        let mask_id = MaterialId::new(format!("{}-mask-{index}", tracking_id.as_str()));
        follow_mask(draft, id, clip_path, anchor, &mask_id)?;
        let (t, s) = locate(draft, id)?;
        let segment = &mut draft.tracks[t].segments[s];
        list_record(segment, &draft.materials.video_trackings, tracking_id);
    }
    let mask_ids: Vec<&SegmentId> = masks.iter().map(|(id, _)| id).collect();
    draft
        .materials
        .video_trackings
        .push(record(tracking_id, target, anchor, path, &mask_ids));
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

/// List `tracking_id` on `segment`, in place of any tracking it listed.
fn list_record(segment: &mut Segment, trackings: &[RawMaterial], tracking_id: &MaterialId) {
    segment
        .extra_material_refs
        .retain(|id| !trackings.iter().any(|raw| &raw.id == id));
    segment.extra_material_refs.push(tracking_id.clone());
}

/// The target's box in one frame of a clip's own frame: its middle, its
/// turn in degrees (unwrapped), and half its width and height.
#[derive(Clone, Copy, Debug)]
struct BoxPose {
    middle: (f64, f64),
    turn: f64,
    half: (f64, f64),
}

fn box_poses(path: &TrackPath) -> Option<Vec<BoxPose>> {
    let mut poses: Vec<BoxPose> = Vec::with_capacity(path.samples.len());
    let length = |a: (f64, f64), b: (f64, f64)| (b.0 - a.0).hypot(b.1 - a.1);
    for sample in &path.samples {
        let c = sample.corners;
        let middle = (
            c.iter().map(|p| p.0).sum::<f64>() / 4.0,
            c.iter().map(|p| p.1).sum::<f64>() / 4.0,
        );
        let across = (
            (c[1].0 - c[0].0) + (c[2].0 - c[3].0),
            (c[1].1 - c[0].1) + (c[2].1 - c[3].1),
        );
        let half = (
            (length(c[0], c[1]) + length(c[3], c[2])) / 4.0,
            (length(c[0], c[3]) + length(c[1], c[2])) / 4.0,
        );
        if !(half.0 > 0.0 && half.1 > 0.0 && half.0.is_finite() && half.1.is_finite()) {
            return None;
        }
        let mut turn = across.1.atan2(across.0).to_degrees();
        if let Some(before) = poses.last() {
            turn = before.turn + (turn - before.turn + 180.0).rem_euclid(360.0) - 180.0;
        }
        poses.push(BoxPose { middle, turn, half });
    }
    Some(poses)
}

/// Key clip `id`'s mask along `clip_path`, the target's box in the clip's
/// own frame, keeping the mask's place, size and turn relative to the box
/// at `anchor`. A clip with no mask gets a rectangle the size of the box,
/// stored as `mask_id`.
fn follow_mask(
    draft: &mut Draft,
    id: &SegmentId,
    clip_path: &TrackPath,
    anchor: Micros,
    mask_id: &MaterialId,
) -> Result<(), EditError> {
    let poses = box_poses(clip_path).ok_or(EditError::Track)?;
    let at = poses[nearest(clip_path, anchor)];
    let (t, s) = locate(draft, id)?;
    let segment = &draft.tracks[t].segments[s];
    let (start, end) = (segment.target.start, segment.target.end());
    let anchor_time = anchor
        .0
        .clamp(start.0, end.0.saturating_sub(1).max(start.0));
    let graphs = draft.keyframe_graph_list.clone();
    let existing = cutlass_effects::segment_mask(draft, segment)
        .map(|mask| mask.at(segment, &graphs, Micros(anchor_time)));
    #[allow(clippy::cast_possible_truncation)]
    let base = existing.unwrap_or(Mask {
        x: at.middle.0 as f32,
        y: at.middle.1 as f32,
        width: at.half.0 as f32,
        height: at.half.1 as f32,
        rotation: at.turn as f32,
        ..Mask::new(MaskShape::Rectangle)
    });
    if existing.is_none() {
        super::mask::set_mask(draft, id, mask_id, Some(&base))?;
    }
    // The mask's middle in the box's own turned, unit frame at the anchor.
    let (sin0, cos0) = at.turn.to_radians().sin_cos();
    let (dx, dy) = (
        f64::from(base.x) - at.middle.0,
        f64::from(base.y) - at.middle.1,
    );
    let local = (
        (cos0 * dx + sin0 * dy) / at.half.0,
        (-sin0 * dx + cos0 * dy) / at.half.1,
    );
    let segment = &mut draft.tracks[t].segments[s];
    let mut rows: Vec<(i64, [f64; 5])> = Vec::new();
    for (sample, pose) in clip_path.samples.iter().zip(&poses) {
        if sample.time.0 < start.0 || sample.time.0 >= end.0 {
            continue;
        }
        let (sin, cos) = pose.turn.to_radians().sin_cos();
        let (u, v) = (local.0 * pose.half.0, local.1 * pose.half.1);
        rows.push((
            sample.time.0 - start.0,
            [
                pose.middle.0 + cos * u - sin * v,
                pose.middle.1 + sin * u + cos * v,
                f64::from(base.width) * pose.half.0 / at.half.0,
                f64::from(base.height) * pose.half.1 / at.half.1,
                f64::from(base.rotation) + pose.turn - at.turn,
            ],
        ));
    }
    let names = &MASK_KEYS[..5];
    let tolerances = [
        PLACE_TOLERANCE,
        PLACE_TOLERANCE,
        PLACE_TOLERANCE,
        PLACE_TOLERANCE,
        TURN_TOLERANCE,
    ];
    key_rows(segment, &rows, names, &tolerances, anchor_time - start.0)
}

/// Key `rows` (offset and values) on `names`, thinned per value within
/// `tolerances` and always keyed at `anchor`, replacing those keys inside
/// the rows' span.
fn key_rows(
    segment: &mut Segment,
    rows: &[(i64, [f64; 5])],
    names: &[&str],
    tolerances: &[f64; 5],
    anchor: i64,
) -> Result<(), EditError> {
    if rows.is_empty()
        || rows
            .iter()
            .any(|(_, values)| values.iter().any(|v| !v.is_finite()))
    {
        return Err(EditError::Track);
    }
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
    if let Some(i) = rows.iter().position(|row| row.0 == anchor) {
        keep[i] = true;
    }
    let (from, to) = (
        rows.first().map_or(0, |row| row.0),
        rows.last().map_or(0, |row| row.0),
    );
    for group in &mut segment.common_keyframes {
        if names.contains(&group.property_type.as_str()) {
            group
                .keyframe_list
                .retain(|key| key.time_offset.0 < from || key.time_offset.0 > to);
        }
    }
    segment
        .common_keyframes
        .retain(|group| !group.keyframe_list.is_empty());
    for ((time, values), _) in rows.iter().zip(&keep).filter(|(_, keep)| **keep) {
        for (property, value) in names.iter().zip(values) {
            put_key(segment, property, Micros(*time), *value);
        }
    }
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
    // A key at the anchor keeps the follower's own pose there exactly, so
    // tracking again from the same anchor finds the same offset.
    if let Some(i) = rows.iter().position(|row| row.0 == anchor_time - start.0) {
        keep[i] = true;
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
fn record(
    id: &MaterialId,
    target: &SegmentId,
    anchor: Micros,
    path: &TrackPath,
    masks: &[&SegmentId],
) -> RawMaterial {
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
    if !masks.is_empty() {
        let ids: Vec<&str> = masks.iter().map(|id| id.as_str()).collect();
        fields.insert("masks".into(), json!(ids));
    }
    fields.insert("path".into(), Value::Array(points));
    RawMaterial {
        id: id.clone(),
        kind: VIDEO_TRACKING.to_owned(),
        fields,
    }
}

#[cfg(test)]
mod tests;
