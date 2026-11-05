use cutlass_model::{
    Draft, MaterialId, MaterialRef, Micros, Segment, SegmentId, TimeRange, blend_window,
};

use crate::edit::locate;
use crate::edit::speed::{curve_points, map_timeline_to_source, timeline_speed};

use super::{EditError, ensure_unlocked};

/// Split one segment. The right piece keeps a transition that ended at the old cut.
pub fn split(
    draft: &mut Draft,
    segment_id: &SegmentId,
    at: Micros,
    right_id: &SegmentId,
) -> Result<(), EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    if right_id.is_empty()
        || draft
            .tracks
            .iter()
            .any(|track| track.segments.iter().any(|segment| &segment.id == right_id))
    {
        return Err(EditError::DuplicateSegment);
    }
    if window_contains(draft, track_index, at) {
        return Err(EditError::TransitionWindow);
    }
    let current = draft.tracks[track_index].segments[segment_index].clone();
    let start = current.target.start.0;
    let end = start
        .checked_add(current.target.duration.0)
        .ok_or(EditError::Range(cutlass_model::RangeError::EndOverflow))?;
    let left_duration =
        at.0.checked_sub(start)
            .ok_or(EditError::Range(cutlass_model::RangeError::EndOverflow))?;
    let right_duration = end
        .checked_sub(at.0)
        .ok_or(EditError::Range(cutlass_model::RangeError::EndOverflow))?;
    if left_duration <= 0 || right_duration <= 0 {
        return Err(EditError::Range(
            cutlass_model::RangeError::NonPositiveDuration,
        ));
    }
    let speed = timeline_speed(draft, &current);
    let mut left = current.clone();
    let mut right = current.clone();
    right.id = right_id.clone();
    left.target.duration = Micros(left_duration);
    right.target.start = at;
    right.target.duration = Micros(right_duration);
    cutlass_model::shift_keys(&mut right, -left_duration);
    let curve = curve_points(draft, &current);
    let curve_cut = match &curve {
        Some((_, points)) => {
            let share = left_duration as f64 / current.target.duration.0 as f64;
            let cut = cutlass_model::split_curve(points, share).map_err(EditError::Speed)?;
            split_curve_source(&mut left, &mut right, &current, cut.0)?;
            Some(cut)
        }
        None => {
            split_source(&mut left, &mut right, &current, speed)?;
            None
        }
    };
    super::mark::split_time_marks(draft, &mut left, &mut right, left_duration, right_id)?;
    super::beat::split_beats(draft, &mut left, &mut right, right_id)?;
    let transitions = transition_ids(draft, &current);
    left.extra_material_refs
        .retain(|material_id| !transitions.iter().any(|id| id == material_id));
    draft.tracks[track_index].segments[segment_index] = left;
    draft.tracks[track_index]
        .segments
        .insert(segment_index + 1, right);
    if let (Some((speed_id, _)), Some((_, before, after))) = (curve, curve_cut) {
        // Each half plays its own stretch of the curve.
        let right_speed = super::own::own(draft, track_index, segment_index + 1, &speed_id)?;
        let left_speed = super::own::own(draft, track_index, segment_index, &speed_id)?;
        for (id, points) in [(left_speed, before), (right_speed, after)] {
            if let Some(curve) = draft
                .materials
                .speeds
                .iter_mut()
                .find(|speed| speed.id == id)
                .and_then(|speed| speed.curve_speed.as_mut())
            {
                curve.speed_points = points;
            }
        }
    }
    Ok(())
}

/// Cut the source where a curve has played the share `x` of it.
fn split_curve_source(
    left: &mut Segment,
    right: &mut Segment,
    original: &Segment,
    x: f64,
) -> Result<(), EditError> {
    let Some(source) = original.source_range.clone() else {
        return Ok(());
    };
    let left_source = (source.duration.0 as f64 * x).round() as i64;
    let right_source = source.duration.0 - left_source;
    if left_source <= 0 || right_source <= 0 {
        return Err(EditError::Range(
            cutlass_model::RangeError::NonPositiveDuration,
        ));
    }
    left.source_range = Some(TimeRange::new(source.start.0, left_source));
    right.source_range = Some(TimeRange::new(source.start.0 + left_source, right_source));
    Ok(())
}

fn split_source(
    left: &mut Segment,
    right: &mut Segment,
    original: &Segment,
    speed: f64,
) -> Result<(), EditError> {
    let Some(source) = original.source_range.clone() else {
        return Ok(());
    };
    let left_source = map_timeline_to_source(left.target.duration.0, speed)?;
    let right_source = map_timeline_to_source(right.target.duration.0, speed)?;
    if left_source <= 0 || right_source <= 0 {
        return Err(EditError::Range(
            cutlass_model::RangeError::NonPositiveDuration,
        ));
    }
    let right_start = source
        .start
        .0
        .checked_add(left_source)
        .ok_or(EditError::Range(cutlass_model::RangeError::EndOverflow))?;
    left.source_range = Some(TimeRange::new(source.start.0, left_source));
    right.source_range = Some(TimeRange::new(right_start, right_source));
    Ok(())
}

fn window_contains(draft: &Draft, track_index: usize, at: Micros) -> bool {
    let track = &draft.tracks[track_index];
    for segment in &track.segments {
        let Some(cut) = segment
            .target
            .start
            .0
            .checked_add(segment.target.duration.0)
        else {
            continue;
        };
        for material_id in &segment.extra_material_refs {
            let Some(MaterialRef::Transition(transition)) =
                draft.materials.get(material_id.as_str())
            else {
                continue;
            };
            let Some((start, end)) =
                blend_window(Micros(cut), transition.duration, transition.is_overlap)
            else {
                continue;
            };
            if at.0 >= start.0 && at.0 < end.0 {
                return true;
            }
        }
    }
    false
}

fn transition_ids(draft: &Draft, segment: &Segment) -> Vec<MaterialId> {
    segment
        .extra_material_refs
        .iter()
        .filter(|material_id| {
            matches!(
                draft.materials.get(material_id.as_str()),
                Some(MaterialRef::Transition(_))
            )
        })
        .cloned()
        .collect()
}
