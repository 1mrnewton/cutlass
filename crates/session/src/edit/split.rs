use cutlass_model::{
    Draft, MaterialId, MaterialRef, Micros, Segment, SegmentId, TimeRange, blend_window,
};

use crate::edit::speed::{locate, map_timeline_to_source, timeline_speed};

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
    split_source(&mut left, &mut right, &current, speed)?;
    let transitions = transition_ids(draft, &current);
    left.extra_material_refs
        .retain(|material_id| !transitions.iter().any(|id| id == material_id));
    draft.tracks[track_index].segments[segment_index] = left;
    draft.tracks[track_index]
        .segments
        .insert(segment_index + 1, right);
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
