use cutlass_model::{ConstantSpeedError, Draft, Micros, Segment, SegmentId, SpeedMode};

use super::{EditError, ensure_unlocked};

/// Timeline microseconds become source microseconds at `speed`.
///
/// One timeline microsecond covers `speed` source microseconds. The product
/// rounds to the nearest microsecond, the same rounding as
/// [`cutlass_model::constant_target_duration`].
pub(crate) fn map_timeline_to_source(delta: i64, speed: f64) -> Result<i64, EditError> {
    if !speed.is_finite() || speed <= 0.0 {
        return Err(EditError::Speed(ConstantSpeedError::NonPositiveSpeed));
    }
    if delta == 0 {
        return Ok(0);
    }
    if delta.unsigned_abs() > (1_u64 << 53) {
        return Err(EditError::Speed(ConstantSpeedError::Unrepresentable));
    }
    let mapped = (delta as f64) * speed;
    let mapped = mapped.round();
    if !(mapped.is_finite() && mapped > i64::MIN as f64 && mapped < i64::MAX as f64) {
        return Err(EditError::Speed(ConstantSpeedError::Unrepresentable));
    }
    Ok(mapped as i64)
}

pub fn set_constant_speed(
    draft: &mut Draft,
    segment_id: &SegmentId,
    speed: f64,
) -> Result<(), EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let current = draft.tracks[track_index].segments[segment_index].clone();
    let mut next = current.clone();
    if let Some(source) = next.source_range.clone() {
        next.target.duration = cutlass_model::constant_target_duration(source.duration, speed)
            .map_err(EditError::Speed)?;
    } else if let Err(error) = cutlass_model::constant_target_duration(Micros(1), speed) {
        return Err(EditError::Speed(error));
    }
    if next.target.duration.0 <= 0 {
        return Err(EditError::Range(
            cutlass_model::RangeError::NonPositiveDuration,
        ));
    }
    next.speed = speed;
    if overlaps_except(&draft.tracks[track_index], &next, segment_id)? {
        return Err(EditError::Overlap);
    }
    let refs = next.extra_material_refs.clone();
    draft.tracks[track_index].segments[segment_index] = next;
    for material_id in refs {
        if let Some(material) = draft
            .materials
            .speeds
            .iter_mut()
            .find(|material| material.id == material_id)
        {
            material.mode = SpeedMode::Normal;
            material.speed = speed;
        }
    }
    Ok(())
}

fn locate(draft: &Draft, segment_id: &SegmentId) -> Result<(usize, usize), EditError> {
    for (track_index, track) in draft.tracks.iter().enumerate() {
        if let Some(segment_index) = track
            .segments
            .iter()
            .position(|segment| &segment.id == segment_id)
        {
            return Ok((track_index, segment_index));
        }
    }
    Err(EditError::MissingSegment)
}

fn overlaps_except(
    track: &cutlass_model::Track,
    candidate: &Segment,
    ignore: &SegmentId,
) -> Result<bool, EditError> {
    if candidate.target.duration.0 <= 0 {
        return Err(EditError::Range(
            cutlass_model::RangeError::NonPositiveDuration,
        ));
    }
    for segment in &track.segments {
        if &segment.id == ignore {
            continue;
        }
        if candidate
            .target
            .overlaps(&segment.target)
            .map_err(EditError::Range)?
        {
            return Ok(true);
        }
    }
    Ok(false)
}
