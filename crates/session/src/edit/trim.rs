use cutlass_model::{Draft, Micros, Segment, SegmentId, TimeRange};

use crate::request::TrimEdge;

use super::speed::map_timeline_to_source;
use super::{EditError, ensure_unlocked};

pub fn trim(
    draft: &mut Draft,
    segment_id: &SegmentId,
    edge: TrimEdge,
    time: Micros,
) -> Result<(), EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let current = draft.tracks[track_index].segments[segment_index].clone();
    let mut next = current.clone();
    let speed = super::speed::timeline_speed(draft, &current);
    apply_edge(&mut next, edge, time, speed)?;
    if overlaps_except(&draft.tracks[track_index], &next, segment_id)? {
        return Err(EditError::Overlap);
    }
    draft.tracks[track_index].segments[segment_index] = next;
    Ok(())
}

fn apply_edge(
    segment: &mut Segment,
    edge: TrimEdge,
    time: Micros,
    speed: f64,
) -> Result<(), EditError> {
    let end = segment
        .target
        .start
        .0
        .checked_add(segment.target.duration.0)
        .ok_or(EditError::Range(cutlass_model::RangeError::EndOverflow))?;
    match edge {
        TrimEdge::Start => {
            let duration = end
                .checked_sub(time.0)
                .ok_or(EditError::Range(cutlass_model::RangeError::EndOverflow))?;
            if duration <= 0 {
                return Err(EditError::Range(
                    cutlass_model::RangeError::NonPositiveDuration,
                ));
            }
            let delta = time
                .0
                .checked_sub(segment.target.start.0)
                .ok_or(EditError::Range(cutlass_model::RangeError::EndOverflow))?;
            segment.target.start = time;
            segment.target.duration = Micros(duration);
            if let Some(source) = segment.source_range.as_mut() {
                let source_delta = map_timeline_to_source(delta, speed)?;
                let source_duration = map_timeline_to_source(duration, speed)?;
                shift_source_start(source, source_delta, source_duration)?;
            }
        }
        TrimEdge::End => {
            let duration = time
                .0
                .checked_sub(segment.target.start.0)
                .ok_or(EditError::Range(cutlass_model::RangeError::EndOverflow))?;
            if duration <= 0 {
                return Err(EditError::Range(
                    cutlass_model::RangeError::NonPositiveDuration,
                ));
            }
            segment.target.duration = Micros(duration);
            if let Some(source) = segment.source_range.as_mut() {
                let source_duration = map_timeline_to_source(duration, speed)?;
                if source_duration <= 0 {
                    return Err(EditError::Range(
                        cutlass_model::RangeError::NonPositiveDuration,
                    ));
                }
                source.duration = Micros(source_duration);
            }
        }
    }
    Ok(())
}

fn shift_source_start(source: &mut TimeRange, delta: i64, duration: i64) -> Result<(), EditError> {
    if duration <= 0 {
        return Err(EditError::Range(
            cutlass_model::RangeError::NonPositiveDuration,
        ));
    }
    let start = source
        .start
        .0
        .checked_add(delta)
        .ok_or(EditError::Range(cutlass_model::RangeError::EndOverflow))?;
    source.start = Micros(start);
    source.duration = Micros(duration);
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
