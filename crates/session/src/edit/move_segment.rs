use cutlass_model::{Draft, Micros, SegmentId, TrackId};

use super::{EditError, ensure_unlocked};

pub fn move_segment(
    draft: &mut Draft,
    segment_id: &SegmentId,
    track_id: &TrackId,
    start: Micros,
) -> Result<(), EditError> {
    let (from_index, segment_index) = locate(draft, segment_id)?;
    let segment = draft.tracks[from_index].segments[segment_index].clone();
    if segment.target.duration.0 <= 0 {
        return Err(EditError::Range(
            cutlass_model::RangeError::NonPositiveDuration,
        ));
    }
    let to_index = draft
        .tracks
        .iter()
        .position(|track| &track.id == track_id)
        .ok_or(EditError::MissingTrack)?;
    ensure_unlocked(draft.tracks[from_index].attribute)?;
    ensure_unlocked(draft.tracks[to_index].attribute)?;

    let mut moved = segment;
    moved.target.start = start;
    if overlaps_except(&draft.tracks[to_index], &moved, segment_id)? {
        return Err(EditError::Overlap);
    }

    draft.tracks[from_index].segments.remove(segment_index);
    let dest = if from_index == to_index {
        &mut draft.tracks[from_index]
    } else {
        &mut draft.tracks[to_index]
    };
    let index = dest
        .segments
        .iter()
        .position(|existing| existing.target.start > moved.target.start)
        .unwrap_or(dest.segments.len());
    dest.segments.insert(index, moved);
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
    candidate: &cutlass_model::Segment,
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
