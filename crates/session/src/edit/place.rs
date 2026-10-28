use cutlass_model::{Draft, Segment, TrackId};

use super::{EditError, ensure_unlocked};

pub fn place(draft: &mut Draft, track_id: &TrackId, segment: Segment) -> Result<(), EditError> {
    if segment.target.duration.0 <= 0 {
        return Err(EditError::Range(
            cutlass_model::RangeError::NonPositiveDuration,
        ));
    }
    if draft.tracks.iter().any(|track| {
        track
            .segments
            .iter()
            .any(|existing| existing.id == segment.id)
    }) {
        return Err(EditError::DuplicateSegment);
    }
    let track = draft
        .tracks
        .iter_mut()
        .find(|track| &track.id == track_id)
        .ok_or(EditError::MissingTrack)?;
    ensure_unlocked(track.attribute)?;
    for existing in &track.segments {
        if segment
            .target
            .overlaps(&existing.target)
            .map_err(EditError::Range)?
        {
            return Err(EditError::Overlap);
        }
    }
    let index = track
        .segments
        .iter()
        .position(|existing| existing.target.start > segment.target.start)
        .unwrap_or(track.segments.len());
    track.segments.insert(index, segment);
    Ok(())
}
