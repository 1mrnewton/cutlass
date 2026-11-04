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
    let adsorb = draft.config.maintrack_adsorb;
    let index = draft
        .tracks
        .iter()
        .position(|track| &track.id == track_id)
        .ok_or(EditError::MissingTrack)?;
    ensure_unlocked(draft.tracks[index].attribute)?;
    super::lane::check(draft, &draft.tracks[index], &segment, segment.target.start)?;
    let track = &mut draft.tracks[index];
    let packed = super::magnet::packs(adsorb, track);
    if !packed {
        for existing in &track.segments {
            if segment
                .target
                .overlaps(&existing.target)
                .map_err(EditError::Range)?
            {
                return Err(EditError::Overlap);
            }
        }
    }
    let index = super::magnet::insert_at(&track.segments, segment.target.start.0, packed);
    track.segments.insert(index, segment);
    Ok(())
}
