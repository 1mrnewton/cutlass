use cutlass_model::{Draft, Micros, SegmentId, TrackId};

use super::{EditError, ensure_unlocked};
use super::{locate, overlaps_except};

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
    super::lane::check(draft, &draft.tracks[to_index], &segment, start)?;

    let mut moved = segment;
    moved.target.start = start;
    let packed = super::magnet::packs(draft.config.maintrack_adsorb, &draft.tracks[to_index]);
    if !packed && overlaps_except(&draft.tracks[to_index], &moved, segment_id)? {
        return Err(EditError::Overlap);
    }

    draft.tracks[from_index].segments.remove(segment_index);
    let dest_index = draft
        .tracks
        .iter()
        .position(|track| &track.id == track_id)
        .ok_or(EditError::MissingTrack)?;
    let packed = super::magnet::packs(draft.config.maintrack_adsorb, &draft.tracks[dest_index]);
    let index = super::magnet::insert_at(
        &draft.tracks[dest_index].segments,
        moved.target.start.0,
        packed,
    );
    draft.tracks[dest_index].segments.insert(index, moved);
    Ok(())
}
