mod move_segment;
mod place;
mod trim;

use cutlass_model::{Draft, Micros, RangeError, TrackAttribute};

use crate::request::Request;

pub use move_segment::move_segment;
pub use place::place;
pub use trim::trim;

/// Why one request did not change the draft.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditError {
    /// The caller read a different revision than the open draft.
    RevisionMismatch {
        expected: u64,
        actual: u64,
    },
    MissingTrack,
    MissingSegment,
    /// The segment id is already on the draft.
    DuplicateSegment,
    /// The track attribute has the lock bit set.
    LockedTrack,
    /// Two positive spans on one track overlap. Touching does not.
    Overlap,
    /// The source range is set and speed is not 1, so the trim has no mapping yet.
    SpeedUnmapped,
    Range(RangeError),
    EmptyUndo,
    EmptyRedo,
}

pub fn apply_request(draft: &mut Draft, request: &Request) -> Result<(), EditError> {
    match request {
        Request::Place { track_id, segment } => place(draft, track_id, (**segment).clone()),
        Request::Move {
            segment_id,
            track_id,
            start,
        } => move_segment(draft, segment_id, track_id, *start),
        Request::Trim {
            segment_id,
            edge,
            time,
        } => trim(draft, segment_id, *edge, *time),
    }
}

pub fn apply_all(draft: &mut Draft, requests: &[Request]) -> Result<(), EditError> {
    for request in requests {
        apply_request(draft, request)?;
    }
    sync_duration(draft);
    Ok(())
}

/// Stored duration becomes the latest target end. Playback reads that field.
pub fn sync_duration(draft: &mut Draft) {
    let mut end = 0_i64;
    for track in &draft.tracks {
        for segment in &track.segments {
            end = end.max(segment.target.end().0);
        }
    }
    draft.duration = Micros(end);
}

pub fn ensure_unlocked(attribute: TrackAttribute) -> Result<(), EditError> {
    if attribute.contains(TrackAttribute::LOCK) {
        Err(EditError::LockedTrack)
    } else {
        Ok(())
    }
}
