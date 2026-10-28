mod move_segment;
mod place;
mod speed;
mod split;
mod track;
mod transition;
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
    /// The segment id is empty or already on the draft.
    DuplicateSegment,
    /// The track attribute has the lock bit set.
    LockedTrack,
    /// Two positive spans on one track overlap. Touching does not.
    Overlap,
    /// The constant-speed mapping refused the duration or the speed.
    Speed(cutlass_model::ConstantSpeedError),
    Range(RangeError),
    EmptyUndo,
    EmptyRedo,
    /// `begin` or `record` was called while a transaction was already open.
    HistoryOpen,
    /// `commit`, `amend`, or `rollback` needs the transaction `begin` or `record` opened.
    HistoryClosed,
    /// `amend` folds into a recorded step, and none exists yet.
    NothingToAmend,
    /// The bit is not mute, hide, lock, solo, or muted-by-solo.
    UnknownAttribute,
    MissingMaterial,
    /// The material id is already in a pool.
    DuplicateMaterial,
    /// The split time falls inside a blend window.
    TransitionWindow,
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
        Request::SetTrackAttribute { track_id, bit, on } => {
            track::set_track_attribute(draft, track_id, *bit, *on)
        }
        Request::SetConstantSpeed { segment_id, speed } => {
            speed::set_constant_speed(draft, segment_id, *speed)
        }
        Request::SetCurveSpeed {
            segment_id,
            speed_id,
            points,
        } => speed::set_curve_speed(draft, segment_id, speed_id, points),
        Request::SetTransition {
            segment_id,
            transition,
        } => transition::set_transition(draft, segment_id, (**transition).clone()),
        Request::Split {
            segment_id,
            at,
            right_id,
        } => split::split(draft, segment_id, *at, right_id),
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
