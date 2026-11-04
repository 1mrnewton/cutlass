//! Ripple delete and close gap: the requests that take clips out of a row,
//! or a gap out of it, and pull the later clips on that row left to fill
//! the space. Clips on other rows stay, apart from those linked to a moved
//! clip, which the session moves with it.

use std::collections::HashSet;

use cutlass_model::{Draft, Micros, SegmentId, TrackId};
use cutlass_session::Request;

/// Remove `segment_ids` and pull each later clip on their rows left by the
/// length removed before it. One undo step when applied together. Empty
/// when none of the ids is on the timeline.
#[must_use]
pub fn ripple_delete(draft: &Draft, segment_ids: &[SegmentId]) -> Vec<Request> {
    let doomed: HashSet<&SegmentId> = segment_ids.iter().collect();
    let mut removes = Vec::new();
    let mut moves = Vec::new();
    for track in &draft.tracks {
        let cut: Vec<(i64, i64)> = track
            .segments
            .iter()
            .filter(|segment| doomed.contains(&segment.id))
            .map(|segment| {
                let start = segment.target.start.0;
                (start, start.saturating_add(segment.target.duration.0))
            })
            .collect();
        if cut.is_empty() {
            continue;
        }
        removes.extend(
            track
                .segments
                .iter()
                .filter(|segment| doomed.contains(&segment.id))
                .map(|segment| Request::Remove {
                    segment_id: segment.id.clone(),
                }),
        );
        let mut kept: Vec<_> = track
            .segments
            .iter()
            .filter(|segment| !doomed.contains(&segment.id))
            .collect();
        kept.sort_by_key(|segment| segment.target.start.0);
        for segment in kept {
            let start = segment.target.start.0;
            let shift = removed_before(&cut, start);
            if shift > 0 {
                moves.push(moved(&segment.id, &track.id, start - shift));
            }
        }
    }
    removes.extend(moves);
    removes
}

/// The empty stretch on row `track_id` that `time` is in: from the end of
/// the clip before it, or the start, to the start of the clip after it.
/// `None` on a clip or past the last one.
#[must_use]
pub fn gap_at(draft: &Draft, track_id: &TrackId, time: Micros) -> Option<(Micros, Micros)> {
    let track = draft.tracks.iter().find(|track| &track.id == track_id)?;
    let mut start = 0_i64;
    let mut spans: Vec<(i64, i64)> = track
        .segments
        .iter()
        .map(|segment| {
            let from = segment.target.start.0;
            (from, from.saturating_add(segment.target.duration.0))
        })
        .collect();
    spans.sort_unstable();
    for (from, to) in spans {
        if time.0 < from {
            return (time.0 >= start).then_some((Micros(start), Micros(from)));
        }
        start = start.max(to);
    }
    None
}

/// Pull every clip after the gap at `time` on row `track_id` left by the
/// gap's length. Empty when there is no gap there.
#[must_use]
pub fn close_gap(draft: &Draft, track_id: &TrackId, time: Micros) -> Vec<Request> {
    let Some((start, end)) = gap_at(draft, track_id, time) else {
        return Vec::new();
    };
    let Some(track) = draft.tracks.iter().find(|track| &track.id == track_id) else {
        return Vec::new();
    };
    let shift = end.0 - start.0;
    let mut later: Vec<_> = track
        .segments
        .iter()
        .filter(|segment| segment.target.start.0 >= end.0)
        .collect();
    later.sort_by_key(|segment| segment.target.start.0);
    later
        .into_iter()
        .map(|segment| moved(&segment.id, track_id, segment.target.start.0 - shift))
        .collect()
}

/// How much of the removed spans `cut` lies before `time`.
fn removed_before(cut: &[(i64, i64)], time: i64) -> i64 {
    cut.iter()
        .map(|&(from, to)| (to.min(time) - from).max(0))
        .sum()
}

fn moved(segment_id: &SegmentId, track_id: &TrackId, start: i64) -> Request {
    Request::Move {
        segment_id: segment_id.clone(),
        track_id: track_id.clone(),
        start: Micros(start.max(0)),
    }
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Segment, TimeRange, Track};
    use cutlass_session::Session;

    use super::*;

    /// An overlay row with clips at 0–1, 2–3 and 5–6 seconds, and a main
    /// row clip that stays put.
    fn draft() -> Draft {
        let mut draft = Draft::empty("D");
        draft.config.maintrack_adsorb = false;
        draft.tracks[0]
            .segments
            .push(Segment::new("main", "m", TimeRange::new(0, 9_000_000)));
        let mut row = Track::main("row");
        for (id, start) in [("a", 0), ("b", 2_000_000), ("c", 5_000_000)] {
            row.segments
                .push(Segment::new(id, "m", TimeRange::new(start, 1_000_000)));
        }
        draft.tracks.push(row);
        draft
    }

    fn starts(draft: &Draft) -> Vec<(String, i64)> {
        draft.tracks[1]
            .segments
            .iter()
            .map(|segment| (segment.id.as_str().to_owned(), segment.target.start.0))
            .collect()
    }

    #[test]
    fn a_ripple_delete_pulls_the_later_clips_on_its_row_left() {
        let draft = draft();
        let requests = ripple_delete(&draft, &[SegmentId::new("a")]);
        let mut session = Session::open(draft);
        session.apply(0, &requests).unwrap();
        assert_eq!(
            starts(session.draft()),
            [("b".to_owned(), 1_000_000), ("c".to_owned(), 4_000_000)]
        );
        assert_eq!(
            session.draft().tracks[0].segments[0].target.start,
            Micros(0)
        );
        session.undo().unwrap();
        assert_eq!(starts(session.draft()).len(), 3);
    }

    #[test]
    fn deleting_two_clips_pulls_each_later_one_by_what_went_before_it() {
        let draft = draft();
        let requests = ripple_delete(&draft, &[SegmentId::new("a"), SegmentId::new("b")]);
        let mut session = Session::open(draft);
        session.apply(0, &requests).unwrap();
        assert_eq!(starts(session.draft()), [("c".to_owned(), 3_000_000)]);
        assert!(ripple_delete(session.draft(), &[SegmentId::new("gone")]).is_empty());
    }

    #[test]
    fn a_gap_closes_and_the_clips_after_it_move_up() {
        let draft = draft();
        let row = TrackId::new("row");
        assert_eq!(
            gap_at(&draft, &row, Micros(3_500_000)),
            Some((Micros(3_000_000), Micros(5_000_000)))
        );
        assert_eq!(gap_at(&draft, &row, Micros(500_000)), None);
        assert_eq!(gap_at(&draft, &row, Micros(7_000_000)), None);
        let requests = close_gap(&draft, &row, Micros(3_500_000));
        let mut session = Session::open(draft);
        session.apply(0, &requests).unwrap();
        assert_eq!(
            starts(session.draft()),
            [
                ("a".to_owned(), 0),
                ("b".to_owned(), 2_000_000),
                ("c".to_owned(), 3_000_000)
            ]
        );
        assert!(close_gap(session.draft(), &row, Micros(500_000)).is_empty());
    }
}
