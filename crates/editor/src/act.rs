//! Which clip a playhead edit acts on, and the edits that cut a clip there:
//! split, and delete the part before or after the playhead.

use cutlass_model::{Draft, Micros, Segment, SegmentId, Track};
use cutlass_session::{Request, TrimEdge};

/// The shortest piece a cut may leave: one frame at the draft's rate, or one
/// microsecond when the draft has no rate.
#[must_use]
pub fn frame_margin(draft: &Draft) -> Micros {
    Micros(
        draft
            .frame_rate()
            .and_then(|rate| rate.frame_start(1))
            .map_or(1, |time| time.0.max(1)),
    )
}

/// The clip a playhead edit acts on at `time`: the latest selected clip that
/// covers it, else the main-track clip there, else the first clip there.
#[must_use]
pub fn clip_under(draft: &Draft, selection: &[SegmentId], time: Micros) -> Option<SegmentId> {
    let picked = selection.iter().rev().find(|id| {
        draft
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .any(|segment| &segment.id == *id && covers(segment, time))
    });
    if let Some(id) = picked {
        return Some(id.clone());
    }
    draft
        .main_track()
        .and_then(|track| clip_on(track, time))
        .or_else(|| draft.tracks.iter().find_map(|track| clip_on(track, time)))
}

/// A cut at `at` leaves at least a frame on each side of the clip.
#[must_use]
pub fn can_cut_at(draft: &Draft, segment_id: &SegmentId, at: Micros) -> bool {
    let margin = frame_margin(draft).0;
    draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .find(|segment| &segment.id == segment_id)
        .is_some_and(|segment| {
            let start = segment.target.start.0;
            let end = start.saturating_add(segment.target.duration.0);
            at.0 >= start.saturating_add(margin) && at.0 <= end.saturating_sub(margin)
        })
}

/// Delete the part of the clip under `at` before it (`Start`) or after it
/// (`End`). `None` when no clip is there or a side would keep less than a frame.
#[must_use]
pub fn delete_side(
    draft: &Draft,
    selection: &[SegmentId],
    at: Micros,
    edge: TrimEdge,
) -> Option<Request> {
    let segment_id = clip_under(draft, selection, at)?;
    can_cut_at(draft, &segment_id, at).then_some(Request::Trim {
        segment_id,
        edge,
        time: at,
    })
}

fn covers(segment: &Segment, time: Micros) -> bool {
    let start = segment.target.start.0;
    time.0 >= start && time.0 < start.saturating_add(segment.target.duration.0)
}

fn clip_on(track: &Track, time: Micros) -> Option<SegmentId> {
    track
        .segments
        .iter()
        .find(|segment| covers(segment, time))
        .map(|segment| segment.id.clone())
}

#[cfg(test)]
mod tests {
    use cutlass_model::{TimeRange, TrackFlag, TrackId};

    use super::*;

    fn draft() -> Draft {
        let mut draft = Draft::empty("D");
        draft.tracks[0]
            .segments
            .push(Segment::new("main", "m", TimeRange::new(0, 2_000_000)));
        let mut upper = Track::main(TrackId::new("v2"));
        upper.flag = TrackFlag::SubVideo;
        upper.segments.push(Segment::new(
            "over",
            "m",
            TimeRange::new(500_000, 1_000_000),
        ));
        draft.tracks.push(upper);
        draft
    }

    #[test]
    fn the_selection_wins_then_the_main_track() {
        let draft = draft();
        let at = Micros(700_000);
        assert_eq!(clip_under(&draft, &[], at), Some(SegmentId::new("main")));
        assert_eq!(
            clip_under(&draft, &[SegmentId::new("over")], at),
            Some(SegmentId::new("over"))
        );
        assert_eq!(
            clip_under(&draft, &[SegmentId::new("over")], Micros(1_800_000)),
            Some(SegmentId::new("main"))
        );
        assert_eq!(clip_under(&draft, &[], Micros(5_000_000)), None);
    }

    #[test]
    fn a_cut_keeps_a_frame_on_each_side() {
        let draft = draft();
        let id = SegmentId::new("main");
        assert!(can_cut_at(&draft, &id, Micros(1_000_000)));
        assert!(!can_cut_at(&draft, &id, Micros(0)));
        assert!(!can_cut_at(&draft, &id, Micros(frame_margin(&draft).0 - 1)));
        assert!(!can_cut_at(&draft, &SegmentId::new("z"), Micros(1_000_000)));
    }

    #[test]
    fn delete_side_trims_the_clip_under_the_playhead() {
        let draft = draft();
        assert_eq!(
            delete_side(&draft, &[], Micros(1_000_000), TrimEdge::Start),
            Some(Request::Trim {
                segment_id: SegmentId::new("main"),
                edge: TrimEdge::Start,
                time: Micros(1_000_000),
            })
        );
        assert_eq!(delete_side(&draft, &[], Micros(0), TrimEdge::End), None);
    }
}
