//! Copy and paste of clips. A copy keeps each clip, its row, its gap to the
//! earliest copied clip, and its own markers and beats. A paste places the
//! copies at a time with fresh ids, as one undo step.

use cutlass_model::{
    Draft, MaterialId, Micros, Segment, SegmentId, TimeMarkItem, TimeRange, TrackAttribute,
    TrackId, beat_timeline_time, segment_beats, segment_time_mark, time_mark_items,
    track_holds_beats, user_beats,
};
use cutlass_session::Request;

/// One clip kept for paste.
#[derive(Clone, Debug, PartialEq)]
pub struct CopiedClip {
    pub track_id: TrackId,
    /// Its start after the earliest copied clip's start.
    pub offset: Micros,
    /// The clip without its own marker and beat materials.
    pub segment: Segment,
    /// Offsets from the clip start. A paste stores them on a new material.
    pub marks: Vec<TimeMarkItem>,
    /// Source times. A paste stores them on a new material.
    pub beats: Vec<i64>,
}

/// Why a copy or a paste does nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClipboardMiss {
    /// No selected clip is on the timeline, or nothing was copied.
    Empty,
    /// A clip's row is locked.
    Locked,
    /// A copied clip's row was removed.
    LaneGone,
}

/// Copy the selected clips in row order.
pub fn copy_clips(
    draft: &Draft,
    selection: &[SegmentId],
) -> Result<Vec<CopiedClip>, ClipboardMiss> {
    let mut copies = Vec::new();
    for track in &draft.tracks {
        for segment in &track.segments {
            if !selection.contains(&segment.id) {
                continue;
            }
            if track.attribute.contains(TrackAttribute::LOCK) {
                return Err(ClipboardMiss::Locked);
            }
            let mark_raw = segment_time_mark(&draft.materials, segment);
            let beat_raw = segment_beats(&draft.materials, segment);
            let marks = mark_raw.map(time_mark_items).unwrap_or_default();
            let beats = match beat_raw {
                Some(raw) if track_holds_beats(&track.kind) => user_beats(raw),
                _ => Vec::new(),
            };
            let owned: Vec<&str> = mark_raw
                .iter()
                .chain(beat_raw.iter())
                .map(|raw| raw.id.as_str())
                .collect();
            let mut segment = segment.clone();
            segment
                .extra_material_refs
                .retain(|ref_id| !owned.contains(&ref_id.as_str()));
            copies.push(CopiedClip {
                track_id: track.id.clone(),
                offset: segment.target.start,
                segment,
                marks,
                beats,
            });
        }
    }
    let earliest = copies
        .iter()
        .map(|item| item.offset.0)
        .min()
        .ok_or(ClipboardMiss::Empty)?;
    for item in &mut copies {
        item.offset = Micros(item.offset.0.saturating_sub(earliest));
    }
    Ok(copies)
}

/// The requests that paste `copies` from `at` (time zero at the earliest),
/// and the new clips' ids. `fresh` makes an unused id from a prefix such as
/// `seg`. Markers past a clip's end and beats outside it are dropped.
pub fn paste_requests(
    draft: &Draft,
    copies: &[CopiedClip],
    at: Micros,
    fresh: &mut dyn FnMut(&str) -> String,
) -> Result<(Vec<Request>, Vec<SegmentId>), ClipboardMiss> {
    if copies.is_empty() {
        return Err(ClipboardMiss::Empty);
    }
    for item in copies {
        let track = draft
            .tracks
            .iter()
            .find(|track| track.id == item.track_id)
            .ok_or(ClipboardMiss::LaneGone)?;
        if track.attribute.contains(TrackAttribute::LOCK) {
            return Err(ClipboardMiss::Locked);
        }
    }
    let at = at.0.max(0);
    let mut places = Vec::new();
    let mut marks = Vec::new();
    let mut beats = Vec::new();
    let mut ids = Vec::new();
    for item in copies {
        let id = SegmentId::new(fresh("seg"));
        let start = at.saturating_add(item.offset.0);
        let mut segment = item.segment.clone();
        segment.id = id.clone();
        segment.target = TimeRange::new(start, segment.target.duration.0);
        let duration = segment.target.duration.0;
        for mark in &item.marks {
            if (0..duration).contains(&mark.start.0) {
                let time = Micros(start.saturating_add(mark.start.0));
                marks.push((id.clone(), time, mark.color.clone()));
            }
        }
        for source in &item.beats {
            if beat_timeline_time(&segment, *source).is_some() {
                beats.push((id.clone(), Micros(*source)));
            }
        }
        places.push(Request::Place {
            track_id: item.track_id.clone(),
            segment: Box::new(segment),
        });
        ids.push(id);
    }
    for (segment_id, at, color) in marks {
        places.push(Request::AddTimeMark {
            segment_id: Some(segment_id),
            at,
            color,
            material_id: MaterialId::new(fresh("time-mark")),
        });
    }
    for (segment_id, source) in beats {
        places.push(Request::AddBeat {
            segment_id,
            source,
            material_id: MaterialId::new(fresh("beats")),
        });
    }
    Ok((places, ids))
}

#[cfg(test)]
mod tests {
    use cutlass_session::Session;

    use super::*;

    fn session() -> Session {
        let mut draft = Draft::empty("D");
        draft.config.maintrack_adsorb = false;
        draft.tracks[0].id = TrackId::new("main");
        for (id, start) in [("a", 1_000_000), ("b", 3_000_000)] {
            draft.tracks[0]
                .segments
                .push(Segment::new(id, "m", TimeRange::new(start, 1_000_000)));
        }
        Session::open(draft)
    }

    fn counter() -> impl FnMut(&str) -> String {
        let mut next = 0;
        move |prefix| {
            next += 1;
            format!("{prefix}-{next}")
        }
    }

    #[test]
    fn a_paste_keeps_the_gap_and_lands_as_one_step() {
        let mut session = session();
        let copies =
            copy_clips(session.draft(), &[SegmentId::new("a"), SegmentId::new("b")]).unwrap();
        assert_eq!(copies[1].offset, Micros(2_000_000));
        let (requests, ids) =
            paste_requests(session.draft(), &copies, Micros(10_000_000), &mut counter()).unwrap();
        assert_eq!(ids, vec![SegmentId::new("seg-1"), SegmentId::new("seg-2")]);
        let revision = session.revision();
        session.apply(revision, &requests).unwrap();
        let starts: Vec<i64> = session.draft().tracks[0]
            .segments
            .iter()
            .map(|segment| segment.target.start.0)
            .collect();
        assert!(starts.contains(&10_000_000) && starts.contains(&12_000_000));
        assert_eq!(session.revision(), revision + 1);
    }

    #[test]
    fn nothing_selected_a_locked_row_and_a_gone_row_refuse() {
        let session = session();
        assert_eq!(copy_clips(session.draft(), &[]), Err(ClipboardMiss::Empty));
        let copies = copy_clips(session.draft(), &[SegmentId::new("a")]).unwrap();
        let mut locked = session.draft().clone();
        locked.tracks[0].attribute = TrackAttribute(TrackAttribute::LOCK);
        assert_eq!(
            copy_clips(&locked, &[SegmentId::new("a")]),
            Err(ClipboardMiss::Locked)
        );
        assert_eq!(
            paste_requests(&locked, &copies, Micros(0), &mut counter()).map(|_| ()),
            Err(ClipboardMiss::Locked)
        );
        let mut gone = session.draft().clone();
        gone.tracks[0].id = TrackId::new("other");
        assert_eq!(
            paste_requests(&gone, &copies, Micros(0), &mut counter()).map(|_| ()),
            Err(ClipboardMiss::LaneGone)
        );
    }
}
