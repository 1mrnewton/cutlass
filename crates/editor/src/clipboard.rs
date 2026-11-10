//! Copy and paste of clips. A copy keeps each clip, its row, its gap to the
//! earliest copied clip, and its own markers and beats. A paste places the
//! copies at a time with fresh ids, as one undo step.
//!
//! A pasted clip goes on the row it was copied from when that row is clear
//! where it lands. When a clip is in the way it goes on the nearest row of
//! the same kind above that is clear, and on a new row when none is, the
//! way a dragged clip does. Nothing on the timeline moves to make room.

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

/// Whether `range` runs into a clip already on `track`, or one this paste
/// has put there. A range that cannot be compared counts as taken.
fn taken(
    track: Option<&cutlass_model::Track>,
    track_id: &TrackId,
    range: &TimeRange,
    pasted: &[(TrackId, TimeRange)],
) -> bool {
    let hits = |other: &TimeRange| range.overlaps(other).unwrap_or(true);
    track.is_some_and(|track| track.segments.iter().any(|segment| hits(&segment.target)))
        || pasted
            .iter()
            .any(|(on, other)| on == track_id && hits(other))
}

/// The requests that paste `copies` from `at` (time zero at the earliest),
/// and the new clips' ids. `fresh` makes an unused id from a prefix such as
/// `seg` or `track`. Markers past a clip's end and beats outside it are
/// dropped. A clip whose row is taken where it lands goes on the nearest
/// clear row of its kind above, or on a new row this paste opens.
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
    let rows = crate::lanes_for(draft, true);
    // Where this paste has put a clip so far, and the rows it has opened,
    // each with the row its clips were copied from.
    let mut pasted: Vec<(TrackId, TimeRange)> = Vec::new();
    let mut opened: Vec<(TrackId, TrackId)> = Vec::new();
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
        let range = segment.target.clone();
        let track_of = |id: &TrackId| draft.tracks.iter().find(|track| &track.id == id);
        let clear = |id: &TrackId, pasted: &[(TrackId, TimeRange)]| {
            !taken(track_of(id), id, &range, pasted)
        };
        let own = rows.iter().find(|lane| lane.track_id == item.track_id);
        let above = own.and_then(|own| {
            rows.iter()
                .rev()
                .filter(|lane| lane.row < own.row && lane.kind == own.kind)
                .find(|lane| {
                    track_of(&lane.track_id)
                        .is_some_and(|track| !track.attribute.contains(TrackAttribute::LOCK))
                })
        });
        let track_id = if clear(&item.track_id, &pasted) {
            item.track_id.clone()
        } else if let Some(lane) = above.filter(|lane| clear(&lane.track_id, &pasted)) {
            lane.track_id.clone()
        } else if let Some((_, row)) = opened
            .iter()
            .find(|(from, row)| from == &item.track_id && clear(row, &pasted))
        {
            row.clone()
        } else {
            let kind = track_of(&item.track_id)
                .map(|track| track.kind.clone())
                .ok_or(ClipboardMiss::LaneGone)?;
            let row = TrackId::new(fresh("track"));
            places.push(Request::InsertTrack {
                track_id: row.clone(),
                kind,
            });
            opened.push((item.track_id.clone(), row.clone()));
            row
        };
        pasted.push((track_id.clone(), range));
        places.push(Request::Place {
            track_id,
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
        draft
            .materials
            .videos
            .push(cutlass_model::MaterialVideo::new("m"));
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

    fn rows(session: &Session) -> Vec<(String, Vec<(i64, i64)>)> {
        session
            .draft()
            .tracks
            .iter()
            .map(|track| {
                let mut clips: Vec<(i64, i64)> = track
                    .segments
                    .iter()
                    .map(|segment| (segment.target.start.0, segment.target.duration.0))
                    .collect();
                clips.sort_unstable();
                (track.id.as_str().to_owned(), clips)
            })
            .collect()
    }

    #[test]
    fn a_paste_onto_a_clip_goes_on_a_new_row_and_moves_nothing() {
        for magnet in [false, true] {
            let mut session = session();
            if magnet {
                let revision = session.revision();
                session
                    .apply(revision, &[Request::SetMagnet { on: true }])
                    .unwrap();
            }
            let before = rows(&session);
            let copies = copy_clips(session.draft(), &[SegmentId::new("a")]).unwrap();
            // The playhead is inside the second clip of the same row.
            let at = session.draft().tracks[0].segments[1].target.start.0 + 500_000;
            let mut fresh = counter();
            let (requests, ids) =
                paste_requests(session.draft(), &copies, Micros(at), &mut fresh).unwrap();
            assert!(
                matches!(requests[0], Request::InsertTrack { ref kind, .. }
                    if *kind == session.draft().tracks[0].kind),
                "{requests:?}"
            );
            let revision = session.revision();
            session.apply(revision, &requests).unwrap();
            assert_eq!(session.revision(), revision + 1, "one undo step");
            let after = rows(&session);
            assert_eq!(after.len(), 2, "magnet {magnet}: {after:?}");
            // The new row is above, holding the copy where the playhead is.
            assert_eq!(after[0].1, vec![(at, 1_000_000)]);
            assert_eq!(after[1], before[0], "the row pasted over is as it was");
            assert_eq!(ids.len(), 1);

            // Pasted there again, the row just opened is taken too, so
            // another opens; pasted where it is clear, the copy goes home.
            let (again, _) =
                paste_requests(session.draft(), &copies, Micros(at), &mut fresh).unwrap();
            assert!(matches!(again[0], Request::InsertTrack { .. }));
            let (clear, _) =
                paste_requests(session.draft(), &copies, Micros(20_000_000), &mut fresh).unwrap();
            assert!(
                matches!(&clear[0], Request::Place { track_id, .. } if track_id.as_str() == "main"),
                "{clear:?}"
            );
        }
    }

    #[test]
    fn a_paste_takes_the_clear_row_above_before_opening_one() {
        let mut session = session();
        let revision = session.revision();
        let kind = session.draft().tracks[0].kind.clone();
        session
            .apply(
                revision,
                &[
                    Request::InsertTrack {
                        track_id: TrackId::new("upper"),
                        kind: kind.clone(),
                    },
                    Request::InsertTrack {
                        track_id: TrackId::new("sound"),
                        kind: cutlass_model::TrackType::Audio,
                    },
                ],
            )
            .unwrap();
        let copies =
            copy_clips(session.draft(), &[SegmentId::new("a"), SegmentId::new("b")]).unwrap();
        // Both copies land on clips of their own row: a at 1s, b at 3s.
        let mut fresh = counter();
        let (requests, _) =
            paste_requests(session.draft(), &copies, Micros(1_200_000), &mut fresh).unwrap();
        let on: Vec<&str> = requests
            .iter()
            .filter_map(|request| match request {
                Request::Place { track_id, .. } => Some(track_id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            on,
            ["upper", "upper"],
            "a row of another kind is passed over"
        );
        assert!(
            !requests
                .iter()
                .any(|request| matches!(request, Request::InsertTrack { .. }))
        );
        let revision = session.revision();
        session.apply(revision, &requests).unwrap();

        // The row above now taken there, and locked besides, a new one opens
        // and both copies share it.
        let mut draft = session.draft().clone();
        let upper = draft
            .tracks
            .iter_mut()
            .find(|track| track.id.as_str() == "upper")
            .unwrap();
        upper.segments.clear();
        upper.attribute = TrackAttribute(TrackAttribute::LOCK);
        let (requests, _) = paste_requests(&draft, &copies, Micros(1_200_000), &mut fresh).unwrap();
        let opened: Vec<&TrackId> = requests
            .iter()
            .filter_map(|request| match request {
                Request::InsertTrack { track_id, .. } => Some(track_id),
                _ => None,
            })
            .collect();
        assert_eq!(opened.len(), 1, "{requests:?}");
        assert!(requests.iter().all(|request| match request {
            Request::Place { track_id, .. } => track_id == opened[0],
            _ => true,
        }));
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
