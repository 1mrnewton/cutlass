use cutlass_model::{Draft, Micros, Segment, SegmentId, Track, TrackId, TrackType};
use cutlass_session::Request;

use crate::hit::{BadScale, Pointer, TimelineScale, hit_test};
use crate::lane::{Lane, lanes};
use crate::snap::snap;

/// Why a gesture could not name a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GestureError {
    /// No segment has this id.
    UnknownSegment,
    /// The pixel size or the lane height cannot map the pointer.
    BadScale,
    /// The media duration is zero or negative, so it cannot occupy a lane.
    EmptyDuration,
}

/// What the drag became. A miss and a new lane do not rewrite the draft.
#[derive(Clone, Debug, PartialEq)]
pub enum GestureOutcome {
    /// One request. The session applies it.
    Edit(Request),
    /// The pointer is outside every lane.
    Miss,
    /// The lane under the pointer and the upper lane of this type are taken.
    /// This crate does not insert the track.
    NewLane { kind: TrackType },
}

/// Where a segment can land. The draft's tracks are not changed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Placement {
    Lane { track_id: TrackId },
    NewLane { kind: TrackType },
}

/// Where a library drop lands. The draft's tracks are not changed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DropOutcome {
    /// `start` is the target start, in microseconds, after snap.
    Place { track_id: TrackId, start: Micros },
    /// The lane under the pointer and the upper lane of this type are taken.
    /// `start` is the target start the new lane would use.
    NewLane { kind: TrackType, start: Micros },
    /// The pointer is outside every lane.
    Miss,
}

/// A drop of library media onto the timeline.
#[derive(Clone, Debug, PartialEq)]
pub struct DropGesture {
    pub kind: TrackType,
    pub duration: Micros,
    pub pointer: Pointer,
    pub scale: TimelineScale,
    pub playhead: Micros,
    /// How far, in microseconds, a snap target may sit from the pointer time.
    pub snap_threshold: Micros,
}

/// A drag of a segment that is already on the draft.
#[derive(Clone, Debug, PartialEq)]
pub struct MoveGesture {
    pub segment_id: SegmentId,
    pub pointer: Pointer,
    pub scale: TimelineScale,
    pub playhead: Micros,
    /// How far, in microseconds, a snap target may sit from the pointer time.
    pub snap_threshold: Micros,
}

/// Turn the drag into one session request.
///
/// Placement tries the lane under the pointer, then the nearest upper lane
/// of the same type, then a new lane. Snap may replace the pointer time.
/// Neither step writes the draft.
pub fn move_gesture(draft: &Draft, gesture: &MoveGesture) -> Result<GestureOutcome, GestureError> {
    let located = locate(draft, &gesture.segment_id).ok_or(GestureError::UnknownSegment)?;
    let hit = match hit_test(draft, gesture.scale, gesture.pointer) {
        Ok(Some(hit)) => hit,
        Ok(None) => return Ok(GestureOutcome::Miss),
        Err(BadScale) => return Err(GestureError::BadScale),
    };
    let start = match snap(
        draft,
        hit.time,
        gesture.playhead,
        gesture.snap_threshold,
        Some(&gesture.segment_id),
    ) {
        Some(target) => target.time,
        None => hit.time,
    };
    let rows = lanes(draft);
    let placed = placement(
        draft,
        &rows,
        hit.lane.row,
        &located.track.kind,
        start,
        located.segment.target.duration,
        &gesture.segment_id,
    );
    Ok(match placed {
        Placement::Lane { track_id } => GestureOutcome::Edit(Request::Move {
            segment_id: gesture.segment_id.clone(),
            track_id,
            start,
        }),
        Placement::NewLane { kind } => GestureOutcome::NewLane { kind },
    })
}

/// Turn a library drop into a lane and a start time.
///
/// Placement tries the lane under the pointer, then the nearest upper lane
/// of `kind`, then a new lane. Snap may replace the pointer time. A negative
/// time becomes the start of the timeline. This does not write the draft
/// and does not build the segment.
pub fn drop_gesture(draft: &Draft, gesture: &DropGesture) -> Result<DropOutcome, GestureError> {
    if gesture.duration.0 <= 0 {
        return Err(GestureError::EmptyDuration);
    }
    let hit = match hit_test(draft, gesture.scale, gesture.pointer) {
        Ok(Some(hit)) => hit,
        Ok(None) => return Ok(DropOutcome::Miss),
        Err(BadScale) => return Err(GestureError::BadScale),
    };
    let snapped = snap(
        draft,
        hit.time,
        gesture.playhead,
        gesture.snap_threshold,
        None,
    )
    .map(|target| target.time)
    .unwrap_or(hit.time);
    let start = Micros(snapped.0.max(0));
    let rows = lanes(draft);
    let placed = placement(
        draft,
        &rows,
        hit.lane.row,
        &gesture.kind,
        start,
        gesture.duration,
        &SegmentId::new(""),
    );
    Ok(match placed {
        Placement::Lane { track_id } => DropOutcome::Place { track_id, start },
        Placement::NewLane { kind } => DropOutcome::NewLane { kind, start },
    })
}

/// The lane under `pointer_row`, then the nearest upper lane of `kind`, then a new lane.
///
/// A lane of another type does not count. The segment named by `ignore` does
/// not block the lane it already sits on.
#[must_use]
pub fn placement(
    draft: &Draft,
    rows: &[Lane],
    pointer_row: usize,
    kind: &TrackType,
    start: Micros,
    duration: Micros,
    ignore: &SegmentId,
) -> Placement {
    if let Some(track_id) = open_lane(draft, rows.get(pointer_row), kind, start, duration, ignore) {
        return Placement::Lane { track_id };
    }
    let upper = rows
        .iter()
        .rev()
        .find(|lane| lane.row < pointer_row && &lane.kind == kind);
    if let Some(track_id) = open_lane(draft, upper, kind, start, duration, ignore) {
        return Placement::Lane { track_id };
    }
    Placement::NewLane { kind: kind.clone() }
}

fn open_lane(
    draft: &Draft,
    lane: Option<&Lane>,
    kind: &TrackType,
    start: Micros,
    duration: Micros,
    ignore: &SegmentId,
) -> Option<TrackId> {
    let lane = lane.filter(|lane| &lane.kind == kind)?;
    let track = draft
        .tracks
        .iter()
        .find(|track| track.id == lane.track_id)?;
    if blocked(track, start, duration, ignore) {
        None
    } else {
        Some(lane.track_id.clone())
    }
}

fn blocked(track: &Track, start: Micros, duration: Micros, ignore: &SegmentId) -> bool {
    if duration.0 <= 0 {
        return true;
    }
    let candidate = cutlass_model::TimeRange::new(start.0, duration.0);
    for segment in &track.segments {
        if &segment.id == ignore {
            continue;
        }
        match candidate.overlaps(&segment.target) {
            Ok(false) => {}
            Ok(true) | Err(_) => return true,
        }
    }
    false
}

struct Located<'a> {
    track: &'a Track,
    segment: &'a Segment,
}

fn locate<'a>(draft: &'a Draft, id: &SegmentId) -> Option<Located<'a>> {
    for track in &draft.tracks {
        if let Some(segment) = track.segments.iter().find(|segment| &segment.id == id) {
            return Some(Located { track, segment });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        Draft, Segment, TimeRange, Track, TrackAttribute, TrackFlag, TrackId, TrackType,
    };
    use cutlass_session::{EditError, Request, Session};

    use super::*;

    fn scale() -> TimelineScale {
        TimelineScale {
            lane_height: 40.0,
            pixel_size: 1.0,
        }
    }

    fn gesture(id: &str, x: f64, y: f64, threshold: i64) -> MoveGesture {
        MoveGesture {
            segment_id: SegmentId::new(id),
            pointer: Pointer { x, y },
            scale: scale(),
            playhead: Micros(9_000_000),
            snap_threshold: Micros(threshold),
        }
    }

    fn place(track: &TrackId, id: &str, start: i64, duration: i64) -> Request {
        Request::Place {
            track_id: track.clone(),
            segment: Box::new(Segment::new(id, "media", TimeRange::new(start, duration))),
        }
    }

    #[test]
    fn a_move_becomes_one_request_and_the_draft_matches() {
        let mut draft = Draft::empty("D");
        let track = TrackId::new("main");
        draft.tracks[0].id = track.clone();
        let mut session = Session::open(draft);
        session
            .apply(0, &[place(&track, "a", 0, 1_000_000)])
            .unwrap();

        let outcome = move_gesture(session.draft(), &gesture("a", 2_000_000.0, 8.0, 0)).unwrap();
        let GestureOutcome::Edit(request) = outcome else {
            panic!("the drag ends as one request");
        };
        assert_eq!(
            request,
            Request::Move {
                segment_id: SegmentId::new("a"),
                track_id: track,
                start: Micros(2_000_000),
            }
        );
        session
            .apply(session.revision(), std::slice::from_ref(&request))
            .unwrap();
        assert_eq!(
            session.draft().tracks[0].segments[0].target,
            TimeRange::new(2_000_000, 1_000_000)
        );
        assert_eq!(session.draft().duration, Micros(3_000_000));
        assert_eq!(session.draft().tracks.len(), 1);
    }

    #[test]
    fn a_miss_leaves_the_draft() {
        let mut draft = Draft::empty("D");
        let track = TrackId::new("main");
        draft.tracks[0].id = track.clone();
        let mut session = Session::open(draft);
        session
            .apply(0, &[place(&track, "a", 0, 1_000_000)])
            .unwrap();
        let before = session.draft().clone();
        let outcome = move_gesture(session.draft(), &gesture("a", 100.0, 80.0, 0)).unwrap();
        assert_eq!(outcome, GestureOutcome::Miss);
        assert_eq!(session.draft(), &before);
        assert_eq!(session.revision(), 1);
    }

    #[test]
    fn a_snap_lands_on_an_edge() {
        let mut draft = Draft::empty("D");
        let track = TrackId::new("main");
        draft.tracks[0].id = track.clone();
        let mut session = Session::open(draft);
        session
            .apply(
                0,
                &[
                    place(&track, "a", 0, 1_000_000),
                    place(&track, "b", 3_000_000, 1_000_000),
                ],
            )
            .unwrap();
        let outcome =
            move_gesture(session.draft(), &gesture("b", 1_200_000.0, 8.0, 300_000)).unwrap();
        let GestureOutcome::Edit(request) = outcome else {
            panic!("the drag ends as one request");
        };
        assert_eq!(
            request,
            Request::Move {
                segment_id: SegmentId::new("b"),
                track_id: track,
                start: Micros(1_000_000),
            }
        );
        session
            .apply(session.revision(), std::slice::from_ref(&request))
            .unwrap();
        assert_eq!(
            session.draft().tracks[0].segments[0].target,
            TimeRange::new(0, 1_000_000)
        );
        assert_eq!(
            session.draft().tracks[0].segments[1].target,
            TimeRange::new(1_000_000, 1_000_000)
        );
    }

    #[test]
    fn a_locked_lane_refuses_the_gesture_and_leaves_the_draft() {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        draft.tracks[0].attribute = TrackAttribute(TrackAttribute::LOCK);
        draft.tracks[0]
            .segments
            .push(Segment::new("a", "media", TimeRange::new(0, 1_000_000)));
        let before = draft.clone();
        let mut session = Session::open(draft);
        let outcome = move_gesture(session.draft(), &gesture("a", 2_000_000.0, 8.0, 0)).unwrap();
        let GestureOutcome::Edit(request) = outcome else {
            panic!("the drag still names one request");
        };
        let error = session
            .apply(session.revision(), std::slice::from_ref(&request))
            .unwrap_err();
        assert_eq!(error, EditError::LockedTrack);
        assert_eq!(session.draft(), &before);
        assert_eq!(session.revision(), 0);
    }

    #[test]
    fn placement_tries_the_pointer_then_the_upper_lane_then_a_new_lane() {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        draft.tracks[0]
            .segments
            .push(Segment::new("a", "media", TimeRange::new(0, 1_000_000)));
        draft.tracks[0].segments.push(Segment::new(
            "b",
            "media",
            TimeRange::new(3_000_000, 1_000_000),
        ));
        let mut upper = Track::main("upper");
        upper.flag = TrackFlag::SubVideo;
        upper.segments.push(Segment::new(
            "later",
            "media",
            TimeRange::new(5_000_000, 1_000_000),
        ));
        let mut blocked = Track::main("blocked");
        blocked.flag = TrackFlag::SubVideo;
        blocked
            .segments
            .push(Segment::new("taken", "media", TimeRange::new(0, 1_000_000)));
        // File order is top to bottom: a free overlay, a taken overlay, then main.
        draft.tracks.insert(0, blocked);
        draft.tracks.insert(0, upper);
        let before = draft.clone();
        let rows = lanes(&draft);
        let video = TrackType::Video;
        let dragged = SegmentId::new("b");

        let on_main = placement(
            &draft,
            &rows,
            2,
            &video,
            Micros(4_000_000),
            Micros(1_000_000),
            &dragged,
        );
        assert_eq!(
            on_main,
            Placement::Lane {
                track_id: TrackId::new("main")
            }
        );

        // The pointer lane and the nearest upper lane are both taken at time zero.
        let taken = placement(
            &draft,
            &rows,
            2,
            &video,
            Micros(0),
            Micros(1_000_000),
            &dragged,
        );
        assert_eq!(
            taken,
            Placement::NewLane {
                kind: TrackType::Video
            }
        );

        let outcome = move_gesture(&draft, &gesture("b", 0.0, 90.0, 0)).unwrap();
        assert_eq!(
            outcome,
            GestureOutcome::NewLane {
                kind: TrackType::Video
            }
        );
        assert_eq!(draft, before);
        assert_eq!(draft.tracks.len(), 3);
    }

    #[test]
    fn an_upper_lane_move_is_one_request() {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        draft.tracks[0]
            .segments
            .push(Segment::new("a", "media", TimeRange::new(0, 1_000_000)));
        draft.tracks[0].segments.push(Segment::new(
            "b",
            "media",
            TimeRange::new(3_000_000, 1_000_000),
        ));
        let mut overlay = Track::main("over");
        overlay.flag = TrackFlag::SubVideo;
        overlay.segments.push(Segment::new(
            "later",
            "media",
            TimeRange::new(5_000_000, 1_000_000),
        ));
        draft.tracks.insert(0, overlay);
        let mut session = Session::open(draft);
        let outcome = move_gesture(session.draft(), &gesture("b", 0.0, 50.0, 0)).unwrap();
        let GestureOutcome::Edit(request) = outcome else {
            panic!("the upper lane is one move");
        };
        assert_eq!(
            request,
            Request::Move {
                segment_id: SegmentId::new("b"),
                track_id: TrackId::new("over"),
                start: Micros(0),
            }
        );
        session
            .apply(session.revision(), std::slice::from_ref(&request))
            .unwrap();
        assert_eq!(session.draft().tracks[0].segments.len(), 2);
        assert_eq!(session.draft().tracks[0].segments[0].id.as_str(), "b");
        assert_eq!(
            session.draft().tracks[0].segments[0].target,
            TimeRange::new(0, 1_000_000)
        );
        assert_eq!(session.draft().tracks[1].segments.len(), 1);
        assert_eq!(session.draft().tracks[1].segments[0].id.as_str(), "a");
    }

    #[test]
    fn a_segment_does_not_block_its_own_lane() {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        draft.tracks[0]
            .segments
            .push(Segment::new("a", "media", TimeRange::new(0, 2_000_000)));
        let rows = lanes(&draft);
        let placed = placement(
            &draft,
            &rows,
            0,
            &TrackType::Video,
            Micros(500_000),
            Micros(2_000_000),
            &SegmentId::new("a"),
        );
        assert_eq!(
            placed,
            Placement::Lane {
                track_id: TrackId::new("main")
            }
        );
    }

    #[test]
    fn an_unknown_segment_is_not_a_request() {
        let draft = Draft::empty("D");
        let error = move_gesture(&draft, &gesture("missing", 0.0, 8.0, 0)).unwrap_err();
        assert_eq!(error, GestureError::UnknownSegment);
        assert_eq!(draft.tracks.len(), 1);
    }

    fn drop_at(
        kind: TrackType,
        duration: i64,
        x: f64,
        y: f64,
        playhead: i64,
        threshold: i64,
    ) -> DropGesture {
        DropGesture {
            kind,
            duration: Micros(duration),
            pointer: Pointer { x, y },
            scale: scale(),
            playhead: Micros(playhead),
            snap_threshold: Micros(threshold),
        }
    }

    #[test]
    fn a_drop_on_the_main_lane_names_that_lane_and_the_snapped_time() {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        let outcome = drop_gesture(
            &draft,
            &drop_at(TrackType::Video, 1_000_000, 50.0, 8.0, 0, 100),
        )
        .unwrap();
        assert_eq!(
            outcome,
            DropOutcome::Place {
                track_id: TrackId::new("main"),
                start: Micros(0),
            }
        );
        assert!(draft.tracks[0].segments.is_empty());
    }

    #[test]
    fn a_drop_keeps_a_time_outside_the_snap_window() {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        let outcome = drop_gesture(
            &draft,
            &drop_at(TrackType::Video, 1_000_000, 2_000_000.0, 8.0, 9_000_000, 0),
        )
        .unwrap();
        assert_eq!(
            outcome,
            DropOutcome::Place {
                track_id: TrackId::new("main"),
                start: Micros(2_000_000),
            }
        );
    }

    #[test]
    fn a_drop_before_the_origin_starts_at_zero() {
        let draft = Draft::empty("D");
        let outcome = drop_gesture(
            &draft,
            &drop_at(TrackType::Video, 1_000_000, -10.0, 8.0, 0, 0),
        )
        .unwrap();
        assert_eq!(
            outcome,
            DropOutcome::Place {
                track_id: draft.tracks[0].id.clone(),
                start: Micros(0),
            }
        );
    }

    #[test]
    fn an_audio_drop_on_a_video_lane_asks_for_a_new_lane() {
        let draft = Draft::empty("D");
        let outcome = drop_gesture(
            &draft,
            &drop_at(TrackType::Audio, 1_000_000, 2_000_000.0, 8.0, 0, 0),
        )
        .unwrap();
        assert_eq!(
            outcome,
            DropOutcome::NewLane {
                kind: TrackType::Audio,
                start: Micros(2_000_000),
            }
        );
        assert_eq!(draft.tracks.len(), 1);
    }

    #[test]
    fn a_blocked_video_lane_asks_for_a_new_lane() {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        draft.tracks[0]
            .segments
            .push(Segment::new("a", "media", TimeRange::new(0, 1_000_000)));
        let outcome = drop_gesture(
            &draft,
            &drop_at(TrackType::Video, 1_000_000, 0.0, 8.0, 0, 0),
        )
        .unwrap();
        assert_eq!(
            outcome,
            DropOutcome::NewLane {
                kind: TrackType::Video,
                start: Micros(0),
            }
        );
    }

    #[test]
    fn a_drop_below_the_lanes_is_a_miss() {
        let draft = Draft::empty("D");
        let before = draft.clone();
        let outcome = drop_gesture(
            &draft,
            &drop_at(TrackType::Video, 1_000_000, 10.0, 80.0, 0, 0),
        )
        .unwrap();
        assert_eq!(outcome, DropOutcome::Miss);
        assert_eq!(draft, before);
    }

    #[test]
    fn a_drop_with_no_duration_is_refused() {
        let draft = Draft::empty("D");
        let error =
            drop_gesture(&draft, &drop_at(TrackType::Video, 0, 0.0, 8.0, 0, 0)).unwrap_err();
        assert_eq!(error, GestureError::EmptyDuration);
        assert!(draft.tracks[0].segments.is_empty());
    }
}
