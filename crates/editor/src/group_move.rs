//! A body drag of one clip, or of every selected clip with it.
//!
//! One clip lands as `move_gesture` places it. When the dragged clip is one
//! of several selected clips, each selected clip keeps its row and its gap
//! and moves by the same amount. The planner does not write the draft.

use cutlass_model::{Draft, Micros, SegmentId, TrackId, TrackType};
use cutlass_session::{Request, lane_fits, new_lane_index};

use crate::gesture::{GestureError, GestureOutcome, MoveGesture, move_gesture};
use crate::hit::{BadScale, Pointer, hit_test, time_at};
use crate::lane::{Lane, lanes_for};
use crate::snap::snap_span;

/// A lane a drag opens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NewLane {
    pub kind: TrackType,
    /// Where it opens in the track list, which is the row order.
    pub index: usize,
}

/// Where one clip of a drag lands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClipMove {
    pub segment_id: SegmentId,
    pub track_id: TrackId,
    /// The target start after snap, in microseconds.
    pub start: Micros,
    pub duration: Micros,
    /// Where the snap line sits when a target, not the pointer, set `start`:
    /// the clip's start or its end. Only the dragged clip snaps.
    pub snap_line: Option<Micros>,
    /// `track_id` is this new lane. The requests insert it first.
    pub new_lane: Option<NewLane>,
}

impl ClipMove {
    #[must_use]
    pub fn request(&self) -> Request {
        Request::Move {
            segment_id: self.segment_id.clone(),
            track_id: self.track_id.clone(),
            start: self.start,
        }
    }
}

/// What a body drag became. A miss and a new lane do not rewrite the draft.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MovePlan {
    /// The dragged clip first, then the selected clips that follow it.
    Moves(Vec<ClipMove>),
    /// The pointer is outside every lane.
    Miss,
    /// The dragged clip opens `lane` and lands at `start`: the drop was
    /// between lanes, or the lane under the pointer and the upper lane of
    /// its type are taken.
    NewLane {
        lane: NewLane,
        start: Micros,
        duration: Micros,
        snap_line: Option<Micros>,
    },
}

impl MovePlan {
    /// The moves this plan makes. A new lane becomes `new_track`, inserted by
    /// `move_requests` ahead of the move. A miss makes none.
    #[must_use]
    pub fn into_moves(self, segment_id: &SegmentId, new_track: TrackId) -> Vec<ClipMove> {
        match self {
            Self::Moves(moves) => moves,
            Self::Miss => Vec::new(),
            Self::NewLane {
                lane,
                start,
                duration,
                snap_line,
            } => vec![ClipMove {
                segment_id: segment_id.clone(),
                track_id: new_track,
                start,
                duration,
                snap_line,
                new_lane: Some(lane),
            }],
        }
    }
}

/// Plan a body drag of `gesture.segment_id`.
///
/// When that clip is selected with at least one other clip, the group moves:
/// the dragged clip's start is the pointer time, pulled to a snap, and every
/// selected clip keeps its row and moves by the same amount. A clip that
/// would pass time zero stops the whole group there, so the gaps stay.
/// Otherwise the clip lands as `move_gesture` places it.
pub fn plan_move(
    draft: &Draft,
    gesture: &MoveGesture,
    selection: &[SegmentId],
) -> Result<MovePlan, GestureError> {
    let dragged = &gesture.segment_id;
    let (track_id, origin, duration) = clip(draft, dragged).ok_or(GestureError::UnknownSegment)?;
    let (raw, pointer_row) = match hit_test(draft, gesture.scale, gesture.pointer) {
        Ok(Some(hit)) => (hit.time, hit.lane),
        Ok(None) => return Ok(MovePlan::Miss),
        Err(BadScale) => return Err(GestureError::BadScale),
    };
    let snapped = snap_span(
        draft,
        raw,
        duration,
        gesture.playhead,
        gesture.snap_threshold,
        Some(dragged),
    );
    if selection.contains(dragged) && selection.len() > 1 {
        let start = Micros(snapped.map_or(raw, |(start, _)| start).0.max(0));
        let lead = ClipMove {
            segment_id: dragged.clone(),
            track_id,
            start,
            duration,
            snap_line: line_for(snapped, start),
            new_lane: None,
        };
        return Ok(MovePlan::Moves(group(draft, lead, origin, selection)));
    }
    match move_gesture(draft, gesture)? {
        GestureOutcome::Edit(Request::Move {
            segment_id,
            track_id,
            start,
        }) => Ok(MovePlan::Moves(vec![ClipMove {
            segment_id,
            track_id,
            start,
            duration,
            snap_line: line_for(snapped, start),
            new_lane: None,
        }])),
        GestureOutcome::Edit(_) => Err(GestureError::UnknownSegment),
        GestureOutcome::Miss => Ok(MovePlan::Miss),
        GestureOutcome::NewLane { kind, start } => {
            // The lane opens just above the pointer's row when the kind may sit
            // there, else where the session opens a lane of that kind.
            let index = if lane_fits(&draft.tracks, &kind, pointer_row.track_index) {
                pointer_row.track_index
            } else {
                new_lane_index(&draft.tracks, &kind)
            };
            Ok(MovePlan::NewLane {
                lane: NewLane { kind, index },
                start,
                duration,
                snap_line: line_for(snapped, start),
            })
        }
    }
}

/// The snap line, when the planned start is the one the snap chose.
fn line_for(snapped: Option<(Micros, Micros)>, start: Micros) -> Option<Micros> {
    snapped
        .filter(|(snapped_start, _)| *snapped_start == start)
        .map(|(_, line)| line)
}

/// Plan a body drag dropped between two rows, or past the first or last.
///
/// `boundary` counts the rows above the drop: zero is above the top row and
/// the row count is under the bottom one. When the dragged clip's kind may
/// sit there, it opens a lane at that spot and lands at the pointer time,
/// pulled to a snap. A selected group keeps its rows. A clip alone on a lane
/// other than main next to the spot stays on its lane, and a spot outside its kind's block
/// falls back to the nearest row; both plan as `plan_move` would there.
pub fn plan_move_between(
    draft: &Draft,
    gesture: &MoveGesture,
    boundary: usize,
    selection: &[SegmentId],
) -> Result<MovePlan, GestureError> {
    let dragged = &gesture.segment_id;
    let (track_id, _, duration) = clip(draft, dragged).ok_or(GestureError::UnknownSegment)?;
    let rows = lanes_for(draft, gesture.scale.empty_lanes);
    let group = selection.contains(dragged) && selection.len() > 1;
    let nearest = boundary
        .min(rows.len())
        .saturating_sub(usize::from(boundary >= rows.len()));
    let on_row = MoveGesture {
        pointer: Pointer {
            x: gesture.pointer.x,
            y: row_middle(gesture.scale.lane_height, nearest),
        },
        ..gesture.clone()
    };
    let index = slot_index(draft, &rows, boundary);
    let kind = draft
        .tracks
        .iter()
        .find(|track| track.id == track_id)
        .map(|track| track.kind.clone())
        .ok_or(GestureError::UnknownSegment)?;
    // Main stays when empty, so a clip leaving it for a new lane is a real move.
    let alone_beside = draft.tracks.iter().enumerate().any(|(at, track)| {
        track.id == track_id
            && !track.is_main()
            && track.segments.len() == 1
            && (at == index || at.saturating_add(1) == index)
    });
    if group || rows.is_empty() || alone_beside || !lane_fits(&draft.tracks, &kind, index) {
        return plan_move(draft, &on_row, selection);
    }
    let raw = time_at(gesture.scale.pixel_size, gesture.pointer.x).ok_or(GestureError::BadScale)?;
    let snapped = snap_span(
        draft,
        raw,
        duration,
        gesture.playhead,
        gesture.snap_threshold,
        Some(dragged),
    );
    let start = Micros(snapped.map_or(raw, |(start, _)| start).0.max(0));
    Ok(MovePlan::NewLane {
        lane: NewLane { kind, index },
        start,
        duration,
        snap_line: line_for(snapped, start),
    })
}

/// The track index a lane opened at `boundary` takes.
fn slot_index(draft: &Draft, rows: &[Lane], boundary: usize) -> usize {
    match rows.get(boundary) {
        Some(row) => row.track_index,
        None => rows
            .last()
            .map_or(draft.tracks.len(), |row| row.track_index.saturating_add(1)),
    }
}

fn row_middle(lane_height: f64, row: usize) -> f64 {
    f64::from(u32::try_from(row).unwrap_or(u32::MAX)) * lane_height + lane_height / 2.0
}

/// The requests for a planned drag, as one undo step.
///
/// A new lane a move names is inserted first. When the group heads right the
/// later clips move first, so a clip does not land on a selected clip that
/// has not moved yet. The session takes a lane the drag leaves empty. Empty
/// when every clip is already where it would land.
#[must_use]
pub fn move_requests(draft: &Draft, moves: &[ClipMove]) -> Vec<Request> {
    let current = |plan: &ClipMove| {
        draft.tracks.iter().any(|track| {
            track.id == plan.track_id
                && track.segments.iter().any(|segment| {
                    segment.id == plan.segment_id && segment.target.start == plan.start
                })
        })
    };
    if moves.iter().all(current) {
        return Vec::new();
    }
    let rightward = moves
        .iter()
        .any(|plan| clip(draft, &plan.segment_id).is_some_and(|(_, start, _)| plan.start > start));
    let mut ordered: Vec<&ClipMove> = moves.iter().collect();
    if rightward {
        ordered.sort_by_key(|plan| std::cmp::Reverse(plan.start));
    } else {
        ordered.sort_by_key(|plan| plan.start);
    }
    let lanes = moves.iter().filter_map(|plan| {
        plan.new_lane.as_ref().map(|lane| Request::InsertTrackAt {
            track_id: plan.track_id.clone(),
            kind: lane.kind.clone(),
            index: lane.index,
        })
    });
    lanes
        .chain(ordered.into_iter().map(ClipMove::request))
        .collect()
}

/// Every selected clip on the timeline, shifted by the lead's delta. The lead
/// comes first. Fewer than two selected clips on the timeline leaves the lead
/// alone.
fn group(draft: &Draft, lead: ClipMove, origin: Micros, selection: &[SegmentId]) -> Vec<ClipMove> {
    let mut members = Vec::new();
    let mut min_start = origin.0;
    for track in &draft.tracks {
        for segment in &track.segments {
            if !selection.contains(&segment.id) {
                continue;
            }
            min_start = min_start.min(segment.target.start.0);
            members.push(ClipMove {
                segment_id: segment.id.clone(),
                track_id: track.id.clone(),
                start: segment.target.start,
                duration: segment.target.duration,
                snap_line: None,
                new_lane: None,
            });
        }
    }
    if members.len() < 2 {
        return vec![lead];
    }
    let mut delta = lead.start.0.saturating_sub(origin.0);
    if min_start.saturating_add(delta) < 0 {
        delta = min_start.saturating_neg();
    }
    for member in &mut members {
        member.start = Micros(member.start.0.saturating_add(delta));
        if member.segment_id == lead.segment_id {
            member.snap_line = lead.snap_line.filter(|_| member.start == lead.start);
        }
    }
    members.sort_by_key(|member| member.segment_id != lead.segment_id);
    members
}

fn clip(draft: &Draft, id: &SegmentId) -> Option<(TrackId, Micros, Micros)> {
    draft.tracks.iter().find_map(|track| {
        track
            .segments
            .iter()
            .find(|segment| &segment.id == id)
            .map(|segment| {
                (
                    track.id.clone(),
                    segment.target.start,
                    segment.target.duration,
                )
            })
    })
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Segment, TimeRange};
    use cutlass_session::Session;

    use super::*;
    use crate::hit::{Pointer, TimelineScale};

    fn gesture(id: &str, x: f64, threshold: i64) -> MoveGesture {
        MoveGesture {
            segment_id: SegmentId::new(id),
            pointer: Pointer { x, y: 8.0 },
            scale: TimelineScale {
                lane_height: 40.0,
                pixel_size: 1.0,
                empty_lanes: false,
            },
            playhead: Micros(50_000_000),
            snap_threshold: Micros(threshold),
        }
    }

    fn session(clips: &[(&str, i64, i64)]) -> Session {
        let mut draft = Draft::empty("D");
        draft.config.maintrack_adsorb = false;
        let track = TrackId::new("main");
        draft.tracks[0].id = track.clone();
        let mut session = Session::open(draft);
        let places: Vec<Request> = clips
            .iter()
            .map(|(id, start, duration)| Request::Place {
                track_id: track.clone(),
                segment: Box::new(Segment::new(
                    *id,
                    "media",
                    TimeRange::new(*start, *duration),
                )),
            })
            .collect();
        session.apply(0, &places).unwrap();
        session
    }

    fn ids(names: &[&str]) -> Vec<SegmentId> {
        names.iter().map(|name| SegmentId::new(*name)).collect()
    }

    fn moves(plan: MovePlan) -> Vec<ClipMove> {
        let MovePlan::Moves(moves) = plan else {
            panic!("the drag lands on a lane");
        };
        moves
    }

    fn starts(session: &Session) -> Vec<(String, i64)> {
        session.draft().tracks[0]
            .segments
            .iter()
            .map(|segment| (segment.id.as_str().to_owned(), segment.target.start.0))
            .collect()
    }

    #[test]
    fn one_clip_lands_where_the_gesture_places_it() {
        let session = session(&[("a", 0, 1_000_000)]);
        let plan = plan_move(session.draft(), &gesture("a", 2_000_000.0, 0), &[]).unwrap();
        assert_eq!(
            moves(plan),
            vec![ClipMove {
                segment_id: SegmentId::new("a"),
                track_id: TrackId::new("main"),
                start: Micros(2_000_000),
                duration: Micros(1_000_000),
                snap_line: None,
                new_lane: None,
            }]
        );
    }

    #[test]
    fn a_drag_onto_a_taken_lane_opens_a_new_lane_in_one_undo_step() {
        let mut session = session(&[("a", 0, 1_000_000), ("b", 2_000_000, 1_000_000)]);
        let before = session.draft().clone();
        let plan = plan_move(session.draft(), &gesture("b", 0.0, 0), &[]).unwrap();
        assert_eq!(
            plan,
            MovePlan::NewLane {
                lane: NewLane {
                    kind: TrackType::Video,
                    index: 0,
                },
                start: Micros(0),
                duration: Micros(1_000_000),
                snap_line: Some(Micros(0)),
            }
        );
        let lane = TrackId::new("lane");
        let moves = plan.into_moves(&SegmentId::new("b"), lane.clone());
        let requests = move_requests(session.draft(), &moves);
        assert_eq!(
            requests,
            vec![
                Request::InsertTrackAt {
                    track_id: lane.clone(),
                    kind: TrackType::Video,
                    index: 0,
                },
                Request::Move {
                    segment_id: SegmentId::new("b"),
                    track_id: lane.clone(),
                    start: Micros(0),
                },
            ]
        );
        let revision = session.revision();
        session.apply(revision, &requests).unwrap();
        let draft = session.draft();
        let opened = draft.tracks.iter().find(|track| track.id == lane).unwrap();
        assert_eq!(opened.segments.len(), 1);
        assert_eq!(opened.segments[0].id, SegmentId::new("b"));
        assert_eq!(opened.segments[0].target.start, Micros(0));
        let main = draft
            .tracks
            .iter()
            .find(|track| track.id == TrackId::new("main"))
            .unwrap();
        assert_eq!(main.segments.len(), 1);
        assert_eq!(main.segments[0].id, SegmentId::new("a"));

        session.undo().unwrap();
        assert_eq!(session.draft().tracks, before.tracks);
    }

    fn with_sound(session: &mut Session) {
        let revision = session.revision();
        session
            .apply(
                revision,
                &[
                    Request::InsertTrack {
                        track_id: TrackId::new("sound"),
                        kind: TrackType::Audio,
                    },
                    Request::Place {
                        track_id: TrackId::new("sound"),
                        segment: Box::new(Segment::new("s", "song", TimeRange::new(0, 500_000))),
                    },
                ],
            )
            .unwrap();
    }

    #[test]
    fn a_drop_between_lanes_opens_a_lane_there() {
        let mut session = session(&[("a", 0, 1_000_000), ("b", 2_000_000, 1_000_000)]);
        with_sound(&mut session);
        let plan =
            plan_move_between(session.draft(), &gesture("b", 4_000_000.0, 0), 0, &[]).unwrap();
        assert_eq!(
            plan,
            MovePlan::NewLane {
                lane: NewLane {
                    kind: TrackType::Video,
                    index: 0,
                },
                start: Micros(4_000_000),
                duration: Micros(1_000_000),
                snap_line: None,
            }
        );
        let moves = plan.into_moves(&SegmentId::new("b"), TrackId::new("lane"));
        let revision = session.revision();
        session
            .apply(revision, &move_requests(session.draft(), &moves))
            .unwrap();
        let ids: Vec<&str> = session
            .draft()
            .tracks
            .iter()
            .map(|track| track.id.as_str())
            .collect();
        assert_eq!(ids, ["lane", "main", "sound"]);
        assert_eq!(
            session.draft().tracks[0].segments[0].id,
            SegmentId::new("b")
        );
    }

    #[test]
    fn a_drop_between_lanes_keeps_to_the_kinds_block() {
        let mut session = session(&[("a", 0, 1_000_000)]);
        with_sound(&mut session);
        // Under the sound lane: a picture may not sit under main, so it plans on
        // the sound row, which sends it up to the main lane it has room on.
        let plan =
            plan_move_between(session.draft(), &gesture("a", 3_000_000.0, 0), 2, &[]).unwrap();
        assert_eq!(
            moves(plan),
            vec![ClipMove {
                segment_id: SegmentId::new("a"),
                track_id: TrackId::new("main"),
                start: Micros(3_000_000),
                duration: Micros(1_000_000),
                snap_line: None,
                new_lane: None,
            }]
        );
        // A sound alone on its lane, dropped just under it, stays on that lane.
        let plan =
            plan_move_between(session.draft(), &gesture("s", 3_000_000.0, 0), 2, &[]).unwrap();
        assert_eq!(
            moves(plan),
            vec![ClipMove {
                segment_id: SegmentId::new("s"),
                track_id: TrackId::new("sound"),
                start: Micros(3_000_000),
                duration: Micros(500_000),
                snap_line: None,
                new_lane: None,
            }]
        );
    }

    #[test]
    fn the_only_main_clip_dropped_above_main_opens_an_overlay() {
        let session = session(&[("a", 0, 1_000_000)]);
        let mut draft = session.draft().clone();
        draft.config.maintrack_adsorb = true;
        let plan = plan_move_between(&draft, &gesture("a", 0.0, 0), 0, &[]).unwrap();
        assert_eq!(
            plan,
            MovePlan::NewLane {
                lane: NewLane {
                    kind: TrackType::Video,
                    index: 0,
                },
                start: Micros(0),
                duration: Micros(1_000_000),
                snap_line: Some(Micros(0)),
            }
        );
    }

    #[test]
    fn a_selected_group_dropped_between_lanes_keeps_its_rows() {
        let session = session(&[("a", 0, 1_000_000), ("b", 2_000_000, 1_000_000)]);
        let plan = plan_move_between(
            session.draft(),
            &gesture("a", 500_000.0, 0),
            0,
            &ids(&["a", "b"]),
        )
        .unwrap();
        let moves = moves(plan);
        assert_eq!(moves.len(), 2);
        assert!(
            moves
                .iter()
                .all(|plan| plan.new_lane.is_none() && plan.track_id == TrackId::new("main"))
        );
    }

    #[test]
    fn a_drag_that_empties_a_lane_takes_the_lane_in_the_same_step() {
        let mut session = session(&[("a", 0, 1_000_000)]);
        with_sound(&mut session);
        let revision = session.revision();
        session
            .apply(
                revision,
                &[
                    Request::InsertTrack {
                        track_id: TrackId::new("more"),
                        kind: TrackType::Audio,
                    },
                    Request::Place {
                        track_id: TrackId::new("more"),
                        segment: Box::new(Segment::new("t", "song", TimeRange::new(0, 500_000))),
                    },
                ],
            )
            .unwrap();
        let before = session.draft().tracks.clone();
        let onto_sound = ClipMove {
            segment_id: SegmentId::new("t"),
            track_id: TrackId::new("sound"),
            start: Micros(2_000_000),
            duration: Micros(500_000),
            snap_line: None,
            new_lane: None,
        };
        let requests = move_requests(session.draft(), &[onto_sound]);
        let revision = session.revision();
        session.apply(revision, &requests).unwrap();
        let ids: Vec<&str> = session
            .draft()
            .tracks
            .iter()
            .map(|track| track.id.as_str())
            .collect();
        assert_eq!(ids, ["main", "sound"]);
        session.undo().unwrap();
        assert_eq!(session.draft().tracks, before);
    }

    #[test]
    fn a_miss_plans_no_moves() {
        assert!(
            MovePlan::Miss
                .into_moves(&SegmentId::new("a"), TrackId::new("lane"))
                .is_empty()
        );
    }

    #[test]
    fn a_selected_group_keeps_its_gaps_and_the_dragged_clip_comes_first() {
        let session = session(&[("a", 0, 1_000_000), ("b", 2_000_000, 1_000_000)]);
        let plan = plan_move(
            session.draft(),
            &gesture("b", 5_000_000.0, 0),
            &ids(&["a", "b"]),
        )
        .unwrap();
        let moves = moves(plan);
        let landed: Vec<_> = moves
            .iter()
            .map(|plan| (plan.segment_id.as_str(), plan.start.0))
            .collect();
        assert_eq!(landed, vec![("b", 5_000_000), ("a", 3_000_000)]);
    }

    #[test]
    fn the_group_stops_at_time_zero() {
        let session = session(&[("a", 1_000_000, 1_000_000), ("b", 4_000_000, 1_000_000)]);
        let plan = plan_move(session.draft(), &gesture("b", 0.0, 0), &ids(&["a", "b"])).unwrap();
        let moves = moves(plan);
        let landed: Vec<_> = moves
            .iter()
            .map(|plan| (plan.segment_id.as_str(), plan.start.0))
            .collect();
        assert_eq!(landed, vec![("b", 3_000_000), ("a", 0)]);
    }

    #[test]
    fn only_the_dragged_clip_reports_a_snap() {
        let session = session(&[
            ("a", 0, 1_000_000),
            ("b", 3_000_000, 1_000_000),
            ("c", 6_000_000, 1_000_000),
        ]);
        let plan = plan_move(
            session.draft(),
            &gesture("c", 4_100_000.0, 300_000),
            &ids(&["b", "c"]),
        )
        .unwrap();
        let moves = moves(plan);
        assert_eq!(moves[0].segment_id, SegmentId::new("c"));
        assert_eq!(moves[0].start, Micros(4_000_000));
        assert!(moves[0].snap_line.is_some());
        assert!(moves.iter().skip(1).all(|plan| plan.snap_line.is_none()));
    }

    #[test]
    fn a_group_heading_right_applies_as_one_step() {
        let mut session = session(&[("a", 0, 1_000_000), ("b", 1_000_000, 1_000_000)]);
        let plan = plan_move(
            session.draft(),
            &gesture("a", 500_000.0, 0),
            &ids(&["a", "b"]),
        )
        .unwrap();
        let requests = move_requests(session.draft(), &moves(plan));
        assert_eq!(requests.len(), 2);
        let revision = session.revision();
        session.apply(revision, &requests).unwrap();
        assert_eq!(
            starts(&session),
            vec![("a".to_owned(), 500_000), ("b".to_owned(), 1_500_000)]
        );
        assert_eq!(session.revision(), revision + 1);
    }

    #[test]
    fn a_group_heading_left_applies_as_one_step() {
        let mut session = session(&[("a", 2_000_000, 1_000_000), ("b", 3_000_000, 1_000_000)]);
        let plan = plan_move(
            session.draft(),
            &gesture("b", 2_500_000.0, 0),
            &ids(&["a", "b"]),
        )
        .unwrap();
        let requests = move_requests(session.draft(), &moves(plan));
        session.apply(session.revision(), &requests).unwrap();
        assert_eq!(
            starts(&session),
            vec![("a".to_owned(), 1_500_000), ("b".to_owned(), 2_500_000)]
        );
    }

    #[test]
    fn a_drag_back_to_the_same_place_makes_no_request() {
        let session = session(&[("a", 1_000_000, 1_000_000)]);
        let plan = plan_move(session.draft(), &gesture("a", 1_000_000.0, 0), &[]).unwrap();
        assert!(move_requests(session.draft(), &moves(plan)).is_empty());
    }

    #[test]
    fn a_miss_and_an_unknown_clip_name_no_moves() {
        let session = session(&[("a", 0, 1_000_000)]);
        let mut below = gesture("a", 100.0, 0);
        below.pointer.y = 80.0;
        assert_eq!(
            plan_move(session.draft(), &below, &[]).unwrap(),
            MovePlan::Miss
        );
        assert_eq!(
            plan_move(session.draft(), &gesture("z", 0.0, 0), &[]),
            Err(GestureError::UnknownSegment)
        );
    }
}
