use cutlass_model::{Draft, LinkageSettings};

use crate::edit::{EditError, apply_all};
use crate::history::History;
use crate::request::Request;

/// The open draft, the revision a write must name, and one undo stack.
#[derive(Clone, Debug)]
pub struct Session {
    draft: Draft,
    revision: u64,
    history: History,
    linkage: LinkageSettings,
}

impl Session {
    /// Revision starts at 0. The first apply must send 0.
    #[must_use]
    pub fn open(draft: Draft) -> Self {
        Self {
            draft,
            revision: 0,
            history: History::default(),
            linkage: LinkageSettings::default(),
        }
    }

    /// Follow and cut switches. A new draft starts with all of them on.
    #[must_use]
    pub fn linkage(&self) -> LinkageSettings {
        self.linkage
    }

    /// Replace the switches. The open draft and its pairs stay as they are.
    pub fn set_linkage(&mut self, linkage: LinkageSettings) {
        self.linkage = linkage;
    }

    #[must_use]
    pub fn draft(&self) -> &Draft {
        &self.draft
    }

    #[must_use]
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Apply every request, or leave the draft untouched.
    ///
    /// An empty list does not push history and does not change the revision.
    /// A mismatch returns before any request runs.
    pub fn apply(&mut self, expected: u64, requests: &[Request]) -> Result<u64, EditError> {
        if expected != self.revision {
            return Err(EditError::RevisionMismatch {
                expected,
                actual: self.revision,
            });
        }
        if requests.is_empty() {
            return Ok(self.revision);
        }
        let mut next = self.draft.clone();
        apply_all(&mut next, requests, &self.linkage)?;
        self.history
            .push_undo(std::mem::replace(&mut self.draft, next));
        self.history.clear_redo();
        self.revision = self.revision.saturating_add(1);
        Ok(self.revision)
    }

    /// Restore the draft from before the last apply. The revision still advances.
    pub fn undo(&mut self) -> Result<u64, EditError> {
        let previous = self.history.pop_undo().ok_or(EditError::EmptyUndo)?;
        let current = std::mem::replace(&mut self.draft, previous);
        self.history.push_redo(current);
        self.revision = self.revision.saturating_add(1);
        Ok(self.revision)
    }

    /// Reapply the draft that undo removed. The revision advances again.
    pub fn redo(&mut self) -> Result<u64, EditError> {
        let next = self.history.pop_redo().ok_or(EditError::EmptyRedo)?;
        let current = std::mem::replace(&mut self.draft, next);
        self.history.push_undo(current);
        self.revision = self.revision.saturating_add(1);
        Ok(self.revision)
    }

    /// Snapshot the open draft. `apply` still pushes undo on its own.
    pub fn begin(&mut self) -> Result<(), EditError> {
        self.history.begin(&self.draft)
    }

    /// Open a recording, or mark the transaction `begin` already opened.
    pub fn record(&mut self) -> Result<(), EditError> {
        self.history.record(&self.draft)
    }

    #[must_use]
    pub fn is_recording(&self) -> bool {
        self.history.is_recording()
    }

    /// Close the transaction. A changed draft is one recorded step, not a second undo push.
    pub fn commit(&mut self) -> Result<(), EditError> {
        self.history.commit(&self.draft)
    }

    /// Fold the open draft into the last recorded step.
    pub fn amend(&mut self) -> Result<(), EditError> {
        self.history.amend(&self.draft)
    }

    /// Put the open draft back where it was at [`Self::begin`] or [`Self::record`].
    pub fn rollback(&mut self) -> Result<(), EditError> {
        self.draft = self.history.rollback()?;
        Ok(())
    }

    #[must_use]
    pub fn recorded_steps(&self) -> usize {
        self.history.recorded_steps()
    }

    #[must_use]
    pub fn recorded_before(&self) -> Option<&Draft> {
        self.history.recorded_before()
    }

    #[must_use]
    pub fn recorded_after(&self) -> Option<&Draft> {
        self.history.recorded_after()
    }
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        ConstantSpeedError, Draft, MaterialAudioFade, MaterialId, MaterialSpeed,
        MaterialTransition, Micros, Segment, SpeedMode, SpeedPoint, TimeRange, Track,
        TrackAttribute, TrackId,
    };

    use super::*;
    use crate::request::{Request, TrimEdge};

    fn session() -> (Session, TrackId) {
        let mut draft = Draft::empty("D");
        let track_id = TrackId::new("main");
        draft.tracks[0].id = track_id.clone();
        (Session::open(draft), track_id)
    }

    fn place(track_id: &TrackId, id: &str, start: i64, duration: i64) -> Request {
        Request::Place {
            track_id: track_id.clone(),
            segment: Box::new(Segment::new(id, "media", TimeRange::new(start, duration))),
        }
    }

    #[test]
    fn a_place_is_one_undo_and_sets_the_duration() {
        let (mut session, track) = session();
        let revision = session
            .apply(0, &[place(&track, "a", 0, 1_000_000)])
            .unwrap();
        assert_eq!(revision, 1);
        assert_eq!(session.draft().duration, Micros(1_000_000));
        assert_eq!(session.draft().tracks[0].segments.len(), 1);
        session.undo().unwrap();
        assert_eq!(session.draft().duration, Micros(0));
        assert!(session.draft().tracks[0].segments.is_empty());
        assert_eq!(session.revision(), 2);
    }

    #[test]
    fn two_places_in_one_apply_undo_together() {
        let (mut session, track) = session();
        session
            .apply(
                0,
                &[
                    place(&track, "a", 0, 1_000_000),
                    place(&track, "b", 1_000_000, 500_000),
                ],
            )
            .unwrap();
        assert_eq!(session.draft().tracks[0].segments.len(), 2);
        session.undo().unwrap();
        assert!(session.draft().tracks[0].segments.is_empty());
    }

    #[test]
    fn a_failed_request_keeps_the_earlier_requests_off_the_draft() {
        let (mut session, track) = session();
        let error = session
            .apply(
                0,
                &[
                    place(&track, "a", 0, 1_000_000),
                    place(&track, "b", 500_000, 500_000),
                ],
            )
            .unwrap_err();
        assert_eq!(error, EditError::Overlap);
        assert!(session.draft().tracks[0].segments.is_empty());
        assert_eq!(session.revision(), 0);
    }

    #[test]
    fn a_stale_revision_writes_nothing() {
        let (mut session, track) = session();
        session
            .apply(0, &[place(&track, "a", 0, 1_000_000)])
            .unwrap();
        let error = session
            .apply(0, &[place(&track, "b", 1_000_000, 1_000_000)])
            .unwrap_err();
        assert_eq!(
            error,
            EditError::RevisionMismatch {
                expected: 0,
                actual: 1
            }
        );
        assert_eq!(session.draft().tracks[0].segments.len(), 1);
    }

    #[test]
    fn ranges_that_only_touch_both_land() {
        let (mut session, track) = session();
        session
            .apply(
                0,
                &[
                    place(&track, "a", 0, 1_000_000),
                    place(&track, "b", 1_000_000, 1_000_000),
                ],
            )
            .unwrap();
        assert_eq!(session.draft().duration, Micros(2_000_000));
    }

    #[test]
    fn a_locked_track_refuses_the_place() {
        let (mut session, track) = session();
        session.draft.tracks[0].attribute = TrackAttribute(TrackAttribute::LOCK);
        let error = session
            .apply(0, &[place(&track, "a", 0, 1_000_000)])
            .unwrap_err();
        assert_eq!(error, EditError::LockedTrack);
    }

    #[test]
    fn move_and_trim_then_undo_restores_the_first_span() {
        let (mut session, track) = session();
        session
            .apply(0, &[place(&track, "a", 0, 2_000_000)])
            .unwrap();
        session
            .apply(
                1,
                &[Request::Move {
                    segment_id: "a".into(),
                    track_id: track.clone(),
                    start: Micros(500_000),
                }],
            )
            .unwrap();
        session
            .apply(
                2,
                &[Request::Trim {
                    segment_id: "a".into(),
                    edge: TrimEdge::End,
                    time: Micros(1_500_000),
                }],
            )
            .unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        assert_eq!(segment.target, TimeRange::new(500_000, 1_000_000));
        assert_eq!(session.draft().duration, Micros(1_500_000));
        session.undo().unwrap();
        assert_eq!(
            session.draft().tracks[0].segments[0].target,
            TimeRange::new(500_000, 2_000_000)
        );
        session.redo().unwrap();
        assert_eq!(
            session.draft().tracks[0].segments[0].target,
            TimeRange::new(500_000, 1_000_000)
        );
    }

    #[test]
    fn trimming_the_start_moves_a_unit_speed_source() {
        let (mut session, track) = session();
        let mut segment = Segment::new("a", "media", TimeRange::new(0, 2_000_000));
        segment.source_range = Some(TimeRange::new(100, 2_000_000));
        session
            .apply(
                0,
                &[Request::Place {
                    track_id: track,
                    segment: Box::new(segment),
                }],
            )
            .unwrap();
        session
            .apply(
                1,
                &[Request::Trim {
                    segment_id: "a".into(),
                    edge: TrimEdge::Start,
                    time: Micros(500_000),
                }],
            )
            .unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        assert_eq!(segment.target, TimeRange::new(500_000, 1_500_000));
        assert_eq!(
            segment.source_range.as_ref().map(|range| range.start),
            Some(Micros(500_100))
        );
        assert_eq!(
            segment.source_range.as_ref().map(|range| range.duration),
            Some(Micros(1_500_000))
        );
    }

    #[test]
    fn a_constant_speed_maps_target_duration_and_trim_maps_the_source() {
        let (mut session, track) = session();
        let mut segment = Segment::new("a", "media", TimeRange::new(0, 2_000_000));
        segment.source_range = Some(TimeRange::new(0, 2_000_000));
        segment.volume = 0.4;
        segment.extra_material_refs.push(MaterialId::new("spd"));
        session.draft.materials.speeds.push(MaterialSpeed {
            id: MaterialId::new("spd"),
            mode: SpeedMode::Normal,
            speed: 1.0,
            ..MaterialSpeed::default()
        });
        session
            .apply(
                0,
                &[Request::Place {
                    track_id: track.clone(),
                    segment: Box::new(segment),
                }],
            )
            .unwrap();
        session
            .apply(
                1,
                &[Request::SetConstantSpeed {
                    segment_id: "a".into(),
                    speed: 2.0,
                }],
            )
            .unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        assert_eq!(segment.speed, 2.0);
        assert_eq!(segment.target, TimeRange::new(0, 1_000_000));
        assert_eq!(segment.source_range, Some(TimeRange::new(0, 2_000_000)));
        assert_eq!(segment.volume, 0.4);
        assert_eq!(session.draft().materials.speeds[0].speed, 2.0);
        assert_eq!(session.draft().materials.speeds[0].mode, SpeedMode::Normal);
        assert_eq!(session.draft().duration, Micros(1_000_000));

        session
            .apply(
                2,
                &[Request::Trim {
                    segment_id: "a".into(),
                    edge: TrimEdge::Start,
                    time: Micros(200_000),
                }],
            )
            .unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        assert_eq!(segment.target, TimeRange::new(200_000, 800_000));
        assert_eq!(
            segment.source_range,
            Some(TimeRange::new(400_000, 1_600_000))
        );
        assert_eq!(segment.volume, 0.4);

        session
            .apply(
                3,
                &[Request::Trim {
                    segment_id: "a".into(),
                    edge: TrimEdge::End,
                    time: Micros(600_000),
                }],
            )
            .unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        assert_eq!(segment.target, TimeRange::new(200_000, 400_000));
        assert_eq!(segment.source_range, Some(TimeRange::new(400_000, 800_000)));
    }

    fn speed_point(x: f64, y: f64) -> SpeedPoint {
        SpeedPoint {
            x,
            y,
            rest: Default::default(),
        }
    }

    #[test]
    fn a_curve_speed_maps_duration_by_the_integral_and_trim_uses_that_factor() {
        let (mut session, track) = session();
        let mut segment = Segment::new("a", "media", TimeRange::new(0, 2_000_000));
        segment.source_range = Some(TimeRange::new(0, 2_000_000));
        segment.volume = 0.4;
        session
            .apply(
                0,
                &[Request::Place {
                    track_id: track.clone(),
                    segment: Box::new(segment),
                }],
            )
            .unwrap();
        session
            .apply(
                1,
                &[Request::SetCurveSpeed {
                    segment_id: "a".into(),
                    speed_id: MaterialId::new("curve"),
                    points: vec![speed_point(0.0, 2.0), speed_point(1.0, 2.0)],
                }],
            )
            .unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        assert_eq!(segment.speed, 1.0);
        assert_eq!(segment.volume, 0.4);
        assert_eq!(segment.target, TimeRange::new(0, 1_000_000));
        assert_eq!(segment.source_range, Some(TimeRange::new(0, 2_000_000)));
        assert_eq!(session.draft().materials.speeds[0].mode, SpeedMode::Curve);
        assert_eq!(
            session.draft().materials.speeds[0]
                .curve_speed
                .as_ref()
                .map(|curve| curve.speed_points.clone()),
            Some(vec![speed_point(0.0, 2.0), speed_point(1.0, 2.0)])
        );
        session
            .apply(
                2,
                &[Request::Trim {
                    segment_id: "a".into(),
                    edge: TrimEdge::Start,
                    time: Micros(200_000),
                }],
            )
            .unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        assert_eq!(segment.target, TimeRange::new(200_000, 800_000));
        assert_eq!(
            segment.source_range,
            Some(TimeRange::new(400_000, 1_600_000))
        );
        assert_eq!(segment.volume, 0.4);
    }

    #[test]
    fn a_curve_point_at_or_below_zero_leaves_the_draft() {
        let (mut session, track) = session();
        let mut segment = Segment::new("a", "media", TimeRange::new(0, 2_000_000));
        segment.source_range = Some(TimeRange::new(0, 2_000_000));
        session
            .apply(
                0,
                &[
                    Request::Place {
                        track_id: track.clone(),
                        segment: Box::new(segment),
                    },
                    place(&track, "b", 2_000_000, 100_000),
                ],
            )
            .unwrap();
        let before = session.draft().clone();
        let error = session
            .apply(
                1,
                &[Request::SetCurveSpeed {
                    segment_id: "a".into(),
                    speed_id: MaterialId::new("curve"),
                    points: vec![speed_point(0.0, 0.0), speed_point(1.0, 1.0)],
                }],
            )
            .unwrap_err();
        assert_eq!(
            error,
            EditError::Speed(ConstantSpeedError::NonPositiveSpeed)
        );
        assert_eq!(session.draft(), &before);
        let error = session
            .apply(
                1,
                &[Request::SetCurveSpeed {
                    segment_id: "a".into(),
                    speed_id: MaterialId::new("curve"),
                    points: vec![speed_point(0.0, 0.5), speed_point(1.0, 0.5)],
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::Overlap);
        assert_eq!(session.draft(), &before);
    }

    #[test]
    fn a_speed_that_cannot_map_leaves_the_draft() {
        let (mut session, track) = session();
        let mut segment = Segment::new("a", "media", TimeRange::new(0, 2_000_000));
        segment.source_range = Some(TimeRange::new(0, 2_000_000));
        session
            .apply(
                0,
                &[
                    Request::Place {
                        track_id: track.clone(),
                        segment: Box::new(segment),
                    },
                    place(&track, "b", 2_000_000, 100_000),
                ],
            )
            .unwrap();
        let before = session.draft().clone();
        let error = session
            .apply(
                1,
                &[Request::SetConstantSpeed {
                    segment_id: "a".into(),
                    speed: 0.5,
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::Overlap);
        assert_eq!(session.draft(), &before);
        let error = session
            .apply(
                1,
                &[Request::SetConstantSpeed {
                    segment_id: "a".into(),
                    speed: 0.0,
                }],
            )
            .unwrap_err();
        assert_eq!(
            error,
            EditError::Speed(ConstantSpeedError::NonPositiveSpeed)
        );
        assert_eq!(session.draft(), &before);
        session.draft.tracks[0].segments[0].speed = 0.0;
        let before = session.draft().clone();
        let error = session
            .apply(
                1,
                &[Request::Trim {
                    segment_id: "a".into(),
                    edge: TrimEdge::End,
                    time: Micros(1_000_000),
                }],
            )
            .unwrap_err();
        assert_eq!(
            error,
            EditError::Speed(ConstantSpeedError::NonPositiveSpeed)
        );
        assert_eq!(session.draft(), &before);
    }

    #[test]
    fn an_empty_apply_does_not_move_the_revision() {
        let (mut session, _) = session();
        assert_eq!(session.apply(0, &[]).unwrap(), 0);
        assert_eq!(session.undo().unwrap_err(), EditError::EmptyUndo);
    }

    #[test]
    fn rollback_restores_the_draft_from_begin_and_drops_the_undo_apply_pushed() {
        let (mut session, track) = session();
        session
            .apply(0, &[place(&track, "a", 0, 1_000_000)])
            .unwrap();
        let at_begin = session.draft().clone();
        session.begin().unwrap();
        session
            .apply(1, &[place(&track, "b", 1_000_000, 500_000)])
            .unwrap();
        assert_eq!(session.draft().tracks[0].segments.len(), 2);
        session.rollback().unwrap();
        assert_eq!(session.draft(), &at_begin);
        assert_eq!(session.revision(), 2);
        assert_eq!(session.recorded_steps(), 0);
        session.undo().unwrap();
        assert!(session.draft().tracks[0].segments.is_empty());
        assert_eq!(session.undo().unwrap_err(), EditError::EmptyUndo);
    }

    #[test]
    fn commit_records_one_step_and_apply_still_pushes_one_undo() {
        let (mut session, track) = session();
        session.begin().unwrap();
        session
            .apply(0, &[place(&track, "a", 0, 1_000_000)])
            .unwrap();
        session.commit().unwrap();
        assert_eq!(session.recorded_steps(), 1);
        assert_eq!(
            session.recorded_before().map(|draft| draft.duration),
            Some(Micros(0))
        );
        assert_eq!(
            session.recorded_after().map(|draft| draft.duration),
            Some(Micros(1_000_000))
        );
        assert!(session.commit().is_err());
        session.undo().unwrap();
        assert_eq!(session.draft().duration, Micros(0));
        assert_eq!(session.undo().unwrap_err(), EditError::EmptyUndo);
    }

    #[test]
    fn amend_folds_into_the_last_recorded_step() {
        let (mut session, track) = session();
        session.begin().unwrap();
        session
            .apply(0, &[place(&track, "a", 0, 1_000_000)])
            .unwrap();
        session.commit().unwrap();
        session.begin().unwrap();
        session
            .apply(1, &[place(&track, "b", 1_000_000, 500_000)])
            .unwrap();
        session.amend().unwrap();
        assert_eq!(session.recorded_steps(), 1);
        assert_eq!(session.draft().tracks[0].segments.len(), 2);
        assert_eq!(
            session
                .recorded_after()
                .map(|draft| draft.tracks[0].segments.len()),
            Some(2)
        );
        assert_eq!(
            session
                .recorded_before()
                .map(|draft| draft.tracks[0].segments.len()),
            Some(0)
        );
        session.undo().unwrap();
        assert_eq!(session.draft().tracks[0].segments.len(), 1);
        session.undo().unwrap();
        assert!(session.draft().tracks[0].segments.is_empty());
    }

    #[test]
    fn record_then_commit_is_one_step_and_a_second_record_keeps_the_snapshot() {
        let (mut session, track) = session();
        session.record().unwrap();
        assert!(session.is_recording());
        session
            .apply(0, &[place(&track, "a", 0, 1_000_000)])
            .unwrap();
        session.record().unwrap();
        session
            .apply(1, &[place(&track, "b", 1_000_000, 500_000)])
            .unwrap();
        session.rollback().unwrap();
        assert!(!session.is_recording());
        assert!(session.draft().tracks[0].segments.is_empty());
        assert_eq!(session.revision(), 2);
        session.record().unwrap();
        session.apply(2, &[place(&track, "c", 0, 250_000)]).unwrap();
        session.commit().unwrap();
        assert!(!session.is_recording());
        assert_eq!(session.recorded_steps(), 1);
        assert_eq!(session.draft().tracks[0].segments.len(), 1);
    }

    #[test]
    fn a_closed_history_call_leaves_the_draft() {
        let (mut session, track) = session();
        let before = session.draft().clone();
        assert_eq!(session.rollback().unwrap_err(), EditError::HistoryClosed);
        assert_eq!(session.commit().unwrap_err(), EditError::HistoryClosed);
        assert_eq!(session.amend().unwrap_err(), EditError::NothingToAmend);
        session.begin().unwrap();
        assert_eq!(session.begin().unwrap_err(), EditError::HistoryOpen);
        assert_eq!(session.draft(), &before);
        session
            .apply(0, &[place(&track, "a", 0, 1_000_000)])
            .unwrap();
        let placed = session.draft().clone();
        assert_eq!(session.begin().unwrap_err(), EditError::HistoryOpen);
        assert_eq!(session.draft(), &placed);
    }

    #[test]
    fn mute_hide_and_solo_do_not_change_volume_or_fade() {
        let (mut session, track) = session();
        let other = TrackId::new("over");
        session.draft.tracks.push(Track::main(other.clone()));
        session
            .apply(0, &[place(&track, "a", 0, 1_000_000)])
            .unwrap();
        session.draft.tracks[0].segments[0].volume = 0.25;
        session.draft.materials.audio_fades.push(MaterialAudioFade {
            id: MaterialId::new("fade"),
            fade_in_duration: Micros(80_000),
            fade_out_duration: Micros(120_000),
            ..MaterialAudioFade::default()
        });
        session.draft.tracks[0].segments[0]
            .extra_material_refs
            .push(MaterialId::new("fade"));

        let mut expected = session.draft().clone();
        expected.tracks[0].attribute = TrackAttribute(TrackAttribute::MUTE);
        session
            .apply(
                1,
                &[Request::SetTrackAttribute {
                    track_id: track.clone(),
                    bit: TrackAttribute::MUTE,
                    on: true,
                }],
            )
            .unwrap();
        assert_eq!(session.draft(), &expected);
        assert_eq!(session.draft().tracks[0].segments[0].volume, 0.25);

        expected.tracks[0].attribute = TrackAttribute(TrackAttribute::MUTE | TrackAttribute::HIDE);
        session
            .apply(
                2,
                &[Request::SetTrackAttribute {
                    track_id: track.clone(),
                    bit: TrackAttribute::HIDE,
                    on: true,
                }],
            )
            .unwrap();
        assert_eq!(session.draft(), &expected);

        expected.tracks[0].attribute =
            TrackAttribute(TrackAttribute::MUTE | TrackAttribute::HIDE | TrackAttribute::SOLO);
        expected.tracks[1].attribute = TrackAttribute(TrackAttribute::MUTE_BY_SOLO);
        session
            .apply(
                3,
                &[Request::SetTrackAttribute {
                    track_id: track.clone(),
                    bit: TrackAttribute::SOLO,
                    on: true,
                }],
            )
            .unwrap();
        assert_eq!(session.draft(), &expected);
        assert_eq!(
            session.draft().materials.audio_fades[0].fade_in_duration,
            Micros(80_000)
        );

        expected.tracks[1].attribute = TrackAttribute(TrackAttribute::SOLO);
        expected.tracks[0].attribute = TrackAttribute(
            TrackAttribute::MUTE | TrackAttribute::HIDE | TrackAttribute::MUTE_BY_SOLO,
        );
        session
            .apply(
                4,
                &[
                    Request::SetTrackAttribute {
                        track_id: track.clone(),
                        bit: TrackAttribute::SOLO,
                        on: false,
                    },
                    Request::SetTrackAttribute {
                        track_id: other.clone(),
                        bit: TrackAttribute::SOLO,
                        on: true,
                    },
                ],
            )
            .unwrap();
        assert_eq!(session.draft(), &expected);

        expected.tracks[1].attribute = TrackAttribute(0);
        expected.tracks[0].attribute = TrackAttribute(TrackAttribute::MUTE | TrackAttribute::HIDE);
        session
            .apply(
                5,
                &[Request::SetTrackAttribute {
                    track_id: other,
                    bit: TrackAttribute::SOLO,
                    on: false,
                }],
            )
            .unwrap();
        assert_eq!(session.draft(), &expected);
    }

    #[test]
    fn a_missing_track_attribute_leaves_the_draft() {
        let (mut session, track) = session();
        session
            .apply(0, &[place(&track, "a", 0, 1_000_000)])
            .unwrap();
        let before = session.draft().clone();
        let error = session
            .apply(
                1,
                &[
                    place(&track, "b", 1_000_000, 100_000),
                    Request::SetTrackAttribute {
                        track_id: TrackId::new("missing"),
                        bit: TrackAttribute::MUTE,
                        on: true,
                    },
                ],
            )
            .unwrap_err();
        assert_eq!(error, EditError::MissingTrack);
        assert_eq!(session.draft(), &before);
        let error = session
            .apply(
                1,
                &[Request::SetTrackAttribute {
                    track_id: track.clone(),
                    bit: 32,
                    on: true,
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::UnknownAttribute);
        assert_eq!(session.draft(), &before);

        let mut expected = before.clone();
        expected.tracks[0].attribute = TrackAttribute(TrackAttribute::MUTE_BY_SOLO);
        session
            .apply(
                1,
                &[Request::SetTrackAttribute {
                    track_id: track,
                    bit: TrackAttribute::MUTE_BY_SOLO,
                    on: true,
                }],
            )
            .unwrap();
        assert_eq!(session.draft(), &expected);
        assert_eq!(session.draft().tracks[0].segments[0].volume, 1.0);
    }

    #[test]
    fn a_failed_request_inside_a_transaction_keeps_begin() {
        let (mut session, track) = session();
        session
            .apply(0, &[place(&track, "a", 0, 1_000_000)])
            .unwrap();
        session.begin().unwrap();
        let error = session
            .apply(1, &[place(&track, "b", 500_000, 500_000)])
            .unwrap_err();
        assert_eq!(error, EditError::Overlap);
        assert_eq!(session.draft().tracks[0].segments.len(), 1);
        session.rollback().unwrap();
        assert_eq!(session.draft().tracks[0].segments.len(), 1);
        assert_eq!(session.revision(), 1);
    }

    fn blend(id: &str, duration: i64, overlap: bool) -> MaterialTransition {
        MaterialTransition {
            id: MaterialId::new(id),
            duration: Micros(duration),
            is_overlap: overlap,
            ..MaterialTransition::default()
        }
    }

    #[test]
    fn a_blend_stays_on_the_outgoing_segment_and_a_split_inside_it_is_refused() {
        let (mut session, track) = session();
        let mut outgoing = Segment::new("a", "media", TimeRange::new(0, 1_000_000));
        outgoing.source_range = Some(TimeRange::new(0, 1_000_000));
        outgoing.volume = 0.4;
        session
            .apply(
                0,
                &[
                    Request::Place {
                        track_id: track.clone(),
                        segment: Box::new(outgoing),
                    },
                    place(&track, "b", 1_000_000, 1_000_000),
                ],
            )
            .unwrap();
        session
            .apply(
                1,
                &[Request::SetTransition {
                    segment_id: "a".into(),
                    transition: Box::new(blend("blend", 200_000, false)),
                }],
            )
            .unwrap();
        assert_eq!(
            session.draft().tracks[0].segments[0].target,
            TimeRange::new(0, 1_000_000)
        );
        assert_eq!(
            session.draft().tracks[0].segments[1].target,
            TimeRange::new(1_000_000, 1_000_000)
        );
        assert_eq!(session.draft().tracks[0].segments[0].volume, 0.4);
        assert_eq!(session.draft().materials.transitions[0].path, "");
        assert_eq!(
            session.draft().materials.transitions[0].duration,
            Micros(200_000)
        );
        assert!(!session.draft().materials.transitions[0].is_overlap);
        let before = session.draft().clone();
        let error = session
            .apply(
                2,
                &[Request::Split {
                    segment_id: "a".into(),
                    at: Micros(800_000),
                    right_id: "a-right".into(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::TransitionWindow);
        assert_eq!(session.draft(), &before);
        let error = session
            .apply(
                2,
                &[
                    Request::SetTransition {
                        segment_id: "a".into(),
                        transition: Box::new(blend("other", 10, false)),
                    },
                    Request::Split {
                        segment_id: "a".into(),
                        at: Micros(999_995),
                        right_id: "a-right".into(),
                    },
                ],
            )
            .unwrap_err();
        assert_eq!(error, EditError::TransitionWindow);
        assert_eq!(session.draft(), &before);
        session
            .apply(
                2,
                &[Request::Split {
                    segment_id: "a".into(),
                    at: Micros(700_000),
                    right_id: "a-right".into(),
                }],
            )
            .unwrap();
        let segments = &session.draft().tracks[0].segments;
        assert_eq!(segments[0].target, TimeRange::new(0, 700_000));
        assert_eq!(segments[0].source_range, Some(TimeRange::new(0, 700_000)));
        assert!(segments[0].extra_material_refs.is_empty());
        assert_eq!(segments[0].volume, 0.4);
        assert_eq!(segments[1].target, TimeRange::new(700_000, 300_000));
        assert_eq!(
            segments[1].source_range,
            Some(TimeRange::new(700_000, 300_000))
        );
        assert_eq!(
            segments[1].extra_material_refs,
            vec![MaterialId::new("blend")]
        );
        assert_eq!(segments[1].volume, 0.4);
        assert_eq!(segments[2].target.start, Micros(1_000_000));
        assert_eq!(session.draft().materials.transitions[0].path, "");
    }

    #[test]
    fn an_overlap_window_crosses_the_cut_and_a_split_there_leaves_the_draft() {
        let (mut session, track) = session();
        session
            .apply(
                0,
                &[
                    place(&track, "a", 0, 1_000_000),
                    place(&track, "b", 1_000_000, 1_000_000),
                ],
            )
            .unwrap();
        session
            .apply(
                1,
                &[Request::SetTransition {
                    segment_id: "a".into(),
                    transition: Box::new(blend("blend", 101, true)),
                }],
            )
            .unwrap();
        assert!(session.draft().materials.transitions[0].is_overlap);
        assert_eq!(
            session.draft().tracks[0].segments[0].target.end(),
            session.draft().tracks[0].segments[1].target.start
        );
        let before = session.draft().clone();
        for (segment_id, at) in [("a", 999_950_i64), ("b", 1_000_050)] {
            let error = session
                .apply(
                    2,
                    &[Request::Split {
                        segment_id: segment_id.into(),
                        at: Micros(at),
                        right_id: "right".into(),
                    }],
                )
                .unwrap_err();
            assert_eq!(error, EditError::TransitionWindow);
            assert_eq!(session.draft(), &before);
        }
    }

    #[test]
    fn a_hard_cut_keeps_a_missing_package_and_still_splits() {
        let (mut session, track) = session();
        session
            .apply(0, &[place(&track, "a", 0, 1_000_000)])
            .unwrap();
        session
            .apply(
                1,
                &[Request::SetTransition {
                    segment_id: "a".into(),
                    transition: Box::new(blend("blend", 0, false)),
                }],
            )
            .unwrap();
        session
            .apply(
                2,
                &[Request::Split {
                    segment_id: "a".into(),
                    at: Micros(400_000),
                    right_id: "a-right".into(),
                }],
            )
            .unwrap();
        assert_eq!(session.draft().materials.transitions[0].duration, Micros(0));
        assert_eq!(session.draft().materials.transitions[0].path, "");
        assert_eq!(
            session.draft().tracks[0].segments[0].target,
            TimeRange::new(0, 400_000)
        );
        assert_eq!(
            session.draft().tracks[0].segments[1].target,
            TimeRange::new(400_000, 600_000)
        );
        assert_eq!(
            session.draft().tracks[0].segments[1].extra_material_refs,
            vec![MaterialId::new("blend")]
        );
    }

    #[test]
    fn a_negative_duration_is_stored_and_played_as_a_hard_cut() {
        let (mut session, track) = session();
        session
            .apply(0, &[place(&track, "a", 0, 1_000_000)])
            .unwrap();
        session
            .apply(
                1,
                &[Request::SetTransition {
                    segment_id: "a".into(),
                    transition: Box::new(blend("blend", -5, true)),
                }],
            )
            .unwrap();
        let stored = session.draft().clone();
        session
            .apply(
                2,
                &[Request::Split {
                    segment_id: "a".into(),
                    at: Micros(500_000),
                    right_id: "a-right".into(),
                }],
            )
            .unwrap();
        assert_eq!(
            session.draft().materials.transitions,
            stored.materials.transitions
        );
        assert_eq!(
            session.draft().materials.transitions[0].duration,
            Micros(-5)
        );
    }

    #[test]
    fn a_locked_track_refuses_the_blend_and_the_split() {
        let (mut session, track) = session();
        session
            .apply(0, &[place(&track, "a", 0, 1_000_000)])
            .unwrap();
        session.draft.tracks[0].attribute = TrackAttribute(TrackAttribute::LOCK);
        let before = session.draft().clone();
        let error = session
            .apply(
                1,
                &[Request::SetTransition {
                    segment_id: "a".into(),
                    transition: Box::new(blend("blend", 10, false)),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::LockedTrack);
        assert_eq!(session.draft(), &before);
        let error = session
            .apply(
                1,
                &[Request::Split {
                    segment_id: "a".into(),
                    at: Micros(400_000),
                    right_id: "a-right".into(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::LockedTrack);
        assert_eq!(session.draft(), &before);
    }
}
