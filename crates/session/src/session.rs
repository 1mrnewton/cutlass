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

    /// Open a project read from disk: `open`, after dropping every lane other
    /// than main that holds no clips. Edits already drop a lane they empty;
    /// this clears the ones a project saved before that rule still holds. It
    /// is a load repair, not an edit: the revision is 0 and there is nothing
    /// to undo.
    #[must_use]
    pub fn open_project(mut draft: Draft) -> Self {
        drop_empty_lanes(&mut draft);
        Self::open(draft)
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

    /// Apply every request into the last undo step instead of a new one, so
    /// a gesture that changes the draft many times undoes in one go. Redo
    /// is cleared and the revision advances, as for [`Self::apply`]. With no
    /// undo step yet this is [`Self::apply`].
    pub fn fold(&mut self, expected: u64, requests: &[Request]) -> Result<u64, EditError> {
        if !self.history.can_undo() {
            return self.apply(expected, requests);
        }
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
        self.draft = next;
        self.history.clear_redo();
        self.revision = self.revision.saturating_add(1);
        Ok(self.revision)
    }

    /// Whether [`Session::undo`] has a step to restore.
    #[must_use]
    pub fn can_undo(&self) -> bool {
        self.history.can_undo()
    }

    /// Whether [`Session::redo`] has a step to reapply.
    #[must_use]
    pub fn can_redo(&self) -> bool {
        self.history.can_redo()
    }

    /// Restore the draft from before the last apply. The revision still
    /// advances. Refused while a transaction is open.
    pub fn undo(&mut self) -> Result<u64, EditError> {
        if self.history.is_open() {
            return Err(EditError::HistoryOpen);
        }
        let previous = self.history.pop_undo().ok_or(EditError::EmptyUndo)?;
        let current = std::mem::replace(&mut self.draft, previous);
        self.history.push_redo(current);
        self.revision = self.revision.saturating_add(1);
        Ok(self.revision)
    }

    /// Reapply the draft that undo removed. The revision advances again.
    /// Refused while a transaction is open.
    pub fn redo(&mut self) -> Result<u64, EditError> {
        if self.history.is_open() {
            return Err(EditError::HistoryOpen);
        }
        let next = self.history.pop_redo().ok_or(EditError::EmptyRedo)?;
        let current = std::mem::replace(&mut self.draft, next);
        self.history.push_undo(current);
        self.revision = self.revision.saturating_add(1);
        Ok(self.revision)
    }

    /// Open a transaction: the applies until [`Self::commit`] are one undo
    /// step.
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

    /// Close the transaction. A changed draft is one undo step.
    pub fn commit(&mut self) -> Result<(), EditError> {
        self.history.commit(&self.draft)
    }

    /// Close the transaction into the undo step before it.
    pub fn amend(&mut self) -> Result<(), EditError> {
        self.history.amend(&self.draft)
    }

    /// Put the open draft back where it was at [`Self::begin`] or
    /// [`Self::record`]. The revision advances, as for any change a reader
    /// of the draft must notice.
    pub fn rollback(&mut self) -> Result<(), EditError> {
        self.draft = self.history.rollback()?;
        self.revision = self.revision.saturating_add(1);
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

/// Take every lane other than main that holds no clips.
pub fn drop_empty_lanes(draft: &mut Draft) {
    draft
        .tracks
        .retain(|track| track.is_main() || !track.segments.is_empty());
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        ConstantSpeedError, Draft, MaterialAudioFade, MaterialId, MaterialSpeed, MaterialText,
        MaterialTransition, Micros, Segment, SegmentId, SpeedMode, SpeedPoint, TimeRange, Track,
        TrackAttribute, TrackId, WordColumns,
    };

    use super::*;
    use crate::request::{Request, TrimEdge};

    fn session() -> (Session, TrackId) {
        let mut draft = Draft::empty("D");
        // These cases check the requested span. Packing is covered in the magnet tests.
        draft.config.maintrack_adsorb = false;
        let track_id = TrackId::new("main");
        draft.tracks[0].id = track_id.clone();
        (Session::open(draft), track_id)
    }

    #[test]
    fn a_project_opens_without_its_empty_lanes_but_keeps_main_and_full_ones() {
        let mut draft = Draft::empty("D");
        draft.tracks[0].segments.clear();
        let mut sound = Track::main(TrackId::new("sound"));
        sound.kind = cutlass_model::TrackType::Audio;
        sound.flag = cutlass_model::TrackFlag::None;
        sound.attribute = TrackAttribute(TrackAttribute::MUTE);
        let mut words = Track::main(TrackId::new("words"));
        words.kind = cutlass_model::TrackType::Text;
        words.flag = cutlass_model::TrackFlag::None;
        words
            .segments
            .push(Segment::new("hello", "text", TimeRange::new(0, 1_000_000)));
        draft.tracks.insert(0, words);
        draft.tracks.push(sound.clone());
        sound.id = TrackId::new("sound-2");
        draft.tracks.push(sound);

        let session = Session::open_project(draft);
        let ids: Vec<&str> = session
            .draft()
            .tracks
            .iter()
            .map(|track| track.id.as_str())
            .collect();
        assert_eq!(
            ids.len(),
            2,
            "the empty main stays and the text lane stays: {ids:?}"
        );
        assert_eq!(ids[0], "words");
        assert!(session.draft().tracks[1].is_main());
        assert_eq!(session.revision(), 0);
        assert!(!session.can_undo());
    }

    fn place(track_id: &TrackId, id: &str, start: i64, duration: i64) -> Request {
        Request::Place {
            track_id: track_id.clone(),
            segment: Box::new(Segment::new(id, "media", TimeRange::new(start, duration))),
        }
    }

    #[test]
    fn a_fold_joins_the_last_undo_step() {
        let (mut session, track) = session();
        session.apply(0, &[place(&track, "a", 0, 10)]).unwrap();
        session.apply(1, &[place(&track, "b", 10, 10)]).unwrap();
        let revision = session.fold(2, &[place(&track, "c", 20, 10)]).unwrap();
        assert_eq!(revision, 3);
        assert_eq!(session.draft().tracks[0].segments.len(), 3);
        assert!(matches!(
            session.fold(2, &[]),
            Err(EditError::RevisionMismatch { .. })
        ));
        session.undo().unwrap();
        assert_eq!(session.draft().tracks[0].segments.len(), 1);
    }

    #[test]
    fn a_fold_with_nothing_to_undo_is_a_step_of_its_own() {
        let (mut session, track) = session();
        session.fold(0, &[place(&track, "a", 0, 10)]).unwrap();
        session.undo().unwrap();
        assert!(session.draft().tracks[0].segments.is_empty());
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
    fn set_words_is_one_undo_and_a_failed_combo_keeps_the_material() {
        let (mut session, track) = session();
        let material_id = MaterialId::new("caption");
        session
            .draft
            .materials
            .texts
            .push(MaterialText::new(material_id.clone()));
        session
            .apply(
                0,
                &[Request::SetWords {
                    material_id: material_id.clone(),
                    words: WordColumns {
                        start_time: vec![0],
                        end_time: vec![250],
                        text: vec!["hi".to_owned()],
                        rest: Default::default(),
                    },
                }],
            )
            .unwrap();
        assert_eq!(session.draft().materials.texts[0].content, "hi");
        assert_eq!(session.revision(), 1);
        let error = session
            .apply(
                1,
                &[
                    Request::SetWords {
                        material_id,
                        words: WordColumns {
                            start_time: vec![0],
                            end_time: vec![10],
                            text: vec!["no".to_owned()],
                            rest: Default::default(),
                        },
                    },
                    place(&track, "overlap-a", 0, 1_000_000),
                    place(&track, "overlap-b", 0, 1_000_000),
                ],
            )
            .unwrap_err();
        assert_eq!(error, EditError::Overlap);
        assert_eq!(session.draft().materials.texts[0].content, "hi");
        assert_eq!(session.revision(), 1);
        session.undo().unwrap();
        assert!(session.draft().materials.texts[0].content.is_empty());
        assert!(session.draft().materials.texts[0].words.text.is_empty());
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
    fn undo_and_redo_say_when_they_have_a_step() {
        let (mut session, track) = session();
        assert!(!session.can_undo() && !session.can_redo());
        session
            .apply(0, &[place(&track, "a", 0, 1_000_000)])
            .unwrap();
        assert!(session.can_undo() && !session.can_redo());
        session.undo().unwrap();
        assert!(!session.can_undo() && session.can_redo());
    }

    #[test]
    fn undo_keeps_only_the_newest_steps_up_to_the_limit() {
        let (mut session, track) = session();
        let steps = crate::UNDO_LIMIT + 5;
        for i in 0..steps {
            let start = i64::try_from(i).unwrap() * 10;
            session
                .apply(
                    session.revision(),
                    &[place(&track, &format!("s{i}"), start, 10)],
                )
                .unwrap();
        }
        for _ in 0..crate::UNDO_LIMIT {
            session.undo().unwrap();
        }
        // The five oldest steps were dropped, so undo stops at five clips.
        assert!(matches!(session.undo(), Err(EditError::EmptyUndo)));
        assert_eq!(session.draft().tracks[0].segments.len(), 5);
        for _ in 0..crate::UNDO_LIMIT {
            session.redo().unwrap();
        }
        assert_eq!(session.draft().tracks[0].segments.len(), steps);
    }

    #[test]
    fn a_split_curve_clip_gives_each_half_its_stretch_of_the_curve() {
        let (mut session, track) = session();
        let mut segment = Segment::new("a", "media", TimeRange::new(0, 4_000_000));
        segment.source_range = Some(TimeRange::new(0, 4_000_000));
        session
            .apply(
                0,
                &[Request::Place {
                    track_id: track.clone(),
                    segment: Box::new(segment),
                }],
            )
            .unwrap();
        let revision = session
            .apply(
                1,
                &[Request::SetCurveSpeed {
                    segment_id: "a".into(),
                    speed_id: MaterialId::new("curve"),
                    points: vec![speed_point(0.0, 0.5), speed_point(1.0, 4.0)],
                }],
            )
            .unwrap();
        let whole = session.draft().tracks[0].segments[0].target.clone();
        let at = whole.start.0 + whole.duration.0 / 2;
        session
            .apply(
                revision,
                &[Request::Split {
                    segment_id: "a".into(),
                    at: Micros(at),
                    right_id: "b".into(),
                }],
            )
            .unwrap();
        let draft = session.draft();
        let (left, right) = (&draft.tracks[0].segments[0], &draft.tracks[0].segments[1]);
        let points = |segment: &Segment| {
            let id = &segment.extra_material_refs[0];
            draft
                .materials
                .speeds
                .iter()
                .find(|speed| &speed.id == id)
                .and_then(|speed| speed.curve_speed.clone())
                .unwrap()
                .speed_points
        };
        assert_ne!(left.extra_material_refs, right.extra_material_refs);
        for segment in [left, right] {
            let source = segment.source_range.clone().unwrap().duration;
            let mapped = cutlass_model::curve_target_duration(source, &points(segment)).unwrap();
            assert!(
                (mapped.0 - segment.target.duration.0).abs() <= 2,
                "{mapped:?}"
            );
        }
        // Slow at the start, so the first half of the time covers less than
        // half the footage.
        let left_source = left.source_range.clone().unwrap().duration.0;
        assert!(left_source < 2_000_000, "{left_source}");
        assert_eq!(
            left_source + right.source_range.clone().unwrap().duration.0,
            4_000_000
        );
    }

    #[test]
    fn change_pitch_turns_on_and_off_and_a_preset_curve_sets_the_length() {
        let (mut session, track) = session();
        let mut segment = Segment::new("a", "media", TimeRange::new(0, 2_000_000));
        segment.source_range = Some(TimeRange::new(0, 2_000_000));
        session
            .apply(
                0,
                &[Request::Place {
                    track_id: track.clone(),
                    segment: Box::new(segment),
                }],
            )
            .unwrap();
        let points = cutlass_model::SpeedPreset::FlashIn.points();
        session
            .apply(
                1,
                &[
                    Request::SetChangePitch {
                        segment_id: "a".into(),
                        on: true,
                    },
                    Request::SetCurveSpeed {
                        segment_id: "a".into(),
                        speed_id: MaterialId::new("curve"),
                        points: points.clone(),
                    },
                ],
            )
            .unwrap();
        let segment = &session.draft().tracks[0].segments[0];
        assert!(segment.is_tone_modify);
        let factor = cutlass_model::curve_speed_factor(&points).unwrap();
        #[allow(clippy::cast_possible_truncation)]
        let length = (2_000_000.0 * factor).round() as i64;
        assert_eq!(segment.target.duration, Micros(length));
        session
            .apply(
                2,
                &[Request::SetChangePitch {
                    segment_id: "a".into(),
                    on: false,
                }],
            )
            .unwrap();
        assert!(!session.draft().tracks[0].segments[0].is_tone_modify);
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
    fn rollback_restores_the_draft_from_begin_and_bumps_the_revision() {
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
        // The apply was 2; the rollback changed the draft again.
        assert_eq!(session.revision(), 3);
        assert_eq!(session.recorded_steps(), 0);
        session.undo().unwrap();
        assert!(session.draft().tracks[0].segments.is_empty());
        assert_eq!(session.undo().unwrap_err(), EditError::EmptyUndo);
    }

    #[test]
    fn a_committed_transaction_is_one_undo_step() {
        let (mut session, track) = session();
        session.begin().unwrap();
        session
            .apply(0, &[place(&track, "a", 0, 1_000_000)])
            .unwrap();
        session
            .apply(1, &[place(&track, "b", 1_000_000, 500_000)])
            .unwrap();
        assert!(!session.can_undo(), "no step until the commit");
        assert_eq!(session.undo().unwrap_err(), EditError::HistoryOpen);
        session.commit().unwrap();
        assert_eq!(session.recorded_steps(), 1);
        assert_eq!(
            session.recorded_before().map(|draft| draft.duration),
            Some(Micros(0))
        );
        assert_eq!(
            session.recorded_after().map(|draft| draft.duration),
            Some(Micros(1_500_000))
        );
        assert!(session.commit().is_err());
        session.undo().unwrap();
        assert!(session.draft().tracks[0].segments.is_empty());
        assert_eq!(session.undo().unwrap_err(), EditError::EmptyUndo);
        session.redo().unwrap();
        assert_eq!(session.draft().tracks[0].segments.len(), 2);
    }

    #[test]
    fn a_transaction_that_changes_nothing_adds_no_step() {
        let (mut session, track) = session();
        session.begin().unwrap();
        session.commit().unwrap();
        assert!(!session.can_undo());
        session.begin().unwrap();
        session
            .apply(0, &[place(&track, "a", 0, 1_000_000)])
            .unwrap();
        session
            .apply(
                1,
                &[Request::Remove {
                    segment_id: SegmentId::new("a"),
                }],
            )
            .unwrap();
        session.commit().unwrap();
        assert!(!session.can_undo());
    }

    #[test]
    fn amend_joins_the_open_transaction_to_the_step_before_it() {
        let (mut edited, track) = session();
        edited
            .apply(0, &[place(&track, "a", 0, 1_000_000)])
            .unwrap();
        edited.begin().unwrap();
        edited
            .apply(1, &[place(&track, "b", 1_000_000, 500_000)])
            .unwrap();
        edited.amend().unwrap();
        assert_eq!(edited.draft().tracks[0].segments.len(), 2);
        edited.undo().unwrap();
        assert!(edited.draft().tracks[0].segments.is_empty());
        assert_eq!(edited.undo().unwrap_err(), EditError::EmptyUndo);

        // With no step before it, amend is a commit.
        let (mut fresh, track) = session();
        fresh.begin().unwrap();
        fresh.apply(0, &[place(&track, "a", 0, 1_000_000)]).unwrap();
        fresh.amend().unwrap();
        fresh.undo().unwrap();
        assert!(fresh.draft().tracks[0].segments.is_empty());
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
        assert_eq!(session.revision(), 3);
        session.record().unwrap();
        session.apply(3, &[place(&track, "c", 0, 250_000)]).unwrap();
        session.commit().unwrap();
        assert!(!session.is_recording());
        assert_eq!(session.recorded_steps(), 1);
        assert_eq!(session.draft().tracks[0].segments.len(), 1);
        session.undo().unwrap();
        assert!(session.draft().tracks[0].segments.is_empty());
    }

    #[test]
    fn a_closed_history_call_leaves_the_draft() {
        let (mut session, track) = session();
        let before = session.draft().clone();
        assert_eq!(session.rollback().unwrap_err(), EditError::HistoryClosed);
        assert_eq!(session.commit().unwrap_err(), EditError::HistoryClosed);
        assert_eq!(session.amend().unwrap_err(), EditError::HistoryClosed);
        assert_eq!(session.revision(), 0);
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
        assert_eq!(session.revision(), 2);
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
