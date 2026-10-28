use cutlass_model::Draft;

use crate::edit::{EditError, apply_all};
use crate::history::History;
use crate::request::Request;

/// The open draft, the revision a write must name, and one undo stack.
#[derive(Clone, Debug)]
pub struct Session {
    draft: Draft,
    revision: u64,
    history: History,
}

impl Session {
    /// Revision starts at 0. The first apply must send 0.
    #[must_use]
    pub fn open(draft: Draft) -> Self {
        Self {
            draft,
            revision: 0,
            history: History::default(),
        }
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
        apply_all(&mut next, requests)?;
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
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        Draft, MaterialId, MaterialText, Micros, Segment, TimeRange, TrackAttribute, TrackId,
        WordColumns,
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
    fn an_empty_apply_does_not_move_the_revision() {
        let (mut session, _) = session();
        assert_eq!(session.apply(0, &[]).unwrap(), 0);
        assert_eq!(session.undo().unwrap_err(), EditError::EmptyUndo);
    }
}
