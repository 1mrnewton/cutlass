//! The job that turns clicks on one frame into a selection on the draft.
//!
//! The job records the draft revision it started from. The model runs
//! without the session ([`SelectJob::select`]), so a shell can run it on
//! another thread, then one `SetSelection` request stores the result
//! ([`SelectJob::store`]). A result for a draft that moved is ignored.

use cutlass_effects::Selection;
use cutlass_model::{MaterialId, Micros, SegmentId};
use cutlass_session::{EditError, Request, Session};

use crate::{Click, Image, Selector, VisionError};

/// A selection that started at one draft revision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectJob {
    revision: u64,
}

/// Why the job did not store a selection.
#[derive(Clone, Debug, PartialEq)]
pub enum JobError {
    /// The draft moved after the job recorded its revision.
    StaleRevision {
        expected: u64,
        actual: u64,
    },
    Vision(VisionError),
    Edit(EditError),
}

impl SelectJob {
    /// Records the open revision. The write is ignored if that revision moves.
    #[must_use]
    pub fn start(session: &Session) -> Self {
        Self {
            revision: session.revision(),
        }
    }

    #[must_use]
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Runs `selector` on `frame`, the clip's picture at source time `at`,
    /// and codes the mask for `clicks`. Needs no session.
    pub fn select(
        &self,
        selector: &mut dyn Selector,
        frame: &Image<'_>,
        at: Micros,
        clicks: &[Click],
    ) -> Result<Selection, JobError> {
        selector.encode(frame).map_err(JobError::Vision)?;
        let mask = selector.select(clicks).map_err(JobError::Vision)?;
        Selection::from_alpha(mask.width, mask.height, &mask.alpha, at).ok_or_else(|| {
            JobError::Vision(VisionError::Model {
                message: "the mask does not fit its frame".into(),
            })
        })
    }

    /// Stores `selection` on `segment_id` with one request, as `material_id`.
    /// Returns the new revision.
    pub fn store(
        &self,
        session: &mut Session,
        segment_id: &SegmentId,
        material_id: &MaterialId,
        selection: Selection,
    ) -> Result<u64, JobError> {
        session
            .apply(
                self.revision,
                &[Request::SetSelection {
                    segment_id: segment_id.clone(),
                    material_id: material_id.clone(),
                    selection: Some(selection),
                }],
            )
            .map_err(|error| match error {
                EditError::RevisionMismatch { actual, .. } => JobError::StaleRevision {
                    expected: self.revision,
                    actual,
                },
                other => JobError::Edit(other),
            })
    }
}

#[cfg(test)]
mod tests {
    use cutlass_effects::segment_selection;
    use cutlass_model::{Draft, Segment, TimeRange};

    use super::*;
    use crate::Mask;

    /// Selects the left half of whatever frame it is given.
    struct LeftHalf {
        size: Option<(u32, u32)>,
        fail: bool,
    }

    impl Selector for LeftHalf {
        fn encode(&mut self, image: &Image<'_>) -> Result<(), VisionError> {
            if self.fail {
                return Err(VisionError::NoGpu);
            }
            self.size = Some((image.width, image.height));
            Ok(())
        }

        fn select(&self, clicks: &[Click]) -> Result<Mask, VisionError> {
            let (width, height) = self.size.ok_or(VisionError::NoFrame)?;
            crate::check_clicks(clicks, width, height)?;
            let alpha = (0..width * height)
                .map(|i| if i % width < width / 2 { 255 } else { 0 })
                .collect();
            Ok(Mask {
                width,
                height,
                alpha,
                score: 0.9,
            })
        }
    }

    fn session() -> Session {
        let mut draft = Draft::empty("D");
        draft.tracks[0]
            .segments
            .push(Segment::new("a", "m", TimeRange::new(0, 1_000_000)));
        Session::open(draft)
    }

    fn frame() -> Vec<u8> {
        vec![0; 4 * 2 * 3]
    }

    const CLICK: [Click; 1] = [Click {
        x: 1.0,
        y: 1.0,
        keep: true,
    }];

    #[test]
    fn clicks_become_a_selection_on_the_clip() {
        let mut session = session();
        let job = SelectJob::start(&session);
        let rgb = frame();
        let image = Image {
            width: 4,
            height: 2,
            rgb: &rgb,
        };
        let mut selector = LeftHalf {
            size: None,
            fail: false,
        };
        let selection = job
            .select(&mut selector, &image, Micros(250_000), &CLICK)
            .unwrap();
        assert_eq!(selection.runs, vec![0, 2, 2, 2, 2]);
        assert_eq!(selection.frame, Micros(250_000));
        let revision = job
            .store(&mut session, &"a".into(), &"s".into(), selection.clone())
            .unwrap();
        assert_eq!(revision, job.revision() + 1);
        assert_eq!(
            segment_selection(session.draft(), &session.draft().tracks[0].segments[0]),
            Some(selection)
        );
    }

    #[test]
    fn a_result_for_a_draft_that_moved_is_ignored() {
        let mut session = session();
        let job = SelectJob::start(&session);
        let rgb = frame();
        let image = Image {
            width: 4,
            height: 2,
            rgb: &rgb,
        };
        let mut selector = LeftHalf {
            size: None,
            fail: false,
        };
        let selection = job
            .select(&mut selector, &image, Micros(0), &CLICK)
            .unwrap();
        // Another edit lands while the model runs.
        session
            .apply(
                job.revision(),
                &[Request::SetVolume {
                    segment_id: "a".into(),
                    volume: 0.5,
                }],
            )
            .unwrap();
        let before = session.draft().clone();
        assert_eq!(
            job.store(&mut session, &"a".into(), &"s".into(), selection),
            Err(JobError::StaleRevision {
                expected: job.revision(),
                actual: job.revision() + 1
            })
        );
        assert_eq!(session.draft(), &before);
    }

    #[test]
    fn a_failed_model_or_a_bad_click_names_the_cause_and_stores_nothing() {
        let session = session();
        let job = SelectJob::start(&session);
        let rgb = frame();
        let image = Image {
            width: 4,
            height: 2,
            rgb: &rgb,
        };
        let mut failing = LeftHalf {
            size: None,
            fail: true,
        };
        assert_eq!(
            job.select(&mut failing, &image, Micros(0), &CLICK),
            Err(JobError::Vision(VisionError::NoGpu))
        );
        let mut selector = LeftHalf {
            size: None,
            fail: false,
        };
        assert!(matches!(
            job.select(&mut selector, &image, Micros(0), &[]),
            Err(JobError::Vision(VisionError::Clicks { .. }))
        ));
    }

    #[test]
    fn a_selection_on_a_missing_clip_is_an_edit_error() {
        let mut session = session();
        let job = SelectJob::start(&session);
        let selection = Selection::from_alpha(2, 1, &[255, 0], Micros(0)).unwrap();
        let before = session.draft().clone();
        assert!(matches!(
            job.store(&mut session, &"missing".into(), &"s".into(), selection),
            Err(JobError::Edit(_))
        ));
        assert_eq!(session.draft(), &before);
    }
}
