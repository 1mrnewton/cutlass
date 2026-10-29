use cutlass_model::{MaterialId, TimeRange};
use cutlass_session::{EditError, Request, Session};

use crate::prepare::{self, Pcm, PcmSource, PrepareError};
use crate::words;
use crate::{RecognizeError, Recognizer};

/// The recognition that started at one draft revision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Job {
    revision: u64,
}

/// Why the job did not write words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobError {
    /// The draft moved after the job recorded its revision.
    StaleRevision {
        expected: u64,
        actual: u64,
    },
    Prepare(PrepareError),
    Recognize(RecognizeError),
    Edit(EditError),
}

impl Job {
    /// Record the open revision. The write is ignored if that revision moves.
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

    /// Read `range`, fit it to the recognizer, and write one word request.
    pub fn recognize(
        &self,
        session: &mut Session,
        material_id: &MaterialId,
        source: &mut dyn PcmSource,
        range: &TimeRange,
        recognizer: &dyn Recognizer,
    ) -> Result<u64, JobError> {
        self.still_current(session)?;
        let pcm = source.read(range).map_err(JobError::Prepare)?;
        let pcm = prepare::fit(pcm, recognizer.sample_rate()).map_err(JobError::Prepare)?;
        self.finish(session, material_id, &pcm, recognizer)
    }

    /// Write words for PCM that is already at the recognizer's rate.
    pub fn finish(
        &self,
        session: &mut Session,
        material_id: &MaterialId,
        pcm: &Pcm,
        recognizer: &dyn Recognizer,
    ) -> Result<u64, JobError> {
        self.still_current(session)?;
        if pcm.sample_rate != recognizer.sample_rate() {
            return Err(JobError::Recognize(RecognizeError::SampleRate {
                expected: recognizer.sample_rate(),
                actual: pcm.sample_rate,
            }));
        }
        let utterances = recognizer.recognize(pcm).map_err(JobError::Recognize)?;
        let words = words::columns(&utterances)
            .map_err(|_| JobError::Recognize(RecognizeError::WordTime))?;
        session
            .apply(
                self.revision,
                &[Request::SetWords {
                    material_id: material_id.clone(),
                    words,
                }],
            )
            .map_err(|error| map_edit(error, self.revision))
    }

    fn still_current(&self, session: &Session) -> Result<(), JobError> {
        let actual = session.revision();
        if actual == self.revision {
            Ok(())
        } else {
            Err(JobError::StaleRevision {
                expected: self.revision,
                actual,
            })
        }
    }
}

fn map_edit(error: EditError, expected: u64) -> JobError {
    match error {
        EditError::RevisionMismatch { actual, .. } => JobError::StaleRevision { expected, actual },
        other => JobError::Edit(other),
    }
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, MaterialText, Micros, TimeRange};

    use super::*;
    use crate::words::{RecognizedWord, Utterance};

    struct Fixed {
        rate: u32,
        words: Vec<Utterance>,
        seen_rate: std::sync::Mutex<u32>,
        seen_len: std::sync::Mutex<usize>,
        fail: bool,
    }

    impl Recognizer for Fixed {
        fn sample_rate(&self) -> u32 {
            self.rate
        }

        fn recognize(&self, pcm: &Pcm) -> Result<Vec<Utterance>, RecognizeError> {
            *self.seen_rate.lock().expect("rate") = pcm.sample_rate;
            *self.seen_len.lock().expect("len") = pcm.samples.len();
            if pcm.sample_rate != self.rate {
                return Err(RecognizeError::SampleRate {
                    expected: self.rate,
                    actual: pcm.sample_rate,
                });
            }
            if self.fail {
                return Err(RecognizeError::Model);
            }
            Ok(self.words.clone())
        }
    }

    struct Buffer {
        pcm: Pcm,
        fail: bool,
    }

    impl PcmSource for Buffer {
        fn read(&mut self, range: &TimeRange) -> Result<Pcm, PrepareError> {
            if range.duration.0 <= 0 {
                return Err(PrepareError::Range);
            }
            if self.fail {
                return Err(PrepareError::Decode);
            }
            Ok(self.pcm.clone())
        }
    }

    fn session_with_caption() -> (Session, MaterialId) {
        let mut draft = Draft::empty("D");
        let material_id = MaterialId::new("caption");
        draft
            .materials
            .texts
            .push(MaterialText::new(material_id.clone()));
        (Session::open(draft), material_id)
    }

    fn word() -> Vec<Utterance> {
        vec![Utterance {
            words: vec![RecognizedWord {
                text: "hi".to_owned(),
                start_ms: 0,
                end_ms: 1_000,
            }],
        }]
    }

    #[test]
    fn a_fixture_buffer_writes_words_and_times() {
        let (mut session, material_id) = session_with_caption();
        let job = Job::start(&session);
        let recognizer = Fixed {
            rate: 16_000,
            words: word(),
            seen_rate: std::sync::Mutex::new(0),
            seen_len: std::sync::Mutex::new(0),
            fail: false,
        };
        let mut source = Buffer {
            pcm: Pcm {
                sample_rate: 16_000,
                samples: vec![0.0; 16_000],
            },
            fail: false,
        };
        let revision = job
            .recognize(
                &mut session,
                &material_id,
                &mut source,
                &TimeRange::new(0, 1_000_000),
                &recognizer,
            )
            .unwrap();
        assert_eq!(revision, 1);
        let text = &session.draft().materials.texts[0];
        assert_eq!(text.words.text, ["hi"]);
        assert_eq!(text.words.start_time, [0]);
        assert_eq!(text.words.end_time, [1_000]);
        assert_eq!(text.current_words.text, ["hi"]);
        assert_eq!(text.content, "hi");
        assert_eq!(crate::SPEECH_LOCAL, "local");
    }

    #[test]
    fn a_moved_revision_writes_nothing() {
        let (mut session, material_id) = session_with_caption();
        let job = Job::start(&session);
        session
            .apply(
                0,
                &[Request::Place {
                    track_id: session.draft().tracks[0].id.clone(),
                    segment: Box::new(cutlass_model::Segment::new(
                        "a",
                        "media",
                        TimeRange::new(0, 1_000_000),
                    )),
                }],
            )
            .unwrap();
        let recognizer = Fixed {
            rate: 16_000,
            words: word(),
            seen_rate: std::sync::Mutex::new(0),
            seen_len: std::sync::Mutex::new(0),
            fail: false,
        };
        let error = job
            .finish(
                &mut session,
                &material_id,
                &Pcm {
                    sample_rate: 16_000,
                    samples: vec![0.0; 8],
                },
                &recognizer,
            )
            .unwrap_err();
        assert_eq!(
            error,
            JobError::StaleRevision {
                expected: 0,
                actual: 1
            }
        );
        assert!(session.draft().materials.texts[0].words.text.is_empty());
        assert_eq!(*recognizer.seen_len.lock().unwrap(), 0);
        assert_eq!(session.revision(), 1);
    }

    #[test]
    fn a_failed_recognizer_leaves_the_draft() {
        let (mut session, material_id) = session_with_caption();
        let job = Job::start(&session);
        let recognizer = Fixed {
            rate: 16_000,
            words: word(),
            seen_rate: std::sync::Mutex::new(0),
            seen_len: std::sync::Mutex::new(0),
            fail: true,
        };
        let error = job
            .finish(
                &mut session,
                &material_id,
                &Pcm {
                    sample_rate: 16_000,
                    samples: vec![0.0],
                },
                &recognizer,
            )
            .unwrap_err();
        assert_eq!(error, JobError::Recognize(RecognizeError::Model));
        assert_eq!(session.revision(), 0);
        assert!(session.draft().materials.texts[0].content.is_empty());
    }

    #[test]
    fn a_missing_material_leaves_the_revision() {
        let mut session = Session::open(Draft::empty("D"));
        let job = Job::start(&session);
        let recognizer = Fixed {
            rate: 16_000,
            words: word(),
            seen_rate: std::sync::Mutex::new(0),
            seen_len: std::sync::Mutex::new(0),
            fail: false,
        };
        let error = job
            .finish(
                &mut session,
                &MaterialId::new("gone"),
                &Pcm {
                    sample_rate: 16_000,
                    samples: vec![0.0],
                },
                &recognizer,
            )
            .unwrap_err();
        assert_eq!(error, JobError::Edit(EditError::MissingMaterial));
        assert_eq!(session.revision(), 0);
    }

    #[test]
    fn the_source_is_fit_before_the_recognizer_sees_it() {
        let (mut session, material_id) = session_with_caption();
        let job = Job::start(&session);
        let recognizer = Fixed {
            rate: 16_000,
            words: Vec::new(),
            seen_rate: std::sync::Mutex::new(0),
            seen_len: std::sync::Mutex::new(0),
            fail: false,
        };
        let mut source = Buffer {
            pcm: Pcm {
                sample_rate: 8_000,
                samples: vec![0.0, 1.0],
            },
            fail: false,
        };
        job.recognize(
            &mut session,
            &material_id,
            &mut source,
            &TimeRange::new(0, Micros(500_000).0),
            &recognizer,
        )
        .unwrap();
        assert_eq!(*recognizer.seen_rate.lock().unwrap(), 16_000);
        assert_eq!(*recognizer.seen_len.lock().unwrap(), 4);
        assert!(session.draft().materials.texts[0].words.text.is_empty());
    }

    #[test]
    fn a_decode_failure_writes_nothing() {
        let (mut session, material_id) = session_with_caption();
        let job = Job::start(&session);
        let recognizer = Fixed {
            rate: 16_000,
            words: word(),
            seen_rate: std::sync::Mutex::new(0),
            seen_len: std::sync::Mutex::new(0),
            fail: false,
        };
        let mut source = Buffer {
            pcm: Pcm {
                sample_rate: 16_000,
                samples: vec![0.0],
            },
            fail: true,
        };
        let error = job
            .recognize(
                &mut session,
                &material_id,
                &mut source,
                &TimeRange::new(0, 1_000_000),
                &recognizer,
            )
            .unwrap_err();
        assert_eq!(error, JobError::Prepare(PrepareError::Decode));
        assert_eq!(session.revision(), 0);
        assert_eq!(*recognizer.seen_len.lock().unwrap(), 0);
    }

    #[test]
    fn an_empty_range_is_refused() {
        let (mut session, material_id) = session_with_caption();
        let job = Job::start(&session);
        let recognizer = Fixed {
            rate: 16_000,
            words: word(),
            seen_rate: std::sync::Mutex::new(0),
            seen_len: std::sync::Mutex::new(0),
            fail: false,
        };
        let mut source = Buffer {
            pcm: Pcm {
                sample_rate: 16_000,
                samples: vec![0.0],
            },
            fail: false,
        };
        let error = job
            .recognize(
                &mut session,
                &material_id,
                &mut source,
                &TimeRange::new(0, 0),
                &recognizer,
            )
            .unwrap_err();
        assert_eq!(error, JobError::Prepare(PrepareError::Range));
        assert_eq!(session.revision(), 0);
    }
}
