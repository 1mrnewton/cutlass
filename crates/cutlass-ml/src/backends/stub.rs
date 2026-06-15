//! A deterministic transcribe backend for tests and headless development.
//!
//! [`StubTranscriber`] returns a fixed [`Transcript`] regardless of the audio
//! it's handed, after honoring the cancel flag and reporting `0.0 → 1.0`
//! progress like a real backend. It lets the downstream features that consume
//! transcripts — the Phase 3 transcript panel, Phase 4 captions — be built and
//! tested without a multi-hundred-MB model on disk or a network call.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::transcribe::{
    Segment, Transcribe, TranscribeError, TranscribeOptions, Transcript, Word,
};

/// Returns a canned transcript for any input.
#[derive(Debug, Clone)]
pub struct StubTranscriber {
    transcript: Transcript,
}

impl StubTranscriber {
    /// A stub that always returns `transcript`.
    pub fn new(transcript: Transcript) -> Self {
        Self { transcript }
    }

    /// A trivial one-segment, two-word transcript ("hello world"), handy for
    /// smoke tests of code that just needs *some* word-timed result.
    pub fn canned() -> Self {
        Self::new(Transcript {
            segments: vec![Segment {
                text: "hello world".into(),
                start: 0.0,
                end: 1.0,
                words: vec![
                    Word {
                        text: "hello".into(),
                        start: 0.0,
                        end: 0.5,
                        confidence: Some(1.0),
                    },
                    Word {
                        text: "world".into(),
                        start: 0.5,
                        end: 1.0,
                        confidence: Some(1.0),
                    },
                ],
            }],
            language: Some("en".into()),
        })
    }
}

impl Transcribe for StubTranscriber {
    fn transcribe(
        &self,
        _audio: &[f32],
        _sample_rate: u32,
        _options: &TranscribeOptions,
        cancel: &AtomicBool,
        on_progress: &mut dyn FnMut(f32),
    ) -> Result<Transcript, TranscribeError> {
        if cancel.load(Ordering::Relaxed) {
            return Err(TranscribeError::Cancelled);
        }
        on_progress(0.0);
        on_progress(1.0);
        Ok(self.transcript.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canned_returns_its_transcript_and_reports_progress() {
        let stub = StubTranscriber::canned();
        let cancel = AtomicBool::new(false);
        let mut progress = Vec::new();
        let out = stub
            .transcribe(
                &[0.0; 16_000],
                16_000,
                &TranscribeOptions::default(),
                &cancel,
                &mut |p| progress.push(p),
            )
            .unwrap();
        assert_eq!(out.text(), "hello world");
        assert_eq!(out.words().count(), 2);
        assert_eq!(progress.first(), Some(&0.0));
        assert_eq!(progress.last(), Some(&1.0));
    }

    #[test]
    fn honors_the_cancel_flag() {
        let stub = StubTranscriber::canned();
        let cancel = AtomicBool::new(true);
        let err = stub
            .transcribe(
                &[],
                16_000,
                &TranscribeOptions::default(),
                &cancel,
                &mut |_| {},
            )
            .unwrap_err();
        assert!(matches!(err, TranscribeError::Cancelled));
    }
}
