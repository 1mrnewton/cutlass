//! The transcription seam: speech audio → word-timed transcript, behind one
//! trait so a local whisper.cpp runtime and a cloud API are interchangeable.
//!
//! Blocking by design — transcription runs on a worker thread, never the UI
//! thread, mirroring `cutlass-ai`'s `ChatProvider`. Input is sample-domain
//! (`&[f32]` mono at a known rate), the M8 DSP convention: no media, model, or
//! timeline types cross the seam, so the engine owns decode and the
//! seconds → tick mapping and the tricky parts unit-test on synthetic audio.
//! Output is plain data the caller maps onto clips — the substrate both auto
//! captions (Phase 4) and transcript-based editing (Phase 3) consume.

use std::sync::atomic::AtomicBool;

use serde::{Deserialize, Serialize};

/// One recognized word with its time span, in seconds from the start of the
/// transcribed audio. The caller maps these onto source ticks through the
/// clip's window — the same way beat seconds become source ticks in M8.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Word {
    pub text: String,
    /// Start offset in seconds from the start of the transcribed audio.
    pub start: f64,
    /// End offset in seconds (`end >= start`).
    pub end: f64,
    /// Model confidence in `[0, 1]`, or `None` when the backend doesn't report
    /// one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

/// A contiguous run of speech (whisper's "segment"): its text, time span, and
/// the words inside it. A backend without word timing leaves `words` empty and
/// the segment span still carries the timing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub text: String,
    pub start: f64,
    pub end: f64,
    #[serde(default)]
    pub words: Vec<Word>,
}

/// A full transcription result: ordered segments plus the detected language.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Transcript {
    pub segments: Vec<Segment>,
    /// Language tag the backend detected (e.g. `"en"`), if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

impl Transcript {
    /// Every word across all segments, in order.
    pub fn words(&self) -> impl Iterator<Item = &Word> {
        self.segments.iter().flat_map(|s| s.words.iter())
    }

    /// The full transcript text: segment texts trimmed and joined with single
    /// spaces.
    pub fn text(&self) -> String {
        let mut out = String::new();
        for seg in &self.segments {
            let trimmed = seg.text.trim();
            if trimmed.is_empty() {
                continue;
            }
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(trimmed);
        }
        out
    }

    /// No segments at all (the backend heard nothing).
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }
}

/// How to transcribe: the spoken-language hint and whether to translate.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TranscribeOptions {
    /// Spoken-language hint as a language tag (e.g. `"en"`); `None` asks the
    /// backend to auto-detect.
    pub language: Option<String>,
    /// Translate to English instead of transcribing verbatim (whisper's
    /// "translate" task). Local-only backends may ignore it.
    pub translate: bool,
}

/// Transcription failures, kept distinct so the UI can say "the base.en model
/// isn't downloaded yet" instead of "something failed".
#[derive(Debug, thiserror::Error)]
pub enum TranscribeError {
    /// No transcribe backend is configured / selected.
    #[error("transcription is not configured: {0}")]
    NotConfigured(String),
    /// The selected model isn't available locally and couldn't be fetched.
    #[error("transcription model is unavailable: {0}")]
    ModelUnavailable(String),
    /// The backend ran but failed (decode error, runtime error, bad audio).
    #[error("transcription backend failed: {0}")]
    Backend(String),
    /// The cancel flag was raised mid-run.
    #[error("transcription cancelled")]
    Cancelled,
}

/// Speech audio → word-timed transcript.
///
/// `audio` is mono PCM in `[-1, 1]` at `sample_rate`; whisper-class backends
/// expect 16 kHz and may resample or reject other rates. Implementations must
/// check `cancel` periodically and return [`TranscribeError::Cancelled`]
/// promptly when it goes true. `on_progress` receives a fraction in `[0, 1]`
/// as work proceeds — best-effort, so a backend that can't estimate may report
/// only `0.0` then `1.0`.
pub trait Transcribe {
    fn transcribe(
        &self,
        audio: &[f32],
        sample_rate: u32,
        options: &TranscribeOptions,
        cancel: &AtomicBool,
        on_progress: &mut dyn FnMut(f32),
    ) -> Result<Transcript, TranscribeError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(text: &str, start: f64, end: f64) -> Word {
        Word {
            text: text.into(),
            start,
            end,
            confidence: None,
        }
    }

    #[test]
    fn words_iterates_across_segments_in_order() {
        let t = Transcript {
            segments: vec![
                Segment {
                    text: "hello world".into(),
                    start: 0.0,
                    end: 1.0,
                    words: vec![word("hello", 0.0, 0.5), word("world", 0.5, 1.0)],
                },
                Segment {
                    text: "again".into(),
                    start: 1.0,
                    end: 1.5,
                    words: vec![word("again", 1.0, 1.5)],
                },
            ],
            language: Some("en".into()),
        };
        let words: Vec<&str> = t.words().map(|w| w.text.as_str()).collect();
        assert_eq!(words, ["hello", "world", "again"]);
    }

    #[test]
    fn text_trims_and_joins_skipping_blanks() {
        let t = Transcript {
            segments: vec![
                Segment {
                    text: "  hello ".into(),
                    start: 0.0,
                    end: 1.0,
                    words: vec![],
                },
                Segment {
                    text: "   ".into(),
                    start: 1.0,
                    end: 1.2,
                    words: vec![],
                },
                Segment {
                    text: "world".into(),
                    start: 1.2,
                    end: 2.0,
                    words: vec![],
                },
            ],
            language: None,
        };
        assert_eq!(t.text(), "hello world");
        assert!(!t.is_empty());
        assert!(Transcript::default().is_empty());
    }

    #[test]
    fn transcript_round_trips_through_json() {
        let t = Transcript {
            segments: vec![Segment {
                text: "hi".into(),
                start: 0.0,
                end: 0.4,
                words: vec![Word {
                    text: "hi".into(),
                    start: 0.0,
                    end: 0.4,
                    confidence: Some(0.9),
                }],
            }],
            language: Some("en".into()),
        };
        let json = serde_json::to_string(&t).unwrap();
        let back: Transcript = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }
}
