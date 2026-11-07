//! The recognizer and the job that writes words onto the draft.
//!
//! The recognizer sees PCM. It does not open a file. [`PcmSource`] is that
//! boundary: the file reader is the first source, and `container` plus
//! `decode` replace it when they can return PCM for a range.
//!
//! Speech stays local. [`ParakeetUnified`] is that path: log-mel frames, a
//! FastConformer on a device handle, then RNN-T. The transcript keeps
//! punctuation and capitals. [`Parakeet`] is the CTC checkpoint. This build
//! passes the CPU device. A later Apple build can pass a Metal device
//! through the same handle. [`LocalRecognizer`] is the small head the job
//! tests use. A remote submit-and-query recognizer is another implementor
//! of [`Recognizer`] and is not built while that switch is off.

#![forbid(unsafe_code)]

mod captions;
mod features;
mod file;
mod job;
mod model;
mod parakeet;
mod prepare;
mod search;
mod timestamp;
mod unified;

/// Set in a tagged run, the tests that need the real speech weights fail
/// when the weights are missing instead of skipping.
pub const REQUIRE_SPEECH_MODEL: &str = "CUTLASS_REQUIRE_SPEECH_MODEL";

/// Called by a real-weights test that found `what` missing, before it
/// skips: in a tagged run ([`REQUIRE_SPEECH_MODEL`]) it fails instead.
///
/// # Panics
///
/// When [`REQUIRE_SPEECH_MODEL`] is set.
pub fn weights_missing(what: &str) {
    assert!(
        std::env::var_os(REQUIRE_SPEECH_MODEL).is_none(),
        "{REQUIRE_SPEECH_MODEL} is set and {what} is missing"
    );
}
mod words;

pub use captions::{
    HeardClip, HeardMiss, caption_request, caption_row, caption_source, heard_clip, rerun_request,
    words_on_timeline,
};
pub use file::FileSource;
pub use job::{Job, JobError};
pub use model::{AcousticModel, AcousticWeights, LocalRecognizer, ModelError};
pub use parakeet::Parakeet;
pub use prepare::{Pcm, PcmSource, PrepareError, fit};
pub use unified::ParakeetUnified;
pub use words::{RecognizedWord, Utterance, WordError, columns};

/// The speech setting. Remote recognition stays unwired.
pub const SPEECH_LOCAL: &str = "local";

/// PCM in, utterances out. The sample rate is the rate [`fit`] must produce.
pub trait Recognizer: Send + Sync {
    fn sample_rate(&self) -> u32;

    fn recognize(&self, pcm: &Pcm) -> Result<Vec<Utterance>, RecognizeError>;
}

/// Why recognition did not produce words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecognizeError {
    /// `pcm` is not at the rate this recognizer accepts.
    SampleRate { expected: u32, actual: u32 },
    /// A word clock is negative or runs backwards.
    WordTime,
    /// The acoustic model failed.
    Model,
    /// Token search failed.
    Search,
}
