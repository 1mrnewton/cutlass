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

mod features;
mod file;
mod job;
mod model;
mod parakeet;
mod prepare;
mod search;
mod timestamp;
mod unified;
mod words;

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
