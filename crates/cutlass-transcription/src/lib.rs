//! Offline speech transcription for Cutlass.
//!
//! This crate has two deliberately separate responsibilities:
//!
//! - [`transcribe_pcm`] runs `whisper.cpp` against caller-supplied 16 kHz mono
//!   floating-point PCM ([`transcribe_pcm_observed`] additionally reports decode
//!   progress). Audio decoding and resampling belong to Cutlass's native media
//!   pipeline and are intentionally not included here.
//! - [`ModelManager`] installs catalogued Whisper models transactionally after
//!   exact size and SHA-256 verification.
//! - [`transcript_to_moment_batch`] converts normalized output into one atomic,
//!   analyzer-scoped moments replacement batch without depending on moments
//!   persistence.
//!
//! No model is bundled in the binary and no model download occurs implicitly
//! during transcription. Callers decide where models live and when to install
//! them.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod model;
mod moments;
mod transcribe;
mod transcript;

pub use model::{
    DownloadError, DownloadReader, HttpDownloader, ModelDownloader, ModelIntegrityError,
    ModelManager, ModelManagerError, ModelSpec, ModelStatus, WhisperModel,
};
pub use moments::{
    TRANSCRIPT_MOMENT_PAYLOAD_VERSION, TRANSCRIPT_SEGMENT_FALLBACK_CONFIDENCE,
    TranscriptMomentConversionError, TranscriptMomentErrorField, TranscriptMomentErrorReason,
    TranscriptMomentRecordIndex, transcript_to_moment_batch,
};
pub use transcribe::{
    CancellationCheck, IgnoreProgress, NeverCancel, ProgressObserver, TranscriptionError,
    TranscriptionOptions, TranscriptionOptionsError, transcribe_pcm, transcribe_pcm_observed,
};
pub use transcript::{Transcript, TranscriptSegment, TranscriptWord};

/// The only PCM sample rate accepted by [`transcribe_pcm`].
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;
