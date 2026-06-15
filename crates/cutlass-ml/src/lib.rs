//! `cutlass-ml`: local-first, provider-abstracted media inference for Cutlass
//! (AI media roadmap M9 Phase 2+).
//!
//! This crate is where model-backed capabilities live — transcription first,
//! then matting and text-to-speech — each behind a trait so a local runtime
//! (whisper.cpp, ONNX Runtime, a Piper/Kokoro-class TTS) and a cloud adapter
//! are interchangeable. It deliberately knows nothing about projects,
//! timelines, or the compositor: inference is sample/pixel-domain in and plain
//! data out, mirroring the M8 DSP seam (`detect_beats`, `detect_silences`).
//! The engine owns decode and the seconds → tick mapping; the worker owns the
//! thread.
//!
//! Architecture invariants (from `docs/ai-media-roadmap.md`):
//!
//! - **Provider-abstracted, local-first, never local-only.** The traits here
//!   ([`Transcribe`] today) define the seam; local runtimes land first, cloud
//!   adapters are additive, and no feature hard-codes a runtime.
//! - **Models are data, downloaded on demand.** Weights live under
//!   `~/.cutlass/models/`, fetched on first use with a checksum, never bundled
//!   into the binary or a project file.
//! - **Off the default build.** The crate is a workspace member but not a
//!   default member (like the planned `cutlass-py`), so the editor build stays
//!   lean; heavy native backends sit behind opt-in features.

pub mod backends;
pub mod transcribe;

pub use backends::StubTranscriber;
pub use transcribe::{Segment, Transcribe, TranscribeError, TranscribeOptions, Transcript, Word};
