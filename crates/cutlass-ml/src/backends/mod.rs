//! Concrete inference backends behind the crate's traits.
//!
//! The seam is local-first but never local-only: a local runtime and a cloud
//! adapter both implement the same trait (e.g. [`crate::Transcribe`]), so a
//! feature swaps backends without touching the feature code. Today only the
//! deterministic [`StubTranscriber`] lives here; the whisper.cpp-backed local
//! runtime and cloud adapters land as added backends behind opt-in features.

pub mod stub;

pub use stub::StubTranscriber;
