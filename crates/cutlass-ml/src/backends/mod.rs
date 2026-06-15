//! Concrete inference backends behind the crate's traits.
//!
//! The seam is local-first but never local-only: a local runtime and a cloud
//! adapter both implement the same trait (e.g. [`crate::Transcribe`]), so a
//! feature swaps backends without touching the feature code. The deterministic
//! [`StubTranscriber`] and the pure-HTTP [`CloudTranscriber`] always compile;
//! the whisper.cpp-backed local runtime sits behind the opt-in `whisper`
//! feature so the lean build never pulls a native toolchain.

pub mod cloud;
pub mod stub;
#[cfg(feature = "whisper")]
pub mod whisper;

pub use cloud::CloudTranscriber;
pub use stub::StubTranscriber;
#[cfg(feature = "whisper")]
pub use whisper::WhisperTranscriber;
